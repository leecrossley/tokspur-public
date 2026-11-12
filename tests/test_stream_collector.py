import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stream_collector", ROOT / "scripts/stream_collector.py"
)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)

REPLY = "The quick brown fox jumps over the lazy dog " * 4
THINKING = "First, consider the animals. "


def _event(delta, finish=None, usage=None, ident="chatcmpl-fake"):
    value = {
        "id": ident,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        value = {"id": ident, "choices": [], "usage": usage}
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def _chunk(data):
    return b"%x\r\n" % len(data) + data + b"\r\n"


def _tokens():
    """One delta per token: four reasoning tokens, then the reply word by word."""
    thinking = [THINKING[i : i + 8] for i in range(0, len(THINKING), 8)]
    words = [word + " " for word in REPLY.split(" ") if word]
    return [("reasoning_content", piece) for piece in thinking] + [
        ("content", word) for word in words
    ]


class FakeServer:
    """Writes SSE events as HTTP chunks in bursts of `packet` tokens, `gap` seconds
    apart, like a server that feeds a whole output packet before the next one."""

    def __init__(self, mode="stream", packet=8, gap=0.03, tool=False, coalesce=False):
        self.coalesce = coalesce
        self.mode = mode
        self.packet = packet
        self.gap = gap
        self.tool = tool
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def _serve(self):
        while True:
            try:
                connection, _ = self.sock.accept()
            except OSError:
                return
            with connection:
                self._answer(connection)

    def _read_request(self, connection):
        data = b""
        while b"\r\n\r\n" not in data:
            data += connection.recv(4096)
        head, body = data.split(b"\r\n\r\n", 1)
        length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        while len(body) < length:
            body += connection.recv(4096)
        return json.loads(body)

    def _answer(self, connection):
        payload = self._read_request(connection)
        if self.mode in ("429", "408", "502"):
            body = json.dumps({"error": {"message": "refused"}}).encode()
            connection.sendall(
                b"HTTP/1.1 %s Refused\r\nContent-Type: application/json\r\n"
                b"Content-Length: %d\r\n\r\n" % (self.mode.encode(), len(body)) + body
            )
            return
        tokens = _tokens()
        if not payload.get("stream"):
            message = {
                "role": "assistant",
                "content": "".join(t for k, t in tokens if k == "content"),
                "reasoning_content": "".join(
                    t for k, t in tokens if k == "reasoning_content"
                ),
            }
            body = json.dumps(
                {
                    "choices": [{"index": 0, "message": message}],
                    "usage": {"completion_tokens": len(tokens)},
                }
            ).encode()
            connection.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: %d\r\n\r\n" % len(body) + body
            )
            return
        connection.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        connection.sendall(_chunk(_event({"role": "assistant"})))
        time.sleep(self.gap)
        try:
            ident = f"chatcmpl-{self.packet}"
            for start in range(0, len(tokens), self.packet):
                deltas = []
                for kind, text in tokens[start : start + self.packet]:
                    if self.coalesce and deltas and kind in deltas[-1]:
                        deltas[-1][kind] += text
                    else:
                        deltas.append({kind: text})
                burst = b"".join(_chunk(_event(delta, ident=ident)) for delta in deltas)
                connection.sendall(burst)
                if self.mode == "drop" and start >= self.packet:
                    return
                if self.mode == "error" and start >= self.packet:
                    connection.sendall(
                        _chunk(b'data: {"error":{"message":"failed"}}\n\n')
                        + b"0\r\n\r\n"
                    )
                    return
                time.sleep(self.gap)
            if self.tool:
                call = {"index": 0, "id": f"call_{self.packet}", "type": "function"}
                connection.sendall(
                    _chunk(
                        _event(
                            {
                                "tool_calls": [
                                    {**call, "function": {"name": "f", "arguments": ""}}
                                ]
                            }
                        )
                    )
                    + _chunk(
                        _event(
                            {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '{"a":1}'}}
                                ]
                            }
                        )
                    )
                )
            connection.sendall(
                _chunk(_event({}, finish="stop"))
                + _chunk(_event({}, usage={"completion_tokens": len(tokens)}))
                + _chunk(b"data: [DONE]\n\n")
                + b"0\r\n\r\n"
            )
        except OSError:
            return

    def close(self):
        self.sock.close()


def run(mode="stream", packet=8, stream=True, raw=False, **options):
    server = FakeServer(mode, packet, **options)
    try:
        record = collector.request(
            server.endpoint, {"stream": stream, "messages": []}, timeout=10
        )
        line = collector.summarise(record, packet_tokens=packet)
        return (record, line) if raw else line
    finally:
        server.close()


def test_packet_gaps_are_burst_gaps_not_chunk_gaps():
    tokens = len(_tokens())
    for packet in (8, 64):
        line = run(packet=packet, gap=0.03)
        assert line["outcome"] == "complete"
        expected = -(-tokens // packet)
        assert line["output_packets"] == expected
        assert line["data_events"] > expected * 2
        if expected > 1:
            assert line["packet_gap_ms"]["p50"] >= 20.0, line["packet_gap_ms"]
        assert line["tokens_per_packet"] == round(tokens / expected, 3)
        assert "per_token_latency_ms" not in line
        assert line["per_token_latency_unavailable"] == "declared packet size is not 1"


def test_per_token_latency_needs_packet_size_one():
    line = run(packet=1, gap=0.01)
    assert line["output_packets"] == len(_tokens())
    assert line["per_token_latency_ms"] == line["packet_gap_ms"]


def test_firsts_are_ordered_and_role_delta_is_only_first_payload():
    # Four reasoning tokens fill the first packet of four; content starts the next.
    line = run(packet=4, gap=0.02)
    assert line["first_payload_ms"] < line["first_reasoning_ms"]
    assert line["first_reasoning_ms"] < line["first_content_ms"]
    assert line["first_content_ms"] <= line["last_output_ms"]
    assert line["first_reasoning_ms"] - line["first_payload_ms"] >= 15.0
    assert line["first_tool_call_ms"] is None


def test_no_prefill_speed_and_no_text_in_the_receipt():
    line = run(packet=8, gap=0.001)
    text = json.dumps(line)
    for name in ("prefill", "prompt_tokens", "tokens_per_second"):
        assert name not in text
    for word in ("quick", "brown", "animals"):
        assert word not in text


def test_digest_is_independent_of_chunking():
    small_raw, small = run(packet=8, gap=0.001, tool=True, raw=True)
    large_raw, large = run(packet=64, gap=0.001, tool=True, coalesce=True, raw=True)
    whole = run(packet=8, stream=False)
    # The raw SSE lines differ (chunk IDs, tool-call IDs and delta splits), so a
    # digest over raw lines would differ; the assembled digest must not.
    small_lines = [data for _, data in small_raw["events"]]
    large_lines = [data for _, data in large_raw["events"]]
    assert len(small_lines) > len(large_lines)
    assert small["digest"] == large["digest"]
    assembly = collector.Assembly()
    for kind, text in _tokens():
        assembly.add_delta({"choices": [{"delta": {kind: text}}]})
    assembly.add_delta(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "another-id",
                                "type": "function",
                                "function": {"name": "f", "arguments": '{"a":1}'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert assembly.digest() == small["digest"]
    # The non-streaming reply has no tool call, so its digest differs by exactly that.
    plain = collector.Assembly()
    for kind, text in _tokens():
        plain.add_delta({"choices": [{"delta": {kind: text}}]})
    assert whole["digest"] == plain.digest() != small["digest"]


def test_every_request_is_classified_into_the_denominator():
    lines = []
    for mode in ("429", "408", "502"):
        lines.append(run(mode=mode))
    lines.append(run(mode="drop", gap=0.001))
    lines.append(run(mode="error", gap=0.001))
    lines.append(run(gap=0.001))
    server = FakeServer(gap=0.001)
    try:
        cut = collector.request(
            server.endpoint, {"stream": True}, timeout=10, cut_after=3
        )
    finally:
        server.close()
    lines.append(collector.summarise(cut))
    refused = collector.request("http://127.0.0.1:9/v1", {"stream": True}, timeout=1)
    lines.append(collector.summarise(refused))
    outcomes = [line["outcome"] for line in lines]
    assert outcomes == [
        "http_429",
        "http_408",
        "http_502",
        "incomplete",
        "mid_stream_error",
        "complete",
        "client_cancel",
        "connect_error",
    ]
    summary = collector.denominators(lines)
    assert summary["sent"] == len(lines) == sum(summary["outcomes"].values())
    assert summary["outcomes"]["http_429"] == 1
    assert all(
        line["digest"] is None for line in lines if line["outcome"] != "complete"
    )


def test_main_writes_a_private_receipt(tmp_path, monkeypatch, capsys):
    server = FakeServer(packet=8, gap=0.001)
    body = tmp_path / "body.json"
    body.write_text(json.dumps({"stream": True, "messages": []}))
    out = tmp_path / "receipt.jsonl"
    monkeypatch.setenv("TOKSPUR_API_KEY", "test")
    try:
        assert (
            collector.main(
                [
                    "--endpoint",
                    server.endpoint,
                    "--body-file",
                    str(body),
                    "--repeat",
                    "2",
                    "--packet-tokens",
                    "8",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
    finally:
        server.close()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row.get("index") for row in rows[:2]] == [0, 1]
    assert rows[2]["summary"]["outcomes"]["complete"] == 2
    assert json.loads(capsys.readouterr().out)["sent"] == 2
