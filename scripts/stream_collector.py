#!/usr/bin/env python3
"""Private served-stream collector for an OpenAI-compatible chat endpoint.

Standard library only. It sends chat completion requests, stamps each step of the
reply on a monotonic clock, and writes a receipt line per request with no prompt
or reply text in it.

What it measures and what it refuses to claim:

- Times are milliseconds from the send: response headers, first payload, first
  reasoning, first content, first tool call and last output. The first payload is
  the role delta, which a server may write before prefill.
- Chunks that arrive together form one packet. A server writes all chunks of an
  output packet back to back, so the gap between chunks is not the gap between
  tokens. Gaps are reported between packets only.
- Tokens per packet is the reported completion token count over output packets.
  Per-token latency is reported only when the declared packet size is 1.
- Prompt tokens over first-token time is never reported, because first-token time
  includes queueing, admission and the first packet, not prefill alone.
- The digest is a SHA-256 of the assembled content, reasoning and tool calls
  (index, type, name and arguments; IDs excluded), so it does not depend on how
  the reply was split into chunks.
- Every request sent is classified, refusals included, so the outcome counts are
  a denominator.
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from urllib.parse import urlsplit

SCHEMA = 1
OUTCOMES = (
    "complete",
    "http_429",
    "http_408",
    "http_502",
    "http_other",
    "client_cancel",
    "incomplete",
    "mid_stream_error",
    "connect_error",
)
MAX_HEADER_BYTES = 64 * 1024
MAX_BODY_BYTES = 64 * 1024 * 1024
DEFAULT_BURST_GAP_MS = 2.0


class CollectorError(Exception):
    pass


def need(condition, message):
    if not condition:
        raise CollectorError(message)


def _request_bytes(endpoint, body, api_key):
    parts = urlsplit(endpoint)
    need(parts.scheme == "http", "only plain http to a local endpoint is supported")
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 80
    path = parts.path.rstrip("/") + "/chat/completions"
    lines = [
        f"POST {path} HTTP/1.1",
        f"Host: {host}:{port}",
        "Content-Type: application/json",
        "Accept: text/event-stream, application/json",
        f"Content-Length: {len(body)}",
        "Connection: close",
    ]
    if api_key:
        lines.append(f"Authorization: Bearer {api_key}")
    return host, port, ("\r\n".join(lines) + "\r\n\r\n").encode() + body


class _ChunkedDecoder:
    """Incremental HTTP/1.1 chunked transfer decoder."""

    def __init__(self):
        self.buffer = b""
        self.remaining = None
        self.finished = False

    def feed(self, data):
        self.buffer += data
        out = []
        while not self.finished:
            if self.remaining is None:
                end = self.buffer.find(b"\r\n")
                if end < 0:
                    break
                size = int(self.buffer[:end].split(b";")[0].strip() or b"0", 16)
                self.buffer = self.buffer[end + 2 :]
                if size == 0:
                    self.finished = True
                    break
                self.remaining = size
            if len(self.buffer) < self.remaining + 2:
                break
            out.append(self.buffer[: self.remaining])
            self.buffer = self.buffer[self.remaining + 2 :]
            self.remaining = None
        return b"".join(out)


def request(
    endpoint,
    payload,
    api_key=None,
    timeout=600.0,
    cut_after=None,
    clock=time.monotonic,
):
    """Sends one request and returns the raw exchange record (in memory only).

    `cut_after` closes the connection after that many SSE data events.
    """
    body = json.dumps(payload, separators=(",", ":")).encode()
    record = {
        "status": None,
        "stream": bool(payload.get("stream")),
        "sent": None,
        "headers_at": None,
        "events": [],
        "body": b"",
        "cut": False,
        "closed_early": False,
        "connect_error": None,
    }
    try:
        host, port, raw = _request_bytes(endpoint, body, api_key)
        sock = socket.create_connection((host, port), timeout=timeout)
    except (OSError, ValueError) as error:
        record["connect_error"] = type(error).__name__
        return record
    with sock:
        sock.settimeout(timeout)
        record["sent"] = clock()
        sock.sendall(raw)
        head = b""
        decoder = None
        length = None
        pending = b""
        received = 0
        while True:
            try:
                data = sock.recv(65536)
            except OSError:
                record["closed_early"] = True
                break
            now = clock()
            if not data:
                record["closed_early"] = True
                break
            if record["status"] is None:
                head += data
                need(len(head) <= MAX_HEADER_BYTES, "response header too large")
                end = head.find(b"\r\n\r\n")
                if end < 0:
                    continue
                header_lines = head[:end].decode("latin-1").split("\r\n")
                record["status"] = int(header_lines[0].split()[1])
                record["headers_at"] = now
                headers = {}
                for line in header_lines[1:]:
                    name, _, value = line.partition(":")
                    headers[name.strip().lower()] = value.strip()
                if "chunked" in headers.get("transfer-encoding", "").lower():
                    decoder = _ChunkedDecoder()
                elif "content-length" in headers:
                    length = int(headers["content-length"])
                data = head[end + 4 :]
            chunk = decoder.feed(data) if decoder else data
            received += len(chunk)
            need(received <= MAX_BODY_BYTES, "response body too large")
            if record["status"] == 200 and record["stream"]:
                pending += chunk
                while b"\n\n" in pending:
                    event, pending = pending.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        if line.startswith(b"data:"):
                            record["events"].append((now, line[5:].strip()))
                    if cut_after is not None and len(record["events"]) >= cut_after:
                        record["cut"] = True
                        break
                if record["cut"]:
                    break
            else:
                record["body"] += chunk
            if decoder and decoder.finished:
                break
            if length is not None and len(record["body"]) >= length:
                break
    return record


def _json(data):
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return None


def _delta_kinds(value):
    choices = value.get("choices") if isinstance(value, dict) else None
    if not choices or not isinstance(choices, list):
        return set()
    delta = choices[0].get("delta") or {}
    kinds = set()
    if delta.get("reasoning_content"):
        kinds.add("reasoning")
    if delta.get("content"):
        kinds.add("content")
    if delta.get("tool_calls"):
        kinds.add("tool_call")
    return kinds


class Assembly:
    """The reply assembled independently of how it was split into chunks."""

    def __init__(self):
        self.content = []
        self.reasoning = []
        self.tools = {}

    def add_message(self, message):
        if message.get("content"):
            self.content.append(message["content"])
        if message.get("reasoning_content"):
            self.reasoning.append(message["reasoning_content"])
        for position, call in enumerate(message.get("tool_calls") or []):
            self._tool(call, position)

    def _tool(self, call, position):
        index = call.get("index", position)
        slot = self.tools.setdefault(index, {"type": "", "name": "", "arguments": ""})
        if call.get("type"):
            slot["type"] = call["type"]
        function = call.get("function") or {}
        slot["name"] += function.get("name") or ""
        slot["arguments"] += function.get("arguments") or ""

    def add_delta(self, value):
        for choice in value.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.content.append(delta["content"])
            if delta.get("reasoning_content"):
                self.reasoning.append(delta["reasoning_content"])
            for position, call in enumerate(delta.get("tool_calls") or []):
                self._tool(call, position)

    def canonical(self):
        tools = [{"index": index, **slot} for index, slot in sorted(self.tools.items())]
        return json.dumps(
            {
                "content": "".join(self.content),
                "reasoning": "".join(self.reasoning),
                "tool_calls": tools,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()

    def digest(self):
        return hashlib.sha256(self.canonical()).hexdigest()


def group_packets(arrivals, burst_gap_ms=DEFAULT_BURST_GAP_MS):
    """Groups event arrival times into packets: a gap above `burst_gap_ms` starts
    a new packet. Returns lists of event indexes."""
    packets = []
    previous = None
    for index, at in enumerate(arrivals):
        if previous is None or (at - previous) * 1000.0 > burst_gap_ms:
            packets.append([])
        packets[-1].append(index)
        previous = at
    return packets


def percentiles(values):
    if not values:
        return None
    ordered = sorted(values)

    def rank(fraction):
        position = max(1, -(-int(fraction * 1000) * len(ordered) // 1000))
        return ordered[min(position, len(ordered)) - 1]

    return {
        "count": len(ordered),
        "p50": round(rank(0.50), 3),
        "p90": round(rank(0.90), 3),
        "p99": round(rank(0.99), 3),
        "max": round(ordered[-1], 3),
    }


def classify(record, values):
    if record["connect_error"]:
        return "connect_error"
    status = record["status"]
    if status is None:
        return "incomplete"
    if status != 200:
        return f"http_{status}" if status in (429, 408, 502) else "http_other"
    if not record["stream"]:
        body = _json(record["body"])
        return (
            "complete"
            if isinstance(body, dict) and "choices" in body
            else ("incomplete")
        )
    if record["cut"]:
        return "client_cancel"
    if any(isinstance(value, dict) and "error" in value for value in values):
        return "mid_stream_error"
    if record["events"] and record["events"][-1][1] == b"[DONE]":
        return "complete"
    return "incomplete"


def summarise(record, packet_tokens=None, burst_gap_ms=DEFAULT_BURST_GAP_MS):
    """One receipt line. It holds stamps, counts, the outcome and the digest, and no
    text."""
    values = [
        None if data == b"[DONE]" else _json(data) for _, data in record["events"]
    ]
    outcome = classify(record, values)
    out = {"schema": SCHEMA, "outcome": outcome, "stream": record["stream"]}
    if record["status"] is not None:
        out["status"] = record["status"]
    sent = record["sent"]

    def since(at):
        return None if at is None or sent is None else round((at - sent) * 1000, 3)

    out["headers_ms"] = since(record["headers_at"])
    assembly = Assembly()
    usage = None
    if record["stream"] and record["status"] == 200:
        firsts = {"reasoning": None, "content": None, "tool_call": None}
        last_output = None
        output_arrivals = []
        for (at, _), value in zip(record["events"], values):
            if not isinstance(value, dict):
                continue
            if value.get("usage"):
                usage = value["usage"]
            kinds = _delta_kinds(value)
            assembly.add_delta(value)
            for kind in kinds:
                if firsts[kind] is None:
                    firsts[kind] = at
            if kinds:
                last_output = at
                output_arrivals.append(at)
        payload_times = [at for at, data in record["events"] if data != b"[DONE]"]
        out["first_payload_ms"] = since(payload_times[0] if payload_times else None)
        out["first_reasoning_ms"] = since(firsts["reasoning"])
        out["first_content_ms"] = since(firsts["content"])
        out["first_tool_call_ms"] = since(firsts["tool_call"])
        out["last_output_ms"] = since(last_output)
        out["data_events"] = len(record["events"])
        packets = group_packets(output_arrivals, burst_gap_ms)
        starts = [output_arrivals[packet[0]] for packet in packets]
        gaps = [(b - a) * 1000.0 for a, b in zip(starts, starts[1:])]
        out["output_packets"] = len(packets)
        out["burst_gap_ms"] = burst_gap_ms
        out["packet_gap_ms"] = percentiles(gaps)
    elif record["status"] == 200:
        body = _json(record["body"]) or {}
        usage = body.get("usage")
        for choice in body.get("choices") or []:
            assembly.add_message(choice.get("message") or {})
    completion_tokens = (usage or {}).get("completion_tokens")
    details = (usage or {}).get("prompt_tokens_details") or {}
    out["usage_cached_tokens"] = details.get("cached_tokens")
    out["completion_tokens"] = completion_tokens
    if record["stream"] and out.get("output_packets") and completion_tokens:
        out["tokens_per_packet"] = round(completion_tokens / out["output_packets"], 3)
    out["declared_packet_tokens"] = packet_tokens
    if packet_tokens == 1 and out.get("packet_gap_ms"):
        out["per_token_latency_ms"] = out["packet_gap_ms"]
    else:
        out["per_token_latency_unavailable"] = (
            "declared packet size is not 1" if packet_tokens != 1 else "no gaps"
        )
    out["digest"] = assembly.digest() if outcome == "complete" else None
    return out


def denominators(lines):
    """Counts every receipt line by outcome; the counts sum to the requests sent."""
    counts = {name: 0 for name in OUTCOMES}
    for line in lines:
        counts[line["outcome"]] += 1
    sent = len(lines)
    need(sum(counts.values()) == sent, "an outcome escaped the denominator")
    return {"sent": sent, "outcomes": counts}


def _api_key(args):
    if args.api_key_file:
        with open(args.api_key_file, encoding="utf-8") as handle:
            return handle.read().strip()
    return os.environ.get("TOKSPUR_API_KEY")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--endpoint", required=True, help="for example http://127.0.0.1:8080/v1"
    )
    parser.add_argument("--body-file", required=True, help="request JSON; kept private")
    parser.add_argument("--api-key-file", help="otherwise TOKSPUR_API_KEY")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--packet-tokens", type=int, choices=(1, 8, 64))
    parser.add_argument("--burst-gap-ms", type=float, default=DEFAULT_BURST_GAP_MS)
    parser.add_argument("--cut-after", type=int)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", required=True, help="receipt JSONL, outside Git")
    args = parser.parse_args(argv)
    with open(args.body_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    api_key = _api_key(args)
    lines = []
    with open(args.out, "x", encoding="utf-8") as out:
        for index in range(args.repeat):
            record = request(
                args.endpoint, payload, api_key, args.timeout, args.cut_after
            )
            line = summarise(record, args.packet_tokens, args.burst_gap_ms)
            line["index"] = index
            lines.append(line)
            out.write(json.dumps(line, sort_keys=True) + "\n")
        out.write(json.dumps({"summary": denominators(lines)}, sort_keys=True) + "\n")
    json.dump(denominators(lines), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
