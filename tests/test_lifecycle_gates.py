import base64
import hashlib
import importlib.util
import json
import socket
import threading
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lifecycle_gates", ROOT / "scripts/lifecycle_gates.py"
)
gates = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gates)

PACKET = 4


def _words(prompt, seed, count):
    """Deterministic generated reply text: a function of prompt and seed only."""
    out = []
    state = hashlib.sha256(f"{prompt}|{seed}".encode()).digest()
    for _ in range(count):
        state = hashlib.sha256(state).digest()
        out.append(f"w{state[0]:02x} ")
    return out


def _chunk(data):
    return b"%x\r\n" % len(data) + data + b"\r\n"


def _sse(value):
    return _chunk(b"data: " + json.dumps(value).encode() + b"\n\n")


class FakeServed:
    """A fake served API with /health, a seeded generator, a prefix cache and a
    capture-format worker exchange."""

    def __init__(
        self,
        tmp_path,
        abort_on_cut=True,
        health_after_cut=200,
        cache_capacity=2,
        stale_after_eviction=False,
    ):
        self.exchange = tmp_path / f"exchange-{id(self)}.jsonl"
        self.abort_on_cut = abort_on_cut
        self.health_after_cut = health_after_cut
        self.cache_capacity = cache_capacity
        self.stale_after_eviction = stale_after_eviction
        self.cut_seen = False
        self.seen = {}
        self.cache = OrderedDict()
        self.evicted = set()
        self.generation = 0
        self.lock = threading.Lock()
        self._record("worker", {"event": "ready"})
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.endpoint = f"http://127.0.0.1:{self.sock.getsockname()[1]}/v1"
        threading.Thread(target=self._serve, daemon=True).start()

    def _record(self, direction, value):
        line = json.dumps(value).encode() + b"\n"
        with open(self.exchange, "ab") as out:
            out.write(
                json.dumps(
                    {
                        "schema": 1,
                        "direction": direction,
                        "bytes": base64.b64encode(line).decode(),
                    }
                ).encode()
                + b"\n"
            )

    def _serve(self):
        while True:
            try:
                connection, _ = self.sock.accept()
            except OSError:
                return
            with connection:
                try:
                    self._answer(connection)
                except OSError:
                    pass

    def _read(self, connection):
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
        return head.split(b"\r\n")[0].decode(), json.loads(body) if body else None

    def _answer(self, connection):
        start, payload = self._read(connection)
        if start.startswith("GET /health"):
            code = self.health_after_cut if self.cut_seen else 200
            body = json.dumps(
                {"status": "ready" if code == 200 else "worker_failed"}
            ).encode()
            connection.sendall(
                b"HTTP/1.1 %d X\r\nContent-Length: %d\r\n\r\n" % (code, len(body))
                + body
            )
            return
        messages = payload["messages"]
        prefix = messages[0]["content"]
        prompt = json.dumps(messages)
        cached = self._cache(prefix)
        stale = self.stale_after_eviction and prefix in self.evicted and not cached
        words = _words("stale" if stale else prompt, payload.get("seed", 0), 16)
        usage = {
            "completion_tokens": len(words),
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        with self.lock:
            self.generation += 1
            ident = self.generation
        if not payload.get("stream"):
            body = json.dumps(
                {
                    "choices": [{"index": 0, "message": {"content": "".join(words)}}],
                    "usage": usage,
                }
            ).encode()
            connection.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body
            )
            return
        self._record(
            "host",
            {
                "tokens": [],
                "generate": PACKET,
                "mtp_generation": {"id": ident, "begin": True, "remaining": len(words)},
            },
        )
        connection.sendall(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            + _sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        )
        for start in range(0, len(words), PACKET):
            if start:
                self._record(
                    "host",
                    {
                        "tokens": [],
                        "generate": PACKET,
                        "mtp_generation": {"id": ident, "begin": False},
                    },
                )
            self._record(
                "worker",
                {"event": "done", "generated_ids": list(range(PACKET))},
            )
            try:
                connection.sendall(
                    b"".join(
                        _sse({"choices": [{"index": 0, "delta": {"content": word}}]})
                        for word in words[start : start + PACKET]
                    )
                )
                time.sleep(0.02)
                connection.sendall(b"")
                if self._peer_closed(connection):
                    raise BrokenPipeError
            except OSError:
                self.cut_seen = True
                if self.abort_on_cut:
                    self._record(
                        "host",
                        {
                            "tokens": [],
                            "generate": 0,
                            "mtp_generation": {"id": ident, "abort": True},
                        },
                    )
                    self._record("worker", {"event": "done", "generated_ids": []})
                return
        connection.sendall(
            _sse({"choices": [], "usage": usage})
            + _chunk(b"data: [DONE]\n\n")
            + b"0\r\n\r\n"
        )

    @staticmethod
    def _peer_closed(connection):
        connection.setblocking(False)
        try:
            return connection.recv(1, socket.MSG_PEEK) == b""
        except BlockingIOError:
            return False
        except OSError:
            return True
        finally:
            connection.setblocking(True)

    def _cache(self, prefix):
        """A prefix is captured on its second appearance; least recent goes first."""
        with self.lock:
            if prefix in self.cache:
                self.cache.move_to_end(prefix)
                return len(prefix)
            self.seen[prefix] = self.seen.get(prefix, 0) + 1
            if self.seen[prefix] >= 2 and self.cache_capacity:
                self.cache[prefix] = True
                while len(self.cache) > self.cache_capacity:
                    self.evicted.add(self.cache.popitem(last=False)[0])
            return 0

    def close(self):
        self.sock.close()


def body(prefix="system prefix", user="hello", seed=7, stream=True):
    return {
        "stream": stream,
        "seed": seed,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": user},
        ],
    }


def cancel(tmp_path, cut_after=6, repeat=None, **server):
    fake = FakeServed(tmp_path, **server)
    try:
        gates_report = gates.cancel_gate(
            fake.endpoint,
            body(),
            rounds=2,
            cut_after=cut_after,
            exchange=fake.exchange,
            repeat_payload=repeat,
            timeout=10,
            ready_timeout=0.5,
        )
    finally:
        fake.close()
    return gates_report


def failing(report):
    return sorted(
        {
            check["check"].split("_", 1)[1]
            if check["check"].startswith("round")
            else check["check"]
            for check in report["checks"]
            if check["result"] == "fail"
        }
    )


def test_cancel_gate_passes_against_a_correct_fake(tmp_path):
    report = cancel(tmp_path)
    assert report["verdict"] == "pass", report
    assert all(r["cut_outcome"] == "client_cancel" for r in report["rounds"])


def test_planted_seed_change_fails_only_equality(tmp_path):
    report = cancel(tmp_path, repeat=body(seed=8))
    assert failing(report) == ["repeat_equals_solo"], report


def test_planted_late_cut_fails_only_liveness(tmp_path):
    report = cancel(tmp_path, cut_after=10_000)
    assert failing(report) == ["cut_was_live"], report
    assert report["checks"][-1]["result"] == "not_evaluated"


def test_planted_missing_abort_fails_only_the_exchange_check(tmp_path):
    report = cancel(tmp_path, abort_on_cut=False)
    assert failing(report) == ["exchange_shows_interrupts"], report


def test_planted_unhealthy_server_fails_only_health(tmp_path):
    report = cancel(tmp_path, health_after_cut=503)
    assert failing(report) == ["health_ready"], report


def test_no_exchange_is_inconclusive_not_a_pass(tmp_path):
    fake = FakeServed(tmp_path)
    try:
        report = gates.cancel_gate(fake.endpoint, body(), rounds=1, timeout=10)
    finally:
        fake.close()
    assert report["verdict"] == "inconclusive"


def test_exchange_parser_counts_only_live_aborted_generations(tmp_path):
    fake = FakeServed(tmp_path)
    fake.close()
    for value in (
        (
            "host",
            {"generate": 4, "mtp_generation": {"id": 1, "begin": True, "remaining": 8}},
        ),
        ("worker", {"event": "done", "generated_ids": [1, 2, 3, 4]}),
        ("host", {"generate": 4, "mtp_generation": {"id": 1, "begin": False}}),
        ("worker", {"event": "done", "generated_ids": [5, 6, 7, 8]}),
        ("host", {"generate": 0, "mtp_generation": {"id": 1, "abort": True}}),
        (
            "host",
            {"generate": 4, "mtp_generation": {"id": 2, "begin": True, "remaining": 8}},
        ),
        ("worker", {"event": "done", "generated_ids": [1, 2, 3, 4]}),
        ("host", {"generate": 0, "mtp_generation": {"id": 2, "abort": True}}),
    ):
        fake._record(*value)
    generations = gates.read_exchange(fake.exchange)
    assert [g["generated"] for g in generations] == [8, 4]
    # Generation 1 reached its budget before the abort, so only 2 was interrupted.
    assert [g["id"] for g in gates.interrupted_generations(generations)] == [2]


def evict(tmp_path, fills=3, **server):
    kept = body("kept prefix", stream=False)
    other = body("other prefix", user="unrelated", stream=False)
    fill_bodies = [body(f"fill {i}", stream=False) for i in range(fills)]
    off = FakeServed(tmp_path, cache_capacity=0)
    try:
        reference = gates.evict_reference(off.endpoint, kept, other, timeout=10)
    finally:
        off.close()
    on = FakeServed(tmp_path, **server)
    try:
        return gates.evict_gate(
            on.endpoint, kept, other, fill_bodies, reference, timeout=10
        )
    finally:
        on.close()


def test_evict_gate_passes_when_fills_evict_the_kept_prefix(tmp_path):
    report = evict(tmp_path)
    assert report["verdict"] == "pass", report
    assert report["cached"][-2] > 0 and report["cached"][-1] < report["cached"][-2]


def test_planted_no_fills_fails_eviction(tmp_path):
    report = evict(tmp_path, fills=0)
    assert failing(report) == ["evicted"], report


def test_planted_stale_answer_fails_equality(tmp_path):
    report = evict(tmp_path, stale_after_eviction=True)
    assert failing(report) == ["replies_equal_cache_off"], report


def test_nothing_cached_is_vacuous(tmp_path):
    report = evict(tmp_path, cache_capacity=0)
    assert report["verdict"] == "inconclusive", report
    assert failing(report) == []
