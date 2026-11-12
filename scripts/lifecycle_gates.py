#!/usr/bin/env python3
"""Private served lifecycle gates: cancellation and eviction.

Standard library only. It drives one running server at a time and writes receipts
that hold outcomes, counts and digests, never prompt or reply text.

cancel: on a freshly started server, a solo reply is the reference. Each round cuts
a streamed reply after N data events, polls /health, sends the same request again
and compares its digest with the reference. The worker exchange recorded by the
existing `runtime_checks.py capture` wrapper must show that each cut interrupted a
live generation: a generation that produced fewer tokens than its budget and was
then aborted by the host. Without that proof the gate is inconclusive, not passed.

evict: run `evict-reference` against a server started with
`--shared-prefix-cache off`, then `evict` against a server with the cache on (never
both at once). A kept conversation must first resume from cache through a non-append
path (an unrelated conversation in between); fills then present unrelated prefixes
twice each; the kept conversation must afterwards resume fewer tokens, and every
reply must equal the cache-off reference. Do not run it until a served run has
shown that a non-append turn reports cached tokens; otherwise it is vacuous.
"""

import argparse
import base64
import http.client
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_collector as collector  # noqa: E402

SCHEMA = 1
MAX_EXCHANGE_BYTES = 256 * 1024 * 1024
MAX_FILLS = 16
MAX_WARM_ATTEMPTS = 4


class GateError(Exception):
    pass


def need(condition, message):
    if not condition:
        raise GateError(message)


def health(endpoint, timeout=5.0):
    """The /health status code and its `status` member."""
    parts = urlsplit(endpoint)
    connection = http.client.HTTPConnection(
        parts.hostname or "127.0.0.1", parts.port or 80, timeout=timeout
    )
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read()
    except OSError:
        return None, None
    finally:
        connection.close()
    try:
        status = json.loads(body).get("status")
    except (ValueError, AttributeError):
        status = None
    return response.status, status


def wait_ready(endpoint, timeout=30.0, interval=0.1, clock=time.monotonic):
    deadline = clock() + timeout
    last = (None, None)
    while clock() < deadline:
        last = health(endpoint)
        if last == (200, "ready"):
            return True, last
        time.sleep(interval)
    return False, last


def read_exchange(path):
    """Generations in a capture trace: for each `mtp_generation` id, its budget, the
    tokens its completions produced and whether the host aborted it."""
    need(Path(path).stat().st_size <= MAX_EXCHANGE_BYTES, "exchange too large")
    generations = {}
    order = []
    pending = None
    for line in Path(path).read_bytes().splitlines():
        record = json.loads(line)
        need(
            record.get("schema") == 1 and record.get("direction") in {"host", "worker"},
            "invalid exchange record",
        )
        value = json.loads(base64.b64decode(record["bytes"], validate=True))
        if record["direction"] == "host":
            command = value.get("mtp_generation")
            pending = None
            if not isinstance(command, dict) or "id" not in command:
                continue
            ident = command["id"]
            if ident not in generations:
                generations[ident] = {
                    "id": ident,
                    "budget": None,
                    "generated": 0,
                    "commands": 0,
                    "aborted": False,
                    "generate_after_abort": False,
                }
                order.append(ident)
            generation = generations[ident]
            if command.get("abort"):
                generation["aborted"] = True
                continue
            if generation["aborted"] and value.get("generate", 0) > 0:
                generation["generate_after_abort"] = True
            if command.get("begin") and generation["budget"] is None:
                generation["budget"] = command.get("remaining")
            if value.get("generate", 0) > 0:
                generation["commands"] += 1
                pending = generation
        elif pending is not None and value.get("event") == "done":
            pending["generated"] += len(value.get("generated_ids") or [])
            pending = None
    return [generations[ident] for ident in order]


def interrupted_generations(generations):
    """Generations that were live when aborted: they ran, stopped short of their
    budget, and were not driven further after the abort."""
    return [
        generation
        for generation in generations
        if generation["aborted"]
        and generation["commands"] > 0
        and not generation["generate_after_abort"]
        and generation["budget"] is not None
        and generation["generated"] < generation["budget"]
    ]


def _check(name, passed, detail=None):
    result = {"check": name, "result": "pass" if passed else "fail"}
    if detail is not None:
        result["detail"] = detail
    return result


def _not_evaluated(name, reason):
    return {"check": name, "result": "not_evaluated", "detail": reason}


def cancel_gate(
    endpoint,
    payload,
    rounds=2,
    cut_after=4,
    api_key=None,
    exchange=None,
    repeat_payload=None,
    timeout=600.0,
    ready_timeout=30.0,
):
    """Runs the cancellation gate. `repeat_payload` replaces the repeat request (a
    planted fault in tests). The verdict is pass, fail or inconclusive."""
    need(payload.get("stream") is True, "the cancel gate needs a streaming request")
    solo = collector.summarise(collector.request(endpoint, payload, api_key, timeout))
    checks = [_check("solo_complete", solo["outcome"] == "complete", solo["outcome"])]
    rounds_out = []
    for index in range(rounds):
        cut = collector.summarise(
            collector.request(endpoint, payload, api_key, timeout, cut_after=cut_after)
        )
        live = (
            cut["outcome"] == "client_cancel" and cut.get("last_output_ms") is not None
        )
        ready, last = wait_ready(endpoint, ready_timeout)
        repeat = collector.summarise(
            collector.request(endpoint, repeat_payload or payload, api_key, timeout)
        )
        equal = (
            repeat["outcome"] == "complete"
            and solo["digest"] is not None
            and repeat["digest"] == solo["digest"]
        )
        rounds_out.append(
            {
                "round": index,
                "cut_outcome": cut["outcome"],
                "cut_events": cut.get("data_events"),
                "health": list(last),
                "repeat_outcome": repeat["outcome"],
            }
        )
        checks.append(_check(f"round{index}_cut_was_live", live, cut["outcome"]))
        checks.append(_check(f"round{index}_health_ready", ready, list(last)))
        checks.append(_check(f"round{index}_repeat_equals_solo", equal))
    live_cuts = sum(
        1
        for check in checks
        if check["check"].endswith("_cut_was_live") and check["result"] == "pass"
    )
    if exchange is None:
        checks.append(_not_evaluated("exchange_shows_interrupts", "no exchange given"))
    elif live_cuts == 0:
        checks.append(
            _not_evaluated("exchange_shows_interrupts", "no live cut to look for")
        )
    else:
        found = len(interrupted_generations(read_exchange(exchange)))
        checks.append(
            _check(
                "exchange_shows_interrupts",
                found >= live_cuts,
                {"interrupted": found, "live_cuts": live_cuts},
            )
        )
    return {
        "schema": SCHEMA,
        "gate": "cancel",
        "verdict": verdict(checks),
        "checks": checks,
        "rounds": rounds_out,
        "solo_digest": solo["digest"],
    }


def verdict(checks):
    if any(check["result"] == "fail" for check in checks):
        return "fail"
    if any(check["result"] == "not_evaluated" for check in checks):
        return "inconclusive"
    return "pass"


def _ask(endpoint, payload, api_key, timeout):
    line = collector.summarise(collector.request(endpoint, payload, api_key, timeout))
    return {
        "outcome": line["outcome"],
        "digest": line["digest"],
        "cached": line.get("usage_cached_tokens"),
    }


def evict_reference(endpoint, kept, other, api_key=None, timeout=600.0):
    """The kept turn's reply from a `--shared-prefix-cache off` server, asked twice
    with the unrelated turn in between; both replies must agree."""
    replies = []
    for _ in range(2):
        replies.append(_ask(endpoint, kept, api_key, timeout))
        _ask(endpoint, other, api_key, timeout)
    digests = {reply["digest"] for reply in replies}
    need(
        len(digests) == 1 and None not in digests,
        "cache-off replies are incomplete or differ",
    )
    return {
        "schema": SCHEMA,
        "gate": "evict-reference",
        "kept_digest": replies[0]["digest"],
        "cached": [reply["cached"] for reply in replies],
    }


def evict_gate(
    endpoint,
    kept,
    other,
    fills,
    reference,
    api_key=None,
    timeout=600.0,
    ready_timeout=30.0,
):
    """The kept turn is asked, with the unrelated turn in between, until it resumes
    from cache (at most MAX_WARM_ATTEMPTS times). Each fill is then sent twice and
    the kept turn asked again."""
    need(len(fills) <= MAX_FILLS, f"at most {MAX_FILLS} fills")
    need(reference.get("gate") == "evict-reference", "not an evict reference")
    observed = []
    resumed = None
    for _ in range(MAX_WARM_ATTEMPTS):
        reply = _ask(endpoint, kept, api_key, timeout)
        observed.append(reply)
        _ask(endpoint, other, api_key, timeout)
        if reply["cached"]:
            resumed = reply
            break
    for fill in fills:
        for _ in range(2):
            _ask(endpoint, fill, api_key, timeout)
    _ask(endpoint, other, api_key, timeout)
    after = _ask(endpoint, kept, api_key, timeout)
    observed.append(after)
    ready, last = wait_ready(endpoint, ready_timeout)
    checks = []
    if resumed is None:
        checks.append(
            _not_evaluated("resumed_from_cache", "the kept turn never resumed tokens")
        )
    else:
        checks.append(_check("resumed_from_cache", True, resumed["cached"]))
        checks.append(
            _check(
                "evicted",
                (after["cached"] or 0) < resumed["cached"],
                {"resumed": resumed["cached"], "after": after["cached"]},
            )
        )
    checks.append(
        _check(
            "replies_equal_cache_off",
            all(reply["digest"] == reference["kept_digest"] for reply in observed),
        )
    )
    checks.append(_check("health_ready", ready, list(last)))
    return {
        "schema": SCHEMA,
        "gate": "evict",
        "verdict": verdict(checks),
        "checks": checks,
        "fills": len(fills),
        "cached": [reply["cached"] for reply in observed],
    }


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write(path, value):
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-file")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", required=True, help="receipt JSON, outside Git")
    sub = parser.add_subparsers(dest="mode", required=True)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("--body-file", required=True)
    cancel.add_argument("--rounds", type=int, default=2)
    cancel.add_argument("--cut-after", type=int, default=4)
    cancel.add_argument("--exchange", help="capture trace from runtime_checks.py")
    for name in ("evict-reference", "evict"):
        mode = sub.add_parser(name)
        mode.add_argument("--kept-file", required=True)
        mode.add_argument("--other-file", required=True)
        if name == "evict":
            mode.add_argument("--fill-file", action="append", default=[])
            mode.add_argument("--reference", required=True)
    args = parser.parse_args(argv)
    api_key = None
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    else:
        api_key = collector.os.environ.get("TOKSPUR_API_KEY")
    if args.mode == "cancel":
        result = cancel_gate(
            args.endpoint,
            _load(args.body_file),
            args.rounds,
            args.cut_after,
            api_key,
            args.exchange,
            timeout=args.timeout,
        )
    elif args.mode == "evict-reference":
        result = evict_reference(
            args.endpoint,
            _load(args.kept_file),
            _load(args.other_file),
            api_key,
            args.timeout,
        )
    else:
        result = evict_gate(
            args.endpoint,
            _load(args.kept_file),
            _load(args.other_file),
            [_load(path) for path in args.fill_file],
            _load(args.reference),
            api_key,
            args.timeout,
        )
    _write(args.out, result)
    json.dump({"verdict": result.get("verdict")}, sys.stdout)
    sys.stdout.write("\n")
    return 0 if result.get("verdict") in ("pass", None) else 1


if __name__ == "__main__":
    sys.exit(main())
