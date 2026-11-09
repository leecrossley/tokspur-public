"""CPU-only checks of the actual branch coordinator; no Metal/model evidence."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = r"""
struct FakeResult { std::vector<float> logits; };
struct FakeHandle { uint32_t slot = 0; uint64_t generation = 0; };
static_assert(std::is_trivially_copyable_v<FakeHandle>);
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void refuses(F&& action) {
    bool refused = false;
    try { action(); } catch (const FormatError&) { refused = true; }
    check(refused, "invalid branch operation admitted");
}
struct Native {
    struct Slot { uint32_t prefix = 0; uint64_t generation = 0; bool valid = false; };
    std::vector<Slot> slots;
    uint32_t prefix = 0;
    uint64_t generation = 0;
    size_t retains = 0, restores = 0;
    explicit Native(uint32_t count) : slots(count) {}
    FakeHandle retain() {
        ++retains; size_t index = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            if (!slots[i].valid) { index = i; break; }
            if (slots[i].generation < slots[index].generation) index = i;
        }
        slots[index] = {prefix, ++generation, true};
        return {uint32_t(index), generation};
    }
    uint32_t restore(FakeHandle handle) {
        ++restores;
        require(handle.slot < slots.size(), "foreign fake handle");
        const auto saved = slots[handle.slot];
        require(saved.valid && saved.generation == handle.generation && saved.prefix <= prefix, "stale fake handle");
        for (auto& slot : slots) if (slot.valid && slot.prefix > saved.prefix) slot.valid = false;
        prefix = saved.prefix; return prefix;
    }
};
struct Fixture {
    BranchSession<FakeHandle, FakeResult> branch;
    CommittedResult<FakeResult> retained;
    TargetSampler sampler;
    Native native;
    explicit Fixture(uint32_t count = 4) : branch(count, 4), native(count) {}
    void append(std::initializer_list<uint32_t> tokens) {
        const auto before = native.prefix;
        native.prefix += uint32_t(tokens.size());
        retained.commit(FakeResult{{float(native.prefix), 8, 8, 0}}, native.prefix);
        for (uint32_t p = before + 1; p <= native.prefix; ++p)
            check(sampler.committed_input(p), "input sampler publication failed");
        check(branch.committed(native.prefix, std::span(tokens.begin(), tokens.size())), "token history publication failed");
    }
    uint32_t generate() {
        return sampled_step(sampler, retained.get(native.prefix).logits, native.prefix,
            [&](uint32_t id) { append({id}); });
    }
    FakeHandle capture() {
        FakeHandle handle;
        branch.capture(native.prefix, retained, [&] { handle = native.retain(); return handle; });
        return handle;
    }
    std::optional<size_t> select(std::initializer_list<uint32_t> prompt, uint32_t generate = 0) const {
        return branch.select(std::span(prompt.begin(), prompt.size()), generate);
    }
    void restore(size_t index, TargetSampling options = {}) {
        check(branch.restore(index, retained, sampler, options, [&](FakeHandle handle) {
            return native.restore(handle);
        }) == native.prefix, "restore prefix differs");
    }
};
void tracking() {
    Fixture f;
    const auto charged = f.branch.charged_bytes();
    check(charged >= decltype(f.branch)::minimum_bytes(4, 4), "branch ledger omits payload");
    f.append({10, 11}); const auto two = f.capture(); const auto saved_two = f.retained.get(2).logits;
    f.append({12}); const auto three = f.capture();
    check(f.generate() == 1 && f.sampler.random().words_consumed() == 0, "greedy generated token changed");
    const auto four = f.capture(); const auto saved_four = f.retained.get(4).logits;
    f.append({14}); const auto five = f.capture();
    check(f.select({10, 11, 12, 1, 99}, 1) == four.slot, "generated token omitted from exact LCP");
    check(f.select({10, 11, 99}, 1) == two.slot && !f.select({10, 99}), "deep-edit LCP selection differs");
    f.restore(four.slot, {{0, 0, 1, 0}, 55});
    check(f.retained.get(4).logits == saved_four && f.branch.matches(4) && f.sampler.matches(4) &&
        f.sampler.random().seed() == 55 && f.sampler.random().words_consumed() == 0, "saved logits/fresh sampler differ");
    f.restore(four.slot); f.append({99});
    check(f.select({10, 11, 12, 1, 14}) == four.slot, "old descendant survived branch overwrite");
    refuses([&] { f.native.restore(five); });
    refuses([&] { f.restore(five.slot); });
    check(f.select({10, 11}) == two.slot, "shortened prompt missed ancestor");
    f.restore(two.slot);
    check(f.retained.get(2).logits == saved_two && f.branch.retained_count() == 1, "ancestor payload/descendants differ");
    refuses([&] { f.native.restore(three); });
    f.append({42, 43}); const auto edited = f.capture();
    check(f.select({10, 11, 42, 43}) == edited.slot && f.select({10, 11, 12}) == two.slot,
        "restored token suffix was not overwritten");
    check(f.branch.charged_bytes() == charged, "capture/restore changed reserved host payload charge");
}
void terminal_and_duplicates() {
    Fixture f;
    f.append({1, 2}); f.retained.stopped(0); const auto terminal = f.capture();
    f.append({3}); f.capture();
    check(f.retained.continuable(3), "new input retained old terminal metadata");
    const auto current = f.retained.get(3).logits;
    refuses([&] { f.select({1, 2}, 1); });
    check(f.native.prefix == 3 && f.retained.get(3).logits == current, "terminal admission mutated session");
    check(f.select({1, 2}, 0) == terminal.slot, "terminal no-work query lost checkpoint");
    f.restore(terminal.slot); f.restore(terminal.slot);
    check(f.retained.stop_id() == 0 && !f.retained.continuable(2), "terminal zero was not restored");
    check(f.select({1, 2, 4}, 1) == terminal.slot, "new suffix refused after terminal");
    f.append({4});
    check(f.retained.continuable(3) && !f.retained.stop_id() && f.generate() == 1,
        "new suffix did not reopen greedy generation");

    Fixture fifo(2);
    fifo.append({7}); const auto oldest = fifo.capture(); const auto duplicate = fifo.capture();
    check(fifo.select({7}) == duplicate.slot, "duplicate-prefix tie did not choose newest capture");
    fifo.append({8}); const auto descendant = fifo.capture();
    refuses([&] { fifo.native.restore(oldest); });
    check(fifo.select({7}) == duplicate.slot, "FIFO evicted newer equal-prefix record");
    fifo.restore(duplicate.slot);
    refuses([&] { fifo.restore(descendant.slot); });
    const auto equal = fifo.capture();
    check(fifo.select({7}) == equal.slot && fifo.branch.retained_count() == 2, "equal-prefix recapture lost surviving handle");
    fifo.restore(duplicate.slot); fifo.restore(equal.slot);
    check(fifo.branch.retained_count() == 2, "equal-prefix restores invalidated peers");
}
void refusals_and_bounds() {
    Fixture f(2);
    f.append({7, 8}); const auto saved = f.capture();
    f.append({9}); f.capture();
    const auto before = f.retained.get(3).logits; const auto random = f.sampler.random();
    const auto charged = f.branch.charged_bytes(); size_t callbacks = 0;
    refuses([&] { f.branch.capture(3, f.retained, [&]() -> FakeHandle {
        ++callbacks; throw FormatError("retain refused before publication");
    }); });
    refuses([&] { f.branch.restore(saved.slot, f.retained, f.sampler, {}, [&](FakeHandle) -> uint32_t {
        ++callbacks; throw FormatError("restore refused before publication");
    }); });
    check(callbacks == 2 && f.native.prefix == 3 && f.branch.matches(3) && f.branch.retained_count() == 2 &&
        f.retained.get(3).logits == before && f.sampler.matches(3) && f.sampler.random() == random,
        "precommit refusal changed current or saved session");
    refuses([&] { f.branch.restore(saved.slot, f.retained, f.sampler, {{0.5, 0, 1, 0}, 99}, [&](FakeHandle) {
        ++callbacks; return uint32_t(2);
    }); });
    refuses([&] { f.branch.restore(99, f.retained, f.sampler, {}, [&](FakeHandle) {
        ++callbacks; return uint32_t(2);
    }); });
    check(callbacks == 2 && f.retained.get(3).logits == before, "invalid restore reached native mutation");
    f.restore(saved.slot);
    check(f.branch.matches(2) && f.retained.get(2).logits[0] == 2 && f.branch.charged_bytes() == charged,
        "failed retention overwrote older saved payload");
    refuses([&] { f.branch.restore(saved.slot, f.retained, f.sampler, {}, [&](FakeHandle) {
        f.native.prefix = 3; return uint32_t(3);
    }); });
    check(!f.branch.matches(2) && !f.branch.matches(3) && !f.branch.retained_count() &&
        !f.retained.matches(2) && !f.sampler.valid() && !f.select({7, 8}), "wrong native prefix did not invalidate session");
    f.branch.reset(); check(f.branch.matches(0), "fresh reset did not reopen tracking");
    check(!f.branch.committed(2, std::array<uint32_t, 1>{1}) && !f.branch.matches(0), "mismatched tracking prefix admitted");
    f.branch.reset(); f.branch.invalidate();
    check(!f.branch.committed(1, std::array<uint32_t, 1>{1}), "invalid tracking accepted appended token");

    Fixture bad_logits(1); bad_logits.append({1}); const auto finite = bad_logits.capture();
    const auto native_retains = bad_logits.native.retains;
    bad_logits.retained.commit(FakeResult{{std::numeric_limits<float>::infinity(), 8, 8, 0}}, 1);
    refuses([&] { bad_logits.capture(); });
    const auto native_restores = bad_logits.native.restores;
    refuses([&] { bad_logits.restore(finite.slot); });
    check(bad_logits.native.restores == native_restores, "nonfinite current payload reached native restoration");
    bad_logits.retained.commit(FakeResult{{1}}, 1);
    refuses([&] { bad_logits.capture(); });
    check(bad_logits.native.retains == native_retains && bad_logits.branch.retained_count() == 1,
        "malformed logits reached native retention or lost saved record");
    refuses([&] { bad_logits.restore(finite.slot); });
    check(bad_logits.native.restores == native_restores, "bad current payload reached native restoration");
    bad_logits.retained.commit(FakeResult{{1, 8, 8, 0}}, 1);
    bad_logits.restore(finite.slot);
    check(bad_logits.retained.get(1).logits == std::vector<float>({1, 8, 8, 0}), "logit refusal damaged saved payload");

    BranchSession<FakeHandle, FakeResult> bounded(1, 4);
    std::vector<uint32_t> full(65536, 1);
    check(bounded.committed(65536, full) && bounded.matches(65536), "full context history refused");
    check(!bounded.committed(65537, std::array<uint32_t, 1>{1}) && !bounded.matches(65536), "history overflow did not invalidate");
    for (uint32_t slots : {0, 5}) refuses([&] { BranchSession<FakeHandle, FakeResult> bad(slots, 4); });
    for (uint32_t vocabulary : {0U, uint32_t(maximum_sampling_vocabulary + 1)})
        refuses([&] { BranchSession<FakeHandle, FakeResult> bad(1, vocabulary); });
    for (uint32_t prefix : {0, 65537}) refuses([&] { TargetSampler::fresh_branch({}, prefix); });
    check(TargetSampler::fresh_branch({}, 65536).matches(65536), "last bounded sampler prefix refused");
}
Request parse_branch(std::string_view body, uint32_t slots = 2, bool entropy = true) {
    return request(parse_json(body), 7, 16, 3, false, true, true, entropy,
        0, false, false, 0, false, false, slots);
}
void admission() {
    branch_preparation(0, 0, 2, true, true, true, true);
    branch_preparation(4, 512ull << 20, 0, false, false, false, false);
    for (uint32_t slots : {1, 4}) branch_preparation(slots, 1, 0, false, false, false, false);
    refuses([] { branch_preparation(5, 1, 0, false, false, false, false); });
    refuses([] { branch_preparation(1, 0, 0, false, false, false, false); });
    refuses([] { branch_preparation(0, 1, 0, false, false, false, false); });
    refuses([] { branch_preparation(1, (512ull << 20) + 1, 0, false, false, false, false); });
    for (uint32_t execution : {1, 2}) refuses([&] { branch_preparation(1, 1, execution, false, false, false, false); });
    for (uint32_t flag = 0; flag < 4; ++flag)
        refuses([&] { branch_preparation(1, 1, 0, flag == 0, flag == 1, flag == 2, flag == 3); });
    const auto branch = R"({"tokens":[1,2],"generate":1,"reset":true,"checkpoint_mode":"branch"})";
    check(parse_branch(branch).branch && parse_branch(branch).tokens == std::vector<uint32_t>({1, 2}), "full greedy branch request refused");
    check(parse_branch(branch, 2, false).branch, "fresh branch admission depended on previous entropy");
    refuses([&] { parse_branch(branch, 0); }); refuses([&] { parse_branch(branch, 5); });
    check(!parse_branch(R"({"tokens":[1],"generate":0,"reset":false})").branch, "ordinary append silently became branch");
    check(!parse_branch(R"({"tokens":[1],"generate":0,"reset":true})", 0).branch, "disabled default changed");
    check(!parse_branch(R"({"tokens":[],"generate":0,"reset":false})").branch, "ordinary no-work query refused");
    check(parse_branch(R"({"tokens":[1],"generate":0,"reset":true,"sampling":{"temperature":0.5,"top_k":0,"top_p":1,"min_p":0},"seed":3})").sampling.policy.temperature == 0.5,
        "ordinary fresh sampling policy changed under checkpoint preparation");
    refuses([&] { parse_branch(R"({"tokens":[1],"generate":0,"reset":false})", 2, false); });
    for (const auto body : {
        R"({"tokens":[1],"generate":0,"reset":false,"checkpoint_mode":"branch"})",
        R"({"tokens":[],"generate":0,"reset":true,"checkpoint_mode":"branch"})",
        R"({"tokens":[1],"generate":4097,"reset":true,"checkpoint_mode":"branch"})",
        R"({"tokens":[248320],"generate":0,"reset":true,"checkpoint_mode":"branch"})",
        R"({"tokens":[1],"generate":0,"reset":true,"checkpoint_mode":"branch","sampling":{"temperature":0.5,"top_k":0,"top_p":1,"min_p":0},"seed":3})"})
        refuses([&] { parse_branch(body); });
    for (const auto mode : {"null", "true", "1", "[]", "{}", "\"off\""}) {
        const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":true,"checkpoint_mode":)") + mode + "}";
        refuses([&] { parse_branch(body); });
    }
    for (const auto override_value : {"\"small_gemv_splits\":16", "\"block_gemv_formats\":3", "\"stack_execution\":\"direct\"",
        "\"known_prefill\":false", "\"known_residual_reuse\":false"}) {
        const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":true,)") + override_value + "}";
        refuses([&] { parse_branch(body); });
    }
}
Json broker_wire(Json body, uint64_t command, uint64_t lease, uint32_t prefix, uint64_t epoch = 77) {
    auto fields = body.object(); fields.emplace_back("broker_epoch", Json::integer(epoch));
    fields.emplace_back("command_id", Json::integer(command)); fields.emplace_back("lease_id", Json::integer(lease));
    fields.emplace_back("expected_prefix", Json::integer(prefix)); return Json(std::move(fields));
}
void broker_admission() {
    TargetBroker broker(77);
    const auto rejected = [&](const Json& wire, uint32_t prefix) { refuses([&] { static_cast<void>(broker.admit(wire, prefix)); }); };
    const auto begin = parse_json(R"({"op":"branch_begin","prompt_ids":[1,2,3],"generation_requested":true,"allow_restore":false,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":42})");
    const auto step = parse_json(R"({"tokens":[1],"generate":0,"reset":false})");
    const auto capture = parse_json(R"({"op":"branch_capture"})");
    rejected(broker_wire(begin, 1, 1, 0, 78), 0);
    rejected(broker_wire(begin, 2, 1, 0), 0);
    rejected(broker_wire(begin, 1, 1, 1), 0);
    auto command = broker.admit(broker_wire(begin, 1, 1, 0), 0);
    check(command.begin && !command.capture && command.fresh && !command.body.find("tokens"), "future metadata became a model command");
    broker.started(command); broker.published(command);
    rejected(broker_wire(capture, 2, 1, 0), 0);
    rejected(broker_wire(step, 2, 2, 0), 0);
    command = broker.admit(broker_wire(step, 2, 1, 0), 0); broker.started(command); broker.published(command); broker.input();
    command = broker.admit(broker_wire(capture, 3, 1, 1), 1); check(command.capture && !command.fresh, "explicit capture refused");
    broker.started(command); broker.published(command); broker.captured();
    rejected(broker_wire(capture, 4, 1, 1), 1);
    for (const auto body : {
        R"({"tokens":[1,2],"generate":0,"reset":false})",
        R"({"tokens":[1],"generate":1,"reset":false})",
        R"({"tokens":[],"generate":2,"reset":false})",
        R"({"op":"branch_capture","tokens":[1]})",
        R"({"op":"branch_begin","prompt_ids":[1],"tokens":[1],"generation_requested":true,"allow_restore":false,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":42})",
        R"({"op":"branch_begin","prompt_ids":[248320],"generation_requested":true,"allow_restore":false,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":42})"})
        rejected(broker_wire(parse_json(body), 4, 2, 1), 1);
    broker.revoke();
    const auto restore = parse_json(R"({"op":"branch_begin","prompt_ids":[1],"generation_requested":true,"allow_restore":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":42})");
    rejected(broker_wire(restore, 4, 2, 1), 1);
    rejected(broker_wire(step, 4, 1, 1), 1);
    command = broker.admit(broker_wire(begin, 4, 2, 1), 1); broker.started(command); broker.published(command);
    const auto identity = broker.decorate(Json(Json::Object{{"event", Json("control_done")}}));
    check(identity.find("broker_epoch")->unsigned_integer() == 77 && identity.find("command_id")->unsigned_integer() == 4 &&
        identity.find("lease_id")->unsigned_integer() == 2, "recovery reused old broker identity");
}
void run_branches() { tracking(); terminal_and_duplicates(); refusals_and_bounds(); admission(); broker_admission(); }
"""


def test_branch_sessions_execute_actual_owned_cpu_protocol(tmp_path):
    """Compile extracted owned source with fake native commits, without Metal."""
    spec = importlib.util.spec_from_file_location(
        "branch_protocol_extract", ROOT / "scripts/check_owned_target_protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    unit = (
        '#include "tokspur/json.hpp"\n#include "tokspur/gguf.hpp"\n#include "tokspur/sampling.hpp"\n#include "tokspur/prompt_lookup.hpp"\n'
        "#include <algorithm>\n#include <array>\n#include <charconv>\n#include <cmath>\n"
        "#include <initializer_list>\n#include <iostream>\n#include <limits>\n#include <optional>\n"
        "#include <set>\n#include <span>\n#include <type_traits>\n#include <utility>\n#include <vector>\n"
        "namespace tokspur { namespace {\n"
        + protocol.extract((ROOT / "native/src/target.cpp").read_text())
        + CHECKS
        + "\n} }\nint main() { try { tokspur::run_branches(); return 0; } "
        + "catch (const std::exception& error) { std::cerr << error.what() << '\\n'; return 1; } }\n"
    )
    path, executable = tmp_path / "branch protocol.cpp", tmp_path / "branch protocol"
    path.write_text(unit)
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I" + str(ROOT / "native/include"),
            str(path),
            *[
                str(ROOT / ("native/src/" + name + ".cpp"))
                for name in ("json", "unicode", "sampling")
            ],
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
