"""Actual extracted branch8 broker admission, without Metal or model assets."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = r"""
void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class F> void refuse(F&& f) { bool failed=false; try { f(); } catch (const FormatError&) { failed=true; }
    check(failed,"unsafe actual branch8 broker command admitted"); }
Json wire(Json body,uint64_t command,uint64_t lease,uint32_t prefix,uint32_t width=8) {
    auto fields=body.object(); fields.emplace_back("broker_epoch",Json::integer(77)); fields.emplace_back("command_id",Json::integer(command));
    fields.emplace_back("lease_id",Json::integer(lease)); fields.emplace_back("expected_prefix",Json::integer(prefix));
    fields.emplace_back("broker_protocol",Json("branch8-v1")); fields.emplace_back("broker_batch_width",Json::integer(width)); return Json(std::move(fields));
}
Json input(uint32_t n) { Json::Array ids; for (uint32_t i=0;i<n;++i) ids.push_back(Json::integer(i));
    return Json(Json::Object{{"tokens",Json(std::move(ids))},{"generate",Json::integer(0)},{"reset",Json(false)}}); }
Json output(uint32_t n) { return Json(Json::Object{{"tokens",Json(Json::Array{})},{"generate",Json::integer(n)},{"reset",Json(false)}}); }
Json actual_ready_seed(TargetBroker* broker) {
    // The two broker fields below are extracted from the actual target startup
    // producer, not a hand-authored synthetic branch8 ready/snapshot packet.
    return Json(Json::Object{{"event",Json("ready")}, /* ACTUAL READY BROKER FIELDS */});
}
struct EmptyResult { std::vector<float> logits; };
struct EmptyHandle { uint32_t slot=0; };
void initial_readiness() {
    TargetSampler sampler; CommittedResult<EmptyResult> retained;
    BranchSession<EmptyHandle,EmptyResult> branches(4,4);
    TargetLookup<EmptyResult> lookup(4,2,2,8ull<<20,7);
    // Witness the real constructor states used by branch8_snapshot. Readiness
    // needs no model step, extra reset, invented logits or new lookup epoch.
    check(sampler.matches(0) && sampler.valid() && !sampler.prefix() &&
        branches.matches(0) && !branches.retained_count() && lookup.matches(0) &&
        lookup.epoch()==1 && lookup.ledger()==LookupLedger{} && !retained.matches(0) &&
        !retained.continuable(0) && !retained.stop_id() && !sampler.random().words_consumed(),
        "actual initial branch8 coordinators are not coherent and empty");
    TargetBroker broker(77,true);
    const auto before=lookup.ledger(); const auto epoch=lookup.epoch();
    const auto text=broker.decorate(actual_ready_seed(&broker)).dump();
    const auto ready=parse_json(text);
    check(ready.object().size()==7 &&
        std::count_if(ready.object().begin(),ready.object().end(),[](const auto& item){return item.first=="broker_epoch";})==1 &&
        ready.find("checkpoint_broker")->boolean() && ready.find("broker_epoch")->unsigned_integer()==77 &&
        !ready.find("command_id")->unsigned_integer() && !ready.find("lease_id")->unsigned_integer() &&
        ready.find("broker_protocol")->string()=="branch8-v1" && ready.find("broker_batch_width")->unsigned_integer()==8,
        "actual readiness decoration did not serialize one authoritative zero-command epoch");
    check(sampler.matches(0) && branches.matches(0) && lookup.matches(0) && lookup.epoch()==epoch &&
        lookup.ledger()==before && !retained.matches(0) && !branches.retained_count(),
        "readiness decoration changed the actual empty session or lifetime lookup cost");
    refuse([&]{static_cast<void>(broker.decorate(parse_json(R"({"event":"ready","checkpoint_broker":true,"broker_epoch":78})")));});
    refuse([&]{static_cast<void>(broker.decorate(parse_json(R"({"event":"ready","checkpoint_broker":false,"broker_epoch":77})")));});
    refuse([]{static_cast<void>(Json(Json::Object{{"broker_epoch",Json::integer(77)},
        {"broker_epoch",Json::integer(77)}}).dump());}); // Global duplicate-output rejection remains strict.
    refuse([]{static_cast<void>(parse_json(R"({"broker_epoch":77,"broker_epoch":77})"));});
    refuse([&]{static_cast<void>(broker.admit(wire(input(1),1,1,0),0));}); // Readiness is not a lease grant.
    const auto begin=parse_json(R"({"op":"branch_begin","prompt_ids":[1],"generation_requested":true,"allow_restore":false,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":0})");
    const auto ticket=broker.admit(wire(begin,1,1,0),0); broker.started(ticket); broker.published(ticket);
    refuse([&]{static_cast<void>(broker.decorate(actual_ready_seed(&broker)));}); // A later reply cannot impersonate initial readiness.
    const auto ordinary=actual_ready_seed(nullptr);
    check(!ordinary.find("checkpoint_broker")->boolean() && !ordinary.find("broker_epoch")->unsigned_integer(),
        "ordinary startup broker fields changed");
}
void reject_fresh_model_resets(TargetBroker& broker,uint64_t command,uint64_t fresh_lease,uint32_t prefix) {
    // The old fresh-lease condition is satisfied. These cases must fail from
    // the new reset/selector boundary, not an unrelated stale lease.
    for (const auto body : {
        R"({"tokens":[1],"generate":0,"reset":true})",
        R"({"tokens":[1],"generate":0,"reset":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":0})",
        R"({"tokens":[1],"generate":0,"reset":true,"prompt_lookup":true})",
        R"({"tokens":[1],"generate":0,"reset":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":0,"prompt_lookup":true})"})
        refuse([&]{static_cast<void>(broker.admit(wire(parse_json(body),command,fresh_lease,prefix),prefix));});
}
void run() {
    initial_readiness();
    const auto begin=parse_json(R"({"op":"branch_begin","prompt_ids":[1,2,3],"generation_requested":true,"allow_restore":false,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":0})");
    const auto warm=parse_json(R"({"op":"branch_begin","prompt_ids":[1,2,3],"generation_requested":true,"allow_restore":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":0})");
    const auto capture=parse_json(R"({"op":"branch_capture"})");
    TargetBroker old(77); refuse([&]{static_cast<void>(old.admit(wire(begin,1,1,0),0));});
    const auto old_reset=parse_json(R"({"tokens":[1],"generate":0,"reset":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":0,"broker_epoch":77,"command_id":1,"lease_id":1,"expected_prefix":0})");
    check(old.admit(old_reset,0).fresh,"new build changed the old broker's explicit fresh model reset");
    TargetBroker broker(77,true); reject_fresh_model_resets(broker,1,1,0); // Valid new lease before the first control.
    refuse([&]{static_cast<void>(broker.admit(wire(warm,1,1,0),0));});
    for (uint32_t width : {0u,1u,2u,7u,9u}) refuse([&]{static_cast<void>(broker.admit(wire(begin,1,1,0,width),0));});
    auto ticket=broker.admit(wire(begin,1,1,0),0); broker.started(ticket); broker.published(ticket);
    reject_fresh_model_resets(broker,2,2,0); // Valid strictly newer lease after an established session.
    for (const auto body : {
        R"({"tokens":[1],"generate":0,"reset":false,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0}})",
        R"({"tokens":[1],"generate":0,"reset":false,"seed":0})",
        R"({"tokens":[1],"generate":0,"reset":false,"prompt_lookup":true})"})
        refuse([&]{static_cast<void>(broker.admit(wire(parse_json(body),2,1,0),0));}); // Existing lease is valid for append.
    uint64_t command=1; uint32_t prefix=0;
    for (uint32_t n=1;n<=8;++n) {
        ticket=broker.admit(wire(input(n),++command,1,prefix),prefix);
        // Feed the actual stripped broker payload into the actual Request parser
        // under the fixed prepared branch8 policy, not just a symbol reference.
        const auto parsed=request(ticket.body,prefix,16,3,false,prefix!=0,prefix!=0,true,
            0,false,false,8,false,true,4,true,true,true);
        check(ticket.body.object().size()==3 && parsed.tokens.size()==n && parsed.stops.empty() && !parsed.generate && !parsed.reset &&
            parsed.small_splits==16 && parsed.block_formats==3 && !parsed.execution && parsed.known_prefill && parsed.residual_reuse &&
            !parsed.branch && parsed.lookup && !parsed.sampling.policy.temperature && !parsed.sampling.policy.top_k &&
            parsed.sampling.policy.top_p==1 && !parsed.sampling.policy.min_p && !parsed.sampling.seed,
            "valid branch8 input payload changed during actual request parsing");
        for (uint32_t i=0;i<n;++i) check(parsed.tokens[i]==i,"broker identity became an input token or changed its order");
        broker.started(ticket); broker.published(ticket); prefix+=n; broker.input();
        ticket=broker.admit(wire(capture,++command,1,prefix),prefix); broker.started(ticket); broker.published(ticket); broker.captured();
        refuse([&]{static_cast<void>(broker.admit(wire(capture,command+1,1,prefix),prefix));});
        ticket=broker.admit(wire(output(n),++command,1,prefix),prefix); broker.started(ticket); broker.published(ticket); prefix+=n; broker.generated();
        refuse([&]{static_cast<void>(broker.admit(wire(capture,command+1,1,prefix),prefix));});
    }
    for (const auto& bad : {input(0),input(9),output(0),output(9),parse_json(R"({"tokens":[1],"generate":1,"reset":false})"),
        parse_json(R"({"tokens":[248320],"generate":0,"reset":false})"),parse_json(R"({"tokens":[1],"generate":0,"reset":false,"unknown":true})")})
        refuse([&]{static_cast<void>(broker.admit(wire(bad,command+1,1,prefix),prefix));});
    refuse([&]{static_cast<void>(broker.admit(wire(input(8),command,1,prefix),prefix));});
    refuse([&]{static_cast<void>(broker.admit(wire(input(8),command+1,2,prefix),prefix));});
    refuse([&]{static_cast<void>(broker.admit(wire(input(8),command+1,1,prefix+1),prefix));});
    broker.revoke(); refuse([&]{static_cast<void>(broker.admit(wire(warm,command+1,2,prefix),prefix));});
    refuse([&]{static_cast<void>(broker.admit(wire(input(8),command+1,1,prefix),prefix));});
    ticket=broker.admit(wire(begin,++command,2,prefix),prefix); broker.started(ticket); broker.published(ticket);
    const auto decorated=broker.decorate(Json(Json::Object{{"event",Json("control_done")}}));
    check(decorated.find("broker_protocol")->string()=="branch8-v1" && decorated.find("broker_batch_width")->unsigned_integer()==8 &&
        decorated.find("command_id")->unsigned_integer()==command && decorated.find("lease_id")->unsigned_integer()==2,"new identity not decorated authoritatively");
    lookup_preparation(true,128ull<<20,0,false,16,3,false,false,false,true,true,true);
    refuse([]{lookup_preparation(true,128ull<<20,0,false,16,3,false,false,false,true,true);});
    for (uint32_t flag=0;flag<8;++flag) refuse([&]{lookup_preparation(true,128ull<<20,flag==0,flag==1,flag==2?0:16,flag==3?0:3,
        flag==4,flag==5,flag==6,true,true,flag!=7);});
    for (uint32_t slots : {0u,1u,4u,5u}) for (uint32_t cap : {0u,8u}) for (bool history : {false,true}) for (bool consent : {false,true}) {
        const bool expected=slots && slots<=4 && cap==8 && history && consent;
        check(hc_exact_startup_allowed(HcExactMode::combined,0,false,16,3,0,cap,false,false,false,true,slots,history,consent)==expected,
            "new HC broker consent disagrees with independent table");
    }
}
"""


def test_actual_branch8_broker_admission(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "branch8_protocol_extract", ROOT / "scripts/check_owned_target_protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    source = (ROOT / "native/src/target.cpp").read_text()
    ready_start = source.index('{"checkpoint_broker", Json(bool(broker))}')
    ready_end = source.index('{"prompt_lookup",', ready_start)
    ready_fields = source[ready_start:ready_end].rstrip(" ,\n")
    assert ready_fields == (
        '{"checkpoint_broker", Json(bool(broker))}, '
        '{"broker_epoch", Json::integer(broker ? broker->epoch() : 0)}'
    )
    unit = (
        '#include "tokspur/json.hpp"\n#include "tokspur/gguf.hpp"\n'
        '#include "tokspur/sampling.hpp"\n#include "tokspur/prompt_lookup.hpp"\n'
        '#include "tokspur/graph.hpp"\n'
        "#include <algorithm>\n#include <array>\n#include <charconv>\n#include <cmath>\n"
        "#include <initializer_list>\n#include <iostream>\n#include <limits>\n#include <optional>\n"
        "#include <set>\n#include <span>\n#include <type_traits>\n#include <utility>\n#include <vector>\n"
        "namespace tokspur { namespace {\n"
        + protocol.extract(source)
        + CHECKS.replace("/* ACTUAL READY BROKER FIELDS */", ready_fields)
        + "\n} }\nint main() { try { tokspur::run(); return 0; } "
        + "catch (const std::exception& error) { std::cerr << error.what() << '\\n'; return 1; } }\n"
    )
    path, executable = tmp_path / "branch8 protocol.cpp", tmp_path / "branch8 protocol"
    path.write_text(unit)
    compiled = subprocess.run(
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
            "-DTOKSPUR_PRIVATE_HTTP_BRANCH8=1",
            "-DTOKSPUR_PRIVATE_VERIFY8_BRANCH=1",
            "-DTOKSPUR_PRIVATE_HTTP_VERIFY8=1",
            "-DTOKSPUR_PRIVATE_VERIFY_EIGHT=1",
            "-DTOKSPUR_PRIVATE_HC_EXACT=1",
            "-I" + str(ROOT / "native/include"),
            str(path),
            *[
                str(ROOT / ("native/src/" + name + ".cpp"))
                for name in ("json", "unicode", "sampling", "prompt_lookup")
            ],
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compiled.returncode == 0, (
        f"actual branch8 protocol compilation exited {compiled.returncode}\n"
        f"stdout (last 16 KiB):\n{compiled.stdout[-16384:]}\n"
        f"stderr (last 16 KiB):\n{compiled.stderr[-16384:]}"
    )
    executed = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=30
    )
    assert executed.returncode == 0, (
        f"actual branch8 protocol execution exited {executed.returncode}\n"
        f"stdout (last 16 KiB):\n{executed.stdout[-16384:]}\n"
        f"stderr (last 16 KiB):\n{executed.stderr[-16384:]}"
    )
