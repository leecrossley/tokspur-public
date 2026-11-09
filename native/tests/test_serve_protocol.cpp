#include "tokspur/serve.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <source_location>

using namespace tokspur;
namespace {
size_t checks = 0;
void check(bool value, std::source_location at = std::source_location::current()) { ++checks; if (!value) throw std::runtime_error("serving protocol check failed at line " + std::to_string(at.line())); }
double parse_number(const Json& value) {
    check(value.kind() == Json::Kind::number);
    const auto text = value.number_lexeme(); double result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    check(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(result));
    return result;
}
template<class F> void refusal(F&& function, std::source_location at = std::source_location::current()) {
    bool refused = false; try { function(); } catch (const FormatError&) { refused = true; } check(refused, at);
}
Json with(Json input, std::string_view key, Json value) {
    auto fields = input.object();
    for (auto& [name, item] : fields) if (name == key) { item = std::move(value); return Json(std::move(fields)); }
    fields.emplace_back(key, std::move(value)); return Json(std::move(fields));
}
Json without(const Json& input, std::string_view key) {
    Json::Object result; for (const auto& item : input.object()) if (item.first != key) result.push_back(item);
    return Json(std::move(result));
}
Json replay_metadata(Json input) {
    input = with(std::move(input), "replay_prepared", Json(true));
    input = with(std::move(input), "recurrent_replay_captures", Json::integer(24));
    return with(std::move(input), "mapped_weight_region_limit_bytes", Json::integer(uint64_t(4) * 1024 * 1024 * 1024));
}
void worker_ready_admission() {
    const auto ready = parse_json(R"({"event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct","maximum_generated_tokens":4096})");
    validate_worker_ready(ready); ++checks;
    validate_worker_ready(with(ready, "unrelated_diagnostic", Json(true))); ++checks;
    for (const auto& [key, expected] : ready.object()) {
        Json::Object missing;
        for (const auto& item : ready.object()) if (item.first != key) missing.push_back(item);
        refusal([&] { validate_worker_ready(Json(missing)); });
        for (const auto& invalid : {Json(), Json(true), Json(false), Json(Json::Array{}), Json(Json::Object{})})
            refusal([&] { validate_worker_ready(with(ready, key, invalid)); });
        if (expected.kind() == Json::Kind::string) {
            for (const auto& invalid : {Json(""), Json("wrong"), Json::integer(1)})
                refusal([&] { validate_worker_ready(with(ready, key, invalid)); });
        } else {
            const auto n = expected.unsigned_integer();
            for (const auto& invalid : {Json::integer(0), Json::integer(n - 1), Json::integer(n + 1), Json(std::to_string(n)),
                parse_json(std::to_string(n) + ".0"), parse_json(std::to_string(n) + "e0")})
                refusal([&] { validate_worker_ready(with(ready, key, invalid)); });
        }
    }
    for (const auto* mode : {"indirect-compatible", "recurrent-replay", "capture"})
        refusal([&] { validate_worker_ready(with(ready, "stack_execution", Json(mode))); });
    for (const auto& invalid : {Json(), Json(true), Json("ready"), Json(Json::Array{}), Json(Json::Object{})})
        refusal([&] { validate_worker_ready(invalid); });
    // Refusals are pure: a later exact ready record remains admissible.
    validate_worker_ready(ready); ++checks;
    const auto replay = replay_metadata(with(ready, "stack_execution", Json("recurrent-replay")));
    validate_worker_ready(replay, "recurrent-replay"); ++checks;
    refusal([&] { validate_worker_ready(replay); });
    refusal([&] { validate_worker_ready(ready, "recurrent-replay"); });
    for (const auto* mode : {"", "capture", "indirect-compatible", "unknown"}) {
        refusal([&] { validate_worker_ready(replay, mode); });
        refusal([&] { validate_worker_ready(ready, mode); });
    }
    for (const auto* key : {"replay_prepared", "recurrent_replay_captures", "mapped_weight_region_limit_bytes"}) {
        refusal([&] { validate_worker_ready(without(replay, key), "recurrent-replay"); });
        for (const auto& invalid : {Json(), Json("true"), Json(Json::Array{}), Json(Json::Object{}), Json(false)})
            refusal([&] { validate_worker_ready(with(replay, key, invalid), "recurrent-replay"); });
        if (std::string_view(key) == "replay_prepared") {
            refusal([&] { validate_worker_ready(with(replay, key, Json::integer(1)), "recurrent-replay"); });
        } else {
            const auto n = replay.find(key)->unsigned_integer();
            for (const auto& invalid : {Json(true), Json::integer(0), Json::integer(n - 1), Json::integer(n + 1),
                Json(std::to_string(n)), parse_json(std::to_string(n) + ".0"), parse_json(std::to_string(n) + "e0")})
                refusal([&] { validate_worker_ready(with(replay, key, invalid), "recurrent-replay"); });
        }
    }
    validate_worker_ready(replay, "recurrent-replay"); ++checks;
}
std::vector<uint32_t> encode(std::string_view text) {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < text.size();) {
        if (text.substr(i).starts_with("<|im_end|>")) { result.push_back(1000); i += 10; }
        else if (text.substr(i).starts_with("<|endoftext|>")) { result.push_back(1001); i += 13; }
        else if (text.substr(i).starts_with("<|im_start|>")) { result.push_back(1002); i += 12; }
        else result.push_back(static_cast<unsigned char>(text[i++]) + 1U);
    }
    return result;
}
std::string decode(std::span<const uint32_t> ids) {
    std::string result;
    for (const auto id : ids) {
        if (id == 1000) result += "<|im_end|>";
        else if (id == 1001) result += "<|endoftext|>";
        else if (id == 1002) result += "<|im_start|>";
        else { check(id > 0 && id <= 256); result += char(id - 1); }
    }
    return result;
}
ServeCodec codec() {
    return {[](const Json& request) {
        auto text = std::string(request.find("messages")->array().back().find("content")->string());
        for (size_t at = 0; (at = text.find("[fixture_end]", at)) != std::string::npos;) text.replace(at, 13, "<|im_end|>");
        return RenderedChat{std::move(text), Json(Json::Array{}), 0, 0};
    }, [](const RenderedChat& rendered) { return encode(rendered.prompt); }, decode, {4096, 1000, 1001, 1002}};
}
Json request(std::string text = "A", bool stream = false, bool thinking = false) {
    for (size_t at = 0; (at = text.find("<|im_end|>", at)) != std::string::npos;) text.replace(at, 10, "[fixture_end]");
    return Json(Json::Object{{"model", Json("tokspur")}, {"messages", Json(Json::Array{Json(Json::Object{{"role", Json("user")}, {"content", Json(std::move(text))}})})},
        {"max_tokens", Json::integer(256)}, {"stream", Json(stream)}, {"enable_thinking", Json(thinking)}});
}
Json tools() {
    return parse_json(R"([{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}])");
}
struct Fake {
    uint32_t position = 0;
    uint64_t seed = 0;
    Json sampling;
    std::vector<uint32_t> output;
    size_t next = 0, resets = 0, prefills = 0, generations = 0, maximum_generation = 0;
    std::string corrupt, stack_execution = "direct";
    void reply(std::string_view text) { output = encode(text); next = 0; }
    Json operator()(const Json& job, const std::function<void(const Json&)>& event) {
        if (job.find("reset")->boolean()) { position = 0; ++resets; sampling = *job.find("sampling"); seed = job.find("seed")->unsigned_integer(); }
        const auto budget = job.find("generate")->unsigned_integer();
        check(budget <= 4096); maximum_generation = std::max(maximum_generation, size_t(budget));
        const bool generation = budget != 0;
        check(!generation || budget == 1); // The ordinary serving scheduler remains packet-bounded.
        check(job.find("tokens")->array().size() == (generation ? 0U : 1U));
        const bool stopped = generation && (output.at(next) == 1000 || output.at(next) == 1001);
        Json::Array ids;
        ++position;
        if (generation) {
            ++generations; const auto id = output.at(next++); ids.push_back(Json::integer(id));
            event(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)}, {"committed_prefix", Json::integer(position)}}));
        } else ++prefills;
        auto policy = sampling.object(); policy.emplace_back("valid", Json(true)); policy.emplace_back("entropy_prefix", Json::integer(position)); policy.emplace_back("seed", Json::integer(seed));
        Json done(Json::Object{{"event", Json("done")}, {"committed_prefix", Json::integer(position)}, {"generated_ids", Json(std::move(ids))},
            {"status", Json(generation ? stopped ? "stopped" : "length" : "prefilled")}, {"finish_reason", generation ? Json(stopped ? "stop" : "length") : Json()},
            {"stop_id", stopped ? Json::integer(output[next - 1]) : Json()}, {"continuation_allowed", Json(!stopped)}, {"sampling", Json(std::move(policy))},
            {"small_gemv_column_splits", Json::integer(16)}, {"block_gemv_formats", Json::integer(3)}, {"stack_execution", Json(stack_execution)}});
        if (stack_execution == "recurrent-replay") done = replay_metadata(std::move(done));
        if (generation && corrupt == "prefix") done = with(std::move(done), "committed_prefix", Json::integer(position + 1));
        if (generation && corrupt == "policy") done = with(std::move(done), "block_gemv_formats", Json::integer(1));
        if (generation && corrupt == "execution") done = with(std::move(done), "stack_execution", Json(stack_execution == "direct" ? "recurrent-replay" : "direct"));
        if (generation && corrupt == "replay-prepared") done = with(std::move(done), "replay_prepared", Json(false));
        if (generation && corrupt == "replay-captures") done = with(std::move(done), "recurrent_replay_captures", Json::integer(23));
        if (generation && corrupt == "replay-region") done = with(std::move(done), "mapped_weight_region_limit_bytes", Json::integer(0));
        if (generation && corrupt == "replay-missing") done = without(done, "replay_prepared");
        if (stopped && corrupt == "terminal") done = with(std::move(done), "stop_id", Json::integer(1001));
        if (generation && corrupt == "throw") throw FormatError("fixture final failure after token");
        return done;
    }
};
Json broker_metadata(Json value) {
    value = with(std::move(value), "checkpoint_slots", Json::integer(2));
    value = with(std::move(value), "checkpoint_combined_budget_bytes", Json::integer(1 << 20));
    value = with(std::move(value), "checkpoint_host_bytes", Json::integer(512));
    value = with(std::move(value), "checkpoint_store_host_bytes", Json::integer(256));
    return with(std::move(value), "checkpoint_gpu_bytes", Json::integer(1024));
}
Json broker_ready() {
    return broker_metadata(parse_json(R"({"event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct","maximum_generated_tokens":4096,"checkpoint_broker":true,"broker_epoch":77})"));
}
struct BrokerFake : Fake {
    struct Saved { size_t prefix = 0, order = 0; bool valid = false; };
    std::array<Saved, 2> saved{};
    std::vector<uint32_t> history;
    uint64_t command = 0, lease = 0;
    size_t order = 0, begins = 0, captures = 0, hits = 0;
    bool last_restore = false, fail_suffix = false;
    std::string control_corrupt;
    Json stamp(Json value) const {
        value = broker_metadata(std::move(value));
        value = with(std::move(value), "checkpoint_retained", Json::integer(size_t(saved[0].valid) + size_t(saved[1].valid)));
        value = with(std::move(value), "broker_epoch", Json::integer(control_corrupt == "epoch" ? 78 : 77));
        value = with(std::move(value), "command_id", Json::integer(command));
        if (control_corrupt == "storage") value = with(std::move(value), "checkpoint_gpu_bytes", Json::integer(1025));
        return with(std::move(value), "lease_id", Json::integer(control_corrupt == "lease" ? lease - 1 : lease));
    }
    Json operator()(const Json& job, const std::function<void(const Json&)>& event) {
        ++command;
        if (const auto* op = job.find("op")) {
            check(!job.find("tokens") && !job.find("generate"));
            const bool begin = op->string() == "branch_begin"; std::string outcome = "captured"; size_t common = position;
            if (begin) {
                ++begins; ++lease; last_restore = job.find("allow_restore")->boolean();
                const auto& future = job.find("prompt_ids")->array(); common = 0;
                if (last_restore) while (common < future.size() && common < history.size() && history[common] == future[common].unsigned_integer()) ++common;
                std::optional<size_t> selected;
                for (size_t i = 0; last_restore && i < saved.size(); ++i) if (saved[i].valid && saved[i].prefix <= common &&
                    (!selected || saved[i].prefix > saved[*selected].prefix ||
                     (saved[i].prefix == saved[*selected].prefix && saved[i].order > saved[*selected].order))) selected = i;
                if (selected) {
                    position = uint32_t(saved[*selected].prefix); ++hits; outcome = "restored";
                    for (auto& record : saved) if (record.valid && record.prefix > position) record.valid = false;
                } else { position = 0; ++resets; for (auto& record : saved) record.valid = false; outcome = "cold_miss"; }
                history.resize(position); sampling = *job.find("sampling"); seed = job.find("seed")->unsigned_integer();
            } else {
                check(op->string() == "branch_capture" && position); ++captures; size_t at = 0;
                for (size_t i = 0; i < saved.size(); ++i) {
                    if (!saved[i].valid) { at = i; break; }
                    if (saved[i].order < saved[at].order) at = i;
                }
                saved[at] = {position, ++order, true};
            }
            auto policy = sampling.object(); policy.emplace_back("valid", Json(true)); policy.emplace_back("entropy_prefix", Json::integer(position)); policy.emplace_back("seed", Json::integer(seed));
            if (control_corrupt == "token") event(stamp(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(1)}, {"committed_prefix", Json::integer(position + 1)}})));
            return stamp(Json(Json::Object{{"event", Json("control_done")}, {"op", *op}, {"outcome", Json(outcome)}, {"model_steps", Json::integer(0)},
                {"generated_ids", Json(Json::Array{})}, {"committed_prefix", Json::integer(position)},
                {"common_prefix", Json::integer(common + (control_corrupt == "lcp"))}, {"sampling", Json(std::move(policy))},
                {"logits_available", Json(bool(position))}, {"stop_id", Json()}, {"continuation_allowed", Json(bool(position))}}));
        }
        if (job.find("reset")->boolean()) { ++lease; history.clear(); for (auto& record : saved) record.valid = false; }
        const bool generated = job.find("generate")->unsigned_integer() != 0;
        auto done = Fake::operator()(job, [&](const Json& token) { history.push_back(uint32_t(token.find("id")->unsigned_integer())); event(stamp(token)); });
        if (!generated) history.push_back(uint32_t(job.find("tokens")->array()[0].unsigned_integer()));
        if (fail_suffix && !generated) throw FormatError("fake suffix failure after committed model input");
        return stamp(std::move(done));
    }
};
const Json& message(const Json& result) { return *result.find("choices")->array()[0].find("message"); }
const Json& finish(const Json& result) { return *result.find("choices")->array()[0].find("finish_reason"); }
void original_agent_contract() {
    // The frozen production agent's initial Rust request shape/settings, with
    // synthetic messages only: no task transcript, persona or fixture sources.
    const auto wire = parse_json(R"({"model":"tokspur","messages":[{"role":"system","content":"contract fixture"},{"role":"user","content":"Read the project."}],"tools":[
      {"type":"function","function":{"name":"list_files","description":"List files in this isolated project.","strict":true,"parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false}}},
      {"type":"function","function":{"name":"read_file","description":"Read a project file by relative path.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}},
      {"type":"function","function":{"name":"edit_file","description":"Replace exactly one occurrence in an allowed production source file. Tests/config must not change.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"],"additionalProperties":false}}},
      {"type":"function","function":{"name":"run_command","description":"Run the project's permitted verification command.","strict":true,"parameters":{"type":"object","properties":{"command":{"type":"string","enum":["cargo test --offline"]}},"required":["command"],"additionalProperties":false}}}
    ],"stream":false,"max_tokens":32768,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0,"repetition_penalty":1,"seed":44,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}})");
    const auto admitted = admit_chat(wire, "tokspur");
    check(admitted.maximum_tokens == 32768 && admitted.thinking && admitted.seed == 44);
    check(admitted.render.find("enable_thinking")->boolean() && admitted.render.find("preserve_thinking")->boolean());
    check(admitted.render.find("reasoning_effort")->string() == "xhigh");
    check(admitted.render.find("messages")->dump() == wire.find("messages")->dump());
    check(admitted.render.find("tools")->dump() == wire.find("tools")->dump() && admitted.manifest.size() == 4);
    check(admitted.sampling.dump() == R"({"temperature":1,"top_k":20,"top_p":0.95,"min_p":0})");
    auto legacy = without(wire, "chat_template_kwargs");
    legacy = with(with(legacy, "enable_thinking", Json(true)), "preserve_thinking", Json(true));
    check(admit_chat(legacy, "tokspur").render.dump() == admitted.render.dump());
    const auto defaults = admit_chat(without(without(request(), "max_tokens"), "enable_thinking"), "tokspur");
    check(defaults.maximum_tokens == 8192 && defaults.thinking && !defaults.render.find("preserve_thinking") && !defaults.render.find("reasoning_effort"));
    check(admit_chat(with(without(wire, "max_tokens"), "max_completion_tokens", Json::integer(32768)), "tokspur").maximum_tokens == 32768);
    check(admit_chat(with(wire, "max_tokens", Json::integer(4097)), "tokspur").maximum_tokens == 4097);
    for (const auto name : {"presence_penalty", "repetition_penalty"}) {
        const bool presence = std::string_view(name) == "presence_penalty";
        for (const auto value : {presence ? "-0.0" : "1.0", presence ? "0e10" : "1e0"})
            check(admit_chat(with(wire, name, parse_json(value)), "tokspur").sampling.dump() == admitted.sampling.dump());
        for (const auto value : {"0.5", presence ? "0.00000000000000000001" : "1.00000000000000000001", "null", "true", "\"0\""})
            refusal([&] { admit_chat(with(wire, name, parse_json(value)), "tokspur"); });
    }
    for (const auto value : {"0", "32769", "32768.0", "\"32768\"", "-1"})
        refusal([&] { admit_chat(with(wire, "max_tokens", parse_json(value)), "tokspur"); });
    for (const auto& value : {Json(), Json(true), Json("options"), Json(Json::Array{})})
        refusal([&] { admit_chat(with(wire, "chat_template_kwargs", value), "tokspur"); });
    for (const auto name : {"enable_thinking", "preserve_thinking"}) {
        refusal([&] { admit_chat(with(wire, name, Json(false)), "tokspur"); });
        auto bad = with(*wire.find("chat_template_kwargs"), name, Json("true"));
        refusal([&] { admit_chat(with(wire, "chat_template_kwargs", bad), "tokspur"); });
    }
    refusal([&] { admit_chat(with(wire, "reasoning_effort", Json("low")), "tokspur"); });
    refusal([&] { admit_chat(with(wire, "chat_template_kwargs", with(*wire.find("chat_template_kwargs"), "thinking_mode", Json("enabled"))), "tokspur"); });
    refusal([&] { admit_chat(with(without(wire, "reasoning_effort"), "chat_template_kwargs", with(*wire.find("chat_template_kwargs"), "reasoning_effort", Json("unknown"))), "tokspur"); });
    refusal([&] { admit_chat(with(wire, "frequency_penalty", Json::integer(0)), "tokspur"); });
    auto disabled = with(request(), "reasoning_effort", Json("none"));
    check(!admit_chat(disabled, "tokspur").thinking);
    refusal([&] { admit_chat(with(disabled, "enable_thinking", Json(true)), "tokspur"); });

    ServeSession session(codec(), "tokspur"); Fake fake;
    fake.reply("Read it.</think>\n\n<tool_call><function=read_file><parameter=path>\nsrc/lib.rs\n</parameter></function></tool_call><|im_end|>");
    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(999); }};
    const auto result = session.complete(wire, "frozen-wire", [&](const Json& job, const auto& event) { return fake(job, event); }, sink);
    check(finish(result).string() == "tool_calls" && fake.seed == 44 && fake.maximum_generation == 1);
    check(message(result).find("reasoning_content")->string() == "Read it.");
    const auto& call = message(result).find("tool_calls")->array()[0];
    check(call.find("function")->find("name")->string() == "read_file" && call.find("function")->find("arguments")->string() == "{\"path\":\"src/lib.rs\"}");
}
void extended_output_budget() {
    ServeSession session(codec()); Fake fake;
    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(42); }};
    const auto target = [&](const Json& job, const auto& event) { return fake(job, event); };
    const std::string text(32768, 'x'); fake.reply(text);
    const auto result = session.complete(with(request(), "max_tokens", Json::integer(32768)), "long", target, sink);
    check(finish(result).string() == "length" && message(result).find("content")->string() == text);
    check(result.find("usage")->find("completion_tokens")->unsigned_integer() == 32768);
    check(fake.generations == 32768 && fake.maximum_generation == 1 && fake.resets == 1 && session.committed_tokens() == 32769);
    fake.reply("Next<|im_end|>");
    const auto appended = session.complete(with(request("A" + text + "B"), "max_tokens", Json::integer(8)), "append", target, sink);
    check(finish(appended).string() == "stop" && fake.resets == 1 && fake.prefills == 2);
    check(appended.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == 32769);
    // A requested upper bound is clamped to the remaining native context.
    // The original agent can keep asking for32K turns above a32K prompt.
    ServeSession over_half(codec()); Fake over_half_fake; over_half_fake.reply("z<|im_end|>");
    const auto over_half_result = over_half.complete(with(request(std::string(32769,'A')), "max_tokens", Json::integer(32768)), "over-half-context",
        [&](const Json& job, const auto& event) { return over_half_fake(job,event); }, sink);
    check(finish(over_half_result).string()=="stop" && over_half_fake.prefills==32769 && over_half_fake.generations==2);
    ServeSession boundary(codec()); Fake boundary_fake; boundary_fake.reply("<|im_end|>");
    const auto exact = boundary.complete(with(request(std::string(32768, 'A')), "max_tokens", Json::integer(32768)), "exact-context",
        [&](const Json& job, const auto& event) { return boundary_fake(job, event); }, sink);
    check(finish(exact).string() == "stop" && boundary_fake.prefills == 32768 && boundary_fake.generations == 1);
    for (uint32_t remaining : {1u,2u}) {
        ServeSession full(codec()); Fake full_fake; full_fake.reply("xyz");
        const auto limited=full.complete(with(request(std::string(65536-remaining,'A')), "max_tokens", Json::integer(32768)), "context-tail",
            [&](const Json& job,const auto& event) { return full_fake(job,event); },sink);
        check(finish(limited).string()=="length" && full_fake.generations==remaining && full.committed_tokens()==65536);
        check(limited.find("usage")->find("completion_tokens")->unsigned_integer()==remaining && limited.find("usage")->find("total_tokens")->unsigned_integer()==65536);
        check(message(limited).find("content")->string()==std::string_view("xyz").substr(0,remaining));
    }
    const auto before=fake.prefills+fake.generations;
    refusal([&] { session.complete(with(request(std::string(65536,'A')), "max_tokens", Json::integer(32768)), "no-context",target,sink); });
    check(before==fake.prefills+fake.generations);
}
void startup_execution_policy() {
    for (const auto* invalid : {"", "capture", "indirect-compatible", "unknown"})
        refusal([&] { ServeSession rejected(codec(), "tokspur", invalid); });
    for (const auto* mode : {"direct", "recurrent-replay"}) {
        ServeSession session(codec(), "tokspur", mode); Fake fake; fake.stack_execution = mode;
        std::vector<Json> chunks;
        const ServeSink sink{[&](const Json& part) { chunks.push_back(part); }, [] { return false; }, [] { return uint64_t(42); }};
        const auto call = [&](const Json& job, const auto& event) { return fake(job, event); };
        fake.reply("valid<|im_end|>");
        check(finish(session.complete(request("A", true), "startup-policy", call, sink)).string() == "stop");
        const auto submitted = fake.prefills + fake.generations;
        for (const auto* body_key : {"stack_execution", "stack-execution"})
            refusal([&] { session.complete(with(request(), body_key, Json(mode)), "body-switch", call, sink); });
        check(submitted == fake.prefills + fake.generations);
        const auto tool_request = with(request("B", true), "tools", tools());
        for (const auto* corrupt : {"execution", "replay-prepared", "replay-captures", "replay-region", "replay-missing"}) {
            if (std::string_view(mode) == "direct" && std::string_view(corrupt) != "execution") continue;
            chunks.clear(); fake.corrupt = corrupt;
            fake.reply("<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>");
            refusal([&] { session.complete(tool_request, "execution-mismatch", call, sink); });
            check(session.committed_tokens() == 0);
            for (const auto& part : chunks) check(!part.find("choices")->array()[0].find("delta")->find("tool_calls"));
            fake.corrupt.clear(); fake.reply("recovered<|im_end|>");
            check(finish(session.complete(request(), "after-policy-refusal", call, sink)).string() == "stop");
        }
        session.invalidate(); fake.stack_execution = std::string_view(mode) == "direct" ? "recurrent-replay" : "direct";
        fake.reply("never accepted<|im_end|>"); const auto generations = fake.generations;
        refusal([&] { session.complete(request(), "startup-mode-mismatch", call, sink); });
        check(session.committed_tokens() == 0 && fake.generations == generations);
    }
}
void broker_sessions() {
    const auto ready = broker_ready(); validate_worker_ready(ready, "direct", 2, 1 << 20); ++checks;
    refusal([&] { validate_worker_ready(ready); });
    for (const auto* key : {"checkpoint_broker", "broker_epoch", "checkpoint_host_bytes", "checkpoint_gpu_bytes"})
        refusal([&] { validate_worker_ready(without(ready, key), "direct", 2, 1 << 20); });
    ServeSession session(codec(), "tokspur", "direct", ready); BrokerFake fake; bool cancel = false;
    const ServeSink sink{{}, [&] { return cancel; }, [] { return uint64_t(42); }};
    const auto call = [&](const Json& job, const auto& event) { return fake(job, event); };
    fake.reply("X<|im_end|>"); session.complete(request("AB"), "broker-first", call, sink);
    check(fake.begins == 1 && fake.resets == 1 && fake.captures == 1 && fake.prefills == 2);
    fake.reply("Y<|im_end|>"); session.complete(request("ABX<|im_end|>C"), "broker-append", call, sink);
    check(fake.begins == 1 && fake.resets == 1 && fake.captures == 2);
    const auto before = fake.prefills; fake.reply("Z<|im_end|>");
    const auto branch = session.complete(request("ABD"), "broker-branch", call, sink);
    check(fake.hits == 1 && fake.prefills == before + 1 &&
        branch.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == 2);
    const auto captures = fake.captures; fake.reply("Z<|im_end|>"); session.complete(request("ABD"), "broker-exact", call, sink);
    check(fake.captures == captures && fake.hits == 2);
    fake.fail_suffix = true; fake.reply("unused<|im_end|>");
    refusal([&] { session.complete(request("ABE"), "broker-suffix-failed", call, sink); });
    check(!session.committed_tokens()); fake.fail_suffix = false;
    fake.reply("safe<|im_end|>"); session.complete(request("AB"), "broker-after-suffix", call, sink); check(!fake.last_restore);
    cancel = true; const auto controls = fake.begins;
    refusal([&] { session.complete(request("Q"), "broker-before-cancel", call, sink); }); check(fake.begins == controls); cancel = false;
    const auto after_control = [&](const Json& job, const auto& event) {
        auto done = fake(job, event); if (job.find("op") && job.find("op")->string() == "branch_begin") cancel = true; return done;
    };
    refusal([&] { session.complete(request("Q"), "broker-after-control-cancel", after_control, sink); });
    check(!session.committed_tokens()); cancel = false;
    fake.reply("safe<|im_end|>"); session.complete(request("Q"), "broker-after-cancel", call, sink); check(!fake.last_restore);
    const auto greedy_begins = fake.begins, ordinary_resets = fake.resets; fake.reply("safe<|im_end|>");
    auto stochastic = with(request("S"), "temperature", parse_json("0.5")); stochastic = with(std::move(stochastic), "seed", Json::integer(7));
    session.complete(stochastic, "broker-stochastic-reset", call, sink);
    check(fake.begins == greedy_begins && fake.resets == ordinary_resets + 1);
    for (const auto* corrupt : {"epoch", "lease", "token", "lcp", "storage"}) {
        ServeSession invalid(codec(), "tokspur", "direct", ready); BrokerFake broken; broken.control_corrupt = corrupt;
        const auto bad = [&](const Json& job, const auto& event) { return broken(job, event); };
        refusal([&] { invalid.complete(request("A"), "broker-invalid-control", bad, sink); }); check(!invalid.committed_tokens());
    }
}
ServeSession enum_session(ToolEnumValidation policy) {
    return ServeSession(codec(), "tokspur", "direct", Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        , false, Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , false
#endif
        , policy);
}
void executor_enum_contract() {
    constexpr auto deferred = ToolEnumValidation::executor;
    check(parse_tool_enum_validation("strict") == ToolEnumValidation::strict && parse_tool_enum_validation("executor") == deferred);
    check(tool_enum_validation_policy(deferred) == "original-agent-executor-enums-v1");
    refusal([&] { parse_tool_enum_validation("permissive"); });
    refusal([&] { enum_session(static_cast<ToolEnumValidation>(99)); });
    const auto declarations = parse_json(R"([{"type":"function","function":{"name":"run_command","parameters":{"type":"object","properties":{"command":{"type":"string","enum":["cargo test --offline"]}},"required":["command"],"additionalProperties":false}}}])");
    auto job = with(request(), "tools", declarations);
    const std::string literal = "cargo test --offline 2>&1 | head -60";
    const std::string frame = "<tool_call><function=run_command><parameter=command>\n" + literal + "\n</parameter></function></tool_call><|im_end|>";
    ServeSink sink{[](const Json&){}, []{return false;}, []{return uint64_t(44);}};
    auto strict = enum_session(ToolEnumValidation::strict); Fake rejected; rejected.reply(frame);
    refusal([&] { strict.complete(job, "strict-enum", [&](const Json& command, const auto& event){return rejected(command,event);}, sink); });
    auto executor = enum_session(deferred); Fake accepted; accepted.reply(frame);
    const auto result = executor.complete(job, "executor-enum", [&](const Json& command, const auto& event){return accepted(command,event);}, sink);
    const auto proposed = message(result).find("tool_calls")->array()[0];
    check(parse_json(proposed.find("function")->find("arguments")->string()).find("command")->string() == literal);
    check(literal != "cargo test --offline"); // The unchanged executor must reject; this fixture executes nothing.
    auto prior = Json::Array{message(result), Json(Json::Object{{"role",Json("tool")}, {"tool_call_id",*proposed.find("id")},
        {"content",Json(R"({"error":"command not allowed"})")}}), Json(Json::Object{{"role",Json("user")},{"content",Json("continue")}})};
    auto next = with(job, "messages", Json(prior));
    check(admit_chat(next, "tokspur", deferred).manifest.size() == 1);
    refusal([&] { admit_chat(next, "tokspur"); });
    accepted.reply("done<|im_end|>");
    check(finish(executor.complete(next, "after-executor-error", [&](const Json& command, const auto& event){return accepted(command,event);}, sink)).string() == "stop");
    for (const auto& [value, allowed] : {std::pair{R"({"command":1})", false}, std::pair{R"({"command":"bad","extra":true})", false}, std::pair{R"({})", false}, std::pair{R"({"command":"<|im_start|>"})", true}, std::pair{R"({"command":"<|endoftext|>"})", true}}) {
        auto function = with(*proposed.find("function"), "arguments", Json(value));
        auto bad = prior; bad[0] = with(bad[0], "tool_calls", Json(Json::Array{with(proposed,"function",function)}));
        if (allowed) { static_cast<void>(admit_chat(with(next,"messages",Json(bad)),"tokspur",deferred)); ++checks; }
        else refusal([&] { admit_chat(with(next,"messages",Json(bad)),"tokspur",deferred); });
    }
    for (const auto key : {"tool_enum_validation", "tool-enum-validation"})
        refusal([&] { admit_chat(with(job,key,Json("executor")),"tokspur",deferred); });
    const auto nested = parse_json(R"({"type":"object","properties":{"items":{"type":"array","items":{"type":"integer","enum":[1],"maximum":3}}},"required":["items"],"additionalProperties":false})");
    validate_tool_value(nested,parse_json(R"({"items":[2,3]})"),deferred);++checks;
    refusal([&] { validate_tool_value(nested,parse_json(R"({"items":[2]})")); });
    for (const auto value : {R"({"items":[4]})",R"({"items":["2"]})",R"({"items":[2],"extra":true})",R"({})"})
        refusal([&] { validate_tool_value(nested,parse_json(value),deferred); });
    const auto malformed = parse_json(R"([{"type":"function","function":{"name":"run_command","parameters":{"type":"object","properties":{"command":{"type":"string","enum":[1]}}}}}])");
    refusal([&] { admit_chat(with(job,"tools",malformed),"tokspur",deferred); });
    for (const auto raw : {"<tool_call><function=unknown></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=extra>\nbad\n</parameter></function></tool_call><|im_end|>",
        "<tool_call><function=run_command></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=command>\n<|im_start|>\n</parameter></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=command>\n<|endoftext|>\n</parameter></function></tool_call><|im_end|>"}) {
        accepted.reply(raw);
        refusal([&] { executor.complete(job,"still-refused",[&](const Json& command,const auto& event){return accepted(command,event);},sink); });
    }
    accepted.reply("done<|im_end|>");
    refusal([&] { executor.complete(with(job,"tool_choice",Json("required")),"required",[&](const Json& command,const auto& event){return accepted(command,event);},sink); });
    accepted.reply(frame);
    const auto partial = executor.complete(with(job,"max_tokens",Json::integer(10)),"partial",[&](const Json& command,const auto& event){return accepted(command,event);},sink);
    check(finish(partial).string() == "length" && !message(partial).find("tool_calls"));
}

ServeSession arg_session(ToolArgValidation policy) {
    return ServeSession(codec(), "tokspur", "direct", Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        , false, Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , false
#endif
        , ToolEnumValidation::strict
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        , false
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , Json()
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        , Json()
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , std::nullopt
#endif
        , policy);
}
void executor_arg_contract() {
    constexpr auto deferred = ToolArgValidation::executor;
    check(parse_tool_arg_validation("strict") == ToolArgValidation::strict && parse_tool_arg_validation("executor") == deferred);
    check(tool_arg_validation_policy(ToolArgValidation::strict) == "strict-server-args-v1");
    check(tool_arg_validation_policy(deferred) == "executor-deferred-args-v1");
    refusal([&] { parse_tool_arg_validation("permissive"); });
    refusal([&] { arg_session(static_cast<ToolArgValidation>(99)); });
    const auto declarations = parse_json(R"([{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"],"additionalProperties":false}}}])");
    auto job = with(request(), "tools", declarations);
    const std::string extra_frame = "<tool_call><function=read_file><parameter=path>\na.rs\n</parameter><parameter=line>\n10\n</parameter><parameter=limit>\nten\n</parameter></function></tool_call><|im_end|>";
    ServeSink sink{[](const Json&){}, []{return false;}, []{return uint64_t(44);}};
    auto strict = arg_session(ToolArgValidation::strict); Fake rejected; rejected.reply(extra_frame);
    refusal([&] { strict.complete(job, "strict-arg", [&](const Json& command, const auto& event){return rejected(command,event);}, sink); });
    auto executor = arg_session(deferred); Fake accepted; accepted.reply(extra_frame);
    const auto result = executor.complete(job, "executor-arg", [&](const Json& command, const auto& event){return accepted(command,event);}, sink);
    const auto proposed = message(result).find("tool_calls")->array()[0];
    const auto args_json = parse_json(proposed.find("function")->find("arguments")->string());
    check(args_json.find("path")->string() == "a.rs" && args_json.find("line")->unsigned_integer() == 10 && args_json.find("limit")->string() == "ten");
    // Missing required parameter also passes to the executor as an empty object.
    accepted.reply("<tool_call><function=read_file></function></tool_call><|im_end|>");
    const auto missing_res = executor.complete(job, "executor-missing-req", [&](const Json& command, const auto& event){return accepted(command,event);}, sink);
    check(parse_json(message(missing_res).find("tool_calls")->array()[0].find("function")->find("arguments")->string()).object().empty());
    // Turn 2 history round-trip succeeds under executor arg policy and refuses under strict.
    auto prior = Json::Array{message(result), Json(Json::Object{{"role",Json("tool")}, {"tool_call_id",*proposed.find("id")},
        {"content",Json(R"({"error":"unexpected argument line"})")}}), Json(Json::Object{{"role",Json("user")},{"content",Json("retry without line")}})};
    auto next = with(job, "messages", Json(prior));
    check(admit_chat(next, "tokspur", ToolEnumValidation::strict, deferred).manifest.size() == 1);
    refusal([&] { admit_chat(next, "tokspur"); });
    accepted.reply("done<|im_end|>");
    check(finish(executor.complete(next, "after-executor-arg-error", [&](const Json& command, const auto& event){return accepted(command,event);}, sink)).string() == "stop");
    // Raw structural delimiters and unknown tools remain refused. JSON-escaped string values are inert data.
    for (const auto& [raw, allowed] : {std::pair{"<tool_call><function=unknown><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>", false},
        std::pair{"<tool_call><function=read_file><parameter=path>\n<|im_start|>\n</parameter></function></tool_call><|im_end|>", false},
        std::pair{"<tool_call><function=read_file><parameter=extra>\n{\"nested\":\"\\u003ctool_call\\u003e\"}\n</parameter></function></tool_call><|im_end|>", true},
        std::pair{"<tool_call><function=read_file><parameter=extra>\n{\"nested\":\"\\u003c|endoftext|\\u003e\"}\n</parameter></function></tool_call><|im_end|>", true}}) {
        accepted.reply(raw);
        const auto run = [&] { return executor.complete(job,"escaped-value",[&](const Json& command,const auto& event){return accepted(command,event);},sink); };
        if (allowed) check(message(run()).find("tool_calls")->array().size() == 1);
        else refusal([&] { static_cast<void>(run()); });
    }
    for (const auto& [value, allowed] : {std::pair{R"("not-an-object")", false}, std::pair{R"([1,2])", false}, std::pair{R"(123)", false}, std::pair{R"({"path":"<|im_start|>"})", true}, std::pair{R"({"path":"<|endoftext|>"})", true}}) {
        auto function = with(*proposed.find("function"), "arguments", Json(value));
        auto bad = prior; bad[0] = with(bad[0], "tool_calls", Json(Json::Array{with(proposed,"function",function)}));
        if (allowed) { static_cast<void>(admit_chat(with(next,"messages",Json(bad)),"tokspur",ToolEnumValidation::strict,deferred)); ++checks; }
        else refusal([&] { admit_chat(with(next,"messages",Json(bad)),"tokspur",ToolEnumValidation::strict,deferred); });
    }
}
void server_generation_clock_contract() {
    ServeSession session(codec(), "tokspur");
    Fake fake;
    fake.reply("Generated token sequence for testing.<|im_end|>");
    std::vector<Json> traces;
    const ServeSink sink{[](const Json& chunk) { check(chunk.dump().find("server_generation_wall_ms") == std::string::npos); },
        [] { return false; }, [] { return uint64_t(44); }, [&](const Json& row) { traces.push_back(row); }};
    const auto target = [&](const Json& job, const auto& event) {
        return fake(job, event);
    };

    const auto result = session.complete(request("Test input prompt", false), "clock-contract-01", target, sink);
    check(finish(result).string() == "stop");
    check(result.find("tokspur_private") == nullptr);
    check(traces.size() == 2 && traces[0].find("event")->string() == "prompt_prefix" && traces[1].find("event")->string() == "completion");

    const auto& trace = traces[1];
    check(trace.find("response_id")->string() == "clock-contract-01");
    check(trace.find("prefill_generation_overlap")->string() == "none");
    check(trace.find("stopped_on_eos")->boolean());

    const double admission_ms = parse_number(*trace.find("server_admission_wall_ms"));
    const double prefill_ms = parse_number(*trace.find("server_prefill_wall_ms"));
    const double wall_ms = parse_number(*trace.find("server_generation_wall_ms"));
    const double request_ms = parse_number(*trace.find("server_request_wall_ms"));
    const double tp = parse_number(*trace.find("server_generation_throughput"));
    check(admission_ms >= 0.0 && prefill_ms >= 0.0 && wall_ms > 0.0 && request_ms >= prefill_ms + wall_ms - 1e-4);
    check(std::abs((admission_ms + prefill_ms + wall_ms) - request_ms) < 1e-3);
    check(tp > 0.0);

    const auto completion_tokens = result.find("usage")->find("completion_tokens")->unsigned_integer();
    check(completion_tokens > 0 && completion_tokens == fake.generations);
    check(trace.find("completion_tokens")->unsigned_integer() == completion_tokens);

    const double expected_tp = (static_cast<double>(completion_tokens) * 1000.0) / wall_ms;
    check(std::abs(tp - expected_tp) / expected_tp < 1e-4);
}

void server_generation_terminal_and_determinism_contract() {
    Json::Array outputs;
    for (int repeat = 0; repeat < 2; ++repeat) {
        ServeSession session(codec(), "tokspur");
        Fake fake;
        fake.reply("<|im_end|>");
        std::vector<Json> traces;
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); },
            [&](const Json& row) { traces.push_back(row); }};
        const auto target = [&](const Json& job, const auto& event) { return fake(job, event); };

        const auto result = session.complete(request("A", false), "terminal-contract-01", target, sink);
        check(finish(result).string() == "stop");
        check(result.find("tokspur_private") == nullptr);
        check(traces.size() == 2 && traces[1].find("event")->string() == "completion");

        const auto completion_tokens = result.find("usage")->find("completion_tokens")->unsigned_integer();
        check(completion_tokens == 1 && traces[1].find("completion_tokens")->unsigned_integer() == 1);
        const double wall_ms = parse_number(*traces[1].find("server_generation_wall_ms"));
        const double tp = parse_number(*traces[1].find("server_generation_throughput"));
        check(wall_ms > 0.0 && tp > 0.0);
        const double expected_tp = (static_cast<double>(completion_tokens) * 1000.0) / wall_ms;
        check(std::abs(tp - expected_tp) / expected_tp < 1e-4);

        Json::Object canonical;
        for (const auto& [name, value] : result.object()) if (name != "created") canonical.emplace_back(name, value);
        outputs.emplace_back(std::move(canonical));
    }
    check(outputs[0].dump() == outputs[1].dump());
}
} // namespace

void literal_marker_serving() {
    const std::string body = "```xml\n<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call>\n```\nDone.";
    for (const auto terminal : {"<|im_end|>", "<|endoftext|>"}) {
        ServeSession serving(codec(), "tokspur"); Fake fake; fake.reply(body + terminal);
        std::vector<Json> chunks;
        ServeSink sink{[&](const Json& value) { chunks.push_back(value); }, [] { return false; }, [] { return uint64_t(44); }};
        const auto result = serving.complete(with(request("literal", true, false), "tools", tools()), "literal-stop", [&](const Json& job, const auto& event) { return fake(job, event); }, sink);
        check(message(result).find("content")->string() == body);
        check(!message(result).find("tool_calls")); check(finish(result).string() == "stop");
        std::string streamed;
        for (const auto& chunk : chunks) {
            const auto* delta = chunk.find("choices")->array()[0].find("delta");
            if (const auto* content = delta->find("content")) streamed += content->string();
        }
        check(streamed == body);
    }
}

void translated_history_metadata() {
    for (const auto* blocks : {
        R"([{"type":"thinking","thinking":"fixture &amp; <tool_call>","signature":""}])",
        R"([{"type":"thinking","thinking":"fixture &amp; "},{"type":"thinking","thinking":"<tool_call>"}])"}) {
        const auto job = parse_json(std::string(R"({"model":"m","messages":[{"role":"assistant","content":"done","reasoning_content":"fixture &amp; <tool_call>","thinking_blocks":)") + blocks + R"(},{"role":"user","content":"next"}]})");
        const auto admitted = admit_chat(job, "m");
        check(admitted.render.find("messages")->dump() == job.find("messages")->dump());
    }
    for (const auto* message : {
        R"({"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"hidden"}]})",
        R"({"role":"assistant","content":"done","reasoning_content":"visible","thinking_blocks":[{"type":"thinking","thinking":"different"}]})",
        R"({"role":"user","content":"done","thinking_blocks":[]})",
        R"({"role":"assistant","content":"done","thinking_blocks":null})",
        R"({"role":"assistant","content":"done","thinking_blocks":[{"type":"redacted_thinking","data":"opaque"}]})",
        R"({"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"","signature":"unverified"}]})",
        R"({"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"","extra":true}]})"}) {
        refusal([&] { admit_chat(parse_json(std::string(R"({"model":"m","messages":[)") + message + "]}"), "m"); });
    }
    static_cast<void>(admit_chat(parse_json(R"({"model":"m","messages":[{"role":"assistant","content":"done","thinking_blocks":[]}]})"), "m")); ++checks;
    for (const auto* index : {"0", "1", "127", "128", "-1", "0.0", "\"0\""}) {
        const auto job = parse_json(std::string(R"({"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"call-a","type":"function","index":)") + index + R"(,"function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-a","content":"done"}],"tools":[{"type":"function","function":{"name":"f"}}]})");
        if (std::string_view(index) == "0" || std::string_view(index) == "1" || std::string_view(index) == "127") { static_cast<void>(admit_chat(job, "m")); ++checks; }
        else refusal([&] { admit_chat(job, "m"); });
    }
    for (const auto* index : {"0", "1", "2", "128"}) {
        const auto job = parse_json(std::string(R"({"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"a","type":"function","index":0,"function":{"name":"f","arguments":"{}"}},{"id":"b","type":"function","index":)") + index + R"(,"function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"a","content":"one"},{"role":"tool","tool_call_id":"b","content":"two"}],"tools":[{"type":"function","function":{"name":"f"}}]})");
        if (std::string_view(index) != "128") { static_cast<void>(admit_chat(job, "m")); ++checks; }
        else refusal([&] { admit_chat(job, "m"); });
    }
}
void parallel_tool_streams() {
    for (const bool valid : {true, false}) {
        ServeSession session(codec()); Fake fake; Json::Array chunks;
        ServeSink sink{[&](const Json& value) { chunks.push_back(value); }, [] { return false; }, [] { return uint64_t(42); }};
        const auto frame = [](std::string_view name, std::string_view path) { return "<tool_call><function=" + std::string(name) + "><parameter=path>\n" + std::string(path) + "\n</parameter></function></tool_call>"; };
        fake.reply(frame("read_file", "a.rs") + "\n" + frame(valid ? "read_file" : "unknown", "b.rs") + "<|im_end|>");
        const auto job = with(with(request("fixture", true), "tools", tools()), "max_tokens", Json::integer(512));
        const auto call = [&](const Json& command, const auto& event) { return fake(command, event); };
        if (valid) {
            const auto result = session.complete(job, "parallel", call, sink);
            check(message(result).find("tool_calls")->array().size() == 2);
            size_t index = 0;
            for (const auto& chunk : chunks) {
                const auto* calls = chunk.find("choices")->array()[0].find("delta")->find("tool_calls");
                if (!calls) continue;
                check(calls->array().size() == 1 && calls->array()[0].find("index")->unsigned_integer() == index);
                check(parse_json(calls->array()[0].find("function")->find("arguments")->string()).find("path")->string() == (index ? "b.rs" : "a.rs"));
                ++index;
            }
            check(index == 2);
        } else {
            refusal([&] { session.complete(job, "parallel-invalid", call, sink); });
            for (const auto& chunk : chunks) check(!chunk.find("choices")->array()[0].find("delta")->find("tool_calls"));
        }
    }
}
int main() {
    try {
        parallel_tool_streams();
        translated_history_metadata();
        literal_marker_serving();
        worker_ready_admission();
        original_agent_contract();
        extended_output_budget();
        startup_execution_policy();
        broker_sessions();
        executor_enum_contract();
        executor_arg_contract();
        server_generation_clock_contract();
        server_generation_terminal_and_determinism_contract();
        for (const auto text : {
            R"({"model":"wrong","messages":[]})", R"({"model":"tokspur","messages":[],"n":2})",
            R"({"model":"tokspur","messages":[],"max_tokens":0})", R"({"model":"tokspur","messages":[],"max_tokens":32769})",
            R"({"model":"tokspur","messages":[],"temperature":-1})", R"({"model":"tokspur","messages":[],"top_p":0})",
            R"({"model":"tokspur","messages":[],"max_tokens":1,"max_completion_tokens":1})",
            R"({"model":"tokspur","messages":[{"role":"tool","content":"x","tool_call_id":"fake"}]})",
            R"({"model":"tokspur","messages":[{"role":"user","content":[{"type":"image_url"}]}]})"
        }) refusal([&] { static_cast<void>(admit_chat(parse_json(text), "tokspur")); });
        auto admitted = admit_chat(with(request(), "tools", tools()), "tokspur"); check(admitted.manifest.size() == 1 && admitted.names[0] == "read_file");
        auto text_parts = with(request(), "messages", parse_json(R"([{"role":"user","content":[{"type":"text","text":"hello"}]}])"));
        check(admit_chat(text_parts, "tokspur").maximum_tokens == 256);
        const auto usage_option = parse_json(R"({"include_usage":true})");
        check(admit_chat(with(with(request("A",true), "stream_options", usage_option), "keep_alive", parse_json("-1.0")), "tokspur").include_usage);
        refusal([&] { admit_chat(with(request(), "stream_options", usage_option), "tokspur"); });
        check(admit_chat(with(request(), "keep_alive", Json("5m")), "tokspur").maximum_tokens == 256);
        for (const auto& invalid : {Json(true), Json(), Json("tomorrow"), parse_json("-2")})
            refusal([&] { admit_chat(with(request(), "keep_alive", invalid), "tokspur"); });
        for (const auto& invalid : {Json(true), Json(), parse_json(R"({"include_usage":1})"), parse_json(R"({"other":true})")})
            refusal([&] { admit_chat(with(request("A",true), "stream_options", invalid), "tokspur"); });
        {
            ServeSession usage_session(codec()); Fake worker; worker.reply("Hello<|im_end|>"); std::vector<Json> events;
            const auto target = [&](const Json& job, const auto& event) { return worker(job,event); };
            const ServeSink output{[&](const Json& value) { events.push_back(value); }, [] { return false; }, [] { return uint64_t(44); }};
            const auto response = usage_session.complete(with(with(request("A",true), "stream_options", usage_option), "keep_alive", parse_json("-1")), "usage", target, output);
            check(events.size() > 2 && events.back().find("choices")->array().empty());
            check(events.back().find("usage")->dump() == response.find("usage")->dump());
            size_t finishes = 0;
            for (size_t n = 0; n + 1 < events.size(); ++n) {
                check(events[n].find("usage") && events[n].find("usage")->is_null());
                if (!events[n].find("choices")->array()[0].find("finish_reason")->is_null()) ++finishes;
            }
            check(finishes == 1);
            const auto old = usage_session.committed_tokens(); events.clear(); worker.reply("Next<|im_end|>");
            const auto next = usage_session.complete(with(with(request("AHello<|im_end|>B",true), "stream_options", parse_json(R"({"include_usage":false})")), "keep_alive", Json::integer(0)), "no-usage", target, output);
            check(worker.resets == 1 && next.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == old);
            for (const auto& event : events) check(!event.find("usage") && event.find("choices")->array().size() == 1);
        }
        for (const auto schema : {
            R"({"type":"object","oneOf":[]})", R"({"type":"object","additionalProperties":true})",
            R"({"type":"array"})", R"({"type":["string","null"]})", R"({"type":"object","required":["missing"]})"
        }) {
            auto declaration = tools().array()[0]; auto function = with(*declaration.find("function"), "parameters", parse_json(schema));
            refusal([&] { static_cast<void>(admit_chat(with(request(), "tools", Json(Json::Array{with(declaration, "function", function)})), "tokspur")); });
        }
        const auto schema = parse_json(R"({"type":"object","properties":{"n":{"type":"integer","minimum":1,"maximum":3},"values":{"type":"array","items":{"type":"string","enum":["x","y"]}}},"required":["n"]})");
        validate_tool_value(schema, parse_json(R"({"n":2,"values":["x","y"]})")); ++checks;
        for (const auto value : {R"({"n":1.5})", R"({"n":1.0})", R"({"n":4})", R"({"n":9007199254740993})", R"({"n":2,"values":["z"]})", R"({"n":2,"extra":true})", R"({})"}) refusal([&] { validate_tool_value(schema, parse_json(value)); });
        refusal([&] { validate_tool_value(parse_json(R"({"type":"integer"})"), parse_json("1.00000000000000001")); });
        refusal([&] { validate_tool_value(parse_json(R"({"type":"number","enum":[1]})"), parse_json("1.00000000000000001")); });
        refusal([&] { validate_tool_value(parse_json(R"({"type":"number","enum":[9007199254740992]})"), parse_json("9007199254740993")); });
        refusal([&] { validate_tool_value(parse_json(R"({"type":"number","maximum":1})"), parse_json("1.00000000000000001")); });
        validate_tool_value(parse_json(R"({"type":"number","enum":[1.0]})"), parse_json("1e0")); ++checks;
        validate_tool_value(parse_json(R"({"type":"number","minimum":-1,"maximum":0})"), parse_json("-0.1")); ++checks;
        const auto opencode_schema = parse_json(R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"timeout":{"type":"number","exclusiveMinimum":0,"maximum":60}},"required":["timeout"]})");
        auto opencode_declaration = tools().array()[0];
        auto opencode_function = with(*opencode_declaration.find("function"), "parameters", opencode_schema);
        check(admit_chat(with(request(), "tools", Json(Json::Array{with(opencode_declaration, "function", opencode_function)})), "tokspur").manifest.size() == 1);
        for (const auto value : {R"({"timeout":0.0001})", R"({"timeout":1})", R"({"timeout":60})"}) { validate_tool_value(opencode_schema, parse_json(value)); ++checks; }
        for (const auto value : {R"({"timeout":0})", R"({"timeout":-0.0})", R"({"timeout":-1})", R"({"timeout":60.01})"}) refusal([&] { validate_tool_value(opencode_schema, parse_json(value)); });
        std::vector<Json> chunks; bool cancel = false; size_t seeds = 0;
        ServeSink sink{[&](const Json& chunk) { chunks.push_back(chunk); }, [&] { return cancel; }, [&] { ++seeds; return uint64_t(42); }};
        ServeSession session(codec()); Fake fake; fake.reply("Hello 🌍<|im_end|>");
        auto call = [&](const Json& job, const auto& event) { return fake(job, event); };
        auto result = session.complete(request("A", true), "first", call, sink);
        check(message(result).find("content")->string() == "Hello 🌍" && finish(result).string() == "stop");
        std::string streamed;
        for (const auto& event : chunks) {
            const auto reparsed = parse_json(event.dump()); // Every SSE JSON payload has complete UTF-8.
            if (const auto* part = reparsed.find("choices")->array()[0].find("delta")->find("content")) streamed += part->string();
        }
        check(streamed == "Hello 🌍");
        check(fake.resets == 1 && seeds == 1 && !chunks.empty());
        const size_t old = session.committed_tokens(); fake.reply("Next<|im_end|>");
        result = session.complete(request("AHello 🌍<|im_end|>B"), "second", call, sink);
        check(fake.resets == 1 && seeds == 1 && result.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == old);
        fake.reply("Changed<|im_end|>");
        result = session.complete(with(request("AHello 🌍<|im_end|>BNext<|im_end|>C"), "temperature", parse_json("0.5")), "third", call, sink);
        check(fake.resets == 2 && seeds == 2);
        const auto before = fake.prefills + fake.generations;
        refusal([&] { session.complete(with(request(), "n", Json::integer(2)), "invalid", call, sink); }); check(before == fake.prefills + fake.generations);
        refusal([&] { session.complete(request(std::string(65536, 'A')), "overflow", call, sink); }); check(before == fake.prefills + fake.generations);
        chunks.clear(); fake.reply("why</think>\n\nVisible<|im_end|>");
        result = session.complete(request("T", true, true), "thinking", call, sink);
        check(message(result).find("reasoning_content")->string() == "why" && message(result).find("content")->string() == "\n\nVisible");
        auto tool_request = with(request("U", true), "tools", tools());
        const std::string tool_response = "<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>";
        chunks.clear(); fake.reply(tool_response); result = session.complete(tool_request, "tool", call, sink);
        check(finish(result).string() == "tool_calls" && message(result).find("tool_calls")->array()[0].find("id")->string() == "tool_call_0");
        size_t tool_chunks = 0; for (const auto& chunk : chunks) if (chunk.find("choices")->array()[0].find("delta")->find("tool_calls")) ++tool_chunks; check(tool_chunks == 1);
        auto history_request = with(request("R"), "tools", tools());
        history_request = with(std::move(history_request), "messages", parse_json(R"([{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]},{"role":"tool","tool_call_id":"a","content":"file"},{"role":"user","content":"R"}])"));
        check(admit_chat(history_request, "tokspur").manifest.size() == 1);
        auto bad_history = history_request.find("messages")->array(); bad_history.push_back(parse_json(R"({"role":"tool","tool_call_id":"a","content":"duplicate"})"));
        refusal([&] { admit_chat(with(history_request, "messages", Json(bad_history)), "tokspur"); });
        bad_history = history_request.find("messages")->array(); bad_history[1] = with(bad_history[1], "name", Json("other"));
        refusal([&] { admit_chat(with(history_request, "messages", Json(bad_history)), "tokspur"); });
        for (const auto corrupt : {"prefix", "policy", "terminal", "throw"}) {
            chunks.clear(); fake.corrupt = corrupt; fake.reply(tool_response);
            refusal([&] { session.complete(tool_request, "bad", call, sink); }); check(session.committed_tokens() == 0);
            for (const auto& chunk : chunks) check(!chunk.find("choices")->array()[0].find("delta")->find("tool_calls"));
        }
        fake.corrupt.clear(); fake.reply("safe<|im_end|>"); result = session.complete(request(), "recovery", call, sink); check(finish(result).string() == "stop");
        chunks.clear(); fake.reply(tool_response); auto limited = with(tool_request, "max_tokens", Json::integer(10));
        result = session.complete(limited, "truncated", call, sink); check(finish(result).string() == "length" && !message(result).find("tool_calls"));
        chunks.clear(); fake.reply("not a call<|im_end|>");
        refusal([&] { session.complete(with(tool_request, "tool_choice", Json("required")), "required", call, sink); });
        for (const auto* name : {"auto", "none", "required"}) {
            auto declarations = tools().array();
            declarations.push_back(Json(Json::Object{{"type", Json("function")}, {"function", Json(Json::Object{{"name", Json(name)}})}}));
            auto named = with(request("named"), "tools", Json(std::move(declarations)));
            named = with(named, "tool_choice", Json(Json::Object{{"type", Json("function")}, {"function", Json(Json::Object{{"name", Json(name)}})}}));
            const auto admitted = admit_chat(named, "tokspur");
            check(admitted.names.size() == 1 && admitted.names[0] == name && admitted.render.find("tools")->array().size() == 1);
            check(admitted.render.find("messages")->array()[0].find("content")->string().find(std::string("named ") + name) != std::string_view::npos);
            fake.reply("not a call<|im_end|>");
            refusal([&] { session.complete(named, "named-missing", call, sink); });
            fake.reply(std::string("<tool_call><function=") + name + "></function></tool_call><|im_end|>");
            const auto selected = session.complete(named, "named-ok", call, sink);
            check(finish(selected).string() == "tool_calls" && message(selected).find("tool_calls")->array()[0].find("function")->find("name")->string() == name);
        }
        cancel = true; const auto previous = fake.prefills + fake.generations; const auto previous_tokens = session.committed_tokens();
        refusal([&] { session.complete(request(), "cancelled", call, sink); }); check(fake.prefills + fake.generations == previous && session.committed_tokens() == previous_tokens); cancel = false;
        fake.reply("more<|im_end|>");
        auto cancel_after_commit = [&](const Json& job, const auto& event) { auto done = fake(job, event); if (job.find("generate")->unsigned_integer()) cancel = true; return done; };
        refusal([&] { session.complete(request("C", true), "midcancel", cancel_after_commit, sink); }); check(session.committed_tokens() == 0); cancel = false;
        fake.reply("<tool_call><function=unknown></function></tool_call><|im_end|>");
        refusal([&] { session.complete(tool_request, "unknown", call, sink); });
        fake.reply("<tool_call><function=read_file><parameter=path>\ntrue\n</parameter></function></tool_call><|im_end|>");
        result = session.complete(tool_request, "rawstring", call, sink); check(message(result).find("tool_calls")->array()[0].find("function")->find("arguments")->string() == "{\"path\":\"true\"}");
        std::cout << "PASS owned serving protocol checks=" << checks << " (model-free; no tools executed)\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
