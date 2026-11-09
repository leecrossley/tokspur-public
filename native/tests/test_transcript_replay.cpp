#include "tokspur/transcript_replay.hpp"
#include "tokspur/gguf.hpp"

#include <iostream>
#include <stdexcept>
#include <source_location>

using namespace tokspur;
namespace {
size_t checks = 0;
const Json& field(const Json& value, std::string_view key) {
    const auto* found = value.find(key); if (!found) throw std::runtime_error("missing replay result field"); return *found;
}
void check(bool ok, std::source_location at = std::source_location::current()) {
    ++checks; if (!ok) throw std::runtime_error("transcript replay check failed at line " + std::to_string(at.line()));
}
Json with(Json value, std::string_view key, Json replacement) {
    auto fields = value.object(); for (auto& [name, item] : fields) if (name == key) { item = std::move(replacement); return Json(std::move(fields)); }
    fields.emplace_back(key, std::move(replacement)); return Json(std::move(fields));
}
ServeCodec codec() {
    const ChatRenderer renderer({"synthetic xhigh", "synthetic low", "synthetic tools", "synthetic instructions"});
    return {[renderer](const Json& value) { return renderer.render(value); },
        [](const RenderedChat& rendered) { const auto& bytes = rendered.prompt; std::vector<uint32_t> ids; for (const unsigned char c : bytes) ids.push_back(uint32_t(c) + 1); return ids; },
        [](std::span<const uint32_t> ids) { std::string bytes; for (const auto id : ids) { if (!id || id > 256) throw FormatError("fake byte ID"); bytes += char(id - 1); } return bytes; },
        {257, 256, 255, 254}};
}
Json wire() {
    return parse_json(R"({"model":"tokspur","messages":[{"role":"user","content":"A"}],"stream":false,"max_tokens":32768,"temperature":1,"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0,"repetition_penalty":1,"seed":44,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"},"tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{"command":{"type":"string","enum":["ok"]}},"required":["command"],"additionalProperties":false}}}]})");
}
Json response(std::string_view arguments = R"({"command":"ok"})") {
    const Json function(Json::Object{{"name", Json("run")}, {"arguments", Json(std::string(arguments))}});
    const Json call(Json::Object{{"id", Json("call0")}, {"type", Json("function")}, {"function", function}});
    const Json message(Json::Object{{"role", Json("assistant")}, {"content", Json("")}, {"reasoning_content", Json("Reason")}, {"tool_calls", Json(Json::Array{call})}});
    const Json choice(Json::Object{{"finish_reason", Json("tool_calls")}, {"message", message}});
    return Json(Json::Object{{"choices", Json(Json::Array{choice})}, {"usage", parse_json(R"({"prompt_tokens":99,"completion_tokens":7})")}});
}
Json finished(Json reply, const char* finish) {
    const auto choice = field(reply, "choices").array()[0];
    return with(reply, "choices", Json(Json::Array{with(choice, "finish_reason", Json(finish))}));
}
Json bundle(Json request = wire(), Json reply = response(), uint64_t turn = 0) {
    auto messages = field(request, "messages").array();
    messages.insert(messages.begin(), parse_json(R"({"role":"system","content":"synthetic persona supplied once"})"));
    return Json(Json::Object{{"turn", Json::integer(turn)}, {"wire_request", std::move(request)},
        {"effective_messages", Json(std::move(messages))}, {"response", std::move(reply)}});
}
void fixtures() {
    TranscriptReplay replay(codec(), "tokspur");
    const auto valid = replay.turn(bundle());
    check(field(valid, "passed").boolean());
    check(field(valid, "wire_request_admitted").boolean() && field(valid, "effective_request_admitted").boolean());
    check(field(valid, "native_prompt_tokens").unsigned_integer() > field(valid, "wire_native_prompt_tokens").unsigned_integer());
    check(field(valid, "reported_completion_tokens").unsigned_integer() == 7);
    check(field(valid, "canonical_output_tokens").unsigned_integer() > 7);
    check(!field(valid, "exact_generated_ids_observed").boolean() && !field(valid, "actual_append_proven").boolean());
    check(field(valid, "actual_input_scalar_steps").is_null() && field(valid, "actual_output_scalar_steps").is_null());
    check(!field(valid, "tools_executed").boolean() && !field(valid, "model_loaded").boolean());
    check(field(valid, "canonical_terminal_synthesized").boolean());
    check(field(valid, "canonical_tool_records").unsigned_integer() == 1);
    for (const auto bad : {R"({"command":"ok","unknown":true})", R"({"command":"wrong-enum"})", R"({"command":1})"}) {
        const auto refused = replay.turn(bundle(wire(), response(bad), 1));
        check(!field(refused, "passed").boolean() && !field(refused, "response_tool_arguments_admitted").boolean());
        check(!field(refused, "refusals").array().empty());
        const auto later = replay.turn(bundle(wire(), response(), 2));
        check(field(later, "passed").boolean());
        check(!field(later, "canonical_append_candidate").boolean());
    }
    const auto extra = replay.turn(bundle(with(wire(), "unsupported", Json(true))));
    check(!field(extra, "wire_request_admitted").boolean() && !field(extra, "effective_request_admitted").boolean());
    auto large = wire(); large = with(large, "messages", Json(Json::Array{Json(Json::Object{{"role", Json("user")}, {"content", Json(std::string(66000, 'x'))}})}));
    const auto over = replay.turn(bundle(large));
    check(!field(over, "effective_request_admitted").boolean());
    check(field(over, "native_prompt_tokens").unsigned_integer() + field(over, "reserved_output_tokens").unsigned_integer() > 65536);
    const auto length = parse_json(R"({"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"partial","reasoning_content":"Reason"}}]})");
    const auto capped = replay.turn(bundle(wire(), length));
    check(field(capped, "passed").boolean() && field(capped, "canonical_response_parser_passed").boolean());
    check(!field(capped, "canonical_terminal_synthesized").boolean() && !field(capped, "canonical_response_complete").boolean());
    check(field(capped, "canonical_tool_records").unsigned_integer() == 0);
    auto tool_length = response(); const auto tool_choice = field(tool_length, "choices").array()[0];
    tool_length = with(tool_length, "choices", Json(Json::Array{with(tool_choice, "finish_reason", Json("length"))}));
    const auto withheld = replay.turn(bundle(wire(), tool_length));
    check(field(withheld, "canonical_response_parser_passed").boolean() && field(withheld, "canonical_tool_records").unsigned_integer() == 0);
    const auto marker = replay.turn(bundle(wire(), response(R"({"command":"<|im_end|>"})")));
    check(field(marker, "canonical_response_parser_passed").boolean());
    check(field(replay.turn(bundle()), "passed").boolean());
    TranscriptReplay append(codec(), "tokspur");
    const auto first = append.turn(bundle());
    auto next = wire(); auto messages = field(next, "messages").array();
    messages.push_back(field(field(response(), "choices").array()[0], "message"));
    messages.push_back(parse_json(R"({"role":"tool","tool_call_id":"call0","content":"done"})"));
    next = with(next, "messages", Json(std::move(messages)));
    const auto second = append.turn(bundle(next, response(), 1));
    check(field(second, "passed").boolean() && field(second, "canonical_append_candidate").boolean());
    check(field(second, "canonical_append_input_scalar_steps").unsigned_integer() > 0);
    check(!field(second, "actual_append_proven").boolean());
    check(field(second, "previous_request_token_lcp").unsigned_integer() == field(first, "native_prompt_tokens").unsigned_integer());
    for (const auto& changed : {with(next, "seed", Json::integer(45)), with(next, "temperature", Json::integer(0))}) {
        TranscriptReplay policy(codec(), "tokspur"); check(field(policy.turn(bundle()), "passed").boolean());
        const auto incompatible = policy.turn(bundle(changed, response(), 1));
        check(field(incompatible, "passed").boolean() && field(incompatible, "canonical_token_prefix_candidate").boolean());
        check(!field(incompatible, "canonical_append_candidate").boolean() && field(incompatible, "canonical_append_input_scalar_steps").is_null());
    }
    const auto prose = parse_json(R"({"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]})");
    for (const auto& mismatch : {finished(prose, "tool_calls"), finished(response(), "stop")}) {
        const auto refused = replay.turn(bundle(wire(), mismatch));
        check(!field(refused, "passed").boolean() && !field(refused, "response_postconditions_admitted").boolean());
    }
    const auto required = replay.turn(bundle(with(wire(), "tool_choice", Json("required")), prose));
    check(!field(required, "passed").boolean() && !field(required, "response_postconditions_admitted").boolean());
    check(field(replay.turn(bundle(with(wire(), "tool_choice", Json("none")), prose)), "passed").boolean());
    check(!field(replay.turn(bundle(with(wire(), "tool_choice", Json("none")))), "passed").boolean());
    const auto named = with(wire(), "tool_choice", parse_json(R"({"type":"function","function":{"name":"run"}})"));
    check(field(replay.turn(bundle(named)), "passed").boolean());
    const auto other = parse_json(R"({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"other0","type":"function","function":{"name":"other","arguments":"{}"}}]}}]})");
    check(!field(replay.turn(bundle(named, other)), "passed").boolean());
    TranscriptReplay executor(codec(), "tokspur", ToolEnumValidation::executor);
    constexpr std::string_view literal = "cargo test --offline 2>&1 | head -60";
    const auto invalid_enum = response(R"({"command":"cargo test --offline 2>&1 | head -60"})");
    const auto transported = executor.turn(bundle(wire(), invalid_enum));
    check(field(transported,"passed").boolean() && field(transported,"response_tool_arguments_admitted").boolean());
    check(field(transported,"tool_enum_validation").string() == "executor" &&
        field(transported,"tool_enum_validation_policy").string() == "original-agent-executor-enums-v1" &&
        field(transported,"counterfactual_executor_enum_policy").boolean());
    const auto& literal_call=field(field(field(invalid_enum,"choices").array()[0],"message"),"tool_calls").array()[0];
    check(parse_json(field(field(literal_call,"function"),"arguments").string()).find("command")->string() == literal);
    auto error_history = field(wire(),"messages").array();
    error_history.push_back(field(field(invalid_enum,"choices").array()[0],"message"));
    error_history.push_back(parse_json(R"({"role":"tool","tool_call_id":"call0","content":"{\"error\":\"command not allowed\"}"})"));
    const auto errored = with(wire(),"messages",Json(error_history));
    check(field(executor.turn(bundle(errored,response(),1)),"passed").boolean());
    check(!field(replay.turn(bundle(errored,response(),1)),"passed").boolean());
    for(const auto& [bad, allowed] : {std::pair{R"({"command":1})",false},std::pair{R"({"command":"bad","extra":true})",false},std::pair{R"({})",false},std::pair{R"({"command":"<|im_start|>"})",true},std::pair{R"({"command":"<|endoftext|>"})",true}})
        check(field(executor.turn(bundle(wire(),response(bad))),"passed").boolean() == allowed);
    check(!field(executor.turn(bundle(named,other)),"passed").boolean());
    check(!field(executor.turn(bundle(with(wire(),"tool_choice",Json("required")),prose)),"passed").boolean());
    const auto executor_length=executor.turn(bundle(wire(),finished(invalid_enum,"length")));
    check(field(executor_length,"canonical_response_parser_passed").boolean() && field(executor_length,"canonical_tool_records").unsigned_integer()==0);
    for(const auto key : {"tool_enum_validation","tool-enum-validation"})
        check(!field(executor.turn(bundle(with(wire(),key,Json("executor")))) ,"passed").boolean());
}
}
int main() {
    try { fixtures(); std::cout << "offline transcript replay synthetic checks=" << checks << '\n'; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
