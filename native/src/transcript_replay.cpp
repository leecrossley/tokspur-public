#include "tokspur/transcript_replay.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <set>

namespace tokspur {
namespace {
void need(bool ok, const char* message) { if (!ok) throw FormatError(message); }
const Json& field(const Json& value, std::string_view key) {
    const auto* found = value.find(key); need(found, "missing replay field"); return *found;
}
void keys(const Json& value, const std::set<std::string_view>& allowed) {
    for (const auto& [key, ignored] : value.object()) { static_cast<void>(ignored); need(allowed.contains(key), "unknown replay field"); }
}
Json with(Json value, std::string_view key, Json replacement) {
    auto fields = value.object();
    for (auto& [name, item] : fields) if (name == key) { item = std::move(replacement); return Json(std::move(fields)); }
    fields.emplace_back(key, std::move(replacement)); return Json(std::move(fields));
}
size_t common(std::span<const uint32_t> a, std::span<const uint32_t> b) {
    size_t n = 0; while (n < a.size() && n < b.size() && a[n] == b[n]) ++n; return n;
}
void response_arguments(const Json& message, const AdmittedChat& admitted, ToolEnumValidation tool_enums) {
    keys(message, {"role", "content", "reasoning_content", "tool_calls"});
    need(field(message, "role").string() == "assistant", "recorded response is not assistant");
    if (const auto* calls = message.find("tool_calls")) {
        need(calls->array().size() <= 128, "replay response tool count");
        for (const auto& call : calls->array()) {
            keys(call, {"id", "type", "function"});
            static_cast<void>(field(call, "id").string());
            need(field(call, "type").string() == "function", "unsupported replay response call type");
            const auto& function = field(call, "function"); keys(function, {"name", "arguments"});
            const auto name = field(function, "name").string();
            const auto at = std::find(admitted.names.begin(), admitted.names.end(), name);
            need(at != admitted.names.end(), "unknown recorded response tool");
            const auto& value = field(function, "arguments");
            const auto arguments = value.kind() == Json::Kind::string ? parse_json(value.string(), {65536, 16384, 32}) : value;
            validate_tool_value(admitted.schemas.at(size_t(at - admitted.names.begin())), arguments, tool_enums);
        }
    }
}
}

TranscriptReplay::TranscriptReplay(ServeCodec codec, std::string alias, ToolEnumValidation tool_enums)
    : codec_(std::move(codec)), alias_(std::move(alias)), tool_enums_(tool_enums) {
    static_cast<void>(tool_enum_validation_name(tool_enums_));
    need(bool(codec_.render) && bool(codec_.encode) && bool(codec_.decode) && !alias_.empty(), "replay requires explicit codec/alias");
}

Json TranscriptReplay::turn(const Json& record) {
    keys(record, {"turn", "wire_request", "effective_messages", "response"});
    Json::Object out{{"event", Json("transcript_replay")}, {"turn", Json::integer(field(record, "turn").unsigned_integer())},
        {"scope", Json("offline structured-message diagnostic; canonical reconstruction only")},
        {"tool_enum_validation", Json(std::string(tool_enum_validation_name(tool_enums_)))},
        {"tool_enum_validation_policy", Json(std::string(tool_enum_validation_policy(tool_enums_)))},
        {"counterfactual_executor_enum_policy", Json(tool_enums_ == ToolEnumValidation::executor)},
        {"tools_executed", Json(false)}, {"model_loaded", Json(false)}, {"exact_generated_ids_observed", Json(false)},
        {"raw_response_bytes_observed", Json(false)}, {"raw_stop_id_observed", Json(false)},
        {"actual_append_proven", Json(false)}, {"actual_input_scalar_steps", Json()}, {"actual_output_scalar_steps", Json()}};
    Json::Array refusals;
    const auto attempt = [&](std::string_view stage, const auto& operation) {
        try { operation(); return true; }
        catch (const FormatError& error) {
            refusals.emplace_back(Json::Object{{"stage", Json(std::string(stage))}, {"message", Json(std::string(error.what()).substr(0, 2048))}});
            return false;
        }
    };
    const auto& wire = field(record, "wire_request");
    out.emplace_back("wire_request_admitted", Json(attempt("wire_request", [&] {
        const auto admitted_wire = admit_chat(wire, alias_, tool_enums_);
        const auto ids = codec_.encode(codec_.render(admitted_wire.render));
        out.emplace_back("wire_native_prompt_tokens", Json::integer(ids.size()));
        need(!ids.empty() && uint64_t(ids.size()) + admitted_wire.maximum_tokens <= 65536, "wire prompt plus reserved output exceeds 65536");
    })));
    std::optional<AdmittedChat> admitted; std::vector<uint32_t> prompt, canonical; std::string prompt_bytes;
    RenderedChat rendered_prompt;
    std::optional<uint64_t> canonical_seed;
    const auto effective = with(wire, "messages", field(record, "effective_messages"));
    const bool input_ok = attempt("effective_request", [&] {
        admitted = admit_chat(effective, alias_, tool_enums_);
        rendered_prompt = codec_.render(admitted->render); prompt_bytes = rendered_prompt.prompt; prompt = codec_.encode(rendered_prompt);
        need(!prompt.empty(), "empty replay prompt");
        for (const auto id : prompt) need(id < codec_.admission.vocabulary, "replay prompt token outside vocabulary");
        out.emplace_back("native_prompt_tokens", Json::integer(prompt.size()));
        out.emplace_back("reserved_output_tokens", Json::integer(admitted->maximum_tokens));
        out.emplace_back("projected_cold_input_scalar_steps", Json::integer(prompt.size()));
        out.emplace_back("previous_request_token_lcp", Json::integer(common(previous_prompt_, prompt)));
        const bool token_prefix = !previous_canonical_committed_.empty() &&
            common(previous_canonical_committed_, prompt) == previous_canonical_committed_.size();
        const bool policy_matches = previous_sampling_ == admitted->sampling.dump();
        const bool seed_known = !admitted->seed || previous_seed_.has_value();
        const bool seed_matches = !admitted->seed || (previous_seed_ && *admitted->seed == *previous_seed_);
        const bool hypothetical = token_prefix && policy_matches && seed_matches;
        out.emplace_back("canonical_token_prefix_candidate", Json(token_prefix));
        out.emplace_back("canonical_sampling_policy_matches", Json(policy_matches));
        out.emplace_back("canonical_seed_compatible", seed_known ? Json(seed_matches) : Json());
        out.emplace_back("canonical_append_candidate", Json(hypothetical));
        out.emplace_back("canonical_append_input_scalar_steps", hypothetical ? Json::integer(prompt.size() - previous_canonical_committed_.size()) : Json());
        canonical_seed = admitted->seed ? admitted->seed : hypothetical ? previous_seed_ : std::nullopt;
        need(uint64_t(prompt.size()) + admitted->maximum_tokens <= 65536, "replay prompt plus reserved output exceeds 65536");
    });
    out.emplace_back("effective_request_admitted", Json(input_ok));
    previous_prompt_ = prompt; previous_canonical_committed_.clear(); previous_sampling_.clear(); previous_seed_.reset();
    const auto& response = field(record, "response");
    attempt("reported_usage", [&] {
        if (const auto* usage = response.find("usage")) {
            if (const auto* n = usage->find("prompt_tokens")) out.emplace_back("reported_prompt_tokens", Json::integer(n->unsigned_integer()));
            if (const auto* n = usage->find("completion_tokens")) out.emplace_back("reported_completion_tokens", Json::integer(n->unsigned_integer()));
        }
    });
    if (input_ok) {
        const Json* message = nullptr; std::string finish;
        const bool envelope_ok = attempt("response_envelope", [&] {
            const auto& choices = field(response, "choices").array(); need(choices.size() == 1, "one replay response choice required");
            message = &field(choices[0], "message"); finish = field(choices[0], "finish_reason").string();
            need(finish == "stop" || finish == "tool_calls" || finish == "length", "unsupported recorded response finish reason");
        });
        const bool conditions_ok = envelope_ok && attempt("response_postconditions", [&] {
            const auto* calls = message->find("tool_calls"); const auto count = calls ? calls->array().size() : 0;
            need(finish == "length" || (finish == "tool_calls") == bool(count), "recorded finish reason disagrees with tool calls");
            need(admitted->tool_choice == "auto" || admitted->tool_choice == "none" || finish == "length" || count,
                "required or named tool choice produced no stopped tool call");
        });
        out.emplace_back("response_postconditions_admitted", Json(conditions_ok));
        const bool arguments_ok = envelope_ok && attempt("response_tool_arguments", [&] { response_arguments(*message, *admitted, tool_enums_); });
        out.emplace_back("response_tool_arguments_admitted", Json(arguments_ok));
        const bool parser_ok = envelope_ok && attempt("canonical_response_parser", [&] {
            auto rendered = codec_.render(Json(Json::Object{{"messages", Json(Json::Array{*message})}, {"enable_thinking", Json(false)},
                {"preserve_thinking", Json(true)}, {"add_generation_prompt", Json(false)}}));
            constexpr std::string_view header = "<|im_start|>assistant\n", tail = "<|im_end|>\n", thinking = "<think>\n";
            need(rendered.prompt.starts_with(header) && rendered.prompt.ends_with(tail), "canonical assistant envelope differs");
            size_t first = header.size(), last = rendered.prompt.size() - 1;
            auto body = std::string_view(rendered.prompt).substr(first, last - first);
            if (admitted->thinking) { need(body.starts_with(thinking), "canonical thinking prefix differs"); first += thinking.size(); }
            else if (body.starts_with("<think>\n\n</think>\n\n")) first += std::string_view("<think>\n\n</think>\n\n").size();
            if (finish == "length") last -= std::string_view("<|im_end|>").size();
            body = std::string_view(rendered.prompt).substr(first, last - first);
            RenderedChat canonical_chat; canonical_chat.prompt = body;
            for (const auto& [start, end] : rendered.literal_ranges) {
                const auto lo = std::max(first, start), hi = std::min(last, end);
                if (lo < hi) canonical_chat.literal_ranges.emplace_back(lo - first, hi - first);
            }
            canonical = codec_.encode(canonical_chat);
            need(canonical.size() <= response_maximum_accumulated_ids && codec_.decode(canonical) == body, "canonical response IDs/byte roundtrip exceeds HTTP limit");
            ResponseParser parser(admitted->manifest, admitted->thinking, {}, false, true);
            const auto parsed_body = finish == "length" ? body : body.substr(0, body.size() - std::string_view("<|im_end|>").size());
            if (parser.feed(parsed_body) == ResponseStatus::error) throw FormatError("canonical response grammar: " + std::string(parser.error()));
            if (finish != "length" && parser.finish() != ResponseStatus::complete) throw FormatError("canonical response finish: " + std::string(parser.error()));
            out.emplace_back("canonical_output_tokens", Json::integer(canonical.size()));
            out.emplace_back("canonical_output_bytes", Json::integer(body.size()));
            out.emplace_back("canonical_output_scalar_steps", Json::integer(canonical.size()));
            auto split = prompt; split.insert(split.end(), canonical.begin(), canonical.end());
            auto joint = rendered_prompt; joint.prompt += body;
            for (const auto& [start, end] : canonical_chat.literal_ranges) joint.literal_ranges.emplace_back(prompt_bytes.size() + start, prompt_bytes.size() + end);
            out.emplace_back("canonical_split_matches_joint_tokenization", Json(split == codec_.encode(joint)));
            out.emplace_back("canonical_terminal_synthesized", Json(finish != "length"));
            out.emplace_back("canonical_response_complete", Json(parser.status() == ResponseStatus::complete));
            out.emplace_back("canonical_tool_records", Json::integer(size_t(std::count_if(parser.records().begin(), parser.records().end(),
                [](const auto& row) { return row.kind == ResponseKind::tool; }))));
        });
        out.emplace_back("canonical_response_parser_passed", Json(parser_ok));
        if (arguments_ok && parser_ok && finish != "length" && refusals.empty()) {
            previous_canonical_committed_ = prompt;
            previous_canonical_committed_.insert(previous_canonical_committed_.end(), canonical.begin(), canonical.end());
            previous_sampling_ = admitted->sampling.dump(); previous_seed_ = canonical_seed;
        }
    }
    out.emplace_back("passed", Json(refusals.empty())); out.emplace_back("refusals", Json(std::move(refusals)));
    return Json(std::move(out));
}
} // namespace tokspur
