#include "tokspur/serve.hpp"
#if defined(TOKSPUR_PRIVATE_LANE_LAYOUT) || defined(TOKSPUR_PRIVATE_Q8_THREE_ROWS)
#include "tokspur/owned_worker.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#include "tokspur/mtp_wire.hpp"
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
#include "tokspur/raw_decode_wire.hpp"
#endif
#endif
#include "tokspur/gguf.hpp"
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
#include "tokspur/serve_verify8.hpp"
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cmath>
#include <chrono>
#include <map>
#include <limits>
#include <set>

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
const Json& field(const Json& value, std::string_view key) {
    const auto* p = value.find(key); require(p, "missing required API field"); return *p;
}
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
Json project_weight_lanes(Json value, const Json& identity) {
    if (identity.is_null()) return value; // Preserve unselected response bytes and shape.
    auto out = value.object();
    Json::Object details;
    if (const auto* present = value.find("tokspur_private")) details = present->object();
    for (const auto name : owned_weight_lane_fields) details.emplace_back(std::string(name), field(identity, name));
    if (identity.find(owned_lane_auth_fields[0])) for (const auto name : owned_lane_auth_fields)
        details.emplace_back(std::string(name), field(identity, name));
    for (auto& [name, item] : out) if (name == "tokspur_private") { item = Json(std::move(details)); return Json(std::move(out)); }
    out.emplace_back("tokspur_private", Json(std::move(details))); return Json(std::move(out));
}
#endif
void keys(const Json& value, std::initializer_list<std::string_view> allowed) {
    for (const auto& [key, item] : value.object()) {
        static_cast<void>(item);
        require(std::find(allowed.begin(), allowed.end(), key) != allowed.end(), "unsupported API/schema field");
    }
}
double number(const Json& value, double low, double high) {
    const auto text = value.number_lexeme(); double result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(result) && result >= low && result <= high,
        "API number outside supported range"); return result;
}
Json real(double value) {
    std::array<char, 64> text{}; const auto r = std::to_chars(text.data(), text.data() + text.size(), value);
    require(r.ec == std::errc{}, "cannot encode API number"); return parse_json({text.data(), size_t(r.ptr - text.data())});
}
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
struct PrivateMtpRequestTrace {
    struct ClockSum {
        double sum = 0;
        uint32_t expected = 0, observed = 0;
        void add(const Json& source, std::string_view name) {
            require(expected < 65536, "private native trace sample bound"); ++expected;
            if (const auto* value = source.find(name)) {
                const auto sample = number(*value, 0, std::numeric_limits<double>::max());
                require(sample <= std::numeric_limits<double>::max() - sum, "private native trace clock overflow");
                sum += sample; ++observed;
            }
        }
        Json report() const {
            return Json(Json::Object{{"total_ms", expected && observed == expected ? real(sum) : Json()},
                {"observed_samples", Json::integer(observed)}, {"expected_samples", Json::integer(expected)}});
        }
    };
    inline static constexpr std::array<std::string_view, 8> round_clocks{"draft_wall_ms", "draft_gpu_ms", "verification_wall_ms", "verification_gpu_ms",
        "recovery_wall_ms", "recovery_gpu_ms", "materialization_wall_ms", "round_wall_ms"};
    inline static constexpr std::array<std::string_view, 4> counter_names{"native_mtp_rounds", "native_mtp_proposed", "native_mtp_accepted", "native_mtp_recovery_rounds"};
    ClockSum input_request, input_prefill, output_request, output_generation_wall, output_generation_gpu;
    std::array<ClockSum, round_clocks.size()> components{};
    std::array<uint64_t, counter_names.size()> counters{};
    uint64_t round_committed_tokens = 0, ordinary_tokens = 0, ordinary_tail_tokens = 0;
    void admit(const Json& previous, const Json& done, const Json& job, std::span<const Json> events) {
        const bool known_input = !field(job, "tokens").array().empty();
        if (known_input) {
            input_request.add(done, "request_wall_ms"); input_prefill.add(done, "prefill_wall_ms"); return;
        }
        output_request.add(done, "request_wall_ms"); output_generation_wall.add(done, "generation_batch_wall_ms"); output_generation_gpu.add(done, "generation_batch_gpu_ms");
        // A cold input reset may catch up unadmitted lifetime work. Count output deltas only against this packet's current admitted predecessor.
        for (size_t i = 0; i < counters.size(); ++i) {
            const auto before = field(previous, counter_names[i]).unsigned_integer(), after = field(done, counter_names[i]).unsigned_integer();
            require(after >= before && after - before <= 131072 && counters[i] <= 131072 - (after - before), "private native trace counter bound");
            counters[i] += after - before;
        }
        uint64_t round_tokens = 0, packet_tokens = 0, packet_ordinary = 0, packet_tails = 0;
        const auto maximum = field(job, "generate").unsigned_integer();
        for (const auto& event : events) {
            if (field(event, "event").string() == "mtp_round") {
                round_tokens += field(event, "count").unsigned_integer();
                for (size_t i = 0; i < components.size(); ++i) components[i].add(event, round_clocks[i]);
            } else {
                require(packet_tokens < maximum, "private native trace packet token bound");
                if (!event.find("round_committed_prefix")) { ++packet_ordinary; packet_tails += maximum - packet_tokens <= mtp_depth(done); }
                ++packet_tokens;
            }
        }
        const auto outputs = field(done, "generated_ids").array().size();
        require(packet_tokens == outputs && round_tokens <= outputs && packet_ordinary == outputs - round_tokens &&
            round_committed_tokens <= 65536 - round_tokens && ordinary_tokens <= 65536 - packet_ordinary &&
            ordinary_tail_tokens <= 65536 - packet_tails, "private native trace ordinary-token bound");
        round_committed_tokens += round_tokens; ordinary_tokens += packet_ordinary; ordinary_tail_tokens += packet_tails;
    }
    Json report() const {
        const bool complete_round_wall = counters[0] && components[7].expected == counters[0] && components[7].observed == counters[0];
        Json::Object rounds{{"rounds", Json::integer(counters[0])}, {"proposed_tokens", Json::integer(counters[1])},
            {"accepted_tokens", Json::integer(counters[2])}, {"round_committed_tokens", Json::integer(round_committed_tokens)},
            {"recovery_rounds", Json::integer(counters[3])}, {"ordinary_tokens", Json::integer(ordinary_tokens)},
            {"ordinary_tail_tokens", Json::integer(ordinary_tail_tokens)}, {"ordinary_non_tail_tokens", Json::integer(ordinary_tokens - ordinary_tail_tokens)},
            {"committed_tokens_per_round", counters[0] ? real(double(round_committed_tokens) / double(counters[0])) : Json()},
            {"wall_ms_per_round", complete_round_wall ? real(components[7].sum / double(counters[0])) : Json()},
            {"round_tokens_per_sec", (complete_round_wall && components[7].sum > 0.0) ? real((double(round_committed_tokens) * 1000.0) / components[7].sum) : Json()}};
        for (size_t i = 0; i < components.size(); ++i) rounds.emplace_back(std::string(round_clocks[i]), components[i].report());
        rounds.emplace_back("materialization_gpu_ms", Json()); rounds.emplace_back("round_gpu_ms", Json());
        return Json(Json::Object{{"packet_wall_scope", Json("admitted native worker packets, including each first output packet")},
            {"generation_batch_scope", Json("worker-reported generation batches, including MTP priming and ordinary tails")},
            {"round_component_scope", Json("admitted MTP round events, incomplete as a whole-generation GPU clock")},
            {"ordinary_tail_scope", Json("ordinary output tokens with at most native depth remaining in their output packet")},
            {"isolated_decode_measured", Json(false)}, {"timings_qualified", Json(false)},
            {"input_packets", Json(Json::Object{{"count", Json::integer(input_request.expected)}, {"request_wall_ms", input_request.report()}, {"prefill_wall_ms", input_prefill.report()}})},
            {"output_packets", Json(Json::Object{{"count", Json::integer(output_request.expected)}, {"request_wall_ms", output_request.report()},
                {"generation_batch_wall_ms", output_generation_wall.report()}, {"generation_batch_gpu_ms", output_generation_gpu.report()}})}, {"mtp", Json(std::move(rounds))}});
    }
};
#endif
struct Decimal { bool negative; std::string digits; int64_t exponent; };
Decimal decimal(const Json& value) {
    const auto text = value.number_lexeme(); require(text.size() <= 128, "tool numeric lexeme bound");
    size_t i = text.front() == '-' ? 1 : 0; const bool negative = i != 0;
    std::string digits; int64_t fraction = 0; bool after_point = false;
    for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; ++i) {
        if (text[i] == '.') after_point = true;
        else { digits += text[i]; if (after_point) ++fraction; }
    }
    int64_t exponent = 0;
    if (i < text.size()) {
        ++i; bool minus = false;
        if (text[i] == '-' || text[i] == '+') { minus = text[i] == '-'; ++i; }
        const auto r = std::from_chars(text.data() + i, text.data() + text.size(), exponent);
        require(r.ec == std::errc{} && r.ptr == text.data() + text.size() && exponent <= 1000000, "tool numeric exponent bound");
        if (minus) exponent = -exponent;
    }
    exponent -= fraction;
    const auto first = digits.find_first_not_of('0');
    if (first == std::string::npos) return {false, "0", 0};
    digits.erase(0, first);
    while (digits.back() == '0') { digits.pop_back(); ++exponent; }
    return {negative, std::move(digits), exponent};
}
int compare(const Decimal& left, const Decimal& right) {
    if (left.digits == "0" && right.digits == "0") return 0;
    if (left.negative != right.negative) return left.negative ? -1 : 1;
    int order = 0;
    if (left.digits == "0") order = -1;
    else if (right.digits == "0") order = 1;
    else {
        const int64_t l = int64_t(left.digits.size()) + left.exponent, r = int64_t(right.digits.size()) + right.exponent;
        if (l != r) order = l < r ? -1 : 1;
        else for (size_t i = 0; i < std::max(left.digits.size(), right.digits.size()); ++i) {
            const char a = i < left.digits.size() ? left.digits[i] : '0', b = i < right.digits.size() ? right.digits[i] : '0';
            if (a != b) { order = a < b ? -1 : 1; break; }
        }
    }
    return left.negative ? -order : order;
}
void safe_name(std::string_view name, bool function) {
    require(!name.empty() && name.size() <= (function ? 64 : 1024), "tool name bound");
    for (const unsigned char c : name) {
        require(c >= 32 && c < 127 && c != '<' && c != '>', "unsupported tool name");
        if (function) require(std::isalnum(c) || c == '_' || c == '-' || c == '.', "unsupported function name");
    }
}
std::string_view schema_type(const Json& schema) {
    if (schema.find("anyOf")) return "any";
    const auto value = field(schema, "type").string();
    require(value == "string" || value == "boolean" || value == "number" || value == "integer" || value == "object" || value == "array" || value == "null",
        "unsupported tool schema type"); return value;
}
void schema_admit(const Json& schema, size_t depth, size_t& nodes) {
    require(depth <= 16 && ++nodes <= 4096, "tool schema complexity bound");
    keys(schema, {"type", "anyOf", "title", "default", "examples", "format", "$schema", "description", "properties", "required", "additionalProperties", "items", "enum", "minimum", "exclusiveMinimum", "maximum"});
    const auto type = schema_type(schema);
    if (const auto* dialect = schema.find("$schema")) require(depth == 0 && dialect->string() == "https://json-schema.org/draft/2020-12/schema", "unsupported tool schema dialect");
    if (const auto* p = schema.find("description")) require(p->string().size() <= 16384, "schema description bound");
    for (const auto name : {"title", "format"}) if (const auto* value = schema.find(name)) require(value->string().size() <= 16384, "schema annotation bound");
    if (const auto* value = schema.find("examples")) require(value->array().size() <= 128, "schema examples bound");
    if (type == "any") {
        keys(schema, {"anyOf", "$schema", "title", "description", "default", "examples", "format"});
        const auto& alternatives = field(schema, "anyOf").array();
        require(!alternatives.empty() && alternatives.size() <= 8, "schema union bound");
        for (const auto& branch : alternatives) schema_admit(branch, depth + 1, nodes);
        return;
    }
    if (type == "object") {
        const auto* properties = schema.find("properties");
        if (properties) {
            require(properties->object().size() <= 32, "too many tool parameters");
            for (const auto& [name, child] : properties->object()) { safe_name(name, false); schema_admit(child, depth + 1, nodes); }
        }
        if (const auto* additional = schema.find("additionalProperties")) require(!additional->boolean(), "additional tool properties are unsupported");
        if (const auto* required = schema.find("required")) {
            std::set<std::string_view> seen;
            for (const auto& item : required->array()) {
                const auto name = item.string(); require(properties && properties->find(name) && seen.insert(name).second, "invalid required tool parameter");
            }
        }
    } else require(!schema.find("properties") && !schema.find("required") && !schema.find("additionalProperties"), "object schema keyword on other type");
    if (type == "array") schema_admit(field(schema, "items"), depth + 1, nodes);
    else require(!schema.find("items"), "array schema keyword on other type");
    if (const auto* values = schema.find("enum")) {
        require(type != "object" && type != "array" && !values->array().empty() && values->array().size() <= 128, "unsupported enum");
        // Admission checks enum members against the same type, without enum.
        Json::Object base; for (const auto& [key, value] : schema.object()) if (key != "enum") base.emplace_back(key, value);
        for (const auto& value : values->array()) validate_tool_value(Json(base), value);
    }
    if (schema.find("minimum") || schema.find("maximum") || schema.find("exclusiveMinimum")) {
        require(type == "integer" || type == "number", "numeric bound on other schema type");
        if (const auto* low = schema.find("minimum")) { static_cast<void>(number(*low, -1e300, 1e300)); static_cast<void>(decimal(*low)); }
        if (const auto* high = schema.find("maximum")) { static_cast<void>(number(*high, -1e300, 1e300)); static_cast<void>(decimal(*high)); }
        if (const auto* low = schema.find("exclusiveMinimum")) {
            static_cast<void>(number(*low, -1e300, 1e300)); static_cast<void>(decimal(*low));
            if (const auto* high = schema.find("maximum")) require(compare(decimal(*low), decimal(*high)) < 0, "inverted schema bounds");
        }
        if (schema.find("minimum") && schema.find("maximum")) require(compare(decimal(*schema.find("minimum")), decimal(*schema.find("maximum"))) <= 0, "inverted schema bounds");
    }
}
ParameterType parameter_type(std::string_view type) {
    if (type == "string") return ParameterType::raw_string;
    if (type == "boolean") return ParameterType::boolean;
    if (type == "number" || type == "integer") return ParameterType::number;
    if (type == "object") return ParameterType::object;
    if (type == "array") return ParameterType::array;
    if (type == "any") return ParameterType::json_value;
    return ParameterType::null_value;
}
void validate_executor_string(std::string_view raw) {
    require(raw.find('\0') == std::string_view::npos, "NUL byte inside tool parameter");
}
void validate_executor_node(const Json& value, size_t depth, size_t& nodes) {
    require(depth <= 32 && ++nodes <= 16384, "executor tool argument complexity bound");
    if (value.kind() == Json::Kind::string) {
        validate_executor_string(value.string());
    } else if (value.kind() == Json::Kind::number) {
        static_cast<void>(number(value, -1e300, 1e300));
    } else if (value.kind() == Json::Kind::array) {
        require(value.array().size() <= 4096, "tool array bound");
        for (const auto& item : value.array()) validate_executor_node(item, depth + 1, nodes);
    } else if (value.kind() == Json::Kind::object) {
        require(value.object().size() <= 256, "too many tool parameters");
        std::set<std::string_view> seen;
        for (const auto& [key, item] : value.object()) {
            if (depth == 0) safe_name(key, false);
            require(seen.insert(key).second, "duplicate executor tool property");
            validate_executor_string(key);
            validate_executor_node(item, depth + 1, nodes);
        }
    }
}
void thinking_history(const Json& message, std::string_view role) {
    const auto* blocks = message.find("thinking_blocks");
    if (!blocks) return;
    require(role == "assistant", "thinking blocks require assistant history");
    require(blocks->array().size() <= 4096, "thinking block count bound");
    std::string text;
    for (const auto& block : blocks->array()) {
        keys(block, {"type", "thinking", "signature"});
        require(field(block, "type").string() == "thinking", "unsupported thinking block type");
        if (const auto* signature = block.find("signature")) require(signature->string().empty(), "signed thinking blocks are unsupported");
        const auto part = field(block, "thinking").string();
        require(part.size() <= 1024 * 1024 - text.size(), "thinking history bound");
        text += part;
    }
    const auto* reasoning = message.find("reasoning_content");
    require(text == (reasoning && !reasoning->is_null() ? reasoning->string() : std::string_view{}), "thinking history differs from reasoning content");
}
void history(const Json& messages, const std::vector<std::string>& tools, const std::vector<Json>& schemas, ToolEnumValidation tool_enums, ToolArgValidation tool_args) {
    require(!messages.array().empty() && messages.array().size() <= 4096, "API history bound");
    std::map<std::string, std::string, std::less<>> pending;
    std::set<std::string, std::less<>> used;
    for (const auto& message : messages.array()) {
        keys(message, {"role", "content", "reasoning_content", "thinking_blocks", "tool_calls", "tool_call_id", "name"});
        const auto role = field(message, "role").string();
        require(role == "system" || role == "developer" || role == "user" || role == "assistant" || role == "tool", "unsupported API message role");
        thinking_history(message, role);
        if (const auto* content = message.find("content")) {
            if (content->kind() == Json::Kind::array) {
                require(content->array().size() <= 4096, "text content part bound");
                for (const auto& part : content->array()) { keys(part, {"type", "text"}); require(field(part, "type").string() == "text", "only text content parts supported"); static_cast<void>(field(part, "text").string()); }
            } else require(content->is_null() || content->kind() == Json::Kind::string, "unsupported message content");
        }
        if (role == "tool") {
            require(!message.find("tool_calls") && !message.find("reasoning_content"), "invalid tool result fields");
            require(!field(message, "content").is_null(), "tool result requires text content");
            const auto id = field(message, "tool_call_id").string(); const auto found = pending.find(id);
            require(found != pending.end(), "unsolicited or duplicate tool result");
            if (const auto* name = message.find("name")) require(name->string() == found->second, "tool result name/ID disagreement");
            pending.erase(found);
        } else {
            require(pending.empty(), "missing tool results before next message");
            require(!message.find("tool_call_id") && !message.find("name"), "tool identity field outside tool result");
            if (role != "assistant") require(!field(message, "content").is_null(), "message requires text content");
            if (const auto* reasoning = message.find("reasoning_content")) require(role == "assistant" && (reasoning->is_null() || reasoning->kind() == Json::Kind::string), "invalid reasoning history");
            if (const auto* calls = message.find("tool_calls")) {
                require(role == "assistant" && !calls->array().empty() && calls->array().size() <= 128, "invalid assistant tool history");
                for (const auto& call : calls->array()) {
                    keys(call, {"id", "type", "function", "index"});
                    if (const auto* index = call.find("index")) require(index->unsigned_integer() < 128, "historical tool index bound");
                    require(field(call, "type").string() == "function", "unsupported historical call type");
                    const auto id = std::string(field(call, "id").string()); require(!id.empty() && id.size() <= 128 && used.insert(id).second, "invalid/duplicate historical tool ID");
                    const auto& function = field(call, "function"); keys(function, {"name", "arguments"});
                    const auto name = field(function, "name").string();
                    const auto found = std::find(tools.begin(), tools.end(), name); require(found != tools.end(), "historical function absent from admitted tools");
                    const auto& arguments = field(function, "arguments"); require(arguments.kind() == Json::Kind::object || arguments.kind() == Json::Kind::string, "invalid historical tool arguments");
                    const Json parsed = arguments.kind() == Json::Kind::string ? parse_json(arguments.string(), {65536, 16384, 32}) : arguments;
                    if (tool_args == ToolArgValidation::executor) validate_executor_tool_arguments(parsed);
                    else validate_tool_value(schemas.at(size_t(found - tools.begin())), parsed, tool_enums);
                    pending.emplace(id, name);
                }
            }
        }
    }
    require(pending.empty(), "request ends with unanswered tool calls");
}
Json parser_manifest(const std::vector<ToolSpec>& tools) {
    Json::Array result;
    for (const auto& tool : tools) {
        Json::Array parameters;
        for (const auto& p : tool.parameters) {
            static constexpr const char* types[]{"string", "boolean", "number", "object", "array", "null", "any"};
            parameters.emplace_back(Json::Object{{"name", Json(p.name)}, {"type", Json(types[uint32_t(p.type)])}, {"required", Json(p.required)}});
        }
        result.emplace_back(Json::Object{{"name", Json(tool.name)}, {"parameters", Json(std::move(parameters))}});
    }
    return Json(std::move(result));
}
Json chunk(std::string_view id, std::string_view alias, uint64_t created, Json delta, Json finish = {}) {
    return Json(Json::Object{{"id", Json(std::string(id))}, {"object", Json("chat.completion.chunk")}, {"created", Json::integer(created)},
        {"model", Json(std::string(alias))}, {"choices", Json(Json::Array{Json(Json::Object{{"index", Json::integer(0)}, {"delta", std::move(delta)}, {"finish_reason", std::move(finish)}})})}});
}
void worker_execution(const Json& value, std::string_view selected) {
    require(selected == "direct" || selected == "recurrent-replay", "unsupported serving startup execution mode");
    require(field(value, "small_gemv_column_splits").unsigned_integer() == 16 &&
        field(value, "block_gemv_formats").unsigned_integer() == 3 &&
        field(value, "stack_execution").string() == selected, "worker execution policy changed");
    if (selected == "recurrent-replay") {
        require(field(value, "replay_prepared").boolean() &&
            field(value, "recurrent_replay_captures").unsigned_integer() == 24 &&
            field(value, "mapped_weight_region_limit_bytes").unsigned_integer() == uint64_t(4) * 1024 * 1024 * 1024,
            "owned recurrent replay preparation/capture/region identity mismatch");
    }
}
void broker_storage(const Json& value, const Json& ready) {
    const auto slots = field(ready, "checkpoint_slots").unsigned_integer();
    const auto budget = field(ready, "checkpoint_combined_budget_bytes").unsigned_integer();
    require(slots && slots <= 4 && budget && budget <= (512ull << 20), "invalid serving broker storage preparation");
    require(field(value, "checkpoint_slots").unsigned_integer() == slots &&
        field(value, "checkpoint_combined_budget_bytes").unsigned_integer() == budget, "serving broker configuration changed");
    uint64_t total = 0;
    for (const auto name : {"checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes"}) {
        const auto bytes = field(value, name).unsigned_integer();
        require(bytes && bytes <= budget - total && bytes == field(ready, name).unsigned_integer(), "serving broker storage charge changed");
        total += bytes;
    }
}
} // namespace

void validate_worker_ready(const Json& ready, std::string_view stack_execution, uint32_t checkpoint_slots, uint64_t checkpoint_budget) {
    require(field(ready, "event").string() == "ready" &&
        field(ready, "maximum_generated_tokens").unsigned_integer() == 4096,
        "owned worker readiness/execution identity mismatch");
    worker_execution(ready, stack_execution);
    require(checkpoint_slots <= 4 && checkpoint_budget <= (512ull << 20) && bool(checkpoint_slots) == bool(checkpoint_budget),
        "invalid serving broker preparation");
    if (checkpoint_slots) {
        require(stack_execution == "direct" && field(ready, "checkpoint_broker").boolean() &&
            field(ready, "broker_epoch").unsigned_integer() && field(ready, "checkpoint_slots").unsigned_integer() == checkpoint_slots &&
            field(ready, "checkpoint_combined_budget_bytes").unsigned_integer() == checkpoint_budget, "owned broker readiness identity mismatch");
        broker_storage(ready, ready);
    } else if (const auto* enabled = ready.find("checkpoint_broker")) require(!enabled->boolean(), "unexpected serving broker capability");
}

ToolEnumValidation parse_tool_enum_validation(std::string_view value) {
    if (value == "strict") return ToolEnumValidation::strict;
    if (value == "executor") return ToolEnumValidation::executor;
    throw FormatError("tool-enum-validation must be strict or executor");
}
std::string_view tool_enum_validation_name(ToolEnumValidation value) {
    switch (value) {
        case ToolEnumValidation::strict: return "strict";
        case ToolEnumValidation::executor: return "executor";
    }
    throw FormatError("invalid tool enum validation policy");
}
std::string_view tool_enum_validation_policy(ToolEnumValidation value) {
    return tool_enum_validation_name(value) == "executor" ? "original-agent-executor-enums-v1" : "strict-server-enums-v1";
}
ToolArgValidation parse_tool_arg_validation(std::string_view value) {
    if (value == "strict") return ToolArgValidation::strict;
    if (value == "executor") return ToolArgValidation::executor;
    throw FormatError("tool-arg-validation must be strict or executor");
}
std::string_view tool_arg_validation_name(ToolArgValidation value) {
    switch (value) {
        case ToolArgValidation::strict: return "strict";
        case ToolArgValidation::executor: return "executor";
    }
    throw FormatError("invalid tool arg validation policy");
}
std::string_view tool_arg_validation_policy(ToolArgValidation value) {
    return tool_arg_validation_name(value) == "executor" ? "executor-deferred-args-v1" : "strict-server-args-v1";
}
void validate_executor_tool_arguments(const Json& value) {
    require(value.kind() == Json::Kind::object, "executor tool arguments must be a JSON object");
    size_t nodes = 0;
    validate_executor_node(value, 0, nodes);
}
void validate_tool_value(const Json& schema, const Json& value, ToolEnumValidation tool_enums) {
    static_cast<void>(tool_enum_validation_name(tool_enums));
    const auto type = schema_type(schema);
    if (type == "any") {
        for (const auto& branch : field(schema, "anyOf").array()) {
            try { validate_tool_value(branch, value, tool_enums); return; }
            catch (const FormatError&) {}
        }
        require(false, "tool value matches no schema alternative");
    }
    if (type == "string") {
        require(value.kind() == Json::Kind::string, "tool value is not string");
        if (tool_enums == ToolEnumValidation::executor) validate_executor_string(value.string());
    }
    else if (type == "boolean") require(value.kind() == Json::Kind::boolean, "tool value is not boolean");
    else if (type == "null") require(value.is_null(), "tool value is not null");
    else if (type == "number" || type == "integer") {
        static_cast<void>(number(value, -1e300, 1e300)); const auto n = decimal(value);
        const auto safe = decimal(parse_json("9007199254740991")); auto magnitude = n; magnitude.negative = false;
        require(type != "integer" || ((n.digits == "0" || n.exponent >= 0) && compare(magnitude, safe) <= 0 && value.number_lexeme().find_first_of(".eE") == std::string_view::npos), "tool integer must use exact safe-integer syntax");
        if (const auto* minimum = schema.find("minimum")) require(compare(n, decimal(*minimum)) >= 0, "tool value below minimum");
        if (const auto* maximum = schema.find("maximum")) require(compare(n, decimal(*maximum)) <= 0, "tool value above maximum");
        if (const auto* minimum = schema.find("exclusiveMinimum")) require(compare(n, decimal(*minimum)) > 0, "tool value at or below exclusive minimum");
    } else if (type == "array") {
        require(value.array().size() <= 4096, "tool array bound");
        for (const auto& item : value.array()) validate_tool_value(field(schema, "items"), item, tool_enums);
    } else {
        const auto* properties = schema.find("properties");
        for (const auto& [name, item] : value.object()) {
            const auto* child = properties ? properties->find(name) : nullptr; require(child, "unknown tool property"); validate_tool_value(*child, item, tool_enums);
        }
        if (const auto* required = schema.find("required")) for (const auto& name : required->array()) require(value.find(name.string()), "missing required tool property");
    }
    if (const auto* values = schema.find("enum"); values && tool_enums == ToolEnumValidation::strict) {
        bool found = false;
        for (const auto& candidate : values->array()) {
            if (candidate.kind() == Json::Kind::number && value.kind() == Json::Kind::number) found |= compare(decimal(candidate), decimal(value)) == 0;
            else found |= candidate.dump() == value.dump();
        }
        require(found, "tool value outside enum");
    }
}

AdmittedChat admit_chat(const Json& request, std::string_view alias, ToolEnumValidation tool_enums, ToolArgValidation tool_args) {
    static_cast<void>(tool_enum_validation_name(tool_enums));
    static_cast<void>(tool_arg_validation_name(tool_args));
    static_cast<void>(request.dump(1024 * 1024));
    keys(request, {"model", "messages", "n", "parallel_tool_calls", "stream", "stream_options", "keep_alive", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "min_p", "seed", "tools", "tool_choice", "enable_thinking", "preserve_thinking", "reasoning_effort", "chat_template_kwargs", "presence_penalty", "repetition_penalty"});
    require(field(request, "model").string() == alias, "unknown model alias");
    AdmittedChat out;
    if (const auto* n = request.find("n")) require(n->unsigned_integer() == 1, "only n=1 is supported");
    if (const auto* parallel = request.find("parallel_tool_calls")) out.single_tool_call = !parallel->boolean();
    if (const auto* p = request.find("stream")) out.stream = p->boolean();
    if (const auto* options = request.find("stream_options")) {
        require(out.stream, "stream_options requires streaming"); keys(*options, {"include_usage"});
        if (const auto* include = options->find("include_usage")) out.include_usage = include->boolean();
    }
    // Numeric lifetime hints do not change the owned worker's supervised lifetime.
    if (const auto* keep = request.find("keep_alive")) {
        if (keep->kind() == Json::Kind::string) {
            const auto value = keep->string(); auto numeric = value;
            for (const auto suffix : {"ms", "s", "m", "h"}) if (numeric.ends_with(suffix)) { numeric.remove_suffix(std::string_view(suffix).size()); break; }
            require(!value.empty() && value.size() <= 64 && (value == "-1" || (!numeric.empty() && std::all_of(numeric.begin(), numeric.end(), [](unsigned char c) { return std::isdigit(c) || c == '.'; }) && std::count(numeric.begin(), numeric.end(), '.') <= 1 && std::any_of(numeric.begin(), numeric.end(), [](unsigned char c) { return std::isdigit(c); }))), "invalid keep_alive duration hint");
        } else static_cast<void>(number(*keep, -1, 1e12));
    }
    const auto* template_options = request.find("chat_template_kwargs");
    if (template_options) keys(*template_options, {"enable_thinking", "preserve_thinking", "reasoning_effort"});
    Json::Object canonical_options;
    for (const auto name : {"enable_thinking", "preserve_thinking", "reasoning_effort"}) {
        const auto* outer = request.find(name);
        const auto* inner = template_options ? template_options->find(name) : nullptr;
        if (std::string_view(name) == "reasoning_effort") {
            if (outer) static_cast<void>(outer->string());
            if (inner) static_cast<void>(inner->string());
        } else {
            if (outer) static_cast<void>(outer->boolean());
            if (inner) static_cast<void>(inner->boolean());
        }
        require(!outer || !inner || outer->dump() == inner->dump(), "conflicting chat template options");
        if (const auto* selected = outer ? outer : inner) canonical_options.emplace_back(name, *selected);
    }
    const Json options(std::move(canonical_options));
    if (const auto* p = options.find("enable_thinking")) out.thinking = p->boolean();
    if (const auto* p = options.find("reasoning_effort")) {
        const auto effort = p->string();
        require(effort == "xhigh" || effort == "high" || effort == "medium" || effort == "low" ||
            (!out.thinking && effort == "none"), "unsupported reasoning effort");
    }
    require(!(request.find("max_tokens") && request.find("max_completion_tokens")), "provide only one output limit");
    const auto* limit = request.find("max_tokens"); if (!limit) limit = request.find("max_completion_tokens");
    // The HTTP request budget is separate from the scalar, V8 or explicitly negotiated MTP packet capacity.
    if (limit) { const auto n = limit->unsigned_integer(); require(n && n <= 32768, "output limit must be 1..32768"); out.maximum_tokens = uint32_t(n); }
    for (const auto name : {"presence_penalty", "repetition_penalty"}) if (const auto* p = request.find(name)) {
        // Compare decimal values exactly: a non-neutral value must not become
        // neutral merely because parsing it as double would round it to 0/1.
        const auto neutral = parse_json(std::string_view(name) == "presence_penalty" ? "0" : "1");
        require(compare(decimal(*p), decimal(neutral)) == 0, "non-neutral sampling penalty is unsupported");
    }
    const double temperature = request.find("temperature") ? number(*request.find("temperature"), 0, 100) : 0;
    const double top_p = request.find("top_p") ? number(*request.find("top_p"), 0, 1) : 1; require(top_p > 0, "top_p must be positive");
    const double min_p = request.find("min_p") ? number(*request.find("min_p"), 0, 1) : 0;
    const auto top_k = request.find("top_k") ? request.find("top_k")->unsigned_integer() : 0; require(top_k <= 248320, "top_k bound");
    out.sampling = Json(Json::Object{{"temperature", real(temperature)}, {"top_k", Json::integer(top_k)}, {"top_p", real(top_p)}, {"min_p", real(min_p)}});
    if (const auto* seed = request.find("seed")) out.seed = seed->unsigned_integer();
    Json::Array declarations; std::set<std::string, std::less<>> names;
    if (const auto* tools = request.find("tools")) {
        require(tools->array().size() <= 64, "API tool limit");
        for (const auto& declaration : tools->array()) {
            keys(declaration, {"type", "function"}); require(field(declaration, "type").string() == "function", "unsupported tool type");
            const auto& function = field(declaration, "function"); keys(function, {"name", "description", "parameters", "strict"});
            const auto name = std::string(field(function, "name").string()); safe_name(name, true); require(names.insert(name).second, "duplicate tool name");
            if (const auto* description = function.find("description")) require(description->string().size() <= 16384, "tool description bound");
            if (const auto* strict = function.find("strict")) static_cast<void>(strict->boolean());
            Json schema = function.find("parameters") ? *function.find("parameters") : parse_json("{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}");
            size_t nodes = 0; schema_admit(schema, 0, nodes); require(schema_type(schema) == "object", "function parameters must be object");
            ToolSpec manifest{name, {}}; std::set<std::string_view> required;
            if (const auto* p = schema.find("required")) for (const auto& value : p->array()) required.insert(value.string());
            if (const auto* p = schema.find("properties")) for (const auto& [key, value] : p->object()) manifest.parameters.push_back({key, parameter_type(schema_type(value)), required.contains(key)});
            out.manifest.push_back(std::move(manifest)); out.schemas.push_back(std::move(schema)); out.names.push_back(name); declarations.push_back(declaration);
        }
    }
    out.tool_choice = "auto";
    if (const auto* choice = request.find("tool_choice")) {
        if (choice->kind() == Json::Kind::string) { out.tool_choice = choice->string(); require(out.tool_choice == "auto" || out.tool_choice == "none" || out.tool_choice == "required", "unsupported tool_choice"); }
        else { out.named_tool_choice = true; keys(*choice, {"type", "function"}); require(field(*choice, "type").string() == "function", "unsupported named tool_choice"); const auto& function = field(*choice, "function"); keys(function, {"name"}); out.tool_choice = field(function, "name").string(); require(names.contains(out.tool_choice), "unknown named tool_choice"); }
    }
    require(out.tool_choice == "auto" || out.tool_choice == "none" || !declarations.empty(), "tool choice requires tools");
    const auto& messages = field(request, "messages"); history(messages, out.names, out.schemas, tool_enums, tool_args);
    Json::Array rendered_messages = messages.array();
    if (out.named_tool_choice || out.tool_choice == "required") {
        const std::string instruction = !out.named_tool_choice ? "For this response, call at least one of the declared tools. Do not answer without a tool call." : "For this response, call only the declared function named " + out.tool_choice + ". Do not answer without that tool call.";
        rendered_messages.insert(rendered_messages.begin(), Json(Json::Object{{"role", Json("system")}, {"content", Json(instruction)}}));
    }
    if (!out.named_tool_choice && out.tool_choice == "none") { declarations.clear(); out.manifest.clear(); out.schemas.clear(); out.names.clear(); }
    else if (out.named_tool_choice) {
        const auto index = size_t(std::find(out.names.begin(), out.names.end(), out.tool_choice) - out.names.begin());
        declarations = {declarations.at(index)}; out.manifest = {out.manifest.at(index)}; out.schemas = {out.schemas.at(index)}; out.names = {out.names.at(index)};
    }
    Json::Object render{{"messages", Json(std::move(rendered_messages))}, {"tools", Json(std::move(declarations))}, {"enable_thinking", Json(out.thinking)}};
    for (const auto name : {"preserve_thinking", "reasoning_effort"}) if (const auto* p = options.find(name)) render.emplace_back(name, *p);
    out.render = Json(std::move(render)); return out;
}

ServeSession::ServeSession(ServeCodec codec, std::string alias, std::string stack_execution, Json broker_ready
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    , bool verify8, Json verify_ready
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    , bool branch8
#endif
    , ToolEnumValidation tool_enums
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    , bool sampled_prefill
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    , Json mtp_ready
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    , Json weight_lanes_ready
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    , std::optional<uint32_t> mtp_first_output_packet
#endif
    , ToolArgValidation tool_args
    )
    : codec_(std::move(codec)), alias_(std::move(alias)), stack_execution_(std::move(stack_execution)), tool_enums_(tool_enums), tool_args_(tool_args)
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    , sampled_prefill_(sampled_prefill)
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    , mtp_(!mtp_ready.is_null()), mtp_ready_(std::move(mtp_ready)), mtp_previous_(mtp_ready_), mtp_first_output_packet_(mtp_first_output_packet)
#endif
    , broker_ready_(std::move(broker_ready))
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    , verify8_(verify8), verify_ready_(std::move(verify_ready))
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    , branch8_(branch8)
#endif
    {
    static_cast<void>(tool_enum_validation_name(tool_enums_));
    static_cast<void>(tool_arg_validation_name(tool_args_));
    require(bool(codec_.render) && bool(codec_.encode) && bool(codec_.decode) && !alias_.empty(), "invalid serving codec");
    require(stack_execution_ == "direct" || stack_execution_ == "recurrent-replay", "unsupported serving startup execution mode");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (!weight_lanes_ready.is_null()) {
        const auto workers = owned_lane_auth_workers_requested(weight_lanes_ready);
        validate_owned_weight_lanes_ready(weight_lanes_ready, true, workers);
        Json::Object identity;
        for (const auto name : owned_weight_lane_fields) identity.emplace_back(std::string(name), field(weight_lanes_ready, name));
        if (workers) for (const auto name : owned_lane_auth_fields)
            identity.emplace_back(std::string(name), field(weight_lanes_ready, name));
        weight_lanes_identity_ = Json(std::move(identity));
    }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    if (mtp_) {
        validate_owned_weight_lanes_ready(mtp_ready_, !weight_lanes_ready.is_null(), owned_lane_auth_workers_requested(mtp_ready_));
        if (!weight_lanes_ready.is_null()) validate_owned_weight_lanes_done(mtp_ready_, weight_lanes_ready);
    }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    require(!sampled_prefill_ || (stack_execution_ == "direct" && !verify8_ && !branch8_ && broker_ready_.is_null()),
        "sampled input8 refuses greedy verification, replay and checkpoint brokers");
    if (sampled_prefill_) committed_.reserve(65536); // Before all-N publication.
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    require(!mtp_ || (stack_execution_ == "direct" && !verify8_ && !branch8_ && !sampled_prefill_ && broker_ready_.is_null()),
        "sampled MTP refuses another transport, replay and checkpoint brokers");
    if (mtp_) { validate_mtp_ready(mtp_ready_, mtp_draft_attention(mtp_ready_), mtp_request_carry(mtp_ready_), mtp_depth(mtp_ready_)); committed_.reserve(65536); }
    if (mtp_first_output_packet_) {
        require(mtp_, "MTP first-output-packet requires native MTP readiness");
        const auto cap = mtp_output_packet(mtp_ready_);
        require(*mtp_first_output_packet_ >= 1 && *mtp_first_output_packet_ <= cap, "MTP first-output-packet outside negotiated output packet bound");
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    if (verify8_) {
        require(stack_execution_ == "direct", "private HTTP V8 cannot use replay");
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (branch8_) {
            require(!broker_ready_.is_null() && broker_ready_.dump() == verify_ready_.dump(), "branch8 requires the same explicit broker/verifier readiness");
            validate_branch8_ready(verify_ready_); verify_previous_ = verify_ready_;
        } else
#endif
        { require(broker_ready_.is_null(), "private HTTP V8 cannot use checkpoints without explicit branch8 consent"); validate_verify8_ready(verify_ready_); }
        committed_.reserve(65536);
    } else require(verify_ready_.is_null(), "unexpected private V8 readiness");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    require(!branch8_ || verify8_, "branch8 requires explicit private V8 scheduling");
#endif
    if (!broker_ready_.is_null()) {
        const auto slots = field(broker_ready_, "checkpoint_slots").unsigned_integer();
        require(slots && slots <= 4, "serving broker slot bound");
        validate_worker_ready(broker_ready_, stack_execution_, uint32_t(slots), field(broker_ready_, "checkpoint_combined_budget_bytes").unsigned_integer());
        broker_epoch_ = field(broker_ready_, "broker_epoch").unsigned_integer();
        committed_.reserve(65536); // Before any restore publication.
    }
}
void ServeSession::invalidate() noexcept { saved_reasoning_.clear(); committed_.clear(); sampling_.clear(); reusable_ = false; seed_ = 0; broker_resync_ = true;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    mtp_resync_ = true;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    verify_resync_ = true;
#endif
}

Json ServeSession::complete(const Json& input, std::string_view response_id, const ServeTarget& target, const ServeSink& sink) {
    const auto request_start = std::chrono::steady_clock::now();
    const auto request = admit_chat(input, alias_, tool_enums_, tool_args_);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    uint64_t gdn_old_order_admitted_calls = 0;
    uint64_t hc_down_f32_admitted_calls = 0;
    std::array<uint64_t, 3> hc_rows_admitted{};
    std::array<uint64_t, 4> router_window_admitted{};
    uint64_t hc_down_window_admitted = 0;
    std::array<uint64_t, 2> qsa_sparse_admitted{};
    std::array<uint64_t, 5> gdn_ported_admitted{};
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    RawDecodeRouteCounts raw_decode_admitted{};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    Json::Object q8_three_admitted_routes{{"raw",Json::integer(0)},{"packed",Json::integer(0)},{"hc_raw",Json::integer(0)},{"wide_qkv_raw",Json::integer(0)},{"wide_output_raw",Json::integer(0)}};
#endif
#endif
    [[maybe_unused]] uint64_t input_batches = 0, output_batches = 0, generation_rounds = 0, verified_rounds = 0, accepted_tokens = 0;
    [[maybe_unused]] size_t input_width = 0, output_width = 0;
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    require(!verify8_ || number(field(request.sampling, "temperature"), 0, 100) == 0,
        "private HTTP V8 is greedy-only; nonzero temperature is explicitly refused");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    uint64_t checkpoint_captures = 0, checkpoint_restores = 0, checkpoint_cold_resyncs = 0;
#endif
    require(bool(target) && bool(sink.cancelled) && bool(sink.new_seed), "missing serving callbacks");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    if (const auto* measured = mtp_ready_.is_null() ? nullptr : mtp_ready_.find("native_measured_controller"); measured && measured->boolean()) {
        const auto temperature = number(field(request.sampling, "temperature"), 0, 100);
        std::string_view mode = "greedy";
        if (temperature != 0) {
            require(temperature == 1 && number(field(request.sampling, "min_p"), 0, 1) == 0, "sampling policy is not calibrated by measured profile");
            const auto top_k = field(request.sampling, "top_k").unsigned_integer(); const auto top_p = number(field(request.sampling, "top_p"), 0, 1);
            if (top_k == 20 && top_p == 0.95) mode = "coding";
            else { require(top_p == 1 && (top_k == 0 || top_k == 20), "sampling policy is not calibrated by measured profile"); mode = top_k == 20 ? "topk" : "full"; }
        }
        const auto& modes = field(mtp_ready_, "native_controller_sampling_modes").array();
        require(!modes.empty() && modes.size() <= 4 && std::any_of(modes.begin(), modes.end(), [mode](const auto& value) { return value.string() == mode; }), "sampling mode is not calibrated by measured profile");
    }
#endif
    const auto rendered = codec_.render(request.render); auto prompt = codec_.encode(rendered);
    const auto matches_committed = [&](const auto& value) { return value.size() >= committed_.size() && std::equal(committed_.begin(), committed_.end(), value.begin()); };
    if (reusable_ && !matches_committed(prompt) && !saved_reasoning_.empty()) {
        auto fields = request.render.object();
        for (auto& [name, value] : fields) if (name == "messages") {
            auto messages = value.array(); size_t ordinal = 0;
            for (auto& message : messages) {
                if (field(message, "role").string() != "assistant") continue;
                const auto* present = message.find("reasoning_content");
                const auto found = std::find_if(saved_reasoning_.begin(), saved_reasoning_.end(), [&](const auto& entry) { return entry.first == ordinal; });
                if ((!present || present->is_null()) && found != saved_reasoning_.end()) {
                    auto entry = message.object();
                    std::erase_if(entry, [](const auto& field) { return field.first == "reasoning_content"; });
                    entry.emplace_back("reasoning_content", Json(found->second)); message = Json(std::move(entry));
                }
                ++ordinal;
            }
            value = Json(std::move(messages));
        }
        auto candidate = codec_.encode(codec_.render(Json(std::move(fields))));
        if (candidate.size() < 65536 && matches_committed(candidate)) prompt = std::move(candidate);
    }
    require(!prompt.empty() && prompt.size() < 65536, "request leaves no generation space in the 64K context");
    const auto maximum_tokens = std::min(request.maximum_tokens, 65536u - uint32_t(prompt.size()));
    for (const auto id : prompt) require(id < codec_.admission.vocabulary, "rendered token outside vocabulary");
    const auto policy = request.sampling.dump();
    const auto created_signed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    require(created_signed >= 0, "invalid API clock"); const auto created = uint64_t(created_signed);
    const bool append = reusable_ && sampling_ == policy && (!request.seed || *request.seed == seed_) &&
        prompt.size() >= committed_.size() && std::equal(committed_.begin(), committed_.end(), prompt.begin());
    size_t cached_prefix = append ? committed_.size() : 0;
    size_t common = 0;
    if (reusable_) while (common < committed_.size() && common < prompt.size() && committed_[common] == prompt[common]) ++common;
    const uint64_t seed = append ? seed_ : request.seed ? *request.seed : sink.new_seed();
    require(!response_id.empty() && response_id.size() <= 96, "response ID bound");
    ResponseParser parser(request.manifest, request.thinking, {}, tool_args_ == ToolArgValidation::executor, true); std::string streamed_content, streamed_thinking;
    std::vector<uint32_t> generated; generated.reserve(maximum_tokens);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::optional<PrivateMtpRequestTrace> native_trace;
    if (sink.prefix_trace && mtp_) native_trace.emplace();
#endif
    const auto cancelled = [&] { require(!sink.cancelled(), "request cancelled or deadline expired"); };
    const auto emit = [&](Json delta, Json finish = Json()) {
        if (request.stream) {
            require(bool(sink.chunk), "missing streaming sink"); cancelled();
            auto value = chunk(response_id, alias_, created, std::move(delta), finish);
            if (request.include_usage) { auto fields = value.object(); fields.emplace_back("usage", Json()); value = Json(std::move(fields)); }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
            if (!finish.is_null()) value = project_weight_lanes(std::move(value), weight_lanes_identity_);
#endif
            sink.chunk(value);
        }
    };
    const auto flush = [&] {
        std::string content, thinking;
        for (const auto& record : parser.records()) {
            if (record.kind == ResponseKind::text) content += record.text;
            else if (record.kind == ResponseKind::thinking) thinking += record.text;
        }
        require(content.starts_with(streamed_content) && thinking.starts_with(streamed_thinking), "nonmonotonic response parser");
        if (thinking.size() != streamed_thinking.size()) emit(Json(Json::Object{{"reasoning_content", Json(thinking.substr(streamed_thinking.size()))}}));
        if (content.size() != streamed_content.size()) emit(Json(Json::Object{{"content", Json(content.substr(streamed_content.size()))}}));
        streamed_content = std::move(content); streamed_thinking = std::move(thinking);
    };
    bool state_touched = false;
    try {
        cancelled();
        if (sink.prefix_trace) {
            Json::Array ids; ids.reserve(prompt.size()); for (const auto id : prompt) ids.push_back(Json::integer(id));
            Json::Array prior; prior.reserve(committed_.size()); for (const auto id : committed_) prior.push_back(Json::integer(id));
            sink.prefix_trace(Json(Json::Object{{"event", Json("prompt_prefix")}, {"schema", Json::integer(1)}, {"response_id", Json(std::string(response_id))},
                {"prompt_ids", Json(std::move(ids))}, {"prior_committed_ids", Json(std::move(prior))}, {"prior_committed_tokens", Json::integer(committed_.size())}, {"common_prefix", Json::integer(common)},
                {"reusable", Json(reusable_)}, {"append", Json(append)}, {"sampling_matches", Json(sampling_ == policy)}, {"seed_matches", Json(!request.seed || *request.seed == seed_)},
                {"prompt_tokens", Json::integer(prompt.size())}, {"cached_tokens", Json::integer(cached_prefix)}, {"new_tokens", Json::integer(prompt.size() - cached_prefix)}}));
        }
        emit(Json(Json::Object{{"role", Json("assistant")}}));
        state_touched = true;
        size_t position = append ? committed_.size() : 0;
        const auto broker_reply = [&](const Json& value, bool fresh) {
            if (!broker_epoch_) return;
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (branch8_) require(field(value, "broker_protocol").string() == "branch8-v1" &&
                field(value, "broker_batch_width").unsigned_integer() == 8, "serving branch8 wire identity changed");
#endif
            require(field(value, "broker_epoch").unsigned_integer() == broker_epoch_, "serving worker epoch changed");
            const auto command = field(value, "command_id").unsigned_integer(), lease = field(value, "lease_id").unsigned_integer();
            require(command > broker_command_ && (broker_resync_ || command == broker_command_ + 1) &&
                (fresh ? lease > broker_lease_ : lease && lease == broker_lease_), "stale serving broker command/lease");
            broker_storage(value, broker_ready_);
            require(field(value, "checkpoint_retained").unsigned_integer() <= field(broker_ready_, "checkpoint_slots").unsigned_integer(),
                "serving broker retained count exceeds capacity");
            broker_command_ = command; broker_lease_ = lease; broker_resync_ = false;
        };
        const auto sampling_matches = [&](const Json& value, size_t prefix) {
            const auto& sampled = field(value, "sampling");
            require(field(sampled, "valid").boolean() && field(sampled, "entropy_prefix").unsigned_integer() == prefix &&
                field(sampled, "seed").unsigned_integer() == seed, "worker sampling prefix/seed mismatch");
            for (const auto& [name, value] : request.sampling.object())
                require(number(field(sampled, name), 0, 1e9) == number(value, 0, 1e9), "worker sampling policy changed");
        };
        const auto control = [&](Json job, std::string_view operation, bool fresh) {
            cancelled();
            const auto done = target(job, [](const Json&) { throw FormatError("zero-step broker control emitted a token"); });
            cancelled();
            require(field(done, "event").string() == "control_done" && field(done, "op").string() == operation &&
                field(done, "model_steps").unsigned_integer() == 0 && field(done, "generated_ids").array().empty(),
                "invalid zero-step broker completion");
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (branch8_) {
                validate_branch8_control(verify_ready_, verify_previous_, job, done, verify_resync_);
                if (operation == "branch_capture") ++checkpoint_captures;
                else { checkpoint_restores += field(done, "outcome").string() == "restored"; checkpoint_cold_resyncs += verify_resync_; }
                verify_previous_ = done; verify_resync_ = false;
            }
#endif
            broker_reply(done, fresh);
            const auto prefix = field(done, "committed_prefix").unsigned_integer();
            require(prefix <= prompt.size(), "broker restored beyond future prompt"); sampling_matches(done, prefix);
            const bool logits = field(done, "logits_available").boolean(); const auto& stop = field(done, "stop_id");
            require(logits == bool(prefix) && (stop.is_null() || (logits && codec_.admission.is_stop(stop.unsigned_integer()))) &&
                field(done, "continuation_allowed").boolean() == (logits && stop.is_null()), "invalid broker terminal metadata");
            return done;
        };
        const auto prefill_start = std::chrono::steady_clock::now();
        bool began = false;
        if (broker_epoch_ && !append && number(field(request.sampling, "temperature"), 0, 100) == 0) {
            Json::Array future; future.reserve(prompt.size()); for (auto id : prompt) future.push_back(Json::integer(id));
            const bool reuse = reusable_;
            const auto done = control(Json(Json::Object{{"op", Json("branch_begin")}, {"prompt_ids", Json(std::move(future))},
                {"generation_requested", Json(true)}, {"allow_restore", Json(reuse)}, {"sampling", request.sampling}, {"seed", Json::integer(seed)}}), "branch_begin", true);
            const auto prefix = field(done, "committed_prefix").unsigned_integer(), lcp = field(done, "common_prefix").unsigned_integer();
            const auto outcome = field(done, "outcome").string();
            require(lcp == (reuse ? common : 0) && prefix <= lcp &&
                ((outcome == "cold_miss" && !prefix) || (outcome == "restored" && prefix && reuse)), "broker exact-LCP transaction mismatch");
            require(prefix < prompt.size() || field(done, "stop_id").is_null(), "terminal restored prompt cannot generate without new input");
            // Reserved at preparation; shrinking cannot allocate after restore.
            committed_.resize(size_t(prefix)); position = size_t(prefix); cached_prefix = position; began = true;
        } else if (!append) committed_.clear();
        if (!append) reusable_ = false;
        const auto input_start = position;
        const auto run = [&](Json job, std::optional<uint32_t> input_token, bool decode) {
            cancelled(); std::vector<uint32_t> events;
            const auto done = target(job, [&](const Json& event) {
                require(decode && field(event, "event").string() == "token", "unexpected worker token event");
                const auto id = field(event, "id").unsigned_integer(); require(id < codec_.admission.vocabulary && events.empty(), "invalid worker token count/ID");
                require(field(event, "committed_prefix").unsigned_integer() == position + 1, "worker token prefix mismatch"); events.push_back(uint32_t(id));
                if (broker_epoch_) require(field(event, "broker_epoch").unsigned_integer() == broker_epoch_ &&
                    field(event, "command_id").unsigned_integer() == broker_command_ + 1 &&
                    field(event, "lease_id").unsigned_integer() == broker_lease_, "stale serving broker token event");
            });
            require(field(done, "event").string() == "done" && field(done, "committed_prefix").unsigned_integer() == position + 1, "worker completion prefix mismatch");
            worker_execution(done, stack_execution_);
            broker_reply(done, field(job, "reset").boolean()); sampling_matches(done, position + 1);
            const auto& ids = field(done, "generated_ids").array(); require(ids.size() == (decode ? 1U : 0U) && events.size() == ids.size(), "worker generated count mismatch");
            if (decode) require(ids[0].unsigned_integer() == events[0], "worker token/done disagreement");
            ++position; committed_.push_back(input_token ? *input_token : events.at(0));
            return done;
        };
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        uint64_t controller_committed = 0, controller_emitted = 0, controller_pending = 0;
        Json controller_feedback;
        const auto run_mtp = [&](Json job) {
            cancelled(); MtpWireReply reply(mtp_ready_, mtp_previous_, job, mtp_resync_);
            const auto done = target(job, [&](const Json& event) { reply.event(event); });
            reply.finish(done); cancelled();
            if (const auto* generation = done.find("native_mtp_generation")) {
                if (const auto* committed = generation->find("committed")) controller_committed = committed->unsigned_integer();
                if (const auto* emitted = generation->find("emitted")) controller_emitted = emitted->unsigned_integer();
                if (const auto* pending = generation->find("overflow")) controller_pending = pending->unsigned_integer();
                if (const auto* feedback = generation->find("stage_a_controller")) controller_feedback = *feedback;
            }
            worker_execution(done, stack_execution_); sampling_matches(done, field(done, "committed_prefix").unsigned_integer());
            const auto& inputs = field(job, "tokens").array();
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
            if (mtp_ready_.find("native_raw_decode_mode")) {
                const auto routes = raw_decode_routes_value(field(done, "native_raw_decode_request_routes"));
                for (size_t i = 0; i < routes.size(); ++i) {
                    require(routes[i] <= std::numeric_limits<uint64_t>::max() - raw_decode_admitted[i], "admitted raw decode route overflow");
                    raw_decode_admitted[i] += routes[i];
                }
            }
#endif
            if (mtp_hc_rows(mtp_ready_)) for (size_t i = 0; i < 3; ++i) {
                const auto calls = field(done, mtp_hc_rows_request_names[i]).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - hc_rows_admitted[i], "admitted HC rows overflow"); hc_rows_admitted[i] += calls;
            }
            if (mtp_router_window(mtp_ready_)) for (size_t i = 0; i < 4; ++i) {
                const auto calls = field(done, mtp_router_window_request_names[i]).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - router_window_admitted[i], "admitted router window overflow"); router_window_admitted[i] += calls;
            }
            if (mtp_hc_down_window(mtp_ready_)) {
                const auto calls = field(done, mtp_hc_down_window_request_name).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - hc_down_window_admitted, "admitted HC down window overflow"); hc_down_window_admitted += calls;
            }
            if (mtp_qsa_sparse(mtp_ready_)) for (size_t i = 0; i < 2; ++i) {
                const auto calls = field(done, mtp_qsa_sparse_request_names[i]).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - qsa_sparse_admitted[i], "admitted sparse QSA overflow"); qsa_sparse_admitted[i] += calls;
            }
            if (mtp_gdn_ported(mtp_ready_)) for (size_t i = 0; i < 5; ++i) {
                const auto calls = field(done, mtp_gdn_ported_request_names[i]).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - gdn_ported_admitted[i], "admitted ported GDN overflow"); gdn_ported_admitted[i] += calls;
            }
            if (mtp_hc_down_f32(mtp_ready_)) {
                const auto calls = field(done, "native_prefill_hc_down_f32_request_encoded_calls").unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - hc_down_f32_admitted_calls, "admitted HC down encoded call overflow"); hc_down_f32_admitted_calls += calls;
            }
            if (mtp_gdn_old_order(mtp_ready_)) {
                const auto calls = field(done, "native_gdn_old_order_request_encoded_calls").unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - gdn_old_order_admitted_calls, "admitted old-order GDN call count overflow");
                gdn_old_order_admitted_calls += calls;
            }
            if (native_trace) native_trace->admit(mtp_previous_, done, job, reply.events());
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
            if (const auto* selected=mtp_ready_.find("q8_three_rows_mode");selected&&selected->string()=="on") {
                const auto& routes=field(done,"q8_three_rows_request_routes");
                for(auto& [name,total]:q8_three_admitted_routes){const auto n=field(routes,name).unsigned_integer(),before=total.unsigned_integer();require(n<=std::numeric_limits<uint64_t>::max()-before,"admitted N3 route overflow");total=Json::integer(before+n);}
            }
#endif
            if (!inputs.empty()) {
                ++input_batches; input_width = std::max(input_width, inputs.size());
                for (const auto& id : inputs) committed_.push_back(uint32_t(id.unsigned_integer()));
            } else {
                const auto& outputs = field(done, "generated_ids").array();
                ++output_batches; output_width = std::max(output_width, outputs.size());
                generation_rounds += field(done, "native_mtp_rounds").unsigned_integer() - field(mtp_previous_, "native_mtp_rounds").unsigned_integer();
                accepted_tokens += field(done, "native_mtp_accepted").unsigned_integer() - field(mtp_previous_, "native_mtp_accepted").unsigned_integer();
                for (const auto& id : outputs) committed_.push_back(uint32_t(id.unsigned_integer()));
            }
            position = size_t(field(done, "committed_prefix").unsigned_integer()); mtp_previous_ = done; mtp_resync_ = false;
            return done;
        };
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        const auto run_many = [&](Json job) {
            cancelled();
            Verify8Reply reply(verify_ready_, verify_previous_, job, uint32_t(position), request.sampling, seed, verify_resync_
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                , branch8_
#endif
                );
            const auto done = target(job, [&](const Json& event) {
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                if (branch8_) require(field(event, "broker_epoch").unsigned_integer() == broker_epoch_ &&
                    field(event, "command_id").unsigned_integer() == broker_command_ + 1 && field(event, "lease_id").unsigned_integer() == broker_lease_ &&
                    field(event, "broker_protocol").string() == "branch8-v1" && field(event, "broker_batch_width").unsigned_integer() == 8,
                    "stale serving branch8 batched event");
#endif
                reply.event(event);
            });
            reply.finish(done); cancelled(); // No response parsing or history publication before full admission.
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (branch8_) broker_reply(done, field(job, "reset").boolean());
#endif
            const auto count = field(job, "tokens").array().size();
            if (count) {
                ++input_batches; input_width = std::max(input_width, count);
                for (const auto& id : field(job, "tokens").array()) committed_.push_back(uint32_t(id.unsigned_integer()));
            } else {
                ++output_batches; output_width = std::max(output_width, field(done, "generated_ids").array().size());
                generation_rounds += field(done, "lookup_rounds").unsigned_integer();
                verified_rounds += field(done, "lookup_verified_rounds").unsigned_integer();
                accepted_tokens += field(done, "lookup_accepted_tokens").unsigned_integer();
                for (const auto& id : field(done, "generated_ids").array()) committed_.push_back(uint32_t(id.unsigned_integer()));
            }
            position = size_t(field(done, "committed_prefix").unsigned_integer()); verify_previous_ = done; verify_resync_ = false;
            return done;
        };
        if (verify8_) {
            while (position < prompt.size()) {
                size_t n = std::min<size_t>(8, prompt.size() - position);
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                if (branch8_) n = std::min(n, 512 - position % 512); // Never step over a useful ancestor boundary.
#endif
                Json::Array ids; ids.reserve(n);
                for (size_t t = 0; t < n; ++t) ids.push_back(Json::integer(prompt[position + t]));
                const bool fresh = !append && !began && !position;
                Json::Object job{{"tokens", Json(std::move(ids))}, {"generate", Json::integer(0)}, {"reset", Json(fresh)}};
                if (fresh) { job.emplace_back("sampling", request.sampling); job.emplace_back("seed", Json::integer(seed)); job.emplace_back("prompt_lookup", Json(true)); }
                static_cast<void>(run_many(Json(std::move(job))));
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                if (branch8_ && (position % 512 == 0 || position == prompt.size())) {
                    const auto captured = control(Json(Json::Object{{"op", Json("branch_capture")}}), "branch_capture", false);
                    require(field(captured, "outcome").string() == "captured" && field(captured, "committed_prefix").unsigned_integer() == position &&
                        field(captured, "common_prefix").unsigned_integer() == position, "branch8 scheduled capture mismatch");
                }
#endif
            }
        } else
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        if (sampled_prefill_
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            || mtp_
#endif
            ) {
            while (position < prompt.size()) {
                cancelled();
                size_t input_packet = 8;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (mtp_) input_packet = mtp_input_packet(mtp_ready_);
#endif
                const auto n = std::min(input_packet, prompt.size() - position);
                Json::Array ids; ids.reserve(n);
                for (size_t t = 0; t < n; ++t) ids.push_back(Json::integer(prompt[position + t]));
                Json::Object job{{"tokens", Json(std::move(ids))}, {"generate", Json::integer(0)},
                    {"reset", Json(!append && !began && position == 0)}};
                if (!append && !began && position == 0) {
                    job.emplace_back("sampling", request.sampling); job.emplace_back("seed", Json::integer(seed));
                }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (mtp_) {
                    if (mtp_final_input_logits(mtp_ready_)) job.emplace_back("prefill_final", Json(position + n == prompt.size()));
                    static_cast<void>(run_mtp(Json(std::move(job)))); continue;
                }
#endif
                const auto done = target(Json(std::move(job)), [](const Json&) { throw FormatError("known input emitted a generated token"); });
                cancelled(); // Never publish serving history after cancelled/ambiguous device work.
                require(field(done, "event").string() == "done" &&
                    field(done, "sampled_prefill_protocol").string() == "sampled-prefill-input8-v1" &&
                    field(done, "committed_prefix").unsigned_integer() == position + n &&
                    field(done, "generated_ids").array().empty() && field(done, "status").string() == "prefilled" &&
                    field(done, "finish_reason").is_null() && field(done, "stop_id").is_null(), "invalid sampled input8 completion");
                worker_execution(done, stack_execution_); sampling_matches(done, position + n);
                for (size_t t = 0; t < n; ++t) committed_.push_back(prompt[position + t]);
                position += n;
            }
        } else
#endif
        for (size_t i = position; i < prompt.size(); ++i) {
            Json::Object job{{"tokens", Json(Json::Array{Json::integer(prompt[i])})}, {"generate", Json::integer(0)}, {"reset", Json(!append && !began && i == 0)}};
            if (!append && !began && i == 0) { job.emplace_back("sampling", request.sampling); job.emplace_back("seed", Json::integer(seed)); }
            const auto done = run(Json(std::move(job)), prompt[i], false);
            require(field(done, "status").string() == "prefilled" && field(done, "finish_reason").is_null() && field(done, "stop_id").is_null(), "invalid prefill termination");
        }
        if (broker_epoch_ && position > input_start
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            && !branch8_
#endif
            ) {
            const auto done = control(Json(Json::Object{{"op", Json("branch_capture")}}), "branch_capture", false);
            require(field(done, "outcome").string() == "captured" && field(done, "committed_prefix").unsigned_integer() == position &&
                field(done, "common_prefix").unsigned_integer() == position, "broker input-boundary capture mismatch");
        }
        const size_t prefill_committed_end = position;
        const auto generation_start = std::chrono::steady_clock::now();
        bool stopped = false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_) {
            controller_committed = controller_emitted = controller_pending = 0; controller_feedback = Json();
            const bool scoped_carry = mtp_request_carry(mtp_ready_);
            if (scoped_carry) { require(mtp_generation_id_ < std::numeric_limits<uint64_t>::max(), "MTP generation ID exhausted"); ++mtp_generation_id_; }
            const size_t output_cap = mtp_output_packet(mtp_ready_);
            while (generated.size() < maximum_tokens) {
                const size_t packet_cap = (generated.empty() && mtp_first_output_packet_)
                    ? std::min<size_t>(*mtp_first_output_packet_, output_cap)
                    : output_cap;
                const auto n = std::min<size_t>(packet_cap, maximum_tokens - generated.size());
                Json::Object job{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(n)}, {"reset", Json(false)},
                    {"stop_ids", Json(Json::Array{Json::integer(codec_.admission.im_end), Json::integer(codec_.admission.end_of_text)})}};
                if (scoped_carry) {
                    Json::Object generation{{"id", Json::integer(mtp_generation_id_)},
                        {"begin", Json(generated.empty())}, {"remaining", Json::integer(maximum_tokens - generated.size())}};
                    if (field(mtp_ready_, "native_mtp_carry_scope").string() == "http-generation-v2") {
                        generation.emplace_back("committed", Json::integer(controller_committed));
                        generation.emplace_back("emitted", Json::integer(controller_emitted));
                        generation.emplace_back("pending", Json::integer(controller_pending));
                        generation.emplace_back("remaining_compute", Json::integer(maximum_tokens - std::min<uint64_t>(maximum_tokens, controller_committed)));
                        generation.emplace_back("total_budget", Json::integer(maximum_tokens));
                        if (!controller_feedback.is_null()) generation.emplace_back("stage_a_controller", controller_feedback);
                    }
                    job.emplace_back("mtp_generation", Json(std::move(generation)));
                }
                const auto done = run_mtp(Json(std::move(job)));
                for (const auto& token : field(done, "generated_ids").array()) {
                    const auto id = uint32_t(token.unsigned_integer());
                    require(id != codec_.admission.im_start && !stopped, "MTP unexpected generated role/terminal");
                    stopped = codec_.admission.is_stop(id); generated.push_back(id);
                    require(stopped || parser.feed(codec_.decode(std::span<const uint32_t>(&generated.back(), 1))) != ResponseStatus::error,
                        "MTP response bytes/frame validation failed"); flush();
                }
                if (stopped) break;
            }
        } else
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        if (verify8_) {
            while (generated.size() < maximum_tokens) {
                const auto n = std::min<size_t>(8, maximum_tokens - generated.size());
                const auto done = run_many(Json(Json::Object{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(n)}, {"reset", Json(false)},
                    {"stop_ids", Json(Json::Array{Json::integer(codec_.admission.im_end), Json::integer(codec_.admission.end_of_text)})}}));
                for (const auto& token : field(done, "generated_ids").array()) {
                    const auto id = uint32_t(token.unsigned_integer());
                    require(id != codec_.admission.im_start && !stopped, "private V8 unexpected generated role/terminal");
                    stopped = codec_.admission.is_stop(id); generated.push_back(id);
                    require(stopped || parser.feed(codec_.decode(std::span<const uint32_t>(&generated.back(), 1))) != ResponseStatus::error, "response bytes/frame validation failed"); flush();
                }
                if (stopped) break;
            }
        } else
#endif
        for (uint32_t i = 0; i < maximum_tokens; ++i) {
            const auto done = run(Json(Json::Object{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(1)}, {"reset", Json(false)},
                {"stop_ids", Json(Json::Array{Json::integer(codec_.admission.im_end), Json::integer(codec_.admission.end_of_text)})}}), {}, true);
            const auto id = committed_.back();
            require(id != codec_.admission.im_start, "unexpected generated role/terminal token");
            stopped = codec_.admission.is_stop(id);
            require(field(done, "finish_reason").string() == (stopped ? "stop" : "length") && field(done, "status").string() == (stopped ? "stopped" : "length"), "worker termination mismatch");
            if (stopped) require(field(done, "stop_id").unsigned_integer() == id && !field(done, "continuation_allowed").boolean(), "invalid terminal admission");
            else require(field(done, "stop_id").is_null() && field(done, "continuation_allowed").boolean(), "invalid continuation admission");
            generated.push_back(id); require(stopped || parser.feed(codec_.decode(std::span<const uint32_t>(&generated.back(), 1))) != ResponseStatus::error, "response bytes/frame validation failed"); flush();
            if (stopped) break;
        }
        Json::Array ids; for (const auto id : generated) ids.push_back(Json::integer(id));
        Json::Object final_req{{"ids", Json(std::move(ids))}, {"tools", parser_manifest(request.manifest)}, {"initial_thinking", Json(request.thinking)},
            {"finish_reason", Json(stopped ? "stop" : "length")}, {"stop_id", stopped ? Json::integer(generated.back()) : Json()}};
        if (tool_args_ == ToolArgValidation::executor) final_req.emplace_back("executor_parameters", Json(true));
        const auto final = parse_native_response(Json(std::move(final_req)), codec_.admission, codec_.decode, response_maximum_accumulated_ids);
        require(!request.single_tool_call || field(final, "tool_calls").array().size() <= 1, "parallel tool calls are disabled for this request");
        Json::Array calls;
        for (const auto& call : field(final, "tool_calls").array()) {
            const auto& function = field(call, "function"); const auto name = field(function, "name").string();
            const auto found = std::find(request.names.begin(), request.names.end(), name); require(found != request.names.end(), "unadmitted final tool");
            if (tool_args_ == ToolArgValidation::executor) validate_executor_tool_arguments(field(function, "arguments"));
            else validate_tool_value(request.schemas.at(size_t(found - request.names.begin())), field(function, "arguments"), tool_enums_);
            calls.emplace_back(Json::Object{{"id", Json(std::string(response_id) + "_call_" + std::to_string(calls.size()))}, {"type", Json("function")},
                {"function", Json(Json::Object{{"name", Json(std::string(name))}, {"arguments", Json(field(function, "arguments").dump())}})}});
        }
        require((!request.named_tool_choice && (request.tool_choice == "auto" || request.tool_choice == "none")) || !stopped || !calls.empty(), "required tool was not produced");
        cancelled();
        const auto content = std::string(field(final, "content").string()), thinking = std::string(field(final, "reasoning_content").string());
        require(content.starts_with(streamed_content) && thinking.starts_with(streamed_thinking), "final response disagrees with streamed prefix");
        if (thinking.size() > streamed_thinking.size()) emit(Json(Json::Object{{"reasoning_content", Json(thinking.substr(streamed_thinking.size()))}}));
        if (content.size() > streamed_content.size()) emit(Json(Json::Object{{"content", Json(content.substr(streamed_content.size()))}}));
        for (size_t i = 0; i < calls.size(); ++i) {
            auto fields = calls[i].object(); fields.emplace_back("index", Json::integer(i));
            emit(Json(Json::Object{{"tool_calls", Json(Json::Array{Json(std::move(fields))})}}));
        }
        const std::string finish = !calls.empty() ? "tool_calls" : stopped ? "stop" : "length";
        emit(Json(Json::Object{}), Json(finish));
        sampling_ = policy; seed_ = seed; reusable_ = true;
        if (!append) saved_reasoning_.clear();
        if (!thinking.empty()) {
            size_t ordinal = 0;
            for (const auto& message : field(request.render, "messages").array()) ordinal += field(message, "role").string() == "assistant";
            saved_reasoning_.emplace_back(ordinal, thinking);
        }
        const auto generation_end = std::chrono::steady_clock::now();
        const auto admission_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(prefill_start - request_start).count();
        const auto prefill_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(generation_start - prefill_start).count();
        const auto generation_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(generation_end - generation_start).count();
        const auto request_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(generation_end - request_start).count();
        const double server_admission_wall_ms = static_cast<double>(admission_ns) / 1e6;
        const double server_prefill_wall_ms = static_cast<double>(prefill_ns) / 1e6;
        const double server_generation_wall_ms = static_cast<double>(generation_ns) / 1e6;
        const double server_request_wall_ms = static_cast<double>(request_ns) / 1e6;
        const double server_generation_throughput = (!generated.empty() && generation_ns > 0)
            ? (static_cast<double>(generated.size()) * 1e9) / static_cast<double>(generation_ns) : 0.0;
        std::string_view prefill_generation_overlap = "none";
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_ && prefill_committed_end > input_start) prefill_generation_overlap = "known_input_mtp_priming";
#else
        static_cast<void>(prefill_committed_end);
#endif
        Json::Object message{{"role", Json("assistant")}, {"content", Json(content)}};
        if (!thinking.empty()) message.emplace_back("reasoning_content", Json(thinking));
        if (!calls.empty()) message.emplace_back("tool_calls", Json(std::move(calls)));
        Json::Object result{{"id", Json(std::string(response_id))}, {"object", Json("chat.completion")}, {"created", Json::integer(created)}, {"model", Json(alias_)},
            {"choices", Json(Json::Array{Json(Json::Object{{"index", Json::integer(0)}, {"message", Json(std::move(message))}, {"finish_reason", Json(finish)}})})},
            {"usage", Json(Json::Object{{"prompt_tokens", Json::integer(prompt.size())}, {"completion_tokens", Json::integer(generated.size())}, {"total_tokens", Json::integer(prompt.size() + generated.size())},
                {"prompt_tokens_details", Json(Json::Object{{"cached_tokens", Json::integer(cached_prefix)}})}})}};
        if (sink.prefix_trace) {
            Json::Object trace{{"event", Json("completion")}, {"schema", Json::integer(1)}, {"response_id", Json(std::string(response_id))},
                {"prompt_tokens", Json::integer(prompt.size())}, {"cached_tokens", Json::integer(cached_prefix)}, {"new_tokens", Json::integer(prompt.size() - cached_prefix)},
                {"completion_tokens", Json::integer(generated.size())}, {"total_tokens", Json::integer(prompt.size() + generated.size())},
                {"finish_reason", Json(finish)}, {"stopped_on_eos", Json(stopped)},
                {"server_admission_wall_ms", real(server_admission_wall_ms)},
                {"server_prefill_wall_ms", real(server_prefill_wall_ms)},
                {"server_generation_wall_ms", real(server_generation_wall_ms)},
                {"server_request_wall_ms", real(server_request_wall_ms)},
                {"server_generation_throughput", real(server_generation_throughput)},
                {"prefill_generation_overlap", Json(std::string(prefill_generation_overlap))}};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (native_trace) trace.emplace_back("native_worker", native_trace->report());
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            if (mtp_ && mtp_ready_.find("native_prefill_dense_kernel")) {
                trace.emplace_back("native_prefill_dense", field(mtp_ready_, "native_prefill_dense"));
                trace.emplace_back("native_prefill_dense_kernel", field(mtp_ready_, "native_prefill_dense_kernel"));
            }
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
            if (mtp_ && mtp_ready_.find("native_raw_decode_mode")) {
                Json::Object route;
                for (const auto name : raw_decode_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                route.emplace_back("admitted_routes", raw_decode_routes_json(raw_decode_admitted));
                route.emplace_back("scope", Json("ported-kernel successful completed physical dispatches from admitted native packets in this HTTP request, only where eligible raw routed calls execute; cancelled unadmitted packets excluded"));
                trace.emplace_back("raw_decode", Json(std::move(route)));
            }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
            if (mtp_) if (const auto* selected=mtp_ready_.find("q8_three_rows_mode");selected&&selected->string()=="on") trace.emplace_back("q8_three_rows",Json(Json::Object{
                {"schema",field(mtp_ready_,"q8_three_rows_schema")},{"mode",*selected},{"padding_rows",field(mtp_ready_,"q8_three_rows_padding_rows")},{"hc_route",field(mtp_ready_,"q8_three_rows_hc_route")},
                {"admitted_routes",Json(q8_three_admitted_routes)},{"scope",Json("completed routes from admitted native packets in this HTTP request only")}}));
#endif
            if (mtp_ && mtp_hc_rows(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_hc_rows_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : hc_rows_admitted) calls.push_back(Json::integer(n));
                route.emplace_back("admitted_encoded_calls", Json(std::move(calls)));
                route.emplace_back("scope", Json("encoded inject/read/write calls from admitted packets only; not completed GPU work"));
                trace.emplace_back("prefill_hc_rows", Json(std::move(route)));
            }
            if (mtp_ && mtp_router_window(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_router_window_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : router_window_admitted) calls.push_back(Json::integer(n));
                route.emplace_back("admitted_encoded_calls", Json(std::move(calls)));
                route.emplace_back("scope", Json("encoded router/route/shared/mixture calls from admitted packets only; not completed GPU work"));
                trace.emplace_back("prefill_router_window", Json(std::move(route)));
            }
            if (mtp_ && mtp_hc_down_window(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_hc_down_window_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                route.emplace_back("admitted_encoded_calls", Json::integer(hc_down_window_admitted));
                route.emplace_back("scope", Json("encoded HC down window calls from admitted packets only; not completed GPU work"));
                trace.emplace_back("prefill_hc_down_window", Json(std::move(route)));
            }
            if (mtp_ && mtp_qsa_sparse(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_qsa_sparse_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : qsa_sparse_admitted) calls.push_back(Json::integer(n));
                route.emplace_back("admitted_completed_calls", Json(std::move(calls)));
                route.emplace_back("scope", Json("completed decode/verify sparse QSA calls from admitted packets only; cancelled unadmitted packets excluded"));
                trace.emplace_back("qsa_sparse", Json(std::move(route)));
            }
            if (mtp_ && mtp_gdn_ported(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_gdn_ported_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : gdn_ported_admitted) calls.push_back(Json::integer(n));
                route.emplace_back("admitted_completed_calls", Json(std::move(calls)));
                route.emplace_back("scope", Json("completed prefill/decode/verify/replay/norm_gate ported GDN calls from admitted packets only; cancelled unadmitted packets excluded"));
                trace.emplace_back("gdn_ported", Json(std::move(route)));
            }
            if (mtp_ && mtp_hc_down_f32(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_hc_down_f32_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                route.emplace_back("admitted_encoded_calls", Json::integer(hc_down_f32_admitted_calls));
                route.emplace_back("scope", Json("encoded calls from admitted native packets only; cancelled/rejected packets excluded; not completed GPU work"));
                trace.emplace_back("prefill_hc_down_f32", Json(std::move(route)));
            }
            if (mtp_ && mtp_gdn_old_order(mtp_ready_)) trace.emplace_back("gdn_old_order", Json(Json::Object{
                {"version", field(mtp_ready_, "native_gdn_old_order_version")}, {"kernel", field(mtp_ready_, "native_gdn_old_order_kernel")},
                {"admitted_encoded_calls", Json::integer(gdn_old_order_admitted_calls)}, {"scope", Json("admitted native packets in this HTTP request only")}}));
#endif
#endif
            sink.prefix_trace(Json(std::move(trace)));
        }
        if (request.stream && request.include_usage) {
            cancelled();
            sink.chunk(Json(Json::Object{{"id", Json(std::string(response_id))}, {"object", Json("chat.completion.chunk")},
                {"created", Json::integer(created)}, {"model", Json(alias_)}, {"choices", Json(Json::Array{})},
                {"usage", field(Json(result), "usage")}}));
        }
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        if (verify8_) result.emplace_back("tokspur_private", Json(Json::Object{{"native_verify8", Json(true)}, {"verification_max_tokens", Json::integer(8)},
            {"known_input_batches", Json::integer(input_batches)}, {"maximum_known_input_width", Json::integer(input_width)},
            {"generation_batches", Json::integer(output_batches)}, {"maximum_generation_width", Json::integer(output_width)},
            {"lookup_rounds", Json::integer(generation_rounds)}, {"lookup_verified_rounds", Json::integer(verified_rounds)}, {"lookup_accepted_tokens", Json::integer(accepted_tokens)},
            {"hc_exact_final_logit_sha256", field(verify_previous_, "hc_exact_final_logit_sha256")},
            {"performance_qualified", Json(false)}}));
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_) result.emplace_back("tokspur_private", Json(Json::Object{{"native_mtp", Json(true)},
            {"native_mtp_protocol", field(mtp_ready_, "native_mtp_protocol")}, {"known_input_batches", Json::integer(input_batches)},
            {"maximum_known_input_width", Json::integer(input_width)}, {"generation_batches", Json::integer(output_batches)},
            {"maximum_generation_width", Json::integer(output_width)}, {"mtp_rounds", Json::integer(generation_rounds)},
            {"mtp_accepted_tokens", Json::integer(accepted_tokens)},
            {"performance_qualified", Json(false)}}));
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if (mtp_ && mtp_ready_.find("native_raw_decode_mode")) {
            auto details = field(Json(result), "tokspur_private").object();
            for (const auto name : raw_decode_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
            details.emplace_back("native_raw_decode_admitted_routes", raw_decode_routes_json(raw_decode_admitted));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if (mtp_) if (const auto* selected=mtp_ready_.find("q8_three_rows_mode");selected&&selected->string()=="on") {
            auto details=field(Json(result),"tokspur_private").object();
            for(const auto name:{"q8_three_rows_schema","q8_three_rows_mode","q8_three_rows_padding_rows","q8_three_rows_hc_route"})details.emplace_back(name,field(mtp_ready_,name));
            details.emplace_back("q8_three_rows_admitted_routes",Json(q8_three_admitted_routes));
            for(auto& [name,value]:result)if(name=="tokspur_private")value=Json(std::move(details));
        }
#endif
        if (mtp_ && mtp_input_packet(mtp_ready_) >= 512) {
            auto details = field(Json(result), "tokspur_private").object();
            for (const auto name : {"native_mtp_input_packet_schema", "native_mtp_input_packet_max", "native_mtp_priming_batch_capacity", "native_mtp_priming_batch_bytes", "native_prefill_dense"})
                details.emplace_back(name, field(mtp_ready_, name));
            for (const auto name : {"native_prefill_grouped", "native_prefill_grouped_kernel", "native_prefill_dense_kernel"})
                if (const auto* value = mtp_ready_.find(name)) details.emplace_back(name, *value);
            if (mtp_hc_rows(mtp_ready_)) {
                for (const auto name : mtp_hc_rows_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : hc_rows_admitted) calls.push_back(Json::integer(n));
                details.emplace_back("native_prefill_hc_rows_admitted_encoded_calls", Json(std::move(calls)));
                details.emplace_back("native_prefill_hc_rows_admitted_scope", Json("encoded inject/read/write calls from admitted packets only; not completed GPU work"));
            }
            if (mtp_router_window(mtp_ready_)) {
                for (const auto name : mtp_router_window_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : router_window_admitted) calls.push_back(Json::integer(n));
                details.emplace_back("native_prefill_router_window_admitted_encoded_calls", Json(std::move(calls)));
                details.emplace_back("native_prefill_router_window_admitted_scope", Json("encoded router/route/shared/mixture calls from admitted packets only; not completed GPU work"));
            }
            if (mtp_hc_down_window(mtp_ready_)) {
                for (const auto name : mtp_hc_down_window_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                details.emplace_back("native_prefill_hc_down_window_admitted_encoded_calls", Json::integer(hc_down_window_admitted));
                details.emplace_back("native_prefill_hc_down_window_admitted_scope", Json("encoded HC down window calls from admitted packets only; not completed GPU work"));
            }
            if (mtp_qsa_sparse(mtp_ready_)) {
                for (const auto name : mtp_qsa_sparse_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : qsa_sparse_admitted) calls.push_back(Json::integer(n));
                details.emplace_back("native_qsa_sparse_admitted_completed_calls", Json(std::move(calls)));
                details.emplace_back("native_qsa_sparse_admitted_scope", Json("completed decode/verify sparse QSA calls from admitted packets only; cancelled unadmitted packets excluded"));
            }
            if (mtp_gdn_ported(mtp_ready_)) {
                for (const auto name : mtp_gdn_ported_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : gdn_ported_admitted) calls.push_back(Json::integer(n));
                details.emplace_back("native_gdn_ported_admitted_completed_calls", Json(std::move(calls)));
                details.emplace_back("native_gdn_ported_admitted_scope", Json("completed prefill/decode/verify/replay/norm_gate ported GDN calls from admitted packets only; cancelled unadmitted packets excluded"));
            }
            if (mtp_hc_down_f32(mtp_ready_)) {
                for (const auto name : mtp_hc_down_f32_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                details.emplace_back("native_prefill_hc_down_f32_admitted_encoded_calls", Json::integer(hc_down_f32_admitted_calls));
                details.emplace_back("native_prefill_hc_down_f32_admitted_scope", Json("encoded calls from admitted native packets only; cancelled/rejected packets excluded; not completed GPU work"));
            }
            if (mtp_gdn_old_order(mtp_ready_)) {
                for (const auto name : {"native_gdn_old_order_version", "native_gdn_old_order_kernel", "native_gdn_old_order_layers", "native_gdn_old_order_scratch_bytes"}) details.emplace_back(name, field(mtp_ready_, name));
                details.emplace_back("native_gdn_old_order_admitted_encoded_calls", Json::integer(gdn_old_order_admitted_calls));
            }
            if (mtp_final_input_logits(mtp_ready_)) {
                details.emplace_back("native_prefill_output_policy", field(mtp_ready_, "native_prefill_output_policy"));
                details.emplace_back("logits_available", field(mtp_previous_, "logits_available"));
            }
            details.emplace_back("native_mtp_priming_batch_submits", field(mtp_previous_, "native_mtp_priming_batch_submits"));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        if (mtp_ && (mtp_depth(mtp_ready_) == 2 || mtp_depth(mtp_ready_) == 3 || mtp_depth(mtp_ready_) == 7)) {
            auto details = field(Json(result), "tokspur_private").object();
            details.emplace_back("native_mtp_depth", Json::integer(mtp_depth(mtp_ready_)));
            details.emplace_back("native_mtp_tail_policy", field(mtp_ready_, "native_mtp_tail_policy"));
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
            if (mtp_ready_.find("native_mtp_output_packet")) {
                details.emplace_back("native_mtp_output_packet", field(mtp_ready_, "native_mtp_output_packet"));
                details.emplace_back("native_mtp_output_packet_schema", field(mtp_ready_, "native_mtp_output_packet_schema"));
            }
#endif
            if (mtp_first_output_packet_) details.emplace_back("native_mtp_first_output_packet", Json::integer(*mtp_first_output_packet_));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#endif
        if (mtp_ && mtp_request_carry(mtp_ready_)) {
            auto details = field(Json(result), "tokspur_private").object();
            details.emplace_back("native_mtp_carry_scope", field(mtp_ready_, "native_mtp_carry_scope"));
            details.emplace_back("native_mtp_generation", field(mtp_previous_, "native_mtp_generation"));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (branch8_) {
            auto details = field(Json(result), "tokspur_private").object();
            details.emplace_back("branch8_reuse", Json(true)); details.emplace_back("broker_protocol", Json("branch8-v1"));
            details.emplace_back("checkpoint_captures", Json::integer(checkpoint_captures)); details.emplace_back("checkpoint_restores", Json::integer(checkpoint_restores));
            details.emplace_back("checkpoint_cold_resyncs", Json::integer(checkpoint_cold_resyncs));
            details.emplace_back("checkpoint_slots", field(verify_ready_, "checkpoint_slots"));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        return project_weight_lanes(Json(std::move(result)), weight_lanes_identity_);
#else
        return Json(std::move(result));
#endif
    } catch (...) { if (state_touched) invalidate(); throw; }
}
} // namespace tokspur
