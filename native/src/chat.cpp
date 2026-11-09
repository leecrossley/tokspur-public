#include "tokspur/chat.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/unicode.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <functional>
#include <limits>
#include <set>

namespace tokspur {
namespace {

constexpr size_t output_limit = 8 * 1024 * 1024;

class Builder {
public:
    void add(std::string_view value) {
        if (value.size() > output_limit - value_.size()) throw FormatError("rendered chat exceeds byte limit");
        value_.append(value);
    }
    void literal(std::string_view value) {
        const auto start = value_.size(); add(value);
        if (start == value_.size()) return;
        if (!literal_ranges.empty() && literal_ranges.back().second == start) literal_ranges.back().second = value_.size();
        else literal_ranges.emplace_back(start, value_.size());
    }
    void parameter(std::string_view value) {
        if (value.find("\n</parameter>") != std::string_view::npos) throw FormatError("raw tool argument contains parameter delimiter");
        literal(value);
    }
    std::vector<std::pair<size_t, size_t>> literal_ranges;
    std::string finish() { return std::move(value_); }
private:
    std::string value_;
};

bool whitespace(uint32_t c) {
    // Python-style str.strip semantics used by the pinned template, not NFC.
    return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

void require(bool condition, const char* message) {
    if (!condition) throw FormatError(message);
}

bool option(const Json& request, std::string_view name, bool fallback) {
    const auto* value = request.find(name);
    return value ? value->boolean() : fallback;
}

std::string_view role(const Json& message) {
    const auto* value = message.find("role");
    if (!value) throw FormatError("chat message lacks role");
    const auto name = value->string();
    if (name != "system" && name != "developer" && name != "user" && name != "assistant" && name != "tool") throw FormatError("unexpected message role");
    return name;
}

std::string content(const Json* value) {
    if (!value || value->is_null()) return {};
    if (value->kind() == Json::Kind::string) return std::string(value->string());
    if (value->kind() != Json::Kind::array) throw FormatError("unexpected text content type");
    Builder output;
    for (const auto& item : value->array()) {
        const auto* type = item.find("type");
        if (item.find("image") || item.find("image_url") || item.find("video") ||
            (type && (type->string() == "image" || type->string() == "video"))) throw FormatError("text-only chat does not accept images or video");
        require(!type || type->string() == "text", "unsupported text content item type");
        const auto* text = item.find("text");
        if (!text) throw FormatError("unexpected text content item");
        output.add(text->string());
    }
    return output.finish();
}

std::string message_content(const Json& message) { return chat_trim(content(message.find("content"))); }

Json replace(Json::Object fields, std::string_view name, Json value) {
    for (auto& [key, item] : fields) {
        if (key == name) { item = std::move(value); return Json(std::move(fields)); }
    }
    fields.emplace_back(std::string(name), std::move(value));
    return Json(std::move(fields));
}

Json normalize_arguments(const Json& message, size_t& parsed) {
    const auto* calls = message.find("tool_calls");
    if (!calls || calls->is_null()) return message;
    Json::Array result;
    require(calls->array().size() <= 128, "too many tool calls in one message");
    for (const auto& call : calls->array()) {
        const auto* wrapped = call.find("function");
        const Json& function = wrapped ? *wrapped : call;
        const auto* args = function.find("arguments");
        if (args && args->kind() == Json::Kind::string && !chat_trim(args->string()).empty()) {
            Json object = parse_json(args->string());
            static_cast<void>(object.object());
            auto changed = replace(function.object(), "arguments", std::move(object));
            result.push_back(wrapped ? replace(call.object(), "function", std::move(changed)) : std::move(changed));
            ++parsed;
        } else result.push_back(call);
    }
    return replace(message.object(), "tool_calls", Json(std::move(result)));
}

void safe_name(std::string_view name, bool function) {
    require(!name.empty() && name.size() <= (function ? 64 : 1024), "invalid tool or parameter name length");
    require(name.find_first_of("<>\r\n\0", 0, 5) == std::string_view::npos, "unsafe tool or parameter name");
    if (function) {
        for (unsigned char c : name) require((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.', "unsupported tool function name");
    }
}

std::string float_json(std::string_view lexeme) {
    double value = 0;
    const auto parsed = std::from_chars(lexeme.data(), lexeme.data() + lexeme.size(), value, std::chars_format::general);
    if (parsed.ec == std::errc::result_out_of_range) {
        // A decimal too small for binary64 rounds to signed zero. Overflow is
        // rejected, rather than emitting a non-JSON Infinity into tool schemas.
        size_t cursor = lexeme.front() == '-' ? 1 : 0;
        int64_t before_point = 0;
        int64_t first_nonzero = -1;
        int64_t digits_seen = 0;
        bool fraction = false;
        for (; cursor < lexeme.size() && lexeme[cursor] != 'e' && lexeme[cursor] != 'E'; ++cursor) {
            if (lexeme[cursor] == '.') { fraction = true; continue; }
            if (!fraction) ++before_point;
            if (first_nonzero < 0 && lexeme[cursor] != '0') first_nonzero = digits_seen;
            ++digits_seen;
        }
        int64_t exponent = 0;
        bool negative_exponent = false;
        if (cursor < lexeme.size()) {
            ++cursor;
            if (lexeme[cursor] == '-' || lexeme[cursor] == '+') negative_exponent = lexeme[cursor++] == '-';
            for (; cursor < lexeme.size(); ++cursor) exponent = std::min(int64_t(1'000'000), exponent * 10 + lexeme[cursor] - '0');
        }
        exponent = (negative_exponent ? -exponent : exponent) + before_point - first_nonzero - 1;
        if (first_nonzero < 0 || exponent <= -324) value = lexeme.front() == '-' ? -0.0 : 0.0;
        else throw FormatError("nonfinite JSON number in chat");
    } else if (parsed.ec != std::errc{} || parsed.ptr != lexeme.data() + lexeme.size() || !std::isfinite(value)) {
        throw FormatError("unsupported JSON number in chat");
    }
    const bool negative = std::signbit(value);
    if (value == 0) return negative ? "-0.0" : "0.0";
    std::array<char, 128> buffer{};
    const auto rendered = std::to_chars(buffer.data(), buffer.data() + buffer.size(), std::abs(value), std::chars_format::scientific);
    if (rendered.ec != std::errc{}) throw FormatError("cannot format chat number");
    const std::string_view scientific(buffer.data(), rendered.ptr);
    const auto separator = scientific.find('e');
    require(separator != std::string_view::npos, "missing scientific exponent");
    int exponent = 0;
    size_t at = separator + 1;
    const bool exponent_negative = scientific[at++] == '-';
    for (; at < scientific.size(); ++at) exponent = exponent * 10 + scientific[at] - '0';
    if (exponent_negative) exponent = -exponent;
    std::string output = negative ? "-" : "";
    if (exponent < -4 || exponent >= 16) {
        output.append(scientific.substr(0, separator));
        output += exponent < 0 ? "e-" : "e+";
        const auto magnitude = std::abs(exponent);
        if (magnitude < 10) output += '0';
        output += std::to_string(magnitude);
    } else {
        std::string digits;
        for (const char c : scientific.substr(0, separator)) if (c != '.') digits += c;
        const auto position = exponent + 1;
        if (position <= 0) output += "0." + std::string(static_cast<size_t>(-position), '0') + digits;
        else if (static_cast<size_t>(position) >= digits.size()) output += digits + std::string(static_cast<size_t>(position) - digits.size(), '0') + ".0";
        else output += digits.substr(0, static_cast<size_t>(position)) + "." + digits.substr(static_cast<size_t>(position));
    }
    return output;
}

std::vector<std::string> literals(std::string_view source) {
    require(source.size() <= 64 * 1024, "template source exceeds profile limit");
    std::vector<std::string> result;
    for (size_t i = 0; i < source.size();) {
        if (source.substr(i, 2) == "{#") {
            const auto end = source.find("#}", i + 2);
            if (end == std::string_view::npos) throw FormatError("unterminated template comment");
            i = end + 2;
            continue;
        }
        const char quote = source[i++];
        if (quote != '\'' && quote != '"') continue;
        std::string value;
        bool closed = false;
        while (i < source.size()) {
            const char c = source[i++];
            if (c == quote) { closed = true; break; }
            if (c != '\\') { value += c; continue; }
            if (i == source.size()) throw FormatError("truncated template literal");
            switch (const char escaped = source[i++]) {
                case 'n': value += '\n'; break;
                case 'r': value += '\r'; break;
                case 't': value += '\t'; break;
                case '\\': value += '\\'; break;
                case '\'': value += '\''; break;
                case '"': value += '"'; break;
                default: static_cast<void>(escaped); throw FormatError("unsupported template literal escape");
            }
        }
        require(closed, "unterminated template literal");
        result.push_back(std::move(value));
    }
    return result;
}

} // namespace

std::string chat_trim(std::string_view value) {
    const auto points = unicode_scalars(value);
    size_t start = 0;
    size_t end = points.size();
    while (start < end && whitespace(points[start])) ++start;
    while (end > start && whitespace(points[end - 1])) --end;
    std::string output;
    for (size_t i = start; i < end; ++i) output += scalar_utf8(points[i]);
    return output;
}

std::string template_json(const Json& value) {
    Builder output;
    const auto quoted = [&](std::string_view text) {
        static_cast<void>(unicode_scalars(text));
        output.add("\"");
        for (unsigned char c : text) {
            switch (c) {
                case '"': output.add("\\\""); break;
                case '\\': output.add("\\\\"); break;
                case '\b': output.add("\\b"); break;
                case '\f': output.add("\\f"); break;
                case '\n': output.add("\\n"); break;
                case '\r': output.add("\\r"); break;
                case '\t': output.add("\\t"); break;
                default:
                    if (c < 32) {
                        constexpr char hex[] = "0123456789abcdef";
                        const std::array<char, 6> escaped{'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
                        output.add({escaped.data(), escaped.size()});
                    } else { const char byte = static_cast<char>(c); output.add({&byte, 1}); }
            }
        }
        output.add("\"");
    };
    std::function<void(const Json&, size_t)> visit = [&](const Json& item, size_t depth) {
        require(depth <= 64, "chat JSON exceeds nesting limit");
        switch (item.kind()) {
            case Json::Kind::null_value: output.add("null"); break;
            case Json::Kind::boolean: output.add(item.boolean() ? "true" : "false"); break;
            case Json::Kind::number: {
                const auto text = item.number_lexeme();
                if (text.find_first_of(".eE") != std::string_view::npos) output.add(float_json(text));
                else { require(text.size() <= 4300, "chat integer exceeds digit limit"); output.add(text == "-0" ? "0" : text); }
                break;
            }
            case Json::Kind::string: quoted(item.string()); break;
            case Json::Kind::array:
                output.add("[");
                for (size_t i = 0; i < item.array().size(); ++i) { if (i) output.add(", "); visit(item.array()[i], depth + 1); }
                output.add("]");
                break;
            case Json::Kind::object:
                output.add("{");
                for (size_t i = 0; i < item.object().size(); ++i) {
                    if (i) output.add(", ");
                    quoted(item.object()[i].first); output.add(": "); visit(item.object()[i].second, depth + 1);
                }
                output.add("}");
                break;
        }
    };
    visit(value, 0);
    return output.finish();
}

ChatStrings extract_flash_next_literals(std::string_view source) {
    const auto values = literals(source);
    const auto select = [&](std::string_view prefix) {
        std::string result;
        size_t matches = 0;
        for (const auto& value : values) if (value.starts_with(prefix)) { result = value; ++matches; }
        require(matches == 1, "pinned template literal missing or ambiguous");
        return result;
    };
    return {select("Reasoning effort is set to xhigh."), select("Reasoning effort is set to low."),
            select("# Tools\n\n"), select("\n\nIf you choose to call a function ONLY reply")};
}

ChatRenderer::ChatRenderer(ChatStrings strings) : strings_(std::move(strings)) {
    require(!strings_.reasoning_xhigh.empty() && !strings_.reasoning_low.empty() &&
            !strings_.tools_header.empty() && !strings_.tool_instructions.empty(), "chat profile lacks model literals");
}

RenderedChat ChatRenderer::render(const Json& request) const {
    for (const auto& [name, value] : request.object()) {
        static_cast<void>(value);
        require(name == "messages" || name == "tools" || name == "enable_thinking" || name == "preserve_thinking" ||
                name == "add_generation_prompt" || name == "reasoning_effort" || name == "late_system_policy" ||
                name == "parse_tool_arguments", "unknown chat request field");
    }
    const auto* raw_messages = request.find("messages");
    require(raw_messages, "no messages provided");
    auto messages = raw_messages->array();
    require(!messages.empty() && messages.size() <= 4096, "chat message count is empty or exceeds limit");
    RenderedChat result;
    const bool parse_arguments = option(request, "parse_tool_arguments", true);
    for (auto& message : messages) {
        const auto who = role(message);
        static_cast<void>(message_content(message));
        if (parse_arguments && who == "assistant") message = normalize_arguments(message, result.parsed_tool_arguments);
    }
    std::string late_policy = "reject";
    if (const auto* value = request.find("late_system_policy")) late_policy = value->string();
    require(late_policy == "reject" || late_policy == "hoist", "unsupported late system message policy");
    if (late_policy == "hoist") {
        Json::Array leading;
        Json::Array rest;
        bool seen_non_system = false;
        for (auto& message : messages) {
            const auto who = role(message);
            if (who == "system" || who == "developer") {
                if (seen_non_system) ++result.hoisted_system_messages;
                leading.push_back(std::move(message));
            } else { seen_non_system = true; rest.push_back(std::move(message)); }
        }
        leading.insert(leading.end(), std::make_move_iterator(rest.begin()), std::make_move_iterator(rest.end()));
        messages = std::move(leading);
    }
    const bool thinking = option(request, "enable_thinking", true);
    const bool preserve = option(request, "preserve_thinking", true);
    std::string instructions;
    if (thinking) {
        std::string_view effort = "xhigh";
        if (const auto* value = request.find("reasoning_effort")) effort = value->string();
        if (effort == "high") effort = "xhigh";
        require(effort == "xhigh" || effort == "medium" || effort == "low", "unexpected reasoning effort");
        if (effort == "xhigh") instructions = strings_.reasoning_xhigh;
        else if (effort == "low") instructions = strings_.reasoning_low;
    }
    size_t leading_count = 0;
    Builder system;
    bool has_system = false;
    while (leading_count < messages.size() && (role(messages[leading_count]) == "system" || role(messages[leading_count]) == "developer")) {
        const auto text = message_content(messages[leading_count++]);
        if (!text.empty()) { if (has_system) system.add("\n"); system.add(text); has_system = true; }
    }
    const auto merged_system = system.finish();
    const Json::Array* tools = nullptr;
    if (const auto* value = request.find("tools"); value && !value->is_null()) tools = &value->array();
    if (tools) require(tools->size() <= 512, "too many tools");
    Builder output;
    if (tools && !tools->empty()) {
        output.add("<|im_start|>system\n");
        if (!instructions.empty()) { output.add(instructions); output.add("\n\n"); }
        output.add(strings_.tools_header);
        for (const auto& tool : *tools) { static_cast<void>(tool.object()); output.add("\n"); output.literal(template_json(tool)); }
        output.add("\n</tools>"); output.add(strings_.tool_instructions);
        if (!merged_system.empty()) { output.add("\n\n"); output.literal(merged_system); }
        output.add("<|im_end|>\n");
    } else if (!merged_system.empty() || !instructions.empty()) {
        output.add("<|im_start|>system\n");
        if (!instructions.empty()) { output.add(instructions); if (!merged_system.empty()) output.add("\n\n"); }
        output.literal(merged_system); output.add("<|im_end|>\n");
    }
    size_t last_query = messages.size() - 1;
    for (size_t i = messages.size(); i-- > 0;) {
        if (role(messages[i]) != "user") continue;
        last_query = i; break;
    }
    for (size_t i = leading_count; i < messages.size(); ++i) {
        const auto& message = messages[i];
        const auto who = role(message);
        const auto text = message_content(message);
        require(who != "system" && who != "developer", "system message must be at the beginning or explicitly hoisted");
        if (who == "user") { output.add("<|im_start|>user\n"); output.literal(text); output.add("<|im_end|>\n"); }
        else if (who == "assistant") {
            output.add("<|im_start|>assistant\n");
            if (preserve || i > last_query) {
                std::string reasoning;
                if (const auto* value = message.find("reasoning_content"); value && !value->is_null()) reasoning = chat_trim(value->string());
                output.add("<think>\n"); output.literal(reasoning); output.add("\n</think>\n\n");
            }
            output.literal(text);
            if (const auto* value = message.find("tool_calls"); value && !value->is_null()) {
                const auto& calls = value->array();
                require(calls.size() <= 128, "too many tool calls");
                for (size_t call_index = 0; call_index < calls.size(); ++call_index) {
                    const auto& call = calls[call_index];
                    const auto* wrapped = call.find("function");
                    const Json& function = wrapped ? *wrapped : call;
                    const auto* name = function.find("name");
                    require(name && !name->is_null(), "tool call lacks function name");
                    safe_name(name->string(), true);
                    output.add(call_index ? "\n<tool_call>\n<function=" : (!text.empty() ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function="));
                    output.add(name->string()); output.add(">\n");
                    if (const auto* args = function.find("arguments"); args && !args->is_null()) {
                        if (args->kind() == Json::Kind::object) {
                            require(args->object().size() <= 256, "too many tool parameters");
                            for (const auto& [key, argument] : args->object()) {
                                safe_name(key, false);
                                output.add("<parameter="); output.add(key); output.add(">\n");
                                if (argument.kind() == Json::Kind::string) output.parameter(argument.string());
                                else output.parameter(template_json(argument));
                                output.add("\n</parameter>\n");
                            }
                        } else if (args->kind() == Json::Kind::string) require(chat_trim(args->string()).empty(), "parse nonempty JSON-string tool arguments before raw template rendering");
                        else throw FormatError("tool arguments must be an object or empty string");
                    }
                    output.add("</function>\n</tool_call>");
                }
            }
            output.add("<|im_end|>\n");
        } else if (who == "tool") {
            // Preserve the pinned formatter's exact treatment of history fragments.
            // Tool-call ID/sequence admission is a separate API integrity gate.
            // The pinned template checks loop.previtem, not loop.first; a leading
            // tool fragment therefore has no user opener. API admission is separate.
            if (i && role(messages[i - 1]) != "tool") output.add("<|im_start|>user");
            output.add("\n<tool_response>\n"); output.literal(text); output.add("\n</tool_response>");
            if (i + 1 == messages.size() || role(messages[i + 1]) != "tool") output.add("<|im_end|>\n");
        }
    }
    if (option(request, "add_generation_prompt", true)) {
        output.add("<|im_start|>assistant\n");
        output.add(thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
    }
    result.literal_ranges = std::move(output.literal_ranges);
    result.prompt = output.finish();
    result.normalized_messages = Json(std::move(messages));
    return result;
}

} // namespace tokspur
