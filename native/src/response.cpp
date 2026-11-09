#include "tokspur/response.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/unicode.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <set>

namespace tokspur {
namespace {
bool space(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
std::string_view trim(std::string_view value) {
    while (!value.empty() && space(value.front())) value.remove_prefix(1);
    while (!value.empty() && space(value.back())) value.remove_suffix(1);
    return value;
}
void require(bool condition, const char* message) { if (!condition) throw FormatError(message); }
void name(std::string_view value, bool function) {
    require(!value.empty() && value.size() <= (function ? 64 : 1024), "response name length limit");
    static_cast<void>(unicode_scalars(value));
    for (const unsigned char c : value) {
        require(c >= 32 && c != '<' && c != '>' && c != 127, "unsafe response name");
        if (function) require((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.', "unsupported response function name");
    }
}
void finite_numbers(const Json& value) {
    if (value.kind() == Json::Kind::number) {
        const auto text = value.number_lexeme();
        double number = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), number);
        require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(number),
                "unrepresentable response argument number");
    } else if (value.kind() == Json::Kind::array) {
        for (const auto& item : value.array()) finite_numbers(item);
    } else if (value.kind() == Json::Kind::object) {
        for (const auto& [key, item] : value.object()) { static_cast<void>(key); finite_numbers(item); }
    }
}
constexpr std::array<std::string_view, 9> markers{
    "<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>",
    "<parameter=", "</parameter>", "<|im_end|>"};
size_t held_suffix(std::string_view value, std::span<const std::string_view> tokens) {
    size_t held = 0;
    for (const auto token : tokens) {
        for (size_t n = 1; n < token.size() && n <= value.size(); ++n)
            if (value.ends_with(token.substr(0, n))) held = std::max(held, n);
    }
    return held;
}
size_t complete_utf8(std::string_view value) {
    if (value.empty()) return 0;
    size_t start = value.size() - 1;
    while (start > 0 && (static_cast<unsigned char>(value[start]) & 0xc0) == 0x80) --start;
    const auto c = static_cast<unsigned char>(value[start]);
    const size_t expected = c < 128 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
    return value.size() - start < expected ? start : value.size();
}
} // namespace

void validate_response_parameter_markers(std::string_view raw) {
    require(raw.find("<tool_call>") == std::string_view::npos && raw.find("</tool_call>") == std::string_view::npos &&
        raw.find("<function=") == std::string_view::npos && raw.find("<parameter=") == std::string_view::npos &&
        raw.find("</function>") == std::string_view::npos && raw.find("</parameter>") == std::string_view::npos &&
        raw.find("<think>") == std::string_view::npos && raw.find("</think>") == std::string_view::npos &&
        raw.find("<|im_end|>") == std::string_view::npos && raw.find("<|im_start|>") == std::string_view::npos,
        "reserved response marker inside parameter");
}

ResponseParser::ResponseParser(std::vector<ToolSpec> tools, bool initial_thinking, ResponseLimits limits, bool executor_parameters, bool token_framing)
    : tools_(std::move(tools)), limits_(limits), mode_(initial_thinking ? Mode::thinking : Mode::text), thought_(initial_thinking), executor_parameters_(executor_parameters), token_framing_(token_framing) {
    require(limits.bytes && limits.bytes <= 8 * 1024 * 1024 && limits.call_bytes && limits.call_bytes <= limits.bytes &&
        limits.records && limits.records <= 65536 && limits.calls && limits.calls <= 128 &&
        limits.json_nodes && limits.json_nodes <= 1'000'000 && limits.json_depth && limits.json_depth <= 64,
        "invalid response limits");
    require(tools_.size() <= 512, "too many response tools");
    std::set<std::string, std::less<>> names;
    size_t schema_bytes = 0;
    for (const auto& tool : tools_) {
        name(tool.name, true);
        require(names.insert(tool.name).second && tool.parameters.size() <= 256, "duplicate tool or too many response parameters");
        schema_bytes += tool.name.size();
        std::set<std::string, std::less<>> parameters;
        for (const auto& parameter : tool.parameters) {
            name(parameter.name, false);
            require(uint32_t(parameter.type) <= uint32_t(ParameterType::json_value) && parameters.insert(parameter.name).second,
                    "invalid or duplicate response parameter spec");
            schema_bytes += parameter.name.size();
            require(schema_bytes <= 1024 * 1024, "response schema byte limit");
        }
    }
}

void ResponseParser::fail(std::string_view message) {
    status_ = ResponseStatus::error;
    error_ = message;
    calls_.clear();
    std::erase_if(records_, [](const auto& record) { return record.kind == ResponseKind::tool; });
}

void ResponseParser::utf8(std::string_view bytes) {
    for (const unsigned char c : bytes) {
        if (remaining_) {
            require((c & 0xc0) == 0x80, "invalid response UTF-8 continuation");
            point_ = (point_ << 6) | (c & 63);
            if (--remaining_ == 0) require(point_ >= minimum_ && point_ <= 0x10ffff &&
                !(point_ >= 0xd800 && point_ <= 0xdfff), "invalid response Unicode scalar");
        } else if (c >= 128) {
            require(c >= 0xc2 && c <= 0xf4, "invalid response UTF-8 lead");
            remaining_ = c < 0xe0 ? 1 : c < 0xf0 ? 2 : 3;
            minimum_ = remaining_ == 1 ? 0x80 : remaining_ == 2 ? 0x800 : 0x10000;
            point_ = c & ((1U << (6 - remaining_)) - 1);
        } else require(c != 0, "NUL byte in response");
    }
}

void ResponseParser::emit(ResponseKind kind, std::string_view value) {
    if (value.empty()) return;
    if (kind == ResponseKind::text && !trim(value).empty()) visible_ = true;
    if (!records_.empty() && records_.back().kind == kind) records_.back().text.append(value);
    else {
        require(records_.size() + calls_.size() < limits_.records, "response record limit");
        records_.push_back({kind, std::string(value), 0, {}, {}});
    }
}

void ResponseParser::parse_call(std::string_view body) {
    require(calls_.size() < limits_.calls, "response call count limit");
    body = trim(body);
    require(body.starts_with("<function="), "response call lacks function frame");
    body.remove_prefix(10);
    const auto close = body.find('>');
    require(close != std::string_view::npos, "response function name is incomplete");
    const auto function_name = body.substr(0, close);
    name(function_name, true);
    const auto tool = std::find_if(tools_.begin(), tools_.end(), [&](const auto& item) { return item.name == function_name; });
    require(tool != tools_.end(), "unknown response tool");
    body.remove_prefix(close + 1);
    Json::Object arguments;
    std::set<std::string, std::less<>> seen;
    while (true) {
        body = trim(body);
        if (body.starts_with("</function>")) {
            body.remove_prefix(11);
            require(trim(body).empty(), "data after response function frame");
            break;
        }
        require(body.starts_with("<parameter="), "invalid response parameter frame");
        body.remove_prefix(11);
        const auto end = body.find('>');
        require(end != std::string_view::npos, "incomplete response parameter name");
        const auto key = body.substr(0, end);
        name(key, false);
        require(seen.insert(std::string(key)).second, "duplicate response parameter");
        const auto spec = std::find_if(tool->parameters.begin(), tool->parameters.end(), [&](const auto& item) { return item.name == key; });
        require(executor_parameters_ || spec != tool->parameters.end(), "unknown response parameter");
        body.remove_prefix(end + 1);
        require(body.starts_with('\n'), "response parameter lacks template newline");
        body.remove_prefix(1);
        const auto stop = body.find("\n</parameter>");
        require(stop != std::string_view::npos, "incomplete response parameter value");
        const auto framed = body.substr(0, stop);
        const auto raw = framed;
        if (!token_framing_) validate_response_parameter_markers(framed);
        Json value;
        if (spec != tool->parameters.end() && spec->type == ParameterType::raw_string) value = Json(std::string(raw));
        else if (spec != tool->parameters.end() && spec->type == ParameterType::json_value) {
            try { value = parse_json(trim(raw), {limits_.call_bytes, limits_.json_nodes, limits_.json_depth}); }
            catch (const FormatError&) { value = Json(std::string(raw)); }
            finite_numbers(value);
        }
        else if (!executor_parameters_) {
            value = parse_json(trim(raw), {limits_.call_bytes, limits_.json_nodes, limits_.json_depth});
            const auto expected = spec->type == ParameterType::boolean ? Json::Kind::boolean :
                spec->type == ParameterType::number ? Json::Kind::number : spec->type == ParameterType::object ? Json::Kind::object :
                spec->type == ParameterType::array ? Json::Kind::array : Json::Kind::null_value;
            require(value.kind() == expected, "wrong response parameter JSON type");
            finite_numbers(value);
        } else {
            bool parsed_ok = false;
            try {
                Json candidate = parse_json(trim(raw), {limits_.call_bytes, limits_.json_nodes, limits_.json_depth});
                finite_numbers(candidate);
                if (spec == tool->parameters.end()) {
                    value = std::move(candidate);
                    parsed_ok = true;
                } else {
                    const auto expected = spec->type == ParameterType::boolean ? Json::Kind::boolean :
                        spec->type == ParameterType::number ? Json::Kind::number : spec->type == ParameterType::object ? Json::Kind::object :
                        spec->type == ParameterType::array ? Json::Kind::array : Json::Kind::null_value;
                    if (candidate.kind() == expected) {
                        value = std::move(candidate);
                        parsed_ok = true;
                    }
                }
            } catch (const FormatError&) {}
            if (!parsed_ok) value = Json(std::string(raw));
        }
        arguments.emplace_back(std::string(key), std::move(value));
        body.remove_prefix(stop + 13);
    }
    if (!executor_parameters_) {
        for (const auto& parameter : tool->parameters)
            require(!parameter.required || seen.contains(parameter.name), "missing required response parameter");
    }
    require(records_.size() + calls_.size() < limits_.records, "response record limit");
    calls_.push_back({ResponseKind::tool, {}, calls_.size(), std::string(function_name), Json(std::move(arguments))});
}

bool ResponseParser::LiteralState::controls(char c) {
    if (run && c != run) {
        if (!delimiter) {
            if (run == '`' || (run_line && run_width >= 3)) { delimiter = run; width = run_width; fenced = run_line && run_width >= 3; }
        } else if (delimiter == run) {
            if (fenced && run_line && run_width >= width) fence_close = true;
            else if (!fenced && run_width == width) delimiter = 0;
        }
        run = 0;
    }
    return !delimiter && !escaped && !indented && !quoted_line && previous != '"' && previous != '\'';
}
void ResponseParser::LiteralState::advance(char c) {
    if (fence_close && c != ' ' && c != '\t' && c != '\r' && c != '\n') fence_close = false;
    if (!delimiter && line_start && c == '>') quoted_line = true;
    if (!escaped && !indented && !quoted_line && (c == '`' || (c == '~' && (line_start || delimiter == '~' || run == '~')))) {
        if (run == c) ++run_width;
        else { run = c; run_width = 1; run_line = line_start && indent <= 3; }
    }
    escaped = !delimiter && c == '\\' && !escaped;
    if (c == '\n' || c == '\r') {
        if (fence_close) { delimiter = 0; fence_close = false; }
        line_start = true; indent = 0; indented = false; quoted_line = false;
    } else if (c == '\t' && line_start) { line_start = false; indent = 4; if (!delimiter) indented = true; }
    else if (c == ' ' && line_start) { indent = std::min<size_t>(indent + 1, 4); if (indent == 4) { line_start = false; if (!delimiter) indented = true; } }
    else line_start = false;
    previous = c;
}

size_t ResponseParser::token_framed_call_end() {
    while (true) {
        if (call_in_parameter_) {
            const auto end = pending_.find("\n</parameter>", call_search_);
            if (end == std::string::npos) {
                call_search_ = std::max(call_search_, pending_.size() > 12 ? pending_.size() - 12 : 0);
                return std::string::npos;
            }
            call_search_ = end + 13; call_in_parameter_ = false;
        } else {
            const auto parameter = pending_.find("<parameter=", call_search_);
            const auto end = pending_.find("</tool_call>", call_search_);
            if (end != std::string::npos && (parameter == std::string::npos || end < parameter)) return end;
            if (parameter == std::string::npos) {
                call_search_ = std::max(call_search_, pending_.size() > 11 ? pending_.size() - 11 : 0);
                return std::string::npos;
            }
            const auto name_end = pending_.find('>', parameter + 11);
            if (name_end == std::string::npos) {
                require(pending_.size() - parameter <= 11 + 1024, "response name length limit");
                call_search_ = parameter; return std::string::npos;
            }
            call_search_ = name_end + 1; call_in_parameter_ = true;
        }
    }
}

void ResponseParser::process_token_framed(bool final) {
    while (!pending_.empty() || mode_ == Mode::possible_call) {
        if (mode_ == Mode::possible_call) {
            require(pending_.size() <= limits_.call_bytes, "response call byte limit");
            while (call_search_ < pending_.size() && (pending_[call_search_] == ' ' || pending_[call_search_] == '\t' || pending_[call_search_] == '\r' || pending_[call_search_] == '\n')) ++call_search_;
            const auto rest = std::string_view(pending_).substr(call_search_);
            if (rest.starts_with("<function=")) { mode_ = Mode::call; call_search_ = 0; call_in_parameter_ = false; continue; }
            if (!final && std::string_view("<function=").starts_with(rest)) return;
            require(rest.empty() || !std::string_view("<function=").starts_with(rest), "truncated response tool call");
            require(!had_calls_, "text after response tool calls"); emit(ResponseKind::text, "<tool_call>");
            mode_ = Mode::text; call_search_ = 0; continue;
        }
        if (mode_ == Mode::call) {
            const auto end = token_framed_call_end();
            require(pending_.size() <= limits_.call_bytes || (end != std::string::npos && end <= limits_.call_bytes), "response call byte limit");
            if (end == std::string::npos) { require(!final, "truncated response tool call"); return; }
            parse_call(std::string_view(pending_).substr(0, end)); pending_.erase(0, end + 12);
            mode_ = Mode::text; had_calls_ = true; call_search_ = 0; call_in_parameter_ = false; literal_ = {}; continue;
        }
        auto state = literal_; bool visible = visible_;
        size_t position = pending_.size(); std::string_view token; bool held = false;
        for (size_t i = 0; i < pending_.size(); ++i) {
            const char c = pending_[i];
            if (state.controls(c)) {
                const std::array<std::string_view, 2> candidates = mode_ == Mode::thinking ? std::array<std::string_view, 2>{"</think>", ""} :
                    (!thought_ && !visible && !had_calls_ ? std::array<std::string_view, 2>{"<think>", "<tool_call>"} : std::array<std::string_view, 2>{"<tool_call>", ""});
                for (const auto candidate : candidates) {
                    if (candidate.empty() || (candidate == "<tool_call>" && tools_.empty())) continue;
                    const auto rest = std::string_view(pending_).substr(i);
                    if (rest.starts_with(candidate)) { position = i; token = candidate; break; }
                    if (!final && candidate.starts_with(rest)) { position = i; held = true; break; }
                }
                if (position != pending_.size()) break;
            }
            state.advance(c);
            if (mode_ == Mode::text && c != ' ' && c != '\t' && c != '\r' && c != '\n') visible = true;
        }
        const size_t safe = complete_utf8(std::string_view(pending_).substr(0, position));
        const auto text = std::string_view(pending_).substr(0, safe);
        for (const char c : text) { literal_.controls(c); literal_.advance(c); }
        if (had_calls_) require(trim(text).empty(), "text after response tool calls");
        else emit(mode_ == Mode::thinking ? ResponseKind::thinking : ResponseKind::text, text);
        pending_.erase(0, safe);
        if (token.empty() || held || safe != position) break;
        pending_.erase(0, token.size()); literal_ = {};
        if (token == "</think>") mode_ = Mode::text;
        else if (token == "<think>") { mode_ = Mode::thinking; thought_ = true; }
        else { mode_ = Mode::possible_call; call_search_ = 0; }
    }
    if (final) require(mode_ != Mode::call && mode_ != Mode::thinking, "truncated response control frame");
}

void ResponseParser::process(bool final) {
    if (token_framing_) { process_token_framed(final); return; }
    while (!pending_.empty()) {
        if (mode_ == Mode::ended) { require(trim(pending_).empty(), "data after response terminal marker"); pending_.clear(); return; }
        if (mode_ == Mode::call) {
            const auto end = pending_.find("</tool_call>", call_search_);
            require(pending_.size() <= limits_.call_bytes || (end != std::string::npos && end <= limits_.call_bytes), "response call byte limit");
            if (end == std::string::npos) {
                call_search_ = pending_.size() > 11 ? pending_.size() - 11 : 0;
                require(!final, "truncated response tool call"); return;
            }
            parse_call(std::string_view(pending_).substr(0, end));
            pending_.erase(0, end + 12);
            mode_ = Mode::text; had_calls_ = true; call_search_ = 0;
            continue;
        }
        const std::array<std::string_view, 2> thinking_markers{"</think>", "<|im_end|>"};
        const auto active = mode_ == Mode::thinking ? std::span<const std::string_view>(thinking_markers) : std::span<const std::string_view>(markers);
        size_t position = std::string::npos;
        std::string_view token;
        for (const auto candidate : active) {
            const auto found = pending_.find(candidate);
            if (found < position) { position = found; token = candidate; }
        }
        const size_t size = position != std::string::npos ? position : final ? pending_.size() : pending_.size() - held_suffix(pending_, active);
        const size_t safe = complete_utf8(std::string_view(pending_).substr(0, size));
        if (had_calls_) require(trim(std::string_view(pending_).substr(0, safe)).empty(), "text after response tool calls");
        else emit(mode_ == Mode::thinking ? ResponseKind::thinking : ResponseKind::text, std::string_view(pending_).substr(0, safe));
        pending_.erase(0, safe);
        if (position == std::string::npos || safe != size) {
            if (final) require(pending_.empty(), "truncated response marker or UTF-8");
            return;
        }
        pending_.erase(0, token.size());
        if (token == "</think>") { require(mode_ == Mode::thinking, "unexpected response thinking close"); mode_ = Mode::text; }
        else if (token == "<think>") {
            require(mode_ == Mode::text && !thought_ && !visible_ && !had_calls_, "unexpected response thinking open");
            thought_ = true; mode_ = Mode::thinking;
        } else if (token == "<tool_call>") mode_ = Mode::call;
        else if (token == "<|im_end|>") { require(mode_ != Mode::thinking, "terminal marker inside thinking"); mode_ = Mode::ended; }
        else throw FormatError("unexpected response control marker");
    }
    if (final) require(mode_ != Mode::call && mode_ != Mode::thinking, "truncated response control frame");
}

ResponseStatus ResponseParser::feed(std::string_view bytes) {
    if (status_ == ResponseStatus::error) return status_;
    if (status_ == ResponseStatus::complete) { fail("input after finished response"); return status_; }
    try {
        require(bytes.size() <= limits_.bytes - received_, "response byte limit");
        utf8(bytes);
        received_ += bytes.size(); pending_.append(bytes);
        process(false);
    } catch (const FormatError& error) { fail(error.what()); }
    return status_;
}

ResponseStatus ResponseParser::finish() {
    if (status_ != ResponseStatus::incomplete) return status_;
    try {
        require(remaining_ == 0, "truncated response UTF-8");
        // A reserved-marker prefix at end is truncation, not visible content.
        if (!token_framing_) require(held_suffix(pending_, markers) == 0, "truncated response marker");
        process(true);
        for (auto& call : calls_) records_.push_back(std::move(call));
        calls_.clear(); status_ = ResponseStatus::complete;
    } catch (const FormatError& error) { fail(error.what()); }
    return status_;
}

} // namespace tokspur
