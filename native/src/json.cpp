#include "tokspur/json.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/unicode.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <functional>
#include <set>

namespace tokspur {

class JsonParser {
public:
    JsonParser(std::string_view input, JsonLimits limits) : input_(input), limits_(limits) {
        if (input.size() > limits.bytes) throw FormatError("JSON input byte limit exceeded");
    }
    Json parse() {
        auto result = value(0);
        whitespace();
        if (position_ != input_.size()) throw FormatError("trailing JSON data");
        return result;
    }
private:
    void whitespace() {
        while (position_ < input_.size() && (input_[position_] == ' ' || input_[position_] == '\t' || input_[position_] == '\r' || input_[position_] == '\n')) ++position_;
    }
    char take() {
        if (position_ == input_.size()) throw FormatError("truncated JSON");
        return input_[position_++];
    }
    bool consume(char c) {
        if (position_ < input_.size() && input_[position_] == c) { ++position_; return true; }
        return false;
    }
    void literal(std::string_view tail) {
        if (input_.substr(position_, tail.size()) != tail) throw FormatError("invalid JSON literal");
        position_ += tail.size();
    }
    uint32_t hex4() {
        uint32_t value = 0;
        for (size_t i = 0; i < 4; ++i) {
            const auto c = take();
            uint32_t digit;
            if (c >= '0' && c <= '9') digit = c - '0';
            else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
            else throw FormatError("invalid JSON Unicode escape");
            value = (value << 4) | digit;
        }
        return value;
    }
    std::string text() {
        std::string output;
        while (true) {
            const auto c = static_cast<unsigned char>(take());
            if (c == '"') {
                static_cast<void>(unicode_scalars(output));
                return output;
            }
            if (c < 0x20) throw FormatError("unescaped JSON control character");
            if (c != '\\') { output.push_back(static_cast<char>(c)); continue; }
            switch (take()) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    uint32_t point = hex4();
                    if (point >= 0xd800 && point <= 0xdbff) {
                        if (take() != '\\' || take() != 'u') throw FormatError("missing JSON low surrogate");
                        const auto low = hex4();
                        if (low < 0xdc00 || low > 0xdfff) throw FormatError("invalid JSON low surrogate");
                        point = 0x10000 + ((point - 0xd800) << 10) + low - 0xdc00;
                    } else if (point >= 0xdc00 && point <= 0xdfff) throw FormatError("unpaired JSON low surrogate");
                    output += scalar_utf8(point);
                    break;
                }
                default: throw FormatError("invalid JSON escape");
            }
        }
    }
    Json number() {
        const auto start = position_;
        consume('-');
        if (!consume('0')) {
            if (position_ == input_.size() || input_[position_] < '1' || input_[position_] > '9') throw FormatError("invalid JSON number");
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
        }
        const auto digits = [&] {
            const auto before = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (before == position_) throw FormatError("missing JSON number digits");
        };
        if (consume('.')) digits();
        if (consume('e') || consume('E')) { if (!consume('+')) consume('-'); digits(); }
        return Json(Json::Number{std::string(input_.substr(start, position_ - start))});
    }
    Json value(size_t depth) {
        if (++nodes_ > limits_.nodes || depth > limits_.depth) throw FormatError("JSON structural limit exceeded");
        whitespace();
        const auto c = take();
        switch (c) {
            case 'n': literal("ull"); return Json();
            case 't': literal("rue"); return Json(true);
            case 'f': literal("alse"); return Json(false);
            case '"': return Json(text());
            case '[': {
                Json::Array result;
                whitespace();
                if (consume(']')) return Json(std::move(result));
                while (true) {
                    result.push_back(value(depth + 1));
                    whitespace();
                    if (consume(']')) break;
                    if (!consume(',')) throw FormatError("expected JSON array delimiter");
                }
                return Json(std::move(result));
            }
            case '{': {
                Json::Object result;
                std::set<std::string, std::less<>> names;
                whitespace();
                if (consume('}')) return Json(std::move(result));
                while (true) {
                    whitespace();
                    if (!consume('"')) throw FormatError("expected JSON object key");
                    auto key = text();
                    if (!names.insert(key).second) throw FormatError("duplicate JSON object key");
                    whitespace();
                    if (!consume(':')) throw FormatError("expected JSON object colon");
                    result.emplace_back(std::move(key), value(depth + 1));
                    whitespace();
                    if (consume('}')) break;
                    if (!consume(',')) throw FormatError("expected JSON object delimiter");
                }
                return Json(std::move(result));
            }
            default:
                if (c == '-' || (c >= '0' && c <= '9')) { --position_; return number(); }
                throw FormatError("invalid JSON value");
        }
    }
    std::string_view input_;
    JsonLimits limits_;
    size_t position_ = 0;
    size_t nodes_ = 0;
};

Json parse_json(std::string_view input, JsonLimits limits) { return JsonParser(input, limits).parse(); }
Json Json::integer(uint64_t value) { return Json(Number{std::to_string(value)}); }
Json Json::number(double value) {
    if (!std::isfinite(value)) throw FormatError("nonfinite JSON number");
    char buffer[64];
    const auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (ec != std::errc{}) throw FormatError("JSON number encoding failed");
    return Json(Number{std::string(buffer, size_t(ptr - buffer))});
}
bool Json::is_null() const { return std::holds_alternative<std::nullptr_t>(data_); }
Json::Kind Json::kind() const { return static_cast<Kind>(data_.index()); }
std::string_view Json::number_lexeme() const {
    const auto* value = std::get_if<Number>(&data_);
    if (!value) throw FormatError("expected JSON number");
    return value->lexeme;
}
bool Json::boolean() const {
    const auto* value = std::get_if<bool>(&data_);
    if (!value) throw FormatError("expected JSON boolean");
    return *value;
}
std::string_view Json::string() const {
    const auto* value = std::get_if<std::string>(&data_);
    if (!value) throw FormatError("expected JSON string");
    return *value;
}
uint64_t Json::unsigned_integer() const {
    const auto* number = std::get_if<Number>(&data_);
    if (!number) throw FormatError("expected JSON unsigned integer");
    uint64_t value = 0;
    const auto& text = number->lexeme;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) throw FormatError("JSON unsigned integer out of range or nonintegral");
    return value;
}
const Json::Array& Json::array() const {
    const auto* value = std::get_if<Array>(&data_);
    if (!value) throw FormatError("expected JSON array");
    return *value;
}
const Json::Object& Json::object() const {
    const auto* value = std::get_if<Object>(&data_);
    if (!value) throw FormatError("expected JSON object");
    return *value;
}
const Json* Json::find(std::string_view key) const {
    for (const auto& [name, value] : object()) if (name == key) return &value;
    return nullptr;
}

std::string Json::dump(size_t byte_limit) const {
    std::string output;
    const auto append = [&](std::string_view text) {
        if (text.size() > byte_limit - output.size()) throw FormatError("JSON output byte limit exceeded");
        output.append(text);
    };
    const auto quoted = [&](std::string_view text) {
        static_cast<void>(unicode_scalars(text));
        append("\"");
        for (const unsigned char c : text) {
            if (c == '"' || c == '\\') { const std::array<char, 2> escaped{'\\', static_cast<char>(c)}; append({escaped.data(), escaped.size()}); }
            else if (c < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                const std::array<char, 6> escaped{'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
                append({escaped.data(), escaped.size()});
            } else { const char byte = static_cast<char>(c); append({&byte, 1}); }
        }
        append("\"");
    };
    std::function<void(const Json&, size_t)> visit = [&](const Json& value, size_t depth) {
        if (depth > 64) throw FormatError("JSON output nesting limit exceeded");
        if (std::holds_alternative<std::nullptr_t>(value.data_)) append("null");
        else if (const auto* b = std::get_if<bool>(&value.data_)) append(*b ? "true" : "false");
        else if (const auto* n = std::get_if<Number>(&value.data_)) append(n->lexeme);
        else if (const auto* s = std::get_if<std::string>(&value.data_)) quoted(*s);
        else if (const auto* a = std::get_if<Array>(&value.data_)) {
            append("[");
            for (size_t i = 0; i < a->size(); ++i) { if (i) append(","); visit((*a)[i], depth + 1); }
            append("]");
        } else if (const auto* o = std::get_if<Object>(&value.data_)) {
            append("{");
            std::set<std::string, std::less<>> names;
            for (size_t i = 0; i < o->size(); ++i) {
                if (!names.insert((*o)[i].first).second) throw FormatError("duplicate JSON output key");
                if (i) append(",");
                quoted((*o)[i].first); append(":"); visit((*o)[i].second, depth + 1);
            }
            append("}");
        }
    };
    visit(*this, 0);
    return output;
}

} // namespace tokspur
