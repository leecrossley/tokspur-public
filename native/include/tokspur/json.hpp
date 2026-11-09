#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace tokspur {

struct JsonLimits {
    size_t bytes = 8 * 1024 * 1024;
    size_t nodes = 1'000'000;
    size_t depth = 64;
};

class Json {
public:
    enum class Kind { null_value, boolean, number, string, array, object };
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;
    Json() : data_(nullptr) {}
    explicit Json(bool value) : data_(value) {}
    explicit Json(std::string value) : data_(std::move(value)) {}
    explicit Json(const char* value) : data_(std::string(value)) {}
    explicit Json(Array value) : data_(std::move(value)) {}
    explicit Json(Object value) : data_(std::move(value)) {}
    static Json integer(uint64_t value);
    static Json number(double value);

    [[nodiscard]] bool is_null() const;
    [[nodiscard]] Kind kind() const;
    [[nodiscard]] std::string_view number_lexeme() const;
    [[nodiscard]] bool boolean() const;
    [[nodiscard]] std::string_view string() const;
    [[nodiscard]] uint64_t unsigned_integer() const;
    [[nodiscard]] const Array& array() const;
    [[nodiscard]] const Object& object() const;
    [[nodiscard]] const Json* find(std::string_view key) const;
    [[nodiscard]] std::string dump(size_t byte_limit = 64 * 1024 * 1024) const;

private:
    struct Number { std::string lexeme; };
    explicit Json(Number value) : data_(std::move(value)) {}
    std::variant<std::nullptr_t, bool, Number, std::string, Array, Object> data_;
    friend class JsonParser;
};

Json parse_json(std::string_view input, JsonLimits limits = {});

} // namespace tokspur
