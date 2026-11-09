#include "tokspur/json.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/unicode.hpp"

#include <array>
#include <functional>
#include <iostream>
#include <random>

namespace {
size_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid JSON request accepted");
}

void basics() {
    const auto object = tokspur::parse_json(R"({"text":"\ufeffusing café\u0000","ids":[0,1,4294967295],"flag":true,"nothing":null})");
    require(object.find("text")->string() == std::string("\ufeffusing café") + std::string(1, '\0'), "JSON must preserve leading BOM and embedded NUL");
    require(object.find("flag")->boolean(), "JSON boolean");
    require(object.find("nothing")->is_null(), "JSON null");
    require(object.find("missing") == nullptr, "JSON absent field");
    require(object.find("ids")->array().at(2).unsigned_integer() == 4294967295ULL, "JSON uint32 maximum");
    require(tokspur::parse_json("18446744073709551615").unsigned_integer() == uint64_t(-1), "JSON uint64 maximum");
    require(tokspur::parse_json(object.dump()).dump() == object.dump(), "JSON stable dump/parse");
    require(tokspur::parse_json(R"("\ud83e\udd89")").string() == "🦉", "JSON surrogate pair");
    require(tokspur::parse_json(R"("\"\\\/\b\f\n\r\t")").string() == "\"\\/\b\f\n\r\t", "JSON escapes");
    require(tokspur::parse_json(" \n [true, false, null, -1.25e+20] \t").dump() == "[true,false,null,-1.25e+20]", "JSON number lexeme and whitespace");
    for (const auto* bad : {"", " ", "{", "[", "[1,]", "{\"x\":1,}", "{x:1}", "01", "-", "1.", "1e", "+1", "NaN", "Infinity", "true false",
                            "\"\\x00\"", "\"\\uD800\"", "\"\\uDC00\"", "\"\\uD800\\u0000\"", "{\"x\":1,\"x\":2}", "{\"x\":1,\"\\u0078\":2}"}) {
        rejected([&] { static_cast<void>(tokspur::parse_json(bad)); });
    }
    for (const auto& bad : {std::string("\"\xc0\x80\"", 4), std::string("\"\xed\xa0\x80\"", 5), std::string("\"\n\"", 3)}) {
        rejected([&] { static_cast<void>(tokspur::parse_json(bad)); });
    }
    for (const auto* number : {"-0", "1.0", "1e2", "18446744073709551616", "-1", "null", "true", "\"1\""}) {
        rejected([&] { static_cast<void>(tokspur::parse_json(number).unsigned_integer()); });
    }
    rejected([&] { static_cast<void>(object.find("ids")->string()); });
    rejected([&] { static_cast<void>(object.find("text")->array()); });
    rejected([&] { static_cast<void>(tokspur::parse_json("[]").find("x")); });
    rejected([&] { static_cast<void>(tokspur::parse_json("[1,2]", {1024, 2, 4})); });
    rejected([&] { static_cast<void>(tokspur::parse_json("[[1]]", {1024, 20, 1})); });
    rejected([&] { static_cast<void>(tokspur::parse_json("null", {3, 20, 1})); });
    rejected([&] { static_cast<void>(object.dump(3)); });
    rejected([&] { static_cast<void>(tokspur::Json(tokspur::Json::Object{{"x", tokspur::Json()}, {"x", tokspur::Json()}}).dump()); });
    rejected([&] { static_cast<void>(tokspur::Json(std::string("\x80", 1)).dump()); });
}

void generated() {
    std::mt19937 random(0x6a736f6e);
    for (size_t run = 0; run < 5000; ++run) {
        std::string text;
        for (size_t i = 0, count = random() % 100; i < count; ++i) {
            auto point = random() % 0x110000;
            if (point >= 0xd800 && point <= 0xdfff) point = 0;
            text += tokspur::scalar_utf8(point);
        }
        const auto value = tokspur::Json(tokspur::Json::Object{{"text", tokspur::Json(text)}, {"id", tokspur::Json::integer(random())}});
        const auto decoded = tokspur::parse_json(value.dump());
        require(decoded.find("text")->string() == text, "generated JSON Unicode round trip");
        require(decoded.dump() == value.dump(), "generated JSON structural round trip");
    }
}
} // namespace

int main() {
    try {
        basics();
        generated();
        std::cout << "Owned JSON checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Owned JSON check failed: " << error.what() << '\n';
        return 1;
    }
}
