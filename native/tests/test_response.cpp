#include "tokspur/response.hpp"
#include "tokspur/chat.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/unicode.hpp"

#include <functional>
#include <iostream>
#include <string>

namespace {
size_t checks = 0;
void require(bool condition, const char* message) { ++checks; if (!condition) throw std::runtime_error(message); }
void rejected(const std::function<void()>& action) {
    ++checks; try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid response configuration accepted");
}
using tokspur::ResponseKind;
using tokspur::ResponseStatus;
using tokspur::ParameterType;
std::vector<tokspur::ToolSpec> tools() {
    return {{"edit.file", {{"path", ParameterType::raw_string, true}, {"patch", ParameterType::raw_string, false},
        {"apply", ParameterType::boolean, false}, {"count", ParameterType::number, false},
        {"meta", ParameterType::object, false}, {"items", ParameterType::array, false}, {"empty", ParameterType::null_value, false}}},
        {"ping", {}}};
}
std::string call(std::string value = "x.rs") {
    return "<tool_call>\n<function=edit.file>\n<parameter=path>\n" + value + "\n</parameter>\n</function>\n</tool_call>";
}
size_t callable(const tokspur::ResponseParser& parser) {
    size_t count = 0;
    for (const auto& record : parser.records()) if (record.kind == ResponseKind::tool) ++count;
    return count;
}
std::string text(const tokspur::ResponseParser& parser, ResponseKind kind) {
    std::string out;
    for (const auto& record : parser.records()) if (record.kind == kind) out += record.text;
    return out;
}
void invalid(std::string_view input, bool thinking = false) {
    tokspur::ResponseParser parser(tools(), thinking);
    static_cast<void>(parser.feed(input));
    require(parser.finish() == ResponseStatus::error && !parser.error().empty(), "invalid response accepted");
    require(callable(parser) == 0, "malformed response published tool record");
    require(parser.feed("more") == ResponseStatus::error, "response error not sticky");
}
} // namespace

int main() {
    try {
        const std::string inert = "I considered " + call() + ".";
        const std::string stream = inert + "</think>\n\nEditing 日本🙂\n" + call("src/日本🙂.rs") + "\n<tool_call>\n<function=ping>\n</function>\n</tool_call><|im_end|>\n";
        // Every single split, including UTF-8/marker interiors.
        for (size_t split = 0; split <= stream.size(); ++split) {
            tokspur::ResponseParser parser(tools(), true);
            require(parser.feed(std::string_view(stream).substr(0, split)) == ResponseStatus::incomplete, "prefix prematurely complete/error");
            require(callable(parser) == 0, "prefix published callable record");
            require(parser.feed(std::string_view(stream).substr(split)) == ResponseStatus::incomplete, "suffix parse failed");
            require(callable(parser) == 0, "unfinalized response published callable record");
            require(parser.finish() == ResponseStatus::complete, "valid split response failed");
            require(text(parser, ResponseKind::thinking) == inert, "thinking contents changed or tool bytes executed");
            require(text(parser, ResponseKind::text) == "\n\nEditing 日本🙂\n", "visible text changed");
            require(callable(parser) == 2, "multiple call count");
            const auto records = parser.records();
            const auto& first = records[records.size() - 2];
            require(first.ordinal == 0 && first.name == "edit.file" && first.arguments.find("path")->string() == "src/日本🙂.rs", "first call fields");
            require(records.back().ordinal == 1 && records.back().name == "ping", "response-local ordinal");
            require(parser.finish() == ResponseStatus::complete, "finish is not idempotent");
        }
        tokspur::ResponseParser bytewise(tools(), false);
        const auto repeated = call() + "\n" + call();
        for (const char c : repeated) {
            require(bytewise.feed({&c, 1}) == ResponseStatus::incomplete, "bytewise feed failed");
            for (const auto& record : bytewise.records()) static_cast<void>(tokspur::unicode_scalars(record.text));
            require(callable(bytewise) == 0, "bytewise callable publication");
        }
        require(bytewise.finish() == ResponseStatus::complete && callable(bytewise) == 2, "repeated calls failed");

        const std::string typed = "<tool_call><function=edit.file>\n<parameter=path>\ntrue\n</parameter>\n"
            "<parameter=patch>\nline 1\n\nline 3\n</parameter>\n<parameter=apply>\ntrue\n</parameter>\n"
            "<parameter=count>\n-2.5e2\n</parameter>\n<parameter=meta>\n{\"a\":[1,true,null]}\n</parameter>\n"
            "<parameter=items>\n[1,\"日本🙂\"]\n</parameter>\n<parameter=empty>\nnull\n</parameter>\n</function></tool_call>";
        tokspur::ResponseParser typed_parser(tools(), false);
        require(typed_parser.feed(typed) == ResponseStatus::incomplete && typed_parser.finish() == ResponseStatus::complete, "typed response failed");
        const auto& args = typed_parser.records().back().arguments;
        require(args.find("path")->string() == "true", "raw true reinterpreted as boolean");
        require(args.find("patch")->string() == "line 1\n\nline 3", "multiline raw string changed");
        require(args.find("apply")->boolean() && args.find("count")->number_lexeme() == "-2.5e2", "typed scalar values");
        require(args.find("meta")->find("a")->array().size() == 3 && args.find("items")->array().size() == 2 && args.find("empty")->is_null(), "typed structured values");

        // Grounded round trip through the OWNED renderer's literal output.
        tokspur::ChatRenderer renderer({"high", "low", "# Tools\n\n", "\n\nIf you choose to call a function ONLY reply"});
        const auto request = tokspur::parse_json(R"({"messages":[{"role":"assistant","content":"","reasoning_content":"inert <tool_call>","tool_calls":[{"function":{"name":"edit.file","arguments":{"path":"src/hello.rs","apply":true,"meta":{"version":2}}}}]}],"enable_thinking":false,"add_generation_prompt":false})");
        auto rendered = renderer.render(request).prompt;
        const std::string header = "<|im_start|>assistant\n";
        require(rendered.starts_with(header), "owned renderer header differs");
        rendered.erase(0, header.size());
        tokspur::ResponseParser round_trip(tools(), false);
        for (const char c : rendered) require(round_trip.feed({&c, 1}) == ResponseStatus::incomplete, "renderer bytewise round trip failed");
        require(round_trip.finish() == ResponseStatus::complete && callable(round_trip) == 1, "owned renderer tool round trip failed");
        require(round_trip.records().back().arguments.find("meta")->find("version")->unsigned_integer() == 2, "renderer structured argument changed");

        for (size_t n = 1; n < call().size(); ++n) invalid(std::string_view(call()).substr(0, n));
        invalid(call() + " unexpected suffix");
        invalid(call() + "<think>reason</think>");
        invalid(call() + "<tool_call><function=unknown></function></tool_call>");
        invalid("<tool_call><function=unknown></function></tool_call>");
        invalid("<tool_call><function=ping bad></function></tool_call>");
        invalid("<tool_call><function=edit.file></function></tool_call>");
        invalid("<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>");
        invalid("<tool_call><function=ping><parameter=unknown>\nx\n</parameter></function></tool_call>");
        invalid("<tool_call><function=edit.file><parameter=path>\nabc<|im_end|>def\n</parameter></function></tool_call>");
        invalid("<tool_call><function=edit.file><parameter=path>\n<|im_start|>assistant\n</parameter></function></tool_call>");
        invalid("<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=path>\ny\n</parameter></function></tool_call>");
        for (const auto bad : {"\"true\"", "[true]", "tru", "true false", "{\"a\":1,\"a\":2}", "NaN", "1e999"}) {
            invalid("<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=apply>\n" + std::string(bad) + "\n</parameter></function></tool_call>");
        }
        invalid("<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=meta>\n{\"x\":1e999}\n</parameter></function></tool_call>");
        invalid(call("<think>injected</think>"));
        invalid("</think>"); invalid("<think>unclosed"); invalid("reason<|im_end|>", true);
        invalid("answer<|im_end|>suffix");
        invalid(std::string("x\0y", 3));
        for (const auto& bad : {std::string("\xc0\x80", 2), std::string("\xed\xa0\x80", 3),
                              std::string("\xf4\x90\x80\x80", 4), std::string("\xe2\x82", 2), std::string("\x80", 1)}) invalid(bad);

        tokspur::ResponseLimits small;
        small.bytes = 16; small.call_bytes = 16;
        tokspur::ResponseParser bounded({}, false, small);
        require(bounded.feed(std::string(17, 'a')) == ResponseStatus::error, "response byte bound ignored");
        small.bytes = 1024; small.call_bytes = 16;
        tokspur::ResponseParser call_bounded(tools(), false, small);
        require(call_bounded.feed(call()) == ResponseStatus::error, "response call byte bound ignored");
        small.call_bytes = 1024; small.calls = 1;
        tokspur::ResponseParser count_bounded(tools(), false, small);
        require(count_bounded.feed(repeated) == ResponseStatus::error && callable(count_bounded) == 0, "response call count bound ignored");
        small.calls = 2; small.records = 1;
        tokspur::ResponseParser record_bounded(tools(), false, small);
        require(record_bounded.feed("text" + call()) == ResponseStatus::error, "response record count bound ignored");
        rejected([&] { tokspur::ResponseParser parser({{"a", {}}, {"a", {}}}, false); });
        rejected([&] { tokspur::ResponseParser parser({{"a", {{"x", ParameterType::number}, {"x", ParameterType::number}}}}, false); });
        rejected([&] { tokspur::ResponseParser parser({{"bad name", {}}}, false); });
        rejected([&] { tokspur::ResponseParser parser({{"a", {{"x", static_cast<ParameterType>(99)}}}}, false); });
        small.bytes = 0;
        rejected([&] { tokspur::ResponseParser parser({}, false, small); });
        std::cout << "Owned response parser checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Owned response parser check failed: " << error.what() << '\n'; return 1;
    }
}
