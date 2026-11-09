#include "tokspur/response_protocol.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/tokenizer.hpp"

#include <functional>
#include <iostream>

namespace {
size_t checks = 0;
void check(bool condition, const char* message) { ++checks; if (!condition) throw std::runtime_error(message); }
void refused(const std::function<void()>& action) {
    ++checks; try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid native response protocol accepted");
}
using namespace tokspur;
class Identity final : public TextProcessor {
    std::vector<std::string> pieces(std::string_view text) const override { return {std::string(text)}; }
};
BpeData fixture() {
    BpeData data;
    for (unsigned byte = 0; byte < 256; ++byte) data.vocabulary.emplace_back(1, char(byte));
    for (const auto marker : {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}) {
        const auto id = uint32_t(data.vocabulary.size()); data.vocabulary.emplace_back(marker); data.added.push_back({id, marker, true});
    }
    return data;
}
Json manifest() { return parse_json(R"([{"name":"read_file","parameters":[{"name":"path","type":"string","required":true}]},{"name":"run_tests","parameters":[]}])"); }
Json envelope(const std::vector<uint32_t>& ids, std::string_view finish, bool thinking = false, Json schema = manifest(), Json stop = Json::integer(256)) {
    Json::Array input; for (auto id : ids) input.push_back(Json::integer(id));
    return Json(Json::Object{{"ids", Json(std::move(input))}, {"tools", std::move(schema)}, {"initial_thinking", Json(thinking)},
        {"finish_reason", Json(std::string(finish))}, {"stop_id", std::move(stop)}});
}
void run() {
    ByteBpe tokenizer(fixture()); Identity identity;
    const ResponseAdmission admission{uint32_t(tokenizer.vocabulary_size()), 256, 257, 258};
    const auto encode = [&](std::string_view text) { return tokenizer.encode(text, identity); };
    const auto decode = [&](std::span<const uint32_t> ids) { return tokenizer.decode_bytes(ids); };
    const auto parse = [&](const Json& input) { return parse_native_response(input, admission, decode); };
    const std::string call = "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/日本🙂.rs\n</parameter>\n</function>\n</tool_call>";
    const std::string batch = "Inspecting café\n" + call + "\n<tool_call><function=run_tests>\n</function></tool_call>";
    const auto all = encode(batch + "<|im_end|>");
    const auto complete = parse(envelope(all, "stop"));
    check(complete.find("status")->string() == "complete" && complete.find("complete")->boolean(), "valid stopped response not complete");
    check(complete.find("tool_calls")->array().size() == 2 && !complete.find("tools_executed")->boolean(), "tool count or execution authority");
    const auto& first = complete.find("tool_calls")->array()[0];
    check(first.find("ordinal")->unsigned_integer() == 0 && first.find("type")->string() == "function", "call identity fields");
    check(first.find("function")->find("arguments")->find("path")->string() == "src/日本🙂.rs", "raw Unicode argument changed");
    check(complete.find("content")->string() == "Inspecting café\n" && complete.find("reasoning_content")->string().empty(), "visible/hidden split");
    // Every token boundary uses the same accumulated IDs, never separately
    // replaced token strings. Some boundaries split UTF-8 or XML-like markers.
    for (size_t split = 0; split < all.size(); ++split) {
        const std::vector<uint32_t> prefix(all.begin(), all.begin() + split);
        const auto pending = parse(envelope(prefix, "length", false, manifest(), Json()));
        check(pending.find("status")->string() == "incomplete" && !pending.find("complete")->boolean(), "length prefix incorrectly completed");
        check(pending.find("tool_calls")->array().empty(), "length prefix released executable call");
        check(parse(envelope(all, "stop")).dump() == complete.dump(), "continuation accumulation changes final parse");
    }
    // Even a syntactically complete call is pending without a verified terminal.
    check(parse(envelope(encode(call), "length", false, manifest(), Json())).find("tool_calls")->array().empty(), "complete-looking capped call was executed");
    const std::string hidden = "considered " + call + " but this is inert.";
    const auto thought = parse(envelope(encode(hidden + "</think>\n\n" + call + "<|im_end|>"), "stop", true));
    check(thought.find("reasoning_content")->string() == hidden && thought.find("tool_calls")->array().size() == 1, "thinking tool text escaped into callable record");
    const auto plain = parse(envelope(encode("finished café 🦉<|im_end|>"), "stop", false, Json(Json::Array{})));
    check(plain.find("content")->string() == "finished café 🦉" && plain.find("tool_calls")->array().empty(), "plain final response");

    const auto typed = parse_json(R"([{"name":"typed","parameters":[{"name":"s","type":"string","required":true},{"name":"b","type":"boolean","required":true},{"name":"n","type":"number","required":true},{"name":"o","type":"object","required":true},{"name":"a","type":"array","required":true},{"name":"z","type":"null","required":true}]}])");
    const auto typed_body = encode("<tool_call><function=typed>\n<parameter=s>\ntrue\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n<parameter=n>\n-2.5\n</parameter>\n<parameter=o>\n{\"x\":1}\n</parameter>\n<parameter=a>\n[1,true]\n</parameter>\n<parameter=z>\nnull\n</parameter>\n</function></tool_call><|im_end|>");
    const auto typed_result = parse(envelope(typed_body, "stop", false, typed));
    const auto& arguments = typed_result.find("tool_calls")->array()[0].find("function")->find("arguments");
    check(arguments->find("s")->string() == "true" && arguments->find("b")->boolean(), "typed manifest string/boolean mapping");
    check(arguments->find("n")->number_lexeme() == "-2.5" && arguments->find("o")->find("x")->unsigned_integer() == 1 &&
        arguments->find("a")->array().size() == 2 && arguments->find("z")->is_null(), "typed manifest structured mapping");

    for (const auto& invalid : {
        envelope({}, "stop"), envelope(all, "stop", false, manifest(), Json()),
        envelope(all, "stop", false, manifest(), Json::integer(257)),
        envelope(all, "length", false, manifest(), Json()),
        envelope(all, "stop", false, manifest(), Json::integer(999999)),
        envelope(encode(call), "stop"), envelope(all, "unknown"),
        envelope(encode("x<|endoftext|>"), "stop"), envelope(encode("<|im_start|>x<|im_end|>"), "stop"),
        envelope(encode("x<|im_end|>y<|im_end|>"), "stop"),
        envelope(encode(call + "unexpected<|im_end|>"), "stop"),
        envelope(encode("<think>unfinished<|im_end|>"), "stop"),
        envelope(encode("<tool_call><function=read_file></function></tool_call><|im_end|>"), "stop"),
        envelope(encode("<tool_call><function=unknown></function></tool_call><|im_end|>"), "stop")}) {
        refused([&] { static_cast<void>(parse(invalid)); });
    }
    auto earlier = all; earlier.insert(earlier.begin(), admission.im_end);
    refused([&] { static_cast<void>(parse(envelope(earlier, "stop"))); });
    refused([&] { static_cast<void>(parse(envelope({admission.vocabulary}, "length", false, manifest(), Json()))); });
    refused([&] { static_cast<void>(parse(envelope(std::vector<uint32_t>(response_maximum_ids + 1, 65), "length", false, manifest(), Json()))); });
    const auto accumulated = envelope(std::vector<uint32_t>(response_maximum_accumulated_ids, 65), "length", false, manifest(), Json());
    check(parse_native_response(accumulated, admission, decode, response_maximum_accumulated_ids).find("content")->string().size() == response_maximum_accumulated_ids,
        "explicit accumulated HTTP response bound was not admitted");
    refused([&] { static_cast<void>(parse_native_response(accumulated, admission, decode, 0)); });
    refused([&] { static_cast<void>(parse_native_response(accumulated, admission, decode, response_maximum_accumulated_ids + 1)); });
    for (const auto schema : {R"([{"name":"x","parameters":[],"extra":1}])", R"([{"name":"x","parameters":[{"name":"p","type":"string"}]}])",
        R"([{"name":"x","parameters":[{"name":"p","type":"string","required":true,"extra":1}]}])",
        R"([{"name":"x","parameters":[{"name":"p","type":"integer","required":true}]}])",
        R"([{"name":"x","parameters":[{"name":"p","type":"string","required":"true"}]}])",
        R"([{"name":"x","parameters":[]},{"name":"x","parameters":[]}])", R"([{"name":"bad name","parameters":[]}])"}) {
        refused([&] { static_cast<void>(parse(envelope(all, "stop", false, parse_json(schema)))); });
    }
    auto extra = envelope(all, "stop").object(); extra.emplace_back("unknown", Json(true));
    refused([&] { static_cast<void>(parse(Json(extra))); });
    refused([&] { static_cast<void>(parse(parse_json(R"({"ids":[]})"))); });
    for (const auto invalid : {ResponseAdmission{0, 0, 1, 2}, ResponseAdmission{259, 256, 256, 258}, ResponseAdmission{259, 256, 257, 259}})
        refused([&] { static_cast<void>(parse_native_response(envelope(all, "stop"), invalid, decode)); });
    size_t decoded = 0;
    const auto errored = parse_native_response(envelope(encode(call), "error", false, manifest(), Json()), admission,
        [&](auto ids) { ++decoded; return decode(ids); });
    check(errored.find("status")->string() == "error" && errored.find("tool_calls")->array().empty() && decoded == 0, "upstream error decoded or released calls");
    refused([&] { static_cast<void>(parse_native_response(envelope(all, "stop"), admission, [](auto) { return std::string("wrong terminal"); })); });
    refused([&] { static_cast<void>(parse_native_response(envelope({}, "length", false, manifest(), Json()), admission,
        [](auto) { return std::string(response_request_bytes + 1, 'x'); })); });
    auto invalid_utf8 = encode(std::string("x\xc0\x80", 3) + "<|im_end|>");
    refused([&] { static_cast<void>(parse(envelope(invalid_utf8, "stop"))); });
    invalid_utf8 = encode(std::string("x\xc3", 2) + "<|im_end|>");
    refused([&] { static_cast<void>(parse(envelope(invalid_utf8, "stop"))); });
    const auto partial = encode(std::string("x\xc3", 2));
    check(tokenizer.decode_bytes(partial) == std::string("x\xc3", 2) && tokenizer.decode_text(partial) != tokenizer.decode_bytes(partial), "fixture does not expose replacement corruption");
    check(parse(envelope(partial, "length", false, manifest(), Json())).find("status")->string() == "incomplete", "valid partial UTF-8 was not retained as incomplete");
    const auto failure = native_response_error("test refusal");
    check(failure.find("tool_calls")->array().empty() && !failure.find("complete")->boolean() && !failure.find("tools_executed")->boolean(), "error envelope contains execution authority");
}
}
int main() {
    try { run(); std::cout << "Owned response protocol checks passed: " << checks << '\n'; return 0; }
    catch (const std::exception& error) { std::cerr << "Owned response protocol check failed: " << error.what() << '\n'; return 1; }
}
