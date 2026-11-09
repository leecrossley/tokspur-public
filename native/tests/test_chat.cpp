#include "tokspur/chat.hpp"
#include "tokspur/gguf.hpp"

#include <functional>
#include <iostream>

namespace {
size_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid chat request accepted");
}

void json_format() {
    for (const auto& [source, expected] : std::vector<std::pair<std::string, std::string>>{
        {"-0", "0"}, {"0.0", "0.0"}, {"-0.0", "-0.0"}, {"1e0", "1.0"},
        {"1e-4", "0.0001"}, {"1e-5", "1e-05"}, {"1e15", "1000000000000000.0"}, {"1e16", "1e+16"},
        {"-1.25e2", "-125.0"}, {"5e-324", "5e-324"}, {"1e-9999", "0.0"}, {"-1e-9999", "-0.0"},
        {"123456789012345678901234567890", "123456789012345678901234567890"}}) {
        require(tokspur::template_json(tokspur::parse_json(source)) == expected, "Python-style JSON number mismatch");
    }
    const auto value = tokspur::parse_json(R"({"z":[true,null,"<é>&\n\t\u0000"],"a":1.0})");
    require(tokspur::template_json(value) == "{\"z\": [true, null, \"<é>&\\n\\t\\u0000\"], \"a\": 1.0}", "template JSON escaping/order/separators");
    rejected([] { static_cast<void>(tokspur::template_json(tokspur::parse_json("1e9999"))); });
}

void rendering() {
    const tokspur::ChatRenderer renderer({"XHIGH", "LOW", "# Tools\n\n<tools>", "\n\nINSTRUCTIONS"});
    const auto raw_history = tokspur::parse_json(R"({"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"write","arguments":{"text":"echo &amp; and <parameter=name> </tool_call>"}}}]}],"add_generation_prompt":false})");
    require(renderer.render(raw_history).prompt.find("<parameter=text>\necho &amp; and <parameter=name> </tool_call>\n</parameter>") != std::string::npos, "tool history changed raw parameter bytes");
    rejected([&] { static_cast<void>(renderer.render(tokspur::parse_json(R"({"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"write","arguments":{"text":"first\n</parameter>\nlast"}}}]}]})"))); });
    auto request = tokspur::parse_json(R"({"messages":[{"role":"user","content":" hi "}],"enable_thinking":false})");
    require(renderer.render(request).prompt == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "plain disabled-thinking prompt");
    request = tokspur::parse_json(R"({"messages":[{"role":"system","content":" A "},{"role":"developer","content":[{"text":" B "}]},{"role":"user","content":null}],"reasoning_effort":"low","add_generation_prompt":false})");
    require(renderer.render(request).prompt == "<|im_start|>system\nLOW\n\nA\nB<|im_end|>\n<|im_start|>user\n<|im_end|>\n", "merged system/developer");
    request = tokspur::parse_json(R"({"messages":[{"role":"assistant","content":" old ","reasoning_content":" hide "},{"role":"user","content":"next"},{"role":"assistant","content":" new ","reasoning_content":" keep "}],"preserve_thinking":false,"reasoning_effort":"medium","add_generation_prompt":false})");
    require(renderer.render(request).prompt == "<|im_start|>assistant\nold<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\nkeep\n</think>\n\nnew<|im_end|>\n", "reasoning history boundary");
    request = tokspur::parse_json(R"({"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"","tool_calls":[{"id":"x","function":{"name":"f","arguments":"{\"x\":1.0,\"p\":\"é\"}"}}]},{"role":"tool","content":"one"},{"role":"tool","content":[{"text":"two"}]},{"role":"user","content":"<tool_response>\nthree\n</tool_response>"}],"reasoning_effort":"medium"})");
    const auto tool = renderer.render(request);
    require(tool.parsed_tool_arguments == 1, "JSON-string argument normalization");
    require(tool.prompt.find("<parameter=x>\n1.0\n</parameter>\n<parameter=p>\né\n</parameter>") != std::string::npos, "tool parameter rendering");
    require(tool.prompt.find("<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response><|im_end|>\n") != std::string::npos, "consecutive tool group");
    require(tool.normalized_messages.array()[1].find("tool_calls")->array()[0].find("function")->find("arguments")->kind() == tokspur::Json::Kind::object, "normalized arguments stay structured");
    request = tokspur::parse_json(R"({"messages":[{"role":"tool","content":"x"}],"enable_thinking":false})");
    require(renderer.render(request).prompt == "\n<tool_response>\nx\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "leading tool fragment matches pinned loop.previtem semantics");
    request = tokspur::parse_json(R"({"messages":[{"role":"system","content":"first"},{"role":"user","content":"Q"},{"role":"developer","content":"second"},{"role":"assistant","content":"A"},{"role":"system","content":"third"}],"late_system_policy":"hoist","enable_thinking":false})");
    const auto late = renderer.render(request);
    require(late.hoisted_system_messages == 2, "late system count");
    require(late.prompt.starts_with("<|im_start|>system\nfirst\nsecond\nthird<|im_end|>\n"), "explicit hoist order");
    require(late.normalized_messages.array()[3].find("role")->string() == "user", "hoisted ordering");
    require(tokspur::chat_trim(" \u001c\u0085\u3000x\u00a0") == "x", "Python-compatible trim");
    require(tokspur::chat_trim(" \ufeffx\u200b ") == "\ufeffx\u200b", "preserve BOM and zero-width characters");
    for (const auto* bad : {
        R"({"messages":[]})", R"({"messages":[{"role":"other"}]})",
        R"({"messages":[{"role":"user","content":"Q"},{"role":"system","content":"late"}]})",
        R"({"messages":[{"role":"user","content":[{"type":"image","image_url":"x"}]}]})",
        R"({"messages":[{"role":"user","content":"Q"}],"reasoning_effort":"bogus"})",
        R"({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"bad>name","arguments":{}}}]}]})",
        R"({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"not JSON"}}]}]})",
        R"({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{\"x\":1}"}}]}],"parse_tool_arguments":false})"
    }) rejected([&] { static_cast<void>(renderer.render(tokspur::parse_json(bad))); });
}

void data_literals() {
    const auto strings = tokspur::extract_flash_next_literals(R"({% set x = 'Reasoning effort is set to xhigh. X' %}
{% set l = 'Reasoning effort is set to low. L' %}
{{ '# Tools\n\nT' }} {{ '\n\nIf you choose to call a function ONLY reply X' }}
{# 'ignored unmatched quote #})");
    require(strings.tools_header == "# Tools\n\nT", "template literal extraction");
    rejected([] { static_cast<void>(tokspur::extract_flash_next_literals("'missing profile'")); });
    rejected([] { static_cast<void>(tokspur::extract_flash_next_literals("'\\x00'")); });
}
} // namespace

int main() {
    try {
        json_format();
        rendering();
        data_literals();
        std::cout << "Chat component checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Chat component check failed: " << error.what() << '\n';
        return 1;
    }
}
