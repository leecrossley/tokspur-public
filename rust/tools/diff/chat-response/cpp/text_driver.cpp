// Differential driver for the Rust port of chat.cpp, response.cpp and response_protocol.cpp. It reads length-prefixed cases from stdin and writes one length-prefixed outcome per case to stdout. The Rust runner in rust/tools/diff/src/main.rs produces the same outcome bytes from the Rust port; the two streams are compared case by case. Only FormatError is caught, as the C++ callers do; anything else aborts the run.
//
// Wire format (little-endian): u8, u32 and u64 integers; byte strings are a u32 length followed by the bytes.

#include "tokspur/chat.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/response.hpp"
#include "tokspur/response_protocol.hpp"
#include "tokspur/tokenizer.hpp"

#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tokspur;

namespace {

struct Reader {
    std::string_view data;
    size_t at = 0;
    uint8_t u8() {
        if (at >= data.size()) throw std::runtime_error("truncated case stream");
        return static_cast<uint8_t>(data[at++]);
    }
    uint32_t u32() {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) value |= uint32_t(u8()) << (8 * i);
        return value;
    }
    uint64_t u64() {
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i) value |= uint64_t(u8()) << (8 * i);
        return value;
    }
    std::string bytes() {
        const auto size = u32();
        if (size > data.size() - at) throw std::runtime_error("truncated case bytes");
        std::string value(data.substr(at, size));
        at += size;
        return value;
    }
};

struct Writer {
    std::string out;
    void u8(uint8_t value) { out.push_back(static_cast<char>(value)); }
    void u32(uint32_t value) { for (int i = 0; i < 4; ++i) u8(static_cast<uint8_t>(value >> (8 * i))); }
    void u64(uint64_t value) { for (int i = 0; i < 8; ++i) u8(static_cast<uint8_t>(value >> (8 * i))); }
    void bytes(std::string_view value) { u32(static_cast<uint32_t>(value.size())); out.append(value); }
};

class Identity final : public TextProcessor {
    std::vector<std::string> pieces(std::string_view text) const override { return {std::string(text)}; }
};

// test_response_protocol.cpp's fixture: 256 byte singletons and three special added markers (IDs 256-258), no merges.
const ByteBpe& fixture() {
    static const ByteBpe tokenizer = [] {
        BpeData data;
        for (unsigned byte = 0; byte < 256; ++byte) data.vocabulary.emplace_back(1, char(byte));
        for (const auto marker : {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}) {
            const auto id = uint32_t(data.vocabulary.size());
            data.vocabulary.emplace_back(marker);
            data.added.push_back({id, marker, true});
        }
        return ByteBpe(std::move(data));
    }();
    return tokenizer;
}

void dump_or_error(Writer& out, const Json& value) {
    try {
        const auto text = value.dump();
        out.u8(0);
        out.bytes(text);
    } catch (const FormatError& error) {
        out.u8(1);
        out.bytes(error.what());
    }
}

// A constructed value: strings and keys may be any bytes and nesting is unbounded, as the Json constructors allow. Numbers are parsed lexemes.
Json tree(Reader& in) {
    switch (in.u8()) {
        case 0: return Json();
        case 1: return Json(in.u8() != 0);
        case 2: {
            const auto lexeme = in.bytes();
            try { return parse_json(lexeme); } catch (const FormatError&) { return Json(); }
        }
        case 3: return Json(in.bytes());
        case 4: {
            Json::Array items(in.u32());
            for (auto& item : items) item = tree(in);
            return Json(std::move(items));
        }
        case 5: {
            Json::Object fields(in.u32());
            for (auto& field : fields) {
                field.first = in.bytes();
                field.second = tree(in);
            }
            return Json(std::move(fields));
        }
        default: throw std::runtime_error("unknown tree tag");
    }
}

void template_tree_case(Reader& in, Writer& out) {
    const auto value = tree(in);
    try {
        const auto rendered = template_json(value);
        out.u8('O');
        out.bytes(rendered);
    } catch (const FormatError& error) { out.u8('E'); out.bytes(error.what()); }
}

void render_tree_case(Reader& in, Writer& out) {
    ChatStrings strings{in.bytes(), in.bytes(), in.bytes(), in.bytes()};
    const auto request = tree(in);
    const ChatRenderer renderer(std::move(strings));
    RenderedChat rendered;
    try { rendered = renderer.render(request); }
    catch (const FormatError& error) { out.u8('E'); out.bytes(error.what()); return; }
    out.u8('O');
    out.bytes(rendered.prompt);
    dump_or_error(out, rendered.normalized_messages);
    out.u64(rendered.parsed_tool_arguments);
    out.u64(rendered.hoisted_system_messages);
}

void template_case(Reader& in, Writer& out) {
    const auto text = in.bytes();
    Json value;
    try { value = parse_json(text); } catch (const FormatError& error) { out.u8('P'); out.bytes(error.what()); return; }
    try {
        const auto rendered = template_json(value);
        out.u8('O');
        out.bytes(rendered);
    } catch (const FormatError& error) { out.u8('E'); out.bytes(error.what()); }
}

void render_case(Reader& in, Writer& out) {
    ChatStrings strings{in.bytes(), in.bytes(), in.bytes(), in.bytes()};
    const auto text = in.bytes();
    Json request;
    try { request = parse_json(text); } catch (const FormatError& error) { out.u8('P'); out.bytes(error.what()); return; }
    std::unique_ptr<ChatRenderer> renderer;
    try { renderer = std::make_unique<ChatRenderer>(std::move(strings)); }
    catch (const FormatError& error) { out.u8('C'); out.bytes(error.what()); return; }
    RenderedChat rendered;
    try { rendered = renderer->render(request); }
    catch (const FormatError& error) { out.u8('E'); out.bytes(error.what()); return; }
    out.u8('O');
    out.bytes(rendered.prompt);
    dump_or_error(out, rendered.normalized_messages);
    out.u64(rendered.parsed_tool_arguments);
    out.u64(rendered.hoisted_system_messages);
}

void extract_case(Reader& in, Writer& out) {
    const auto source = in.bytes();
    try {
        const auto strings = extract_flash_next_literals(source);
        out.u8('O');
        out.bytes(strings.reasoning_xhigh);
        out.bytes(strings.reasoning_low);
        out.bytes(strings.tools_header);
        out.bytes(strings.tool_instructions);
    } catch (const FormatError& error) { out.u8('E'); out.bytes(error.what()); }
}

void trim_case(Reader& in, Writer& out) {
    const auto text = in.bytes();
    try {
        const auto trimmed = chat_trim(text);
        out.u8('O');
        out.bytes(trimmed);
    } catch (const FormatError& error) { out.u8('E'); out.bytes(error.what()); }
}

void snapshot(Writer& out, ResponseStatus returned, const ResponseParser& parser) {
    out.u8(static_cast<uint8_t>(returned));
    out.u8(static_cast<uint8_t>(parser.status()));
    out.bytes(parser.error());
    out.u64(parser.bytes_received());
    const auto records = parser.records();
    out.u32(static_cast<uint32_t>(records.size()));
    for (const auto& record : records) {
        out.u8(static_cast<uint8_t>(record.kind));
        out.bytes(record.text);
        out.u64(record.ordinal);
        out.bytes(record.name);
        dump_or_error(out, record.arguments);
    }
}

void parser_case(Reader& in, Writer& out) {
    std::vector<ToolSpec> tools(in.u32());
    for (auto& tool : tools) {
        tool.name = in.bytes();
        tool.parameters.resize(in.u32());
        for (auto& parameter : tool.parameters) {
            parameter.name = in.bytes();
            parameter.type = static_cast<ParameterType>(in.u8());
            parameter.required = in.u8() != 0;
        }
    }
    const bool thinking = in.u8() != 0;
    ResponseLimits limits{in.u64(), in.u64(), in.u64(), in.u64(), in.u64(), in.u64()};
    struct Operation { bool finish; std::string bytes; };
    std::vector<Operation> operations(in.u32());
    for (auto& operation : operations) {
        operation.finish = in.u8() != 0;
        if (!operation.finish) operation.bytes = in.bytes();
    }
    std::unique_ptr<ResponseParser> parser;
    try { parser = std::make_unique<ResponseParser>(std::move(tools), thinking, limits); }
    catch (const FormatError& error) { out.u8('C'); out.bytes(error.what()); return; }
    out.u8('O');
    for (const auto& operation : operations) {
        const auto status = operation.finish ? parser->finish() : parser->feed(operation.bytes);
        snapshot(out, status, *parser);
    }
}

void protocol_case(Reader& in, Writer& out) {
    ResponseAdmission admission{in.u32(), in.u32(), in.u32(), in.u32()};
    const auto text = in.bytes();
    const auto mode = in.u8();
    Json envelope;
    try { envelope = parse_json(text); } catch (const FormatError& error) { out.u8('P'); out.bytes(error.what()); return; }
    uint64_t calls = 0;
    ResponseDecoder decode;
    if (mode == 0) decode = [&](std::span<const uint32_t> ids) { ++calls; return fixture().decode_bytes(ids); };
    else if (mode == 2) decode = [&](std::span<const uint32_t>) { ++calls; return std::string("wrong terminal"); };
    else if (mode == 3) decode = [&](std::span<const uint32_t>) { ++calls; return std::string(response_request_bytes + 1, 'x'); };
    try {
        const auto result = parse_native_response(envelope, admission, decode);
        out.u8('O');
        dump_or_error(out, result);
    } catch (const FormatError& error) { out.u8('E'); out.bytes(error.what()); }
    out.u64(calls);
}

} // namespace

int main() {
    std::ios::sync_with_stdio(false);
    const std::string input{std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
    Reader in{input};
    std::string output;
    while (in.at < input.size()) {
        Writer out;
        switch (in.u8()) {
            case 1: template_case(in, out); break;
            case 2: render_case(in, out); break;
            case 3: extract_case(in, out); break;
            case 4: trim_case(in, out); break;
            case 5: parser_case(in, out); break;
            case 6: protocol_case(in, out); break;
            case 7: template_tree_case(in, out); break;
            case 8: render_tree_case(in, out); break;
            default: std::cerr << "unknown case kind\n"; return 2;
        }
        Writer frame;
        frame.bytes(out.out);
        output += frame.out;
    }
    std::cout.write(output.data(), static_cast<std::streamsize>(output.size()));
    std::cout.flush();
    return std::cout ? 0 : 3;
}
