// Narrow extern "C" shim over the unchanged C++ parsers in native/src, for the differential fuzz targets in rust/crates/tokspur-fuzz (development tooling, never part of a shipped binary). build.rs compiles it with native/src/json.cpp, unicode.cpp, gguf.cpp and response.cpp, all unchanged.
//
// Each entry point encodes one case's complete observable outcome as bytes in a thread-local buffer and returns a pointer to it; the pointer stays valid until the next call on the same thread. src/oracle.rs copies the bytes out at once, and src/outcome.rs produces the same encoding from the Rust port. No exception ever crosses the C boundary: tokspur::FormatError is part of the encoded outcome, any other exception becomes an 'X' (std::exception) or 'U' (anything else) outcome, which the Rust side never produces, so it fails the comparison.
//
// Encoding (little-endian): u8, u32 and u64 integers; byte strings are a u32 length followed by the bytes.

#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include "tokspur/response.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using tokspur::FormatError;
using tokspur::Json;

thread_local std::string outcome;

struct Writer {
    std::string out;
    void u8(uint8_t value) { out.push_back(static_cast<char>(value)); }
    void u32(uint32_t value) { for (int i = 0; i < 4; ++i) u8(static_cast<uint8_t>(value >> (8 * i))); }
    void u64(uint64_t value) { for (int i = 0; i < 8; ++i) u8(static_cast<uint8_t>(value >> (8 * i))); }
    void bytes(std::string_view value) {
        if (value.size() > UINT32_MAX) throw std::length_error("oracle field exceeds u32");
        u32(static_cast<uint32_t>(value.size()));
        out.append(value);
    }
};

struct Reader {
    std::string_view data;
    size_t at = 0;
    uint8_t u8() {
        if (at >= data.size()) throw std::runtime_error("truncated oracle case");
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
        if (size > data.size() - at) throw std::runtime_error("truncated oracle case bytes");
        std::string value(data.substr(at, size));
        at += size;
        return value;
    }
};

std::string_view view(const uint8_t* data, size_t size) {
    return size ? std::string_view(reinterpret_cast<const char*>(data), size) : std::string_view();
}

// Runs one case; an exception other than those the case encodes itself replaces the partial outcome.
template <typename Action>
const uint8_t* run(size_t* size, Action&& action) {
    Writer out;
    try {
        action(out);
    } catch (const std::exception& error) {
        out.out.clear();
        out.u8('X');
        out.bytes(error.what());
    } catch (...) {
        out.out.clear();
        out.u8('U');
    }
    outcome = std::move(out.out);
    *size = outcome.size();
    return reinterpret_cast<const uint8_t*>(outcome.data());
}

// --- JSON -------------------------------------------------------------------

// Every number, in document order: its lexeme, Json::unsigned_integer, and libc++ std::from_chars/std::to_chars for double (what tokspur-json's from_chars_f64 and to_chars_shortest reproduce).
void numbers(Writer& out, const Json& value) {
    switch (value.kind()) {
        case Json::Kind::number: {
            const auto lexeme = value.number_lexeme();
            out.u8('n');
            out.bytes(lexeme);
            try {
                const auto integer = value.unsigned_integer();
                out.u8('O');
                out.u64(integer);
            } catch (const FormatError& error) {
                out.u8('E');
                out.bytes(error.what());
            }
            double parsed = 0;
            const auto end = lexeme.data() + lexeme.size();
            const auto read = std::from_chars(lexeme.data(), end, parsed);
            if (read.ec != std::errc{} || read.ptr != end) {
                out.u8('N');
                break;
            }
            out.u8('O');
            out.u64(std::bit_cast<uint64_t>(parsed));
            std::array<char, 64> text{};
            const auto written = std::to_chars(text.data(), text.data() + text.size(), parsed);
            if (written.ec != std::errc{}) {
                out.u8('E');
                break;
            }
            out.u8('O');
            out.bytes(std::string_view(text.data(), static_cast<size_t>(written.ptr - text.data())));
            break;
        }
        case Json::Kind::array:
            for (const auto& item : value.array()) numbers(out, item);
            break;
        case Json::Kind::object:
            for (const auto& [key, item] : value.object()) {
                static_cast<void>(key);
                numbers(out, item);
            }
            break;
        default:
            break;
    }
}

void json_case(Writer& out, std::string_view text, tokspur::JsonLimits limits) {
    Json value;
    try {
        value = tokspur::parse_json(text, limits);
    } catch (const FormatError& error) {
        out.u8('E');
        out.bytes(error.what());
        return;
    }
    out.u8('O');
    try {
        const auto dumped = value.dump();
        out.u8('O');
        out.bytes(dumped);
    } catch (const FormatError& error) {
        out.u8('E');
        out.bytes(error.what());
    }
    numbers(out, value);
}

// --- GGUF -------------------------------------------------------------------

// gguf_summary from rust/tools/diff/gguf_driver.cpp: the constructor's refusal, or every accessor on every key and every tensor.
constexpr uint64_t fnv_basis = 0xcbf29ce484222325ULL;
constexpr uint64_t tensor_bytes_limit = 16 * 1024 * 1024;

uint64_t fnv(uint64_t hash, std::span<const std::byte> bytes) {
    for (const auto byte : bytes) {
        hash ^= std::to_integer<uint64_t>(byte);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

uint64_t fnv(uint64_t hash, std::string_view text) {
    return fnv(hash, std::as_bytes(std::span(text.data(), text.size())));
}

uint64_t fnv_number(uint64_t hash, uint64_t value) {
    std::array<std::byte, 8> bytes{};
    for (size_t i = 0; i < 8; ++i) bytes[i] = std::byte((value >> (8 * i)) & 255);
    return fnv(hash, bytes);
}

uint64_t strings_hash(const std::vector<std::string_view>& values) {
    auto hash = fnv_basis;
    for (const auto value : values) hash = fnv(fnv_number(hash, value.size()), value);
    return hash;
}

std::string hex(uint64_t value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string text(16, '0');
    for (size_t i = 0; i < 16; ++i) text[15 - i] = digits[(value >> (4 * i)) & 15];
    return text;
}

void gguf_summary(std::ostream& out, const tokspur::Gguf& file) {
    out << "ok tensors=" << file.tensors().size() << " metadata=" << file.metadata().size()
        << " data_offset=" << file.data_offset() << " file_size=" << file.file_size() << '\n';
    for (const auto& [key, item] : file.metadata()) {
        out << "key " << key << " type=" << item.type << " offset=" << item.offset << " end=" << item.end
            << " element_type=" << item.element_type << " elements=" << item.elements << '\n';
        try {
            const auto value = file.unsigned_value(key);
            out << "unsigned " << value << '\n';
        } catch (const FormatError& error) {
            out << "unsigned error " << error.what() << '\n';
        }
        try {
            const auto value = file.floating_value(key);
            out << "floating " << hex(std::bit_cast<uint64_t>(value)) << '\n';
        } catch (const FormatError& error) {
            out << "floating error " << error.what() << '\n';
        }
        try {
            const auto values = file.unsigned_array(key);
            auto hash = fnv_basis;
            for (const auto value : values) hash = fnv_number(hash, value);
            out << "unsigned_array " << values.size() << ' ' << hex(hash) << '\n';
        } catch (const FormatError& error) {
            out << "unsigned_array error " << error.what() << '\n';
        }
        try {
            const auto value = file.string_value(key);
            out << "string " << value.size() << ' ' << hex(fnv(fnv_basis, value)) << '\n';
        } catch (const FormatError& error) {
            out << "string error " << error.what() << '\n';
        }
        try {
            const auto values = file.string_array(key);
            out << "string_array " << values.size() << ' ' << hex(strings_hash(values)) << '\n';
        } catch (const FormatError& error) {
            out << "string_array error " << error.what() << '\n';
        }
    }
    for (const auto& tensor : file.tensors()) {
        out << "tensor " << tensor.name << " type=" << tensor.type << " offset=" << tensor.offset
            << " bytes=" << tensor.bytes << " elements=" << tensor.elements << " dimensions=";
        for (size_t i = 0; i < tensor.dimensions.size(); ++i) out << (i ? "," : "") << tensor.dimensions[i];
        out << '\n';
        if (tensor.bytes > tensor_bytes_limit) {
            out << "tensor_bytes skipped\n";
        } else {
            try {
                const auto bytes = file.tensor_bytes(tensor);
                out << "tensor_bytes " << bytes.size() << ' ' << hex(fnv(fnv_basis, bytes)) << '\n';
            } catch (const FormatError& error) {
                out << "tensor_bytes error " << error.what() << '\n';
            }
        }
        try {
            const auto copy = tensor;
            static_cast<void>(file.tensor_bytes(copy));
            out << "copy accepted\n";
        } catch (const FormatError& error) {
            out << "copy error " << error.what() << '\n';
        }
    }
    try {
        file.assert_unchanged();
        out << "unchanged ok\n";
    } catch (const FormatError& error) {
        out << "unchanged error " << error.what() << '\n';
    }
}

void gguf_case(Writer& out, std::string_view input) {
    std::ostringstream text;
    try {
        const auto bytes = std::as_bytes(std::span(input.data(), input.size()));
        const tokspur::Gguf file(bytes);
        gguf_summary(text, file);
    } catch (const FormatError& error) {
        text << "error " << error.what() << '\n';
    }
    out.out = text.str();
}

// --- Response parser ----------------------------------------------------------

// parser_case and snapshot from rust/tools/diff/chat-response/cpp/text_driver.cpp.
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

void snapshot(Writer& out, tokspur::ResponseStatus returned, const tokspur::ResponseParser& parser) {
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

void response_case(Writer& out, std::string_view input) {
    Reader in{input};
    std::vector<tokspur::ToolSpec> tools(in.u32());
    for (auto& tool : tools) {
        tool.name = in.bytes();
        tool.parameters.resize(in.u32());
        for (auto& parameter : tool.parameters) {
            parameter.name = in.bytes();
            parameter.type = static_cast<tokspur::ParameterType>(in.u8());
            parameter.required = in.u8() != 0;
        }
    }
    const bool thinking = in.u8() != 0;
    const bool token_framing = in.u8() != 0;
    tokspur::ResponseLimits limits{in.u64(), in.u64(), in.u64(), in.u64(), in.u64(), in.u64()};
    struct Operation { bool finish; std::string bytes; };
    std::vector<Operation> operations(in.u32());
    for (auto& operation : operations) {
        operation.finish = in.u8() != 0;
        if (!operation.finish) operation.bytes = in.bytes();
    }
    if (in.at != input.size()) throw std::runtime_error("trailing oracle case bytes");
    std::unique_ptr<tokspur::ResponseParser> parser;
    try {
        parser = std::make_unique<tokspur::ResponseParser>(std::move(tools), thinking, limits, false, token_framing);
    } catch (const FormatError& error) {
        out.u8('C');
        out.bytes(error.what());
        return;
    }
    out.u8('O');
    for (const auto& operation : operations) {
        const auto status = operation.finish ? parser->finish() : parser->feed(operation.bytes);
        snapshot(out, status, *parser);
    }
}

} // namespace

extern "C" {

const uint8_t* tokspur_oracle_json(const uint8_t* input, size_t input_size, size_t bytes, size_t nodes,
                                   size_t depth, size_t* size) {
    return run(size, [&](Writer& out) { json_case(out, view(input, input_size), {bytes, nodes, depth}); });
}

const uint8_t* tokspur_oracle_gguf(const uint8_t* input, size_t input_size, size_t* size) {
    return run(size, [&](Writer& out) { gguf_case(out, view(input, input_size)); });
}

const uint8_t* tokspur_oracle_response(const uint8_t* input, size_t input_size, size_t* size) {
    return run(size, [&](Writer& out) { response_case(out, view(input, input_size)); });
}

} // extern "C"
