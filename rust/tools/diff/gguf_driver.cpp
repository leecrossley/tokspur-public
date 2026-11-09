// Differential driver for the Rust GGUF port (rust/crates/tokspur-gguf).
//
// Prints, for each case, a canonical summary of what native/src/gguf.cpp accepts, refuses and returns. rust/tools/diff/gguf.sh builds it into build/rust-diff/, and the ignored tests in rust/crates/tokspur-gguf/tests/gguf/differential.rs print the same summary from the Rust reader (tests/gguf/summary.rs) and compare them. Not part of native/Makefile or any shipped binary.
//
//   gguf-driver pack FILE             Gguf(std::span) per length-prefixed input
//   gguf-driver files LIST            Gguf(path) per line
//   gguf-driver models LIST           ModelFiles(paths) per line; tab-separated paths
//   gguf-driver mapped LIST           MappedFile(path) per line
//   gguf-driver fingerprints PATH...  ModelFiles(paths): headers and tokenizer
//                                     fingerprints only, never tensor data
#include "tokspur/gguf.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using tokspur::FormatError;

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
    std::array<char, 17> text{};
    std::snprintf(text.data(), text.size(), "%016" PRIx64, value);
    return text.data();
}

std::string hex_text(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (const unsigned char c : value) {
        result += digits[c >> 4];
        result += digits[c & 15];
    }
    return result;
}

// string_fingerprint from native/src/text.mm.
std::string fingerprint(const std::vector<std::string_view>& values) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    const auto length = [&](uint64_t value) {
        std::array<unsigned char, 8> bytes{};
        for (size_t i = 0; i < 8; ++i) bytes[i] = static_cast<unsigned char>(value >> (8 * i));
        CC_SHA256_Update(&context, bytes.data(), bytes.size());
    };
    length(values.size());
    for (const auto& value : values) {
        length(value.size());
        CC_SHA256_Update(&context, value.data(), static_cast<CC_LONG>(value.size()));
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return hex_text({reinterpret_cast<const char*>(digest.data()), digest.size()});
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

void model_summary(std::ostream& out, const tokspur::ModelFiles& model) {
    out << "ok shards=" << model.shards().size() << " tensors=" << model.tensors().size()
        << " architecture=" << hex_text(model.architecture()) << '\n';
    for (const auto& [name, location] : model.tensors()) {
        out << "location " << name << " shard=" << location.shard << " index=" << location.index << '\n';
        try {
            const auto& tensor = model.tensor(name);
            out << "tensor offset=" << tensor.offset << " bytes=" << tensor.bytes << '\n';
            if (tensor.bytes > tensor_bytes_limit) {
                out << "tensor_bytes skipped\n";
            } else {
                const auto bytes = model.tensor_bytes(name);
                out << "tensor_bytes " << bytes.size() << ' ' << hex(fnv(fnv_basis, bytes)) << '\n';
            }
        } catch (const FormatError& error) {
            out << "tensor error " << error.what() << '\n';
        }
    }
    for (size_t s = 0; s < model.shards().size(); ++s) {
        const auto& shard = model.shards()[s];
        if (!shard.find("tokenizer.ggml.tokens")) continue;
        try {
            const auto tokens = shard.string_array("tokenizer.ggml.tokens");
            const auto merges = shard.string_array("tokenizer.ggml.merges");
            out << "tokenizer " << s << " tokens " << tokens.size() << ' ' << hex(strings_hash(tokens))
                << " merges " << merges.size() << ' ' << hex(strings_hash(merges)) << '\n';
        } catch (const FormatError& error) {
            out << "tokenizer " << s << " error " << error.what() << '\n';
        }
    }
    try {
        static_cast<void>(model.tensor("missing"));
        out << "unknown accepted\n";
    } catch (const FormatError& error) {
        out << "unknown error " << error.what() << '\n';
    }
    try {
        static_cast<void>(model.tensor_bytes("missing"));
        out << "unknown_bytes accepted\n";
    } catch (const FormatError& error) {
        out << "unknown_bytes error " << error.what() << '\n';
    }
    try {
        model.assert_unchanged();
        out << "unchanged ok\n";
    } catch (const FormatError& error) {
        out << "unchanged error " << error.what() << '\n';
    }
}

void mapped_summary(std::ostream& out, const tokspur::MappedFile& file) {
    out << "ok size=" << file.size() << " bytes=" << hex(fnv(fnv_basis, file.bytes())) << '\n';
    try {
        file.assert_unchanged();
        out << "unchanged ok\n";
    } catch (const FormatError& error) {
        out << "unchanged error " << error.what() << '\n';
    }
}

void fingerprint_summary(std::ostream& out, const tokspur::ModelFiles& model) {
    out << "ok shards=" << model.shards().size() << " tensors=" << model.tensors().size()
        << " architecture=" << hex_text(model.architecture()) << '\n';
    for (size_t s = 0; s < model.shards().size(); ++s) {
        const auto& shard = model.shards()[s];
        out << "shard " << s << " file_size=" << shard.file_size() << " data_offset=" << shard.data_offset()
            << " tensors=" << shard.tensors().size() << " metadata=" << shard.metadata().size() << '\n';
    }
    for (size_t s = 0; s < model.shards().size(); ++s) {
        const auto& shard = model.shards()[s];
        if (!shard.find("tokenizer.ggml.tokens")) continue;
        try {
            const auto tokens = shard.string_array("tokenizer.ggml.tokens");
            const auto merges = shard.string_array("tokenizer.ggml.merges");
            out << "tokenizer " << s << " tokens " << tokens.size() << ' ' << fingerprint(tokens)
                << " merges " << merges.size() << ' ' << fingerprint(merges) << '\n';
        } catch (const FormatError& error) {
            out << "tokenizer " << s << " error " << error.what() << '\n';
        }
    }
    try {
        model.assert_unchanged();
        out << "unchanged ok\n";
    } catch (const FormatError& error) {
        out << "unchanged error " << error.what() << '\n';
    }
}

// One case: a header line, then the summary or the constructor's refusal.
template <typename Action>
void run_case(std::ostream& out, size_t index, Action&& action) {
    out << "case " << index << '\n';
    try {
        action();
    } catch (const FormatError& error) {
        out << "error " << error.what() << '\n';
    } catch (const std::exception& error) {
        out << "exception " << error.what() << '\n';
    }
}

std::vector<std::string> lines(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read case list");
    std::vector<std::string> result;
    for (std::string line; std::getline(input, line);) result.push_back(line);
    return result;
}

std::vector<std::filesystem::path> paths(const std::string& line) {
    std::vector<std::filesystem::path> result;
    if (line.empty()) return result;
    for (size_t start = 0;;) {
        const auto tab = line.find('\t', start);
        result.emplace_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
        if (tab == std::string::npos) return result;
        start = tab + 1;
    }
}

void pack(std::ostream& out, const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read case pack");
    const std::vector<char> data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto bytes = std::as_bytes(std::span(data));
    size_t offset = 0;
    for (size_t index = 0; offset < bytes.size(); ++index) {
        if (bytes.size() - offset < 8) throw std::runtime_error("truncated case pack");
        uint64_t length = 0;
        for (size_t i = 0; i < 8; ++i) length |= std::to_integer<uint64_t>(bytes[offset + i]) << (8 * i);
        offset += 8;
        if (length > bytes.size() - offset) throw std::runtime_error("truncated case pack");
        const auto input_bytes = bytes.subspan(offset, static_cast<size_t>(length));
        offset += static_cast<size_t>(length);
        run_case(out, index, [&] {
            const tokspur::Gguf file(input_bytes);
            gguf_summary(out, file);
        });
    }
}

} // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    try {
        if (argc < 3) throw std::runtime_error("usage: gguf-driver pack|files|models|mapped LIST | fingerprints PATH...");
        const std::string_view mode(argv[1]);
        auto& out = std::cout;
        if (mode == "pack" && argc == 3) {
            pack(out, argv[2]);
        } else if (mode == "files" && argc == 3) {
            const auto cases = lines(argv[2]);
            for (size_t i = 0; i < cases.size(); ++i) {
                run_case(out, i, [&] {
                    const tokspur::Gguf file{std::filesystem::path(cases[i])};
                    gguf_summary(out, file);
                });
            }
        } else if (mode == "models" && argc == 3) {
            const auto cases = lines(argv[2]);
            for (size_t i = 0; i < cases.size(); ++i) {
                run_case(out, i, [&] {
                    const tokspur::ModelFiles model(paths(cases[i]));
                    model_summary(out, model);
                });
            }
        } else if (mode == "mapped" && argc == 3) {
            const auto cases = lines(argv[2]);
            for (size_t i = 0; i < cases.size(); ++i) {
                run_case(out, i, [&] {
                    const tokspur::MappedFile file{std::filesystem::path(cases[i])};
                    mapped_summary(out, file);
                });
            }
        } else if (mode == "fingerprints") {
            const std::vector<std::filesystem::path> shards(argv + 2, argv + argc);
            run_case(out, 0, [&] {
                const tokspur::ModelFiles model(shards);
                fingerprint_summary(out, model);
            });
        } else {
            throw std::runtime_error("unknown driver mode");
        }
        out.flush();
        return out ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gguf-driver: " << error.what() << '\n';
        return 1;
    }
}
