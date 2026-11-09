#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

namespace {
bool try_input(const uint8_t* data, size_t size) {
    try {
        const auto bytes = std::span(reinterpret_cast<const std::byte*>(data), size);
        tokspur::Gguf file(bytes);
        for (const auto& tensor : file.tensors()) static_cast<void>(file.tensor_bytes(tensor));
        for (const auto& [key, value] : file.metadata()) {
            if (value.type == 8) static_cast<void>(file.string_value(key));
        }
    } catch (const tokspur::FormatError&) {
        // Invalid inputs are expected. Other exceptions and sanitizer findings fail.
        return false;
    }
    return true;
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static_cast<void>(try_input(data, size));
    return 0;
}

#ifdef TOKSPUR_STANDALONE_FUZZ
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected generated seed directory");
        std::vector<std::vector<uint8_t>> seeds;
        std::vector<std::filesystem::path> seed_paths;
        for (const auto& item : std::filesystem::directory_iterator(argv[1])) {
            if (!item.is_regular_file() || item.file_size() > 65536) throw std::runtime_error("invalid fuzz seed");
            seed_paths.push_back(item.path());
        }
        std::sort(seed_paths.begin(), seed_paths.end());
        for (const auto& path : seed_paths) {
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("cannot open generated fuzz seed");
            seeds.emplace_back(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }
        if (seeds.empty()) throw std::runtime_error("no generated fuzz seeds");
        std::mt19937 random(0x746f6b73);
        size_t accepted = 0;
        constexpr size_t rounds = 50000;
        for (size_t run = 0; run < rounds; ++run) {
            auto input = seeds[random() % seeds.size()];
            for (size_t mutation = 0, count = 1 + random() % 5; mutation < count; ++mutation) {
                const auto offset = input.empty() ? 0 : random() % input.size();
                switch (random() % 5) {
                    case 0: if (!input.empty()) input[offset] ^= uint8_t(1 << (random() % 8)); break;
                    case 1: if (!input.empty()) input[offset] = static_cast<uint8_t>(random()); break;
                    case 2: input.resize(offset); break;
                    case 3: input.insert(input.begin() + static_cast<ptrdiff_t>(offset), static_cast<uint8_t>(random())); break;
                    case 4:
                        for (size_t i = offset; i < std::min(offset + 8, input.size()); ++i) input[i] = 255;
                        break;
                }
            }
            accepted += try_input(input.data(), input.size());
        }
        std::cout << "Deterministic mutation checks: " << rounds << ", still valid: " << accepted
                  << " (not coverage-guided fuzzing)\n";
    } catch (const std::exception& error) {
        std::cerr << "Mutation check failed: " << error.what() << '\n';
        return 1;
    }
}
#endif
