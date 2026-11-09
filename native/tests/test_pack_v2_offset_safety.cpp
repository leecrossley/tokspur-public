#include "tokspur/derived_pack.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace tokspur;
using namespace tokspur::derived_pack;

namespace {
using Bytes = std::vector<std::byte>;
size_t checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template <typename F>
void refuses(F&& func, const char* message) {
    ++checks;
    bool caught = false;
    try {
        func();
    } catch (const FormatError&) {
        caught = true;
    } catch (const std::exception&) {
        caught = true;
    }
    if (!caught) throw std::runtime_error(message);
}

void number(Bytes& bytes, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        bytes.push_back(std::byte((value >> (i * 8)) & 255));
    }
}

void string(Bytes& bytes, std::string_view value) {
    number(bytes, value.size(), 8);
    for (const unsigned char c : value) {
        bytes.push_back(std::byte(c));
    }
}

void metadata_string(Bytes& b, const char* key, std::string_view value) {
    string(b, key);
    number(b, 8, 4);
    string(b, value);
}

void metadata_uint(Bytes& b, const char* key, uint32_t value) {
    string(b, key);
    number(b, 4, 4);
    number(b, value, 4);
}

struct TempFileGuard {
    std::filesystem::path path;
    int fd = -1;

    explicit TempFileGuard(const std::string& name_prefix) {
        const auto temp_dir = std::filesystem::temp_directory_path();
        path = temp_dir / (name_prefix + "_" + std::to_string(::getpid()) + "_" + std::to_string(arc4random()) + ".tmp");
        fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("failed to create temporary sparse file");
    }

    ~TempFileGuard() {
        if (fd >= 0) ::close(fd);
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

void pwrite_all(int fd, std::span<const std::byte> bytes, off_t offset) {
    for (size_t n = 0; n < bytes.size();) {
        const auto got = ::pwrite(fd, bytes.data() + n, bytes.size() - n, offset + off_t(n));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) throw std::runtime_error("pwrite failed on sparse fixture");
        n += size_t(got);
    }
}
} // namespace

int main() {
    try {
        std::cout << "=== Running Milestone 4 uint64_t 64-Bit Offset Safety Tests ===" << std::endl;

        // -------------------------------------------------------------
        // 1. Pure Arithmetic and 64-bit Boundary Tests
        // -------------------------------------------------------------
        {
            // 2 GiB boundary (2^31 = 2,147,483,648)
            constexpr uint64_t two_gib = 2ULL * 1024 * 1024 * 1024;
            require(two_gib == 2147483648ULL, "2 GiB exact definition");
            require(two_gib - 1 < two_gib, "2 GiB - 1 under boundary");
            require(two_gib + 1 > two_gib, "2 GiB + 1 exceeds boundary");

            // Verify checked_add around 2 GiB
            require(checked_add(two_gib - 1, 1ULL) == two_gib, "checked_add at 2 GiB boundary");
            require(checked_add(two_gib, 1024ULL) == 2147484672ULL, "checked_add above 2 GiB");

            // 4 GiB boundary (2^32 = 4,294,967,296)
            constexpr uint64_t four_gib = 4ULL * 1024 * 1024 * 1024;
            require(four_gib == 4294967296ULL, "4 GiB exact definition");
            require(four_gib - 1 < four_gib, "4 GiB - 1 under boundary");
            require(four_gib + 1 > four_gib, "4 GiB + 1 exceeds boundary");

            // Verify checked_add and checked_mul around 4 GiB
            require(checked_add(four_gib - 1, 1ULL) == four_gib, "checked_add at 4 GiB boundary");
            require(checked_add(four_gib, 4096ULL) == 4294971392ULL, "checked_add above 4 GiB");
            require(checked_mul(four_gib, 2ULL) == 8ULL * 1024 * 1024 * 1024, "checked_mul to 8 GiB");

            // 85 GB pack planning bound
            constexpr uint64_t eighty_five_gb = 85ULL * 1024 * 1024 * 1024;
            require(eighty_five_gb > four_gib, "85 GB exceeds 32-bit unsigned capacity");
            require(checked_add(eighty_five_gb, 512ULL * 1024 * 1024) > eighty_five_gb, "85 GB + staging buffer");

            // Overflow detection past uint64 max
            refuses([] {
                checked_add(std::numeric_limits<uint64_t>::max() - 10, 20ULL);
            }, "checked_add must throw FormatError on uint64 overflow");

            refuses([] {
                checked_mul(std::numeric_limits<uint64_t>::max() / 2 + 1, 2ULL);
            }, "checked_mul must throw FormatError on uint64 overflow");
        }

        // -------------------------------------------------------------
        // 2. Alignment and Reserve Rule Checks
        // -------------------------------------------------------------
        {
            require(pack_alignment == 16384, "page alignment is 16 KiB");
            require(pack_version == 2, "pack version is 2");

            // Available space reserve rule: available >= peak + 50 GB
            constexpr uint64_t fifty_gb = 50ULL * 1024 * 1024 * 1024;
            constexpr uint64_t peak_alloc = 85ULL * 1024 * 1024 * 1024;
            require(check_disk_reserve(peak_alloc + fifty_gb + 1000, peak_alloc, fifty_gb), "budget check with surplus");
            require(check_disk_reserve(peak_alloc + fifty_gb, peak_alloc, fifty_gb), "budget check exact match");
            require(!check_disk_reserve(peak_alloc + fifty_gb - 1, peak_alloc, fifty_gb), "budget check 1 byte deficit");
            require(!check_disk_reserve(0, peak_alloc, fifty_gb), "zero available space refused");
        }

        // -------------------------------------------------------------
        // 3. APFS Sparse File Fixture with Offsets > 2 GiB and > 4 GiB
        // -------------------------------------------------------------
        {
            TempFileGuard sparse_file("tokspur_offset_safety");

            // Target tensor offsets:
            // Header: 16 KiB (0 .. 16383)
            // Tensor 1: at 2 GiB + 16 KiB relative offset (> 2 GiB)
            // Tensor 2: at 4 GiB + 32 KiB relative offset (> 4 GiB)
            // Tensor 3: at 8 GiB relative offset (> 8 GiB)
            // Tail marker: at 9 GiB to ensure file extent
            constexpr uint64_t data_offset = 16384;
            constexpr uint64_t offset_2gib = 2ULL * 1024 * 1024 * 1024 + 16384;
            constexpr uint64_t offset_4gib = 4ULL * 1024 * 1024 * 1024 + 32768;
            constexpr uint64_t offset_8gib = 8ULL * 1024 * 1024 * 1024 + 16384;
            constexpr uint64_t tail_offset = 9ULL * 1024 * 1024 * 1024;

            require(offset_2gib > 2147483648ULL, "offset_2gib > 2 GiB");
            require(offset_4gib > 4294967296ULL, "offset_4gib > 4 GiB");
            require(offset_8gib > 8589934592ULL, "offset_8gib > 8 GiB");

            // Construct minimal valid GGUF header
            Bytes hdr;
            number(hdr, 0x46554747, 4); // "GGUF"
            number(hdr, 3, 4);          // version 3
            number(hdr, 3, 8);          // 3 tensors
            number(hdr, 3, 8);          // 3 metadata fields

            metadata_string(hdr, "general.architecture", "tokspur_pack");
            metadata_uint(hdr, "general.alignment", uint32_t(pack_alignment));
            metadata_uint(hdr, "tokspur_pack.version", 2);

            // Tensor 1 descriptor
            string(hdr, "tensor_above_2gib");
            number(hdr, 1, 4); // 1 dimension
            number(hdr, 4, 8); // 4 elements
            number(hdr, 0, 4); // F32 (16 bytes)
            number(hdr, offset_2gib, 8);

            // Tensor 2 descriptor
            string(hdr, "tensor_above_4gib");
            number(hdr, 1, 4);
            number(hdr, 4, 8);
            number(hdr, 0, 4);
            number(hdr, offset_4gib, 8);

            // Tensor 3 descriptor
            string(hdr, "tensor_above_8gib");
            number(hdr, 1, 4);
            number(hdr, 4, 8);
            number(hdr, 0, 4);
            number(hdr, offset_8gib, 8);

            require(hdr.size() <= data_offset, "header fits in 16 KiB");
            hdr.resize(data_offset, std::byte(0)); // Pad to 16 KiB alignment

            // Write header at offset 0
            pwrite_all(sparse_file.fd, hdr, 0);

            // Payload test patterns (16 bytes each)
            const std::array<uint8_t, 16> pattern_2gib{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
            const std::array<uint8_t, 16> pattern_4gib{30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45};
            const std::array<uint8_t, 16> pattern_8gib{50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65};
            const std::array<uint8_t, 4> tail_pattern{0xde, 0xad, 0xbe, 0xef};

            // Write payload 1 at data_offset + offset_2gib
            pwrite_all(sparse_file.fd, std::as_bytes(std::span(pattern_2gib)), off_t(data_offset + offset_2gib));

            // Write payload 2 at data_offset + offset_4gib
            pwrite_all(sparse_file.fd, std::as_bytes(std::span(pattern_4gib)), off_t(data_offset + offset_4gib));

            // Write payload 3 at data_offset + offset_8gib
            pwrite_all(sparse_file.fd, std::as_bytes(std::span(pattern_8gib)), off_t(data_offset + offset_8gib));

            // Write tail marker to seal file extent
            pwrite_all(sparse_file.fd, std::as_bytes(std::span(tail_pattern)), off_t(tail_offset));

            // Inspect APFS sparse allocation: physical blocks must be small despite 9 GiB logical size
            struct stat st{};
            require(::fstat(sparse_file.fd, &st) == 0, "fstat on sparse file");
            require(uint64_t(st.st_size) >= tail_offset + 4, "logical file size exceeds 9 GiB");
            const uint64_t physical_bytes = uint64_t(st.st_blocks) * 512;
            require(physical_bytes < 1024 * 1024, "APFS sparse file physical disk consumption under 1 MiB");

            // Open sparse file via Gguf parser
            Gguf parsed(sparse_file.path);
            require(parsed.tensors().size() == 3, "parsed exactly 3 tensors");
            require(parsed.data_offset() == data_offset, "data offset matches 16 KiB alignment");

            // Verify Tensor 1 (> 2 GiB)
            const auto& t1 = parsed.tensors()[0];
            require(t1.name == "tensor_above_2gib", "tensor 1 name");
            require(t1.offset == data_offset + offset_2gib, "tensor 1 absolute offset");
            require(t1.offset > 2147483648ULL, "tensor 1 offset exceeds 2 GiB");
            require(t1.bytes == 16, "tensor 1 size is 16 bytes");
            auto bytes1 = parsed.tensor_bytes(t1);
            require(std::ranges::equal(bytes1, std::as_bytes(std::span(pattern_2gib))),
                    "tensor 1 bytes readback bit-exact across > 2 GiB boundary");

            // Verify Tensor 2 (> 4 GiB)
            const auto& t2 = parsed.tensors()[1];
            require(t2.name == "tensor_above_4gib", "tensor 2 name");
            require(t2.offset == data_offset + offset_4gib, "tensor 2 absolute offset");
            require(t2.offset > 4294967296ULL, "tensor 2 offset exceeds 4 GiB");
            require(t2.bytes == 16, "tensor 2 size is 16 bytes");
            auto bytes2 = parsed.tensor_bytes(t2);
            require(std::ranges::equal(bytes2, std::as_bytes(std::span(pattern_4gib))),
                    "tensor 2 bytes readback bit-exact across > 4 GiB boundary");

            // Verify Tensor 3 (> 8 GiB)
            const auto& t3 = parsed.tensors()[2];
            require(t3.name == "tensor_above_8gib", "tensor 3 name");
            require(t3.offset == data_offset + offset_8gib, "tensor 3 absolute offset");
            require(t3.offset > 8589934592ULL, "tensor 3 offset exceeds 8 GiB");
            require(t3.bytes == 16, "tensor 3 size is 16 bytes");
            auto bytes3 = parsed.tensor_bytes(t3);
            require(std::ranges::equal(bytes3, std::as_bytes(std::span(pattern_8gib))),
                    "tensor 3 bytes readback bit-exact across > 8 GiB boundary");
        }

        std::cout << "PASS " << checks << " pack v2 64-bit uint64_t offset safety checks, no GPU/model execution\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
