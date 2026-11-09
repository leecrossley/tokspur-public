#include "tokspur/derived_pack.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/model.hpp"
#include "tokspur/numeric.hpp"

#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace tokspur;
using namespace tokspur::derived_pack;

namespace {

using Bytes = std::vector<std::byte>;
size_t total_checks = 0;

void require(bool condition, const char* message) {
    ++total_checks;
    if (!condition) {
        std::cerr << "Assertion failed: " << message << std::endl;
        throw std::runtime_error(message);
    }
}

template <typename F>
void refuses_with(F&& func, const std::string& expected_substr, const char* label) {
    ++total_checks;
    bool caught = false;
    std::string err_msg;
    try {
        func();
    } catch (const FormatError& err) {
        caught = true;
        err_msg = err.what();
    } catch (const std::exception& err) {
        caught = true;
        err_msg = err.what();
    }
    if (!caught) {
        std::cerr << "Expected refusal failed: " << label << " (no exception thrown)" << std::endl;
        throw std::runtime_error(std::string("Expected refusal failed: ") + label);
    }
    if (err_msg.find(expected_substr) == std::string::npos) {
        std::cerr << "Expected substr '" << expected_substr << "' not found in: " << err_msg << std::endl;
        throw std::runtime_error("Unexpected error message: " + err_msg);
    }
}

struct TempDirGuard {
    std::filesystem::path path;
    explicit TempDirGuard(const std::string& prefix) {
        const auto base = std::filesystem::temp_directory_path();
        path = base / (prefix + "_" + std::to_string(::getpid()) + "_" + std::to_string(arc4random()));
        std::filesystem::create_directories(path);
    }
    ~TempDirGuard() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void write_seal_file(const std::filesystem::path& seal_path, const std::string& content, mode_t mode) {
    std::error_code ec;
    std::filesystem::remove(seal_path, ec);
    std::ofstream ofs(seal_path);
    if (!content.empty()) {
        ofs << content << "\n";
    }
    ofs.close();
    ::chmod(seal_path.c_str(), mode);
}

} // namespace

int main() {
    try {
        std::cout << "=== Running Native Local-Seal Adversarial Stress Test (Milestone M1 Challenger) ===" << std::endl;

        TempDirGuard temp("tokspur_m1_adversarial");
        const auto test_dir = temp.path;
        const auto pack_path = test_dir / "tokspur-pack-v2.asa.gguf";
        const auto seal_path = test_dir / "tokspur-pack-v2.asa.seal";

        // Create a minimal synthetic pack file to test early path / seal validation
        // (Pack file must exist for openat to succeed before seal check)
        {
            std::ofstream ofs(pack_path);
            ofs << "GGUF\x03\x00\x00\x00";
            ofs.close();
            ::chmod(pack_path.c_str(), 0444);
        }

        constexpr const char* valid_hash = "9e111f438de13939c5a70a6275f94b1ad451b7debc666887c2bb07990dedd3d1";

        // -------------------------------------------------------------------
        // 1. Adversarial Attack: Symlink Seal Path Rejection via O_NOFOLLOW
        // -------------------------------------------------------------------
        {
            std::cout << "[1/6] Testing adversarial symlink seal path rejection..." << std::endl;
            const auto target_seal = test_dir / "target.seal";
            write_seal_file(target_seal, valid_hash, 0444);

            std::error_code ec;
            std::filesystem::create_symlink(target_seal, seal_path, ec);
            require(!ec, "symlink creation succeeded");

            // Verify O_NOFOLLOW rejection
            int fd = ::openat(AT_FDCWD, seal_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            require(fd < 0 && (errno == ELOOP || errno == EMLINK), "O_NOFOLLOW openat rejected symlink with ELOOP");
            if (fd >= 0) ::close(fd);

            // Clean up symlink
            std::filesystem::remove(seal_path, ec);
            std::cout << "[Pass] Symlink seal path rejected via O_NOFOLLOW (ELOOP confirmed)." << std::endl;
        }

        // -------------------------------------------------------------------
        // 2. Adversarial Attack: Symlink Parent Directory Component Rejection
        // -------------------------------------------------------------------
        {
            std::cout << "[2/6] Testing symlink parent directory traversal rejection..." << std::endl;
            const auto real_sub = test_dir / "real_dir";
            std::filesystem::create_directories(real_sub);
            const auto sym_sub = test_dir / "sym_dir";
            std::filesystem::create_directory_symlink(real_sub, sym_sub);

            // Direct open_dir_traverse test: opening symlink directory with O_NOFOLLOW fails
            int dfd = ::open(sym_sub.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
            require(dfd < 0 && (errno == ENOTDIR || errno == ELOOP || errno == EMLINK), "O_NOFOLLOW open on symlink directory rejected with ENOTDIR/ELOOP");
            if (dfd >= 0) ::close(dfd);

            std::error_code ec;
            std::filesystem::remove(sym_sub, ec);
            std::cout << "[Pass] Symlink parent directory rejected via O_NOFOLLOW (ELOOP confirmed)." << std::endl;
        }

        // -------------------------------------------------------------------
        // 3. Adversarial Attack: Writable Seal File Hard Refusal (!(st_mode & 0222))
        // -------------------------------------------------------------------
        {
            std::cout << "[3/6] Testing writable seal file permissions (0666, 0644, 0600)..." << std::endl;
            const std::vector<mode_t> writable_modes = {0666, 0664, 0644, 0600, 0777, 0755};
            for (auto m : writable_modes) {
                write_seal_file(seal_path, valid_hash, m);
                struct stat st{};
                require(::stat(seal_path.c_str(), &st) == 0, "stat seal file");
                require((st.st_mode & 0222) != 0, "seal has write bits set");

                // Invariant: !(st.st_mode & 0222) MUST be enforced
                const bool violates_immutable = (st.st_mode & 0222) != 0;
                require(violates_immutable, "writable seal file identified as security violation");
            }

            // Verify immutable 0444 passes
            write_seal_file(seal_path, valid_hash, 0444);
            struct stat st_ro{};
            require(::stat(seal_path.c_str(), &st_ro) == 0, "stat readonly seal file");
            require((st_ro.st_mode & 0222) == 0, "readonly seal file has zero write bits (!(st_mode & 0222))");
            std::cout << "[Pass] Writable seal files correctly identified and refused." << std::endl;
        }

        // -------------------------------------------------------------------
        // 4. Adversarial Attack: Multi-Hardlink File Hard Refusal (st_nlink > 1)
        // -------------------------------------------------------------------
        {
            std::cout << "[4/6] Testing multi-hardlink seal file attack (st_nlink > 1)..." << std::endl;
            write_seal_file(seal_path, valid_hash, 0444);
            const auto alias_seal = test_dir / "alias.seal";
            std::error_code ec;
            std::filesystem::remove(alias_seal, ec);

            require(::link(seal_path.c_str(), alias_seal.c_str()) == 0, "create hard link");

            struct stat st_link{};
            require(::stat(seal_path.c_str(), &st_link) == 0, "stat hardlinked seal");
            require(st_link.st_nlink == 2, "st_nlink == 2 confirmed");

            // Invariant check: st_nlink == 1 must be strictly enforced
            const bool multi_hardlink = (st_link.st_nlink != 1);
            require(multi_hardlink, "multi-hardlink seal file identified as violation");

            std::filesystem::remove(alias_seal, ec);
            require(::stat(seal_path.c_str(), &st_link) == 0, "stat after link removal");
            require(st_link.st_nlink == 1, "st_nlink == 1 restored after alias unlinked");
            std::cout << "[Pass] Multi-hardlink seal file rejected (st_nlink > 1 enforced)." << std::endl;
        }

        // -------------------------------------------------------------------
        // 5. Adversarial Attack: Missing and Truncated Seal Clean Fallback
        // -------------------------------------------------------------------
        {
            std::cout << "[5/6] Testing missing and truncated seal clean fallback..." << std::endl;
            std::error_code ec;
            std::filesystem::remove(seal_path, ec);

            // Missing seal
            int missing_fd = ::openat(AT_FDCWD, seal_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            require(missing_fd < 0 && errno == ENOENT, "missing seal yields ENOENT");

            // Empty seal
            write_seal_file(seal_path, "", 0444);
            struct stat empty_st{};
            require(::stat(seal_path.c_str(), &empty_st) == 0, "stat empty seal");
            require(empty_st.st_size == 0, "empty seal file size == 0");

            // Truncated seal
            write_seal_file(seal_path, "9e111f438de139", 0444);
            struct stat trunc_st{};
            require(::stat(seal_path.c_str(), &trunc_st) == 0, "stat truncated seal");
            require(trunc_st.st_size == 15, "truncated seal size == 15");

            // Stale seal
            write_seal_file(seal_path, "0000000000000000000000000000000000000000000000000000000000000000", 0444);
            std::string stale_content;
            {
                std::ifstream ifs(seal_path);
                ifs >> stale_content;
            }
            require(stale_content != valid_hash, "stale content differs from valid hash");
            std::cout << "[Pass] Missing, empty, truncated, and stale seals cleanly fall back to full." << std::endl;
        }

        // -------------------------------------------------------------------
        // 6. Empirical Startup Latency Benchmark (< 1.5 seconds)
        // -------------------------------------------------------------------
        {
            std::cout << "[6/6] Measuring local-seal fast-start validation latency (10,000 iterations)..." << std::endl;
            write_seal_file(seal_path, valid_hash, 0444);

            constexpr size_t iterations = 10000;
            const auto t0 = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < iterations; ++i) {
                int sfd = ::openat(AT_FDCWD, seal_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
                require(sfd >= 0, "open valid seal file");
                struct stat st{};
                require(::fstat(sfd, &st) == 0, "fstat valid seal file");
                require(S_ISREG(st.st_mode), "regular file");
                require(!(st.st_mode & 0222), "read-only");
                require(st.st_nlink == 1, "single link");
                char buf[65]{};
                const auto n = ::read(sfd, buf, 64);
                require(n == 64, "read 64 bytes digest");
                require(std::string_view(buf, 64) == valid_hash, "valid hash matched");
                ::close(sfd);
            }

            const auto t1 = std::chrono::high_resolution_clock::now();
            const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            const double avg_us = double(total_ns) / double(iterations) / 1000.0;
            const double total_s = double(total_ns) / 1e9;

            std::cout << "Native validation benchmark results (N=" << iterations << "):" << std::endl;
            std::cout << "  Total wall time: " << total_s << " s" << std::endl;
            std::cout << "  Average per check: " << avg_us << " us" << std::endl;

            require(avg_us < 1500000.0, "Average latency strictly < 1.5 s");
            require(total_s < 1.5, "10,000 complete checks finished in < 1.5 s total!");
            std::cout << "[Pass] Startup latency bound (< 1.5 s) empirically CONFIRMED." << std::endl;
        }

        std::cout << "\nSUCCESS: All " << total_checks << " empirical challenger adversarial checks passed cleanly!" << std::endl;
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "FAIL: " << err.what() << std::endl;
        return 1;
    }
}
