#include "tokspur/ssd_checkpoint.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <string_view>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace tokspur;

static void test_storage_security() {
    std::filesystem::path test_dir = "/tmp/tokspur_test_sec_" + std::to_string(::getpid());
    std::filesystem::remove_all(test_dir);

    std::filesystem::create_directory(test_dir);
    assert(::chmod(test_dir.c_str(), 0755) == 0);
    { std::ofstream unrelated(test_dir / "unrelated.ckpt"); unrelated << "preserve"; }
    const auto legacy = test_dir / "tokspur-checkpoints-v2";
    std::filesystem::create_directory(legacy);
    { std::ofstream file(legacy / "old.ckpt"); file << "legacy fixture"; }

    SsdCheckpointManager mgr(test_dir, 64ull << 20, 1ull << 20);
    assert(mgr.enabled());
    assert(mgr.dir_fd() >= 0);

    // Verify directory permissions 0700
    struct stat st{};
    assert(::fstat(mgr.dir_fd(), &st) == 0);
    assert(S_ISDIR(st.st_mode));
    assert((st.st_mode & 0777) == 0700);
    assert(st.st_uid == ::geteuid());
    assert(::stat(test_dir.c_str(), &st) == 0 && (st.st_mode & 0777) == 0755);
    assert(mgr.occupied_bytes() == 0);
    std::string original;
    { std::ifstream unrelated(test_dir / "unrelated.ckpt"); unrelated >> original; }
    assert(original == "preserve");
    { std::ifstream file(legacy / "old.ckpt"); std::getline(file, original); }
    assert(original == "legacy fixture");
    assert(std::filesystem::is_directory(test_dir / "tokspur-checkpoints-v3"));

    // Path traversal rejection
    SlotHandshake handshake;
    CheckpointFileHeader hdr;
    CheckpointMetadata meta;
    auto read_dummy = [](size_t, std::span<std::byte>) { return true; };

    bool rejected_dotdot = false;
    try {
        mgr.spill_checkpoint("../escaped.ckpt", handshake, hdr, meta, read_dummy);
    } catch (const std::invalid_argument&) {
        rejected_dotdot = true;
    }
    assert(rejected_dotdot);

    bool rejected_slash = false;
    try {
        mgr.spill_checkpoint("sub/dir.ckpt", handshake, hdr, meta, read_dummy);
    } catch (const std::invalid_argument&) {
        rejected_slash = true;
    }
    assert(rejected_slash);

    std::filesystem::remove_all(test_dir);
}

static void test_staging_slab_and_budgets() {
    std::filesystem::path test_dir = "/tmp/tokspur_test_slab_" + std::to_string(::getpid());
    std::filesystem::remove_all(test_dir);

    const uint64_t budget = 64ull << 20;
    SsdCheckpointManager mgr(test_dir, budget, 1ull << 20);

    // Verify staging slab 64 MiB page alignment
    auto slab = mgr.staging_slab();
    assert(slab.size() == staging_slab_bytes);
    assert(reinterpret_cast<uintptr_t>(slab.data()) % 16384 == 0);

    // Verify budget
    assert(mgr.budget_bytes() == budget);
    assert(mgr.occupied_bytes() == 0);

    // Test admit_spill
    assert(mgr.admit_spill(10ull << 20));
    assert(!mgr.admit_spill(100ull << 20)); // Exceeds 64 MiB budget

    std::filesystem::remove_all(test_dir);
}

static void test_eviction_scoring_and_system_tools_anchor() {
    // Calibrated prefill formula check: T_hat_prefill(N) = 1.8e-4 * N + 4.2e-8 * N^2
    const double t4k = calibrated_prefill_time(4096);
    const double expected_t4k = 1.8e-4 * 4096.0 + 4.2e-8 * 4096.0 * 4096.0;
    assert(std::abs(t4k - expected_t4k) < 1e-6);

    // Normal eviction priority
    const double prio1 = compute_eviction_priority(10, 512, 10ull << 20, false);
    const double prio2 = compute_eviction_priority(10, 4096, 10ull << 20, false);
    // 4096 tokens should have higher prefill recompute cost, thus higher priority
    assert(prio2 > prio1);

    // System anchors retain a finite recomputation preference.
    const double prio_anchor = compute_eviction_priority(10, 512, 10ull << 20, true);
    assert(std::isfinite(prio_anchor));
    assert(prio_anchor > prio1 && prio_anchor < prio2);

    // Coalescing check
    std::filesystem::path test_dir = "/tmp/tokspur_test_coal_" + std::to_string(::getpid());
    std::filesystem::remove_all(test_dir);
    SsdCheckpointManager mgr(test_dir, 64ull << 20, 1ull << 20);

    assert(mgr.should_coalesce(5.1, 10));
    assert(mgr.should_coalesce(1.0, 300));
    assert(!mgr.should_coalesce(2.0, 50));

    std::filesystem::remove_all(test_dir);
}

static void test_spill_cancellation_between_chunks() {
    const std::filesystem::path test_dir = "/tmp/tokspur_test_cancel_" + std::to_string(::getpid());
    std::filesystem::remove_all(test_dir);
    SsdCheckpointManager manager(test_dir, 64ull << 20, 1ull << 20);
    SlotHandshake handshake;
    assert(handshake.reader_epoch.load() == 0);
    assert(!handshake.cancel_spill.load());

    CheckpointFileHeader header;
    header.prefix_tokens = 512;
    header.total_payload_bytes = 2 * checkpoint_chunk_bytes;
    CheckpointMetadata metadata;
    std::promise<void> copy_entered, finish_copy;
    auto entered = copy_entered.get_future();
    auto finish = finish_copy.get_future();
    size_t reads = 0;
    bool spilled = true;
    std::thread background([&] {
        spilled = manager.spill_checkpoint("cancel.ckpt", handshake, header, metadata,
            [&](size_t index, std::span<std::byte> target) {
                ++reads;
                assert(index == 0 && reads == 1);
                assert(handshake.reader_epoch.load() & 1);
                copy_entered.set_value();
                finish.wait();
                std::fill(target.begin(), target.end(), std::byte{0x42});
                return true;
            });
    });

    assert(entered.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    assert(handshake.reader_epoch.load() & 1);
    handshake.cancel_spill.store(true);
    finish_copy.set_value();
    background.join();
    assert(!spilled && reads == 1 && handshake.reader_epoch.load() == 2);
    assert(manager.occupied_bytes() == 0 && !manager.has_checkpoint("cancel.ckpt"));
    assert(std::filesystem::is_empty(test_dir / "tokspur-checkpoints-v3"));

    assert(!manager.spill_checkpoint("already-cancelled.ckpt", handshake, header, metadata,
        [&](size_t, std::span<std::byte>) { ++reads; return true; }));
    assert(reads == 1 && handshake.reader_epoch.load() == 2);
    assert(manager.occupied_bytes() == 0 && std::filesystem::is_empty(test_dir / "tokspur-checkpoints-v3"));
    std::filesystem::remove_all(test_dir);
}

static void test_spill_restore_and_sha256_verification() {
    std::filesystem::path test_dir = "/tmp/tokspur_test_sr_" + std::to_string(::getpid());
    std::filesystem::remove_all(test_dir);

    SsdCheckpointManager mgr(test_dir, 128ull << 20, 1ull << 20);

    SlotHandshake handshake;
    CheckpointFileHeader hdr;
    hdr.prefix_tokens = 512;
    hdr.total_payload_bytes = 32ull << 20; // 32 MiB = 2 chunks of 16 MiB
    hdr.is_system_anchor = 1;
    hdr.state_digest.fill(0x42);

    CheckpointMetadata meta;
    meta.rng = SamplingRng(12345);
    meta.controller_state.ewma_alpha1 = 0.48;
    meta.controller_state.current_mode = 0;
    meta.controller_state.consecutive_ordinary_steps = 3;
    meta.controller_state.controller_policy_digest = default_controller_policy_digest;
    meta.controller_state.controller_policy_digest[0] ^= 0x55;
    meta.carried_anchor = SampledPathToken{37, 1.25f, false};
    meta.pending_overflow = 2;
    meta.mtp_first = 1;
    meta.mtp_rows = 511;
    meta.logits.resize(256, 1.5f);

    std::vector<uint8_t> chunk0(16ull << 20, 0xAA);
    std::vector<uint8_t> chunk1(16ull << 20, 0xBB);

    auto read_chunk = [&](size_t idx, std::span<std::byte> target) {
        if (idx == 0) {
            std::memcpy(target.data(), chunk0.data(), chunk0.size());
            return true;
        } else if (idx == 1) {
            std::memcpy(target.data(), chunk1.data(), chunk1.size());
            return true;
        }
        return false;
    };

    bool spill_ok = mgr.spill_checkpoint("test_512.ckpt", handshake, hdr, meta, read_chunk);
    assert(spill_ok);
    assert(mgr.occupied_bytes() > 32ull << 20);

    // Verify listing
    auto entries = mgr.list_entries();
    assert(entries.size() == 1);
    assert(entries[0].filename == "test_512.ckpt");
    assert(entries[0].prefix_tokens == 512);
    assert(entries[0].is_system_anchor);
    assert(std::isfinite(entries[0].priority));

    // Restore checkpoint
    CheckpointFileHeader restored_hdr;
    CheckpointMetadata restored_meta;
    std::vector<uint8_t> restored_chunk0(16ull << 20);
    std::vector<uint8_t> restored_chunk1(16ull << 20);

    auto write_chunk = [&](size_t idx, std::span<const std::byte> source) {
        assert(restored_hdr.prefix_tokens == 512);
        assert(restored_hdr.total_payload_bytes == (32ull << 20));
        if (idx == 0) {
            std::memcpy(restored_chunk0.data(), source.data(), source.size());
            return true;
        } else if (idx == 1) {
            std::memcpy(restored_chunk1.data(), source.data(), source.size());
            return true;
        }
        return false;
    };

    bool restore_ok = mgr.restore_checkpoint("test_512.ckpt", restored_hdr, restored_meta, write_chunk);
    assert(restore_ok);
    assert(restored_hdr.prefix_tokens == 512);
    assert(restored_hdr.state_digest == hdr.state_digest);
    assert(restored_meta.rng.seed() == 12345);
    assert(std::abs(restored_meta.controller_state.ewma_alpha1 - 0.48) < 1e-6);
    assert(restored_meta.controller_state.consecutive_ordinary_steps == 3);
    assert(restored_meta.controller_state.controller_policy_digest == meta.controller_state.controller_policy_digest);
    assert(restored_hdr.controller_policy_digest == meta.controller_state.controller_policy_digest);
    const auto reset = restore_controller_policy(restored_meta.controller_state, default_controller_policy_digest, false);
    assert(reset.ewma_alpha1 == StageACostTable::alpha_0 && reset.consecutive_ordinary_steps == 0);
    assert(reset.controller_policy_digest == default_controller_policy_digest);
    assert(restored_meta.rng == meta.rng && restored_meta.logits == meta.logits);
    assert(restored_meta.carried_anchor && restored_meta.carried_anchor->id == 37 && restored_meta.carried_anchor->source_logit == 1.25f);
    assert(restored_meta.pending_overflow == 2);
    const auto measured_reset = restore_controller_policy(restored_meta.controller_state, restored_meta.controller_state.controller_policy_digest, true);
    assert(measured_reset.ewma_alpha1 == StageACostTable::alpha_0 && !measured_reset.measured && measured_reset.consecutive_ordinary_steps == 0);
    assert(restored_meta.rng == meta.rng && restored_meta.carried_anchor->id == meta.carried_anchor->id && restored_meta.pending_overflow == meta.pending_overflow);
    assert(restored_meta.mtp_rows == 511);
    assert(restored_meta.logits.size() == 256);
    assert(restored_meta.logits[0] == 1.5f);
    assert(restored_chunk0 == chunk0);
    assert(restored_chunk1 == chunk1);

    // Test inline SHA-256 corruption rejection
    // Corrupt one byte of chunk 0 in the checkpoint file
    const std::string file_path = (test_dir / "tokspur-checkpoints-v3" / "test_512.ckpt").string();
    FILE* fp = std::fopen(file_path.c_str(), "r+b");
    assert(fp != nullptr);
    std::fseek(fp, -1000, SEEK_END);
    uint8_t corrupted_byte = 0xFF;
    std::fwrite(&corrupted_byte, 1, 1, fp);
    std::fclose(fp);

    // Attempting restore on corrupt chunk must fail cleanly (reject torn/corrupt file)
    bool corrupt_restore_ok = mgr.restore_checkpoint("test_512.ckpt", restored_hdr, restored_meta, write_chunk);
    assert(!corrupt_restore_ok);
    assert(!std::filesystem::exists(file_path));

    std::filesystem::remove_all(test_dir);
}

static void test_execution_identity_binding_and_stale_eviction() {
    std::filesystem::path test_dir = "/tmp/tokspur_test_exec_id_" + std::to_string(::getpid());
    std::filesystem::remove_all(test_dir);

    std::array<uint8_t, 32> digest_a{};
    digest_a.fill(0xA1);
    std::array<uint8_t, 32> digest_b{};
    digest_b.fill(0xB2);

    std::vector<uint8_t> payload(16ull << 20, 0x5A);
    auto read_chunk = [&](size_t idx, std::span<std::byte> target) {
        if (idx != 0 || target.size() != payload.size()) return false;
        std::memcpy(target.data(), payload.data(), payload.size());
        return true;
    };

    {
        SsdCheckpointManager mgr_a(test_dir, 64ull << 20, 1ull << 20);
        mgr_a.set_execution_identity(digest_a);
        assert(mgr_a.execution_identity() == digest_a);

        SlotHandshake handshake;
        CheckpointFileHeader hdr;
        hdr.prefix_tokens = 256;
        hdr.total_payload_bytes = payload.size();
        hdr.is_system_anchor = 1; // Anchor under model A must not wedge budget under model B
        CheckpointMetadata meta;
        meta.logits.resize(64, 0.25f);

        assert(mgr_a.spill_checkpoint("pfx_rev_a.ckpt", handshake, hdr, meta, read_chunk));
        assert(mgr_a.list_entries().size() == 1);
    }

    // Restart with different model/execution identity: old checkpoint must not list or restore
    {
        SsdCheckpointManager mgr_b(test_dir, 24ull << 20, 1ull << 20);
        mgr_b.set_execution_identity(digest_b);
        assert(mgr_b.list_entries().empty());

        CheckpointFileHeader out_hdr{};
        CheckpointMetadata out_meta{};
        size_t write_calls = 0;
        const bool restored = mgr_b.restore_checkpoint(
            "pfx_rev_a.ckpt", out_hdr, out_meta,
            [&](size_t, std::span<const std::byte>) {
                ++write_calls;
                return true;
            });
        assert(!restored);
        assert(write_calls == 0);

        // Tight 24 MiB budget cannot hold both 16 MiB files, so stale model-A checkpoint must be evicted first.
        SlotHandshake handshake;
        CheckpointFileHeader hdr;
        hdr.prefix_tokens = 256;
        hdr.total_payload_bytes = payload.size();
        CheckpointMetadata meta;
        meta.logits.resize(64, 0.75f);

        assert(mgr_b.spill_checkpoint("pfx_rev_b.ckpt", handshake, hdr, meta, read_chunk));
        assert(!mgr_b.has_checkpoint("pfx_rev_a.ckpt"));
        assert(mgr_b.has_checkpoint("pfx_rev_b.ckpt"));
        assert(mgr_b.list_entries().size() == 1);
    }

    std::filesystem::remove_all(test_dir);
}

static void test_restore_atomicity_on_later_chunk_failures() {
    std::filesystem::path test_dir = "/tmp/tokspur_test_atomic_" + std::to_string(::getpid());
    std::filesystem::remove_all(test_dir);

    SsdCheckpointManager mgr(test_dir, 256ull << 20, 1ull << 20);
    SlotHandshake handshake;

    // 1. Two-chunk checkpoint (32 MiB <= 64 MiB slab): corrupt chunk 1 and verify chunk 0 is never published.
    std::vector<uint8_t> chunk0(16ull << 20, 0xAA);
    std::vector<uint8_t> chunk1(16ull << 20, 0xBB);
    auto read_two_chunks = [&](size_t idx, std::span<std::byte> target) {
        if (idx == 0) {
            std::memcpy(target.data(), chunk0.data(), chunk0.size());
            return true;
        }
        if (idx == 1) {
            std::memcpy(target.data(), chunk1.data(), chunk1.size());
            return true;
        }
        return false;
    };

    CheckpointFileHeader hdr{};
    hdr.prefix_tokens = 512;
    hdr.total_payload_bytes = 32ull << 20;
    CheckpointMetadata meta{};
    meta.logits.resize(128, 2.0f);

    assert(mgr.spill_checkpoint("two_chunk_corrupt.ckpt", handshake, hdr, meta, read_two_chunks));

    std::vector<uint8_t> live_chunk0(16ull << 20, 0x11);
    std::vector<uint8_t> live_chunk1(16ull << 20, 0x22);
    size_t write_calls = 0;
    auto write_to_live = [&](size_t idx, std::span<const std::byte> source) {
        ++write_calls;
        if (idx == 0) std::memcpy(live_chunk0.data(), source.data(), source.size());
        if (idx == 1) std::memcpy(live_chunk1.data(), source.data(), source.size());
        return true;
    };

    // Validation callback rejection must not invoke write_chunk_fn
    CheckpointFileHeader out_hdr{};
    CheckpointMetadata out_meta{};
    assert(!mgr.restore_checkpoint(
        "two_chunk_corrupt.ckpt", out_hdr, out_meta, write_to_live,
        [](const CheckpointFileHeader&, const CheckpointMetadata&) { return false; }));
    assert(write_calls == 0);
    assert(live_chunk0[0] == 0x11);

    // Corrupt a byte inside chunk 1 (near EOF) while chunk 0 remains valid
    {
        const std::string path = (test_dir / "tokspur-checkpoints-v3" / "two_chunk_corrupt.ckpt").string();
        FILE* fp = std::fopen(path.c_str(), "r+b");
        assert(fp != nullptr);
        assert(std::fseek(fp, -512, SEEK_END) == 0);
        const uint8_t bad = 0xFE;
        assert(std::fwrite(&bad, 1, 1, fp) == 1);
        std::fclose(fp);

        assert(!mgr.restore_checkpoint("two_chunk_corrupt.ckpt", out_hdr, out_meta, write_to_live));
        assert(write_calls == 0);
        assert(live_chunk0 == std::vector<uint8_t>(16ull << 20, 0x11));
        assert(live_chunk1 == std::vector<uint8_t>(16ull << 20, 0x22));
        assert(!std::filesystem::exists(path));
    }

    // 2. Truncated second chunk must fail before any write_chunk_fn call
    assert(mgr.spill_checkpoint("two_chunk_trunc.ckpt", handshake, hdr, meta, read_two_chunks));
    {
        const std::string path = (test_dir / "tokspur-checkpoints-v3" / "two_chunk_trunc.ckpt").string();
        struct stat st{};
        assert(::stat(path.c_str(), &st) == 0);
        assert(::truncate(path.c_str(), st.st_size - 4096) == 0);

        assert(!mgr.restore_checkpoint("two_chunk_trunc.ckpt", out_hdr, out_meta, write_to_live));
        assert(write_calls == 0);
        assert(live_chunk0[0] == 0x11);
    }

    // 3. Five-chunk checkpoint (80 MiB > 64 MiB staging slab): corrupt chunk 4 and verify 0 writes
    CheckpointFileHeader large_hdr{};
    large_hdr.prefix_tokens = 2048;
    large_hdr.total_payload_bytes = 80ull << 20;
    assert(mgr.spill_checkpoint(
        "five_chunk_corrupt.ckpt", handshake, large_hdr, meta,
        [](size_t idx, std::span<std::byte> target) {
            std::memset(target.data(), int(0x30 + idx), target.size());
            return true;
        }));
    {
        const std::string path = (test_dir / "tokspur-checkpoints-v3" / "five_chunk_corrupt.ckpt").string();
        FILE* fp = std::fopen(path.c_str(), "r+b");
        assert(fp != nullptr);
        assert(std::fseek(fp, -256, SEEK_END) == 0);
        const uint8_t bad = 0x99;
        assert(std::fwrite(&bad, 1, 1, fp) == 1);
        std::fclose(fp);

        assert(!mgr.restore_checkpoint("five_chunk_corrupt.ckpt", out_hdr, out_meta, write_to_live));
        assert(write_calls == 0);
        assert(!std::filesystem::exists(path));
    }

    std::filesystem::remove_all(test_dir);
}

static void test_free_space_floor_reservation_and_eviction() {
    const std::filesystem::path base_dir = std::filesystem::canonical(std::filesystem::temp_directory_path());
    std::filesystem::path test_dir = base_dir / ("tokspur_test_floor_" + std::to_string(::getpid()));
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directories(test_dir);

    struct statvfs vfs{};
    assert(::statvfs(test_dir.c_str(), &vfs) == 0);
    const uint64_t initial_free = uint64_t(vfs.f_bavail) * uint64_t(vfs.f_frsize);
    assert(initial_free > (64ull << 20));

    // Verify the default fstatvfs query path enforces an unreachable floor and admits normal headroom.
    {
        SsdCheckpointManager unreachable_floor_mgr(test_dir, 128ull << 20, std::numeric_limits<uint64_t>::max());
        assert(!unreachable_floor_mgr.admit_spill(1));

        SsdCheckpointManager normal_floor_mgr(test_dir, 128ull << 20, 1ull << 20);
        assert(normal_floor_mgr.admit_spill(16ull << 20));
    }

    // Set floor 8 MiB below deterministic simulated free space so a 16 MiB spill breaches floor_bytes and is refused.
    {
        const uint64_t simulated_free = 100ull << 20;
        const uint64_t tight_floor = simulated_free - (8ull << 20);
        SsdCheckpointManager mgr(test_dir, 128ull << 20, tight_floor);
        mgr.set_free_space_query([&](int fd, uint64_t& out_free) {
            assert(fd == mgr.dir_fd());
            out_free = simulated_free;
            return true;
        });
        assert(mgr.admit_spill(8ull << 20));
        assert(!mgr.admit_spill(16ull << 20));

        SlotHandshake handshake;
        CheckpointFileHeader hdr{};
        hdr.prefix_tokens = 128;
        hdr.total_payload_bytes = 16ull << 20;
        CheckpointMetadata meta{};
        const bool ok = mgr.spill_checkpoint(
            "blocked_by_floor.ckpt", handshake, hdr, meta,
            [](size_t, std::span<std::byte> target) {
                std::memset(target.data(), 0x44, target.size());
                return true;
            });
        assert(!ok);
        assert(!mgr.has_checkpoint("blocked_by_floor.ckpt"));

        mgr.set_free_space_query([](int, uint64_t&) { return false; });
        assert(!mgr.admit_spill(1));
    }

    // Spill low-priority and high-priority checkpoints under a 256 MiB budget, then verify floor-driven eviction removes only the low-priority checkpoint even when fstatvfs lags behind unlink reclamation.
    {
        SsdCheckpointManager setup_mgr(test_dir, 256ull << 20, 1ull << 20);
        SlotHandshake handshake;
        CheckpointMetadata meta{};

        CheckpointFileHeader low_hdr{};
        low_hdr.prefix_tokens = 128;
        low_hdr.total_payload_bytes = 24ull << 20;
        assert(setup_mgr.spill_checkpoint(
            "low_prio_floor.ckpt", handshake, low_hdr, meta,
            [](size_t, std::span<std::byte> target) {
                std::memset(target.data(), 0x55, target.size());
                return true;
            }));

        CheckpointFileHeader high_hdr{};
        high_hdr.prefix_tokens = 8192;
        high_hdr.total_payload_bytes = 24ull << 20;
        assert(setup_mgr.spill_checkpoint(
            "high_prio_floor.ckpt", handshake, high_hdr, meta,
            [](size_t, std::span<std::byte> target) {
                std::memset(target.data(), 0x56, target.size());
                return true;
            }));
    }

    {
        const uint64_t simulated_free = 100ull << 20;
        const uint64_t eviction_floor = simulated_free - (4ull << 20);
        SsdCheckpointManager mgr(test_dir, 256ull << 20, eviction_floor);
        size_t space_queries = 0;
        mgr.set_free_space_query([&](int fd, uint64_t& out_free) {
            assert(fd == mgr.dir_fd());
            ++space_queries;
            out_free = simulated_free;
            return true;
        });
        assert(mgr.has_checkpoint("low_prio_floor.ckpt"));
        assert(mgr.has_checkpoint("high_prio_floor.ckpt"));

        SlotHandshake handshake;
        CheckpointFileHeader hdr{};
        hdr.prefix_tokens = 512;
        hdr.total_payload_bytes = 16ull << 20;
        CheckpointMetadata meta{};
        const bool ok = mgr.spill_checkpoint(
            "admitted_after_floor_evict.ckpt", handshake, hdr, meta,
            [](size_t, std::span<std::byte> target) {
                std::memset(target.data(), 0x66, target.size());
                return true;
            });
        assert(ok);
        assert(space_queries >= 2);
        assert(!mgr.has_checkpoint("low_prio_floor.ckpt"));
        assert(mgr.has_checkpoint("high_prio_floor.ckpt"));
        assert(mgr.has_checkpoint("admitted_after_floor_evict.ckpt"));
    }

    std::filesystem::remove_all(test_dir);
}

static void test_metadata_corruption_precedes_publication() {
    const auto test_dir = std::filesystem::path("/tmp/tokspur_test_metadata_" + std::to_string(::getpid()));
    std::filesystem::remove_all(test_dir);
    SsdCheckpointManager mgr(test_dir, 64ull << 20, 0);
    SlotHandshake handshake;
    CheckpointFileHeader header{};
    header.prefix_tokens = 512;
    header.total_payload_bytes = 64;
    CheckpointMetadata metadata{};
    metadata.logits = {0.25f, 0.75f};
    const auto fill = [](size_t, std::span<std::byte> bytes) { std::fill(bytes.begin(), bytes.end(), std::byte{0x51}); return true; };
    const auto filename = "metadata.ckpt";
    for (const auto selected : {"seed", "words", "engine", "controller", "logits"}) {
        assert(mgr.spill_checkpoint(filename, handshake, header, metadata, fill));
        const auto file = test_dir / "tokspur-checkpoints-v3" / filename;
        std::fstream stream(file, std::ios::in | std::ios::out | std::ios::binary);
        assert(stream);
        const auto length_offset = sizeof(CheckpointFileHeader) + 32;
        uint32_t length = 0, engine_length = 0;
        stream.seekg(std::streamoff(length_offset));
        stream.read(reinterpret_cast<char*>(&length), sizeof(length));
        stream.seekg(std::streamoff(length_offset + sizeof(length) + 16));
        stream.read(reinterpret_cast<char*>(&engine_length), sizeof(engine_length));
        assert(stream && length > 20 + engine_length + 8);
        const auto field = std::string_view(selected);
        const size_t offset = field == "seed" ? 0 : field == "words" ? 8 : field == "engine" ? 20 : field == "controller" ? 20 + engine_length : length - 8;
        stream.seekg(std::streamoff(length_offset + sizeof(length) + offset));
        char byte = 0;
        stream.read(&byte, 1);
        assert(stream);
        byte = field == "engine" ? (byte == '9' ? '8' : char(byte + 1)) : char(static_cast<unsigned char>(byte) ^ 1u);
        stream.seekp(std::streamoff(length_offset + sizeof(length) + offset));
        stream.write(&byte, 1);
        stream.close();
        size_t writes = 0, validations = 0;
        CheckpointFileHeader restored{};
        CheckpointMetadata restored_metadata{};
        const bool ok = mgr.restore_checkpoint(filename, restored, restored_metadata,
            [&](size_t, std::span<const std::byte>) { ++writes; return true; },
            [&](const auto&, const auto&) { ++validations; return true; });
        assert(!ok && writes == 0 && validations == 0);
        assert(restored.prefix_tokens == 0 && restored_metadata.logits.empty());
    }
    std::filesystem::remove_all(test_dir);
}

static void test_restore_refreshes_persistent_eviction_age() {
    const auto parent = std::filesystem::path("/tmp/tokspur_test_lru_" + std::to_string(::getpid()));
    std::filesystem::remove_all(parent);
    SlotHandshake handshake;
    CheckpointFileHeader header{};
    header.prefix_tokens = 512; header.total_payload_bytes = 64;
    CheckpointMetadata metadata{};
    const auto fill = [](size_t, std::span<std::byte> bytes) { std::fill(bytes.begin(), bytes.end(), std::byte{0x42}); return true; };
    const auto age = [](const auto& manager, std::string_view name) {
        for (const auto& entry : manager.list_entries()) if (entry.filename == name) return entry.access_epoch;
        assert(false); return uint64_t(0);
    };
    uint64_t bytes = 0, newest = 0;
    {
        SsdCheckpointManager manager(parent, 1ull << 20, 0, "lru");
        assert(manager.spill_checkpoint("a.ckpt", handshake, header, metadata, fill));
        assert(manager.spill_checkpoint("b.ckpt", handshake, header, metadata, fill));
        bytes = std::filesystem::file_size(parent / "tokspur-checkpoints-v3" / "a.ckpt");
        assert(age(manager, "a.ckpt") < age(manager, "b.ckpt"));
        CheckpointFileHeader restored{}; CheckpointMetadata restored_metadata{};
        assert(manager.restore_checkpoint("a.ckpt", restored, restored_metadata, [](size_t, std::span<const std::byte>) { return true; }));
        newest = age(manager, "a.ckpt");
        assert(newest > age(manager, "b.ckpt") && restored.access_epoch == newest);
        assert(!manager.restore_checkpoint(
            "a.ckpt", restored, restored_metadata, [](size_t, std::span<const std::byte>) { return false; }));
        assert(manager.has_checkpoint("a.ckpt") && age(manager, "a.ckpt") == newest);
        assert(manager.occupied_bytes() == 2 * bytes);
        assert(manager.restore_checkpoint(
            "a.ckpt", restored, restored_metadata, [](size_t, std::span<const std::byte> bytes) {
                return bytes.size() == 64 &&
                       std::all_of(bytes.begin(), bytes.end(), [](std::byte byte) { return byte == std::byte{0x42}; });
            }));
        assert(age(manager, "a.ckpt") == restored.access_epoch && restored.access_epoch > newest);
        newest = restored.access_epoch;
        const auto alias = parent / "external-hard-link";
        assert(::link((parent / "tokspur-checkpoints-v3" / "a.ckpt").c_str(), alias.c_str()) == 0);
        size_t writes = 0;
        assert(!manager.restore_checkpoint("a.ckpt", restored, restored_metadata, [&](size_t, std::span<const std::byte>) { ++writes; return true; }));
        assert(writes == 0 && age(manager, "a.ckpt") == newest);
        assert(::unlink(alias.c_str()) == 0);
    }
    {
        SsdCheckpointManager manager(parent, bytes * 2, 0, "lru");
        assert(age(manager, "a.ckpt") == newest);
        assert(manager.spill_checkpoint("c.ckpt", handshake, header, metadata, fill));
        assert(manager.has_checkpoint("a.ckpt") && !manager.has_checkpoint("b.ckpt") && manager.has_checkpoint("c.ckpt"));
        assert(age(manager, "c.ckpt") > newest);
        newest = age(manager, "c.ckpt");
    }
    {
        SsdCheckpointManager manager(parent, bytes * 2, 0, "lru");
        assert(manager.spill_checkpoint("d.ckpt", handshake, header, metadata, fill));
        assert(!manager.has_checkpoint("a.ckpt") && manager.has_checkpoint("c.ckpt") && manager.has_checkpoint("d.ckpt"));
        assert(age(manager, "d.ckpt") > newest);
    }
    std::filesystem::remove_all(parent);
}

static std::filesystem::path restore_test_directory(std::string_view test) {
    auto name = (std::filesystem::temp_directory_path() / ("tokspur_" + std::string(test) + "_XXXXXX")).string();
    assert(::mkdtemp(name.data()) != nullptr);
    return name;
}

static void test_late_restore_refusal_preserves_valid_checkpoints(std::string_view policy) {
    const auto parent = restore_test_directory("late_restore");
    const auto checkpoints = parent / "tokspur-checkpoints-v3";
    const auto damaged = checkpoints / "damaged.ckpt";
    const auto healthy = checkpoints / "healthy.ckpt";
    const uint64_t large_payload = staging_slab_bytes + 4096;
    SsdCheckpointManager manager(parent, 128ull << 20, 1ull << 20, std::string(policy));
    SlotHandshake handshake;
    CheckpointFileHeader header{};
    CheckpointMetadata metadata{};
    const auto fill = [](size_t index, std::span<std::byte> bytes) {
        std::fill(bytes.begin(), bytes.end(), std::byte(0x40 + index));
        return true;
    };
    header.prefix_tokens = 1024;
    header.total_payload_bytes = large_payload;
    metadata.mtp_first = 9;
    assert(manager.spill_checkpoint("damaged.ckpt", handshake, header, metadata, fill));
    header.prefix_tokens = 512;
    header.total_payload_bytes = checkpoint_chunk_bytes;
    metadata.mtp_first = 1;
    assert(manager.spill_checkpoint("healthy.ckpt", handshake, header, metadata, fill));
    assert(manager.restore_checkpoint(
        "healthy.ckpt", header, metadata,
        [](size_t index, std::span<const std::byte> bytes) {
            return index == 0 && bytes.size() == checkpoint_chunk_bytes &&
                   std::all_of(bytes.begin(), bytes.end(), [](std::byte byte) { return byte == std::byte{0x40}; });
        }));
    const uint64_t healthy_bytes = std::filesystem::file_size(healthy);
    const uint64_t damaged_bytes = std::filesystem::file_size(damaged);
    const uint64_t shortened = damaged_bytes - large_payload + checkpoint_chunk_bytes;
    assert(healthy_bytes + damaged_bytes + (48ull << 20) > manager.budget_bytes());
    assert(healthy_bytes + shortened + (48ull << 20) <= manager.budget_bytes());

    const int fd = ::open(damaged.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    assert(fd >= 0);
    CheckpointFileHeader original{};
    assert(::pread(fd, &original, sizeof(original), 0) == ssize_t(sizeof(original)));
    size_t writes = 0;
    assert(!manager.restore_checkpoint(
        "damaged.ckpt", header, metadata,
        [&](size_t index, std::span<const std::byte> bytes) {
            assert(index == 0 && writes++ == 0 && bytes.size() == checkpoint_chunk_bytes);
            assert(metadata.mtp_first == 9);
            assert(::ftruncate(fd, off_t(shortened)) == 0);
            return true;
        }));
    assert(writes == 1);
    CheckpointFileHeader failed{};
    assert(::pread(fd, &failed, sizeof(failed), 0) == ssize_t(sizeof(failed)));
    assert(::close(fd) == 0);
    assert(failed.access_epoch == original.access_epoch);
    assert(manager.occupied_bytes() == healthy_bytes + shortened);
    for (int attempt = 0; attempt < 2; ++attempt) {
        const auto entries = manager.list_entries();
        assert(entries.size() == 1 && entries.front().filename == "healthy.ckpt");
        assert(!manager.has_checkpoint("damaged.ckpt"));
        assert(!manager.restore_checkpoint(
            "damaged.ckpt", header, metadata,
            [](size_t, std::span<const std::byte>) { assert(false); return false; }));
        assert(manager.occupied_bytes() == healthy_bytes + shortened);
    }
    assert(manager.admit_spill(48ull << 20));
    assert(std::filesystem::exists(healthy) && std::filesystem::exists(damaged));
    assert(manager.admit_spill(105ull << 20));
    assert(std::filesystem::exists(healthy) && !std::filesystem::exists(damaged));
    assert(manager.occupied_bytes() == healthy_bytes);
    std::filesystem::remove_all(parent);
}

static void test_corrupt_restore_preserves_substituted_checkpoint() {
    const auto parent = restore_test_directory("restore_replacement");
    const auto checkpoints = parent / "tokspur-checkpoints-v3";
    const auto original_path = checkpoints / "candidate.ckpt";
    const auto replacement_path = checkpoints / "replacement.ckpt";
    const uint64_t payload = staging_slab_bytes + 4096;
    SsdCheckpointManager manager(parent, 128ull << 20, 1ull << 20);
    SlotHandshake handshake;
    CheckpointFileHeader header{};
    CheckpointMetadata metadata{};
    header.prefix_tokens = 1024;
    header.total_payload_bytes = payload;
    assert(manager.spill_checkpoint("candidate.ckpt", handshake, header, metadata,
        [](size_t, std::span<std::byte> bytes) {
            std::fill(bytes.begin(), bytes.end(), std::byte{0x21});
            return true;
        }));
    header.total_payload_bytes = 64;
    assert(manager.spill_checkpoint("replacement.ckpt", handshake, header, metadata,
        [](size_t, std::span<std::byte> bytes) {
            std::fill(bytes.begin(), bytes.end(), std::byte{0x53});
            return true;
        }));
    const uint64_t replacement_bytes = std::filesystem::file_size(replacement_path);
    const off_t second_chunk = off_t(std::filesystem::file_size(original_path) - payload + checkpoint_chunk_bytes);
    const int fd = ::open(original_path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    assert(fd >= 0);
    size_t writes = 0;
    assert(!manager.restore_checkpoint("candidate.ckpt", header, metadata,
        [&](size_t index, std::span<const std::byte>) {
            assert(index == 0 && writes++ == 0);
            std::filesystem::rename(original_path, parent / "original");
            std::filesystem::rename(replacement_path, original_path);
            const std::byte corrupt{0x44};
            assert(::pwrite(fd, &corrupt, 1, second_chunk) == 1);
            return true;
        }));
    assert(::close(fd) == 0 && writes == 1);
    assert(std::filesystem::file_size(original_path) == replacement_bytes);
    assert(std::filesystem::exists(parent / "original"));
    assert(manager.occupied_bytes() == replacement_bytes);
    assert(manager.restore_checkpoint("candidate.ckpt", header, metadata,
        [](size_t index, std::span<const std::byte> bytes) {
            return index == 0 && bytes.size() == 64 &&
                   std::all_of(bytes.begin(), bytes.end(), [](std::byte byte) { return byte == std::byte{0x53}; });
        }));
    std::filesystem::remove_all(parent);
}

static void test_eviction_preserves_substituted_checkpoint(std::string_view policy) {
    const auto parent = restore_test_directory("eviction_replacement");
    const auto checkpoints = parent / "tokspur-checkpoints-v3";
    const auto original_path = checkpoints / "second.ckpt";
    const auto held_path = parent / "held";
    SsdCheckpointManager manager(parent, 1ull << 20, 0, std::string(policy));
    SlotHandshake handshake;
    CheckpointFileHeader header{};
    header.prefix_tokens = 512;
    header.total_payload_bytes = 64;
    CheckpointMetadata metadata{};
    const auto fill = [](size_t, std::span<std::byte> bytes) {
        std::fill(bytes.begin(), bytes.end(), std::byte{0x63});
        return true;
    };
    for (const auto* name : {"first.ckpt", "second.ckpt", "replacement.ckpt"}) {
        assert(manager.spill_checkpoint(name, handshake, header, metadata, fill));
    }
    const auto replacement = checkpoints / "replacement.ckpt";
    const auto replacement_bytes = std::filesystem::file_size(replacement);
    std::filesystem::rename(replacement, held_path);

    size_t queries = 0;
    manager.set_free_space_query([&](int, uint64_t& out_free) {
        out_free = 1ull << 30;
        if (++queries == 2) {
            assert(!std::filesystem::exists(checkpoints / "first.ckpt"));
            std::filesystem::rename(original_path, parent / "original");
            std::filesystem::rename(held_path, original_path);
        }
        return true;
    });
    const bool admitted = manager.admit_spill(manager.budget_bytes());
    assert(std::filesystem::exists(original_path));
    assert(!admitted);
    assert(queries == 2 && manager.occupied_bytes() == replacement_bytes);
    assert(std::filesystem::exists(parent / "original"));
    assert(manager.restore_checkpoint("second.ckpt", header, metadata,
        [](size_t index, std::span<const std::byte> bytes) {
            return index == 0 && bytes.size() == 64 &&
                   std::all_of(bytes.begin(), bytes.end(), [](std::byte byte) { return byte == std::byte{0x63}; });
        }));
    std::filesystem::remove_all(parent);
}

int main() {
    test_late_restore_refusal_preserves_valid_checkpoints("lru");
    test_late_restore_refusal_preserves_valid_checkpoints("recompute-cost");
    test_corrupt_restore_preserves_substituted_checkpoint();
    test_eviction_preserves_substituted_checkpoint("lru");
    test_eviction_preserves_substituted_checkpoint("recompute-cost");
    test_restore_refreshes_persistent_eviction_age();
    test_metadata_corruption_precedes_publication();
    std::cout << "Running SSD Checkpoint unit tests..." << std::endl;
    test_storage_security();
    test_staging_slab_and_budgets();
    test_eviction_scoring_and_system_tools_anchor();
    test_spill_cancellation_between_chunks();
    test_spill_restore_and_sha256_verification();
    test_execution_identity_binding_and_stale_eviction();
    test_restore_atomicity_on_later_chunk_failures();
    test_free_space_floor_reservation_and_eviction();
    std::cout << "All SSD Checkpoint unit tests passed successfully." << std::endl;
    return 0;
}
