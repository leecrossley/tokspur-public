#include "grouped_mm_fixture.hpp"
#include "tokspur/derived_pack.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/grouped_mm_f32.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/planar_pack_v3.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <vector>

using namespace tokspur;
using namespace tokspur::planar_v3;

namespace {

using Bytes = std::vector<std::byte>;
size_t total_checks = 0;

void need(bool ok, const char* why) {
    ++total_checks;
    if (!ok) throw FormatError(why);
}

template <typename F>
void refuses(F&& fn, const char* why, std::string_view substr = {}) {
    ++total_checks;
    bool caught = false;
    try {
        fn();
    } catch (const FormatError& e) {
        caught = substr.empty() || std::string_view(e.what()).find(substr) != std::string_view::npos;
    }
    if (!caught) throw FormatError(why);
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

void push_u(Bytes& out, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        out.push_back(std::byte((value >> (8 * i)) & 0xff));
    }
}

void push_str(Bytes& out, std::string_view value) {
    push_u(out, value.size(), 8);
    for (unsigned char c : value) out.push_back(std::byte(c));
}

void push_meta_str(Bytes& out, std::string_view key, std::string_view value) {
    push_str(out, key);
    push_u(out, 8, 4);
    push_str(out, value);
}

void push_meta_u32(Bytes& out, std::string_view key, uint32_t value) {
    push_str(out, key);
    push_u(out, 4, 4);
    push_u(out, value, 4);
}

struct SyntheticTensorSpec {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> dimensions;
    Bytes payload;
};

void write_file_bytes(const std::filesystem::path& path, const Bytes& bytes, mode_t mode = 0600) {
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, mode);
    if (fd < 0) throw FormatError("failed to create synthetic GGUF file");
    for (size_t at = 0; at < bytes.size();) {
        const auto n = ::write(fd, bytes.data() + at, bytes.size() - at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            ::close(fd);
            throw FormatError("failed to write synthetic GGUF file");
        }
        at += size_t(n);
    }
    ::close(fd);
}

void write_gguf_shard(const std::filesystem::path& path, std::string_view architecture, const std::vector<SyntheticTensorSpec>& specs) {
    constexpr uint64_t alignment = 32;
    Bytes header;
    push_u(header, 0x46554747, 4);
    push_u(header, 3, 4);
    push_u(header, specs.size(), 8);
    push_u(header, 2, 8);
    push_meta_str(header, "general.architecture", architecture);
    push_meta_u32(header, "general.alignment", uint32_t(alignment));

    std::vector<uint64_t> offsets;
    uint64_t rel = 0;
    for (const auto& s : specs) {
        rel = align_up(rel, alignment);
        offsets.push_back(rel);
        rel += s.payload.size();
    }
    for (size_t i = 0; i < specs.size(); ++i) {
        push_str(header, specs[i].name);
        push_u(header, specs[i].dimensions.size(), 4);
        for (uint64_t d : specs[i].dimensions) push_u(header, d, 8);
        push_u(header, specs[i].type, 4);
        push_u(header, offsets[i], 8);
    }
    const uint64_t data_offset = align_up(header.size(), alignment);
    header.resize(size_t(data_offset), std::byte(0));
    Bytes file = std::move(header);
    file.resize(size_t(data_offset + rel), std::byte(0));
    for (size_t i = 0; i < specs.size(); ++i) {
        std::memcpy(file.data() + size_t(data_offset + offsets[i]), specs[i].payload.data(), specs[i].payload.size());
    }
    write_file_bytes(path, file);
}

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto base = std::filesystem::canonical(std::filesystem::temp_directory_path());
        path = base / ("tokspur_test_planar_pack_v3_" + std::to_string(::getpid()) + "_" + std::to_string(arc4random()));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path, std::filesystem::directory_options::skip_permission_denied)) {
            if (!entry.is_symlink()) ::chmod(entry.path().c_str(), entry.is_directory() ? 0700 : 0600);
        }
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

struct Arena {
    MetalGraph& g;
    std::vector<GpuView> views;
    ~Arena() {
        try { g.abort(); } catch (...) {}
        for (auto v : views) g.release(v);
    }
    GpuView upload(std::span<const std::byte> data, bool immutable = true) {
        auto v = g.upload(data, "planar v3 test fixture", immutable);
        views.push_back(v);
        return v;
    }
    GpuView floats(std::span<const float> data, bool immutable = true) {
        return upload(std::as_bytes(data), immutable);
    }
    GpuView zeros(uint64_t bytes) {
        auto v = g.zeros(bytes, "planar v3 test scratch");
        views.push_back(v);
        return v;
    }
};

float silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

double nmse_f32(std::span<const float> got, std::span<const float> ref) {
    need(got.size() == ref.size() && !got.empty(), "NMSE span mismatch");
    double err = 0.0, sig = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        need(std::isfinite(got[i]) && std::isfinite(ref[i]), "nonfinite output in planar v3 test");
        const double d = double(got[i]) - double(ref[i]);
        err += d * d;
        sig += double(ref[i]) * double(ref[i]);
    }
    need(sig > 0.0, "zero reference signal in planar v3 test");
    return err / sig;
}

void test_sm16_bijection_and_range_refusal() {
    uint32_t rng = 0x12345678u;
    for (uint32_t iter = 0; iter < 65536; ++iter) {
        std::array<std::byte, 12> src{};
        for (size_t b = 0; b < 12; ++b) {
            rng = rng * 1664525u + 1013904223u;
            src[b] = std::byte((rng >> 19) & 0xffu);
        }
        std::array<std::byte, 16> sm{};
        pack_sm16(src, sm);
        for (size_t j = 0; j < 8; ++j) {
            const uint8_t sc = std::to_integer<uint8_t>(sm[j]);
            const uint8_t mn = std::to_integer<uint8_t>(sm[8 + j]);
            need(sc <= 63u && mn <= 63u, "pack_sm16 produced factor outside 0..63");
            uint8_t ref_sc = 0, ref_mn = 0;
            if (j < 4) {
                ref_sc = std::to_integer<uint8_t>(src[j]) & 63u;
                ref_mn = std::to_integer<uint8_t>(src[j + 4]) & 63u;
            } else {
                const uint8_t packed = std::to_integer<uint8_t>(src[j + 4]);
                ref_sc = uint8_t((packed & 15u) | ((std::to_integer<uint8_t>(src[j - 4]) >> 6) << 4));
                ref_mn = uint8_t((packed >> 4) | ((std::to_integer<uint8_t>(src[j]) >> 6) << 4));
            }
            need(sc == ref_sc && mn == ref_mn, "pack_sm16 disagrees with GGUF scale_min_k4");
        }
        std::array<std::byte, 12> roundtrip{};
        unpack_sm16(sm, roundtrip);
        need(roundtrip == src, "unpack_sm16 failed byte-exact roundtrip");
    }

    std::array<std::byte, 16> valid{};
    std::array<std::byte, 12> out{};
    for (size_t i = 0; i < 16; ++i) {
        for (uint8_t bad_val : {uint8_t(64), uint8_t(127), uint8_t(255)}) {
            auto bad_sm = valid;
            bad_sm[i] = std::byte(bad_val);
            refuses([&] { unpack_sm16(bad_sm, out); }, "unpack_sm16 must refuse factor >= 64");
        }
    }
    std::cout << "PASS planar_v3 sm16 bijection and range refusal\n";
}

void test_cpu_pack_unpack_and_exact_operands() {
    struct Case {
        uint32_t type;
        uint32_t cols;
        uint32_t rows;
        uint32_t experts;
    };
    const std::array<Case, 4> cases{{
        {12u, 512u, 64u, 4u},
        {13u, 512u, 64u, 4u},
        {8u, 512u, 64u, 4u},
        {7u, 320u, 64u, 4u},
    }};
    for (const auto c : cases) {
        const auto spec = make_expert_spec(c.type, c.cols, c.rows);
        const auto raw = grouped_fixture::weights({c.type, c.cols, c.rows, c.experts, 1, 1, true, PrefillWeightLayout::gguf});
        std::vector<float> ref_all(uint64_t(c.cols) * c.rows * c.experts);
        tokspur::dequantize(c.type, raw, ref_all);

        for (uint32_t e = 0; e < c.experts; ++e) {
            const auto src_exp = std::span<const std::byte>(raw).subspan(e * spec.source_expert_bytes, spec.source_expert_bytes);
            Bytes packed(spec.planar_expert_bytes);
            pack_expert_planar_v3(spec, src_exp, packed);

            Bytes unpacked(spec.source_expert_bytes);
            unpack_expert_planar_v3(spec, packed, unpacked);
            need(std::memcmp(src_exp.data(), unpacked.data(), spec.source_expert_bytes) == 0,
                 "unpack_expert_planar_v3 did not reproduce exact GGUF bytes");

            const auto ref_exp = std::span<const float>(ref_all).subspan(uint64_t(e) * c.rows * c.cols, uint64_t(c.rows) * c.cols);
            std::vector<float> got_f64(ref_exp.size()), got_f32(ref_exp.size());
            dequantize_expert_planar_v3(spec, packed, got_f64);
            dequantize_expert_planar_v3_f32(spec, packed, got_f32);

            float max_diff_f64 = 0.0f, max_diff_f32 = 0.0f;
            for (size_t i = 0; i < ref_exp.size(); ++i) {
                max_diff_f64 = std::max(max_diff_f64, std::fabs(got_f64[i] - ref_exp[i]));
                max_diff_f32 = std::max(max_diff_f32, std::fabs(got_f32[i] - ref_exp[i]));
                need(got_f64[i] == ref_exp[i] && got_f32[i] == ref_exp[i],
                     "planar v3 dequantised operand differed from GGUF FP32 reference");
            }
            need(max_diff_f64 == 0.0f && max_diff_f32 == 0.0f, "nonzero max_abs_diff in planar v3 CPU operand test");
        }
    }
    std::cout << "PASS planar_v3 CPU pack/unpack and zero-diff FP32 operands (Q4_K, Q5_K, Q8_0, Q5_1)\n";
}

void test_pack_v3_builder_adoption_and_tamper() {
    TempDir dir;
    const auto src_path = dir.path / "source.gguf";
    const auto mtp_path = dir.path / "mtp.gguf";
    const auto pack_path = dir.path / "tokspur-pack-v3-synth.asa.gguf";
    const auto blk0_path = dir.path / "tokspur-pack-v3-blk0.asa.gguf";

    const auto q4k_gate = grouped_fixture::weights({12u, 256u, 32u, 4u, 1u, 1u, true});
    const auto q4k_up = grouped_fixture::weights({12u, 256u, 32u, 4u, 1u, 1u, true});
    const auto q51_down = grouped_fixture::weights({7u, 32u, 256u, 4u, 1u, 1u, true});
    const auto q5k_gate = grouped_fixture::weights({13u, 256u, 32u, 4u, 1u, 1u, true});
    const auto q5k_up = grouped_fixture::weights({13u, 256u, 32u, 4u, 1u, 1u, true});
    const auto q8_down = grouped_fixture::weights({8u, 32u, 256u, 4u, 1u, 1u, true});
    const auto dense_q8 = grouped_fixture::weights({8u, 512u, 64u, 1u, 1u, 1u, true});
    std::vector<float> norm_vals(256, 1.0f);
    Bytes norm_bytes(norm_vals.size() * 4);
    std::memcpy(norm_bytes.data(), norm_vals.data(), norm_bytes.size());

    const std::vector<SyntheticTensorSpec> src_specs{
        {"blk.0.attn_norm.weight", 0u, {256u}, norm_bytes},
        {"blk.0.attn_qkv.weight", 8u, {512u, 64u}, dense_q8},
        {"blk.0.ffn_down_exps.weight", 7u, {32u, 256u, 4u}, q51_down},
        {"blk.0.ffn_gate_exps.weight", 12u, {256u, 32u, 4u}, q4k_gate},
        {"blk.0.ffn_up_exps.weight", 12u, {256u, 32u, 4u}, q4k_up},
        {"blk.2.ffn_down_exps.weight", 8u, {32u, 256u, 4u}, q8_down},
        {"blk.2.ffn_gate_exps.weight", 13u, {256u, 32u, 4u}, q5k_gate},
        {"blk.2.ffn_up_exps.weight", 13u, {256u, 32u, 4u}, q5k_up},
    };
    const std::vector<SyntheticTensorSpec> mtp_specs{
        {"blk.48.ffn_down_exps.weight", 8u, {32u, 256u, 4u}, q8_down},
        {"blk.48.ffn_gate_exps.weight", 8u, {256u, 32u, 4u}, q8_down},
        {"blk.48.ffn_up_exps.weight", 8u, {256u, 32u, 4u}, q8_down},
    };
    write_gguf_shard(src_path, "qwen35moe", src_specs);
    write_gguf_shard(mtp_path, "qwen35moe", mtp_specs);

    const std::vector<std::filesystem::path> src_paths{src_path};
    const std::vector<std::filesystem::path> mtp_paths{mtp_path};
    const auto source = std::make_shared<const ModelFiles>(src_paths);
    const auto mtp = std::make_shared<const ModelFiles>(mtp_paths);

    const auto budget = measure_pack_v3_budget(*source, mtp.get(), dir.path, {}, 2u, 1024ull * 1024ull);
    need(budget.ram_ceiling_satisfied && budget.reserve_satisfied, "synthetic pack v3 budget check failed");
    need(budget.original_12b_scale_header_bytes * 4 == budget.expanded_16b_scale_header_bytes * 3,
         "12B -> 16B scale header ratio mismatch");

    const auto built = build_derived_pack_v3(source, mtp, pack_path, 2u, 1024ull * 1024ull);
    need(built.tensors.size() == 11, "unexpected synthetic pack v3 tensor count");

    // Verify zero sidecar footer bytes: file size equals header + last aligned tensor end.
    const auto& last_t = built.tensors.back();
    need(built.file_bytes == built.data_offset + last_t.offset + last_t.bytes,
         "pack v3 contains unexpected trailing footer bytes");
    need(std::filesystem::file_size(pack_path) == built.file_bytes, "pack v3 on-disk file size mismatch");

    const auto adopted = adopt_derived_pack_v3(source, mtp, pack_path, 2u, derived_pack::default_chunk_bytes, 4u, derived_pack::TrustMode::full, false);
    need(adopted.authentication_workers == 4u && adopted.plan.tensors.size() == 11, "adopt_derived_pack_v3 metadata mismatch");

    // Verify single-block builder and adopter (blk.0).
    const auto blk0_built = build_block_pack_v3(source, mtp, 0u, blk0_path, 2u, 1024ull * 1024ull);
    need(blk0_built.tensors.size() == 5, "blk.0 single-block tensor count mismatch");
    const auto blk0_adopted = adopt_block_pack_v3(source, mtp, 0u, blk0_path, 2u, derived_pack::default_chunk_bytes, 2u);
    need(blk0_adopted.plan.tensors.size() == 5, "adopt_block_pack_v3 tensor count mismatch");

    // Tamper check: flip one payload bit and verify full authentication refuses.
    ::chmod(blk0_path.c_str(), 0600);
    {
        const int fd = ::open(blk0_path.c_str(), O_RDWR | O_CLOEXEC);
        need(fd >= 0, "open blk0 pack for tamper check");
        std::byte b{};
        need(::pread(fd, &b, 1, off_t(blk0_built.data_offset)) == 1, "pread tamper byte");
        b ^= std::byte{0x01};
        need(::pwrite(fd, &b, 1, off_t(blk0_built.data_offset)) == 1, "pwrite tamper byte");
        ::close(fd);
    }
    ::chmod(blk0_path.c_str(), 0400);
    refuses([&] {
        static_cast<void>(adopt_block_pack_v3(source, mtp, 0u, blk0_path, 2u, derived_pack::default_chunk_bytes, 2u));
    }, "tampered pack v3 must be refused during full authentication");

    std::cout << "PASS planar_v3 builder, multi-worker adoption, zero-footer check and tamper refusal\n";
}

struct PackedColocPair {
    Bytes pair_lo;
    Bytes pair_hi;
    Bytes down;
    std::vector<float> gate_ref;
    std::vector<float> up_ref;
    std::vector<float> down_ref;
};

PackedColocPair build_coloc_fixture(uint32_t gu_type, uint32_t dn_type, uint32_t channels, uint32_t hidden, uint32_t experts, bool distinct_up = false, bool materialise_reference = true) {
    const auto raw_gate = grouped_fixture::weights({gu_type, channels, hidden, experts, 1u, 1u, true});
    auto raw_up = grouped_fixture::weights({gu_type, channels, hidden, experts, 1u, 1u, true});
    if (distinct_up) {
        const uint32_t block_bytes = encoding(gu_type).bytes_per_block;
        const uint32_t codes = gu_type == 12u ? 16u : (gu_type == 13u ? 48u : 2u);
        for (size_t at = 0; at < raw_up.size(); at += block_bytes)
            for (uint32_t i = codes; i < block_bytes; ++i) raw_up[at + i] ^= std::byte{0x5a};
    }
    const auto raw_down = grouped_fixture::weights({dn_type, hidden, channels, experts, 1u, 1u, true});

    PackedColocPair out;
    if (materialise_reference) {
        out.gate_ref.resize(uint64_t(channels) * hidden * experts);
        out.up_ref.resize(uint64_t(channels) * hidden * experts);
        out.down_ref.resize(uint64_t(hidden) * channels * experts);
        tokspur::dequantize(gu_type, raw_gate, out.gate_ref);
        tokspur::dequantize(gu_type, raw_up, out.up_ref);
        tokspur::dequantize(dn_type, raw_down, out.down_ref);
    }

    const auto gu_spec = make_expert_spec(gu_type, channels, hidden);
    const auto dn_spec = make_expert_spec(dn_type, hidden, channels);
    const uint32_t half_e = experts / 2u;
    out.pair_lo.resize(uint64_t(half_e) * 2u * gu_spec.planar_expert_bytes);
    out.pair_hi.resize(uint64_t(half_e) * 2u * gu_spec.planar_expert_bytes);
    out.down.resize(uint64_t(experts) * dn_spec.planar_expert_bytes);

    for (uint32_t e = 0; e < experts; ++e) {
        Bytes& dst_pair = (e < half_e) ? out.pair_lo : out.pair_hi;
        const uint32_t local_e = (e < half_e) ? e : (e - half_e);
        const uint64_t pair_off = uint64_t(local_e) * 2u * gu_spec.planar_expert_bytes;
        pack_expert_planar_v3(
            gu_spec,
            std::span<const std::byte>(raw_gate).subspan(uint64_t(e) * gu_spec.source_expert_bytes, gu_spec.source_expert_bytes),
            std::span<std::byte>(dst_pair).subspan(pair_off, gu_spec.planar_expert_bytes));
        pack_expert_planar_v3(
            gu_spec,
            std::span<const std::byte>(raw_up).subspan(uint64_t(e) * gu_spec.source_expert_bytes, gu_spec.source_expert_bytes),
            std::span<std::byte>(dst_pair).subspan(pair_off + gu_spec.planar_expert_bytes, gu_spec.planar_expert_bytes));
        pack_expert_planar_v3(
            dn_spec,
            std::span<const std::byte>(raw_down).subspan(uint64_t(e) * dn_spec.source_expert_bytes, dn_spec.source_expert_bytes),
            std::span<std::byte>(out.down).subspan(uint64_t(e) * dn_spec.planar_expert_bytes, dn_spec.planar_expert_bytes));
    }
    return out;
}

std::pair<PlanarKind, PlanarKind> coloc_kinds_for(uint32_t gu_type) {
    if (gu_type == 12u) return {PlanarKind::q4k_coloc_gate_v3, PlanarKind::q4k_coloc_up_v3};
    if (gu_type == 13u) return {PlanarKind::q5k_coloc_gate_v3, PlanarKind::q5k_coloc_up_v3};
    return {PlanarKind::q8_coloc_gate_v3, PlanarKind::q8_coloc_up_v3};
}

PlanarKind down_kind_for(uint32_t dn_type) {
    return dn_type == 7u ? PlanarKind::q51_planar_v3 : PlanarKind::q8_planar_v3;
}

void test_gpu_isolation_and_numerical_parity(const char* metallib, std::vector<float>* captured = nullptr,
    std::vector<uint64_t>* resources = nullptr) {
    struct Combo {
        uint32_t gu_type;
        uint32_t dn_type;
    };
    const std::array<Combo, 4> combos{{{12u, 7u}, {13u, 7u}, {13u, 8u}, {8u, 8u}}};
    const uint32_t channels = 256u, hidden = 64u, experts = 8u, selected = 4u;

    for (const auto combo : combos) {
        const auto fx = build_coloc_fixture(combo.gu_type, combo.dn_type, channels, hidden, experts);
        MetalGraph g(metallib, 512ull << 20);
        Arena a{g, {}};
        const auto v_lo = a.upload(fx.pair_lo);
        const auto v_hi = a.upload(fx.pair_hi);
        const auto v_dn = a.upload(fx.down);
        const auto [k_lo, k_hi] = coloc_kinds_for(combo.gu_type);
        const auto k_dn = down_kind_for(combo.dn_type);

        g.register_planar_matrix(v_lo, {combo.gu_type, uint32_t(k_lo), channels, hidden, experts});
        g.register_planar_matrix(v_hi, {combo.gu_type, uint32_t(k_hi), channels, hidden, experts});
        g.register_planar_matrix(v_dn, {combo.dn_type, uint32_t(k_dn), hidden, channels, experts});
        need(g.planar_matrix_count() == 3u, "planar_matrix_count mismatch");
        need(g.planar_matrix_shape(v_lo).has_value() && g.planar_matrix_shape(v_lo)->source_type == combo.gu_type,
             "planar_matrix_shape query mismatch");

        // Consumer isolation checks: non-planar primitives, copy, capture and indirect comparator must refuse.
        const auto dummy_out = a.zeros(fx.down.size());
        refuses([&] { g.copy(std::array{GraphCopy{v_dn, dummy_out}}); }, "copy must refuse planar v3 weights");
        refuses([&] { g.begin_capture(8); }, "begin_capture must refuse planar v3 weights");
        refuses([&] { g.begin_indirect_compatible(); }, "begin_indirect_compatible must refuse planar v3 weights");
        {
            const auto d_in = a.zeros(uint64_t(hidden) * 4u);
            const auto d_out = a.zeros(uint64_t(channels) * experts * 4u);
            g.begin();
            refuses([&] { g.gemv(combo.dn_type, v_dn, d_in, d_out, hidden, channels * experts); },
                    "gemv must refuse planar v3 weights");
            g.abort();
        }
        {
            const std::array<uint32_t, 4> r_ids{0u, 1u, 2u, 3u};
            const auto d_ix = a.upload(std::as_bytes(std::span(r_ids)));
            const auto d_in = a.zeros(uint64_t(hidden) * 4u * 4u);
            const auto d_out = a.zeros(uint64_t(channels) * 4u * 4u);
            g.begin();
            refuses([&] { g.routed_gemv(combo.dn_type, v_dn, d_ix, d_in, d_out, hidden, channels, experts, false); },
                    "routed_gemv must refuse planar v3 weights");
            g.abort();
        }

        g.prepare_planar_moe_v3(combo.gu_type, combo.dn_type, true, 600u, experts, selected, channels, hidden);

        // Every supported width, including duplicate/shared routes and singletons.
        for (const uint32_t m : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
            std::vector<float> x(uint64_t(m) * channels);
            uint32_t rng = 9001u + m * 31u + combo.gu_type;
            for (auto& val : x) {
                rng = rng * 1664525u + 1013904223u;
                val = float(int(rng % 1993) - 996) / 997.0f;
            }
            std::vector<uint32_t> ids(m * selected);
            for (uint32_t t = 0; t < m; ++t) {
                for (uint32_t s = 0; s < selected; ++s) {
                    // Mix shared experts across rows (expert 0 and 5) with singleton experts.
                    ids[t * selected + s] = (s == 0) ? 0u : ((s == 1 && (t & 1u)) ? 5u : ((t * 3u + s * 2u + 1u) % experts));
                }
            }
            const auto vx = a.floats(x);
            const auto vix = a.upload(std::as_bytes(std::span(ids)));
            const auto vh = a.zeros(uint64_t(m * selected) * hidden * 4u);
            const auto vy = a.zeros(uint64_t(m * selected) * channels * 4u);

            g.begin();
            g.planar_moe_routed_m8(
                v_lo, v_hi, v_dn, vix,
                {vx, m, channels, uint64_t(channels) * 4u},
                {vh, m * selected, hidden, uint64_t(hidden) * 4u},
                {vy, m * selected, channels, uint64_t(channels) * 4u},
                experts, selected);
            const auto batch = g.submit();
            need(batch.dispatches == (m == 1u ? 2u : 3u), "planar_moe_routed_m8 dispatch count mismatch");

            std::vector<float> got_h(m * selected * hidden), got_y(m * selected * channels);
            g.read(vh, std::as_writable_bytes(std::span(got_h)));
            g.read(vy, std::as_writable_bytes(std::span(got_y)));
            if (captured) {
                captured->insert(captured->end(), got_h.begin(), got_h.end());
                captured->insert(captured->end(), got_y.begin(), got_y.end());
            }
            if (resources) {
                resources->push_back(g.memory().live_bytes); resources->push_back(g.memory().allocations);
                resources->push_back(batch.dispatches);
            }

            std::vector<float> ref_h(got_h.size()), ref_y(got_y.size());
            for (uint32_t rid = 0; rid < m * selected; ++rid) {
                const uint32_t tok = rid / selected;
                const uint32_t exp = ids[rid];
                for (uint32_t r = 0; r < hidden; ++r) {
                    double dg = 0.0, du = 0.0;
                    for (uint32_t c = 0; c < channels; ++c) {
                        const double xv = x[uint64_t(tok) * channels + c];
                        dg += double(fx.gate_ref[(uint64_t(exp) * hidden + r) * channels + c]) * xv;
                        du += double(fx.up_ref[(uint64_t(exp) * hidden + r) * channels + c]) * xv;
                    }
                    ref_h[uint64_t(rid) * hidden + r] = silu_f32(float(dg)) * float(du);
                }
                for (uint32_t r = 0; r < channels; ++r) {
                    double dd = 0.0;
                    for (uint32_t c = 0; c < hidden; ++c) {
                        dd += double(fx.down_ref[(uint64_t(exp) * channels + r) * hidden + c]) *
                              double(ref_h[uint64_t(rid) * hidden + c]);
                    }
                    ref_y[uint64_t(rid) * channels + r] = float(dd);
                }
            }
            need(nmse_f32(got_h, ref_h) <= 1e-6, "planar_moe_gate_up_swiglu_m8 NMSE exceeded 1e-6");
            need(nmse_f32(got_y, ref_y) <= 1e-6, "planar_moe_routed_m8 NMSE exceeded 1e-6");
        }

        // 1b. All-singleton (n == 1 on every unique expert, U == 32) parity with channels = 512, hidden = 416 (13 blocks = 8 + 4 + 1).
        {
            const uint32_t s_ch = 512u, s_hid = 416u, s_exp = 32u, s_sel = 4u;
            const auto s_fx = build_coloc_fixture(combo.gu_type, combo.dn_type, s_ch, s_hid, s_exp);
            MetalGraph gs(metallib, 512ull << 20);
            Arena as{gs, {}};
            const auto sv_lo = as.upload(s_fx.pair_lo);
            const auto sv_hi = as.upload(s_fx.pair_hi);
            const auto sv_dn = as.upload(s_fx.down);
            gs.register_planar_matrix(sv_lo, {combo.gu_type, uint32_t(k_lo), s_ch, s_hid, s_exp});
            gs.register_planar_matrix(sv_hi, {combo.gu_type, uint32_t(k_hi), s_ch, s_hid, s_exp});
            gs.register_planar_matrix(sv_dn, {combo.dn_type, uint32_t(k_dn), s_hid, s_ch, s_exp});
            gs.prepare_planar_moe_v3(combo.gu_type, combo.dn_type, false, 32u, s_exp, s_sel, s_ch, s_hid);

            for (const uint32_t m : {2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
                std::vector<float> sx_vec(uint64_t(m) * s_ch);
                uint32_t rng = 31337u + m * 23u + combo.gu_type * 7u;
                for (auto& val : sx_vec) {
                    rng = rng * 1664525u + 1013904223u;
                    val = float(int(rng % 1993) - 996) / 997.0f;
                }
                std::vector<uint32_t> s_ids(m * s_sel);
                for (uint32_t rid = 0; rid < m * s_sel; ++rid) {
                    s_ids[rid] = (rid * 7u + 3u) % s_exp; // gcd(7, 32) == 1 -> all 32 routes are distinct singletons!
                }
                const auto vx = as.floats(sx_vec);
                const auto vix = as.upload(std::as_bytes(std::span(s_ids)));
                const auto vh = as.zeros(uint64_t(m * s_sel) * s_hid * 4u);
                const auto vy = as.zeros(uint64_t(m * s_sel) * s_ch * 4u);
                gs.begin();
                gs.planar_moe_routed_m8(
                    sv_lo, sv_hi, sv_dn, vix,
                    {vx, m, s_ch, uint64_t(s_ch) * 4u},
                    {vh, m * s_sel, s_hid, uint64_t(s_hid) * 4u},
                    {vy, m * s_sel, s_ch, uint64_t(s_ch) * 4u},
                    s_exp, s_sel);
                const auto batch = gs.submit();
                need(batch.dispatches == 3u, "singleton planar_moe_routed_m8 dispatch count mismatch");
                std::vector<float> got_h(m * s_sel * s_hid), got_y(m * s_sel * s_ch);
                gs.read(vh, std::as_writable_bytes(std::span(got_h)));
                gs.read(vy, std::as_writable_bytes(std::span(got_y)));
                if (captured) {
                    captured->insert(captured->end(), got_h.begin(), got_h.end());
                    captured->insert(captured->end(), got_y.begin(), got_y.end());
                }
                if (resources) {
                    resources->push_back(gs.memory().live_bytes); resources->push_back(gs.memory().allocations);
                    resources->push_back(batch.dispatches);
                }
                std::vector<float> ref_h(got_h.size()), ref_y(got_y.size());
                for (uint32_t rid = 0; rid < m * s_sel; ++rid) {
                    const uint32_t tok = rid / s_sel;
                    const uint32_t exp = s_ids[rid];
                    for (uint32_t r = 0; r < s_hid; ++r) {
                        double dg = 0.0, du = 0.0;
                        for (uint32_t c = 0; c < s_ch; ++c) {
                            const double xv = sx_vec[uint64_t(tok) * s_ch + c];
                            dg += double(s_fx.gate_ref[(uint64_t(exp) * s_hid + r) * s_ch + c]) * xv;
                            du += double(s_fx.up_ref[(uint64_t(exp) * s_hid + r) * s_ch + c]) * xv;
                        }
                        ref_h[uint64_t(rid) * s_hid + r] = silu_f32(float(dg)) * float(du);
                    }
                    for (uint32_t r = 0; r < s_ch; ++r) {
                        double dd = 0.0;
                        for (uint32_t c = 0; c < s_hid; ++c) {
                            dd += double(s_fx.down_ref[(uint64_t(exp) * s_ch + r) * s_hid + c]) *
                                  double(ref_h[uint64_t(rid) * s_hid + c]);
                        }
                        ref_y[uint64_t(rid) * s_ch + r] = float(dd);
                    }
                }
                need(nmse_f32(got_h, ref_h) <= 1e-6, "singleton planar_moe_gate_up_swiglu NMSE exceeded 1e-6");
                need(nmse_f32(got_y, ref_y) <= 1e-6, "singleton planar_moe_routed_m8 NMSE exceeded 1e-6");
            }
        }

        // 2. NAX grouped MoE parity (m = 1, 4, 33, 64, 96, 544 across 0, 1..15, 16..31, 33..63 and > 128 routes/expert).
        for (const uint32_t n : {1u, 4u, 33u, 64u, 96u, 544u}) {
            std::vector<float> x(uint64_t(n) * channels);
            uint32_t rng = 7777u + n * 19u + combo.dn_type;
            for (auto& val : x) {
                rng = rng * 1664525u + 1013904223u;
                val = float(int(rng % 1993) - 996) / 997.0f;
            }
            std::vector<uint32_t> ids(n * selected);
            for (uint32_t t = 0; t < n; ++t) {
                for (uint32_t s = 0; s < selected; ++s) {
                    if (n == 96u && t < 75u) {
                        ids[t * selected + s] = (s == 0u) ? 0u : (1u + ((t + s * 2u) % (experts - 1u)));
                    } else if (n == 544u) {
                        // Exercise 0 routes (expert 7), partial SG0 (11 routes, expert 6), partial SG0 hi (25 routes, expert 5), partial SG1 (45 routes, expert 4), and multi-tile (>128 routes, experts 0..3).
                        if (t < 11u && s == 0u) ids[t * selected + s] = 6u;
                        else if (t < 25u && s == 1u) ids[t * selected + s] = 5u;
                        else if (t < 45u && s == 2u) ids[t * selected + s] = 4u;
                        else ids[t * selected + s] = (t * 3u + s) % 4u;
                    } else {
                        ids[t * selected + s] = (t * 5u + s * 3u) % experts;
                    }
                }
            }
            const uint32_t max_tasks = (n * selected) / 32u + experts;
            const auto vx = a.floats(x);
            const auto vix = a.upload(std::as_bytes(std::span(ids)));
            const auto vy = a.zeros(uint64_t(n * selected) * channels * 4u);
            const auto vmap = a.zeros(uint64_t(experts) * n * 4u);
            const auto vcnt = a.zeros(uint64_t(experts) * 4u);
            const auto vtsk = a.zeros(uint64_t(2u + max_tasks * 2u) * 4u);

            g.begin();
            g.planar_nax_grouped_moe(
                v_lo, v_hi, v_dn, vix,
                {vx, n, channels, uint64_t(channels) * 4u},
                {vy, n * selected, channels, uint64_t(channels) * 4u},
                vmap, vcnt, vtsk, experts, selected);
            const auto batch = g.submit();
            need(batch.dispatches == 5u, "planar_nax_grouped_moe must execute in 5 dispatches");

            std::vector<float> got_y(n * selected * channels), ref_y(n * selected * channels);
            g.read(vy, std::as_writable_bytes(std::span(got_y)));
            for (uint32_t rid = 0; rid < n * selected; ++rid) {
                const uint32_t tok = rid / selected;
                const uint32_t exp = ids[rid];
                std::vector<float> h_row(hidden);
                for (uint32_t r = 0; r < hidden; ++r) {
                    double dg = 0.0, du = 0.0;
                    for (uint32_t c = 0; c < channels; ++c) {
                        const double xv = x[uint64_t(tok) * channels + c];
                        dg += double(fx.gate_ref[(uint64_t(exp) * hidden + r) * channels + c]) * xv;
                        du += double(fx.up_ref[(uint64_t(exp) * hidden + r) * channels + c]) * xv;
                    }
                    h_row[r] = silu_f32(float(dg)) * float(du);
                }
                for (uint32_t r = 0; r < channels; ++r) {
                    double dd = 0.0;
                    for (uint32_t c = 0; c < hidden; ++c) {
                        dd += double(fx.down_ref[(uint64_t(exp) * channels + r) * hidden + c]) * double(h_row[c]);
                    }
                    ref_y[uint64_t(rid) * channels + r] = float(dd);
                }
            }
            need(nmse_f32(got_y, ref_y) <= 2e-3, "planar_nax_grouped_moe NMSE exceeded 2e-3");
        }
    }
    std::cout << "PASS planar_v3 GPU isolation, small-M (m=1..8) and NAX grouped (5-dispatch) parity\n";
}

void test_sticky_finite_contract(const char* metallib, uint32_t gu_type = 12u, uint32_t dn_type = 7u, uint32_t rows = 4u, bool nax_only = false) {
    const uint32_t channels = 256u, hidden = 64u, experts = 4u, selected = 2u;
    const auto base_fx = build_coloc_fixture(gu_type, dn_type, channels, hidden, experts);
    const auto gu_spec = make_expert_spec(gu_type, channels, hidden);
    const auto dn_spec = make_expert_spec(dn_type, hidden, channels);

    const auto run_case = [&](const Bytes& lo_b, const Bytes& hi_b, const Bytes& dn_b,
                              const std::vector<float>& x, const std::vector<uint32_t>& ids, bool use_nax) {
        MetalGraph g(metallib, 256ull << 20);
        Arena a{g, {}};
        const auto v_lo = a.upload(lo_b);
        const auto v_hi = a.upload(hi_b);
        const auto v_dn = a.upload(dn_b);
        const auto [gate_kind, up_kind] = coloc_kinds_for(gu_type);
        g.register_planar_matrix(v_lo, {gu_type, uint32_t(gate_kind), channels, hidden, experts});
        g.register_planar_matrix(v_hi, {gu_type, uint32_t(up_kind), channels, hidden, experts});
        g.register_planar_matrix(v_dn, {dn_type, uint32_t(down_kind_for(dn_type)), hidden, channels, experts});
        const uint32_t n = uint32_t(x.size() / channels);
        g.prepare_planar_moe_v3(gu_type, dn_type, use_nax, std::max(32u, n), experts, selected, channels, hidden);

        const auto vx = a.floats(x);
        const auto vix = a.upload(std::as_bytes(std::span(ids)));
        const auto vy = a.zeros(uint64_t(n * selected) * channels * 4u);
        if (use_nax) {
            const uint32_t max_tasks = (n * selected) / 32u + experts;
            const auto vmap = a.zeros(uint64_t(experts) * n * 4u);
            const auto vcnt = a.zeros(uint64_t(experts) * 4u);
            const auto vtsk = a.zeros(uint64_t(2u + max_tasks * 2u) * 4u);
            g.begin();
            g.planar_nax_grouped_moe(
                v_lo, v_hi, v_dn, vix,
                {vx, n, channels, uint64_t(channels) * 4u},
                {vy, n * selected, channels, uint64_t(channels) * 4u},
                vmap, vcnt, vtsk, experts, selected);
            static_cast<void>(g.submit());
        } else {
            const auto vh = a.zeros(uint64_t(n * selected) * hidden * 4u);
            g.begin();
            g.planar_moe_routed_m8(
                v_lo, v_hi, v_dn, vix,
                {vx, n, channels, uint64_t(channels) * 4u},
                {vh, n * selected, hidden, uint64_t(hidden) * 4u},
                {vy, n * selected, channels, uint64_t(channels) * 4u},
                experts, selected);
            static_cast<void>(g.submit());
        }
    };

    const std::vector<float> clean_x(rows * channels, 0.25f);
    std::vector<uint32_t> clean_ids(rows * selected);
    for (uint32_t row = 0; row < rows; ++row) {
        clean_ids[row * selected] = 0u;
        clean_ids[row * selected + 1u] = 3u;
    }

    for (const bool use_nax : {false, true}) {
        if (nax_only && !use_nax) continue;
        // Control pass succeeds.
        run_case(base_fx.pair_lo, base_fx.pair_hi, base_fx.down, clean_x, clean_ids, use_nax);

        // 1. Non-finite NaN / Inf in input activation.
        for (const float bad_v : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            auto bad_x = clean_x;
            bad_x[17u] = bad_v;
            refuses([&] { run_case(base_fx.pair_lo, base_fx.pair_hi, base_fx.down, bad_x, clean_ids, use_nax); },
                    "planar MoE must refuse non-finite input", "nonfinite");
        }

        // 2. Stored FP16 NaN (0x7e00) in gate super-scale: even though tile loader substitutes 0.0f so output stays 0, sticky error MUST fire!
        {
            auto bad_lo = base_fx.pair_lo;
            const uint16_t nan_h = 0x7e00u;
            std::memcpy(bad_lo.data() + gu_spec.dm_offset, &nan_h, 2);
            refuses([&] { run_case(bad_lo, base_fx.pair_hi, base_fx.down, clean_x, clean_ids, use_nax); },
                    "planar MoE must refuse non-finite stored gate scale despite zero substitution", "nonfinite");
        }

        // The up scale and finite weights that overflow the FP16 stage must reach the same sticky guard in each staged tile size.
        {
            auto bad_lo = base_fx.pair_lo;
            const uint16_t inf_h = 0x7c00u;
            std::memcpy(bad_lo.data() + gu_spec.planar_expert_bytes + gu_spec.dm_offset, &inf_h, 2);
            refuses([&] { run_case(bad_lo, base_fx.pair_hi, base_fx.down, clean_x, clean_ids, use_nax); },
                    "planar MoE must refuse non-finite stored up scale", "nonfinite");
        }
        if (use_nax && rows >= 3u) {
            auto bad_lo = base_fx.pair_lo;
            const uint16_t large_h = 0x7bffu;
            std::memcpy(bad_lo.data() + gu_spec.dm_offset, &large_h, 2);
            if (gu_type == 12u || gu_type == 13u) bad_lo[size_t(gu_spec.sm_offset)] = std::byte{63};
            std::fill_n(bad_lo.begin() + std::ptrdiff_t(gu_spec.qs_offset), gu_type == 8u ? 32u : 128u, std::byte{0x7f});
            refuses([&] { run_case(bad_lo, base_fx.pair_hi, base_fx.down, clean_x, clean_ids, use_nax); },
                    "planar MoE must refuse finite stored weights overflowing its FP16 stage", "nonfinite");
        }

        // 3. Stored FP16 Inf (0x7c00) in down Q5_1 scale: verifies intermediate/stored check in down kernel.
        {
            auto bad_dn = base_fx.down;
            const uint16_t inf_h = 0x7c00u;
            std::memcpy(bad_dn.data() + dn_spec.dm_offset, &inf_h, 2);
            refuses([&] { run_case(base_fx.pair_lo, base_fx.pair_hi, bad_dn, clean_x, clean_ids, use_nax); },
                    "planar MoE must refuse non-finite stored down scale", "nonfinite");
        }
        if (use_nax && rows >= 3u) {
            auto bad_dn = base_fx.down;
            const uint16_t large_h = 0x7bffu;
            std::memcpy(bad_dn.data() + dn_spec.dm_offset, &large_h, 2);
            std::fill_n(bad_dn.begin() + std::ptrdiff_t(dn_spec.qs_offset), dn_type == 8u ? 32u : 16u, std::byte{0x7f});
            refuses([&] { run_case(base_fx.pair_lo, base_fx.pair_hi, bad_dn, clean_x, clean_ids, use_nax); },
                    "planar MoE must refuse finite down weights overflowing its FP16 stage", "nonfinite");
        }

        // 4. Intermediate SwiGLU overflow (finite large inputs whose gate*up product overflows FP32 / BF16 conversion).
        {
            std::vector<float> huge_x(rows * channels, 3.0e38f);
            refuses([&] { run_case(base_fx.pair_lo, base_fx.pair_hi, base_fx.down, huge_x, clean_ids, use_nax); },
                    "planar MoE must refuse intermediate/narrow-conversion overflow", "nonfinite");
        }

        // 5. Out-of-range expert route index (>= experts).
        {
            auto bad_ids = clean_ids;
            bad_ids.back() = experts;
            refuses([&] { run_case(base_fx.pair_lo, base_fx.pair_hi, base_fx.down, clean_x, bad_ids, use_nax); },
                    "planar MoE must refuse out-of-range route index", "nonfinite");
        }

        // Clean control pass still succeeds afterwards.
        run_case(base_fx.pair_lo, base_fx.pair_hi, base_fx.down, clean_x, clean_ids, use_nax);
    }
    std::cout << "PASS planar_v3 sticky finite-value contract (input, stored scale, narrow conversion, intermediate SwiGLU, route bounds)\n";
}

float represented_bf16(float value) {
    const uint32_t bits = uint32_t(finite_bf16(value)) << 16u;
    float rounded;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

std::vector<float> test_nax_staging(const char* metallib, const char* profile_arm = nullptr, uint32_t leg = 0u,
    std::vector<uint64_t>* resources = nullptr, bool gu_stage64 = false) {
    constexpr uint32_t experts = 4u, selected = 2u;
    constexpr float sentinel = -12345.25f;
    std::vector<float> captured;
    const std::array<std::pair<uint32_t, uint32_t>, 4> formats{{{12u, 7u}, {13u, 7u}, {13u, 8u}, {8u, 8u}}};
    for (const auto [gu_type, dn_type] : formats) {
        for (const bool model_shape : {false, true}) {
            if (model_shape && (!profile_arm || dn_type != 7u)) continue;
            const uint32_t channels = model_shape ? 2560u : 512u;
            const uint32_t hidden = model_shape ? 640u : 96u;
            const uint32_t capacity = model_shape ? 128u : (gu_stage64 ? 34u : 33u);
            const uint32_t input_stride = channels + 4u, output_stride = channels + 8u;
            const auto fx = build_coloc_fixture(gu_type, dn_type, channels, hidden, experts, true);
            MetalGraph g(metallib, 512ull << 20);
            g.prepare_planar_moe_v3(gu_type, dn_type, true, capacity, experts, selected, channels, hidden);
            const auto prepared_bytes = g.memory().live_bytes;
            {
                Arena a{g, {}};
                const auto lo = a.upload(fx.pair_lo), hi = a.upload(fx.pair_hi), down = a.upload(fx.down);
                const auto [gate_kind, up_kind] = coloc_kinds_for(gu_type);
                g.register_planar_matrix(lo, {gu_type, uint32_t(gate_kind), channels, hidden, experts});
                g.register_planar_matrix(hi, {gu_type, uint32_t(up_kind), channels, hidden, experts});
                g.register_planar_matrix(down, {dn_type, uint32_t(down_kind_for(dn_type)), hidden, channels, experts});
                std::vector<float> input(uint64_t(capacity + 1u) * input_stride, sentinel);
                uint32_t rng = 0x173124u;
                for (uint32_t row = 0; row < capacity; ++row) {
                    for (uint32_t k = 0; k < channels; ++k) {
                        rng = rng * 1664525u + 1013904223u;
                        input[uint64_t(row) * input_stride + k] = float(int(rng % 1993u) - 996) / 9970.0f;
                    }
                }
                std::vector<uint32_t> ids(capacity * selected);
                for (uint32_t row = 0; row < capacity; ++row) {
                    ids[row * selected] = 0u;
                    ids[row * selected + 1u] = 3u;
                }
                const auto x = a.floats(input), routes = a.upload(std::as_bytes(std::span(ids)));
                const std::vector<float> poison(uint64_t(capacity * selected + 1u) * output_stride, sentinel);
                const auto y = a.floats(poison, false);
                const uint32_t max_tasks = capacity * selected / 32u + experts;
                const auto map = a.zeros(uint64_t(experts) * capacity * sizeof(uint32_t));
                const auto counts = a.zeros(experts * sizeof(uint32_t));
                const auto tasks = a.zeros((2u + 2u * max_tasks) * sizeof(uint32_t));
                const auto live_bytes = g.memory().live_bytes;
                const std::vector<uint32_t> row_cases = model_shape
                    ? (gu_stage64 ? std::vector<uint32_t>{32u, 128u} : std::vector<uint32_t>{128u})
                    : (gu_stage64 ? std::vector<uint32_t>{1u, 2u, 3u, 8u, 16u, 32u, 33u, 34u} : std::vector<uint32_t>{1u, 2u, 3u, 8u, 16u, 32u, 33u});
                for (const uint32_t rows : row_cases) {
                    if (resources) { resources->push_back(live_bytes); resources->push_back(g.memory().allocations); }
                    g.write(y, std::as_bytes(std::span(poison)));
                    const auto execute = [&](uint32_t iterations = 1u) {
                        g.begin();
                        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
                            g.planar_nax_grouped_moe(lo, hi, down, routes,
                                {x, rows, channels, uint64_t(input_stride) * sizeof(float)},
                                {y, rows * selected, channels, uint64_t(output_stride) * sizeof(float)},
                                map, counts, tasks, experts, selected);
                        }
                        const auto batch = g.submit();
                        need(batch.dispatches == 5u * iterations, "planar staging fixture dispatch count changed");
                        need(g.memory().live_bytes == live_bytes, "planar staging allocated during execution");
                        return batch;
                    };
                    if (profile_arm) need(host_thermal_state() == 0u, "planar staging warm-up must start nominal");
                    static_cast<void>(execute());
                    if (profile_arm) {
                        constexpr uint32_t iterations = 32u;
                        const auto warm_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
                        do {
                            static_cast<void>(execute(iterations));
                            need(host_thermal_state() == 0u, "planar staging warm-up must remain nominal");
                        } while (std::chrono::steady_clock::now() < warm_end);
                        for (uint32_t trial = 0; trial < 8u; ++trial) {
                            const auto before = host_thermal_state();
                            need(before == 0u, "planar staging component must start nominal");
                            const auto batch = execute(iterations);
                            const auto after = host_thermal_state();
                            std::cout << "PLANAR_STAGING_PROFILE {\"arm\":\"" << profile_arm << "\",\"leg\":" << leg
                                      << ",\"gu_stage64_comparison\":" << (gu_stage64 ? 1 : 0)
                                      << ",\"gate_encoding\":" << gu_type << ",\"down_encoding\":" << dn_type
                                      << ",\"channels\":" << channels << ",\"hidden\":" << hidden << ",\"rows\":" << rows
                                      << ",\"trial\":" << trial << ",\"iterations\":" << iterations << ",\"batch_gpu_ms\":" << batch.gpu_ms << ",\"live_metal_bytes\":" << live_bytes
                                      << ",\"thermal_before\":" << before << ",\"thermal_after\":" << after
                                      << ",\"scope\":\"GPU interval for 32 warm five-dispatch grouped MoE iterations with fixed generated routes\"}\n";
                            need(after == 0u, "planar staging component must finish nominal");
                            need(std::isfinite(batch.gpu_ms) && batch.gpu_ms > 0.0, "planar staging GPU interval must be finite and positive");
                        }
                    }
                    std::vector<float> actual(poison.size());
                    g.read(y, std::as_writable_bytes(std::span(actual)));
                    std::vector<float> got(uint64_t(rows) * selected * channels);
                    for (uint32_t row = 0; row <= capacity * selected; ++row) {
                        for (uint32_t col = 0; col < output_stride; ++col) {
                            const float value = actual[uint64_t(row) * output_stride + col];
                            if (row < rows * selected && col < channels) {
                                need(std::isfinite(value), "planar staging produced a nonfinite value");
                                got[uint64_t(row) * channels + col] = value;
                            } else need(value == sentinel, "planar staging wrote outside live output rows or into stride padding");
                        }
                    }
                    std::array<uint32_t, experts> route_counts{};
                    g.read(counts, std::as_writable_bytes(std::span(route_counts)));
                    need(route_counts == std::array<uint32_t, experts>{rows, 0u, 0u, rows}, "planar staging route occupancy differs from fixture");

                    // Scalar FP64 sums use BF16 inputs, FP16 staged weights and the BF16 SwiGLU hand-off; the unchanged <=2-row column paths retain FP32 weights.
                    const uint32_t reference_rows = model_shape ? 3u : rows;
                    std::vector<float> reference(uint64_t(reference_rows) * selected * channels);
                    for (uint32_t row = 0; row < reference_rows; ++row) {
                        const uint32_t task_live = std::min(32u, rows - row / 32u * 32u);
                        const bool gate_column = gu_type != 8u && task_live <= 2u;
                        const bool down_column = dn_type == 7u && task_live <= 2u;
                        for (uint32_t slot = 0; slot < selected; ++slot) {
                            const uint32_t expert = ids[row * selected + slot];
                            std::vector<float> h(hidden);
                            for (uint32_t col = 0; col < hidden; ++col) {
                                double gate_sum = 0.0, up_sum = 0.0;
                                for (uint32_t k = 0; k < channels; ++k) {
                                    const auto at = (uint64_t(expert) * hidden + col) * channels + k;
                                    const float xv = represented_bf16(input[uint64_t(row) * input_stride + k]);
                                    const float gw = gate_column ? fx.gate_ref[at] : float(_Float16(fx.gate_ref[at]));
                                    const float uw = gate_column ? fx.up_ref[at] : float(_Float16(fx.up_ref[at]));
                                    gate_sum += double(xv) * gw;
                                    up_sum += double(xv) * uw;
                                }
                                h[col] = represented_bf16(silu_f32(float(gate_sum)) * float(up_sum));
                            }
                            for (uint32_t col = 0; col < channels; ++col) {
                                double sum = 0.0;
                                for (uint32_t k = 0; k < hidden; ++k) {
                                    const auto at = (uint64_t(expert) * channels + col) * hidden + k;
                                    const float dw = down_column ? fx.down_ref[at] : float(_Float16(fx.down_ref[at]));
                                    sum += double(h[k]) * dw;
                                }
                                reference[(uint64_t(row) * selected + slot) * channels + col] = float(sum);
                            }
                        }
                    }
                    need(nmse_f32(std::span(got).first(reference.size()), reference) <= 1e-4,
                        "planar staging exceeds BF16 NMSE bound against declared-precision scalar reference");
                    captured.insert(captured.end(), got.begin(), got.end());
                }
            }
            need(g.memory().live_bytes == prepared_bytes, "planar staging fixture leaked registered buffers");
        }
    }
    std::cout << "PASS planar staging varied codes, distinct gate/up, live-row tiles, scalar reference and output guards\n";
    return captured;
}

void test_nax_staging_finite(const char* metallib) {
    for (const uint32_t rows : {1u, 2u, 3u, 8u, 16u, 32u}) {
        test_sticky_finite_contract(metallib, 12u, 7u, rows, true);
        test_sticky_finite_contract(metallib, 13u, 7u, rows, true);
    }
    test_sticky_finite_contract(metallib, 8u, 8u, 3u, true);
}

void test_union_tiles(const char* candidate, const char* control) {
    constexpr uint32_t channels = 2560, hidden = 640, experts = 128, selected = 10;
    constexpr float sentinel = 91.25f;
    for (const uint32_t gu_type : {12u, 13u}) {
        // Build bounded generated chunks, then repeat their varied expert codes at distinct weight addresses.
        const auto chunk = build_coloc_fixture(gu_type, 7, channels, hidden, 16, true, false);
        PackedColocPair fx;
        const auto pair_bytes = 2 * make_expert_spec(gu_type, channels, hidden).planar_expert_bytes;
        const auto down_bytes = make_expert_spec(7, hidden, channels).planar_expert_bytes;
        fx.pair_lo.resize(experts / 2 * pair_bytes); fx.pair_hi.resize(fx.pair_lo.size()); fx.down.resize(experts * down_bytes);
        for (uint32_t expert = 0; expert < experts; ++expert) {
            const auto& input = expert % 16 < 8 ? chunk.pair_lo : chunk.pair_hi;
            auto& output = expert < experts / 2 ? fx.pair_lo : fx.pair_hi;
            std::memcpy(output.data() + (expert % (experts / 2)) * pair_bytes, input.data() + (expert % 8) * pair_bytes, pair_bytes);
            std::memcpy(fx.down.data() + expert * down_bytes, chunk.down.data() + (expert % 16) * down_bytes, down_bytes);
        }
        std::vector<float> expected;
        uint64_t expected_bytes = 0;
        for (uint32_t leg = 0; leg < 5; ++leg) {
            const bool use_candidate = leg == 2 || leg == 3;
            MetalGraph g(use_candidate ? candidate : control, 1024ull << 20);
            Arena a{g, {}};
            const auto lo = a.upload(fx.pair_lo), hi = a.upload(fx.pair_hi), down = a.upload(fx.down);
            const auto [lo_kind, hi_kind] = coloc_kinds_for(gu_type);
            g.register_planar_matrix(lo, {gu_type, uint32_t(lo_kind), channels, hidden, experts});
            g.register_planar_matrix(hi, {gu_type, uint32_t(hi_kind), channels, hidden, experts});
            g.register_planar_matrix(down, {7, uint32_t(PlanarKind::q51_planar_v3), hidden, channels, experts});
            g.prepare_planar_moe_v3(gu_type, 7, true, 8, experts, selected, channels, hidden);
            if (leg == 0) expected_bytes = g.memory().live_bytes;
            need(g.memory().live_bytes == expected_bytes, "narrow union changed prepared Metal bytes");
            std::vector<float> captured;
            for (const uint32_t rows : {2u, 4u, 5u, 8u}) for (uint32_t pattern = 0; pattern < 3; ++pattern) {
                const uint32_t input_stride = channels + 4, hidden_stride = hidden + 4, output_stride = channels + 4;
                std::vector<float> x(uint64_t(rows) * input_stride, sentinel);
                std::vector<uint32_t> ids(rows * selected);
                for (uint32_t row = 0; row < rows; ++row) {
                    for (uint32_t k = 0; k < channels; ++k) x[uint64_t(row) * input_stride + k] = float(int((row * 71 + k * 17) % 127) - 63) / 512.0f;
                    for (uint32_t slot = 0; slot < selected; ++slot)
                        ids[row * selected + slot] = pattern == 0 ? row * selected + slot : (pattern == 2 || slot < 2 ? slot : 2 + row * 8 + slot - 2);
                }
                const std::vector<float> h0(uint64_t(rows) * selected * hidden_stride, sentinel), y0(uint64_t(rows) * selected * output_stride, sentinel);
                const auto vx = a.upload(std::as_bytes(std::span(x))), vi = a.upload(std::as_bytes(std::span(ids)));
                const auto vh = a.upload(std::as_bytes(std::span(h0)), false), vy = a.upload(std::as_bytes(std::span(y0)), false);
                const auto execute = [&](uint32_t iterations) {
                    g.begin();
                    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
                        g.planar_moe_routed_m8(lo, hi, down, vi, {vx, rows, channels, uint64_t(input_stride) * 4},
                            {vh, rows * selected, hidden, uint64_t(hidden_stride) * 4}, {vy, rows * selected, channels, uint64_t(output_stride) * 4}, experts, selected);
                    const auto batch = g.submit();
                    need(batch.dispatches == 3u * iterations, "narrow union changed dispatch count");
                    return batch;
                };
                static_cast<void>(execute(1));
                for (const auto& [view, stride, columns] : {std::tuple{vh, hidden_stride, hidden}, std::tuple{vy, output_stride, channels}}) {
                    std::vector<float> values(uint64_t(rows) * selected * stride);
                    g.read(view, std::as_writable_bytes(std::span(values)));
                    for (uint32_t row = 0; row < rows * selected; ++row) for (uint32_t col = 0; col < stride; ++col) {
                        const auto value = values[uint64_t(row) * stride + col];
                        need(std::isfinite(value) && (col < columns || value == sentinel), "narrow union changed padding or produced nonfinite output");
                    }
                    captured.insert(captured.end(), values.begin(), values.end());
                }
                if (leg) {
                    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(120);
                    do { static_cast<void>(execute(16)); } while (std::chrono::steady_clock::now() < until);
                    for (uint32_t trial = 0; trial < 5; ++trial) {
                        const auto batch = execute(16);
                        std::cout << "{\"event\":\"union_tile_component\",\"arm\":\"" << (use_candidate ? "narrow" : "control")
                            << "\",\"leg\":" << leg << ",\"gu_type\":" << gu_type << ",\"rows\":" << rows << ",\"pattern\":" << pattern
                            << ",\"trial\":" << trial << ",\"iterations\":16,\"batch_gpu_ms\":" << batch.gpu_ms << ",\"live_metal_bytes\":" << g.memory().live_bytes << "}\n";
                    }
                }
                for (const auto view : {vx, vi, vh, vy}) { g.release(view); std::erase_if(a.views, [&](GpuView v) { return v.slot == view.slot; }); }
            }
            if (leg == 0) expected = std::move(captured);
            else need(captured.size() == expected.size() && std::memcmp(captured.data(), expected.data(), captured.size() * sizeof(float)) == 0, "narrow union changed ordered output bytes");
        }
    }
    std::cout << "PASS union tiles exact outputs, padding, fixed routes and generated component ABBA\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[2], "--narrow-rows-control") == 0) {
        std::vector<float> reference, candidate; std::vector<uint64_t> reference_resources, candidate_resources;
        test_gpu_isolation_and_numerical_parity(argv[3], &reference, &reference_resources);
        test_gpu_isolation_and_numerical_parity(argv[1], &candidate, &candidate_resources);
        need(candidate_resources == reference_resources, "narrow rows changed workspace, allocation or dispatch counts");
        need(candidate.size() == reference.size() && !reference.empty() &&
            std::memcmp(candidate.data(), reference.data(), candidate.size() * sizeof(float)) == 0,
            "narrow rows changed complete represented hidden or output values");
        for (const auto* library : {argv[3], argv[1]}) for (uint32_t rows : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
            test_sticky_finite_contract(library, 12, 7, rows);
            test_sticky_finite_contract(library, 13, 7, rows);
            test_sticky_finite_contract(library, 13, 8, rows);
            test_sticky_finite_contract(library, 8, 8, rows);
        }
        std::cout << "PASS narrow-row exact complete outputs, all widths/encodings, unchanged resources and finite guards\n";
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[2], "--union-tiles-control") == 0) {
        test_union_tiles(argv[1], argv[3]);
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[2], "--nax-staging-only") == 0) {
        static_cast<void>(test_nax_staging(argv[1]));
        test_nax_staging_finite(argv[1]);
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[2], "--nax-gu-stage64-control") == 0) {
        test_nax_staging_finite(argv[3]);
        test_nax_staging_finite(argv[1]);
        std::vector<uint64_t> reference_resources;
        const auto reference = test_nax_staging(argv[3], "column32", 0u, &reference_resources, true);
        for (uint32_t leg = 1; leg <= 4u; ++leg) {
            const bool candidate = leg == 2u || leg == 3u;
            std::vector<uint64_t> actual_resources;
            const auto actual = test_nax_staging(argv[candidate ? 1 : 3], candidate ? "gu_stage64" : "column32", leg, &actual_resources, true);
            need(actual_resources == reference_resources, "GU stage64 changed grouped scratch bytes or allocation counts");
            need(actual.size() == reference.size() && std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) == 0,
                "GU stage64 candidate/control output bytes differ");
        }
        std::cout << "PASS GU stage64 exact output parity, unchanged grouped resources and nominal component ABBA\n";
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[2], "--nax-staging-control") == 0) {
        test_nax_staging_finite(argv[3]);
        test_nax_staging_finite(argv[1]);
        const auto reference = test_nax_staging(argv[3], "reference", 0u);
        for (uint32_t leg = 1; leg <= 4u; ++leg) {
            const bool candidate = leg == 2u || leg == 3u;
            const auto actual = test_nax_staging(argv[candidate ? 1 : 3], candidate ? "candidate" : "reference", leg);
            need(actual.size() == reference.size() && std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) == 0,
                "planar staging candidate/reference output bytes differ");
        }
        std::cout << "PASS planar staging exact candidate/reference output parity and nominal component ABBA\n";
        return 0;
    }
    if (argc != 2) {
        std::cerr << "usage: test-planar_pack_v3 <tokspur-owned.metallib> [--narrow-rows-control <reference.metallib> | --nax-staging-only | --nax-staging-control <reference.metallib> | --nax-gu-stage64-control <column32.metallib> | --union-tiles-control <reference.metallib>]\n";
        return 2;
    }
    test_sm16_bijection_and_range_refusal();
    test_cpu_pack_unpack_and_exact_operands();
    test_pack_v3_builder_adoption_and_tamper();
    test_gpu_isolation_and_numerical_parity(argv[1]);
    test_sticky_finite_contract(argv[1]);
    static_cast<void>(test_nax_staging(argv[1]));
    std::cout << "PASS test-planar_pack_v3 (" << total_checks << " checks)\n";
    return 0;
}
