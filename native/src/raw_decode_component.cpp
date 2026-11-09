// The packed-product routines adapt llama.cpp 6050b467 mul_mv.metal. See native/third_party/llama-cpp/LICENSE.
#include "tokspur/raw_decode.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tokspur::raw_decode {
namespace {
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
std::uint64_t extent(std::uint64_t count, std::uint64_t stride, std::uint64_t width) {
    require(count && stride >= width && !(stride % 4), "invalid row stride");
    require((count - 1) <= (std::numeric_limits<std::uint64_t>::max() - width) / stride, "extent overflow");
    return (count - 1) * stride + width;
}
std::uint32_t random(std::uint32_t& state) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
}

std::uint32_t block_columns(Format f) {
    if (f == Format::q4_k) return 256;
    if (f == Format::q5_1) return 32;
    throw std::invalid_argument("unsupported raw encoding");
}
std::uint32_t block_bytes(Format f) { return block_columns(f) == 256 ? 144 : 24; }

Plan make_plan(Format f, std::uint32_t k, std::uint32_t r, std::uint32_t e, std::uint32_t s, bool broadcast,
    std::uint64_t input_stride, std::uint64_t output_stride) {
    const auto qk = block_columns(f);
    require(k && k <= 65536 && !(k % qk), "columns must be complete quant blocks");
    require(r && r <= 1000000 && e && e <= 4096 && s && s <= 64, "invalid rows or route geometry");
    const std::uint64_t row_bytes = std::uint64_t(k / qk) * block_bytes(f);
    const std::uint64_t expert_stride = row_bytes * r;
    require(expert_stride <= matrix_limit / e, "matrix budget exceeded");
    if (!input_stride) input_stride = std::uint64_t(k) * 4;
    if (!output_stride) output_stride = std::uint64_t(r) * 4;
    const auto input_bytes = extent(broadcast ? 1 : s, input_stride, std::uint64_t(k) * 4);
    const auto output_bytes = extent(s, output_stride, std::uint64_t(r) * 4);
    require(input_bytes <= vector_limit && output_bytes <= vector_limit, "vector budget exceeded");
    const std::uint32_t rows_per_group = f == Format::q4_k ? 4 : 8;
    return {f, {k, r, e, s, std::uint32_t(row_bytes), broadcast ? 1u : 0u, 0, 0, input_stride, output_stride,
        expert_stride, expert_stride * e}, input_bytes, output_bytes, std::uint64_t(s) * 4,
        (r + rows_per_group - 1) / rows_per_group, s, 64,
        f == Format::q4_k ? "raw_decode_q4_k" : "raw_decode_q5_1"};
}

void validate_bindings(const Plan& p, std::uint64_t w, std::uint64_t x, std::uint64_t y, std::uint64_t ids,
    std::uint64_t wo, std::uint64_t xo, std::uint64_t yo, std::uint64_t io) {
    const auto expected = make_plan(p.format, p.args.columns, p.args.rows, p.args.experts, p.args.selected,
        p.args.broadcast != 0, p.args.input_stride, p.args.output_stride);
    require(p.args.broadcast <= 1 && !p.args.reserved0 && !p.args.reserved1, "noncanonical arguments");
    require(p.args.row_bytes == expected.args.row_bytes && p.args.expert_stride == expected.args.expert_stride &&
        p.args.matrix_bytes == expected.args.matrix_bytes && p.input_bytes == expected.input_bytes &&
        p.output_bytes == expected.output_bytes && p.index_bytes == expected.index_bytes &&
        p.grid_x == expected.grid_x && p.grid_y == expected.grid_y && p.threads == expected.threads &&
        p.kernel && !std::strcmp(p.kernel, expected.kernel),
        "noncanonical plan");
    const auto fit = [](std::uint64_t n, std::uint64_t offset, std::uint64_t need) {
        return !(offset % 4) && offset <= n && need <= n - offset;
    };
    require(fit(w, wo, p.args.matrix_bytes) && fit(x, xo, p.input_bytes) && fit(y, yo, p.output_bytes) &&
        fit(ids, io, p.index_bytes), "short or misaligned buffer");
}

std::uint16_t read16(const std::uint8_t* p) { return std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8); }
void write16(std::uint8_t* p, std::uint16_t v) { p[0] = std::uint8_t(v); p[1] = std::uint8_t(v >> 8); }
std::uint32_t read32(const std::uint8_t* p) { return read16(p) | (std::uint32_t(read16(p + 2)) << 16); }
void write32(std::uint8_t* p, std::uint32_t v) { write16(p, std::uint16_t(v)); write16(p + 2, std::uint16_t(v >> 16)); }
float half_value(std::uint16_t bits) {
    const unsigned e = (bits >> 10) & 31, m = bits & 1023;
    const float magnitude = e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
        : e == 0 ? std::ldexp(float(m), -24) : std::ldexp(float(1024 + m), int(e) - 25);
    return bits & 32768 ? -magnitude : magnitude;
}

void set_q4_group(std::uint8_t* b, unsigned g, unsigned scale, unsigned minimum) {
    require(g < 8 && scale < 64 && minimum < 64, "invalid Q4_K group");
    auto* s = b + 4;
    if (g < 4) { s[g] = std::uint8_t((s[g] & 192) | scale); s[g + 4] = std::uint8_t((s[g + 4] & 192) | minimum); }
    else {
        const auto j = g - 4;
        s[j] = std::uint8_t((s[j] & 63) | ((scale >> 4) << 6));
        s[j + 4] = std::uint8_t((s[j + 4] & 63) | ((minimum >> 4) << 6));
        s[j + 8] = std::uint8_t((scale & 15) | ((minimum & 15) << 4));
    }
}

void set_quant(Format f, std::uint8_t* b, unsigned c, unsigned value) {
    require(c < block_columns(f) && value < (f == Format::q4_k ? 16u : 32u), "invalid quant");
    if (f == Format::q4_k) {
        const auto at = 16 + (c / 64) * 32 + c % 32, shift = 4 * ((c % 64) / 32);
        b[at] = std::uint8_t((b[at] & ~(15u << shift)) | (value << shift));
    } else {
        const auto at = 8 + c % 16, shift = 4 * (c / 16);
        b[at] = std::uint8_t((b[at] & ~(15u << shift)) | ((value & 15) << shift));
        const auto mask = 1u << c;
        write32(b + 4, (read32(b + 4) & ~mask) | ((value >> 4) << c));
    }
}

double represented_weight(Format f, const std::uint8_t* b, unsigned c) {
    require(c < block_columns(f), "column outside block");
    const double d = half_value(read16(b)), m = half_value(read16(b + 2));
    if (f == Format::q5_1) {
        const auto q = ((b[8 + c % 16] >> (4 * (c / 16))) & 15) | (((read32(b + 4) >> c) & 1) << 4);
        return d * q + m;
    }
    const unsigned g = c / 32;
    const auto* s = b + 4;
    const unsigned scale = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
    const unsigned minimum = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
    const unsigned q = (b[16 + (c / 64) * 32 + c % 32] >> (4 * ((c % 64) / 32))) & 15;
    return d * scale * q - m * minimum;
}

double dot_f64(Format f, const std::uint8_t* row, std::span<const float> x) {
    require(!(x.size() % block_columns(f)), "partial block");
    double sum = 0;
    for (std::size_t c = 0; c < x.size(); ++c)
        sum += double(float(represented_weight(f, row + (c / block_columns(f)) * block_bytes(f), unsigned(c % block_columns(f))))) * x[c];
    return sum;
}

float dot_source_order(Format f, const std::uint8_t* row, std::span<const float> x) {
    require(!(x.size() % block_columns(f)), "partial block");
    std::array<float, 32> lanes{};
    for (unsigned lane = 0; lane < 32; ++lane) {
        if (f == Format::q5_1) {
            const unsigned il = (lane % 2) * 8;
            for (std::size_t ib = lane / 2; ib < x.size() / 32; ib += 16) {
                const auto* b = row + ib * 24;
                const auto* y = x.data() + ib * 32 + il;
                float yl[16], sy0 = 0, sy1 = 0, acc[4]{};
                for (unsigned i = 0; i < 8; i += 2) {
                    sy0 += y[i] + y[i + 1]; sy1 += y[i + 16] + y[i + 17];
                    yl[i] = y[i]; yl[i + 1] = y[i + 1] / 256.f;
                    yl[i + 8] = y[i + 16] / 16.f; yl[i + 9] = y[i + 17] / 4096.f;
                }
                const auto qh = read32(b + 4);
                for (unsigned i = 0; i < 8; i += 2) {
                    const auto qs = read16(b + 8 + il + i);
                    acc[0] += yl[i] * ((qs & 0x000f) | ((qh >> (i + il) << 4) & 0x10));
                    acc[1] += yl[i + 1] * ((qs & 0x0f00) | ((qh >> (i + 1 + il) << 12) & 0x1000));
                    acc[2] += yl[i + 8] * ((qs & 0x00f0) | ((qh >> (i + il + 16) << 8) & 0x100));
                    acc[3] += yl[i + 9] * ((qs & 0xf000) | ((qh >> (i + 1 + il + 16) << 16) & 0x10000));
                }
                lanes[lane] += half_value(read16(b)) * (acc[0] + acc[1] + acc[2] + acc[3]) +
                    (sy0 + sy1) * half_value(read16(b + 2));
            }
        } else {
            const unsigned iq = (lane % 8) / 4, ir = lane % 4;
            for (std::size_t ib = lane / 8; ib < x.size() / 256; ib += 4) {
                const auto* b = row + ib * 144;
                const auto* y = x.data() + ib * 256 + 64 * iq + 8 * ir;
                float yl[16], yh[16], sy[4]{}, a[4]{}, h[4]{};
                for (unsigned i = 0; i < 8; ++i) {
                    yl[i] = y[i]; sy[0] += yl[i]; yl[i + 8] = y[i + 32]; sy[1] += yl[i + 8];
                    yh[i] = y[i + 128]; sy[2] += yh[i]; yh[i + 8] = y[i + 160]; sy[3] += yh[i + 8];
                }
                const auto* s = b + 4 + 2 * iq;
                const std::uint16_t packed[4] = {std::uint16_t(read16(s) & 0x3f3f), std::uint16_t(read16(s + 4) & 0x3f3f),
                    std::uint16_t((read16(s + 8) & 0x0f0f) | ((read16(s) & 0xc0c0) >> 2)),
                    std::uint16_t(((read16(s + 8) >> 4) & 0x0f0f) | ((read16(s + 4) & 0xc0c0) >> 2))};
                unsigned sc[8]; for (unsigned j = 0; j < 4; ++j) { sc[2*j] = packed[j] & 255; sc[2*j+1] = packed[j] >> 8; }
                const auto* q = b + 16 + 32 * iq + 8 * ir;
                for (unsigned i = 0; i < 4; ++i) {
                    const auto q1 = read16(q + 2*i), q2 = read16(q + 64 + 2*i);
                    a[0] += yl[2*i] * (q1 & 0x000f); a[1] += yl[2*i+1] * (q1 & 0x0f00);
                    a[2] += yl[2*i+8] * (q1 & 0x00f0); a[3] += yl[2*i+9] * (q1 & 0xf000);
                    h[0] += yh[2*i] * (q2 & 0x000f); h[1] += yh[2*i+1] * (q2 & 0x0f00);
                    h[2] += yh[2*i+8] * (q2 & 0x00f0); h[3] += yh[2*i+9] * (q2 & 0xf000);
                }
                lanes[lane] += half_value(read16(b)) * ((a[0] + a[1]/256.f)*sc[0] + (a[2] + a[3]/256.f)*sc[1]/16.f +
                    (h[0] + h[1]/256.f)*sc[4] + (h[2] + h[3]/256.f)*sc[5]/16.f) -
                    half_value(read16(b+2))*(sy[0]*sc[2] + sy[1]*sc[3] + sy[2]*sc[6] + sy[3]*sc[7]);
            }
        }
    }
    // This CPU reduction is a diagnostic formula check, not a claim about Metal's SIMD tree.
    float sum = 0; for (const auto value : lanes) sum += value; return sum;
}

void fill_generated(Format f, const Plan& p, std::span<std::uint8_t> weights, std::span<float> input, std::uint32_t seed) {
    require(weights.size() == p.args.matrix_bytes && input.size_bytes() == p.input_bytes, "generated extent mismatch");
    require(seed != 0, "zero generator seed");
    const unsigned qk = block_columns(f), bytes = block_bytes(f);
    for (std::size_t offset = 0; offset < weights.size(); offset += bytes) {
        auto* b = weights.data() + offset;
        for (unsigned j = 0; j < bytes; ++j) b[j] = 0;
        write16(b, std::uint16_t(0x1400 + random(seed) % 1024));
        write16(b + 2, std::uint16_t((f == Format::q5_1 && (random(seed) & 1) ? 0x8000 : 0) | (0x1000 + random(seed) % 1024)));
        if (f == Format::q4_k) for (unsigned g = 0; g < 8; ++g) set_q4_group(b, g, random(seed) % 64, random(seed) % 64);
        for (unsigned c = 0; c < qk; ++c) set_quant(f, b, c, random(seed) % (f == Format::q4_k ? 16 : 32));
    }
    for (auto& value : input) value = float(int(random(seed) % 1993) - 996) / 997.f;
}

std::vector<std::uint32_t> generated_routes(std::uint32_t selected, std::uint32_t experts) {
    require(selected && experts, "empty routes");
    std::vector<std::uint32_t> result(selected);
    // experts-1 is coprime to experts: target S10/E14 visits ten distinct experts.
    for (std::uint32_t i = 0; i < selected; ++i) result[i] = (std::uint64_t(i) * (experts > 1 ? experts - 1 : 1) + 3) % experts;
    return result;
}
}
