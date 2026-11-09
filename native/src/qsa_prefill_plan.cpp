#include "tokspur/qsa_prefill_plan.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>
#include <limits>

namespace tokspur {
namespace {
void need(bool okay, const char* why) { if (!okay) throw FormatError(why); }
}
QsaPrefillPlan qsa_prefill_plan(uint32_t base, uint32_t rows, uint32_t capacity, QsaShape s, bool allow_nax_2k) {
    need(s.channels && s.channels <= 2560 && s.query_heads && s.query_heads <= 24 && s.kv_heads &&
        s.query_heads % s.kv_heads == 0 && s.head_width && s.head_width <= 256 && s.head_width % 2 == 0 &&
        s.index_heads && s.index_heads <= 8 && s.index_width && s.index_width <= 256 && s.index_width % 2 == 0 &&
        s.index_heads * s.index_width <= 512 &&
        s.rotary_width && s.rotary_width % 2 == 0 && s.rotary_width <= std::min(s.head_width,s.index_width) &&
        s.compression == 4 && s.token_budget && s.token_budget % 4 == 0 && s.token_budget / 4 <= 512 &&
        s.capacity && s.capacity <= 65536, "unsupported batched QSA geometry");
    const uint32_t max_cap = allow_nax_2k ? 2048u : qsa_prefill_capacity;
    need(capacity && capacity <= max_cap && rows && rows <= capacity && base < s.capacity &&
        rows <= s.capacity - base, "batched QSA rows exceed the accepted cache frontier");
    const uint32_t blocks = (s.capacity + 3) / 4, budget = s.token_budget / 4;
    const uint32_t selected_stride = ((blocks + 511) / 512) * budget;
    // This includes the largest earlier tail when a later row completes a pool.
    const uint32_t tokens = std::min(s.token_budget + 3, base + rows);
    const uint32_t tiles = (tokens + qsa_prefill_token_tile - 1) / qsa_prefill_token_tile;
    const uint32_t stripe = std::min(rows,qsa_prefill_query_stripe);
    const uint64_t kv_stride = uint64_t(s.kv_heads) * s.head_width * 2;
    QsaPrefillPlan out{base,rows,capacity,base/4,(base+rows)/4-base/4,(base+rows)%4,
        blocks,budget,selected_stride,tokens,tiles,stripe,
        uint64_t(base)*kv_stride,uint64_t(rows)*kv_stride,
        uint64_t(base/4)*s.index_width*2,uint64_t((base+rows)/4-base/4)*s.index_width*2,
        uint64_t(stripe)*blocks*4,uint64_t(stripe)*selected_stride*4,
        uint64_t(capacity)*budget*4,uint64_t(stripe)*s.query_heads*tiles*(s.head_width+2)*4};
    need(out.pool_count <= (capacity + 3) / 4 && out.maximum_attended &&
        out.maximum_attended <= s.token_budget + 3 && out.attention_partials_bytes <= (64ull << 20),
        "batched QSA scratch bound exceeded");
    return out;
}
uint32_t qsa_nax_indexer_stripe(uint32_t base, uint32_t rows, uint32_t block_budget) {
    const uint32_t chunk_max_blocks = (base + rows) / 4u;
    if (chunk_max_blocks <= block_budget) return std::min(2048u, std::max(32u, rows));
    const uint32_t score_stride = ((chunk_max_blocks + 31u) / 32u) * 32u;
    const uint32_t sel_stride = ((score_stride + 511u) / 512u) * block_budget;
    const uint64_t row_bytes = uint64_t(score_stride + 4u * sel_stride) * 4ull;
    const uint32_t fit = uint32_t(41'943'040ull / row_bytes);
    return std::clamp((fit / 32u) * 32u, 32u, 2048u);
}
uint64_t qsa_nax_attention_dispatches(uint32_t base, uint32_t rows, QsaShape shape) {
    if (shape.query_heads != shape.kv_heads * 12u || shape.kv_heads > 2u) return 1ull;
    const uint32_t num_tiles = (rows + 7u) / 8u;
    const uint32_t block_budget = shape.token_budget / 4u;
    const bool sparse_mode = (base + rows) / 4u > block_budget;
    const uint32_t splits = (num_tiles * shape.kv_heads >= 32u) ? 1u : std::min(16u, std::max(1u, 32u / (num_tiles * shape.kv_heads)));
    return 2ull + (sparse_mode ? 1ull : 0ull) + (splits > 1u ? 1ull : 0ull);
}
uint64_t qsa_prefill_projected_dispatches(uint32_t base, uint32_t rows, QsaShape shape, bool allow_nax_2k, bool nax_qsa, bool fused_prep) {
    const auto p=qsa_prefill_plan(base,rows,rows,shape,allow_nax_2k);
    // Split, three RMS, three RoPE, two BF16 stores, pool, and final sigmoid multiply (or 4 fused prep dispatches + final sigmoid multiply).
    const uint64_t mul_dispatches = nax_qsa ? ((uint64_t(rows) * shape.query_heads * shape.head_width + 16'777'215ull) / 16'777'216ull) : 1ull;
    uint64_t count = (fused_prep ? 4ull : (10ull + (p.pool_count ? 3ull : 0ull))) + mul_dispatches;
    if (nax_qsa) {
        const uint32_t idx_stripe = qsa_nax_indexer_stripe(base, rows, p.block_budget);
        for (uint32_t first = 0; first < rows; first += idx_stripe) {
            const auto stripe_rows = std::min(idx_stripe, rows - first);
            const auto blocks = (base + first + stripe_rows) / shape.compression;
            if (blocks > p.block_budget) {
                count += 3; // Scores, local selection sort and canonical IDs.
                for (uint32_t span = 512; span < blocks; span *= 2) ++count;
            }
        }
        count += qsa_nax_attention_dispatches(base, rows, shape);
    } else {
        for(uint32_t first=0;first<rows;first+=qsa_prefill_query_stripe) {
            const auto stripe_rows=std::min(qsa_prefill_query_stripe,rows-first);
            const auto blocks=(base+first+stripe_rows)/shape.compression;
            count+=3; // Canonical IDs, partial attention and reduction.
            if(blocks>p.block_budget) {
                count+=2; // Scores and local selection sort.
                for(uint32_t span=512;span<blocks;span*=2)++count;
            }
        }
    }
    return count;
}
uint64_t qsa_prefill_dispatches(uint32_t base, uint32_t rows, QsaShape shape, bool allow_nax_2k, bool nax_qsa, bool fused_prep) {
    return qsa_prefill_projected_dispatches(base,rows,shape,allow_nax_2k,nax_qsa,fused_prep)+6;
}
void validate_qsa_prefill_rows(GpuRowsF32 r, uint32_t rows, uint32_t columns, bool allow_nax_2k) {
    const uint32_t max_rows = allow_nax_2k ? 2048u : 512u;
    need(r.rows == rows && r.columns == columns && rows && rows <= max_rows && columns &&
        r.view.owner && r.view.bytes && r.view.offset % 4 == 0 && r.stride_bytes % 4 == 0 &&
        r.stride_bytes >= uint64_t(columns)*4 && r.stride_bytes <= std::numeric_limits<uint32_t>::max(),
        "invalid batched QSA row view");
    need(uint64_t(rows-1)*r.stride_bytes+uint64_t(columns)*4 <= r.view.bytes,
        "batched QSA row view is truncated");
}
} // namespace tokspur
