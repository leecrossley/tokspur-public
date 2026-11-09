#pragma once

#include "tokspur/gdn_chunk.hpp"
#include "tokspur/gguf.hpp"

namespace tokspur {

// Recurrence-only ablation. Borrows the fused chunk's current log-decay, row strides and state views. Root owns Graph dispatch and selector integration.
inline constexpr const char* gdn_legacy_batch_kernel = "gdn_chunk_legacy_batch";
GdnChunkSpec gdn_legacy_batch_spec(const GdnChunkDispatch&);
void validate_gdn_legacy_batch_dispatch(const GdnChunkDispatch&);

} // namespace tokspur
