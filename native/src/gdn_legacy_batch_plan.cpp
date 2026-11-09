#include "tokspur/gdn_legacy_batch.hpp"

namespace tokspur {

GdnChunkSpec gdn_legacy_batch_spec(const GdnChunkDispatch& d) {
    if (d.op != GdnChunkOp::fused) throw FormatError("legacy-order GDN batch requires the fused recurrence ABI");
    const auto& p = d.params;
    if (!p.heads || p.heads > 48 || !p.value_width || p.value_width > 128)
        throw FormatError("legacy-order GDN batch geometry");
    return {gdn_legacy_batch_kernel, 0x3f, 0x140, {(p.heads * p.value_width + 127) / 128, 1, 1}, 128};
}

void validate_gdn_legacy_batch_dispatch(const GdnChunkDispatch& d) {
    static_cast<void>(gdn_legacy_batch_spec(d));
    // The existing fused ABI already proves active strided row extents, exact [head,key,value] state extents, reserved flag bindings and disjoint live writes.
    validate_gdn_chunk_dispatch(d);
}

} // namespace tokspur
