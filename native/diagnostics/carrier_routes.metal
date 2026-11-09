#include <metal_stdlib>
using namespace metal;

// Diagnostic state inspection only. Copy the exact requested byte range;
// two-byte cache fields must not cause reads of neighbouring state or padding.
kernel void carrier_readback_bytes(device const uchar* source [[buffer(0)]],
                                   device uchar* output [[buffer(1)]],
                                   constant ulong& offset [[buffer(2)]],
                                   constant uint& count [[buffer(3)]],
                                   constant ulong& source_bytes [[buffer(4)]],
                                   constant ulong& output_offset [[buffer(5)]],
                                   constant ulong& source_begin [[buffer(6)]],
                                   device atomic_uint* error [[buffer(7)]],
                                   uint i [[thread_position_in_grid]]) {
    if (atomic_load_explicit(error, memory_order_relaxed)) return;
    if (!count || count > 16777216u || offset < source_begin || offset > source_bytes || ulong(count) > source_bytes - offset ||
        output_offset > 16777216ul || ulong(count) > 16777216ul - output_offset) {
        atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return;
    }
    if (i < count) output[output_offset + i] = source[offset + i];
}

// Used only by the opt-in correctness recorder. Integer copies preserve the
// router's complete represented IDs and weights before its scratch is reused.
kernel void graph_capture_routes(device const uint* ids [[buffer(0)]],
                                 device const uint* weights [[buffer(1)]],
                                 device uint* output [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 device atomic_uint* error [[buffer(7)]],
                                 uint i [[thread_position_in_grid]]) {
    if (atomic_load_explicit(error, memory_order_relaxed)) return;
    if (!count || count > 512) { atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return; }
    if (i < count) { output[i] = ids[i]; output[count + i] = weights[i]; }
}
