// Bounded TensorOps/dequantisation adaptation of llama.cpp 6050b467.
// Copyright (c) 2023-2026 The ggml authors. MIT notice: ../third_party/llama-cpp/LICENSE.
// Tokspur owns grouping, layout identity, row views, state and scheduling.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using uint8_t = uchar;
using uint16_t = ushort;
using uint32_t = uint;
struct block_q5_1 { half d, m; uchar qh[4], qs[16]; };
struct block_q8_0 { half d; char qs[32]; };
struct block_q4_K { half d, dmin; uchar scales[12], qs[128]; };
struct block_q5_K { half d, dmin; uchar scales[12], qh[32], qs[128]; };
static_assert(sizeof(block_q5_1)==24 && sizeof(block_q8_0)==34 && sizeof(block_q4_K)==144 && sizeof(block_q5_K)==176, "GGUF block bytes");
constant uint mm_encoding [[function_constant(0)]];
constant uint mm_layout [[function_constant(1)]]; // 0 GGUF, 1 Q8 lanes-v1, 2 Q5_1 lanes-v1.
struct GroupedMmArgs {
    uint columns, rows, row_bytes, experts;
    uint selected, tokens, broadcast, maximum_tasks;
    ulong expert_stride_bytes, input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(GroupedMmArgs)==56, "grouped mm ABI");
inline float stored_half(device const uchar* p) { return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1])<<8)))); }
template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }

    reg = (type4x4) reg_f;
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float stored_d = float(xb->d);
    const float d = il < 2 ? stored_d : stored_d / 16.0f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float stored_d = float(xb->d);
    const float d = il < 2 ? stored_d : stored_d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}


inline half4x4 mm_decode16(device const uchar* row, uint ci, uint columns) {
    half4x4 result;
    if (mm_layout==0) {
        if (mm_encoding==7) dequantize_q5_1((device const block_q5_1*)row+ci/2,ci%2,result);
        else if (mm_encoding==8) dequantize_q8_0((device const block_q8_0*)row+ci/2,ci%2,result);
        else if (mm_encoding==12) dequantize_q4_K((device const block_q4_K*)row+ci/16,ci%16,result);
        else dequantize_q5_K((device const block_q5_K*)row+ci/16,ci%16,result);
        return result;
    }
    const uint blocks=columns/32, column=16*ci, b=column/32;
    if (mm_layout==1) {
        const float d=stored_half(row+2*b); const uint run=b/16, width=min(16u,blocks-run*16);
        for (uint i=0;i<16;++i) {
            const uint qindex=(column+i)%32;
            const ulong at=2*blocks+ulong(run)*512+qindex*width+b%16;
            result[i/4][i%4]=half(d*float(as_type<char>(row[at])));
        }
    } else {
        const float d=stored_half(row+8*b), m=stored_half(row+8*b+2);
        device const uchar* h=row+8*b+4;
        const uint qh=uint(h[0])|(uint(h[1])<<8)|(uint(h[2])<<16)|(uint(h[3])<<24);
        const uint run=b/16, width=min(16u,blocks-run*16);
        for (uint i=0;i<16;++i) {
            const uint qi=(column+i)%32;
            const ulong at=8*blocks+ulong(run)*256+(qi%16)*width+b%16;
            const uint q=((uint(row[at])>>(4*(qi/16)))&15)|(((qh>>qi)&1)<<4);
            result[i/4][i%4]=half(d*float(q)+m);
        }
    }
    return result;
}
inline uint mm_route(device const uint* ids, device const uint* map, uint expert, uint at,
    constant GroupedMmArgs& a, device atomic_uint* error) {
    const uint rid=map[ulong(expert)*a.tokens+at];
    if (rid>=a.tokens*a.selected || ids[rid]!=expert) { atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return UINT_MAX; }
    return rid;
}
template<bool Dense>
inline void mm_tile(device const uchar* weights, device const float* input, device float* output,
    constant GroupedMmArgs& a, device const uint* ids, device const uint* map, device const uint* tasks,
    device const uint* counts, device atomic_uint* error, uint2 group, ushort tid, ushort lane, ushort sg,
    threadgroup half* shared) {
    constexpr uint NK=32, NR0=64, NR1=32, HALF_ROWS=16;
    uint expert=0,start=group.y*NR1,n=0;
    if (Dense) {
        if (start>=a.tokens) return; n=min(NR1,a.tokens-start);
    } else {
        if (tasks[0]>a.maximum_tasks || tasks[1]!=a.tokens*a.selected) { if(!tid) atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return; }
        if (group.y>=tasks[0]) return;
        expert=tasks[2+2*group.y]; start=tasks[3+2*group.y];
        if (expert>=a.experts || start>=a.tokens || counts[expert]>a.tokens || start>=counts[expert]) { if(!tid) atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return; }
        n=min(NR1,counts[expert]-start);
    }
    const uint first=group.x*NR0;
    if (first>=a.rows) return;
    const uint nr0=min(NR0,a.rows-first), ar=min(uint(tid/2),nr0-1), ai=tid%2;
    const uint br=min(uint(tid/4),n-1), bi=8*(tid%4);
    uint rid=Dense?start+br:mm_route(ids,map,expert,start+br,a,error);
    const uint input_row=rid==UINT_MAX?0:(a.broadcast?rid/a.selected:rid);
    device const uchar* w=weights+ulong(expert)*a.expert_stride_bytes+ulong(first+ar)*a.row_bytes;
    device const float* x=(device const float*)((device const uchar*)input+ulong(input_row)*a.input_stride_bytes);
    threadgroup half* sa=shared; threadgroup half* sb=shared+2048;
    auto tA=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sa,dextents<int32_t,2>(NK,NR0));
    auto tB0=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sb,dextents<int32_t,2>(NK,HALF_ROWS));
    auto tB1=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sb+HALF_ROWS*NK,dextents<int32_t,2>(NK,HALF_ROWS));
    constexpr auto descriptor=matmul2d_descriptor(HALF_ROWS,NR0,NK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "declared half inputs and FP32 accumulation");
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto c0=op.template get_destination_cooperative_tensor<decltype(tA),decltype(tB0),float>();
    auto c1=op.template get_destination_cooperative_tensor<decltype(tA),decltype(tB1),float>();
    for(auto i=c0.begin();i!=c0.end();++i)*i=0;
    for(auto i=c1.begin();i!=c1.end();++i)*i=0;
    const bool has_hi=n>HALF_ROWS;
    for(uint k=0;k<a.columns;k+=NK) {
        const half4x4 decoded=mm_decode16(w,k/16+ai,a.columns);
        for(uint i=0;i<16;++i) {
            const half value=decoded[i/4][i%4];
            if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            sa[(tid/2)*NK+ai*16+i]=value;
        }
        for(uint i=0;i<8;++i) {
            const float value=rid==UINT_MAX?0:x[k+bi+i]; const half h=half(value);
            if (!isfinite(value)||!isfinite(h)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            sb[(tid/4)*NK+bi+i]=h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto left=tA.slice(0,0), low=tB0.slice(0,0);
        op.run(low,left,c0);
        if(has_hi) { auto high=tB1.slice(0,0); op.run(high,left,c1); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* sc=(threadgroup float*)shared;
    auto out0=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(sc,dextents<int32_t,2>(NR0,HALF_ROWS));
    c0.store(out0);
    if(has_hi) {
        auto out1=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(sc+HALF_ROWS*NR0,dextents<int32_t,2>(NR0,HALF_ROWS));
        c1.store(out1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=sg;j<n;j+=4) {
        const uint dest=Dense?start+j:mm_route(ids,map,expert,start+j,a,error);
        if(dest==UINT_MAX)continue;
        device float* y=(device float*)((device uchar*)output+ulong(dest)*a.output_stride_bytes);
        for(uint r=lane;r<nr0;r+=32) {
            const float value=sc[j*NR0+r];
            if(!isfinite(value))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            y[first+r]=value;
        }
    }
}
kernel void graph_grouped_mm(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],
    constant GroupedMmArgs& a [[buffer(3)]],device const uint* ids [[buffer(4)]],device const uint* map [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]],device const uint* tasks [[buffer(8)]],device const uint* counts [[buffer(9)]],
    threadgroup half* shared [[threadgroup(0)]],uint2 group [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {
    mm_tile<false>(w,x,y,a,ids,map,tasks,counts,error,group,tid,lane,sg,shared);
}
kernel void graph_dense_mm_q8(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],
    constant GroupedMmArgs& a [[buffer(3)]],device atomic_uint* error [[buffer(7)]],threadgroup half* shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {
    mm_tile<true>(w,x,y,a,nullptr,nullptr,nullptr,nullptr,error,group,tid,lane,sg,shared);
}
template<ushort N>
void q8_serial_tile(device const uchar* weights, device const float* input, device float* output,
    constant uint4& s, device atomic_uint* error, uint group, ushort subgroup, ushort lane) {
    const uint row = group * 4 + subgroup;
    if (row >= s.y) return;
    device const uchar* data = weights + ulong(row) * s.z;
    float sums[N];
    for (ushort token = 0; token < N; ++token) sums[token] = 0;
    for (uint block = 0; block < s.x / 32; ++block) {
        device const uchar* encoded = data + ulong(block) * 34;
        const float scale = float(*reinterpret_cast<device const half*>(encoded));
        const float quantized = float(*reinterpret_cast<device const char*>(encoded + 2 + lane));
        const float weight = scale * quantized;
        for (ushort token = 0; token < N; ++token)
            sums[token] += weight * input[ulong(token) * s.x + block * 32 + lane];
    }
    for (ushort token = 0; token < N; ++token) {
        const float total = simd_sum(sums[token]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[ulong(token) * s.y + row] = total;
        }
    }
}
kernel void graph_q8_serial_batch(device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]], device float* output [[buffer(2)]],
    constant uint4& shape [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint base = 0;
    uint left = shape.w;
    while (left) {
        if (left >= 8) {
            q8_serial_tile<8>(weights, input + ulong(base) * shape.x, output + ulong(base) * shape.y, shape, error, group, subgroup, lane);
            base += 8; left -= 8;
        } else if (left >= 4) {
            q8_serial_tile<4>(weights, input + ulong(base) * shape.x, output + ulong(base) * shape.y, shape, error, group, subgroup, lane);
            base += 4; left -= 4;
        } else if (left >= 2) {
            q8_serial_tile<2>(weights, input + ulong(base) * shape.x, output + ulong(base) * shape.y, shape, error, group, subgroup, lane);
            base += 2; left -= 2;
        } else {
            q8_serial_tile<1>(weights, input + ulong(base) * shape.x, output + ulong(base) * shape.y, shape, error, group, subgroup, lane);
            base += 1; left -= 1;
        }
    }
}
inline uint serial_dword(device const uchar* data, uint offset) {
    return uint(data[offset]) | (uint(data[offset + 1]) << 8) | (uint(data[offset + 2]) << 16) | (uint(data[offset + 3]) << 24);
}
inline float serial_half(device const uchar* data, uint offset) {
    return float(as_type<half>(ushort(uint(data[offset]) | (uint(data[offset + 1]) << 8))));
}
inline float serial_element(device const uchar* row, uint column) {
    if (mm_encoding == 8) {
        device const uchar* block = row + (column / 32) * 34;
        return serial_half(block, 0) * float(as_type<char>(block[2 + column % 32]));
    }
    if (mm_encoding == 7) {
        device const uchar* block = row + (column / 32) * 24;
        const uint i = column % 32;
        const uint low = (uint(block[8 + i % 16]) >> (4 * (i / 16))) & 15;
        const uint q = low | (((serial_dword(block, 4) >> i) & 1) << 4);
        return serial_half(block, 0) * float(q) + serial_half(block, 2);
    }
    const uint block_bytes = mm_encoding == 12 ? 144 : 176;
    device const uchar* block = row + (column / 256) * block_bytes;
    const uint i = column % 256;
    const uint group = i / 32;
    uint scale, minimum;
    if (group < 4) {
        scale = uint(block[4 + group]) & 63;
        minimum = uint(block[8 + group]) & 63;
    } else {
        const uint slot = group - 4;
        scale = (uint(block[12 + slot]) & 15) | ((uint(block[4 + slot]) >> 6) << 4);
        minimum = (uint(block[12 + slot]) >> 4) | ((uint(block[8 + slot]) >> 6) << 4);
    }
    const uint payload = mm_encoding == 12 ? 16 : 48;
    uint q = (uint(block[payload + (i / 64) * 32 + i % 32]) >> (4 * ((i % 64) / 32))) & 15;
    if (mm_encoding == 13) q |= ((uint(block[16 + i % 32]) >> (i / 32)) & 1) << 4;
    return serial_half(block, 0) * float(scale) * float(q) - serial_half(block, 2) * float(minimum);
}
kernel void graph_grouped_serial(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant GroupedMmArgs& a [[buffer(3)]], device const uint* ids [[buffer(4)]],
    device const uint* map [[buffer(6)]], device atomic_uint* error [[buffer(7)]], device const uint* tasks [[buffer(8)]],
    device const uint* counts [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]],
    ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if (mm_layout != 0) { if (!lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return; }
    if (tasks[0] > a.maximum_tasks || tasks[1] != a.tokens * a.selected) { if (!lane && !subgroup) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return; }
    const uint row = group.x * 4 + subgroup;
    if (group.y >= tasks[0] || row >= a.rows) return;
    const uint expert = tasks[2 + 2 * group.y], start = tasks[3 + 2 * group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return;
    }
    const uint n = min(32u, counts[expert] - start);
    device const uchar* data = weights + ulong(expert) * a.expert_stride_bytes + ulong(row) * a.row_bytes;
    for (uint base = 0; base < n; base += 8) {
        const uint width = min(8u, n - base);
        uint dests[8];
        bool live[8];
        float sums[8];
        for (uint i = 0; i < 8; ++i) { dests[i] = 0; live[i] = false; sums[i] = 0; }
        for (uint i = 0; i < width; ++i) {
            const uint dest = mm_route(ids, map, expert, start + base + i, a, error);
            live[i] = dest != UINT_MAX; dests[i] = live[i] ? dest : 0;
        }
        for (uint column = lane; column < a.columns; column += 32) {
            const float weight = serial_element(data, column);
            for (uint i = 0; i < width; ++i) {
                if (!live[i]) continue;
                const uint input_row = a.broadcast ? dests[i] / a.selected : dests[i];
                device const float* x = (device const float*)((device const uchar*)input + ulong(input_row) * a.input_stride_bytes);
                sums[i] += weight * x[column];
            }
        }
        for (uint i = 0; i < width; ++i) {
            if (!live[i]) continue;
            const float total = simd_sum(sums[i]);
            if (!lane) {
                if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                device float* y = (device float*)((device uchar*)output + ulong(dests[i]) * a.output_stride_bytes);
                y[row] = total;
            }
        }
    }
}
