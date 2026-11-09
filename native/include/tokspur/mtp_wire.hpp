#pragma once
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#ifndef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
#error Native MTP HTTP requires the explicit sampled known-input build.
#endif

#include "tokspur/json.hpp"
#include "tokspur/mtp_anchor_state.hpp"

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace tokspur {
// Retained pre-observation captures mean serial only. Unknown modes always fail.
[[nodiscard]] std::string_view mtp_draft_attention(const Json&);
[[nodiscard]] bool mtp_request_carry(const Json&);
[[nodiscard]] uint32_t mtp_depth(const Json&);
[[nodiscard]] uint32_t mtp_output_packet(const Json&);
void validate_mtp_output_packet_selection(const Json&, std::optional<uint32_t> expected);
[[nodiscard]] uint32_t mtp_input_packet(const Json&);
void validate_mtp_input_packet_selection(const Json&, bool expected_input512);
[[nodiscard]] bool mtp_final_input_logits(const Json&);
void validate_mtp_final_input_selection(const Json&, bool expected);
[[nodiscard]] bool mtp_gdn_old_order(const Json&);
void validate_mtp_gdn_old_order_selection(const Json&, bool expected);
inline constexpr std::array<std::string_view, 11> mtp_hc_down_f32_identity_names{
    "native_prefill_hc_down_f32_schema", "native_prefill_hc_down_f32_version", "native_prefill_hc_down_f32_kernel", "native_prefill_hc_down_f32_precision",
    "native_prefill_hc_down_f32_layout", "native_prefill_hc_down_f32_input_rows", "native_prefill_hc_down_f32_input_columns", "native_prefill_hc_down_f32_output_columns",
    "native_prefill_hc_down_f32_branches", "native_prefill_hc_down_f32_scratch_bytes", "native_prefill_hc_down_f32_counter_scope"};
inline constexpr std::array<std::string_view, 10> mtp_hc_rows_identity_names{
    "native_prefill_hc_rows_schema", "native_prefill_hc_rows_version", "native_prefill_hc_rows_kernels", "native_prefill_hc_rows_precision",
    "native_prefill_hc_rows_min_rows", "native_prefill_hc_rows_max_rows", "native_prefill_hc_rows_branches", "native_prefill_hc_rows_channels",
    "native_prefill_hc_rows_scratch_bytes", "native_prefill_hc_rows_counter_scope"};
inline constexpr std::array<const char*, 3> mtp_hc_rows_counter_names{
    "native_prefill_hc_rows_inject_encoded", "native_prefill_hc_rows_read_encoded", "native_prefill_hc_rows_write_encoded"};
inline constexpr std::array<const char*, 3> mtp_hc_rows_request_names{
    "native_prefill_hc_rows_inject_request_encoded", "native_prefill_hc_rows_read_request_encoded", "native_prefill_hc_rows_write_request_encoded"};
Json::Object mtp_hc_rows_fields(std::array<uint64_t, 3> encoded = {}, std::array<uint64_t, 3> request = {});
[[nodiscard]] bool mtp_hc_rows(const Json&);
void validate_mtp_hc_rows_selection(const Json&, bool expected);
inline constexpr std::array<std::string_view, 11> mtp_router_window_identity_names{
    "native_prefill_router_window_schema", "native_prefill_router_window_version", "native_prefill_router_window_kernels", "native_prefill_router_window_precision",
    "native_prefill_router_window_min_rows", "native_prefill_router_window_max_rows", "native_prefill_router_window_window_rows",
    "native_prefill_router_window_channels", "native_prefill_router_window_experts", "native_prefill_router_window_scratch_bytes", "native_prefill_router_window_counter_scope"};
inline constexpr std::array<const char*, 4> mtp_router_window_counter_names{
    "native_prefill_router_window_router_encoded", "native_prefill_router_window_route_encoded",
    "native_prefill_router_window_shared_router_encoded", "native_prefill_router_window_mixture_encoded"};
inline constexpr std::array<const char*, 4> mtp_router_window_request_names{
    "native_prefill_router_window_router_request_encoded", "native_prefill_router_window_route_request_encoded",
    "native_prefill_router_window_shared_router_request_encoded", "native_prefill_router_window_mixture_request_encoded"};
Json::Object mtp_router_window_fields(std::array<uint64_t, 4> encoded = {}, std::array<uint64_t, 4> request = {});
[[nodiscard]] bool mtp_router_window(const Json&);
void validate_mtp_router_window_selection(const Json&, bool expected);
inline constexpr std::array<std::string_view, 12> mtp_hc_down_window_identity_names{
    "native_prefill_hc_down_window_schema", "native_prefill_hc_down_window_version", "native_prefill_hc_down_window_kernel", "native_prefill_hc_down_window_precision",
    "native_prefill_hc_down_window_min_rows", "native_prefill_hc_down_window_max_rows", "native_prefill_hc_down_window_window_rows",
    "native_prefill_hc_down_window_columns", "native_prefill_hc_down_window_rows", "native_prefill_hc_down_window_threads",
    "native_prefill_hc_down_window_threadgroup_bytes", "native_prefill_hc_down_window_counter_scope"};
inline constexpr const char* mtp_hc_down_window_counter_name = "native_prefill_hc_down_window_encoded";
inline constexpr const char* mtp_hc_down_window_request_name = "native_prefill_hc_down_window_request_encoded";
Json::Object mtp_hc_down_window_fields(uint64_t encoded = 0, uint64_t request = 0);
[[nodiscard]] bool mtp_hc_down_window(const Json&);
void validate_mtp_hc_down_window_selection(const Json&, bool expected);
#ifndef TOKSPUR_QSA_SPARSE_DECODE_BN
#define TOKSPUR_QSA_SPARSE_DECODE_BN 32
#endif
#ifndef TOKSPUR_QSA_SPARSE_VERIFY_BN
#define TOKSPUR_QSA_SPARSE_VERIFY_BN 32
#endif
inline constexpr std::array<std::string_view, 12> mtp_qsa_sparse_identity_names{
    "native_qsa_sparse_schema", "native_qsa_sparse_version", "native_qsa_sparse_decode_kernel", "native_qsa_sparse_verify_kernel",
    "native_qsa_sparse_precision", "native_qsa_sparse_decode_bn", "native_qsa_sparse_verify_bn",
    "native_qsa_sparse_decode_threads", "native_qsa_sparse_verify_threads",
    "native_qsa_sparse_decode_threadgroup_bytes", "native_qsa_sparse_verify_threadgroup_bytes", "native_qsa_sparse_counter_scope"};
inline constexpr std::array<const char*, 2> mtp_qsa_sparse_counter_names{
    "native_qsa_sparse_decode_calls", "native_qsa_sparse_verify_calls"};
inline constexpr std::array<const char*, 2> mtp_qsa_sparse_request_names{
    "native_qsa_sparse_decode_request_calls", "native_qsa_sparse_verify_request_calls"};
Json::Object mtp_qsa_sparse_fields(std::array<uint64_t, 2> completed = {}, std::array<uint64_t, 2> request = {});
[[nodiscard]] bool mtp_qsa_sparse(const Json&);
void validate_mtp_qsa_sparse_selection(const Json&, bool expected);
#ifndef TOKSPUR_GDN_VERIFY_DB
#define TOKSPUR_GDN_VERIFY_DB 8
#endif
inline constexpr std::array<std::string_view, 13> mtp_gdn_ported_identity_names{
    "native_gdn_ported_schema", "native_gdn_ported_version", "native_gdn_ported_prefill_kernel", "native_gdn_ported_prepare_kernel",
    "native_gdn_ported_verify_kernel", "native_gdn_ported_replay_kernel", "native_gdn_ported_norm_gate_kernel", "native_gdn_ported_precision",
    "native_gdn_ported_prefill_tile_rows", "native_gdn_ported_verify_db", "native_gdn_ported_prefill_threadgroup_bytes",
    "native_gdn_ported_verify_threadgroup_bytes", "native_gdn_ported_counter_scope"};
inline constexpr std::array<const char*, 5> mtp_gdn_ported_counter_names{
    "native_gdn_ported_prefill_calls", "native_gdn_ported_decode_calls", "native_gdn_ported_verify_calls",
    "native_gdn_ported_replay_calls", "native_gdn_ported_norm_gate_calls"};
inline constexpr std::array<const char*, 5> mtp_gdn_ported_request_names{
    "native_gdn_ported_prefill_request_calls", "native_gdn_ported_decode_request_calls", "native_gdn_ported_verify_request_calls",
    "native_gdn_ported_replay_request_calls", "native_gdn_ported_norm_gate_request_calls"};
Json::Object mtp_gdn_ported_fields(std::array<uint64_t, 5> completed = {}, std::array<uint64_t, 5> request = {});
[[nodiscard]] bool mtp_gdn_ported(const Json&);
void validate_mtp_gdn_ported_selection(const Json&, bool expected);
Json::Object mtp_hc_down_f32_fields(uint64_t encoded, uint64_t request);
[[nodiscard]] bool mtp_hc_down_f32(const Json&);
void validate_mtp_hc_down_f32_selection(const Json&, bool expected);
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
void validate_mtp_prefill_selection(const Json&, std::optional<std::string> expected_dense = std::nullopt,
                                  std::optional<std::string> expected_grouped = std::nullopt);
#endif
void validate_mtp_ready(const Json&, std::string_view expected_draft_attention = "serial", bool expected_request_carry = false, uint32_t expected_depth = 1);

// Sampled MTP input defaults to eight, with explicit input512 in its isolated build. Output defaults to eight, or explicitly negotiates eight/64. At most twice the output capacity is buffered, and no callback becomes observable before DONE admission. Counter resynchronisation is only for cold known-input reset after cancellation; immutable resources still match.
class MtpWireReply {
public:
    MtpWireReply(const Json& ready, const Json& previous, const Json& job, bool cold_resync = false);
    void event(const Json&);
    void finish(const Json&);
    [[nodiscard]] std::span<const Json> events() const;
private:
    const Json &ready_, &previous_, &job_;
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    static constexpr size_t packet_capacity = 64;
#else
    static constexpr size_t packet_capacity = 8;
#endif
    std::array<Json, packet_capacity * 2> events_{};
    std::array<uint32_t, packet_capacity> ids_{};
    std::array<uint64_t, packet_capacity> token_words_{};
    size_t event_count_ = 0, count_ = 0;
    uint32_t base_ = 0, position_ = 0, inputs_ = 0, maximum_ = 0;
    uint32_t rounds_ = 0, accepted_ = 0, ordinary_ = 0, carried_ = 0, depth_ = 1, proposed_ = 0, rejected_ = 0;
    uint32_t head_calls_ = 0, prime_extra_ = 0;
    uint32_t pending_tokens_ = 0, round_end_ = 0;
    uint64_t last_words_ = 0, discarded_ = 0;
    bool scoped_ = false, abort_ = false;
    MtpGenerationCommand generation_{};
    bool reset_ = false, resync_ = false, carry_ = false, stopped_ = false, finished_ = false;
    bool final_input_logits_ = false, final_input_ = true;
};
} // namespace tokspur
#endif
