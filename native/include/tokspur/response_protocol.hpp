#pragma once

#include "tokspur/json.hpp"

#include <functional>
#include <span>
#include <string>

namespace tokspur {
inline constexpr size_t response_request_bytes = 1024 * 1024;
inline constexpr size_t response_maximum_ids = 16384;
// HTTP accumulates multiple bounded native packets; the CLI default stays fixed.
inline constexpr size_t response_maximum_accumulated_ids = 32768;
inline constexpr size_t response_output_bytes = 8 * 1024 * 1024;

// Derived by the native CLI from the exact pinned tokenizer, never admitted
// directly from an untrusted JSON stop_id. The CPU test uses owned fixture DATA.
struct ResponseAdmission {
    uint32_t vocabulary, im_end, end_of_text, im_start;
    bool is_stop(uint64_t id) const { return id == im_end || id == end_of_text; }
};
using ResponseDecoder = std::function<std::string(std::span<const uint32_t>)>;

// Stateless accumulated-response adapter. The caller supplies all generated assistant IDs (not prompt IDs), the renderer's initial thinking mode, a typed tool manifest and the actual worker's termination fields. Length boundaries do not publish tool calls. Only a matching final terminal-ID/stop pair admits finish(). Callers own correlation, epochs and tool side effects. Invalid envelopes, bytes or frames throw FormatError. The CLI emits an empty-call error envelope and does not reuse parser state for the next request.
Json parse_native_response(const Json& request, ResponseAdmission admission,
                           const ResponseDecoder& decode, size_t maximum_ids = response_maximum_ids);
Json native_response_error(std::string_view message);
} // namespace tokspur
