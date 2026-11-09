#pragma once

#include "tokspur/chat.hpp"
#include "tokspur/response.hpp"
#include "tokspur/response_protocol.hpp"

#include <functional>
#include <optional>

namespace tokspur {

enum class ToolEnumValidation { strict, executor };
ToolEnumValidation parse_tool_enum_validation(std::string_view);
std::string_view tool_enum_validation_name(ToolEnumValidation);
std::string_view tool_enum_validation_policy(ToolEnumValidation);

enum class ToolArgValidation { strict, executor };
ToolArgValidation parse_tool_arg_validation(std::string_view);
std::string_view tool_arg_validation_name(ToolArgValidation);
std::string_view tool_arg_validation_policy(ToolArgValidation);

// CPU-only seams. Production binds these to the owned tokenizer/renderer and
// exactly one owned target child. Tests supply original, deterministic fixtures.
struct ServeCodec {
    std::function<RenderedChat(const Json&)> render;
    std::function<std::vector<uint32_t>(const RenderedChat&)> encode;
    ResponseDecoder decode;
    ResponseAdmission admission;
};
using ServeTarget = std::function<Json(const Json&, const std::function<void(const Json&)>&)>;
struct ServeSink {
    std::function<void(const Json&)> chunk;
    std::function<bool()> cancelled;
    std::function<uint64_t()> new_seed;
    std::function<void(const Json&)> prefix_trace = {}; // Explicit private capture outside HTTP responses.
};
struct AdmittedChat {
    Json render, sampling;
    std::vector<ToolSpec> manifest;
    std::vector<Json> schemas;
    std::vector<std::string> names;
    std::optional<uint64_t> seed;
    std::string tool_choice;
    bool stream = false, thinking = true, include_usage = false, named_tool_choice = false, single_tool_call = false;
    uint32_t maximum_tokens = 8192;
};

AdmittedChat admit_chat(const Json& request, std::string_view alias,
                        ToolEnumValidation = ToolEnumValidation::strict,
                        ToolArgValidation = ToolArgValidation::strict);
// CPU-only startup admission for the fixed owned target policy. Every field is
// mandatory and type-checked; absent metadata never implies the direct path.
void validate_worker_ready(const Json& ready, std::string_view stack_execution = "direct",
                           uint32_t checkpoint_slots = 0, uint64_t checkpoint_budget = 0);
// Validates the supported schema subset recursively; unknown keywords fail
// admission rather than being silently ignored. No tool is ever executed here.
void validate_tool_value(const Json& schema, const Json& value,
                         ToolEnumValidation = ToolEnumValidation::strict);
void validate_executor_tool_arguments(const Json& value);

class ServeSession {
public:
    explicit ServeSession(ServeCodec codec, std::string alias = "tokspur", std::string stack_execution = "direct",
                          Json broker_ready = Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
                          , bool verify8 = false, Json verify_ready = Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                          , bool branch8 = false
#endif
                          , ToolEnumValidation tool_enums = ToolEnumValidation::strict
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
                          , bool sampled_prefill = false
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                          , Json mtp_ready = Json()
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
                          , Json weight_lanes_ready = Json()
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                          , std::optional<uint32_t> mtp_first_output_packet = std::nullopt
#endif
                          , ToolArgValidation tool_args = ToolArgValidation::strict
                          );
    Json complete(const Json& request, std::string_view response_id,
                  const ServeTarget& target, const ServeSink& sink);
    void invalidate() noexcept;
    [[nodiscard]] size_t committed_tokens() const noexcept { return committed_.size(); }
private:
    ServeCodec codec_;
    std::string alias_;
    std::string stack_execution_; // Immutable startup choice, never a request field.
    const ToolEnumValidation tool_enums_; // Enum membership only; executor owns allowlist.
    const ToolArgValidation tool_args_; // Schema property/type/required enforcement or executor deferral.
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    Json weight_lanes_identity_; // Five identity fields plus optional worker pair; no paths or raw READY.
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    const bool sampled_prefill_; // Immutable input capacity; output sampling stays scalar.
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    const bool mtp_;
    Json mtp_ready_, mtp_previous_;
    const std::optional<uint32_t> mtp_first_output_packet_;
    bool mtp_resync_ = false;
    uint64_t mtp_generation_id_ = 0; // Never reused after session invalidation.
#endif
    std::vector<uint32_t> committed_;
    std::vector<std::pair<size_t, std::string>> saved_reasoning_;
    std::string sampling_;
    uint64_t seed_ = 0;
    bool reusable_ = false;
    Json broker_ready_;
    uint64_t broker_epoch_ = 0, broker_command_ = 0, broker_lease_ = 0;
    bool broker_resync_ = true;
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    const bool verify8_;
    Json verify_ready_, verify_previous_;
    bool verify_resync_ = false; // Only a cold reset may rebase metadata after a drained cancellation.
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    const bool branch8_;
#endif
};

int server_cli(int argc, char** argv);
} // namespace tokspur
