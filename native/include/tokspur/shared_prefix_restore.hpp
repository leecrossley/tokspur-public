#pragma once

#include "tokspur/gguf.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tokspur {

// A failed rollback is fatal: the target must not continue from an uncertain target/drafter pair.
class SharedPrefixRollbackError final : public std::runtime_error {
public:
    SharedPrefixRollbackError() : std::runtime_error("shared prefix rollback failed; target state is invalid") {}
};

// Snapshot the actual checkpoint interfaces before either write. A false result permits cold prefill only after both prior states have been restored.
template<class Stack, class Mtp = std::nullptr_t>
bool restore_shared_prefix_checkpoint(Stack& stack, uint32_t target_prefix, std::span<const std::byte> stack_payload,
    Mtp* mtp = nullptr, uint32_t target_mtp_first = 1, uint32_t target_mtp_rows = 0, std::span<const std::byte> mtp_payload = {}) {
    if (stack.ssd_checkpoint_payload_bytes(target_prefix) != stack_payload.size()) return false;
    if constexpr (!std::is_same_v<Mtp, std::nullptr_t>) {
        if ((mtp ? mtp->ssd_cache_bytes(target_mtp_rows) : 0) != mtp_payload.size()) return false;
    } else if (!mtp_payload.empty()) return false;

    const uint32_t previous_prefix = stack.position();
    std::vector<std::byte> previous_stack;
    if (previous_prefix) {
        previous_stack.resize(size_t(stack.ssd_checkpoint_payload_bytes(previous_prefix)));
        if (!stack.read_ssd_checkpoint_span(0, previous_stack)) return false;
    }
    [[maybe_unused]] uint32_t previous_mtp_first = 1, previous_mtp_rows = 0;
    [[maybe_unused]] std::vector<std::byte> previous_mtp;
    if constexpr (!std::is_same_v<Mtp, std::nullptr_t>) {
        if (mtp) {
            previous_mtp_first = mtp->first_absolute_position();
            previous_mtp_rows = mtp->cache_rows();
            if (previous_mtp_rows) {
                previous_mtp.resize(size_t(mtp->ssd_cache_bytes(previous_mtp_rows)));
                if (!mtp->read_ssd_cache_span(0, previous_mtp)) return false;
            }
        }
    }
    const auto rollback = [&]() {
        bool restored = true;
        try {
            if (!previous_prefix) stack.reset();
            else restored = stack.write_ssd_checkpoint_span(previous_prefix, 0, previous_stack);
        } catch (...) { restored = false; }
        if constexpr (!std::is_same_v<Mtp, std::nullptr_t>) {
            if (mtp) {
                try {
                    if (!previous_mtp_rows) mtp->reset(previous_mtp_first);
                    else if (!mtp->write_ssd_cache_span(previous_mtp_first, previous_mtp_rows, 0, previous_mtp)) restored = false;
                } catch (...) { restored = false; }
            }
        }
        if (!restored) throw SharedPrefixRollbackError();
    };
    bool restored = false;
    try {
        restored = stack.write_ssd_checkpoint_span(target_prefix, 0, stack_payload);
        if constexpr (!std::is_same_v<Mtp, std::nullptr_t>) {
            if (restored && mtp) restored = mtp->write_ssd_cache_span(target_mtp_first, target_mtp_rows, 0, mtp_payload);
        }
    } catch (...) {
        const auto failure = std::current_exception();
        rollback();
        std::rethrow_exception(failure);
    }
    if (!restored) rollback();
    return restored;
}

} // namespace tokspur
