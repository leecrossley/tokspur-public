#include "tokspur/shared_prefix_restore.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>

using namespace tokspur;

namespace {
unsigned checks = 0;
void check(bool value, const char* why) { ++checks; if (!value) throw std::runtime_error(why); }

enum class Fault { none, refuse, exception };
struct Store {
    std::array<std::byte, 8> bytes{};
    std::array<Fault, 2> faults{};
    unsigned writes = 0, resets = 0;
    bool refuse_read = false;
    bool read(std::span<std::byte> output) const {
        if (refuse_read) return false;
        std::copy(bytes.begin(), bytes.end(), output.begin());
        return true;
    }
    bool write(std::span<const std::byte> input) {
        const auto fault = writes < faults.size() ? faults[writes] : Fault::none;
        ++writes;
        if (!input.empty()) bytes[0] = input[0];
        if (fault == Fault::exception) throw FormatError("injected partial checkpoint write");
        if (fault == Fault::refuse) return false;
        if (!input.empty()) std::copy(input.begin(), input.end(), bytes.begin());
        return true;
    }
    void clear() { ++resets; bytes.fill(std::byte{0}); }
};
struct Stack {
    Store state;
    uint32_t prefix = 0;
    uint32_t position() const { return prefix; }
    uint64_t ssd_checkpoint_payload_bytes(uint32_t) const { return state.bytes.size(); }
    bool read_ssd_checkpoint_span(uint64_t, std::span<std::byte> bytes) const { return state.read(bytes); }
    bool write_ssd_checkpoint_span(uint32_t target, uint64_t, std::span<const std::byte> bytes) {
        prefix = target;
        return state.write(bytes);
    }
    void reset() { prefix = 0; state.clear(); }
};
struct Mtp {
    Store state;
    uint32_t first = 7, rows = 0;
    uint32_t first_absolute_position() const { return first; }
    uint32_t cache_rows() const { return rows; }
    uint64_t ssd_cache_bytes(uint32_t count) const { return count ? state.bytes.size() : 0; }
    bool read_ssd_cache_span(uint64_t, std::span<std::byte> bytes) const { return state.read(bytes); }
    bool write_ssd_cache_span(uint32_t start, uint32_t count, uint64_t, std::span<const std::byte> bytes) {
        first = start; rows = count;
        return state.write(bytes);
    }
    void reset(uint32_t start) { first = start; rows = 0; state.clear(); }
};
const std::array<std::byte, 8> anchor_stack{std::byte{10}, std::byte{11}, std::byte{12}, std::byte{13}, std::byte{14}, std::byte{15}, std::byte{16}, std::byte{17}};
const std::array<std::byte, 8> anchor_mtp{std::byte{20}, std::byte{21}, std::byte{22}, std::byte{23}, std::byte{24}, std::byte{25}, std::byte{26}, std::byte{27}};

void seed(Stack& stack, Mtp& mtp, bool empty) {
    stack.prefix = empty ? 0 : 12;
    mtp.first = 7; mtp.rows = empty ? 0 : 5;
    stack.state.bytes.fill(empty ? std::byte{0} : std::byte{2});
    mtp.state.bytes.fill(empty ? std::byte{0} : std::byte{3});
}

void failed_restore(bool empty, bool fail_stack, Fault failure) {
    Stack stack; Mtp mtp; seed(stack, mtp, empty);
    const auto old_stack = stack.state.bytes, old_mtp = mtp.state.bytes;
    const auto old_prefix = stack.prefix, old_first = mtp.first, old_rows = mtp.rows;
    (fail_stack ? stack.state : mtp.state).faults[0] = failure;
    bool returned_false = false, threw = false;
    try { returned_false = !restore_shared_prefix_checkpoint(stack, 512, anchor_stack, &mtp, 1, 511, anchor_mtp); }
    catch (const FormatError&) { threw = true; }
    check(failure == Fault::exception ? threw : returned_false, "failed apply lost its refusal/exception");
    check(stack.prefix == old_prefix && stack.state.bytes == old_stack, "partial restore changed prior stack payload/frontier");
    check(mtp.first == old_first && mtp.rows == old_rows && mtp.state.bytes == old_mtp, "partial restore changed prior MTP bytes/frontier");
    check(empty ? stack.state.resets == 1 : stack.state.writes == 2, "stack rollback did not run");
    check(empty ? mtp.state.resets == 1 : mtp.state.writes == unsigned(fail_stack ? 1 : 2), "MTP rollback did not run");
    // Execute a fresh restore on the same participants, without rebuilding or resetting them.
    stack.state.faults = {}; mtp.state.faults = {};
    check(restore_shared_prefix_checkpoint(stack, 512, anchor_stack, &mtp, 1, 511, anchor_mtp), "retry after restored failure was refused");
    check(stack.prefix == 512 && mtp.first == 1 && mtp.rows == 511 && stack.state.bytes == anchor_stack && mtp.state.bytes == anchor_mtp, "retry published an inconsistent pair");
}

void rollback_failures() {
    for (bool stack_rollback : {false, true}) for (Fault fault : {Fault::refuse, Fault::exception}) {
        Stack stack; Mtp mtp; seed(stack, mtp, false);
        mtp.state.faults[0] = Fault::refuse;
        (stack_rollback ? stack.state : mtp.state).faults[1] = fault;
        bool fatal = false, continued = false;
        try {
            static_cast<void>(restore_shared_prefix_checkpoint(stack, 512, anchor_stack, &mtp, 1, 511, anchor_mtp));
            continued = true;
        } catch (const SharedPrefixRollbackError&) { fatal = true; }
        check(fatal && !continued, "failed rollback permitted cold continuation");
        check(stack.state.writes == 2 && mtp.state.writes == 2, "rollback failure skipped the other participant");
    }
}

void refusal_and_success() {
    Stack stack; Mtp mtp; seed(stack, mtp, false);
    stack.state.refuse_read = true;
    check(!restore_shared_prefix_checkpoint(stack, 512, anchor_stack, &mtp, 1, 511, anchor_mtp), "failed stack snapshot admitted restore");
    check(!stack.state.writes && !mtp.state.writes, "snapshot failure mutated state");
    stack.state.refuse_read = false; mtp.state.refuse_read = true;
    check(!restore_shared_prefix_checkpoint(stack, 512, anchor_stack, &mtp, 1, 511, anchor_mtp), "failed MTP snapshot admitted restore");
    check(!stack.state.writes && !mtp.state.writes, "MTP snapshot failure mutated stack");
    mtp.state.refuse_read = false;
    check(!restore_shared_prefix_checkpoint(stack, 512, std::span(anchor_stack).first(7), &mtp, 1, 511, anchor_mtp), "short stack payload admitted");
    check(!restore_shared_prefix_checkpoint(stack, 512, anchor_stack, &mtp, 1, 511, std::span(anchor_mtp).first(7)), "short MTP payload admitted");
    check(!stack.state.writes && !mtp.state.writes, "malformed payload mutated state");
    check(restore_shared_prefix_checkpoint(stack, 512, anchor_stack, &mtp, 1, 511, anchor_mtp), "valid pair restore failed");
    check(stack.state.writes == 1 && mtp.state.writes == 1, "successful restore performed rollback");
    check(stack.state.bytes == anchor_stack && mtp.state.bytes == anchor_mtp, "successful restore changed payload bytes");
    check(restore_shared_prefix_checkpoint(stack, 1024, anchor_stack), "stack-only restore failed");
    Mtp* absent = nullptr;
    check(restore_shared_prefix_checkpoint(stack, 2048, anchor_stack, absent), "optional MTP restore failed");
    check(!restore_shared_prefix_checkpoint(stack, 2048, anchor_stack, absent, 1, 511, anchor_mtp), "MTP payload without owner admitted");

}
}

int main() {
    try {
        for (bool empty : {false, true}) for (bool fail_stack : {false, true}) for (Fault fault : {Fault::refuse, Fault::exception}) failed_restore(empty, fail_stack, fault);
        rollback_failures(); refusal_and_success();
        std::cout << "Shared-prefix production transaction checks=" << checks << "; fault-injected checkpoint interfaces, no GPU state qualification\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
