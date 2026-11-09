#pragma once

#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"

namespace tokspur {

struct TensorRequirement {
    std::string name;
    std::vector<uint64_t> shape;
    std::vector<uint32_t> encodings;
};

std::vector<TensorRequirement> flash_next_requirements(uint64_t ple_rows);

struct FlashLayout {
    size_t tensors = 0;
    uint64_t resident_matrix_bytes = 0;
    uint64_t ple_source_bytes = 0;
    uint64_t ple_addressable_rows = 0;
    uint64_t ple_storage_rows = 0;
    double rms_epsilon = 0;
    double rope_base = 0;
    std::vector<uint64_t> ple_multipliers;
    std::vector<uint64_t> ple_offsets;
    std::vector<uint64_t> ple_vocabularies;
    [[nodiscard]] Json describe() const;
};

// Closed geometry contract for the current adopted variant. No allocation,
// quantization/repack, GPU inference or claim of graph numerical equivalence.
FlashLayout inspect_flash_next(const ModelFiles& files, const ModelFiles* pack = nullptr);

} // namespace tokspur
