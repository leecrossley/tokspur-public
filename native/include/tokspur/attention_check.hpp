#pragma once

#include "tokspur/attention_reference.hpp"

#include <string>
#include <utility>

namespace tokspur {
struct AttentionComparison {
    std::vector<std::pair<std::string, double>> scaled_errors;
    bool routes_match, selected_blocks_match, storage_matches;
    uint64_t compared_values, bf16_reference_boundary_differences;
    double maximum;
    bool passed;
};
AttentionComparison compare_attention_layer(MetalGraph& graph, const AttentionLayerScratch& scratch, const QsaCache& cache,
                                              GpuView next_sum, GpuView output, uint32_t position, AttentionLayerShape shape,
                                              const ReferenceAttentionLayer& expected);
int attention_check_cli(int argc, char** argv);
} // namespace tokspur
