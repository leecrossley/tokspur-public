#pragma once

#include "tokspur/recurrent_reference.hpp"

#include <string>
#include <utility>

namespace tokspur {
struct LayerComparison {
    std::vector<std::pair<std::string, double>> scaled_errors;
    bool routes_match;
    uint64_t compared_values;
    double maximum;
    bool passed;
};
// Only call after submit completed. Readback is diagnostic, never in decoding.
LayerComparison compare_recurrent_layer(MetalGraph& graph, const RecurrentScratch& scratch,
                                         const GdnState& next, GpuView output_streams, const ReferenceLayer& expected);
int recurrent_check_cli(int argc, char** argv);
} // namespace tokspur
