#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace tokspur {

double stable_sigmoid(double value);
double stable_silu(double value);
double stable_softplus(double value);

// Mathematical references use double accumulation. Gains here are EFFECTIVE
// gains; zero-centered checkpoint gains must be converted explicitly by a loader.
std::vector<double> group_rms_reference(std::span<const double> input, std::span<const double> effective_gains,
                                        size_t group_width, double epsilon);

struct ResidualRead {
    std::vector<double> input;
    std::vector<double> write_scales;
};

// Matrices are row-major [output,input]. Stream layout is [branch,channel].
ResidualRead gated_residual_reference(std::span<const double> streams, std::span<const double> effective_gains,
                                      std::span<const double> down, std::span<const double> up,
                                      std::span<const double> inject, size_t branches, size_t channels,
                                      size_t rank, double epsilon);
std::vector<double> residual_write_reference(std::span<const double> streams, std::span<const double> output,
                                              std::span<const double> scales);

struct DeltaStep {
    std::vector<double> state;
    std::vector<double> output;
};

// State layout [head,key_channel,value_channel]. q/k are already normalized,
// expanded per value head and scaled as required by the model's projection path.
// Pure result ownership supports independently checking commit/rollback later.
DeltaStep gated_delta_reference(std::span<const double> state, std::span<const double> query,
                                std::span<const double> key, std::span<const double> value,
                                std::span<const double> decay, std::span<const double> beta,
                                size_t heads, size_t key_width, size_t value_width);

struct ConvStep {
    std::vector<double> history;
    std::vector<double> output;
};

// History is chronological [dilation*(kernel-1),channel]; weights [channel,kernel].
ConvStep causal_depthwise_reference(std::span<const double> history, std::span<const double> input,
                                    std::span<const double> weights, size_t kernel, size_t dilation = 1);

} // namespace tokspur
