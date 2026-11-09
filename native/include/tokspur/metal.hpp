#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace tokspur {
// Read-only Apple ProcessInfo state; creates no Metal context or model worker.
[[nodiscard]] uint32_t host_thermal_state();

struct MetalResult {
    std::vector<float> output;
    uint64_t buffer_bytes = 0;
    double setup_ms = 0;
    double first_submission_ms = 0;
    double median_submission_ms = 0;
    double median_gpu_ms = 0;
};

class MetalGemv {
public:
    explicit MetalGemv(const std::filesystem::path& owned_library);
    ~MetalGemv();
    MetalGemv(const MetalGemv&) = delete;
    MetalGemv& operator=(const MetalGemv&) = delete;
    [[nodiscard]] std::string device_name() const;
    [[nodiscard]] MetalResult run(uint32_t type, std::span<const std::byte> matrix, uint32_t columns,
                                  uint32_t rows, std::span<const float> input, uint32_t repetitions);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tokspur
