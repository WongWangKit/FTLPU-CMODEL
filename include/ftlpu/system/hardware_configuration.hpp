#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <cstddef>
#include <stdexcept>

namespace ftlpu {

// Runtime-selectable logical hardware exposed by one CModel instance. The
// static constants in hw:: remain the compiled physical capacity and ISA
// geometry; each test may select a supported subset without rebuilding.
struct SystemHardwareConfiguration {
    std::size_t sram_depth_rows{hw::kSramDepthRows};
    std::size_t mxms_per_hemisphere{hw::kMxmsPerHemisphere};
    std::size_t mxm_weight_buffers{2};
    std::size_t vxm_alus{16};
    std::size_t c2c_streams_per_direction{hw::kC2cStreamsPerDirection};
    bool mxm_local_dequant_enabled{true};
    bool mxm_block_compute_enabled{true};
    bool mxm_weight_activation_overlap_enabled{true};

    void validate() const
    {
        if (sram_depth_rows == 0 || sram_depth_rows > hw::kSramDepthRows)
            throw std::invalid_argument(
                "configured SRAM depth exceeds the CModel physical capacity");
        if (mxms_per_hemisphere == 0
            || mxms_per_hemisphere > hw::kMxmsPerHemisphere)
            throw std::invalid_argument(
                "configured MXM count exceeds the CModel physical capacity");
        if (mxm_weight_buffers == 0 || mxm_weight_buffers > 2)
            throw std::invalid_argument(
                "configured MXM weight-buffer count is unsupported");
        if (vxm_alus == 0 || vxm_alus > 16)
            throw std::invalid_argument(
                "configured VXM ALU count exceeds the CModel physical capacity");
        if (c2c_streams_per_direction == 0
            || c2c_streams_per_direction > hw::kC2cStreamsPerDirection)
            throw std::invalid_argument(
                "configured C2C stream count exceeds the CModel physical capacity");
    }
};

} // namespace ftlpu
