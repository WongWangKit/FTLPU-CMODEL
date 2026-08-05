#pragma once

#include "ftlpu/vxm/data_format.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ftlpu {

// Boundary hardware after the configured MXM accumulator. One converter is
// instantiated per configured output lane, so one Supercell result vector is
// converted in parallel.
class MxmOutputCast {
public:
    static constexpr std::size_t kLatency = 1;
    static constexpr std::size_t kOutputBytes = 2;

    static float combined_dequant_scale(
        float activation_scale,
        float weight_scale)
    {
        return activation_scale * weight_scale;
    }

    static std::uint16_t cast(
        std::int32_t input,
        float dequant_scale = 1.0f)
    {
        return VxmDataFormat::float_to_fp16_bits(
            static_cast<float>(input) * dequant_scale);
    }

    // The reduced-precision W8A16 model accumulates in float. It shares the
    // same physical FP16 egress format and one-cycle Cast stage.
    static std::uint16_t cast(
        float input,
        float dequant_scale = 1.0f)
    {
        return VxmDataFormat::float_to_fp16_bits(input * dequant_scale);
    }

    static std::array<std::uint8_t, kOutputBytes> bytes(
        std::int32_t input,
        float dequant_scale = 1.0f)
    {
        const auto bits = cast(input, dequant_scale);
        return {static_cast<std::uint8_t>(bits),
                static_cast<std::uint8_t>(bits >> 8)};
    }


    static std::array<std::uint8_t, kOutputBytes> bytes(
        float input,
        float dequant_scale = 1.0f)
    {
        const auto bits = cast(input, dequant_scale);
        return {static_cast<std::uint8_t>(bits),
                static_cast<std::uint8_t>(bits >> 8)};
    }
};

} // namespace ftlpu
