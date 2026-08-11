#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu {

struct C2cVector {
    StreamPayloadSliceVector payload{};
    std::uint64_t vector_tag{0};
};

struct C2cLinkConfig {
    std::size_t beat_bytes{hw::kPhysicalVectorBytes};
    std::size_t flight_latency_cycles{0};
    std::size_t rx_fifo_depth_vectors{4};

    void validate() const
    {
        if (beat_bytes == 0 || beat_bytes > hw::kPhysicalVectorBytes) {
            throw std::invalid_argument(
                "C2C beat_bytes must fit within one physical vector");
        }
        if (rx_fifo_depth_vectors == 0) {
            throw std::invalid_argument(
                "C2C RX FIFO depth must be at least one vector");
        }
    }

    std::size_t serialization_cycles() const noexcept
    {
        return (hw::kPhysicalVectorBytes + beat_bytes - 1) / beat_bytes;
    }
};

static_assert(hw::kPhysicalVectorBytes == 32);

} // namespace ftlpu
