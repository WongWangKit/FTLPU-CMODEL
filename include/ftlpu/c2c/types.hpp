#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu {

// C2C architecture-visible transaction. One instruction moves one physical
// 320-byte vector, represented as twenty 16-byte tile segments.
struct C2cVector {
    StreamPayloadVector320 payload{};
    std::uint64_t vector_tag{0};
    bool last{false};
};

struct C2cLinkConfig {
    // Physical payload transferred by the external link per core cycle.
    // The first functional model uses 320, so serialization takes one cycle.
    std::size_t beat_bytes{hw::kPhysicalVectorBytes};

    // Additional fixed link-flight cycles after serialization completes.
    std::size_t flight_latency_cycles{0};

    // Number of complete vectors that may wait at the receive side.
    std::size_t rx_fifo_depth_vectors{4};

    void validate() const
    {
        if (beat_bytes == 0 || beat_bytes > hw::kPhysicalVectorBytes) {
            throw std::invalid_argument(
                "C2C beat_bytes must be in the range [1, 320]");
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

static_assert(hw::kPhysicalVectorBytes == 320);
static_assert(hw::kTileRows * hw::kLanesPerTile == 320);

} // namespace ftlpu
