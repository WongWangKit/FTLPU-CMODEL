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
    std::size_t beat_bytes{hw::kC2cBytesPerStreamPerCycle};
    std::size_t flight_latency_cycles{0};
    std::size_t rx_fifo_depth_vectors{4};

    // Nonzero enables a one-entry retransmit window (physical link ticks).
    std::size_t retry_timeout{0};
    std::size_t max_retries{2};
    std::size_t control_latency{1};
    // Zero keeps the ACK latency for the receive-credit path. A nonzero
    // override models an independent credit return delay after RX consumption.
    std::size_t rx_credit_latency{0};

    std::size_t effective_rx_credit_latency() const noexcept
    {
        return rx_credit_latency ? rx_credit_latency : control_latency;
    }

    void validate() const
    {
        if (retry_timeout && (control_latency==0 || control_latency>retry_timeout))
            throw std::invalid_argument("reliable Peer requires ACK latency within timeout");
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

} // namespace ftlpu
