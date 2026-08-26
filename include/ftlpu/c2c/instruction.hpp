#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ftlpu {

enum class C2cOpcode : std::uint8_t {
    Send,
    Receive,
};

struct C2cConsumer {
    Hemisphere hemisphere{Hemisphere::East};
    std::size_t mem_slice{0};
    std::size_t mem_bank{0};
    std::size_t base_row{0};
    std::size_t vector_count{1};
    std::size_t row_stride{1};
    bool notify_mem{true};
};

struct C2cInstruction {
    C2cOpcode opcode{C2cOpcode::Send};
    // The external transport lane and the ordinary SR index are independent.
    std::size_t stream_index{0};
    std::size_t fabric_stream_index{0};
    C2cConsumer consumer{};

    static C2cInstruction Send(std::size_t stream_index)
    {
        validate_stream(stream_index);
        return C2cInstruction {
            C2cOpcode::Send, stream_index, stream_index, {}};
    }

    static C2cInstruction Receive(
        std::size_t stream_index,
        Hemisphere consumer_hemisphere,
        std::size_t consumer_mem_slice,
        std::size_t consumer_mem_bank = 0,
        bool notify_mem = true,
        std::size_t base_row = 0,
        std::size_t vector_count = 1,
        std::size_t row_stride = 1,
        std::size_t fabric_stream_index =
            std::numeric_limits<std::size_t>::max())
    {
        validate_stream(stream_index);
        if (fabric_stream_index == std::numeric_limits<std::size_t>::max())
            fabric_stream_index = stream_index;
        if (fabric_stream_index >= hw::kStreamsPerDirection) {
            throw std::out_of_range(
                "C2C Receive stream is outside the ordinary SR file");
        }
        if (consumer_mem_slice >= hw::kMemSliceColumns) {
            throw std::out_of_range(
                "C2C Receive consumer is outside the MEM slice array");
        }
        if (consumer_mem_bank >= hw::kMemBanksPerSlice) {
            throw std::out_of_range(
                "C2C Receive consumer is outside the MEM bank array");
        }
        if (vector_count == 0) {
            throw std::invalid_argument(
                "C2C Receive vector_count must be non-zero");
        }
        if (base_row >= hw::kSramDepthRows
            || (vector_count - 1) * row_stride
                > hw::kSramDepthRows - 1 - base_row) {
            throw std::out_of_range(
                "C2C Receive row sequence exceeds the MEM bank");
        }
        return C2cInstruction {
            C2cOpcode::Receive,
            stream_index,
            fabric_stream_index,
            C2cConsumer {
                consumer_hemisphere, consumer_mem_slice, consumer_mem_bank,
                base_row, vector_count, row_stride, notify_mem},
        };
    }

private:
    static void validate_stream(std::size_t stream_index)
    {
        if (stream_index >= hw::kC2cStreamsPerDirection) {
            throw std::out_of_range(
                "C2C stream index is outside the directional transport lanes");
        }
    }
};

} // namespace ftlpu
