#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"

#include <cstddef>
#include <cstdint>
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
    bool notify_mem{true};
};

struct C2cInstruction {
    C2cOpcode opcode{C2cOpcode::Send};
    std::size_t stream_index{0};
    C2cConsumer consumer{};

    static C2cInstruction Send(std::size_t stream_index)
    {
        validate_stream(stream_index);
        return C2cInstruction {C2cOpcode::Send, stream_index, {}};
    }

    static C2cInstruction Receive(
        std::size_t stream_index,
        Hemisphere consumer_hemisphere,
        std::size_t consumer_mem_slice,
        std::size_t consumer_mem_bank = 0,
        bool notify_mem = true)
    {
        validate_stream(stream_index);
        if (consumer_mem_slice >= hw::kMemSliceColumns) {
            throw std::out_of_range(
                "C2C Receive consumer is outside the MEM slice array");
        }
        if (consumer_mem_bank >= hw::kMemBanksPerSlice) {
            throw std::out_of_range(
                "C2C Receive consumer is outside the MEM bank array");
        }
        return C2cInstruction {
            C2cOpcode::Receive,
            stream_index,
            C2cConsumer {
                consumer_hemisphere, consumer_mem_slice, consumer_mem_bank,
                notify_mem},
        };
    }

private:
    static void validate_stream(std::size_t stream_index)
    {
        if (stream_index >= hw::kStreamsPerDirection) {
            throw std::out_of_range(
                "C2C stream index is outside one directional stream file");
        }
    }
};

} // namespace ftlpu
