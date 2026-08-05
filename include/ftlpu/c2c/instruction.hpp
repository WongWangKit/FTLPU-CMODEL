#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu {

enum class C2cOpcode : std::uint8_t {
    Send,
    Receive,
};

struct C2cInstruction {
    C2cOpcode opcode{C2cOpcode::Send};
    std::size_t stream_index{0};

    static C2cInstruction Send(std::size_t stream_index)
    {
        validate_stream(stream_index);
        return C2cInstruction{C2cOpcode::Send, stream_index};
    }

    static C2cInstruction Receive(std::size_t stream_index)
    {
        validate_stream(stream_index);
        return C2cInstruction{C2cOpcode::Receive, stream_index};
    }

private:
    static void validate_stream(std::size_t stream_index)
    {
        if (stream_index >= hw::kStreamsPerDirection) {
            throw std::out_of_range(
                "C2C stream index is outside the 32 streams in one direction");
        }
    }
};

} // namespace ftlpu
