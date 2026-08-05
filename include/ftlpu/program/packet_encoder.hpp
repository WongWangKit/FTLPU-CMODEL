#pragma once

#include "ftlpu/core/instruction_packet.hpp"

namespace ftlpu::program {

template <typename Instruction>
isa::EncodedInstructionPacket encode_packet(const Instruction& instruction)
{
    return isa::encode_packet(instruction);
}

inline isa::EncodedInstructionPacket padding_nop_packet()
{
    return isa::encode_packet(IcuControlInstruction::Nop(0));
}

} // namespace ftlpu::program
