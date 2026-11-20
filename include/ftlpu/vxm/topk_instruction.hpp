#pragma once

#include "ftlpu/vxm/compact_instruction.hpp"

namespace ftlpu {

// One operation on a row batch. Each lane reduces one row over the streamed
// columns. Input group sources are fixed for the duration of the operation.
struct VxmTopKInstruction {
    static constexpr std::uint64_t opcode = 10;
    static constexpr std::size_t queue = 0;
    static constexpr std::size_t max_rows = 128, max_columns = 1 << 20, max_k = 16;
    static constexpr std::size_t index_stream = 0, value_stream = 4;

    std::uint32_t rows{}, columns{}, k{};
    std::uint16_t west_input_groups{};
    bool has_mask{};

    friend bool operator==(const VxmTopKInstruction&, const VxmTopKInstruction&) = default;
};

inline bool is_vxm_topk(VxmCompactInstruction packet)
{
    return (packet.control & VxmCompactInstructionCodec::kOpcodeMask)
        == VxmTopKInstruction::opcode;
}

inline VxmCompactInstruction encode_vxm_topk(const VxmTopKInstruction& instruction)
{
    const auto [rows, columns, k, west, mask] = instruction;
    if (!rows || rows > instruction.max_rows || !columns ||
        columns > instruction.max_columns || !k || k > columns || k > instruction.max_k)
        throw std::invalid_argument("invalid VXM TOPK shape");
    return {instruction.opcode | (std::uint64_t(rows - 1) << 4)
        | (std::uint64_t(columns - 1) << 11) | (std::uint64_t(k - 1) << 31)
        | (std::uint64_t(west) << 35) | (std::uint64_t(mask) << 51), 0};
}

inline VxmTopKInstruction decode_vxm_topk(VxmCompactInstruction packet)
{
    if (!is_vxm_topk(packet) || packet.control >> 52 || packet.immediate_bits)
        throw std::invalid_argument("invalid VXM TOPK encoding");
    const VxmTopKInstruction instruction{
        static_cast<std::uint32_t>((packet.control >> 4) & 0x7f) + 1,
        static_cast<std::uint32_t>((packet.control >> 11) & 0xfffff) + 1,
        static_cast<std::uint32_t>((packet.control >> 31) & 0xf) + 1,
        static_cast<std::uint16_t>(packet.control >> 35),
        ((packet.control >> 51) & 1) != 0};
    (void)encode_vxm_topk(instruction);
    return instruction;
}

} // namespace ftlpu
