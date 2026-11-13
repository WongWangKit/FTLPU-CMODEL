#pragma once

#include "ftlpu/vxm/compact_instruction.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace ftlpu::isa {

// One physical 96-bit iMEM word. VXM has no ICU launch loops. The FU
// repeat_count owns a contiguous run of identical ALU configurations.
// [1:0] is the common ICU envelope (00 = instruction), [55:2] is the
// compact configuration's 54 used control bits, [87:56] is its immediate,
// and [95:88] is reserved. WAIT/SYNC use the existing common envelope.
struct EncodedVxmIcuWord {
    std::array<std::uint32_t, 3> lanes{};
};
static_assert(sizeof(EncodedVxmIcuWord) == 12);

inline EncodedVxmIcuWord encode_vxm_icu_instruction(
    const VxmCompactInstruction& instruction)
{
    if (instruction.control >> 54)
        throw std::invalid_argument("VXM compact control has reserved bits set");
    const auto low = (instruction.control << 2)
        | (std::uint64_t(instruction.immediate_bits & 0xffU) << 56);
    return {{static_cast<std::uint32_t>(low),
        static_cast<std::uint32_t>(low >> 32),
        instruction.immediate_bits >> 8}};
}

inline VxmCompactInstruction decode_vxm_icu_instruction(
    const EncodedVxmIcuWord& word)
{
    if ((word.lanes[0] & 3U) || (word.lanes[2] >> 24))
        throw std::invalid_argument("invalid VXM sequential instruction envelope/reserved bits");
    const auto low = std::uint64_t(word.lanes[0])
        | (std::uint64_t(word.lanes[1]) << 32);
    return {(low >> 2) & ((std::uint64_t{1} << 54) - 1),
        static_cast<std::uint32_t>(low >> 56) | (word.lanes[2] << 8)};
}

} // namespace ftlpu::isa
