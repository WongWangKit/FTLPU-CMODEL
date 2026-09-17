#pragma once

#include "ftlpu/icu/fu_3d_codec.hpp"
#include "ftlpu/vxm/compact_instruction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ftlpu {

// A VXM functional instruction already owns its contiguous run length through
// VxmLaneAluInstruction::repeat_count. The ICU therefore carries only two
// outer launch counters. The normalized third coordinate is an implementation
// detail used by the common queue engine and is never encoded as a VXM
// counter.
struct VxmIcuRun2DInstruction {
    IcuLoop3D loop{};
    VxmCompactInstruction instruction{};

    static VxmIcuRun2DInstruction Run2D(
        std::size_t startCycle,
        std::array<std::size_t, 2> counts,
        std::array<std::size_t, 2> cycleStrides,
        VxmCompactInstruction instruction)
    {
        // startCycle remains a schedule coordinate. Physical packets use
        // zero and carry any queue-local launch delay in loop.wait_cycle.
        return {{startCycle, {counts[0], counts[1], 1},
                    {cycleStrides[0], cycleStrides[1], 1}},
            instruction};
    }
};

namespace detail {

inline void validate_vxm_icu_run_2d_instruction(
    const VxmIcuRun2DInstruction& instruction)
{
    validate_icu_loop_3d(instruction.loop);
    if (instruction.loop.counts[2] != 1
        || instruction.loop.cycle_strides[2] != 1)
        throw std::invalid_argument(
            "VXM RUN_2D cannot carry a third launch counter");
}

} // namespace detail

namespace isa {

struct EncodedVxmIcuRun2DWord {
    std::array<std::uint32_t, 3> lanes{};
};

struct EncodedVxmIcuRun2DPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 3;
    static constexpr std::size_t kLanesPerWord = 3;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 88;
    std::array<EncodedVxmIcuRun2DWord, kWordCount> words{};
};

static_assert(sizeof(EncodedVxmIcuRun2DWord) == 96 / 8);
static_assert(sizeof(EncodedVxmIcuRun2DPacket) == 3 * 96 / 8);
static_assert(std::is_trivially_copyable_v<EncodedVxmIcuRun2DPacket>);

inline EncodedVxmIcuRun2DPacket encode_vxm_icu_run_2d_instruction(
    const VxmIcuRun2DInstruction& instruction)
{
    ::ftlpu::detail::validate_vxm_icu_run_2d_instruction(instruction);
    namespace codec = fu_3d_codec_detail;
    EncodedVxmIcuRun2DPacket packet{};
    codec::initialize_word_headers(packet, 0);
    codec::write_unsigned(packet, 0, 24, instruction.loop.wait_cycle,
        "VXM RUN_2D wait_cycle exceeds 24 bits");
    for (std::size_t dimension = 0; dimension < 2; ++dimension) {
        codec::write_unsigned(packet, 24 + 16 * dimension, 16,
            instruction.loop.counts[dimension] - 1,
            "VXM RUN_2D count");
        codec::write_unsigned(packet, 56 + 24 * dimension, 24,
            instruction.loop.cycle_strides[dimension],
            "VXM RUN_2D cycle stride");
    }
    codec::write_unsigned(packet, 104, 32,
        static_cast<std::uint32_t>(instruction.instruction.control),
        "VXM RUN_2D compact control low");
    codec::write_unsigned(packet, 136, 32,
        static_cast<std::uint32_t>(instruction.instruction.control >> 32),
        "VXM RUN_2D compact control high");
    codec::write_unsigned(packet, 168, 32,
        instruction.instruction.immediate_bits,
        "VXM RUN_2D compact immediate");
    return packet;
}

inline VxmIcuRun2DInstruction decode_vxm_icu_run_2d_instruction(
    const EncodedVxmIcuRun2DPacket& packet)
{
    namespace codec = fu_3d_codec_detail;
    codec::validate_word_headers(
        packet, 0, "invalid VXM RUN_2D packet header");
    codec::require_reserved_zero(
        packet, 200, "VXM RUN_2D reserved field is non-zero");
    auto instruction = VxmIcuRun2DInstruction::Run2D(
        0,
        {static_cast<std::size_t>(codec::read_unsigned(packet, 24, 16)) + 1,
            static_cast<std::size_t>(codec::read_unsigned(packet, 40, 16)) + 1},
        {static_cast<std::size_t>(codec::read_unsigned(packet, 56, 24)),
            static_cast<std::size_t>(codec::read_unsigned(packet, 80, 24))},
        VxmCompactInstruction {
            codec::read_unsigned(packet, 104, 32)
                | (codec::read_unsigned(packet, 136, 32) << 32),
            static_cast<std::uint32_t>(
                codec::read_unsigned(packet, 168, 32))});
    instruction.loop.wait_cycle = static_cast<std::size_t>(
        codec::read_unsigned(packet, 0, 24));
    ::ftlpu::detail::validate_vxm_icu_run_2d_instruction(instruction);
    return instruction;
}

} // namespace isa
} // namespace ftlpu
