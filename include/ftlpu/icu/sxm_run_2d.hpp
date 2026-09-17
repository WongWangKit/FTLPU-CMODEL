#pragma once

#include "ftlpu/icu/fu_3d_codec.hpp"
#include "ftlpu/sxm/instruction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ftlpu {

// SXM owns the 32-lane map inside its tile-local instruction. The ICU repeats
// that tile operation over at most two outer launch coordinates.
struct SxmIcuRun2DInstruction {
    IcuLoop3D loop{};
    SxmInstruction instruction{};
    // Add this modulo-32 delta to every non-zero-fill permute-map entry after
    // each launch. This lets one blocking coarse instruction express the
    // four-phase 8-lane rotations used by transpose/permute pipelines.
    std::size_t permute_map_stride{0};

    static SxmIcuRun2DInstruction Run2D(
        std::size_t startCycle,
        std::array<std::size_t, 2> counts,
        std::array<std::size_t, 2> cycleStrides,
        SxmInstruction instruction,
        std::size_t permuteMapStride = 0)
    {
        // startCycle remains a schedule coordinate. Physical packets use
        // zero and carry any queue-local launch delay in loop.wait_cycle.
        return {{startCycle, {counts[0], counts[1], 1},
                    {cycleStrides[0], cycleStrides[1], 1}},
            std::move(instruction), permuteMapStride};
    }
};

namespace detail {

inline void validate_sxm_icu_run_2d_instruction(
    const SxmIcuRun2DInstruction& instruction)
{
    validate_icu_loop_3d(instruction.loop);
    if (instruction.loop.counts[2] != 1
        || instruction.loop.cycle_strides[2] != 1)
        throw std::invalid_argument(
            "SXM RUN_2D cannot carry a third launch counter");
    if (instruction.permute_map_stride >= SxmInstruction::kTotalLanes
        || instruction.permute_map_stride % hw::kLanesPerTile != 0)
        throw std::invalid_argument(
            "SXM RUN_2D permute-map stride must be a modulo-32 multiple of eight lanes");
    if (instruction.instruction.opcode != SxmOpcode::Permute
        && instruction.permute_map_stride != 0)
        throw std::invalid_argument(
            "SXM RUN_2D map stride is only valid for permute");
    static_cast<void>(isa::encode_sxm_instruction(instruction.instruction));
}

} // namespace detail

namespace isa {

namespace sxm_run_2d_codec_detail {

template <typename Packet>
void initialize_headers(Packet& packet)
{
    // SXM needs six words, so its FU-local header uses a three-bit word index.
    // The queue identifies SXM and the tile opcode is in the payload; no
    // separate local-operation field is needed in this header.
    for (std::size_t word = 0; word < Packet::kWordCount; ++word)
        packet.words[word].lanes[0] =
            static_cast<std::uint32_t>(IcuCommandOpcode::Extended)
            | (static_cast<std::uint32_t>(word) << 2)
            | (std::uint32_t {1} << 6);
    packet.words[0].lanes[2] |= std::uint32_t {2} << 24;
}

template <typename Packet>
void validate_headers(const Packet& packet, const char* message)
{
    if (((packet.words[0].lanes[2] >> 24) & 0xfU) != 2U)
        throw std::logic_error(message);
    for (std::size_t word = 0; word < Packet::kWordCount; ++word) {
        const auto expected =
            static_cast<std::uint32_t>(IcuCommandOpcode::Extended)
            | (static_cast<std::uint32_t>(word) << 2)
            | (std::uint32_t {1} << 6);
        if ((packet.words[word].lanes[0] & 0xffU) != expected)
            throw std::logic_error(message);
    }
}

} // namespace sxm_run_2d_codec_detail

struct EncodedSxmIcuRun2DWord {
    std::array<std::uint32_t, 3> lanes{};
};

struct EncodedSxmIcuRun2DPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 6;
    static constexpr std::size_t kLanesPerWord = 3;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 88;
    std::array<EncodedSxmIcuRun2DWord, kWordCount> words{};
};

static_assert(sizeof(EncodedSxmIcuRun2DWord) == 96 / 8);
static_assert(sizeof(EncodedSxmIcuRun2DPacket) == 6 * 96 / 8);
static_assert(std::is_trivially_copyable_v<EncodedSxmIcuRun2DPacket>);

inline EncodedSxmIcuRun2DPacket encode_sxm_icu_run_2d_instruction(
    const SxmIcuRun2DInstruction& instruction)
{
    ::ftlpu::detail::validate_sxm_icu_run_2d_instruction(instruction);
    namespace codec = fu_3d_codec_detail;
    EncodedSxmIcuRun2DPacket packet{};
    sxm_run_2d_codec_detail::initialize_headers(packet);
    codec::write_unsigned(packet, 0, 24, instruction.loop.wait_cycle,
        "SXM RUN_2D wait_cycle exceeds 24 bits");
    for (std::size_t dimension = 0; dimension < 2; ++dimension) {
        codec::write_unsigned(packet, 24 + 16 * dimension, 16,
            instruction.loop.counts[dimension] - 1,
            "SXM RUN_2D count");
        codec::write_unsigned(packet, 56 + 24 * dimension, 24,
            instruction.loop.cycle_strides[dimension],
            "SXM RUN_2D cycle stride");
    }
    const auto encoded = encode_sxm_instruction(instruction.instruction);
    for (std::size_t word = 0; word < encoded.words.size(); ++word)
        codec::write_unsigned(packet, 104 + 32 * word, 32,
            encoded.words[word], "SXM RUN_2D tile instruction");
    codec::write_unsigned(packet, 520, 2,
        instruction.permute_map_stride / hw::kLanesPerTile,
        "SXM RUN_2D permute-map group stride");
    return packet;
}

inline SxmIcuRun2DInstruction decode_sxm_icu_run_2d_instruction(
    const EncodedSxmIcuRun2DPacket& packet)
{
    namespace codec = fu_3d_codec_detail;
    sxm_run_2d_codec_detail::validate_headers(
        packet, "invalid SXM RUN_2D packet header");
    codec::require_reserved_zero(
        packet, 522, "SXM RUN_2D reserved field is non-zero");
    EncodedSxmInstruction encoded{};
    for (std::size_t word = 0; word < encoded.words.size(); ++word)
        encoded.words[word] = static_cast<std::uint32_t>(
            codec::read_unsigned(packet, 104 + 32 * word, 32));
    auto result = SxmIcuRun2DInstruction::Run2D(
        0,
        {static_cast<std::size_t>(codec::read_unsigned(packet, 24, 16)) + 1,
            static_cast<std::size_t>(codec::read_unsigned(packet, 40, 16)) + 1},
        {static_cast<std::size_t>(codec::read_unsigned(packet, 56, 24)),
            static_cast<std::size_t>(codec::read_unsigned(packet, 80, 24))},
        decode_sxm_instruction(encoded),
        static_cast<std::size_t>(codec::read_unsigned(packet, 520, 2))
            * hw::kLanesPerTile);
    result.loop.wait_cycle = static_cast<std::size_t>(
        codec::read_unsigned(packet, 0, 24));
    ::ftlpu::detail::validate_sxm_icu_run_2d_instruction(result);
    return result;
}

} // namespace isa
} // namespace ftlpu
