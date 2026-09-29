#pragma once

#include "ftlpu/icu/fu_3d_codec.hpp"
#include "ftlpu/sxm/instruction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ftlpu {

// Transpose and Permute have independent physical ICU queues. Their packets
// therefore carry only the fields consumed by that queue. The ICU repeats the
// tile operation over at most two outer launch coordinates.
struct SxmIcuRun2DInstruction {
    IcuLoop3D loop{};
    SxmInstruction instruction{};
    // Add this modulo-32 delta to every permute-map entry after each launch.
    std::size_t permute_map_stride{0};

    static SxmIcuRun2DInstruction Run2D(
        std::size_t startCycle,
        std::array<std::size_t, 2> counts,
        std::array<std::size_t, 2> cycleStrides,
        SxmInstruction instruction,
        std::size_t permuteMapStride = 0)
    {
        return {{startCycle, {counts[0], counts[1], 1},
                    {cycleStrides[0], cycleStrides[1], 1}},
            std::move(instruction), permuteMapStride};
    }
};

namespace detail {

inline void validate_sxm_stream_range(
    const SxmInstruction::StreamList& streams,
    std::size_t expectedCount, const char* role)
{
    if (streams.size() != expectedCount)
        throw std::invalid_argument(
            std::string("SXM ") + role + " requires "
            + std::to_string(expectedCount) + " consecutive streams");
    if (streams.empty()) return;
    if (streams.front().stream >= hw::kStreams
        || streams.front().stream + streams.size() > hw::kStreams)
        throw std::invalid_argument(
            std::string("SXM ") + role + " stream range is outside hardware");
    for (std::size_t index = 0; index < streams.size(); ++index) {
        if (streams[index].stream != streams.front().stream + index)
            throw std::invalid_argument(
                std::string("SXM ") + role
                + " streams must be one consecutive hardware range");
    }
}

inline void validate_sxm_icu_run_2d_loop(
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
}

inline void validate_sxm_transpose_icu_run_2d_instruction(
    const SxmIcuRun2DInstruction& run)
{
    validate_sxm_icu_run_2d_loop(run);
    const auto& instruction = run.instruction;
    if (instruction.opcode != SxmOpcode::Transpose)
        throw std::invalid_argument(
            "SXM transpose RUN_2D requires a Transpose instruction");
    if (run.permute_map_stride != 0)
        throw std::invalid_argument(
            "SXM transpose RUN_2D cannot carry a permute-map stride");
    if (instruction.output_row != SxmInstruction::kAllOutputRows
        || instruction.output_tile != SxmInstruction::kAllOutputTiles)
        throw std::invalid_argument(
            "SXM transpose RUN_2D cannot carry permute output selectors");
    const auto sourceCount =
        instruction.input_row == SxmInstruction::kAllInputRows
        ? 2 * hw::kLanesPerTile : std::size_t {2};
    if (instruction.input_row != SxmInstruction::kAllInputRows
        && instruction.input_row >= hw::kLanesPerTile)
        throw std::invalid_argument(
            "SXM transpose input row is outside the physical tile");
    validate_sxm_stream_range(
        instruction.src_streams, sourceCount, "transpose source");
    validate_sxm_stream_range(
        instruction.dst_streams, 2 * hw::kLanesPerTile,
        "transpose destination");
}

inline void validate_sxm_permute_icu_run_2d_instruction(
    const SxmIcuRun2DInstruction& run)
{
    validate_sxm_icu_run_2d_loop(run);
    const auto& instruction = run.instruction;
    if (instruction.opcode != SxmOpcode::Permute)
        throw std::invalid_argument(
            "SXM permute RUN_2D requires a Permute instruction");
    if (instruction.input_row != SxmInstruction::kAllInputRows)
        throw std::invalid_argument(
            "SXM permute RUN_2D cannot carry a transpose input selector");
    if (instruction.output_row != SxmInstruction::kAllOutputRows
        && instruction.output_row >= hw::kLanesPerTile)
        throw std::invalid_argument(
            "SXM permute output row is outside the physical tile");
    if (instruction.output_tile != SxmInstruction::kAllOutputTiles
        && instruction.output_tile >= hw::kTileRows)
        throw std::invalid_argument(
            "SXM permute output tile is outside the physical SXM");
    validate_sxm_stream_range(
        instruction.src_streams, 2 * hw::kLanesPerTile,
        "permute source");
    validate_sxm_stream_range(
        instruction.dst_streams, 2 * hw::kLanesPerTile,
        "permute destination");
    for (const auto source : instruction.permute_map) {
        if (source >= SxmInstruction::kTotalLanes)
            throw std::invalid_argument(
                "SXM permute map source is outside the physical vector plane");
    }
}

inline void validate_sxm_icu_run_2d_instruction(
    const SxmIcuRun2DInstruction& instruction)
{
    if (instruction.instruction.opcode == SxmOpcode::Transpose)
        validate_sxm_transpose_icu_run_2d_instruction(instruction);
    else if (instruction.instruction.opcode == SxmOpcode::Permute)
        validate_sxm_permute_icu_run_2d_instruction(instruction);
    else
        throw std::invalid_argument(
            "SXM RUN_2D supports only the physical transpose and permute queues");
}

} // namespace detail

namespace isa {

namespace sxm_run_2d_codec_detail {

template <typename Packet>
void initialize_headers(Packet& packet)
{
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

template <typename Packet>
void encode_loop(Packet& packet, const SxmIcuRun2DInstruction& instruction)
{
    namespace codec = fu_3d_codec_detail;
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
}

template <typename Packet>
SxmIcuRun2DInstruction decode_loop(const Packet& packet,
    SxmInstruction instruction, std::size_t permuteMapStride = 0)
{
    namespace codec = fu_3d_codec_detail;
    auto result = SxmIcuRun2DInstruction::Run2D(
        0,
        {static_cast<std::size_t>(codec::read_unsigned(packet, 24, 16)) + 1,
            static_cast<std::size_t>(codec::read_unsigned(packet, 40, 16)) + 1},
        {static_cast<std::size_t>(codec::read_unsigned(packet, 56, 24)),
            static_cast<std::size_t>(codec::read_unsigned(packet, 80, 24))},
        std::move(instruction), permuteMapStride);
    result.loop.wait_cycle = static_cast<std::size_t>(
        codec::read_unsigned(packet, 0, 24));
    return result;
}

inline SxmInstruction::StreamList stream_range(
    std::size_t base, std::size_t count)
{
    auto result = SxmInstruction::StreamList{};
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        result.push_back(SxmStreamId {base + index});
    return result;
}

} // namespace sxm_run_2d_codec_detail

struct EncodedSxmIcuRun2DWord {
    std::array<std::uint32_t, 3> lanes{};
};

struct EncodedSxmTransposeIcuRun2DPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 2;
    static constexpr std::size_t kLanesPerWord = 3;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 88;
    std::array<EncodedSxmIcuRun2DWord, kWordCount> words{};
};

struct EncodedSxmPermuteIcuRun2DPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 4;
    static constexpr std::size_t kLanesPerWord = 3;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 88;
    std::array<EncodedSxmIcuRun2DWord, kWordCount> words{};
};

static_assert(sizeof(EncodedSxmIcuRun2DWord) == 96 / 8);
static_assert(sizeof(EncodedSxmTransposeIcuRun2DPacket) == 2 * 96 / 8);
static_assert(sizeof(EncodedSxmPermuteIcuRun2DPacket) == 4 * 96 / 8);
static_assert(std::is_trivially_copyable_v<EncodedSxmTransposeIcuRun2DPacket>);
static_assert(std::is_trivially_copyable_v<EncodedSxmPermuteIcuRun2DPacket>);

inline EncodedSxmTransposeIcuRun2DPacket
encode_sxm_transpose_icu_run_2d_instruction(
    const SxmIcuRun2DInstruction& instruction)
{
    ::ftlpu::detail::validate_sxm_transpose_icu_run_2d_instruction(instruction);
    namespace codec = fu_3d_codec_detail;
    EncodedSxmTransposeIcuRun2DPacket packet{};
    sxm_run_2d_codec_detail::initialize_headers(packet);
    sxm_run_2d_codec_detail::encode_loop(packet, instruction);
    codec::write_unsigned(packet, 104, 6,
        instruction.instruction.src_streams.front().stream,
        "SXM transpose source stream base");
    codec::write_unsigned(packet, 110, 6,
        instruction.instruction.dst_streams.front().stream,
        "SXM transpose destination stream base");
    const auto inputRow =
        instruction.instruction.input_row == SxmInstruction::kAllInputRows
        ? hw::kLanesPerTile : instruction.instruction.input_row;
    codec::write_unsigned(packet, 116, 4, inputRow,
        "SXM transpose input row");
    return packet;
}

inline SxmIcuRun2DInstruction decode_sxm_transpose_icu_run_2d_instruction(
    const EncodedSxmTransposeIcuRun2DPacket& packet)
{
    namespace codec = fu_3d_codec_detail;
    sxm_run_2d_codec_detail::validate_headers(
        packet, "invalid SXM transpose RUN_2D packet header");
    codec::require_reserved_zero(
        packet, 120, "SXM transpose RUN_2D reserved field is non-zero");
    const auto inputRow = static_cast<std::size_t>(
        codec::read_unsigned(packet, 116, 4));
    if (inputRow > hw::kLanesPerTile)
        throw std::logic_error("encoded SXM transpose input row is invalid");
    const auto sourceCount = inputRow == hw::kLanesPerTile
        ? 2 * hw::kLanesPerTile : std::size_t {2};
    auto instruction = SxmInstruction::Transpose(
        sxm_run_2d_codec_detail::stream_range(
            static_cast<std::size_t>(codec::read_unsigned(packet, 104, 6)),
            sourceCount),
        sxm_run_2d_codec_detail::stream_range(
            static_cast<std::size_t>(codec::read_unsigned(packet, 110, 6)),
            2 * hw::kLanesPerTile));
    instruction.input_row = inputRow == hw::kLanesPerTile
        ? SxmInstruction::kAllInputRows : inputRow;
    auto result = sxm_run_2d_codec_detail::decode_loop(
        packet, std::move(instruction));
    ::ftlpu::detail::validate_sxm_transpose_icu_run_2d_instruction(result);
    return result;
}

inline EncodedSxmPermuteIcuRun2DPacket
encode_sxm_permute_icu_run_2d_instruction(
    const SxmIcuRun2DInstruction& instruction)
{
    ::ftlpu::detail::validate_sxm_permute_icu_run_2d_instruction(instruction);
    namespace codec = fu_3d_codec_detail;
    EncodedSxmPermuteIcuRun2DPacket packet{};
    sxm_run_2d_codec_detail::initialize_headers(packet);
    sxm_run_2d_codec_detail::encode_loop(packet, instruction);
    codec::write_unsigned(packet, 104, 6,
        instruction.instruction.src_streams.front().stream,
        "SXM permute source stream base");
    codec::write_unsigned(packet, 110, 6,
        instruction.instruction.dst_streams.front().stream,
        "SXM permute destination stream base");
    for (std::size_t lane = 0; lane < SxmInstruction::kTotalLanes; ++lane)
        codec::write_unsigned(packet, 116 + lane * 5, 5,
            instruction.instruction.permute_map[lane],
            "SXM permute map source");
    const auto outputRow =
        instruction.instruction.output_row == SxmInstruction::kAllOutputRows
        ? hw::kLanesPerTile : instruction.instruction.output_row;
    const auto outputTile =
        instruction.instruction.output_tile == SxmInstruction::kAllOutputTiles
        ? hw::kTileRows : instruction.instruction.output_tile;
    codec::write_unsigned(packet, 276, 4, outputRow,
        "SXM permute output row");
    codec::write_unsigned(packet, 280, 3, outputTile,
        "SXM permute output tile");
    codec::write_unsigned(packet, 283, 2,
        instruction.permute_map_stride / hw::kLanesPerTile,
        "SXM RUN_2D permute-map group stride");
    return packet;
}

inline SxmIcuRun2DInstruction decode_sxm_permute_icu_run_2d_instruction(
    const EncodedSxmPermuteIcuRun2DPacket& packet)
{
    namespace codec = fu_3d_codec_detail;
    sxm_run_2d_codec_detail::validate_headers(
        packet, "invalid SXM permute RUN_2D packet header");
    codec::require_reserved_zero(
        packet, 285, "SXM permute RUN_2D reserved field is non-zero");
    auto map = SxmInstruction::PermuteMap{};
    for (std::size_t lane = 0; lane < map.size(); ++lane)
        map[lane] = static_cast<std::size_t>(
            codec::read_unsigned(packet, 116 + lane * 5, 5));
    auto instruction = SxmInstruction::Permute(
        sxm_run_2d_codec_detail::stream_range(
            static_cast<std::size_t>(codec::read_unsigned(packet, 104, 6)),
            2 * hw::kLanesPerTile),
        sxm_run_2d_codec_detail::stream_range(
            static_cast<std::size_t>(codec::read_unsigned(packet, 110, 6)),
            2 * hw::kLanesPerTile),
        map);
    const auto outputRow = static_cast<std::size_t>(
        codec::read_unsigned(packet, 276, 4));
    const auto outputTile = static_cast<std::size_t>(
        codec::read_unsigned(packet, 280, 3));
    if (outputRow > hw::kLanesPerTile || outputTile > hw::kTileRows)
        throw std::logic_error("encoded SXM permute output selector is invalid");
    instruction.output_row = outputRow == hw::kLanesPerTile
        ? SxmInstruction::kAllOutputRows : outputRow;
    instruction.output_tile = outputTile == hw::kTileRows
        ? SxmInstruction::kAllOutputTiles : outputTile;
    const auto mapStride = static_cast<std::size_t>(
        codec::read_unsigned(packet, 283, 2)) * hw::kLanesPerTile;
    auto result = sxm_run_2d_codec_detail::decode_loop(
        packet, std::move(instruction), mapStride);
    ::ftlpu::detail::validate_sxm_permute_icu_run_2d_instruction(result);
    return result;
}

} // namespace isa
} // namespace ftlpu
