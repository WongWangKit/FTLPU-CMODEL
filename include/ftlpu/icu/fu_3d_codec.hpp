#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/icu/fu_3d_instruction.hpp"
#include "ftlpu/icu/packet_bit_codec.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ftlpu {

namespace isa {

struct MxmIcuEncodingLayout {
    static constexpr unsigned kAddressBits = MxmEncodingLayout::kAddressBits;
    static constexpr unsigned kColumnBits = MxmEncodingLayout::kColumnBits;
    static constexpr unsigned kStrideBits = MxmEncodingLayout::kStrideBits;
    static constexpr unsigned kLoadStrides = 147 + kColumnBits;
    static constexpr unsigned kLoadStream = kLoadStrides + 3 * 16;
    static constexpr unsigned kLoadInputMode = kLoadStream + 5;
    static constexpr unsigned kLoadEnd = kLoadInputMode + 1;
    static constexpr unsigned kComputeStrides = 157 + kAddressBits;
    static constexpr unsigned kRowStride = kComputeStrides + 3 * kStrideBits;
    static constexpr unsigned kFormat = kRowStride + kAddressBits;
    static constexpr unsigned kRegularMode = kFormat + 1;
    static constexpr unsigned kTerminalDimension = kRegularMode + 3;
    static constexpr unsigned kTerminalMode = kTerminalDimension + 2;
    static constexpr unsigned kComputeEnd = kTerminalMode + 3;
    static constexpr unsigned kDecodeStrides = 152 + kAddressBits;
    static constexpr unsigned kDecodeColumn = kDecodeStrides + 3 * kStrideBits;
    static constexpr unsigned kDecodeDestination = kDecodeColumn + kColumnBits;
    static constexpr unsigned kDecodeClear = kDecodeDestination + 1;
    static constexpr unsigned kDecodeFormat = kDecodeClear + 1;
    static constexpr unsigned kDecodeLayout = kDecodeFormat + 1;
    static constexpr unsigned kDecodeEnd = kDecodeLayout + 1;
    static_assert(kDecodeEnd <= kComputeEnd);
};

// Each functional queue has its own physical word and packet type. The packet
// type supplies the instruction kind for the three MXM queues; consequently no
// unit selector is carried in any word. Word 0 uses the existing global
// Extended-control discriminator at physical [91:88]: subtype 1 is Repeat2D
// and subtype 2 is an FU-specific 3-D packet. Testing that nibble before any
// local header bits makes the two formats disjoint. Every physical word then
// starts with this local 8-bit header:
//   [1:0] global ICU envelope opcode 0b11 (Extended)
//   [3:2] word index within the fixed-size packet
//   [5:4] FU-local operation (MEM uses 0=read, 1=write, 2=write-tap;
//         MXM load/dequant use 0, and MXM compute uses 0=compute,
//         1=accumulator-read)
//   [6]   FU-local 3-D packet marker
//   [7]   format version, currently zero
// A raw local-iMEM decoder can classify words[0] from this byte before it
// collects the continuation words. Every continuation repeats marker, version,
// and local operation, and carries its expected word index. The logical packet
// payload excludes every low-byte local header and word-0 physical [91:88].
// Layouts below use P[n] to number that payload; payload numbering closes the
// four-bit subtype hole.
//
// The common loop layout in every packet is:
//   [ 23:  0] wait_cycle before the first FU issue, relative to packet decode
//              and the previous descriptor's retirement.
//   [ 39: 24] count[0] - 1
//   [ 55: 40] count[1] - 1
//   [ 71: 56] count[2] - 1
//   [ 95: 72] cycle_stride[0]
//   [119: 96] cycle_stride[1]
//   [143:120] cycle_stride[2]
// Counter 0 is innermost. A count field of zero therefore represents one
// iteration, while all cycle strides are encoded directly and must be nonzero.

struct EncodedMemIcu3DWord {
    std::array<std::uint32_t, 3> lanes{};
};

struct EncodedMxmLoadIcu3DWord {
    std::array<std::uint32_t, 4> lanes{};
};

struct EncodedMxmDequantIcu3DWord {
    std::array<std::uint32_t, 4> lanes{};
};

struct EncodedMxmComputeIcu3DWord {
    std::array<std::uint32_t, 4> lanes{};
};

struct EncodedMemIcu3DPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 3;
    static constexpr std::size_t kLanesPerWord = 3;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 88;
    std::array<EncodedMemIcu3DWord, kWordCount> words{};
};

struct EncodedMemIcuWriteRead2DPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 3;
    static constexpr std::size_t kLanesPerWord = 3;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 88;
    std::array<EncodedMemIcu3DWord, kWordCount> words{};
};

struct EncodedMxmLoadIcu3DPacket {
    static constexpr std::size_t kWordBits = 128;
    static constexpr std::size_t kWordCount = 2;
    static constexpr std::size_t kLanesPerWord = 4;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 120;
    std::array<EncodedMxmLoadIcu3DWord, kWordCount> words{};
};

struct EncodedMxmDequantIcu3DPacket {
    static constexpr std::size_t kWordBits = 128;
    static constexpr std::size_t kWordCount = 2;
    static constexpr std::size_t kLanesPerWord = 4;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 120;
    std::array<EncodedMxmDequantIcu3DWord, kWordCount> words{};
};

struct EncodedMxmComputeIcu3DPacket {
    static constexpr std::size_t kWordBits = 128;
    static constexpr std::size_t kRequiredWords =
        (MxmIcuEncodingLayout::kComputeEnd + 4 + 119) / 120;
    static constexpr std::size_t kWordCount = kRequiredWords < 2 ? 2 : kRequiredWords;
    static constexpr std::size_t kLanesPerWord = 4;
    static constexpr std::size_t kHeaderBits = 8;
    static constexpr std::size_t kPayloadBitsPerWord = 120;
    std::array<EncodedMxmComputeIcu3DWord, kWordCount> words{};
};

static_assert(hw::kIcuMemInstructionBits == 96);
static_assert(hw::kIcuMxmInstructionBits == 128);
static_assert(hw::kSramDepthRows <= (std::size_t {1} << 13));
static_assert(hw::kMxmAccumulatorRows <= (std::size_t {1} << MxmEncodingLayout::kAddressBits));
static_assert(hw::kStreams <= (std::size_t {1} << 6));
static_assert(hw::kEastStreams <= (std::size_t {1} << 5));
static_assert(hw::kWestStreams <= (std::size_t {1} << 5));
static_assert(hw::kMxmSupercellsPerPlane <= (std::size_t {1} << MxmEncodingLayout::kColumnBits));
static_assert(sizeof(EncodedMemIcu3DWord) == 96 / 8);
static_assert(sizeof(EncodedMxmLoadIcu3DWord) == 128 / 8);
static_assert(sizeof(EncodedMxmDequantIcu3DWord) == 128 / 8);
static_assert(sizeof(EncodedMxmComputeIcu3DWord) == 128 / 8);
static_assert(sizeof(EncodedMemIcu3DPacket) == 3 * 96 / 8);
static_assert(sizeof(EncodedMemIcuWriteRead2DPacket) == 3 * 96 / 8);
static_assert(sizeof(EncodedMxmLoadIcu3DPacket) == 2 * 128 / 8);
static_assert(sizeof(EncodedMxmDequantIcu3DPacket) == 2 * 128 / 8);
static_assert(sizeof(EncodedMxmComputeIcu3DPacket)
    == EncodedMxmComputeIcu3DPacket::kWordCount * 128 / 8);
static_assert(std::is_trivially_copyable_v<EncodedMemIcu3DPacket>);
static_assert(std::is_trivially_copyable_v<EncodedMemIcuWriteRead2DPacket>);
static_assert(std::is_trivially_copyable_v<EncodedMxmLoadIcu3DPacket>);
static_assert(std::is_trivially_copyable_v<EncodedMxmDequantIcu3DPacket>);
static_assert(std::is_trivially_copyable_v<EncodedMxmComputeIcu3DPacket>);

namespace fu_3d_codec_detail {

constexpr std::size_t kLoopBits = 144;
constexpr std::uint8_t kFu3DExtendedSubtype = 2;
constexpr std::uint8_t kMemWriteRead2DExtendedSubtype = 7;
constexpr std::uint8_t kExtendedEnvelope =
    static_cast<std::uint8_t>(IcuCommandOpcode::Extended);
constexpr std::uint8_t kPacketClass3D = 1;
constexpr std::uint8_t kFormatVersion = 0;
constexpr std::uint8_t kMxmLocalOperation = 0;
static_assert(kExtendedEnvelope == 0b11);

template <typename Packet>
void initialize_word_headers(Packet& packet, std::uint8_t local_operation,
    std::uint8_t subtype = kFu3DExtendedSubtype)
{
    static_assert(Packet::kWordCount <= 4);
    if (local_operation > 3)
        throw std::logic_error("FU 3-D local operation exceeds its header field");
    for (std::size_t word = 0; word < Packet::kWordCount; ++word)
        packet.words[word].lanes[0] =
            static_cast<std::uint32_t>(kExtendedEnvelope)
            | (static_cast<std::uint32_t>(word) << 2)
            | (static_cast<std::uint32_t>(local_operation) << 4)
            | (static_cast<std::uint32_t>(kPacketClass3D) << 6)
            | (static_cast<std::uint32_t>(kFormatVersion) << 7);
    constexpr auto lane = kExtendedSubtypePhysicalOffset / 32;
    constexpr auto shift = kExtendedSubtypePhysicalOffset % 32;
    packet.words[0].lanes[lane] |=
        static_cast<std::uint32_t>(subtype) << shift;
}

template <typename Packet>
void validate_word_headers(const Packet& packet,
    std::uint8_t local_operation, const char* message,
    std::uint8_t subtype = kFu3DExtendedSubtype)
{
    if (local_operation > 3) throw std::logic_error(message);
    constexpr auto subtype_lane = kExtendedSubtypePhysicalOffset / 32;
    constexpr auto subtype_shift = kExtendedSubtypePhysicalOffset % 32;
    constexpr auto subtype_mask =
        (std::uint32_t {1} << kExtendedSubtypeBits) - 1;
    if (((packet.words[0].lanes[subtype_lane] >> subtype_shift)
            & subtype_mask)
        != subtype)
        throw std::logic_error(message);
    for (std::size_t word = 0; word < Packet::kWordCount; ++word) {
        const auto header = packet.words[word].lanes[0] & 0xffU;
        const auto expected =
            static_cast<std::uint32_t>(kExtendedEnvelope)
            | (static_cast<std::uint32_t>(word) << 2)
            | (static_cast<std::uint32_t>(local_operation) << 4)
            | (static_cast<std::uint32_t>(kPacketClass3D) << 6)
            | (static_cast<std::uint32_t>(kFormatVersion) << 7);
        if (header != expected) throw std::logic_error(message);
    }
}

template <typename Packet>
std::uint8_t decode_word0_local_operation(
    const Packet& packet, const char* message,
    std::uint8_t expected_subtype = kFu3DExtendedSubtype)
{
    const auto header = packet.words[0].lanes[0] & 0xffU;
    const auto envelope = static_cast<std::uint8_t>(header & 3U);
    const auto word_index = static_cast<std::uint8_t>((header >> 2) & 3U);
    const auto packet_class = static_cast<std::uint8_t>((header >> 6) & 1U);
    const auto version = static_cast<std::uint8_t>((header >> 7) & 1U);
    constexpr auto subtype_lane = kExtendedSubtypePhysicalOffset / 32;
    constexpr auto subtype_shift = kExtendedSubtypePhysicalOffset % 32;
    constexpr auto subtype_mask =
        (std::uint32_t {1} << kExtendedSubtypeBits) - 1;
    const auto subtype = static_cast<std::uint8_t>(
        (packet.words[0].lanes[subtype_lane] >> subtype_shift)
        & subtype_mask);
    if (envelope != kExtendedEnvelope || word_index != 0
        || packet_class != kPacketClass3D || version != kFormatVersion
        || subtype != expected_subtype)
        throw std::logic_error(message);
    return static_cast<std::uint8_t>((header >> 4) & 3U);
}

template <typename Packet>
void encode_loop(Packet& packet, const IcuLoop3D& loop)
{
    ::ftlpu::detail::validate_icu_loop_3d(loop);
    write_unsigned(packet, 0, 24, loop.wait_cycle,
        "FU 3-D wait_cycle does not fit 24 bits");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
        write_unsigned(packet, 24 + 16 * dimension, 16,
            loop.counts[dimension] - 1,
            "FU 3-D loop count does not fit 16 bits");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
        write_unsigned(packet, 72 + 24 * dimension, 24,
            loop.cycle_strides[dimension],
            "FU 3-D cycle stride does not fit 24 bits");
}

template <typename Packet>
IcuLoop3D decode_loop(const Packet& packet)
{
    IcuLoop3D loop{};
    loop.start_cycle = 0;
    loop.wait_cycle = static_cast<std::size_t>(
        read_unsigned(packet, 0, 24));
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
        loop.counts[dimension] = static_cast<std::size_t>(
            read_unsigned(packet, 24 + 16 * dimension, 16) + 1);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
        loop.cycle_strides[dimension] = static_cast<std::size_t>(
            read_unsigned(packet, 72 + 24 * dimension, 24));
    ::ftlpu::detail::validate_icu_loop_3d(loop);
    return loop;
}

inline void validate_buffer_mode(MxmIcuBufferMode mode)
{
    switch (mode) {
    case MxmIcuBufferMode::Fixed:
    case MxmIcuBufferMode::ToggleDimension0:
    case MxmIcuBufferMode::ToggleDimension1:
    case MxmIcuBufferMode::ToggleDimension2:
        return;
    }
    throw std::invalid_argument("MXM ICU 3-D buffer mode is invalid");
}

inline void validate_weight_input_mode(MxmWeightInputMode mode)
{
    switch (mode) {
    case MxmWeightInputMode::Int8DequantBf16:
    case MxmWeightInputMode::Direct16:
        return;
    }
    throw std::invalid_argument("MXM ICU 3-D weight input mode is invalid");
}

inline std::uint64_t encode_compute_mode(MxmComputeIcuMode mode)
{
    ::ftlpu::detail::validate_mxm_compute_mode(mode);
    return static_cast<std::uint64_t>(mode.accumulator_destination)
        | (static_cast<std::uint64_t>(mode.accumulator_clear) << 1)
        | (static_cast<std::uint64_t>(mode.accumulator_output_format) << 2);
}

inline MxmComputeIcuMode decode_compute_mode(std::uint64_t encoded)
{
    auto mode = MxmComputeIcuMode {};
    mode.accumulator_destination =
        static_cast<MxmAccumulatorDestination>(encoded & 1U);
    mode.accumulator_clear = ((encoded >> 1) & 1U) != 0;
    mode.accumulator_output_format =
        static_cast<MxmAccumulatorOutputFormat>((encoded >> 2) & 1U);
    ::ftlpu::detail::validate_mxm_compute_mode(mode);
    return mode;
}

} // namespace fu_3d_codec_detail

// Subtypes 8/9/10 are sequential singleton/1D/2D packets. These are
// independent physical encodings, not multiple in-flight descriptors.
template <typename Word>
int short_icu_rank(const Word& word)
{
    const auto subtype = (word.lanes[2] >> 24) & 15U;
    return subtype >= 8 && subtype <= 10 ? static_cast<int>(subtype - 8) : -1;
}

template <typename Packet>
constexpr bool supports_short_icu_packet =
    std::is_same_v<Packet, EncodedMemIcu3DPacket>
    || std::is_same_v<Packet, EncodedMxmLoadIcu3DPacket>
    || std::is_same_v<Packet, EncodedMxmDequantIcu3DPacket>
    || std::is_same_v<Packet, EncodedMxmComputeIcu3DPacket>;

template <typename Packet>
constexpr std::size_t short_icu_tail_bits()
{
    if constexpr (std::is_same_v<Packet, EncodedMxmLoadIcu3DPacket>)
        return MxmIcuEncodingLayout::kLoadEnd - 144;
    else if constexpr (std::is_same_v<Packet, EncodedMxmComputeIcu3DPacket>)
        return MxmIcuEncodingLayout::kComputeEnd - 144;
    else return 16; // Dequant scale.
}

template <typename Packet>
std::size_t short_icu_used_bits(std::size_t rank)
{
    if constexpr (std::is_same_v<Packet, EncodedMemIcu3DPacket>)
        return 43 + 60 * rank; // wait + axes + stream/base + address strides
    else return 24 + (rank ? 2 : 0) + 40 * rank + short_icu_tail_bits<Packet>();
}

template <typename Packet, typename Word>
std::size_t fu_loop_packet_word_count(const Word& first)
{
    if constexpr (supports_short_icu_packet<Packet>) {
        const auto rank = short_icu_rank(first);
        if (rank >= 0)
            return (short_icu_used_bits<Packet>(rank) + 4
                + Packet::kPayloadBitsPerWord - 1) / Packet::kPayloadBitsPerWord;
    }
    return Packet::kWordCount;
}

template <typename Packet>
std::size_t fu_loop_packet_word_count(const Packet& packet)
{
    return fu_loop_packet_word_count<Packet>(packet.words[0]);
}

template <typename Packet>
void initialize_short_icu_packet(Packet& packet, unsigned operation, unsigned rank)
{
    fu_3d_codec_detail::initialize_word_headers(packet, operation, 8 + rank);
    const auto words = fu_loop_packet_word_count(packet);
    for (auto i = words; i < Packet::kWordCount; ++i) packet.words[i] = {};
}

template <typename Packet>
IcuLoop3D decode_short_icu_loop(const Packet& packet)
{
    namespace codec = fu_3d_codec_detail;
    const auto rank = short_icu_rank(packet.words[0]);
    if (rank < 0) throw std::logic_error("invalid short ICU subtype");
    const auto op = codec::decode_word0_local_operation(packet,
        "invalid short ICU header", 8 + rank);
    Packet headers{};
    initialize_short_icu_packet(headers, op, rank);
    for (std::size_t w = 0; w < fu_loop_packet_word_count(packet); ++w)
        if ((packet.words[w].lanes[0] & 255U) != (headers.words[w].lanes[0] & 255U))
            throw std::logic_error("invalid short ICU continuation header");
    codec::require_reserved_zero(packet, short_icu_used_bits<Packet>(rank),
        "nonzero short ICU reserved bits");
    IcuLoop3D loop{};
    loop.wait_cycle = codec::read_unsigned(packet, 0, 24);
    unsigned selector = 0;
    constexpr bool mxm = !std::is_same_v<Packet, EncodedMemIcu3DPacket>;
    const auto axisBits = mxm && rank ? 2 : 0;
    if (axisBits) {
        selector = static_cast<unsigned>(codec::read_unsigned(packet, 24, 2));
        if (selector > 2) throw std::logic_error("invalid MXM short axis selector");
    }
    int n = 0;
    for (int d = 0; d < 3; ++d) {
        const bool active = mxm ? (rank == 1 ? d == selector : rank == 2 && d != selector) : d < rank;
        if (!active) continue;
        loop.counts[d] = codec::read_unsigned(packet, 24 + axisBits + n * 16, 16) + 1;
        loop.cycle_strides[d] = codec::read_unsigned(packet, 24 + axisBits + rank * 16 + n * 24, 24);
        ++n;
    }
    ::ftlpu::detail::validate_icu_loop_3d(loop);
    return loop;
}

template <typename Packet>
void encode_short_icu_loop(Packet& packet, const IcuLoop3D& loop, unsigned rank)
{
    namespace codec = fu_3d_codec_detail;
    codec::write_unsigned(packet, 0, 24, loop.wait_cycle, "short ICU wait overflow");
    constexpr bool mxm = !std::is_same_v<Packet, EncodedMemIcu3DPacket>;
    const auto axisBits = mxm && rank ? 2 : 0;
    if (axisBits) {
        unsigned selector = 0;
        for (unsigned d = 0; d < 3; ++d)
            if ((rank == 1 && loop.counts[d] > 1) || (rank == 2 && loop.counts[d] == 1)) selector = d;
        codec::write_unsigned(packet, 24, 2, selector, "MXM axis selector");
    }
    unsigned n = 0;
    for (unsigned d = 0; d < 3; ++d) {
        if (mxm ? loop.counts[d] == 1 : d >= rank) continue;
        codec::write_unsigned(packet, 24 + axisBits + n * 16, 16, loop.counts[d] - 1, "short ICU count overflow");
        codec::write_unsigned(packet, 24 + axisBits + rank * 16 + n * 24, 24,
            loop.cycle_strides[d], "short ICU cycle stride overflow");
        ++n;
    }
}

// MXM short packets keep the FU-specific payload, including buffer/terminal
// modes, and omit the inactive loop axes. The software storage object is a
// maximum-sized buffer; only fu_loop_packet_word_count() words enter iMEM.
template <typename Packet>
Packet expand_short_mxm_packet(const Packet& packet)
{
    namespace codec = fu_3d_codec_detail;
    const auto rank = short_icu_rank(packet.words[0]);
    const auto loop = decode_short_icu_loop(packet);
    Packet full{};
    codec::initialize_word_headers(full, (packet.words[0].lanes[0] >> 4) & 3U);
    codec::encode_loop(full, loop);
    for (std::size_t b = 0; b < short_icu_tail_bits<Packet>(); ++b)
        codec::write_unsigned(full, 144 + b, 1,
            codec::read_unsigned(packet, 24 + (rank ? 2 : 0) + 40 * rank + b, 1), "MXM payload bit");
    return full;
}

template <typename Packet>
Packet compact_mxm_packet(const Packet& full, const IcuLoop3D& loop)
{
    namespace codec = fu_3d_codec_detail;
    unsigned rank = 0;
    for (const auto count : loop.counts) if (count > 1) ++rank;
    if (rank == 3) return full;
    Packet packet{};
    initialize_short_icu_packet(packet, (full.words[0].lanes[0] >> 4) & 3U, rank);
    encode_short_icu_loop(packet, loop, rank);
    for (std::size_t b = 0; b < short_icu_tail_bits<Packet>(); ++b)
        codec::write_unsigned(packet, 24 + (rank ? 2 : 0) + 40 * rank + b, 1,
            codec::read_unsigned(full, 144 + b, 1), "MXM payload bit");
    return packet;
}

// MEM packet, 3 consecutive 96-bit words (260 payload bits):
//   P[143:  0] common loop
//   P[149:144] packed stream selector
//   P[162:150] bank-local base row
//   P[178:163] outer_group_size - 1
//   P[198:179] signed inner stride
//   P[218:199] signed middle stride
//   P[238:219] signed outer-within-group stride
//   P[258:239] signed outer-group stride
//   P[259]     reserved, must be zero
inline EncodedMemIcu3DPacket encode_mem_icu_3d_instruction(
    const MemIcuInstruction& instruction)
{
    namespace codec = fu_3d_codec_detail;
    ::ftlpu::detail::validate_mem_icu_instruction(instruction);
    EncodedMemIcu3DPacket packet{};
    codec::initialize_word_headers(packet,
        static_cast<std::uint8_t>(instruction.opcode));
    codec::encode_loop(packet, instruction.loop);
    codec::write_unsigned(packet, 144, 6, instruction.stream,
        "MEM ICU 3-D stream does not fit 6 bits");
    codec::write_unsigned(packet, 150, 13,
        instruction.address.base_address,
        "MEM ICU 3-D base address does not fit 13 bits");
    if (instruction.address.outer_group_size == 0)
        throw std::invalid_argument(
            "MEM ICU 3-D outer group size must be non-zero");
    codec::write_unsigned(packet, 163, 16,
        instruction.address.outer_group_size - 1,
        "MEM ICU 3-D outer group size does not fit 16 bits");
    codec::write_signed(packet, 179, 20,
        instruction.address.inner_stride,
        "MEM ICU 3-D inner stride does not fit 20 bits");
    codec::write_signed(packet, 199, 20,
        instruction.address.middle_stride,
        "MEM ICU 3-D middle stride does not fit 20 bits");
    codec::write_signed(packet, 219, 20,
        instruction.address.outer_inner_stride,
        "MEM ICU 3-D outer-within-group stride does not fit 20 bits");
    codec::write_signed(packet, 239, 20,
        instruction.address.outer_group_stride,
        "MEM ICU 3-D outer-group stride does not fit 20 bits");
    return packet;
}

inline MemIcuInstruction decode_mem_icu_3d_instruction(
    const EncodedMemIcu3DPacket& packet)
{
    if (short_icu_rank(packet.words[0]) >= 0) {
        namespace codec = fu_3d_codec_detail;
        const auto rank = short_icu_rank(packet.words[0]);
        MemIcuInstruction instruction{};
        instruction.loop = decode_short_icu_loop(packet);
        const auto op = (packet.words[0].lanes[0] >> 4) & 3U;
        if (op > 2) throw std::logic_error("invalid short MEM operation");
        instruction.opcode = static_cast<MemIcuOpcode>(op);
        const auto tail = 24 + 40 * rank;
        instruction.stream = codec::read_unsigned(packet, tail, 6);
        instruction.address.base_address = codec::read_unsigned(packet, tail + 6, 13);
        if (rank > 0) instruction.address.inner_stride = codec::read_signed(packet, tail + 19, 20);
        if (rank > 1) instruction.address.middle_stride = codec::read_signed(packet, tail + 39, 20);
        ::ftlpu::detail::validate_mem_icu_instruction(instruction);
        return instruction;
    }
    namespace codec = fu_3d_codec_detail;
    const auto local_operation = codec::decode_word0_local_operation(packet,
        "encoded MEM ICU 3-D packet has an invalid word-0 header");
    if (local_operation > static_cast<std::uint8_t>(
            MemIcuOpcode::WriteTap3D))
        throw std::logic_error(
            "encoded MEM ICU 3-D packet has an invalid local operation");
    codec::validate_word_headers(packet, local_operation,
        "encoded MEM ICU 3-D packet has an invalid word header");
    codec::require_reserved_zero(packet, 259,
        "encoded MEM ICU 3-D packet has non-zero reserved bits");
    auto instruction = MemIcuInstruction {};
    instruction.loop = codec::decode_loop(packet);
    instruction.opcode = static_cast<MemIcuOpcode>(local_operation);
    instruction.stream = static_cast<std::size_t>(
        codec::read_unsigned(packet, 144, 6));
    instruction.address.base_address = static_cast<std::size_t>(
        codec::read_unsigned(packet, 150, 13));
    instruction.address.outer_group_size = static_cast<std::size_t>(
        codec::read_unsigned(packet, 163, 16) + 1);
    instruction.address.inner_stride =
        codec::read_signed(packet, 179, 20);
    instruction.address.middle_stride =
        codec::read_signed(packet, 199, 20);
    instruction.address.outer_inner_stride =
        codec::read_signed(packet, 219, 20);
    instruction.address.outer_group_stride =
        codec::read_signed(packet, 239, 20);
    ::ftlpu::detail::validate_mem_icu_instruction(instruction);
    return instruction;
}

// MEM WRITE_READ_2D packet, 3 consecutive 96-bit words. Word 0 carries
// Extended subtype 7 and all words carry FU-local operation 3. The 260-bit
// payload has 247 used bits; P[259:247] must be zero.
inline EncodedMemIcuWriteRead2DPacket
encode_mem_icu_write_read_2d_instruction(
    const MemIcuWriteRead2DInstruction& instruction)
{
    namespace codec = fu_3d_codec_detail;
    ::ftlpu::detail::validate_mem_icu_write_read_2d_instruction(
        instruction);
    EncodedMemIcuWriteRead2DPacket packet{};
    codec::initialize_word_headers(packet, 3,
        codec::kMemWriteRead2DExtendedSubtype);
    codec::write_unsigned(packet, 0, 24, instruction.start_wait,
        "MEM WRITE_READ_2D start wait exceeds 24 bits");
    codec::write_unsigned(packet, 24, 16, instruction.counts[0] - 1,
        "MEM WRITE_READ_2D inner count exceeds 16 bits");
    codec::write_unsigned(packet, 40, 16, instruction.counts[1] - 1,
        "MEM WRITE_READ_2D outer count exceeds 16 bits");
    codec::write_unsigned(packet, 56, 24,
        instruction.write_cycle_strides[0],
        "MEM WRITE_READ_2D inner write stride exceeds 24 bits");
    codec::write_unsigned(packet, 80, 24,
        instruction.write_cycle_strides[1],
        "MEM WRITE_READ_2D outer write stride exceeds 24 bits");
    codec::write_unsigned(packet, 104, 24,
        instruction.read_cycle_strides[0],
        "MEM WRITE_READ_2D inner read stride exceeds 24 bits");
    codec::write_unsigned(packet, 128, 24,
        instruction.read_cycle_strides[1],
        "MEM WRITE_READ_2D outer read stride exceeds 24 bits");
    codec::write_unsigned(packet, 152, 24,
        instruction.read_start_offset,
        "MEM WRITE_READ_2D read offset exceeds 24 bits");
    codec::write_unsigned(packet, 176, 13, instruction.base_address,
        "MEM WRITE_READ_2D bank address exceeds 13 bits");
    codec::write_signed(packet, 189, 20, instruction.address_strides[0],
        "MEM WRITE_READ_2D inner address stride exceeds 20 bits");
    codec::write_signed(packet, 209, 20, instruction.address_strides[1],
        "MEM WRITE_READ_2D outer address stride exceeds 20 bits");
    codec::write_unsigned(packet, 229, 6, instruction.write_stream,
        "MEM WRITE_READ_2D write stream exceeds 6 bits");
    codec::write_unsigned(packet, 235, 6, instruction.read_stream_base,
        "MEM WRITE_READ_2D read stream exceeds 6 bits");
    codec::write_signed(packet, 241, 6,
        instruction.read_stream_outer_stride,
        "MEM WRITE_READ_2D read stream stride exceeds 6 bits");
    return packet;
}

inline MemIcuWriteRead2DInstruction
decode_mem_icu_write_read_2d_instruction(
    const EncodedMemIcuWriteRead2DPacket& packet)
{
    namespace codec = fu_3d_codec_detail;
    if (codec::decode_word0_local_operation(packet,
            "encoded MEM WRITE_READ_2D has an invalid word-0 header",
            codec::kMemWriteRead2DExtendedSubtype) != 3)
        throw std::logic_error(
            "encoded MEM WRITE_READ_2D has an invalid local operation");
    codec::validate_word_headers(packet, 3,
        "encoded MEM WRITE_READ_2D has an invalid continuation header",
        codec::kMemWriteRead2DExtendedSubtype);
    codec::require_reserved_zero(packet, 247,
        "encoded MEM WRITE_READ_2D has non-zero reserved bits");
    MemIcuWriteRead2DInstruction instruction{};
    instruction.start_wait = codec::read_unsigned(packet, 0, 24);
    instruction.counts = {
        codec::read_unsigned(packet, 24, 16) + 1,
        codec::read_unsigned(packet, 40, 16) + 1};
    instruction.write_cycle_strides = {
        codec::read_unsigned(packet, 56, 24),
        codec::read_unsigned(packet, 80, 24)};
    instruction.read_cycle_strides = {
        codec::read_unsigned(packet, 104, 24),
        codec::read_unsigned(packet, 128, 24)};
    instruction.read_start_offset = codec::read_unsigned(packet, 152, 24);
    instruction.base_address = codec::read_unsigned(packet, 176, 13);
    instruction.address_strides = {
        codec::read_signed(packet, 189, 20),
        codec::read_signed(packet, 209, 20)};
    instruction.write_stream = codec::read_unsigned(packet, 229, 6);
    instruction.read_stream_base = codec::read_unsigned(packet, 235, 6);
    instruction.read_stream_outer_stride = codec::read_signed(packet, 241, 6);
    ::ftlpu::detail::validate_mem_icu_write_read_2d_instruction(
        instruction);
    return instruction;
}

// MXM load-ICU packet, 2 consecutive 128-bit words (236 payload bits).
// Local operation 0 is LOAD_3D:
//   P[143:  0] common loop
//   P[144]     weight-buffer base
//   P[146:145] buffer parity mode
//   P[148:147] weight-column base
//   P[164:149] signed column stride[0]
//   P[180:165] signed column stride[1]
//   P[196:181] signed column stride[2]
//   P[201:197] east weight input stream base
//   P[202]     input mode: 0=INT8 dequant, 1=direct 16-bit
//   P[235:203] reserved, must be zero
// Local operation 1 is DECODE_LOAD_ACTIVATION_3D:
//   P[143:  0] common loop
//   P[144]     activation-buffer base
//   P[146:145] buffer parity mode
//   P[151:147] east activation stream base
//   P[152]     data format
//   P[153]     decode layout
//   P[235:154] reserved, must be zero
inline EncodedMxmLoadIcu3DPacket encode_mxm_load_icu_3d_instruction(
    const MxmLoadIcuInstruction& instruction)
{
    namespace codec = fu_3d_codec_detail;
    ::ftlpu::detail::validate_mxm_load_icu_instruction(instruction);
    codec::validate_buffer_mode(instruction.weight_buffer_mode);
    EncodedMxmLoadIcu3DPacket packet{};
    codec::initialize_word_headers(packet,
        static_cast<std::uint8_t>(instruction.opcode));
    codec::encode_loop(packet, instruction.loop);
    if (instruction.opcode
        == MxmLoadIcuOpcode::DecodeLoadActivation3D) {
        codec::validate_buffer_mode(instruction.weight_buffer_mode);
        codec::write_unsigned(packet, 144, 1,
            instruction.weight_buffer_base,
            "MXM DECODE_LOAD_ACTIVATION_3D buffer base does not fit 1 bit");
        codec::write_unsigned(packet, 145, 2,
            static_cast<std::uint8_t>(instruction.weight_buffer_mode),
            "MXM DECODE_LOAD_ACTIVATION_3D buffer mode does not fit 2 bits");
        codec::write_unsigned(packet, 147, 5,
            instruction.weight_stream_base,
            "MXM DECODE_LOAD_ACTIVATION_3D stream base does not fit 5 bits");
        codec::write_unsigned(packet, 152, 1,
            static_cast<std::uint8_t>(instruction.data_format),
            "MXM DECODE_LOAD_ACTIVATION_3D data format does not fit 1 bit");
        codec::write_unsigned(packet, 153, 1,
            static_cast<std::uint8_t>(instruction.decode_layout),
            "MXM DECODE_LOAD_ACTIVATION_3D layout does not fit 1 bit");
        return packet;
    }
    codec::validate_weight_input_mode(instruction.weight_input_mode);
    codec::write_unsigned(packet, 144, 1,
        instruction.weight_buffer_base,
        "MXM LOAD_3D weight-buffer base does not fit 1 bit");
    codec::write_unsigned(packet, 145, 2,
        static_cast<std::uint8_t>(instruction.weight_buffer_mode),
        "MXM LOAD_3D buffer mode does not fit 2 bits");
    codec::write_unsigned(packet, 147, MxmIcuEncodingLayout::kColumnBits,
        instruction.weight_column_base /
            (instruction.opcode == MxmLoadIcuOpcode::LoadColumn3D ? 8 : 1),
        "MXM LOAD_3D weight-column base does not fit 2 bits");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
        codec::write_signed(packet, MxmIcuEncodingLayout::kLoadStrides + 16 * dimension, 16,
            instruction.weight_column_strides[dimension],
            "MXM LOAD_3D column stride does not fit 16 bits");
    codec::write_unsigned(packet, MxmIcuEncodingLayout::kLoadStream, 5,
        instruction.weight_stream_base,
        "MXM LOAD_3D weight stream base does not fit 5 bits");
    codec::write_unsigned(packet, MxmIcuEncodingLayout::kLoadInputMode, 1,
        static_cast<std::uint8_t>(instruction.weight_input_mode),
        "MXM LOAD_3D input mode does not fit 1 bit");
    if (instruction.opcode == MxmLoadIcuOpcode::LoadColumn3D)
        codec::write_unsigned(packet, MxmIcuEncodingLayout::kLoadEnd, 3,
            instruction.weight_column_base % 8, "MXM inner column overflow");
    return packet;
}

inline MxmLoadIcuInstruction decode_mxm_load_icu_3d_instruction(
    const EncodedMxmLoadIcu3DPacket& packet)
{
    if (short_icu_rank(packet.words[0]) >= 0)
        return decode_mxm_load_icu_3d_instruction(expand_short_mxm_packet(packet));
    namespace codec = fu_3d_codec_detail;
    const auto local_operation = codec::decode_word0_local_operation(packet,
        "encoded MXM LOAD_3D packet has an invalid word-0 header");
    if (local_operation > static_cast<std::uint8_t>(
            MxmLoadIcuOpcode::LoadColumn3D))
        throw std::logic_error(
            "encoded MXM LOAD_3D packet has an invalid local operation");
    codec::validate_word_headers(packet, local_operation,
        "encoded MXM LOAD_3D packet has an invalid word header");
    const auto loop = codec::decode_loop(packet);
    if (local_operation == static_cast<std::uint8_t>(
            MxmLoadIcuOpcode::DecodeLoadActivation3D)) {
        codec::require_reserved_zero(packet, 154,
            "encoded MXM DECODE_LOAD_ACTIVATION_3D packet has non-zero reserved bits");
        auto instruction = MxmLoadIcuInstruction::DecodeLoadActivation3D(
            loop,
            static_cast<std::size_t>(
                codec::read_unsigned(packet, 144, 1)),
            static_cast<MxmIcuBufferMode>(
                codec::read_unsigned(packet, 145, 2)),
            static_cast<std::size_t>(
                codec::read_unsigned(packet, 147, 5)),
            static_cast<MxmDataFormat>(
                codec::read_unsigned(packet, 152, 1)),
            static_cast<MxmDecodeLayout>(
                codec::read_unsigned(packet, 153, 1)));
        codec::validate_buffer_mode(instruction.weight_buffer_mode);
        ::ftlpu::detail::validate_mxm_load_icu_instruction(instruction);
        return instruction;
    }
    const bool columnLoad = local_operation == static_cast<std::uint8_t>(
        MxmLoadIcuOpcode::LoadColumn3D);
    codec::require_reserved_zero(packet, MxmIcuEncodingLayout::kLoadEnd + (columnLoad ? 3 : 0),
        "encoded MXM LOAD_3D packet has non-zero reserved bits");
    auto instruction = MxmLoadIcuInstruction {};
    instruction.opcode = static_cast<MxmLoadIcuOpcode>(local_operation);
    instruction.loop = loop;
    instruction.weight_buffer_base = static_cast<std::size_t>(
        codec::read_unsigned(packet, 144, 1));
    instruction.weight_buffer_mode = static_cast<MxmIcuBufferMode>(
        codec::read_unsigned(packet, 145, 2));
    instruction.weight_column_base = static_cast<std::size_t>(
        codec::read_unsigned(packet, 147, MxmIcuEncodingLayout::kColumnBits));
    if (columnLoad)
        instruction.weight_column_base = instruction.weight_column_base * 8
            + codec::read_unsigned(packet, MxmIcuEncodingLayout::kLoadEnd, 3);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
        instruction.weight_column_strides[dimension] =
            codec::read_signed(packet, MxmIcuEncodingLayout::kLoadStrides + 16 * dimension, 16);
    instruction.weight_stream_base = static_cast<std::size_t>(
        codec::read_unsigned(packet, MxmIcuEncodingLayout::kLoadStream, 5));
    instruction.weight_input_mode = static_cast<MxmWeightInputMode>(
        codec::read_unsigned(packet, MxmIcuEncodingLayout::kLoadInputMode, 1));
    codec::validate_buffer_mode(instruction.weight_buffer_mode);
    codec::validate_weight_input_mode(instruction.weight_input_mode);
    ::ftlpu::detail::validate_mxm_load_icu_instruction(instruction);
    return instruction;
}

// MXM DEQUANT_3D packet, 2 consecutive 128-bit words (236 payload bits):
//   P[143:  0] common loop
//   P[159:144] BF16 scale bits
//   P[235:160] reserved, must be zero
inline EncodedMxmDequantIcu3DPacket
encode_mxm_dequant_icu_3d_instruction(
    const MxmDequantIcuInstruction& instruction)
{
    namespace codec = fu_3d_codec_detail;
    ::ftlpu::detail::validate_mxm_dequant_icu_instruction(instruction);
    EncodedMxmDequantIcu3DPacket packet{};
    codec::initialize_word_headers(packet, codec::kMxmLocalOperation);
    codec::encode_loop(packet, instruction.loop);
    codec::write_unsigned(packet, 144, 16,
        instruction.instruction.scale_bf16,
        "MXM DEQUANT_3D scale does not fit 16 bits");
    return packet;
}

inline MxmDequantIcuInstruction decode_mxm_dequant_icu_3d_instruction(
    const EncodedMxmDequantIcu3DPacket& packet)
{
    if (short_icu_rank(packet.words[0]) >= 0)
        return decode_mxm_dequant_icu_3d_instruction(expand_short_mxm_packet(packet));
    namespace codec = fu_3d_codec_detail;
    const auto local_operation = codec::decode_word0_local_operation(packet,
        "encoded MXM DEQUANT_3D packet has an invalid word-0 header");
    if (local_operation != codec::kMxmLocalOperation)
        throw std::logic_error(
            "encoded MXM DEQUANT_3D packet has an invalid local operation");
    codec::validate_word_headers(packet, local_operation,
        "encoded MXM DEQUANT_3D packet has an invalid word header");
    codec::require_reserved_zero(packet, 160,
        "encoded MXM DEQUANT_3D packet has non-zero reserved bits");
    auto instruction = MxmDequantIcuInstruction {};
    instruction.loop = codec::decode_loop(packet);
    instruction.instruction = MxmDequantInstruction::ScaleBits(
        static_cast<std::uint16_t>(
            codec::read_unsigned(packet, 144, 16)));
    ::ftlpu::detail::validate_mxm_dequant_icu_instruction(instruction);
    return instruction;
}

// MXM compute-ICU packet. The diagram below is the compatible 32-lane layout,
// two 128-bit words. MxmIcuEncodingLayout derives all widened fields; 64 lanes
// use three words to retain 14-bit addresses and 15-bit signed strides.
// FU-local operation 0 is COMPUTE_3D:
//   P[143:  0] common loop
//   P[144]     weight-buffer base
//   P[146:145] buffer parity mode
//   P[151:147] east activation stream base
//   P[156:152] west result stream base
//   P[169:157] accumulator base row
//   P[183:170] signed accumulator stride[0]
//   P[197:184] signed accumulator stride[1]
//   P[211:198] signed accumulator stride[2]
//   P[224:212] accumulator row stride
//   P[225]     weight/activation data format
//   P[228:226] regular mode {destination, clear, output format}
//   P[230:229] terminal dimension (3 means disabled)
//   P[233:231] terminal mode {destination, clear, output format}
//   P[235:234] reserved, must be zero
//
// FU-local operation 1 is ACCUMULATOR_READ_3D and deliberately reuses the
// result/address/mode fields decoded by the same physical compute ICU:
//   P[143:  0] common loop
//   P[151:144] reserved, must be zero
//   P[156:152] west result stream base
//   P[169:157] accumulator base row
//   P[211:170] signed accumulator stride[0..2], 14 bits each
//   P[225:212] reserved, must be zero
//   P[228:226] read mode {destination, clear, output format}
//   P[235:229] reserved, must be zero
//
// FU-local operation 2 is DECODE_STREAM_COMPUTE_3D:
//   P[143:  0] common loop
//   P[144]     activation-buffer base
//   P[146:145] buffer parity mode
//   P[151:147] west result stream base
//   P[164:152] accumulator base row
//   P[206:165] signed accumulator stride[0..2], 14 bits each
//   P[208:207] accumulator column
//   P[209]     accumulator destination
//   P[210]     accumulator clear
//   P[211]     data format
//   P[212]     decode layout
//   P[235:213] reserved, must be zero
inline EncodedMxmComputeIcu3DPacket
encode_mxm_compute_icu_3d_instruction(
    const MxmComputeIcuInstruction& instruction)
{
    namespace codec = fu_3d_codec_detail;
    ::ftlpu::detail::validate_mxm_compute_icu_instruction(instruction);
    EncodedMxmComputeIcu3DPacket packet{};
    codec::initialize_word_headers(packet,
        static_cast<std::uint8_t>(instruction.opcode));
    codec::encode_loop(packet, instruction.loop);
    if (instruction.opcode == MxmComputeIcuOpcode::AccumulatorRead3D) {
        codec::write_unsigned(packet, 152, 5,
            instruction.result_stream_base,
            "MXM ACCUMULATOR_READ_3D result stream base does not fit 5 bits");
        codec::write_unsigned(packet, 157, MxmIcuEncodingLayout::kAddressBits,
            instruction.accumulator_address_base,
            "MXM ACCUMULATOR_READ_3D accumulator base does not fit target address width");
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
            codec::write_signed(packet, MxmIcuEncodingLayout::kComputeStrides + MxmIcuEncodingLayout::kStrideBits * dimension, MxmIcuEncodingLayout::kStrideBits,
                instruction.accumulator_address_strides[dimension],
                "MXM ACCUMULATOR_READ_3D accumulator stride does not fit target signed-stride width");
        codec::write_unsigned(packet, MxmIcuEncodingLayout::kRegularMode, 3,
            codec::encode_compute_mode(instruction.regular_mode),
            "MXM ACCUMULATOR_READ_3D mode does not fit 3 bits");
        return packet;
    }
    if (instruction.opcode
        == MxmComputeIcuOpcode::DecodeStreamCompute3D) {
        codec::validate_buffer_mode(instruction.weight_buffer_mode);
        codec::write_unsigned(packet, 144, 1,
            instruction.weight_buffer_base,
            "MXM DECODE_STREAM_COMPUTE_3D buffer base does not fit 1 bit");
        codec::write_unsigned(packet, 145, 2,
            static_cast<std::uint8_t>(instruction.weight_buffer_mode),
            "MXM DECODE_STREAM_COMPUTE_3D buffer mode does not fit target field width");
        codec::write_unsigned(packet, 147, 5,
            instruction.result_stream_base,
            "MXM DECODE_STREAM_COMPUTE_3D result stream does not fit 5 bits");
        codec::write_unsigned(packet, 152, MxmIcuEncodingLayout::kAddressBits,
            instruction.accumulator_address_base,
            "MXM DECODE_STREAM_COMPUTE_3D accumulator base does not fit target address width");
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
            codec::write_signed(packet, MxmIcuEncodingLayout::kDecodeStrides + MxmIcuEncodingLayout::kStrideBits * dimension, MxmIcuEncodingLayout::kStrideBits,
                instruction.accumulator_address_strides[dimension],
                "MXM DECODE_STREAM_COMPUTE_3D accumulator stride does not fit target signed-stride width");
        codec::write_unsigned(packet, MxmIcuEncodingLayout::kDecodeColumn, MxmIcuEncodingLayout::kColumnBits,
            instruction.accumulator_column,
            "MXM DECODE_STREAM_COMPUTE_3D accumulator column does not fit target field width");
        codec::write_unsigned(packet, MxmIcuEncodingLayout::kDecodeDestination, 1,
            static_cast<std::uint8_t>(
                instruction.regular_mode.accumulator_destination),
            "MXM DECODE_STREAM_COMPUTE_3D destination does not fit 1 bit");
        codec::write_unsigned(packet, MxmIcuEncodingLayout::kDecodeClear, 1,
            instruction.regular_mode.accumulator_clear ? 1U : 0U,
            "MXM DECODE_STREAM_COMPUTE_3D clear flag does not fit 1 bit");
        codec::write_unsigned(packet, MxmIcuEncodingLayout::kDecodeFormat, 1,
            static_cast<std::uint8_t>(instruction.data_format),
            "MXM DECODE_STREAM_COMPUTE_3D data format does not fit 1 bit");
        codec::write_unsigned(packet, MxmIcuEncodingLayout::kDecodeLayout, 1,
            static_cast<std::uint8_t>(instruction.decode_layout),
            "MXM DECODE_STREAM_COMPUTE_3D layout does not fit 1 bit");
        return packet;
    }
    codec::validate_buffer_mode(instruction.weight_buffer_mode);
    codec::write_unsigned(packet, 144, 1,
        instruction.weight_buffer_base,
        "MXM COMPUTE_3D weight-buffer base does not fit 1 bit");
    codec::write_unsigned(packet, 145, 2,
        static_cast<std::uint8_t>(instruction.weight_buffer_mode),
        "MXM COMPUTE_3D buffer mode does not fit target field width");
    codec::write_unsigned(packet, 147, 5,
        instruction.activation_stream_base,
        "MXM COMPUTE_3D activation stream base does not fit 5 bits");
    codec::write_unsigned(packet, 152, 5,
        instruction.result_stream_base,
        "MXM COMPUTE_3D result stream base does not fit 5 bits");
    codec::write_unsigned(packet, 157, MxmIcuEncodingLayout::kAddressBits,
        instruction.accumulator_address_base,
        "MXM COMPUTE_3D accumulator base does not fit target address width");
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
        codec::write_signed(packet, MxmIcuEncodingLayout::kComputeStrides + MxmIcuEncodingLayout::kStrideBits * dimension, MxmIcuEncodingLayout::kStrideBits,
            instruction.accumulator_address_strides[dimension],
            "MXM COMPUTE_3D accumulator stride does not fit target signed-stride width");
    codec::write_unsigned(packet, MxmIcuEncodingLayout::kRowStride, MxmIcuEncodingLayout::kAddressBits,
        instruction.accumulator_row_stride,
        "MXM COMPUTE_3D accumulator row stride does not fit target address width");
    codec::write_unsigned(packet, MxmIcuEncodingLayout::kFormat, 1,
        static_cast<std::uint8_t>(instruction.data_format),
        "MXM COMPUTE_3D data format does not fit 1 bit");
    codec::write_unsigned(packet, MxmIcuEncodingLayout::kRegularMode, 3,
        codec::encode_compute_mode(instruction.regular_mode),
        "MXM COMPUTE_3D regular mode does not fit 3 bits");
    codec::write_unsigned(packet, MxmIcuEncodingLayout::kTerminalDimension, 2,
        instruction.terminal_dimension,
        "MXM COMPUTE_3D terminal dimension does not fit target field width");
    codec::write_unsigned(packet, MxmIcuEncodingLayout::kTerminalMode, 3,
        codec::encode_compute_mode(instruction.terminal_mode),
        "MXM COMPUTE_3D terminal mode does not fit 3 bits");
    return packet;
}

inline MxmComputeIcuInstruction decode_mxm_compute_icu_3d_instruction(
    const EncodedMxmComputeIcu3DPacket& packet)
{
    if (short_icu_rank(packet.words[0]) >= 0)
        return decode_mxm_compute_icu_3d_instruction(expand_short_mxm_packet(packet));
    namespace codec = fu_3d_codec_detail;
    const auto local_operation = codec::decode_word0_local_operation(packet,
        "encoded MXM compute ICU 3-D packet has an invalid word-0 header");
    if (local_operation > static_cast<std::uint8_t>(
            MxmComputeIcuOpcode::DecodeStreamCompute3D))
        throw std::logic_error(
            "encoded MXM compute ICU 3-D packet has an invalid local operation");
    codec::validate_word_headers(packet, local_operation,
        "encoded MXM compute ICU 3-D packet has an invalid word header");
    const auto loop = codec::decode_loop(packet);
    if (local_operation == static_cast<std::uint8_t>(
            MxmComputeIcuOpcode::AccumulatorRead3D)) {
        if (codec::read_unsigned(packet, 144, 8) != 0
            || codec::read_unsigned(packet, MxmIcuEncodingLayout::kRowStride,
                MxmIcuEncodingLayout::kRegularMode - MxmIcuEncodingLayout::kRowStride) != 0)
            throw std::logic_error(
                "encoded MXM ACCUMULATOR_READ_3D packet has non-zero reserved bits");
        codec::require_reserved_zero(packet, MxmIcuEncodingLayout::kTerminalDimension,
            "encoded MXM ACCUMULATOR_READ_3D packet has non-zero reserved bits");
        std::array<std::int64_t, 3> strides{};
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
            strides[dimension] =
                codec::read_signed(packet, MxmIcuEncodingLayout::kComputeStrides + MxmIcuEncodingLayout::kStrideBits * dimension, MxmIcuEncodingLayout::kStrideBits);
        const auto mode = codec::decode_compute_mode(
            codec::read_unsigned(packet, MxmIcuEncodingLayout::kRegularMode, 3));
        auto instruction = MxmComputeIcuInstruction::AccumulatorRead3D(
            loop,
            static_cast<std::size_t>(
                codec::read_unsigned(packet, 152, 5)),
            static_cast<std::size_t>(
                codec::read_unsigned(packet, 157, MxmIcuEncodingLayout::kAddressBits)),
            strides,
            mode.accumulator_clear,
            mode.accumulator_output_format,
            mode.accumulator_destination);
        ::ftlpu::detail::validate_mxm_compute_icu_instruction(instruction);
        return instruction;
    }
    if (local_operation == static_cast<std::uint8_t>(
            MxmComputeIcuOpcode::DecodeStreamCompute3D)) {
        codec::require_reserved_zero(packet, MxmIcuEncodingLayout::kDecodeEnd,
            "encoded MXM DECODE_STREAM_COMPUTE_3D packet has non-zero reserved bits");
        std::array<std::int64_t, 3> strides{};
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
            strides[dimension] =
                codec::read_signed(packet, MxmIcuEncodingLayout::kDecodeStrides + MxmIcuEncodingLayout::kStrideBits * dimension, MxmIcuEncodingLayout::kStrideBits);
        auto instruction =
            MxmComputeIcuInstruction::DecodeStreamCompute3D(
                loop,
                static_cast<std::size_t>(
                    codec::read_unsigned(packet, 144, 1)),
                static_cast<MxmIcuBufferMode>(
                    codec::read_unsigned(packet, 145, 2)),
                static_cast<std::size_t>(
                    codec::read_unsigned(packet, 147, 5)),
                static_cast<MxmDataFormat>(
                    codec::read_unsigned(packet, MxmIcuEncodingLayout::kDecodeFormat, 1)),
                static_cast<std::size_t>(
                    codec::read_unsigned(packet, 152, MxmIcuEncodingLayout::kAddressBits)),
                strides,
                static_cast<std::size_t>(
                    codec::read_unsigned(packet, MxmIcuEncodingLayout::kDecodeColumn, MxmIcuEncodingLayout::kColumnBits)),
                static_cast<MxmAccumulatorDestination>(
                    codec::read_unsigned(packet, MxmIcuEncodingLayout::kDecodeDestination, 1)),
                codec::read_unsigned(packet, MxmIcuEncodingLayout::kDecodeClear, 1) != 0,
                static_cast<MxmDecodeLayout>(
                    codec::read_unsigned(packet, MxmIcuEncodingLayout::kDecodeLayout, 1)));
        codec::validate_buffer_mode(instruction.weight_buffer_mode);
        ::ftlpu::detail::validate_mxm_compute_icu_instruction(instruction);
        return instruction;
    }
    codec::require_reserved_zero(packet, MxmIcuEncodingLayout::kComputeEnd,
        "encoded MXM COMPUTE_3D packet has non-zero reserved bits");
    auto instruction = MxmComputeIcuInstruction {};
    instruction.opcode = MxmComputeIcuOpcode::Compute3D;
    instruction.loop = loop;
    instruction.weight_buffer_base = static_cast<std::size_t>(
        codec::read_unsigned(packet, 144, 1));
    instruction.weight_buffer_mode = static_cast<MxmIcuBufferMode>(
        codec::read_unsigned(packet, 145, 2));
    instruction.activation_stream_base = static_cast<std::size_t>(
        codec::read_unsigned(packet, 147, 5));
    instruction.result_stream_base = static_cast<std::size_t>(
        codec::read_unsigned(packet, 152, 5));
    instruction.accumulator_address_base = static_cast<std::size_t>(
        codec::read_unsigned(packet, 157, MxmIcuEncodingLayout::kAddressBits));
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
        instruction.accumulator_address_strides[dimension] =
            codec::read_signed(packet, MxmIcuEncodingLayout::kComputeStrides + MxmIcuEncodingLayout::kStrideBits * dimension, MxmIcuEncodingLayout::kStrideBits);
    instruction.accumulator_row_stride = static_cast<std::size_t>(
        codec::read_unsigned(packet, MxmIcuEncodingLayout::kRowStride, MxmIcuEncodingLayout::kAddressBits));
    instruction.data_format = static_cast<MxmDataFormat>(
        codec::read_unsigned(packet, MxmIcuEncodingLayout::kFormat, 1));
    instruction.regular_mode = codec::decode_compute_mode(
        codec::read_unsigned(packet, MxmIcuEncodingLayout::kRegularMode, 3));
    instruction.terminal_dimension = static_cast<std::size_t>(
        codec::read_unsigned(packet, MxmIcuEncodingLayout::kTerminalDimension, 2));
    instruction.terminal_mode = codec::decode_compute_mode(
        codec::read_unsigned(packet, MxmIcuEncodingLayout::kTerminalMode, 3));
    codec::validate_buffer_mode(instruction.weight_buffer_mode);
    ::ftlpu::detail::validate_mxm_compute_icu_instruction(instruction);
    return instruction;
}


// Canonicalize closed-form domains only; never materialize FU issues.
inline MemIcuInstruction canonical_mem_icu_domain(MemIcuInstruction instruction)
{
    ::ftlpu::detail::validate_mem_icu_instruction(instruction);
    const auto a = instruction.address;
    std::int64_t outer = a.outer_group_stride;
    if (instruction.loop.counts[2] > 1 && a.outer_group_size > 1) {
        if (instruction.loop.counts[2] <= a.outer_group_size)
            outer = a.outer_inner_stride;
        else if (a.outer_group_stride == static_cast<std::int64_t>(a.outer_group_size) * a.outer_inner_stride)
            outer = a.outer_inner_stride;
        else return instruction; // Non-affine blocked outer stays 3D.
    }
    auto loop = instruction.loop;
    loop.counts = {1, 1, 1};
    loop.cycle_strides = {1, 1, 1};
    std::array<std::int64_t, 3> address{};
    const std::array<std::int64_t, 3> oldAddress{a.inner_stride, a.middle_stride, outer};
    unsigned rank = 0;
    for (unsigned d = 0; d < 3; ++d) {
        const auto count = instruction.loop.counts[d];
        if (count == 1) continue;
        if (rank > 0 && count <= 65536 / loop.counts[rank - 1]
            && instruction.loop.cycle_strides[d] == loop.counts[rank - 1] * loop.cycle_strides[rank - 1]
            && oldAddress[d] == static_cast<std::int64_t>(loop.counts[rank - 1]) * address[rank - 1]) {
            loop.counts[rank - 1] *= count;
        } else {
            loop.counts[rank] = count;
            loop.cycle_strides[rank] = instruction.loop.cycle_strides[d];
            address[rank++] = oldAddress[d];
        }
    }
    instruction.loop = loop;
    instruction.address = MemIcuAddress3D::Affine(a.base_address, address);
    return instruction;
}

inline EncodedMemIcu3DPacket encode_mem_icu_shortest_instruction(MemIcuInstruction instruction)
{
    instruction = canonical_mem_icu_domain(instruction);
    // Validate the complete domain and original field limits before selecting.
    const auto full = encode_mem_icu_3d_instruction(instruction);
    if (instruction.address.outer_group_size != 1) return full;
    unsigned rank = 0;
    while (rank < 3 && instruction.loop.counts[rank] > 1) ++rank;
    if (rank == 3) return full;
    EncodedMemIcu3DPacket packet{};
    initialize_short_icu_packet(packet, static_cast<unsigned>(instruction.opcode), rank);
    encode_short_icu_loop(packet, instruction.loop, rank);
    namespace codec = fu_3d_codec_detail;
    const auto tail = 24 + 40 * rank;
    codec::write_unsigned(packet, tail, 6, instruction.stream, "MEM stream overflow");
    codec::write_unsigned(packet, tail + 6, 13, instruction.address.base_address, "MEM address overflow");
    if (rank > 0) codec::write_signed(packet, tail + 19, 20, instruction.address.inner_stride, "MEM stride overflow");
    if (rank > 1) codec::write_signed(packet, tail + 39, 20, instruction.address.middle_stride, "MEM stride overflow");
    return packet;
}

// MXM row, K and N axes retain their identities via the short axis selector.
inline EncodedMxmLoadIcu3DPacket encode_mxm_load_icu_shortest_instruction(const MxmLoadIcuInstruction& instruction)
{
    // Preserve existing short-packet layouts. The column opcode uses three
    // additional tail bits and currently has only the full packet encoding.
    if (instruction.opcode == MxmLoadIcuOpcode::LoadColumn3D)
        return encode_mxm_load_icu_3d_instruction(instruction);
    return compact_mxm_packet(encode_mxm_load_icu_3d_instruction(instruction), instruction.loop);
}
inline EncodedMxmDequantIcu3DPacket encode_mxm_dequant_icu_shortest_instruction(const MxmDequantIcuInstruction& instruction)
{
    return compact_mxm_packet(encode_mxm_dequant_icu_3d_instruction(instruction), instruction.loop);
}
inline EncodedMxmComputeIcu3DPacket encode_mxm_compute_icu_shortest_instruction(const MxmComputeIcuInstruction& instruction)
{
    return compact_mxm_packet(encode_mxm_compute_icu_3d_instruction(instruction), instruction.loop);
}

// Relocations preserve the selected physical format and all following PCs.
// write_unsigned initializes zeroed buffers, so replacement must clear bits.
template <typename Packet>
void replace_icu_payload_field(Packet& packet, std::size_t offset, unsigned width,
    std::uint64_t value, const char* message)
{
    auto updated = packet;
    for (unsigned bit = 0; bit < width; ++bit) {
        const auto location = fu_3d_codec_detail::packet_bit_location<Packet>(offset + bit);
        updated.words[location.word].lanes[location.lane] &= ~(std::uint32_t{1} << location.bit);
    }
    fu_3d_codec_detail::write_unsigned(updated, offset, width, value, message);
    packet = updated;
}

inline EncodedMemIcu3DPacket patch_mem_icu_base_address(
    EncodedMemIcu3DPacket packet, std::size_t base)
{
    auto instruction = decode_mem_icu_3d_instruction(packet);
    instruction.address.base_address = base;
    const auto rank = short_icu_rank(packet.words[0]);
    if (rank < 0) return encode_mem_icu_3d_instruction(instruction);
    replace_icu_payload_field(packet, 24 + 40 * rank + 6, 13, base,
        "short MEM relocation address overflow");
    static_cast<void>(decode_mem_icu_3d_instruction(packet));
    return packet;
}

inline EncodedMxmDequantIcu3DPacket patch_mxm_dequant_icu_scale(
    EncodedMxmDequantIcu3DPacket packet, MxmDequantInstruction scale)
{
    auto instruction = decode_mxm_dequant_icu_3d_instruction(packet);
    instruction.instruction = scale;
    const auto rank = short_icu_rank(packet.words[0]);
    if (rank < 0) return encode_mxm_dequant_icu_3d_instruction(instruction);
    replace_icu_payload_field(packet, 24 + (rank ? 2 : 0) + 40 * rank, 16,
        encode_mxm_dequant_instruction(scale), "short MXM scale relocation overflow");
    static_cast<void>(decode_mxm_dequant_icu_3d_instruction(packet));
    return packet;
}

} // namespace isa

} // namespace ftlpu
