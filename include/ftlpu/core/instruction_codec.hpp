#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/sxm/instruction.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/vxm/compact_instruction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ftlpu {

namespace isa {

// FTLPU hardware ISA encoding for the modeled slice.
//
// MEM instruction word (32b normally, 48b for ReadWrite):
//   [2:0] opcode, [8:3] stream, [14:9] map stream,
//   [30:15] slice-local SRAM row address.
//   ReadWrite repurposes [14:9] as write stream and [46:31] as write address;
//   [47] preserves the write stream for passive forwarding. Plain Write uses
//   [31] for the same preserve behavior.
// MXM control 49b:
//   IW      [1:0] opcode, [2] weight buffer, [4:3] weight column,
//           [5] column mode, [8:6] inner column, [9] Direct16 input mode.
//   Compute [1:0] opcode, [2] weight buffer, [8:3] activation stream base,
//           [14:9] output stream base, [27:15] accumulator address,
//           [43:28] accumulator row stride, [44] destination,
//           [45] weight/activation data format, [46] compute mode,
//           [47] retain accumulator after stream output,
//           [48] accumulator stream format (0=FP32, 1=BF16).
//   AccRead [1:0] opcode, [14:9] output stream base,
//           [27:15] accumulator address, [28] clear.
//   Decode  [1:0] opcode, [2] activation buffer, [3] operation,
//           [9:4] activation/output stream base, [25] data format,
//           [28] layout (0=Linear1x16, 1=Native4x4).
// VXM compact instruction 96b:
//   control [3:0] opcode, [6:4] lhs, [9:7] rhs, [10] precision,
//           [12:11] output type, [13] output enable,
//           [16:14] accumulator controls, [18:17] chain depth,
//           [50:19] repeat count, [51] local scalar write.
//   immediate_bits carries the one shared FP32 immediate.
// ICU queue command 32b:
//   NOP    [1:0] opcode, [31:2] cycle count.
//   Repeat [1:0] opcode, [11:2] count, [19:12] interval,
//          [31:20] signed MEM address stride.
//   Loop   [1:0] opcode, [7:2] window size, [15:8] count,
//          [23:16] interval, [31:24] signed MEM address stride.
// SXM control 13x32b:
//   header, 16 source selectors, 16 destination selectors, 8 lane-map
//   selectors, and 32 cross-tile permute selectors. The fixed packet avoids
//   host-sized vector/index fields in the hardware-facing ISA.
using EncodedMemInstruction = std::uint64_t;
using EncodedMxmInstruction = std::uint64_t;
using EncodedMxmDequantInstruction = std::uint16_t;
using EncodedIcuCommand = std::uint32_t;

struct EncodedIcuRepeat2D {
    std::array<std::uint32_t, 3> words{};
};

using EncodedVxmInstruction = VxmCompactInstruction;

struct EncodedSxmInstruction {
    std::array<std::uint32_t, 13> words{};
};

enum class IcuCommandOpcode : std::uint8_t {
    Instruction = 0,
    Nop = 1,
    Repeat = 2,
    Loop = 3,
};

namespace detail {

inline std::uint32_t float_to_bits(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float bits_to_float(std::uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void require_unsigned_fit(std::uint64_t value, std::uint64_t max, const char* field)
{
    if (value > max) {
        throw std::out_of_range(field);
    }
}

inline void require_signed_fit(std::int64_t value, std::int64_t min, std::int64_t max, const char* field)
{
    if (value < min || value > max) {
        throw std::out_of_range(field);
    }
}

inline std::uint64_t low_bits(std::uint64_t word, unsigned shift, std::uint64_t mask)
{
    return (word >> shift) & mask;
}

inline void require_reserved_zero(std::uint64_t word, std::uint64_t used_mask, const char* instruction)
{
    if ((word & ~used_mask) != 0) {
        throw std::logic_error(instruction);
    }
}

template <std::size_t N>
inline void write_bits(
    std::array<std::uint32_t, N>& words,
    std::size_t offset,
    unsigned width,
    std::uint32_t value)
{
    for (unsigned bit = 0; bit < width; ++bit) {
        if (((value >> bit) & 1u) != 0) {
            const auto position = offset + bit;
            words[position / 32] |= 1u << (position % 32);
        }
    }
}

template <std::size_t N>
inline std::uint32_t read_bits(
    const std::array<std::uint32_t, N>& words,
    std::size_t offset,
    unsigned width)
{
    std::uint32_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const auto position = offset + bit;
        value |= ((words[position / 32] >> (position % 32)) & 1u) << bit;
    }
    return value;
}

} // namespace detail

inline EncodedMemInstruction encode_mem_instruction(const MemInstruction& instruction)
{
    constexpr std::uint64_t kOpcodeMask = 0x7;
    constexpr std::uint64_t kStreamMask = 0x3f;
    constexpr std::uint64_t kAddressMask = hw::kSramDepthWords - 1;

    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.opcode),
        kOpcodeMask,
        "MEM opcode does not fit encoded instruction");
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.stream),
        kStreamMask,
        "MEM stream does not fit encoded instruction");
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.address),
        kAddressMask,
        "MEM row address does not fit encoded instruction");
    if (instruction.preserve_stream
        && instruction.opcode != MemOpcode::Write
        && instruction.opcode != MemOpcode::ReadWrite)
        throw std::logic_error(
            "only MEM Write or ReadWrite can preserve its input stream");

    if (instruction.opcode == MemOpcode::ReadWrite) {
        if (instruction.address == instruction.write_address) {
            throw std::logic_error("MEM ReadWrite encodes identical read and write addresses");
        }
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.write_stream),
            kStreamMask,
            "MEM write stream does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.write_address),
            kAddressMask,
            "MEM write row address does not fit encoded instruction");
        return static_cast<std::uint64_t>(instruction.opcode)
            | (static_cast<std::uint64_t>(instruction.stream) << 3)
            | (static_cast<std::uint64_t>(instruction.write_stream) << 9)
            | (static_cast<std::uint64_t>(instruction.address) << 15)
            | (static_cast<std::uint64_t>(instruction.write_address) << 31)
            | (static_cast<std::uint64_t>(instruction.preserve_stream) << 47);
    }
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.map_stream),
        kStreamMask,
        "MEM map stream does not fit encoded instruction");
    return static_cast<std::uint64_t>(
        static_cast<std::uint64_t>(instruction.opcode)
        | (static_cast<std::uint64_t>(instruction.stream) << 3)
        | (static_cast<std::uint64_t>(instruction.map_stream) << 9)
        | (static_cast<std::uint64_t>(instruction.address) << 15)
        | (static_cast<std::uint64_t>(instruction.preserve_stream) << 31));
}

inline MemInstruction decode_mem_instruction(EncodedMemInstruction word)
{
    const auto opcode = static_cast<MemOpcode>(detail::low_bits(word, 0, 0x7));
    const auto stream = static_cast<std::size_t>(detail::low_bits(word, 3, 0x3f));
    const auto address = static_cast<std::size_t>(detail::low_bits(word, 15, 0xffff));
    if (opcode == MemOpcode::ReadWrite) {
        constexpr std::uint64_t kReadWriteMask = (std::uint64_t {1} << 48) - 1;
        detail::require_reserved_zero(
            word, kReadWriteMask, "encoded MEM ReadWrite instruction has non-zero reserved bits");
        const auto write_stream = static_cast<std::size_t>(detail::low_bits(word, 9, 0x3f));
        const auto write_address = static_cast<std::size_t>(detail::low_bits(word, 31, 0xffff));
        const bool preserve_stream = detail::low_bits(word, 47, 0x1) != 0;
        return preserve_stream
            ? MemInstruction::ReadWriteTap(
                  address, stream, write_address, write_stream)
            : MemInstruction::ReadWrite(
                  address, stream, write_address, write_stream);
    }
    constexpr std::uint64_t kUsedMask = 0xffffffffull;
    detail::require_reserved_zero(word, kUsedMask, "encoded MEM instruction has non-zero reserved bits");
    const auto map_stream = static_cast<std::size_t>(detail::low_bits(word, 9, 0x3f));
    const bool preserve_stream = detail::low_bits(word, 31, 0x1) != 0;
    if (preserve_stream && opcode != MemOpcode::Write)
        throw std::logic_error(
            "only encoded MEM Write can preserve its input stream");
    switch (opcode) {
    case MemOpcode::Read:
        return MemInstruction::Read(address, stream);
    case MemOpcode::Write:
        return preserve_stream
            ? MemInstruction::WriteTap(address, stream)
            : MemInstruction::Write(address, stream);
    case MemOpcode::ReadWrite:
        break;
    case MemOpcode::Gather:
        return MemInstruction::Gather(stream, map_stream);
    case MemOpcode::Scatter:
        return MemInstruction::Scatter(stream, map_stream);
    }
    throw std::logic_error("unknown encoded MEM opcode");
}

inline EncodedMxmInstruction encode_mxm_instruction(const MxmControlInstruction& instruction)
{
    constexpr std::uint64_t kOpcodeMask = 0x3;
    constexpr std::uint64_t kWeightBufferMask = 0x1;
    constexpr std::uint64_t kWeightColumnMask = 0x3;
    constexpr std::uint64_t kWeightInnerColumnMask = 0x7;
    constexpr std::uint64_t kStreamBaseMask = 0x3f;
    constexpr std::uint64_t kActivationStreamMask = 0x3f;

    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.opcode),
        kOpcodeMask,
        "MXM opcode does not fit encoded instruction");
    const auto opcode = static_cast<std::uint64_t>(instruction.opcode);
    switch (instruction.opcode) {
    case MxmControlOpcode::IW:
        MxmControlInstruction::check_weight_load(
            instruction.weight_load_mode,
            instruction.weight_inner_column);
        MxmControlInstruction::check_weight_input_mode(
            instruction.weight_input_mode);
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_column),
            kWeightColumnMask,
            "MXM weight column does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_inner_column),
            kWeightInnerColumnMask,
            "MXM weight inner column does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint64_t>(instruction.weight_buffer) << 2)
            | (static_cast<std::uint64_t>(instruction.weight_column) << 3)
            | (static_cast<std::uint64_t>(
                   instruction.weight_load_mode == MxmWeightLoadMode::Column)
               << 5)
            | (static_cast<std::uint64_t>(instruction.weight_inner_column) << 6)
            | (static_cast<std::uint64_t>(instruction.weight_input_mode) << 9);
    case MxmControlOpcode::Compute:
        MxmControlInstruction::check_data_format(instruction.data_format);
        MxmControlInstruction::check_compute_mode(instruction.compute_mode);
        MxmControlInstruction::check_activation_stream_base(
            instruction.activation_stream_base,
            instruction.compute_mode);
        MxmControlInstruction::check_compute_destination(
            instruction.compute_mode,
            instruction.accumulator_destination);
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.activation_stream_base),
            kActivationStreamMask,
            "MXM activation stream base does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.stream_base),
            kStreamBaseMask,
            "MXM output stream base does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.data_format),
            0x1,
            "MXM data format does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.compute_mode),
            0x1,
            "MXM compute mode does not fit encoded instruction");
        MxmControlInstruction::check_accumulator_address(
            instruction.accumulator_address,
            instruction.compute_mode);
        detail::require_unsigned_fit(
            instruction.accumulator_address,
            hw::kMxmAccumulatorRows - 1,
            "MXM accumulator address does not fit encoded instruction");
        detail::require_unsigned_fit(
            instruction.accumulator_row_stride,
            0xffff,
            "MXM accumulator row stride does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint64_t>(instruction.weight_buffer) << 2)
            | (static_cast<std::uint64_t>(instruction.activation_stream_base) << 3)
            | (static_cast<std::uint64_t>(instruction.stream_base) << 9)
            | (static_cast<std::uint64_t>(instruction.accumulator_address) << 15)
            | (static_cast<std::uint64_t>(instruction.accumulator_row_stride) << 28)
            | (static_cast<std::uint64_t>(instruction.accumulator_destination) << 44)
            | (static_cast<std::uint64_t>(instruction.data_format) << 45)
            | (static_cast<std::uint64_t>(instruction.compute_mode) << 46)
            | (static_cast<std::uint64_t>(!instruction.accumulator_clear) << 47)
            | (static_cast<std::uint64_t>(
                   instruction.accumulator_output_format) << 48);
    case MxmControlOpcode::Decode:
        MxmControlInstruction::check_data_format(instruction.data_format);
        MxmControlInstruction::check_decode_layout(instruction.decode_layout);
        detail::require_unsigned_fit(
            instruction.weight_buffer,
            kWeightBufferMask,
            "MXM decode activation buffer does not fit encoded instruction");
        if (instruction.decode_operation
            == MxmDecodeOperation::LoadActivation) {
            MxmControlInstruction::check_decode_activation_stream_base(
                instruction.activation_stream_base,
                instruction.decode_layout);
            detail::require_unsigned_fit(
                instruction.activation_stream_base,
                kActivationStreamMask,
                "MXM decode activation stream does not fit encoded instruction");
            return opcode
                | (static_cast<std::uint64_t>(instruction.weight_buffer) << 2)
                | (static_cast<std::uint64_t>(
                       instruction.decode_operation) << 3)
                | (static_cast<std::uint64_t>(
                       instruction.activation_stream_base) << 4)
                | (static_cast<std::uint64_t>(
                       instruction.data_format) << 25)
                | (static_cast<std::uint64_t>(
                       instruction.decode_layout) << 28);
        }
        if (instruction.decode_operation
            != MxmDecodeOperation::StreamCompute) {
            throw std::invalid_argument("MXM decode operation is invalid");
        }
        if (instruction.accumulator_destination
            == MxmAccumulatorDestination::Stream) {
            MxmControlInstruction::check_decode_output_stream_base(
                instruction.stream_base);
        }
        MxmControlInstruction::check_accumulator_address(
            instruction.accumulator_address,
            MxmComputeMode::Vector);
        MxmControlInstruction::check_column(
            instruction.weight_column);
        detail::require_unsigned_fit(
            instruction.stream_base,
            kStreamBaseMask,
            "MXM decode output stream does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint64_t>(instruction.weight_buffer) << 2)
            | (static_cast<std::uint64_t>(
                   instruction.decode_operation) << 3)
            | (static_cast<std::uint64_t>(instruction.stream_base) << 4)
            | (static_cast<std::uint64_t>(
                   instruction.accumulator_address) << 10)
            | (static_cast<std::uint64_t>(
                   instruction.weight_column) << 23)
            | (static_cast<std::uint64_t>(
                   instruction.data_format) << 25)
            | (static_cast<std::uint64_t>(
                   instruction.accumulator_destination) << 26)
            | (static_cast<std::uint64_t>(
                   !instruction.accumulator_clear) << 27)
            | (static_cast<std::uint64_t>(
                   instruction.decode_layout) << 28);
    case MxmControlOpcode::AccumulatorRead:
        MxmControlInstruction::check_compute_mode(instruction.compute_mode);
        MxmControlInstruction::check_accumulator_read_stream_base(
            instruction.stream_base,
            instruction.compute_mode);
        MxmControlInstruction::check_accumulator_address(
            instruction.accumulator_address,
            instruction.compute_mode);
        detail::require_unsigned_fit(
            instruction.stream_base,
            kStreamBaseMask,
            "MXM accumulator read stream does not fit encoded instruction");
        detail::require_unsigned_fit(
            instruction.accumulator_address,
            hw::kMxmAccumulatorRows - 1,
            "MXM accumulator read address does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint64_t>(instruction.stream_base) << 9)
            | (static_cast<std::uint64_t>(instruction.accumulator_address) << 15)
            | (static_cast<std::uint64_t>(instruction.accumulator_clear) << 28)
            | (static_cast<std::uint64_t>(instruction.compute_mode) << 46);
    }
    throw std::logic_error("unknown MXM opcode");
}

inline MxmControlInstruction decode_mxm_instruction(EncodedMxmInstruction word)
{
    const auto opcode = static_cast<MxmControlOpcode>(word & 0x3u);
    const auto iw_weight_buffer = static_cast<std::size_t>((word >> 2) & 0x1u);
    const auto iw_weight_column = static_cast<std::size_t>((word >> 3) & 0x3u);
    const auto iw_column_mode = ((word >> 5) & 0x1u) != 0;
    const auto iw_inner_column = static_cast<std::size_t>((word >> 6) & 0x7u);
    const auto iw_input_mode =
        static_cast<MxmWeightInputMode>((word >> 9) & 0x1u);
    const auto compute_weight_buffer = static_cast<std::size_t>((word >> 2) & 0x1u);
    const auto compute_activation_stream_base = static_cast<std::size_t>((word >> 3) & 0x3fu);
    const auto stream_base = static_cast<std::size_t>((word >> 9) & 0x3fu);
    const auto compute_data_format =
        static_cast<MxmDataFormat>((word >> 45) & 0x1u);
    const auto compute_mode =
        static_cast<MxmComputeMode>((word >> 46) & 0x1u);

    switch (opcode) {
    case MxmControlOpcode::IW:
        detail::require_reserved_zero(word, 0x000003ffu, "encoded MXM IW instruction has non-zero reserved bits");
        return iw_column_mode
            ? MxmControlInstruction::IWColumn(
                  iw_weight_buffer,
                  iw_weight_column,
                  iw_inner_column,
                  iw_input_mode)
            : MxmControlInstruction::IW(
                  iw_weight_buffer,
                  iw_weight_column,
                  iw_input_mode);
    case MxmControlOpcode::Compute:
        detail::require_reserved_zero(
            word,
            (std::uint64_t {1} << 49) - 1,
            "encoded MXM Compute instruction has non-zero reserved bits");
        return MxmControlInstruction::Compute(
            compute_weight_buffer,
            compute_activation_stream_base,
            stream_base,
            static_cast<std::size_t>((word >> 15) & 0x1fffu),
            static_cast<std::size_t>((word >> 28) & 0xffffu),
            static_cast<MxmAccumulatorDestination>((word >> 44) & 0x1u),
            compute_data_format,
            compute_mode,
            ((word >> 47) & 0x1u) == 0,
            static_cast<MxmAccumulatorOutputFormat>(
                (word >> 48) & 0x1u));
    case MxmControlOpcode::AccumulatorRead:
        detail::require_reserved_zero(
            word,
            0x40001ffffe03ull,
            "encoded MXM AccumulatorRead instruction has non-zero reserved bits");
        return MxmControlInstruction::AccumulatorRead(
            static_cast<std::size_t>((word >> 15) & 0x1fffu),
            stream_base,
            ((word >> 28) & 0x1u) != 0,
            compute_mode);
    case MxmControlOpcode::Decode: {
        const auto activation_buffer =
            static_cast<std::size_t>((word >> 2) & 0x1u);
        const auto operation =
            static_cast<MxmDecodeOperation>((word >> 3) & 0x1u);
        const auto decode_stream_base =
            static_cast<std::size_t>((word >> 4) & 0x3fu);
        const auto data_format =
            static_cast<MxmDataFormat>((word >> 25) & 0x1u);
        const auto decode_layout =
            static_cast<MxmDecodeLayout>((word >> 28) & 0x1u);
        if (operation == MxmDecodeOperation::LoadActivation) {
            detail::require_reserved_zero(
                word,
                (std::uint64_t {1} << 28)
                    | (std::uint64_t {1} << 25) | 0x3ffu,
                "encoded MXM DecodeLoadActivation instruction has non-zero reserved bits");
            return MxmControlInstruction::DecodeLoadActivation(
                activation_buffer,
                decode_stream_base,
                data_format,
                decode_layout);
        }
        detail::require_reserved_zero(
            word,
            (std::uint64_t {1} << 29) - 1,
            "encoded MXM DecodeStreamCompute instruction has non-zero reserved bits");
        return MxmControlInstruction::DecodeStreamCompute(
            activation_buffer,
            decode_stream_base,
            data_format,
            static_cast<std::size_t>((word >> 10) & 0x1fffu),
            static_cast<std::size_t>((word >> 23) & 0x3u),
            static_cast<MxmAccumulatorDestination>(
                (word >> 26) & 0x1u),
            ((word >> 27) & 0x1u) == 0,
            decode_layout);
    }
    }
    throw std::logic_error("unknown encoded MXM opcode");
}

inline EncodedMxmDequantInstruction encode_mxm_dequant_instruction(
    MxmDequantInstruction instruction)
{
    return instruction.scale_bf16;
}

inline MxmDequantInstruction decode_mxm_dequant_instruction(
    EncodedMxmDequantInstruction word)
{
    return MxmDequantInstruction::ScaleBits(word);
}

namespace detail {
inline void require_default_float(float value, const char* field)
{
    if (value != 1.0f) throw std::logic_error(field);
}

inline void require_zero_float(float value, const char* field)
{
    if (value != 0.0f) throw std::logic_error(field);
}

} // namespace detail

inline EncodedVxmInstruction encode_vxm_instruction(
    std::size_t queue,
    VxmChainDepth depth,
    const VxmLaneAluInstruction& instruction)
{
    return VxmCompactInstructionCodec::encode(queue, depth, instruction);
}

inline VxmDecodedInstruction decode_vxm_instruction(
    std::size_t queue,
    const EncodedVxmInstruction& encoded)
{
    return VxmCompactInstructionCodec::decode(queue, encoded);
}

inline EncodedSxmInstruction encode_sxm_instruction(const SxmInstruction& instruction)
{
    constexpr std::size_t kMaxStreams = 16;
    constexpr std::size_t kSrcOffset = 16;
    constexpr std::size_t kDstOffset = kSrcOffset + kMaxStreams * 6;
    constexpr std::size_t kLaneMapOffset = kDstOffset + kMaxStreams * 6;
    constexpr std::size_t kPermuteMapOffset =
        kLaneMapOffset + hw::kLanesPerTile * 4;

    detail::require_unsigned_fit(
        static_cast<std::size_t>(instruction.opcode), 0x3, "SXM opcode does not fit");
    detail::require_unsigned_fit(
        static_cast<std::size_t>(instruction.shift_source), 0x3, "SXM shift source does not fit");
    detail::require_unsigned_fit(
        instruction.shift_distance, 0x3, "SXM shift distance does not fit");
    detail::require_unsigned_fit(
        instruction.src_streams.size(), kMaxStreams, "SXM source stream count does not fit");
    detail::require_unsigned_fit(
        instruction.dst_streams.size(), kMaxStreams, "SXM destination stream count does not fit");

    EncodedSxmInstruction encoded{};
    detail::write_bits(encoded.words, 0, 2, static_cast<std::uint32_t>(instruction.opcode));
    detail::write_bits(encoded.words, 2, 2, static_cast<std::uint32_t>(instruction.shift_source));
    detail::write_bits(encoded.words, 4, 2, static_cast<std::uint32_t>(instruction.shift_distance));
    detail::write_bits(encoded.words, 6, 5, static_cast<std::uint32_t>(instruction.src_streams.size()));
    detail::write_bits(encoded.words, 11, 5, static_cast<std::uint32_t>(instruction.dst_streams.size()));

    for (std::size_t index = 0; index < instruction.src_streams.size(); ++index) {
        detail::require_unsigned_fit(
            instruction.src_streams[index].stream, 0x3f, "SXM source stream does not fit");
        detail::write_bits(
            encoded.words, kSrcOffset + index * 6, 6,
            static_cast<std::uint32_t>(instruction.src_streams[index].stream));
    }
    for (std::size_t index = 0; index < instruction.dst_streams.size(); ++index) {
        detail::require_unsigned_fit(
            instruction.dst_streams[index].stream, 0x3f, "SXM destination stream does not fit");
        detail::write_bits(
            encoded.words, kDstOffset + index * 6, 6,
            static_cast<std::uint32_t>(instruction.dst_streams[index].stream));
    }
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        const auto source = instruction.lane_map[lane] == SxmInstruction::kZeroFill
            ? hw::kLanesPerTile
            : instruction.lane_map[lane];
        detail::require_unsigned_fit(source, hw::kLanesPerTile, "SXM lane map does not fit");
        detail::write_bits(
            encoded.words, kLaneMapOffset + lane * 4, 4,
            static_cast<std::uint32_t>(source));
    }
    for (std::size_t lane = 0; lane < SxmInstruction::kTotalLanes; ++lane) {
        detail::require_unsigned_fit(
            instruction.permute_map[lane],
            SxmInstruction::kTotalLanes - 1,
            "SXM permute map does not fit");
        detail::write_bits(
            encoded.words, kPermuteMapOffset + lane * 5, 5,
            static_cast<std::uint32_t>(instruction.permute_map[lane]));
    }
    return encoded;
}

inline SxmInstruction decode_sxm_instruction(const EncodedSxmInstruction& encoded)
{
    constexpr std::size_t kMaxStreams = 16;
    constexpr std::size_t kSrcOffset = 16;
    constexpr std::size_t kDstOffset = kSrcOffset + kMaxStreams * 6;
    constexpr std::size_t kLaneMapOffset = kDstOffset + kMaxStreams * 6;
    constexpr std::size_t kPermuteMapOffset =
        kLaneMapOffset + hw::kLanesPerTile * 4;
    constexpr std::size_t kUsedBits =
        kPermuteMapOffset + SxmInstruction::kTotalLanes * 5;

    for (std::size_t bit = kUsedBits; bit < encoded.words.size() * 32; ++bit) {
        if (detail::read_bits(encoded.words, bit, 1) != 0) {
            throw std::logic_error("encoded SXM instruction has non-zero reserved bits");
        }
    }

    SxmInstruction instruction{};
    instruction.opcode = static_cast<SxmOpcode>(detail::read_bits(encoded.words, 0, 2));
    instruction.shift_source =
        static_cast<SxmShiftSource>(detail::read_bits(encoded.words, 2, 2));
    instruction.shift_distance = detail::read_bits(encoded.words, 4, 2);
    const auto src_count = detail::read_bits(encoded.words, 6, 5);
    const auto dst_count = detail::read_bits(encoded.words, 11, 5);
    if (src_count > kMaxStreams || dst_count > kMaxStreams) {
        throw std::logic_error("encoded SXM stream count is invalid");
    }
    for (std::size_t index = 0; index < src_count; ++index) {
        instruction.src_streams.push_back(
            SxmStreamId {detail::read_bits(encoded.words, kSrcOffset + index * 6, 6)});
    }
    for (std::size_t index = 0; index < dst_count; ++index) {
        instruction.dst_streams.push_back(
            SxmStreamId {detail::read_bits(encoded.words, kDstOffset + index * 6, 6)});
    }
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        const auto source = detail::read_bits(encoded.words, kLaneMapOffset + lane * 4, 4);
        if (source > hw::kLanesPerTile) {
            throw std::logic_error("encoded SXM lane map is invalid");
        }
        instruction.lane_map[lane] =
            source == hw::kLanesPerTile ? SxmInstruction::kZeroFill : source;
    }
    for (std::size_t lane = 0; lane < SxmInstruction::kTotalLanes; ++lane) {
        const auto source =
            detail::read_bits(encoded.words, kPermuteMapOffset + lane * 5, 5);
        if (source >= SxmInstruction::kTotalLanes) {
            throw std::logic_error("encoded SXM permute map is invalid");
        }
        instruction.permute_map[lane] = source;
    }
    return instruction;
}

inline EncodedIcuCommand encode_icu_nop(std::size_t cycles)
{
    constexpr std::uint64_t kCountMask = 0x3fffffffull;
    detail::require_unsigned_fit(cycles, kCountMask, "ICU NOP cycle count does not fit encoded command");
    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(IcuCommandOpcode::Nop)
        | (static_cast<std::uint64_t>(cycles) << 2));
}

inline EncodedIcuCommand encode_icu_repeat(const InstructionControlUnit::Repeat& repeat)
{
    constexpr std::uint64_t kCountMask = 0x3ffull;
    constexpr std::uint64_t kIntervalMask = 0xffull;
    constexpr auto kStrideMin = static_cast<std::int64_t>(-2048);
    constexpr auto kStrideMax = static_cast<std::int64_t>(2047);
    detail::require_unsigned_fit(repeat.count, kCountMask, "ICU Repeat count does not fit encoded command");
    detail::require_unsigned_fit(repeat.interval, kIntervalMask, "ICU Repeat interval does not fit encoded command");
    detail::require_signed_fit(
        repeat.address_stride,
        kStrideMin,
        kStrideMax,
        "ICU Repeat address stride does not fit encoded command");

    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(IcuCommandOpcode::Repeat)
        | (static_cast<std::uint64_t>(repeat.count) << 2)
        | (static_cast<std::uint64_t>(repeat.interval) << 12)
        | (static_cast<std::uint64_t>(static_cast<std::uint16_t>(repeat.address_stride) & 0x0fffu) << 20));
}

inline EncodedIcuCommand encode_icu_loop(const IcuLoop& loop)
{
    detail::require_unsigned_fit(loop.window_size, 0x3full,
        "ICU Loop window size does not fit encoded command");
    detail::require_unsigned_fit(loop.count, 0xffull,
        "ICU Loop count does not fit encoded command");
    detail::require_unsigned_fit(loop.interval, 0xffull,
        "ICU Loop interval does not fit encoded command");
    detail::require_signed_fit(loop.address_stride, -128, 127,
        "ICU Loop address stride does not fit encoded command");
    if (loop.window_size == 0 || loop.count == 0
        || loop.interval < loop.window_size) {
        throw std::invalid_argument(
            "ICU Loop requires count > 0 and interval >= window size > 0");
    }
    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(IcuCommandOpcode::Loop)
        | (static_cast<std::uint64_t>(loop.window_size) << 2)
        | (static_cast<std::uint64_t>(loop.count) << 8)
        | (static_cast<std::uint64_t>(loop.interval) << 16)
        | (static_cast<std::uint64_t>(
               static_cast<std::uint8_t>(loop.address_stride))
            << 24));
}

inline EncodedIcuRepeat2D encode_icu_repeat_2d(const IcuRepeat2D& repeat)
{
    detail::require_unsigned_fit(repeat.inner_count, 0x3ff,
        "ICU Repeat2D inner count does not fit encoded command");
    detail::require_unsigned_fit(repeat.outer_count, 0x3ff,
        "ICU Repeat2D outer count does not fit encoded command");
    detail::require_unsigned_fit(repeat.inner_interval, 0xffff,
        "ICU Repeat2D inner interval does not fit encoded command");
    detail::require_unsigned_fit(repeat.outer_interval, 0xffff,
        "ICU Repeat2D outer interval does not fit encoded command");
    detail::require_signed_fit(repeat.inner_stride, -32768, 32767,
        "ICU Repeat2D inner stride does not fit encoded command");
    detail::require_signed_fit(repeat.outer_stride, -32768, 32767,
        "ICU Repeat2D outer stride does not fit encoded command");
    if (repeat.inner_count == 0 || repeat.outer_count == 0
        || repeat.inner_interval == 0 || repeat.outer_interval == 0
        || repeat.inner_count * repeat.outer_count <= 1
        || (repeat.outer_count > 1
            && repeat.outer_interval
                <= (repeat.inner_count - 1) * repeat.inner_interval)) {
        throw std::invalid_argument("ICU Repeat2D has an invalid iteration space");
    }

    EncodedIcuRepeat2D encoded;
    auto write = [&](std::size_t offset, std::size_t width,
                     std::uint64_t value) {
        for (std::size_t bit = 0; bit < width; ++bit) {
            if ((value & (std::uint64_t {1} << bit)) != 0)
                encoded.words[(offset + bit) / 32]
                    |= std::uint32_t {1} << ((offset + bit) % 32);
        }
    };
    write(0, 2, static_cast<std::uint64_t>(IcuCommandOpcode::Loop));
    write(2, 10, repeat.inner_count);
    write(12, 10, repeat.outer_count);
    write(22, 16, repeat.inner_interval);
    write(38, 16, repeat.outer_interval);
    write(54, 16, static_cast<std::uint16_t>(repeat.inner_stride));
    write(70, 16, static_cast<std::uint16_t>(repeat.outer_stride));
    write(86, 2, static_cast<std::uint8_t>(repeat.induction_target));
    write(88, 4, 1); // Extended-control subtype: Repeat2D.
    return encoded;
}

inline IcuCommandOpcode decode_icu_command_opcode(EncodedIcuCommand command)
{
    return static_cast<IcuCommandOpcode>(command & 0x3u);
}

inline std::size_t decode_icu_nop_cycles(EncodedIcuCommand command)
{
    if (decode_icu_command_opcode(command) != IcuCommandOpcode::Nop) {
        throw std::logic_error("encoded ICU command is not NOP");
    }
    return static_cast<std::size_t>((command >> 2) & 0x3fffffffull);
}

inline InstructionControlUnit::Repeat decode_icu_repeat(EncodedIcuCommand command)
{
    if (decode_icu_command_opcode(command) != IcuCommandOpcode::Repeat) {
        throw std::logic_error("encoded ICU command is not Repeat");
    }
    auto stride = static_cast<std::int32_t>((command >> 20) & 0x0fffu);
    if ((stride & 0x800) != 0) {
        stride |= ~0x0fff;
    }
    return InstructionControlUnit::Repeat {
        static_cast<std::size_t>((command >> 2) & 0x3ffull),
        static_cast<std::size_t>((command >> 12) & 0xffull),
        stride,
    };
}

inline IcuLoop decode_icu_loop(EncodedIcuCommand command)
{
    if (decode_icu_command_opcode(command) != IcuCommandOpcode::Loop) {
        throw std::logic_error("encoded ICU command is not Loop");
    }
    const auto loop = IcuLoop {
        static_cast<std::size_t>((command >> 2) & 0x3fu),
        static_cast<std::size_t>((command >> 8) & 0xffu),
        static_cast<std::size_t>((command >> 16) & 0xffu),
        static_cast<std::int8_t>(command >> 24),
    };
    if (loop.window_size == 0 || loop.count == 0
        || loop.interval < loop.window_size) {
        throw std::logic_error("encoded ICU Loop has invalid fields");
    }
    return loop;
}

inline IcuRepeat2D decode_icu_repeat_2d(const EncodedIcuRepeat2D& encoded)
{
    auto read = [&](std::size_t offset, std::size_t width) {
        std::uint64_t value = 0;
        for (std::size_t bit = 0; bit < width; ++bit) {
            if ((encoded.words[(offset + bit) / 32]
                    & (std::uint32_t {1} << ((offset + bit) % 32))) != 0)
                value |= std::uint64_t {1} << bit;
        }
        return value;
    };
    if (read(0, 2) != static_cast<std::uint8_t>(IcuCommandOpcode::Loop)
        || read(88, 4) != 1)
        throw std::logic_error("encoded ICU control is not Repeat2D");
    auto signed16 = [&](std::size_t offset) {
        return static_cast<std::int16_t>(read(offset, 16));
    };
    const auto repeat = IcuRepeat2D {
        static_cast<std::size_t>(read(2, 10)),
        static_cast<std::size_t>(read(22, 16)),
        signed16(54),
        static_cast<std::size_t>(read(12, 10)),
        static_cast<std::size_t>(read(38, 16)),
        signed16(70),
        static_cast<IcuInductionTarget>(read(86, 2)),
    };
    if (repeat.inner_count == 0 || repeat.outer_count == 0
        || repeat.inner_interval == 0 || repeat.outer_interval == 0
        || repeat.inner_count * repeat.outer_count <= 1
        || repeat.induction_target > IcuInductionTarget::MxmWeightColumn)
        throw std::logic_error("encoded ICU Repeat2D has invalid fields");
    return repeat;
}

} // namespace isa

} // namespace ftlpu
