#include "ftlpu/icu/fu_3d_codec.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool require(bool condition, const std::string& message)
{
    if (condition) return true;
    std::cerr << message << '\n';
    return false;
}

template <typename Fn>
bool require_throws(Fn&& fn, const std::string& message)
{
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    std::cerr << message << '\n';
    return false;
}

bool same_loop(const ftlpu::IcuLoop3D& lhs,
    const ftlpu::IcuLoop3D& rhs)
{
    return lhs.start_cycle == rhs.start_cycle
        && lhs.counts == rhs.counts
        && lhs.cycle_strides == rhs.cycle_strides;
}

bool same_mem(const ftlpu::MemIcuInstruction& lhs,
    const ftlpu::MemIcuInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && same_loop(lhs.loop, rhs.loop)
        && lhs.address.base_address == rhs.address.base_address
        && lhs.address.inner_stride == rhs.address.inner_stride
        && lhs.address.middle_stride == rhs.address.middle_stride
        && lhs.address.outer_group_size == rhs.address.outer_group_size
        && lhs.address.outer_inner_stride
            == rhs.address.outer_inner_stride
        && lhs.address.outer_group_stride
            == rhs.address.outer_group_stride
        && lhs.stream == rhs.stream;
}

bool same_load(const ftlpu::MxmLoadIcuInstruction& lhs,
    const ftlpu::MxmLoadIcuInstruction& rhs)
{
    return same_loop(lhs.loop, rhs.loop)
        && lhs.weight_buffer_base == rhs.weight_buffer_base
        && lhs.weight_buffer_mode == rhs.weight_buffer_mode
        && lhs.weight_column_base == rhs.weight_column_base
        && lhs.weight_column_strides == rhs.weight_column_strides
        && lhs.weight_stream_base == rhs.weight_stream_base
        && lhs.weight_input_mode == rhs.weight_input_mode;
}

bool same_mode(const ftlpu::MxmComputeIcuMode& lhs,
    const ftlpu::MxmComputeIcuMode& rhs)
{
    return lhs.accumulator_destination == rhs.accumulator_destination
        && lhs.accumulator_clear == rhs.accumulator_clear
        && lhs.accumulator_output_format
            == rhs.accumulator_output_format;
}

bool same_compute(const ftlpu::MxmComputeIcuInstruction& lhs,
    const ftlpu::MxmComputeIcuInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && same_loop(lhs.loop, rhs.loop)
        && lhs.weight_buffer_base == rhs.weight_buffer_base
        && lhs.weight_buffer_mode == rhs.weight_buffer_mode
        && lhs.activation_stream_base == rhs.activation_stream_base
        && lhs.result_stream_base == rhs.result_stream_base
        && lhs.accumulator_address_base == rhs.accumulator_address_base
        && lhs.accumulator_address_strides
            == rhs.accumulator_address_strides
        && lhs.accumulator_row_stride == rhs.accumulator_row_stride
        && lhs.data_format == rhs.data_format
        && same_mode(lhs.regular_mode, rhs.regular_mode)
        && lhs.terminal_dimension == rhs.terminal_dimension
        && same_mode(lhs.terminal_mode, rhs.terminal_mode);
}

template <typename Packet, std::size_t WordCount, std::size_t LaneCount>
bool require_golden(const Packet& packet,
    const std::array<std::array<std::uint32_t, LaneCount>, WordCount>& golden,
    const std::string& message)
{
    if (packet.words.size() != WordCount) return require(false, message);
    for (std::size_t word = 0; word < WordCount; ++word) {
        if (packet.words[word].lanes != golden[word])
            return require(false, message);
    }
    return true;
}

bool verify_golden_vectors_and_extended_discriminator()
{
    const auto mem = ftlpu::MemIcuInstruction::WriteTap3D(
        ftlpu::IcuLoop3D {
            0, {1, 1, 1}, {0xabcdef, 0x123456, 0xfedcba}},
        ftlpu::MemIcuAddress3D::BlockedOuter(
            0x1234, -524288, 524287, 65536, -1, -0x12345),
        ftlpu::StreamId::West(31));
    const auto encoded_mem =
        ftlpu::isa::encode_mem_icu_3d_instruction(mem);
    const std::array<std::array<std::uint32_t, 3>, 3> mem_golden {{
        {0x00000063U, 0x00000000U, 0xd2ef0000U},
        {0x456abc67U, 0xedcba123U, 0xffc8d3ffU},
        {0x00007f6bU, 0xffbffffcU, 0x76e5dfffU},
    }};
    if (!require_golden(encoded_mem, mem_golden,
            "MEM ICU 3-D golden bit vector changed")
        || !require(same_mem(mem,
                ftlpu::isa::decode_mem_icu_3d_instruction(encoded_mem)),
            "MEM ICU 3-D signed-boundary golden vector did not round-trip"))
        return false;

    auto absolute_start = mem;
    absolute_start.loop.start_cycle = 1;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::encode_mem_icu_3d_instruction(
                        absolute_start));
            },
            "FU 3-D encoder accepted an absolute start cycle"))
        return false;
    auto legacy_start_bits = encoded_mem;
    legacy_start_bits.words[0].lanes[0] |= std::uint32_t {1} << 8;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mem_icu_3d_instruction(
                        legacy_start_bits));
            },
            "FU 3-D decoder accepted legacy absolute-start bits"))
        return false;

    const auto load = ftlpu::MxmLoadIcuInstruction::Load3D(
        ftlpu::IcuLoop3D {
            0, {1, 1, 1}, {0x010203, 0x040506, 0x070809}},
        1,
        ftlpu::MxmIcuBufferMode::ToggleDimension2,
        3,
        {-32768, 32767, -1},
        16,
        ftlpu::MxmWeightInputMode::Direct16);
    const auto encoded_load =
        ftlpu::isa::encode_mxm_load_icu_3d_instruction(load);
    const std::array<std::array<std::uint32_t, 4>, 2> load_golden {{
        {0x00000043U, 0x00000000U, 0x22030000U, 0x40506010U},
        {0x70809047U, 0xff0001f0U, 0x61fffeffU, 0x00000000U},
    }};
    if (!require_golden(encoded_load, load_golden,
            "MXM LOAD_3D golden bit vector changed")
        || !require(same_load(load,
                ftlpu::isa::decode_mxm_load_icu_3d_instruction(
                    encoded_load)),
            "MXM LOAD_3D signed-boundary golden vector did not round-trip"))
        return false;

    const auto dequant = ftlpu::MxmDequantIcuInstruction::Dequant3D(
        ftlpu::IcuLoop3D {
            0, {1, 1, 1}, {0x112233, 0x445566, 0x778899}},
        ftlpu::MxmDequantInstruction::ScaleBits(0xbeef));
    const auto encoded_dequant =
        ftlpu::isa::encode_mxm_dequant_icu_3d_instruction(dequant);
    const std::array<std::array<std::uint32_t, 4>, 2> dequant_golden {{
        {0x00000043U, 0x00000000U, 0x22330000U, 0x45566112U},
        {0x78899447U, 0x000beef7U, 0x00000000U, 0x00000000U},
    }};
    const auto decoded_dequant =
        ftlpu::isa::decode_mxm_dequant_icu_3d_instruction(encoded_dequant);
    if (!require_golden(encoded_dequant, dequant_golden,
            "MXM DEQUANT_3D golden bit vector changed")
        || !require(same_loop(dequant.loop, decoded_dequant.loop)
                && dequant.instruction.scale_bf16
                    == decoded_dequant.instruction.scale_bf16,
            "MXM DEQUANT_3D golden vector did not round-trip"))
        return false;

    const auto regular = ftlpu::MxmComputeIcuMode {
        ftlpu::MxmAccumulatorDestination::Sram,
        false,
        ftlpu::MxmAccumulatorOutputFormat::Float32};
    const auto terminal = ftlpu::MxmComputeIcuMode {
        ftlpu::MxmAccumulatorDestination::Stream,
        true,
        ftlpu::MxmAccumulatorOutputFormat::BFloat16};
    const auto compute = ftlpu::MxmComputeIcuInstruction::Compute3D(
        ftlpu::IcuLoop3D {
            0, {1, 1, 1}, {0x102030, 0x405060, 0x708090}},
        1,
        ftlpu::MxmIcuBufferMode::ToggleDimension1,
        30,
        28,
        8191,
        {0, -8192, 8191},
        8191,
        ftlpu::MxmDataFormat::BFloat16,
        regular,
        ftlpu::MxmComputeIcuInstruction::kNoTerminalDimension,
        terminal);
    const auto encoded_compute =
        ftlpu::isa::encode_mxm_compute_icu_3d_instruction(compute);
    const std::array<std::array<std::uint32_t, 4>, 2> compute_golden {{
        {0x00000043U, 0x00000000U, 0x02300000U, 0x05060102U},
        {0x08090447U, 0x3fffcf57U, 0xfe000000U, 0x3e3fff7fU},
    }};
    if (!require_golden(encoded_compute, compute_golden,
            "MXM COMPUTE_3D golden bit vector changed")
        || !require(same_compute(compute,
                ftlpu::isa::decode_mxm_compute_icu_3d_instruction(
                    encoded_compute)),
            "MXM COMPUTE_3D signed-boundary golden vector did not round-trip"))
        return false;

    const auto repeat = ftlpu::isa::encode_icu_repeat_2d(ftlpu::IcuRepeat2D {
        16, 1, 0, 2, 17, 0, ftlpu::IcuInductionTarget::None});
    auto read_packet = encoded_mem;
    read_packet.words[0].lanes[0] &= ~0x30U;
    if (!require((repeat.words[0] & 0xffU)
                == (read_packet.words[0].lanes[0] & 0xffU),
            "Repeat2D collision fixture does not share the 3-D low header")
        || !require(((repeat.words[2] >> 24) & 0xfU) == 1U
                && ((read_packet.words[0].lanes[2] >> 24) & 0xfU) == 2U,
            "Extended subtype registry does not separate Repeat2D and FU 3-D")
        || !require(ftlpu::isa::decode_icu_command_opcode(
                read_packet.words[0].lanes[0])
                == ftlpu::isa::IcuCommandOpcode::Extended,
            "FU 3-D word does not use the global Extended envelope"))
        return false;

    const ftlpu::isa::EncodedIcuRepeat2D first_3d_word {{
        read_packet.words[0].lanes[0],
        read_packet.words[0].lanes[1],
        read_packet.words[0].lanes[2],
    }};
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_icu_repeat_2d(first_3d_word));
            },
            "Repeat2D decoder accepted the FU 3-D Extended subtype"))
        return false;

    auto repeat_as_3d = read_packet;
    repeat_as_3d.words[0].lanes = repeat.words;
    return require_throws(
        [&] {
            static_cast<void>(
                ftlpu::isa::decode_mem_icu_3d_instruction(repeat_as_3d));
        },
        "FU 3-D decoder accepted the Repeat2D Extended subtype");
}

bool verify_mem_codec()
{
    const auto blocked_loop = ftlpu::IcuLoop3D {
        0, {4, 48, 14}, {1, 32, 1536}};
    const auto blocked_address = ftlpu::MemIcuAddress3D::BlockedOuter(
        0, 1, 8, 2, 4, 384);
    const auto negative_loop = ftlpu::IcuLoop3D {
        0, {2, 2, 3}, {1, 4, 16}};
    const auto negative_address = ftlpu::MemIcuAddress3D::BlockedOuter(
        100, -1, 4, 2, -8, 16);
    const auto affine_loop = ftlpu::IcuLoop3D {
        0, {2, 2, 2}, {1, 4, 16}};

    const std::array instructions {
        ftlpu::MemIcuInstruction::Read3D(
            blocked_loop, blocked_address, ftlpu::StreamId::East(8)),
        ftlpu::MemIcuInstruction::Write3D(
            negative_loop, negative_address, ftlpu::StreamId::West(13)),
        ftlpu::MemIcuInstruction::WriteTap3D(
            affine_loop,
            ftlpu::MemIcuAddress3D::Affine(64, {1, 4, 32}),
            ftlpu::StreamId::West(21)),
    };

    for (std::size_t index = 0; index < instructions.size(); ++index) {
        const auto encoded =
            ftlpu::isa::encode_mem_icu_3d_instruction(instructions[index]);
        const auto decoded =
            ftlpu::isa::decode_mem_icu_3d_instruction(encoded);
        if (!require(same_mem(instructions[index], decoded),
                "MEM ICU 3-D codec round-trip failed"))
            return false;

        const auto expected_operation = static_cast<std::uint32_t>(index);
        for (std::size_t word = 0; word < encoded.words.size(); ++word) {
            const auto header = encoded.words[word].lanes[0] & 0xffU;
            const auto expected = 0x43U
                | (static_cast<std::uint32_t>(word) << 2)
                | (expected_operation << 4);
            if (!require(header == expected,
                    "MEM ICU 3-D physical-word header is incorrect"))
                return false;
        }
    }

    auto bad_reserved =
        ftlpu::isa::encode_mem_icu_3d_instruction(instructions[0]);
    bad_reserved.words[2].lanes[2] |= std::uint32_t {1} << 31;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mem_icu_3d_instruction(
                        bad_reserved));
            },
            "MEM ICU 3-D decoder accepted a non-zero reserved bit"))
        return false;

    auto bad_continuation =
        ftlpu::isa::encode_mem_icu_3d_instruction(instructions[0]);
    bad_continuation.words[1].lanes[0] ^= 0x04U;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mem_icu_3d_instruction(
                        bad_continuation));
            },
            "MEM ICU 3-D decoder accepted a bad continuation index"))
        return false;

    auto bad_class =
        ftlpu::isa::encode_mem_icu_3d_instruction(instructions[0]);
    bad_class.words[0].lanes[0] ^= 0x40U;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mem_icu_3d_instruction(bad_class));
            },
            "MEM ICU 3-D decoder accepted a non-3-D word-0 class"))
        return false;

    auto bad_operation =
        ftlpu::isa::encode_mem_icu_3d_instruction(instructions[0]);
    bad_operation.words[0].lanes[0]
        = (bad_operation.words[0].lanes[0] & ~0x30U) | 0x30U;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mem_icu_3d_instruction(
                        bad_operation));
            },
            "MEM ICU 3-D decoder accepted an unknown local operation"))
        return false;

    auto wide_stride = ftlpu::MemIcuInstruction::Read3D(
        ftlpu::IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
        ftlpu::MemIcuAddress3D::Affine(
            0, {std::int64_t {1} << 19, 0, 0}),
        ftlpu::StreamId::East(0));
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::encode_mem_icu_3d_instruction(wide_stride));
            },
            "MEM ICU 3-D encoder accepted a stride wider than 20 bits"))
        return false;

    auto non_power_of_two_group = ftlpu::MemIcuInstruction::Read3D(
        ftlpu::IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
        ftlpu::MemIcuAddress3D::BlockedOuter(
            0, 0, 0, 3, 0, 0),
        ftlpu::StreamId::East(0));
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::encode_mem_icu_3d_instruction(
                        non_power_of_two_group));
            },
            "MEM ICU 3-D encoder accepted a non-power-of-two group"))
        return false;

    auto wide_group = ftlpu::MemIcuInstruction::Read3D(
        ftlpu::IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
        ftlpu::MemIcuAddress3D::BlockedOuter(
            0, 0, 0, 65537, 0, 0),
        ftlpu::StreamId::East(0));
    return require_throws(
        [&] {
            static_cast<void>(
                ftlpu::isa::encode_mem_icu_3d_instruction(wide_group));
        },
        "MEM ICU 3-D encoder accepted a group size wider than 16 bits");
}

bool verify_mxm_load_codec()
{
    const auto instruction = ftlpu::MxmLoadIcuInstruction::Load3D(
        ftlpu::IcuLoop3D {0, {4, 5, 3}, {1, 8, 64}},
        1,
        ftlpu::MxmIcuBufferMode::ToggleDimension1,
        3,
        {-1, 0, 0},
        8,
        ftlpu::MxmWeightInputMode::Direct16);
    const auto encoded =
        ftlpu::isa::encode_mxm_load_icu_3d_instruction(instruction);
    const auto decoded =
        ftlpu::isa::decode_mxm_load_icu_3d_instruction(encoded);
    if (!require(same_load(instruction, decoded),
            "MXM LOAD_3D codec round-trip failed"))
        return false;
    if (!require((encoded.words[0].lanes[0] & 0xffU) == 0x43U
            && (encoded.words[1].lanes[0] & 0xffU) == 0x47U,
            "MXM LOAD_3D physical-word headers are incorrect"))
        return false;

    auto bad_reserved = encoded;
    bad_reserved.words[1].lanes[3] |= std::uint32_t {1} << 31;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mxm_load_icu_3d_instruction(
                        bad_reserved));
            },
            "MXM LOAD_3D decoder accepted a non-zero reserved bit"))
        return false;

    auto bad_version = encoded;
    bad_version.words[0].lanes[0] ^= 0x80U;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mxm_load_icu_3d_instruction(
                        bad_version));
            },
            "MXM LOAD_3D decoder accepted an unknown format version"))
        return false;

    auto wide_stride = instruction;
    wide_stride.loop = {0, {1, 1, 1}, {1, 1, 1}};
    wide_stride.weight_column_base = 0;
    wide_stride.weight_column_strides = {0, 32768, 0};
    return require_throws(
        [&] {
            static_cast<void>(
                ftlpu::isa::encode_mxm_load_icu_3d_instruction(
                    wide_stride));
        },
        "MXM LOAD_3D encoder accepted a stride wider than 16 bits");
}

bool verify_mxm_dequant_codec()
{
    const auto instruction = ftlpu::MxmDequantIcuInstruction::Dequant3D(
        ftlpu::IcuLoop3D {0, {65536, 1, 1}, {1, 1, 1}},
        ftlpu::MxmDequantInstruction::ScaleBits(0x3c00));
    const auto encoded =
        ftlpu::isa::encode_mxm_dequant_icu_3d_instruction(instruction);
    const auto decoded =
        ftlpu::isa::decode_mxm_dequant_icu_3d_instruction(encoded);
    if (!require(same_loop(instruction.loop, decoded.loop)
            && instruction.instruction.scale_bf16
                == decoded.instruction.scale_bf16,
            "MXM DEQUANT_3D codec round-trip failed"))
        return false;
    if (!require((encoded.words[0].lanes[0] & 0xffU) == 0x43U
            && (encoded.words[1].lanes[0] & 0xffU) == 0x47U,
            "MXM DEQUANT_3D physical-word headers are incorrect"))
        return false;

    auto bad_reserved = encoded;
    bad_reserved.words[1].lanes[3] |= std::uint32_t {1} << 31;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mxm_dequant_icu_3d_instruction(
                        bad_reserved));
            },
            "MXM DEQUANT_3D decoder accepted a reserved bit"))
        return false;

    auto zero_stride = encoded;
    // P[72] is physical bit 80 in word 0.
    zero_stride.words[0].lanes[2] &= ~(std::uint32_t {1} << 16);
    return require_throws(
        [&] {
            static_cast<void>(
                ftlpu::isa::decode_mxm_dequant_icu_3d_instruction(
                    zero_stride));
        },
        "MXM DEQUANT_3D decoder accepted a zero cycle stride");
}

bool verify_mxm_compute_codec()
{
    const auto regular = ftlpu::MxmComputeIcuMode {
        ftlpu::MxmAccumulatorDestination::Sram,
        false,
        ftlpu::MxmAccumulatorOutputFormat::Float32};
    const auto terminal = ftlpu::MxmComputeIcuMode {
        ftlpu::MxmAccumulatorDestination::Stream,
        true,
        ftlpu::MxmAccumulatorOutputFormat::BFloat16};
    const auto instruction = ftlpu::MxmComputeIcuInstruction::Compute3D(
        ftlpu::IcuLoop3D {0, {32, 3, 2}, {1, 64, 256}},
        1,
        ftlpu::MxmIcuBufferMode::ToggleDimension2,
        16,
        12,
        100,
        {0, 64, -32},
        2,
        ftlpu::MxmDataFormat::BFloat16,
        regular,
        2,
        terminal);
    const auto encoded =
        ftlpu::isa::encode_mxm_compute_icu_3d_instruction(instruction);
    const auto decoded =
        ftlpu::isa::decode_mxm_compute_icu_3d_instruction(encoded);
    if (!require(same_compute(instruction, decoded),
            "MXM COMPUTE_3D codec round-trip failed"))
        return false;
    if (!require((encoded.words[0].lanes[0] & 0xffU) == 0x43U
            && (encoded.words[1].lanes[0] & 0xffU) == 0x47U,
            "MXM COMPUTE_3D physical-word headers are incorrect"))
        return false;

    const auto accumulatorRead =
        ftlpu::MxmComputeIcuInstruction::AccumulatorRead3D(
            ftlpu::IcuLoop3D {0, {2, 3, 1}, {1, 4, 16}},
            28,
            100,
            {1, 16, 0},
            false,
            ftlpu::MxmAccumulatorOutputFormat::BFloat16,
            ftlpu::MxmAccumulatorDestination::Stream);
    const auto encodedAccumulatorRead =
        ftlpu::isa::encode_mxm_compute_icu_3d_instruction(
            accumulatorRead);
    const auto decodedAccumulatorRead =
        ftlpu::isa::decode_mxm_compute_icu_3d_instruction(
            encodedAccumulatorRead);
    if (!require(same_compute(accumulatorRead, decodedAccumulatorRead),
            "MXM ACCUMULATOR_READ_3D codec round-trip failed")
        || !require((encodedAccumulatorRead.words[0].lanes[0] & 0xffU)
                    == 0x53U
                && (encodedAccumulatorRead.words[1].lanes[0] & 0xffU)
                    == 0x57U,
            "MXM ACCUMULATOR_READ_3D local operation headers are incorrect"))
        return false;

    auto accumulatorReadReserved = encodedAccumulatorRead;
    // P[144] is physical bit 36 in the second 128-bit word.
    accumulatorReadReserved.words[1].lanes[1]
        |= std::uint32_t {1} << 4;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mxm_compute_icu_3d_instruction(
                        accumulatorReadReserved));
            },
            "MXM ACCUMULATOR_READ_3D decoder accepted a reserved bit"))
        return false;

    auto invalidLocalOperation = encodedAccumulatorRead;
    invalidLocalOperation.words[0].lanes[0] ^= 0x30U;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mxm_compute_icu_3d_instruction(
                        invalidLocalOperation));
            },
            "MXM compute ICU decoder accepted an unknown local operation"))
        return false;

    auto bad_reserved = encoded;
    bad_reserved.words[1].lanes[3] |= std::uint32_t {1} << 31;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mxm_compute_icu_3d_instruction(
                        bad_reserved));
            },
            "MXM COMPUTE_3D decoder accepted its reserved payload bit"))
        return false;

    auto bad_continuation = encoded;
    bad_continuation.words[1].lanes[0] ^= 0x10U;
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::decode_mxm_compute_icu_3d_instruction(
                        bad_continuation));
            },
            "MXM COMPUTE_3D decoder accepted a mismatched continuation"))
        return false;

    auto wide_stride = instruction;
    wide_stride.loop = {0, {1, 1, 1}, {1, 1, 1}};
    wide_stride.accumulator_address_strides = {0, 8192, 0};
    if (!require_throws(
            [&] {
                static_cast<void>(
                    ftlpu::isa::encode_mxm_compute_icu_3d_instruction(
                        wide_stride));
            },
            "MXM COMPUTE_3D encoder accepted a stride wider than 14 bits"))
        return false;

    auto wide_row_stride = instruction;
    wide_row_stride.loop = {0, {1, 1, 1}, {1, 1, 1}};
    wide_row_stride.accumulator_address_strides = {0, 0, 0};
    wide_row_stride.accumulator_row_stride = 8192;
    return require_throws(
        [&] {
            static_cast<void>(
                ftlpu::isa::encode_mxm_compute_icu_3d_instruction(
                    wide_row_stride));
        },
        "MXM COMPUTE_3D encoder accepted a row stride wider than 13 bits");
}

} // namespace

int main()
{
    try {
        if (!verify_golden_vectors_and_extended_discriminator()
            || !verify_mem_codec()
            || !verify_mxm_load_codec()
            || !verify_mxm_dequant_codec()
            || !verify_mxm_compute_codec())
            return 1;
    } catch (const std::exception& ex) {
        std::cerr << "fu_3d_codec_test failed: " << ex.what() << '\n';
        return 1;
    }

    std::cout << "FU-specific 3-D coarse instruction codecs passed\n";
    return 0;
}
