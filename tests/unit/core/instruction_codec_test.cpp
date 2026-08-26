#include "ftlpu/core/instruction_codec.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool same_mem(const ftlpu::MemInstruction& lhs, const ftlpu::MemInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && lhs.address == rhs.address
        && lhs.stream == rhs.stream
        && lhs.map_stream == rhs.map_stream
        && lhs.preserve_stream == rhs.preserve_stream;
}

bool same_mxm(const ftlpu::MxmControlInstruction& lhs, const ftlpu::MxmControlInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && lhs.weight_buffer == rhs.weight_buffer
        && lhs.weight_stream_base == rhs.weight_stream_base
        && lhs.stream_base == rhs.stream_base
        && lhs.activation_stream_base == rhs.activation_stream_base
        && lhs.weight_column == rhs.weight_column
        && lhs.weight_load_mode == rhs.weight_load_mode
        && lhs.weight_input_mode == rhs.weight_input_mode
        && lhs.weight_inner_column == rhs.weight_inner_column
        && lhs.accumulator_address == rhs.accumulator_address
        && lhs.accumulator_row_stride == rhs.accumulator_row_stride
        && lhs.accumulator_destination == rhs.accumulator_destination
        && lhs.accumulator_clear == rhs.accumulator_clear
        && lhs.data_format == rhs.data_format
        && lhs.accumulator_output_format == rhs.accumulator_output_format
        && lhs.decode_operation == rhs.decode_operation
        && lhs.decode_layout == rhs.decode_layout;
}

bool same_vxm(const ftlpu::VxmLaneAluInstruction& lhs, const ftlpu::VxmLaneAluInstruction& rhs)
{
    return lhs.operation == rhs.operation
        && lhs.lhs.kind == rhs.lhs.kind
        && lhs.lhs.immediate == rhs.lhs.immediate
        && lhs.lhs.stream_group == rhs.lhs.stream_group
        && lhs.rhs.kind == rhs.rhs.kind
        && lhs.rhs.immediate == rhs.rhs.immediate
        && lhs.rhs.stream_group == rhs.rhs.stream_group
        && lhs.precision == rhs.precision
        && lhs.output_type == rhs.output_type
        && lhs.output_stream == rhs.output_stream
        && lhs.accumulator_reset == rhs.accumulator_reset
        && lhs.accumulator_write == rhs.accumulator_write
        && lhs.accumulator_emit == rhs.accumulator_emit
        && lhs.repeat_count == rhs.repeat_count
        && lhs.local_scalar_write == rhs.local_scalar_write;
}

bool require(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

template <typename Fn>
bool require_throws(Fn fn, const std::string& message)
{
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }

    std::cerr << message << '\n';
    return false;
}

bool verify_mem_codec()
{
    const ftlpu::MemInstruction instructions[] {
        ftlpu::MemInstruction::Read(
            ftlpu::hw::kSramDepthRows / 2, 45),
        ftlpu::MemInstruction::Write(ftlpu::hw::kSramDepthRows - 1, 63),
        ftlpu::MemInstruction::WriteTap(123, 34),
        ftlpu::MemInstruction::Gather(7, 55),
        ftlpu::MemInstruction::Scatter(36, 12),
    };

    for (const auto& instruction : instructions) {
        const auto encoded = ftlpu::isa::encode_mem_instruction(instruction);
        const auto decoded = ftlpu::isa::decode_mem_instruction(encoded);
        if (!require(same_mem(instruction, decoded), "MEM instruction codec round-trip failed")) {
            return false;
        }
    }

    if (!require_throws(
        [] {
            ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Read(
                    ftlpu::hw::kSramDepthRows, 0));
        },
        "MEM codec should reject row addresses outside the configured bank")) {
        return false;
    }
    return require_throws(
        [] {
            const auto reserved_address_bit =
                static_cast<std::uint64_t>(ftlpu::hw::kSramDepthRows) << 15;
            static_cast<void>(ftlpu::isa::decode_mem_instruction(
                reserved_address_bit));
        },
        "MEM codec should reject address bits above the modeled row depth");
}

bool verify_mxm_codec()
{
    const ftlpu::MxmControlInstruction instructions[] {
        ftlpu::MxmControlInstruction::IW(1, 3,
            ftlpu::MxmWeightInputMode::Int8DequantBf16, 8),
        ftlpu::MxmControlInstruction::IWColumn(1, 2, 7,
            ftlpu::MxmWeightInputMode::Int8DequantBf16, 15),
        ftlpu::MxmControlInstruction::IWDirect16(0, 1, 16),
        ftlpu::MxmControlInstruction::IWColumnDirect16(0, 3, 4, 30),
        ftlpu::MxmControlInstruction::Compute(
            1,
            30,
            20,
            ftlpu::hw::kMxmAccumulatorRows - 1,
            48,
            ftlpu::MxmAccumulatorDestination::Sram),
        ftlpu::MxmControlInstruction::Compute(
            0,
            4,
            8,
            ftlpu::hw::kMxmAccumulatorRows / 2,
            1,
            ftlpu::MxmAccumulatorDestination::Stream,
            ftlpu::MxmDataFormat::BFloat16),
        ftlpu::MxmControlInstruction::AccumulatorRead(
            ftlpu::hw::kMxmAccumulatorRows - 1, 16, false),
        ftlpu::MxmControlInstruction::AccumulatorRead(
            0,
            0,
            true,
            ftlpu::MxmAccumulatorOutputFormat::BFloat16,
            ftlpu::MxmAccumulatorDestination::Sram),
        ftlpu::MxmControlInstruction::DecodeLoadActivation(
            1,
            4,
            ftlpu::MxmDataFormat::BFloat16),
        ftlpu::MxmControlInstruction::DecodeStreamCompute(
            1,
            8,
            ftlpu::MxmDataFormat::BFloat16,
            ftlpu::hw::kMxmAccumulatorRows / 3,
            2,
            ftlpu::MxmAccumulatorDestination::Sram,
            false),
        ftlpu::MxmControlInstruction::DecodeLoadActivation(
            0,
            30,
            ftlpu::MxmDataFormat::BFloat16,
            ftlpu::MxmDecodeLayout::Native4x4),
        ftlpu::MxmControlInstruction::DecodeStreamCompute(
            0,
            4,
            ftlpu::MxmDataFormat::BFloat16,
            ftlpu::hw::kMxmAccumulatorRows / 4,
            0,
            ftlpu::MxmAccumulatorDestination::Stream,
            true,
            ftlpu::MxmDecodeLayout::Native4x4),
    };

    for (const auto& instruction : instructions) {
        const auto encoded = ftlpu::isa::encode_mxm_instruction(instruction);
        const auto decoded = ftlpu::isa::decode_mxm_instruction(encoded);
        if (!require(same_mxm(instruction, decoded), "MXM instruction codec round-trip failed")) {
            return false;
        }
    }

    if (!require(
            ftlpu::isa::encode_mxm_instruction(
                ftlpu::MxmControlInstruction::IW(1, 3))
                == 0x1cu,
            "MXM base-zero IW encoding changed")) {
        return false;
    }
    if (!require(
            (ftlpu::isa::encode_mxm_instruction(
                 ftlpu::MxmControlInstruction::IW(1, 3,
                     ftlpu::MxmWeightInputMode::Int8DequantBf16, 8))
                >> 10)
                    == 8,
            "MXM IW weight stream base was not encoded")) {
        return false;
    }
    const auto direct_iw = ftlpu::isa::encode_mxm_instruction(
        ftlpu::MxmControlInstruction::IWDirect16(1, 3));
    if (!require(
            (direct_iw & (std::uint64_t {1} << 9)) != 0,
            "MXM Direct16 IW input-mode bit was not encoded")) {
        return false;
    }
    const auto dequant = ftlpu::MxmDequantInstruction::Scale(0.125f);
    const auto encoded_dequant =
        ftlpu::isa::encode_mxm_dequant_instruction(dequant);
    const auto decoded_dequant =
        ftlpu::isa::decode_mxm_dequant_instruction(encoded_dequant);
    if (!require(
            decoded_dequant.scale_bf16 == dequant.scale_bf16,
            "MXM Dequant scale immediate codec round-trip failed")) {
        return false;
    }
    const auto bf16_compute = ftlpu::isa::encode_mxm_instruction(
        ftlpu::MxmControlInstruction::Compute(
            0,
            0,
            0,
            0,
            1,
            ftlpu::MxmAccumulatorDestination::Stream,
            ftlpu::MxmDataFormat::BFloat16));
    if (!require(
            (bf16_compute & (std::uint64_t {1} << 45)) != 0,
            "MXM BF16 Compute format bit was not encoded")) {
        return false;
    }
    if (!require(
            (bf16_compute & (std::uint64_t {1} << 46)) == 0,
            "MXM Compute must keep reserved bit 46 clear")) {
        return false;
    }
    const auto bf16_output_compute =
        ftlpu::MxmControlInstruction::Compute(
            0,
            0,
            0,
            0,
            1,
            ftlpu::MxmAccumulatorDestination::Stream,
            ftlpu::MxmDataFormat::BFloat16,
            true,
            ftlpu::MxmAccumulatorOutputFormat::BFloat16);
    const auto encoded_bf16_output =
        ftlpu::isa::encode_mxm_instruction(bf16_output_compute);
    const auto decoded_bf16_output =
        ftlpu::isa::decode_mxm_instruction(encoded_bf16_output);
    if (!require(
            (encoded_bf16_output & (std::uint64_t {1} << 48)) != 0
                && decoded_bf16_output.accumulator_output_format
                    == ftlpu::MxmAccumulatorOutputFormat::BFloat16,
            "MXM accumulator BF16 output codec round-trip failed")) {
        return false;
    }
    const auto retain_compute = ftlpu::isa::encode_mxm_instruction(
        ftlpu::MxmControlInstruction::Compute(
            0,
            0,
            0,
            0,
            1,
            ftlpu::MxmAccumulatorDestination::Stream,
            ftlpu::MxmDataFormat::BFloat16,
            false));
    if (!require(
            (retain_compute & (std::uint64_t {1} << 47)) != 0,
            "MXM Compute retain-accumulator bit was not encoded")) {
        return false;
    }
    const auto vector_read = ftlpu::isa::encode_mxm_instruction(
        ftlpu::MxmControlInstruction::AccumulatorRead(
            ftlpu::hw::kMxmAccumulatorRows - 1, 16, false));
    if (!require(
            (vector_read & (std::uint64_t {1} << 46)) == 0,
            "MXM AccumulatorRead must keep reserved bit 46 clear")) {
        return false;
    }
    const auto sram_read = ftlpu::isa::encode_mxm_instruction(
        ftlpu::MxmControlInstruction::AccumulatorRead(
            0,
            0,
            true,
            ftlpu::MxmAccumulatorOutputFormat::BFloat16,
            ftlpu::MxmAccumulatorDestination::Sram));
    const auto decoded_sram_read =
        ftlpu::isa::decode_mxm_instruction(sram_read);
    if (!require(
            (sram_read & (std::uint64_t {1} << 44)) == 0
                && (sram_read & (std::uint64_t {1} << 48)) != 0
                && decoded_sram_read.accumulator_destination
                    == ftlpu::MxmAccumulatorDestination::Sram
                && decoded_sram_read.accumulator_output_format
                    == ftlpu::MxmAccumulatorOutputFormat::BFloat16,
            "MXM SRAM AccumulatorRead fields were not encoded")) {
        return false;
    }
    return require_throws(
        [] {
            ftlpu::isa::encode_mxm_instruction(ftlpu::MxmControlInstruction::IW(32));
        },
        "MXM codec should reject weight buffers outside the two-buffer set")
        && require_throws(
            [] {
                auto invalid = ftlpu::MxmControlInstruction::IW(0, 0);
                invalid.weight_inner_column = 1;
                ftlpu::isa::encode_mxm_instruction(invalid);
            },
            "MXM codec should reject an inner column in full-supercell mode");
}

bool verify_vxm_codec()
{
    constexpr auto queue = std::size_t {1};
    constexpr auto depth = ftlpu::VxmChainDepth::Two;
    auto instruction = ftlpu::VxmLaneAluInstruction {};
    instruction.operation = ftlpu::VxmAluOpcode::Multiply;
    instruction.lhs = ftlpu::VxmLaneOperand::Previous();
    instruction.rhs = ftlpu::VxmLaneOperand::Imm(0.125f);
    instruction.precision = ftlpu::VxmAluPrecision::Float32;
    instruction.output_type = ftlpu::VxmCastTarget::BFloat16;
    instruction.output_stream =
        ftlpu::VxmLane::fixed_output_stream_for_block(
            ftlpu::VxmLane::block_for_stage(queue));
    instruction.repeat_count = 7;

    const auto encoded = ftlpu::isa::encode_vxm_instruction(
        queue, depth, instruction);
    const auto decoded = ftlpu::isa::decode_vxm_instruction(queue, encoded);
    if (!require(
            decoded.chain_depth == depth
                && same_vxm(instruction, decoded.instruction),
            "VXM compact instruction codec round-trip failed")) {
        return false;
    }

    auto selected_streams = ftlpu::VxmLaneAluInstruction {};
    selected_streams.operation = ftlpu::VxmAluOpcode::Add;
    selected_streams.lhs =
        ftlpu::VxmLaneOperand::StreamBFloat16(1.0f, 3);
    selected_streams.rhs =
        ftlpu::VxmLaneOperand::StreamFloat16(1.0f, 7);
    const auto selected_packet = ftlpu::isa::encode_vxm_instruction(
        0, depth, selected_streams);
    const auto selected_decoded = ftlpu::isa::decode_vxm_instruction(
        0, selected_packet);
    if (!require(
            selected_decoded.chain_depth == depth
                && same_vxm(selected_streams,
                    selected_decoded.instruction),
            "VXM stream-group selectors were not preserved")) {
        return false;
    }

    return require_throws(
        [] {
            auto invalid = ftlpu::VxmLaneAluInstruction {};
            invalid.lhs = ftlpu::VxmLaneOperand::Imm(1.0f);
            invalid.rhs = ftlpu::VxmLaneOperand::Imm(2.0f);
            ftlpu::isa::encode_vxm_instruction(
                0, ftlpu::VxmChainDepth::Two, invalid);
        },
        "VXM codec should reject two distinct immediate values")
        && require_throws(
            [] {
                auto invalid = ftlpu::VxmLaneAluInstruction {
                    ftlpu::VxmAluOpcode::Bypass,
                    ftlpu::VxmLaneOperand::StreamFloat16(1.0f, 8)};
                ftlpu::isa::encode_vxm_instruction(
                    0, ftlpu::VxmChainDepth::Two, invalid);
            },
            "VXM codec should reject stream groups outside 0..7");
}

bool verify_icu_command_codec()
{
    const auto nop = ftlpu::isa::encode_icu_nop(1234);
    if (!require(ftlpu::isa::decode_icu_command_opcode(nop) == ftlpu::isa::IcuCommandOpcode::Nop, "ICU NOP opcode decode failed")) {
        return false;
    }
    if (!require(ftlpu::isa::decode_icu_nop_cycles(nop) == 1234, "ICU NOP cycle decode failed")) {
        return false;
    }

    const auto repeat = ftlpu::InstructionControlUnit::Repeat {7, 3, -16};
    const auto decoded = ftlpu::isa::decode_icu_repeat(ftlpu::isa::encode_icu_repeat(repeat));
    if (!require(
            decoded.count == repeat.count
                && decoded.interval == repeat.interval
                && decoded.address_stride == repeat.address_stride,
            "ICU Repeat codec round-trip failed")) {
        return false;
    }

    return require_throws(
        [] {
            ftlpu::isa::encode_icu_repeat(ftlpu::InstructionControlUnit::Repeat {1, 1, 2048});
        },
        "ICU Repeat codec should reject strides wider than signed 12 bits");
}

bool verify_sxm_codec()
{
    auto src = ftlpu::SxmInstruction::StreamList {};
    auto dst = ftlpu::SxmInstruction::StreamList {};
    for (std::size_t stream = 0; stream < 16; ++stream) {
        src.push_back(ftlpu::SxmStreamId {stream});
        dst.push_back(ftlpu::SxmStreamId {32 + stream});
    }
    auto map = ftlpu::SxmInstruction::PermuteMap {};
    for (std::size_t lane = 0; lane < map.size(); ++lane) {
        map[lane] = map.size() - lane - 1;
    }
    auto instruction =
        ftlpu::SxmInstruction::Permute(std::move(src), std::move(dst), map);
    instruction.output_row = 5;
    instruction.input_row = 3;
    instruction.output_tile = 2;
    const auto decoded =
        ftlpu::isa::decode_sxm_instruction(ftlpu::isa::encode_sxm_instruction(instruction));
    return require(
        decoded.opcode == instruction.opcode
            && decoded.src_streams == instruction.src_streams
            && decoded.dst_streams == instruction.dst_streams
            && decoded.permute_map == instruction.permute_map
            && decoded.output_row == instruction.output_row
            && decoded.input_row == instruction.input_row
            && decoded.output_tile == instruction.output_tile,
        "SXM instruction codec round-trip failed");
}

} // namespace

int main()
try
{
    if (!verify_mem_codec()) {
        return 1;
    }
    if (!verify_mxm_codec()) {
        return 1;
    }
    if (!verify_vxm_codec()) {
        return 1;
    }
    if (!verify_icu_command_codec()) {
        return 1;
    }
    if (!verify_sxm_codec()) {
        return 1;
    }
    return 0;
}
catch (const std::exception& ex) {
    std::cerr << "instruction_codec_test failed: " << ex.what() << '\n';
    return 1;
}
