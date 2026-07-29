#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/core/instruction_packet.hpp"

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
        && lhs.accumulator_destination == rhs.accumulator_destination
        && lhs.write_address == rhs.write_address
        && lhs.write_stream == rhs.write_stream;
}

bool same_mxm(const ftlpu::MxmControlInstruction& lhs, const ftlpu::MxmControlInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && lhs.weight_buffer == rhs.weight_buffer
        && lhs.stream_base == rhs.stream_base
        && lhs.activation_stream_base == rhs.activation_stream_base
        && lhs.accumulator_bank == rhs.accumulator_bank
        && lhs.accumulate == rhs.accumulate
        && lhs.reduce == rhs.reduce
        && lhs.start_of_k_block == rhs.start_of_k_block;
}

bool same_operand(const ftlpu::VxmLaneOperand& lhs, const ftlpu::VxmLaneOperand& rhs)
{
    return lhs.kind == rhs.kind
        && lhs.immediate == rhs.immediate
        && lhs.scale == rhs.scale
        && lhs.zero_point == rhs.zero_point;
}

bool same_vxm(const ftlpu::VxmLaneAluInstruction& lhs, const ftlpu::VxmLaneAluInstruction& rhs)
{
    return lhs.operation == rhs.operation
        && same_operand(lhs.lhs, rhs.lhs)
        && same_operand(lhs.rhs, rhs.rhs)
        && lhs.precision == rhs.precision
        && lhs.output_type == rhs.output_type
        && lhs.output_scale == rhs.output_scale
        && lhs.output_zero_point == rhs.output_zero_point
        && lhs.output_stream == rhs.output_stream
        && lhs.accumulator_reset == rhs.accumulator_reset
        && lhs.accumulator_write == rhs.accumulator_write
        && lhs.accumulator_emit == rhs.accumulator_emit
        && lhs.repeat_count == rhs.repeat_count
        && lhs.input_hemisphere == rhs.input_hemisphere
        && lhs.output_hemisphere == rhs.output_hemisphere;
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
        ftlpu::MemInstruction::Read(4096, 45),
        ftlpu::MemInstruction::Write(ftlpu::hw::kMemLocalWordAddressCount - 1, 63),
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

    return require_throws(
        [] {
            ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Read(ftlpu::hw::kMemLocalWordAddressCount, 0));
        },
        "MEM codec should reject addresses outside the 13-bit local word range");
}

bool same_sxm(
    const ftlpu::SxmInstruction& lhs,
    const ftlpu::SxmInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && lhs.shift_source == rhs.shift_source
        && lhs.shift_distance == rhs.shift_distance
        && lhs.lane_map == rhs.lane_map
        && lhs.permute_map == rhs.permute_map
        && lhs.src_streams == rhs.src_streams
        && lhs.dst_streams == rhs.dst_streams;
}

ftlpu::SxmInstruction::StreamList stream_range(
    std::size_t first,
    std::size_t count)
{
    auto result = ftlpu::SxmInstruction::StreamList{};
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(ftlpu::SxmStreamId{first + index});
    }
    return result;
}

ftlpu::SxmInstruction::PermuteMap block_diagonal_map(
    std::size_t diagonal)
{
    auto map = ftlpu::SxmInstruction::PermuteMap{};
    for (std::size_t destination = 0;
         destination < ftlpu::hw::kTileRows;
         ++destination) {
        const auto source =
            (diagonal + ftlpu::hw::kTileRows - destination)
            % ftlpu::hw::kTileRows;
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            map[destination * ftlpu::hw::kLanesPerTile + lane] =
                source * ftlpu::hw::kLanesPerTile + lane;
        }
    }
    return map;
}

bool same_icu(const ftlpu::IcuControlInstruction& lhs, const ftlpu::IcuControlInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && lhs.count == rhs.count
        && lhs.interval == rhs.interval
        && lhs.address_stride == rhs.address_stride
        && lhs.source_stream == rhs.source_stream;
}

bool verify_mxm_codec()
{
    const ftlpu::MxmControlInstruction instructions[] {
        ftlpu::MxmControlInstruction::IW(1),
        ftlpu::MxmControlInstruction::LoadScales(1),
        ftlpu::MxmControlInstruction::Compute(1, 31, 36),
        ftlpu::MxmControlInstruction::ComputeToAccumulator(0, 1, 17, 24, true, false, true),
    };

    for (const auto& instruction : instructions) {
        const auto encoded = ftlpu::isa::encode_mxm_instruction(instruction);
        const auto decoded = ftlpu::isa::decode_mxm_instruction(encoded);
        if (!require(same_mxm(instruction, decoded), "MXM instruction codec round-trip failed")) {
            return false;
        }
    }

    return require_throws(
        [] {
            ftlpu::isa::encode_mxm_instruction(ftlpu::MxmControlInstruction::IW(32));
        },
        "MXM codec should reject weight buffers outside the two-buffer set");
}

bool verify_vxm_codec()
{
    auto instruction = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::StreamFloat32(0.5f),
        ftlpu::VxmLaneOperand::StreamFloat16(2.0f),
    };
    instruction.precision = ftlpu::VxmAluPrecision::Float32;
    instruction.output_type = ftlpu::VxmCastTarget::Float16;
    instruction.output_scale = 0.25f;
    instruction.output_stream = 8;
    instruction.repeat_count = 37;
    instruction.input_hemisphere = ftlpu::Hemisphere::West;
    instruction.output_hemisphere = ftlpu::Hemisphere::East;

    const auto encoded = ftlpu::isa::encode_vxm_instruction(instruction);
    const auto decoded = ftlpu::isa::decode_vxm_instruction(encoded);
    if (!require(same_vxm(instruction, decoded), "VXM instruction codec round-trip failed")) {
        return false;
    }

    auto special = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmSpecialAluOpcode::Rsqrt,
        ftlpu::VxmLaneOperand::Previous(),
    };
    special.accumulator_emit = false;
    const auto decoded_special =
        ftlpu::isa::decode_vxm_instruction(
            ftlpu::isa::encode_vxm_instruction(
                special));
    if (!require(
            same_vxm(special, decoded_special),
            "VXM special-ALU codec round-trip failed")) {
        return false;
    }

    return require_throws(
        [] {
            auto invalid = ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::Previous(),
            };
            invalid.output_stream = 6;
            ftlpu::isa::encode_vxm_instruction(invalid);
        },
        "VXM codec should reject non-physical output stream groups")
        && require_throws(
            [] {
                auto invalid = ftlpu::VxmLaneAluInstruction {
                    ftlpu::VxmAluOpcode::Bypass,
                    ftlpu::VxmLaneOperand::StreamInt32(
                        1.0f, 1),
                };
                ftlpu::isa::encode_vxm_instruction(invalid);
            },
            "VXM codec should reject asymmetric operand zero points")
        && require_throws(
            [] {
                auto invalid = ftlpu::VxmLaneAluInstruction {
                    ftlpu::VxmAluOpcode::Bypass,
                    ftlpu::VxmLaneOperand::Previous(),
                };
                invalid.output_zero_point = 1;
                ftlpu::isa::encode_vxm_instruction(invalid);
            },
            "VXM codec should reject asymmetric output zero points")
        && require_throws(
            [] {
                auto invalid =
                    ftlpu::VxmLaneAluInstruction{};
                invalid.repeat_count = 2049;
                ftlpu::isa::encode_vxm_instruction(
                    invalid);
            },
            "VXM codec should reject repeat counts above 2048");
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

    const auto repeat = ftlpu::IcuRepeat {7, 3, -16};
    const auto decoded = ftlpu::isa::decode_icu_repeat(ftlpu::isa::encode_icu_repeat(repeat));
    if (!require(
            decoded.count == repeat.count
                && decoded.interval == repeat.interval
                && decoded.address_stride == repeat.address_stride,
            "ICU Repeat codec round-trip failed")) {
        return false;
    }

    const ftlpu::IcuControlInstruction controls[] {
        ftlpu::IcuControlInstruction::Fetch(ftlpu::StreamId::West(17)),
        ftlpu::IcuControlInstruction::Nop(65535),
        ftlpu::IcuControlInstruction::Repeat(31, 7, -32),
        ftlpu::IcuControlInstruction::Sync(),
        ftlpu::IcuControlInstruction::Notify(),
    };
    for (const auto& control : controls) {
        const auto decoded_control = ftlpu::isa::decode_icu_control_instruction(
            ftlpu::isa::encode_icu_control_instruction(control));
        if (!require(same_icu(control, decoded_control), "ICU control round-trip failed")) {
            return false;
        }
    }

    auto vxm_routed = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::StreamFloat32(),
        ftlpu::VxmLaneOperand::StreamFloat32(),
    };
    vxm_routed.output_stream = 12;
    vxm_routed.input_hemisphere =
        ftlpu::Hemisphere::West;
    vxm_routed.output_hemisphere =
        ftlpu::Hemisphere::East;
    const auto routed_packet =
        ftlpu::isa::encode_packet(vxm_routed);
    if (!require(
            ftlpu::isa::packet_kind(routed_packet)
                    == ftlpu::isa::InstructionPacketKind::Vxm
                && same_vxm(
                    vxm_routed,
                    ftlpu::isa::decode_vxm_packet(
                        routed_packet)),
            "VXM packet/hemisphere round-trip failed")) {
        return false;
    }

    const ftlpu::MemInstruction mem_extended[] {
        ftlpu::MemInstruction::ReadWrite(
            17,
            ftlpu::StreamId::East(3),
            ftlpu::hw::kMemLocalWordAddressCount - 1,
            ftlpu::StreamId::West(31)),
        ftlpu::MemInstruction::Accumulate(
            29,
            ftlpu::StreamId::West(8),
            ftlpu::MemAccumulatorDestination::Sram),
        ftlpu::MemInstruction::Accumulate(
            30,
            ftlpu::StreamId::West(12),
            ftlpu::MemAccumulatorDestination::Stream),
    };
    for (const auto& instruction : mem_extended) {
        const auto packet = ftlpu::isa::encode_packet(instruction);
        if (!require(
                packet.bytes[1] == 12
                    && same_mem(
                        instruction,
                        ftlpu::isa::decode_mem_packet(packet)),
                "extended MEM packet round-trip failed")) {
            return false;
        }
    }

    return require_throws(
        [] {
            ftlpu::isa::encode_icu_repeat(ftlpu::IcuRepeat {1, 1, 2048});
        },
        "ICU Repeat codec should reject strides wider than signed 12 bits")
        && require_throws(
            [] { (void)ftlpu::isa::encode_icu_nop(65536); },
            "ICU NOP codec should reject counts wider than 16 bits");
}

bool verify_sxm_packet_codec()
{
    auto lane_map = ftlpu::SxmInstruction::LaneMap{};
    for (std::size_t lane = 0; lane < lane_map.size(); ++lane) {
        lane_map[lane] = lane_map.size() - 1 - lane;
    }
    const ftlpu::SxmInstruction instructions[] {
        ftlpu::SxmInstruction::Distribute(
            {3}, {ftlpu::StreamId::West(7).packed()}, lane_map),
        ftlpu::SxmInstruction::Transpose(
            stream_range(0, ftlpu::hw::kLanesPerTile),
            stream_range(16, ftlpu::hw::kLanesPerTile)),
        ftlpu::SxmInstruction::ShiftSelect(
            stream_range(4, ftlpu::hw::kTileRows),
            stream_range(24, ftlpu::hw::kTileRows),
            ftlpu::SxmShiftSource::NorthShifted,
            3),
        ftlpu::SxmInstruction::Permute(
            stream_range(0, ftlpu::hw::kTileRows),
            stream_range(
                ftlpu::StreamId::West(0).packed(),
                ftlpu::hw::kTileRows),
            block_diagonal_map(3)),
    };
    for (const auto& instruction : instructions) {
        const auto packet = ftlpu::isa::encode_packet(instruction);
        if (!require(
                ftlpu::isa::packet_kind(packet)
                        == ftlpu::isa::InstructionPacketKind::Sxm
                    && same_sxm(
                        instruction,
                        ftlpu::isa::decode_sxm_packet(packet)),
                "SXM compact packet round-trip failed")) {
            return false;
        }
    }
    return true;
}

bool verify_instruction_packets()
{
    const auto mem = ftlpu::MemInstruction::Read(123, ftlpu::StreamId::West(7));
    const auto mxm = ftlpu::MxmControlInstruction::Compute(1, 3, 8);
    auto vxm = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::StreamInt32(4),
        ftlpu::VxmLaneOperand::Imm(2.0f),
    };
    const auto control = ftlpu::IcuControlInstruction::Fetch(ftlpu::StreamId::East(11));

    const auto mem_packet = ftlpu::isa::encode_packet(mem);
    const auto mxm_packet = ftlpu::isa::encode_packet(mxm);
    const auto vxm_packet = ftlpu::isa::encode_packet(vxm);
    const auto control_packet = ftlpu::isa::encode_packet(control);
    if (!require(same_mem(mem, ftlpu::isa::decode_mem_packet(mem_packet)), "MEM packet round-trip failed")
        || !require(same_mxm(mxm, ftlpu::isa::decode_mxm_packet(mxm_packet)), "MXM packet round-trip failed")
        || !require(same_vxm(vxm, ftlpu::isa::decode_vxm_packet(vxm_packet)), "VXM packet round-trip failed")
        || !require(same_icu(control, ftlpu::isa::decode_icu_packet(control_packet)), "ICU packet round-trip failed")) {
        return false;
    }
    for (std::size_t byte = 8; byte < mem_packet.bytes.size(); ++byte) {
        if (!require(mem_packet.bytes[byte] == 0, "short instruction packet padding is not zero")) {
            return false;
        }
    }
    return true;
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
    if (!verify_sxm_packet_codec()) {
        return 1;
    }
    if (!verify_instruction_packets()) {
        return 1;
    }
    return 0;
}
catch (const std::exception& ex) {
    std::cerr << "instruction_codec_test failed: " << ex.what() << '\n';
    return 1;
}
