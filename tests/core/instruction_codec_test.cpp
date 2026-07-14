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
        && lhs.map_stream == rhs.map_stream;
}

bool same_mxm(const ftlpu::MxmControlInstruction& lhs, const ftlpu::MxmControlInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && lhs.weight_buffer == rhs.weight_buffer
        && lhs.stream_base == rhs.stream_base
        && lhs.activation_stream_base == rhs.activation_stream_base;
}

bool same_operand(const ftlpu::VxmLaneOperand& lhs, const ftlpu::VxmLaneOperand& rhs)
{
    return lhs.kind == rhs.kind
        && lhs.index == rhs.index
        && lhs.immediate == rhs.immediate
        && lhs.scale == rhs.scale;
}

bool same_vxm(const ftlpu::VxmLaneAluInstruction& lhs, const ftlpu::VxmLaneAluInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && same_operand(lhs.lhs, rhs.lhs)
        && same_operand(lhs.rhs, rhs.rhs)
        && lhs.scale == rhs.scale
        && lhs.output_zero_point == rhs.output_zero_point
        && lhs.cast_target == rhs.cast_target
        && lhs.output_stream == rhs.output_stream;
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
        ftlpu::MemInstruction::Write((ftlpu::hw::kSramDepthWords - 1) << 4, 63),
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
            ftlpu::isa::encode_mem_instruction(ftlpu::MemInstruction::Read(17, 0));
        },
        "MEM codec should reject unaligned SRAM byte addresses");
}

bool verify_mxm_codec()
{
    const ftlpu::MxmControlInstruction instructions[] {
        ftlpu::MxmControlInstruction::IW(1),
        ftlpu::MxmControlInstruction::Compute(1, 31, 36),
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
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::StreamInt32(32),
        ftlpu::VxmLaneOperand::Alu(13),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float32,
    };
    instruction.output_stream = 31;

    const auto encoded = ftlpu::isa::encode_vxm_instruction(instruction);
    const auto decoded = ftlpu::isa::decode_vxm_instruction(encoded);
    if (!require(same_vxm(instruction, decoded), "VXM instruction codec round-trip failed")) {
        return false;
    }

    auto cast = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::Alu(14),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Int8,
        9,
    };
    const auto decoded_cast = ftlpu::isa::decode_vxm_instruction(ftlpu::isa::encode_vxm_instruction(cast));
    if (!require(same_vxm(cast, decoded_cast), "VXM cast/output codec round-trip failed")) {
        return false;
    }

    return require_throws(
        [] {
            auto invalid = ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::Alu(16),
            };
            ftlpu::isa::encode_vxm_instruction(invalid);
        },
        "VXM codec should reject ALU indexes outside the 16-ALU lane")
        && require_throws(
            [] {
                auto invalid = ftlpu::VxmLaneAluInstruction {
                    ftlpu::VxmAluOpcode::Pass,
                    ftlpu::VxmLaneOperand::StreamInt32(61),
                };
                ftlpu::isa::encode_vxm_instruction(invalid);
            },
            "VXM codec should reject int32 stream operands that cross the 64-stream boundary")
        && require_throws(
            [] {
                auto invalid = ftlpu::VxmLaneAluInstruction {
                    ftlpu::VxmAluOpcode::Pass,
                    ftlpu::VxmLaneOperand::Alu(0),
                };
                invalid.output_zero_point = 1;
                ftlpu::isa::encode_vxm_instruction(invalid);
            },
            "VXM codec should reject model-only output zero point metadata");
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
    return 0;
}
catch (const std::exception& ex) {
    std::cerr << "instruction_codec_test failed: " << ex.what() << '\n';
    return 1;
}
