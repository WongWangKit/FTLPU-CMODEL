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
        && lhs.weight_column == rhs.weight_column;
}

bool same_vxm(const ftlpu::VxmLaneAluInstruction& lhs, const ftlpu::VxmLaneAluInstruction& rhs)
{
    return lhs.opcode == rhs.opcode
        && lhs.lhs.kind == rhs.lhs.kind && lhs.lhs.index == rhs.lhs.index
        && lhs.rhs.kind == rhs.rhs.kind && lhs.rhs.immediate == rhs.rhs.immediate
        && lhs.cast_target == rhs.cast_target
        && lhs.output_stream == rhs.output_stream
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
        ftlpu::MemInstruction::Write(ftlpu::hw::kSramDepthRows - 1, 63),
        ftlpu::MemInstruction::ReadWrite(4095, 7, 4096, 55),
        ftlpu::MemInstruction::Gather(7, 55),
        ftlpu::MemInstruction::Scatter(36, 12),
        ftlpu::MemInstruction::Accumulate(
            6143,
            ftlpu::StreamId::West(28),
            ftlpu::MemAccumulatorDestination::Stream),
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
                ftlpu::MemInstruction::Read(ftlpu::hw::kSramDepthRows, 0));
        },
        "MEM codec should reject row addresses outside the 8192-row bank")) {
        return false;
    }
    return require_throws(
        [] {
            const auto encoded_read = ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Read(0, 0));
            ftlpu::isa::decode_mem_instruction(encoded_read | (1u << 28));
        },
        "MEM codec should reject an accumulator destination on Read");
}

bool verify_mxm_codec()
{
    const ftlpu::MxmControlInstruction instructions[] {
        ftlpu::MxmControlInstruction::IW(1, 3),
        ftlpu::MxmControlInstruction::Compute(1, 30, 20),
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
        ftlpu::VxmLaneOperand::StreamInt8(32),
        ftlpu::VxmLaneOperand::Imm(0.125f),
        1.0f, 0, ftlpu::VxmCastTarget::Float16, 16,
        ftlpu::Hemisphere::West, ftlpu::Hemisphere::East};

    const auto encoded = ftlpu::isa::encode_vxm_instruction(instruction);
    const auto decoded = ftlpu::isa::decode_vxm_instruction(encoded);
    if (!require(same_vxm(instruction, decoded), "VXM ALU instruction codec round-trip failed")) {
        return false;
    }

    return require_throws(
        [] {
            auto invalid = ftlpu::VxmLaneAluInstruction {};
            invalid.lhs = ftlpu::VxmLaneOperand::StreamInt8(64);
            ftlpu::isa::encode_vxm_instruction(invalid);
        },
        "VXM codec should reject stream indexes outside the stream set");
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
