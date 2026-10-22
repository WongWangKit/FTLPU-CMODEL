#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/fp16.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace {

constexpr std::size_t kVxmCycle = 8;
constexpr std::size_t kSwigluCycle = 12;
constexpr std::size_t kSourceAddress = 0;
constexpr float kAddend = 0.1f;
constexpr float kSubtrahend = -0.35f;
constexpr float kMultiplier = 1.3f;
constexpr float kDivisor = 1.7f;
using Record = std::array<std::uint32_t, 15>;

float input_value(std::size_t tile, std::size_t lane)
{
    return 0.25f * static_cast<float>(
        static_cast<int>(tile * ftlpu::hw::kLanesPerTile + lane) - 15);
}

std::uint64_t read_word(
    const ftlpu::TspSliceSystem& system,
    std::size_t column,
    std::size_t address,
    std::size_t tile)
{
    std::uint64_t result = 0;
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane)
        result |= static_cast<std::uint64_t>(
            system.read_mem_sram_lane_byte(column, tile, address, lane))
            << (lane * 8);
    return result;
}

void write_init(const char* path, const ftlpu::TspSliceSystem& system)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open VXM arithmetic init");
    output << std::hex << std::setfill('0');
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile)
            output << std::setw(16)
                   << read_word(system, column, kSourceAddress, tile) << '\n';
}

void write_golden(const char* path, const ftlpu::TspSliceSystem& system)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open VXM arithmetic golden");
    output << std::hex << std::setfill('0');
    for (std::size_t address = 1; address <= 6; ++address)
        for (std::size_t column = 0;
             column < (address >= 5 ? 2 : 4);
             ++column)
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile)
                output << std::setw(16)
                       << read_word(system, column, address, tile) << '\n';
}

void set_record_bits(
    Record& record,
    std::size_t offset,
    std::uint64_t value,
    std::size_t width)
{
    for (std::size_t bit = 0; bit < width; ++bit)
        if (((value >> bit) & 1u) != 0)
            record[(offset + bit) / 32] |=
                std::uint32_t {1} << ((offset + bit) % 32);
}

Record command_record(std::uint8_t queue, std::uint32_t command)
{
    Record result {};
    set_record_bits(result, 0, queue, 8);
    set_record_bits(result, 9, command, 32);
    return result;
}

Record mem_record(std::uint8_t queue, const ftlpu::MemInstruction& instruction)
{
    Record result {};
    set_record_bits(result, 0, queue, 8);
    set_record_bits(result, 8, 1, 1);
    set_record_bits(result, 41, ftlpu::isa::encode_mem_instruction(instruction), 47);
    return result;
}

Record vxm_record(
    std::uint8_t queue,
    const ftlpu::isa::EncodedVxmInstruction& instruction)
{
    Record result {};
    set_record_bits(result, 0, queue, 8);
    set_record_bits(result, 8, 1, 1);
    for (std::size_t word = 0; word < instruction.words.size(); ++word)
        set_record_bits(result, 41 + word * 32, instruction.words[word], 32);
    return result;
}

void write_record(std::ofstream& output, const Record& record)
{
    for (std::size_t word = record.size(); word-- > 0;)
        output << std::setw(8) << record[word];
    output << '\n';
}

ftlpu::VxmLaneAluInstruction arithmetic_instruction(
    ftlpu::VxmAluOpcode opcode,
    ftlpu::VxmLaneOperand lhs,
    float rhs)
{
    return ftlpu::VxmLaneAluInstruction {
        opcode,
        lhs,
        ftlpu::VxmLaneOperand::Imm(rhs),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float32,
        0,
        ftlpu::Hemisphere::East,
        ftlpu::Hemisphere::East,
    };
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4) {
        std::cerr << "usage: vxm_fp32_arithmetic <init.hex> <golden.hex> <schedule.hex>\n";
        return 2;
    }

    auto system = ftlpu::TspSliceSystem {};
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto bits = ftlpu::Fp16::from_float(input_value(tile, lane)).bits();
            system.initialize_mem_sram_lane_byte(
                0, tile, kSourceAddress, lane,
                static_cast<std::uint8_t>(bits & 0xffu));
            system.initialize_mem_sram_lane_byte(
                1, tile, kSourceAddress, lane,
                static_cast<std::uint8_t>((bits >> 8) & 0xffu));
        }
    }

    const auto add = arithmetic_instruction(
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::StreamFloat16(32), kAddend);
    const auto subtract = arithmetic_instruction(
        ftlpu::VxmAluOpcode::Subtract,
        ftlpu::VxmLaneOperand::Alu(0), kSubtrahend);
    const auto multiply = arithmetic_instruction(
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Alu(1), kMultiplier);
    const auto divide = arithmetic_instruction(
        ftlpu::VxmAluOpcode::Divide,
        ftlpu::VxmLaneOperand::Alu(2), kDivisor);
    const auto exponential = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Exp,
        ftlpu::VxmLaneOperand::Alu(3),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::BFloat16,
        0,
        ftlpu::Hemisphere::East,
        ftlpu::Hemisphere::East,
    };

    write_init(argv[1], system);
    for (std::size_t column = 0; column < 4; ++column) {
        if (column < 2) {
            system.icu().enqueue_mem_nop(column, kVxmCycle - 2);
            system.icu().enqueue_mem(column, ftlpu::MemInstruction::Read(
                kSourceAddress, ftlpu::StreamId::West(column)));
            system.icu().enqueue_mem_nop(column, 2);
        } else {
            system.icu().enqueue_mem_nop(column, kVxmCycle + 1);
        }
        for (std::size_t address = 1; address <= 4; ++address)
            system.icu().enqueue_mem(column, ftlpu::MemInstruction::Write(
                address, ftlpu::StreamId::East(column)));
        if (column < 2)
            system.icu().enqueue_mem(column, ftlpu::MemInstruction::Write(
                5, ftlpu::StreamId::East(column)));
    }
    const std::array instructions {
        add, subtract, multiply, divide, exponential};
    const auto feedback_instruction = [](
        ftlpu::VxmAluOpcode opcode,
        ftlpu::VxmLaneOperand lhs,
        ftlpu::VxmLaneOperand rhs,
        ftlpu::VxmCastTarget cast = ftlpu::VxmCastTarget::Float32,
        std::optional<std::size_t> output = std::nullopt) {
        return ftlpu::VxmLaneAluInstruction {
            opcode, lhs, rhs, 1.0f, 0, cast, output,
            ftlpu::Hemisphere::East, ftlpu::Hemisphere::East};
    };
    struct TimedInstruction {
        std::size_t alu;
        std::size_t cycle;
        ftlpu::VxmLaneAluInstruction instruction;
    };
    const std::array swiglu {
        TimedInstruction {0, kSwigluCycle, feedback_instruction(
            ftlpu::VxmAluOpcode::Negate,
            ftlpu::VxmLaneOperand::Alu(3), ftlpu::VxmLaneOperand::Imm(0.0f))},
        TimedInstruction {1, kSwigluCycle, feedback_instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(3), ftlpu::VxmLaneOperand::Alu(2))},
        TimedInstruction {2, kSwigluCycle + 1, feedback_instruction(
            ftlpu::VxmAluOpcode::Exp,
            ftlpu::VxmLaneOperand::Alu(0), ftlpu::VxmLaneOperand::Imm(0.0f))},
        TimedInstruction {5, kSwigluCycle + 1, feedback_instruction(
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(1), ftlpu::VxmLaneOperand::Imm(0.0f))},
        TimedInstruction {3, kSwigluCycle + 2, feedback_instruction(
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(2), ftlpu::VxmLaneOperand::Imm(1.0f))},
        TimedInstruction {6, kSwigluCycle + 2, feedback_instruction(
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(5), ftlpu::VxmLaneOperand::Imm(0.0f))},
        TimedInstruction {4, kSwigluCycle + 3, feedback_instruction(
            ftlpu::VxmAluOpcode::Divide,
            ftlpu::VxmLaneOperand::Imm(1.0f), ftlpu::VxmLaneOperand::Alu(3))},
        TimedInstruction {7, kSwigluCycle + 3, feedback_instruction(
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(6), ftlpu::VxmLaneOperand::Imm(0.0f))},
        TimedInstruction {8, kSwigluCycle + 4, feedback_instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(7), ftlpu::VxmLaneOperand::Alu(4))},
        TimedInstruction {9, kSwigluCycle + 5, feedback_instruction(
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(8), ftlpu::VxmLaneOperand::Imm(0.0f),
            ftlpu::VxmCastTarget::BFloat16, 0)},
    };
    for (std::size_t alu = 0; alu < instructions.size(); ++alu) {
        system.icu().enqueue_vxm_nop(alu, kVxmCycle + alu);
        system.icu().enqueue_vxm(alu, instructions[alu]);
    }
    auto vxm_cursor = std::array<std::size_t, 16> {};
    for (std::size_t alu = 0; alu < instructions.size(); ++alu)
        vxm_cursor[alu] = kVxmCycle + alu + 1;
    for (const auto& timed : swiglu) {
        system.icu().enqueue_vxm_nop(
            timed.alu, timed.cycle - vxm_cursor[timed.alu]);
        system.icu().enqueue_vxm(timed.alu, timed.instruction);
        vxm_cursor[timed.alu] = timed.cycle + 1;
    }
    for (std::size_t column = 0; column < 2; ++column) {
        system.icu().enqueue_mem_nop(column, 4);
        system.icu().enqueue_mem(column, ftlpu::MemInstruction::Write(
            6, ftlpu::StreamId::East(column)));
    }

    for (std::size_t cycle = 0; cycle < 32; ++cycle)
        system.tick(ftlpu::TspSliceSystem::LogSinks {});

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const float expected[4] {
                input_value(tile, lane) + kAddend,
                (input_value(tile, lane) + kAddend) - kSubtrahend,
                ((input_value(tile, lane) + kAddend) - kSubtrahend) * kMultiplier,
                (((input_value(tile, lane) + kAddend) - kSubtrahend)
                    * kMultiplier) / kDivisor,
            };
            for (std::size_t result = 0; result < 4; ++result) {
                std::uint32_t actual = 0;
                for (std::size_t byte = 0; byte < 4; ++byte)
                    actual |= static_cast<std::uint32_t>(
                        system.read_mem_sram_lane_byte(
                            byte, tile, result + 1, lane)) << (byte * 8);
                if (actual != std::bit_cast<std::uint32_t>(expected[result]))
                    throw std::runtime_error("C model VXM FP32 arithmetic mismatch");
            }
            std::uint16_t actual_exp = 0;
            for (std::size_t byte = 0; byte < 2; ++byte)
                actual_exp |= static_cast<std::uint16_t>(
                    system.read_mem_sram_lane_byte(
                        byte, tile, 5, lane)) << (byte * 8);
            if (actual_exp != ftlpu::Bf16::from_float(
                    std::exp(expected[3])).bits())
                throw std::runtime_error("C model VXM BF16 Exp mismatch");
            std::uint16_t actual_swiglu = 0;
            for (std::size_t byte = 0; byte < 2; ++byte)
                actual_swiglu |= static_cast<std::uint16_t>(
                    system.read_mem_sram_lane_byte(
                        byte, tile, 6, lane)) << (byte * 8);
            const auto swiglu_expected = (expected[3] * expected[2])
                / (1.0f + std::exp(-expected[3]));
            if (actual_swiglu !=
                ftlpu::Bf16::from_float(swiglu_expected).bits())
                throw std::runtime_error("C model VXM BF16 SwiGLU mismatch");
        }
    }
    write_golden(argv[2], system);

    std::ofstream schedule(argv[3], std::ios::trunc);
    if (!schedule) throw std::runtime_error("failed to open VXM arithmetic schedule");
    schedule << std::hex << std::setfill('0');
    for (std::size_t column = 0; column < 4; ++column) {
        if (column < 2) {
            write_record(schedule, command_record(column,
                ftlpu::isa::encode_icu_nop(kVxmCycle - 2)));
            write_record(schedule, mem_record(column, ftlpu::MemInstruction::Read(
                kSourceAddress, ftlpu::StreamId::West(column))));
            write_record(schedule, command_record(
                column, ftlpu::isa::encode_icu_nop(2)));
        } else {
            write_record(schedule, command_record(
                column, ftlpu::isa::encode_icu_nop(kVxmCycle + 1)));
        }
        for (std::size_t address = 1; address <= 4; ++address)
            write_record(schedule, mem_record(column, ftlpu::MemInstruction::Write(
                address, ftlpu::StreamId::East(column))));
        if (column < 2)
            write_record(schedule, mem_record(column, ftlpu::MemInstruction::Write(
                5, ftlpu::StreamId::East(column))));
    }
    for (std::size_t alu = 0; alu < instructions.size(); ++alu) {
        write_record(schedule, command_record(
            static_cast<std::uint8_t>(112 + alu),
            ftlpu::isa::encode_icu_nop(kVxmCycle + alu)));
        write_record(schedule, vxm_record(
            static_cast<std::uint8_t>(112 + alu),
            ftlpu::isa::encode_vxm_instruction(instructions[alu])));
    }
    vxm_cursor = {};
    for (std::size_t alu = 0; alu < instructions.size(); ++alu)
        vxm_cursor[alu] = kVxmCycle + alu + 1;
    for (const auto& timed : swiglu) {
        write_record(schedule, command_record(
            static_cast<std::uint8_t>(112 + timed.alu),
            ftlpu::isa::encode_icu_nop(
                timed.cycle - vxm_cursor[timed.alu])));
        write_record(schedule, vxm_record(
            static_cast<std::uint8_t>(112 + timed.alu),
            ftlpu::isa::encode_vxm_instruction(timed.instruction)));
        vxm_cursor[timed.alu] = timed.cycle + 1;
    }
    for (std::size_t column = 0; column < 2; ++column) {
        write_record(schedule, command_record(
            column, ftlpu::isa::encode_icu_nop(4)));
        write_record(schedule, mem_record(
            column, ftlpu::MemInstruction::Write(
                6, ftlpu::StreamId::East(column))));
    }

    std::cout << "C model VXM arithmetic and BF16 SwiGLU golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
