#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/fp16.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kVxmCycle = 8;
constexpr std::size_t kSourceAddress = 0;
constexpr std::size_t kFp16OutputAddress = 1;
constexpr std::size_t kBf16OutputAddress = 2;
constexpr std::size_t kFp32OutputAddress = 3;
using Record = std::array<std::uint32_t, 15>;

float input_value(std::size_t tile, std::size_t lane)
{
    return 0.5f * static_cast<float>(
        static_cast<int>(tile * ftlpu::hw::kLanesPerTile + lane) - 16);
}

std::uint64_t read_word(
    const ftlpu::TspSliceSystem& system,
    std::size_t column,
    std::size_t address,
    std::size_t tile)
{
    std::uint64_t result = 0;
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        result |= static_cast<std::uint64_t>(
            system.read_mem_sram_lane_byte(column, tile, address, lane))
            << (lane * 8);
    }
    return result;
}

void write_vectors(
    const char* path,
    const ftlpu::TspSliceSystem& system,
    std::size_t address)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open VXM FP16 vector output");
    output << std::hex << std::setfill('0');
    for (std::size_t column = 0; column < 8; ++column)
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile)
            output << std::setw(16)
                   << read_word(system, column, address, tile) << '\n';
}

void write_golden(const char* path, const ftlpu::TspSliceSystem& system)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open VXM floating golden output");
    output << std::hex << std::setfill('0');
    for (std::size_t column = 0; column < 8; ++column) {
        const auto address = column < 2 ? kFp16OutputAddress
            : (column < 4 ? kBf16OutputAddress : kFp32OutputAddress);
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile)
            output << std::setw(16)
                   << read_word(system, column, address, tile) << '\n';
    }
}

void set_record_bits(
    Record& record,
    std::size_t offset,
    std::uint64_t value,
    std::size_t width)
{
    for (std::size_t bit = 0; bit < width; ++bit) {
        if (((value >> bit) & 1u) != 0)
            record[(offset + bit) / 32] |=
                std::uint32_t {1} << ((offset + bit) % 32);
    }
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
    set_record_bits(
        result, 41, ftlpu::isa::encode_mem_instruction(instruction), 47);
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

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4) {
        std::cerr << "usage: vxm_fp16_relu <init.hex> <golden.hex> <schedule.hex>\n";
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

    const auto read_low =
        ftlpu::MemInstruction::Read(kSourceAddress, ftlpu::StreamId::West(0));
    const auto read_high =
        ftlpu::MemInstruction::Read(kSourceAddress, ftlpu::StreamId::West(1));
    const auto fp16_relu = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Relu,
        ftlpu::VxmLaneOperand::StreamFloat16(32),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float16,
        0,
        ftlpu::Hemisphere::East,
        ftlpu::Hemisphere::East,
    };
    const auto bf16_cast = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::Alu(0),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::BFloat16,
        2,
        ftlpu::Hemisphere::East,
        ftlpu::Hemisphere::East,
    };
    const auto fp32_cast = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::Alu(1),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float32,
        4,
        ftlpu::Hemisphere::East,
        ftlpu::Hemisphere::East,
    };

    write_vectors(argv[1], system, kSourceAddress);
    for (std::size_t column = 0; column < 8; ++column) {
        if (column < 2) {
            const auto read = column == 0 ? read_low : read_high;
            system.icu().enqueue_mem_nop(column, kVxmCycle - 2);
            system.icu().enqueue_mem(column, read);
            system.icu().enqueue_mem_nop(column, 2);
        } else {
            const auto write_cycle = column < 4 ? kVxmCycle + 2
                                                 : kVxmCycle + 4;
            system.icu().enqueue_mem_nop(column, write_cycle);
        }
        const auto address = column < 2 ? kFp16OutputAddress
            : (column < 4 ? kBf16OutputAddress : kFp32OutputAddress);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Write(
                address, ftlpu::StreamId::East(column)));
    }
    system.icu().enqueue_vxm_nop(0, kVxmCycle);
    system.icu().enqueue_vxm(0, fp16_relu);
    system.icu().enqueue_vxm_nop(1, kVxmCycle + 1);
    system.icu().enqueue_vxm(1, bf16_cast);
    system.icu().enqueue_vxm_nop(2, kVxmCycle + 2);
    system.icu().enqueue_vxm(2, fp32_cast);

    for (std::size_t cycle = 0; cycle < 28; ++cycle)
        system.tick(ftlpu::TspSliceSystem::LogSinks {});

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto relu = std::max(0.0f, input_value(tile, lane));
            const auto fp16_expected = ftlpu::Fp16::from_float(relu).bits();
            const auto bf16_expected = ftlpu::Bf16::from_float(relu).bits();
            const auto rounded = ftlpu::Bf16::from_bits(bf16_expected).to_float();
            const auto fp32_expected = std::bit_cast<std::uint32_t>(rounded);
            const auto fp16_actual = static_cast<std::uint16_t>(
                system.read_mem_sram_lane_byte(
                    0, tile, kFp16OutputAddress, lane)) |
                (static_cast<std::uint16_t>(system.read_mem_sram_lane_byte(
                    1, tile, kFp16OutputAddress, lane)) << 8);
            const auto bf16_actual = static_cast<std::uint16_t>(
                system.read_mem_sram_lane_byte(
                    2, tile, kBf16OutputAddress, lane)) |
                (static_cast<std::uint16_t>(system.read_mem_sram_lane_byte(
                    3, tile, kBf16OutputAddress, lane)) << 8);
            std::uint32_t fp32_actual = 0;
            for (std::size_t byte = 0; byte < 4; ++byte)
                fp32_actual |= static_cast<std::uint32_t>(
                    system.read_mem_sram_lane_byte(
                        4 + byte, tile, kFp32OutputAddress, lane)) << (byte * 8);
            if (fp16_actual != fp16_expected || bf16_actual != bf16_expected
                || fp32_actual != fp32_expected)
                throw std::runtime_error("C model VXM floating format mismatch");
        }
    }
    write_golden(argv[2], system);

    std::ofstream schedule(argv[3], std::ios::trunc);
    if (!schedule) throw std::runtime_error("failed to open VXM floating schedule output");
    schedule << std::hex << std::setfill('0');
    for (std::size_t column = 0; column < 8; ++column) {
        if (column < 2) {
            write_record(schedule, command_record(
                static_cast<std::uint8_t>(column),
                ftlpu::isa::encode_icu_nop(kVxmCycle - 2)));
            write_record(schedule, mem_record(
                static_cast<std::uint8_t>(column),
                column == 0 ? read_low : read_high));
            write_record(schedule, command_record(
                static_cast<std::uint8_t>(column),
                ftlpu::isa::encode_icu_nop(2)));
        } else {
            const auto write_cycle = column < 4 ? kVxmCycle + 2
                                                 : kVxmCycle + 4;
            write_record(schedule, command_record(
                static_cast<std::uint8_t>(column),
                ftlpu::isa::encode_icu_nop(write_cycle)));
        }
        const auto address = column < 2 ? kFp16OutputAddress
            : (column < 4 ? kBf16OutputAddress : kFp32OutputAddress);
        write_record(schedule, mem_record(
            static_cast<std::uint8_t>(column),
            ftlpu::MemInstruction::Write(
                address, ftlpu::StreamId::East(column))));
    }
    write_record(schedule, command_record(112, ftlpu::isa::encode_icu_nop(kVxmCycle)));
    write_record(schedule, vxm_record(112, ftlpu::isa::encode_vxm_instruction(fp16_relu)));
    write_record(schedule, command_record(113, ftlpu::isa::encode_icu_nop(kVxmCycle + 1)));
    write_record(schedule, vxm_record(113, ftlpu::isa::encode_vxm_instruction(bf16_cast)));
    write_record(schedule, command_record(114, ftlpu::isa::encode_icu_nop(kVxmCycle + 2)));
    write_record(schedule, vxm_record(114, ftlpu::isa::encode_vxm_instruction(fp32_cast)));

    std::cout << "C model VXM FP16/BF16/FP32 chain golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
