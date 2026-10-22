#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kVxmCycle = 8;
constexpr std::size_t kSourceAddress = 0;
constexpr std::size_t kOutputAddress = 1;
constexpr std::int8_t kAddend = 5;
using Record = std::array<std::uint32_t, 15>;

std::int8_t input_value(std::size_t tile, std::size_t lane)
{
    return static_cast<std::int8_t>(
        static_cast<int>(tile * ftlpu::hw::kLanesPerTile + lane) - 16);
}

std::uint64_t read_word(
    const ftlpu::TspSliceSystem& system,
    std::size_t address,
    std::size_t tile)
{
    std::uint64_t result = 0;
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        result |= static_cast<std::uint64_t>(
            system.read_mem_sram_lane_byte(0, tile, address, lane))
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
    if (!output) throw std::runtime_error("failed to open VXM vector output");
    output << std::hex << std::setfill('0');
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile)
        output << std::setw(16) << read_word(system, address, tile) << '\n';
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
        std::cerr << "usage: vxm_int8_add <init.hex> <golden.hex> <schedule.hex>\n";
        return 2;
    }

    auto system = ftlpu::TspSliceSystem {};
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            system.initialize_mem_sram_lane_byte(
                0, tile, kSourceAddress, lane,
                static_cast<std::uint8_t>(input_value(tile, lane)));
        }
    }

    const auto read =
        ftlpu::MemInstruction::Read(kSourceAddress, ftlpu::StreamId::West(0));
    const auto write =
        ftlpu::MemInstruction::Write(kOutputAddress, ftlpu::StreamId::East(0));
    const auto instruction = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::StreamInt8(32),
        ftlpu::VxmLaneOperand::Imm(static_cast<float>(kAddend)),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Int8,
        0,
        ftlpu::Hemisphere::East,
        ftlpu::Hemisphere::East,
    };

    write_vectors(argv[1], system, kSourceAddress);
    system.icu().enqueue_mem_nop(0, kVxmCycle - 2);
    system.icu().enqueue_mem(0, read);
    system.icu().enqueue_mem_nop(0, 2);
    system.icu().enqueue_mem(0, write);
    system.icu().enqueue_vxm_nop(0, kVxmCycle);
    system.icu().enqueue_vxm(0, instruction);

    for (std::size_t cycle = 0; cycle < 20; ++cycle)
        system.tick(ftlpu::TspSliceSystem::LogSinks {});

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto expected = static_cast<std::uint8_t>(
                static_cast<std::int8_t>(input_value(tile, lane) + kAddend));
            const auto actual = system.read_mem_sram_lane_byte(
                0, tile, kOutputAddress, lane);
            if (actual != expected)
                throw std::runtime_error("C model VXM INT8 Add mismatch");
        }
    }
    write_vectors(argv[2], system, kOutputAddress);

    std::ofstream schedule(argv[3], std::ios::trunc);
    if (!schedule) throw std::runtime_error("failed to open VXM schedule output");
    schedule << std::hex << std::setfill('0');
    write_record(schedule, command_record(
        0, ftlpu::isa::encode_icu_nop(kVxmCycle - 2)));
    write_record(schedule, mem_record(0, read));
    write_record(schedule, command_record(0, ftlpu::isa::encode_icu_nop(2)));
    write_record(schedule, mem_record(0, write));
    write_record(schedule, command_record(
        112, ftlpu::isa::encode_icu_nop(kVxmCycle)));
    write_record(schedule, vxm_record(
        112, ftlpu::isa::encode_vxm_instruction(instruction)));

    std::cout << "C model MEM -> VXM INT8 Add -> MEM golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
