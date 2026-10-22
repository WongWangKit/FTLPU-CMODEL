#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/core/stream.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kCyclesPerDirection = 16;

std::uint8_t source_byte(std::size_t tile, std::size_t lane)
{
    return static_cast<std::uint8_t>(tile * 0x10 + lane);
}

void initialize_row(
    ftlpu::TspSliceSystem& system,
    std::size_t column,
    std::size_t row)
{
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            system.initialize_mem_sram_lane_byte(
                column, tile, row, lane, source_byte(tile, lane));
        }
    }
}

void check_row(
    const ftlpu::TspSliceSystem& system,
    std::size_t column,
    std::size_t row)
{
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto actual =
                system.read_mem_sram_lane_byte(column, tile, row, lane);
            const auto expected = source_byte(tile, lane);
            if (actual != expected) {
                throw std::runtime_error("C model MEM/stream round trip mismatch");
            }
        }
    }
}

void tick_cycles(ftlpu::TspSliceSystem& system, std::size_t cycles)
{
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        system.tick(ftlpu::TspSliceSystem::LogSinks {});
    }
}

std::uint64_t packed_tile_word(std::size_t tile)
{
    std::uint64_t word = 0;
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        word |= static_cast<std::uint64_t>(source_byte(tile, lane))
            << (lane * 8);
    }
    return word;
}

std::uint64_t instruction_record(
    std::uint8_t queue,
    const ftlpu::MemInstruction& instruction)
{
    return static_cast<std::uint64_t>(queue)
        | (std::uint64_t {1} << 8)
        | (ftlpu::isa::encode_mem_instruction(instruction) << 9);
}

std::uint64_t command_record(std::uint8_t queue, std::uint32_t command)
{
    return static_cast<std::uint64_t>(queue)
        | (static_cast<std::uint64_t>(command) << 9);
}

} // namespace

int main(int argc, char** argv)
try
{
    if (argc != 3) {
        std::cerr << "usage: mem_stream_roundtrip <data.hex> <schedule.hex>\n";
        return 2;
    }

    ftlpu::TspSliceSystem system;

    // East: group 0 -> passive link -> group 2.
    const auto east_read =
        ftlpu::MemInstruction::Read(3, ftlpu::StreamId::East(5));
    const auto east_write =
        ftlpu::MemInstruction::Write(9, ftlpu::StreamId::East(5));
    const auto west_read =
        ftlpu::MemInstruction::Read(4, ftlpu::StreamId::West(6));
    const auto west_write =
        ftlpu::MemInstruction::Write(10, ftlpu::StreamId::West(6));
    const auto nop3 = ftlpu::isa::encode_icu_nop(3);

    initialize_row(system, 0, 3);
    system.icu().enqueue_mem(0, east_read);
    system.icu().enqueue_mem_nop(8, 3);
    system.icu().enqueue_mem(8, east_write);
    tick_cycles(system, kCyclesPerDirection);
    check_row(system, 8, 9);

    // West mirror: group 2 -> passive link -> group 0.
    system.reset_execution_state();
    initialize_row(system, 8, 4);
    system.icu().enqueue_mem(8, west_read);
    system.icu().enqueue_mem_nop(0, 3);
    system.icu().enqueue_mem(0, west_write);
    tick_cycles(system, kCyclesPerDirection);
    check_row(system, 0, 10);

    std::ofstream output(argv[1], std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open C model vector output");
    }
    output << std::hex << std::setfill('0');
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        output << std::setw(16) << packed_tile_word(tile) << '\n';
    }

    std::ofstream schedule_output(argv[2], std::ios::trunc);
    if (!schedule_output) {
        throw std::runtime_error("failed to open C model schedule output");
    }
    schedule_output << std::hex << std::setfill('0');
    schedule_output << std::setw(16) << instruction_record(0, east_read) << '\n';
    schedule_output << std::setw(16) << command_record(8, nop3) << '\n';
    schedule_output << std::setw(16) << instruction_record(8, east_write) << '\n';
    schedule_output << std::setw(16) << instruction_record(8, west_read) << '\n';
    schedule_output << std::setw(16) << command_record(0, nop3) << '\n';
    schedule_output << std::setw(16) << instruction_record(0, west_write) << '\n';

    std::cout << "C model East/West MEM-stream golden generated\n";
    return 0;
}
catch (const std::exception& ex)
{
    std::cerr << ex.what() << '\n';
    return 1;
}
