#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kBank = 1;
constexpr std::size_t kBaseAddress = 128;
constexpr std::size_t kFirstBridgeCycle = 32;
constexpr std::size_t kBlockBeats = 4;
constexpr std::size_t kBlocks = 48;
constexpr std::array<std::size_t, 16> kSlices {
    51, 50, 49, 48, 47, 46, 45, 44,
    43, 42, 41, 40, 39, 38, 37, 36};
constexpr std::size_t kMaxSliceGroup =
    kSlices.front() / ftlpu::hw::kMemSlicesPerGroup;
constexpr std::size_t kBlockInterval =
    kBlockBeats + 2 * kMaxSliceGroup + 3;

constexpr std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

constexpr std::size_t east_write_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 1;
}

std::uint8_t value(std::size_t block, std::size_t stream,
    std::size_t beat, std::size_t tile, std::size_t lane)
{
    return static_cast<std::uint8_t>(
        1 + block * 29 + stream * 7 + beat * 3 + tile + lane);
}

void enqueue_wave(ftlpu::InstructionControlUnit& icu,
    std::array<std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues>& cursors,
    std::size_t queue, std::size_t cycle,
    ftlpu::MemInstruction instruction)
{
    if (cycle < cursors[queue])
        throw std::logic_error("distributed16 copy MEM queue overlap");
    icu.enqueue_mem_nop(queue, cycle - cursors[queue]);
    icu.enqueue_mem(queue, std::move(instruction));
    icu.enqueue_mem_repeat(queue, kBlockBeats - 1, 1, 1);
    cursors[queue] = cycle + kBlockBeats;
}

} // namespace

int main()
try {
    auto system = ftlpu::TspSliceSystem {};
    auto cursors = std::array<std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues> {};
    std::size_t runEnd = 0;

    for (std::size_t block = 0; block < kBlocks; ++block) {
        const auto owner = static_cast<ftlpu::Hemisphere>(block % 2);
        const auto replica = static_cast<ftlpu::Hemisphere>((block + 1) % 2);
        const std::size_t address = kBaseAddress + block * kBlockBeats;
        const std::size_t bridgeCycle =
            kFirstBridgeCycle + block * kBlockInterval;
        for (std::size_t stream = 0; stream < kSlices.size(); ++stream) {
            const std::size_t slice = kSlices[stream];
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    for (std::size_t beat = 0;
                         beat < kBlockBeats; ++beat) {
                        system.initialize_mem_sram_lane_byte(owner, slice,
                            kBank, tile, address + beat, lane,
                            value(block, stream, beat, tile, lane));
                    }
                }
            }

            enqueue_wave(system.icu(), cursors,
                ftlpu::InstructionControlUnit::mem_queue(owner, slice, kBank),
                bridgeCycle - west_read_latency(slice),
                ftlpu::MemInstruction::Read(
                    address, ftlpu::StreamId::West(stream)));
            const std::size_t writeCycle =
                bridgeCycle + east_write_latency(slice);
            enqueue_wave(system.icu(), cursors,
                ftlpu::InstructionControlUnit::mem_queue(replica, slice, kBank),
                writeCycle,
                ftlpu::MemInstruction::Write(
                    address, ftlpu::StreamId::East(stream)));
            runEnd = std::max(runEnd, writeCycle + kBlockBeats);
        }
    }

    for (std::size_t cycle = 0;
         cycle < runEnd + ftlpu::hw::kTileRows + 3; ++cycle) {
        try {
            system.tick({});
        } catch (const std::exception& error) {
            throw std::logic_error("cycle " + std::to_string(cycle)
                + ": " + error.what());
        }
    }

    for (std::size_t block = 0; block < kBlocks; ++block) {
        const auto replica = static_cast<ftlpu::Hemisphere>((block + 1) % 2);
        const std::size_t address = kBaseAddress + block * kBlockBeats;
        for (std::size_t stream = 0; stream < kSlices.size(); ++stream) {
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    for (std::size_t beat = 0;
                         beat < kBlockBeats; ++beat) {
                        const auto actual = system.read_mem_sram_lane_byte(
                            replica, kSlices[stream], kBank, tile,
                            address + beat, lane);
                        const auto expected =
                            value(block, stream, beat, tile, lane);
                        if (actual != expected) {
                            std::cerr << "distributed16 passive copy mismatch"
                                      << " block=" << block
                                      << " stream=" << stream
                                      << " beat=" << beat
                                      << " actual="
                                      << static_cast<unsigned>(actual)
                                      << " expected="
                                      << static_cast<unsigned>(expected)
                                      << '\n';
                            return 1;
                        }
                    }
                }
            }
        }
    }

    std::cout << "distributed16 passive copy passed: blocks="
              << kBlocks << " streams=" << kSlices.size() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "distributed16 passive copy test failed: "
              << error.what() << '\n';
    return 1;
}
