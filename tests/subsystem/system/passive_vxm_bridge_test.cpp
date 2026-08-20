#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kSourceAddress = 17;
constexpr std::size_t kDestinationAddress = 23;
constexpr std::size_t kBridgeCycle = 12;
constexpr std::size_t kTransferCount = 8;
constexpr std::array<std::size_t, 2> kSlices {18, 19};
constexpr std::array<std::size_t, 2> kStreams {0, 1};
constexpr std::array<ftlpu::Hemisphere, 2> kSources {
    ftlpu::Hemisphere::East,
    ftlpu::Hemisphere::West,
};

constexpr std::size_t opposite(std::size_t side)
{
    return side ^ 1U;
}

constexpr std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

constexpr std::size_t east_write_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 1;
}

std::uint8_t input_value(
    std::size_t transfer, std::size_t tile, std::size_t lane)
{
    return static_cast<std::uint8_t>(
        1 + transfer * 73 + tile * ftlpu::hw::kLanesPerTile + lane);
}

void enqueue_mem_at(
    ftlpu::InstructionControlUnit& icu,
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues>& cursors,
    std::size_t queue,
    std::size_t cycle,
    ftlpu::MemInstruction instruction)
{
    if (cycle < cursors[queue]) {
        throw std::logic_error("passive bridge schedule overlaps a MEM queue");
    }
    icu.enqueue_mem_nop(queue, cycle - cursors[queue]);
    icu.enqueue_mem(queue, std::move(instruction));
    cursors[queue] = cycle + 1;
}

} // namespace

int main()
try {
    auto system = ftlpu::TspSliceSystem {};
    auto cursors = std::array<
        std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues> {};

    for (std::size_t transfer = 0; transfer < kSources.size(); ++transfer) {
        const auto source = kSources[transfer];
        const auto destination = kSources[opposite(transfer)];
        const auto slice = kSlices[transfer];
        const auto stream = kStreams[transfer];
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                for (std::size_t beat = 0; beat < kTransferCount; ++beat) {
                    system.initialize_mem_sram_lane_byte(
                        source, slice, tile, kSourceAddress + beat, lane,
                        static_cast<std::uint8_t>(
                            input_value(transfer, tile, lane) + beat));
                }
            }
        }

        const auto source_queue = ftlpu::InstructionControlUnit::mem_queue(
            source, slice);
        enqueue_mem_at(
            system.icu(), cursors, source_queue,
            kBridgeCycle - west_read_latency(slice),
            ftlpu::MemInstruction::Read(
                kSourceAddress, ftlpu::StreamId::West(stream)));
        system.icu().enqueue_mem_repeat(
            source_queue, kTransferCount - 1, 1, 1);

        const auto destination_queue = ftlpu::InstructionControlUnit::mem_queue(
            destination, slice);
        enqueue_mem_at(
            system.icu(), cursors, destination_queue,
            kBridgeCycle + east_write_latency(slice),
            ftlpu::MemInstruction::Write(
                kDestinationAddress, ftlpu::StreamId::East(stream)));
        system.icu().enqueue_mem_repeat(
            destination_queue, kTransferCount - 1, 1, 1);
    }

    constexpr auto kRunCycles = kBridgeCycle
        + east_write_latency(kSlices.back()) + kTransferCount
        + ftlpu::hw::kTileRows + 3;
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle) {
        system.tick({});
    }

    for (std::size_t transfer = 0; transfer < kSources.size(); ++transfer) {
        const auto destination = kSources[opposite(transfer)];
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                for (std::size_t beat = 0; beat < kTransferCount; ++beat) {
                    const auto actual = system.read_mem_sram_lane_byte(
                        destination, kSlices[transfer], tile,
                        kDestinationAddress + beat, lane);
                    const auto expected = static_cast<std::uint8_t>(
                        input_value(transfer, tile, lane) + beat);
                    if (actual != expected) {
                        std::cerr << "passive VXM bridge mismatch: transfer="
                                  << transfer << " beat=" << beat
                                  << " tile=" << tile << " lane=" << lane
                                  << " actual=" << static_cast<unsigned>(actual)
                                  << " expected=" << static_cast<unsigned>(expected)
                                  << '\n';
                        return 1;
                    }
                }
            }
        }
    }

    std::cout << "bidirectional passive VXM bridge passed at cycle "
              << kBridgeCycle << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "passive VXM bridge test failed: " << error.what() << '\n';
    return 1;
}
