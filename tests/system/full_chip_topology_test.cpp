#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "../integration/vxm_alu_program.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

constexpr std::size_t kVxmCycle = 10;
constexpr std::size_t kWriteCycle = kVxmCycle + 2;
constexpr std::size_t kInputAddress = 7;
constexpr std::size_t kOutputAddress = 9;
constexpr std::array<std::size_t, 8> kInputSlices {16, 17, 18, 19, 20, 21, 22, 23};

std::size_t read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

} // namespace

int main()
{
    static_assert(ftlpu::InstructionControlUnit::kMemQueues == 88);
    static_assert(ftlpu::InstructionControlUnit::kMxmQueues == 4);
    static_assert(ftlpu::TspSliceSystem::kMxmCount == 4);

    auto system = ftlpu::TspSliceSystem {};
    auto& icu = system.icu();

    std::array<float, ftlpu::hw::kLanesPerTile> scales {};
    for (std::size_t element = 0; element < scales.size(); ++element) {
        scales[element] = 0.125f * static_cast<float>(element + 1);
        const auto slice = kInputSlices[element];
        const auto read_cycle = kVxmCycle - read_latency(slice);
        const auto queue = ftlpu::InstructionControlUnit::mem_queue(ftlpu::Hemisphere::West, slice);
        icu.enqueue_mem_nop(queue, read_cycle);
        icu.enqueue_mem(
            queue,
            ftlpu::MemInstruction::Read(kInputAddress, ftlpu::StreamId::West(element)));

        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto value = static_cast<std::int8_t>(
                    static_cast<int>(tile * ftlpu::hw::kLanesPerTile + lane + element) - 16);
                system.initialize_mem_sram_lane_byte(
                    ftlpu::Hemisphere::West,
                    slice,
                    tile,
                    kInputAddress,
                    lane,
                    static_cast<std::uint8_t>(value));
            }
        }
    }

    auto dequant = ftlpu::test::DequantSpec {};
    dequant.input_stream_base = ftlpu::hw::kEastStreams;
    dequant.output_stream_base = 0;
    dequant.scales = scales;
    dequant.input_hemisphere = ftlpu::Hemisphere::West;
    dequant.output_hemisphere = ftlpu::Hemisphere::West;
    auto vxm_cursors = std::array<std::size_t, ftlpu::VxmLane::kAluCount> {};
    ftlpu::test::enqueue_dequant(icu, vxm_cursors, kVxmCycle, dequant);

    for (std::size_t stream = 0; stream < 16; ++stream) {
        const auto queue = ftlpu::InstructionControlUnit::mem_queue(ftlpu::Hemisphere::West, stream);
        const auto write_cycle = kWriteCycle + stream / ftlpu::hw::kMemSlicesPerGroup;
        icu.enqueue_mem_nop(queue, write_cycle);
        icu.enqueue_mem(
            queue,
            ftlpu::MemInstruction::Write(kOutputAddress, ftlpu::StreamId::East(stream)));
    }

    for (std::size_t cycle = 0; cycle < kWriteCycle + 4 + ftlpu::hw::kTileRows + 2; ++cycle) {
        system.tick({});
    }

    for (std::size_t element = 0; element < scales.size(); ++element) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto input = static_cast<std::int8_t>(
                    static_cast<int>(tile * ftlpu::hw::kLanesPerTile + lane + element) - 16);
                const auto expected = ftlpu::Fp16::from_float(
                    static_cast<float>(input) * scales[element]).bits();
                const auto low = system.read_mem_sram_lane_byte(
                    ftlpu::Hemisphere::West, element * 2, tile, kOutputAddress, lane);
                const auto high = system.read_mem_sram_lane_byte(
                    ftlpu::Hemisphere::West, element * 2 + 1, tile, kOutputAddress, lane);
                const auto actual = static_cast<std::uint16_t>(low)
                    | (static_cast<std::uint16_t>(high) << 8);
                if (actual != expected) {
                    std::cerr << "west hemisphere round-trip mismatch at element=" << element
                              << " tile=" << tile << " lane=" << lane << '\n';
                    return 1;
                }
            }
        }
    }

    std::cout << "full-chip mirrored topology passed: 88 MEM queues, 4 MXMs, dual SXM/MEM edges\n";
    return 0;
}
