#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kIssueCycle = 20;
constexpr std::size_t kRows = 32;
constexpr std::size_t kLhsAddress = 64;
constexpr std::size_t kRhsAddress = 128;
constexpr std::array<std::size_t, 2> kLhsSlices {51, 50};
constexpr std::array<std::size_t, 16> kRhsSlices {
    0, 2, 4, 8, 12, 16, 18, 20,
    24, 25, 26, 27, 28, 29, 30, 31};

constexpr std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

void enqueue_mem_at(
    ftlpu::InstructionControlUnit& icu,
    std::array<std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues>& cursors,
    ftlpu::Hemisphere hemisphere,
    std::size_t slice,
    std::size_t cycle,
    ftlpu::MemInstruction instruction)
{
    const auto queue = ftlpu::InstructionControlUnit::mem_queue(
        hemisphere, slice);
    if (cycle < cursors[queue])
        throw std::logic_error("VXM repeated-input MEM queue overlap");
    icu.enqueue_mem_nop(queue, cycle - cursors[queue]);
    icu.enqueue_mem(queue, std::move(instruction));
    cursors[queue] = cycle + 1;
}

void initialize_inputs(ftlpu::TspSliceSystem& system)
{
    for (std::size_t side = 0; side < ftlpu::hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<ftlpu::Hemisphere>(side);
        for (std::size_t row = 0; row < kRows; ++row) {
            const auto rhsPair = row % 8;
            const auto rhsAddress = kRhsAddress + row / 8;
            for (std::size_t tile = 0;
                 tile < ftlpu::hw::kTileRows; ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    for (std::size_t byte = 0; byte < 2; ++byte) {
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kLhsSlices[byte], tile,
                            kLhsAddress + row, lane,
                            static_cast<std::uint8_t>(
                                1 + byte + row + lane));
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kRhsSlices[2 * rhsPair + byte],
                            tile, rhsAddress, lane,
                            static_cast<std::uint8_t>(
                                33 + byte + row + lane));
                    }
                }
            }
        }
    }
}

} // namespace

int main()
try {
    auto system = ftlpu::TspSliceSystem {};
    system.vxm_unit().configure_input_group_source(
        0, ftlpu::Hemisphere::East);
    system.vxm_unit().configure_input_group_source(
        1, ftlpu::Hemisphere::East);
    system.vxm_unit().configure_input_group_source(
        8, ftlpu::Hemisphere::West);
    system.vxm_unit().configure_input_group_source(
        9, ftlpu::Hemisphere::West);
    initialize_inputs(system);

    auto& icu = system.icu();
    auto cursors = std::array<std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues> {};
    for (std::size_t side = 0; side < ftlpu::hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<ftlpu::Hemisphere>(side);
        const auto streamOffset = side * 16;
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kLhsSlices[byte];
            enqueue_mem_at(icu, cursors, hemisphere, slice,
                kIssueCycle - west_read_latency(slice),
                ftlpu::MemInstruction::Read(kLhsAddress,
                    ftlpu::StreamId::West(streamOffset + byte)));
            const auto queue = ftlpu::InstructionControlUnit::mem_queue(
                hemisphere, slice);
            icu.enqueue_mem_repeat(queue, kRows - 1, 1, 1);
        }
        for (std::size_t pair = 0; pair < 8; ++pair) {
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kRhsSlices[2 * pair + byte];
                enqueue_mem_at(icu, cursors, hemisphere, slice,
                    kIssueCycle + pair - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(kRhsAddress,
                        ftlpu::StreamId::West(
                            streamOffset + 2 + byte)));
                const auto queue =
                    ftlpu::InstructionControlUnit::mem_queue(
                        hemisphere, slice);
                icu.enqueue_mem_repeat(queue, 3, 8, 1);
            }
        }
    }

    auto add = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::StreamBFloat16(),
        ftlpu::VxmLaneOperand::StreamBFloat16()};
    add.repeat_count = kRows;
    auto cast = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Bypass,
        ftlpu::VxmLaneOperand::Previous()};
    cast.output_type = ftlpu::VxmCastTarget::BFloat16;
    cast.output_stream = 0;
    cast.repeat_count = kRows;

    icu.enqueue_vxm_nop(0, kIssueCycle - 1);
    icu.enqueue_vxm(0, ftlpu::VxmChainDepth::Two, add);
    icu.enqueue_vxm_nop(1, kIssueCycle - 1);
    icu.enqueue_vxm(1, ftlpu::VxmChainDepth::Two, cast);

    constexpr auto kRunCycles =
        kIssueCycle + kRows + ftlpu::hw::kTileRows + 4;
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle) {
        try {
            system.tick({});
        } catch (const std::exception& error) {
            throw std::logic_error("cycle " + std::to_string(cycle)
                + ": " + error.what());
        }
    }

    std::cout << "VXM repeated fixed/distributed BF16 inputs passed: rows="
              << kRows << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "VXM repeated distributed-input test failed: "
              << error.what() << '\n';
    return 1;
}
