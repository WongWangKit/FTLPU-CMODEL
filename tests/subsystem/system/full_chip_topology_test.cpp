#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kInputAddress = 7;
constexpr std::size_t kOutputAddress = 9;
constexpr std::size_t kConfigCycle = 8;
constexpr std::size_t kInputCycle = kConfigCycle + 1;
constexpr std::size_t kFirstOutputCycle = kInputCycle + 1;
constexpr std::size_t kWriteCycle = kFirstOutputCycle + 1;

// The two logical compact queues are mirrored onto physical C0/C1 and C8/C9.
// Their fixed input groups are 0 and 8 and their fixed output groups are 0
// and 4 (streams 0/1 and 8/9).
constexpr std::array<std::size_t, 4> kInputStreams {0, 1, 16, 17};
constexpr std::array<std::size_t, 4> kOutputStreams {0, 1, 8, 9};
constexpr std::array<std::size_t, 4> kMemSlices {0, 1, 2, 3};

std::size_t west_read_latency(std::size_t mem_slice)
{
    return mem_slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

float input_value(std::size_t physical_chain, std::size_t tile, std::size_t lane)
{
    return static_cast<float>(
        physical_chain * 100 + tile * 10 + lane);
}

void enqueue_mem_at(
    ftlpu::InstructionControlUnit& icu,
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues>& cursors,
    std::size_t queue,
    std::size_t cycle,
    ftlpu::MemInstruction instruction)
{
    if (cycle < cursors[queue]) {
        throw std::logic_error("MEM test schedule overlaps one ICU queue");
    }
    icu.enqueue_mem_nop(queue, cycle - cursors[queue]);
    icu.enqueue_mem(queue, std::move(instruction));
    cursors[queue] = cycle + 1;
}

void initialize_inputs(ftlpu::TspSliceSystem& system)
{
    for (std::size_t physical_chain = 0; physical_chain < 2; ++physical_chain) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto bits = ftlpu::Fp16::from_float(
                    input_value(physical_chain, tile, lane)).bits();
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    const auto slice = kMemSlices[physical_chain * 2 + byte];
                    system.initialize_mem_sram_lane_byte(
                        ftlpu::Hemisphere::East,
                        slice,
                        tile,
                        kInputAddress,
                        lane,
                        static_cast<std::uint8_t>(bits >> (8 * byte)));
                }
            }
        }
    }
}

bool verify_outputs(const ftlpu::TspSliceSystem& system)
{
    for (std::size_t physical_chain = 0; physical_chain < 2; ++physical_chain) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto low = system.read_mem_sram_lane_byte(
                    ftlpu::Hemisphere::East,
                    kMemSlices[physical_chain * 2],
                    tile,
                    kOutputAddress,
                    lane);
                const auto high = system.read_mem_sram_lane_byte(
                    ftlpu::Hemisphere::East,
                    kMemSlices[physical_chain * 2 + 1],
                    tile,
                    kOutputAddress,
                    lane);
                const auto actual = static_cast<std::uint16_t>(low)
                    | (static_cast<std::uint16_t>(high) << 8);
                const auto expected = ftlpu::Fp16::from_float(
                    input_value(physical_chain, tile, lane) + 1.0f).bits();
                if (actual != expected) {
                    std::cerr << "MEM->VXM->MEM mismatch: chain="
                              << physical_chain << " tile=" << tile
                              << " lane=" << lane << " actual=0x"
                              << std::hex << actual << " expected=0x"
                              << expected << std::dec << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

int main()
try {
    auto system = ftlpu::TspSliceSystem {};
    auto& vxm = system.vxm_unit();
    vxm.set_chain_depth(ftlpu::VxmChainDepth::Two);
    vxm.configure_input_group_source(0, ftlpu::Hemisphere::East);
    vxm.configure_input_group_source(8, ftlpu::Hemisphere::East);
    vxm.configure_output_block_destination(0, ftlpu::Hemisphere::East);
    vxm.configure_output_block_destination(4, ftlpu::Hemisphere::East);

    initialize_inputs(system);
    auto& icu = system.icu();
    auto mem_cursors = std::array<
        std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues> {};

    for (std::size_t byte = 0; byte < kInputStreams.size(); ++byte) {
        const auto slice = kMemSlices[byte];
        const auto queue = ftlpu::InstructionControlUnit::mem_queue(
            ftlpu::Hemisphere::East, slice);
        enqueue_mem_at(
            icu,
            mem_cursors,
            queue,
            kInputCycle - west_read_latency(slice),
            ftlpu::MemInstruction::Read(
                kInputAddress,
                ftlpu::StreamId::West(kInputStreams[byte])));
        enqueue_mem_at(
            icu,
            mem_cursors,
            queue,
            kWriteCycle,
            ftlpu::MemInstruction::Write(
                kOutputAddress,
                ftlpu::StreamId::East(kOutputStreams[byte])));
    }

    auto head = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::StreamFloat16(),
        ftlpu::VxmLaneOperand::Imm(1.0f)};
    auto tail = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Bypass,
        ftlpu::VxmLaneOperand::Previous()};
    tail.output_type = ftlpu::VxmCastTarget::Float16;
    tail.output_stream = 0;

    icu.enqueue_vxm_nop(0, kConfigCycle);
    icu.enqueue_vxm(0, ftlpu::VxmChainDepth::Two, head);
    icu.enqueue_vxm_nop(1, kConfigCycle);
    icu.enqueue_vxm(1, ftlpu::VxmChainDepth::Two, tail);

    constexpr auto kRunCycles = kWriteCycle + ftlpu::hw::kTileRows + 3;
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle) {
        system.tick({});

        if (cycle == kConfigCycle) {
            const auto& required = system.vxm_unit().required_streams_at(0);
            if (!required.has_value()) {
                throw std::logic_error(
                    "VXM did not publish its input requirements after decode");
            }
            auto count = std::size_t {0};
            for (const auto bit : *required) {
                count += bit ? 1U : 0U;
            }
            if (count != 4 || !(*required)[0] || !(*required)[1]
                || !(*required)[16] || !(*required)[17]) {
                throw std::logic_error(
                    "VXM required-stream mask did not select exactly groups 0 and 8");
            }
        }

        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            const auto should_output = cycle == kFirstOutputCycle + tile;
            const auto& outputs = system.vxm_unit().outputs_at(tile);
            if (should_output) {
                if (outputs.size() != 2
                    || outputs[0].stream != 0
                    || outputs[1].stream != 8) {
                    throw std::logic_error(
                        "VXM output timing or mirrored fixed-stream binding is incorrect");
                }
            } else if (!outputs.empty()) {
                throw std::logic_error(
                    "VXM produced an output outside the expected tile wavefront cycle");
            }
        }
    }

    if (!verify_outputs(system)) {
        return 1;
    }

    std::cout
        << "MEM -> VXM -> MEM FP16 passed: required_groups=2, "
        << "tile0_input_cycle=" << kInputCycle
        << ", tile0_output_cycle=" << kFirstOutputCycle
        << ", tile_wavefront_interval=1\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "MEM -> VXM -> MEM system test failed: "
              << error.what() << '\n';
    return 1;
}
