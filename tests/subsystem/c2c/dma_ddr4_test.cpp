#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using namespace ftlpu;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

C2cVector make_vector(std::uint8_t base, std::uint64_t tag = 0)
{
    auto vector = C2cVector {};
    vector.vector_tag = tag;
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            vector.payload[tile][lane] = static_cast<std::uint8_t>(
                base + tile * hw::kLanesPerTile + lane);
        }
    }
    return vector;
}
void test_ddr4_transfers_one_beat_per_cycle()
{
    constexpr auto kAddress = std::uint64_t {0x800};
    auto ddr4 = Ddr4Model(Ddr4Config {8, 2, 2, 2});
    const auto expected = make_vector(0x20);
    const auto request = ddr4.request_write(kAddress, expected);

    ddr4.tick();
    ddr4.tick();
    require(!ddr4.last_beat().has_value(),
        "DDR4 transferred data during write latency");

    ddr4.tick();
    require(ddr4.last_beat().has_value(),
        "DDR4 did not expose its first write beat");
    require(ddr4.last_beat()->address == kAddress
            && ddr4.last_beat()->byte_count == 8,
        "DDR4 first beat trace has an incorrect address or size");
    const auto partial = ddr4.read_vector(kAddress);
    for (std::size_t byte = 0; byte < hw::kPhysicalVectorBytes; ++byte) {
        const auto tile = byte / hw::kLanesPerTile;
        const auto lane = byte % hw::kLanesPerTile;
        const auto expected_byte =
            byte < 8 ? expected.payload[tile][lane] : 0;
        require(partial.payload[tile][lane] == expected_byte,
            "DDR4 write did not commit exactly one beat");
    }
    ddr4.tick();
    ddr4.tick();
    ddr4.tick();
    require(ddr4.write_completion_ready(request),
        "DDR4 write completion did not follow the final beat");
    require(ddr4.read_vector(kAddress).payload == expected.payload,
        "DDR4 beat sequence did not commit the complete vector");
}


void test_ddr4_dma_rx_sr_mem()
{
    constexpr auto kHemisphere = Hemisphere::West;
    constexpr auto kStream = 5U;
    constexpr auto kTargetSlice = 16U;
    constexpr auto kTargetRow = 41U;
    constexpr auto kDdr4Address = std::uint64_t {0x1000};
    constexpr auto kTargetGroup =
        kTargetSlice / hw::kMemSlicesPerGroup;
    constexpr auto kRxToTargetNops =
        hw::kMemEastBoundaryStreamRegisterColumn
        - (kTargetGroup + 1) - 1;

    auto system = C2cDmaSystem(Ddr4Config {8, 2, 2, 4});
    auto& chip = system.chip();
    const auto expected = make_vector(0x40, 77);
    system.ddr4().initialize_vector(kDdr4Address, expected);

    chip.icu().enqueue_c2c_dma(
        kHemisphere,
        C2cDmaInstruction::Load(kDdr4Address, 1, 32, 77));
    chip.icu().enqueue_control(
        IcuLocation::C2cDma(kHemisphere),
        IcuControlInstruction::Sync());
    chip.icu().enqueue_c2c_receive(
        kHemisphere, kStream, kHemisphere, kTargetSlice);

    const auto target_queue = InstructionControlUnit::mem_queue(
        kHemisphere, kTargetSlice);
    chip.icu().enqueue_control(
        IcuLocation::Mem(kHemisphere, kTargetSlice),
        IcuControlInstruction::Sync());
    chip.icu().enqueue_mem_nop(target_queue, kRxToTargetNops);
    chip.icu().enqueue_mem(
        target_queue,
        MemInstruction::Write(kTargetRow, StreamId::West(kStream)));

    bool observed_dma_sync_wait = false;
    bool observed_dma_sync_release = false;
    bool observed_mem_sync_release = false;
    for (std::size_t cycle = 0; cycle < 40; ++cycle) {
        system.tick();
        const auto dma_action =
            chip.icu().c2c_dma_iq(kHemisphere).last_trace().action;
        observed_dma_sync_wait = observed_dma_sync_wait
            || dma_action == IcuQueueAction::SyncWait;
        observed_dma_sync_release = observed_dma_sync_release
            || dma_action == IcuQueueAction::SyncRelease;
        observed_mem_sync_release = observed_mem_sync_release
            || chip.icu().mem_iq(target_queue).last_trace().action
                == IcuQueueAction::SyncRelease;
    }

    require(observed_dma_sync_wait,
        "DMA ICU did not wait for the DDR4 read to complete");
    require(observed_dma_sync_release,
        "DDR4 read completion did not release the DMA ICU");
    require(observed_mem_sync_release,
        "C2C RX did not notify the target MEM ICU");
    require(system.dma(kHemisphere).idle(),
        "DDR4-to-C2C DMA did not become idle");

    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto actual = chip.read_mem_sram_lane_byte(
                kHemisphere, kTargetSlice, tile, kTargetRow, lane);
            if (actual != expected.payload[tile][lane]) {
                throw std::runtime_error(
                    "DDR4 -> DMA -> RX -> SR -> MEM mismatch at tile "
                    + std::to_string(tile) + ", lane "
                    + std::to_string(lane));
            }
        }
    }
}

void test_mem_sr_tx_dma_ddr4()
{
    constexpr auto kHemisphere = Hemisphere::East;
    constexpr auto kStream = 7U;
    constexpr auto kSourceSlice = 48U;
    constexpr auto kSourceRow = 23U;
    constexpr auto kDdr4Address = std::uint64_t {0x2000};

    auto system = C2cDmaSystem(Ddr4Config {8, 2, 2, 4});
    auto& chip = system.chip();
    const auto expected = make_vector(0x80, 99);

    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            chip.initialize_mem_sram_lane_byte(
                kHemisphere,
                kSourceSlice,
                tile,
                kSourceRow,
                lane,
                expected.payload[tile][lane]);
        }
    }

    const auto source_queue = InstructionControlUnit::mem_queue(
        kHemisphere, kSourceSlice);
    chip.icu().enqueue_mem(
        source_queue,
        MemInstruction::Read(kSourceRow, StreamId::East(kStream)));
    chip.icu().enqueue_c2c_send(kHemisphere, kStream);
    chip.icu().enqueue_c2c_dma(
        kHemisphere, C2cDmaInstruction::Store(kDdr4Address));
    chip.icu().enqueue_control(
        IcuLocation::C2cDma(kHemisphere),
        IcuControlInstruction::Sync());

    bool observed_dma_sync_wait = false;
    bool observed_dma_sync_release = false;
    for (std::size_t cycle = 0; cycle < 40; ++cycle) {
        system.tick();
        const auto action =
            chip.icu().c2c_dma_iq(kHemisphere).last_trace().action;
        observed_dma_sync_wait = observed_dma_sync_wait
            || action == IcuQueueAction::SyncWait;
        observed_dma_sync_release = observed_dma_sync_release
            || action == IcuQueueAction::SyncRelease;
    }

    require(observed_dma_sync_wait,
        "DMA ICU did not wait for the DDR4 write to complete");
    require(observed_dma_sync_release,
        "DDR4 write completion did not release the DMA ICU");
    require(system.dma(kHemisphere).idle(),
        "C2C-to-DDR4 DMA did not become idle");
    require(
        system.ddr4().read_vector(kDdr4Address).payload == expected.payload,
        "MEM -> SR -> TX -> DMA -> DDR4 changed the vector payload");
}

} // namespace

int main()
{
    test_ddr4_transfers_one_beat_per_cycle();
    test_ddr4_dma_rx_sr_mem();
    test_mem_sr_tx_dma_ddr4();
}
