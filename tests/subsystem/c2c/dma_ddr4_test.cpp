#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <array>
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

void test_ddr4_exposes_eight_vector_channels()
{
    constexpr auto kBaseAddress = std::uint64_t {0x4000};
    constexpr auto kChannels = std::size_t {8};
    auto ddr4 = Ddr4Model(Ddr4Config {32, 2, 2, 64, kChannels});
    auto requests = std::array<Ddr4Model::RequestId, kChannels> {};
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        const auto address =
            kBaseAddress + channel * hw::kPhysicalVectorBytes;
        ddr4.initialize_vector(
            address, make_vector(static_cast<std::uint8_t>(channel * 16)));
        requests[channel] = ddr4.request_read(address);
    }

    ddr4.tick();
    ddr4.tick();
    for (const auto request : requests)
        require(!ddr4.read_completion_ready(request),
            "parallel C2C transfer completed during external latency");
    ddr4.tick();
    for (const auto request : requests)
        require(ddr4.read_completion_ready(request),
            "C2C did not complete eight 32-byte vectors in one cycle");
}

void test_c2c_stream_count_is_runtime_selectable()
{
    auto hardware = SystemHardwareConfiguration {};
    hardware.c2c_streams_per_direction = 4;
    auto system = C2cDmaSystem(
        Ddr4Config {32, 2, 2, 64, 4}, 16, hardware);
    system.dma(Hemisphere::East).issue(
        C2cDmaInstruction::Load(0x6000, 1, 32, 0, 3));

    bool rejected = false;
    try {
        system.dma(Hemisphere::East).issue(
            C2cDmaInstruction::Load(0x6020, 1, 32, 0, 4));
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    require(rejected,
        "C2C DMA accepted a stream disabled by hardware configuration");
}

void test_dedicated_c2c_overlaps_all_compute_streams()
{
    constexpr auto kTargetSlice = std::size_t {40};
    constexpr auto kTargetRow = std::size_t {17};
    constexpr auto kDdr4Address = std::uint64_t {0x7000};
    auto system = C2cDmaSystem(Ddr4Config {32, 2, 2, 64, 8});
    const auto expected = make_vector(0xa0, 91);
    system.ddr4().initialize_vector(kDdr4Address, expected);

    for (std::size_t stream = 0; stream < hw::kWestStreams; ++stream) {
        const auto queue = InstructionControlUnit::mem_queue(
            Hemisphere::West, stream, 0);
        for (std::size_t cycle = 0; cycle < 16; ++cycle)
            system.chip().icu().enqueue_mem(queue,
                MemInstruction::Read(0, StreamId::West(stream)));
    }
    system.chip().icu().enqueue_c2c_dma(Hemisphere::West,
        C2cDmaInstruction::Load(kDdr4Address, 1, 32, 91, 0));
    system.chip().icu().enqueue_c2c_receive(
        Hemisphere::West, 0, Hemisphere::West, kTargetSlice,
        1, false, kTargetRow);

    bool observed_overlap = false;
    for (std::size_t cycle = 0; cycle < 32; ++cycle) {
        system.tick();
        std::size_t issued = 0;
        for (std::size_t stream = 0; stream < hw::kWestStreams; ++stream) {
            const auto queue = InstructionControlUnit::mem_queue(
                Hemisphere::West, stream, 0);
            issued += system.chip().icu().mem_iq(queue).last_trace().action
                    == IcuQueueAction::FunctionalIssue
                ? 1
                : 0;
        }
        observed_overlap = observed_overlap
            || (issued == hw::kWestStreams
                && system.dma(Hemisphere::West).last_beat().has_value());
    }
    require(observed_overlap,
        "all 32 compute streams blocked the dedicated C2C data path");
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            require(system.chip().read_mem_sram_lane_byte(
                        Hemisphere::West, kTargetSlice, 1, tile,
                        kTargetRow, lane)
                    == expected.payload[tile][lane],
                "dedicated C2C write changed data under compute-stream load");
}


void test_ddr4_dma_rx_sr_mem()
{
    constexpr auto kHemisphere = Hemisphere::West;
    constexpr auto kStream = 5U;
    constexpr auto kTargetSlice = 16U;
    constexpr auto kTargetRow = 41U;
    constexpr auto kDdr4Address = std::uint64_t {0x1000};

    auto system = C2cDmaSystem(Ddr4Config {8, 2, 2, 4});
    auto& chip = system.chip();
    const auto expected = make_vector(0x40, 77);
    system.ddr4().initialize_vector(kDdr4Address, expected);

    chip.icu().enqueue_c2c_dma(
        kHemisphere,
        C2cDmaInstruction::Load(kDdr4Address, 1, 32, 77, kStream));
    chip.icu().enqueue_control(
        IcuLocation::C2cDma(kHemisphere),
        IcuControlInstruction::Sync());
    chip.icu().enqueue_c2c_receive(
        kHemisphere, kStream, kHemisphere, kTargetSlice,
        0, true, kTargetRow);
    const auto target_queue = InstructionControlUnit::mem_queue(
        kHemisphere, kTargetSlice);
    chip.icu().enqueue_control(
        IcuLocation::Mem(kHemisphere, kTargetSlice),
        IcuControlInstruction::Sync());

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
        "dedicated C2C RX did not notify the target MEM ICU");
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
    test_ddr4_exposes_eight_vector_channels();
    test_c2c_stream_count_is_runtime_selectable();
    test_dedicated_c2c_overlaps_all_compute_streams();
    test_ddr4_dma_rx_sr_mem();
    test_mem_sr_tx_dma_ddr4();
}
