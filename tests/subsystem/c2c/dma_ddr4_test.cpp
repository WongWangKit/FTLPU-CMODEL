#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
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

Ddr4Config ideal_ddr4(std::size_t beat_bytes,
    std::size_t read_latency, std::size_t write_latency,
    std::size_t queue_depth,
    std::size_t channels = 1)
{
    Ddr4Config config;
    config.beat_bytes = beat_bytes;
    config.read_latency_cycles = read_latency;
    config.write_latency_cycles = write_latency;
    config.request_queue_depth = queue_depth;
    config.transfer_channels = channels;
    config.peak_bandwidth_bytes_per_second =
        config.lpu_clock_hz * beat_bytes * channels;
    config.read_latency_jitter_cycles = 0;
    config.write_latency_jitter_cycles = 0;
    return config;
}

void test_ddr4_transfers_one_beat_per_cycle()
{
    constexpr auto kAddress = std::uint64_t {0x800};
    auto ddr4 = Ddr4Model(ideal_ddr4(8, 2, 2, 2));
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
    auto ddr4 = Ddr4Model(
        ideal_ddr4(32, 2, 2, 64, kChannels));
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

void test_default_ddr4_bandwidth_and_latency_jitter()
{
    auto config = Ddr4Config {};
    require(config.lpu_clock_hz == 500'000'000
            && config.peak_bandwidth_bytes_per_second == 51'200'000'000,
        "default DDR4 platform is not 500 MHz dual-channel DDR4-3200");

    config.read_latency_cycles = 0;
    config.read_latency_jitter_cycles = 0;
    auto bandwidthModel = Ddr4Model(config);
    constexpr auto kRequestCount = std::size_t {128};
    for (std::size_t index = 0; index < kRequestCount; ++index) {
        const auto address = std::uint64_t {0x80000}
            + index * hw::kPhysicalVectorBytes;
        bandwidthModel.initialize_vector(
            address, make_vector(static_cast<std::uint8_t>(index)));
        (void)bandwidthModel.request_read(address);
    }
    for (std::size_t cycle = 0; cycle < 10; ++cycle)
        bandwidthModel.tick();
    require(bandwidthModel.read_bytes_transferred() == 1024,
        "DDR4 did not enforce an average 102.4-byte/cycle read ceiling");

    const auto completionCycles = [](Ddr4Config jitterConfig) {
        auto model = Ddr4Model(jitterConfig);
        auto requests = std::array<Ddr4Model::RequestId, 16> {};
        auto completed = std::array<std::size_t, 16> {};
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const auto address = std::uint64_t {0x90000}
                + index * hw::kPhysicalVectorBytes;
            model.initialize_vector(
                address, make_vector(static_cast<std::uint8_t>(index * 3)));
            requests[index] = model.request_read(address);
        }
        std::size_t remaining = requests.size();
        for (std::size_t elapsed = 1; elapsed <= 256 && remaining != 0;
             ++elapsed) {
            model.tick();
            for (std::size_t index = 0; index < requests.size(); ++index) {
                if (completed[index] == 0
                    && model.read_completion_ready(requests[index])) {
                    completed[index] = elapsed;
                    --remaining;
                }
            }
        }
        require(remaining == 0,
            "DDR4 jitter test did not complete every request");
        return completed;
    };

    config = Ddr4Config {};
    config.peak_bandwidth_bytes_per_second =
        config.lpu_clock_hz * hw::kPhysicalVectorBytes
        * config.transfer_channels;
    const auto first = completionCycles(config);
    const auto second = completionCycles(config);
    require(first == second,
        "DDR4 latency jitter is not deterministic for a fixed seed");
    bool observedVariation = false;
    for (std::size_t index = 0; index < first.size(); ++index) {
        require(first[index] >= config.read_latency_cycles + 1
                && first[index] <= config.read_latency_cycles
                        + config.read_latency_jitter_cycles + 1,
            "DDR4 read completion escaped the configured jitter window");
        observedVariation = observedVariation
            || (index != 0 && first[index] != first[0]);
    }
    require(observedVariation,
        "DDR4 latency jitter produced no request-level variation");
}

void test_c2c_stream_count_is_runtime_selectable()
{
    auto hardware = SystemHardwareConfiguration {};
    hardware.c2c_streams_per_direction = 4;
    auto system = C2cDmaSystem(
        ideal_ddr4(32, 2, 2, 64, 4), 16, hardware);
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

void test_ddr4_dma_rx_sr_mem()
{
    constexpr auto kHemisphere = Hemisphere::West;
    constexpr auto kStream = 5U;
    constexpr auto kFabricStream = 29U;
    constexpr auto kTargetSlice = 16U;
    constexpr auto kTargetRow = 41U;
    constexpr auto kDdr4Address = std::uint64_t {0x1000};

    auto system = C2cDmaSystem(ideal_ddr4(8, 2, 2, 4));
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
        0, true, kTargetRow, 1, 1, kFabricStream);
    const auto target_queue = InstructionControlUnit::mem_queue(
        kHemisphere, kTargetSlice);
    constexpr auto kTargetGroup =
        kTargetSlice / hw::kMemSlicesPerGroup;
    constexpr auto kTransportNops =
        hw::kMemEastBoundaryStreamRegisterColumn - (kTargetGroup + 1);
    chip.icu().enqueue_c2c_mem_stream_write(target_queue,
        MemInstruction::Write(
            kTargetRow, StreamId::West(kFabricStream)),
        1, kTransportNops);

    bool observed_dma_sync_wait = false;
    bool observed_dma_sync_release = false;
    bool observed_mem_synchronized_issue = false;
    for (std::size_t cycle = 0; cycle < 64; ++cycle) {
        system.tick();
        const auto dma_action =
            chip.icu().c2c_dma_iq(kHemisphere).last_trace().action;
        observed_dma_sync_wait = observed_dma_sync_wait
            || dma_action == IcuQueueAction::SyncWait;
        observed_dma_sync_release = observed_dma_sync_release
            || dma_action == IcuQueueAction::SyncRelease;
        observed_mem_synchronized_issue = observed_mem_synchronized_issue
            || chip.icu().c2c_mem_iq(target_queue).last_trace().action
                == IcuQueueAction::SynchronizedIssue;
    }

    require(observed_dma_sync_wait,
        "DMA ICU did not wait for the DDR4 read to complete");
    require(observed_dma_sync_release,
        "DDR4 read completion did not release the DMA ICU");
    require(observed_mem_synchronized_issue,
        "shared C2C RX did not notify the target MEM ICU");
    require(system.dma(kHemisphere).idle(),
        "DDR4-to-C2C DMA did not become idle");

    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto actual = chip.read_mem_sram_lane_byte(
                kHemisphere, kTargetSlice, tile, kTargetRow, lane);
            if (actual != expected.payload[tile][lane]) {
                throw std::runtime_error(
                    "DDR4 -> DMA -> shared SR -> MEM Write mismatch at tile "
                    + std::to_string(tile) + ", lane "
                    + std::to_string(lane));
            }
        }
    }
}

void test_eight_shared_c2c_lanes_write_through_mem()
{
    constexpr auto kHemisphere = Hemisphere::East;
    constexpr auto kLaneCount = std::size_t {8};
    constexpr auto kFabricBase = std::size_t {24};
    constexpr auto kTargetRow = std::size_t {73};
    constexpr auto kDdr4Base = std::uint64_t {0x10000};
    auto system = C2cDmaSystem(
        ideal_ddr4(32, 2, 2, 64, kLaneCount));
    auto& chip = system.chip();

    for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
        const auto targetSlice = std::size_t {20} + lane;
        const auto address = kDdr4Base
            + lane * hw::kPhysicalVectorBytes;
        const auto expected = make_vector(
            static_cast<std::uint8_t>(0x10 + lane * 0x10), 200 + lane);
        system.ddr4().initialize_vector(address, expected);
        chip.icu().enqueue_c2c_dma(kHemisphere,
            C2cDmaInstruction::Load(address, 1,
                hw::kPhysicalVectorBytes, 200 + lane, lane));
        chip.icu().enqueue_c2c_receive(kHemisphere, lane,
            kHemisphere, targetSlice, 1, true, kTargetRow,
            1, 1, kFabricBase + lane);

        const auto queue = InstructionControlUnit::mem_queue(
            kHemisphere, targetSlice, 1);
        const auto group = targetSlice / hw::kMemSlicesPerGroup;
        const auto transportNops =
            hw::kMemEastBoundaryStreamRegisterColumn - (group + 1);
        chip.icu().enqueue_c2c_mem_stream_write(queue,
            MemInstruction::Write(
                kTargetRow, StreamId::West(kFabricBase + lane)),
            1, transportNops);
    }

    for (std::size_t cycle = 0; cycle < 80; ++cycle) system.tick();

    for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
        const auto targetSlice = std::size_t {20} + lane;
        const auto expected = make_vector(
            static_cast<std::uint8_t>(0x10 + lane * 0x10), 200 + lane);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t byte = 0; byte < hw::kLanesPerTile; ++byte)
                require(chip.read_mem_sram_lane_byte(kHemisphere,
                            targetSlice, 1, tile, kTargetRow, byte)
                        == expected.payload[tile][byte],
                    "parallel shared C2C lane did not write through MEM");
    }
}

void test_mem_sr_tx_dma_ddr4()
{
    constexpr auto kHemisphere = Hemisphere::East;
    constexpr auto kStream = 7U;
    constexpr auto kSourceSlice = 48U;
    constexpr auto kSourceRow = 23U;
    constexpr auto kDdr4Address = std::uint64_t {0x2000};

    auto system = C2cDmaSystem(ideal_ddr4(8, 2, 2, 4));
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
        kHemisphere, C2cDmaInstruction::Store(
            kDdr4Address, 1, hw::kPhysicalVectorBytes, kStream));
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

void test_eight_indexed_tx_lanes_store_to_ddr4()
{
    constexpr auto kHemisphere = Hemisphere::East;
    constexpr auto kLaneCount = std::size_t {8};
    constexpr auto kSourceRow = std::size_t {91};
    constexpr auto kDdr4Base = std::uint64_t {0x20000};
    auto system = C2cDmaSystem(
        ideal_ddr4(32, 2, 2, 128, kLaneCount));
    auto& chip = system.chip();

    for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
        const auto sourceSlice = std::size_t {36} + lane;
        const auto expected = make_vector(
            static_cast<std::uint8_t>(0x20 + lane * 7), 300 + lane);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t byte = 0; byte < hw::kLanesPerTile; ++byte)
                chip.initialize_mem_sram_lane_byte(kHemisphere,
                    sourceSlice, tile, kSourceRow, byte,
                    expected.payload[tile][byte]);

        const auto queue = InstructionControlUnit::mem_queue(
            kHemisphere, sourceSlice);
        chip.icu().enqueue_mem_nop(queue, kLaneCount);
        chip.icu().enqueue_mem(queue,
            MemInstruction::Read(
                kSourceRow, StreamId::East(lane)));
        chip.icu().enqueue_c2c_send(kHemisphere, lane, 1, lane);
        chip.icu().enqueue_c2c_dma(kHemisphere,
            C2cDmaInstruction::Store(
                kDdr4Base + lane * hw::kPhysicalVectorBytes,
                1, hw::kPhysicalVectorBytes, lane));
    }

    for (std::size_t cycle = 0; cycle < 80; ++cycle) system.tick();

    for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
        const auto expected = make_vector(
            static_cast<std::uint8_t>(0x20 + lane * 7), 300 + lane);
        const auto actual = system.ddr4().read_vector(
            kDdr4Base + lane * hw::kPhysicalVectorBytes);
        if (actual.payload != expected.payload)
            throw std::runtime_error(
                "indexed C2C TX lane stored the wrong payload: lane="
                + std::to_string(lane)
                + " actual0=" + std::to_string(actual.payload[0][0])
                + " expected0=" + std::to_string(expected.payload[0][0])
                + " tx_idle=" + std::to_string(
                    chip.c2c_endpoint(kHemisphere).tx().idle())
                + " tx_queued=" + std::to_string(
                    chip.c2c_endpoint(kHemisphere).tx()
                        .queued_instruction_count())
                + " dma_idle=" + std::to_string(
                    system.dma(kHemisphere).idle())
                + " dma_outbound=" + std::to_string(
                    system.dma(kHemisphere).outbound_queue_size(lane))
                + " mem_done=" + std::to_string(
                    chip.icu().mem_iq(InstructionControlUnit::mem_queue(
                        kHemisphere, 36 + lane)).done()));
    }
}

void test_sixteen_lane_shared_ingress_sustains_qwen_page()
{
    constexpr std::size_t kVectorsPerLane = 768;
    constexpr std::size_t kLaneCount = 8;
    constexpr std::size_t kFabricBase = 24;
    auto system = C2cDmaSystem {};
    auto& chip = system.chip();

    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            const auto slice = std::size_t {20} + lane;
            const auto ddrBase =
                (side * kLaneCount + lane) * kVectorsPerLane
                * hw::kPhysicalVectorBytes;
            for (std::size_t vector = 0;
                 vector < kVectorsPerLane; ++vector)
                system.ddr4().initialize_vector(
                    ddrBase + vector * hw::kPhysicalVectorBytes,
                    make_vector(static_cast<std::uint8_t>(
                        side * 64 + lane * 4 + vector), vector));

            chip.icu().enqueue_c2c_dma(hemisphere,
                C2cDmaInstruction::Load(ddrBase, kVectorsPerLane,
                    hw::kPhysicalVectorBytes, 0, lane));
            chip.icu().enqueue_c2c_receive(hemisphere, lane,
                hemisphere, slice, 1, true, 0,
                kVectorsPerLane, 1, kFabricBase + lane);
            const auto queue = InstructionControlUnit::mem_queue(
                hemisphere, slice, 1);
            const auto group = slice / hw::kMemSlicesPerGroup;
            const auto transport =
                hw::kMemEastBoundaryStreamRegisterColumn - (group + 1);
            chip.icu().enqueue_c2c_mem_stream_write(queue,
                MemInstruction::Write(
                    0, StreamId::West(kFabricBase + lane)),
                kVectorsPerLane, transport, 1);
        }
    }

    std::size_t cycles = 0;
    for (; cycles < 10000; ++cycles) {
        system.tick();
        bool done = system.ddr4().idle();
        for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
            const auto hemisphere = static_cast<Hemisphere>(side);
            done = done && system.dma(hemisphere).idle()
                && chip.c2c_endpoint(hemisphere).rx().idle();
            for (std::size_t lane = 0; lane < kLaneCount; ++lane)
                done = done && chip.icu().c2c_mem_iq(
                    InstructionControlUnit::mem_queue(
                        hemisphere, 20 + lane, 1)).done();
        }
        if (done) break;
    }
    if (cycles >= 4400)
        throw std::runtime_error(
            "16-lane shared C2C page missed its bandwidth bound: cycles="
            + std::to_string(cycles));
    for (std::size_t side = 0; side < hw::kHemispheres; ++side)
        for (std::size_t lane = 0; lane < kLaneCount; ++lane)
            require(chip.c2c_endpoint(
                        static_cast<Hemisphere>(side))
                        .rx().completed_instruction_count(lane)
                    == 1,
                "shared C2C RX did not publish burst completion");
}

} // namespace

int main()
try {
    test_ddr4_transfers_one_beat_per_cycle();
    test_ddr4_exposes_eight_vector_channels();
    test_default_ddr4_bandwidth_and_latency_jitter();
    test_c2c_stream_count_is_runtime_selectable();
    test_ddr4_dma_rx_sr_mem();
    test_eight_shared_c2c_lanes_write_through_mem();
    test_mem_sr_tx_dma_ddr4();
    test_eight_indexed_tx_lanes_store_to_ddr4();
    test_sixteen_lane_shared_ingress_sustains_qwen_page();
    return 0;
} catch (const std::exception& error) {
    std::cerr << "c2c_dma_ddr4_test failed: "
              << error.what() << '\n';
    return 1;
}
