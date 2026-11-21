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

// Keep all eight lanes in distinct legal slices; preserve group-distance tests.
static_assert(hw::kMemGroups >= 4);
constexpr auto kMiddleLaneBase = hw::kMemSlicesPerGroup * (hw::kMemGroups / 2 - 1);
constexpr auto kTxLaneBase = hw::kMemSliceColumns - 4 * hw::kMemSlicesPerGroup;

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
    config.beat_bytes = std::min(beat_bytes,hw::kPhysicalVectorBytes);
    config.read_latency_cycles = read_latency;
    config.write_latency_cycles = write_latency;
    config.request_queue_depth = queue_depth;
    config.transfer_channels = channels;
    config.peak_bandwidth_bytes_per_second =
        config.lpu_clock_hz * config.beat_bytes * channels;
    config.read_latency_jitter_cycles = 0;
    config.write_latency_jitter_cycles = 0;
    return config;
}

void test_ddr4_transfers_one_beat_per_cycle()
{
    Ddr4Model ddr(ideal_ddr4(8,2,2,2));const auto expected=make_vector(0x20);
    const auto id=ddr.request_write(0x800,expected);std::size_t committed=0,cycles=0;
    while(!ddr.write_completion_ready(id)) {
        require(++cycles<100,"AXI write did not retire");ddr.tick();
        if(ddr.last_beat()) {
            require(ddr.last_beat()->write && ddr.last_beat()->address==0x800+committed
                && ddr.last_beat()->byte_count==8,"AXI write beat/ordering mismatch");
            committed+=8;
            require(!ddr.write_completion_ready(id),"write retired before B response");
        }
        const auto partial=ddr.read_vector(0x800);
        for(std::size_t b=0;b<hw::kPhysicalVectorBytes;++b)
            require(partial.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]
                ==(b<committed?expected.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]:0),
                "DDR byte committed without W handshake");
    }
    require(committed==hw::kPhysicalVectorBytes,"write returned short");
    ddr.pop_write_completion(id);require(ddr.drained(),"write credit leaked");
}

void test_ddr4_exposes_eight_vector_channels()
{
    Ddr4Model ddr(ideal_ddr4(32,2,2,64,8));std::array<Ddr4Model::RequestId,8> ids{};
    std::array<bool,8> done{};
    for(std::size_t c=0;c<8;++c) {ddr.initialize_vector(0x4000+c*hw::kPhysicalVectorBytes,make_vector(c*16));
        ids[c]=ddr.request_read(0x4000+c*hw::kPhysicalVectorBytes,2*c);}
    std::size_t retired=0;
    for(std::size_t t=0;retired<8;++t) {
        require(t<100,"eight AXI clients did not progress");const auto before=ddr.read_bytes_transferred();ddr.tick();
        require(ddr.read_bytes_transferred()-before<=32,"one R channel transferred more than one beat");
        for(std::size_t c=0;c<8;++c)if(!done[c] && ddr.read_completion_ready(ids[c])) {
            require(ddr.pop_read_completion(ids[c]).vector.payload==make_vector(c*16).payload,"RID routing error");
            done[c]=true;++retired;
        }
    }
    require(ddr.axi().counters().ar==8 && ddr.drained(),"eight-client accounting");
}

void test_default_ddr4_bandwidth_and_latency_jitter()
{
    auto config=Ddr4Config{};config.peak_bandwidth_bytes_per_second=config.lpu_clock_hz*7;
    auto run=[&] {
        Ddr4Model ddr(config);std::array<Ddr4Model::RequestId,16> ids{};std::array<std::size_t,16> times{};
        for(std::size_t i=0;i<ids.size();++i){ddr.initialize_vector(0x90000+i*hw::kPhysicalVectorBytes,make_vector(i*3));
            ids[i]=ddr.request_read(0x90000+i*hw::kPhysicalVectorBytes,2*i);}
        std::size_t left=16;
        for(std::size_t t=1;left;++t) {
            require(t<2048,"rate-limited AXI requests starved");const auto before=ddr.read_bytes_transferred();ddr.tick();
            require(ddr.read_bytes_transferred()<=t*7 && ddr.read_bytes_transferred()-before<=config.beat_bytes,
                "DDR token budget or physical R bandwidth exceeded");
            for(std::size_t i=0;i<ids.size();++i)if(!times[i] && ddr.read_completion_ready(ids[i])) {
                require(t>=config.read_latency_cycles+2,"read beat preceded address plus response latency");
                require(ddr.pop_read_completion(ids[i]).vector.payload==make_vector(i*3).payload,"jitter corrupted payload");
                times[i]=t;--left;
            }
        }
        require(ddr.drained(),"read credits leaked");return times;
    };
    require(run()==run(),"seeded AXI latency not reproducible");
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

void test_unified_mem_icu_insertion_and_transport_only_tick()
{
    using MemIcu = InstructionControlUnit::MemIcu;
    constexpr std::size_t kSyncTag = 77;
    constexpr std::size_t kTransportDelay = 2;
    constexpr std::size_t kReservationCycles = 8;
    const auto packet = MemIcu::encode_synchronized_raw_packet(
        2, kSyncTag, kTransportDelay, 3,
        MemInstruction::Write(40, StreamId::West(29)),
        kReservationCycles);

    MemIcu queue;
    queue.push_instruction(
        MemInstruction::Write(10, StreamId::West(28)));
    queue.push_instruction(
        MemInstruction::Write(90, StreamId::West(30)));
    queue.insert_encoded_synchronized_packet(1, packet);
    require(queue.imem_occupancy() == 4,
        "MEM_WRITE_SYNC insertion did not occupy two words in the one MEM i-MEM");

    const auto early = queue.tick();
    require(early.has_value() && early->address == 10,
        "unified MEM ICU did not issue the instruction before MEM_WRITE_SYNC");
    require(queue.cycle() == 1,
        "normal MEM tick did not advance queue-local logical time");

    queue.notify(kSyncTag + 1);
    require(!queue.tick_transport_only().has_value()
            && queue.last_trace().action
                == IcuQueueAction::SynchronizedWait,
        "MEM_WRITE_SYNC accepted a notification with the wrong tag");
    require(queue.cycle() == 1,
        "transport-only wait advanced queue-local logical time");

    queue.notify(kSyncTag);
    queue.notify(kSyncTag);
    for (std::size_t delay = 0; delay < kTransportDelay + 1; ++delay) {
        require(!queue.tick_transport_only().has_value(),
            "MEM_WRITE_SYNC issued before its physical transport delay");
    }
    const auto firstWrite = queue.tick_transport_only();
    require(firstWrite.has_value() && firstWrite->address == 40,
        "MEM_WRITE_SYNC did not issue its first delayed FU write");
    const auto secondWrite = queue.tick_transport_only();
    require(secondWrite.has_value() && secondWrite->address == 43,
        "MEM_WRITE_SYNC did not apply its address stride");
    require(queue.synchronized_issued_count() == 2,
        "unified MEM ICU did not count synchronized FU writes separately");

    for (std::size_t hold = 0; hold < 2; ++hold) {
        require(!queue.tick_transport_only().has_value()
                && queue.last_trace().action
                    == IcuQueueAction::SynchronizedDelay,
            "MEM_WRITE_SYNC did not hold its reserved single-ICU window");
    }

    require(!queue.tick_transport_only().has_value()
            && queue.last_trace().action == IcuQueueAction::ProgramPaused,
        "transport-only mode issued an ordinary MEM command after the transfer");
    require(queue.cycle() == 1,
        "transport-only execution advanced the ordinary MEM logical cycle");
    const auto late = queue.tick();
    require(late.has_value() && late->address == 90,
        "unified MEM ICU did not preserve the instruction after MEM_WRITE_SYNC");

    const auto encodedWait = MemIcu::encode_control_raw_word(
        IcuControlInstruction::WaitEvent(0x1234));
    const auto decodedWait = MemIcu::decode_control_raw_word(encodedWait);
    require(decodedWait.opcode == IcuControlOpcode::WaitEvent
            && decodedWait.event_tag == 0x1234,
        "public raw control codec did not preserve WAIT_EVENT");

    MemIcu boundaryQueue;
    boundaryQueue.push_encoded_synchronized_packet(packet);
    const auto occupancy = boundaryQueue.imem_occupancy();
    bool splitRejected = false;
    try {
        boundaryQueue.insert_encoded_synchronized_packet(1, packet);
    } catch (const StaticScheduleError&) {
        splitRejected = true;
    }
    require(splitRejected && boundaryQueue.imem_occupancy() == occupancy,
        "MEM_WRITE_SYNC insertion split a synchronized transport packet or changed i-MEM on failure");

    bool launchedRejected = false;
    try {
        queue.insert_encoded_synchronized_packet(
            queue.imem_occupancy(), packet);
    } catch (const StaticScheduleError&) {
        launchedRejected = true;
    }
    require(launchedRejected,
        "MEM_WRITE_SYNC insertion modified i-MEM after launch");

    // When data arrives early during normal execution, the synchronized
    // command must still consume its compiler-reserved queue window. The next
    // ordinary command therefore starts at the same logical cycle regardless
    // of DDR response time.
    constexpr std::size_t kEarlyTag = 91;
    constexpr std::size_t kEarlyWindow = 6;
    const auto earlyPacket = MemIcu::encode_synchronized_raw_packet(
        2, kEarlyTag, 0, 1,
        MemInstruction::Write(120, StreamId::West(27)), kEarlyWindow);
    MemIcu reservedQueue;
    reservedQueue.push_encoded_synchronized_packet(earlyPacket);
    reservedQueue.push_instruction(
        MemInstruction::Read(150, StreamId::East(1)));
    reservedQueue.configure_all();
    for (std::size_t cycle = 0;
         cycle < MemIcu::fetch_latency
                 + MemIcu::synchronized_packet_word_count;
         ++cycle)
        reservedQueue.prefetch_only();
    reservedQueue.notify(kEarlyTag);
    reservedQueue.notify(kEarlyTag);
    std::size_t activeSyncCycles = 0;
    std::optional<MemInstruction> afterReservation;
    for (std::size_t cycle = 0; cycle < 32; ++cycle) {
        const auto issue = reservedQueue.tick();
        const auto action = reservedQueue.last_trace().action;
        if (action == IcuQueueAction::SynchronizedWait
            || action == IcuQueueAction::SynchronizedDelay
            || action == IcuQueueAction::SynchronizedIssue)
            ++activeSyncCycles;
        if (issue.has_value() && issue->opcode == MemOpcode::Read) {
            afterReservation = issue;
            break;
        }
        require(!issue.has_value() || issue->opcode == MemOpcode::Write,
            "ordinary MEM command escaped the MEM_WRITE_SYNC reservation");
    }
    if (activeSyncCycles != kEarlyWindow)
        throw std::runtime_error(
            "MEM_WRITE_SYNC reserved-window cycles: expected "
            + std::to_string(kEarlyWindow) + ", got "
            + std::to_string(activeSyncCycles));
    require(afterReservation.has_value()
            && afterReservation->address == 150,
        "early MEM_WRITE_SYNC completion shifted the following MEM command");

    // A notification may arrive while an earlier local command still owns the
    // queue. Its transport delay must age relative to that arrival, without a
    // free-running absolute-cycle comparison inside the ICU.
    constexpr std::size_t kAgedTag = 92;
    const auto agedPacket = MemIcu::encode_synchronized_raw_packet(
        1, kAgedTag, kTransportDelay, 1,
        MemInstruction::Write(180, StreamId::West(26)), 1);
    MemIcu agedQueue;
    agedQueue.push_nop(kTransportDelay + 2);
    agedQueue.push_encoded_synchronized_packet(agedPacket);
    agedQueue.prefetch_only();
    agedQueue.notify(kAgedTag);
    for (std::size_t delay = 0; delay < kTransportDelay + 2; ++delay) {
        require(!agedQueue.tick().has_value(),
            "an early synchronized token bypassed the preceding local NOP");
    }
    const auto agedWrite = agedQueue.tick();
    require(agedWrite.has_value() && agedWrite->address == 180,
        "an early synchronized token did not age while waiting behind a NOP");

    // Same-tag tokens can arrive on different cycles. Each token carries its
    // own relative age so the second vector's transport delay overlaps the
    // first vector's delay and issue instead of being restarted serially.
    constexpr std::size_t kStaggeredTag = 93;
    const auto staggeredPacket = MemIcu::encode_synchronized_raw_packet(
        2, kStaggeredTag, kTransportDelay, 5,
        MemInstruction::Write(200, StreamId::West(25)), 8);
    MemIcu staggeredQueue;
    staggeredQueue.push_encoded_synchronized_packet(staggeredPacket);
    staggeredQueue.prefetch_only();
    staggeredQueue.notify(kStaggeredTag);
    require(!staggeredQueue.tick().has_value(),
        "first staggered token ignored its relative transport delay");
    staggeredQueue.notify(kStaggeredTag);
    require(!staggeredQueue.tick().has_value(),
        "first staggered token matured before D+1 relative cycles");
    require(!staggeredQueue.tick().has_value(),
        "first staggered token matured before D+1 relative cycles");
    const auto staggeredFirst = staggeredQueue.tick();
    const auto staggeredSecond = staggeredQueue.tick();
    require(staggeredFirst.has_value() && staggeredFirst->address == 200
            && staggeredSecond.has_value()
            && staggeredSecond->address == 205,
        "staggered synchronized tokens did not retain independent relative ages");
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
    chip.icu().enqueue_mem_synchronized_write(target_queue,
        MemInstruction::Write(
            kTargetRow, StreamId::West(kFabricStream)),
        1, kTransportNops);
    chip.icu().set_program_issue_enabled(false);

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
            || chip.icu().mem_iq(target_queue).last_trace().action
                == IcuQueueAction::SynchronizedIssue;
    }

    require(observed_dma_sync_wait,
        "DMA ICU did not wait for the DDR4 read to complete");
    require(observed_dma_sync_release,
        "DDR4 read completion did not release the DMA ICU");
    require(observed_mem_synchronized_issue,
        "shared C2C RX did not notify the target MEM ICU");
    require(chip.icu().mem_iq(target_queue).cycle() == 0,
        "transport-only MEM execution advanced logical program time");
    require(chip.icu().mem_iq(target_queue).last_trace().action
            == IcuQueueAction::ProgramPaused,
        "paused MEM ICU did not remain in transport-only mode");
    require(chip.icu().mem_iq(target_queue).synchronized_issued_count() == 1,
        "unified MEM ICU did not count its synchronized write");
    chip.icu().set_program_issue_enabled(true);
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
        const auto targetSlice = kMiddleLaneBase + lane;
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
        chip.icu().enqueue_mem_synchronized_write(queue,
            MemInstruction::Write(
                kTargetRow, StreamId::West(kFabricBase + lane)),
            1, transportNops);
    }

    for (std::size_t cycle = 0; cycle < 80; ++cycle) system.tick();

    for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
        const auto targetSlice = kMiddleLaneBase + lane;
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
    constexpr auto kSourceSlice = hw::kMemSliceColumns - hw::kMemSlicesPerGroup;
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

void test_shared_rx_waits_for_mem_write_sync_head()
{
    constexpr auto kHemisphere = Hemisphere::East;
    constexpr auto kStream = std::size_t {0};
    constexpr auto kFabricStream = std::size_t {24};
    constexpr auto kTargetSlice = kTxLaneBase;
    constexpr auto kTargetBank = std::size_t {1};
    constexpr auto kTargetRow = std::size_t {19};
    constexpr auto kDdr4Address = std::uint64_t {0x8000};
    constexpr auto kHeadDelay = std::size_t {96};

    auto system = C2cDmaSystem(ideal_ddr4(32, 2, 2, 64));
    auto& chip = system.chip();
    const auto expected = make_vector(0x5a, 303);
    system.ddr4().initialize_vector(kDdr4Address, expected);

    chip.icu().enqueue_c2c_dma(kHemisphere,
        C2cDmaInstruction::Load(kDdr4Address, 1,
            hw::kPhysicalVectorBytes, 303, kStream));
    chip.icu().enqueue_c2c_receive(kHemisphere, kStream,
        kHemisphere, kTargetSlice, kTargetBank, true, kTargetRow,
        1, 1, kFabricStream);

    const auto queue = InstructionControlUnit::mem_queue(
        kHemisphere, kTargetSlice, kTargetBank);
    chip.icu().mem_iq(queue).push_nop(kHeadDelay);
    const auto group = kTargetSlice / hw::kMemSlicesPerGroup;
    const auto transportDelay =
        hw::kMemEastBoundaryStreamRegisterColumn - (group + 1);
    chip.icu().enqueue_mem_synchronized_write(queue,
        MemInstruction::Write(
            kTargetRow, StreamId::West(kFabricStream)),
        1, transportDelay);

    for (std::size_t cycle = 0; cycle < 256; ++cycle)
        system.tick();

    require(chip.c2c_endpoint(kHemisphere).rx()
                .completed_instruction_count(kStream) == 1,
        "shared C2C RX did not resume when MEM_WRITE_SYNC reached the head");
    require(chip.icu().mem_iq(queue).synchronized_issued_count() == 1,
        "delayed MEM_WRITE_SYNC did not consume the held C2C vector");
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            require(chip.read_mem_sram_lane_byte(kHemisphere,
                        kTargetSlice, kTargetBank, tile, kTargetRow, lane)
                    == expected.payload[tile][lane],
                "C2C RX released a vector before MEM_WRITE_SYNC was ready");
}

void test_mem_read_sync_tx_dma_ddr4()
{
    constexpr auto kHemisphere = Hemisphere::East;
    constexpr auto kLane = std::size_t {3};
    constexpr auto kSourceSlice = kTxLaneBase;
    constexpr auto kSourceRow = std::size_t {41};
    constexpr auto kVectorCount = std::size_t {128};
    constexpr auto kSyncTag = std::uint32_t {0x8123};
    constexpr auto kDdr4Address = std::uint64_t {0x8000};

    auto system = C2cDmaSystem(ideal_ddr4(8, 2, 2, 16));
    auto& chip = system.chip();
    std::array<C2cVector, kVectorCount> expected{};
    for (std::size_t vector = 0; vector < kVectorCount; ++vector) {
        expected[vector] = make_vector(
            static_cast<std::uint8_t>(0x40 + vector * 0x20),
            500 + vector);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t byte = 0; byte < hw::kLanesPerTile; ++byte)
                chip.initialize_mem_sram_lane_byte(kHemisphere,
                    kSourceSlice, tile, kSourceRow + vector, byte,
                    expected[vector].payload[tile][byte]);
    }

    const auto sourceQueue = InstructionControlUnit::mem_queue(
        kHemisphere, kSourceSlice);
    chip.icu().enqueue_mem_synchronized_read(sourceQueue,
        MemInstruction::Read(kSourceRow, StreamId::East(kLane)),
        kVectorCount, kSyncTag, 0, 1, kVectorCount);
    chip.icu().enqueue_c2c_tx_raw(kHemisphere,
        C2cIcuPacketCodec::encode(C2cTxIcuInstruction::Send(
            kHemisphere, kLane, kLane, kVectorCount, kSyncTag,
            C2cMemNotifyRoute::Mem(
                kHemisphere, kSourceSlice, 0))));
    chip.icu().enqueue_c2c_dma(kHemisphere,
        C2cDmaInstruction::Store(kDdr4Address, kVectorCount,
            hw::kPhysicalVectorBytes, kLane));
    chip.icu().enqueue_control(IcuLocation::C2cDma(kHemisphere),
        IcuControlInstruction::Sync());

    bool ready = false;
    for (std::size_t cycle = 0; cycle < 4096; ++cycle) {
        system.tick();
        ready = chip.icu().mem_iq(sourceQueue).done()
            && chip.icu().c2c_tx_iq(kHemisphere).done()
            && chip.icu().c2c_dma_iq(kHemisphere).done()
            && chip.c2c_endpoint(kHemisphere).tx().idle()
            && system.dma(kHemisphere).idle() && system.ddr4().idle();
        if (ready) break;
    }
    if (!ready)
        throw std::runtime_error(
            "MEM_READ_SYNC page-out did not retire: mem_done="
            + std::to_string(chip.icu().mem_iq(sourceQueue).done())
            + " mem_issued=" + std::to_string(
                chip.icu().mem_iq(sourceQueue).synchronized_issued_count())
            + " tx_iq_done=" + std::to_string(
                chip.icu().c2c_tx_iq(kHemisphere).done())
            + " tx_idle=" + std::to_string(
                chip.c2c_endpoint(kHemisphere).tx().idle())
            + " dma_iq_done=" + std::to_string(
                chip.icu().c2c_dma_iq(kHemisphere).done())
            + " dma_idle=" + std::to_string(system.dma(kHemisphere).idle())
            + " ddr_idle=" + std::to_string(system.ddr4().idle()));
    require(chip.icu().mem_iq(sourceQueue).synchronized_issued_count()
            == kVectorCount,
        "MEM_READ_SYNC did not issue one MEM read per C2C vector token");
    for (std::size_t vector = 0; vector < kVectorCount; ++vector)
        require(system.ddr4().read_vector(
                    kDdr4Address + vector * hw::kPhysicalVectorBytes)
                    .payload == expected[vector].payload,
            "MEM_READ_SYNC -> C2C TX -> DMA -> DDR4 changed data");
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
        const auto sourceSlice = kTxLaneBase + lane;
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
                        kHemisphere, kTxLaneBase + lane)).done()));
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
            const auto slice = kMiddleLaneBase + lane;
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
            chip.icu().enqueue_mem_synchronized_write(queue,
                MemInstruction::Write(
                    0, StreamId::West(kFabricBase + lane)),
                kVectorsPerLane, transport, 1);
        }
    }

    std::size_t cycles = 0;
    // A single 32-byte R channel replaces the old 16 parallel vector returns.
    constexpr auto minimumBeats=hw::kHemispheres*kLaneCount*kVectorsPerLane
        *hw::kPhysicalVectorBytes/hw::kC2cBytesPerStreamPerCycle;
    for (; cycles < 2*minimumBeats+256; ++cycles) {
        system.tick();
        bool done = system.ddr4().idle();
        for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
            const auto hemisphere = static_cast<Hemisphere>(side);
            done = done && system.dma(hemisphere).idle()
                && chip.c2c_endpoint(hemisphere).rx().idle();
            for (std::size_t lane = 0; lane < kLaneCount; ++lane)
                done = done && chip.icu().mem_iq(
                    InstructionControlUnit::mem_queue(
                        hemisphere, kMiddleLaneBase + lane, 1)).done();
        }
        if (done) break;
    }
    require(cycles>=minimumBeats,"physical AXI R channel exceeded one beat per cycle");
    if (cycles >= 2*minimumBeats+256)
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

void test_vector_crosses_4k() {
    Ddr4Model ddr(ideal_ddr4(hw::kPhysicalVectorBytes/2,2,2,4));
    const auto address=4096-ddr.config().beat_bytes;
    const auto expected=make_vector(71);
    ddr.initialize_vector(address,expected);
    const auto read=ddr.request_read(address);
    for(std::size_t t=0;!ddr.read_completion_ready(read);++t) {
        require(t<200,"split read timed out");ddr.tick();
    }
    require(ddr.pop_read_completion(read).vector.payload==expected.payload
        && ddr.axi().counters().ar==2,"4K read fragments not reassembled exactly once");
    const auto value=make_vector(123);
    const auto write=ddr.request_write(address,value);
    for(std::size_t t=0;!ddr.write_completion_ready(write);++t) {
        require(t<200,"split write timed out");ddr.tick();
    }
    ddr.pop_write_completion(write);
    require(ddr.read_vector(address).payload==value.payload && ddr.axi().counters().aw==2
        && ddr.axi().counters().b==2 && ddr.drained(),"4K write retired before both B responses");
}
void test_read_write_service_fairness() {
    Ddr4Model ddr(ideal_ddr4(hw::kPhysicalVectorBytes,0,0,64));
    for(std::size_t n=0;n<32;++n)ddr.request_read(0x10000+n*hw::kPhysicalVectorBytes,0);
    const auto write=ddr.request_write(0x20000,make_vector(17),1);
    for(std::size_t n=0;!ddr.write_completion_ready(write);++n) {
        require(n<8,"continuous R traffic starved W/B service");ddr.tick();
    }
    require(ddr.pop_write_completion(write).address==0x20000,"mixed-service write owner");
}

void test_long_sr_store_reserves_complete_returns() {
    constexpr std::size_t vectors=32, lanes=8;
    C2cDmaSystem system(ideal_ddr4(8,2,2,16),2);
    auto& chip=system.chip();
    for(std::size_t side=0;side<2;++side)for(std::size_t lane=0;lane<lanes;++lane) {
        const auto h=static_cast<Hemisphere>(side);const auto slice=8+lane;
        const auto base=0x40000+(side*lanes+lane)*vectors*hw::kPhysicalVectorBytes;
        for(std::size_t v=0;v<vectors;++v) {
            const auto data=make_vector(static_cast<std::uint8_t>(side*79+lane*11+v*37));
            for(std::size_t t=0;t<hw::kTileRows;++t)for(std::size_t b=0;b<hw::kLanesPerTile;++b)
                chip.initialize_mem_sram_lane_byte(h,slice,t,64+v,b,data.payload[t][b]);
        }
        const auto tag=static_cast<std::uint32_t>(0x9000+lane);
        chip.icu().enqueue_mem_synchronized_read(InstructionControlUnit::mem_queue(h,slice),
            MemInstruction::Read(64,StreamId::East(lane)),vectors,tag,0,1,vectors);
        chip.icu().enqueue_c2c_tx_raw(h,C2cIcuPacketCodec::encode(C2cTxIcuInstruction::Send(
            h,lane,lane,vectors,tag,C2cMemNotifyRoute::Mem(h,slice,0))));
        chip.icu().enqueue_c2c_dma(h,C2cDmaInstruction::Store(base,vectors,hw::kPhysicalVectorBytes,lane));
    }
    bool done=false;
    for(std::size_t t=0;t<20000;++t) {
        system.tick();done=system.ddr4().drained();
        for(std::size_t side=0;side<2;++side) {
            const auto h=static_cast<Hemisphere>(side);
            done &= system.dma(h).idle() && chip.c2c_endpoint(h).tx().idle()
                && chip.icu().c2c_dma_iq(h).done();
        }
        if(done)break;
    }
    require(done,"long SR STORE lost an admitted vector under DDR pressure");
    for(std::size_t side=0;side<2;++side)for(std::size_t lane=0;lane<lanes;++lane)
        for(std::size_t v=0;v<vectors;++v) {
            const auto address=0x40000+((side*lanes+lane)*vectors+v)*hw::kPhysicalVectorBytes;
            const auto expected=make_vector(static_cast<std::uint8_t>(side*79+lane*11+v*37));
            require(system.ddr4().read_vector(address).payload==expected.payload,
                "long SR STORE reordered/overwrote a returned vector");
        }
}

int main()
try {
    // Legal bus errors must retire accepted bursts without false success,
    // and explicit recovery must not reset time or undo external writes.
    for(const bool write:{false,true}) {
        Ddr4Model ddr(ideal_ddr4(8,2,2,16));C2cDmaEngine dma(ddr,4);
        ddr.set_response_fault([](auto,bool){return std::uint8_t{2};});
        const auto data=make_vector(91);
        dma.issue(write?C2cDmaInstruction::Store(0x10000,4):C2cDmaInstruction::Load(0x10000,4));
        if(write)for(std::size_t n=0;n<4;++n)dma.send(data);
        for(std::size_t n=0;!dma.error_drained();++n) {
            require(n<1000,"error did not drain accepted AXI bursts");dma.tick();ddr.tick();
        }
        require(ddr.drained() && !dma.take_completion_notification() && !dma.can_issue(),"failed DMA reported success/leaked credits");
        if(write)require(ddr.read_vector(0x10000).payload==data.payload,"B error incorrectly rolled back DDR bytes");
        const auto cycle=dma.cycle();dma.recover_error();ddr.set_response_fault({});
        require(dma.cycle()==cycle && dma.can_issue(),"recovery reset time or left admission stopped");
        dma.issue(C2cDmaInstruction::Load(0x10000,1));
        for(std::size_t n=0;!dma.receive_ready();++n) {
            require(n<1000,"DMA retry stalled");dma.tick();ddr.tick();
        }
        dma.pop_received();require(dma.take_completion_notification(),"new descriptor failed after recovery");
    }
    {
        C2cTxSlice tx(C2cStreamPortMap::EastEdge(0).tx_input,"bounded",true);
        for(std::size_t n=0;n<C2cTxSlice::kDescriptorSlots;++n)tx.issue(C2cInstruction::Send(0,1000000));
        require(tx.queued_instruction_count()==C2cTxSlice::kDescriptorSlots && !tx.can_issue(),"TX expanded vectors or exceeded bounded descriptor count");
        tx.reset();require(tx.can_issue(),"TX descriptor retirement failed");
    }
    test_vector_crosses_4k();
    test_read_write_service_fairness();
    test_long_sr_store_reserves_complete_returns();
    test_ddr4_transfers_one_beat_per_cycle();
    test_ddr4_exposes_eight_vector_channels();
    test_default_ddr4_bandwidth_and_latency_jitter();
    test_c2c_stream_count_is_runtime_selectable();
    test_unified_mem_icu_insertion_and_transport_only_tick();
    test_ddr4_dma_rx_sr_mem();
    test_eight_shared_c2c_lanes_write_through_mem();
    test_shared_rx_waits_for_mem_write_sync_head();
    test_mem_sr_tx_dma_ddr4();
    test_mem_read_sync_tx_dma_ddr4();
    test_eight_indexed_tx_lanes_store_to_ddr4();
    test_sixteen_lane_shared_ingress_sustains_qwen_page();
    return 0;
} catch (const std::exception& error) {
    std::cerr << "c2c_dma_ddr4_test failed: "
              << error.what() << '\n';
    return 1;
}
