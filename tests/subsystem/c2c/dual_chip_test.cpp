#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/system/dual_chip_c2c_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu {
struct C2cLinkBoundaryFixture {
    static void seed_empty_link(C2cLink& link, std::uint64_t epoch, std::uint64_t sequence) {
        link.epoch_ = epoch;
        link.current_.issued = link.current_.rx_next = sequence;
    }
};
}

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

C2cVector make_vector(std::uint8_t base, std::uint64_t tag)
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

class VectorTransport {
public:
    bool can_send() const noexcept { return true; }
    void send(C2cVector vector) { sent.push_back(std::move(vector)); }
    bool receive_ready() const noexcept { return !received.empty(); }
    C2cVector pop_received()
    {
        auto vector = std::move(received.front());
        received.pop_front();
        return vector;
    }

    std::deque<C2cVector> received{};
    std::vector<C2cVector> sent{};
};

void test_rx_diagonal_pipeline_accepts_one_vector_per_cycle()
{
    constexpr auto kColumn = 0U;
    constexpr auto kStream = 5U;
    constexpr auto kVectorCount = std::size_t {4};
    auto fabric = StreamRegisterFabric(1);
    auto rx = C2cRxSlice(
        C2cStreamPortMap::OutputEndpoint {
            kColumn, StreamDirection::West});
    auto transport = VectorTransport {};

    for (std::size_t vector = 0; vector < kVectorCount; ++vector) {
        transport.received.push_back(make_vector(
            static_cast<std::uint8_t>(vector * 0x20), vector));
        rx.issue(C2cInstruction::Receive(
            kStream, Hemisphere::East, 0));
    }

    for (std::size_t cycle = 0; cycle < kVectorCount; ++cycle) {
        fabric.low();
        rx.evaluate(fabric, transport);
        rx.validate_posedge();
        fabric.validate_posedge();
        rx.posedge();
        fabric.posedge();

        require(
            transport.received.size() == kVectorCount - cycle - 1,
            "C2C RX did not accept one complete 32-byte vector per cycle");
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto valid = tile <= cycle;
            require(
                fabric.segment_valid(
                    kColumn, tile, StreamId::West(kStream)) == valid,
                "C2C RX diagonal pipeline produced an invalid tile pattern");
            if (!valid) continue;

            const auto source_vector = cycle - tile;
            const auto segment = fabric.segment(
                kColumn, tile, StreamId::West(kStream));
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto expected = static_cast<std::uint8_t>(
                    source_vector * 0x20
                    + tile * hw::kLanesPerTile + lane);
                require(
                    segment[lane].data == expected
                        && segment[lane].vector_tag == source_vector,
                    "C2C RX diagonal pipeline selected the wrong vector");
            }
        }
    }
}

void test_tx_diagonal_pipeline_emits_one_vector_per_cycle()
{
    constexpr auto kColumn = 0U;
    constexpr auto kStream = 7U;
    constexpr auto kVectorCount = std::size_t {4};
    auto fabric = StreamRegisterFabric(1);
    auto tx = C2cTxSlice(
        C2cStreamPortMap::InputEndpoint {
            kColumn, StreamDirection::East});
    auto transport = VectorTransport {};

    for (std::size_t vector = 0; vector < kVectorCount; ++vector) {
        tx.issue(C2cInstruction::Send(kStream));
    }

    for (std::size_t cycle = 0;
         cycle < kVectorCount + hw::kTileRows - 1;
         ++cycle) {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (cycle < tile || cycle - tile >= kVectorCount) continue;
            const auto source_vector = cycle - tile;
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                fabric.initialize_cell(
                    kColumn,
                    tile,
                    lane,
                    StreamId::East(kStream),
                    StreamCell::Valid(
                        static_cast<std::uint8_t>(
                            source_vector * 0x20
                            + tile * hw::kLanesPerTile + lane),
                        lane + 1 == hw::kLanesPerTile,
                        source_vector));
            }
        }

        fabric.low();
        tx.evaluate(fabric, transport);
        tx.validate_posedge();
        fabric.validate_posedge();
        tx.posedge();
        fabric.posedge();

        const auto expected_completed = cycle + 1 < hw::kTileRows
            ? 0
            : std::min(
                kVectorCount,
                cycle - hw::kTileRows + 2);
        require(
            transport.sent.size() == expected_completed,
            "C2C TX did not emit one complete 32-byte vector per cycle");
    }

    require(tx.idle(), "C2C TX diagonal pipeline did not drain");
    for (std::size_t vector = 0; vector < kVectorCount; ++vector) {
        const auto expected = make_vector(
            static_cast<std::uint8_t>(vector * 0x20), vector);
        require(
            transport.sent[vector].payload == expected.payload
                && transport.sent[vector].vector_tag == vector,
            "C2C TX diagonal pipeline assembled the wrong vector");
    }
}

void test_link_credit_and_serialization()
{
    auto link = C2cLink(C2cLinkConfig {8, 2, 1});
    auto vector = C2cVector {};
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            vector.payload[tile][lane] = static_cast<std::uint8_t>(
                tile * hw::kLanesPerTile + lane);
        }
    }

    link.send(vector);
    require(!link.can_send(), "C2C accepted a second vector without credit");
    const auto arrival_cycles = (hw::kPhysicalVectorBytes + 7) / 8 + 2;
    for (std::size_t cycle = 1; cycle < arrival_cycles; ++cycle) {
        link.tick();
        require(!link.receive_ready(), "C2C vector arrived before serialization and flight completed");
    }
    link.tick();
    require(link.receive_ready(), "C2C vector did not reach the RX FIFO");
    const auto received = link.pop_received();
    require(received.payload == vector.payload, "C2C link changed vector payload bytes");
    require(link.can_send(), "C2C RX pop did not return link credit");
}

void test_dual_chip_rx_notifies_mem_and_writes_sram(bool reliable=false)
{
    constexpr auto kStream = 5U;
    constexpr auto kSourceSlice = hw::kMemSliceColumns - hw::kMemSlicesPerGroup;
    constexpr auto kTargetSlice = 16U;
    constexpr auto kSourceRow = 23U;
    constexpr auto kTargetRow = 41U;
    constexpr auto kTargetGroup =
        kTargetSlice / hw::kMemSlicesPerGroup;
    constexpr auto kRxToTargetNops =
        hw::kMemEastBoundaryStreamRegisterColumn
        - (kTargetGroup + 1) - 1;
    static_assert(kTargetGroup == 4);

    auto system = std::make_unique<DualChipC2cSystem>(C2cLinkConfig {
        hw::kPhysicalVectorBytes,
        0,
        2, reliable ? 3U : 0U, 2, 1,
    });
    if(reliable) system->link_0_to_1().set_peer_fault_schedule([](auto,auto,auto attempt){
        using F=C2cLink::FrameResult;
        return attempt==0?F::Corrupt:attempt==1?F::LostAck:F::Good;
    });
    auto& source = system->chip(0);
    auto& destination = system->chip(1);

    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            source.initialize_mem_sram_lane_byte(
                Hemisphere::East,
                kSourceSlice,
                tile,
                kSourceRow,
                lane,
                static_cast<std::uint8_t>(0x40 + tile * 8 + lane));
        }
    }

    const auto source_queue = InstructionControlUnit::mem_queue(
        Hemisphere::East, kSourceSlice);
    const auto target_queue = InstructionControlUnit::mem_queue(
        Hemisphere::West, kTargetSlice);

    source.icu().enqueue_mem(
        source_queue,
        MemInstruction::Read(kSourceRow, StreamId::East(kStream)));
    source.icu().enqueue_c2c_send(Hemisphere::East, kStream);

    destination.icu().enqueue_c2c_receive(
        Hemisphere::West,
        kStream, Hemisphere::West, kTargetSlice);
    destination.icu().enqueue_control(
        IcuLocation::Mem(Hemisphere::West, kTargetSlice),
        IcuControlInstruction::Sync());
    destination.icu().enqueue_mem_nop(
        target_queue,
        kRxToTargetNops);
    destination.icu().enqueue_mem(
        target_queue,
        MemInstruction::Write(kTargetRow, StreamId::West(kStream)));

    bool observed_sync_wait = false;
    bool observed_sync_release = false;
    const auto transfer_cycles = 16 + 2 * hw::kTileRows + (reliable ? 24 : 0);
    for (std::size_t cycle = 0; cycle < transfer_cycles; ++cycle) {
        system->tick();
        const auto action = destination.icu().mem_iq(target_queue).last_trace().action;
        observed_sync_wait = observed_sync_wait
            || action == IcuQueueAction::SyncWait;
        observed_sync_release = observed_sync_release
            || action == IcuQueueAction::SyncRelease;
    }

    if(reliable) require(system->link_0_to_1().peer_retries()==2 &&
        system->link_0_to_1().peer_duplicates()==1 && !system->link_0_to_1().receive_ready()
        && !system->link_0_to_1().awaiting_ack() && !system->link_0_to_1().peer_error(), "reliable peer did not retire exactly once");
    require(observed_sync_wait, "destination MEM did not wait on C2C receive");
    require(observed_sync_release, "C2C receive did not notify destination MEM ICU");
    require(source.c2c_endpoint(Hemisphere::East).tx().idle(),
        "C2C TX did not finish four-tile gather");
    require(destination.c2c_endpoint(Hemisphere::West).rx().idle(),
        "C2C RX did not finish four-tile replay");

    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto expected = static_cast<std::uint8_t>(
                0x40 + tile * hw::kLanesPerTile + lane);
            const auto actual = destination.read_mem_sram_lane_byte(
                Hemisphere::West,
                kTargetSlice,
                tile,
                kTargetRow,
                lane);
            if (actual != expected) {
                throw std::runtime_error(
                    "C2C RX -> SR -> MEM wrote an incorrect byte at tile "
                    + std::to_string(tile) + ", lane "
                    + std::to_string(lane));
            }
        }
    }
}

void test_dual_hemisphere_endpoints_transfer_in_parallel()
{
    constexpr auto kStream = 7U;
    constexpr auto kSourceSlice = hw::kMemSliceColumns - hw::kMemSlicesPerGroup;
    constexpr auto kTargetSlice = 16U;
    constexpr auto kTargetGroup =
        kTargetSlice / hw::kMemSlicesPerGroup;
    constexpr auto kRxToTargetNops =
        hw::kMemEastBoundaryStreamRegisterColumn
        - (kTargetGroup + 1) - 1;
    constexpr std::array kSourceRows {51U, 52U};
    constexpr std::array kTargetRows {61U, 62U};

    auto system = std::make_unique<DualChipC2cSystem>(C2cLinkConfig {
        hw::kPhysicalVectorBytes,
        0,
        2,
    });
    auto& source = system->chip(0);
    auto& destination = system->chip(1);

    for (std::size_t source_index = 0; source_index < hw::kHemispheres;
         ++source_index) {
        const auto source_hemisphere =
            static_cast<Hemisphere>(source_index);
        const auto destination_hemisphere = source_hemisphere == Hemisphere::East
            ? Hemisphere::West
            : Hemisphere::East;

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                source.initialize_mem_sram_lane_byte(
                    source_hemisphere,
                    kSourceSlice,
                    tile,
                    kSourceRows[source_index],
                    lane,
                    static_cast<std::uint8_t>(
                        0x20 + source_index * 0x40 + tile * 8 + lane));
            }
        }

        const auto source_queue = InstructionControlUnit::mem_queue(
            source_hemisphere, kSourceSlice);
        const auto target_queue = InstructionControlUnit::mem_queue(
            destination_hemisphere, kTargetSlice);
        source.icu().enqueue_mem(
            source_queue,
            MemInstruction::Read(
                kSourceRows[source_index], StreamId::East(kStream)));
        source.icu().enqueue_c2c_send(source_hemisphere, kStream);
        destination.icu().enqueue_c2c_receive(
            destination_hemisphere,
            kStream,
            destination_hemisphere,
            kTargetSlice);
        destination.icu().enqueue_control(
            IcuLocation::Mem(destination_hemisphere, kTargetSlice),
            IcuControlInstruction::Sync());
        destination.icu().enqueue_mem_nop(target_queue, kRxToTargetNops);
        destination.icu().enqueue_mem(
            target_queue,
            MemInstruction::Write(
                kTargetRows[source_index], StreamId::West(kStream)));
    }

    for (std::size_t cycle = 0; cycle < 16 + 2 * hw::kTileRows; ++cycle)
        system->tick();

    for (std::size_t source_index = 0; source_index < hw::kHemispheres;
         ++source_index) {
        const auto destination_hemisphere = source_index == 0
            ? Hemisphere::West
            : Hemisphere::East;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto expected = static_cast<std::uint8_t>(
                    0x20 + source_index * 0x40 + tile * 8 + lane);
                require(
                    destination.read_mem_sram_lane_byte(
                        destination_hemisphere,
                        kTargetSlice,
                        tile,
                        kTargetRows[source_index],
                        lane) == expected,
                    "parallel C2C hemisphere transfer wrote an incorrect byte");
            }
        }
    }
}

void test_reliable_window() {
    using F=C2cLink::FrameResult;
    C2cLink link({hw::kPhysicalVectorBytes/2,2,3,5,2,3});
    link.set_peer_fault_schedule([](auto,auto seq,auto attempt){
        return seq%2==0 && !attempt?F::LostAck:(!attempt?F::Lost:F::Good);
    });
    std::size_t sent=0,received=0;
    for(unsigned t=0;t<512;++t) {
        if(sent<10 && link.can_send()) {C2cVector v{}; v.vector_tag=sent++;link.send(v);}
        if(t==80) require(sent==3 && !link.can_send() && link.receive_queue_size()==3
            && !link.awaiting_ack(),"ACK improperly returned receive credit");
        if(t>80 && t%4==0 && link.receive_ready())
            require(link.pop_received().vector_tag==received++,"duplicate or out-of-order vector");
        require(link.outstanding_vector_count()<=3,"overcommitted peer window");
        link.tick();
    }
    require(sent==10 && received==10 && !link.peer_error() && !link.awaiting_ack()
        && !link.outstanding_vector_count() && link.peer_duplicates()==5 && link.peer_retries()==10,
        "failed peer window accounting");
    link.set_peer_fault_schedule([](auto,auto,auto){return F::Corrupt;});
    link.send({});
    for(unsigned t=0;t<80;++t)link.tick();
    require(link.peer_error() && link.awaiting_ack() && !link.receive_ready() && !link.can_send()
        && link.peer_retries()==12,"retry limit silently retired a failed vector");
    link.reset();link.set_peer_fault_schedule({});link.send({});
    for(unsigned t=0;t<20;++t)link.tick();
    require(!link.peer_error() && link.receive_ready() && !link.awaiting_ack(),"peer reset/restart");
}

void test_reliable_sequence_exhaustion() {
    constexpr auto limit = std::numeric_limits<std::uint64_t>::max();
    C2cLink link({hw::kPhysicalVectorBytes,1,2,5,2,2});
    C2cLinkBoundaryFixture::seed_empty_link(link,limit-1,limit-1);
    std::vector<std::pair<std::uint64_t,std::uint64_t>> seen;
    link.set_peer_fault_schedule([&](auto epoch,auto seq,auto) {
        seen.emplace_back(epoch,seq);
        return C2cLink::FrameResult::Good;
    });
    auto transfer = [&](unsigned tag) {
        link.send(make_vector(0,tag));
        for(unsigned i=0;i<12;++i)link.tick();
        require(link.pop_received().vector_tag==tag, "boundary payload");
        for(unsigned i=0;i<12;++i)link.tick();
        require(!link.awaiting_ack() && !link.outstanding_vector_count() &&
                !link.receive_ready(), "boundary retirement");
    };
    transfer(7);
    require(!link.can_send(), "sequence exhaustion wrapped in the same generation");
    link.reset();
    transfer(8);
    bool refused=false;
    try {link.reset();} catch(const std::overflow_error&) {refused=true;}
    require(refused, "generation wrapped");
    transfer(9);
    require(seen==decltype(seen){{limit-1,limit-1},{limit,0},{limit,1}},
            "generation exhaustion aliased old sequence or changed live state");
}
} // namespace

int main()
try {
    test_rx_diagonal_pipeline_accepts_one_vector_per_cycle();
    test_tx_diagonal_pipeline_emits_one_vector_per_cycle();
    test_link_credit_and_serialization();
    test_dual_chip_rx_notifies_mem_and_writes_sram();
    test_dual_chip_rx_notifies_mem_and_writes_sram(true);
    test_reliable_window();
    test_reliable_sequence_exhaustion();
    test_dual_hemisphere_endpoints_transfer_in_parallel();
    return 0;
} catch (const std::exception& error) {
    std::cerr << "c2c_test failed: " << error.what() << '\n';
    return 1;
}
