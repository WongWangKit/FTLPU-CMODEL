#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/system/dual_chip_c2c_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu;

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
        fabric.begin_cycle();
        rx.evaluate(fabric, transport);
        fabric.commit_cycle();

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

        fabric.begin_cycle();
        tx.evaluate(fabric, transport);
        fabric.commit_cycle();

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
    for (std::size_t cycle = 0; cycle < 5; ++cycle) {
        link.tick();
        require(!link.receive_ready(), "C2C vector arrived before serialization and flight completed");
    }
    link.tick();
    require(link.receive_ready(), "C2C vector did not reach the RX FIFO");
    const auto received = link.pop_received();
    require(received.payload == vector.payload, "C2C link changed vector payload bytes");
    require(link.can_send(), "C2C RX pop did not return link credit");
}

void test_dual_chip_rx_notifies_mem_and_writes_sram()
{
    constexpr auto kStream = 5U;
    constexpr auto kSourceSlice = 48U;
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
        2,
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
    for (std::size_t cycle = 0; cycle < 24; ++cycle) {
        system->tick();
        const auto action = destination.icu().mem_iq(target_queue).last_trace().action;
        observed_sync_wait = observed_sync_wait
            || action == IcuQueueAction::SyncWait;
        observed_sync_release = observed_sync_release
            || action == IcuQueueAction::SyncRelease;
    }

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
    constexpr auto kSourceSlice = 48U;
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

    for (std::size_t cycle = 0; cycle < 24; ++cycle) system->tick();

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

} // namespace

int main()
try {
    test_rx_diagonal_pipeline_accepts_one_vector_per_cycle();
    test_tx_diagonal_pipeline_emits_one_vector_per_cycle();
    test_link_credit_and_serialization();
    test_dual_chip_rx_notifies_mem_and_writes_sram();
    test_dual_hemisphere_endpoints_transfer_in_parallel();
    return 0;
} catch (const std::exception& error) {
    std::cerr << "c2c_test failed: " << error.what() << '\n';
    return 1;
}
