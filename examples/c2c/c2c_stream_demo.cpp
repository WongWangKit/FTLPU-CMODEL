#include "ftlpu/c2c/c2c.hpp"
#include "ftlpu/system/stream_topology.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

void initialize_sender(
    ftlpu::StreamRegisterFabric& fabric,
    std::size_t column,
    std::size_t stream_index)
{
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto byte = static_cast<std::uint8_t>(tile * 16 + lane);
            fabric.initialize_cell(
                column,
                tile,
                lane,
                ftlpu::StreamId::East(stream_index),
                ftlpu::StreamCell::Valid(byte, lane + 1 == 16, 1001));
        }
    }
}

} // namespace

int main()
{
    constexpr std::size_t kTxStream = 3;
    constexpr std::size_t kRxStream = 9;

    // Sender side only needs the TX attachment column for this demo.
    ftlpu::StreamRegisterFabric sender_fabric(1);
    initialize_sender(sender_fabric, 0, kTxStream);

    // Receiver topology: RX injects at the edge column. A passive westward
    // route then carries the vector one SR hop inward to the MEM boundary.
    ftlpu::StreamTopology receiver_topology;
    const auto mem_boundary = receiver_topology.add_column("mem_boundary");
    const auto rx_boundary = receiver_topology.add_column("rx_boundary");
    receiver_topology.bind_slice({
        "C2C_RX",
        {},
        ftlpu::StreamTopology::DirectionalPorts{std::nullopt, rx_boundary},
    });
    receiver_topology.add_route({
        "rx_to_mem_west",
        rx_boundary,
        mem_boundary,
        ftlpu::StreamDirection::West,
        ftlpu::StreamTopology::RouteKind::Normal,
        true,
        false,
    });

    ftlpu::StreamRegisterFabric receiver_fabric(
        receiver_topology.column_count());
    auto route_selection = receiver_topology.default_route_selection();

    ftlpu::C2cLink link(ftlpu::C2cLinkConfig{
        320, // one full physical vector per link cycle
        0,
        2,
    });

    ftlpu::C2cTxSlice tx(
        {0, ftlpu::StreamDirection::East},
        "east-chip TX");
    ftlpu::C2cRxSlice rx(
        {rx_boundary, ftlpu::StreamDirection::West},
        "west-chip RX");

    // Cycle 0: TX consumes one complete East stream vector.
    tx.enqueue_send(kTxStream);
    sender_fabric.begin_cycle();
    tx.evaluate(sender_fabric, link);
    sender_fabric.commit_cycle();
    link.tick();
    std::cout << "cycle 0: TX sent 320 bytes; link transfer completed\n";

    // Cycle 1: RX injects twenty 16-byte segments into one West stream at
    // rx_boundary. They become current after the global commit.
    rx.enqueue_receive(kRxStream);
    receiver_fabric.begin_cycle();
    rx.evaluate(receiver_fabric, link);
    receiver_topology.stage_active_routes(receiver_fabric, route_selection);
    receiver_fabric.commit_cycle();
    link.tick();
    std::cout << "cycle 1: RX staged 20 x 16-byte segments at rx_boundary\n";

    // Cycle 2: the normal topology route moves the West stream one SR hop
    // inward, where a MEM Write instruction can consume it next cycle.
    receiver_fabric.begin_cycle();
    receiver_topology.stage_active_routes(receiver_fabric, route_selection);
    receiver_fabric.commit_cycle();
    link.tick();
    std::cout << "cycle 2: West stream reached mem_boundary\n";

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto& cell = receiver_fabric.cell(
                mem_boundary,
                tile,
                lane,
                ftlpu::StreamId::West(kRxStream));
            assert(cell.valid);
            assert(cell.data == static_cast<std::uint8_t>(tile * 16 + lane));
            assert(cell.vector_tag == 1001);
        }
    }

    std::cout << "PASS: C2C 320-byte vector is ready for MEM Write\n";
    return 0;
}
