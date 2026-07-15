#include "ftlpu/system/stream_topology.hpp"

#include <cassert>
#include <stdexcept>

int main()
{
    ftlpu::StreamTopology topology;
    const auto tx_out = topology.add_column("tx.out");
    const auto dst_out = topology.add_column("dst.out");
    const auto acc_in = topology.add_column("acc.in");

    topology.bind_slice({
        "DST",
        {tx_out, dst_out},
        {},
    });

    topology.add_route({
        "normal.tx_to_dst",
        tx_out,
        dst_out,
        ftlpu::StreamDirection::East,
        ftlpu::StreamTopology::RouteKind::Normal,
        true,
        false,
    });
    topology.add_route({
        "bypass.tx_to_acc",
        tx_out,
        acc_in,
        ftlpu::StreamDirection::East,
        ftlpu::StreamTopology::RouteKind::Bypass,
        false,
        false,
    });

    ftlpu::StreamRegisterFabric fabric(topology.column_count());
    const auto stream = ftlpu::StreamId::East(3);
    fabric.initialize_cell(
        tx_out,
        0,
        0,
        stream,
        ftlpu::StreamCell::Valid(42));

    auto routes = topology.default_route_selection();
    fabric.begin_cycle();
    topology.stage_active_routes(fabric, routes);
    fabric.commit_cycle();
    assert(fabric.cell(dst_out, 0, 0, stream).valid);
    assert(fabric.cell(dst_out, 0, 0, stream).data == 42);
    assert(!fabric.cell(acc_in, 0, 0, stream).valid);

    fabric.reset();
    fabric.initialize_cell(
        tx_out,
        0,
        0,
        stream,
        ftlpu::StreamCell::Valid(99));
    routes.disable("normal.tx_to_dst");
    routes.enable("bypass.tx_to_acc");
    fabric.begin_cycle();
    topology.stage_active_routes(fabric, routes);
    fabric.commit_cycle();
    assert(!fabric.cell(dst_out, 0, 0, stream).valid);
    assert(fabric.cell(acc_in, 0, 0, stream).data == 99);

    // Enabling normal and bypass together is illegal unless the topology
    // explicitly declares multicast.
    fabric.reset();
    routes.enable("normal.tx_to_dst");
    bool rejected = false;
    try {
        fabric.begin_cycle();
        topology.stage_active_routes(fabric, routes);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);

    return 0;
}
