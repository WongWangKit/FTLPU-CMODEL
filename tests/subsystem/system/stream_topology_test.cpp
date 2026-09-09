#include "ftlpu/core/hardware_config.hpp"
#include "ftlpu/system/stream_topology.hpp"
#include "ftlpu/system/stream_topology_builder.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <stdexcept>

int main()
{
    const auto& descriptor = ftlpu::hw::config::kStreamTopology;
    assert(descriptor.routes[0].flow == ftlpu::target::StreamFlow::Forward);
    assert(descriptor.routes[1].flow == ftlpu::target::StreamFlow::Reverse);
    assert(descriptor.sxm.forward_input.flow
        == ftlpu::target::StreamFlow::Forward);
    assert(descriptor.sxm.reverse_input.flow
        == ftlpu::target::StreamFlow::Reverse);
    assert(descriptor.system_transfers[0].source.flow
        == ftlpu::target::StreamFlow::Reverse);
    assert(descriptor.system_transfers[0].destination.flow
        == ftlpu::target::StreamFlow::Forward);

    const auto lpu32 = ftlpu::make_configured_stream_layout();
    assert(lpu32.fabric_names.size() == 2);
    assert(lpu32.fabric_names[0] == "east");
    assert(lpu32.fabric_names[1] == "west");
    assert(lpu32.topology.column_count() == 16);
    assert(lpu32.topology.routes().size() == 30);
    assert(lpu32.topology.column_name(0) == "mem.b0");
    assert(lpu32.topology.column_name(15) == "sxm_mxm");

    assert(lpu32.mem_ports.input_column(
        0, ftlpu::StreamDirection::East) == 0);
    assert(lpu32.mem_ports.output_column(
        51, ftlpu::StreamDirection::East) == 13);
    assert(lpu32.sxm_ports.input_column(
        ftlpu::StreamDirection::East) == 14);
    assert(lpu32.sxm_ports.output_column(
        ftlpu::StreamDirection::West) == 14);
    assert(lpu32.mxm_weight_input.column == 15);
    assert(lpu32.mxm_activation_input.column == 15);
    assert(lpu32.mxm_result_output.direction
        == ftlpu::StreamDirection::West);
    assert(lpu32.vxm_input.column == 0);
    assert(lpu32.vxm_output.direction == ftlpu::StreamDirection::East);
    assert(lpu32.c2c_ports.tx_input.column == 13);
    assert(lpu32.c2c_ports.tx_input.direction
        == ftlpu::StreamDirection::East);
    assert(lpu32.c2c_ports.rx_output.column == 13);
    assert(lpu32.c2c_ports.rx_output.direction
        == ftlpu::StreamDirection::West);

    assert(lpu32.system_transfers.size() == 2);
    assert(lpu32.system_transfers[0].source_fabric == 0);
    assert(lpu32.system_transfers[0].destination_fabric == 1);
    assert(lpu32.system_transfers[0].source.direction
        == ftlpu::StreamDirection::West);
    assert(lpu32.system_transfers[0].destination.direction
        == ftlpu::StreamDirection::East);

    // TspSliceSystem must stage the generated route selection, rather than
    // silently falling back to the old implicit linear fabric.
    ftlpu::TspSliceSystem system;
    const auto system_stream = ftlpu::StreamId::East(7);
    system.stream_fabric(ftlpu::Hemisphere::East).initialize_cell(
        0, 0, 0, system_stream, ftlpu::StreamCell::Valid(61));
    system.stream_route_selection(ftlpu::Hemisphere::East).disable(
        "main.forward.0");
    system.tick({});
    assert(!system.stream_fabric(ftlpu::Hemisphere::East)
                .cell(1, 0, 0, system_stream).valid);

    system.reset_execution_state();
    system.stream_route_selection(ftlpu::Hemisphere::East).enable(
        "main.forward.0");
    system.stream_fabric(ftlpu::Hemisphere::East).initialize_cell(
        0, 0, 0, system_stream, ftlpu::StreamCell::Valid(62));
    system.tick({});
    const auto& forwarded = system.stream_fabric(ftlpu::Hemisphere::East)
                                .cell(1, 0, 0, system_stream);
    assert(forwarded.valid);
    assert(forwarded.data == 62);

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
