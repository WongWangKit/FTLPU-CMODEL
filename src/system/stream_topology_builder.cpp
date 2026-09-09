#include "ftlpu/system/stream_topology_builder.hpp"

#include "ftlpu/core/hardware_config.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace ftlpu {
namespace {

StreamDirection to_runtime_direction(target::StreamFlow flow)
{
    switch (flow) {
    case target::StreamFlow::Forward:
        return StreamDirection::East;
    case target::StreamFlow::Reverse:
        return StreamDirection::West;
    }
    throw std::invalid_argument("unknown target stream flow");
}

StreamTopology::RouteKind to_runtime_kind(target::StreamRouteKind kind)
{
    switch (kind) {
    case target::StreamRouteKind::Normal:
        return StreamTopology::RouteKind::Normal;
    case target::StreamRouteKind::Bypass:
        return StreamTopology::RouteKind::Bypass;
    }
    throw std::invalid_argument("unknown target stream route kind");
}

void check_column(
    const target::StreamTopologyDescriptor& descriptor,
    std::size_t column,
    const char* context)
{
    if (column >= descriptor.columns.size()) {
        throw std::out_of_range(
            std::string(context) + " references an unknown stream column");
    }
}

StreamEndpoint make_endpoint(
    const target::StreamTopologyDescriptor& descriptor,
    target::StreamPortDescriptor port,
    const char* context)
{
    check_column(descriptor, port.column, context);
    return StreamEndpoint {port.column, to_runtime_direction(port.flow)};
}

void require_flow(
    target::StreamPortDescriptor port,
    target::StreamFlow expected,
    const char* context)
{
    if (port.flow != expected) {
        throw std::invalid_argument(
            std::string(context) + " has an incompatible stream flow");
    }
}

} // namespace

StreamLayout make_stream_layout(
    const target::StreamTopologyDescriptor& descriptor)
{
    if (descriptor.columns.empty()) {
        throw std::invalid_argument("stream topology has no columns");
    }
    if (descriptor.fabric_names.empty()) {
        throw std::invalid_argument("stream topology has no fabric replicas");
    }

    StreamTopology topology;
    for (const auto& column : descriptor.columns) {
        topology.add_column(std::string(column.name));
    }
    for (const auto& route : descriptor.routes) {
        check_column(descriptor, route.source_column, "stream route source");
        check_column(
            descriptor, route.destination_column, "stream route destination");
        if (route.latency_cycles != 1) {
            throw std::invalid_argument(
                "the current StreamTopology runtime supports one-cycle routes only");
        }
        topology.add_route(StreamTopology::Route {
            std::string(route.name),
            route.source_column,
            route.destination_column,
            to_runtime_direction(route.flow),
            to_runtime_kind(route.kind),
            route.enabled_by_default,
            route.multicast_allowed,
        });
    }

    if (descriptor.mem_slices_per_group != hw::kMemSlicesPerGroup) {
        throw std::invalid_argument(
            "stream topology MEM group size does not match the CModel");
    }
    if (descriptor.mem_boundary_columns.size()
        != MemStreamPortMap::BoundaryColumns {}.size()) {
        throw std::invalid_argument(
            "stream topology has the wrong number of MEM boundaries");
    }
    MemStreamPortMap::BoundaryColumns mem_boundaries{};
    for (std::size_t i = 0; i < mem_boundaries.size(); ++i) {
        check_column(
            descriptor, descriptor.mem_boundary_columns[i], "MEM boundary");
        mem_boundaries[i] = descriptor.mem_boundary_columns[i];
    }

    require_flow(
        descriptor.sxm.forward_input, target::StreamFlow::Forward,
        "SXM forward input");
    require_flow(
        descriptor.sxm.forward_output, target::StreamFlow::Forward,
        "SXM forward output");
    require_flow(
        descriptor.sxm.reverse_input, target::StreamFlow::Reverse,
        "SXM reverse input");
    require_flow(
        descriptor.sxm.reverse_output, target::StreamFlow::Reverse,
        "SXM reverse output");
    check_column(descriptor, descriptor.sxm.forward_input.column,
        "SXM forward input");
    check_column(descriptor, descriptor.sxm.forward_output.column,
        "SXM forward output");
    check_column(descriptor, descriptor.sxm.reverse_input.column,
        "SXM reverse input");
    check_column(descriptor, descriptor.sxm.reverse_output.column,
        "SXM reverse output");
    auto sxm_ports = SxmStreamPortMap::BetweenColumns(
        descriptor.sxm.forward_input.column,
        descriptor.sxm.forward_output.column,
        descriptor.sxm.reverse_input.column,
        descriptor.sxm.reverse_output.column);

    const auto c2c_tx = make_endpoint(
        descriptor, descriptor.c2c.tx_input, "C2C TX input");
    const auto c2c_rx = make_endpoint(
        descriptor, descriptor.c2c.rx_output, "C2C RX output");
    C2cStreamPortMap c2c_ports {
        {c2c_tx.column, c2c_tx.direction},
        {c2c_rx.column, c2c_rx.direction},
    };

    std::vector<std::string> fabric_names;
    fabric_names.reserve(descriptor.fabric_names.size());
    std::unordered_set<std::string> unique_fabric_names;
    for (const auto name : descriptor.fabric_names) {
        if (name.empty()) {
            throw std::invalid_argument("stream fabric name must not be empty");
        }
        auto value = std::string(name);
        if (!unique_fabric_names.insert(value).second) {
            throw std::invalid_argument("duplicate stream fabric name: " + value);
        }
        fabric_names.push_back(std::move(value));
    }

    std::vector<StreamSystemTransfer> system_transfers;
    system_transfers.reserve(descriptor.system_transfers.size());
    std::unordered_set<std::string> unique_transfer_names;
    for (const auto& transfer : descriptor.system_transfers) {
        if (transfer.source_fabric >= descriptor.fabric_names.size()
            || transfer.destination_fabric >= descriptor.fabric_names.size()) {
            throw std::out_of_range(
                "stream system transfer references an unknown fabric");
        }
        if (transfer.latency_cycles == 0) {
            throw std::invalid_argument(
                "stream system transfer latency must be positive");
        }
        auto name = std::string(transfer.name);
        if (name.empty()) {
            throw std::invalid_argument(
                "stream system transfer name must not be empty");
        }
        if (!unique_transfer_names.insert(name).second) {
            throw std::invalid_argument(
                "duplicate stream system transfer name: " + name);
        }
        system_transfers.push_back(StreamSystemTransfer {
            std::move(name),
            transfer.kind,
            transfer.source_fabric,
            make_endpoint(descriptor, transfer.source, "system transfer source"),
            transfer.destination_fabric,
            make_endpoint(
                descriptor, transfer.destination, "system transfer destination"),
            transfer.latency_cycles,
        });
    }

    return StreamLayout {
        std::move(topology),
        std::move(fabric_names),
        MemStreamPortMap(std::move(mem_boundaries)),
        std::move(sxm_ports),
        c2c_ports,
        make_endpoint(
            descriptor, descriptor.mxm.weight_input, "MXM weight input"),
        make_endpoint(
            descriptor, descriptor.mxm.activation_input, "MXM activation input"),
        make_endpoint(
            descriptor, descriptor.mxm.result_output, "MXM result output"),
        make_endpoint(descriptor, descriptor.vxm.input, "VXM input"),
        make_endpoint(descriptor, descriptor.vxm.output, "VXM output"),
        std::move(system_transfers),
    };
}

StreamLayout make_configured_stream_layout()
{
    return make_stream_layout(hw::config::kStreamTopology);
}

} // namespace ftlpu
