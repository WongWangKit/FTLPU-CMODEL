#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/system/stream_topology.hpp"

#include <array>
#include <iostream>
#include <string>

int main()
{
    ftlpu::StreamTopology topology;

    // Demonstration only: the final Figure-4 integration should add every
    // physical SR column once, then map the MEM boundaries by name.
    std::array<ftlpu::StreamTopology::ColumnId,
        ftlpu::hw::kMemBoundaryStreamRegisterColumns> mem_boundary{};
    for (std::size_t i = 0; i < mem_boundary.size(); ++i) {
        mem_boundary[i] = topology.add_column("MEM.B" + std::to_string(i));
    }
    const auto tx_post = topology.add_column("TX.POST");
    const auto acc_in = topology.add_column("ACC.IN");

    for (std::size_t i = 0; i + 1 < mem_boundary.size(); ++i) {
        topology.add_route({
            "mem.east." + std::to_string(i),
            mem_boundary[i],
            mem_boundary[i + 1],
            ftlpu::StreamDirection::East,
            ftlpu::StreamTopology::RouteKind::Normal,
            true,
            false,
        });
        topology.add_route({
            "mem.west." + std::to_string(i),
            mem_boundary[i + 1],
            mem_boundary[i],
            ftlpu::StreamDirection::West,
            ftlpu::StreamTopology::RouteKind::Normal,
            true,
            false,
        });
    }

    topology.add_route({
        "bypass.tx_to_acc",
        tx_post,
        acc_in,
        ftlpu::StreamDirection::East,
        ftlpu::StreamTopology::RouteKind::Bypass,
        false,
        false,
    });

    ftlpu::MemStreamPortMap::BoundaryColumns mapped{};
    for (std::size_t i = 0; i < mapped.size(); ++i) {
        mapped[i] = mem_boundary[i];
    }

    ftlpu::StreamRegisterFabric streams(topology.column_count());
    ftlpu::MemArrayModel mem{ftlpu::MemStreamPortMap(mapped)};
    auto routes = topology.default_route_selection();

    std::cout << "physical SR columns: " << streams.column_count() << '\n';
    std::cout << "MEM group 0 east input/output: "
              << mem.ports().input_column(0, ftlpu::StreamDirection::East)
              << "/"
              << mem.ports().output_column(0, ftlpu::StreamDirection::East)
              << '\n';
    (void)routes;
    return 0;
}
