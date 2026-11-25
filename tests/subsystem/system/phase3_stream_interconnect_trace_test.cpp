#include "ftlpu/system/stream_topology_builder.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ftlpu::StreamCell;
using ftlpu::StreamDirection;
using ftlpu::StreamId;

struct Boundary {
    const char* name;
    std::size_t column;
    StreamDirection direction;
};

constexpr std::array<Boundary, 6> kBoundaries {{
    {"vxm_east_to_mem", 0, StreamDirection::East},
    {"mem_east_to_sxm", 14, StreamDirection::East},
    {"sxm_east_to_mxm", 15, StreamDirection::East},
    {"mxm_west_to_sxm", 15, StreamDirection::West},
    {"sxm_west_to_mem", 14, StreamDirection::West},
    {"mem_west_to_vxm", 0, StreamDirection::West},
}};

StreamId stream_id(StreamDirection direction)
{
    return direction == StreamDirection::East
        ? StreamId::East(0) : StreamId::West(0);
}

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    const auto output = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path();
    std::filesystem::create_directories(output);

    const auto layout = ftlpu::make_configured_stream_layout();
    require(layout.topology.column_count() == 16,
        "R7 trace requires the current 16-column Stream topology");
    auto routes = layout.topology.default_route_selection();
    ftlpu::StreamRegisterFabric fabric(layout.topology.column_count());

    constexpr std::uint8_t kEastData = 0xa5;
    constexpr std::uint8_t kWestData = 0x5a;
    fabric.initialize_cell(
        0, 0, 0, StreamId::East(0), StreamCell::Valid(kEastData));
    fabric.initialize_cell(
        15, 0, 0, StreamId::West(0), StreamCell::Valid(kWestData));

    const auto trace_name = "phase3_r7_stream_interconnect_trace.csv";
    std::ofstream trace(output / trace_name);
    require(static_cast<bool>(trace), "cannot open R7 Stream trace");
    trace << "cycle,phase";
    for (const auto& boundary : kBoundaries) {
        trace << ',' << boundary.name << "_valid"
              << ',' << boundary.name << "_data";
    }
    trace << ",stream_conflict\n";

    constexpr std::size_t kTraceCycles = 20;
    for (std::size_t cycle = 0; cycle < kTraceCycles; ++cycle) {
        trace << cycle << ",PRE_POSEDGE";
        for (const auto& boundary : kBoundaries) {
            const auto& cell = fabric.cell(
                boundary.column, 0, 0, stream_id(boundary.direction));
            trace << ',' << (cell.valid ? 1 : 0)
                  << ',' << (cell.valid ? static_cast<unsigned>(cell.data) : 0);
        }
        trace << ",0\n";

        fabric.low();
        layout.topology.stage_active_routes(fabric, routes);
        fabric.validate_posedge();
        fabric.posedge();
    }

    require(fabric.empty(), "R7 trace ended with a held Stream cell");
    std::ofstream manifest(output / "phase3_r7_stream_interconnect_manifest.json");
    require(static_cast<bool>(manifest), "cannot open R7 manifest");
    manifest
        << "{\n"
        << "  \"schema\": \"ftlpu.phase3.stream_interconnect.v1\",\n"
        << "  \"sample_point\": \"PRE_POSEDGE\",\n"
        << "  \"cycle_offset\": 0,\n"
        << "  \"trace\": \"" << trace_name << "\",\n"
        << "  \"rows\": " << kTraceCycles << ",\n"
        << "  \"interfaces\": 6\n"
        << "}\n";

    std::cout << "PHASE3_R7_STREAM_TRACE_PASS rows=" << kTraceCycles
              << " interfaces=" << kBoundaries.size() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "R7 Stream trace failed: " << error.what() << '\n';
    return 1;
}
