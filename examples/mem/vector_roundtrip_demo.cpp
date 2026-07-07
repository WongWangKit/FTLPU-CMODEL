#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/core/topology.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

namespace {

std::array<std::uint8_t, ftlpu::hw::kLanesPerTile> tile_vector(std::size_t tile)
{
    std::array<std::uint8_t, ftlpu::hw::kLanesPerTile> bytes{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        bytes[lane] = static_cast<std::uint8_t>(tile * ftlpu::hw::kLanesPerTile + lane);
    }
    return bytes;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string log_path = argc > 1 ? argv[1] : "vector_roundtrip.log";
    std::ofstream log(log_path);
    if (!log) {
        std::cerr << "failed to open log file: " << log_path << '\n';
        return 1;
    }

    constexpr std::uint32_t kSeed = 0x46544c50;
    constexpr std::size_t kSramAddress = 128;
    constexpr std::size_t kReadDelayCycles = 4;

    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<std::size_t> stream_dist(0, ftlpu::hw::kEastStreams - 1);
    std::uniform_int_distribution<std::size_t> slice_dist(0, ftlpu::hw::kSliceColumns - 1);

    const auto stream = stream_dist(rng);
    const auto mem_slice = slice_dist(rng);
    const auto target_sreg = ftlpu::stream_register_before_slice(mem_slice);
    const auto store_cycle = target_sreg + 1;
    const auto read_cycle = store_cycle + kReadDelayCycles;
    const auto last_output_cycle = read_cycle + ftlpu::hw::kTileRows + ftlpu::hw::kStreamRegisterColumns - target_sreg;

    auto model = std::make_unique<ftlpu::TileArrayModel>();

    log << "scenario seed=0x46544c50 stream=E" << stream
        << " mem_slice=" << mem_slice
        << " target_sreg=" << target_sreg
        << " store_issue_cycle=" << store_cycle
        << " read_issue_cycle=" << read_cycle
        << " read_delay=" << kReadDelayCycles << '\n';

    for (std::size_t cycle = 0; cycle <= last_output_cycle; ++cycle) {
        if (cycle < ftlpu::hw::kTileRows) {
            const auto bytes = tile_vector(cycle);
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                model->set_east_stream_input(cycle, lane, stream, {bytes[lane], lane == ftlpu::hw::kLanesPerTile - 1});
            }
        }

        if (cycle == store_cycle) {
            model->enqueue_instruction(mem_slice, ftlpu::MemInstruction::Write(kSramAddress, stream));
        }

        if (cycle == read_cycle) {
            model->enqueue_instruction(mem_slice, ftlpu::MemInstruction::Read(kSramAddress, stream));
        }

        model->tick(log);
    }

    std::cout << "wrote vector roundtrip log: " << log_path << '\n';
    std::cout << "stream=E" << stream << " mem_slice=" << mem_slice
              << " store_issue_cycle=" << store_cycle
              << " read_issue_cycle=" << read_cycle << '\n';
    return 0;
}
