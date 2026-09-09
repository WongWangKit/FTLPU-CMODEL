#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/mem/mem_array.hpp"

#include <array>
#include <cassert>
#include <cstdint>
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

int main()
{
    constexpr std::uint32_t kSeed = 0x46544c50;
    constexpr std::size_t kSramAddress = 128;
    constexpr std::size_t kReadDelayCycles = 4;

    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<std::size_t> stream_dist(0, ftlpu::hw::kEastStreams - 1);
    std::uniform_int_distribution<std::size_t> slice_dist(
        0, ftlpu::hw::kMemSliceColumns - 1);

    const auto stream = stream_dist(rng);
    const auto mem_slice = slice_dist(rng);
    auto mem = std::make_unique<ftlpu::MemArrayModel>();
    const auto target_sreg = mem->ports().input_column(
        mem_slice, ftlpu::StreamDirection::East);
    const auto store_cycle = target_sreg + 1;
    const auto read_cycle = store_cycle + kReadDelayCycles;
    const auto last_output_cycle = read_cycle + ftlpu::hw::kTileRows
        + ftlpu::hw::kSystemStreamRegisterColumns - target_sreg;

    auto fabric = std::make_unique<ftlpu::StreamRegisterFabric>(
        ftlpu::hw::kSystemStreamRegisterColumns);
    ftlpu::StreamOutputPort input(
        *fabric, 0, ftlpu::StreamDirection::East,
        "vector roundtrip test input");
    std::array<bool, ftlpu::hw::kTileRows> load_seen{};
    std::ostringstream log;

    log << "scenario seed=0x46544c50 stream=E" << stream
        << " mem_slice=" << mem_slice
        << " target_sreg=" << target_sreg
        << " read_delay=" << kReadDelayCycles << '\n';

    for (std::size_t cycle = 0; cycle <= last_output_cycle; ++cycle) {
        fabric->begin_cycle();
        if (cycle < ftlpu::hw::kTileRows) {
            const auto bytes = tile_vector(cycle);
            input.write_payload_segment(cycle, stream, bytes);
        }

        if (cycle == store_cycle) {
            mem->enqueue_instruction(
                mem_slice, ftlpu::MemInstruction::Write(kSramAddress, stream));
        }

        if (cycle == read_cycle) {
            mem->enqueue_instruction(
                mem_slice, ftlpu::MemInstruction::Read(kSramAddress, stream));
        }

        mem->evaluate(*fabric);
        for (const auto& transfer : mem->executed_transfers()) {
            if (transfer.kind
                != ftlpu::MemArrayModel::MemTransfer::Kind::LoadSramToStream) {
                continue;
            }
            assert(transfer.mem_slice == mem_slice);
            assert(transfer.bytes == tile_vector(transfer.tile));
            load_seen[transfer.tile] = true;
        }
        fabric->stage_linear_links();
        fabric->commit_cycle();
        mem->log_cycle(log);

        if (cycle == store_cycle) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                assert(!fabric->cell(
                    target_sreg, 0, lane,
                    ftlpu::StreamId::East(stream)).valid);
            }
        }
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        const auto bytes = tile_vector(tile);
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            assert(mem->sram_lane_byte(
                       mem_slice, tile, kSramAddress, lane)
                == bytes[lane]);
        }
        assert(load_seen[tile]);
    }

    const auto text = log.str();

    assert(text.find("scenario seed=0x46544c50") != std::string::npos);
    assert(text.find("c" + std::to_string(mem_slice)
        + ".b0.t0=Write(a=128,s=" + std::to_string(stream) + ")")
        != std::string::npos);
    assert(text.find("c" + std::to_string(mem_slice)
        + ".b0.t0=Read(a=128,s=" + std::to_string(stream) + ")")
        != std::string::npos);
    assert(text.find("store E" + std::to_string(stream)
        + " addr=128 tag=") != std::string::npos);
    assert(text.find("load E" + std::to_string(stream)
        + " addr=128 tag=") != std::string::npos);

    return 0;
}
