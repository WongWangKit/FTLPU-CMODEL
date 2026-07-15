#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/core/topology.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>

namespace {

std::string hex_bytes(const std::array<std::uint8_t, ftlpu::hw::kLanesPerTile>& bytes)
{
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (const auto byte : bytes) {
        os << std::setw(2) << static_cast<unsigned>(byte);
    }
    return os.str();
}

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
    std::uniform_int_distribution<std::size_t> slice_dist(0, ftlpu::hw::kSliceColumns - 1);

    const auto stream = stream_dist(rng);
    const auto mem_slice = slice_dist(rng);
    const auto target_sreg = ftlpu::stream_register_before_slice(mem_slice);
    const auto store_cycle = target_sreg + 1;
    const auto read_cycle = store_cycle + kReadDelayCycles;
    const auto last_output_cycle = read_cycle + ftlpu::hw::kStreamRegisterColumns - target_sreg + 1;

    auto model = std::make_unique<ftlpu::TileArrayModel>();
    std::ostringstream log;

    log << "scenario seed=0x46544c50 stream=E" << stream
        << " mem_slice=" << mem_slice
        << " target_sreg=" << target_sreg
        << " read_delay=" << kReadDelayCycles << '\n';

    for (std::size_t cycle = 0; cycle <= last_output_cycle; ++cycle) {
        if (cycle == 0) {
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                const auto bytes = tile_vector(tile);
                for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    model->set_east_stream_input(tile, lane, stream, {bytes[lane], lane == ftlpu::hw::kLanesPerTile - 1});
                }
            }
        }

        if (cycle == store_cycle) {
            model->enqueue_instruction(mem_slice, ftlpu::MemInstruction::Write(kSramAddress, stream));
        }

        if (cycle == read_cycle) {
            model->enqueue_instruction(mem_slice, ftlpu::MemInstruction::Read(kSramAddress, stream));
        }

        model->tick(log);

        if (cycle == store_cycle) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                assert(!model->east_register(0, lane, target_sreg, stream).has_value());
            }
        }
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        const auto bytes = tile_vector(tile);
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            assert(model->sram_lane_byte(mem_slice, tile, kSramAddress, lane) == bytes[lane]);
        }
    }

    const auto last_tile_bytes = tile_vector(ftlpu::hw::kTileRows - 1);
    const auto last_tile_hex = hex_bytes(last_tile_bytes);
    const auto text = log.str();

    assert(text.find("scenario seed=0x46544c50") != std::string::npos);
    assert(text.find("c" + std::to_string(mem_slice) + ".t0=Write(a=128,s=" + std::to_string(stream) + ")") != std::string::npos);
    assert(text.find("c" + std::to_string(mem_slice) + ".t0=Read(a=128,s=" + std::to_string(stream) + ")") != std::string::npos);
    assert(text.find("store E" + std::to_string(stream) + " addr=128 bytes=0x") != std::string::npos);
    assert(text.find("load E" + std::to_string(stream) + " addr=128 bytes=0x") != std::string::npos);
    assert(text.find("tile 19:") != std::string::npos);
    assert(text.find("sreg 11: E" + std::to_string(stream) + "=0x" + last_tile_hex) != std::string::npos);

    return 0;
}
