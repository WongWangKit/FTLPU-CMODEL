#include "ftlpu/mem/tile_array.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

int main()
{
    auto model = std::make_unique<ftlpu::TileArrayModel>();
    std::ostringstream log;
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            model->set_east_stream_input(tile, lane, 3,
                {static_cast<std::uint8_t>(tile * 16 + lane), lane == 15});
            model->set_sram_lane_byte(43, tile, 32, lane,
                static_cast<std::uint8_t>(255 - (tile * 16 + lane)));
        }
    }

    // External input becomes visible at E sreg0, then needs two more ticks to E sreg2.
    model->tick(log);
    model->tick(log);
    model->tick(log);
    model->enqueue_instruction(8, ftlpu::MemInstruction::Write(16, 3));
    model->enqueue_instruction(43, ftlpu::MemInstruction::Read(32, 32 + 4));
    model->tick(log);

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        assert(!model->instruction_at(8, tile).has_value());
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            assert(model->sram_lane_byte(8, tile, 16, lane)
                == static_cast<std::uint8_t>(tile * 16 + lane));
            assert(model->west_register(tile, lane, 11, 4).has_value());
            assert(model->west_register(tile, lane, 11, 4)->data
                == static_cast<std::uint8_t>(255 - (tile * 16 + lane)));
        }
    }

    bool caught = false;
    try {
        model->enqueue_instruction(0, ftlpu::MemInstruction::Read(0, 64));
        model->tick(log);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    model->set_sram_lane_byte(0, 19, ftlpu::hw::kSramDepthRows - 1, 15, 0xa5);
    assert(model->sram_lane_byte(0, 19, ftlpu::hw::kSramDepthRows - 1, 15) == 0xa5);

    const std::string text = log.str();
    assert(text.find("c8.t0=Write(a=16,s=3)") != std::string::npos);
    assert(text.find("c8.t19=Write(a=16,s=3)") != std::string::npos);
    assert(text.find("c43.t0=Read(a=32,s=36)") != std::string::npos);
    assert(text.find("c43.t19=Read(a=32,s=36)") != std::string::npos);
    return 0;
}
