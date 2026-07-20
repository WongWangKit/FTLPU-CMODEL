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
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        model->set_east_stream_input(2, lane, 3, {
            static_cast<std::uint8_t>(10 + lane), lane + 1 == ftlpu::hw::kLanesPerTile});
        model->set_sram_lane_byte(43, 2, 32, lane, static_cast<std::uint8_t>(40 + lane));
    }

    model->tick(log);
    assert(!model->instruction_at(8, 1).has_value());
    assert(model->east_register(2, 0, 0, 3).has_value());
    assert(model->sram_lane_byte(8, 2, 16, 0) == 0);
    assert(!model->west_register(2, 0, 11, 4).has_value());

    model->enqueue_instruction(8, ftlpu::MemInstruction::Write(16, 3));
    model->enqueue_instruction(43, ftlpu::MemInstruction::Read(32, 32 + 4));
    model->tick(log);
    assert(model->instruction_at(8, 1).has_value());
    assert(model->sram_lane_byte(8, 2, 16, 0) == 0);
    assert(!model->west_register(2, 0, 11, 4).has_value());

    model->tick(log);
    assert(model->instruction_at(8, 2).has_value());

    model->tick(log);
    assert(model->instruction_at(8, 3).has_value());

    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        assert(model->sram_lane_byte(8, 2, 16, lane) == 10 + lane);
        assert(model->west_register(2, lane, 11, 4).has_value());
        assert(model->west_register(2, lane, 11, 4)->data == 40 + lane);
        assert(model->west_register(2, lane, 11, 4)->last == (lane + 1 == ftlpu::hw::kLanesPerTile));
    }
    bool caught = false;
    try {
        model->enqueue_instruction(0, ftlpu::MemInstruction::Read(0, 64));
        model->tick(log);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    model->set_sram_lane_byte(
        0, ftlpu::hw::kTileRows - 1, ftlpu::hw::kSramDepthRows - 1,
        ftlpu::hw::kLanesPerTile - 1, 0xa5);
    assert(model->sram_lane_byte(
        0, ftlpu::hw::kTileRows - 1, ftlpu::hw::kSramDepthRows - 1,
        ftlpu::hw::kLanesPerTile - 1) == 0xa5);

    caught = false;
    try {
        model->set_sram_lane_byte(0, 0, ftlpu::hw::kSramDepthRows, 0, 1);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    const std::string text = log.str();
    assert(text.find("cycle 0") != std::string::npos);
    assert(text.find("c8.t0=Write(a=16,s=3)") != std::string::npos);
    assert(text.find("c8.t1=Write(a=16,s=3)") != std::string::npos);
    assert(text.find("c8.t2=Write(a=16,s=3)") != std::string::npos);
    assert(text.find("c43.t2=Read(a=32,s=36)") != std::string::npos);
    assert(text.find("c8.t2 store E3 addr=16 bytes=0x0a0b0c0d0e0f1011") != std::string::npos);
    assert(text.find("c43.t2 load W4 addr=32 bytes=0x28292a2b2c2d2e2f") != std::string::npos);
    assert(text.find("tile 2:") != std::string::npos);
    assert(text.find("sreg 11: W4=0x28292a2b2c2d2e2f") != std::string::npos);

    return 0;
}
