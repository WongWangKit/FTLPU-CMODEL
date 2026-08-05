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
        model->set_east_stream_input(2, lane, 3, {static_cast<std::uint8_t>(10 + lane), lane == 15});
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
        assert(model->west_register(2, lane, 11, 4)->last == (lane == 15));
    }
    bool caught = false;
    try {
        model->enqueue_instruction(0, ftlpu::MemInstruction::Read(0, 64));
        model->tick(log);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    const auto bank0_word = ftlpu::MemLocalWordAddress13::FromFields(0, 7);
    const auto bank1_word = ftlpu::MemLocalWordAddress13::FromFields(1, 7);
    model->set_sram_lane_byte(0, 5, bank0_word, 3, 0x11);
    model->set_sram_byte(0, 5, bank1_word.slice_byte_address(3), 0x22);
    assert(model->sram_lane_byte(0, 5, bank0_word, 3) == 0x11);
    assert(model->sram_byte(0, 5, bank1_word.slice_byte_address(3)) == 0x22);
    assert(model->sram_lane_byte(0, 6, bank1_word, 3) == 0);

    const auto last_address = ftlpu::MemLocalWordAddress13::FromFields(1, 4095);
    model->set_sram_lane_byte(0, 19, last_address, 15, 0xa5);
    assert(model->sram_lane_byte(0, 19, last_address, 15) == 0xa5);

    caught = false;
    try {
        model->set_sram_lane_byte(
            0, 0, ftlpu::hw::kMemLocalWordAddressCount, 0, 1);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    const std::string text = log.str();
    assert(text.find("cycle 0") != std::string::npos);
    assert(text.find("c8.t0=Write(addr=b0:w16 encoded=0x0010,s=3)") != std::string::npos);
    assert(text.find("c8.t1=Write(addr=b0:w16 encoded=0x0010,s=3)") != std::string::npos);
    assert(text.find("c8.t2=Write(addr=b0:w16 encoded=0x0010,s=3)") != std::string::npos);
    assert(text.find("c43.t2=Read(addr=b0:w32 encoded=0x0020,s=36)") != std::string::npos);
    assert(text.find("c8.t2 store E3 addr=b0:w16 encoded=0x0010 bytes=0x0a0b0c0d0e0f10111213141516171819") != std::string::npos);
    assert(text.find("c43.t2 load W4 addr=b0:w32 encoded=0x0020 bytes=0x28292a2b2c2d2e2f3031323334353637") != std::string::npos);
    assert(text.find("tile 2:") != std::string::npos);
    assert(text.find("sreg 11: W4=0x28292a2b2c2d2e2f3031323334353637") != std::string::npos);

    return 0;
}
