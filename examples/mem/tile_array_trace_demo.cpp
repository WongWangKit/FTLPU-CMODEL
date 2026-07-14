#include "ftlpu/mem/tile_array.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv)
{
    const std::string log_path = argc > 1 ? argv[1] : "array_trace.log";
    std::ofstream log(log_path);
    if (!log) {
        std::cerr << "failed to open log file: " << log_path << '\n';
        return 1;
    }

    auto model = std::make_unique<ftlpu::TileArrayModel>();

    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        model->set_east_stream_input(0, lane, 0, {static_cast<std::uint8_t>(100 + lane), lane == 15});
        model->set_west_stream_input(0, lane, 0, {static_cast<std::uint8_t>(200 + lane), lane == 15});
        model->set_sram_lane_byte(43, 0, 16, lane, static_cast<std::uint8_t>(210 + lane));
    }

    model->enqueue_instruction(0, ftlpu::MemInstruction::Write(0, 0));
    model->enqueue_instruction(3, ftlpu::MemInstruction::Gather(2, 5));
    model->enqueue_instruction(17, ftlpu::MemInstruction::Write(640, 7));
    model->enqueue_instruction(43, ftlpu::MemInstruction::Read(16, 32));

    for (int cycle = 0; cycle < 8; ++cycle) {
        model->tick(log);
    }

    std::cout << "wrote trace log: " << log_path << '\n';
    return 0;
}
