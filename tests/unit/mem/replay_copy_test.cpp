#include "ftlpu/mem/mem_array.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

int main() try {
    using namespace ftlpu;
    constexpr std::size_t rows = 67;
    for (std::size_t test = 0; test < 7; ++test) {
        const std::size_t source = test < 5 ? test * 2 : (test - 5) * 2;
        const std::size_t destination = test >= 5 ? 16 : source == 8 ? 0 : 8;
        const std::size_t sourceBank = test >= 5 ? 1 : 0;
        const std::size_t sourceGroup = source / hw::kMemSlicesPerGroup;
        const std::size_t destinationGroup = destination / hw::kMemSlicesPerGroup;
        const bool east = sourceGroup < destinationGroup;
        const std::size_t latency = east
            ? destinationGroup - sourceGroup : sourceGroup - destinationGroup;
        const auto stream = east ? StreamId::East(0) : StreamId::West(0);
        StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);
        auto mem = std::make_unique<MemArrayModel>();
        const auto expected = [](std::size_t row, std::size_t tile, std::size_t lane) {
            return static_cast<std::uint8_t>(row * 17 + tile * 7 + lane);
        };
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                    mem->set_sram_lane_byte(source, sourceBank, tile, row, lane,
                        expected(row, tile, lane));
        for (std::size_t cycle = 0; cycle < rows + latency + hw::kTileRows; ++cycle) {
            if (cycle < rows)
                mem->enqueue_instruction(source, sourceBank, MemInstruction::Read(cycle, stream));
            if (cycle >= latency && cycle - latency < rows)
                mem->enqueue_instruction(destination,
                    MemInstruction::Write(768 + cycle - latency, stream));
            fabric.begin_cycle();
            mem->evaluate(fabric);
            fabric.stage_linear_links();
            fabric.commit_cycle();
        }
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                    if (mem->sram_lane_byte(destination, tile, 768 + row, lane)
                        != expected(row, tile, lane))
                        throw std::logic_error("MEM replay copy lost or shifted data");
        std::cout << "MEM copy " << source << " -> " << destination
                  << " latency=" << latency << " PASS\n";
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
