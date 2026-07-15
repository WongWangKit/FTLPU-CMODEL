#include "ftlpu/mem/mem_array.hpp"

#include <cassert>
#include <cstdint>

int main()
{
    constexpr std::size_t kMemSlice = 8;
    constexpr std::size_t kReadAddress = 64;
    constexpr std::size_t kWriteAddress = 128;
    const auto stream = ftlpu::StreamId::East(5);

    ftlpu::StreamRegisterFabric fabric(ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    ftlpu::MemArrayModel mem;
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            mem.set_sram_lane_byte(kMemSlice, tile, kReadAddress, lane,
                static_cast<std::uint8_t>(tile * 16 + lane));
        }
    }

    mem.enqueue_instruction(kMemSlice, ftlpu::MemInstruction::Read(kReadAddress, stream));
    fabric.begin_cycle();
    mem.evaluate(fabric);
    fabric.stage_linear_links();
    fabric.commit_cycle();

    const auto read_column = mem.ports().output_column(kMemSlice, stream.direction());
    assert(read_column == 3);
    assert(fabric.vector_valid(read_column, stream));
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        assert(mem.instruction_at(kMemSlice, tile) == std::nullopt);
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            assert(fabric.cell(read_column, tile, lane, stream).data
                == static_cast<std::uint8_t>(tile * 16 + lane));
        }
    }

    // One commit moves the complete vector by exactly one SR boundary.
    fabric.begin_cycle();
    mem.evaluate(fabric);
    fabric.stage_linear_links();
    fabric.commit_cycle();
    assert(!fabric.vector_valid(read_column, stream));
    assert(fabric.vector_valid(read_column + 1, stream));

    ftlpu::StreamRegisterFabric write_fabric(ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    auto payload = ftlpu::StreamPayloadVector320 {};
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            payload[tile][lane] = static_cast<std::uint8_t>(255 - (tile * 16 + lane));
        }
    }
    write_fabric.begin_cycle();
    write_fabric.stage_payload_vector(0, stream, payload, 1, "test source");
    write_fabric.stage_linear_links();
    write_fabric.commit_cycle();
    for (std::size_t hop = 0; hop < 2; ++hop) {
        write_fabric.begin_cycle();
        write_fabric.stage_linear_links();
        write_fabric.commit_cycle();
    }
    assert(write_fabric.vector_valid(mem.ports().input_column(kMemSlice, stream.direction()), stream));

    mem.enqueue_instruction(kMemSlice, ftlpu::MemInstruction::Write(kWriteAddress, stream));
    write_fabric.begin_cycle();
    mem.evaluate(write_fabric);
    write_fabric.stage_linear_links();
    write_fabric.commit_cycle();
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            assert(mem.sram_lane_byte(kMemSlice, tile, kWriteAddress, lane) == payload[tile][lane]);
        }
    }
    return 0;
}
