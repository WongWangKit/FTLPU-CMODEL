#include "ftlpu/mem/mem_array.hpp"

#include <cassert>
#include <cstdint>

int main()
{
    constexpr std::size_t kMemSlice = 8; // group 2: east input SR2, east output SR3
    constexpr std::size_t kAddress = 64;
    const auto stream = ftlpu::StreamId::East(5);

    {
        ftlpu::StreamRegisterFabric fabric(ftlpu::hw::kMemBoundaryStreamRegisterColumns);
        ftlpu::MemArrayModel mem;
        constexpr std::size_t kTile = 3;

        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            mem.set_sram_lane_byte(
                kMemSlice,
                kTile,
                kAddress,
                lane,
                static_cast<std::uint8_t>(100 + lane));
        }

        // Instruction takes three cycles to reach tile 3.
        mem.enqueue_instruction(kMemSlice, ftlpu::MemInstruction::Read(kAddress, stream));
        for (std::size_t cycle = 0; cycle <= kTile; ++cycle) {
            fabric.begin_cycle();
            mem.evaluate(fabric);
            fabric.stage_linear_links();
            fabric.commit_cycle();
        }

        const auto source_column = mem.ports().output_column(kMemSlice, stream.direction());
        assert(mem.ports().input_column(kMemSlice, stream.direction()) == 2);
        assert(source_column == 3);
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto& cell = fabric.cell(source_column, kTile, lane, stream);
            assert(cell.valid);
            assert(cell.data == static_cast<std::uint8_t>(100 + lane));
        }
    }

    {
        ftlpu::StreamRegisterFabric fabric(ftlpu::hw::kMemBoundaryStreamRegisterColumns);
        ftlpu::MemArrayModel mem;
        constexpr std::size_t kWriteAddress = 128;
        constexpr std::size_t kWriteIssueCycle = 3; // source SR boundary is column 2

        // Inject tile t at cycle t. It reaches SR2 at cycle t+3, exactly when
        // the Write issued at cycle 3 reaches tile t.
        for (std::size_t cycle = 0;
             cycle < kWriteIssueCycle + ftlpu::hw::kTileRows;
             ++cycle) {
            if (cycle == kWriteIssueCycle) {
                mem.enqueue_instruction(
                    kMemSlice,
                    ftlpu::MemInstruction::Write(kWriteAddress, stream));
            }

            fabric.begin_cycle();
            mem.evaluate(fabric);
            fabric.stage_linear_links();

            if (cycle < ftlpu::hw::kTileRows) {
                ftlpu::StreamPayloadSegment16 bytes{};
                for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    bytes[lane] = static_cast<std::uint8_t>(cycle * 16 + lane);
                }
                fabric.stage_payload_segment(
                    0,
                    cycle,
                    stream,
                    bytes,
                    cycle,
                    "test source");
            }
            fabric.commit_cycle();
        }

        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                assert(mem.sram_lane_byte(kMemSlice, tile, kWriteAddress, lane)
                    == static_cast<std::uint8_t>(tile * 16 + lane));
            }
        }
    }

    {
        ftlpu::StreamRegisterFabric fabric(ftlpu::hw::kMemBoundaryStreamRegisterColumns);
        ftlpu::MemArrayModel mem;
        constexpr std::size_t kReadAddress = 192;
        constexpr std::size_t kWriteAddress = 193;
        constexpr std::size_t kIssueCycle = 3;
        constexpr std::size_t kLastTile = ftlpu::hw::kTileRows - 1;
        const auto read_stream = ftlpu::StreamId::East(9);
        const auto write_stream = ftlpu::StreamId::East(10);

        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                mem.set_sram_lane_byte(
                    kMemSlice, tile, kReadAddress, lane,
                    static_cast<std::uint8_t>(0xa0 + tile * 8 + lane));
            }
        }

        for (std::size_t cycle = 0;
             cycle < kIssueCycle + ftlpu::hw::kTileRows;
             ++cycle) {
            if (cycle == kIssueCycle) {
                mem.enqueue_instruction(
                    kMemSlice,
                    ftlpu::MemInstruction::ReadWrite(
                        kReadAddress, read_stream, kWriteAddress, write_stream));
            }

            fabric.begin_cycle();
            mem.evaluate(fabric);
            fabric.stage_linear_links();
            if (cycle < ftlpu::hw::kTileRows) {
                ftlpu::StreamPayloadSegment16 bytes{};
                for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    bytes[lane] = static_cast<std::uint8_t>(0x40 + cycle * 8 + lane);
                }
                fabric.stage_payload_segment(0, cycle, write_stream, bytes, cycle, "dual-port source");
            }
            fabric.commit_cycle();
        }

        const auto output_column = mem.ports().output_column(kMemSlice, read_stream.direction());
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto& read_cell = fabric.cell(output_column, kLastTile, lane, read_stream);
            assert(read_cell.valid);
            assert(read_cell.data == static_cast<std::uint8_t>(0xa0 + kLastTile * 8 + lane));
        }
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                assert(mem.sram_lane_byte(kMemSlice, tile, kWriteAddress, lane)
                    == static_cast<std::uint8_t>(0x40 + tile * 8 + lane));
            }
        }

        bool rejected_same_address = false;
        try {
            static_cast<void>(ftlpu::MemInstruction::ReadWrite(
                kReadAddress, read_stream, kReadAddress, write_stream));
        } catch (const std::invalid_argument&) {
            rejected_same_address = true;
        }
        assert(rejected_same_address);
    }

    return 0;
}
