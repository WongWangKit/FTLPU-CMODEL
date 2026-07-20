#include "ftlpu/mem/mem_array.hpp"

#include <cassert>
#include <cstdint>

int main()
{
    constexpr std::size_t kMemSlice = 8; // group 2: east input SR2, east output SR3
    constexpr auto kAddress =
        ftlpu::MemLocalWordAddress13::FromFields(1, 64);
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
        assert(mem.executed_transfers().size() == 1);
        const auto& transfer = mem.executed_transfers().front();
        assert(transfer.address == kAddress);
        assert(transfer.bank == 1);
        assert(transfer.word == 64);
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

        mem.enqueue_instruction(
            kMemSlice,
            ftlpu::MemInstruction::Read(kWriteAddress, stream));
        mem.reset();
        assert(mem.cycle() == 0);
        assert(mem.sram_lane_byte(kMemSlice, 7, kWriteAddress, 3) == 7 * 16 + 3);
        fabric.begin_cycle();
        mem.evaluate(fabric);
        fabric.stage_linear_links();
        fabric.commit_cycle();
        assert(mem.executed_instructions().empty());

        mem.clear_sram();
        assert(mem.sram_lane_byte(kMemSlice, 7, kWriteAddress, 3) == 0);

        mem.set_sram_lane_byte(kMemSlice, 7, kWriteAddress, 3, 0xa5);
        mem.reset_and_clear_sram();
        assert(mem.sram_lane_byte(kMemSlice, 7, kWriteAddress, 3) == 0);
    }

    return 0;
}
