#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/block_accumulator.hpp"
#include "ftlpu/mxm/mxm.hpp"

#include <bit>
#include <cassert>
#include <cstdint>
#include <sstream>
#include <stdexcept>

int main()
{
    static_assert(ftlpu::hw::kMxmBlockAccumulatorRows
        == ftlpu::hw::kMxmAccumulatorBlockCount * 4);
    static_assert(ftlpu::hw::kMxmBlockAccumulatorColumns == 256);
    static_assert(ftlpu::hw::kMxmBlockAccumulatorBytes
        == ftlpu::hw::kMxmAccumulatorBlockCount * 32 * 32 * sizeof(float));
    constexpr auto kTestAddress = ftlpu::hw::kMxmBlockAccumulatorRows / 2;

    ftlpu::MxmBlockAccumulator accumulator;
    ftlpu::MxmBlockAccumulator::Segment first {};
    ftlpu::MxmBlockAccumulator::Segment second {};
    for (std::size_t row = 0; row < first.size(); ++row) {
        for (std::size_t lane = 0; lane < first[row].size(); ++lane) {
            first[row][lane] = static_cast<float>(row * 10 + lane);
            second[row][lane] = 0.5f;
        }
    }

    accumulator.accumulate(kTestAddress, 2, first);
    accumulator.accumulate(kTestAddress, 2, second);
    const auto values = accumulator.read(kTestAddress, 2);
    for (std::size_t row = 0; row < values.size(); ++row) {
        for (std::size_t lane = 0; lane < values[row].size(); ++lane) {
            assert(values[row][lane] == first[row][lane] + 0.5f);
            assert(
                accumulator.value(kTestAddress, row, 16 + lane)
                == values[row][lane]);
        }
    }

    accumulator.clear_segment(kTestAddress, 2);
    for (std::size_t row = 0; row < values.size(); ++row) {
        for (std::size_t lane = 0; lane < values[row].size(); ++lane) {
            assert(accumulator.value(kTestAddress, row, 16 + lane) == 0.0f);
        }
    }

    bool rejected = false;
    try {
        (void)accumulator.read(
            ftlpu::hw::kMxmBlockAccumulatorRows,
            0);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);

    ftlpu::Mxm mxm;
    ftlpu::TileArrayModel mem;
    ftlpu::MxmBlockAccumulator::Segment stream_values {};
    for (std::size_t row = 0; row < stream_values.size(); ++row) {
        for (std::size_t lane = 0; lane < stream_values[row].size(); ++lane) {
            stream_values[row][lane] =
                static_cast<float>(100 * row + lane) + 0.25f;
        }
    }
    mxm.block_accumulator().accumulate(kTestAddress, 0, stream_values);
    mxm.control().issue_south(
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kTestAddress,
            0,
            true,
            ftlpu::MxmComputeMode::Block8));
    std::ostringstream log;
    mxm.control().tick(log);
    mxm.tick_datapath(mem, 0);
    mem.tick();

    for (std::size_t row = 0; row < stream_values.size(); ++row) {
        for (std::size_t lane = 0; lane < stream_values[row].size(); ++lane) {
            const auto raw =
                std::bit_cast<std::uint32_t>(stream_values[row][lane]);
            for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                const auto& cell = mem.west_register(
                    0,
                    lane,
                    ftlpu::hw::kMxmBoundaryStreamRegisterColumn,
                    row * sizeof(float) + byte);
                assert(cell.valid);
                assert(
                    cell.data
                    == static_cast<std::uint8_t>(
                        (raw >> (byte * 8)) & 0xffu));
            }
            assert(mxm.block_accumulator().value(
                kTestAddress, row, lane) == 0.0f);
        }
    }

    return 0;
}
