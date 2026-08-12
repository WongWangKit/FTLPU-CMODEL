#include "ftlpu/sxm/slice.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace {

constexpr std::size_t kRows = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kPlanes = ftlpu::SxmSlice::kTransposeBytePlanes;
constexpr std::size_t kBlocks = 3;

ftlpu::SxmInstruction::StreamList east_streams(
    std::size_t first,
    std::size_t count)
{
    auto streams = ftlpu::SxmInstruction::StreamList {};
    streams.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        streams.push_back(ftlpu::SxmStreamId {
            ftlpu::StreamId::East(first + index).packed()});
    }
    return streams;
}

std::uint8_t input_value(
    std::size_t plane,
    std::size_t block,
    std::size_t tile,
    std::size_t row,
    std::size_t lane)
{
    return static_cast<std::uint8_t>(
        plane * 101 + block * 37 + tile * 11 + row * 5 + lane);
}

} // namespace

int main()
{
    auto fabric = ftlpu::StreamRegisterFabric(2);
    auto sxm = ftlpu::SxmSlice(
        ftlpu::SxmStreamPortMap::SameDirection(0, 1));
    sxm.set_trace_enabled(false);

    const auto transpose = ftlpu::SxmInstruction::Transpose(
        east_streams(0, kPlanes * kRows),
        east_streams(16, kPlanes * kRows));

    auto reverse_tiles_and_lanes = ftlpu::Permute320::identity_map();
    for (std::size_t destination_tile = 0;
         destination_tile < ftlpu::hw::kTileRows;
         ++destination_tile) {
        const auto source_tile =
            ftlpu::hw::kTileRows - 1 - destination_tile;
        for (std::size_t lane = 0; lane < kRows; ++lane) {
            reverse_tiles_and_lanes[destination_tile * kRows + lane] =
                source_tile * kRows + (kRows - 1 - lane);
        }
    }
    const auto permute = ftlpu::SxmInstruction::Permute(
        east_streams(16, kPlanes * kRows),
        east_streams(0, kPlanes * kRows),
        reverse_tiles_and_lanes);

    auto invalid_transpose_rejected = false;
    try {
        sxm.issue(ftlpu::SxmInstruction::Transpose(
            east_streams(0, 2), east_streams(2, 2)));
    }
    catch (const std::invalid_argument&) {
        invalid_transpose_rejected = true;
    }
    assert(invalid_transpose_rejected);

    // A block starts every eight cycles.  Each tile receives the instruction
    // one cycle after its southern neighbour, matching the physical SR wave.
    constexpr auto kLastOutputCycle =
        kBlocks * kRows + (ftlpu::hw::kTileRows - 1) + 2 + (kRows - 1);
    auto captured_rows = std::size_t {0};
    auto transpose_bank_loads = std::size_t {0};
    auto transpose_rows = std::size_t {0};
    auto permute_rows = std::size_t {0};
    for (std::size_t cycle = 0; cycle <= kLastOutputCycle; ++cycle) {
        if (cycle < kBlocks * kRows && cycle % kRows == 0) {
            assert(sxm.can_issue(transpose));
            sxm.issue(transpose);
            assert(!sxm.can_issue(transpose));
            assert(sxm.can_issue(permute));
            sxm.issue(permute);
        }

        for (std::size_t tile = 0;
             tile < ftlpu::hw::kTileRows;
             ++tile) {
            if (cycle < tile) continue;
            const auto local_cycle = cycle - tile;
            if (local_cycle >= kBlocks * kRows) continue;
            const auto block = local_cycle / kRows;
            const auto row = local_cycle % kRows;
            for (std::size_t plane = 0; plane < kPlanes; ++plane) {
                const auto stream = ftlpu::StreamId::East(
                    row * kPlanes + plane);
                for (std::size_t lane = 0; lane < kRows; ++lane) {
                    fabric.initialize_cell(
                        0,
                        tile,
                        lane,
                        stream,
                        ftlpu::StreamCell::Valid(
                            input_value(
                                plane, block, tile, row, lane),
                            lane + 1 == kRows));
                }
            }
        }

        fabric.begin_cycle();
        sxm.evaluate(fabric);
        fabric.commit_cycle();
        const auto timing = sxm.timing_snapshot();
        captured_rows += timing.captured_rows;
        transpose_bank_loads += timing.transpose_bank_loads;
        transpose_rows += timing.transpose_rows;
        permute_rows += timing.permute_rows;

        for (std::size_t destination_tile = 0;
             destination_tile < ftlpu::hw::kTileRows;
             ++destination_tile) {
            const auto source_tile =
                ftlpu::hw::kTileRows - 1 - destination_tile;
            auto output_block = std::optional<std::size_t> {};
            std::size_t output_row = 0;
            for (std::size_t block = 0; block < kBlocks; ++block) {
                const auto first_output =
                    (block + 1) * kRows + source_tile + 2;
                if (cycle >= first_output
                    && cycle < first_output + kRows) {
                    output_block = block;
                    output_row = cycle - first_output;
                    break;
                }
            }

            for (std::size_t row = 0; row < kRows; ++row) {
                for (std::size_t plane = 0; plane < kPlanes; ++plane) {
                    const auto stream = ftlpu::StreamId::East(
                        row * kPlanes + plane);
                    const auto valid = output_block && row == output_row;
                    assert(fabric.segment_valid(
                               1, destination_tile, stream)
                        == valid);
                    if (!valid) continue;

                    for (std::size_t lane = 0; lane < kRows; ++lane) {
                        const auto source_lane = kRows - 1 - lane;
                        assert(fabric.cell(
                                   1,
                                   destination_tile,
                                   lane,
                                   stream).data
                            == input_value(
                                plane,
                                *output_block,
                                source_tile,
                                source_lane,
                                output_row));
                    }
                }
            }
        }
    }

    assert(captured_rows
        == kBlocks * kRows * ftlpu::hw::kTileRows);
    assert(transpose_bank_loads
        == kBlocks * ftlpu::hw::kTileRows);
    assert(transpose_rows
        == kBlocks * kRows * ftlpu::hw::kTileRows);
    assert(permute_rows
        == kBlocks * kRows * ftlpu::hw::kTileRows);

    return 0;
}
