#include "ftlpu/sxm/slice.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace {

using TileVector = ftlpu::SxmSlice::TileVector<std::int32_t>;
using StreamVector = ftlpu::SxmSlice::StreamVector<std::int32_t>;
using Matrix16 = ftlpu::SxmSlice::Matrix16<std::int32_t>;

TileVector lane_vector(std::int32_t base)
{
    TileVector vector{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        vector[lane] = base + static_cast<std::int32_t>(lane);
    }
    return vector;
}

StreamVector stream_vector()
{
    StreamVector data{};
    for (std::size_t row = 0; row < ftlpu::hw::kTileRows; ++row) {
        data[row] = lane_vector(static_cast<std::int32_t>(row * ftlpu::hw::kLanesPerTile));
    }
    return data;
}

ftlpu::SxmInstruction::StreamList stream_range(std::size_t first, std::size_t count)
{
    ftlpu::SxmInstruction::StreamList streams;
    streams.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        streams.push_back(ftlpu::SxmStreamId {first + index});
    }
    return streams;
}

} // namespace

int main()
{
    auto map = ftlpu::Distribute16::identity_map();
    map[0] = ftlpu::hw::kLanesPerTile - 1;
    map[1] = 0;
    map[2] = ftlpu::Distribute16::kZeroFill;
    const auto distributed = ftlpu::SxmSlice::distribute(lane_vector(100), map, -1);
    assert(distributed[0] == 107);
    assert(distributed[1] == 100);
    assert(distributed[2] == -1);
    assert(distributed[3] == 103);

    Matrix16 matrix{};
    for (std::size_t stream = 0; stream < ftlpu::hw::kLanesPerTile; ++stream) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            matrix[stream][lane] = static_cast<std::int32_t>(stream * 100 + lane);
        }
    }
    const auto transposed = ftlpu::SxmSlice::transpose(matrix);
    assert(transposed[0][1] == 100);
    assert(transposed[7][3] == 307);
    assert(transposed[7][6] == 607);

    const auto input_stream = stream_vector();
    const auto north = ftlpu::SxmSlice::shift_select(input_stream, ftlpu::SxmShiftSource::NorthShifted, 1, -1);
    assert(north[0][0] == 1);
    assert(north[0][6] == 7);
    assert(north[0][7] == 8);
    assert(north[ftlpu::hw::kTileRows - 2][7] == 24);
    assert(north[ftlpu::hw::kTileRows - 1][0] == 25);
    assert(north[ftlpu::hw::kTileRows - 1][7] == -1);

    const auto south = ftlpu::SxmSlice::shift_select(input_stream, ftlpu::SxmShiftSource::SouthShifted, 2, -1);
    assert(south[0][0] == -1);
    assert(south[0][1] == -1);
    assert(south[0][2] == 0);
    assert(south[1][0] == 6);
    assert(south[ftlpu::hw::kTileRows - 1][7] == 29);

    auto permutation = ftlpu::Permute320::identity_map();
    for (std::size_t index = 0; index < ftlpu::Permute320::kTotalLanes; ++index) {
        permutation[index] = ftlpu::Permute320::kTotalLanes - 1 - index;
    }
    const auto permuted = ftlpu::SxmSlice::permute(input_stream, permutation);
    assert(permuted[0][0] == 31);
    assert(permuted[0][7] == 24);
    assert(permuted[ftlpu::hw::kTileRows - 1][7] == 0);

    bool caught = false;
    permutation[1] = permutation[0];
    try {
        static_cast<void>(ftlpu::SxmSlice::permute(input_stream, permutation));
    }
    catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    using IntUnitGroup = ftlpu::SxmUnitGroup<std::int32_t>;
    auto set_input = [](IntUnitGroup::StreamState& state, std::size_t stream, const TileVector& values) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            state[stream][lane] = IntUnitGroup::Word {
                values[lane],
                lane + 1 == ftlpu::hw::kLanesPerTile,
            };
        }
    };

    IntUnitGroup sxm;
    IntUnitGroup::StreamState inputs{};
    for (std::size_t stream = 0; stream < ftlpu::hw::kLanesPerTile; ++stream) {
        set_input(inputs, stream, lane_vector(static_cast<std::int32_t>(stream * 100)));
    }

    sxm.issue(ftlpu::SxmInstruction::Transpose(stream_range(0, 8), stream_range(8, 8)));
    auto evaluation = sxm.evaluate(inputs);
    sxm.complete_cycle();
    assert(evaluation.produced[8]);
    assert(evaluation.outputs[8][3]->data == 300);
    assert(evaluation.outputs[15][3]->data == 307);

    auto chained_map = ftlpu::Distribute16::identity_map();
    chained_map[0] = 3;
    sxm.issue(ftlpu::SxmInstruction::Distribute(
        ftlpu::SxmStreamId {8},
        ftlpu::SxmStreamId {40},
        chained_map));
    evaluation = sxm.evaluate(evaluation.outputs);
    sxm.complete_cycle();
    assert(evaluation.produced[40]);

    sxm.issue(ftlpu::SxmInstruction::Distribute(
        ftlpu::SxmStreamId {40},
        ftlpu::SxmStreamId {41},
        ftlpu::Distribute16::identity_map()));
    evaluation = sxm.evaluate(evaluation.outputs);
    sxm.complete_cycle();
    assert(evaluation.consumed[40]);
    assert(evaluation.produced[41]);
    assert(evaluation.outputs[41][0]->data == 300);

    IntUnitGroup same_cycle_dependency_sxm;
    IntUnitGroup::StreamState dependency_inputs{};
    set_input(dependency_inputs, 0, lane_vector(0));
    same_cycle_dependency_sxm.issue(ftlpu::SxmInstruction::Distribute(
        ftlpu::SxmStreamId {0},
        ftlpu::SxmStreamId {1},
        ftlpu::Distribute16::identity_map()));
    same_cycle_dependency_sxm.issue(ftlpu::SxmInstruction::Distribute(
        ftlpu::SxmStreamId {1},
        ftlpu::SxmStreamId {2},
        ftlpu::Distribute16::identity_map()));
    caught = false;
    try {
        static_cast<void>(same_cycle_dependency_sxm.evaluate(dependency_inputs));
    }
    catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    IntUnitGroup width_limited_sxm;
    for (std::size_t op = 0; op < ftlpu::hw::kSxmConcurrentStreamOps; ++op) {
        width_limited_sxm.issue(ftlpu::SxmInstruction::Distribute(
            ftlpu::SxmStreamId {0},
            ftlpu::SxmStreamId {1},
            ftlpu::Distribute16::identity_map()));
    }
    caught = false;
    try {
        width_limited_sxm.issue(ftlpu::SxmInstruction::Distribute(
            ftlpu::SxmStreamId {0},
            ftlpu::SxmStreamId {1},
            ftlpu::Distribute16::identity_map()));
    }
    catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    // The SR-facing slice consumes from the global fabric and stages its
    // output without retaining a second copy of either stream.
    ftlpu::StreamRegisterFabric fabric(2);
    const auto east0 = ftlpu::StreamId::East(0);
    const auto east1 = ftlpu::StreamId::East(1);
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            fabric.initialize_cell(
                0,
                tile,
                lane,
                east0,
                ftlpu::StreamCell::Valid(
                    static_cast<std::uint8_t>(tile * ftlpu::hw::kLanesPerTile + lane),
                    lane + 1 == ftlpu::hw::kLanesPerTile,
                    77));
        }
    }

    auto sr_map = ftlpu::Distribute16::identity_map();
    sr_map[0] = ftlpu::hw::kLanesPerTile - 1;
    ftlpu::SxmSlice sr_sxm(ftlpu::SxmStreamPortMap::SameDirection(0, 1));
    sr_sxm.issue(ftlpu::SxmInstruction::Distribute(
        ftlpu::SxmStreamId {east0.packed()},
        ftlpu::SxmStreamId {east1.packed()},
        sr_map));

    fabric.begin_cycle();
    sr_sxm.evaluate(fabric);
    fabric.commit_cycle();
    assert(sr_sxm.cycle() == 1);
    assert(!fabric.cell(0, 0, 0, east0).valid);
    assert(fabric.cell(1, 0, 0, east1).valid);
    assert(fabric.cell(1, 0, 0, east1).data == 7);
    assert(fabric.cell(1, ftlpu::hw::kTileRows - 1, 0, east1).data == 31);

    // FP16 transpose accepts one low/high byte-stream row per cycle into one
    // buffer.  Tile t sees logical row r at physical cycle r+t.
    ftlpu::StreamRegisterFabric transpose_fabric(2);
    ftlpu::SxmSlice streaming_transpose(ftlpu::SxmStreamPortMap::SameDirection(0, 1));
    const auto low_in = ftlpu::StreamId::East(0);
    const auto high_in = ftlpu::StreamId::East(1);
    const auto low_out = ftlpu::StreamId::East(2);
    const auto high_out = ftlpu::StreamId::East(3);
    const auto routed_low_out = ftlpu::StreamId::East(4);
    const auto routed_high_out = ftlpu::StreamId::East(5);
    const auto transpose_instruction = ftlpu::SxmInstruction::Transpose(
        {{low_in.packed()}, {high_in.packed()}},
        {{low_out.packed()}, {high_out.packed()}});

    // Transpose control is a four-tile south-to-north wave.  Data capture is
    // therefore enabled in tile n exactly n cycles after ICU issues it.
    ftlpu::StreamRegisterFabric transpose_control_fabric(2);
    ftlpu::SxmSlice transpose_control(ftlpu::SxmStreamPortMap::SameDirection(0, 1));
    transpose_control.issue(transpose_instruction);
    for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows; ++cycle) {
        transpose_control_fabric.begin_cycle();
        transpose_control.evaluate(transpose_control_fabric);
        transpose_control_fabric.commit_cycle();
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            const auto expected = tile == cycle + 1;
            assert(transpose_control.transpose_instruction_at(tile).has_value() == expected);
        }
    }

    const auto transpose_value = [](
                                     std::size_t block,
                                     std::size_t plane,
                                     std::size_t tile,
                                     std::size_t row,
                                     std::size_t lane) {
        return static_cast<std::uint8_t>(
            block * 101 + plane * 43 + tile * 17 + row * 8 + lane);
    };

    constexpr auto kLastCaptureCycle =
        ftlpu::hw::kLanesPerTile + ftlpu::hw::kTileRows - 2;
    constexpr auto kPermuteStart = kLastCaptureCycle + 2;
    constexpr auto kTestCycles = kPermuteStart + ftlpu::hw::kLanesPerTile;
    for (std::size_t cycle = 0; cycle < kTestCycles; ++cycle) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            if (cycle < tile) continue;
            const auto row = cycle - tile;
            if (row >= ftlpu::hw::kLanesPerTile) continue;
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                transpose_fabric.initialize_cell(
                    0, tile, lane, low_in,
                    ftlpu::StreamCell::Valid(
                        transpose_value(0, 0, tile, row, lane),
                        lane + 1 == ftlpu::hw::kLanesPerTile));
                transpose_fabric.initialize_cell(
                    0, tile, lane, high_in,
                    ftlpu::StreamCell::Valid(
                        transpose_value(0, 1, tile, row, lane),
                        lane + 1 == ftlpu::hw::kLanesPerTile));
            }
        }
        if (cycle < ftlpu::hw::kLanesPerTile) {
            streaming_transpose.issue(transpose_instruction);
        }
        if (cycle >= kPermuteStart) {
            streaming_transpose.issue(ftlpu::SxmInstruction::Permute(
                {{low_out.packed()}, {high_out.packed()}},
                {{routed_low_out.packed()}, {routed_high_out.packed()}},
                ftlpu::Permute320::identity_map()));
        }

        transpose_fabric.begin_cycle();
        streaming_transpose.evaluate(transpose_fabric);
        transpose_fabric.commit_cycle();

        if (cycle < kPermuteStart) {
            assert(!transpose_fabric.segment_valid(1, 0, low_out));
            assert(!transpose_fabric.segment_valid(1, 0, high_out));
            continue;
        }

        const auto column = cycle - kPermuteStart;
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                assert(transpose_fabric.cell(1, tile, lane, routed_low_out).data
                    == transpose_value(0, 0, tile, lane, column));
                assert(transpose_fabric.cell(1, tile, lane, routed_high_out).data
                    == transpose_value(0, 1, tile, lane, column));
            }
        }
    }

    // A parallel FP16 beat carries all eight low/high row pairs.  Transpose
    // produces the canonical block.  Permute exchanges complete 8x8 blocks
    // across superlanes on the following cycle.
    ftlpu::StreamRegisterFabric parallel_fabric(2);
    ftlpu::SxmSlice parallel_sxm(ftlpu::SxmStreamPortMap::SameDirection(0, 1));
    const auto parallel_value = [](
                                    std::size_t plane,
                                    std::size_t row,
                                    std::size_t tile,
                                    std::size_t lane) {
        return static_cast<std::uint8_t>(plane * 97 + row * 19 + tile * 7 + lane);
    };
    for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows; ++cycle) {
        const auto tile = cycle;
        for (std::size_t row = 0; row < ftlpu::hw::kLanesPerTile; ++row) {
            for (std::size_t plane = 0; plane < ftlpu::SxmSlice::kTransposeBytePlanes; ++plane) {
                const auto input_stream = ftlpu::StreamId::East(
                    row * ftlpu::SxmSlice::kTransposeBytePlanes + plane);
                for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    parallel_fabric.initialize_cell(
                        0, tile, lane, input_stream,
                        ftlpu::StreamCell::Valid(
                            parallel_value(plane, row, tile, lane),
                            lane + 1 == ftlpu::hw::kLanesPerTile));
                }
            }
        }
        if (cycle == 0) {
            parallel_sxm.issue(ftlpu::SxmInstruction::Transpose(
                stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
                stream_range(16, 2 * ftlpu::hw::kLanesPerTile)));
        }
        parallel_fabric.begin_cycle();
        parallel_sxm.evaluate(parallel_fabric);
        parallel_fabric.commit_cycle();
    }

    auto reverse_superlanes = ftlpu::Permute320::identity_map();
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            reverse_superlanes[tile * ftlpu::hw::kLanesPerTile + lane] =
                (ftlpu::hw::kTileRows - 1 - tile) * ftlpu::hw::kLanesPerTile + lane;
        }
    }
    parallel_sxm.issue(ftlpu::SxmInstruction::Permute(
        stream_range(16, 2 * ftlpu::hw::kLanesPerTile),
        stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
        reverse_superlanes));
    parallel_fabric.begin_cycle();
    parallel_sxm.evaluate(parallel_fabric);
    parallel_fabric.commit_cycle();

    for (std::size_t row = 0; row < ftlpu::hw::kLanesPerTile; ++row) {
        for (std::size_t plane = 0; plane < ftlpu::SxmSlice::kTransposeBytePlanes; ++plane) {
            const auto output_stream = ftlpu::StreamId::East(
                row * ftlpu::SxmSlice::kTransposeBytePlanes + plane);
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                const auto source_tile = ftlpu::hw::kTileRows - 1 - tile;
                for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    assert(parallel_fabric.cell(1, tile, lane, output_stream).data
                        == parallel_value(plane, lane, source_tile, row));
                }
            }
        }
    }

    // A 32x32 transpose is presented as four cyclic block diagonals.  Beat p
    // carries B[i][(i+p)%4] on source superlane i; Permute moves the locally
    // transposed block to destination superlane (i+p)%4.
    ftlpu::StreamRegisterFabric diagonal_fabric(2);
    ftlpu::SxmSlice diagonal_sxm(ftlpu::SxmStreamPortMap::SameDirection(0, 1));
    const auto diagonal_value = [](
                                    std::size_t plane,
                                    std::size_t block_row,
                                    std::size_t block_column,
                                    std::size_t row,
                                    std::size_t column) {
        return static_cast<std::uint8_t>(
            plane * 113 + block_row * 29 + block_column * 11 + row * 7 + column);
    };
    for (std::size_t diagonal = 0; diagonal < ftlpu::hw::kTileRows; ++diagonal) {
        for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows; ++cycle) {
            const auto source_tile = cycle;
            for (std::size_t row = 0; row < ftlpu::hw::kLanesPerTile; ++row) {
                for (std::size_t plane = 0;
                     plane < ftlpu::SxmSlice::kTransposeBytePlanes;
                     ++plane) {
                    const auto input_stream = ftlpu::StreamId::East(
                        row * ftlpu::SxmSlice::kTransposeBytePlanes + plane);
                    const auto block_column =
                        (source_tile + diagonal) % ftlpu::hw::kTileRows;
                    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                        diagonal_fabric.initialize_cell(
                            0, source_tile, lane, input_stream,
                            ftlpu::StreamCell::Valid(
                                diagonal_value(
                                    plane, source_tile, block_column, row, lane),
                                lane + 1 == ftlpu::hw::kLanesPerTile));
                    }
                }
            }
            if (cycle == 0) {
                diagonal_sxm.issue(ftlpu::SxmInstruction::Transpose(
                    stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
                    stream_range(16, 2 * ftlpu::hw::kLanesPerTile)));
            }
            diagonal_fabric.begin_cycle();
            diagonal_sxm.evaluate(diagonal_fabric);
            diagonal_fabric.commit_cycle();
        }

        auto diagonal_map = ftlpu::Permute320::identity_map();
        for (std::size_t destination_tile = 0;
             destination_tile < ftlpu::hw::kTileRows;
             ++destination_tile) {
            const auto source_tile =
                (destination_tile + ftlpu::hw::kTileRows - diagonal)
                % ftlpu::hw::kTileRows;
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                diagonal_map[destination_tile * ftlpu::hw::kLanesPerTile + lane] =
                    source_tile * ftlpu::hw::kLanesPerTile + lane;
            }
        }
        diagonal_sxm.issue(ftlpu::SxmInstruction::Permute(
            stream_range(16, 2 * ftlpu::hw::kLanesPerTile),
            stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
            diagonal_map));
        diagonal_fabric.begin_cycle();
        diagonal_sxm.evaluate(diagonal_fabric);
        diagonal_fabric.commit_cycle();

        for (std::size_t destination_tile = 0;
             destination_tile < ftlpu::hw::kTileRows;
             ++destination_tile) {
            const auto source_tile =
                (destination_tile + ftlpu::hw::kTileRows - diagonal)
                % ftlpu::hw::kTileRows;
            for (std::size_t row = 0; row < ftlpu::hw::kLanesPerTile; ++row) {
                for (std::size_t plane = 0;
                     plane < ftlpu::SxmSlice::kTransposeBytePlanes;
                     ++plane) {
                    const auto output_stream = ftlpu::StreamId::East(
                        row * ftlpu::SxmSlice::kTransposeBytePlanes + plane);
                    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                        assert(diagonal_fabric.cell(
                                   1, destination_tile, lane, output_stream).data
                            == diagonal_value(
                                plane, source_tile, destination_tile, lane, row));
                    }
                }
            }
        }
    }

    // With 16-stream ingress, one beat is one complete 8x8 block per
    // superlane.  A south-to-north one-cycle skew turns row-major block beats
    // B[p][i] into the physical-cycle diagonal p+i=c.
    ftlpu::StreamRegisterFabric wavefront_fabric(2);
    ftlpu::SxmSlice wavefront_sxm(ftlpu::SxmStreamPortMap::SameDirection(0, 1));
    const auto wavefront_value = [](
                                     std::size_t plane,
                                     std::size_t block_row,
                                     std::size_t block_column,
                                     std::size_t row,
                                     std::size_t column) {
        return static_cast<std::uint8_t>(
            plane * 109 + block_row * 31 + block_column * 13 + row * 7 + column);
    };
    constexpr auto kWavefrontCycles = 2 * ftlpu::hw::kTileRows - 1;
    for (std::size_t cycle = 0; cycle <= kWavefrontCycles; ++cycle) {
        if (cycle < kWavefrontCycles) {
            for (std::size_t source_tile = 0;
                 source_tile < ftlpu::hw::kTileRows;
                 ++source_tile) {
                if (cycle < source_tile) continue;
                const auto block_row = cycle - source_tile;
                if (block_row >= ftlpu::hw::kTileRows) continue;
                for (std::size_t row = 0; row < ftlpu::hw::kLanesPerTile; ++row) {
                    for (std::size_t plane = 0;
                         plane < ftlpu::SxmSlice::kTransposeBytePlanes;
                         ++plane) {
                        const auto input_stream = ftlpu::StreamId::East(
                            row * ftlpu::SxmSlice::kTransposeBytePlanes + plane);
                        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                            wavefront_fabric.initialize_cell(
                                0, source_tile, lane, input_stream,
                                ftlpu::StreamCell::Valid(
                                    wavefront_value(
                                        plane, block_row, source_tile, row, lane),
                                    lane + 1 == ftlpu::hw::kLanesPerTile));
                        }
                    }
                }
            }
            wavefront_sxm.issue(ftlpu::SxmInstruction::Transpose(
                stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
                stream_range(16, 2 * ftlpu::hw::kLanesPerTile)));
        }

        const auto permute_wave = cycle - 1;
        auto cycle_map = ftlpu::Permute320::identity_map();
        for (std::size_t destination_tile = 0;
             destination_tile < ftlpu::hw::kTileRows;
             ++destination_tile) {
            const auto source_tile =
                (permute_wave + ftlpu::hw::kTileRows - destination_tile)
                % ftlpu::hw::kTileRows;
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                cycle_map[destination_tile * ftlpu::hw::kLanesPerTile + lane] =
                    source_tile * ftlpu::hw::kLanesPerTile + lane;
            }
        }
        if (cycle > 0) {
            wavefront_sxm.issue(ftlpu::SxmInstruction::Permute(
                stream_range(16, 2 * ftlpu::hw::kLanesPerTile),
                stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
                cycle_map));
        }
        wavefront_fabric.begin_cycle();
        wavefront_sxm.evaluate(wavefront_fabric);
        wavefront_fabric.commit_cycle();

        if (cycle == 0) continue;
        for (std::size_t destination_tile = 0;
             destination_tile < ftlpu::hw::kTileRows;
             ++destination_tile) {
            const auto source_tile =
                (permute_wave + ftlpu::hw::kTileRows - destination_tile)
                % ftlpu::hw::kTileRows;
            if (permute_wave < source_tile) continue;
            const auto block_row = permute_wave - source_tile;
            if (block_row != destination_tile) continue;
            for (std::size_t row = 0; row < ftlpu::hw::kLanesPerTile; ++row) {
                for (std::size_t plane = 0;
                     plane < ftlpu::SxmSlice::kTransposeBytePlanes;
                     ++plane) {
                    const auto output_stream = ftlpu::StreamId::East(
                        row * ftlpu::SxmSlice::kTransposeBytePlanes + plane);
                    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                        assert(wavefront_fabric.cell(
                                   1, destination_tile, lane, output_stream).data
                            == wavefront_value(
                                plane, block_row, source_tile, lane, row));
                    }
                }
            }
        }
    }

    return 0;
}
