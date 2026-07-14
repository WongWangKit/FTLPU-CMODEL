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
    map[0] = 15;
    map[1] = 0;
    map[2] = ftlpu::Distribute16::kZeroFill;
    const auto distributed = ftlpu::SxmSlice::distribute(lane_vector(100), map, -1);
    assert(distributed[0] == 115);
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
    assert(transposed[15][14] == 1415);

    const auto input_stream = stream_vector();
    const auto north = ftlpu::SxmSlice::shift_select(input_stream, ftlpu::SxmShiftSource::NorthShifted, 1, -1);
    assert(north[0][0] == 1);
    assert(north[0][14] == 15);
    assert(north[0][15] == 16);
    assert(north[18][15] == 304);
    assert(north[19][0] == 305);
    assert(north[19][15] == -1);

    const auto south = ftlpu::SxmSlice::shift_select(input_stream, ftlpu::SxmShiftSource::SouthShifted, 2, -1);
    assert(south[0][0] == -1);
    assert(south[0][1] == -1);
    assert(south[0][2] == 0);
    assert(south[1][0] == 14);
    assert(south[19][15] == 317);

    auto permutation = ftlpu::Permute320::identity_map();
    for (std::size_t index = 0; index < ftlpu::Permute320::kTotalLanes; ++index) {
        permutation[index] = ftlpu::Permute320::kTotalLanes - 1 - index;
    }
    const auto permuted = ftlpu::SxmSlice::permute(input_stream, permutation);
    assert(permuted[0][0] == 319);
    assert(permuted[0][15] == 304);
    assert(permuted[19][15] == 0);

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

    sxm.issue(ftlpu::SxmInstruction::Transpose(stream_range(0, 16), stream_range(16, 16)));
    auto evaluation = sxm.evaluate(inputs);
    sxm.complete_cycle();
    assert(evaluation.produced[16]);
    assert(evaluation.outputs[16][3]->data == 300);
    assert(evaluation.outputs[23][3]->data == 307);

    auto chained_map = ftlpu::Distribute16::identity_map();
    chained_map[0] = 3;
    sxm.issue(ftlpu::SxmInstruction::Distribute(
        ftlpu::SxmStreamId {16},
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
    sr_map[0] = 15;
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
    assert(fabric.cell(1, 0, 0, east1).data == 15);
    assert(fabric.cell(1, 19, 0, east1).data == 63);

    return 0;
}
