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

    ftlpu::SxmUnitGroup<std::int32_t> sxm;
    for (std::size_t stream = 0; stream < ftlpu::hw::kLanesPerTile; ++stream) {
        sxm.set_stream_input(
            ftlpu::SxmStreamId {stream},
            lane_vector(static_cast<std::int32_t>(stream * 100)));
    }

    sxm.issue(ftlpu::SxmInstruction::Transpose(stream_range(0, 16), stream_range(16, 16)));
    sxm.tick();
    assert(sxm.stream_available(ftlpu::SxmStreamId {16}));
    assert(sxm.stream(ftlpu::SxmStreamId {16})[3]->data == 300);
    assert(sxm.stream(ftlpu::SxmStreamId {23})[3]->data == 307);

    auto chained_map = ftlpu::Distribute16::identity_map();
    chained_map[0] = 3;
    sxm.issue(ftlpu::SxmInstruction::Distribute(
        ftlpu::SxmStreamId {16},
        ftlpu::SxmStreamId {40},
        chained_map));
    sxm.tick();
    assert(sxm.stream_available(ftlpu::SxmStreamId {40}));

    sxm.issue(ftlpu::SxmInstruction::Distribute(
        ftlpu::SxmStreamId {40},
        ftlpu::SxmStreamId {41},
        ftlpu::Distribute16::identity_map()));
    sxm.tick();
    assert(!sxm.stream_occupied(ftlpu::SxmStreamId {40}));
    assert(sxm.stream_available(ftlpu::SxmStreamId {41}));
    assert(sxm.stream(ftlpu::SxmStreamId {41})[0]->data == 300);

    ftlpu::SxmUnitGroup<std::int32_t> same_cycle_dependency_sxm;
    same_cycle_dependency_sxm.set_stream_input(ftlpu::SxmStreamId {0}, lane_vector(0));
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
        same_cycle_dependency_sxm.tick();
    }
    catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    ftlpu::SxmUnitGroup<std::int32_t> width_limited_sxm;
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

    return 0;
}
