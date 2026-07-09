#include "ftlpu/sxm/slice.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace {

using Vector16 = ftlpu::SxmSlice::Vector<std::int32_t>;
using Plane320 = ftlpu::SxmSlice::Plane<std::int32_t>;
using Matrix16 = ftlpu::SxmSlice::Matrix16<std::int32_t>;

Vector16 lane_vector(std::int32_t base)
{
    Vector16 vector{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        vector[lane] = base + static_cast<std::int32_t>(lane);
    }
    return vector;
}

Plane320 plane()
{
    Plane320 data{};
    for (std::size_t row = 0; row < ftlpu::hw::kTileRows; ++row) {
        data[row] = lane_vector(static_cast<std::int32_t>(row * ftlpu::hw::kLanesPerTile));
    }
    return data;
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

    const auto input_plane = plane();
    const auto north = ftlpu::SxmSlice::shift_select(input_plane, ftlpu::SxmShiftSource::NorthShifted, 1, -1);
    assert(north[0][0] == 16);
    assert(north[18][15] == 319);
    assert(north[19][0] == -1);

    const auto south = ftlpu::SxmSlice::shift_select(input_plane, ftlpu::SxmShiftSource::SouthShifted, 2, -1);
    assert(south[0][0] == -1);
    assert(south[1][0] == -1);
    assert(south[2][0] == 0);
    assert(south[19][15] == 287);

    auto permutation = ftlpu::Permute320::identity_map();
    for (std::size_t index = 0; index < ftlpu::Permute320::kTotalLanes; ++index) {
        permutation[index] = ftlpu::Permute320::kTotalLanes - 1 - index;
    }
    const auto permuted = ftlpu::SxmSlice::permute(input_plane, permutation);
    assert(permuted[0][0] == 319);
    assert(permuted[0][15] == 304);
    assert(permuted[19][15] == 0);

    bool caught = false;
    permutation[1] = permutation[0];
    try {
        static_cast<void>(ftlpu::SxmSlice::permute(input_plane, permutation));
    }
    catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    return 0;
}
