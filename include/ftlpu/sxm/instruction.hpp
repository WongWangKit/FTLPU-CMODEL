#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>

namespace ftlpu {

enum class SxmOpcode {
    ShiftSelect,
    Distribute,
    Transpose,
    Permute,
};

enum class SxmShiftSource {
    Unshifted,
    NorthShifted,
    SouthShifted,
};

struct SxmInstruction {
    static constexpr std::size_t kZeroFill = static_cast<std::size_t>(-1);
    static constexpr std::size_t kTotalLanes = hw::kTileRows * hw::kLanesPerTile;

    using LaneMap = std::array<std::size_t, hw::kLanesPerTile>;
    using PermuteMap = std::array<std::size_t, kTotalLanes>;

    SxmOpcode opcode{SxmOpcode::ShiftSelect};
    SxmShiftSource shift_source{SxmShiftSource::Unshifted};
    std::size_t shift_distance{1};
    LaneMap lane_map{};
    PermuteMap permute_map{};
};

} // namespace ftlpu
