#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

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

enum class SxmWeightLayout {
    VectorColumns,
    MatrixColumns,
};

struct SxmStreamId {
    std::size_t stream{0};
};

inline bool operator==(SxmStreamId lhs, SxmStreamId rhs)
{
    return lhs.stream == rhs.stream;
}

inline bool operator!=(SxmStreamId lhs, SxmStreamId rhs)
{
    return !(lhs == rhs);
}

struct SxmInstruction {
    static constexpr std::size_t kZeroFill = static_cast<std::size_t>(-1);
    static constexpr std::size_t kTotalLanes = hw::kTileRows * hw::kLanesPerTile;

    using LaneMap = std::array<std::size_t, hw::kLanesPerTile>;
    using PermuteMap = std::array<std::size_t, kTotalLanes>;
    using StreamList = std::vector<SxmStreamId>;

    SxmOpcode opcode{SxmOpcode::ShiftSelect};
    SxmShiftSource shift_source{SxmShiftSource::Unshifted};
    std::size_t shift_distance{1};
    LaneMap lane_map{};
    PermuteMap permute_map{};
    SxmWeightLayout weight_layout{SxmWeightLayout::VectorColumns};
    StreamList src_streams{};
    StreamList dst_streams{};

    static SxmInstruction Distribute(SxmStreamId src, SxmStreamId dst, LaneMap map)
    {
        SxmInstruction instruction{};
        instruction.opcode = SxmOpcode::Distribute;
        instruction.lane_map = map;
        instruction.src_streams = {src};
        instruction.dst_streams = {dst};
        return instruction;
    }

    static SxmInstruction Transpose(StreamList srcs, StreamList dsts)
    {
        SxmInstruction instruction{};
        instruction.opcode = SxmOpcode::Transpose;
        instruction.src_streams = std::move(srcs);
        instruction.dst_streams = std::move(dsts);
        return instruction;
    }

    static SxmInstruction TransposeStream(SxmStreamId src, SxmStreamId dst)
    {
        return Transpose(StreamList {src}, StreamList {dst});
    }

    static SxmInstruction ShiftSelect(
        StreamList srcs,
        StreamList dsts,
        SxmShiftSource source,
        std::size_t distance = 1)
    {
        SxmInstruction instruction{};
        instruction.opcode = SxmOpcode::ShiftSelect;
        instruction.shift_source = source;
        instruction.shift_distance = distance;
        instruction.src_streams = std::move(srcs);
        instruction.dst_streams = std::move(dsts);
        return instruction;
    }

    static SxmInstruction Permute(
        StreamList srcs,
        StreamList dsts,
        PermuteMap map,
        SxmWeightLayout weight_layout = SxmWeightLayout::VectorColumns)
    {
        SxmInstruction instruction{};
        instruction.opcode = SxmOpcode::Permute;
        instruction.permute_map = map;
        instruction.weight_layout = weight_layout;
        instruction.src_streams = std::move(srcs);
        instruction.dst_streams = std::move(dsts);
        return instruction;
    }

    static SxmInstruction PermuteStream(SxmStreamId src, SxmStreamId dst, PermuteMap map)
    {
        return Permute(StreamList {src}, StreamList {dst}, std::move(map));
    }

    static SxmInstruction PermuteToStreams(
        SxmStreamId src,
        StreamList dsts,
        PermuteMap map)
    {
        SxmInstruction instruction{};
        instruction.opcode = SxmOpcode::Permute;
        instruction.permute_map = std::move(map);
        instruction.src_streams = {src};
        instruction.dst_streams = std::move(dsts);
        return instruction;
    }
};

} // namespace ftlpu
