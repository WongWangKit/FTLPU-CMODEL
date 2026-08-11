#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace ftlpu {

enum class SxmOpcode {
    // Preserve the hardware encoding formerly assigned to these operations.
    Transpose = 2,
    Permute = 3,
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
    static constexpr std::size_t kTotalLanes = hw::kTileRows * hw::kLanesPerTile;

    using PermuteMap = std::array<std::size_t, kTotalLanes>;
    using StreamList = std::vector<SxmStreamId>;

    SxmOpcode opcode{SxmOpcode::Transpose};
    PermuteMap permute_map{};
    StreamList src_streams{};
    StreamList dst_streams{};

    static SxmInstruction Transpose(StreamList srcs, StreamList dsts)
    {
        SxmInstruction instruction{};
        instruction.opcode = SxmOpcode::Transpose;
        instruction.src_streams = std::move(srcs);
        instruction.dst_streams = std::move(dsts);
        return instruction;
    }

    static SxmInstruction Permute(
        StreamList srcs,
        StreamList dsts,
        PermuteMap map)
    {
        SxmInstruction instruction{};
        instruction.opcode = SxmOpcode::Permute;
        instruction.permute_map = map;
        instruction.src_streams = std::move(srcs);
        instruction.dst_streams = std::move(dsts);
        return instruction;
    }
};

} // namespace ftlpu
