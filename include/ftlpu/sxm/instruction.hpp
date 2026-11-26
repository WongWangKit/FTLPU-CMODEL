#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

// Legacy opcode values are reserved by serialized IR. The hardware accepts
// only a complete stream-to-stream Transpose; decoding rejects retired modes.
enum class SxmOpcode {
    ShiftSelect,
    Distribute,
    Transpose,
    Permute,
    LocalTransposeToStream,
    VectorRotate,
};

enum class SxmShiftSource {
    Unshifted,
    NorthShifted,
    SouthShifted,
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
    static constexpr std::size_t kAllOutputRows = static_cast<std::size_t>(-1);
    static constexpr std::size_t kAllInputRows = static_cast<std::size_t>(-1);
    // Select one byte-stream pair and fan it out to every transpose row.
    // Selector 8 retains the existing all-rows encoding; 9 is broadcast.
    static constexpr std::size_t kBroadcastInputRows = hw::kLanesPerTile + 1;
    // Selectors 9..15 emit row 0..6 and release the complete input block.
    // Row 7 already releases it; selector 8 remains all rows.
    static constexpr std::size_t kConsumeOutputRowBase = hw::kLanesPerTile + 1;
    static constexpr std::size_t kLastConsumeRow = 2 * hw::kLanesPerTile - 1;
    static constexpr bool valid_output_row(std::size_t row)
    {
        return row == kAllOutputRows || row < hw::kLanesPerTile
            || (row >= kConsumeOutputRowBase && row <= kLastConsumeRow);
    }
    static constexpr std::size_t kAllOutputTiles = static_cast<std::size_t>(-1);
    static constexpr std::size_t kTotalLanes = hw::kTileRows * hw::kLanesPerTile;

    using LaneMap = std::array<std::size_t, hw::kLanesPerTile>;
    using PermuteMap = std::array<std::size_t, kTotalLanes>;
    using StreamList = std::vector<SxmStreamId>;

    SxmOpcode opcode{SxmOpcode::ShiftSelect};
    SxmShiftSource shift_source{SxmShiftSource::Unshifted};
    std::size_t shift_distance{1};
    LaneMap lane_map{};
    PermuteMap permute_map{};
    std::size_t output_row{kAllOutputRows};
    std::size_t input_row{kAllInputRows};
    std::size_t output_tile{kAllOutputTiles};
    StreamList src_streams{};
    StreamList dst_streams{};

    bool is_stream_operation() const noexcept
    {
        return opcode == SxmOpcode::LocalTransposeToStream || opcode == SxmOpcode::VectorRotate;
    }

    // One byte stream or a consecutive low/high pair carrying a 16-bit element.
    static SxmInstruction StreamOperation(SxmOpcode operation, SxmStreamId src,
        SxmStreamId dst, std::size_t planes, std::size_t rotation = 0)
    {
        if ((operation != SxmOpcode::LocalTransposeToStream && operation != SxmOpcode::VectorRotate)
            || (planes != 1 && planes != 2) || rotation >= hw::kTileRows
            || (operation == SxmOpcode::LocalTransposeToStream && rotation != 0))
            throw std::invalid_argument("invalid SXM stream operation");
        for (auto base : {src.stream, dst.stream})
            if (base >= hw::kStreams || base % hw::kStreamsPerDirection + planes > hw::kStreamsPerDirection)
                throw std::invalid_argument("SXM stream range crosses direction boundary");
        SxmInstruction instruction{};
        instruction.opcode = operation;
        instruction.shift_distance = rotation;
        for (std::size_t p = 0; p < planes; ++p) {
            instruction.src_streams.push_back({src.stream+p});
            instruction.dst_streams.push_back({dst.stream+p});
        }
        return instruction;
    }

    void validate_stream_operation() const
    {
        if (src_streams.empty() || dst_streams.empty())
            throw std::invalid_argument("missing SXM stream");
        const auto expected = StreamOperation(opcode, src_streams.front(), dst_streams.front(),
            src_streams.size(), shift_distance);
        if (src_streams != expected.src_streams || dst_streams != expected.dst_streams)
            throw std::invalid_argument("SXM byte planes must be consecutive");
    }

    static SxmInstruction LocalTranspose(SxmStreamId src, SxmStreamId dst, std::size_t planes = 2)
    {
        return StreamOperation(SxmOpcode::LocalTransposeToStream, src, dst, planes);
    }

    static SxmInstruction VectorRotate(SxmStreamId src, SxmStreamId dst,
        std::size_t distance, std::size_t planes = 2)
    {
        return StreamOperation(SxmOpcode::VectorRotate, src, dst, planes, distance);
    }

    static PermuteMap identity_map() {
        PermuteMap result{};
        for (std::size_t i=0; i<result.size(); ++i) result[i]=i;
        return result;
    }

    bool is_vector_permute() const noexcept
    {
        return opcode == SxmOpcode::Permute && src_streams.size() <= 1
            && dst_streams.size() == 1;
    }

    bool reads_transpose_bank() const noexcept
    {
        return is_vector_permute() && src_streams.empty();
    }

    // With no SR source, slot 2*r+p selects transposed row r, byte plane p.
    static SxmInstruction PermuteFromBank(std::size_t row, std::size_t plane,
                                          SxmStreamId dst, PermuteMap map)
    {
        if (row >= hw::kLanesPerTile || plane >= 2)
            throw std::out_of_range("vector Permute bank row/plane outside the local block");
        auto instruction = Permute({}, {dst}, std::move(map));
        instruction.input_row = 2 * row + plane;
        return instruction;
    }

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
