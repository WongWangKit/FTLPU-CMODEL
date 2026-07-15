#pragma once

#include "ftlpu/core/stream.hpp"
#include "ftlpu/sxm/distributor.hpp"
#include "ftlpu/sxm/permute.hpp"
#include "ftlpu/sxm/shift.hpp"
#include "ftlpu/sxm/transpose.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

// Stateless SXM data-path executor.  It owns only the instructions issued for
// the current cycle; all stream data is supplied explicitly by the caller and
// returned in Evaluation.  In particular, this class is not an SR store.
template <typename T>
class SxmUnitGroup {
public:
    static constexpr std::size_t kMaxConcurrentOps = hw::kSxmConcurrentStreamOps;

    using Word = StreamWord<T>;
    using LaneSlot = StreamValue<T>;
    using TileVectorSlot = std::array<LaneSlot, hw::kLanesPerTile>;
    using StreamState = std::array<TileVectorSlot, hw::kStreams>;

    struct Evaluation {
        StreamState outputs{};
        std::array<bool, hw::kStreams> consumed{};
        std::array<bool, hw::kStreams> produced{};
    };

    void reset()
    {
        issued_.clear();
        issued_stream_ops_ = 0;
        cycle_ = 0;
    }

    std::size_t cycle() const noexcept
    {
        return cycle_;
    }

    bool can_issue() const noexcept
    {
        return issued_stream_ops_ < kMaxConcurrentOps;
    }

    bool can_issue(const SxmInstruction& instruction) const
    {
        return issued_stream_ops_ + stream_op_count(instruction) <= kMaxConcurrentOps;
    }

    std::size_t issued_count() const noexcept
    {
        return issued_.size();
    }

    std::size_t issued_stream_ops() const noexcept
    {
        return issued_stream_ops_;
    }

    const std::vector<SxmInstruction>& issued_instructions() const noexcept
    {
        return issued_;
    }

    void issue(SxmInstruction instruction)
    {
        validate_instruction_shape(instruction);
        if (!can_issue(instruction)) {
            throw std::logic_error("SXM issue width exceeded for this cycle");
        }
        check_streams(instruction.src_streams);
        check_streams(instruction.dst_streams);
        issued_stream_ops_ += stream_op_count(instruction);
        issued_.push_back(std::move(instruction));
    }

    Evaluation evaluate(const StreamState& inputs) const
    {
        Evaluation result{};
        for (const auto& instruction : issued_) {
            execute(inputs, result, instruction);
        }
        return result;
    }

    // Called once after the owning slice has evaluated every physical tile.
    void complete_cycle()
    {
        issued_.clear();
        issued_stream_ops_ = 0;
        ++cycle_;
    }

private:
    using TileWordVector = std::array<Word, hw::kLanesPerTile>;
    using Matrix16 = std::array<TileWordVector, hw::kLanesPerTile>;
    using StreamWordVector = std::array<TileWordVector, hw::kTileRows>;

    static bool vector_available(const TileVectorSlot& vector)
    {
        return std::all_of(vector.begin(), vector.end(), [](const auto& lane) {
            return lane.has_value();
        });
    }

    static bool vector_occupied(const TileVectorSlot& vector)
    {
        return std::any_of(vector.begin(), vector.end(), [](const auto& lane) {
            return lane.has_value();
        });
    }

    static void check_stream(SxmStreamId id)
    {
        if (id.stream >= hw::kStreams) {
            throw std::out_of_range("SXM stream id is outside the 64 encoded streams");
        }
    }

    static void check_streams(const SxmInstruction::StreamList& streams)
    {
        for (const auto stream : streams) {
            check_stream(stream);
        }
    }

    static void validate_instruction_shape(const SxmInstruction& instruction)
    {
        switch (instruction.opcode) {
        case SxmOpcode::Distribute:
            require_stream_count(instruction, 1);
            break;
        case SxmOpcode::Transpose:
            require_stream_count(instruction, hw::kLanesPerTile);
            break;
        case SxmOpcode::ShiftSelect:
        case SxmOpcode::Permute:
            require_stream_count(instruction, hw::kTileRows);
            break;
        }
    }

    static std::size_t stream_op_count(const SxmInstruction& instruction)
    {
        return std::max(instruction.src_streams.size(), instruction.dst_streams.size());
    }

    static void require_stream_count(const SxmInstruction& instruction, std::size_t count)
    {
        if (instruction.src_streams.size() != count || instruction.dst_streams.size() != count) {
            throw std::invalid_argument("SXM instruction has an invalid source/destination stream count");
        }
    }

    static TileWordVector read_vector(const StreamState& inputs, SxmStreamId id)
    {
        const auto& slot = inputs[id.stream];
        if (!vector_available(slot)) {
            throw std::logic_error("SXM instruction source stream is not available");
        }

        TileWordVector vector{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            vector[lane] = *slot[lane];
        }
        return vector;
    }

    static void mark_consumed(Evaluation& result, SxmStreamId id)
    {
        result.consumed[id.stream] = true;
    }

    static void write_vector(Evaluation& result, SxmStreamId id, const TileWordVector& vector)
    {
        auto& slot = result.outputs[id.stream];
        if (vector_occupied(slot)) {
            throw std::logic_error("two SXM operations write the same destination stream");
        }
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            slot[lane] = vector[lane];
        }
        result.produced[id.stream] = true;
    }

    static void execute(const StreamState& inputs, Evaluation& result, const SxmInstruction& instruction)
    {
        switch (instruction.opcode) {
        case SxmOpcode::Distribute:
            execute_distribute(inputs, result, instruction);
            break;
        case SxmOpcode::Transpose:
            execute_transpose(inputs, result, instruction);
            break;
        case SxmOpcode::ShiftSelect:
            execute_shift_select(inputs, result, instruction);
            break;
        case SxmOpcode::Permute:
            execute_permute(inputs, result, instruction);
            break;
        }
    }

    static void execute_distribute(
        const StreamState& inputs,
        Evaluation& result,
        const SxmInstruction& instruction)
    {
        const auto input = read_vector(inputs, instruction.src_streams[0]);
        mark_consumed(result, instruction.src_streams[0]);
        write_vector(
            result,
            instruction.dst_streams[0],
            Distribute16::apply(input, instruction.lane_map, Word {}));
    }

    static void execute_transpose(
        const StreamState& inputs,
        Evaluation& result,
        const SxmInstruction& instruction)
    {
        Matrix16 input{};
        for (std::size_t stream = 0; stream < hw::kLanesPerTile; ++stream) {
            input[stream] = read_vector(inputs, instruction.src_streams[stream]);
            mark_consumed(result, instruction.src_streams[stream]);
        }

        const auto output = Transpose16x16::apply(input);
        for (std::size_t stream = 0; stream < hw::kLanesPerTile; ++stream) {
            write_vector(result, instruction.dst_streams[stream], output[stream]);
        }
    }

    static void execute_shift_select(
        const StreamState& inputs,
        Evaluation& result,
        const SxmInstruction& instruction)
    {
        StreamWordVector input{};
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            input[row] = read_vector(inputs, instruction.src_streams[row]);
            mark_consumed(result, instruction.src_streams[row]);
        }

        const auto output = ShiftSelect::apply(
            input,
            instruction.shift_source,
            instruction.shift_distance,
            Word {});
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            write_vector(result, instruction.dst_streams[row], output[row]);
        }
    }

    static void execute_permute(
        const StreamState& inputs,
        Evaluation& result,
        const SxmInstruction& instruction)
    {
        StreamWordVector input{};
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            input[row] = read_vector(inputs, instruction.src_streams[row]);
            mark_consumed(result, instruction.src_streams[row]);
        }

        const auto output = Permute320::apply(input, instruction.permute_map);
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            write_vector(result, instruction.dst_streams[row], output[row]);
        }
    }

    std::vector<SxmInstruction> issued_{};
    std::size_t issued_stream_ops_{0};
    std::size_t cycle_{0};
};

} // namespace ftlpu
