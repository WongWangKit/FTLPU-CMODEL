#pragma once

#include "ftlpu/core/stream.hpp"
#include "ftlpu/sxm/distributor.hpp"
#include "ftlpu/sxm/permute.hpp"
#include "ftlpu/sxm/shift.hpp"
#include "ftlpu/sxm/transpose.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

template <typename T>
class SxmUnitGroup {
public:
    static constexpr std::size_t kMaxConcurrentOps = hw::kSxmConcurrentStreamOps;

    using Word = StreamWord<T>;
    using LaneSlot = StreamValue<T>;
    using TileVectorSlot = std::array<LaneSlot, hw::kLanesPerTile>;

    void reset()
    {
        for (auto& stream : streams_) {
            clear_vector(stream);
        }
        issued_.clear();
        issued_stream_ops_ = 0;
        cycle_ = 0;
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    bool can_issue() const
    {
        return issued_stream_ops_ < kMaxConcurrentOps;
    }

    bool can_issue(const SxmInstruction& instruction) const
    {
        return issued_stream_ops_ + stream_op_count(instruction) <= kMaxConcurrentOps;
    }

    std::size_t issued_count() const
    {
        return issued_.size();
    }

    std::size_t issued_stream_ops() const
    {
        return issued_stream_ops_;
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

    void set_stream_input(SxmStreamId id, const TileVectorSlot& vector)
    {
        check_stream(id);
        if (vector_occupied(streams_[id.stream])) {
            throw std::logic_error("SXM input stream is already occupied");
        }
        if (!vector_available(vector)) {
            throw std::logic_error("SXM input stream requires all 16 lanes to be valid");
        }
        streams_[id.stream] = vector;
    }

    void set_stream_input(SxmStreamId id, const std::array<T, hw::kLanesPerTile>& values, bool last = true)
    {
        TileVectorSlot vector{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            vector[lane] = Word {values[lane], last && lane + 1 == hw::kLanesPerTile};
        }
        set_stream_input(id, vector);
    }

    bool stream_available(SxmStreamId id) const
    {
        check_stream(id);
        return vector_available(streams_[id.stream]);
    }

    bool stream_occupied(SxmStreamId id) const
    {
        check_stream(id);
        return vector_occupied(streams_[id.stream]);
    }

    const TileVectorSlot& stream(SxmStreamId id) const
    {
        check_stream(id);
        return streams_[id.stream];
    }

    TileVectorSlot take_stream(SxmStreamId id)
    {
        check_stream(id);
        auto vector = streams_[id.stream];
        clear_vector(streams_[id.stream]);
        return vector;
    }

    bool ready(const SxmInstruction& instruction) const
    {
        validate_instruction_shape(instruction);
        check_streams(instruction.src_streams);
        check_streams(instruction.dst_streams);

        for (const auto src : instruction.src_streams) {
            if (!stream_available(src)) {
                return false;
            }
        }
        for (const auto dst : instruction.dst_streams) {
            if (stream_occupied(dst) && !contains_stream(instruction.src_streams, dst)) {
                return false;
            }
        }
        return true;
    }

    void tick()
    {
        const auto input_snapshot = streams_;
        auto next_streams = streams_;
        for (const auto& instruction : issued_) {
            execute(input_snapshot, next_streams, instruction);
        }

        streams_ = std::move(next_streams);
        issued_.clear();
        issued_stream_ops_ = 0;
        ++cycle_;
    }

private:
    using TileWordVector = std::array<Word, hw::kLanesPerTile>;
    using Matrix16 = std::array<TileWordVector, hw::kLanesPerTile>;
    using StreamWordVector = std::array<TileWordVector, hw::kTileRows>;

    static void clear_vector(TileVectorSlot& vector)
    {
        for (auto& lane : vector) {
            lane.reset();
        }
    }

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

    static bool contains_stream(const SxmInstruction::StreamList& streams, SxmStreamId id)
    {
        return std::find(streams.begin(), streams.end(), id) != streams.end();
    }

    static void check_stream(SxmStreamId id)
    {
        if (id.stream >= hw::kStreams) {
            throw std::out_of_range("SXM stream id is outside the 64-stream file");
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

    static TileWordVector read_vector(const std::array<TileVectorSlot, hw::kStreams>& streams, SxmStreamId id)
    {
        const auto& slot = streams[id.stream];
        if (!vector_available(slot)) {
            throw std::logic_error("SXM instruction source stream is not available");
        }

        TileWordVector vector{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            vector[lane] = *slot[lane];
        }
        return vector;
    }

    static void consume_vector(std::array<TileVectorSlot, hw::kStreams>& streams, SxmStreamId id)
    {
        clear_vector(streams[id.stream]);
    }

    static void write_vector(
        std::array<TileVectorSlot, hw::kStreams>& streams,
        SxmStreamId id,
        const TileWordVector& vector)
    {
        auto& slot = streams[id.stream];
        if (vector_occupied(slot)) {
            throw std::logic_error("SXM instruction destination stream is occupied");
        }
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            slot[lane] = vector[lane];
        }
    }

    static void execute(
        const std::array<TileVectorSlot, hw::kStreams>& input_streams,
        std::array<TileVectorSlot, hw::kStreams>& output_streams,
        const SxmInstruction& instruction)
    {
        switch (instruction.opcode) {
        case SxmOpcode::Distribute:
            execute_distribute(input_streams, output_streams, instruction);
            break;
        case SxmOpcode::Transpose:
            execute_transpose(input_streams, output_streams, instruction);
            break;
        case SxmOpcode::ShiftSelect:
            execute_shift_select(input_streams, output_streams, instruction);
            break;
        case SxmOpcode::Permute:
            execute_permute(input_streams, output_streams, instruction);
            break;
        }
    }

    static void execute_distribute(
        const std::array<TileVectorSlot, hw::kStreams>& input_streams,
        std::array<TileVectorSlot, hw::kStreams>& output_streams,
        const SxmInstruction& instruction)
    {
        const auto input = read_vector(input_streams, instruction.src_streams[0]);
        consume_vector(output_streams, instruction.src_streams[0]);
        const auto output = Distribute16::apply(input, instruction.lane_map, Word {});
        write_vector(output_streams, instruction.dst_streams[0], output);
    }

    static void execute_transpose(
        const std::array<TileVectorSlot, hw::kStreams>& input_streams,
        std::array<TileVectorSlot, hw::kStreams>& output_streams,
        const SxmInstruction& instruction)
    {
        Matrix16 input{};
        for (std::size_t stream = 0; stream < hw::kLanesPerTile; ++stream) {
            input[stream] = read_vector(input_streams, instruction.src_streams[stream]);
        }
        for (const auto src : instruction.src_streams) {
            consume_vector(output_streams, src);
        }

        const auto output = Transpose16x16::apply(input);
        for (std::size_t stream = 0; stream < hw::kLanesPerTile; ++stream) {
            write_vector(output_streams, instruction.dst_streams[stream], output[stream]);
        }
    }

    static void execute_shift_select(
        const std::array<TileVectorSlot, hw::kStreams>& input_streams,
        std::array<TileVectorSlot, hw::kStreams>& output_streams,
        const SxmInstruction& instruction)
    {
        StreamWordVector input{};
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            input[row] = read_vector(input_streams, instruction.src_streams[row]);
        }
        for (const auto src : instruction.src_streams) {
            consume_vector(output_streams, src);
        }

        const auto output = ShiftSelect::apply(input, instruction.shift_source, instruction.shift_distance, Word {});
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            write_vector(output_streams, instruction.dst_streams[row], output[row]);
        }
    }

    static void execute_permute(
        const std::array<TileVectorSlot, hw::kStreams>& input_streams,
        std::array<TileVectorSlot, hw::kStreams>& output_streams,
        const SxmInstruction& instruction)
    {
        StreamWordVector input{};
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            input[row] = read_vector(input_streams, instruction.src_streams[row]);
        }
        for (const auto src : instruction.src_streams) {
            consume_vector(output_streams, src);
        }

        const auto output = Permute320::apply(input, instruction.permute_map);
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            write_vector(output_streams, instruction.dst_streams[row], output[row]);
        }
    }

    std::array<TileVectorSlot, hw::kStreams> streams_{};
    std::vector<SxmInstruction> issued_{};
    std::size_t issued_stream_ops_{0};
    std::size_t cycle_{0};
};

} // namespace ftlpu
