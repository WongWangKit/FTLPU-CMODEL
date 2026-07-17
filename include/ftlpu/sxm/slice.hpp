#pragma once

#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/sxm/distributor.hpp"
#include "ftlpu/sxm/permute.hpp"
#include "ftlpu/sxm/shift.hpp"
#include "ftlpu/sxm/transpose.hpp"
#include "ftlpu/sxm/unit_group.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

class SxmStreamPortMap {
public:
    static SxmStreamPortMap BetweenColumns(
        std::size_t east_input,
        std::size_t east_output,
        std::size_t west_input,
        std::size_t west_output)
    {
        return SxmStreamPortMap(east_input, east_output, west_input, west_output);
    }

    static SxmStreamPortMap SameDirection(std::size_t input, std::size_t output)
    {
        return SxmStreamPortMap(input, output, input, output);
    }

    std::size_t input_column(StreamDirection direction) const noexcept
    {
        return direction == StreamDirection::East ? east_input_ : west_input_;
    }

    std::size_t output_column(StreamDirection direction) const noexcept
    {
        return direction == StreamDirection::East ? east_output_ : west_output_;
    }

    void validate_for(const StreamRegisterFabric& fabric) const
    {
        if (east_input_ >= fabric.column_count()
            || east_output_ >= fabric.column_count()
            || west_input_ >= fabric.column_count()
            || west_output_ >= fabric.column_count()) {
            throw std::out_of_range("SXM port maps outside stream-register fabric");
        }
    }

private:
    SxmStreamPortMap(
        std::size_t east_input,
        std::size_t east_output,
        std::size_t west_input,
        std::size_t west_output)
        : east_input_(east_input)
        , east_output_(east_output)
        , west_input_(west_input)
        , west_output_(west_output)
    {
    }

    std::size_t east_input_{0};
    std::size_t east_output_{0};
    std::size_t west_input_{0};
    std::size_t west_output_{0};
};

// SR-facing SXM functional slice.  It owns instruction issue state only; the
// StreamRegisterFabric remains the sole owner of all live stream data.
class SxmSlice {
public:
    using UnitGroup = SxmUnitGroup<std::uint8_t>;

    explicit SxmSlice(SxmStreamPortMap ports)
        : ports_(std::move(ports))
    {
    }

    void reset()
    {
        units_.reset();
        transpose_instruction_.reset();
        permute_instruction_.reset();
        transpose_capture_count_.fill(0);
        transpose_block_count_.fill(0);
        for (auto& output : transpose_output_) {
            output.clear();
        }
        for (auto& source_tile : transpose_history_valid_) {
            for (auto& block : source_tile) {
                block.fill(false);
            }
        }
        transpose_history_blocks_ = 0;
        transpose_history_query_ = 0;
        transpose_history_weight_column_ = 0;
        delayed_output_segments_.clear();
    }

    std::size_t cycle() const noexcept
    {
        return units_.cycle();
    }

    bool can_issue(const SxmInstruction& instruction) const
    {
        if (is_streaming_transpose(instruction)) {
            return !transpose_instruction_.has_value();
        }
        if (is_global_permute(instruction)) {
            return !permute_instruction_.has_value();
        }
        return units_.can_issue(instruction);
    }

    void issue(SxmInstruction instruction)
    {
        if (is_streaming_transpose(instruction)) {
            if (transpose_instruction_.has_value()) {
                throw std::logic_error("SXM transpose queue issued twice in one cycle");
            }
            check_east_stream(instruction.src_streams[0]);
            for (const auto stream : instruction.dst_streams) {
                check_east_stream(stream);
            }
            transpose_instruction_ = std::move(instruction);
            return;
        }
        if (is_global_permute(instruction)) {
            if (permute_instruction_.has_value()) {
                throw std::logic_error("SXM permute queue issued twice in one cycle");
            }
            check_east_stream(instruction.src_streams[0]);
            check_east_stream(instruction.dst_streams[0]);
            permute_instruction_ = std::move(instruction);
            return;
        }
        units_.issue(std::move(instruction));
    }

    const SxmStreamPortMap& ports() const noexcept
    {
        return ports_;
    }

    void evaluate(StreamRegisterFabric& fabric)
    {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) {
            throw std::logic_error("SxmSlice::evaluate requires an open SR cycle");
        }

        flush_delayed_outputs(fabric);

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto inputs = read_and_consume_inputs(fabric, tile);
            const auto result = units_.evaluate(inputs);
            write_outputs(fabric, tile, result);
        }

        if (transpose_instruction_.has_value()) {
            capture_transpose_input(fabric, *transpose_instruction_);
        }

        const auto emitted_transpose = emit_transpose_outputs(fabric);
        if (!emitted_transpose && permute_instruction_.has_value()) {
            if (transpose_history_blocks_ != 0) {
                if (permute_instruction_->dst_streams.size() == hw::kMxmLoadStreamsPerCycle) {
                    emit_transpose_weight_streams(fabric, *permute_instruction_);
                } else {
                    auto output = assemble_transpose_history(*permute_instruction_);
                    schedule_skewed_output(fabric, output);
                }
            } else {
                const auto input = read_and_consume_vector(fabric, permute_instruction_->src_streams[0]);
                schedule_skewed_output(
                    fabric,
                    PendingVector {
                        permute_instruction_->dst_streams[0],
                        Permute320::apply(input, permute_instruction_->permute_map),
                    });
            }
        }

        transpose_instruction_.reset();
        permute_instruction_.reset();
        units_.complete_cycle();
    }

    template <typename T>
    using TileVector = std::array<T, hw::kLanesPerTile>;

    template <typename T>
    using StreamVector = std::array<TileVector<T>, hw::kTileRows>;

    template <typename T>
    using Matrix16 = std::array<TileVector<T>, hw::kLanesPerTile>;

    template <typename T>
    static TileVector<T> distribute(const TileVector<T>& input, const Distribute16::Map& map, T zero = T{})
    {
        return Distribute16::apply(input, map, zero);
    }

    template <typename T>
    static Matrix16<T> transpose(const Matrix16<T>& input)
    {
        return Transpose16x16::apply(input);
    }

    template <typename T>
    static StreamVector<T> shift_select(
        const StreamVector<T>& input,
        SxmShiftSource source,
        std::size_t distance = 1,
        T zero = T{})
    {
        return ShiftSelect::apply(input, source, distance, zero);
    }

    template <typename T>
    static StreamVector<T> permute(const StreamVector<T>& input, const Permute320::Map& map)
    {
        return Permute320::apply(input, map);
    }

private:
    using StreamState = UnitGroup::StreamState;
    using Evaluation = UnitGroup::Evaluation;
    using Vector = StreamVector320;

    struct PendingVector {
        SxmStreamId stream{};
        Vector vector{};
    };

    struct PendingSegment {
        SxmStreamId stream{};
        StreamSegment16 segment{};
        std::size_t block{0};
        std::size_t row{0};
    };

    struct DelayedSegment {
        std::size_t ready_cycle{0};
        std::size_t tile{0};
        SxmStreamId stream{};
        StreamSegment16 segment{};
    };

    static bool is_streaming_transpose(const SxmInstruction& instruction)
    {
        return instruction.opcode == SxmOpcode::Transpose
            && instruction.src_streams.size() == 1
            && instruction.dst_streams.size() == 1;
    }

    static bool is_global_permute(const SxmInstruction& instruction)
    {
        return instruction.opcode == SxmOpcode::Permute
            && instruction.src_streams.size() == 1
            && (instruction.dst_streams.size() == 1
                || instruction.dst_streams.size() == hw::kMxmLoadStreamsPerCycle);
    }

    static void check_east_stream(SxmStreamId stream)
    {
        const auto physical = StreamId::from_packed(stream.stream);
        if (physical.direction() != StreamDirection::East) {
            throw std::invalid_argument("system SXM accepts eastward streams only");
        }
    }

    Vector read_and_consume_vector(StreamRegisterFabric& fabric, SxmStreamId stream) const
    {
        const auto physical = StreamId::from_packed(stream.stream);
        Vector vector{};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!fabric.segment_valid(ports_.input_column(StreamDirection::East), tile, physical)) {
                throw std::logic_error("SXM streaming instruction source vector is incomplete");
            }
            vector[tile] = fabric.segment(ports_.input_column(StreamDirection::East), tile, physical);
        }
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            fabric.consume_segment(
                ports_.input_column(StreamDirection::East),
                tile,
                physical,
                "SXM streaming instruction");
        }
        return vector;
    }

    void capture_transpose_input(StreamRegisterFabric& fabric, const SxmInstruction& instruction)
    {
        const auto source = StreamId::from_packed(instruction.src_streams[0].stream);
        const auto input_column = ports_.input_column(StreamDirection::East);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!fabric.segment_valid(input_column, tile, source)) {
                continue;
            }
            const auto input = fabric.segment(input_column, tile, source);
            fabric.consume_segment(input_column, tile, source, "SXM streaming Transpose");

            const auto phase = transpose_capture_count_[tile];
            transpose_capture_[tile][phase] = input;
            ++transpose_capture_count_[tile];
            if (transpose_capture_count_[tile] != hw::kLanesPerTile) {
                continue;
            }

            for (std::size_t row = 0; row < hw::kLanesPerTile; ++row) {
                auto output = PendingSegment {
                    instruction.dst_streams[0],
                    {},
                    transpose_block_count_[tile],
                    row,
                };
                for (std::size_t column = 0; column < hw::kLanesPerTile; ++column) {
                    output.segment[column] = transpose_capture_[tile][column][row];
                }
                transpose_output_[tile].push_back(std::move(output));
            }
            transpose_capture_count_[tile] = 0;
            ++transpose_block_count_[tile];
        }
    }

    StreamSegment16 permute_local_segment(
        std::size_t tile,
        const StreamSegment16& input,
        const SxmInstruction::PermuteMap& map) const
    {
        auto output = StreamSegment16 {};
        const auto tile_base = tile * hw::kLanesPerTile;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto source = map[tile_base + lane];
            if (source / hw::kLanesPerTile != tile) {
                throw std::logic_error(
                    "streaming SXM Permute cannot move a wavefront segment across tiles");
            }
            output[lane] = input[source % hw::kLanesPerTile];
        }
        return output;
    }

    bool emit_transpose_outputs(StreamRegisterFabric& fabric)
    {
        auto emitted = false;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (transpose_output_[tile].empty()) {
                continue;
            }
            auto output = std::move(transpose_output_[tile].front());
            transpose_output_[tile].pop_front();
            if (permute_instruction_.has_value()) {
                if (permute_instruction_->src_streams[0] != output.stream) {
                    throw std::logic_error("SXM Permute source does not match pending Transpose output stream");
                }
                if (output.block >= hw::kTileRows) {
                    throw std::logic_error("SXM Transpose history exceeds the configured plane");
                }
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    transpose_history_[tile][output.block][output.row][lane] =
                        output.segment[lane].data;
                }
                transpose_history_valid_[tile][output.block][output.row] = true;
                transpose_history_blocks_ = std::max(
                    transpose_history_blocks_,
                    output.block + 1);
                emitted = true;
                continue;
            }
            fabric.stage_segment(
                ports_.output_column(StreamDirection::East),
                tile,
                StreamId::from_packed(output.stream.stream),
                output.segment,
                "SXM transposed east output");
            emitted = true;
        }
        return emitted;
    }

    PendingVector assemble_transpose_history(const SxmInstruction& instruction)
    {
        if (transpose_history_query_ >= hw::kMxmRows) {
            throw std::logic_error("SXM Permute requested more rows than the physical vector width");
        }
        const auto source_tile = transpose_history_query_ / hw::kLanesPerTile;
        const auto row = transpose_history_query_ % hw::kLanesPerTile;
        auto assembled = Vector {};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                assembled[tile][lane] = StreamCell::Valid(
                    0,
                    lane + 1 == hw::kLanesPerTile);
            }
        }
        for (std::size_t block = 0; block < transpose_history_blocks_; ++block) {
            if (!transpose_history_valid_[source_tile][block][row]) {
                throw std::logic_error("SXM Permute requested an incomplete transposed row");
            }
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                assembled[block][lane] = StreamCell::Valid(
                    transpose_history_[source_tile][block][row][lane],
                    lane + 1 == hw::kLanesPerTile);
            }
        }
        ++transpose_history_query_;
        auto output = PendingVector {
            instruction.dst_streams[0],
            Permute320::apply(assembled, instruction.permute_map),
        };
        if (transpose_history_query_ == transpose_history_blocks_ * hw::kLanesPerTile) {
            clear_transpose_history();
        }
        return output;
    }

    void emit_transpose_weight_streams(
        StreamRegisterFabric& fabric,
        const SxmInstruction& instruction)
    {
        if (transpose_history_weight_column_ >= hw::kTileRows) {
            throw std::logic_error("SXM weight Permute requested too many column blocks");
        }
        const auto source_tile = hw::kTileRows - 1 - transpose_history_weight_column_;
        for (std::size_t stream = 0; stream < hw::kMxmLoadStreamsPerCycle; ++stream) {
            auto assembled = Vector {};
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    assembled[tile][lane] = StreamCell::Valid(
                        0,
                        lane + 1 == hw::kLanesPerTile);
                }
            }
            for (std::size_t block = 0; block < transpose_history_blocks_; ++block) {
                if (!transpose_history_valid_[source_tile][block][stream]) {
                    throw std::logic_error("SXM weight Permute requested an incomplete column");
                }
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    assembled[block][lane] = StreamCell::Valid(
                        transpose_history_[source_tile][block][stream][lane],
                        lane + 1 == hw::kLanesPerTile);
                }
            }
            schedule_skewed_output(
                fabric,
                PendingVector {
                    instruction.dst_streams[stream],
                    Permute320::apply(assembled, instruction.permute_map),
                });
        }
        ++transpose_history_weight_column_;
        if (transpose_history_weight_column_ == hw::kTileRows) {
            clear_transpose_history();
        }
    }

    void clear_transpose_history()
    {
        transpose_capture_count_.fill(0);
        transpose_block_count_.fill(0);
        for (auto& output : transpose_output_) {
            output.clear();
        }
        for (auto& source_tile : transpose_history_valid_) {
            for (auto& block : source_tile) {
                block.fill(false);
            }
        }
        transpose_history_blocks_ = 0;
        transpose_history_query_ = 0;
        transpose_history_weight_column_ = 0;
    }

    void schedule_skewed_output(StreamRegisterFabric& fabric, const PendingVector& output)
    {
        const auto physical = StreamId::from_packed(output.stream.stream);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (tile == 0) {
                fabric.stage_segment(
                    ports_.output_column(StreamDirection::East),
                    tile,
                    physical,
                    output.vector[tile],
                    "SXM east output");
                continue;
            }
            delayed_output_segments_.push_back(DelayedSegment {
                units_.cycle() + tile,
                tile,
                output.stream,
                output.vector[tile],
            });
        }
    }

    void flush_delayed_outputs(StreamRegisterFabric& fabric)
    {
        auto remaining = std::vector<DelayedSegment> {};
        remaining.reserve(delayed_output_segments_.size());
        for (auto& output : delayed_output_segments_) {
            if (output.ready_cycle != units_.cycle()) {
                remaining.push_back(std::move(output));
                continue;
            }
            fabric.stage_segment(
                ports_.output_column(StreamDirection::East),
                output.tile,
                StreamId::from_packed(output.stream.stream),
                output.segment,
                "SXM skewed east output");
        }
        delayed_output_segments_ = std::move(remaining);
    }

    static StreamId physical_stream(SxmStreamId stream)
    {
        return StreamId::from_packed(stream.stream);
    }

    StreamState read_and_consume_inputs(StreamRegisterFabric& fabric, std::size_t tile) const
    {
        StreamState inputs{};
        std::array<bool, hw::kStreams> required{};
        for (const auto& instruction : units_.issued_instructions()) {
            for (const auto stream : instruction.src_streams) {
                required[stream.stream] = true;
            }
        }

        // Validate every operand before consuming any of them, so a missing
        // operand cannot leave a partially consumed fabric cycle.
        for (std::size_t packed = 0; packed < required.size(); ++packed) {
            if (!required[packed]) {
                continue;
            }
            const auto id = StreamId::from_packed(packed);
            StreamInputPort input(
                fabric,
                ports_.input_column(id.direction()),
                id.direction(),
                "SXM");
            if (!input.segment_valid(tile, id.index())) {
                throw std::logic_error("SXM source stream segment is not available");
            }
        }

        for (std::size_t packed = 0; packed < required.size(); ++packed) {
            if (!required[packed]) {
                continue;
            }
            const auto id = StreamId::from_packed(packed);
            StreamInputPort input(
                fabric,
                ports_.input_column(id.direction()),
                id.direction(),
                "SXM");
            const auto segment = input.consume_segment(tile, id.index());
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                inputs[packed][lane] = UnitGroup::Word {
                    segment[lane].data,
                    segment[lane].last,
                };
            }
        }
        return inputs;
    }

    void write_outputs(
        StreamRegisterFabric& fabric,
        std::size_t tile,
        const Evaluation& result) const
    {
        for (std::size_t packed = 0; packed < result.produced.size(); ++packed) {
            if (!result.produced[packed]) {
                continue;
            }
            const auto id = StreamId::from_packed(packed);
            StreamSegment16 segment{};
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto& word = result.outputs[packed][lane];
                if (!word.has_value()) {
                    throw std::logic_error("SXM produced an incomplete stream segment");
                }
                segment[lane] = StreamCell::Valid(word->data, word->last);
            }
            StreamOutputPort output(
                fabric,
                ports_.output_column(id.direction()),
                id.direction(),
                "SXM");
            output.write_segment(tile, id.index(), segment);
        }
    }

    SxmStreamPortMap ports_;
    UnitGroup units_{};
    std::optional<SxmInstruction> transpose_instruction_{};
    std::optional<SxmInstruction> permute_instruction_{};
    std::array<std::array<StreamSegment16, hw::kLanesPerTile>, hw::kTileRows> transpose_capture_{};
    std::array<std::size_t, hw::kTileRows> transpose_capture_count_{};
    std::array<std::size_t, hw::kTileRows> transpose_block_count_{};
    std::array<std::deque<PendingSegment>, hw::kTileRows> transpose_output_{};
    std::array<
        std::array<std::array<StreamPayloadSegment16, hw::kLanesPerTile>, hw::kTileRows>,
        hw::kTileRows>
        transpose_history_{};
    std::array<
        std::array<std::array<bool, hw::kLanesPerTile>, hw::kTileRows>,
        hw::kTileRows>
        transpose_history_valid_{};
    std::size_t transpose_history_blocks_{0};
    std::size_t transpose_history_query_{0};
    std::size_t transpose_history_weight_column_{0};
    std::vector<DelayedSegment> delayed_output_segments_{};
};

} // namespace ftlpu
