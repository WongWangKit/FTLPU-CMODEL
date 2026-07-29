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
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
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
    static constexpr std::size_t kTransposeBytePlanes = 2;
    using UnitGroup = SxmUnitGroup<std::uint8_t>;

    explicit SxmSlice(SxmStreamPortMap ports)
        : ports_(std::move(ports))
    {
    }

    void reset()
    {
        units_.reset();
        for (auto& instruction : transpose_instruction_rows_) instruction.reset();
        permute_instruction_.reset();
        transpose_bank_ = TransposeBank {};
        delayed_output_segments_.clear();
    }

    std::size_t cycle() const noexcept
    {
        return units_.cycle();
    }

    bool can_issue(const SxmInstruction& instruction) const
    {
        if (is_streaming_transpose(instruction)) {
            return !transpose_instruction_rows_[0].has_value();
        }
        if (is_block_permute(instruction) || is_global_permute(instruction)) {
            return !permute_instruction_.has_value();
        }
        return units_.can_issue(instruction);
    }

    void issue(SxmInstruction instruction)
    {
        if (is_streaming_transpose(instruction)) {
            if (transpose_instruction_rows_[0].has_value()) {
                throw std::logic_error("SXM transpose south row issued twice in one cycle");
            }
            check_uniform_direction(instruction.src_streams, "Transpose source");
            check_uniform_direction(instruction.dst_streams, "Transpose destination");
            transpose_instruction_rows_[0] = std::move(instruction);
            return;
        }
        if (is_block_permute(instruction) || is_global_permute(instruction)) {
            if (permute_instruction_.has_value()) {
                throw std::logic_error("SXM permute queue issued twice in one cycle");
            }
            check_uniform_direction(instruction.src_streams, "Permute source");
            check_uniform_direction(instruction.dst_streams, "Permute destination");
            permute_instruction_ = std::move(instruction);
            return;
        }
        units_.issue(std::move(instruction));
    }

    const SxmStreamPortMap& ports() const noexcept
    {
        return ports_;
    }

    const std::optional<SxmInstruction>& transpose_instruction_at(std::size_t tile) const
    {
        if (tile >= hw::kTileRows) {
            throw std::out_of_range("SXM transpose instruction tile is outside the slice");
        }
        return transpose_instruction_rows_[tile];
    }

    void evaluate(StreamRegisterFabric& fabric)
    {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) {
            throw std::logic_error("SxmSlice::evaluate requires an open SR cycle");
        }

        cycle_events_.clear();
        flush_delayed_outputs(fabric);

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto inputs = read_and_consume_inputs(fabric, tile);
            const auto result = units_.evaluate(inputs);
            write_outputs(fabric, tile, result);
        }

        if (permute_instruction_.has_value()) {
            if (is_block_permute(*permute_instruction_)) {
                auto processed_tiles = std::array<bool, hw::kTileRows> {};
                execute_block_permute(fabric, *permute_instruction_, processed_tiles);
                capture_transpose_inputs(fabric);
            } else {
                capture_transpose_inputs(fabric);
                execute_global_permute(fabric, *permute_instruction_);
            }
        } else {
            capture_transpose_inputs(fabric);
        }

        permute_instruction_.reset();
        units_.complete_cycle();
        advance_transpose_instructions();
    }

    void log_cycle(std::ostream& os) const
    {
        os << "sxm cycle " << (units_.cycle() == 0 ? 0 : units_.cycle() - 1) << '\n';
        if (cycle_events_.empty()) {
            os << "  idle\n";
            return;
        }
        for (const auto& event : cycle_events_) os << "  " << event << '\n';
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

    struct TransposeBank {
        using Block = std::array<
            std::array<StreamPayloadVector320, hw::kLanesPerTile>,
            kTransposeBytePlanes>;

        Block block{};
        std::array<SxmInstruction::StreamList, hw::kTileRows> dst_streams{};
        std::array<std::size_t, hw::kTileRows> input_stream_count{};
        std::array<std::size_t, hw::kTileRows> capture_count{};
        std::array<bool, hw::kTileRows> tile_ready{};
        std::array<std::size_t, hw::kTileRows> ready_cycle{};
        std::array<std::size_t, hw::kTileRows> emit_row{};
        std::array<std::size_t, hw::kTileRows> permute_destination{};
        std::array<std::array<std::size_t, hw::kLanesPerTile>, hw::kTileRows>
            permute_source_lanes{};
        std::array<SxmInstruction::StreamList, hw::kTileRows> permute_outputs{};
        std::array<bool, hw::kTileRows> permute_active{};
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
            && ((instruction.src_streams.size() == kTransposeBytePlanes
                    && instruction.dst_streams.size() == kTransposeBytePlanes)
                || (instruction.src_streams.size()
                        == kTransposeBytePlanes * hw::kLanesPerTile
                    && instruction.dst_streams.size()
                        == kTransposeBytePlanes * hw::kLanesPerTile));
    }

    static bool is_block_permute(const SxmInstruction& instruction)
    {
        return instruction.opcode == SxmOpcode::Permute
            && ((instruction.src_streams.size() == kTransposeBytePlanes
                    && instruction.dst_streams.size() == kTransposeBytePlanes)
                || (instruction.src_streams.size()
                        == kTransposeBytePlanes * hw::kLanesPerTile
                    && instruction.dst_streams.size()
                        == kTransposeBytePlanes * hw::kLanesPerTile));
    }

    static bool is_global_permute(const SxmInstruction& instruction)
    {
        return instruction.opcode == SxmOpcode::Permute
            && instruction.src_streams.size() == 1
            && instruction.dst_streams.size() == 1;
    }

    static StreamDirection uniform_direction(
        const SxmInstruction::StreamList& streams,
        const char* role)
    {
        if (streams.empty()) {
            throw std::invalid_argument(std::string("SXM ") + role + " streams are empty");
        }
        const auto direction = StreamId::from_packed(streams.front().stream).direction();
        for (const auto stream : streams) {
            if (StreamId::from_packed(stream.stream).direction() != direction) {
                throw std::invalid_argument(
                    std::string("SXM ") + role + " streams must share one direction");
            }
        }
        return direction;
    }

    static void check_uniform_direction(
        const SxmInstruction::StreamList& streams,
        const char* role)
    {
        static_cast<void>(uniform_direction(streams, role));
    }

    Vector read_and_consume_vector(StreamRegisterFabric& fabric, SxmStreamId stream) const
    {
        const auto physical = StreamId::from_packed(stream.stream);
        const auto input_column = ports_.input_column(physical.direction());
        Vector vector{};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!fabric.segment_valid(input_column, tile, physical)) {
                throw std::logic_error("SXM streaming instruction source vector is incomplete");
            }
            vector[tile] = fabric.segment(input_column, tile, physical);
        }
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            fabric.consume_segment(
                input_column,
                tile,
                physical,
                "SXM streaming instruction");
        }
        return vector;
    }

    void capture_transpose_inputs(StreamRegisterFabric& fabric)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto& instruction = transpose_instruction_rows_[tile];
            if (instruction.has_value()) capture_transpose_input(fabric, tile, *instruction);
        }
    }

    void capture_transpose_input(
        StreamRegisterFabric& fabric,
        std::size_t tile,
        const SxmInstruction& instruction)
    {
        const auto input_direction = uniform_direction(instruction.src_streams, "Transpose source");
        const auto input_column = ports_.input_column(input_direction);
        const auto parallel_input = instruction.src_streams.size()
            == kTransposeBytePlanes * hw::kLanesPerTile;
        auto all_valid = true;
        for (const auto source_stream : instruction.src_streams) {
            all_valid = all_valid && fabric.segment_valid(
                input_column,
                tile,
                StreamId::from_packed(source_stream.stream));
        }
        if (!all_valid) return;

            auto& bank = transpose_bank_;
            if (bank.tile_ready[tile]) {
                throw std::logic_error("SXM transpose buffer is full");
            }
            if (bank.capture_count[tile] == 0) {
                bank.dst_streams[tile] = instruction.dst_streams;
                bank.input_stream_count[tile] = instruction.src_streams.size();
            } else if (bank.input_stream_count[tile] != instruction.src_streams.size()
                || bank.dst_streams[tile] != instruction.dst_streams) {
                throw std::logic_error("SXM transpose mode changed within a block");
            }

            if (parallel_input) {
                for (std::size_t row = 0; row < hw::kLanesPerTile; ++row) {
                    for (std::size_t plane = 0; plane < kTransposeBytePlanes; ++plane) {
                        const auto source = StreamId::from_packed(
                            instruction.src_streams[row * kTransposeBytePlanes + plane].stream);
                        const auto input = fabric.segment(input_column, tile, source);
                        fabric.consume_segment(
                            input_column, tile, source, "SXM parallel FP16 Transpose");
                        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                            bank.block[plane][lane][tile][row] = input[lane].data;
                        }
                    }
                }
                bank.capture_count[tile] = hw::kLanesPerTile;
            } else {
                const auto row = bank.capture_count[tile];
                if (row >= hw::kLanesPerTile) {
                    throw std::logic_error("SXM serial Transpose captured too many rows");
                }
                for (std::size_t plane = 0; plane < kTransposeBytePlanes; ++plane) {
                    const auto source = StreamId::from_packed(instruction.src_streams[plane].stream);
                    const auto input = fabric.segment(input_column, tile, source);
                    fabric.consume_segment(
                        input_column, tile, source, "SXM serial FP16 Transpose");
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        bank.block[plane][lane][tile][row] = input[lane].data;
                    }
                }
                ++bank.capture_count[tile];
            }

        if (bank.capture_count[tile] != hw::kLanesPerTile) return;
        bank.tile_ready[tile] = true;
        bank.ready_cycle[tile] = units_.cycle();
        std::ostringstream event;
        event << "transpose tile=" << tile
              << " input_streams=" << instruction.src_streams.size()
              << " block=ready";
        cycle_events_.push_back(event.str());
    }

    void advance_transpose_instructions()
    {
        for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile) {
            transpose_instruction_rows_[tile] = std::move(transpose_instruction_rows_[tile - 1]);
        }
        transpose_instruction_rows_[0].reset();
    }

    void stage_block_segment(
        StreamRegisterFabric& fabric,
        std::size_t tile,
        const StreamPayloadSegment16& segment,
        SxmStreamId output_stream,
        const char* producer)
    {
        const auto destination = StreamId::from_packed(output_stream.stream);
        fabric.stage_payload_segment(
            ports_.output_column(destination.direction()),
            tile,
            destination,
            segment,
            0,
            producer);
    }

    static std::size_t block_permute_source_tile(
        std::size_t destination_tile,
        const SxmInstruction::PermuteMap& map)
    {
        const auto destination_base = destination_tile * hw::kLanesPerTile;
        const auto source_tile = map[destination_base] / hw::kLanesPerTile;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            if (map[destination_base + lane] / hw::kLanesPerTile != source_tile) {
                throw std::logic_error(
                    "SXM block Permute must move complete superlane blocks");
            }
        }
        return source_tile;
    }

    static StreamPayloadSegment16 permute_block_segment(
        const TransposeBank& bank,
        std::size_t plane,
        std::size_t row,
        std::size_t source_tile,
        std::size_t destination_tile,
        const SxmInstruction::PermuteMap& map)
    {
        auto output = StreamPayloadSegment16 {};
        const auto destination_base = destination_tile * hw::kLanesPerTile;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto source = map[destination_base + lane];
            output[lane] = bank.block[plane][row][source_tile]
                [source % hw::kLanesPerTile];
        }
        return output;
    }

    void execute_block_permute(
        StreamRegisterFabric& fabric,
        const SxmInstruction& instruction,
        std::array<bool, hw::kTileRows>& processed_tiles)
    {
        Permute320::validate_bijection(instruction.permute_map);
        const auto parallel_output = instruction.dst_streams.size()
            == kTransposeBytePlanes * hw::kLanesPerTile;

        for (std::size_t destination_tile = 0;
             destination_tile < hw::kTileRows;
            ++destination_tile) {
            if (processed_tiles[destination_tile]) continue;
            const auto source_tile = block_permute_source_tile(
                destination_tile, instruction.permute_map);
            auto& bank = transpose_bank_;
            // A transpose result must spend one full cycle in its output
            // register before Permute may consume it.
            if (!bank.tile_ready[source_tile]
                || bank.ready_cycle[source_tile] >= units_.cycle()) continue;
            if (bank.dst_streams[source_tile] != instruction.src_streams) {
                throw std::logic_error("SXM Permute source does not match the ready Transpose block");
            }

            if (parallel_output) {
                for (std::size_t row = 0; row < hw::kLanesPerTile; ++row) {
                    for (std::size_t plane = 0; plane < kTransposeBytePlanes; ++plane) {
                        stage_block_segment(
                            fabric,
                            destination_tile,
                            permute_block_segment(
                                bank,
                                plane,
                                row,
                                source_tile,
                                destination_tile,
                                instruction.permute_map),
                            instruction.dst_streams[row * kTransposeBytePlanes + plane],
                            "SXM parallel block output");
                    }
                }
                bank.tile_ready[source_tile] = false;
                bank.capture_count[source_tile] = 0;
                processed_tiles[destination_tile] = true;
                std::ostringstream event;
                event << "permute source_tile=" << source_tile
                      << " destination_tile=" << destination_tile
                      << " output_streams=" << instruction.dst_streams.size();
                cycle_events_.push_back(event.str());
                continue;
            }

            const auto destination_base = destination_tile * hw::kLanesPerTile;
            if (!bank.permute_active[source_tile]) {
                bank.permute_active[source_tile] = true;
                bank.permute_destination[source_tile] = destination_tile;
                bank.permute_outputs[source_tile] = instruction.dst_streams;
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    bank.permute_source_lanes[source_tile][lane] =
                        instruction.permute_map[destination_base + lane];
                }
            } else {
                auto same_map = bank.permute_destination[source_tile] == destination_tile
                    && bank.permute_outputs[source_tile] == instruction.dst_streams;
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    same_map = same_map
                        && bank.permute_source_lanes[source_tile][lane]
                            == instruction.permute_map[destination_base + lane];
                }
                if (!same_map) {
                    throw std::logic_error(
                        "SXM Permute route changed while serializing a source block");
                }
            }

            const auto row = bank.emit_row[source_tile];
            for (std::size_t plane = 0; plane < kTransposeBytePlanes; ++plane) {
                stage_block_segment(
                    fabric,
                    destination_tile,
                    permute_block_segment(
                        bank,
                        plane,
                        row,
                        source_tile,
                        destination_tile,
                        instruction.permute_map),
                    instruction.dst_streams[plane],
                    "SXM serial block output");
            }
            ++bank.emit_row[source_tile];
            processed_tiles[destination_tile] = true;
            if (bank.emit_row[source_tile] == hw::kLanesPerTile) {
                bank.emit_row[source_tile] = 0;
                bank.permute_active[source_tile] = false;
                bank.permute_outputs[source_tile].clear();
                bank.tile_ready[source_tile] = false;
                bank.capture_count[source_tile] = 0;
            }
        }
    }

    void execute_global_permute(
        StreamRegisterFabric& fabric,
        const SxmInstruction& instruction)
    {
        auto inputs = std::vector<Vector> {};
        inputs.reserve(instruction.src_streams.size());
        for (const auto source : instruction.src_streams) {
            inputs.push_back(read_and_consume_vector(fabric, source));
        }
        for (std::size_t output_index = 0;
             output_index < instruction.dst_streams.size();
             ++output_index) {
            const auto plane = output_index % inputs.size();
            schedule_skewed_output(
                fabric,
                PendingVector {
                    instruction.dst_streams[output_index],
                    Permute320::apply(inputs[plane], instruction.permute_map),
                });
        }
    }

    void schedule_skewed_output(StreamRegisterFabric& fabric, const PendingVector& output)
    {
        const auto physical = StreamId::from_packed(output.stream.stream);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (tile == 0) {
                fabric.stage_segment(
                    ports_.output_column(physical.direction()),
                    tile,
                    physical,
                    output.vector[tile],
                    "SXM output");
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
                ports_.output_column(StreamId::from_packed(output.stream.stream).direction()),
                output.tile,
                StreamId::from_packed(output.stream.stream),
                output.segment,
                "SXM skewed output");
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
    std::array<std::optional<SxmInstruction>, hw::kTileRows> transpose_instruction_rows_{};
    std::optional<SxmInstruction> permute_instruction_{};
    TransposeBank transpose_bank_{};
    std::vector<std::string> cycle_events_{};
    std::vector<DelayedSegment> delayed_output_segments_{};
};

} // namespace ftlpu
