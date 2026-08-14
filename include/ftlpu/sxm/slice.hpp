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
    }

    std::size_t cycle() const noexcept
    {
        return units_.cycle();
    }

    bool can_issue(const SxmInstruction& instruction) const
    {
        if (instruction.opcode == SxmOpcode::Transpose) {
            return is_streaming_transpose(instruction)
                && !transpose_instruction_rows_[0].has_value();
        }
        if (instruction.opcode == SxmOpcode::Permute) {
            return is_block_permute(instruction)
                && !permute_instruction_.has_value();
        }
        return units_.can_issue(instruction);
    }

    void issue(SxmInstruction instruction)
    {
        if (instruction.opcode == SxmOpcode::Transpose) {
            if (!is_streaming_transpose(instruction)) {
                throw std::invalid_argument(
                    "SXM Transpose requires exactly 16 input and 16 output streams");
            }
            if (transpose_instruction_rows_[0].has_value()) {
                throw std::logic_error("SXM transpose south row issued twice in one cycle");
            }
            check_uniform_direction(instruction.src_streams, "Transpose source");
            check_uniform_direction(instruction.dst_streams, "Transpose destination");
            transpose_instruction_rows_[0] = std::move(instruction);
            return;
        }
        if (instruction.opcode == SxmOpcode::Permute) {
            if (!is_block_permute(instruction)) {
                throw std::invalid_argument(
                    "SXM Permute requires exactly 16 input and 16 output streams");
            }
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

    void set_trace_enabled(bool enabled) noexcept
    {
        trace_enabled_ = enabled;
    }

    void evaluate(StreamRegisterFabric& fabric)
    {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) {
            throw std::logic_error("SxmSlice::evaluate requires an open SR cycle");
        }

        cycle_events_.clear();

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto inputs = read_and_consume_inputs(fabric, tile);
            const auto result = units_.evaluate(inputs);
            write_outputs(fabric, tile, result);
        }

        if (permute_instruction_.has_value()) {
            auto processed_tiles = std::array<bool, hw::kTileRows> {};
            execute_block_permute(fabric, *permute_instruction_, processed_tiles);
            capture_transpose_inputs(fabric);
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
    struct TransposeBank {
        using Block = std::array<
            std::array<StreamPayloadSliceVector, hw::kLanesPerTile>,
            kTransposeBytePlanes>;

        Block block{};
        std::array<SxmInstruction::StreamList, hw::kTileRows> dst_streams{};
        std::array<bool, hw::kTileRows> tile_ready{};
        std::array<std::uint8_t, hw::kTileRows> input_row_mask{};
        std::array<std::size_t, hw::kTileRows> ready_cycle{};
    };

    static bool is_streaming_transpose(const SxmInstruction& instruction)
    {
        if (instruction.opcode != SxmOpcode::Transpose
            || instruction.dst_streams.size()
                != kTransposeBytePlanes * hw::kLanesPerTile)
            return false;
        if (instruction.input_row == SxmInstruction::kAllInputRows)
            return instruction.src_streams.size()
                == kTransposeBytePlanes * hw::kLanesPerTile;
        return instruction.input_row < hw::kLanesPerTile
            && instruction.src_streams.size() == kTransposeBytePlanes;
    }

    static bool is_block_permute(const SxmInstruction& instruction)
    {
        return instruction.opcode == SxmOpcode::Permute
            && instruction.src_streams.size()
                == kTransposeBytePlanes * hw::kLanesPerTile
            && instruction.dst_streams.size()
                == kTransposeBytePlanes * hw::kLanesPerTile;
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

    void capture_transpose_inputs(StreamRegisterFabric& fabric)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto& instruction = transpose_instruction_rows_[tile];
            if (instruction.has_value())
                capture_transpose_input(fabric, tile, *instruction);
        }
    }

    void capture_transpose_input(
        StreamRegisterFabric& fabric,
        std::size_t tile,
        const SxmInstruction& instruction)
    {
        const auto input_direction = uniform_direction(instruction.src_streams, "Transpose source");
        const auto input_column = ports_.input_column(input_direction);
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
            if (bank.input_row_mask[tile] == 0)
                bank.dst_streams[tile] = instruction.dst_streams;
            else if (bank.dst_streams[tile] != instruction.dst_streams)
                throw std::logic_error("SXM serial Transpose destination changed");

            const std::size_t first_row =
                instruction.input_row == SxmInstruction::kAllInputRows
                ? 0 : instruction.input_row;
            const std::size_t row_end =
                instruction.input_row == SxmInstruction::kAllInputRows
                ? hw::kLanesPerTile : first_row + 1;
            for (std::size_t row = first_row; row < row_end; ++row) {
                for (std::size_t plane = 0; plane < kTransposeBytePlanes; ++plane) {
                    const auto source = StreamId::from_packed(
                        instruction.src_streams[
                            instruction.input_row == SxmInstruction::kAllInputRows
                                ? row * kTransposeBytePlanes + plane : plane].stream);
                    const auto input = fabric.segment(input_column, tile, source);
                    fabric.consume_segment(
                        input_column, tile, source, "SXM parallel FP16 Transpose");
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        bank.block[plane][lane][tile][row] = input[lane].data;
                    }
                }
                bank.input_row_mask[tile] |= static_cast<std::uint8_t>(1u << row);
            }
        bank.tile_ready[tile] = bank.input_row_mask[tile]
            == static_cast<std::uint8_t>((1u << hw::kLanesPerTile) - 1u);
        if (bank.tile_ready[tile]) bank.ready_cycle[tile] = units_.cycle();
        if (trace_enabled_) {
            std::ostringstream event;
            event << "transpose tile=" << tile
                  << " input_streams=" << instruction.src_streams.size()
                  << " block=ready";
            cycle_events_.push_back(event.str());
        }
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
        const StreamPayloadTileSegment& segment,
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

    static StreamPayloadTileSegment permute_block_segment(
        const TransposeBank& bank,
        std::size_t plane,
        std::size_t row,
        std::size_t source_tile,
        std::size_t destination_tile,
        const SxmInstruction::PermuteMap& map)
    {
        auto output = StreamPayloadTileSegment {};
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
        if (instruction.output_row != SxmInstruction::kAllOutputRows
            && instruction.output_row >= hw::kLanesPerTile) {
            throw std::out_of_range("SXM Permute output row is outside the block");
        }
        if (instruction.output_tile != SxmInstruction::kAllOutputTiles
            && instruction.output_tile >= hw::kTileRows) {
            throw std::out_of_range("SXM Permute output tile is outside the block");
        }

        for (std::size_t destination_tile = 0;
             destination_tile < hw::kTileRows;
            ++destination_tile) {
            if (processed_tiles[destination_tile]) continue;
            if (instruction.output_tile != SxmInstruction::kAllOutputTiles
                && instruction.output_tile != destination_tile)
                continue;
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

            const std::size_t first_row =
                instruction.output_row == SxmInstruction::kAllOutputRows
                ? 0 : instruction.output_row;
            const std::size_t row_end =
                instruction.output_row == SxmInstruction::kAllOutputRows
                ? hw::kLanesPerTile : first_row + 1;
            for (std::size_t row = first_row; row < row_end; ++row) {
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
            if (instruction.output_row == SxmInstruction::kAllOutputRows
                || instruction.output_row + 1 == hw::kLanesPerTile) {
                bank.tile_ready[source_tile] = false;
                bank.input_row_mask[source_tile] = 0;
            }
            processed_tiles[destination_tile] = true;
            if (trace_enabled_) {
                std::ostringstream event;
                event << "permute source_tile=" << source_tile
                      << " destination_tile=" << destination_tile
                      << " output_streams=" << instruction.dst_streams.size();
                cycle_events_.push_back(event.str());
            }
        }
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
            StreamTileSegment segment{};
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
    bool trace_enabled_{true};
};

} // namespace ftlpu
