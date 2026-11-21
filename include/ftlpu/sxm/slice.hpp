#pragma once

#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/core/timing.hpp"
#include "ftlpu/sxm/distributor.hpp"
#include "ftlpu/sxm/permute.hpp"
#include "ftlpu/sxm/shift.hpp"
#include "ftlpu/sxm/transpose.hpp"
#include "ftlpu/sxm/unit_group.hpp"
#include "ftlpu/sxm/transpose_architecture.hpp"
#include "ftlpu/sxm/wide_transpose.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <memory>
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
class SxmSlice : public timing::ClockedModule {
public:
    static constexpr std::size_t kTransposeBytePlanes = 2;
    using UnitGroup = SxmUnitGroup<std::uint8_t>;

private:
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

    struct FanoutOutput {
        std::size_t cycle;
        std::size_t tile;
        StreamPayloadTileSegment segment;
        SxmStreamId stream;
    };

    struct State {
        explicit State(SxmTransposeArchitecture architecture)
        {
            if (architecture == SxmTransposeArchitecture::WideStreaming)
                wide.emplace();
        }

        std::optional<SxmWideTranspose> wide{};
        UnitGroup units{};
        std::vector<FanoutOutput> fanout_outputs{};
        std::array<std::optional<SxmInstruction>, hw::kTileRows>
            transpose_instruction_rows{};
        std::optional<SxmInstruction> permute_instruction{};
        TransposeBank transpose_bank{};
    };

public:

    explicit SxmSlice(SxmStreamPortMap ports,
        SxmTransposeArchitecture architecture = SxmTransposeArchitecture::LocalBlocks)
        : ports_(std::move(ports))
        , architecture_(architecture)
        , current_(std::make_unique<State>(architecture))
    {
    }

    SxmSlice(const SxmSlice&) = delete;
    SxmSlice& operator=(const SxmSlice&) = delete;
    SxmSlice(SxmSlice&&) = delete;
    SxmSlice& operator=(SxmSlice&&) = delete;

    void reset(std::size_t cycle = 0)
    {
        current_ = std::make_unique<State>(architecture_);
        current_->units.reset(cycle);
        next_.reset();
        pending_issues_.clear();
        active_fabric_ = nullptr;
        low_started_ = false;
        low_complete_ = false;
        cycle_events_.clear();
    }

    std::size_t cycle() const noexcept
    {
        return current_->units.cycle();
    }

    SxmTransposeArchitecture transpose_architecture() const noexcept
    {
        return architecture_;
    }

    bool idle() const noexcept
    {
        if (!pending_issues_.empty()) return false;
        const auto& state = *current_;
        if (state.wide && !state.wide->idle()) return false;
        if (state.units.issued_count() != 0 || state.permute_instruction
            || !state.fanout_outputs.empty()) return false;
        for (const auto& row : state.transpose_instruction_rows) if (row) return false;
        // A partially captured block still owns its source slot. Match the
        // VModel compat engine's occupied-row busy rule; drain must not wait
        // only for tile_ready (all eight rows).
        for (const auto rows : state.transpose_bank.input_row_mask) if (rows != 0) return false;
        return true;
    }

    bool can_issue(const SxmInstruction& instruction) const
    {
        if (low_started_) return false;
        auto preview = *current_;
        for (const auto& pending : pending_issues_)
            issue_into(preview, pending);
        return can_issue_from(preview, instruction);
    }

    void issue(SxmInstruction instruction)
    {
        if (low_started_)
            throw std::logic_error("SXM instruction issue must precede LOW evaluation");
        auto preview = *current_;
        for (const auto& pending : pending_issues_)
            issue_into(preview, pending);
        issue_into(preview, instruction);
        pending_issues_.push_back(std::move(instruction));
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
        return current_->transpose_instruction_rows[tile];
    }

    void set_trace_enabled(bool enabled) noexcept
    {
        trace_enabled_ = enabled;
    }

    void evaluate(StreamRegisterFabric& fabric)
    {
        if (active_fabric_ != nullptr || low_started_)
            throw std::logic_error("SxmSlice LOW evaluated more than once");
        active_fabric_ = &fabric;
        low();
    }

    void low() override
    {
        if (active_fabric_ == nullptr)
            throw std::logic_error("SxmSlice LOW requires an attached SR fabric");
        auto& fabric = *active_fabric_;
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) {
            throw std::logic_error("SxmSlice::evaluate requires an open SR cycle");
        }

        next_ = std::make_unique<State>(*current_);
        for (auto& pending : pending_issues_)
            issue_into(*next_, std::move(pending));
        pending_issues_.clear();
        low_started_ = true;

        auto& state = *next_;
        cycle_events_.clear();
        if (auto instruction=state.wide ? state.wide->active_instruction() : std::nullopt) {
            state.wide->evaluate(fabric,state.units.cycle(),
                ports_.input_column(uniform_direction(instruction->src_streams,"wide source")),
                ports_.output_column(uniform_direction(instruction->dst_streams,"wide destination")));
            if(trace_enabled_)cycle_events_.push_back("wide transpose input_fifo_bytes="
                +std::to_string(state.wide->input_buffer_bytes())+" output_fifo_bytes="
                +std::to_string(state.wide->output_buffer_bytes()));
        }

        // Fanout traverses the same four superlane pipeline positions as a
        // MEM vector. Keep its output beat skew explicit rather than making
        // all tiles appear on the SR fabric in one cycle.
        for (const auto& output : state.fanout_outputs)
            if (output.cycle == state.units.cycle())
                stage_block_segment(fabric, output.tile, output.segment,
                    output.stream, "SXM block fanout pipeline");
        state.fanout_outputs.erase(std::remove_if(state.fanout_outputs.begin(), state.fanout_outputs.end(),
            [&](const auto& output) { return output.cycle <= state.units.cycle(); }), state.fanout_outputs.end());

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto inputs = read_and_consume_inputs(fabric, tile);
            const auto result = state.units.evaluate(inputs);
            write_outputs(fabric, tile, result);
        }

        if (state.permute_instruction.has_value()) {
            auto processed_tiles = std::array<bool, hw::kTileRows> {};
            execute_block_permute(fabric, *state.permute_instruction, processed_tiles);
            capture_transpose_inputs(fabric);
        } else {
            capture_transpose_inputs(fabric);
        }

        state.permute_instruction.reset();
        state.units.complete_cycle();
        advance_transpose_instructions();
        low_complete_ = true;
    }

    void validate_posedge() const override
    {
        if (!low_started_ || !low_complete_ || !next_ || active_fabric_ == nullptr)
            throw std::logic_error("SXM posedge requires one complete LOW evaluation");
    }

    void posedge() override
    {
        validate_posedge();
        current_.swap(next_);
        next_.reset();
        active_fabric_ = nullptr;
        low_started_ = false;
        low_complete_ = false;
    }

    void high() override {}

    void log_cycle(std::ostream& os) const
    {
        os << "sxm cycle " << (current_->units.cycle() == 0 ? 0 : current_->units.cycle() - 1) << '\n';
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

    static bool is_streaming_transpose(const SxmInstruction& instruction)
    {
        if (instruction.opcode != SxmOpcode::Transpose
            || instruction.dst_streams.size()
                != kTransposeBytePlanes * hw::kLanesPerTile)
            return false;
        if (instruction.input_row == SxmInstruction::kAllInputRows)
            return instruction.src_streams.size()
                == kTransposeBytePlanes * hw::kLanesPerTile;
        return (instruction.input_row < hw::kLanesPerTile
                || instruction.input_row == SxmInstruction::kBroadcastInputRows)
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

    bool can_issue_from(const State& state, const SxmInstruction& instruction) const
    {
        if (architecture_ == SxmTransposeArchitecture::WideStreaming) {
            if (instruction.opcode == SxmOpcode::Transpose)
                return state.wide->can_issue(instruction, state.units.cycle())
                    && state.units.issued_count() == 0
                    && !state.permute_instruction;
            if (state.wide && !state.wide->idle()) return false;
        }
        if (instruction.opcode == SxmOpcode::Transpose) {
            return is_streaming_transpose(instruction)
                && !state.transpose_instruction_rows[0].has_value();
        }
        if (instruction.opcode == SxmOpcode::Permute) {
            return is_block_permute(instruction)
                && !state.permute_instruction.has_value();
        }
        return state.units.can_issue(instruction);
    }

    void issue_into(State& state, SxmInstruction instruction) const
    {
        if (architecture_ == SxmTransposeArchitecture::WideStreaming) {
            if (!can_issue_from(state, instruction))
                throw std::logic_error("wide SXM instruction cannot issue");
            if (instruction.opcode == SxmOpcode::Transpose) {
                state.wide->issue(instruction, state.units.cycle());
                return;
            }
        }
        if (instruction.opcode == SxmOpcode::Transpose) {
            if (!is_streaming_transpose(instruction)) {
                throw std::invalid_argument(
                    "SXM Transpose requires exactly 16 input and 16 output streams");
            }
            if (state.transpose_instruction_rows[0].has_value()) {
                throw std::logic_error("SXM transpose south row issued twice in one cycle");
            }
            check_uniform_direction(instruction.src_streams, "Transpose source");
            check_uniform_direction(instruction.dst_streams, "Transpose destination");
            state.transpose_instruction_rows[0] = std::move(instruction);
            return;
        }
        if (instruction.opcode == SxmOpcode::Permute) {
            if (!is_block_permute(instruction)) {
                throw std::invalid_argument(
                    "SXM Permute requires exactly 16 input and 16 output streams");
            }
            if (state.permute_instruction.has_value()) {
                throw std::logic_error("SXM permute queue issued twice in one cycle");
            }
            check_uniform_direction(instruction.src_streams, "Permute source");
            check_uniform_direction(instruction.dst_streams, "Permute destination");
            state.permute_instruction = std::move(instruction);
            return;
        }
        state.units.issue(std::move(instruction));
    }

    void capture_transpose_inputs(StreamRegisterFabric& fabric)
    {
        auto& state = next_state();
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto& instruction = state.transpose_instruction_rows[tile];
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
        if (!all_valid) {
            if (trace_enabled_) {
                std::ostringstream event;
                event << "transpose wait tile=" << tile << " valid=";
                for (std::size_t index = 0;
                     index < instruction.src_streams.size(); ++index) {
                    if (index != 0) event << ',';
                    const auto source = StreamId::from_packed(
                        instruction.src_streams[index].stream);
                    event << (fabric.segment_valid(
                        input_column, tile, source) ? '1' : '0')
                          << '@' << fabric.last_segment_cycles(
                              input_column, tile, source);
                }
                cycle_events_.push_back(event.str());
            }
            return;
        }

            auto& bank = next_state().transpose_bank;
            if (bank.tile_ready[tile]) {
                throw std::logic_error(
                    "SXM transpose buffer is full at tile "
                    + std::to_string(tile)
                    + " (ready_cycle="
                    + std::to_string(bank.ready_cycle[tile]) + ")");
            }
            if (bank.input_row_mask[tile] == 0)
                bank.dst_streams[tile] = instruction.dst_streams;
            else if (bank.dst_streams[tile] != instruction.dst_streams)
                throw std::logic_error("SXM serial Transpose destination changed");

            const std::size_t first_row =
                instruction.input_row == SxmInstruction::kAllInputRows
                    || instruction.input_row == SxmInstruction::kBroadcastInputRows
                ? 0 : instruction.input_row;
            const std::size_t row_end =
                instruction.input_row == SxmInstruction::kAllInputRows
                    || instruction.input_row == SxmInstruction::kBroadcastInputRows
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
        if (bank.tile_ready[tile]) bank.ready_cycle[tile] = next_state().units.cycle();
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
        auto& rows = next_state().transpose_instruction_rows;
        for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile) {
            rows[tile] = std::move(rows[tile - 1]);
        }
        rows[0].reset();
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
        // The block crossbar may fan out one source superlane to several
        // destinations. Each destination still selects one complete block;
        // this requires no arithmetic or additional transpose storage.
        for (const auto source : instruction.permute_map)
            if (source >= SxmInstruction::kTotalLanes)
                throw std::out_of_range("SXM block source is outside the physical plane");
        std::array<bool, hw::kTileRows> used_sources{};
        bool fanout = false;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto source = block_permute_source_tile(tile, instruction.permute_map);
            fanout = fanout || used_sources[source];
            used_sources[source] = true;
        }
        if (instruction.output_row != SxmInstruction::kAllOutputRows
            && instruction.output_row > 15) {
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
            auto& bank = next_state().transpose_bank;
            // A transpose result must spend one full cycle in its output
            // register before Permute may consume it.
            if (!bank.tile_ready[source_tile]
                || bank.ready_cycle[source_tile] >= next_state().units.cycle()) continue;
            if (bank.dst_streams[source_tile] != instruction.src_streams) {
                throw std::logic_error("SXM Permute source does not match the ready Transpose block");
            }

            const std::size_t first_row =
                instruction.output_row == SxmInstruction::kAllOutputRows
                ? 0 : instruction.output_row >= SxmInstruction::kConsumeOutputRowBase
                    ? instruction.output_row - SxmInstruction::kConsumeOutputRowBase : instruction.output_row;
            const std::size_t row_end =
                instruction.output_row == SxmInstruction::kAllOutputRows
                ? hw::kLanesPerTile : first_row + 1;
            for (std::size_t row = first_row; row < row_end; ++row) {
                for (std::size_t plane = 0; plane < kTransposeBytePlanes; ++plane) {
                    if (fanout && destination_tile != 0) {
                        next_state().fanout_outputs.push_back(FanoutOutput {
                            next_state().units.cycle() + destination_tile,
                            destination_tile,
                            permute_block_segment(bank, plane, row, source_tile,
                                destination_tile, instruction.permute_map),
                            instruction.dst_streams[row * kTransposeBytePlanes + plane]});
                        continue;
                    }
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
            processed_tiles[destination_tile] = true;
            if (trace_enabled_) {
                std::ostringstream event;
                event << "permute source_tile=" << source_tile
                      << " destination_tile=" << destination_tile
                      << " output_streams=" << instruction.dst_streams.size();
                cycle_events_.push_back(event.str());
            }
        }
        // Release only after every destination has observed the source. A
        // full-plane operation consumes even the source blocks it discards.
        if (instruction.output_row == SxmInstruction::kAllOutputRows
            || instruction.output_row >= SxmInstruction::kConsumeOutputRowBase
            || instruction.output_row + 1 == hw::kLanesPerTile) {
            if (instruction.output_tile == SxmInstruction::kAllOutputTiles) {
                for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                    auto& bank = next_state().transpose_bank;
                    if (!bank.tile_ready[tile]
                        || bank.ready_cycle[tile] >= next_state().units.cycle()
                        || bank.dst_streams[tile] != instruction.src_streams) continue;
                    bank.tile_ready[tile] = false;
                    bank.input_row_mask[tile] = 0;
                }
            } else {
                const auto source = block_permute_source_tile(instruction.output_tile, instruction.permute_map);
                next_state().transpose_bank.tile_ready[source] = false;
                next_state().transpose_bank.input_row_mask[source] = 0;
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
        for (const auto& instruction : next_state().units.issued_instructions()) {
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
    SxmTransposeArchitecture architecture_;
    std::unique_ptr<State> current_{};
    std::unique_ptr<State> next_{};
    std::vector<SxmInstruction> pending_issues_{};
    StreamRegisterFabric* active_fabric_{};
    bool low_started_{};
    bool low_complete_{};
    std::vector<std::string> cycle_events_{};
    bool trace_enabled_{true};

    State& next_state()
    {
        if (!next_) throw std::logic_error("SXM Next state is not prepared");
        return *next_;
    }

    const State& next_state() const
    {
        if (!next_) throw std::logic_error("SXM Next state is not prepared");
        return *next_;
    }
};

} // namespace ftlpu
