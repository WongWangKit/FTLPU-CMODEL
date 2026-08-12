#pragma once

#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/sxm/permute.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
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

// Physical SR-facing SXM slice: an 8x8 capture bank feeds a fixed-wiring 8x8
// transpose output bank, followed by the explicit Permute vector stage.
class SxmSlice {
public:
    static constexpr std::size_t kTransposeBytePlanes = 2;

    struct TimingSnapshot {
        std::size_t cycle{0};
        std::size_t captured_rows{0};
        std::size_t transpose_bank_loads{0};
        std::size_t transpose_rows{0};
        std::size_t permute_rows{0};
    };

    explicit SxmSlice(SxmStreamPortMap ports)
        : ports_(std::move(ports))
    {
    }

    void reset()
    {
        cycle_ = 0;
        for (auto& instruction : transpose_instruction_rows_) instruction.reset();
        transpose_block_rows_.fill(std::nullopt);
        transpose_windows_ = {};
        transpose_pipeline_.fill(std::nullopt);
        permute_pipeline_.fill(std::nullopt);
        blocks_.clear();
        pending_permutes_.clear();
        next_block_id_ = 0;
        next_transpose_issue_cycle_ = 0;
        permute_issue_cycle_.reset();
        captured_rows_ = 0;
        transpose_bank_loads_ = 0;
        transpose_rows_ = 0;
        permute_rows_ = 0;
    }

    std::size_t cycle() const noexcept
    {
        return cycle_;
    }

    TimingSnapshot timing_snapshot() const noexcept
    {
        return TimingSnapshot {
            cycle_ == 0 ? 0 : cycle_ - 1,
            captured_rows_, transpose_bank_loads_,
            transpose_rows_, permute_rows_};
    }

    bool can_issue(const SxmInstruction& instruction) const
    {
        if (instruction.opcode == SxmOpcode::Transpose) {
            return is_streaming_transpose(instruction)
                && !transpose_instruction_rows_[0].has_value()
                && cycle_ >= next_transpose_issue_cycle_;
        }
        if (instruction.opcode == SxmOpcode::Permute) {
            return is_block_permute(instruction)
                && permute_issue_cycle_ != cycle_;
        }
        return false;
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
            if (cycle_ < next_transpose_issue_cycle_) {
                throw std::logic_error(
                    "SXM streaming Transpose accepts one block instruction every eight cycles");
            }
            check_uniform_direction(instruction.src_streams, "Transpose source");
            check_uniform_direction(instruction.dst_streams, "Transpose destination");

            const auto block_id = next_block_id_++;
            auto control = BlockControl{block_id, instruction};
            if (!pending_permutes_.empty()) {
                control.permute = std::move(pending_permutes_.front());
                pending_permutes_.pop_front();
            }
            blocks_.push_back(std::move(control));
            transpose_block_rows_[0] = block_id;
            transpose_instruction_rows_[0] = std::move(instruction);
            next_transpose_issue_cycle_ = cycle_ + hw::kLanesPerTile;
            return;
        }
        if (instruction.opcode == SxmOpcode::Permute) {
            if (!is_block_permute(instruction)) {
                throw std::invalid_argument(
                    "SXM Permute requires exactly 16 input and 16 output streams");
            }
            if (permute_issue_cycle_ == cycle_) {
                throw std::logic_error("SXM Permute issued twice in one cycle");
            }
            check_uniform_direction(instruction.src_streams, "Permute source");
            check_uniform_direction(instruction.dst_streams, "Permute destination");
            Permute320::validate_bijection(instruction.permute_map);
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                static_cast<void>(
                    block_permute_source_tile(tile, instruction.permute_map));
            }

            auto attached = false;
            for (auto& block : blocks_) {
                if (!block.permute) {
                    block.permute = std::move(instruction);
                    attached = true;
                    break;
                }
            }
            if (!attached) {
                pending_permutes_.push_back(std::move(instruction));
            }
            permute_issue_cycle_ = cycle_;
            return;
        }
        throw std::invalid_argument("SXM supports only physical Transpose and Permute instructions");
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
        captured_rows_ = 0;
        transpose_bank_loads_ = 0;
        transpose_rows_ = 0;
        permute_rows_ = 0;

        drain_permute_pipeline(fabric);
        advance_transpose_pipeline();
        start_transpose_captures();
        produce_transpose_rows();
        capture_transpose_rows(fabric);
        retire_completed_blocks();

        ++cycle_;
        advance_transpose_instructions();
    }

    void log_cycle(std::ostream& os) const
    {
        os << "sxm cycle " << (cycle_ == 0 ? 0 : cycle_ - 1) << '\n';
        if (cycle_events_.empty()) {
            os << "  idle\n";
            return;
        }
        for (const auto& event : cycle_events_) os << "  " << event << '\n';
    }

private:
    using TransposeVector = std::array<
        std::array<std::uint8_t, hw::kLanesPerTile>,
        kTransposeBytePlanes>;

    using TransposeBlock = std::array<
        std::array<
            std::array<std::uint8_t, hw::kLanesPerTile>,
            hw::kLanesPerTile>,
        kTransposeBytePlanes>;

    struct BlockControl {
        std::size_t id{0};
        SxmInstruction transpose{};
        std::optional<SxmInstruction> permute{};
        std::size_t completed_destination_tiles{0};
        bool completed{false};
    };

    // Two fixed-role 8x8 BF16 register banks per tile.  capture_bank always
    // receives one flat MXM row per cycle.  Once all eight rows are present,
    // fixed transpose wiring loads the complete transposed matrix into
    // output_bank in one register stage.  While output_bank drains one row
    // per cycle, capture_bank collects the following block.
    struct TransposeWindow {
        TransposeBlock capture_bank{};
        TransposeBlock output_bank{};
        std::optional<std::size_t> resident_block{};
        std::size_t output_row{0};
        bool resident_drained{false};

        std::optional<std::size_t> capture_block{};
        std::size_t capture_row{0};
    };

    struct TransposePipelineEntry {
        std::size_t block_id{0};
        std::size_t source_tile{0};
        std::size_t row{0};
        TransposeVector data{};
    };

    struct PermutePipelineEntry {
        std::size_t block_id{0};
        std::size_t destination_tile{0};
        std::size_t row{0};
        TransposeVector data{};
        SxmInstruction::StreamList dst_streams{};
    };

    static bool is_streaming_transpose(const SxmInstruction& instruction)
    {
        return instruction.opcode == SxmOpcode::Transpose
            && instruction.src_streams.size()
                == kTransposeBytePlanes * hw::kLanesPerTile
            && instruction.dst_streams.size()
                == kTransposeBytePlanes * hw::kLanesPerTile;
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

    BlockControl& block_control(std::size_t id)
    {
        for (auto& block : blocks_) {
            if (block.id == id) return block;
        }
        throw std::logic_error("SXM streaming block control retired too early");
    }

    const BlockControl& block_control(std::size_t id) const
    {
        for (const auto& block : blocks_) {
            if (block.id == id) return block;
        }
        throw std::logic_error("SXM streaming block control retired too early");
    }

    void start_transpose_captures()
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto& instruction = transpose_instruction_rows_[tile];
            if (!instruction) continue;
            const auto block_id = transpose_block_rows_[tile];
            if (!block_id) {
                throw std::logic_error("SXM transpose instruction lost its block id");
            }

            auto& window = transpose_windows_[tile];
            if (window.capture_block) {
                throw std::logic_error(
                    "SXM transpose instruction reached a tile while its "
                    "previous block was still collecting");
            }
            window.capture_block = *block_id;
            window.capture_row = 0;

            if (trace_enabled_) {
                std::ostringstream event;
                event << "transpose tile=" << tile
                      << " block=" << *block_id
                      << " collection=start";
                cycle_events_.push_back(event.str());
            }
        }
    }

    void capture_transpose_rows(StreamRegisterFabric& fabric)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto& window = transpose_windows_[tile];
            if (!window.capture_block) continue;

            const auto block_id = *window.capture_block;
            const auto& instruction = block_control(block_id).transpose;
            const auto row = window.capture_row;
            const auto input_direction = uniform_direction(
                instruction.src_streams, "Transpose source");
            const auto input_column = ports_.input_column(input_direction);

            auto ready = true;
            for (std::size_t plane = 0;
                 plane < kTransposeBytePlanes;
                ++plane) {
                const auto source = StreamId::from_packed(
                    instruction.src_streams[
                        row * kTransposeBytePlanes + plane].stream);
                ready = ready && fabric.segment_valid(
                    input_column, tile, source);
            }
            if (!ready) continue;

            for (std::size_t plane = 0;
                 plane < kTransposeBytePlanes;
                ++plane) {
                const auto source = StreamId::from_packed(
                    instruction.src_streams[
                        row * kTransposeBytePlanes + plane].stream);
                const auto input = fabric.segment(
                    input_column, tile, source);
                fabric.consume_segment(
                    input_column,
                    tile,
                    source,
                    "SXM serial FP16 Transpose input");
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    window.capture_bank[plane][row][lane] =
                        input[lane].data;
                }
            }

            ++window.capture_row;
            ++captured_rows_;
            if (trace_enabled_) {
                std::ostringstream event;
                event << "transpose tile=" << tile
                      << " block=" << block_id
                      << " input_row=" << row;
                cycle_events_.push_back(event.str());
            }
            if (window.capture_row == hw::kLanesPerTile) {
                if (window.resident_block && !window.resident_drained) {
                    throw std::logic_error(
                        "SXM capture bank filled before the transpose output bank drained");
                }
                // The transpose is fixed wiring, not an ALU operation.  All
                // 64 BF16 elements cross into the output bank on this edge.
                for (std::size_t plane = 0;
                     plane < kTransposeBytePlanes;
                     ++plane) {
                    for (std::size_t output_row = 0;
                         output_row < hw::kLanesPerTile;
                         ++output_row) {
                        for (std::size_t lane = 0;
                             lane < hw::kLanesPerTile;
                             ++lane) {
                            window.output_bank[plane][output_row][lane] =
                                window.capture_bank[plane][lane][output_row];
                        }
                    }
                }
                window.resident_block = block_id;
                window.output_row = 0;
                window.resident_drained = false;
                window.capture_block.reset();
                ++transpose_bank_loads_;

                if (trace_enabled_) {
                    std::ostringstream event;
                    event << "transpose tile=" << tile
                          << " block=" << block_id
                          << " capture->output-bank";
                    cycle_events_.push_back(event.str());
                }
            }
        }
    }

    void advance_transpose_instructions()
    {
        for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile) {
            transpose_instruction_rows_[tile] = std::move(transpose_instruction_rows_[tile - 1]);
            transpose_block_rows_[tile] = transpose_block_rows_[tile - 1];
        }
        transpose_instruction_rows_[0].reset();
        transpose_block_rows_[0].reset();
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

    void produce_transpose_rows()
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto& window = transpose_windows_[tile];
            if (transpose_pipeline_[tile]
                || !window.resident_block
                || window.resident_drained) {
                continue;
            }

            const auto block_id = *window.resident_block;
            const auto& control = block_control(block_id);
            if (!control.permute) continue;
            if (control.permute->src_streams
                != control.transpose.dst_streams) {
                throw std::logic_error(
                    "SXM Permute input streams do not match Transpose output streams");
            }

            const auto row = window.output_row;
            auto entry = TransposePipelineEntry{block_id, tile, row, {}};
            for (std::size_t plane = 0;
                 plane < kTransposeBytePlanes;
                 ++plane) {
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    entry.data[plane][lane] =
                        window.output_bank[plane][row][lane];
                }
            }
            transpose_pipeline_[tile] = std::move(entry);
            ++transpose_rows_;

            ++window.output_row;
            if (window.output_row == hw::kLanesPerTile) {
                window.resident_drained = true;
            }

            if (trace_enabled_) {
                std::ostringstream event;
                event << "transpose tile=" << tile
                      << " block=" << block_id
                      << " output_row=" << row;
                cycle_events_.push_back(event.str());
            }
        }
    }

    void advance_transpose_pipeline()
    {
        for (std::size_t source_tile = 0;
             source_tile < hw::kTileRows;
             ++source_tile) {
            if (!transpose_pipeline_[source_tile]) continue;

            const auto& entry = *transpose_pipeline_[source_tile];
            const auto& control = block_control(entry.block_id);
            if (!control.permute) continue;
            const auto& instruction = *control.permute;

            std::optional<std::size_t> destination_tile{};
            for (std::size_t candidate = 0;
                 candidate < hw::kTileRows;
                 ++candidate) {
                if (block_permute_source_tile(
                        candidate, instruction.permute_map)
                    == source_tile) {
                    destination_tile = candidate;
                    break;
                }
            }
            if (!destination_tile) {
                throw std::logic_error(
                    "SXM Permute map does not route a source superlane");
            }
            if (permute_pipeline_[*destination_tile]) {
                throw std::logic_error(
                    "SXM Permute pipeline collision between streaming blocks");
            }

            auto output = PermutePipelineEntry{
                entry.block_id,
                *destination_tile,
                entry.row,
                {},
                instruction.dst_streams,
            };
            const auto destination_base =
                *destination_tile * hw::kLanesPerTile;
            for (std::size_t plane = 0;
                 plane < kTransposeBytePlanes;
                 ++plane) {
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    const auto source = instruction.permute_map[
                        destination_base + lane];
                    output.data[plane][lane] =
                        entry.data[plane][source % hw::kLanesPerTile];
                }
            }
            permute_pipeline_[*destination_tile] = std::move(output);
            transpose_pipeline_[source_tile].reset();
        }
    }

    void drain_permute_pipeline(StreamRegisterFabric& fabric)
    {
        for (std::size_t destination_tile = 0;
             destination_tile < hw::kTileRows;
             ++destination_tile) {
            if (!permute_pipeline_[destination_tile]) continue;
            const auto entry = std::move(*permute_pipeline_[destination_tile]);
            permute_pipeline_[destination_tile].reset();
            ++permute_rows_;

            for (std::size_t plane = 0;
                 plane < kTransposeBytePlanes;
                 ++plane) {
                auto segment = StreamPayloadTileSegment{};
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    segment[lane] = entry.data[plane][lane];
                }
                stage_block_segment(
                    fabric,
                    destination_tile,
                    segment,
                    entry.dst_streams[
                        entry.row * kTransposeBytePlanes + plane],
                    "SXM streaming FP16 Permute output");
            }

            if (entry.row + 1 == hw::kLanesPerTile) {
                auto& control = block_control(entry.block_id);
                ++control.completed_destination_tiles;
                if (control.completed_destination_tiles
                    == hw::kTileRows) {
                    control.completed = true;
                }
            }

            if (trace_enabled_) {
                std::ostringstream event;
                event << "permute tile=" << destination_tile
                      << " block=" << entry.block_id
                      << " output_row=" << entry.row;
                cycle_events_.push_back(event.str());
            }
        }
    }

    void retire_completed_blocks()
    {
        while (!blocks_.empty() && blocks_.front().completed) {
            blocks_.pop_front();
        }
    }

    static StreamId physical_stream(SxmStreamId stream)
    {
        return StreamId::from_packed(stream.stream);
    }

    SxmStreamPortMap ports_;
    std::size_t cycle_{0};
    std::array<std::optional<SxmInstruction>, hw::kTileRows> transpose_instruction_rows_{};
    std::array<std::optional<std::size_t>, hw::kTileRows>
        transpose_block_rows_{};
    std::array<TransposeWindow, hw::kTileRows> transpose_windows_{};
    std::array<std::optional<TransposePipelineEntry>, hw::kTileRows>
        transpose_pipeline_{};
    std::array<std::optional<PermutePipelineEntry>, hw::kTileRows>
        permute_pipeline_{};
    std::deque<BlockControl> blocks_{};
    std::deque<SxmInstruction> pending_permutes_{};
    std::size_t next_block_id_{0};
    std::size_t next_transpose_issue_cycle_{0};
    std::optional<std::size_t> permute_issue_cycle_{};
    std::vector<std::string> cycle_events_{};
    std::size_t captured_rows_{0};
    std::size_t transpose_bank_loads_{0};
    std::size_t transpose_rows_{0};
    std::size_t permute_rows_{0};
    bool trace_enabled_{true};
};

} // namespace ftlpu
