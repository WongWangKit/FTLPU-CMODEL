#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream_fabric.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mem/sram.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

// Maps the SR boundaries surrounding the groups of four MEM slices
// into arbitrary physical SR column IDs in a whole-chip StreamRegisterFabric.
class MemStreamPortMap {
public:
    using BoundaryColumns =
        std::array<std::size_t, hw::kMemBoundaryStreamRegisterColumns>;

    static MemStreamPortMap BetweenBoundaries()
    {
        BoundaryColumns columns{};
        for (std::size_t i = 0; i < columns.size(); ++i) {
            columns[i] = i;
        }
        return MemStreamPortMap(columns);
    }

    explicit MemStreamPortMap(BoundaryColumns columns)
        : columns_(std::move(columns))
    {
    }

    std::size_t input_column(
        std::size_t mem_slice,
        StreamDirection direction) const
    {
        const auto group = group_for(mem_slice);
        return direction == StreamDirection::East
            ? columns_[group]
            : columns_[group + 1];
    }

    std::size_t output_column(
        std::size_t mem_slice,
        StreamDirection direction) const
    {
        const auto group = group_for(mem_slice);
        return direction == StreamDirection::East
            ? columns_[group + 1]
            : columns_[group];
    }

    void validate_for(const StreamRegisterFabric& fabric) const
    {
        for (const auto column : columns_) {
            if (column >= fabric.column_count()) {
                throw std::out_of_range("MEM SR boundary maps outside stream-register fabric");
            }
        }
    }

private:
    static std::size_t group_for(std::size_t mem_slice)
    {
        if (mem_slice >= hw::kMemSliceColumns) {
            throw std::out_of_range("MEM slice is outside the configured hemisphere");
        }
        return mem_slice / hw::kMemSlicesPerGroup;
    }

    BoundaryColumns columns_{};
};

class MemArrayModel {
public:
    using InstructionSlot = std::optional<MemInstruction>;

    enum class MissingStreamPolicy {
        Error,
        ZeroFill,
    };

    struct MemTransfer {
        enum class Kind {
            StoreStreamToSram,
            LoadSramToStream,
        };

        Kind kind{Kind::StoreStreamToSram};
        std::size_t mem_slice{0};
        std::size_t bank{0};
        std::size_t tile{0};
        std::size_t sr_column{0};
        StreamId stream{StreamId::East(0)};
        std::size_t address{0};
        std::uint64_t vector_tag{0};
        StreamPayloadTileSegment bytes{};
    };

    struct InstructionTrace {
        std::size_t mem_slice{0};
        std::size_t bank{0};
        std::size_t tile{0};
        MemInstruction instruction{};
    };

    explicit MemArrayModel(
        MemStreamPortMap ports = MemStreamPortMap::BetweenBoundaries(),
        MissingStreamPolicy missing_stream_policy = MissingStreamPolicy::Error)
        : ports_(std::move(ports))
        , missing_stream_policy_(missing_stream_policy)
    {
    }

    void reset()
    {
        reset_execution_state();
        sram_.clear();
    }

    void reset_execution_state()
    {
        cycle_ = 0;
        for (auto& slice : instruction_queues_) {
            for (auto& bank : slice)
                for (auto& queue : bank) queue.clear();
        }
        for (auto& slice : instruction_rows_) {
            for (auto& bank : slice) {
                for (auto& port : bank)
                    for (auto& slot : port) slot.reset();
            }
        }
        executed_mem_.clear();
        executed_instructions_.clear();
        transport_commits_.clear();
    }

    void set_sram_depth_rows(std::size_t rows)
    {
        sram_.set_active_rows(rows);
    }

    std::size_t cycle() const noexcept
    {
        return cycle_;
    }

    const MemStreamPortMap& ports() const noexcept
    {
        return ports_;
    }

    bool idle() const noexcept
    {
        for (const auto& slice : instruction_queues_)
            for (const auto& bank : slice)
                for (const auto& queue : bank)
                    if (!queue.empty()) return false;
        for (const auto& slice : instruction_rows_)
            for (const auto& bank : slice)
                for (const auto& port : bank)
                    for (const auto& slot : port)
                        if (slot) return false;
        return true;
    }

    bool compute_idle() const noexcept
    {
        for (const auto& slice : instruction_queues_)
            for (const auto& bank : slice)
                for (const auto& queue : bank)
                    for (const auto& instruction : queue)
                        if (!instruction.transport_generation) return false;
        for (const auto& slice : instruction_rows_)
            for (const auto& bank : slice)
                for (const auto& port : bank)
                    for (const auto& slot : port)
                        if (slot && !slot->transport_generation) return false;
        return true;
    }

    bool transport_idle() const noexcept
    {
        for (const auto& slice : instruction_queues_)
            for (const auto& bank : slice)
                for (const auto& queue : bank)
                    for (const auto& instruction : queue)
                        if (instruction.transport_generation) return false;
        for (const auto& slice : instruction_rows_)
            for (const auto& bank : slice)
                for (const auto& port : bank)
                    for (const auto& slot : port)
                        if (slot && slot->transport_generation) return false;
        return true;
    }

    void configure_transport_commit(std::uint64_t generation, std::uint32_t launch,
        std::size_t slice, std::size_t bank, std::size_t vectors)
    {
        check_mem_slice(slice); check_bank(bank);
        if (!generation || !launch || !vectors || transport_commits_.size() >= 65535
            || vectors > std::numeric_limits<std::uint64_t>::max() / hw::kTileRows)
            throw std::invalid_argument("invalid or oversized MEM commit scoreboard");
        for (const auto& s : transport_commits_)
            if (s.launch == launch && s.slice == slice && s.bank == bank)
                throw std::invalid_argument("duplicate MEM commit scoreboard owner");
        transport_commits_.push_back({generation, launch, slice, bank,
            static_cast<std::uint64_t>(vectors) * hw::kTileRows, 0});
    }

    std::uint64_t transport_committed_segments(std::uint64_t generation,
        std::uint32_t launch, std::size_t slice, std::size_t bank) const
    {
        for (const auto& s : transport_commits_)
            if (s.generation == generation && s.launch == launch
                && s.slice == slice && s.bank == bank) return s.committed;
        throw std::invalid_argument("unregistered MEM commit scoreboard owner");
    }

    // Physical SRAM writes, independent of optional debug trace capture.
    std::size_t last_transport_commit_segments() const noexcept
    {
        return last_transport_commit_segments_;
    }

    void enqueue_instruction(std::size_t mem_slice, MemInstruction instruction)
    {
        enqueue_instruction(mem_slice, 0, std::move(instruction));
    }

    void enqueue_instruction(
        std::size_t mem_slice,
        std::size_t bank,
        MemInstruction instruction)
    {
        check_mem_slice(mem_slice);
        check_bank(bank);
        const std::size_t port = instruction.opcode == MemOpcode::Write
            || instruction.opcode == MemOpcode::Scatter ? 1 : 0;
        instruction_queues_[mem_slice][bank][port].push_back(
            std::move(instruction));
    }

    void set_sram_byte(
        std::size_t mem_slice,
        std::size_t row,
        std::size_t byte_offset,
        std::uint8_t value)
    {
        set_sram_byte(mem_slice, 0, row, byte_offset, value);
    }

    void set_sram_byte(
        std::size_t mem_slice,
        std::size_t bank,
        std::size_t row,
        std::size_t byte_offset,
        std::uint8_t value)
    {
        check_mem_slice(mem_slice);
        check_bank(bank);
        sram_.bank(mem_slice, bank).set_byte(row, byte_offset, value);
    }

    std::uint8_t sram_byte(
        std::size_t mem_slice,
        std::size_t row,
        std::size_t byte_offset) const
    {
        return sram_byte(mem_slice, 0, row, byte_offset);
    }

    std::uint8_t sram_byte(
        std::size_t mem_slice,
        std::size_t bank,
        std::size_t row,
        std::size_t byte_offset) const
    {
        check_mem_slice(mem_slice);
        check_bank(bank);
        return sram_.bank(mem_slice, bank).byte(row, byte_offset);
    }

    void set_sram_lane_byte(
        std::size_t mem_slice,
        std::size_t tile,
        std::size_t row,
        std::size_t lane,
        std::uint8_t value)
    {
        set_sram_lane_byte(mem_slice, 0, tile, row, lane, value);
    }

    void set_sram_lane_byte(
        std::size_t mem_slice,
        std::size_t bank,
        std::size_t tile,
        std::size_t row,
        std::size_t lane,
        std::uint8_t value)
    {
        check_tile(tile);
        check_lane(lane);
        set_sram_byte(
            mem_slice, bank, row,
            tile * hw::kLanesPerTile + lane, value);
    }

    std::uint8_t sram_lane_byte(
        std::size_t mem_slice,
        std::size_t tile,
        std::size_t row,
        std::size_t lane) const
    {
        return sram_lane_byte(mem_slice, 0, tile, row, lane);
    }

    std::uint8_t sram_lane_byte(
        std::size_t mem_slice,
        std::size_t bank,
        std::size_t tile,
        std::size_t row,
        std::size_t lane) const
    {
        check_tile(tile);
        check_lane(lane);
        return sram_byte(
            mem_slice, bank, row,
            tile * hw::kLanesPerTile + lane);
    }

    const InstructionSlot& instruction_at(std::size_t mem_slice, std::size_t tile) const
    {
        return instruction_at(mem_slice, 0, tile);
    }

    const InstructionSlot& instruction_at(
        std::size_t mem_slice,
        std::size_t bank,
        std::size_t tile) const
    {
        check_mem_slice(mem_slice);
        check_bank(bank);
        check_tile(tile);
        const auto& read = instruction_rows_[mem_slice][bank][0][tile];
        return read.has_value()
            ? read : instruction_rows_[mem_slice][bank][1][tile];
    }

    const std::vector<MemTransfer>& executed_transfers() const noexcept
    {
        return executed_mem_;
    }

    const std::vector<InstructionTrace>& executed_instructions() const noexcept
    {
        return executed_instructions_;
    }

    // Evaluate one MEM cycle against the current SR state.  This method never
    // propagates or commits the SR fabric: the owning system performs one
    // global fabric commit after all functional slices have evaluated.
    void evaluate(StreamRegisterFabric& fabric, bool capture_trace = true)
    {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) {
            throw std::logic_error("MemArrayModel::evaluate requires an open SR cycle");
        }

        executed_mem_.clear();
        executed_instructions_.clear();
        last_transport_commit_segments_ = 0;
        dispatch_from_queues();
        capture_trace_ = capture_trace;
        execute_current_instructions(fabric);
        advance_instructions();
        ++cycle_;
    }

    void log_cycle(std::ostream& os, std::optional<std::size_t> log_tile = std::nullopt) const
    {
        if (log_tile.has_value()) {
            check_tile(*log_tile);
        }
        os << "mem cycle " << (cycle_ == 0 ? 0 : cycle_ - 1) << '\n';
        log_instructions(os, log_tile);
        log_mem(os, log_tile);
    }

private:
    static void check_mem_slice(std::size_t mem_slice)
    {
        if (mem_slice >= hw::kMemSliceColumns) {
            throw std::out_of_range("MEM slice is outside the configured hemisphere");
        }
    }

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kTileRows) {
            throw std::out_of_range("tile is outside the configured MEM slice");
        }
    }

    static void check_bank(std::size_t bank)
    {
        if (bank >= hw::kMemBanksPerSlice) {
            throw std::out_of_range(
                "bank is outside the configured MEM slice");
        }
    }

    static void check_lane(std::size_t lane)
    {
        if (lane >= hw::kLanesPerTile) {
            throw std::out_of_range("lane is outside the 8-byte MEM tile segment");
        }
    }

    static const char* direction_name(StreamDirection direction) noexcept
    {
        return direction == StreamDirection::East ? "E" : "W";
    }

    static const char* opcode_name(MemOpcode opcode) noexcept
    {
        switch (opcode) {
        case MemOpcode::Read:
            return "Read";
        case MemOpcode::Write:
            return "Write";
        case MemOpcode::Gather:
            return "Gather";
        case MemOpcode::Scatter:
            return "Scatter";
        }
        return "Unknown";
    }

    static void print_instruction(std::ostream& os, const MemInstruction& instruction)
    {
        os << opcode_name(instruction.opcode);
        switch (instruction.opcode) {
        case MemOpcode::Read:
        case MemOpcode::Write:
            os << "(a=" << instruction.address << ",s=" << instruction.stream << ")";
            break;
        case MemOpcode::Gather:
        case MemOpcode::Scatter:
            os << "(s=" << instruction.stream << ",map=" << instruction.map_stream << ")";
            break;
        }
    }

    void dispatch_from_queues()
    {
        for (std::size_t mem_slice = 0; mem_slice < hw::kMemSliceColumns; ++mem_slice) {
            for (std::size_t bank = 0; bank < hw::kMemBanksPerSlice; ++bank) {
                for (std::size_t port = 0; port < 2; ++port) {
                    if (instruction_rows_[mem_slice][bank][port][0].has_value()
                        || instruction_queues_[mem_slice][bank][port].empty())
                        continue;
                    instruction_rows_[mem_slice][bank][port][0] =
                        instruction_queues_[mem_slice][bank][port].front();
                    instruction_queues_[mem_slice][bank][port].pop_front();
                }
            }
        }
    }

    void execute_current_instructions(StreamRegisterFabric& fabric)
    {
        for (std::size_t mem_slice = 0; mem_slice < hw::kMemSliceColumns; ++mem_slice) {
            for (std::size_t bank = 0; bank < hw::kMemBanksPerSlice; ++bank) {
                // Read executes before write when both ports target one SRAM
                // row in the same cycle, matching synchronous read-before-
                // write behavior.
                for (std::size_t port = 0; port < 2; ++port)
                    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                        const auto& instruction =
                            instruction_rows_[mem_slice][bank][port][tile];
                        if (!instruction.has_value()) continue;

                        if (capture_trace_) {
                            executed_instructions_.push_back(
                                InstructionTrace {
                                    mem_slice, bank, tile, *instruction});
                        }
                        switch (instruction->opcode) {
                        case MemOpcode::Read:
                            execute_read(fabric, mem_slice, bank, tile,
                                *instruction);
                            break;
                        case MemOpcode::Write:
                            execute_write(fabric, mem_slice, bank, tile,
                                *instruction);
                            break;
                        case MemOpcode::Gather:
                        case MemOpcode::Scatter:
                            throw std::logic_error("Gather/Scatter require a separate address-stream datapath model");
                        }
                    }
            }
        }
    }

    void execute_read(
        StreamRegisterFabric& fabric,
        std::size_t mem_slice,
        std::size_t bank,
        std::size_t tile,
        const MemInstruction& instruction)
    {
        const auto stream = instruction.stream_id();
        const auto sr_column = ports_.output_column(mem_slice, stream.direction());
        const auto physical_address = instruction.address;
        const auto bytes =
            sram_.bank(mem_slice, bank).read_segment(tile, physical_address);
        const auto vector_tag = static_cast<std::uint64_t>(cycle_) * hw::kTileRows + tile;

        StreamOutputPort output(
            fabric,
            sr_column,
            stream.direction(),
            "MEM Read");
        output.write_payload_segment(
            tile,
            stream.index(),
            bytes,
            vector_tag);

        if (capture_trace_) {
            executed_mem_.push_back(MemTransfer {
                MemTransfer::Kind::LoadSramToStream,
                mem_slice,
                bank,
                tile,
                sr_column,
                stream,
                physical_address,
                vector_tag,
                bytes,
            });
        }
    }

    void execute_write(
        StreamRegisterFabric& fabric,
        std::size_t mem_slice,
        std::size_t bank,
        std::size_t tile,
        const MemInstruction& instruction)
    {
        const auto stream = instruction.stream_id();
        const auto sr_column = ports_.input_column(mem_slice, stream.direction());
        StreamPayloadTileSegment bytes{};
        StreamInputPort input(
            fabric,
            sr_column,
            stream.direction(),
            "MEM Write");
        std::uint64_t vector_tag = 0;

        if (input.segment_valid(tile, stream.index())) {
            const auto segment = instruction.preserve_stream
                ? input.peek_segment(tile, stream.index())
                : input.consume_segment(tile, stream.index());
            vector_tag = segment.front().vector_tag;
            if (instruction.transport_generation
                && !std::all_of(segment.begin(), segment.end(),
                    [](const auto& cell) { return cell.transport; }))
                throw std::logic_error("transport MEM write consumed computation SR data");
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                bytes[lane] = segment[lane].data;
            }
        } else if (instruction.transport_generation) {
            throw std::logic_error("transport MEM commit has no valid C2C payload");
        } else if (instruction.preserve_stream) {
            // A tap is a conditional observation of an existing stream.  The
            // ICU command traverses every tile, but only the tile carrying the
            // selected stream may update SRAM.
            return;
        } else if (missing_stream_policy_ == MissingStreamPolicy::Error) {
            std::string valid_streams;
            for (std::size_t candidate = 0;
                 candidate < hw::kStreamsPerDirection; ++candidate) {
                if (!input.segment_valid(tile, candidate)) continue;
                if (!valid_streams.empty()) valid_streams += ",";
                valid_streams += std::to_string(candidate);
            }
            std::string valid_tiles;
            for (std::size_t candidate = 0;
                 candidate < hw::kTileRows; ++candidate) {
                if (!input.segment_valid(candidate, stream.index())) continue;
                if (!valid_tiles.empty()) valid_tiles += ",";
                valid_tiles += std::to_string(candidate);
            }
            std::string valid_cells;
            for (std::size_t candidate_tile = 0;
                 candidate_tile < hw::kTileRows; ++candidate_tile) {
                for (std::size_t candidate_stream = 0;
                     candidate_stream < hw::kStreamsPerDirection;
                     ++candidate_stream) {
                    if (!input.segment_valid(
                            candidate_tile, candidate_stream))
                        continue;
                    if (!valid_cells.empty()) valid_cells += ",";
                    valid_cells += std::to_string(candidate_tile) + ":"
                        + std::to_string(candidate_stream);
                }
            }
            std::string stream_columns;
            for (std::size_t candidate_column = 0;
                 candidate_column < fabric.column_count();
                 ++candidate_column) {
                for (std::size_t candidate_tile = 0;
                     candidate_tile < hw::kTileRows; ++candidate_tile) {
                    if (!fabric.segment_valid(
                            candidate_column, candidate_tile, stream))
                        continue;
                    if (!stream_columns.empty()) stream_columns += ",";
                    stream_columns += std::to_string(candidate_column) + ":"
                        + std::to_string(candidate_tile);
                }
            }
            std::string staged_stream_columns;
            std::string last_stream_column_cycles;
            for (std::size_t candidate_column = 0;
                 candidate_column < fabric.column_count();
                 ++candidate_column) {
                for (std::size_t candidate_tile = 0;
                     candidate_tile < hw::kTileRows; ++candidate_tile) {
                    if (!last_stream_column_cycles.empty())
                        last_stream_column_cycles += ";";
                    last_stream_column_cycles +=
                        std::to_string(candidate_column) + ":"
                        + std::to_string(candidate_tile) + "="
                        + fabric.last_segment_cycles(
                            candidate_column, candidate_tile, stream);
                    if (!fabric.staged_segment_valid(
                            candidate_column, candidate_tile, stream))
                        continue;
                    if (!staged_stream_columns.empty())
                        staged_stream_columns += ",";
                    staged_stream_columns +=
                        std::to_string(candidate_column) + ":"
                        + std::to_string(candidate_tile);
                }
            }
            throw std::logic_error(
                "MEM Write reached an invalid stream segment: slice="
                + std::to_string(mem_slice)
                + " bank=" + std::to_string(bank)
                + " tile=" + std::to_string(tile)
                + " sr_column=" + std::to_string(sr_column)
                + " stream=" + std::to_string(stream.packed())
                + " address=" + std::to_string(instruction.address)
                + " valid_stream_indices_at_tile="
                + (valid_streams.empty() ? "none" : valid_streams)
                + " valid_tiles_for_stream="
                + (valid_tiles.empty() ? "none" : valid_tiles)
                + " valid_cells_at_column="
                + (valid_cells.empty() ? "none" : valid_cells)
                + " stream_columns="
                + (stream_columns.empty() ? "none" : stream_columns)
                + " staged_stream_columns="
                + (staged_stream_columns.empty()
                    ? "none" : staged_stream_columns)
                + " last_target_segment_cycles="
                + fabric.last_segment_cycles(sr_column, tile, stream)
                + " last_stream_column_cycles="
                + last_stream_column_cycles);
        }

        sram_.bank(mem_slice, bank).write_segment(
            tile, instruction.address, bytes);
        if (instruction.transport_generation) {
            auto found = std::find_if(transport_commits_.begin(), transport_commits_.end(),
                [&](const auto& s) { return s.generation == instruction.transport_generation
                    && s.launch == instruction.transport_launch
                    && s.slice == mem_slice && s.bank == bank; });
            if (found == transport_commits_.end() || found->committed >= found->expected)
                throw std::logic_error("unbound or duplicate transport SRAM commit");
            ++found->committed;
            ++last_transport_commit_segments_;
        }
        if (capture_trace_) {
            executed_mem_.push_back(MemTransfer {
                MemTransfer::Kind::StoreStreamToSram,
                mem_slice,
                bank,
                tile,
                sr_column,
                stream,
                instruction.address,
                vector_tag,
                bytes,
            });
        }
    }

    void advance_instructions()
    {
        for (auto& mem_slice : instruction_rows_) {
            for (auto& bank : mem_slice) {
                for (auto& port : bank) {
                    for (std::size_t tile = hw::kTileRows - 1;
                         tile > 0; --tile)
                        port[tile] = port[tile - 1];
                    port[0].reset();
                }
            }
        }
    }

    static void print_hex_bytes(std::ostream& os, const StreamPayloadTileSegment& bytes)
    {
        const auto old_flags = os.flags();
        const auto old_fill = os.fill();
        os << std::hex << std::setfill('0');
        for (const auto byte : bytes) {
            os << std::setw(2) << static_cast<unsigned>(byte);
        }
        os.flags(old_flags);
        os.fill(old_fill);
    }

    void log_instructions(std::ostream& os, std::optional<std::size_t> log_tile) const
    {
        os << "  instructions:";
        bool any = false;
        for (const auto& trace : executed_instructions_) {
            if (!log_tile.has_value() || trace.tile == *log_tile) {
                any = true;
                break;
            }
        }
        if (!any) {
            os << " idle\n";
            return;
        }
        os << '\n';
        for (const auto& trace : executed_instructions_) {
            if (log_tile.has_value() && trace.tile != *log_tile) {
                continue;
            }
            os << "    c" << trace.mem_slice << ".b" << trace.bank
               << ".t" << trace.tile << '=';
            print_instruction(os, trace.instruction);
            os << '\n';
        }
    }

    void log_mem(std::ostream& os, std::optional<std::size_t> log_tile) const
    {
        os << "  mem:";
        bool any = false;
        for (const auto& transfer : executed_mem_) {
            if (!log_tile.has_value() || transfer.tile == *log_tile) {
                any = true;
                break;
            }
        }
        if (!any) {
            os << " idle\n";
            return;
        }
        os << '\n';
        for (const auto& transfer : executed_mem_) {
            if (log_tile.has_value() && transfer.tile != *log_tile) {
                continue;
            }
            os << "    c" << transfer.mem_slice << ".b" << transfer.bank
               << ".t" << transfer.tile << ' '
               << (transfer.kind == MemTransfer::Kind::StoreStreamToSram
                       ? "store"
                       : "load")
               << ' ' << direction_name(transfer.stream.direction()) << transfer.stream.index()
               << " addr=" << transfer.address
               << " tag=" << transfer.vector_tag << " bytes=0x";
            print_hex_bytes(os, transfer.bytes);
            os << '\n';
        }
    }

    MemStreamPortMap ports_;
    MissingStreamPolicy missing_stream_policy_{MissingStreamPolicy::Error};
    SramArray sram_{};
    std::array<
        std::array<std::array<std::deque<MemInstruction>, 2>,
            hw::kMemBanksPerSlice>,
        hw::kMemSliceColumns> instruction_queues_{};
    std::array<
        std::array<
            std::array<std::array<InstructionSlot, hw::kTileRows>, 2>,
            hw::kMemBanksPerSlice>,
        hw::kMemSliceColumns> instruction_rows_{};
    std::vector<MemTransfer> executed_mem_{};
    std::vector<InstructionTrace> executed_instructions_{};
    std::size_t cycle_{0};
    bool capture_trace_{true};
    struct TransportCommitState {
        std::uint64_t generation;
        std::uint32_t launch;
        std::size_t slice, bank;
        std::uint64_t expected, committed;
    };
    std::vector<TransportCommitState> transport_commits_{};
    std::size_t last_transport_commit_segments_{0};
};

} // namespace ftlpu
