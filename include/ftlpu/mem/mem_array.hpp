#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream_fabric.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mem/sram.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

// Maps the twelve SR boundaries surrounding the 11 groups of four MEM slices
// into arbitrary physical SR column IDs in a whole-chip StreamRegisterFabric.
class MemStreamPortMap {
public:
    using BoundaryColumns =
        std::array<std::size_t, hw::kMemBoundaryStreamRegisterColumns>;

    enum class OutputPlacement {
        // A MEM group is located between boundary g (west) and g+1 (east).
        // Incoming east streams are consumed at g and newly produced east
        // streams appear at g+1; west streams use the mirrored mapping.
        DownstreamBoundary,

        // Compatibility with the original TileArrayModel, which injected a
        // Read at the same upstream boundary used by Write.  Whole-system
        // integration should not select this mode.
        LegacyInputBoundary,
    };

    static MemStreamPortMap BetweenBoundaries()
    {
        BoundaryColumns columns{};
        for (std::size_t i = 0; i < columns.size(); ++i) {
            columns[i] = i;
        }
        return MemStreamPortMap(columns, OutputPlacement::DownstreamBoundary);
    }

    static MemStreamPortMap LegacyLocalLinear()
    {
        BoundaryColumns columns{};
        for (std::size_t i = 0; i < columns.size(); ++i) {
            columns[i] = i;
        }
        return MemStreamPortMap(columns, OutputPlacement::LegacyInputBoundary);
    }

    // Retained as a source-compatible alias.  New code should use the
    // semantically explicit BetweenBoundaries() factory.
    static MemStreamPortMap LocalLinear()
    {
        return BetweenBoundaries();
    }

    explicit MemStreamPortMap(
        BoundaryColumns columns,
        OutputPlacement output_placement = OutputPlacement::DownstreamBoundary)
        : columns_(std::move(columns))
        , output_placement_(output_placement)
    {
    }

    std::size_t boundary_column(std::size_t boundary) const
    {
        return columns_.at(boundary);
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
        if (output_placement_ == OutputPlacement::LegacyInputBoundary) {
            return input_column(mem_slice, direction);
        }

        const auto group = group_for(mem_slice);
        return direction == StreamDirection::East
            ? columns_[group + 1]
            : columns_[group];
    }

    // Compatibility for code that treated every MEM access as an input-side
    // attachment.  Prefer input_column()/output_column() in new code.
    std::size_t attachment_column(
        std::size_t mem_slice,
        StreamDirection direction) const
    {
        return input_column(mem_slice, direction);
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
            throw std::out_of_range("MEM slice is outside the 44-slice hemisphere");
        }
        return mem_slice / hw::kMemSlicesPerGroup;
    }

    BoundaryColumns columns_{};
    OutputPlacement output_placement_{OutputPlacement::DownstreamBoundary};
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
        std::size_t tile{0};
        std::size_t sr_column{0};
        StreamId stream{StreamId::East(0)};
        MemLocalWordAddress13 address{};
        std::size_t bank{0};
        std::size_t word{0};
        StreamPayloadSegment16 bytes{};
        std::uint64_t vector_tag{0};
    };

    struct InstructionTrace {
        std::size_t mem_slice{0};
        std::size_t tile{0};
        MemInstruction instruction{};
        std::uint64_t issue_tag{0};
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
    }

    void reset_execution_state()
    {
        cycle_ = 0;
        for (auto& queue : instruction_queues_) {
            queue.clear();
        }
        for (auto& slice : instruction_rows_) {
            for (auto& slot : slice) {
                slot.reset();
            }
        }
        for (auto& slice : instruction_tag_rows_) {
            for (auto& tag : slice) {
                tag.reset();
            }
        }
        executed_mem_.clear();
        executed_instructions_.clear();
        next_issue_tag_ = 1;
    }

    void clear_sram()
    {
        sram_.clear();
    }

    void reset_and_clear_sram()
    {
        reset_execution_state();
        clear_sram();
    }

    std::size_t cycle() const noexcept
    {
        return cycle_;
    }

    const MemStreamPortMap& ports() const noexcept
    {
        return ports_;
    }

    void enqueue_instruction(std::size_t mem_slice, MemInstruction instruction)
    {
        check_mem_slice(mem_slice);
        if (next_issue_tag_ == 0) {
            throw std::overflow_error("MEM issue tag space is exhausted");
        }
        instruction_queues_[mem_slice].push_back(
            InstructionToken {std::move(instruction), next_issue_tag_++});
    }

    void set_sram_byte(
        std::size_t mem_slice,
        std::size_t tile,
        MemSliceByteAddress17 address,
        std::uint8_t value)
    {
        check_mem_slice(mem_slice);
        check_tile(tile);
        sram_.slice(mem_slice).tile_block(tile).set_byte(address, value);
    }

    std::uint8_t sram_byte(
        std::size_t mem_slice,
        std::size_t tile,
        MemSliceByteAddress17 address) const
    {
        check_mem_slice(mem_slice);
        check_tile(tile);
        return sram_.slice(mem_slice).tile_block(tile).byte(address);
    }

    void set_sram_lane_byte(
        std::size_t mem_slice,
        std::size_t tile,
        MemLocalWordAddress13 address,
        std::size_t lane,
        std::uint8_t value)
    {
        check_tile(tile);
        check_lane(lane);
        set_sram_byte(mem_slice, tile, address.slice_byte_address(lane), value);
    }

    void set_sram_lane_byte(
        std::size_t mem_slice,
        std::size_t tile,
        std::size_t address,
        std::size_t lane,
        std::uint8_t value)
    {
        set_sram_lane_byte(
            mem_slice, tile, MemLocalWordAddress13(address), lane, value);
    }

    std::uint8_t sram_lane_byte(
        std::size_t mem_slice,
        std::size_t tile,
        MemLocalWordAddress13 address,
        std::size_t lane) const
    {
        check_tile(tile);
        check_lane(lane);
        return sram_byte(mem_slice, tile, address.slice_byte_address(lane));
    }

    std::uint8_t sram_lane_byte(
        std::size_t mem_slice,
        std::size_t tile,
        std::size_t address,
        std::size_t lane) const
    {
        return sram_lane_byte(
            mem_slice, tile, MemLocalWordAddress13(address), lane);
    }

    void write_sram_vector(
        std::size_t mem_slice,
        MemLocalWordAddress13 address,
        const StreamPayloadVector320& values)
    {
        check_mem_slice(mem_slice);
        sram_.slice(mem_slice).write_vector(address, values);
    }

    StreamPayloadVector320 read_sram_vector(
        std::size_t mem_slice,
        MemLocalWordAddress13 address) const
    {
        check_mem_slice(mem_slice);
        return sram_.slice(mem_slice).read_vector(address);
    }

    const InstructionSlot& instruction_at(std::size_t mem_slice, std::size_t tile) const
    {
        check_mem_slice(mem_slice);
        check_tile(tile);
        return instruction_rows_[mem_slice][tile];
    }

    std::optional<std::uint64_t> issue_tag_at(
        std::size_t mem_slice,
        std::size_t tile) const
    {
        check_mem_slice(mem_slice);
        check_tile(tile);
        return instruction_tag_rows_[mem_slice][tile];
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
    void evaluate(StreamRegisterFabric& fabric)
    {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) {
            throw std::logic_error("MemArrayModel::evaluate requires an open SR cycle");
        }

        executed_mem_.clear();
        executed_instructions_.clear();
        dispatch_from_queues();
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
    struct InstructionToken {
        MemInstruction instruction{};
        std::uint64_t issue_tag{0};
    };

    static void check_mem_slice(std::size_t mem_slice)
    {
        if (mem_slice >= hw::kMemSliceColumns) {
            throw std::out_of_range("MEM slice is outside the 44-slice hemisphere");
        }
    }

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kTileRows) {
            throw std::out_of_range("tile is outside the 20-row MEM slice");
        }
    }

    static void check_lane(std::size_t lane)
    {
        if (lane >= hw::kLanesPerTile) {
            throw std::out_of_range("lane is outside the 16-byte MEM tile segment");
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
            os << "(";
            print_address(os, instruction.address);
            os << ",s=" << instruction.stream << ")";
            break;
        case MemOpcode::Gather:
        case MemOpcode::Scatter:
            os << "(s=" << instruction.stream << ",map=" << instruction.map_stream << ")";
            break;
        }
    }

    static void print_address(std::ostream& os, MemLocalWordAddress13 address)
    {
        const auto old_flags = os.flags();
        const auto old_fill = os.fill();
        os << "addr=b" << address.bank() << ":w" << address.word()
           << " encoded=0x" << std::hex << std::setw(4) << std::setfill('0')
           << address.encoded();
        os.flags(old_flags);
        os.fill(old_fill);
    }

    void dispatch_from_queues()
    {
        for (std::size_t mem_slice = 0; mem_slice < hw::kMemSliceColumns; ++mem_slice) {
            if (instruction_rows_[mem_slice][0].has_value()
                || instruction_queues_[mem_slice].empty()) {
                continue;
            }
            instruction_rows_[mem_slice][0] = instruction_queues_[mem_slice].front().instruction;
            instruction_tag_rows_[mem_slice][0] = instruction_queues_[mem_slice].front().issue_tag;
            instruction_queues_[mem_slice].pop_front();
        }
    }

    void execute_current_instructions(StreamRegisterFabric& fabric)
    {
        for (std::size_t mem_slice = 0; mem_slice < hw::kMemSliceColumns; ++mem_slice) {
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                const auto& instruction = instruction_rows_[mem_slice][tile];
                if (!instruction.has_value()) {
                    continue;
                }

                const auto issue_tag = instruction_tag_rows_[mem_slice][tile];
                if (!issue_tag.has_value()) {
                    throw std::logic_error("MEM instruction lost its stable issue tag");
                }
                executed_instructions_.push_back(
                    InstructionTrace {mem_slice, tile, *instruction, *issue_tag});
                switch (instruction->opcode) {
                case MemOpcode::Read:
                    execute_read(fabric, mem_slice, tile, *instruction, *issue_tag);
                    break;
                case MemOpcode::Write:
                    execute_write(fabric, mem_slice, tile, *instruction);
                    break;
                case MemOpcode::Gather:
                case MemOpcode::Scatter:
                    throw std::logic_error("Gather/Scatter require a separate address-stream datapath model");
                }
            }
        }
    }

    void execute_read(
        StreamRegisterFabric& fabric,
        std::size_t mem_slice,
        std::size_t tile,
        const MemInstruction& instruction,
        std::uint64_t issue_tag)
    {
        const auto stream = instruction.stream_id();
        const auto sr_column = ports_.output_column(mem_slice, stream.direction());
        const auto bytes =
            sram_.slice(mem_slice).tile_block(tile).read_word(instruction.address);

        StreamOutputPort output(
            fabric,
            sr_column,
            stream.direction(),
            "MEM Read");
        output.write_payload_segment(
            tile,
            stream.index(),
            bytes,
            issue_tag);

        executed_mem_.push_back(MemTransfer {
            MemTransfer::Kind::LoadSramToStream,
            mem_slice,
            tile,
            sr_column,
            stream,
            instruction.address,
            instruction.address.bank(),
            instruction.address.word(),
            bytes,
            issue_tag,
        });
    }

    void execute_write(
        StreamRegisterFabric& fabric,
        std::size_t mem_slice,
        std::size_t tile,
        const MemInstruction& instruction)
    {
        const auto stream = instruction.stream_id();
        const auto sr_column = ports_.input_column(mem_slice, stream.direction());
        StreamPayloadSegment16 bytes{};
        StreamInputPort input(
            fabric,
            sr_column,
            stream.direction(),
            "MEM Write");

        if (input.segment_valid(tile, stream.index())) {
            const auto segment = input.consume_segment(tile, stream.index());
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                bytes[lane] = segment[lane].data;
            }
        } else if (missing_stream_policy_ == MissingStreamPolicy::Error) {
            throw std::logic_error("MEM Write reached an invalid stream segment");
        }

        sram_.slice(mem_slice).tile_block(tile).write_word(instruction.address, bytes);
        executed_mem_.push_back(MemTransfer {
            MemTransfer::Kind::StoreStreamToSram,
            mem_slice,
            tile,
            sr_column,
            stream,
            instruction.address,
            instruction.address.bank(),
            instruction.address.word(),
            bytes,
        });
    }

    void advance_instructions()
    {
        for (auto& mem_slice : instruction_rows_) {
            for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile) {
                mem_slice[tile] = mem_slice[tile - 1];
            }
            mem_slice[0].reset();
        }
        for (auto& mem_slice : instruction_tag_rows_) {
            for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile) {
                mem_slice[tile] = mem_slice[tile - 1];
            }
            mem_slice[0].reset();
        }
    }

    static void print_hex_bytes(std::ostream& os, const StreamPayloadSegment16& bytes)
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
            os << "    c" << trace.mem_slice << ".t" << trace.tile << '=';
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
            os << "    c" << transfer.mem_slice << ".t" << transfer.tile << ' '
               << (transfer.kind == MemTransfer::Kind::StoreStreamToSram ? "store" : "load")
               << ' ' << direction_name(transfer.stream.direction()) << transfer.stream.index()
               << ' ';
            print_address(os, transfer.address);
            os << " bytes=0x";
            print_hex_bytes(os, transfer.bytes);
            os << '\n';
        }
    }

    MemStreamPortMap ports_;
    MissingStreamPolicy missing_stream_policy_{MissingStreamPolicy::Error};
    SramArray sram_{};
    std::array<std::deque<InstructionToken>, hw::kMemSliceColumns> instruction_queues_{};
    std::array<std::array<InstructionSlot, hw::kTileRows>, hw::kMemSliceColumns>
        instruction_rows_{};
    std::array<std::array<std::optional<std::uint64_t>, hw::kTileRows>, hw::kMemSliceColumns>
        instruction_tag_rows_{};
    std::vector<MemTransfer> executed_mem_{};
    std::vector<InstructionTrace> executed_instructions_{};
    std::size_t cycle_{0};
    std::uint64_t next_issue_tag_{1};
};

} // namespace ftlpu
