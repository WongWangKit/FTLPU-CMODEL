#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/core/stream.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

enum class StreamDirection {
    East,
    West,
};

class TileArrayModel {
public:
    using Data = std::uint8_t;
    using DataWord = StreamWord<Data>;
    using StreamSlot = std::optional<DataWord>;
    using InstructionSlot = std::optional<MemInstruction>;

    struct MemTransfer {
        enum class Kind {
            StoreStreamToSram,
            LoadSramToStream,
        };

        Kind kind{Kind::StoreStreamToSram};
        std::size_t column{0};
        std::size_t tile{0};
        StreamDirection direction{StreamDirection::East};
        std::size_t stream{0};
        std::size_t address{0};
        std::array<Data, hw::kLanesPerTile> bytes{};
    };

    struct InstructionTrace {
        std::size_t column{0};
        std::size_t tile{0};
        MemInstruction instruction{};
    };

    void reset()
    {
        cycle_ = 0;
        for (auto& queue : instruction_queues_) {
            queue.clear();
        }
        for (auto& column : instruction_rows_) {
            for (auto& slot : column) {
                slot.reset();
            }
        }
        clear_streams();
        clear_pending_inputs();
        executed_mem_.clear();
        executed_instructions_.clear();
        pending_stream_writes_.clear();
        std::fill(sram_.begin(), sram_.end(), 0);
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    void enqueue_instruction(std::size_t column, MemInstruction instruction)
    {
        check_column(column);
        instruction_queues_[column].push_back(instruction);
    }

    void set_east_stream_input(std::size_t tile, std::size_t lane, std::size_t stream, DataWord word)
    {
        check_tile(tile);
        check_lane(lane);
        check_east_stream(stream);
        pending_east_inputs_[tile][lane][stream] = word;
    }

    void set_west_stream_input(std::size_t tile, std::size_t lane, std::size_t stream, DataWord word)
    {
        check_tile(tile);
        check_lane(lane);
        check_west_stream(stream);
        pending_west_inputs_[tile][lane][stream] = word;
    }

    void set_sram_byte(std::size_t column, std::size_t tile, std::size_t address, Data value)
    {
        check_column(column);
        check_tile(tile);
        sram_[sram_index(column, tile, address)] = value;
    }

    const StreamSlot& east_register(std::size_t tile, std::size_t lane, std::size_t register_file, std::size_t stream) const
    {
        check_tile(tile);
        check_lane(lane);
        check_register_file(register_file);
        check_east_stream(stream);
        return east_regs_[tile][lane][register_file][stream];
    }

    StreamSlot consume_east_register(std::size_t tile, std::size_t lane, std::size_t register_file, std::size_t stream)
    {
        check_tile(tile);
        check_lane(lane);
        check_register_file(register_file);
        check_east_stream(stream);
        auto value = east_regs_[tile][lane][register_file][stream];
        east_regs_[tile][lane][register_file][stream].reset();
        return value;
    }

    const StreamSlot& west_register(std::size_t tile, std::size_t lane, std::size_t register_file, std::size_t stream) const
    {
        check_tile(tile);
        check_lane(lane);
        check_register_file(register_file);
        check_west_stream(stream);
        return west_regs_[tile][lane][register_file][stream];
    }

    Data sram_byte(std::size_t column, std::size_t tile, std::size_t address) const
    {
        check_column(column);
        check_tile(tile);
        return sram_[sram_index(column, tile, address)];
    }

    const InstructionSlot& instruction_at(std::size_t column, std::size_t tile) const
    {
        check_column(column);
        check_tile(tile);
        return instruction_rows_[column][tile];
    }

    void tick(std::ostream& os, std::optional<std::size_t> log_tile = std::nullopt)
    {
        if (log_tile.has_value()) {
            check_tile(*log_tile);
        }
        dispatch_from_queues();
        execute_current_instructions();
        advance_instructions();
        advance_streams();
        apply_pending_stream_writes();
        apply_pending_inputs();
        log_cycle(os, log_tile);
        executed_mem_.clear();
        executed_instructions_.clear();
        pending_stream_writes_.clear();
        ++cycle_;
    }

private:
    struct DecodedStream {
        StreamDirection direction{StreamDirection::East};
        std::size_t stream{0};
    };

    struct DecodedSramAddress {
        std::size_t bank{0};
        std::size_t word{0};
        std::size_t byte{0};
    };

    struct PendingStreamWrite {
        StreamDirection direction{StreamDirection::East};
        std::size_t tile{0};
        std::size_t lane{0};
        std::size_t register_file{0};
        std::size_t stream{0};
        DataWord word{};
    };

    static void check_column(std::size_t column)
    {
        if (column >= hw::kSliceColumns) {
            throw std::out_of_range("slice column is outside the TSP grid");
        }
    }

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kTileRows) {
            throw std::out_of_range("tile is outside the TSP grid");
        }
    }

    static void check_lane(std::size_t lane)
    {
        if (lane >= hw::kLanesPerTile) {
            throw std::out_of_range("lane is outside the tile");
        }
    }

    static void check_register_file(std::size_t register_file)
    {
        if (register_file >= hw::kStreamRegisterColumns) {
            throw std::out_of_range("stream register column is outside the TSP grid");
        }
    }

    static void check_east_stream(std::size_t stream)
    {
        if (stream >= hw::kEastStreams) {
            throw std::out_of_range("east stream index is outside the lane register file");
        }
    }

    static void check_west_stream(std::size_t stream)
    {
        if (stream >= hw::kWestStreams) {
            throw std::out_of_range("west stream index is outside the lane register file");
        }
    }

    static DecodedSramAddress decode_sram_address(std::size_t address)
    {
        return DecodedSramAddress {
            (address >> 16) & (hw::kSramBanksPerTile - 1),
            (address >> 4) & (hw::kSramBankDepthWords - 1),
            address & (hw::kSramWordBytes - 1),
        };
    }

    static void check_sram_word_aligned(std::size_t address)
    {
        if ((address & (hw::kSramWordBytes - 1)) != 0) {
            throw std::out_of_range("MEM SRAM Read/Write address must be 16-byte word aligned");
        }
    }

    static std::size_t sram_index(std::size_t column, std::size_t tile, std::size_t address)
    {
        const auto decoded = decode_sram_address(address);
        return (((column * hw::kTileRows + tile) * hw::kSramBanksPerTile + decoded.bank)
                   * hw::kSramBankDepthWords
                   + decoded.word)
                * hw::kSramWordBytes
            + decoded.byte;
    }

    static DecodedStream decode_stream(std::size_t stream)
    {
        if (stream >= hw::kStreams) {
            throw std::out_of_range("MEM stream is outside the tile stream set");
        }
        if (stream < hw::kEastStreams) {
            return DecodedStream {StreamDirection::East, stream};
        }
        return DecodedStream {StreamDirection::West, stream - hw::kEastStreams};
    }

    static const char* direction_name(StreamDirection direction)
    {
        return direction == StreamDirection::East ? "E" : "W";
    }

    static const char* opcode_name(MemOpcode opcode)
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

    static void print_word(std::ostream& os, const DataWord& word)
    {
        os << static_cast<unsigned>(word.data);
        if (word.last) {
            os << ":last";
        }
    }

    static std::size_t stream_register_before_column(std::size_t column)
    {
        return column / hw::kSlicesPerGroup;
    }

    static std::size_t stream_register_after_column(std::size_t column)
    {
        return column / hw::kSlicesPerGroup + 1;
    }

    StreamSlot& stream_slot(
        StreamDirection direction,
        std::size_t tile,
        std::size_t lane,
        std::size_t column,
        std::size_t stream)
    {
        return direction == StreamDirection::East
            ? east_regs_[tile][lane][stream_register_before_column(column)][stream]
            : west_regs_[tile][lane][stream_register_after_column(column)][stream];
    }

    const StreamSlot& stream_slot(
        StreamDirection direction,
        std::size_t tile,
        std::size_t lane,
        std::size_t column,
        std::size_t stream) const
    {
        return direction == StreamDirection::East
            ? east_regs_[tile][lane][stream_register_before_column(column)][stream]
            : west_regs_[tile][lane][stream_register_after_column(column)][stream];
    }

    void clear_streams()
    {
        for (auto& tile : east_regs_) {
            for (auto& lane : tile) {
                for (auto& reg : lane) {
                    for (auto& slot : reg) {
                        slot.reset();
                    }
                }
            }
        }
        for (auto& tile : west_regs_) {
            for (auto& lane : tile) {
                for (auto& reg : lane) {
                    for (auto& slot : reg) {
                        slot.reset();
                    }
                }
            }
        }
    }

    void clear_pending_inputs()
    {
        for (auto& tile : pending_east_inputs_) {
            for (auto& lane : tile) {
                for (auto& slot : lane) {
                    slot.reset();
                }
            }
        }
        for (auto& tile : pending_west_inputs_) {
            for (auto& lane : tile) {
                for (auto& slot : lane) {
                    slot.reset();
                }
            }
        }
    }

    void dispatch_from_queues()
    {
        for (std::size_t column = 0; column < hw::kSliceColumns; ++column) {
            if (instruction_rows_[column][0].has_value() || instruction_queues_[column].empty()) {
                continue;
            }
            instruction_rows_[column][0] = instruction_queues_[column].front();
            instruction_queues_[column].pop_front();
        }
    }

    void execute_current_instructions()
    {
        for (std::size_t column = 0; column < hw::kSliceColumns; ++column) {
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                const auto& instruction = instruction_rows_[column][tile];
                if (!instruction.has_value()) {
                    continue;
                }

                executed_instructions_.push_back(InstructionTrace {column, tile, *instruction});
                if (instruction->opcode == MemOpcode::Read) {
                    execute_read(column, tile, *instruction);
                } else if (instruction->opcode == MemOpcode::Write) {
                    execute_write(column, tile, *instruction);
                }
            }
        }
    }

    void execute_read(std::size_t column, std::size_t tile, const MemInstruction& instruction)
    {
        const auto decoded = decode_stream(instruction.stream);
        check_sram_word_aligned(instruction.address);

        MemTransfer transfer {
            MemTransfer::Kind::LoadSramToStream,
            column,
            tile,
            decoded.direction,
            decoded.stream,
            instruction.address,
            {},
        };

        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto value = sram_[sram_index(column, tile, instruction.address + lane)];
            const auto source_register = decoded.direction == StreamDirection::East
                ? stream_register_before_column(column)
                : stream_register_after_column(column);
            pending_stream_writes_.push_back(PendingStreamWrite {
                decoded.direction,
                tile,
                lane,
                source_register,
                decoded.stream,
                DataWord {
                    value,
                    lane + 1 == hw::kLanesPerTile,
                },
            });
            transfer.bytes[lane] = value;
        }

        executed_mem_.push_back(transfer);
    }

    void execute_write(std::size_t column, std::size_t tile, const MemInstruction& instruction)
    {
        const auto decoded = decode_stream(instruction.stream);
        check_sram_word_aligned(instruction.address);

        MemTransfer transfer {
            MemTransfer::Kind::StoreStreamToSram,
            column,
            tile,
            decoded.direction,
            decoded.stream,
            instruction.address,
            {},
        };

        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            auto& slot = stream_slot(decoded.direction, tile, lane, column, decoded.stream);
            const auto value = slot.has_value() ? slot->data : 0;
            sram_[sram_index(column, tile, instruction.address + lane)] = value;
            transfer.bytes[lane] = value;
            slot.reset();
        }

        executed_mem_.push_back(transfer);
    }

    void advance_instructions()
    {
        for (auto& column : instruction_rows_) {
            for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile) {
                column[tile] = column[tile - 1];
            }
            column[0].reset();
        }
    }

    void advance_streams()
    {
        for (auto& tile : east_regs_) {
            for (auto& lane : tile) {
                for (std::size_t reg = hw::kStreamRegisterColumns - 1; reg > 0; --reg) {
                    lane[reg] = lane[reg - 1];
                }
                for (auto& slot : lane[0]) {
                    slot.reset();
                }
            }
        }

        for (auto& tile : west_regs_) {
            for (auto& lane : tile) {
                for (std::size_t reg = 0; reg + 1 < hw::kStreamRegisterColumns; ++reg) {
                    lane[reg] = lane[reg + 1];
                }
                for (auto& slot : lane[hw::kStreamRegisterColumns - 1]) {
                    slot.reset();
                }
            }
        }
    }

    void apply_pending_stream_writes()
    {
        for (const auto& write : pending_stream_writes_) {
            auto& slot = write.direction == StreamDirection::East
                ? east_regs_[write.tile][write.lane][write.register_file][write.stream]
                : west_regs_[write.tile][write.lane][write.register_file][write.stream];
            if (slot.has_value()) {
                std::ostringstream os;
                os << "stream register write collision after MEM read"
                   << " tile=" << write.tile
                   << " lane=" << write.lane
                   << " sreg=" << write.register_file
                   << " stream=" << direction_name(write.direction) << write.stream;
                throw std::logic_error(os.str());
            }
            slot = write.word;
        }
    }

    void apply_pending_inputs()
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                for (std::size_t stream = 0; stream < hw::kEastStreams; ++stream) {
                    auto& input = pending_east_inputs_[tile][lane][stream];
                    if (!input.has_value()) {
                        continue;
                    }
                    auto& slot = east_regs_[tile][lane][0][stream];
                    if (slot.has_value()) {
                        throw std::logic_error("east stream input collided with register file 0");
                    }
                    slot = *input;
                    input.reset();
                }

                for (std::size_t stream = 0; stream < hw::kWestStreams; ++stream) {
                    auto& input = pending_west_inputs_[tile][lane][stream];
                    if (!input.has_value()) {
                        continue;
                    }
                    auto& slot = west_regs_[tile][lane][hw::kStreamRegisterColumns - 1][stream];
                    if (slot.has_value()) {
                        throw std::logic_error("west stream input collided with last register file");
                    }
                    slot = *input;
                    input.reset();
                }
            }
        }
    }

    void log_cycle(std::ostream& os, std::optional<std::size_t> log_tile) const
    {
        os << "mem cycle " << cycle_ << '\n';
        log_instructions(os, log_tile);
        log_mem(os, log_tile);
        log_streams(os, log_tile);
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
            os << "    c" << trace.column << ".t" << trace.tile << "=";
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
            os << "    c" << transfer.column << ".t" << transfer.tile << ' ';
            os << (transfer.kind == MemTransfer::Kind::StoreStreamToSram ? "store" : "load");
            os << " " << direction_name(transfer.direction) << transfer.stream
               << " addr=" << transfer.address << " bytes=0x";
            print_hex_bytes(os, transfer.bytes);
            os << '\n';
        }
    }

    template <typename Bytes>
    static void print_hex_bytes(std::ostream& os, const Bytes& bytes)
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

    template <typename Regs>
    static bool collect_stream_bytes(
        const Regs& regs,
        std::size_t tile,
        std::size_t reg_column,
        std::size_t stream,
        std::array<Data, hw::kLanesPerTile>& bytes)
    {
        bool any = false;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto& slot = regs[tile][lane][reg_column][stream];
            bytes[lane] = slot.has_value() ? slot->data : 0;
            any = any || slot.has_value();
        }
        return any;
    }

    void log_streams(std::ostream& os, std::optional<std::size_t> log_tile) const
    {
        os << "  stream_registers (E/W combined):\n";
        const auto first_tile = log_tile.value_or(0);
        const auto end_tile = log_tile.has_value() ? first_tile + 1 : hw::kTileRows;
        for (std::size_t tile = first_tile; tile < end_tile; ++tile) {
            os << "    tile " << tile << ":\n";
            for (std::size_t reg = 0; reg < hw::kStreamRegisterColumns; ++reg) {
                os << "      sreg " << reg << ":";
                bool any = false;

                for (std::size_t stream = 0; stream < hw::kEastStreams; ++stream) {
                    std::array<Data, hw::kLanesPerTile> bytes{};
                    if (collect_stream_bytes(east_regs_, tile, reg, stream, bytes)) {
                        any = true;
                        os << " E" << stream << "=0x";
                        print_hex_bytes(os, bytes);
                    }
                }

                for (std::size_t stream = 0; stream < hw::kWestStreams; ++stream) {
                    std::array<Data, hw::kLanesPerTile> bytes{};
                    if (collect_stream_bytes(west_regs_, tile, reg, stream, bytes)) {
                        any = true;
                        os << " W" << stream << "=0x";
                        print_hex_bytes(os, bytes);
                    }
                }

                if (!any) {
                    os << " empty";
                }
                os << '\n';
            }
        }
    }

    std::size_t cycle_{0};
    std::array<std::deque<MemInstruction>, hw::kSliceColumns> instruction_queues_{};
    std::array<std::array<InstructionSlot, hw::kTileRows>, hw::kSliceColumns> instruction_rows_{};
    std::array<
        std::array<std::array<std::array<StreamSlot, hw::kEastStreams>, hw::kStreamRegisterColumns>, hw::kLanesPerTile>,
        hw::kTileRows>
        east_regs_{};
    std::array<
        std::array<std::array<std::array<StreamSlot, hw::kWestStreams>, hw::kStreamRegisterColumns>, hw::kLanesPerTile>,
        hw::kTileRows>
        west_regs_{};
    std::array<std::array<std::array<StreamSlot, hw::kEastStreams>, hw::kLanesPerTile>, hw::kTileRows>
        pending_east_inputs_{};
    std::array<std::array<std::array<StreamSlot, hw::kWestStreams>, hw::kLanesPerTile>, hw::kTileRows>
        pending_west_inputs_{};
    std::vector<Data> sram_ = std::vector<Data>(hw::kSliceColumns * hw::kTileRows * hw::kSramBytesPerTile);
    std::vector<MemTransfer> executed_mem_{};
    std::vector<InstructionTrace> executed_instructions_{};
    std::vector<PendingStreamWrite> pending_stream_writes_{};
};

} // namespace ftlpu
