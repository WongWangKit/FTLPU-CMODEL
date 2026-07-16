#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream_fabric.hpp"
#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/sxm/slice.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ftlpu {

// Compatibility facade for existing examples/tests.  Unlike the previous
// implementation, it no longer owns intermingled SRAM and SR arrays:
//   - MemArrayModel owns SRAM and MEM instruction pipelines.
//   - StreamRegisterFabric owns all physical stream-register state.
// New whole-chip integration should own these two objects directly rather
// than treating TileArrayModel as the system topology.
class TileArrayModel {
public:
    using Data = std::uint8_t;
    using DataWord = StreamWord<Data>;
    using StreamSlot = StreamCell;
    using InstructionSlot = MemArrayModel::InstructionSlot;
    using MemTransfer = MemArrayModel::MemTransfer;
    using InstructionTrace = MemArrayModel::InstructionTrace;

    TileArrayModel()
        : mem_(
            MemStreamPortMap::LegacyLocalLinear(),
            MemArrayModel::MissingStreamPolicy::ZeroFill)
        , streams_(hw::kSystemStreamRegisterColumns)
    {
    }

    void reset()
    {
        cycle_ = 0;
        mem_.reset();
        streams_.reset();
        pending_inputs_.clear();
        pending_consumptions_.clear();
    }

    std::size_t cycle() const noexcept
    {
        return cycle_;
    }

    MemArrayModel& memory_model() noexcept
    {
        return mem_;
    }

    const MemArrayModel& memory_model() const noexcept
    {
        return mem_;
    }

    StreamRegisterFabric& stream_fabric() noexcept
    {
        return streams_;
    }

    const StreamRegisterFabric& stream_fabric() const noexcept
    {
        return streams_;
    }

    void enqueue_instruction(std::size_t column, MemInstruction instruction)
    {
        mem_.enqueue_instruction(column, std::move(instruction));
    }

    void set_east_stream_input(
        std::size_t tile,
        std::size_t lane,
        std::size_t stream,
        DataWord word)
    {
        pending_inputs_.push_back(PendingInput {
            0,
            tile,
            lane,
            StreamId::East(stream),
            StreamCell::Valid(word.data, word.last),
        });
    }

    void set_west_stream_input(
        std::size_t tile,
        std::size_t lane,
        std::size_t stream,
        DataWord word)
    {
        pending_inputs_.push_back(PendingInput {
            hw::kMxmBoundaryStreamRegisterColumn,
            tile,
            lane,
            StreamId::West(stream),
            StreamCell::Valid(word.data, word.last),
        });
    }

    void set_sram_byte(
        std::size_t column,
        std::size_t row,
        std::size_t byte_offset,
        Data value)
    {
        mem_.set_sram_byte(column, row, byte_offset, value);
    }

    void set_sram_lane_byte(
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane,
        Data value)
    {
        mem_.set_sram_lane_byte(column, tile, row, lane, value);
    }

    const StreamSlot& east_register(
        std::size_t tile,
        std::size_t lane,
        std::size_t register_file,
        std::size_t stream) const
    {
        return streams_.cell(register_file, tile, lane, StreamId::East(stream));
    }

    const StreamSlot& west_register(
        std::size_t tile,
        std::size_t lane,
        std::size_t register_file,
        std::size_t stream) const
    {
        return streams_.cell(register_file, tile, lane, StreamId::West(stream));
    }

    StreamSlot consume_east_register(
        std::size_t tile,
        std::size_t lane,
        std::size_t register_file,
        std::size_t stream)
    {
        const auto id = StreamId::East(stream);
        const auto cell = streams_.cell(register_file, tile, lane, id);
        if (cell.valid) {
            pending_consumptions_.push_back(
                PendingConsumption {register_file, tile, lane, id});
        }
        return cell;
    }

    Data sram_byte(std::size_t column, std::size_t row, std::size_t byte_offset) const
    {
        return mem_.sram_byte(column, row, byte_offset);
    }

    Data sram_lane_byte(
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane) const
    {
        return mem_.sram_lane_byte(column, tile, row, lane);
    }

    const InstructionSlot& instruction_at(std::size_t column, std::size_t tile) const
    {
        return mem_.instruction_at(column, tile);
    }

    void tick()
    {
        tick_impl(nullptr, nullptr, std::nullopt);
    }

    void tick(std::ostream& os, std::optional<std::size_t> log_tile = std::nullopt)
    {
        tick_impl(nullptr, &os, log_tile);
    }

    void tick(SxmSlice& sxm)
    {
        tick_impl(&sxm, nullptr, std::nullopt);
    }

    void tick(SxmSlice& sxm, std::ostream& os, std::optional<std::size_t> log_tile = std::nullopt)
    {
        tick_impl(&sxm, &os, log_tile);
    }

private:
    void tick_impl(SxmSlice* sxm, std::ostream* os, std::optional<std::size_t> log_tile)
    {
        streams_.begin_cycle();

        // Legacy MXM/VXM bridges run before TileArrayModel::tick().  Defer
        // their consume requests until the shared fabric cycle is open.
        for (const auto& consumption : pending_consumptions_) {
            streams_.consume(
                consumption.column,
                consumption.tile,
                consumption.lane,
                consumption.stream,
                "legacy TileArray consumer");
        }
        pending_consumptions_.clear();

        // Functional slices read current SR state, consume operands and stage
        // results into next state.  No functional slice can see another
        // slice's same-cycle result through an SR boundary.
        mem_.evaluate(streams_);
        if (sxm != nullptr) {
            sxm->evaluate(streams_);
        }

        // Unconsumed streams move one physical SR hop in their direction.
        streams_.stage_linear_links();

        // Edge producers (legacy test hooks, later VXM/C2C ports) also write
        // next state and therefore participate in collision detection.
        for (const auto& input : pending_inputs_) {
            streams_.stage_write(
                input.column,
                input.tile,
                input.lane,
                input.stream,
                input.cell,
                "external stream input");
        }

        streams_.commit_cycle();
        pending_inputs_.clear();

        if (os != nullptr) {
            mem_.log_cycle(*os, log_tile);
            log_streams(*os, log_tile);
        }
        ++cycle_;
    }

    struct PendingInput {
        std::size_t column{0};
        std::size_t tile{0};
        std::size_t lane{0};
        StreamId stream{StreamId::East(0)};
        StreamCell cell{};
    };

    struct PendingConsumption {
        std::size_t column{0};
        std::size_t tile{0};
        std::size_t lane{0};
        StreamId stream{StreamId::East(0)};
    };

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

    bool collect_stream_bytes(
        std::size_t tile,
        std::size_t reg_column,
        StreamId stream,
        StreamPayloadSegment16& bytes) const
    {
        bool any = false;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto& slot = streams_.cell(reg_column, tile, lane, stream);
            bytes[lane] = slot.valid ? slot.data : 0;
            any = any || slot.valid;
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
            for (std::size_t reg = 0; reg < streams_.column_count(); ++reg) {
                os << "      sreg " << reg << ":";
                bool any = false;

                for (std::size_t stream = 0; stream < hw::kEastStreams; ++stream) {
                    StreamPayloadSegment16 bytes{};
                    if (collect_stream_bytes(tile, reg, StreamId::East(stream), bytes)) {
                        any = true;
                        os << " E" << stream << "=0x";
                        print_hex_bytes(os, bytes);
                    }
                }

                for (std::size_t stream = 0; stream < hw::kWestStreams; ++stream) {
                    StreamPayloadSegment16 bytes{};
                    if (collect_stream_bytes(tile, reg, StreamId::West(stream), bytes)) {
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

    MemArrayModel mem_;
    StreamRegisterFabric streams_;
    std::vector<PendingInput> pending_inputs_{};
    std::vector<PendingConsumption> pending_consumptions_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
