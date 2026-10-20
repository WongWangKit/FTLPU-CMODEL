#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm/sr_datapath.hpp"
#include "ftlpu/vxm/sr_instruction.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

// Hardware-facing VXM slice.  The sixteen ICU channels configure sixteen
// independent VALU columns.  Data state is held exclusively by the ordinary
// SR fabric in VxmSrDatapath; there is no token, chain, or private VXM SR.
class VxmSrSlice {
public:
    static constexpr std::size_t kTileCount = hw::kTileRows;
    static constexpr std::size_t kAluQueues = hw::kVxmAluCount;
    static constexpr std::size_t kConfigFifoDepth = 3;
    using CompactInstruction = VxmSrCompactInstruction;
    using InstructionSlot = std::optional<CompactInstruction>;
    using ExecutionMask = VxmSrDatapath::ExecutionMask;

    VxmSrSlice() : datapath_(std::make_unique<VxmSrDatapath>()) {}

    void reset()
    {
        for (auto& queue : south_queues_) queue.clear();
        for (auto& rows : instruction_rows_)
            for (auto& slot : rows) slot.reset();
        for (auto& tile : controls_)
            for (auto& control : tile) control.reset();
        datapath_->reset();
        last_execution_ = {};
        cycle_ = 0;
        prepared_ = false;
        cycle_open_ = false;
    }

    void issue_south(std::size_t column, CompactInstruction instruction)
    {
        check_column(column);
        south_queues_[column].push_back(std::move(instruction));
    }

    void configure_special_lut(VxmSpecialAluOpcode opcode,
        VxmLutConfig config, const std::vector<VxmLutEntry>& entries)
    {
        datapath_->configure_special_lut(opcode, config, entries);
    }

    void prepare_cycle()
    {
        if (prepared_) return;
        for (std::size_t column = 0; column < kAluQueues; ++column) {
            if (instruction_rows_[column][0]
                || south_queues_[column].empty())
                continue;
            instruction_rows_[column][0] = south_queues_[column].front();
            south_queues_[column].pop_front();
        }
        prepared_ = true;
    }

    void begin_cycle()
    {
        if (cycle_open_)
            throw std::logic_error("VXM SR slice cycle is already open");
        prepare_cycle();
        datapath_->begin_cycle();
        cycle_open_ = true;
    }

    void stage_ingress(std::size_t tile, StreamId stream,
        const StreamTileSegment& values,
        const char* producer = "VXM boundary input")
    {
        require_open_cycle();
        check_tile(tile);
        datapath_->fabric().registers().stage_segment(
            VxmStreamFabric::ingress_column(stream.direction()),
            tile, stream, values, producer);
    }

    bool egress_valid(std::size_t tile, StreamId stream) const
    {
        check_tile(tile);
        return datapath_->fabric().registers().segment_valid(
            VxmStreamFabric::egress_column(stream.direction()),
            tile, stream);
    }

    StreamTileSegment egress_segment(
        std::size_t tile, StreamId stream) const
    {
        check_tile(tile);
        return datapath_->fabric().registers().segment(
            VxmStreamFabric::egress_column(stream.direction()),
            tile, stream);
    }

    void consume_egress(std::size_t tile, StreamId stream,
        const char* consumer = "VXM boundary output")
    {
        require_open_cycle();
        check_tile(tile);
        datapath_->fabric().registers().consume_segment(
            VxmStreamFabric::egress_column(stream.direction()),
            tile, stream, consumer);
    }

    void tick(std::ostream* os = nullptr,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
        if (!cycle_open_) begin_cycle();
        if (log_tile) check_tile(*log_tile);

        auto issued = VxmSrDatapath::Instructions {};
        for (std::size_t tile = 0; tile < kTileCount; ++tile)
            for (std::size_t column = 0; column < kAluQueues; ++column)
                if (controls_[tile][column].current)
                {
                    const auto& active =
                        *controls_[tile][column].current;
                    auto instruction = active.instruction;
                    if (instruction.accumulator_reset_first
                        && active.remaining
                            != active.instruction.repeat_count)
                        instruction.accumulator_reset = false;
                    if (instruction.accumulator_emit_last
                        && active.remaining != 1)
                        instruction.accumulator_emit = false;
                    issued[tile][column] = std::move(instruction);
                }

        last_execution_ = datapath_->execute(issued);
        for (std::size_t tile = 0; tile < kTileCount; ++tile) {
            for (std::size_t column = 0; column < kAluQueues; ++column) {
                auto& control = controls_[tile][column];
                if (!control.current) continue;
                // A decoded column configuration may arrive before its
                // explicit SR operands.  Current Config is a hardware
                // resident control register: hold it until ready/valid fires
                // instead of treating operand latency as a schedule error.
                if (!last_execution_[tile][column]) continue;
                if (os && (!log_tile || *log_tile == tile))
                    *os << "  tile " << tile << " C" << column
                        << " launch\n";
                if (--control.current->remaining == 0) {
                    control.current.reset();
                    control.promote();
                }
            }
        }

        // A southbound packet consumes one local decode cycle.  It becomes
        // Current Config only after this cycle's datapath launch decision.
        for (std::size_t tile = 0; tile < kTileCount; ++tile) {
            for (std::size_t column = 0; column < kAluQueues; ++column) {
                if (!instruction_rows_[column][tile]) continue;
                controls_[tile][column].enqueue(
                    VxmSrCompactInstructionCodec::decode(
                        *instruction_rows_[column][tile]));
                if (os && (!log_tile || *log_tile == tile)) {
                    *os << "  tile " << tile << " C" << column
                        << " config\n";
                }
            }
        }

        datapath_->finish_cycle();
        advance_instruction_rows();
        prepared_ = false;
        cycle_open_ = false;
        ++cycle_;
    }

    std::size_t cycle() const noexcept { return cycle_; }
    const ExecutionMask& last_execution() const noexcept
    {
        return last_execution_;
    }
    const InstructionSlot& instruction_at(
        std::size_t column, std::size_t tile) const
    {
        check_column(column);
        check_tile(tile);
        return instruction_rows_[column][tile];
    }
    std::size_t config_entry_count(
        std::size_t tile, std::size_t column) const
    {
        check_tile(tile);
        check_column(column);
        const auto& control = controls_[tile][column];
        return (control.current ? 1U : 0U) + control.next.size();
    }
    bool idle() const
    {
        for (const auto& queue : south_queues_)
            if (!queue.empty()) return false;
        for (const auto& rows : instruction_rows_)
            for (const auto& slot : rows)
                if (slot) return false;
        for (const auto& tile : controls_)
            for (const auto& control : tile)
                if (control.current || !control.next.empty()) return false;
        return datapath_->idle();
    }

    VxmSrDatapath& datapath() noexcept { return *datapath_; }
    const VxmSrDatapath& datapath() const noexcept { return *datapath_; }

private:
    struct ActiveConfig {
        VxmSrInstruction instruction {};
        std::size_t remaining {1};
    };
    struct ColumnControl {
        std::optional<ActiveConfig> current {};
        std::deque<ActiveConfig> next {};

        void reset()
        {
            current.reset();
            next.clear();
        }
        void enqueue(VxmSrInstruction instruction)
        {
            ActiveConfig config {std::move(instruction), 1};
            config.remaining = config.instruction.repeat_count;
            if (!current) {
                current = std::move(config);
                return;
            }
            if (next.size() + 1 >= kConfigFifoDepth)
                throw std::overflow_error(
                    "VXM column Current/Next Config FIFO overflow");
            next.push_back(std::move(config));
        }
        void promote()
        {
            if (next.empty()) return;
            current = std::move(next.front());
            next.pop_front();
        }
    };

    static void check_tile(std::size_t tile)
    {
        if (tile >= kTileCount)
            throw std::out_of_range("VXM tile is outside the slice");
    }
    static void check_column(std::size_t column)
    {
        if (column >= kAluQueues)
            throw std::out_of_range("VXM column exceeds the configured physical column count");
    }
    void require_open_cycle() const
    {
        if (!cycle_open_)
            throw std::logic_error(
                "VXM boundary transfer requires begin_cycle()");
    }
    void advance_instruction_rows()
    {
        for (auto& rows : instruction_rows_) {
            for (std::size_t tile = kTileCount - 1; tile > 0; --tile)
                rows[tile] = rows[tile - 1];
            rows[0].reset();
        }
    }

    std::array<std::deque<CompactInstruction>, kAluQueues> south_queues_ {};
    std::array<std::array<InstructionSlot, kTileCount>, kAluQueues>
        instruction_rows_ {};
    std::array<std::array<ColumnControl, kAluQueues>, kTileCount> controls_ {};
    std::unique_ptr<VxmSrDatapath> datapath_;
    ExecutionMask last_execution_ {};
    std::size_t cycle_ {0};
    bool prepared_ {false};
    bool cycle_open_ {false};
};

} // namespace ftlpu
