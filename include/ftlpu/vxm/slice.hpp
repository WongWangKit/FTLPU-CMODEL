#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm/superlane.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace ftlpu {

class VxmSlice {
public:
    static constexpr std::size_t kTileCount = hw::kTileRows;
    static constexpr std::size_t kRows = hw::kTileRows * hw::kLanesPerTile;
    static constexpr std::size_t kAluQueues = VxmLane::kAluCount;

    using Superlane = VxmSuperlane;
    using AluInstruction = VxmLaneAluInstruction;
    using Int32Vector = Superlane::Int32Vector;
    using Int8Vector = Superlane::Int8Vector;
    using StreamMatrix = Superlane::StreamMatrix;
    using InstructionSlot = std::optional<AluInstruction>;
    using OutputSlot = std::optional<Superlane::Output>;
    using RequiredStreams = std::array<bool, hw::kStreams>;

    struct InputSlot {
        StreamMatrix streams{};
    };

    void reset()
    {
        for (auto& queue : instruction_queues_) {
            queue.clear();
        }
        for (auto& alu_rows : instruction_rows_) {
            for (auto& slot : alu_rows) {
                slot.reset();
            }
        }
        for (auto& superlane : superlanes_) {
            superlane.reset();
        }
        for (auto& input : input_slots_) {
            input.reset();
        }
        for (auto& output : output_slots_) {
            output.reset();
        }
        for (auto& outputs : output_slots_multi_) {
            outputs.clear();
        }
        for (auto& required : required_streams_) {
            required.reset();
        }
        cycle_ = 0;
    }

    void issue_south(std::size_t alu, AluInstruction instruction)
    {
        check_alu(alu);
        instruction_queues_[alu].push_back(instruction);
    }

    void set_swiglu_input(std::size_t tile, const Int32Vector& gates, const Int32Vector& ups)
    {
        check_tile(tile);
        if (input_slots_[tile].has_value()) {
            throw std::logic_error("VXM tile input is already occupied");
        }
        auto streams = StreamMatrix {};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto gate_bytes = VxmLane::pack_int32(gates[lane]);
            const auto up_bytes = VxmLane::pack_int32(ups[lane]);
            for (std::size_t byte = 0; byte < 4; ++byte) {
                streams[lane][byte] = gate_bytes[byte];
                streams[lane][4 + byte] = up_bytes[byte];
            }
        }
        input_slots_[tile] = InputSlot {streams};
    }

    void set_stream_inputs(std::size_t tile, const StreamMatrix& streams)
    {
        check_tile(tile);
        if (input_slots_[tile].has_value()) {
            throw std::logic_error("VXM tile input is already occupied");
        }
        input_slots_[tile] = InputSlot {streams};
    }

    void tick(std::ostream* os = nullptr, std::optional<std::size_t> log_tile = std::nullopt)
    {
        if (log_tile.has_value()) {
            check_tile(*log_tile);
        }
        prepare_cycle();
        if (os != nullptr) {
            *os << "vxm_slice cycle " << cycle_ << '\n';
            log_status(*os, log_tile);
        }

        execute_instructions(os, log_tile);
        tick_superlanes(os, log_tile);
        advance_instructions();
        for (auto& input : input_slots_) {
            input.reset();
        }
        prepared_ = false;
        ++cycle_;
    }

    void prepare_cycle()
    {
        if (prepared_) {
            return;
        }
        dispatch_instruction_queues();
        refresh_required_streams();
        prepared_ = true;
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    const InstructionSlot& instruction_at(std::size_t alu, std::size_t tile) const
    {
        check_alu(alu);
        check_tile(tile);
        return instruction_rows_[alu][tile];
    }

    const OutputSlot& output_at(std::size_t tile) const
    {
        check_tile(tile);
        return output_slots_[tile];
    }

    const std::vector<Superlane::Output>& outputs_at(std::size_t tile) const
    {
        check_tile(tile);
        return output_slots_multi_[tile];
    }

    const std::optional<RequiredStreams>& required_streams_at(std::size_t tile) const
    {
        check_tile(tile);
        return required_streams_[tile];
    }

    const Superlane& superlane(std::size_t tile) const
    {
        check_tile(tile);
        return superlanes_[tile];
    }

private:
    static void check_tile(std::size_t tile)
    {
        if (tile >= kTileCount) {
            throw std::out_of_range("VXM slice tile is outside the 20-row slice");
        }
    }

    static void check_alu(std::size_t alu)
    {
        if (alu >= kAluQueues) {
            throw std::out_of_range("VXM ALU queue is outside the 16-ALU lane");
        }
    }

    void dispatch_instruction_queues()
    {
        for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
            if (instruction_rows_[alu][0].has_value() || instruction_queues_[alu].empty()) {
                continue;
            }
            instruction_rows_[alu][0] = instruction_queues_[alu].front();
            instruction_queues_[alu].pop_front();
        }
    }

    void log_status(std::ostream& os, std::optional<std::size_t> log_tile) const
    {
        std::size_t queued = 0;
        for (const auto& queue : instruction_queues_) {
            queued += queue.size();
        }

        std::size_t active = 0;
        for (const auto& alu_rows : instruction_rows_) {
            for (const auto& slot : alu_rows) {
                if (slot.has_value()) {
                    ++active;
                }
            }
        }

        std::size_t inputs = 0;
        for (const auto& slot : input_slots_) {
            if (slot.has_value()) {
                ++inputs;
            }
        }

        os << "  status:";
        if (log_tile.has_value()) {
            os << " log_tile=" << *log_tile;
        }
        os
           << " queued=" << queued
           << " active_instr=" << active
           << " input_tiles=" << inputs << '\n';
    }

    void execute_instructions(std::ostream* os, std::optional<std::size_t> log_tile)
    {
        bool any = false;
        bool any_logged = false;
        for (std::size_t tile = 0; tile < kTileCount; ++tile) {
            for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
                const auto& instruction = instruction_rows_[alu][tile];
                if (!instruction.has_value()) {
                    continue;
                }

                any = true;
                superlanes_[tile].enqueue_instruction(alu, *instruction);
                if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                    any_logged = true;
                    *os << "  tile " << tile
                        << " alu" << alu
                        << " " << VxmLane::opcode_name(instruction->opcode)
                        << '\n';
                }
            }
        }

        if ((!any || (log_tile.has_value() && !any_logged)) && os != nullptr) {
            *os << "  control idle\n";
        }
    }

    void tick_superlanes(std::ostream* os, std::optional<std::size_t> log_tile)
    {
        for (std::size_t tile = 0; tile < kTileCount; ++tile) {
            output_slots_[tile].reset();
            output_slots_multi_[tile].clear();
            if (input_slots_[tile].has_value()) {
                superlanes_[tile].set_stream_inputs(input_slots_[tile]->streams);
                if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                    *os << "  tile " << tile << " input\n";
                }
            }

            superlanes_[tile].tick();
            output_slots_multi_[tile] = superlanes_[tile].outputs();
            if (!output_slots_multi_[tile].empty()) {
                output_slots_[tile] = output_slots_multi_[tile].front();
                if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                    for (const auto& output : output_slots_multi_[tile]) {
                        *os << "  tile " << tile << " output s" << output.stream;
                        for (const auto value : output.values) {
                            *os << ' ' << static_cast<int>(value);
                        }
                        *os << '\n';
                    }
                }
            }
        }
    }

    void advance_instructions()
    {
        for (auto& alu_rows : instruction_rows_) {
            for (std::size_t tile = kTileCount - 1; tile > 0; --tile) {
                alu_rows[tile] = alu_rows[tile - 1];
            }
            alu_rows[0].reset();
        }
    }

    static void mark_operand_streams(RequiredStreams& required, const VxmLaneOperand& operand)
    {
        if (operand.kind != VxmLaneOperandKind::StreamInt32) {
            return;
        }
        if (operand.index + 3 >= hw::kStreams) {
            throw std::out_of_range("VXM int32 stream operand is outside the 64-stream set");
        }
        for (std::size_t byte = 0; byte < 4; ++byte) {
            required[operand.index + byte] = true;
        }
    }

    static bool operand_uses_stream(const VxmLaneOperand& operand)
    {
        return operand.kind == VxmLaneOperandKind::StreamInt32;
    }

    static bool instruction_uses_stream(const AluInstruction& instruction)
    {
        return operand_uses_stream(instruction.lhs) || operand_uses_stream(instruction.rhs);
    }

    void refresh_required_streams()
    {
        for (auto& required : required_streams_) {
            required.reset();
        }

        for (std::size_t tile = 0; tile < kTileCount; ++tile) {
            auto required = RequiredStreams {};
            bool any = false;
            for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
                const auto& instruction = instruction_rows_[alu][tile];
                if (!instruction.has_value() || !instruction_uses_stream(*instruction)) {
                    continue;
                }
                mark_operand_streams(required, instruction->lhs);
                mark_operand_streams(required, instruction->rhs);
                any = true;
            }
            if (any) {
                required_streams_[tile] = required;
            }
        }
    }

    std::array<std::deque<AluInstruction>, kAluQueues> instruction_queues_{};
    std::array<std::array<InstructionSlot, kTileCount>, kAluQueues> instruction_rows_{};
    std::array<Superlane, kTileCount> superlanes_{};
    std::array<std::optional<InputSlot>, kTileCount> input_slots_{};
    std::array<OutputSlot, kTileCount> output_slots_{};
    std::array<std::vector<Superlane::Output>, kTileCount> output_slots_multi_{};
    std::array<std::optional<RequiredStreams>, kTileCount> required_streams_{};
    std::size_t cycle_{0};
    bool prepared_{false};
};

} // namespace ftlpu
