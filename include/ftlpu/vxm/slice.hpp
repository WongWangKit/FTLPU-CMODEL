#pragma once

#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm/input_buffer.hpp"
#include "ftlpu/vxm/superlane.hpp"

#include <array>
#include <cstddef>
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
    // Eight Slice instruction channels feed eight shared control queues.
    // Superlane-local decode mirrors logical Ci onto physical C(i+8).
    static constexpr std::size_t kAluQueues =
        VxmSuperlaneInstructionControl::kStageCount;

    using Superlane = VxmSuperlane;
    using CompactInstruction = VxmCompactInstruction;
    using StreamMatrix = Superlane::StreamMatrix;
    using InstructionSlot = std::optional<CompactInstruction>;
    using OutputSlot = std::optional<Superlane::Output>;
    using RequiredStreams = std::array<bool, hw::kStreams>;
    using InputBuffer = VxmInputBuffer;

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
        for (auto& buffer : input_buffers_) {
            buffer.reset();
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
        input_group_sources_.fill(Hemisphere::East);
        output_block_destinations_.fill(Hemisphere::East);
        cycle_ = 0;
    }

    void issue_south(
        std::size_t alu, CompactInstruction instruction)
    {
        check_alu(alu);
        instruction_queues_[alu].push_back(instruction);
    }

    void set_chain_depth(VxmChainDepth depth)
    {
        for (auto& superlane : superlanes_) {
            superlane.set_chain_depth(depth);
        }
    }

    void set_chain_depth(std::size_t tile, VxmChainDepth depth)
    {
        check_tile(tile);
        superlanes_[tile].set_chain_depth(depth);
    }

    void request_chain_depth_transition(
        std::size_t tile, VxmChainDepth depth,
        bool feedback_required = true)
    {
        check_tile(tile);
        superlanes_[tile].request_chain_depth_transition(
            depth, feedback_required);
    }

    void configure_special_lut(VxmSpecialAluOpcode opcode, VxmLutConfig config,
                               const std::vector<VxmLutEntry>& entries)
    {
        for (auto& superlane : superlanes_) {
            superlane.configure_special_lut(opcode, config, entries);
        }
    }

    void set_stream_inputs(std::size_t tile, const StreamMatrix& streams)
    {
        check_tile(tile);
        input_buffers_[tile].load_complete(streams);
    }

    void configure_input_buffer(
        std::size_t tile,
        std::size_t expected_group_count)
    {
        check_tile(tile);
        input_buffers_[tile].configure(
            expected_group_count);
    }

    void capture_stream_group(
        std::size_t tile,
        std::size_t group,
        const InputBuffer::GroupVector& values)
    {
        check_tile(tile);
        input_buffers_[tile].capture_group(group, values);
    }

    // Compiler-visible fixed mux configuration. The logical group position
    // inside every lane remains fixed; this bit selects which hemisphere's
    // same-numbered SR group drives that position.
    void configure_input_group_source(
        std::size_t group,
        Hemisphere source)
    {
        if (group >= VxmLane::kStreamGroupCount) {
            throw std::out_of_range(
                "VXM input group source is outside the fixed group set");
        }
        input_group_sources_[group] = source;
    }

    Hemisphere input_group_source(std::size_t group) const
    {
        if (group >= VxmLane::kStreamGroupCount) {
            throw std::out_of_range(
                "VXM input group source is outside the fixed group set");
        }
        return input_group_sources_[group];
    }

    void configure_output_block_destination(
        std::size_t block,
        Hemisphere destination)
    {
        if (block >= VxmLane::kBlockCount) {
            throw std::out_of_range(
                "VXM output block destination is outside the fixed block set");
        }
        output_block_destinations_[block] = destination;
    }

    Hemisphere output_block_destination(std::size_t block) const
    {
        if (block >= VxmLane::kBlockCount) {
            throw std::out_of_range(
                "VXM output block destination is outside the fixed block set");
        }
        return output_block_destinations_[block];
    }

    Hemisphere output_stream_destination(std::size_t stream) const
    {
        return output_block_destination(
            stream / VxmLane::kStreamGroupBytes);
    }

    InputBuffer& input_buffer(std::size_t tile)
    {
        check_tile(tile);
        return input_buffers_[tile];
    }

    const InputBuffer& input_buffer(std::size_t tile) const
    {
        check_tile(tile);
        return input_buffers_[tile];
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

    Superlane& superlane(std::size_t tile)
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
            throw std::out_of_range(
                "VXM shared ALU control queue is outside 0..7");
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
        for (const auto& buffer : input_buffers_) {
            if (!buffer.empty()) {
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
                superlanes_[tile].enqueue_compact_instruction(
                    alu, *instruction);
                if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                    any_logged = true;
                    const auto decoded =
                        VxmCompactInstructionCodec::decode(
                            alu, *instruction);
                    *os << "  tile " << tile
                        << " alu" << alu
                        << " " << VxmLane::operation_name(
                            decoded.instruction.operation)
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

            auto stream_consumers =
                std::array<bool, kAluQueues>{};
            bool will_consume_bundle = false;
            for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
                if (!superlanes_[tile]
                         .configuration_will_issue(alu)) {
                    continue;
                }
                const auto instruction =
                    superlanes_[tile].next_instruction(alu);
                if (!instruction
                    || !instruction_uses_stream(*instruction)) {
                    continue;
                }
                stream_consumers[alu] = true;
                will_consume_bundle = true;
            }

            auto supplied_bundle = false;
            if (will_consume_bundle
                && !input_buffers_[tile].ready()) {
                throw std::logic_error(
                    "VXM compiler/static-schedule error: input Bundle is not ready on its scheduled issue cycle at tile "
                    + std::to_string(tile));
            }
            if (will_consume_bundle) {
                superlanes_[tile].set_stream_inputs(
                    input_buffers_[tile].bundle());
                supplied_bundle = true;
                if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                    *os << "  tile " << tile
                        << " input Bundle groups="
                        << input_buffers_[tile].fill_count()
                        << '\n';
                }
            }

            const auto executed = superlanes_[tile].tick();
            if (supplied_bundle) {
                for (std::size_t alu = 0;
                     alu < kAluQueues;
                     ++alu) {
                    if (!stream_consumers[alu]) {
                        continue;
                    }
                    const auto consumed =
                        executed[alu]
                        || executed[
                            alu
                            + VxmSuperlaneInstructionControl::
                                kMirroredStageOffset];
                    if (!consumed) {
                        throw std::logic_error(
                            "VXM compiler/static-schedule error: a scheduled stream consumer did not execute");
                    }
                }
                // The execution mask is an assertion of the compiler's
                // schedule, not a ready/valid handshake or backpressure path.
                input_buffers_[tile].release_after_issue();
            }
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

    void mark_operand_streams(RequiredStreams& required,
                              const VxmLaneOperand& operand,
                              std::size_t alu, bool rhs_port) const
    {
        if (operand.kind != VxmLaneOperandKind::StreamFloat16) {
            return;
        }
        const auto group = VxmLane::fixed_input_group_for_stage(alu, rhs_port);
        const auto base = group * VxmLane::kStreamGroupBytes
            + (input_group_sources_[group] == Hemisphere::West
                ? hw::kStreamsPerDirection : std::size_t{0});
        for (std::size_t byte = 0; byte < VxmLane::kStreamGroupBytes; ++byte) {
            required[base + byte] = true;
        }
    }

    static bool operand_uses_stream(const VxmLaneOperand& operand)
    {
        return operand.kind == VxmLaneOperandKind::StreamFloat16;
    }

    static bool instruction_uses_stream(
        const VxmLaneAluInstruction& instruction)
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
                // A held Current Config takes precedence over a new packet
                // merely passing this tile toward the shared Superlane queue.
                auto instruction = superlanes_[tile].next_instruction(alu);
                if (!instruction && instruction_rows_[alu][tile]) {
                    instruction =
                        VxmCompactInstructionCodec::decode(
                            alu, *instruction_rows_[alu][tile])
                            .instruction;
                }
                if (!instruction.has_value() || !instruction_uses_stream(*instruction)) {
                    continue;
                }
                mark_operand_streams(required, instruction->lhs, alu, false);
                mark_operand_streams(required, instruction->rhs, alu, true);
                const auto mirrored =
                    alu + VxmSuperlaneInstructionControl::kMirroredStageOffset;
                mark_operand_streams(
                    required, instruction->lhs, mirrored, false);
                mark_operand_streams(
                    required, instruction->rhs, mirrored, true);
                any = true;
            }
            if (any) {
                required_streams_[tile] = required;
            }
        }
    }

    std::array<std::deque<CompactInstruction>, kAluQueues>
        instruction_queues_{};
    std::array<std::array<InstructionSlot, kTileCount>, kAluQueues> instruction_rows_{};
    // A complete Slice contains kTileCount configured Superlanes and every Lane owns real
    // internal ALU pipeline state. Keep the fixed architectural count while
    // placing the large C-model objects on the host heap instead of its stack.
    std::vector<Superlane> superlanes_{kTileCount};
    std::array<InputBuffer, kTileCount> input_buffers_{};
    std::array<OutputSlot, kTileCount> output_slots_{};
    std::array<std::vector<Superlane::Output>, kTileCount> output_slots_multi_{};
    std::array<std::optional<RequiredStreams>, kTileCount> required_streams_{};
    std::array<Hemisphere, VxmLane::kStreamGroupCount>
        input_group_sources_{};
    std::array<Hemisphere, VxmLane::kBlockCount>
        output_block_destinations_{};
    std::size_t cycle_{0};
    bool prepared_{false};
};

} // namespace ftlpu
