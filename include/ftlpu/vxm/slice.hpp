#pragma once

#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/timing.hpp"
#include "ftlpu/vxm/input_buffer.hpp"
#include "ftlpu/vxm/superlane.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace ftlpu {

class VxmSlice : public timing::ClockedModule {
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

    VxmSlice()
        : current_(std::make_unique<State>())
    {}

    VxmSlice(const VxmSlice&) = delete;
    VxmSlice& operator=(const VxmSlice&) = delete;
    VxmSlice(VxmSlice&&) = delete;
    VxmSlice& operator=(VxmSlice&&) = delete;

    void reset(std::size_t cycle = 0)
    {
        current_ = std::make_unique<State>();
        current_->cycle = cycle;
        for(auto& lane:current_->superlanes) lane.reset(cycle);
        next_.reset();
        for (auto& queue : pending_issues_) queue.clear();
        low_started_ = false;
        low_complete_ = false;
    }

    // Epoch release must wait for queued configs, lane pipelines, and any
    // captured input bundle, including data that has not launched yet.
    bool idle() const
    {
        for (const auto& queue : pending_issues_)
            if (!queue.empty()) return false;
        const auto& current = visible_state();
        for (const auto& queue : current.instruction_queues)
            if (!queue.empty()) return false;
        for (const auto& rows : current.instruction_rows)
            for (const auto& slot : rows)
                if (slot) return false;
        for (std::size_t tile = 0; tile < kTileCount; ++tile)
            if (!current.input_buffers[tile].empty()
                || !current.superlanes[tile].idle())
                return false;
        return true;
    }

    void issue_south(
        std::size_t alu, CompactInstruction instruction)
    {
        check_alu(alu);
        if (low_started_) {
            throw std::logic_error(
                "VXM instruction issue must precede prepare_cycle in LOW");
        }
        pending_issues_[alu].push_back(std::move(instruction));
    }

    void set_chain_depth(VxmChainDepth depth)
    {
        for (auto& superlane : writable_state().superlanes) {
            superlane.set_chain_depth(depth);
        }
    }

    void set_chain_depth(std::size_t tile, VxmChainDepth depth)
    {
        check_tile(tile);
        writable_state().superlanes[tile].set_chain_depth(depth);
    }

    void request_chain_depth_transition(
        std::size_t tile, VxmChainDepth depth,
        bool feedback_required = true)
    {
        check_tile(tile);
        writable_state().superlanes[tile].request_chain_depth_transition(
            depth, feedback_required);
    }

    void configure_special_lut(VxmSpecialAluOpcode opcode, VxmLutConfig config,
                               const std::vector<VxmLutEntry>& entries)
    {
        for (auto& superlane : writable_state().superlanes) {
            superlane.configure_special_lut(opcode, config, entries);
        }
    }

    void set_stream_inputs(std::size_t tile, const StreamMatrix& streams)
    {
        check_tile(tile);
        writable_state().input_buffers[tile].load_complete(streams);
    }

    void configure_input_buffer(
        std::size_t tile,
        std::size_t expected_group_count)
    {
        check_tile(tile);
        writable_state().input_buffers[tile].configure(
            expected_group_count);
    }

    void capture_stream_group(
        std::size_t tile,
        std::size_t group,
        const InputBuffer::GroupVector& values)
    {
        check_tile(tile);
        writable_state().input_buffers[tile].capture_group(group, values);
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
        writable_state().input_group_sources[group] = source;
    }

    Hemisphere input_group_source(std::size_t group) const
    {
        if (group >= VxmLane::kStreamGroupCount) {
            throw std::out_of_range(
                "VXM input group source is outside the fixed group set");
        }
        return visible_state().input_group_sources[group];
    }

    void configure_output_block_destination(
        std::size_t block,
        Hemisphere destination)
    {
        if (block >= VxmLane::kBlockCount) {
            throw std::out_of_range(
                "VXM output block destination is outside the fixed block set");
        }
        writable_state().output_block_destinations[block] = destination;
    }

    Hemisphere output_block_destination(std::size_t block) const
    {
        if (block >= VxmLane::kBlockCount) {
            throw std::out_of_range(
                "VXM output block destination is outside the fixed block set");
        }
        return visible_state().output_block_destinations[block];
    }

    Hemisphere output_stream_destination(std::size_t stream) const
    {
        return output_block_destination(
            stream / VxmLane::kStreamGroupBytes);
    }

    InputBuffer& input_buffer(std::size_t tile)
    {
        check_tile(tile);
        return writable_state().input_buffers[tile];
    }

    const InputBuffer& input_buffer(std::size_t tile) const
    {
        check_tile(tile);
        return visible_state().input_buffers[tile];
    }

    // Compatibility wrapper for direct unit-level use. Integrated systems call
    // prepare_cycle()/evaluate() in LOW and let Clock perform the commit.
    void tick(std::ostream* os = nullptr,
              std::optional<std::size_t> log_tile = std::nullopt)
    {
        prepare_cycle();
        evaluate(os, log_tile);
        validate_posedge();
        posedge();
    }

    void low() override
    {
        evaluate();
    }

    void evaluate(std::ostream* os = nullptr,
                  std::optional<std::size_t> log_tile = std::nullopt)
    {
        if (log_tile.has_value()) {
            check_tile(*log_tile);
        }
        prepare_cycle();
        if (low_complete_) {
            throw std::logic_error("VXM LOW evaluated more than once");
        }
        auto& next = next_state();
        if (os != nullptr) {
            *os << "vxm_slice cycle " << next.cycle << '\n';
            log_status(*os, log_tile);
        }

        execute_instructions(os, log_tile);
        tick_superlanes(os, log_tile);
        advance_instructions();
        ++next.cycle;
        low_complete_ = true;
    }

    void prepare_cycle()
    {
        if (low_started_) {
            return;
        }
        next_ = std::make_unique<State>(*current_);
        for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
            auto& destination = next_->instruction_queues[alu];
            auto& pending = pending_issues_[alu];
            while (!pending.empty()) {
                destination.push_back(std::move(pending.front()));
                pending.pop_front();
            }
        }
        low_started_ = true;
        low_complete_ = false;
        dispatch_instruction_queues();
        refresh_required_streams();
    }

    void validate_posedge() const override
    {
        if (!low_started_ || !low_complete_ || !next_) {
            throw std::logic_error(
                "VXM posedge requires one complete LOW evaluation");
        }
    }

    void posedge() override
    {
        validate_posedge();
        current_.swap(next_);
        next_.reset();
        low_started_ = false;
        low_complete_ = false;
    }

    void high() override {}

    std::size_t cycle() const
    {
        return current_->cycle;
    }

    const InstructionSlot& instruction_at(std::size_t alu, std::size_t tile) const
    {
        check_alu(alu);
        check_tile(tile);
        return current_->instruction_rows[alu][tile];
    }

    const OutputSlot& output_at(std::size_t tile) const
    {
        check_tile(tile);
        return visible_state().output_slots[tile];
    }

    const std::vector<Superlane::Output>& outputs_at(std::size_t tile) const
    {
        check_tile(tile);
        return visible_state().output_slots_multi[tile];
    }

    const std::optional<RequiredStreams>& required_streams_at(std::size_t tile) const
    {
        check_tile(tile);
        return visible_state().required_streams[tile];
    }

    const Superlane& superlane(std::size_t tile) const
    {
        check_tile(tile);
        return current_->superlanes[tile];
    }

    Superlane& superlane(std::size_t tile)
    {
        check_tile(tile);
        return writable_state().superlanes[tile];
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
        auto& next = next_state();
        for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
            if (next.instruction_rows[alu][0].has_value()
                || next.instruction_queues[alu].empty()) {
                continue;
            }
            next.instruction_rows[alu][0] =
                next.instruction_queues[alu].front();
            next.instruction_queues[alu].pop_front();
        }
    }

    void log_status(std::ostream& os, std::optional<std::size_t> log_tile) const
    {
        const auto& next = next_state();
        std::size_t queued = 0;
        for (const auto& queue : next.instruction_queues) {
            queued += queue.size();
        }

        std::size_t active = 0;
        for (const auto& alu_rows : next.instruction_rows) {
            for (const auto& slot : alu_rows) {
                if (slot.has_value()) {
                    ++active;
                }
            }
        }

        std::size_t inputs = 0;
        for (const auto& buffer : next.input_buffers) {
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
        auto& next = next_state();
        bool any = false;
        bool any_logged = false;
        for (std::size_t tile = 0; tile < kTileCount; ++tile) {
            for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
                const auto& instruction = next.instruction_rows[alu][tile];
                if (!instruction.has_value()) {
                    continue;
                }

                any = true;
                next.superlanes[tile].enqueue_compact_instruction(
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
        auto& next = next_state();
        for (std::size_t tile = 0; tile < kTileCount; ++tile) {
            next.output_slots[tile].reset();
            next.output_slots_multi[tile].clear();

            auto stream_consumers =
                std::array<bool, kAluQueues>{};
            bool will_consume_bundle = false;
            for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
                if (!next.superlanes[tile]
                         .configuration_will_issue(alu)) {
                    continue;
                }
                const auto instruction =
                    next.superlanes[tile].next_instruction(alu);
                if (!instruction
                    || !instruction_uses_stream(*instruction)) {
                    continue;
                }
                stream_consumers[alu] = true;
                will_consume_bundle = true;
            }

            auto supplied_bundle = false;
            if (will_consume_bundle
                && !next.input_buffers[tile].ready()) {
                throw std::logic_error(
                    "VXM compiler/static-schedule error: input Bundle is not ready on its scheduled issue cycle at tile "
                    + std::to_string(tile));
            }
            if (will_consume_bundle) {
                next.superlanes[tile].set_stream_inputs(
                    next.input_buffers[tile].bundle());
                supplied_bundle = true;
                if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                    *os << "  tile " << tile
                        << " input Bundle groups="
                        << next.input_buffers[tile].fill_count()
                        << '\n';
                }
            }

            const auto executed = next.superlanes[tile].tick();
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
                next.input_buffers[tile].release_after_issue();
            }
            next.output_slots_multi[tile] = next.superlanes[tile].outputs();
            if (!next.output_slots_multi[tile].empty()) {
                next.output_slots[tile] = next.output_slots_multi[tile].front();
                if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                    for (const auto& output : next.output_slots_multi[tile]) {
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
        for (auto& alu_rows : next_state().instruction_rows) {
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
        if (operand.kind != VxmLaneOperandKind::StreamFloat16
            && operand.kind != VxmLaneOperandKind::StreamBFloat16
            && operand.kind != VxmLaneOperandKind::StreamFloat32) {
            return;
        }

        // RequiredStreams describes the fixed logical 32-byte VXM input
        // bundle.  The separately configured group source selects which MEM
        // hemisphere supplies that same-numbered westbound SR group.
        const auto base = VxmLane::fixed_input_base_for_stage(alu, rhs_port, operand.kind);
        for (std::size_t byte = 0; byte < VxmLane::stream_operand_bytes(operand.kind); ++byte) {
            required[base + byte] = true;
        }
    }

    static bool operand_uses_stream(const VxmLaneOperand& operand)
    {
        return operand.kind == VxmLaneOperandKind::StreamFloat16
            || operand.kind == VxmLaneOperandKind::StreamBFloat16
            || operand.kind == VxmLaneOperandKind::StreamFloat32;
    }

    static bool instruction_uses_stream(
        const VxmLaneAluInstruction& instruction)
    {
        return operand_uses_stream(instruction.lhs) || operand_uses_stream(instruction.rhs);
    }

    void refresh_required_streams()
    {
        auto& next = next_state();
        for (auto& required : next.required_streams) {
            required.reset();
        }

        for (std::size_t tile = 0; tile < kTileCount; ++tile) {
            auto required = RequiredStreams {};
            bool any = false;
            for (std::size_t alu = 0; alu < kAluQueues; ++alu) {
                // A held Current Config takes precedence over a new packet
                // merely passing this tile toward the shared Superlane queue.
                auto instruction = next.superlanes[tile].next_instruction(alu);
                if (!instruction && next.instruction_rows[alu][tile]) {
                    instruction =
                        VxmCompactInstructionCodec::decode(
                            alu, *next.instruction_rows[alu][tile])
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
                next.required_streams[tile] = required;
            }
        }
    }

    struct State {
        std::array<std::deque<CompactInstruction>, kAluQueues>
            instruction_queues{};
        std::array<std::array<InstructionSlot, kTileCount>, kAluQueues>
            instruction_rows{};
        // The Slice snapshot owns every hardware-visible Superlane state.
        // VxmLane statistics remain shared software observation state.
        std::vector<Superlane> superlanes{kTileCount};
        std::array<InputBuffer, kTileCount> input_buffers{};
        std::array<OutputSlot, kTileCount> output_slots{};
        std::array<std::vector<Superlane::Output>, kTileCount>
            output_slots_multi{};
        std::array<std::optional<RequiredStreams>, kTileCount>
            required_streams{};
        std::array<Hemisphere, VxmLane::kStreamGroupCount>
            input_group_sources{};
        std::array<Hemisphere, VxmLane::kBlockCount>
            output_block_destinations{};
        std::size_t cycle{0};
    };

    State& writable_state()
    {
        return next_ ? *next_ : *current_;
    }

    const State& visible_state() const
    {
        return next_ ? *next_ : *current_;
    }

    State& next_state()
    {
        if (!next_) throw std::logic_error("VXM Next state is not prepared");
        return *next_;
    }

    const State& next_state() const
    {
        if (!next_) throw std::logic_error("VXM Next state is not prepared");
        return *next_;
    }

    std::unique_ptr<State> current_{};
    std::unique_ptr<State> next_{};
    std::array<std::deque<CompactInstruction>, kAluQueues> pending_issues_{};
    bool low_started_{false};
    bool low_complete_{false};
};

} // namespace ftlpu
