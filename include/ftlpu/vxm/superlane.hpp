#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm/compact_instruction.hpp"
#include "ftlpu/vxm/lane.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

// One shared control unit feeds all configured lanes in a Superlane. The 16 physical
// ALUs are split into two identical 8-stage datapaths. Physical stages Ci and
// C(i+8) share one three-entry instruction FIFO, one-cycle decoder, Next/Current
// Config Registers, and repeat counter. The local decoder mirrors the decoded
// config; data and pipeline state remain independent in the two datapaths.
class VxmSuperlaneInstructionControl {
public:
    static constexpr std::size_t kStageCount = kVxmAluStageCount / 2;
    static constexpr std::size_t kMirroredStageOffset = kStageCount;
    static constexpr std::size_t kFifoDepth = 3;
    static constexpr std::size_t kDecodeLatency = 1;
    using Configs = VxmLaneConfigs;
    using ExecutionMask = VxmLaneExecutionMask;

    void reset()
    {
        for (auto& queue : queues_) queue.clear();
        for (auto& decoding : decoding_) decoding.reset();
        for (auto& next : next_) next.reset();
        for (auto& current : current_) current.reset();
    }

    void enqueue(std::size_t stage, VxmChainDepth depth,
                 VxmLaneAluInstruction instruction)
    {
        check_stage(stage);
        if (instruction.repeat_count == 0) {
            throw std::invalid_argument(
                "VXM instruction repeat_count must be non-zero");
        }

        auto run = ConfigRun{instruction, instruction.repeat_count, depth};
        run.instruction.repeat_count = 1;
        enqueue_pending(stage, std::move(run));
    }

    void enqueue_compact(
        std::size_t stage, VxmCompactInstruction instruction)
    {
        check_stage(stage);
        enqueue_pending(stage, std::move(instruction));
    }

    Configs issue(VxmChainDepth active_depth)
    {
        auto configs = Configs{};
        for (std::size_t stage = 0; stage < kStageCount; ++stage) {
            // A compact packet placed in this Superlane's decoder during the
            // previous control cycle is expanded here. The wide config never
            // travels through the Slice instruction pipeline.
            if (decoding_[stage]) {
                if (next_[stage]) {
                    throw std::logic_error(
                        "VXM decoder completed while Next Config was occupied");
                }
                next_[stage] =
                    decode_pending(stage, std::move(*decoding_[stage]));
                decoding_[stage].reset();
            }

            // A prefetched configuration for a future 2/4/8 routing mode may
            // finish decoding early, but it cannot become Current until the
            // global chain-depth register switches to that mode.
            if (!current_[stage] && next_[stage]
                && next_[stage]->depth == active_depth) {
                current_[stage] = std::move(next_[stage]);
                next_[stage].reset();
            }

            // Prefetch and decode the following compact FIFO entry while
            // Current executes. A newly-started decode is not visible until
            // the next issue().
            if (!decoding_[stage] && !next_[stage]
                && !queues_[stage].empty()) {
                decoding_[stage] = std::move(queues_[stage].front());
                queues_[stage].pop_front();
            }

            if (current_[stage]) {
                configs[stage] = current_[stage]->instruction;
                configs[stage + kMirroredStageOffset] =
                    instruction_for_physical_stage(
                        stage + kMirroredStageOffset,
                        current_[stage]->instruction);
            }
        }
        return configs;
    }

private:
    struct ConfigRun {
        VxmLaneAluInstruction instruction{};
        std::size_t remaining{0};
        VxmChainDepth depth{VxmChainDepth::Eight};
    };

    // Decoded ConfigRun is retained only for the legacy direct-Superlane
    // helper API. The real Slice path stores VxmCompactInstruction.
    using PendingInstruction =
        std::variant<VxmCompactInstruction, ConfigRun>;

    void enqueue_pending(
        std::size_t stage, PendingInstruction instruction)
    {
        if (queues_[stage].size() >= kFifoDepth) {
            throw std::overflow_error(
                "VXM Superlane Config FIFO overflow: compiler scheduled "
                "more than 3 queued entries for one ALU stage");
        }
        queues_[stage].push_back(std::move(instruction));
    }

    static ConfigRun decode_pending(
        std::size_t stage, PendingInstruction pending)
    {
        if (auto* decoded = std::get_if<ConfigRun>(&pending)) {
            return std::move(*decoded);
        }
        auto decoded = VxmCompactInstructionCodec::decode(
            stage, std::get<VxmCompactInstruction>(pending));
        const auto repeat_count = decoded.instruction.repeat_count;
        auto run = ConfigRun{
            std::move(decoded.instruction),
            repeat_count,
            decoded.chain_depth};
        run.instruction.repeat_count = 1;
        return run;
    }

public:

    void consume(const ExecutionMask& executed)
    {
        for (std::size_t stage = 0; stage < kStageCount; ++stage) {
            // The pair shares one repeat counter. A cycle consumes one config
            // entry when either physical copy accepts its token. Normally both
            // copies execute in lockstep; accepting either also supports a
            // partially-filled final token wave.
            if (!executed[stage]
                && !executed[stage + kMirroredStageOffset]) {
                continue;
            }
            if (!current_[stage] || current_[stage]->remaining == 0) {
                throw std::logic_error(
                    "VXM consumed an empty Current Config Register");
            }
            if (--current_[stage]->remaining == 0) {
                current_[stage].reset();
            }
        }
    }

    bool idle() const
    {
        for (const auto& current : current_) if (current) return false;
        for (const auto& next : next_) if (next) return false;
        for (const auto& decoding : decoding_) if (decoding) return false;
        for (const auto& queue : queues_) if (!queue.empty()) return false;
        return true;
    }

    std::size_t remaining_executions(std::size_t stage) const
    {
        check_stage(stage);
        std::size_t total = current_[stage] ? current_[stage]->remaining : 0;
        if (next_[stage]) total += next_[stage]->remaining;
        if (decoding_[stage]) {
            total += pending_remaining(stage, *decoding_[stage]);
        }
        for (const auto& pending : queues_[stage]) {
            total += pending_remaining(stage, pending);
        }
        return total;
    }

    std::size_t remaining_in_current(std::size_t stage) const
    {
        check_stage(stage);
        return current_[stage] ? current_[stage]->remaining : 0;
    }

    std::size_t config_entry_count(std::size_t stage) const
    {
        check_stage(stage);
        return queues_[stage].size()
            + (decoding_[stage] ? 1 : 0)
            + (next_[stage] ? 1 : 0)
            + (current_[stage] ? 1 : 0);
    }

    std::size_t fifo_entry_count(std::size_t stage) const
    {
        check_stage(stage);
        return queues_[stage].size();
    }

    bool decoding(std::size_t stage) const
    {
        check_stage(stage);
        return decoding_[stage].has_value();
    }

    bool next_config_ready(std::size_t stage) const
    {
        check_stage(stage);
        return next_[stage].has_value();
    }

    ExecutionMask feedback_capture_mask(VxmChainDepth depth) const
    {
        auto mask = ExecutionMask{};
        const auto length = static_cast<std::size_t>(depth);
        for (std::size_t stage = 0; stage < kStageCount; ++stage) {
            if (stage % length != 0) continue;
            const ConfigRun* run = nullptr;
            if (next_[stage] && next_[stage]->depth == depth) {
                run = &*next_[stage];
            } else if (current_[stage]
                       && current_[stage]->depth == depth) {
                run = &*current_[stage];
            }
            mask[stage] =
                run
                && run->instruction.lhs.kind
                    == VxmLaneOperandKind::Feedback;
            mask[stage + kMirroredStageOffset] = mask[stage];
        }
        return mask;
    }

    std::optional<VxmLaneAluInstruction> next_instruction(
        std::size_t stage) const
    {
        check_stage(stage);
        if (current_[stage]) return current_[stage]->instruction;
        if (next_[stage]) return next_[stage]->instruction;
        if (decoding_[stage]) {
            return pending_instruction(stage, *decoding_[stage]);
        }
        if (!queues_[stage].empty()) {
            return pending_instruction(stage, queues_[stage].front());
        }
        return std::nullopt;
    }

    bool configuration_will_issue(
        std::size_t stage,
        VxmChainDepth active_depth) const
    {
        check_stage(stage);
        if (current_[stage]) {
            return true;
        }
        if (next_[stage]) {
            return next_[stage]->depth == active_depth;
        }
        if (decoding_[stage]) {
            return pending_depth(stage, *decoding_[stage])
                == active_depth;
        }
        return false;
    }

    static VxmLaneAluInstruction instruction_for_physical_stage(
        std::size_t physical_stage,
        VxmLaneAluInstruction instruction)
    {
        if (physical_stage >= kVxmAluStageCount) {
            throw std::out_of_range(
                "VXM physical ALU stage is outside 0..15");
        }
        if (instruction.output_stream) {
            instruction.output_stream =
                VxmLane::fixed_output_stream_for_block(
                    VxmLane::block_for_stage(physical_stage));
        }
        return instruction;
    }

private:
    static void check_stage(std::size_t stage)
    {
        if (stage >= kStageCount) {
            throw std::out_of_range("VXM control stage is outside 0..7");
        }
    }

    static std::size_t pending_remaining(
        std::size_t stage, const PendingInstruction& pending)
    {
        if (const auto* decoded = std::get_if<ConfigRun>(&pending)) {
            return decoded->remaining;
        }
        return VxmCompactInstructionCodec::decode(
            stage, std::get<VxmCompactInstruction>(pending))
            .instruction.repeat_count;
    }

    static VxmChainDepth pending_depth(
        std::size_t stage,
        const PendingInstruction& pending)
    {
        if (const auto* decoded = std::get_if<ConfigRun>(&pending)) {
            return decoded->depth;
        }
        return VxmCompactInstructionCodec::decode(
            stage, std::get<VxmCompactInstruction>(pending))
            .chain_depth;
    }

    static VxmLaneAluInstruction pending_instruction(
        std::size_t stage, const PendingInstruction& pending)
    {
        if (const auto* decoded = std::get_if<ConfigRun>(&pending)) {
            return decoded->instruction;
        }
        return VxmCompactInstructionCodec::decode(
            stage, std::get<VxmCompactInstruction>(pending)).instruction;
    }

    std::array<std::deque<PendingInstruction>, kStageCount> queues_{};
    std::array<std::optional<PendingInstruction>, kStageCount> decoding_{};
    std::array<std::optional<ConfigRun>, kStageCount> next_{};
    std::array<std::optional<ConfigRun>, kStageCount> current_{};
};

class VxmSuperlane {
public:
    static constexpr std::size_t kLaneCount = hw::kLanesPerTile;
    using Int8Vector = std::array<std::int8_t, kLaneCount>;
    using StreamBytes = VxmLane::StreamBytes;
    using StreamMatrix = std::array<StreamBytes, kLaneCount>;

    struct Output {
        Int8Vector values{};
        std::array<std::array<std::uint8_t, 4>, kLaneCount> byte_values{};
        std::size_t stream{0};
        std::size_t byte_count{1};
    };

    VxmSuperlane()
        : special_alu_(std::make_shared<VxmSpecialAlu>())
    {
        for (auto& lane : lanes_) {
            lane = std::make_unique<VxmLane>(special_alu_);
        }
    }

    void reset()
    {
        instruction_control_.reset();
        for (auto& lane : lanes_) lane->reset();
        output_.reset();
        outputs_.clear();
        cycle_ = 0;
        pending_chain_depth_.reset();
        pending_feedback_required_ = false;
    }

    void set_chain_depth(VxmChainDepth depth)
    {
        if (!datapath_idle()) {
            throw std::logic_error(
                "cannot change Superlane chain depth while data remains in flight");
        }
        for (auto& lane : lanes_) lane->set_chain_depth(depth);
    }

    // Request a no-bubble configuration boundary.  The old depth remains
    // active for the current tick so its chain tails can retire.  A decoded
    // Next Config supplies the new heads' Feedback mux controls, then the new
    // depth becomes active at the same clock edge as feedback capture.
    void request_chain_depth_transition(
        VxmChainDepth depth, bool feedback_required = true)
    {
        if (depth == lanes_[0]->chain_depth()) {
            throw std::invalid_argument(
                "VXM chain-depth transition requires a different depth");
        }
        if (pending_chain_depth_) {
            throw std::logic_error(
                "VXM already has a pending chain-depth transition");
        }
        if (datapath_idle()) {
            throw std::logic_error(
                "VXM no-bubble transition requires old tail data in flight");
        }
        pending_chain_depth_ = depth;
        pending_feedback_required_ = feedback_required;
    }

    void configure_special_lut(VxmSpecialAluOpcode opcode, VxmLutConfig config,
                               std::vector<VxmLutEntry> entries)
    {
        if (!idle()) throw std::logic_error("cannot configure shared LUT while lanes are active");
        special_alu_->configure_lut(opcode, config, std::move(entries));
    }

    VxmSpecialAlu& special_alu() { return *special_alu_; }
    const VxmSpecialAlu& special_alu() const { return *special_alu_; }
    const VxmSuperlaneInstructionControl& instruction_control() const
    {
        return instruction_control_;
    }

    void enqueue_instruction(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        check_control_stage(alu);
        lanes_[0]->validate_broadcast_instruction(alu, instruction);
        lanes_[0]->validate_broadcast_instruction(
            alu + VxmSuperlaneInstructionControl::kMirroredStageOffset,
            VxmSuperlaneInstructionControl::instruction_for_physical_stage(
                alu + VxmSuperlaneInstructionControl::kMirroredStageOffset,
                instruction));
        instruction_control_.enqueue(
            alu, lanes_[0]->chain_depth(), std::move(instruction));
    }

    void enqueue_instruction_for_depth(
        VxmChainDepth depth, std::size_t alu,
        VxmLaneAluInstruction instruction)
    {
        check_control_stage(alu);
        lanes_[0]->validate_broadcast_instruction(depth, alu, instruction);
        lanes_[0]->validate_broadcast_instruction(
            depth,
            alu + VxmSuperlaneInstructionControl::kMirroredStageOffset,
            VxmSuperlaneInstructionControl::instruction_for_physical_stage(
                alu + VxmSuperlaneInstructionControl::kMirroredStageOffset,
                instruction));
        instruction_control_.enqueue(alu, depth, std::move(instruction));
    }

    void enqueue_compact_instruction(
        std::size_t alu, VxmCompactInstruction instruction)
    {
        check_control_stage(alu);
        // Decode once here only to validate the mirrored physical placement.
        // The packet itself remains compact and is stored only in the shared
        // logical control queue.
        const auto decoded = VxmCompactInstructionCodec::decode(
            alu, instruction);
        if (decoded.chain_depth != lanes_[0]->chain_depth()
            && datapath_idle()
            && instruction_control_.idle()) {
            // Chain depth is part of the compact hardware instruction.  Apply
            // it at an idle boundary so an ICU-driven program does not need a
            // direct C-model configuration call before execution.  While the
            // old datapath is active, retain its depth: an explicit scheduled
            // transition may still use the decoded packet for tail feedback.
            set_chain_depth(decoded.chain_depth);
        }
        const auto mirrored_stage =
            alu + VxmSuperlaneInstructionControl::kMirroredStageOffset;
        lanes_[0]->validate_broadcast_instruction(
            decoded.chain_depth, mirrored_stage,
            VxmSuperlaneInstructionControl::instruction_for_physical_stage(
                mirrored_stage, decoded.instruction));
        instruction_control_.enqueue_compact(
            alu, std::move(instruction));
    }

    void set_stream_inputs(const StreamMatrix& streams)
    {
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            lanes_[lane]->set_stream_inputs(streams[lane]);
        }
    }

    VxmLaneExecutionMask tick()
    {
        output_.reset();
        outputs_.clear();
        const auto issued =
            instruction_control_.issue(lanes_[0]->chain_depth());
        const auto feedback_capture = pending_chain_depth_
            ? instruction_control_.feedback_capture_mask(
                *pending_chain_depth_)
            : VxmLaneExecutionMask{};
        if (pending_chain_depth_ && pending_feedback_required_
            && std::none_of(
                feedback_capture.begin(), feedback_capture.end(),
                [](bool enabled) { return enabled; })) {
            throw std::logic_error(
                "VXM chain-depth transition reached the boundary before "
                "the new Feedback configuration finished decoding");
        }
        auto common_execution = VxmLaneExecutionMask{};
        bool first = true;
        for (auto& lane : lanes_) {
            const auto executed =
                lane->tick(issued, feedback_capture);
            if (first) {
                common_execution = executed;
                first = false;
            } else if (executed != common_execution) {
                throw std::logic_error(
                    "VXM Superlane lanes lost lockstep under shared instruction control");
            }
        }
        instruction_control_.consume(common_execution);

        if (pending_chain_depth_) {
            if (pending_feedback_required_
                && lanes_[0]->last_feedback_capture_count() == 0) {
                throw std::logic_error(
                    "VXM no-bubble transition did not capture an old "
                    "chain-tail result");
            }
            for (const auto& lane : lanes_) {
                lane->validate_chain_depth_transition(
                    *pending_chain_depth_);
            }
            for (auto& lane : lanes_) {
                lane->commit_chain_depth_transition(
                    *pending_chain_depth_);
            }
            pending_chain_depth_.reset();
            pending_feedback_required_ = false;
        }

        const auto count = lanes_[0]->outputs().size();
        for (std::size_t lane = 1; lane < kLaneCount; ++lane) {
            if (lanes_[lane]->outputs().size() != count) {
                throw std::logic_error("VXM superlane lanes produced different output counts");
            }
        }
        for (std::size_t item = 0; item < count; ++item) {
            auto result = Output{};
            result.stream = lanes_[0]->outputs()[item].stream;
            result.byte_count = lanes_[0]->outputs()[item].byte_count;
            for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
                const auto& lane_output = lanes_[lane]->outputs()[item];
                if (lane_output.stream != result.stream || lane_output.byte_count != result.byte_count) {
                    throw std::logic_error("VXM lanes disagree on fixed output binding");
                }
                result.values[lane] = lane_output.value;
                result.byte_values[lane] = lane_output.bytes;
            }
            outputs_.push_back(result);
        }
        if (!outputs_.empty()) output_ = outputs_.front();
        ++cycle_;
        return common_execution;
    }

    const std::optional<Output>& output() const { return output_; }
    const std::vector<Output>& outputs() const { return outputs_; }
    const VxmLane& lane(std::size_t index) const
    {
        check_lane(index);
        return *lanes_[index];
    }
    VxmLane& lane(std::size_t index)
    {
        check_lane(index);
        return *lanes_[index];
    }
    std::size_t cycle() const { return cycle_; }
    bool idle() const
    {
        if (pending_chain_depth_) return false;
        if (!instruction_control_.idle()) return false;
        for (const auto& lane : lanes_) {
            if (!lane->idle()) return false;
        }
        return true;
    }
    bool datapath_idle() const
    {
        for (const auto& lane : lanes_) {
            if (!lane->datapath_idle()) return false;
        }
        return true;
    }
    std::size_t remaining_in_current(std::size_t alu) const
    {
        return instruction_control_.remaining_in_current(alu);
    }
    std::size_t remaining_executions(std::size_t alu) const
    {
        return instruction_control_.remaining_executions(alu);
    }
    std::size_t config_entry_count(std::size_t alu) const
    {
        return instruction_control_.config_entry_count(alu);
    }
    std::optional<VxmLaneAluInstruction> next_instruction(
        std::size_t alu) const
    {
        return instruction_control_.next_instruction(alu);
    }
    bool configuration_will_issue(std::size_t alu) const
    {
        return instruction_control_.configuration_will_issue(
            alu, lanes_[0]->chain_depth());
    }
    bool chain_depth_transition_pending() const
    {
        return pending_chain_depth_.has_value();
    }
    void print_lane_trace(std::ostream& os, std::size_t lane_index) const { lane(lane_index).print_last_trace(os); }

private:
    static void check_control_stage(std::size_t stage)
    {
        if (stage >= VxmSuperlaneInstructionControl::kStageCount) {
            throw std::out_of_range(
                "VXM Superlane owns logical control stages 0..7; "
                "Ci is mirrored to physical C(i+8)");
        }
    }

    static void check_lane(std::size_t index)
    {
        if (index >= kLaneCount) {
            throw std::out_of_range(
                "VXM lane exceeds the configured Superlane lane count");
        }
    }

    std::shared_ptr<VxmSpecialAlu> special_alu_;
    VxmSuperlaneInstructionControl instruction_control_{};
    std::array<std::unique_ptr<VxmLane>, kLaneCount> lanes_{};
    std::optional<Output> output_{};
    std::vector<Output> outputs_{};
    std::size_t cycle_{0};
    std::optional<VxmChainDepth> pending_chain_depth_{};
    bool pending_feedback_required_{false};
};

} // namespace ftlpu
