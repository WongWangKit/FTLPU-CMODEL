#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm/column_slice.hpp"
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

// One shared control unit feeds all configured lanes in a Superlane. Every
// configured physical column is an independently scheduled 8-lane VALU slice
// with its own instruction FIFO, one-cycle decoder, Next/Current Config
// Registers, and repeat counter. Chain depth selects the fixed local-register
// topology used by those columns for a program phase.
class VxmSuperlaneInstructionControl {
public:
    static constexpr std::size_t kColumnCount = kVxmAluStageCount;
    // Compatibility name used by existing diagnostics and tests. A control
    // stage is now exactly one independently schedulable physical column.
    static constexpr std::size_t kStageCount = kColumnCount;
    static constexpr std::size_t kFifoDepth = 3;
    static constexpr std::size_t kDecodeLatency = 1;
    using Configs = VxmLaneConfigs;
    using ExecutionMask = VxmLaneExecutionMask;

    VxmSuperlaneInstructionControl()
        : VxmSuperlaneInstructionControl(
            std::make_index_sequence<kColumnCount>{})
    {
    }

    void reset()
    {
        for (auto& column : columns_) column.reset();
    }

    void enqueue(std::size_t stage, VxmChainDepth depth,
                 VxmLaneAluInstruction instruction)
    {
        check_stage(stage);
        if (instruction.repeat_count == 0) {
            throw std::invalid_argument(
                "VXM instruction repeat_count must be non-zero");
        }

        columns_[stage].enqueue(depth, std::move(instruction));
    }

    void enqueue_compact(
        std::size_t stage, VxmCompactInstruction instruction)
    {
        check_stage(stage);
        columns_[stage].enqueue_compact(std::move(instruction));
    }

    Configs issue(VxmChainDepth active_depth)
    {
        auto configs = Configs{};
        for (std::size_t stage = 0; stage < kStageCount; ++stage) {
            configs[stage] = columns_[stage].issue(active_depth);
        }
        return configs;
    }

    void consume(const ExecutionMask& executed)
    {
        for (std::size_t stage = 0; stage < kStageCount; ++stage) {
            columns_[stage].consume(executed[stage]);
        }
    }

    bool idle() const
    {
        for (const auto& column : columns_)
            if (!column.idle()) return false;
        return true;
    }

    std::size_t remaining_executions(std::size_t stage) const
    {
        check_stage(stage);
        return columns_[stage].remaining_executions();
    }

    std::size_t remaining_in_current(std::size_t stage) const
    {
        check_stage(stage);
        return columns_[stage].remaining_in_current();
    }

    std::size_t config_entry_count(std::size_t stage) const
    {
        check_stage(stage);
        return columns_[stage].config_entry_count();
    }

    std::size_t fifo_entry_count(std::size_t stage) const
    {
        check_stage(stage);
        return columns_[stage].fifo_entry_count();
    }

    bool decoding(std::size_t stage) const
    {
        check_stage(stage);
        return columns_[stage].decoding();
    }

    bool next_config_ready(std::size_t stage) const
    {
        check_stage(stage);
        return columns_[stage].next_config_ready();
    }

    ExecutionMask feedback_capture_mask(VxmChainDepth depth) const
    {
        auto mask = ExecutionMask{};
        const auto length = static_cast<std::size_t>(depth);
        for (std::size_t stage = 0; stage < kStageCount; ++stage) {
            if (stage % length != 0) continue;
            mask[stage] = columns_[stage].feedback_capture(depth);
        }
        return mask;
    }

    std::optional<VxmLaneAluInstruction> next_instruction(
        std::size_t stage) const
    {
        check_stage(stage);
        return columns_[stage].next_instruction();
    }

    bool configuration_will_issue(
        std::size_t stage,
        VxmChainDepth active_depth) const
    {
        check_stage(stage);
        return columns_[stage].configuration_will_issue(active_depth);
    }

private:
    template<std::size_t... Columns>
    explicit VxmSuperlaneInstructionControl(
        std::index_sequence<Columns...>)
        : columns_ {VxmColumnSlice {Columns}...}
    {
    }

    static void check_stage(std::size_t stage)
    {
        if (stage >= kStageCount) {
            throw std::out_of_range("VXM control column exceeds the configured physical column count");
        }
    }

    std::array<VxmColumnSlice, kColumnCount> columns_;
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
        instruction_control_.enqueue(
            alu, lanes_[0]->chain_depth(), std::move(instruction));
    }

    void enqueue_instruction_for_depth(
        VxmChainDepth depth, std::size_t alu,
        VxmLaneAluInstruction instruction)
    {
        check_control_stage(alu);
        lanes_[0]->validate_broadcast_instruction(depth, alu, instruction);
        instruction_control_.enqueue(alu, depth, std::move(instruction));
    }

    void enqueue_compact_instruction(
        std::size_t alu, VxmCompactInstruction instruction)
    {
        check_control_stage(alu);
        // Decode once here to validate this physical column. The packet itself
        // remains compact in the column-local control queue.
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
    VxmChainDepth chain_depth() const { return lanes_[0]->chain_depth(); }
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
                "VXM Superlane column exceeds the configured physical column count");
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
