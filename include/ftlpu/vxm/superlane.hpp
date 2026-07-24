#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm/lane.hpp"

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

// One shared control unit feeds all 16 lanes in a Superlane.  Each ALU stage
// has a three-entry instruction FIFO, one-cycle decoder, Next Config Register,
// Current Config Register, and repeat counter.
class VxmSuperlaneInstructionControl {
public:
    static constexpr std::size_t kStageCount = kVxmAluStageCount;
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
        if (!queues_[stage].empty()
            && queues_[stage].back().depth == run.depth
            && same_config(queues_[stage].back().instruction, run.instruction)) {
            queues_[stage].back().remaining += run.remaining;
            return;
        }
        if (queues_[stage].size() >= kFifoDepth) {
            throw std::overflow_error(
                "VXM Superlane Config FIFO overflow: compiler scheduled "
                "more than 3 queued entries for one ALU stage");
        }
        queues_[stage].push_back(std::move(run));
    }

    Configs issue(VxmChainDepth active_depth)
    {
        auto configs = Configs{};
        for (std::size_t stage = 0; stage < kStageCount; ++stage) {
            // The instruction placed in the decoder on the previous control
            // cycle has now completed the explicit one-cycle decode.
            if (decoding_[stage]) {
                if (next_[stage]) {
                    throw std::logic_error(
                        "VXM decoder completed while Next Config was occupied");
                }
                next_[stage] = std::move(decoding_[stage]);
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

            // Prefetch and decode the following entry while Current executes.
            // A newly-started decode is not visible until the next issue().
            if (!decoding_[stage] && !next_[stage]
                && !queues_[stage].empty()) {
                decoding_[stage] = std::move(queues_[stage].front());
                queues_[stage].pop_front();
            }

            if (current_[stage]) {
                configs[stage] = current_[stage]->instruction;
            }
        }
        return configs;
    }

    void consume(const ExecutionMask& executed)
    {
        for (std::size_t stage = 0; stage < kStageCount; ++stage) {
            if (!executed[stage]) continue;
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
        if (decoding_[stage]) total += decoding_[stage]->remaining;
        for (const auto& run : queues_[stage]) total += run.remaining;
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

    std::optional<VxmLaneAluInstruction> next_instruction(
        std::size_t stage) const
    {
        check_stage(stage);
        if (current_[stage]) return current_[stage]->instruction;
        if (next_[stage]) return next_[stage]->instruction;
        if (decoding_[stage]) return decoding_[stage]->instruction;
        if (!queues_[stage].empty()) return queues_[stage].front().instruction;
        return std::nullopt;
    }

private:
    struct ConfigRun {
        VxmLaneAluInstruction instruction{};
        std::size_t remaining{0};
        VxmChainDepth depth{VxmChainDepth::Eight};
    };

    static void check_stage(std::size_t stage)
    {
        if (stage >= kStageCount) {
            throw std::out_of_range("VXM control stage is outside 0..7");
        }
    }

    static bool same_operand(const VxmLaneOperand& lhs,
                             const VxmLaneOperand& rhs)
    {
        return lhs.kind == rhs.kind
            && lhs.immediate == rhs.immediate
            && lhs.scale == rhs.scale
            && lhs.zero_point == rhs.zero_point;
    }

    static bool same_config(const VxmLaneAluInstruction& lhs,
                            const VxmLaneAluInstruction& rhs)
    {
        return lhs.operation == rhs.operation
            && same_operand(lhs.lhs, rhs.lhs)
            && same_operand(lhs.rhs, rhs.rhs)
            && lhs.precision == rhs.precision
            && lhs.output_type == rhs.output_type
            && lhs.output_scale == rhs.output_scale
            && lhs.output_zero_point == rhs.output_zero_point
            && lhs.output_stream == rhs.output_stream
            && lhs.accumulator_reset == rhs.accumulator_reset
            && lhs.accumulator_write == rhs.accumulator_write
            && lhs.accumulator_emit == rhs.accumulator_emit;
    }

    std::array<std::deque<ConfigRun>, kStageCount> queues_{};
    std::array<std::optional<ConfigRun>, kStageCount> decoding_{};
    std::array<std::optional<ConfigRun>, kStageCount> next_{};
    std::array<std::optional<ConfigRun>, kStageCount> current_{};
};

class VxmSuperlane {
public:
    static constexpr std::size_t kLaneCount = hw::kLanesPerTile;
    using Int32Vector = std::array<std::int32_t, kLaneCount>;
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
            lane.bind_special_alu(special_alu_);
        }
    }

    void reset()
    {
        instruction_control_.reset();
        for (auto& lane : lanes_) lane.reset();
        output_.reset();
        outputs_.clear();
        cycle_ = 0;
    }

    void set_chain_depth(VxmChainDepth depth)
    {
        if (!datapath_idle()) {
            throw std::logic_error(
                "cannot change Superlane chain depth while data remains in flight");
        }
        for (auto& lane : lanes_) lane.set_chain_depth(depth);
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

    void load_pipelined_swiglu_program(const VxmLane::SwigluParams& params,
                                       std::size_t token_count,
                                       std::size_t output_stream = 12)
    {
        set_chain_depth(VxmChainDepth::Eight);
        auto repeated = [token_count](VxmLaneAluInstruction instruction) {
            instruction.repeat_count = token_count;
            return instruction;
        };
        enqueue_instruction(0, repeated({VxmAluOpcode::Negate,
            VxmLaneOperand::StreamInt32(params.gate_scale),
            VxmLaneOperand::StreamInt32(params.up_scale)}));
        enqueue_instruction(1, repeated(
            {VxmSpecialAluOpcode::Exp, VxmLaneOperand::Previous()}));
        enqueue_instruction(2, repeated(
            {VxmAluOpcode::Add, VxmLaneOperand::Previous(), VxmLaneOperand::Imm(1.0f)}));
        enqueue_instruction(3, repeated(
            {VxmSpecialAluOpcode::Reciprocal, VxmLaneOperand::Previous()}));
        enqueue_instruction(4, repeated(
            {VxmAluOpcode::Multiply, VxmLaneOperand::Previous(), VxmLaneOperand::Original()}));
        enqueue_instruction(5, repeated(
            {VxmAluOpcode::Multiply, VxmLaneOperand::Previous(), VxmLaneOperand::Aux()}));
        enqueue_instruction(6, repeated(
            {VxmAluOpcode::Bypass, VxmLaneOperand::Previous()}));
        auto tail = VxmLaneAluInstruction{
            VxmAluOpcode::Bypass, VxmLaneOperand::Previous()};
        tail.output_type = VxmCastTarget::Int8;
        tail.output_scale = params.output_scale;
        tail.output_zero_point = params.output_zero_point;
        tail.output_stream = output_stream;
        tail.repeat_count = token_count;
        enqueue_instruction(7, tail);
    }

    void load_pipelined_add_quant_program(const VxmLane::AddQuantParams& params,
                                          std::size_t token_count,
                                          std::size_t output_stream = 0)
    {
        set_chain_depth(VxmChainDepth::Two);
        auto head = VxmLaneAluInstruction{VxmAluOpcode::Add,
            VxmLaneOperand::StreamInt32(params.lhs_scale),
            VxmLaneOperand::StreamInt32(params.rhs_scale)};
        head.repeat_count = token_count;
        enqueue_instruction(0, head);
        auto tail = VxmLaneAluInstruction{
            VxmAluOpcode::Bypass, VxmLaneOperand::Previous()};
        tail.output_type = VxmCastTarget::Int8;
        tail.output_scale = params.output_scale;
        tail.output_zero_point = params.output_zero_point;
        tail.output_stream = output_stream;
        tail.repeat_count = token_count;
        enqueue_instruction(1, tail);
    }

    void enqueue_instruction(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        lanes_[0].validate_broadcast_instruction(alu, instruction);
        instruction_control_.enqueue(
            alu, lanes_[0].chain_depth(), std::move(instruction));
    }

    void enqueue_instruction_for_depth(
        VxmChainDepth depth, std::size_t alu,
        VxmLaneAluInstruction instruction)
    {
        lanes_[0].validate_broadcast_instruction(depth, alu, instruction);
        instruction_control_.enqueue(alu, depth, std::move(instruction));
    }

    void set_stream_inputs(const StreamMatrix& streams)
    {
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) lanes_[lane].set_stream_inputs(streams[lane]);
    }

    void tick()
    {
        output_.reset();
        outputs_.clear();
        const auto issued =
            instruction_control_.issue(lanes_[0].chain_depth());
        auto common_execution = VxmLaneExecutionMask{};
        bool first = true;
        for (auto& lane : lanes_) {
            const auto executed = lane.tick(issued);
            if (first) {
                common_execution = executed;
                first = false;
            } else if (executed != common_execution) {
                throw std::logic_error(
                    "VXM Superlane lanes lost lockstep under shared instruction control");
            }
        }
        instruction_control_.consume(common_execution);

        const auto count = lanes_[0].outputs().size();
        for (std::size_t lane = 1; lane < kLaneCount; ++lane) {
            if (lanes_[lane].outputs().size() != count) {
                throw std::logic_error("VXM superlane lanes produced different output counts");
            }
        }
        for (std::size_t item = 0; item < count; ++item) {
            auto result = Output{};
            result.stream = lanes_[0].outputs()[item].stream;
            result.byte_count = lanes_[0].outputs()[item].byte_count;
            for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
                const auto& lane_output = lanes_[lane].outputs()[item];
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
    }

    const std::optional<Output>& output() const { return output_; }
    const std::vector<Output>& outputs() const { return outputs_; }
    const VxmLane& lane(std::size_t index) const { check_lane(index); return lanes_[index]; }
    VxmLane& lane(std::size_t index) { check_lane(index); return lanes_[index]; }
    std::size_t cycle() const { return cycle_; }
    bool idle() const
    {
        if (!instruction_control_.idle()) return false;
        return datapath_idle();
    }
    bool datapath_idle() const
    {
        for (const auto& lane : lanes_) {
            if (!lane.datapath_idle()) return false;
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
    void print_lane_trace(std::ostream& os, std::size_t lane_index) const { lane(lane_index).print_last_trace(os); }

private:
    static void check_lane(std::size_t index)
    {
        if (index >= kLaneCount) throw std::out_of_range("VXM lane is outside 0..15");
    }

    std::shared_ptr<VxmSpecialAlu> special_alu_;
    VxmSuperlaneInstructionControl instruction_control_{};
    std::array<VxmLane, kLaneCount> lanes_{};
    std::optional<Output> output_{};
    std::vector<Output> outputs_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
