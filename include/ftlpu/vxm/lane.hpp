#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm/alu.hpp"
#include "ftlpu/vxm/special_alu.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace ftlpu {

enum class VxmChainDepth : std::size_t {
    Two = 2,
    Four = 4,
    Eight = 8,
};

enum class VxmLaneOperandKind {
    Previous,//之前运算的结果
    Original,//第一操作数
    Auxiliary,//第二操作数
    Accumulator,//累加器结果
    //stream reg打包出的data
    StreamInt32,
    StreamFloat32,
    //控制流引出的立即数
    Immediate,
};

struct VxmLaneOperand {
    VxmLaneOperandKind kind{VxmLaneOperandKind::Immediate};
    float immediate{0.0f};
    float scale{1.0f};
    std::int32_t zero_point{0};

    static VxmLaneOperand Previous() { return {VxmLaneOperandKind::Previous}; }
    static VxmLaneOperand Original() { return {VxmLaneOperandKind::Original}; }
    static VxmLaneOperand Aux() { return {VxmLaneOperandKind::Auxiliary}; }
    static VxmLaneOperand Acc() { return {VxmLaneOperandKind::Accumulator}; }
    static VxmLaneOperand Imm(float value) { return {VxmLaneOperandKind::Immediate, value}; }
    static VxmLaneOperand StreamInt32(float scale = 1.0f,
                                      std::int32_t zero_point = 0)
    {
        return {VxmLaneOperandKind::StreamInt32, 0.0f, scale, zero_point};
    }
    static VxmLaneOperand StreamFloat32(float scale = 1.0f)
    {
        return {VxmLaneOperandKind::StreamFloat32, 0.0f, scale, 0};
    }
};

using VxmLaneOperation = std::variant<
    VxmAluOpcode,
    VxmSpecialAluOpcode>;

struct VxmLaneAluInstruction {
    VxmLaneOperation operation{VxmAluOpcode::Bypass};
    VxmLaneOperand lhs{VxmLaneOperand::Previous()};
    VxmLaneOperand rhs{VxmLaneOperand::Imm(0.0f)};
    VxmAluPrecision precision{VxmAluPrecision::Float16};
    VxmCastTarget output_type{VxmCastTarget::Float32};
    float output_scale{1.0f};
    std::int32_t output_zero_point{0};
    std::optional<std::size_t> output_stream{};
    bool accumulator_reset{false};
    bool accumulator_write{false};
    bool accumulator_emit{true};
    std::size_t repeat_count{1};
};

enum class VxmLaneAluTraceState { Idle, Stalled, Executed };

struct VxmLaneAluTrace {
    VxmLaneAluTraceState state{VxmLaneAluTraceState::Idle};
    std::optional<float> result{};
};

inline constexpr std::size_t kVxmAluStageCount = 8;
using VxmLaneConfigs =
    std::array<std::optional<VxmLaneAluInstruction>, kVxmAluStageCount>;
using VxmLaneExecutionMask = std::array<bool, kVxmAluStageCount>;

class VxmLane {
public:
    static constexpr std::size_t kAluCount = 8;
    static constexpr std::size_t kBlockCount = 4;
    static constexpr std::size_t kStagesPerBlock = 2;
    static constexpr std::size_t kInputStreams = hw::kStreamsPerDirection;
    static constexpr std::size_t kStreamGroupBytes = 4;
    static constexpr std::size_t kStreamGroupCount = kInputStreams / kStreamGroupBytes;

    using StreamBytes = std::array<std::uint8_t, kInputStreams>;

    struct Utilization {
        std::uint64_t cycles{0};
        std::uint64_t executed_slots{0};
        std::uint64_t useful_slots{0};
        std::uint64_t peak_executed_slots{0};
        std::uint64_t peak_useful_slots{0};
        std::array<std::uint64_t, kAluCount> stage_executions{};

        double active_utilization() const
        {
            return cycles == 0 ? 0.0
                : static_cast<double>(executed_slots)
                    / static_cast<double>(cycles * kAluCount);
        }

        double useful_utilization() const
        {
            return cycles == 0 ? 0.0
                : static_cast<double>(useful_slots)
                    / static_cast<double>(cycles * kAluCount);
        }

        double peak_active_utilization() const
        {
            return static_cast<double>(peak_executed_slots)
                / static_cast<double>(kAluCount);
        }

        double peak_useful_utilization() const
        {
            return static_cast<double>(peak_useful_slots)
                / static_cast<double>(kAluCount);
        }
    };

    struct CycleActivity {
        std::size_t cycle{0};
        VxmChainDepth chain_depth{VxmChainDepth::Eight};
        std::array<VxmLaneAluTraceState, kAluCount> states{};
        std::array<bool, kAluCount> useful{};

        std::size_t active_slots() const
        {
            std::size_t count = 0;
            for (const auto state : states) {
                if (state == VxmLaneAluTraceState::Executed) ++count;
            }
            return count;
        }

        std::size_t useful_slots() const
        {
            std::size_t count = 0;
            for (const auto value : useful) count += value ? 1 : 0;
            return count;
        }
    };

    struct Statistics {
        std::uint64_t cycles{0};
        std::uint64_t executed_slots{0};
        std::uint64_t useful_slots{0};
        std::array<std::uint64_t, kAluCount> stage_executions{};
        std::array<Utilization, 3> by_chain_depth{};
        std::vector<CycleActivity> timeline{};

        double active_utilization() const
        {
            return cycles == 0 ? 0.0
                : static_cast<double>(executed_slots)
                    / static_cast<double>(cycles * kAluCount);
        }

        double useful_utilization() const
        {
            return cycles == 0 ? 0.0
                : static_cast<double>(useful_slots)
                    / static_cast<double>(cycles * kAluCount);
        }

        const Utilization& for_depth(VxmChainDepth depth) const
        {
            return by_chain_depth.at(depth_index(depth));
        }

        Utilization total() const
        {
            auto result = Utilization{};
            result.cycles = cycles;
            result.executed_slots = executed_slots;
            result.useful_slots = useful_slots;
            result.stage_executions = stage_executions;
            for (const auto& activity : timeline) {
                result.peak_executed_slots = std::max<std::uint64_t>(
                    result.peak_executed_slots, activity.active_slots());
                result.peak_useful_slots = std::max<std::uint64_t>(
                    result.peak_useful_slots, activity.useful_slots());
            }
            return result;
        }

    private:
        static constexpr std::size_t depth_index(VxmChainDepth depth)
        {
            switch (depth) {
            case VxmChainDepth::Two: return 0;
            case VxmChainDepth::Four: return 1;
            case VxmChainDepth::Eight: return 2;
            }
            return 0;
        }

        friend class VxmLane;
    };

    struct Output {
        std::int8_t value{0};
        std::array<std::uint8_t, 4> bytes{};
        std::size_t stream{0};
        std::size_t byte_count{1};
    };

    struct SwigluParams {
        float gate_scale{1.0f};
        float up_scale{1.0f};
        float output_scale{1.0f};
        std::int32_t output_zero_point{0};
    };

    struct AddQuantParams {
        float lhs_scale{1.0f};
        float rhs_scale{1.0f};
        float output_scale{1.0f};
        std::int32_t output_zero_point{0};
    };

    VxmLane()
        : special_alu_(std::make_shared<VxmSpecialAlu>())
    {}

    explicit VxmLane(std::shared_ptr<VxmSpecialAlu> special_alu)
        : special_alu_(std::move(special_alu))
    {
        if (!special_alu_) throw std::invalid_argument("VXM lane requires a special ALU");
    }

    void bind_special_alu(std::shared_ptr<VxmSpecialAlu> special_alu)
    {
        if (!idle()) throw std::logic_error("cannot rebind LUT while VXM lane is active");
        if (!special_alu) throw std::invalid_argument("VXM lane requires a special ALU");
        special_alu_ = std::move(special_alu);
    }

    VxmSpecialAlu& special_alu() { return *special_alu_; }
    const VxmSpecialAlu& special_alu() const { return *special_alu_; }

    void reset()
    {
        for (auto& token : stage_inputs_) token.reset();
        for (auto& pipeline : basic_pipelines_) pipeline.reset();
        for (auto& pipeline : special_pipelines_) pipeline.reset();
        for (auto& accumulator : accumulators_) accumulator.reset();
        for (auto& reg : output_registers_) reg.reset();
        stream_inputs_.fill(0);
        stream_inputs_valid_ = false;
        output_.reset();
        outputs_.clear();
        last_trace_.fill({});
        statistics_ = {};
        cycle_ = 0;
    }

    void set_chain_depth(VxmChainDepth depth)
    {
        if (!datapath_idle()) {
            throw std::logic_error("cannot change VXM chain depth while data path is active");
        }
        chain_depth_ = depth;
    }

    VxmChainDepth chain_depth() const { return chain_depth_; }
    std::size_t chain_length() const { return static_cast<std::size_t>(chain_depth_); }
    std::size_t chain_count() const { return kAluCount / chain_length(); }
    bool is_chain_head(std::size_t stage) const { check_stage(stage); return stage % chain_length() == 0; }
    bool is_chain_tail(std::size_t stage) const { check_stage(stage); return stage % chain_length() == chain_length() - 1; }
    bool stream_input_enabled(std::size_t stage) const { return is_chain_head(stage); }
    bool output_register_enabled(std::size_t stage) const { return is_chain_tail(stage); }

    static constexpr std::size_t block_for_stage(std::size_t stage) { return stage / kStagesPerBlock; }
    static constexpr std::size_t fixed_output_stream_for_block(std::size_t block)
    {
        return block * kStreamGroupBytes;
    }
    std::size_t fixed_output_stream_for_stage(std::size_t stage) const
    {
        check_stage(stage);
        return fixed_output_stream_for_block(block_for_stage(stage));
    }

    // Four physical chain heads have two direct-wired Stream Groups each:
    // ALU0 <- G0/G1, ALU2 <- G2/G3, ALU4 <- G4/G5, ALU6 <- G6/G7.
    static constexpr std::size_t fixed_input_group_for_stage(
        std::size_t stage, bool rhs_port)
    {
        return block_for_stage(stage) * 2 + (rhs_port ? 1 : 0);
    }

    void validate_broadcast_instruction(
        std::size_t stage, const VxmLaneAluInstruction& instruction) const
    {
        check_stage(stage);
        validate_instruction(stage, instruction, chain_depth_);
    }

    void validate_broadcast_instruction(
        VxmChainDepth depth, std::size_t stage,
        const VxmLaneAluInstruction& instruction) const
    {
        check_stage(stage);
        validate_instruction(stage, instruction, depth);
    }

    void set_stream_inputs(const StreamBytes& streams)
    {
        if (stream_inputs_valid_) throw std::logic_error("VXM lane input already set for this cycle");
        stream_inputs_ = streams;
        stream_inputs_valid_ = true;
    }

    // Loads a row scalar produced by an earlier scheduled phase into the
    // small local register beside C1/C3.  This models explicit scalar feedback,
    // not a general ALU-to-ALU crossbar.
    void load_local_scalar(std::size_t stage, float value)
    {
        check_stage(stage);
        if (!datapath_idle()) {
            throw std::logic_error("cannot load a local scalar while VXM lane is active");
        }
        if (stage % 4 != 1 && stage % 4 != 3) {
            throw std::invalid_argument("only C1/C3 own a local scalar register");
        }
        accumulators_[stage] = value;
    }

    const std::optional<float>& local_scalar(std::size_t stage) const
    {
        check_stage(stage);
        return accumulators_[stage];
    }

    void set_swiglu_input(const std::array<std::uint8_t, 4>& gate,
                          const std::array<std::uint8_t, 4>& up)
    {
        auto streams = StreamBytes{};
        for (std::size_t byte = 0; byte < 4; ++byte) {
            streams[byte] = gate[byte];
            streams[4 + byte] = up[byte];
        }
        set_stream_inputs(streams);
    }

    using Configs = VxmLaneConfigs;
    using ExecutionMask = VxmLaneExecutionMask;

    // A Lane owns no instruction FIFO, decoder, or repeat counter.  The
    // Superlane broadcasts its already-decoded Current Config Registers here.
    ExecutionMask tick(const Configs& issued)
    {
        output_.reset();
        outputs_.clear();
        last_trace_.fill({});
        if (stream_inputs_valid_) {
            bool decoded_head_available = false;
            for (std::size_t stage = 0; stage < kAluCount; ++stage) {
                decoded_head_available =
                    decoded_head_available
                    || (is_chain_head(stage) && issued[stage].has_value());
            }
            if (!decoded_head_available) {
                throw std::logic_error(
                    "VXM Data arrived before a decoded Superlane configuration");
            }
        }
        const auto executes = execution_mask(issued);
        auto consumed = ExecutionMask{};
        auto next_inputs = std::array<std::optional<Token>, kAluCount>{};
        auto basic_requests =
            std::array<std::optional<BasicPipeline::Request>, kAluCount>{};
        auto special_requests =
            std::array<std::optional<SpecialPipeline::Request>, kAluCount>{};
        auto waiting_for_input = std::array<bool, kAluCount>{};
        auto activity = CycleActivity{};
        activity.cycle = cycle_;
        activity.chain_depth = chain_depth_;

        // Build this cycle's requests. The instruction is consumed when the
        // corresponding ALU accepts its input, not when a multi-cycle result
        // later leaves the internal pipeline.
        for (std::size_t stage = 0; stage < kAluCount; ++stage) {
            if (!issued[stage]) {
                if (stage_inputs_[stage]) next_inputs[stage] = stage_inputs_[stage];
                continue;
            }
            const auto& instruction = *issued[stage];
            std::optional<Token> source;
            if (is_chain_head(stage)) {
                if (!head_operands_ready(instruction)) {
                    waiting_for_input[stage] = true;
                    continue;
                }
                const auto lhs = read_head_operand(instruction.lhs, stage, false);
                const auto rhs = read_head_operand(instruction.rhs, stage, true);
                source = Token {0.0f, lhs, rhs, true};
            } else {
                source = stage_inputs_[stage];
                if (!source) {
                    waiting_for_input[stage] = true;
                    continue;
                }
            }

            prepare_accumulator(stage, instruction);
            const auto a = read_operand(instruction.lhs, *source, stage, false);
            const auto b = read_operand(instruction.rhs, *source, stage, true);
            auto metadata = ExecutionMetadata{*source, instruction};
            if (const auto* special =
                    std::get_if<VxmSpecialAluOpcode>(&instruction.operation)) {
                if (!basic_pipelines_[stage].empty()) {
                    throw std::logic_error(
                        "VXM configuration changed to LUT while a Basic ALU "
                        "result remained in flight");
                }
                special_requests[stage] = SpecialPipeline::Request{
                    *special, a, std::move(metadata)};
            } else {
                if (!special_pipelines_[stage].empty()) {
                    throw std::logic_error(
                        "VXM configuration changed to Basic while a LUT "
                        "result remained in flight");
                }
                basic_requests[stage] = BasicPipeline::Request{
                    {std::get<VxmAluOpcode>(instruction.operation),
                     instruction.precision},
                    a, b, std::move(metadata)};
            }
            consumed[stage] = true;
        }

        // Advance every physical ALU once. Basic Multiply and LUT operations
        // retain their own internal pipeline state in alu.hpp/special_alu.hpp.
        for (std::size_t stage = 0; stage < kAluCount; ++stage) {
            const auto basic_was_busy = !basic_pipelines_[stage].empty();
            const auto special_was_busy = !special_pipelines_[stage].empty();
            auto basic_result =
                basic_pipelines_[stage].tick(std::move(basic_requests[stage]));
            auto special_result = special_pipelines_[stage].tick(
                *special_alu_, std::move(special_requests[stage]));

            if (basic_result && special_result) {
                throw std::logic_error(
                    "VXM Basic and LUT pipelines completed on the same ALU cycle");
            }

            const auto accepted = consumed[stage];
            const auto active = accepted || basic_was_busy || special_was_busy;
            if (active) {
                activity.states[stage] = VxmLaneAluTraceState::Executed;
                last_trace_[stage].state = VxmLaneAluTraceState::Executed;
                ++statistics_.executed_slots;
                ++statistics_.stage_executions[stage];

                const auto accepted_useful = issued[stage]
                    && is_useful_operation(issued[stage]->operation);
                const auto internally_useful =
                    basic_was_busy || special_was_busy;
                if (accepted_useful || internally_useful) {
                    activity.useful[stage] = true;
                    ++statistics_.useful_slots;
                }
            } else if (waiting_for_input[stage]) {
                activity.states[stage] = VxmLaneAluTraceState::Stalled;
                last_trace_[stage].state = VxmLaneAluTraceState::Stalled;
            }

            std::optional<CompletedOperation> completed;
            if (basic_result) {
                completed = CompletedOperation{
                    basic_result->value, std::move(basic_result->metadata)};
            } else if (special_result) {
                completed = CompletedOperation{
                    special_result->value, std::move(special_result->metadata)};
            }
            if (!completed) continue;

            last_trace_[stage].result = completed->value;
            const auto& instruction = completed->metadata.instruction;
            if (instruction.accumulator_write) {
                accumulators_[stage] = completed->value;
            }

            const auto emit = !instruction.accumulator_write
                           || instruction.accumulator_emit;
            if (!emit) continue;

            const auto& source = completed->metadata.source;
            auto token = Token{
                completed->value, source.original, source.auxiliary, true};
            if (is_chain_tail(stage)) {
                output_registers_[block_for_stage(stage)] = completed->value;
                if (instruction.output_stream) {
                    emit_output(stage, instruction, completed->value);
                }
            } else {
                if (next_inputs[stage + 1]) {
                    throw std::logic_error(
                        "VXM static schedule collision: completing ALU"
                        + std::to_string(stage)
                        + " would overwrite the occupied input register of ALU"
                        + std::to_string(stage + 1));
                }
                next_inputs[stage + 1] = token;
            }
        }

        stage_inputs_ = std::move(next_inputs);
        stream_inputs_valid_ = false;
        if (!outputs_.empty()) output_ = outputs_.front();
        ++statistics_.cycles;
        auto& depth_statistics = statistics_.by_chain_depth[
            Statistics::depth_index(chain_depth_)];
        ++depth_statistics.cycles;
        depth_statistics.executed_slots += activity.active_slots();
        depth_statistics.useful_slots += activity.useful_slots();
        depth_statistics.peak_executed_slots = std::max<std::uint64_t>(
            depth_statistics.peak_executed_slots, activity.active_slots());
        depth_statistics.peak_useful_slots = std::max<std::uint64_t>(
            depth_statistics.peak_useful_slots, activity.useful_slots());
        for (std::size_t stage = 0; stage < kAluCount; ++stage) {
            if (activity.states[stage] == VxmLaneAluTraceState::Executed) {
                ++depth_statistics.stage_executions[stage];
            }
        }
        statistics_.timeline.push_back(activity);
        ++cycle_;
        return consumed;
    }

    const std::optional<Output>& output() const { return output_; }
    const std::vector<Output>& outputs() const { return outputs_; }
    const std::array<VxmLaneAluTrace, kAluCount>& last_trace() const { return last_trace_; }
    std::size_t last_trace_cycle() const { return cycle_ == 0 ? 0 : cycle_ - 1; }
    std::size_t cycle() const { return cycle_; }
    const Statistics& statistics() const { return statistics_; }
    void reset_statistics() { statistics_ = {}; }
    bool idle() const
    {
        return datapath_idle();
    }

    bool datapath_idle() const
    {
        for (const auto& token : stage_inputs_) if (token) return false;
        for (const auto& pipeline : basic_pipelines_) {
            if (!pipeline.empty()) return false;
        }
        for (const auto& pipeline : special_pipelines_) {
            if (!pipeline.empty()) return false;
        }
        return true;
    }

    const std::optional<float>& output_register(std::size_t block) const
    {
        if (block >= kBlockCount) throw std::out_of_range("VXM output block is outside 0..3");
        return output_registers_[block];
    }

    void print_last_trace(std::ostream& os) const
    {
        os << "vxm lane cycle " << last_trace_cycle() << '\n';
        for (std::size_t stage = 0; stage < kAluCount; ++stage) {
            os << "  C" << (stage % 4) << " alu" << stage << ' ';
            switch (last_trace_[stage].state) {
            case VxmLaneAluTraceState::Idle: os << "idle"; break;
            case VxmLaneAluTraceState::Stalled: os << "stalled"; break;
            case VxmLaneAluTraceState::Executed:
                os << "executed";
                if (last_trace_[stage].result) {
                    os << ' ' << *last_trace_[stage].result;
                } else {
                    os << " (pipeline)";
                }
                break;
            }
            os << '\n';
        }
    }

    void print_activity_trace(std::ostream& os) const
    {
        for (const auto& activity : statistics_.timeline) {
            os << "cycle " << activity.cycle
               << " depth " << static_cast<std::size_t>(activity.chain_depth)
               << " active ";
            for (std::size_t stage = 0; stage < kAluCount; ++stage) {
                const auto state = activity.states[stage];
                os << (state == VxmLaneAluTraceState::Executed
                    ? (activity.useful[stage] ? 'U' : 'B')
                    : state == VxmLaneAluTraceState::Stalled ? 'S' : '.');
            }
            os << '\n';
        }
    }

    static std::array<std::uint8_t, 4> pack_int32(std::int32_t value)
    {
        const auto bits = static_cast<std::uint32_t>(value);
        return {static_cast<std::uint8_t>(bits), static_cast<std::uint8_t>(bits >> 8),
                static_cast<std::uint8_t>(bits >> 16), static_cast<std::uint8_t>(bits >> 24)};
    }

    static std::int32_t unpack_int32(const std::array<std::uint8_t, 4>& bytes)
    {
        const auto bits = static_cast<std::uint32_t>(bytes[0])
                        | (static_cast<std::uint32_t>(bytes[1]) << 8)
                        | (static_cast<std::uint32_t>(bytes[2]) << 16)
                        | (static_cast<std::uint32_t>(bytes[3]) << 24);
        return static_cast<std::int32_t>(bits);
    }

    static float unpack_float32(const std::array<std::uint8_t, 4>& bytes)
    {
        const auto bits = static_cast<std::uint32_t>(bytes[0])
                        | (static_cast<std::uint32_t>(bytes[1]) << 8)
                        | (static_cast<std::uint32_t>(bytes[2]) << 16)
                        | (static_cast<std::uint32_t>(bytes[3]) << 24);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    static const char* operation_name(const VxmLaneOperation& operation)
    {
        if (const auto* opcode = std::get_if<VxmAluOpcode>(&operation)) {
            switch (*opcode) {
        case VxmAluOpcode::Bypass: return "bypass";
        case VxmAluOpcode::Add: return "add";
        case VxmAluOpcode::Subtract: return "sub";
        case VxmAluOpcode::Multiply: return "mul";
        case VxmAluOpcode::Negate: return "neg";
        case VxmAluOpcode::Max: return "max";
            }
        }
        if (const auto* opcode = std::get_if<VxmSpecialAluOpcode>(&operation)) {
            switch (*opcode) {
            case VxmSpecialAluOpcode::Exp: return "exp_lut";
            case VxmSpecialAluOpcode::Reciprocal: return "reciprocal_lut";
            case VxmSpecialAluOpcode::Rsqrt: return "rsqrt_lut";
            }
        }
        return "unknown";
    }

    static const char* opcode_name(const VxmLaneOperation& operation)
    {
        return operation_name(operation);
    }

    static bool supports(std::size_t stage, const VxmLaneOperation& operation)
    {
        if (std::holds_alternative<VxmAluOpcode>(operation)) return true;
        const auto column = stage % 4;
        if (const auto* special = std::get_if<VxmSpecialAluOpcode>(&operation)) {
            return (column == 1 && *special == VxmSpecialAluOpcode::Exp)
                || (column == 3 && (*special == VxmSpecialAluOpcode::Reciprocal
                                 || *special == VxmSpecialAluOpcode::Rsqrt));
        }
        return false;
    }

    static bool is_useful_operation(const VxmLaneOperation& operation)
    {
        const auto* basic = std::get_if<VxmAluOpcode>(&operation);
        return basic == nullptr || *basic != VxmAluOpcode::Bypass;
    }

private:
    struct Token {
        float value{0.0f};
        float original{0.0f};
        float auxiliary{0.0f};
        bool valid{false};
    };

    struct ExecutionMetadata {
        Token source{};
        VxmLaneAluInstruction instruction{};
    };

    struct CompletedOperation {
        float value{0.0f};
        ExecutionMetadata metadata{};
    };

    using BasicPipeline = VxmAlu::Pipeline<ExecutionMetadata>;
    using SpecialPipeline = VxmSpecialAlu::Pipeline<ExecutionMetadata>;

    static void check_stage(std::size_t stage)
    {
        if (stage >= kAluCount) throw std::out_of_range("VXM ALU stage is outside 0..7");
    }

    static bool is_stream(VxmLaneOperandKind kind)
    {
        return kind == VxmLaneOperandKind::StreamInt32
            || kind == VxmLaneOperandKind::StreamFloat32;
    }

    static bool is_chain_head_for(std::size_t stage, VxmChainDepth depth)
    {
        return stage % static_cast<std::size_t>(depth) == 0;
    }

    static bool is_chain_tail_for(std::size_t stage, VxmChainDepth depth)
    {
        return stage % static_cast<std::size_t>(depth)
            == static_cast<std::size_t>(depth) - 1;
    }

    void validate_instruction(std::size_t stage,
                              const VxmLaneAluInstruction& instruction,
                              VxmChainDepth depth) const
    {
        if (!supports(stage, instruction.operation)) {
            throw std::invalid_argument("opcode is not implemented by this C0/C1/C2/C3 position");
        }
        if (instruction.accumulator_reset && !instruction.accumulator_write) {
            throw std::invalid_argument("accumulator reset requires accumulator write");
        }
        if (instruction.accumulator_write) {
            const auto* opcode = std::get_if<VxmAluOpcode>(&instruction.operation);
            if ((stage % 4 != 1 && stage % 4 != 3)
                || opcode == nullptr
                || (*opcode != VxmAluOpcode::Add && *opcode != VxmAluOpcode::Max)
                || instruction.rhs.kind != VxmLaneOperandKind::Accumulator) {
                throw std::invalid_argument(
                    "only C1/C3 Basic Add/Max may write their local accumulator");
            }
        } else if (instruction.rhs.kind == VxmLaneOperandKind::Accumulator
                   && stage % 4 != 1 && stage % 4 != 3) {
            throw std::invalid_argument("only C1/C3 may read their local scalar register");
        }
        if (is_chain_head_for(stage, depth)) {
            const auto allowed = [](VxmLaneOperandKind kind) {
                return is_stream(kind) || kind == VxmLaneOperandKind::Immediate;
            };
            if (!allowed(instruction.lhs.kind) || !allowed(instruction.rhs.kind)) {
                throw std::invalid_argument("chain head operands must come from its fixed Stream input or immediate");
            }
        } else {
            if (instruction.lhs.kind != VxmLaneOperandKind::Previous) {
                throw std::invalid_argument("internal stage lhs is fixed to the preceding pipeline register");
            }
            const auto rhs = instruction.rhs.kind;
            const auto rhs_allowed = rhs == VxmLaneOperandKind::Original
                                  || rhs == VxmLaneOperandKind::Auxiliary
                                  || rhs == VxmLaneOperandKind::Immediate
                                  || ((stage % 4 == 1 || stage % 4 == 3)
                                      && rhs == VxmLaneOperandKind::Accumulator);
            if (!rhs_allowed) {
                throw std::invalid_argument("internal stage rhs is outside its small local mux");
            }
        }
        if (instruction.output_stream) {
            if (!is_chain_tail_for(stage, depth)) {
                throw std::invalid_argument("only a configured chain tail can drive an output register");
            }
            if (*instruction.output_stream != fixed_output_stream_for_stage(stage)) {
                throw std::invalid_argument("VXM output register has a fixed stream-group binding");
            }
            if (instruction.output_type == VxmCastTarget::Int8 && instruction.output_scale == 0.0f) {
                throw std::invalid_argument("VXM output quantization scale must be non-zero");
            }
        }
    }

    bool head_operands_ready(const VxmLaneAluInstruction& instruction) const
    {
        return (!is_stream(instruction.lhs.kind) && !is_stream(instruction.rhs.kind))
            || stream_inputs_valid_;
    }

    ExecutionMask execution_mask(const Configs& issued) const
    {
        auto executes = ExecutionMask{};
        for (std::size_t stage = 0; stage < kAluCount; ++stage) {
            if (!issued[stage]) continue;
            executes[stage] = is_chain_head(stage)
                ? head_operands_ready(*issued[stage])
                : stage_inputs_[stage].has_value();
        }
        return executes;
    }

    static bool instruction_emits(const VxmLaneAluInstruction& instruction)
    {
        return !instruction.accumulator_write || instruction.accumulator_emit;
    }

    void validate_static_schedule(const Configs& issued,
                                  const ExecutionMask& executes) const
    {
        for (std::size_t stage = 0; stage < kAluCount; ++stage) {
            if (!executes[stage] || is_chain_tail(stage)) continue;
            const auto& instruction = *issued[stage];
            if (!instruction_emits(instruction)) continue;

            const auto downstream = stage + 1;
            if (stage_inputs_[downstream].has_value() && !executes[downstream]) {
                throw std::logic_error(
                    "VXM static schedule collision: ALU"
                    + std::to_string(stage)
                    + " would overwrite the occupied input register of ALU"
                    + std::to_string(downstream));
            }
        }
    }

    float read_head_operand(const VxmLaneOperand& operand, std::size_t stage,
                            bool rhs_port) const
    {
        // No Stream selector: the physical group is fixed by stage and port.
        // The only input MUX selects this fixed Stream value or Immediate.
        if (operand.kind == VxmLaneOperandKind::Immediate) return operand.immediate;
        std::array<std::uint8_t, kStreamGroupBytes> bytes{};
        const auto base = fixed_input_group_for_stage(stage, rhs_port)
                        * kStreamGroupBytes;
        for (std::size_t byte = 0; byte < kStreamGroupBytes; ++byte) {
            bytes[byte] = stream_inputs_[base + byte];
        }
        if (operand.kind == VxmLaneOperandKind::StreamInt32) {
            return static_cast<float>(unpack_int32(bytes) - operand.zero_point) * operand.scale;
        }
        if (operand.kind == VxmLaneOperandKind::StreamFloat32) {
            return unpack_float32(bytes) * operand.scale;
        }
        throw std::logic_error("unsupported operand used at VXM chain head");
    }

    void prepare_accumulator(std::size_t stage,
                             const VxmLaneAluInstruction& instruction)
    {
        if (!instruction.accumulator_write) return;
        if (instruction.accumulator_reset) {
            const auto opcode = std::get<VxmAluOpcode>(instruction.operation);
            accumulators_[stage] = opcode == VxmAluOpcode::Add
                ? 0.0f : -std::numeric_limits<float>::infinity();
            return;
        }
        if (!accumulators_[stage].has_value()) {
            throw std::logic_error(
                "VXM accumulator used before an initializing reset instruction");
        }
    }

    float read_operand(const VxmLaneOperand& operand, const Token& token,
                       std::size_t stage, bool rhs_port) const
    {
        switch (operand.kind) {
        case VxmLaneOperandKind::Previous: return token.value;
        case VxmLaneOperandKind::Original: return token.original;
        case VxmLaneOperandKind::Auxiliary: return token.auxiliary;
        case VxmLaneOperandKind::Accumulator:
            if (!accumulators_[stage]) {
                throw std::logic_error("VXM local scalar read before it was initialized");
            }
            return *accumulators_[stage];
        case VxmLaneOperandKind::Immediate: return operand.immediate;
        case VxmLaneOperandKind::StreamInt32:
        case VxmLaneOperandKind::StreamFloat32:
            return read_head_operand(operand, stage, rhs_port);
        }
        throw std::logic_error("unsupported VXM operand");
    }

    void emit_output(std::size_t stage, const VxmLaneAluInstruction& instruction, float result)
    {
        auto output = Output{};
        output.stream = fixed_output_stream_for_stage(stage);
        switch (instruction.output_type) {
        case VxmCastTarget::Int8: {
            output.value = VxmDataFormat::quantize_int8(result, instruction.output_scale,
                                                        instruction.output_zero_point);
            output.bytes[0] = static_cast<std::uint8_t>(output.value);
            output.byte_count = 1;
            break;
        }
        case VxmCastTarget::Float16: {
            const auto bits = VxmDataFormat::float_to_fp16_bits(result);
            output.bytes[0] = static_cast<std::uint8_t>(bits);
            output.bytes[1] = static_cast<std::uint8_t>(bits >> 8);
            output.byte_count = 2;
            break;
        }
        case VxmCastTarget::Float32: {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &result, sizeof(bits));
            for (std::size_t byte = 0; byte < 4; ++byte) output.bytes[byte] = static_cast<std::uint8_t>(bits >> (8 * byte));
            output.byte_count = 4;
            break;
        }
        }
        outputs_.push_back(output);
    }

    VxmChainDepth chain_depth_{VxmChainDepth::Eight};
    std::shared_ptr<VxmSpecialAlu> special_alu_;
    std::array<std::optional<Token>, kAluCount> stage_inputs_{};
    std::array<BasicPipeline, kAluCount> basic_pipelines_{};
    std::array<SpecialPipeline, kAluCount> special_pipelines_{};
    std::array<std::optional<float>, kAluCount> accumulators_{};
    std::array<std::optional<float>, kBlockCount> output_registers_{};
    StreamBytes stream_inputs_{};
    bool stream_inputs_valid_{false};
    std::optional<Output> output_{};
    std::vector<Output> outputs_{};
    std::array<VxmLaneAluTrace, kAluCount> last_trace_{};
    Statistics statistics_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
