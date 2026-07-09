#pragma once

#include "ftlpu/vxm/alu.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ftlpu {

enum class VxmLaneOperandKind {
    AluOutput,
    StreamInt32,
    Immediate,
};

struct VxmLaneOperand {
    VxmLaneOperandKind kind{VxmLaneOperandKind::Immediate};
    std::size_t index{0};
    float immediate{0.0f};
    float scale{1.0f};

    static VxmLaneOperand Alu(std::size_t alu)
    {
        return VxmLaneOperand {VxmLaneOperandKind::AluOutput, alu, 0.0f, 1.0f};
    }

    static VxmLaneOperand StreamInt32(std::size_t base_stream)
    {
        return VxmLaneOperand {VxmLaneOperandKind::StreamInt32, base_stream, 0.0f, 1.0f};
    }

    static VxmLaneOperand Imm(float value)
    {
        return VxmLaneOperand {VxmLaneOperandKind::Immediate, 0, value, 1.0f};
    }
};

struct VxmLaneAluInstruction {
    VxmAluOpcode opcode{VxmAluOpcode::Pass};
    VxmLaneOperand lhs{VxmLaneOperand::Imm(0.0f)};
    VxmLaneOperand rhs{VxmLaneOperand::Imm(0.0f)};
    float scale{1.0f};
    std::int32_t output_zero_point{0};
    VxmCastTarget cast_target{VxmCastTarget::Float32};
    std::optional<std::size_t> output_stream{};
};

enum class VxmLaneAluTraceState {
    Idle,
    Stalled,
    Executed,
};

struct VxmLaneOperandTrace {
    std::string source{};
    std::optional<float> value{};
};

struct VxmLaneAluTrace {
    VxmLaneAluTraceState state{VxmLaneAluTraceState::Idle};
    VxmAluOpcode opcode{VxmAluOpcode::Pass};
    std::size_t queue_depth_before{0};
    std::size_t queue_depth_after{0};
    VxmLaneOperandTrace lhs{};
    VxmLaneOperandTrace rhs{};
    std::optional<float> value{};
    std::optional<std::int8_t> output{};
};

class VxmLane {
public:
    static constexpr std::size_t kAluCount = 16;
    static constexpr std::size_t kInputStreams = hw::kStreams;
    static constexpr std::size_t kAluStages = kAluCount;

    using Byte = std::uint8_t;
    using Int32Bytes = std::array<Byte, 4>;
    using StreamBytes = std::array<Byte, kInputStreams>;

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

    struct Output {
        std::int8_t value{0};
        std::size_t stream{0};
    };

    void reset()
    {
        for (auto& queue : queues_) {
            queue.clear();
        }
        for (auto& output : alu_outputs_) {
            output.reset();
        }
        pending_streams_.reset();
        streams_.reset();
        output_.reset();
        reset_trace();
        cycle_ = 0;
    }

    void clear_queues()
    {
        for (auto& queue : queues_) {
            queue.clear();
        }
    }

    void enqueue_instruction(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        check_alu(alu);
        queues_[alu].push_back(instruction);
    }

    void load_swiglu_program(const SwigluParams& params)
    {
        load_pipelined_swiglu_program(params, 1);
    }

    void load_pipelined_swiglu_program(
        const SwigluParams& params,
        std::size_t token_count,
        std::size_t gate_stream_base = 0,
        std::size_t up_stream_base = 4,
        std::size_t output_stream = 0)
    {
        clear_queues();
        swiglu_params_ = params;

        for (std::size_t token = 0; token < token_count; ++token) {
            enqueue_instruction(0, VxmLaneAluInstruction {
                VxmAluOpcode::Cast,
                VxmLaneOperand::StreamInt32(gate_stream_base),
                VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                VxmCastTarget::Float32,
            });
            enqueue_instruction(1, VxmLaneAluInstruction {
                VxmAluOpcode::Cast,
                VxmLaneOperand::StreamInt32(up_stream_base),
                VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                VxmCastTarget::Float32,
            });
            enqueue_instruction(2, VxmLaneAluInstruction {
                VxmAluOpcode::Multiply,
                VxmLaneOperand::Alu(0),
                VxmLaneOperand::Imm(params.gate_scale),
            });
            enqueue_instruction(3, VxmLaneAluInstruction {
                VxmAluOpcode::Multiply,
                VxmLaneOperand::Alu(1),
                VxmLaneOperand::Imm(params.up_scale),
            });
            enqueue_instruction(4, VxmLaneAluInstruction {
                VxmAluOpcode::Multiply,
                VxmLaneOperand::Alu(2),
                VxmLaneOperand::Alu(3),
            });
            enqueue_instruction(5, VxmLaneAluInstruction {
                VxmAluOpcode::Negate,
                VxmLaneOperand::Alu(2),
            });
            enqueue_instruction(6, VxmLaneAluInstruction {
                VxmAluOpcode::Exp,
                VxmLaneOperand::Alu(5),
            });
            enqueue_instruction(7, VxmLaneAluInstruction {
                VxmAluOpcode::Add,
                VxmLaneOperand::Alu(6),
                VxmLaneOperand::Imm(1.0f),
            });
            enqueue_instruction(8, VxmLaneAluInstruction {
                VxmAluOpcode::Divide,
                VxmLaneOperand::Imm(1.0f),
                VxmLaneOperand::Alu(7),
            });
            enqueue_instruction(9, VxmLaneAluInstruction {
                VxmAluOpcode::Pass,
                VxmLaneOperand::Alu(4),
            });
            enqueue_instruction(10, VxmLaneAluInstruction {
                VxmAluOpcode::Pass,
                VxmLaneOperand::Alu(9),
            });
            enqueue_instruction(11, VxmLaneAluInstruction {
                VxmAluOpcode::Pass,
                VxmLaneOperand::Alu(10),
            });
            enqueue_instruction(12, VxmLaneAluInstruction {
                VxmAluOpcode::Multiply,
                VxmLaneOperand::Alu(11),
                VxmLaneOperand::Alu(8),
            });
            enqueue_instruction(13, VxmLaneAluInstruction {
                VxmAluOpcode::Multiply,
                VxmLaneOperand::Alu(12),
                VxmLaneOperand::Imm(1.0f / params.output_scale),
            });
            enqueue_instruction(14, VxmLaneAluInstruction {
                VxmAluOpcode::Add,
                VxmLaneOperand::Alu(13),
                VxmLaneOperand::Imm(static_cast<float>(params.output_zero_point)),
            });
            enqueue_instruction(15, VxmLaneAluInstruction {
                VxmAluOpcode::Cast,
                VxmLaneOperand::Alu(14),
                VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                VxmCastTarget::Int8,
                output_stream,
            });
        }
    }

    void load_pipelined_add_quant_program(
        const AddQuantParams& params,
        std::size_t token_count,
        std::size_t lhs_stream_base = 0,
        std::size_t rhs_stream_base = 4,
        std::size_t output_stream = 0)
    {
        clear_queues();

        for (std::size_t token = 0; token < token_count; ++token) {
            enqueue_instruction(0, VxmLaneAluInstruction {
                VxmAluOpcode::Cast,
                VxmLaneOperand::StreamInt32(lhs_stream_base),
                VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                VxmCastTarget::Float32,
            });
            enqueue_instruction(1, VxmLaneAluInstruction {
                VxmAluOpcode::Cast,
                VxmLaneOperand::StreamInt32(rhs_stream_base),
                VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                VxmCastTarget::Float32,
            });
            enqueue_instruction(2, VxmLaneAluInstruction {
                VxmAluOpcode::Multiply,
                VxmLaneOperand::Alu(0),
                VxmLaneOperand::Imm(params.lhs_scale),
            });
            enqueue_instruction(3, VxmLaneAluInstruction {
                VxmAluOpcode::Multiply,
                VxmLaneOperand::Alu(1),
                VxmLaneOperand::Imm(params.rhs_scale),
            });
            enqueue_instruction(4, VxmLaneAluInstruction {
                VxmAluOpcode::Add,
                VxmLaneOperand::Alu(2),
                VxmLaneOperand::Alu(3),
            });
            enqueue_instruction(5, VxmLaneAluInstruction {
                VxmAluOpcode::Multiply,
                VxmLaneOperand::Alu(4),
                VxmLaneOperand::Imm(1.0f / params.output_scale),
            });
            enqueue_instruction(6, VxmLaneAluInstruction {
                VxmAluOpcode::Add,
                VxmLaneOperand::Alu(5),
                VxmLaneOperand::Imm(static_cast<float>(params.output_zero_point)),
            });
            enqueue_instruction(7, VxmLaneAluInstruction {
                VxmAluOpcode::Cast,
                VxmLaneOperand::Alu(6),
                VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                VxmCastTarget::Int8,
                output_stream,
            });
        }
    }

    void set_stream_inputs(StreamBytes streams)
    {
        if (pending_streams_.has_value()) {
            throw std::logic_error("VXM lane stream input is already occupied");
        }
        pending_streams_ = streams;
    }

    void set_swiglu_input(Int32Bytes gate_streams, Int32Bytes up_streams)
    {
        StreamBytes streams{};
        for (std::size_t byte = 0; byte < 4; ++byte) {
            streams[byte] = gate_streams[byte];
            streams[4 + byte] = up_streams[byte];
        }
        set_stream_inputs(streams);
    }

    void tick()
    {
        output_.reset();
        reset_trace();
        last_trace_cycle_ = cycle_;
        if (pending_streams_.has_value()) {
            streams_ = pending_streams_;
            pending_streams_.reset();
        }

        const auto previous_outputs = alu_outputs_;
        auto next_outputs = alu_outputs_;
        for (std::size_t alu = 0; alu < kAluCount; ++alu) {
            auto& trace = last_trace_[alu];
            trace.queue_depth_before = queues_[alu].size();
            trace.queue_depth_after = queues_[alu].size();
            if (queues_[alu].empty()) {
                continue;
            }

            const auto& instruction = queues_[alu].front();
            trace.opcode = instruction.opcode;
            trace.lhs = trace_operand(instruction.lhs, previous_outputs);
            trace.rhs = trace_operand(instruction.rhs, previous_outputs);
            const auto result = try_execute(instruction, previous_outputs);
            if (!result.has_value()) {
                trace.state = VxmLaneAluTraceState::Stalled;
                continue;
            }

            trace.state = VxmLaneAluTraceState::Executed;
            trace.value = result->value;
            next_outputs[alu] = result->value;
            if (result->output_valid && instruction.output_stream.has_value()) {
                output_ = Output {result->output, *instruction.output_stream};
                trace.output = result->output;
            }
            queues_[alu].pop_front();
            trace.queue_depth_after = queues_[alu].size();
        }

        alu_outputs_ = next_outputs;
        ++cycle_;
    }

    const std::optional<Output>& output() const
    {
        return output_;
    }

    const std::optional<Output>& last_output() const
    {
        return output_;
    }

    std::optional<float> alu_output(std::size_t alu) const
    {
        check_alu(alu);
        return alu_outputs_[alu];
    }

    std::optional<float> stage_value(std::size_t stage) const
    {
        return alu_output(stage);
    }

    std::size_t queue_depth(std::size_t alu) const
    {
        check_alu(alu);
        return queues_[alu].size();
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    std::size_t last_trace_cycle() const
    {
        return last_trace_cycle_;
    }

    const std::array<VxmLaneAluTrace, kAluCount>& last_trace() const
    {
        return last_trace_;
    }

    void print_last_trace(std::ostream& os) const
    {
        os << "cycle " << last_trace_cycle_ << '\n';
        for (std::size_t alu = 0; alu < kAluCount; ++alu) {
            const auto& trace = last_trace_[alu];
            os << "  alu" << std::right << std::setw(2) << std::setfill('0') << alu << std::setfill(' ');
            os << " | state=" << std::left << std::setw(5) << trace_state_name(trace.state);
            os << " | queue=" << trace.queue_depth_before << "->" << trace.queue_depth_after << " ";
            if (trace.state != VxmLaneAluTraceState::Idle) {
                os << "| op=" << std::left << std::setw(7) << opcode_name(trace.opcode);
                os << " | lhs=" << std::left << std::setw(13) << trace.lhs.source;
                os << " value=" << std::right << std::setw(10) << operand_value_text(trace.lhs.value);
                os << " | rhs=" << std::left << std::setw(13) << trace.rhs.source;
                os << " value=" << std::right << std::setw(10) << operand_value_text(trace.rhs.value);
            }
            if (trace.value.has_value()) {
                os << " | result=" << std::right << std::setw(10) << operand_value_text(trace.value);
            }
            if (trace.output.has_value()) {
                os << " | out=" << std::right << std::setw(4) << static_cast<int>(*trace.output);
            }
            os << '\n';
        }
        if (output_.has_value()) {
            os << "  lane_out=" << static_cast<int>(output_->value) << '\n';
        }
    }

    static std::int32_t unpack_int32(Int32Bytes streams)
    {
        const auto raw = static_cast<std::uint32_t>(streams[0])
            | (static_cast<std::uint32_t>(streams[1]) << 8)
            | (static_cast<std::uint32_t>(streams[2]) << 16)
            | (static_cast<std::uint32_t>(streams[3]) << 24);
        return static_cast<std::int32_t>(raw);
    }

    static Int32Bytes pack_int32(std::int32_t value)
    {
        const auto raw = static_cast<std::uint32_t>(value);
        return Int32Bytes {
            static_cast<Byte>(raw & 0xffu),
            static_cast<Byte>((raw >> 8) & 0xffu),
            static_cast<Byte>((raw >> 16) & 0xffu),
            static_cast<Byte>((raw >> 24) & 0xffu),
        };
    }

    static const char* opcode_name(VxmAluOpcode opcode)
    {
        switch (opcode) {
        case VxmAluOpcode::Pass:
            return "pass";
        case VxmAluOpcode::Add:
            return "add";
        case VxmAluOpcode::Subtract:
            return "sub";
        case VxmAluOpcode::Multiply:
            return "mul";
        case VxmAluOpcode::Divide:
            return "div";
        case VxmAluOpcode::Negate:
            return "neg";
        case VxmAluOpcode::Abs:
            return "abs";
        case VxmAluOpcode::Min:
            return "min";
        case VxmAluOpcode::Max:
            return "max";
        case VxmAluOpcode::Clamp:
            return "clamp";
        case VxmAluOpcode::Square:
            return "square";
        case VxmAluOpcode::Sqrt:
            return "sqrt";
        case VxmAluOpcode::Rsqrt:
            return "rsqrt";
        case VxmAluOpcode::Exp:
            return "exp";
        case VxmAluOpcode::Log:
            return "log";
        case VxmAluOpcode::Tanh:
            return "tanh";
        case VxmAluOpcode::Relu:
            return "relu";
        case VxmAluOpcode::Gelu:
            return "gelu";
        case VxmAluOpcode::Equal:
            return "eq";
        case VxmAluOpcode::LessThan:
            return "lt";
        case VxmAluOpcode::LessEqual:
            return "le";
        case VxmAluOpcode::Select:
            return "select";
        case VxmAluOpcode::Accumulate:
            return "acc";
        case VxmAluOpcode::Cast:
            return "cast";
        }
        return "unknown";
    }

    static const char* trace_state_name(VxmLaneAluTraceState state)
    {
        switch (state) {
        case VxmLaneAluTraceState::Idle:
            return "idle";
        case VxmLaneAluTraceState::Stalled:
            return "stall";
        case VxmLaneAluTraceState::Executed:
            return "exec";
        }
        return "unknown";
    }

    static std::string operand_value_text(const std::optional<float>& value)
    {
        if (!value.has_value()) {
            return "NA";
        }

        std::ostringstream os;
        os << std::setprecision(6) << *value;
        return os.str();
    }

private:
    struct AluResult {
        float value{0.0f};
        std::int8_t output{0};
        bool output_valid{false};
    };

    static void check_alu(std::size_t alu)
    {
        if (alu >= kAluCount) {
            throw std::out_of_range("VXM lane ALU index is outside the 16-ALU lane");
        }
    }

    void reset_trace()
    {
        for (auto& trace : last_trace_) {
            trace = VxmLaneAluTrace {};
        }
    }

    std::string operand_source_name(const VxmLaneOperand& operand) const
    {
        switch (operand.kind) {
        case VxmLaneOperandKind::AluOutput:
            return "alu" + std::to_string(operand.index);
        case VxmLaneOperandKind::StreamInt32:
            return "stream[" + std::to_string(operand.index) + ".." + std::to_string(operand.index + 3) + "]";
        case VxmLaneOperandKind::Immediate:
            return "imm(" + std::to_string(operand.immediate) + ")";
        }
        return "unknown";
    }

    VxmLaneOperandTrace trace_operand(
        const VxmLaneOperand& operand,
        const std::array<std::optional<float>, kAluCount>& previous_outputs) const
    {
        return VxmLaneOperandTrace {
            operand_source_name(operand),
            resolve_operand(operand, previous_outputs),
        };
    }

    std::optional<float> resolve_operand(
        const VxmLaneOperand& operand,
        const std::array<std::optional<float>, kAluCount>& previous_outputs) const
    {
        switch (operand.kind) {
        case VxmLaneOperandKind::AluOutput:
            check_alu(operand.index);
            return previous_outputs[operand.index];
        case VxmLaneOperandKind::StreamInt32:
            if (!streams_.has_value()) {
                return std::nullopt;
            }
            if (operand.index + 3 >= kInputStreams) {
                throw std::out_of_range("VXM lane int32 stream operand needs four streams");
            }
            return static_cast<float>(unpack_int32(Int32Bytes {
                       (*streams_)[operand.index],
                       (*streams_)[operand.index + 1],
                       (*streams_)[operand.index + 2],
                       (*streams_)[operand.index + 3],
                   }));
        case VxmLaneOperandKind::Immediate:
            return operand.immediate;
        }
        throw std::logic_error("unsupported VXM lane operand kind");
    }

    std::optional<AluResult> try_execute(
        const VxmLaneAluInstruction& instruction,
        const std::array<std::optional<float>, kAluCount>& previous_outputs) const
    {
        const auto lhs = resolve_operand(instruction.lhs, previous_outputs);
        const auto rhs = resolve_operand(instruction.rhs, previous_outputs);
        if (!lhs.has_value() || !rhs.has_value()) {
            return std::nullopt;
        }

        AluResult result{};
        switch (instruction.opcode) {
        case VxmAluOpcode::Pass:
            result.value = *lhs;
            return result;
        case VxmAluOpcode::Add:
            result.value = *lhs + *rhs;
            return result;
        case VxmAluOpcode::Subtract:
            result.value = *lhs - *rhs;
            return result;
        case VxmAluOpcode::Multiply:
            result.value = *lhs * *rhs;
            return result;
        case VxmAluOpcode::Divide:
            result.value = *lhs / *rhs;
            return result;
        case VxmAluOpcode::Negate:
            result.value = -*lhs;
            return result;
        case VxmAluOpcode::Exp:
            result.value = std::exp(*lhs);
            return result;
        case VxmAluOpcode::Cast:
            if (instruction.cast_target == VxmCastTarget::Int8) {
                result.output = VxmAlu::cast_scalar_to_int8(*lhs);
                result.value = static_cast<float>(result.output);
                result.output_valid = true;
                return result;
            }
            result.value = *lhs;
            return result;
        default:
            throw std::logic_error("VXM lane ALU opcode is not implemented in the issue-queue lane");
        }
    }

    SwigluParams swiglu_params_{};
    std::array<std::deque<VxmLaneAluInstruction>, kAluCount> queues_{};
    std::array<std::optional<float>, kAluCount> alu_outputs_{};
    std::optional<StreamBytes> pending_streams_{};
    std::optional<StreamBytes> streams_{};
    std::optional<Output> output_{};
    std::array<VxmLaneAluTrace, kAluCount> last_trace_{};
    std::size_t last_trace_cycle_{0};
    std::size_t cycle_{0};
};

} // namespace ftlpu
