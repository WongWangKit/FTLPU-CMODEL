#pragma once

#include "ftlpu/core/fp16.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/vxm/alu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <cstring>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu {

enum class VxmLaneOperandKind {
    AluOutput,
    StreamInt32,
    Immediate,
    StreamFloat32,
    StreamInt8,
    StreamFloat16,
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

    static VxmLaneOperand StreamFloat32(std::size_t base_stream)
    {
        return VxmLaneOperand {VxmLaneOperandKind::StreamFloat32, base_stream, 0.0f, 1.0f};
    }

    static VxmLaneOperand StreamInt8(std::size_t stream)
    {
        return VxmLaneOperand {VxmLaneOperandKind::StreamInt8, stream, 0.0f, 1.0f};
    }

    static VxmLaneOperand StreamFloat16(std::size_t base_stream)
    {
        return VxmLaneOperand {VxmLaneOperandKind::StreamFloat16, base_stream, 0.0f, 1.0f};
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
    Hemisphere input_hemisphere{Hemisphere::East};
    Hemisphere output_hemisphere{Hemisphere::East};
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

    struct Output {
        std::int8_t value{0};
        std::size_t stream{0};
        std::array<std::uint8_t, 4> bytes{};
        std::size_t byte_count{1};
        Hemisphere hemisphere{Hemisphere::East};
    };

    void reset()
    {
        for (auto& queue : queues_) {
            queue.clear();
        }
        for (auto& output : alu_outputs_) {
            output.reset();
        }
        for (auto& streams : pending_streams_) streams.reset();
        for (auto& streams : streams_) streams.reset();
        output_.reset();
        outputs_.clear();
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

    void set_stream_inputs(Hemisphere hemisphere, StreamBytes streams)
    {
        auto& pending = pending_streams_[hemisphere_index(hemisphere)];
        if (pending.has_value()) {
            throw std::logic_error("VXM lane stream input is already occupied");
        }
        pending = streams;
    }

    void set_stream_inputs(StreamBytes streams)
    {
        set_stream_inputs(Hemisphere::East, streams);
    }

    void tick()
    {
        output_.reset();
        outputs_.clear();
        reset_trace();
        last_trace_cycle_ = cycle_;
        for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
            if (pending_streams_[hemisphere].has_value()) {
                streams_[hemisphere] = pending_streams_[hemisphere];
                pending_streams_[hemisphere].reset();
            }
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
            trace.lhs = trace_operand(instruction.lhs, previous_outputs, instruction.input_hemisphere);
            trace.rhs = trace_operand(instruction.rhs, previous_outputs, instruction.input_hemisphere);
            const auto result = try_execute(instruction, previous_outputs);
            if (!result.has_value()) {
                trace.state = VxmLaneAluTraceState::Stalled;
                continue;
            }

            trace.state = VxmLaneAluTraceState::Executed;
            trace.value = result->value;
            next_outputs[alu] = result->value;
            if (result->output_valid && instruction.output_stream.has_value()) {
                auto output = Output {
                    result->output,
                    *instruction.output_stream,
                    result->output_bytes,
                    result->output_byte_count,
                    instruction.output_hemisphere,
                };
                if (!output_.has_value()) {
                    output_ = output;
                }
                outputs_.push_back(output);
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

    const std::vector<Output>& outputs() const
    {
        return outputs_;
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

    static float unpack_float32(Int32Bytes streams)
    {
        const auto raw = static_cast<std::uint32_t>(streams[0])
            | (static_cast<std::uint32_t>(streams[1]) << 8)
            | (static_cast<std::uint32_t>(streams[2]) << 16)
            | (static_cast<std::uint32_t>(streams[3]) << 24);
        float value = 0.0f;
        std::memcpy(&value, &raw, sizeof(value));
        return value;
    }

    static std::array<std::uint8_t, 4> pack_float32(float value)
    {
        std::uint32_t raw = 0;
        std::memcpy(&raw, &value, sizeof(raw));
        return std::array<std::uint8_t, 4> {
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
        case VxmAluOpcode::Exp:
            return "exp";
        case VxmAluOpcode::Log:
            return "log";
        case VxmAluOpcode::Relu:
            return "relu";
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
        std::array<std::uint8_t, 4> output_bytes{};
        std::size_t output_byte_count{1};
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
        case VxmLaneOperandKind::StreamFloat32:
            return "f32stream[" + std::to_string(operand.index) + ".." + std::to_string(operand.index + 3) + "]";
        case VxmLaneOperandKind::StreamInt8:
            return "i8stream[" + std::to_string(operand.index) + "]";
        case VxmLaneOperandKind::StreamFloat16:
            return "f16stream[" + std::to_string(operand.index) + ".." + std::to_string(operand.index + 1) + "]";
        case VxmLaneOperandKind::Immediate:
            return "imm(" + std::to_string(operand.immediate) + ")";
        }
        return "unknown";
    }

    VxmLaneOperandTrace trace_operand(
        const VxmLaneOperand& operand,
        const std::array<std::optional<float>, kAluCount>& previous_outputs,
        Hemisphere input_hemisphere) const
    {
        return VxmLaneOperandTrace {
            operand_source_name(operand),
            resolve_operand(operand, previous_outputs, input_hemisphere),
        };
    }

    std::optional<float> resolve_operand(
        const VxmLaneOperand& operand,
        const std::array<std::optional<float>, kAluCount>& previous_outputs,
        Hemisphere input_hemisphere) const
    {
        const auto& stream_input = streams_[hemisphere_index(input_hemisphere)];
        switch (operand.kind) {
        case VxmLaneOperandKind::AluOutput:
            check_alu(operand.index);
            return previous_outputs[operand.index];
        case VxmLaneOperandKind::StreamInt32:
            if (!stream_input.has_value()) {
                return std::nullopt;
            }
            if (operand.index + 3 >= kInputStreams) {
                throw std::out_of_range("VXM lane int32 stream operand needs four streams");
            }
            return static_cast<float>(unpack_int32(Int32Bytes {
                       (*stream_input)[operand.index],
                       (*stream_input)[operand.index + 1],
                       (*stream_input)[operand.index + 2],
                       (*stream_input)[operand.index + 3],
                   }));
        case VxmLaneOperandKind::StreamFloat32:
            if (!stream_input.has_value()) {
                return std::nullopt;
            }
            if (operand.index + 3 >= kInputStreams) {
                throw std::out_of_range("VXM lane float32 stream operand needs four streams");
            }
            return unpack_float32(Int32Bytes {
                (*stream_input)[operand.index],
                (*stream_input)[operand.index + 1],
                (*stream_input)[operand.index + 2],
                (*stream_input)[operand.index + 3],
            });
        case VxmLaneOperandKind::StreamInt8:
            if (!stream_input.has_value()) return std::nullopt;
            if (operand.index >= kInputStreams) throw std::out_of_range("VXM lane int8 stream operand is outside the stream set");
            return static_cast<float>(static_cast<std::int8_t>((*stream_input)[operand.index]));
        case VxmLaneOperandKind::StreamFloat16:
            if (!stream_input.has_value()) return std::nullopt;
            if (operand.index + 1 >= kInputStreams) throw std::out_of_range("VXM lane fp16 stream operand needs two streams");
            return Fp16::from_bits(static_cast<std::uint16_t>((*stream_input)[operand.index])
                | (static_cast<std::uint16_t>((*stream_input)[operand.index + 1]) << 8)).to_float();
        case VxmLaneOperandKind::Immediate:
            return operand.immediate;
        }
        throw std::logic_error("unsupported VXM lane operand kind");
    }

    std::optional<AluResult> try_execute(
        const VxmLaneAluInstruction& instruction,
        const std::array<std::optional<float>, kAluCount>& previous_outputs) const
    {
        const auto lhs = resolve_operand(instruction.lhs, previous_outputs, instruction.input_hemisphere);
        const auto rhs = resolve_operand(instruction.rhs, previous_outputs, instruction.input_hemisphere);
        if (!lhs.has_value() || !rhs.has_value()) {
            return std::nullopt;
        }

        AluResult result{};
        switch (instruction.opcode) {
        case VxmAluOpcode::Pass:
            result.value = *lhs;
            break;
        case VxmAluOpcode::Add:
            result.value = *lhs + *rhs;
            break;
        case VxmAluOpcode::Subtract:
            result.value = *lhs - *rhs;
            break;
        case VxmAluOpcode::Multiply:
            result.value = *lhs * *rhs;
            break;
        case VxmAluOpcode::Divide:
            result.value = *lhs / *rhs;
            break;
        case VxmAluOpcode::Max:
            result.value = std::max(*lhs, *rhs);
            break;
        case VxmAluOpcode::Min:
            result.value = std::min(*lhs, *rhs);
            break;
        case VxmAluOpcode::Negate:
            result.value = -*lhs;
            break;
        case VxmAluOpcode::Abs:
            result.value = std::fabs(*lhs);
            break;
        case VxmAluOpcode::Square:
            result.value = *lhs * *lhs;
            break;
        case VxmAluOpcode::Sqrt:
            result.value = std::sqrt(*lhs);
            break;
        case VxmAluOpcode::Exp:
            result.value = std::exp(*lhs);
            break;
        case VxmAluOpcode::Log:
            result.value = std::log(*lhs);
            break;
        case VxmAluOpcode::Relu:
            result.value = std::max(0.0f, *lhs);
            break;
        case VxmAluOpcode::Cast:
            if (instruction.cast_target == VxmCastTarget::Int8) {
                result.value = static_cast<float>(VxmAlu::cast_scalar_to_int8(*lhs));
            } else {
                result.value = *lhs;
            }
            break;
        default:
            throw std::logic_error("VXM lane ALU opcode is not implemented in the issue-queue lane");
        }

        if (!instruction.output_stream.has_value()) {
            return result;
        }

        switch (instruction.cast_target) {
        case VxmCastTarget::Int8:
            result.output = VxmAlu::cast_scalar_to_int8(result.value);
            result.output_bytes[0] = static_cast<std::uint8_t>(result.output);
            result.output_byte_count = 1;
            break;
        case VxmCastTarget::Float16: {
            const auto bits = VxmAlu::cast_scalar_to_float16_bits(result.value);
            result.output = static_cast<std::int8_t>(bits & 0xffu);
            result.output_bytes[0] = static_cast<std::uint8_t>(bits & 0xffu);
            result.output_bytes[1] = static_cast<std::uint8_t>((bits >> 8) & 0xffu);
            result.output_byte_count = 2;
            break;
        }
        case VxmCastTarget::Float32:
            result.output = 0;
            result.output_bytes = pack_float32(result.value);
            result.output_byte_count = 4;
            break;
        }
        result.output_valid = true;
        return result;
    }

    std::array<std::deque<VxmLaneAluInstruction>, kAluCount> queues_{};
    std::array<std::optional<float>, kAluCount> alu_outputs_{};
    std::array<std::optional<StreamBytes>, hw::kHemispheres> pending_streams_{};
    std::array<std::optional<StreamBytes>, hw::kHemispheres> streams_{};
    std::optional<Output> output_{};
    std::vector<Output> outputs_{};
    std::array<VxmLaneAluTrace, kAluCount> last_trace_{};
    std::size_t last_trace_cycle_{0};
    std::size_t cycle_{0};
};

} // namespace ftlpu
