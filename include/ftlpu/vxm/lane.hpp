#pragma once

#include "ftlpu/vxm/alu.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ftlpu {

enum class VxmLaneSource {
    Value,
    Gate,
    Up,
    Immediate,
};

struct VxmLaneAluInstruction {
    VxmAluOpcode opcode{VxmAluOpcode::Pass};
    VxmLaneSource lhs{VxmLaneSource::Value};
    VxmLaneSource rhs{VxmLaneSource::Immediate};
    float immediate{0.0f};
    float scale{1.0f};
    std::int32_t output_zero_point{0};
};

class VxmLane {
public:
    static constexpr std::size_t kAluStages = 16;

    using Byte = std::uint8_t;
    using Int32Bytes = std::array<Byte, 4>;
    using Program = std::array<VxmLaneAluInstruction, kAluStages>;

    struct SwigluParams {
        float gate_scale{1.0f};
        float up_scale{1.0f};
        float output_scale{1.0f};
        std::int32_t output_zero_point{0};
    };

    struct Output {
        std::int8_t value{0};
    };

    void reset()
    {
        for (auto& stage : stages_) {
            stage.reset();
        }
        pending_input_.reset();
        output_.reset();
        cycle_ = 0;
    }

    void set_program(Program program)
    {
        program_ = program;
    }

    void load_swiglu_program(const SwigluParams& params)
    {
        Program program{};
        for (auto& instruction : program) {
            instruction = VxmLaneAluInstruction {VxmAluOpcode::Pass};
        }

        program[0] = VxmLaneAluInstruction {VxmAluOpcode::Negate};
        program[1] = VxmLaneAluInstruction {VxmAluOpcode::Exp};
        program[2] = VxmLaneAluInstruction {
            VxmAluOpcode::Add,
            VxmLaneSource::Value,
            VxmLaneSource::Immediate,
            1.0f,
        };
        program[3] = VxmLaneAluInstruction {
            VxmAluOpcode::Divide,
            VxmLaneSource::Immediate,
            VxmLaneSource::Value,
            1.0f,
        };
        program[4] = VxmLaneAluInstruction {
            VxmAluOpcode::Multiply,
            VxmLaneSource::Gate,
            VxmLaneSource::Value,
        };
        program[5] = VxmLaneAluInstruction {
            VxmAluOpcode::Multiply,
            VxmLaneSource::Up,
            VxmLaneSource::Value,
        };
        program[15] = VxmLaneAluInstruction {
            VxmAluOpcode::QuantizeInt8,
            VxmLaneSource::Value,
            VxmLaneSource::Immediate,
            0.0f,
            params.output_scale,
            params.output_zero_point,
        };
        set_program(program);
        swiglu_params_ = params;
    }

    void set_swiglu_input(Int32Bytes gate_streams, Int32Bytes up_streams)
    {
        if (pending_input_.has_value()) {
            throw std::logic_error("VXM lane input is already occupied");
        }
        const auto gate_int32 = unpack_int32(gate_streams);
        const auto up_int32 = unpack_int32(up_streams);
        pending_input_ = Packet {
            true,
            static_cast<float>(gate_int32) * swiglu_params_.gate_scale,
            static_cast<float>(gate_int32) * swiglu_params_.gate_scale,
            static_cast<float>(up_int32) * swiglu_params_.up_scale,
            0,
            false,
        };
    }

    void tick()
    {
        output_.reset();

        std::array<std::optional<Packet>, kAluStages> next{};
        if (stages_[kAluStages - 1].has_value()) {
            const auto packet = execute_stage(*stages_[kAluStages - 1], program_[kAluStages - 1]);
            if (packet.output_valid) {
                output_ = Output {packet.output};
            }
        }

        for (std::size_t stage = 1; stage + 1 < kAluStages; ++stage) {
            const auto source_stage = kAluStages - 1 - stage;
            if (stages_[source_stage].has_value()) {
                next[source_stage + 1] = execute_stage(*stages_[source_stage], program_[source_stage]);
            }
        }

        if (pending_input_.has_value()) {
            next[1] = execute_stage(*pending_input_, program_[0]);
            pending_input_.reset();
        }

        stages_ = next;
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

    const std::optional<float> stage_value(std::size_t stage) const
    {
        if (stage >= kAluStages) {
            throw std::out_of_range("VXM lane stage is outside the 16-ALU lane");
        }
        if (!stages_[stage].has_value()) {
            return std::nullopt;
        }
        return stages_[stage]->value;
    }

    std::size_t cycle() const
    {
        return cycle_;
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

private:
    struct Packet {
        bool valid{false};
        float value{0.0f};
        float gate{0.0f};
        float up{0.0f};
        std::int8_t output{0};
        bool output_valid{false};
    };

    static float select_source(const Packet& packet, const VxmLaneAluInstruction& instruction, VxmLaneSource source)
    {
        switch (source) {
        case VxmLaneSource::Value:
            return packet.value;
        case VxmLaneSource::Gate:
            return packet.gate;
        case VxmLaneSource::Up:
            return packet.up;
        case VxmLaneSource::Immediate:
            return instruction.immediate;
        }
        throw std::logic_error("unsupported VXM lane ALU source");
    }

    static Packet execute_stage(Packet packet, const VxmLaneAluInstruction& instruction)
    {
        const auto lhs = select_source(packet, instruction, instruction.lhs);
        const auto rhs = select_source(packet, instruction, instruction.rhs);

        switch (instruction.opcode) {
        case VxmAluOpcode::Pass:
            return packet;
        case VxmAluOpcode::Add:
            packet.value = lhs + rhs;
            return packet;
        case VxmAluOpcode::Subtract:
            packet.value = lhs - rhs;
            return packet;
        case VxmAluOpcode::Multiply:
            packet.value = lhs * rhs;
            return packet;
        case VxmAluOpcode::Divide:
            packet.value = lhs / rhs;
            return packet;
        case VxmAluOpcode::Negate:
            packet.value = -lhs;
            return packet;
        case VxmAluOpcode::Exp:
            packet.value = std::exp(lhs);
            return packet;
        case VxmAluOpcode::QuantizeInt8:
            packet.output = VxmAlu::quantize_scalar(lhs, instruction.scale, instruction.output_zero_point);
            packet.output_valid = true;
            return packet;
        default:
            throw std::logic_error("VXM lane ALU opcode is not implemented in the scalar lane pipeline");
        }
    }

    Program program_{};
    SwigluParams swiglu_params_{};
    std::array<std::optional<Packet>, kAluStages> stages_{};
    std::optional<Packet> pending_input_{};
    std::optional<Output> output_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
