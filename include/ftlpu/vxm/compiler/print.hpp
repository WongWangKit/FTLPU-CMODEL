#pragma once

#include "ftlpu/vxm/compiler/codegen.hpp"

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <string_view>
#include <variant>

namespace ftlpu::vxm::compiler {

inline std::string_view ir_opcode_name(VxmIrOpcode opcode)
{
    switch (opcode) {
    case VxmIrOpcode::Input: return "Input";
    case VxmIrOpcode::Constant: return "Constant";
    case VxmIrOpcode::Add: return "Add";
    case VxmIrOpcode::Subtract: return "Subtract";
    case VxmIrOpcode::Multiply: return "Multiply";
    case VxmIrOpcode::Negate: return "Negate";
    case VxmIrOpcode::Maximum: return "Maximum";
    case VxmIrOpcode::Exp: return "Exp";
    case VxmIrOpcode::Reciprocal: return "Reciprocal";
    case VxmIrOpcode::Rsqrt: return "Rsqrt";
    case VxmIrOpcode::ReduceAdd: return "ReduceAdd";
    case VxmIrOpcode::ReduceMax: return "ReduceMax";
    case VxmIrOpcode::Broadcast: return "Broadcast";
    }
    return "Unknown";
}

inline std::string_view scheduled_operand_name(
    VxmScheduledOperandKind kind)
{
    switch (kind) {
    case VxmScheduledOperandKind::StreamValue: return "StreamValue";
    case VxmScheduledOperandKind::Previous: return "Previous";
    case VxmScheduledOperandKind::Original: return "Original";
    case VxmScheduledOperandKind::Auxiliary: return "Auxiliary";
    case VxmScheduledOperandKind::Immediate: return "Immediate";
    case VxmScheduledOperandKind::LocalScalar: return "LocalScalar";
    case VxmScheduledOperandKind::Feedback: return "Feedback";
    }
    return "Unknown";
}

inline std::string_view lane_operand_name(VxmLaneOperandKind kind)
{
    switch (kind) {
    case VxmLaneOperandKind::Previous: return "Previous";
    case VxmLaneOperandKind::Original: return "Original";
    case VxmLaneOperandKind::Auxiliary: return "Auxiliary";
    case VxmLaneOperandKind::Accumulator: return "Accumulator";
    case VxmLaneOperandKind::StreamFloat16: return "StreamFloat16";
    case VxmLaneOperandKind::Reserved: return "Reserved";
    case VxmLaneOperandKind::Immediate: return "Immediate";
    case VxmLaneOperandKind::Feedback: return "Feedback";
    }
    return "Unknown";
}

inline void print_shape(std::ostream& os, const VxmTensorType& type)
{
    os << '[';
    for (std::size_t index = 0; index < type.shape.size(); ++index) {
        if (index != 0) os << ',';
        os << type.shape[index];
    }
    os << "] fp16";
}

inline void print_kernel_ir(std::ostream& os, const VxmKernel& kernel)
{
    os << "Kernel " << kernel.name << '\n';
    for (const auto& node : kernel.nodes) {
        os << "  v" << node.output << " = " << ir_opcode_name(node.opcode);
        if (!node.inputs.empty()) {
            os << '(';
            for (std::size_t index = 0; index < node.inputs.size(); ++index) {
                if (index != 0) os << ", ";
                os << 'v' << node.inputs[index];
            }
            os << ')';
        }
        os << " type=";
        print_shape(os, node.type);
        if (node.axis) os << " axis=" << *node.axis;
        if (node.constant) os << " value=" << *node.constant;
        if (node.epsilon) os << " epsilon_fp32=" << *node.epsilon;
        if (!node.name.empty()) os << " name=" << node.name;
        os << '\n';
    }
    os << "  outputs:";
    for (const auto output : kernel.outputs) os << " v" << output;
    os << "\n\n";
}

inline void print_scheduled_operand(
    std::ostream& os, const VxmScheduledOperand& operand)
{
    os << scheduled_operand_name(operand.kind);
    if (operand.value != kInvalidValue) os << "(v" << operand.value << ')';
    if (operand.kind == VxmScheduledOperandKind::Immediate) {
        os << '(' << operand.immediate << ')';
    }
}

inline void print_schedule_ir(std::ostream& os, const VxmSchedule& schedule)
{
    os << "Schedule " << schedule.kernel_name
       << " total_cycles=" << schedule.total_cycles << '\n';
    for (const auto& phase : schedule.phases) {
        os << "  Phase " << phase.id << ' ' << phase.name
           << " depth=" << static_cast<std::size_t>(phase.chain_depth)
           << " elements=" << phase.element_count
           << " parallel_chains=" << phase.parallel_chain_count
           << " rows=[" << phase.token_begin
           << ',' << phase.token_begin + phase.parallel_chain_count << ')'
           << " data=[" << phase.data_start_cycle
           << ',' << phase.end_cycle << ')'
           << (phase.feedback_from_previous
                   ? " feedback_from_previous" : "")
           << '\n';
        for (const auto& route : phase.feedback_routes) {
            os << "    feedback v" << route.value
               << ": tail" << route.source_tail
               << " -> head" << route.destination_head
               << (route.uses_holding_register
                       ? " holding_register" : " direct")
               << '\n';
        }
        for (const auto& input : phase.inputs) {
            os << "    input v" << input.value
               << " row=" << input.token_index
               << " -> head" << input.chain_head << '.'
               << (input.port == VxmInputPort::Lhs ? "LHS" : "RHS")
               << (input.access == VxmInputAccess::HoldForPhase
                       ? " hold" : " each_cycle")
               << '\n';
        }
        for (const auto& scalar : phase.local_scalars) {
            os << "    local_scalar v" << scalar.value
               << " row=" << scalar.token_index
               << " -> ALU" << scalar.stage << '\n';
        }
        for (const auto& instruction : phase.instructions) {
            os << "    ALU" << instruction.stage << ' '
               << VxmLane::operation_name(instruction.operation)
               << " lhs=";
            print_scheduled_operand(os, instruction.lhs);
            os << " rhs=";
            print_scheduled_operand(os, instruction.rhs);
            os << " repeat=" << instruction.repeat_count;
            if (instruction.accumulator_reset) os << " acc_reset";
            if (instruction.accumulator_write) os << " acc_write";
            if (!instruction.accumulator_emit) os << " no_emit";
            if (instruction.output) os << " output=v" << *instruction.output;
            if (instruction.output && !instruction.stream_output) {
                os << " feedback_only";
            }
            os << '\n';
        }
        for (const auto& output : phase.outputs) {
            os << "    phase_output v" << output.value
               << " row=" << output.token_index
               << " <- tail" << output.chain_tail
               << (output.scalar ? " scalar" : " vector")
               << (output.stream_write ? "" : " feedback_only")
               << '\n';
        }
    }
    os << '\n';
}

inline void print_lane_operand(
    std::ostream& os, const VxmLaneOperand& operand)
{
    os << lane_operand_name(operand.kind);
    if (operand.kind == VxmLaneOperandKind::Immediate) {
        os << '(' << operand.immediate << ')';
    }
}

inline void print_compiled_program(
    std::ostream& os, const VxmCompiledProgram& program)
{
    os << "Compiled program " << program.kernel.name << '\n';
    for (const auto& phase : program.phases) {
        const auto& scheduled =
            program.schedule.phases.at(phase.phase_id);
        os << "  Phase " << phase.phase_id << ' ' << phase.name
           << " depth=" << static_cast<std::size_t>(phase.chain_depth)
           << " parallel_chains=" << scheduled.parallel_chain_count
           << " rows=[" << scheduled.token_begin
           << ',' << scheduled.token_begin
                + scheduled.parallel_chain_count << ')'
           << " config_start=" << phase.config_deadline_cycle
           << " data=[" << phase.data_start_cycle
           << ',' << phase.end_cycle << ')'
           << (phase.feedback_from_previous
                   ? " feedback_from_previous" : "")
           << '\n';
        os << "    Config commands:\n";
        for (const auto& command : phase.config_commands) {
            const auto decoded = VxmCompactInstructionCodec::decode(
                command.stage, command.packet);
            const auto& instruction = decoded.instruction;
            os << "      cycle " << command.arrival_cycle
               << " ALU" << command.stage << ' '
               << "packet={0x" << std::hex << command.packet.control
               << ",0x" << command.packet.immediate_bits << std::dec
               << "} "
               << VxmLane::operation_name(instruction.operation)
               << " lhs=";
            print_lane_operand(os, instruction.lhs);
            os << " rhs=";
            print_lane_operand(os, instruction.rhs);
            os << " repeat=" << instruction.repeat_count;
            if (instruction.accumulator_reset) os << " acc_reset";
            if (instruction.accumulator_write) os << " acc_write";
            if (!instruction.accumulator_emit) os << " no_emit";
            if (instruction.output_stream) {
                os << " output_stream=" << *instruction.output_stream;
            }
            os << '\n';
        }
        os << "    Stream requirements:\n";
        for (const auto& stream : phase.stream_requirements) {
            os << "      v" << stream.value << ' '
               << (stream.direction == VxmStreamDirection::Input
                       ? "input" : "output")
               << " stream=" << stream.stream_base
               << " bytes=" << stream.byte_count
               << " first_cycle=" << stream.first_cycle
               << " transfers=" << stream.transfer_count
               << " source_element=" << stream.element_base;
            if (stream.hold) {
                os << " hold_for=" << stream.reuse_count;
            } else {
                os << " period=" << stream.period;
            }
            os << '\n';
        }
        for (const auto& load : phase.local_scalar_loads) {
            os << "    local scalar v" << load.value
               << ": stream=" << load.source_stream_base
               << " -> ALU" << load.destination_stage
               << " source_element=" << load.element_index
               << " at cycle " << load.load_cycle << '\n';
        }
    }
    os << '\n';
}

} // namespace ftlpu::vxm::compiler
