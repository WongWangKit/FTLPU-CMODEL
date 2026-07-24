#pragma once

#include "ftlpu/vxm/superlane.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

// Standalone Lane tests still need a source of broadcast configurations.
// This test-only driver owns the same Superlane-level controller used by
// VxmSuperlane; the production VxmLane remains a pure configured data path.
class VxmLaneTestDriver : public ftlpu::VxmLane {
public:
    void reset()
    {
        control_.reset();
        ftlpu::VxmLane::reset();
    }

    void set_chain_depth(ftlpu::VxmChainDepth depth)
    {
        if (!datapath_idle()) {
            throw std::logic_error(
                "cannot change test Lane chain depth while data remains in flight");
        }
        ftlpu::VxmLane::set_chain_depth(depth);
    }

    void enqueue_instruction(
        std::size_t stage, ftlpu::VxmLaneAluInstruction instruction)
    {
        validate_broadcast_instruction(stage, instruction);
        control_.enqueue(stage, chain_depth(), std::move(instruction));
    }

    void enqueue_instruction_for_depth(
        ftlpu::VxmChainDepth depth, std::size_t stage,
        ftlpu::VxmLaneAluInstruction instruction)
    {
        validate_broadcast_instruction(depth, stage, instruction);
        control_.enqueue(stage, depth, std::move(instruction));
    }

    void load_pipelined_swiglu_program(
        const SwigluParams& params, std::size_t token_count,
        std::size_t output_stream = 12)
    {
        set_chain_depth(ftlpu::VxmChainDepth::Eight);
        auto repeated = [token_count](
                            ftlpu::VxmLaneAluInstruction instruction) {
            instruction.repeat_count = token_count;
            return instruction;
        };
        enqueue_instruction(0, repeated({ftlpu::VxmAluOpcode::Negate,
            ftlpu::VxmLaneOperand::StreamInt32(params.gate_scale),
            ftlpu::VxmLaneOperand::StreamInt32(params.up_scale)}));
        enqueue_instruction(1, repeated(
            {ftlpu::VxmSpecialAluOpcode::Exp,
             ftlpu::VxmLaneOperand::Previous()}));
        enqueue_instruction(2, repeated(
            {ftlpu::VxmAluOpcode::Add,
             ftlpu::VxmLaneOperand::Previous(),
             ftlpu::VxmLaneOperand::Imm(1.0f)}));
        enqueue_instruction(3, repeated(
            {ftlpu::VxmSpecialAluOpcode::Reciprocal,
             ftlpu::VxmLaneOperand::Previous()}));
        enqueue_instruction(4, repeated(
            {ftlpu::VxmAluOpcode::Multiply,
             ftlpu::VxmLaneOperand::Previous(),
             ftlpu::VxmLaneOperand::Original()}));
        enqueue_instruction(5, repeated(
            {ftlpu::VxmAluOpcode::Multiply,
             ftlpu::VxmLaneOperand::Previous(),
             ftlpu::VxmLaneOperand::Aux()}));
        enqueue_instruction(6, repeated(
            {ftlpu::VxmAluOpcode::Bypass,
             ftlpu::VxmLaneOperand::Previous()}));
        auto tail = ftlpu::VxmLaneAluInstruction{
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::Previous()};
        tail.output_type = ftlpu::VxmCastTarget::Int8;
        tail.output_scale = params.output_scale;
        tail.output_zero_point = params.output_zero_point;
        tail.output_stream = output_stream;
        tail.repeat_count = token_count;
        enqueue_instruction(7, tail);
    }

    void tick()
    {
        const auto configs = control_.issue(chain_depth());
        const auto executed = ftlpu::VxmLane::tick(configs);
        control_.consume(executed);
    }

    bool idle() const
    {
        return control_.idle() && datapath_idle();
    }

    std::size_t queue_depth(std::size_t stage) const
    {
        return control_.remaining_executions(stage);
    }

    std::size_t current_repeat_count(std::size_t stage) const
    {
        return control_.remaining_in_current(stage);
    }

    std::size_t config_entry_count(std::size_t stage) const
    {
        return control_.config_entry_count(stage);
    }

    const ftlpu::VxmSuperlaneInstructionControl& instruction_control() const
    {
        return control_;
    }

private:
    ftlpu::VxmSuperlaneInstructionControl control_{};
};
