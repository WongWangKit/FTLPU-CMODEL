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
