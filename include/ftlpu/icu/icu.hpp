#pragma once

#include "ftlpu/icu/instruction.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

template <typename FuncInstruction>
class IcuUnit {
public:
    void load(std::vector<IcuInstruction> icu_program,
              std::vector<FuncInstruction> func_program)
    {
        icu_program_ = std::move(icu_program);
        func_program_ = std::move(func_program);
        reset_state();
        state_ = State::Synced;
    }

    void reset()
    {
        icu_program_.clear();
        func_program_.clear();
        reset_state();
        state_ = State::Idle;
    }

    std::optional<FuncInstruction> tick()
    {
        switch (state_) {
        case State::Idle:
        case State::Synced:
            return std::nullopt;
        case State::NopDelay:
            ++cycle_;
            return tick_nop_delay();
        case State::RepeatEmit:
            ++cycle_;
            return tick_repeat_emit();
        case State::Running:
            ++cycle_;
            return tick_running();
        }
        return std::nullopt;
    }

    void notify()
    {
        if (state_ == State::Synced) {
            state_ = State::Running;
        }
    }

    bool running() const
    {
        return state_ == State::Running
            || state_ == State::NopDelay
            || state_ == State::RepeatEmit;
    }

    bool synced() const
    {
        return state_ == State::Synced;
    }

    bool done() const
    {
        return state_ == State::Running && ice_pc_ >= icu_program_.size();
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    std::size_t ice_pc() const
    {
        return ice_pc_;
    }

    std::size_t func_pc() const
    {
        return func_pc_;
    }

private:
    enum class State {
        Idle,
        Running,
        NopDelay,
        Synced,
        RepeatEmit,
    };

    void reset_state()
    {
        ice_pc_ = 0;
        func_pc_ = 0;
        cycle_ = 0;
        last_dispatched_valid_ = false;
        last_dispatched_ = FuncInstruction{};
        nop_remaining_ = 0;
        repeat_remaining_ = 0;
        repeat_gap_ = 0;
    }

    std::optional<FuncInstruction> tick_running()
    {
        if (ice_pc_ >= icu_program_.size()) {
            return std::nullopt;
        }

        const auto& inst = icu_program_[ice_pc_++];

        switch (inst.opcode) {
        case IcuOpcode::Dispatch:
            return handle_dispatch();
        case IcuOpcode::Nop:
            return handle_nop(inst);
        case IcuOpcode::Repeat:
            return handle_repeat(inst);
        case IcuOpcode::Sync:
            state_ = State::Synced;
            return std::nullopt;
        case IcuOpcode::Notify:
            return std::nullopt;
        }
        return std::nullopt;
    }

    std::optional<FuncInstruction> handle_dispatch()
    {
        if (func_pc_ >= func_program_.size()) {
            throw std::logic_error("ICU Dispatch at end of functional program");
        }
        auto inst = func_program_[func_pc_++];
        last_dispatched_ = inst;
        last_dispatched_valid_ = true;
        return inst;
    }

    std::optional<FuncInstruction> handle_nop(const IcuInstruction& inst)
    {
        if (inst.repeat_count > 1) {
            nop_remaining_ = inst.repeat_count - 1;
            state_ = State::NopDelay;
        }
        return std::nullopt;
    }

    std::optional<FuncInstruction> handle_repeat(const IcuInstruction& inst)
    {
        if (!last_dispatched_valid_) {
            throw std::logic_error("ICU Repeat without a prior Dispatch");
        }
        if (inst.repeat_count == 0) {
            return std::nullopt;
        }

        repeat_remaining_ = inst.repeat_count - 1;
        repeat_gap_ = inst.repeat_interval;

        if (repeat_remaining_ > 0) {
            state_ = State::RepeatEmit;
        }
        return last_dispatched_;
    }

    std::optional<FuncInstruction> tick_nop_delay()
    {
        if (nop_remaining_ == 0) {
            state_ = State::Running;
            return std::nullopt;
        }
        --nop_remaining_;
        if (nop_remaining_ == 0) {
            state_ = State::Running;
        }
        return std::nullopt;
    }

    std::optional<FuncInstruction> tick_repeat_emit()
    {
        if (repeat_gap_ > 0) {
            --repeat_gap_;
        }
        if (repeat_gap_ == 0) {
            if (repeat_remaining_ == 0) {
                state_ = State::Running;
                return std::nullopt;
            }
            --repeat_remaining_;
            repeat_gap_ = icu_program_[ice_pc_ - 1].repeat_interval;
            if (repeat_remaining_ == 0) {
                state_ = State::Running;
            }
            return last_dispatched_;
        }
        return std::nullopt;
    }

    std::vector<IcuInstruction> icu_program_{};
    std::vector<FuncInstruction> func_program_{};
    std::size_t ice_pc_{0};
    std::size_t func_pc_{0};
    std::size_t cycle_{0};
    State state_{State::Idle};

    bool last_dispatched_valid_{false};
    FuncInstruction last_dispatched_{};

    std::size_t nop_remaining_{0};
    std::size_t repeat_remaining_{0};
    std::size_t repeat_gap_{0};
};

} // namespace ftlpu
