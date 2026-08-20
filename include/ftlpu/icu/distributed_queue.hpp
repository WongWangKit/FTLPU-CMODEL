#pragma once

#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mxm/control_slice.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ftlpu {

template <typename FuncInstruction>
using IqEntry = std::variant<IcuControlInstruction, FuncInstruction>;

struct IcuProgramDescriptor {
    std::size_t base_pc{0};
    std::size_t instruction_count{0};
    std::size_t start_cycle{0};
};

enum class IcuQueueAction : std::uint8_t {
    Idle,
    WaitingForStart,
    PrefetchOnly,
    FunctionalIssue,
    Nop,
    NopWait,
    RepeatIssue,
    RepeatWait,
    LoopIssue,
    LoopWait,
    Repeat2DIssue,
    Repeat2DWait,
    SyncWait,
    SyncRelease,
    Notify,
    Underflow,
};

struct IcuQueueCycleTrace {
    std::size_t cycle{0};
    std::optional<std::size_t> fetch_started_pc{};
    std::optional<std::size_t> fetch_completed_pc{};
    std::optional<std::size_t> issue_pc{};
    std::size_t iq_before{0};
    std::size_t iq_after{0};
    IcuQueueAction action{IcuQueueAction::Idle};
};

namespace detail {

template <typename FuncInstruction>
FuncInstruction apply_icu_repeat_stride(
    FuncInstruction instruction,
    std::int64_t,
    std::size_t)
{
    return instruction;
}

inline MemInstruction apply_icu_repeat_stride(
    MemInstruction instruction,
    std::int64_t address_stride,
    std::size_t repeat_index)
{
    const auto delta = address_stride * static_cast<std::int64_t>(repeat_index);
    const auto address = static_cast<std::int64_t>(instruction.address);
    if (delta < 0 && address < -delta) {
        throw std::out_of_range("ICU MEM Repeat address stride underflow");
    }
    instruction.address = static_cast<std::size_t>(address + delta);
    return instruction;
}

inline MxmControlInstruction apply_icu_repeat_stride(
    MxmControlInstruction instruction,
    std::int64_t address_stride,
    std::size_t repeat_index)
{
    if (address_stride == 0) return instruction;
    const auto delta = address_stride
        * static_cast<std::int64_t>(repeat_index);
    if (instruction.opcode == MxmControlOpcode::Compute
        || instruction.opcode == MxmControlOpcode::AccumulatorRead) {
        const auto address = static_cast<std::int64_t>(
            instruction.accumulator_address) + delta;
        if (address < 0)
            throw std::out_of_range(
                "ICU MXM Loop accumulator-address stride underflow");
        MxmControlInstruction::check_accumulator_address(
            static_cast<std::size_t>(address), instruction.compute_mode);
        instruction.accumulator_address = static_cast<std::size_t>(address);
        return instruction;
    }
    if (instruction.opcode == MxmControlOpcode::IW) {
        const auto column = static_cast<std::int64_t>(
            instruction.weight_column) + delta;
        if (column < 0
            || column >= static_cast<std::int64_t>(hw::kMxmColumns))
            throw std::out_of_range(
                "ICU MXM Loop weight-column stride is outside the MXM");
        instruction.weight_column = static_cast<std::size_t>(column);
        return instruction;
    }
    throw std::invalid_argument(
        "ICU MXM Loop stride requires compute, accumulator-read, or IW");
}

template <typename FuncInstruction>
FuncInstruction apply_icu_repeat_2d_stride(
    FuncInstruction instruction,
    IcuInductionTarget target,
    std::int64_t delta)
{
    if (target != IcuInductionTarget::None || delta != 0)
        throw std::invalid_argument(
            "ICU Repeat2D induction target is invalid for this queue");
    return instruction;
}

inline MemInstruction apply_icu_repeat_2d_stride(
    MemInstruction instruction,
    IcuInductionTarget target,
    std::int64_t delta)
{
    if (target == IcuInductionTarget::None) {
        if (delta != 0)
            throw std::invalid_argument(
                "ICU Repeat2D has a stride without a MEM induction target");
        return instruction;
    }
    if (target != IcuInductionTarget::MemAddress)
        throw std::invalid_argument(
            "ICU Repeat2D induction target is invalid for a MEM queue");
    return apply_icu_repeat_stride(instruction, delta, 1);
}

inline MxmControlInstruction apply_icu_repeat_2d_stride(
    MxmControlInstruction instruction,
    IcuInductionTarget target,
    std::int64_t delta)
{
    if (target == IcuInductionTarget::None) {
        if (delta != 0)
            throw std::invalid_argument(
                "ICU Repeat2D has a stride without an MXM induction target");
        return instruction;
    }
    if (target == IcuInductionTarget::MxmWeightColumn) {
        if (instruction.opcode != MxmControlOpcode::IW)
            throw std::invalid_argument(
                "ICU Repeat2D weight-column induction requires an MXM IW instruction");
        const auto column =
            static_cast<std::int64_t>(instruction.weight_column) + delta;
        if (column < 0
            || column >= static_cast<std::int64_t>(hw::kMxmColumns))
            throw std::out_of_range(
                "ICU Repeat2D MXM weight-column induction is outside the MXM");
        instruction.weight_column = static_cast<std::size_t>(column);
        return instruction;
    }
    if (target == IcuInductionTarget::MxmAccumulatorAddress) {
        if (instruction.opcode != MxmControlOpcode::Compute
            && instruction.opcode != MxmControlOpcode::AccumulatorRead)
            throw std::invalid_argument(
                "ICU Repeat2D accumulator induction requires an MXM compute or accumulator-read instruction");
        const auto address =
            static_cast<std::int64_t>(instruction.accumulator_address)
            + delta;
        if (address < 0)
            throw std::out_of_range(
                "ICU Repeat2D MXM accumulator-address induction underflow");
        MxmControlInstruction::check_accumulator_address(
            static_cast<std::size_t>(address), instruction.compute_mode);
        instruction.accumulator_address = static_cast<std::size_t>(address);
        return instruction;
    }
    throw std::invalid_argument(
        "ICU Repeat2D induction target is invalid for an MXM queue");
}

} // namespace detail

// One independently scheduled ICU endpoint. The instruction memory is local
// to this queue, so runtime fetches never consume MEM capacity or data Stream
// Register bandwidth. InstructionBits, ImemDepth and IqDepth are architectural
// parameters and may differ between function types.
template <
    typename FuncInstruction,
    std::size_t InstructionBits,
    std::size_t ImemDepth,
    std::size_t IqDepth,
    std::size_t FetchLatency = 1>
class DistributedIcuQueue {
public:
    using Entry = IqEntry<FuncInstruction>;

    static_assert(
        InstructionBits >= 32,
        "every distributed ICU word must hold a 32-bit ICU control command");
    static_assert(ImemDepth > 0);
    static_assert(IqDepth > 0);
    static_assert(FetchLatency > 0);

    static constexpr std::size_t instruction_bits = InstructionBits;
    static constexpr std::size_t imem_depth = ImemDepth;
    static constexpr std::size_t iq_depth = IqDepth;
    static constexpr std::size_t fetch_latency = FetchLatency;

    void reset()
    {
        imem_.clear();
        reset_execution();
        configured_ = false;
        cycle_ = 0;
    }

    void reset_execution()
    {
        iq_.clear();
        iq_pcs_.clear();
        pending_fetches_.clear();
        last_dispatched_.reset();
        repeat_instruction_.reset();
        repeat_2d_instruction_.reset();
        nop_remaining_ = 0;
        repeat_remaining_ = 0;
        repeat_interval_ = 1;
        repeat_cooldown_ = 0;
        repeat_address_stride_ = 0;
        repeat_index_ = 0;
        repeat_2d_active_ = false;
        repeat_2d_inner_ = 0;
        repeat_2d_outer_ = 0;
        repeat_2d_cooldown_ = 0;
        static_history_pcs_.clear();
        loop_window_pcs_.clear();
        loop_rounds_remaining_ = 0;
        loop_window_index_ = 0;
        loop_interval_ = 1;
        loop_cooldown_ = 0;
        loop_address_stride_ = 0;
        loop_round_index_ = 0;
        last_dispatched_pc_.reset();
        notification_tokens_ = 0;
        notify_emitted_ = false;
        fetch_pc_ = 0;
        program_end_pc_ = 0;
        start_cycle_ = 0;
        fetched_count_ = 0;
        issued_count_ = 0;
        launched_ = false;
        underflowed_ = false;
    }

    void write_imem(std::size_t address, Entry entry)
    {
        if (launched_) {
            throw StaticScheduleError(
                "ICU local i-MEM cannot be modified after program launch");
        }
        if (address >= ImemDepth) {
            std::ostringstream os;
            os << "ICU i-MEM address " << address
               << " exceeds configured depth " << ImemDepth
               << " for " << InstructionBits << "-bit instructions";
            throw StaticScheduleError(os.str());
        }
        if (imem_.size() <= address) {
            imem_.resize(address + 1);
        }
        imem_[address] = std::move(entry);
    }

    void write_imem(std::size_t address, FuncInstruction instruction)
    {
        write_imem(
            address,
            Entry {std::in_place_type<FuncInstruction>, std::move(instruction)});
    }

    void write_imem_control(
        std::size_t address, IcuControlInstruction instruction)
    {
        write_imem(
            address,
            Entry {std::in_place_type<IcuControlInstruction>, instruction});
    }

    void load_imem(std::size_t base, std::vector<Entry> program)
    {
        if (base > ImemDepth || program.size() > ImemDepth - base) {
            throw StaticScheduleError("ICU program exceeds configured i-MEM depth");
        }
        for (std::size_t index = 0; index < program.size(); ++index) {
            write_imem(base + index, std::move(program[index]));
        }
    }

    // Compatibility/configuration helper: append compiler output to local
    // i-MEM. It does not inject an instruction directly into the runtime IQ.
    void append_program(Entry entry)
    {
        const auto address = imem_.size();
        write_imem(address, std::move(entry));
        configured_ = false;
    }

    void append_program(FuncInstruction instruction)
    {
        append_program(
            Entry {std::in_place_type<FuncInstruction>, std::move(instruction)});
    }

    void append_control(IcuControlInstruction instruction)
    {
        append_program(
            Entry {std::in_place_type<IcuControlInstruction>, instruction});
    }

    // Compatibility surface used by the existing schedule builders. These
    // calls append to local i-MEM; runtime IQ entries are still produced only
    // by the modeled fetch frontend.
    void push_instruction(FuncInstruction instruction)
    {
        append_program(std::move(instruction));
    }

    void push_nop(std::size_t cycles)
    {
        if (cycles != 0) append_control(IcuControlInstruction::Nop(cycles));
    }

    void push_repeat(IcuRepeat repeat)
    {
        if (repeat.count != 0) {
            append_control(IcuControlInstruction::Repeat(
                repeat.count, repeat.interval, repeat.address_stride));
        }
    }

    void push_loop(IcuLoop loop)
    {
        append_control(IcuControlInstruction::Loop(
            loop.window_size, loop.count, loop.interval,
            loop.address_stride));
    }

    void push_repeat_2d(IcuRepeat2D repeat)
    {
        append_control(IcuControlInstruction::Repeat2D(repeat));
    }

    std::optional<FuncInstruction> dispatch_next()
    {
        return tick();
    }
    void configure(IcuProgramDescriptor descriptor)
    {
        if (descriptor.base_pc > ImemDepth
            || descriptor.instruction_count
                > ImemDepth - descriptor.base_pc
            || descriptor.base_pc + descriptor.instruction_count
                > imem_.size()) {
            throw StaticScheduleError(
                "ICU program descriptor exceeds initialized local i-MEM");
        }
        reset_execution();
        fetch_pc_ = descriptor.base_pc;
        program_end_pc_ = descriptor.base_pc + descriptor.instruction_count;
        start_cycle_ = descriptor.start_cycle;
        configured_ = true;
        launched_ = true;
    }

    void configure_all(std::size_t start_cycle = 0)
    {
        configure(IcuProgramDescriptor {0, imem_.size(), start_cycle});
    }

    // Represents compiler/loader controlled pre-launch time. Each call is one
    // frontend cycle: a local i-MEM read completes after FetchLatency cycles.
    void prefetch_only()
    {
        ensure_configured();
        begin_trace(IcuQueueAction::PrefetchOnly);
        commit_ready_fetch();
        begin_fetch_if_possible();
        age_pending_fetches();
        finish_trace();
        ++cycle_;
    }

    std::optional<FuncInstruction> tick()
    {
        ensure_configured();
        notify_emitted_ = false;
        begin_trace(
            cycle_ < start_cycle_
                ? IcuQueueAction::WaitingForStart
                : IcuQueueAction::Idle);
        commit_ready_fetch();

        std::optional<FuncInstruction> result;
        if (cycle_ >= start_cycle_) {
            result = dispatch_ready_entry();
        }

        begin_fetch_if_possible();
        age_pending_fetches();
        finish_trace();
        ++cycle_;
        return result;
    }

    std::optional<FuncInstruction> dispatch()
    {
        return tick();
    }

    void notify() { ++notification_tokens_; }

    bool take_notify()
    {
        const auto emitted = notify_emitted_;
        notify_emitted_ = false;
        return emitted;
    }

    bool blocked_on_sync() const
    {
        if (nop_remaining_ != 0 || repeat_remaining_ != 0
            || loop_rounds_remaining_ != 0 || repeat_2d_active_
            || iq_.empty()) {
            return false;
        }
        const auto* control = std::get_if<IcuControlInstruction>(&iq_.front());
        return control != nullptr
            && control->opcode == IcuControlOpcode::Sync
            && notification_tokens_ == 0;
    }

    bool done() const
    {
        if (!configured_) {
            return imem_.empty();
        }
        return fetch_pc_ == program_end_pc_
            && pending_fetches_.empty()
            && iq_.empty()
            && nop_remaining_ == 0
            && repeat_remaining_ == 0
            && loop_rounds_remaining_ == 0
            && !repeat_2d_active_;
    }

    bool running() const { return !done(); }
    bool underflowed() const noexcept { return underflowed_; }
    std::size_t cycle() const noexcept { return cycle_; }
    std::size_t imem_occupancy() const noexcept { return imem_.size(); }
    std::size_t iq_occupancy() const noexcept { return iq_.size(); }
    std::size_t pending_fetch_count() const noexcept
    {
        return pending_fetches_.size();
    }
    std::size_t fetch_pc() const noexcept { return fetch_pc_; }
    std::size_t fetched_count() const noexcept { return fetched_count_; }
    std::size_t issued_count() const noexcept { return issued_count_; }
    std::size_t start_cycle() const noexcept { return start_cycle_; }
    const IcuQueueCycleTrace& last_trace() const noexcept
    {
        return last_trace_;
    }
    std::size_t free_iq_entries() const noexcept
    {
        return IqDepth - iq_.size() - pending_fetches_.size();
    }
    std::size_t pending_issue_cycles() const noexcept
    {
        if (repeat_remaining_ == 0) {
            if (repeat_2d_active_)
                return nop_remaining_ + repeat_2d_cooldown_
                    + repeat_2d_remaining_points();
            if (loop_rounds_remaining_ == 0) return nop_remaining_;
            const auto current_round = loop_window_pcs_.size()
                - loop_window_index_;
            return nop_remaining_ + loop_cooldown_ + current_round
                + (loop_rounds_remaining_ - 1) * loop_interval_;
        }
        return nop_remaining_ + repeat_cooldown_ + 1
            + (repeat_remaining_ - 1) * repeat_interval_;
    }
    std::size_t queued_count() const noexcept
    {
        if (!configured_) {
            return imem_.size();
        }
        return iq_.size() + pending_fetches_.size()
            + (program_end_pc_ - fetch_pc_)
            + repeat_remaining_ + nop_remaining_
            + repeat_2d_remaining_points()
            + loop_rounds_remaining_ * loop_window_pcs_.size();
    }

private:
    struct PendingFetch {
        std::size_t pc{0};
        Entry entry;
        std::size_t remaining_cycles{FetchLatency};
    };

    void begin_trace(IcuQueueAction action)
    {
        last_trace_ = IcuQueueCycleTrace{};
        last_trace_.cycle = cycle_;
        last_trace_.iq_before = iq_.size();
        last_trace_.action = action;
    }

    void finish_trace()
    {
        last_trace_.iq_after = iq_.size();
    }

    void ensure_configured()
    {
        if (!configured_) {
            configure_all(0);
            // Direct CModel enqueue calls represent a program already loaded
            // before cycle zero. Prime the finite IQ; all subsequent refills
            // still traverse the modeled local one-instruction/cycle frontend.
            while (iq_.size() < IqDepth && fetch_pc_ < program_end_pc_) {
                iq_.push_back(read_imem(fetch_pc_++));
                iq_pcs_.push_back(fetch_pc_ - 1);
                ++fetched_count_;
            }
        }
    }

    Entry read_imem(std::size_t address) const
    {
        if (address >= imem_.size() || !imem_[address].has_value()) {
            throw StaticScheduleError("ICU fetched an uninitialized local i-MEM address");
        }
        return *imem_[address];
    }

    void commit_ready_fetch()
    {
        while (!pending_fetches_.empty()
               && pending_fetches_.front().remaining_cycles == 0) {
            if (iq_.size() == IqDepth) {
                throw std::logic_error("ICU frontend committed into a full IQ");
            }
            iq_.push_back(std::move(pending_fetches_.front().entry));
            iq_pcs_.push_back(pending_fetches_.front().pc);
            last_trace_.fetch_completed_pc = pending_fetches_.front().pc;
            pending_fetches_.pop_front();
            ++fetched_count_;
        }
    }

    void begin_fetch_if_possible()
    {
        if (!launched_ || fetch_pc_ == program_end_pc_
            || free_iq_entries() == 0) {
            return;
        }
        last_trace_.fetch_started_pc = fetch_pc_;
        pending_fetches_.push_back(PendingFetch {
            fetch_pc_, read_imem(fetch_pc_), FetchLatency});
        ++fetch_pc_;
    }

    void age_pending_fetches()
    {
        for (auto& fetch : pending_fetches_) {
            if (fetch.remaining_cycles != 0) {
                --fetch.remaining_cycles;
            }
        }
    }

    std::optional<FuncInstruction> dispatch_ready_entry()
    {
        if (nop_remaining_ > 0) {
            --nop_remaining_;
            last_trace_.action = IcuQueueAction::NopWait;
            return std::nullopt;
        }
        if (repeat_remaining_ > 0) {
            return tick_repeat();
        }
        if (repeat_2d_active_) {
            return tick_repeat_2d();
        }
        if (loop_rounds_remaining_ > 0) {
            return tick_loop();
        }
        if (iq_.empty()) {
            if (fetch_pc_ != program_end_pc_ || !pending_fetches_.empty()) {
                underflowed_ = true;
                last_trace_.action = IcuQueueAction::Underflow;
                std::ostringstream os;
                os << "ICU IQ underflow at cycle " << cycle_
                   << "; compiler did not prefetch before the scheduled issue";
                throw StaticScheduleError(os.str());
            }
            return std::nullopt;
        }

        if (auto* instruction = std::get_if<FuncInstruction>(&iq_.front())) {
            auto result = std::move(*instruction);
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::FunctionalIssue;
            last_dispatched_pc_ = iq_pcs_.front();
            iq_.pop_front();
            iq_pcs_.pop_front();
            last_dispatched_ = result;
            static_history_pcs_.push_back(*last_dispatched_pc_);
            if (static_history_pcs_.size() > 63) {
                static_history_pcs_.pop_front();
            }
            ++issued_count_;
            return result;
        }

        const auto control = std::get<IcuControlInstruction>(iq_.front());
        return execute_control(control);
    }

    std::optional<FuncInstruction> execute_control(
        const IcuControlInstruction& control)
    {
        switch (control.opcode) {
        case IcuControlOpcode::Nop:
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::Nop;
            iq_.pop_front();
            iq_pcs_.pop_front();
            nop_remaining_ = control.count;
            if (nop_remaining_ > 0) {
                --nop_remaining_;
            }
            return std::nullopt;
        case IcuControlOpcode::Repeat:
            return begin_repeat(control);
        case IcuControlOpcode::Loop:
            return begin_loop(control);
        case IcuControlOpcode::Repeat2D:
            return begin_repeat_2d(control);
        case IcuControlOpcode::Sync:
            if (notification_tokens_ == 0) {
                last_trace_.issue_pc = iq_pcs_.front();
                last_trace_.action = IcuQueueAction::SyncWait;
                return std::nullopt;
            }
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::SyncRelease;
            --notification_tokens_;
            iq_.pop_front();
            iq_pcs_.pop_front();
            return std::nullopt;
        case IcuControlOpcode::Notify:
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::Notify;
            iq_.pop_front();
            iq_pcs_.pop_front();
            notify_emitted_ = true;
            return std::nullopt;
        }
        throw std::logic_error("unknown ICU control opcode");
    }

    std::optional<FuncInstruction> begin_repeat(
        const IcuControlInstruction& control)
    {
        if (control.interval == 0) {
            throw std::invalid_argument("ICU Repeat interval must be at least one cycle");
        }
        if (!last_dispatched_.has_value()) {
            throw std::logic_error("ICU Repeat needs a prior functional instruction in the same IQ");
        }
        last_trace_.issue_pc = iq_pcs_.front();
        iq_.pop_front();
        iq_pcs_.pop_front();
        if (control.count == 0) {
            return std::nullopt;
        }
        repeat_remaining_ = control.count;
        repeat_instruction_ = *last_dispatched_;
        repeat_interval_ = control.interval;
        repeat_cooldown_ = control.interval - 1;
        repeat_address_stride_ = control.address_stride;
        repeat_index_ = 1;
        return tick_repeat();
    }

    std::optional<FuncInstruction> tick_repeat()
    {
        if (repeat_cooldown_ > 0) {
            --repeat_cooldown_;
            last_trace_.action = IcuQueueAction::RepeatWait;
            return std::nullopt;
        }
        auto result = detail::apply_icu_repeat_stride(
            *repeat_instruction_, repeat_address_stride_, repeat_index_);
        --repeat_remaining_;
        ++repeat_index_;
        if (repeat_remaining_ > 0) {
            repeat_cooldown_ = repeat_interval_ - 1;
        }
        last_dispatched_ = result;
        last_trace_.issue_pc = last_dispatched_pc_;
        last_trace_.action = IcuQueueAction::RepeatIssue;
        ++issued_count_;
        return result;
    }

    std::optional<FuncInstruction> begin_repeat_2d(
        const IcuControlInstruction& control)
    {
        const auto& repeat = control.repeat_2d;
        if (!last_dispatched_.has_value())
            throw std::logic_error(
                "ICU Repeat2D needs a prior functional instruction");
        if (repeat.inner_count == 0 || repeat.outer_count == 0
            || repeat.inner_interval == 0 || repeat.outer_interval == 0
            || repeat.inner_count * repeat.outer_count <= 1
            || (repeat.outer_count > 1
                && repeat.outer_interval
                    <= (repeat.inner_count - 1) * repeat.inner_interval))
            throw std::invalid_argument(
                "ICU Repeat2D has an invalid iteration space");
        last_trace_.issue_pc = iq_pcs_.front();
        iq_.pop_front();
        iq_pcs_.pop_front();
        repeat_2d_ = repeat;
        repeat_2d_instruction_ = *last_dispatched_;
        repeat_2d_outer_ = 0;
        repeat_2d_inner_ = repeat.inner_count > 1 ? 1 : 0;
        if (repeat.inner_count == 1) repeat_2d_outer_ = 1;
        repeat_2d_active_ = true;
        const auto firstOffset = repeat_2d_issue_offset();
        repeat_2d_cooldown_ = firstOffset - 1;
        return tick_repeat_2d();
    }

    std::optional<FuncInstruction> tick_repeat_2d()
    {
        if (repeat_2d_cooldown_ > 0) {
            --repeat_2d_cooldown_;
            last_trace_.action = IcuQueueAction::Repeat2DWait;
            return std::nullopt;
        }
        const auto delta = static_cast<std::int64_t>(repeat_2d_inner_)
                * repeat_2d_.inner_stride
            + static_cast<std::int64_t>(repeat_2d_outer_)
                * repeat_2d_.outer_stride;
        auto result = detail::apply_icu_repeat_2d_stride(
            *repeat_2d_instruction_, repeat_2d_.induction_target, delta);
        last_trace_.issue_pc = last_dispatched_pc_;
        last_trace_.action = IcuQueueAction::Repeat2DIssue;
        last_dispatched_ = result;
        ++issued_count_;

        const auto previousOffset = repeat_2d_issue_offset();
        ++repeat_2d_inner_;
        if (repeat_2d_inner_ == repeat_2d_.inner_count) {
            repeat_2d_inner_ = 0;
            ++repeat_2d_outer_;
        }
        if (repeat_2d_outer_ == repeat_2d_.outer_count) {
            repeat_2d_active_ = false;
        } else {
            const auto nextOffset = repeat_2d_issue_offset();
            repeat_2d_cooldown_ = nextOffset - previousOffset - 1;
        }
        return result;
    }

    std::size_t repeat_2d_issue_offset() const noexcept
    {
        return repeat_2d_outer_ * repeat_2d_.outer_interval
            + repeat_2d_inner_ * repeat_2d_.inner_interval;
    }

    std::size_t repeat_2d_remaining_points() const noexcept
    {
        if (!repeat_2d_active_) return 0;
        return (repeat_2d_.outer_count - repeat_2d_outer_ - 1)
                * repeat_2d_.inner_count
            + repeat_2d_.inner_count - repeat_2d_inner_;
    }

    std::optional<FuncInstruction> begin_loop(
        const IcuControlInstruction& control)
    {
        if (control.window_size == 0 || control.count == 0
            || control.interval < control.window_size) {
            throw std::invalid_argument(
                "ICU Loop requires count > 0 and interval >= window size > 0");
        }
        if (control.window_size > static_history_pcs_.size()) {
            throw std::logic_error(
                "ICU Loop window exceeds prior functional instructions");
        }
        last_trace_.issue_pc = iq_pcs_.front();
        iq_.pop_front();
        iq_pcs_.pop_front();
        loop_window_pcs_.assign(
            static_history_pcs_.end() - control.window_size,
            static_history_pcs_.end());
        loop_rounds_remaining_ = control.count;
        loop_window_index_ = 0;
        loop_interval_ = control.interval;
        loop_cooldown_ = 0;
        loop_address_stride_ = control.address_stride;
        loop_round_index_ = 1;
        return tick_loop();
    }

    std::optional<FuncInstruction> tick_loop()
    {
        if (loop_cooldown_ > 0) {
            --loop_cooldown_;
            last_trace_.action = IcuQueueAction::LoopWait;
            return std::nullopt;
        }
        auto result = detail::apply_icu_repeat_stride(
            std::get<FuncInstruction>(
                read_imem(loop_window_pcs_[loop_window_index_])),
            loop_address_stride_,
            loop_round_index_);
        last_trace_.issue_pc = loop_window_pcs_[loop_window_index_];
        last_trace_.action = IcuQueueAction::LoopIssue;
        last_dispatched_ = result;
        last_dispatched_pc_ = loop_window_pcs_[loop_window_index_];
        ++issued_count_;
        ++loop_window_index_;
        if (loop_window_index_ == loop_window_pcs_.size()) {
            --loop_rounds_remaining_;
            loop_window_index_ = 0;
            ++loop_round_index_;
            if (loop_rounds_remaining_ > 0) {
                loop_cooldown_ = loop_interval_ - loop_window_pcs_.size();
            }
        }
        return result;
    }

    std::vector<std::optional<Entry>> imem_{};
    std::deque<Entry> iq_{};
    std::deque<std::size_t> iq_pcs_{};
    std::deque<PendingFetch> pending_fetches_{};
    std::optional<FuncInstruction> last_dispatched_{};
    std::optional<FuncInstruction> repeat_instruction_{};
    std::optional<FuncInstruction> repeat_2d_instruction_{};
    std::deque<std::size_t> static_history_pcs_{};
    std::vector<std::size_t> loop_window_pcs_{};
    std::optional<std::size_t> last_dispatched_pc_{};
    std::size_t nop_remaining_{0};
    std::size_t repeat_remaining_{0};
    std::size_t repeat_interval_{1};
    std::size_t repeat_cooldown_{0};
    std::int64_t repeat_address_stride_{0};
    std::size_t repeat_index_{0};
    IcuRepeat2D repeat_2d_{};
    bool repeat_2d_active_{false};
    std::size_t repeat_2d_inner_{0};
    std::size_t repeat_2d_outer_{0};
    std::size_t repeat_2d_cooldown_{0};
    std::size_t loop_rounds_remaining_{0};
    std::size_t loop_window_index_{0};
    std::size_t loop_interval_{1};
    std::size_t loop_cooldown_{0};
    std::int64_t loop_address_stride_{0};
    std::size_t loop_round_index_{0};
    std::size_t notification_tokens_{0};
    bool notify_emitted_{false};
    std::size_t fetch_pc_{0};
    std::size_t program_end_pc_{0};
    std::size_t start_cycle_{0};
    std::size_t fetched_count_{0};
    std::size_t issued_count_{0};
    std::size_t cycle_{0};
    bool configured_{false};
    bool launched_{false};
    bool underflowed_{false};
    IcuQueueCycleTrace last_trace_{};
};

} // namespace ftlpu
