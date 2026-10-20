#pragma once

#include "ftlpu/core/instruction_codec.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ftlpu {

// Physical Active Context RAM entry. Bit numbering is little-endian within
// each 32-bit word and the entry is always exactly eight words wide.
struct IcuPackedActiveMacroContext256 {
    static constexpr std::size_t bit_count = 256;
    static constexpr std::size_t word_count = bit_count / 32;

    std::array<std::uint32_t, word_count> words{};

    std::uint64_t read(std::size_t offset, std::size_t width) const
    {
        if (width > 64 || offset + width > bit_count)
            throw std::logic_error("invalid packed Active Context read");
        std::uint64_t result = 0;
        for (std::size_t bit = 0; bit < width; ++bit) {
            const auto position = offset + bit;
            result |= static_cast<std::uint64_t>(
                (words[position / 32] >> (position % 32)) & 1u) << bit;
        }
        return result;
    }

    void write(std::size_t offset, std::size_t width, std::uint64_t value)
    {
        if (width > 64 || offset + width > bit_count
            || (width < 64 && (value >> width) != 0))
            throw std::logic_error("invalid packed Active Context write");
        for (std::size_t bit = 0; bit < width; ++bit) {
            const auto position = offset + bit;
            const auto mask = std::uint32_t{1} << (position % 32);
            auto& word = words[position / 32];
            if (((value >> bit) & 1u) != 0)
                word |= mask;
            else
                word &= ~mask;
        }
    }
};

template <typename FuncInstruction>
struct IcuActiveMacroContextState {
    FuncInstruction instruction{};
    IcuInductionTarget induction_target{IcuInductionTarget::None};
    std::uint32_t inner_count{1};
    std::uint32_t inner_remaining{1};
    std::uint32_t outer_remaining{1};
    std::uint32_t inner_interval{1};
    std::uint32_t outer_residual{1};
    std::int32_t inner_operand_step{0};
    std::int32_t row_transition_step{0};
    std::uint32_t next_issue_cycle{0};
    std::uint32_t final_issue_cycle{0};
};

template <typename FuncInstruction>
class IcuActiveMacroContextCodec256 {
public:
    using Packed = IcuPackedActiveMacroContext256;
    using State = IcuActiveMacroContextState<FuncInstruction>;

    static constexpr std::size_t bit_count = Packed::bit_count;
    static constexpr std::size_t word_count = Packed::word_count;

    static constexpr std::size_t valid_offset = 0;
    static constexpr std::size_t native_offset = 1;
    static constexpr std::size_t native_bits = 49;
    static constexpr std::size_t induction_target_offset = 50;
    static constexpr std::size_t induction_target_bits = 2;
    static constexpr std::size_t inner_count_offset = 52;
    static constexpr std::size_t inner_count_bits = 16;
    static constexpr std::size_t inner_remaining_offset = 68;
    static constexpr std::size_t inner_remaining_bits = 16;
    static constexpr std::size_t outer_remaining_offset = 84;
    static constexpr std::size_t outer_remaining_bits = 16;
    static constexpr std::size_t inner_interval_offset = 100;
    static constexpr std::size_t inner_interval_bits = 32;
    static constexpr std::size_t outer_residual_offset = 132;
    static constexpr std::size_t outer_residual_bits = 32;
    static constexpr std::size_t inner_operand_step_offset = 164;
    static constexpr std::size_t inner_operand_step_bits = 14;
    static constexpr std::size_t row_transition_step_offset = 178;
    static constexpr std::size_t row_transition_step_bits = 14;
    static constexpr std::size_t next_issue_cycle_offset = 192;
    static constexpr std::size_t final_issue_cycle_offset = 224;
    static constexpr std::size_t issue_cycle_bits = 32;

    static_assert(final_issue_cycle_offset + issue_cycle_bits == bit_count);

    static bool valid(const Packed& packed) noexcept
    {
        return (packed.words[0] & 1u) != 0;
    }

    static std::uint32_t issue_cycle(const Packed& packed)
    {
        return static_cast<std::uint32_t>(packed.read(
            next_issue_cycle_offset, issue_cycle_bits));
    }

    static std::uint32_t release_cycle(const Packed& packed)
    {
        return static_cast<std::uint32_t>(packed.read(
            final_issue_cycle_offset, issue_cycle_bits));
    }

    static Packed pack_initial(
        const IcuMacroSchedule& schedule,
        const FuncInstruction& instruction)
    {
        validate_schedule(schedule);

        const auto innerSteps = schedule.inner_count - 1;
        const auto outerSteps = schedule.outer_count - 1;
        const auto effectiveInnerStep = innerSteps == 0
            ? std::int64_t{0} : schedule.inner_stride;
        require_signed_fit(effectiveInnerStep, inner_operand_step_bits,
            "Active Context inner operand step exceeds 14 bits");
        const auto innerOperandSpan = static_cast<std::int64_t>(innerSteps)
            * effectiveInnerStep;

        std::int64_t rowTransition = 0;
        if (outerSteps != 0) {
            if (schedule.outer_stride
                    < innerOperandSpan + signed_min(row_transition_step_bits)
                || schedule.outer_stride
                    > innerOperandSpan + signed_max(row_transition_step_bits))
                throw StaticScheduleError(
                    "Active Context row transition exceeds 14 bits");
            rowTransition = schedule.outer_stride - innerOperandSpan;
        }

        const auto effectiveInnerInterval = innerSteps == 0
            ? std::size_t{1} : schedule.inner_interval;
        const auto innerCycleSpan = innerSteps * effectiveInnerInterval;
        const auto outerResidual = outerSteps == 0
            ? std::size_t{1}
            : schedule.outer_interval - innerCycleSpan;

        State state;
        state.instruction = instruction;
        state.induction_target = schedule.induction_target;
        state.inner_count = static_cast<std::uint32_t>(schedule.inner_count);
        state.inner_remaining = state.inner_count;
        state.outer_remaining = static_cast<std::uint32_t>(
            schedule.outer_count);
        state.inner_interval = static_cast<std::uint32_t>(
            effectiveInnerInterval);
        state.outer_residual = static_cast<std::uint32_t>(outerResidual);
        state.inner_operand_step = static_cast<std::int32_t>(
            effectiveInnerStep);
        state.row_transition_step = static_cast<std::int32_t>(
            rowTransition);
        state.next_issue_cycle = static_cast<std::uint32_t>(
            schedule.start_cycle);
        state.final_issue_cycle = static_cast<std::uint32_t>(
            schedule.start_cycle
            + outerSteps * schedule.outer_interval + innerCycleSpan);

        validate_induction(state, schedule.outer_stride, outerSteps);
        return pack(state);
    }

    static Packed pack(const State& state)
    {
        validate_mutable_state_fields(state);
        require_signed_fit(state.inner_operand_step,
            inner_operand_step_bits,
            "Active Context inner operand step exceeds 14 bits");
        require_signed_fit(state.row_transition_step,
            row_transition_step_bits,
            "Active Context row transition exceeds 14 bits");
        validate_mutable_state_induction(state);

        Packed result;
        result.write(valid_offset, 1, 1);
        result.write(native_offset, native_bits,
            encode_native(state.instruction));
        result.write(induction_target_offset, induction_target_bits,
            static_cast<std::uint8_t>(state.induction_target));
        result.write(inner_count_offset, inner_count_bits,
            state.inner_count - 1);
        result.write(inner_remaining_offset, inner_remaining_bits,
            state.inner_remaining - 1);
        result.write(outer_remaining_offset, outer_remaining_bits,
            state.outer_remaining - 1);
        result.write(inner_interval_offset, inner_interval_bits,
            state.inner_interval - 1);
        result.write(outer_residual_offset, outer_residual_bits,
            state.outer_residual - 1);
        result.write(inner_operand_step_offset, inner_operand_step_bits,
            encode_signed(state.inner_operand_step,
                inner_operand_step_bits));
        result.write(row_transition_step_offset,
            row_transition_step_bits,
            encode_signed(state.row_transition_step,
                row_transition_step_bits));
        result.write(next_issue_cycle_offset, issue_cycle_bits,
            state.next_issue_cycle);
        result.write(final_issue_cycle_offset, issue_cycle_bits,
            state.final_issue_cycle);
        return result;
    }

    static State unpack(const Packed& packed)
    {
        if (!valid(packed))
            throw std::logic_error("unpack of invalid Active Context");
        const auto target = packed.read(
            induction_target_offset, induction_target_bits);
        if (target > static_cast<std::uint8_t>(
                         IcuInductionTarget::MxmAccumulatorAddress))
            throw std::logic_error(
                "packed Active Context has an invalid induction target");
        const auto innerIntervalMinusOne = packed.read(
            inner_interval_offset, inner_interval_bits);
        const auto outerResidualMinusOne = packed.read(
            outer_residual_offset, outer_residual_bits);
        if (innerIntervalMinusOne
                == std::numeric_limits<std::uint32_t>::max()
            || outerResidualMinusOne
                == std::numeric_limits<std::uint32_t>::max())
            throw std::logic_error(
                "packed Active Context interval is outside 32 bits");

        State result;
        result.instruction = decode_native(packed.read(
            native_offset, native_bits));
        result.induction_target = static_cast<IcuInductionTarget>(target);
        result.inner_count = static_cast<std::uint32_t>(packed.read(
            inner_count_offset, inner_count_bits) + 1);
        result.inner_remaining = static_cast<std::uint32_t>(packed.read(
            inner_remaining_offset, inner_remaining_bits) + 1);
        result.outer_remaining = static_cast<std::uint32_t>(packed.read(
            outer_remaining_offset, outer_remaining_bits) + 1);
        result.inner_interval = static_cast<std::uint32_t>(
            innerIntervalMinusOne + 1);
        result.outer_residual = static_cast<std::uint32_t>(
            outerResidualMinusOne + 1);
        result.inner_operand_step = static_cast<std::int32_t>(decode_signed(
            packed.read(inner_operand_step_offset,
                inner_operand_step_bits), inner_operand_step_bits));
        result.row_transition_step = static_cast<std::int32_t>(decode_signed(
            packed.read(row_transition_step_offset,
                row_transition_step_bits), row_transition_step_bits));
        result.next_issue_cycle = issue_cycle(packed);
        result.final_issue_cycle = release_cycle(packed);
        validate_mutable_state_fields(result);
        validate_mutable_state_induction(result);
        return result;
    }

    // Returns true when the just-issued context has completed.
    static bool advance_after_issue(State& state)
    {
        if (state.inner_remaining > 1) {
            --state.inner_remaining;
            state.next_issue_cycle = checked_cycle_add(
                state.next_issue_cycle, state.inner_interval);
            advance_operand(state, state.inner_operand_step);
            return false;
        }
        if (state.outer_remaining > 1) {
            --state.outer_remaining;
            state.inner_remaining = state.inner_count;
            state.next_issue_cycle = checked_cycle_add(
                state.next_issue_cycle, state.outer_residual);
            advance_operand(state, state.row_transition_step);
            return false;
        }
        return true;
    }

private:
    static constexpr std::size_t max_count = std::size_t{1}
        << inner_count_bits;

    static constexpr std::int64_t signed_min(std::size_t width)
    {
        return -(std::int64_t{1} << (width - 1));
    }

    static constexpr std::int64_t signed_max(std::size_t width)
    {
        return (std::int64_t{1} << (width - 1)) - 1;
    }

    static void require_signed_fit(
        std::int64_t value, std::size_t width, const char* message)
    {
        if (value < signed_min(width) || value > signed_max(width))
            throw StaticScheduleError(message);
    }

    static std::uint64_t encode_signed(
        std::int64_t value, std::size_t width)
    {
        require_signed_fit(value, width,
            "signed Active Context field does not fit");
        return static_cast<std::uint64_t>(value)
            & ((std::uint64_t{1} << width) - 1);
    }

    static std::int64_t decode_signed(
        std::uint64_t value, std::size_t width)
    {
        const auto sign = std::uint64_t{1} << (width - 1);
        if ((value & sign) == 0) return static_cast<std::int64_t>(value);
        return static_cast<std::int64_t>(
            value | (~std::uint64_t{0} << width));
    }

    static void validate_schedule(const IcuMacroSchedule& schedule)
    {
        if (schedule.inner_count == 0 || schedule.outer_count == 0
            || schedule.inner_interval == 0
            || schedule.outer_interval == 0)
            throw StaticScheduleError(
                "Active Context received an invalid iteration space");
        if (schedule.inner_count > max_count
            || schedule.outer_count > max_count)
            throw StaticScheduleError(
                "Macro iteration count exceeds the 16-bit Active Context field");

        constexpr auto maxCycle = static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max());
        const auto innerSteps = schedule.inner_count - 1;
        if (schedule.start_cycle > maxCycle
            || (innerSteps != 0
                && schedule.inner_interval > maxCycle / innerSteps))
            throw StaticScheduleError(
                "Macro issue cycle exceeds the 32-bit Active Context field");
        const auto innerSpan = innerSteps * schedule.inner_interval;
        if (schedule.outer_count > 1
            && schedule.outer_interval <= innerSpan)
            throw StaticScheduleError(
                "Active Context received an overlapping outer iteration");
        const auto outerSteps = schedule.outer_count - 1;
        if (innerSpan > maxCycle - schedule.start_cycle
            || (outerSteps != 0
                && schedule.outer_interval
                    > (maxCycle - schedule.start_cycle - innerSpan)
                        / outerSteps))
            throw StaticScheduleError(
                "Macro final issue cycle exceeds the 32-bit Active Context field");
    }

    static void validate_induction(const State& state,
        std::int64_t outerStride, std::size_t outerSteps)
    {
        if (state.induction_target == IcuInductionTarget::None) {
            if (state.inner_operand_step != 0
                || state.row_transition_step != 0)
                throw StaticScheduleError(
                    "Active Context has operand steps without an induction target");
            return;
        }

        const auto innerSteps = static_cast<std::int64_t>(
            state.inner_count - 1);
        const auto innerDelta = innerSteps * state.inner_operand_step;
        const auto outerDelta = static_cast<std::int64_t>(outerSteps)
            * outerStride;
        const auto base = induction_operand(state.instruction,
            state.induction_target);
        validate_operand(base, state.induction_target);
        validate_operand(base + innerDelta, state.induction_target);
        validate_operand(base + outerDelta, state.induction_target);
        validate_operand(base + innerDelta + outerDelta,
            state.induction_target);
    }

    static void validate_mutable_state_induction(const State& state)
    {
        if (state.induction_target == IcuInductionTarget::None) {
            if (state.inner_operand_step != 0
                || state.row_transition_step != 0)
                throw StaticScheduleError(
                    "Active Context has operand steps without an induction target");
            return;
        }
        const auto operand = induction_operand(
            state.instruction, state.induction_target);
        validate_operand(operand, state.induction_target);
    }

    static void validate_mutable_state_fields(const State& state)
    {
        if (state.inner_count == 0 || state.inner_count > max_count
            || state.inner_remaining == 0
            || state.inner_remaining > state.inner_count
            || state.outer_remaining == 0
            || state.outer_remaining > max_count
            || state.inner_interval == 0 || state.outer_residual == 0
            || state.next_issue_cycle > state.final_issue_cycle)
            throw StaticScheduleError(
                "invalid state at Active Context pack boundary");
    }

    static std::int64_t induction_operand(const FuncInstruction& instruction,
        IcuInductionTarget target)
    {
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            if (target != IcuInductionTarget::MemAddress)
                throw StaticScheduleError(
                    "MEM Active Context has an invalid induction target");
            return static_cast<std::int64_t>(instruction.address);
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            if (target == IcuInductionTarget::MxmWeightColumn) {
                if (instruction.opcode != MxmControlOpcode::IW)
                    throw StaticScheduleError(
                        "MXM weight-column induction requires IW");
                return static_cast<std::int64_t>(instruction.weight_column);
            }
            if (target == IcuInductionTarget::MxmAccumulatorAddress) {
                if (instruction.opcode != MxmControlOpcode::Compute
                    && instruction.opcode
                        != MxmControlOpcode::AccumulatorRead
                    && !(instruction.opcode == MxmControlOpcode::Decode
                        && instruction.decode_operation
                            == MxmDecodeOperation::StreamCompute))
                    throw StaticScheduleError(
                        "MXM accumulator induction requires a compute instruction");
                return static_cast<std::int64_t>(
                    instruction.accumulator_address);
            }
            throw StaticScheduleError(
                "MXM Active Context has an invalid induction target");
        } else {
            throw StaticScheduleError(
                "this queue type does not support Macro induction");
        }
    }

    static void validate_operand(
        std::int64_t operand, IcuInductionTarget target)
    {
        std::int64_t limit = 0;
        switch (target) {
        case IcuInductionTarget::MemAddress:
            limit = static_cast<std::int64_t>(hw::kSramDepthRows);
            break;
        case IcuInductionTarget::MxmWeightColumn:
            limit = static_cast<std::int64_t>(hw::kMxmSupercellsPerPlane);
            break;
        case IcuInductionTarget::MxmAccumulatorAddress:
            limit = static_cast<std::int64_t>(hw::kMxmAccumulatorRows);
            break;
        case IcuInductionTarget::None:
            throw std::logic_error(
                "None induction target has no operand domain");
        }
        if (operand < 0 || operand >= limit)
            throw StaticScheduleError(
                "Macro induction leaves the physical operand range");
    }

    static void advance_operand(State& state, std::int32_t step)
    {
        if (step == 0) return;
        const auto current = induction_operand(
            state.instruction, state.induction_target);
        const auto next = current + step;
        validate_operand(next, state.induction_target);
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            state.instruction.address = static_cast<std::size_t>(next);
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            if (state.induction_target
                == IcuInductionTarget::MxmWeightColumn)
                state.instruction.weight_column =
                    static_cast<std::size_t>(next);
            else
                state.instruction.accumulator_address =
                    static_cast<std::size_t>(next);
        }
    }

    static std::uint32_t checked_cycle_add(
        std::uint32_t cycle, std::uint32_t delta)
    {
        if (delta > std::numeric_limits<std::uint32_t>::max() - cycle)
            throw StaticScheduleError(
                "Active Context next issue cycle overflowed");
        return cycle + delta;
    }

    static std::uint64_t encode_native(
        const FuncInstruction& instruction)
    {
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            return isa::encode_mem_instruction(instruction);
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            return isa::encode_mxm_instruction(instruction);
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmDequantInstruction>) {
            return isa::encode_mxm_dequant_instruction(instruction);
        } else {
            throw StaticScheduleError(
                "this queue instruction has no 256-bit Active Context codec");
        }
    }

    static FuncInstruction decode_native(std::uint64_t native)
    {
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            return isa::decode_mem_instruction(native);
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            return isa::decode_mxm_instruction(native);
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmDequantInstruction>) {
            return isa::decode_mxm_dequant_instruction(
                static_cast<std::uint16_t>(native));
        } else {
            throw StaticScheduleError(
                "this queue instruction has no 256-bit Active Context codec");
        }
    }
};

} // namespace ftlpu
