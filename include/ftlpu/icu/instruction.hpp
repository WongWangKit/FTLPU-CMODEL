#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ftlpu {

// Fail-fast error for a statically scheduled program that exceeds a modeled
// finite resource or requests an impossible overlap.
class StaticScheduleError : public std::logic_error {
public:
    explicit StaticScheduleError(const std::string& message)
        : std::logic_error(message)
    {
    }
};

enum class IcuControlOpcode : std::uint8_t {
    Nop = 1,
    Repeat = 2,
    Sync = 3,
    Notify = 4,
    Loop = 5,
    Repeat2D = 6,
};

enum class IcuInductionTarget : std::uint8_t {
    None = 0,
    MemAddress = 1,
    MxmWeightColumn = 2,
    MxmAccumulatorAddress = 3,
};

struct IcuRepeat {
    std::size_t count{0};
    std::size_t interval{1};
    std::int64_t address_stride{0};
};

struct IcuLoop {
    std::size_t window_size{0};
    std::size_t count{0};
    std::size_t interval{1};
    std::int64_t address_stride{0};
};

// Total iteration counts include the functional instruction that was already
// issued at coordinate (0, 0). The control entry emits all remaining points in
// outer-major, inner-minor order.
struct IcuRepeat2D {
    std::size_t inner_count{1};
    std::size_t inner_interval{1};
    std::int64_t inner_stride{0};
    std::size_t outer_count{1};
    std::size_t outer_interval{1};
    std::int64_t outer_stride{0};
    IcuInductionTarget induction_target{IcuInductionTarget::None};
};

// An absolute-cycle macro entry. Unlike Repeat/Repeat2D, this descriptor owns
// its functional instruction and therefore needs neither a leading
// instruction nor queue-local NOP padding. The local ICU expands one native
// instruction at every point in the two-dimensional iteration space.
struct IcuMacroSchedule {
    std::size_t start_cycle{0};
    std::size_t inner_count{1};
    std::size_t inner_interval{1};
    std::int64_t inner_stride{0};
    std::size_t outer_count{1};
    std::size_t outer_interval{1};
    std::int64_t outer_stride{0};
    IcuInductionTarget induction_target{IcuInductionTarget::None};
};

struct IcuControlInstruction {
    IcuControlOpcode opcode{IcuControlOpcode::Nop};
    std::size_t count{0};
    std::size_t interval{1};
    std::int64_t address_stride{0};
    std::size_t window_size{0};
    IcuRepeat2D repeat_2d{};

    static constexpr IcuControlInstruction Nop(std::size_t cycles) noexcept
    {
        return IcuControlInstruction {IcuControlOpcode::Nop, cycles};
    }

    static constexpr IcuControlInstruction Repeat(
        std::size_t count,
        std::size_t interval = 1,
        std::int64_t address_stride = 0) noexcept
    {
        return IcuControlInstruction {
            IcuControlOpcode::Repeat,
            count,
            interval,
            address_stride};
    }

    static constexpr IcuControlInstruction Loop(
        std::size_t window_size,
        std::size_t count,
        std::size_t interval,
        std::int64_t address_stride = 0) noexcept
    {
        return IcuControlInstruction {
            IcuControlOpcode::Loop,
            count,
            interval,
            address_stride,
            window_size};
    }

    static constexpr IcuControlInstruction Repeat2D(
        IcuRepeat2D repeat) noexcept
    {
        IcuControlInstruction instruction;
        instruction.opcode = IcuControlOpcode::Repeat2D;
        instruction.repeat_2d = repeat;
        return instruction;
    }

    static constexpr IcuControlInstruction Sync() noexcept
    {
        return IcuControlInstruction {IcuControlOpcode::Sync};
    }

    static constexpr IcuControlInstruction Notify() noexcept
    {
        return IcuControlInstruction {IcuControlOpcode::Notify};
    }
};

} // namespace ftlpu
