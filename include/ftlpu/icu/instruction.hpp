#pragma once

#include <array>
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
    WaitEvent = 7,
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

// A Macro owns its native FU template and absolute issue schedule.
// It is encoded by the packed Macro codec, not a separate loop opcode.
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
    std::size_t event_tag{0};

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

    static constexpr IcuControlInstruction Sync() noexcept
    {
        return IcuControlInstruction {IcuControlOpcode::Sync};
    }

    static constexpr IcuControlInstruction Notify() noexcept
    {
        return IcuControlInstruction {IcuControlOpcode::Notify};
    }

    static constexpr IcuControlInstruction WaitEvent(
        std::size_t tag) noexcept
    {
        IcuControlInstruction instruction;
        instruction.opcode = IcuControlOpcode::WaitEvent;
        instruction.event_tag = tag;
        return instruction;
    }
};

} // namespace ftlpu
