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
};

struct IcuRepeat {
    std::size_t count{0};
    std::size_t interval{1};
    std::int64_t address_stride{0};
};

struct IcuControlInstruction {
    IcuControlOpcode opcode{IcuControlOpcode::Nop};
    std::size_t count{0};
    std::size_t interval{1};
    std::int64_t address_stride{0};

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
};

} // namespace ftlpu
