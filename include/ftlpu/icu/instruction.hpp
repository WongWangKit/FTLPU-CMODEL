#pragma once

#include <cstddef>
#include <cstdint>

namespace ftlpu {

enum class IcuControlOpcode : std::uint8_t {
    Fetch,
    Nop,
    Repeat,
    Sync,
    Notify,
};

struct IcuRepeat {
    std::size_t count{0};
    std::size_t interval{1};
    std::int64_t address_stride{0};
};

struct IcuControlInstruction {
    IcuControlOpcode opcode{IcuControlOpcode::Fetch};
    std::size_t count{0};
    std::size_t interval{1};
    std::int64_t address_stride{0};

    static constexpr IcuControlInstruction Fetch() noexcept
    {
        return IcuControlInstruction {IcuControlOpcode::Fetch};
    }

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
