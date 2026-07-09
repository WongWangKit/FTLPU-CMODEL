#pragma once

#include <cstddef>

namespace ftlpu {

enum class IcuOpcode {
    Dispatch,
    Nop,
    Repeat,
    Sync,
    Notify,
};

struct IcuInstruction {
    IcuOpcode opcode{IcuOpcode::Dispatch};
    std::size_t repeat_count{1};
    std::size_t repeat_interval{1};

    static IcuInstruction Dispatch()
    {
        return IcuInstruction{IcuOpcode::Dispatch, 0, 0};
    }

    static IcuInstruction Nop(std::size_t count)
    {
        return IcuInstruction{IcuOpcode::Nop, count, 0};
    }

    static IcuInstruction Repeat(std::size_t count, std::size_t interval = 1)
    {
        return IcuInstruction{IcuOpcode::Repeat, count, interval};
    }

    static IcuInstruction Sync()
    {
        return IcuInstruction{IcuOpcode::Sync, 0, 0};
    }

    static IcuInstruction Notify()
    {
        return IcuInstruction{IcuOpcode::Notify, 0, 0};
    }
};

} // namespace ftlpu
