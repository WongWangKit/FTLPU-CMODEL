#pragma once

#include "ftlpu/icu/icu.hpp"
#include "ftlpu/program/sram_layout.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ftlpu {

using BootstrapInstruction = std::variant<
    IcuControlInstruction,
    MemInstruction,
    MxmControlInstruction,
    VxmLaneAluInstruction>;

struct BootstrapEntry {
    IcuLocation location{};
    BootstrapInstruction instruction{};
};

// Directly loaded bootstrap state is intentionally small.  The body of the
// program remains in SRAM and can enter an IQ only through MEM Read + Fetch.
struct BootstrapPreamble {
    std::vector<BootstrapEntry> entries{};
};

class BootstrapPreambleBuilder {
public:
    static BootstrapPreamble ForProgramBlock(
        const ProgramBlockPlacement& block,
        StreamId instruction_stream,
        IcuLocation notifier)
    {
        const auto local = block.memory_address.slice_byte_address()
                               .local_word_address();
        if (local.bank() != local.next_word().bank()) {
            throw std::logic_error("bootstrap program block unexpectedly crosses an SRAM bank");
        }
        if (notifier == block.target) {
            throw StaticScheduleError(
                "bootstrap notifier cannot be behind Sync in the target IQ");
        }

        BootstrapPreamble preamble;
        const auto mem = IcuLocation::Mem(block.memory_address.mem_slice());
        preamble.entries = {
            {mem, MemInstruction::Read(local, instruction_stream)},
            {mem, MemInstruction::Read(local.next_word(), instruction_stream)},
            {block.target, IcuControlInstruction::Fetch(instruction_stream)},
            {block.target, IcuControlInstruction::Sync()},
            {notifier, IcuControlInstruction::Notify()},
        };
        return preamble;
    }
};

inline void load_bootstrap_preamble(
    InstructionControlUnit& icu,
    const BootstrapPreamble& preamble)
{
    for (const auto& entry : preamble.entries) {
        std::visit(
            [&](const auto& instruction) {
                using T = std::decay_t<decltype(instruction)>;
                if constexpr (std::is_same_v<T, IcuControlInstruction>) {
                    icu.enqueue_control(entry.location, instruction);
                } else if constexpr (std::is_same_v<T, MemInstruction>) {
                    if (entry.location.kind != IcuLocationKind::Mem) {
                        throw StaticScheduleError(
                            "bootstrap MEM instruction targets a non-MEM ICU");
                    }
                    icu.enqueue_mem(entry.location.index, instruction);
                } else if constexpr (std::is_same_v<T, MxmControlInstruction>) {
                    const auto load = instruction.opcode == MxmControlOpcode::IW;
                    if ((load && entry.location.kind != IcuLocationKind::MxmLoad)
                        || (!load && entry.location.kind != IcuLocationKind::MxmCompute)) {
                        throw StaticScheduleError(
                            "bootstrap MXM instruction targets the wrong MXM ICU port");
                    }
                    icu.enqueue_mxm(entry.location.unit, instruction);
                } else if constexpr (std::is_same_v<T, VxmLaneAluInstruction>) {
                    if (entry.location.kind != IcuLocationKind::Vxm) {
                        throw StaticScheduleError(
                            "bootstrap VXM instruction targets a non-VXM ICU");
                    }
                    icu.enqueue_vxm(entry.location.index, instruction);
                }
            },
            entry.instruction);
    }
}

} // namespace ftlpu
