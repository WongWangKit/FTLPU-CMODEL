#pragma once

#include "ftlpu/icu/icu.hpp"
#include "ftlpu/program/sram_layout.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
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

struct MemIcuLocalBootstrap {
    std::size_t mem_slice{0};
    MemLocalWordAddress13 program_address{};
};

// Directly loaded bootstrap state is intentionally small. The MEM loader
// enters its IQ through the local-SRAM reset address; functional program
// blocks remain in SRAM and enter other IQs only through MEM Read + Fetch.
struct BootstrapPreamble {
    std::vector<MemIcuLocalBootstrap> mem_local_bootstraps{};
    std::vector<BootstrapEntry> entries{};
};

struct MemIcuReadRequest {
    ProgramBlockPlacement block{};
    std::size_t delay_before_cycles{0};
    bool notify_after{false};
};

// Builds the small program that a MEM ICU will fetch from its own local SRAM.
// The returned section is still DMA-loaded like every other ProgramSection.
inline ProgramSection build_mem_icu_loader_section(
    std::size_t mem_slice,
    StreamId instruction_stream,
    const std::vector<MemIcuReadRequest>& requests,
    std::string metadata = "MEM ICU local loader")
{
    if (requests.empty()) {
        throw std::invalid_argument("MEM ICU loader needs at least one program block");
    }
    ProgramSection section {IcuLocation::Mem(mem_slice), {}, 0, std::move(metadata)};
    for (const auto& request : requests) {
        if (request.block.memory_address.mem_slice() != mem_slice) {
            throw StaticScheduleError(
                "MEM ICU loader can read program blocks only from its local MEM slice");
        }
        const auto local = request.block.memory_address.slice_byte_address()
                               .local_word_address();
        if (local.bank() != local.next_word().bank()) {
            throw StaticScheduleError(
                "MEM ICU loader program block cannot cross an SRAM bank");
        }
        if (request.delay_before_cycles != 0) {
            section.packets.push_back(program::encode_packet(
                IcuControlInstruction::Nop(request.delay_before_cycles)));
        }
        section.packets.push_back(program::encode_packet(
            MemInstruction::Read(local, instruction_stream)));
        section.packets.push_back(program::encode_packet(
            MemInstruction::Read(local.next_word(), instruction_stream)));
        if (request.notify_after) {
            section.packets.push_back(program::encode_packet(
                IcuControlInstruction::Notify()));
        }
    }
    if (section.packets.size() > hw::kIcuFetchPackets) {
        throw StaticScheduleError(
            "initial MEM ICU local loader exceeds one 640-byte bootstrap block");
    }
    return section;
}

class BootstrapPreambleBuilder {
public:
    static BootstrapPreamble ForAutonomousMem(
        const ProgramBlockPlacement& mem_loader,
        const ProgramBlockPlacement& target_block,
        StreamId instruction_stream)
    {
        if (mem_loader.target.kind != IcuLocationKind::Mem
            || mem_loader.target.index != mem_loader.memory_address.mem_slice()) {
            throw StaticScheduleError(
                "MEM ICU bootstrap program must reside in its own MEM slice");
        }
        const auto loader_address = mem_loader.memory_address
                                        .slice_byte_address()
                                        .local_word_address();
        if (loader_address.bank() != loader_address.next_word().bank()) {
            throw StaticScheduleError(
                "MEM ICU local bootstrap block cannot cross an SRAM bank");
        }

        BootstrapPreamble preamble;
        preamble.mem_local_bootstraps.push_back(MemIcuLocalBootstrap {
            mem_loader.target.index, loader_address});
        preamble.entries = {
            {target_block.target,
             IcuControlInstruction::Fetch(instruction_stream)},
            {target_block.target, IcuControlInstruction::Sync()},
        };
        return preamble;
    }

    // Legacy direct-Read bootstrap retained for source compatibility. New
    // program-loading code should use ForAutonomousMem().
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
    for (const auto& bootstrap : preamble.mem_local_bootstraps) {
        icu.bootstrap_mem_icu_from_local_sram(
            bootstrap.mem_slice, bootstrap.program_address);
    }
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
