#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"
#include "ftlpu/core/bf16.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>

using namespace ftlpu;
namespace rt = ftlpu::software::runtime;

namespace {

rt::QueueCommand mem_macro(
    MemInstruction instruction, std::size_t start, std::size_t count = 4)
{
    const auto encoded = isa::encode_mem_instruction(instruction);
    rt::QueueCommand command {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mem,
        2,
        {static_cast<std::uint32_t>(encoded),
         static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
    return rt::encode_macro_schedule_command(
        std::move(command),
        IcuMacroSchedule {
            start, count, 1, 1, 1, 1, 0, IcuInductionTarget::MemAddress});
}

rt::QueueCommand mxm_macro(
    MxmControlInstruction instruction,
    std::size_t start,
    std::size_t count,
    std::int64_t stride,
    IcuInductionTarget target)
{
    const auto encoded = isa::encode_mxm_instruction(instruction);
    rt::QueueCommand command {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mxm,
        2,
        {static_cast<std::uint32_t>(encoded),
         static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
    return rt::encode_macro_schedule_command(
        std::move(command),
        IcuMacroSchedule {start, count, 1, stride, 1, 1, 0, target});
}

rt::BinaryProgram make_program()
{
    rt::BinaryProgram program;
    const auto latency = [](std::size_t slice) {
        return hw::kMemGroups + hw::kC2cToSxmStreamRegisterColumns + 1
            - slice / hw::kMemSlicesPerGroup;
    };
    for (std::size_t slice = 0; slice < 16; ++slice) {
        program.queues.push_back({
            rt::QueueKind::Mem,
            InstructionControlUnit::mem_queue(Hemisphere::East, slice, 0),
            {mem_macro(
                MemInstruction::Read(64, StreamId::East(slice)),
                20 - latency(slice), 4)},
        });
    }
    program.queues.push_back({
        rt::QueueKind::MxmLoad,
        0,
        {mxm_macro(
            MxmControlInstruction::IWDirect16(0, 0),
            20, 4, 1, IcuInductionTarget::MxmWeightColumn)},
    });
    program.max_cycle = 24;
    return program;
}

void initialize(TspSliceSystem& system)
{
    for (std::size_t row = 0; row < hw::kMxmRows; ++row) {
        for (std::size_t col = 0; col < hw::kMxmColumns; ++col) {
            const auto value = static_cast<float>(
                static_cast<int>((row * 5 + col * 3) % 23) - 11) / 16.0f;
            const auto bits = Bf16::from_float(value).bits();
            for (std::size_t byte = 0; byte < 2; ++byte) {
                system.initialize_mem_sram_lane_byte(
                    Hemisphere::East,
                    (col % hw::kMxmSupercellColumns) * 2 + byte,
                    0,
                    row / hw::kLanesPerTile,
                    64 + col / hw::kMxmSupercellColumns,
                    row % hw::kLanesPerTile,
                    static_cast<std::uint8_t>(bits >> (byte * 8)));
            }
        }
    }
}

} // namespace

int main()
{
    try {
        TspSliceSystem system;
        rt::CModelRuntime runtime(system);
        runtime.load_direct(make_program());
        initialize(system);

        TspSliceSystem::LogSinks sinks;
        sinks.icu = &std::cout;
        sinks.mem = &std::cout;
        sinks.mxm = &std::cout;
        sinks.system = &std::cout;
        sinks.mem_log_tile = 0;
        sinks.mxm_log_tile = 0;
        sinks.capture_mem_trace = true;

        for (std::size_t step = 0; step < 25; ++step) {
            std::cout << "B1_DIAG before_tick cycle=" << system.cycle() << '\n';
            system.tick(sinks);
            const auto& mem = system.mem_array(Hemisphere::East);
            std::cout << "B1_DIAG after_tick cycle=" << system.cycle()
                      << " executed_mem=" << mem.executed_transfers().size()
                      << " staged_responses=" << mem.staged_read_responses().size()
                      << " pending_responses=" << mem.read_responses().size()
                      << '\n';
        }
        std::cout << "B1_DIAG completed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "B1_DIAG failure: " << error.what() << '\n';
        return 1;
    }
}
