#include "ftlpu/core/instruction_pipeline.hpp"
#include "ftlpu/icu/icu.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

int main()
{
    using Control = ftlpu::IcuControlInstruction;
    ftlpu::SliceIcu<ftlpu::MemInstruction> icu;
    icu.load({
        ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(1)),
        Control::Sync(),
        ftlpu::MemInstruction::Write(20, ftlpu::StreamId::East(2)),
    });

    ftlpu::NorthboundInstructionPipeline mem_pipeline;
    std::ostringstream trace;
    bool read_dispatched = false;
    bool write_dispatched = false;

    for (std::size_t cycle = 0; cycle < 10; ++cycle) {
        if (cycle == 4) {
            icu.notify();
            trace << "runtime notifies MEM IQ\n";
        }

        if (const auto instruction = icu.tick(); instruction.has_value()) {
            mem_pipeline.issue_south(*instruction);
            read_dispatched |= instruction->opcode == ftlpu::MemOpcode::Read;
            write_dispatched |= instruction->opcode == ftlpu::MemOpcode::Write;
        }
        mem_pipeline.tick(trace);
    }

    if (!read_dispatched || !write_dispatched || !icu.done()) {
        throw std::logic_error("unified MEM IQ did not complete Read/Sync/Write");
    }

    std::cout << trace.str();
    std::cout << "OK: one MEM IQ preserves Read -> Sync -> Write program order\n";
    return 0;
}
