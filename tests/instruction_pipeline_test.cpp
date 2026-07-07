#include "ftlpu/instruction_pipeline.hpp"

#include <cassert>
#include <sstream>
#include <string>

int main()
{
    ftlpu::NorthboundInstructionPipeline pipe;
    std::ostringstream log;

    pipe.issue_south(ftlpu::MemInstruction::Read(16, 3));
    pipe.tick(log);
    assert(!pipe.row(0).has_value());
    assert(pipe.row(1).has_value());
    assert(pipe.row(1)->opcode == ftlpu::MemOpcode::Read);
    assert(pipe.row(1)->address == 16);
    assert(pipe.row(1)->stream == 3);

    pipe.tick(log);
    assert(!pipe.row(1).has_value());
    assert(pipe.row(2).has_value());

    pipe.issue_south(ftlpu::MemInstruction::Gather(5, 9));
    pipe.tick(log);
    assert(pipe.row(1).has_value());
    assert(pipe.row(1)->opcode == ftlpu::MemOpcode::Gather);
    assert(pipe.row(3).has_value());
    assert(pipe.row(3)->opcode == ftlpu::MemOpcode::Read);

    const std::string text = log.str();
    assert(text.find("cycle 0: row 0 executes Read a=16 s=3;") != std::string::npos);
    assert(text.find("cycle 2: row 0 executes Gather s=5 map=9; row 2 executes Read a=16 s=3;") != std::string::npos);

    return 0;
}
