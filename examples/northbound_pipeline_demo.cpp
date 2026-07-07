#include "ftlpu/instruction_pipeline.hpp"

#include <iostream>

int main()
{
    ftlpu::NorthboundInstructionPipeline pipe;

    pipe.issue_south(ftlpu::MemInstruction::Read(0, 1));
    pipe.tick(std::cout);

    pipe.issue_south(ftlpu::MemInstruction::Write(320, 2));
    pipe.tick(std::cout);

    pipe.issue_south(ftlpu::MemInstruction::Gather(3, 8));
    pipe.tick(std::cout);

    pipe.issue_south(ftlpu::MemInstruction::Scatter(4, 9));
    pipe.tick(std::cout);

    for (int i = 0; i < 4; ++i) {
        pipe.tick(std::cout);
    }

    return 0;
}
