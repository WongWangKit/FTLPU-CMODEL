#include "ftlpu/mem/slice.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main()
{
    ftlpu::MemSlice<std::uint32_t> mem({100, 101, 102, 103, 104}, 3);

    mem.issue(ftlpu::MemInstruction::Read(1, 7));
    assert(mem.busy());

    mem.tick();
    assert(mem.output().has_value());
    assert(mem.output()->stream == 7);
    assert(mem.output()->word.data == 101);
    assert(!mem.output()->word.last);

    mem.tick();
    assert(mem.output().has_value());
    assert(mem.output()->stream == 7);
    assert(mem.output()->word.data == 102);
    assert(!mem.output()->word.last);

    mem.tick();
    assert(mem.output().has_value());
    assert(mem.output()->stream == 7);
    assert(mem.output()->word.data == 103);
    assert(mem.output()->word.last);
    assert(!mem.busy());

    mem.tick();
    assert(!mem.output().has_value());

    bool caught = false;
    try {
        mem.issue(ftlpu::MemInstruction::Read(4, 0));
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    caught = false;
    try {
        mem.issue(ftlpu::MemInstruction::Write(0, 1));
    } catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    return 0;
}
