#include "ftlpu/icu/distributed_queue.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main()
try {
    using namespace ftlpu;
    using Queue = DistributedIcuQueue<MemInstruction, 128, 32, 8>;

    Queue queue;
    queue.push_mem_slice_program(IcuMemSliceProgram {
        IcuStreamNdSchedule {
            3, 1, {3, 1, 1}, {2, 1, 1}, {0, 0, 0},
            IcuInductionTarget::None},
        {
            IcuMemSliceProgramEntry {
                0, {1, 0, 0}, MemInstruction::Read(100, 0)},
            IcuMemSliceProgramEntry {
                1, {2, 0, 0}, MemInstruction::Write(200, 1)},
        },
    });
    require(queue.imem_occupancy() == 1,
        "MEM_SLICE_PROGRAM did not occupy one logical i-MEM entry");

    std::vector<std::pair<std::size_t, MemInstruction>> issues;
    for (std::size_t cycle = 0; cycle < 12; ++cycle) {
        if (const auto instruction = queue.tick())
            issues.emplace_back(cycle, *instruction);
    }
    require(issues.size() == 6,
        "MEM_SLICE_PROGRAM emitted the wrong number of operations");
    require(issues[0].first == 3
            && issues[0].second.opcode == MemOpcode::Read
            && issues[0].second.address == 100
            && issues[1].first == 4
            && issues[1].second.opcode == MemOpcode::Write
            && issues[1].second.address == 200
            && issues[2].first == 5
            && issues[2].second.address == 101
            && issues[3].first == 6
            && issues[3].second.address == 202
            && issues[4].first == 7
            && issues[4].second.address == 102
            && issues[5].first == 8
            && issues[5].second.address == 204,
        "MEM_SLICE_PROGRAM emitted incorrect cycles, opcodes, or addresses");
    require(queue.done(), "MEM_SLICE_PROGRAM queue did not complete");

    bool rejected = false;
    try {
        Queue invalid;
        invalid.push_mem_slice_program(IcuMemSliceProgram {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "empty MEM_SLICE_PROGRAM body was accepted");

    std::cout << "icu_mem_slice_program_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_mem_slice_program_test failed: "
              << error.what() << '\n';
    return 1;
}
