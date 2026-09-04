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

    const IcuMemStreamNdSchedule schedule {
        3,
        3,
        {3, 2, 2},
        {2, 8, 20},
        {1, 16, 64},
    };
    Queue queue;
    queue.push_mem_stream_nd(schedule, MemInstruction::Read(100, 0));
    queue.push_mem_stream_nd(IcuMemStreamNdSchedule {
        4, 1, {3, 1, 1}, {4, 1, 1}, {1, 0, 0}},
        MemInstruction::Read(200, 1));
    require(queue.imem_occupancy() == 2,
        "MEM_STREAM_ND did not remain one FIFO entry per descriptor");

    std::vector<std::pair<std::size_t, std::size_t>> issues;
    for (std::size_t cycle = 0; cycle < 36; ++cycle) {
        if (const auto instruction = queue.tick())
            issues.emplace_back(cycle, instruction->address);
    }
    require(issues == std::vector<std::pair<std::size_t, std::size_t>> {
        {3, 100}, {4, 200}, {5, 101}, {7, 102}, {8, 201},
        {11, 116}, {12, 202}, {13, 117}, {15, 118},
        {23, 164}, {25, 165}, {27, 166}, {31, 180},
        {33, 181}, {35, 182}},
        "MEM_STREAM_ND emitted incorrect cycles or addresses");
    require(queue.done(), "MEM_STREAM_ND queue did not complete");

    std::cout << "icu_mem_stream_nd_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_mem_stream_nd_test failed: " << error.what() << '\n';
    return 1;
}
