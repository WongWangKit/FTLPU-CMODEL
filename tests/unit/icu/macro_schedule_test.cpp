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
    queue.push_macro(IcuMacroSchedule {
        3, 3, 2, 1, 2, 8, 16,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(100, 0));

    std::vector<std::pair<std::size_t, std::size_t>> issues;
    for (std::size_t cycle = 0; cycle < 16; ++cycle) {
        if (const auto instruction = queue.tick())
            issues.emplace_back(cycle, instruction->address);
    }
    require(issues == std::vector<std::pair<std::size_t, std::size_t>> {
        {3, 100}, {5, 101}, {7, 102},
        {11, 116}, {13, 117}, {15, 118}},
        "ICU macro emitted incorrect cycles or addresses");
    require(queue.done(), "ICU macro queue did not complete");

    Queue interleaved;
    interleaved.push_macro(IcuMacroSchedule {
        3, 3, 4, 1, 1, 1, 0,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(100, 0));
    interleaved.push_macro(IcuMacroSchedule {
        4, 3, 4, 1, 1, 1, 0,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(200, 1));
    issues.clear();
    for (std::size_t cycle = 0; cycle < 13; ++cycle) {
        if (const auto instruction = interleaved.tick())
            issues.emplace_back(cycle, instruction->address);
    }
    require(issues == std::vector<std::pair<std::size_t, std::size_t>> {
        {3, 100}, {4, 200}, {7, 101},
        {8, 201}, {11, 102}, {12, 202}},
        "ICU did not interleave independent macro descriptors");
    require(interleaved.done(), "interleaved macro queue did not complete");
    require(interleaved.peak_active_macros() == 2,
        "interleaved macro peak-context count is incorrect");

    using OneContextQueue =
        DistributedIcuQueue<MemInstruction, 128, 32, 8, 1, 1>;
    OneContextQueue constrained;
    constrained.push_macro(IcuMacroSchedule {
        3, 3, 4, 1, 1, 1, 0,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(100, 0));
    constrained.push_macro(IcuMacroSchedule {
        4, 3, 4, 1, 1, 1, 0,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(200, 1));
    bool contextOverflow = false;
    try {
        for (std::size_t cycle = 0; cycle < 6; ++cycle)
            (void)constrained.tick();
    } catch (const StaticScheduleError&) {
        contextOverflow = true;
    }
    require(contextOverflow,
        "finite ICU Macro context capacity was not enforced");

    std::cout << "icu_macro_schedule_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_macro_schedule_test failed: " << error.what() << '\n';
    return 1;
}
