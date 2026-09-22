#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/system/icu.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
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

    static_assert(InstructionControlUnit::MxmIcu::macro_context_depth == 40);
    static_assert(
        InstructionControlUnit::MxmIcu::macro_issue_compare_width == 40);
    static_assert(
        InstructionControlUnit::MxmIcu::macro_issue_cycle_bits == 32);

    Queue cycleWidth;
    bool cycleWidthEnforced = false;
    try {
        cycleWidth.push_macro(IcuMacroSchedule {
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()) + 1,
            1, 1, 0, 1, 1, 0,
            IcuInductionTarget::MemAddress,
        }, MemInstruction::Read(0, 0));
    } catch (const std::overflow_error&) {
        cycleWidthEnforced = true;
    }
    require(cycleWidthEnforced,
        "Macro scheduler accepted a cycle outside its 32-bit compare key");

    using Flat40Queue =
        DistributedIcuQueue<MemInstruction, 128, 128, 64, 1, 40>;
    Flat40Queue flat40;
    for (std::size_t context = 0; context < 41; ++context) {
        flat40.push_macro(IcuMacroSchedule {
            100 + context, 2, 1000, 1, 1, 1, 0,
            IcuInductionTarget::MemAddress,
        }, MemInstruction::Read(1000 + context, 0));
    }
    for (std::size_t cycle = 0; cycle < 140; ++cycle)
        static_cast<void>(flat40.tick());
    require(flat40.peak_active_macros() == 40,
        "flat Macro scheduler did not populate all 40 physical entries");
    bool flat40Overflow = false;
    try {
        static_cast<void>(flat40.tick());
    } catch (const StaticScheduleError&) {
        flat40Overflow = true;
    }
    require(flat40Overflow,
        "flat Macro scheduler accepted a 41st live context");

    Queue collision;
    collision.push_macro(IcuMacroSchedule {
        3, 1, 1, 0, 1, 1, 0,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(300, 0));
    collision.push_macro(IcuMacroSchedule {
        3, 1, 1, 0, 1, 1, 0,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(400, 0));
    bool collisionDetected = false;
    try {
        for (std::size_t cycle = 0; cycle <= 3; ++cycle)
            static_cast<void>(collision.tick());
    } catch (const StaticScheduleError&) {
        collisionDetected = true;
    }
    require(collisionDetected,
        "flat Macro scheduler did not reject a multi-match cycle");
    bool lateDetected = false;
    try {
        static_cast<void>(collision.tick());
    } catch (const StaticScheduleError&) {
        lateDetected = true;
    }
    require(lateDetected,
        "flat Macro scheduler dynamically delayed a missed issue cycle");

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

    InstructionControlUnit icu;
    icu.enqueue_mem_macro(0, IcuMacroSchedule {
        3, 3, 4, 1, 1, 1, 0,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(100, 0));
    icu.enqueue_mem_macro(0, IcuMacroSchedule {
        4, 3, 4, 1, 1, 1, 0,
        IcuInductionTarget::MemAddress,
    }, MemInstruction::Read(200, 1));
    for (std::size_t cycle = 0; cycle < 13; ++cycle)
        static_cast<void>(icu.mem_iq(0).tick());
    const auto statistics = icu.frontend_statistics();
    require(statistics.imem_entries == 2,
        "Macro frontend statistics reported incorrect i-MEM entries");
    require(statistics.fetched_entries == 2,
        "Macro frontend statistics reported incorrect fetched entries");
    require(statistics.issued_instructions == 6,
        "Macro frontend statistics reported incorrect dynamic issues");
    require(statistics.macro_queues == 1,
        "Macro frontend statistics did not identify the Macro queue");
    require(statistics.peak_macro_contexts_per_queue == 2,
        "Macro frontend statistics reported incorrect context pressure");

    std::cout << "icu_macro_schedule_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_macro_schedule_test failed: " << error.what() << '\n';
    return 1;
}
