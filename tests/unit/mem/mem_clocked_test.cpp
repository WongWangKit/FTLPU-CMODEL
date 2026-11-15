#include "ftlpu/core/timing.hpp"
#include "ftlpu/mem/mem_array.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using namespace ftlpu;

struct WriteResult {
    std::uint8_t value{};
    std::size_t mem_cycle{};
    std::size_t fabric_cycle{};
};

WriteResult run_write_edge(bool reverse_registration)
{
    StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);
    MemArrayModel mem;
    timing::Clock clock;
    timing::SystemPhase phases;
    std::vector<int> order;

    if (reverse_registration) {
        clock.register_module(mem);
        clock.register_module(fabric);
    } else {
        clock.register_module(fabric);
        clock.register_module(mem);
    }
    phases.add_node("open MEM fabric", [&] {
        order.push_back(1);
        fabric.low();
    }, [&] { order.push_back(3); });
    phases.add_node("evaluate MEM", [&] {
        order.push_back(2);
        mem.evaluate(fabric);
    }, [&] {
        order.push_back(4);
        mem.high();
    });

    const auto stream = StreamId::East(0);
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        fabric.initialize_cell(0, 0, lane, stream,
            StreamCell::Valid(static_cast<std::uint8_t>(0x50 + lane)));
    }
    mem.enqueue_instruction(0, MemInstruction::Write(7, stream));

    phases.low();
    assert((order == std::vector<int>{1, 2}));
    assert(mem.cycle() == 0);
    assert(mem.sram_lane_byte(0, 0, 7, 0) == 0);
    assert(mem.executed_transfers().size() == 1);
    assert(!mem.instruction_at(0, 0, 0));

    clock.complete_low();
    clock.validate_all();
    clock.commit_all();
    assert(mem.cycle() == 1 && fabric.cycle() == 1);
    assert(mem.sram_lane_byte(0, 0, 7, 0) == 0x50);
    assert(mem.instruction_at(0, 0, 1));

    phases.high();
    assert((order == std::vector<int>{1, 2, 3, 4}));
    assert(mem.sram_lane_byte(0, 0, 7, 0) == 0x50);
    clock.complete_high();
    clock.negedge_all();

    return {mem.sram_lane_byte(0, 0, 7, 0), mem.cycle(), fabric.cycle()};
}

void verify_registered_read_response()
{
    StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);
    MemArrayModel mem(MemStreamPortMap::BetweenBoundaries(),
        MemArrayModel::MissingStreamPolicy::Error,
        MemArrayModel::ReadTiming::RegisteredR1);
    timing::Clock clock;
    timing::SystemPhase phases;
    clock.register_module(fabric);
    clock.register_module(mem);
    phases.add_node("open MEM fabric", [&] { fabric.low(); });
    phases.add_node("evaluate MEM", [&] { mem.evaluate(fabric); });

    const auto stream = StreamId::East(3);
    mem.set_sram_lane_byte(0, 0, 9, 0, 0xa5);
    mem.enqueue_instruction(0, MemInstruction::Read(9, stream));

    phases.low();
    assert(mem.read_responses().empty());
    assert(mem.next_read_responses().size() == 1);
    assert(mem.next_read_responses()[0].bytes[0] == 0xa5);
    assert(!fabric.staged_segment_valid(1, 0, stream));
    clock.complete_low(); clock.validate_all(); clock.commit_all();
    phases.high(); clock.complete_high(); clock.negedge_all();
    assert(mem.read_responses().size() == 1);
    assert(!fabric.cell(1, 0, 0, stream).valid);

    phases.low();
    assert(mem.staged_read_responses().size() == 1);
    assert(fabric.staged_segment_valid(1, 0, stream));
    assert(!fabric.cell(1, 0, 0, stream).valid);
    clock.complete_low(); clock.validate_all(); clock.commit_all();
    phases.high(); clock.complete_high(); clock.negedge_all();
    assert(fabric.cell(1, 0, 0, stream).valid);
    assert(fabric.cell(1, 0, 0, stream).data == 0xa5);
}

void verify_instruction_pipeline_reference()
{
    StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);
    MemArrayModel mem;
    timing::Clock clock;
    timing::SystemPhase phases;
    clock.register_module(fabric);
    clock.register_module(mem);
    phases.add_node("open MEM fabric", [&] { fabric.low(); });
    phases.add_node("evaluate MEM", [&] { mem.evaluate(fabric); });

    const auto stream = StreamId::East(0);
    mem.enqueue_instruction(0, MemInstruction::Write(13, stream));
    std::array<std::uint8_t, hw::kTileRows> reference{};
    for (std::size_t cycle = 0; cycle < hw::kTileRows; ++cycle) {
        const auto expected = static_cast<std::uint8_t>(0x20 + cycle);
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            fabric.initialize_cell(0, cycle, lane, stream,
                StreamCell::Valid(static_cast<std::uint8_t>(expected + lane)));
        }
        phases.low();
        assert(mem.executed_transfers().size() == 1);
        assert(mem.executed_transfers()[0].tile == cycle);
        assert(mem.sram_lane_byte(0, cycle, 13, 0) == reference[cycle]);
        clock.complete_low(); clock.validate_all(); clock.commit_all();
        reference[cycle] = expected;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            assert(mem.sram_lane_byte(0, tile, 13, 0) == reference[tile]);
        if (cycle + 1 < hw::kTileRows)
            assert(mem.instruction_at(0, 0, cycle + 1));
        phases.high(); clock.complete_high(); clock.negedge_all();
    }
    assert(mem.idle());
}

void verify_validation_barrier()
{
    StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);
    MemArrayModel mem;
    timing::Register<int> missing_low;
    timing::Clock clock;
    timing::SystemPhase phases;
    clock.register_module(fabric);
    clock.register_module(mem);
    clock.register_module(missing_low);
    phases.add_node("open MEM fabric", [&] { fabric.low(); });
    phases.add_node("evaluate MEM", [&] { mem.evaluate(fabric); });

    const auto stream = StreamId::East(0);
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        fabric.initialize_cell(0, 0, lane, stream,
            StreamCell::Valid(static_cast<std::uint8_t>(0x70 + lane)));
    }
    mem.enqueue_instruction(0, MemInstruction::Write(11, stream));
    phases.low();
    clock.complete_low();
    bool rejected = false;
    try { clock.validate_all(); }
    catch (const std::logic_error&) { rejected = true; }
    assert(rejected && clock.faulted());
    assert(mem.cycle() == 0);
    assert(fabric.cycle() == 0);
    assert(mem.sram_lane_byte(0, 0, 11, 0) == 0);
}

} // namespace

int main()
{
    const auto forward = run_write_edge(false);
    const auto reverse = run_write_edge(true);
    assert(forward.value == reverse.value);
    assert(forward.mem_cycle == reverse.mem_cycle);
    assert(forward.fabric_cycle == reverse.fabric_cycle);
    verify_registered_read_response();
    verify_instruction_pipeline_reference();
    verify_validation_barrier();
    return 0;
}
