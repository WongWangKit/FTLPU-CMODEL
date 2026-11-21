#include "ftlpu/core/timing.hpp"
#include "ftlpu/mxm/accumulator.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/mxm/supercell.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using namespace ftlpu;

MxmSupercell::InputVector full_weight_input(std::uint16_t base)
{
    auto input = MxmSupercell::InputVector {};
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        for (std::size_t column = 0;
             column < hw::kMxmSupercellColumns;
             ++column) {
            input[lane][column] = MxmSupercell::InputWord {
                static_cast<std::uint16_t>(base + lane * 16 + column),
                column + 1 == hw::kMxmSupercellColumns};
        }
    }
    return input;
}

void finish_cycle(timing::Clock& clock, timing::SystemPhase& phases)
{
    clock.complete_low();
    clock.validate_all();
    clock.commit_all();
    phases.high();
    clock.complete_high();
    clock.negedge_all();
}

void verify_registered_control_rows()
{
    MxmArray array;
    MxmControlSlice control(array);
    timing::Clock clock;
    timing::SystemPhase phases;
    clock.register_module(control);
    phases.add_node("MXM legacy control LOW", [&] { control.low(); });

    control.issue_south(MxmControlInstruction::Compute());
    assert(!control.compute_instruction_at(0));

    phases.low();
    assert(!control.compute_instruction_at(0)); // No architectural Next exposure.
    assert(control.compute_active(0));          // Row-0 combinational consumer.
    finish_cycle(clock, phases);
    assert(control.compute_instruction_at(1));
    assert(control.cycle()==1);
    for(std::size_t row=1;row<hw::kMxmSupercellsPerPlane;++row) {
        phases.low();
        assert(control.compute_active(row));
        assert(control.compute_instruction_at(row));
        if(row+1<hw::kMxmSupercellsPerPlane)
            assert(!control.compute_instruction_at(row+1));
        finish_cycle(clock,phases);
        assert(!control.compute_instruction_at(row));
    }
    assert(control.idle());
    assert(control.cycle()==hw::kMxmSupercellsPerPlane);
}

void verify_supercell_edge_visibility()
{
    MxmSupercell supercell;
    timing::Clock clock;
    timing::SystemPhase phases;
    clock.register_module(supercell);
    phases.add_node("MXM supercell LOW", [&] { supercell.low(); });

    constexpr std::uint16_t kBase = 0x1200;
    supercell.set_input(full_weight_input(kBase));
    supercell.issue(MxmInstruction::IW(0));
    phases.low();
    assert(!supercell.weight_buffer_valid(0));
    assert(supercell.weight_bits(0, 0, 0) == 0);
    finish_cycle(clock, phases);
    assert(supercell.weight_buffer_valid(0));
    assert(supercell.weight_bits(0, 0, 0) == kBase);
}

struct OrderResult {
    std::uint16_t weight{};
    float accumulated{};
    std::vector<int> low_order{};
    bool operator==(const OrderResult&) const = default;
};

OrderResult run_registration_order(bool reverse_registration)
{
    MxmSupercell supercell;
    MxmAccumulator accumulator;
    timing::Clock clock;
    timing::SystemPhase phases;
    std::vector<int> low_order;

    if (reverse_registration) {
        clock.register_module(accumulator);
        clock.register_module(supercell);
    } else {
        clock.register_module(supercell);
        clock.register_module(accumulator);
    }

    MxmAccumulator::Segment segment {};
    segment.fill(2.5f);
    phases.add_node("MXM supercell LOW", [&] {
        low_order.push_back(1);
        supercell.low();
    });
    phases.add_node("MXM accumulator LOW", [&] {
        low_order.push_back(2);
        accumulator.low();
        accumulator.accumulate(3, 1, segment);
        assert(accumulator.read(3, 1)[0] == 2.5f);
    });

    constexpr std::uint16_t kBase = 0x2300;
    supercell.set_input(full_weight_input(kBase));
    supercell.issue(MxmInstruction::IW(0));
    phases.low();
    assert(supercell.weight_bits(0, 0, 0) == 0);
    finish_cycle(clock, phases);
    return {supercell.weight_bits(0, 0, 0), accumulator.value(3, 8), low_order};
}

void verify_validation_barrier()
{
    MxmSupercell supercell;
    timing::Register<int> missing_low;
    timing::Clock clock;
    timing::SystemPhase phases;
    clock.register_module(supercell);
    clock.register_module(missing_low);
    phases.add_node("MXM supercell LOW", [&] { supercell.low(); });

    supercell.set_input(full_weight_input(0x3400));
    supercell.issue(MxmInstruction::IW(0));
    phases.low();
    clock.complete_low();
    bool rejected = false;
    try {
        clock.validate_all();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);
    assert(clock.faulted());
    // Validation happens before every commit, so the earlier valid module did
    // not become visible when a later module lacked a prepared Next value.
    assert(!supercell.weight_buffer_valid(0));
    assert(supercell.weight_bits(0, 0, 0) == 0);
}

} // namespace

int main()
{
    verify_registered_control_rows();
    verify_supercell_edge_visibility();
    const auto forward = run_registration_order(false);
    const auto reverse = run_registration_order(true);
    assert(forward == reverse);
    assert(forward.weight == 0x2300);
    assert(forward.accumulated == 2.5f);
    assert((forward.low_order == std::vector<int> {1, 2}));
    verify_validation_barrier();
    return 0;
}
