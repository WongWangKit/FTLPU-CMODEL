#include "ftlpu/core/timing.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

#undef assert
#define assert(condition)                                                     \
    do {                                                                      \
        if (!(condition))                                                     \
            throw std::logic_error("VXM clocked check failed: " #condition); \
    } while (false)

namespace {

using namespace ftlpu;

VxmCompactInstruction immediate_bypass()
{
    return VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Eight,
        VxmLaneAluInstruction {
            VxmAluOpcode::Bypass,
            VxmLaneOperand::Imm(3.0f),
            VxmLaneOperand::StreamFloat16()});
}

bool run_one_edge(bool reverse_registration)
{
    VxmSlice vxm;
    timing::Register<int> witness(7, 7);
    timing::Clock clock;
    timing::SystemPhase phases;

    if (reverse_registration) {
        clock.register_module(witness);
        clock.register_module(vxm);
    } else {
        clock.register_module(vxm);
        clock.register_module(witness);
    }
    phases.add_node("VXM prepare and evaluate", [&] {
        vxm.prepare_cycle();
        vxm.evaluate();
    });
    phases.add_node("independent witness", [&] { witness.low(); });

    vxm.issue_south(0, immediate_bypass());
    phases.low();

    // LOW has a complete transactional Next image, but the architectural
    // Current image remains unchanged until the common edge.
    assert(vxm.cycle() == 0);
    assert(!vxm.instruction_at(0, 1));
    assert(witness.q() == 7);

    clock.complete_low();
    clock.validate_all();
    clock.commit_all();
    assert(vxm.cycle() == 1);
    assert(vxm.instruction_at(0, 1));
    assert(witness.q() == 7);

    phases.high();
    assert(vxm.cycle() == 1);
    assert(vxm.instruction_at(0, 1));
    clock.complete_high();
    clock.negedge_all();
    return vxm.instruction_at(0, 1).has_value();
}

void verify_validation_barrier()
{
    VxmSlice vxm;
    timing::Register<int> missing_low(0);
    timing::Clock clock;
    timing::SystemPhase phases;
    clock.register_module(vxm);
    clock.register_module(missing_low);
    phases.add_node("VXM only", [&] {
        vxm.prepare_cycle();
        vxm.evaluate();
    });

    vxm.issue_south(0, immediate_bypass());
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
    assert(vxm.cycle() == 0);
    assert(!vxm.instruction_at(0, 1));
}

void verify_failed_low_does_not_commit()
{
    VxmSlice vxm;
    auto stream_consumer = VxmLaneAluInstruction {
        VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamFloat16()};
    vxm.issue_south(0, VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Eight, stream_consumer));

    // First edge moves the compact packet into local decode.
    vxm.tick();
    assert(vxm.cycle() == 1);

    vxm.prepare_cycle();
    bool rejected = false;
    try {
        vxm.evaluate();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);
    assert(vxm.cycle() == 1);
    assert(vxm.input_buffer(0).empty());
}

} // namespace

int main() try
{
    assert(run_one_edge(false));
    assert(run_one_edge(true));
    verify_validation_barrier();
    verify_failed_low_does_not_commit();
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
