#include "ftlpu/system/icu.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <cstddef>

namespace {

using ftlpu::IcuControlInstruction;
using ftlpu::IcuLocation;
using ftlpu::IcuQueueAction;
using ftlpu::InstructionControlUnit;
using ftlpu::VxmSlice;

void tick_vxm(
    InstructionControlUnit& icu,
    VxmSlice& vxm)
{
    icu.dispatch_vxm(vxm);
    vxm.tick();
    icu.advance_barrier_events();
}

void test_delayed_broadcast()
{
    auto icu = InstructionControlUnit{2};
    auto vxm = VxmSlice{};
    icu.enqueue_control(
        IcuLocation::Vxm(0),
        IcuControlInstruction::Notify());
    icu.enqueue_control(
        IcuLocation::Vxm(1),
        IcuControlInstruction::Sync());

    tick_vxm(icu, vxm);
    assert(icu.pending_barrier_event_count() == 1);
    assert(icu.vxm_iq(1).blocked_on_sync());

    tick_vxm(icu, vxm);
    assert(icu.vxm_iq(1).blocked_on_sync());
    tick_vxm(icu, vxm);
    assert(!icu.vxm_iq(1).blocked_on_sync());
    assert(!icu.vxm_iq(1).done());

    tick_vxm(icu, vxm);
    assert(icu.vxm_iq(1).done());
    assert(icu.vxm_iq(1).last_trace().action
           == IcuQueueAction::SyncRelease);
}

void test_zero_latency_and_direct_notify()
{
    auto icu = InstructionControlUnit{0};
    auto vxm = VxmSlice{};
    icu.enqueue_control(
        IcuLocation::Vxm(0),
        IcuControlInstruction::Notify());
    icu.enqueue_control(
        IcuLocation::Vxm(1),
        IcuControlInstruction::Sync());
    tick_vxm(icu, vxm);
    assert(icu.pending_barrier_event_count() == 0);
    tick_vxm(icu, vxm);
    assert(icu.vxm_iq(1).done());

    auto direct = InstructionControlUnit{};
    auto direct_vxm = VxmSlice{};
    direct.enqueue_control(
        IcuLocation::Vxm(2),
        IcuControlInstruction::Sync());
    // Launch/configure the queue first. A notification is a runtime token and
    // therefore must not be injected before the direct CModel queue has
    // entered execution state.
    tick_vxm(direct, direct_vxm);
    assert(direct.vxm_iq(2).blocked_on_sync());
    direct.notify(IcuLocation::Vxm(2));
    tick_vxm(direct, direct_vxm);
    assert(direct.vxm_iq(2).done());
}

void test_tsp_automatic_advance()
{
    auto system = ftlpu::TspSliceSystem{};
    system.icu().enqueue_control(
        IcuLocation::Vxm(0), IcuControlInstruction::Notify());
    system.icu().enqueue_control(
        IcuLocation::Vxm(1), IcuControlInstruction::Sync());

    system.tick(ftlpu::TspSliceSystem::LogSinks{});
    assert(system.icu().pending_barrier_event_count() == 1);
    for (std::size_t cycle = 0;
         cycle < ftlpu::hw::kIcuBarrierLatencyCycles;
         ++cycle) {
        system.tick(ftlpu::TspSliceSystem::LogSinks{});
    }
    assert(!system.icu().vxm_iq(1).blocked_on_sync());
    system.tick(ftlpu::TspSliceSystem::LogSinks{});
    assert(system.icu().vxm_iq(1).done());
}

} // namespace

int main()
{
    test_delayed_broadcast();
    test_zero_latency_and_direct_notify();
    test_tsp_automatic_advance();
}
