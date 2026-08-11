#include "ftlpu/system/icu.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <cstddef>

namespace {

using ftlpu::DistributedVxmSlice;
using ftlpu::IcuControlInstruction;
using ftlpu::IcuLocation;
using ftlpu::IcuQueueAction;
using ftlpu::InstructionControlUnit;

void tick_distributed(
    InstructionControlUnit& icu,
    DistributedVxmSlice& vxm)
{
    icu.dispatch_vxm(vxm);
    vxm.tick();
    icu.advance_barrier_events();
}

void test_delayed_broadcast()
{
    auto icu = InstructionControlUnit{2};
    auto vxm = DistributedVxmSlice{};
    icu.enqueue_control(
        IcuLocation::DistributedVxm(0),
        IcuControlInstruction::Notify());
    icu.enqueue_control(
        IcuLocation::DistributedVxm(1),
        IcuControlInstruction::Sync());

    tick_distributed(icu, vxm);
    assert(icu.pending_barrier_event_count() == 1);
    assert(icu.distributed_vxm_iq(1).blocked_on_sync());

    tick_distributed(icu, vxm);
    assert(icu.distributed_vxm_iq(1).blocked_on_sync());
    tick_distributed(icu, vxm);
    assert(!icu.distributed_vxm_iq(1).blocked_on_sync());
    assert(!icu.distributed_vxm_iq(1).done());

    tick_distributed(icu, vxm);
    assert(icu.distributed_vxm_iq(1).done());
    assert(icu.distributed_vxm_iq(1).last_trace().action
           == IcuQueueAction::SyncRelease);
}

void test_zero_latency_and_direct_notify()
{
    auto icu = InstructionControlUnit{0};
    auto vxm = DistributedVxmSlice{};
    icu.enqueue_control(
        IcuLocation::DistributedVxm(0),
        IcuControlInstruction::Notify());
    icu.enqueue_control(
        IcuLocation::DistributedVxm(1),
        IcuControlInstruction::Sync());
    tick_distributed(icu, vxm);
    assert(icu.pending_barrier_event_count() == 0);
    tick_distributed(icu, vxm);
    assert(icu.distributed_vxm_iq(1).done());

    auto direct = InstructionControlUnit{};
    auto direct_vxm = DistributedVxmSlice{};
    direct.enqueue_control(
        IcuLocation::DistributedVxm(2),
        IcuControlInstruction::Sync());
    direct.notify(IcuLocation::DistributedVxm(2));
    tick_distributed(direct, direct_vxm);
    assert(direct.distributed_vxm_iq(2).done());
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
