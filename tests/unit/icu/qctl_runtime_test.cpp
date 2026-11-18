#include "ftlpu/icu/qctl_runtime.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ftlpu;

namespace {

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

template<class Fn>
void rejects(Fn&& fn)
{
    bool failed = false;
    try { fn(); }
    catch (const StaticScheduleError&) { failed = true; }
    require(failed, "invalid QCTL operation was accepted");
}

void pool_cache_partition_and_offsets()
{
    IcuQctlPool pool;
    pool.configure(16, 2, 2, 1);
    pool.configure_partition(0, 0, 2, 10);
    pool.configure_partition(1, 8, 2, 20);
    pool.load(0, {3, IcuControlOpcode::WaitEvent, 11});
    pool.load(1, {9, IcuControlOpcode::WaitEvent, 12});
    pool.load(8, {4, IcuControlOpcode::WaitEvent, 21});
    pool.load(9, {10, IcuControlOpcode::WaitEvent, 22});

    IcuQctlCache cache0;
    IcuQctlCache cache1;
    cache0.configure(pool, 0, 1);
    cache1.configure(pool, 1, 2);
    for (unsigned cycle = 0; cycle < 16
         && !(cache0.primed() && cache1.primed()); ++cycle) {
        pool.begin_cycle();
        cache0.begin_cycle();
        cache1.begin_cycle();
        cache0.end_cycle();
        cache1.end_cycle();
        pool.end_cycle();
    }
    require(cache0.primed() && cache1.primed(),
        "QCTL caches did not prime through the shared single read port");
    require(cache0.head() && cache0.head()->operand == 11
        && cache1.head() && cache1.head()->operand == 21,
        "QCTL Pool partition data crossed Queue ownership");
    require(cache0.fetch_offset() == 1 && cache0.retire_offset() == 0
        && cache1.fetch_offset() == 2 && cache1.retire_offset() == 0,
        "QCTL Cache offsets are not independently owned");

    cache0.retire();
    for (unsigned cycle = 0; cycle < 8 && cache0.level() == 0; ++cycle) {
        pool.begin_cycle();
        cache0.begin_cycle();
        cache1.begin_cycle();
        cache0.end_cycle();
        cache1.end_cycle();
        pool.end_cycle();
    }
    require(cache0.head() && cache0.head()->operand == 12
        && cache0.fetch_offset() == 2 && cache0.retire_offset() == 1,
        "QCTL Cache refill lost fetch/retire ordering");
    require(cache1.head() && cache1.head()->operand == 21
        && cache1.retire_offset() == 0,
        "one QCTL Cache retirement modified another Queue");

    rejects([&] { pool.configure_partition(0, 12, 1, 30); });
    rejects([&] { pool.request(0, 2, 10); });
}

void scheduler_counter_enable_and_wait_states()
{
    IcuQctlController controller;
    const IcuQueueControl sync{3, IcuControlOpcode::Sync, 0};
    std::uint32_t scheduler_cycle = 0;
    for (unsigned physical = 0; physical < 3; ++physical) {
        IcuQctlControllerInputs inputs{scheduler_cycle, &sync};
        const auto outputs = controller.evaluate(inputs);
        require(outputs.scheduler_counter_ce && !outputs.retire_head,
            "future QCTL record stopped Scheduler early");
        controller.commit(inputs, outputs);
        scheduler_cycle += outputs.scheduler_counter_ce;
    }
    require(scheduler_cycle == 3,
        "Scheduler did not reach QCTL boundary cycle");

    IcuQctlControllerInputs arrival{scheduler_cycle, &sync};
    auto outputs = controller.evaluate(arrival);
    require(!outputs.scheduler_counter_ce && outputs.sync_arrival_valid
        && !outputs.retire_head,
        "SYNC did not stop the Scheduler on the hit cycle");
    controller.commit(arrival, outputs);
    require(controller.state() == IcuQctlControllerState::SYNC_WAIT,
        "SYNC did not enter SYNC_WAIT");

    for (unsigned wait = 0; wait < 3; ++wait) {
        outputs = controller.evaluate(arrival);
        require(!outputs.scheduler_counter_ce && outputs.sync_arrival_valid,
            "SYNC_WAIT did not hold CE low and arrival valid high");
        controller.commit(arrival, outputs);
    }
    arrival.sync_release = true;
    outputs = controller.evaluate(arrival);
    require(outputs.retire_head && !outputs.scheduler_counter_ce,
        "SYNC release incorrectly advanced Scheduler on the release edge");
    controller.commit(arrival, outputs);
    require(controller.state() == IcuQctlControllerState::IDLE,
        "SYNC release did not return Controller to IDLE");

    IcuQctlControllerInputs resumed{scheduler_cycle, nullptr};
    outputs = controller.evaluate(resumed);
    require(outputs.scheduler_counter_ce,
        "released QCTL boundary inserted an extra Scheduler offset");
}

struct BarrierCycleResult {
    std::array<IcuQctlControllerOutputs, 3> controllers{};
    IcuQctlArrivalReleaseBarrier::Outputs barrier{};
};

BarrierCycleResult run_barrier_cycle(
    std::array<IcuQctlController, 3>& controllers,
    IcuQctlArrivalReleaseBarrier& barrier,
    const std::array<IcuQueueControl, 3>& records,
    const std::array<std::uint32_t, 3>& cycles)
{
    std::array<IcuQctlControllerInputs, 3> inputs{};
    std::vector<bool> arrivals(3, false);
    std::vector<bool> notifies(3, false);
    BarrierCycleResult result;
    for (std::size_t queue = 0; queue < controllers.size(); ++queue) {
        inputs[queue].scheduler_cycle = cycles[queue];
        inputs[queue].head = &records[queue];
        result.controllers[queue] = controllers[queue].evaluate(inputs[queue]);
        arrivals[queue] = result.controllers[queue].sync_arrival_valid;
        notifies[queue] = result.controllers[queue].notify_valid;
    }
    result.barrier = barrier.evaluate(arrivals, notifies);
    for (std::size_t queue = 0; queue < controllers.size(); ++queue) {
        inputs[queue].sync_release = result.barrier.release[queue];
        inputs[queue].notify_ready = result.barrier.notify_ready[queue];
        result.controllers[queue] = controllers[queue].evaluate(inputs[queue]);
        controllers[queue].commit(inputs[queue], result.controllers[queue]);
    }
    barrier.commit(arrivals, result.barrier);
    return result;
}

void arrival_release_barrier_rounds_and_reset()
{
    IcuQctlArrivalReleaseBarrier barrier;
    barrier.configure({true, true, false});
    std::array<IcuQctlController, 3> controllers{};
    const std::array<IcuQueueControl, 3> records{{
        {0, IcuControlOpcode::Sync, 0},
        {1, IcuControlOpcode::Sync, 0},
        {0, IcuControlOpcode::Notify, 0}}};

    auto result = run_barrier_cycle(controllers, barrier, records, {0, 0, 0});
    require(!result.barrier.completed && barrier.arrived(0)
        && !barrier.arrived(1),
        "early NOTIFY created a release or historical token");
    require(controllers[0].state() == IcuQctlControllerState::SYNC_WAIT
        && controllers[2].state() == IcuQctlControllerState::NOTIFY_WAIT,
        "early arrivals were not held in explicit wait states");

    result = run_barrier_cycle(controllers, barrier, records, {0, 1, 0});
    require(result.barrier.completed && result.barrier.notifier == 2,
        "all participants did not accept the held NOTIFY request");
    require(result.controllers[0].retire_head
        && result.controllers[1].retire_head
        && result.controllers[2].retire_head,
        "barrier completion did not release every participant and notifier");
    require(!barrier.arrived(0) && !barrier.arrived(1),
        "barrier completion retained arrival history");

    // A second round must require fresh arrivals, and also completes normally.
    result = run_barrier_cycle(controllers, barrier, records, {0, 0, 0});
    require(!result.barrier.completed && barrier.arrived(0)
        && !barrier.arrived(1),
        "second barrier round reused first-round arrival state");
    result = run_barrier_cycle(controllers, barrier, records, {0, 1, 0});
    require(result.barrier.completed && result.controllers[0].retire_head
        && result.controllers[1].retire_head
        && result.controllers[2].retire_head,
        "second barrier round did not independently complete");
    barrier.reset();
    for (auto& controller : controllers) controller.reset();
    require(!barrier.arrived(0) && !barrier.arrived(1)
        && controllers[0].state() == IcuQctlControllerState::IDLE,
        "QCTL reset retained barrier or Controller state");
}

void sync_first_notify_later()
{
    IcuQctlArrivalReleaseBarrier barrier;
    barrier.configure({true, true, false});
    std::array<IcuQctlController, 3> controllers{};
    const std::array<IcuQueueControl, 3> records{{
        {0, IcuControlOpcode::Sync, 0},
        {0, IcuControlOpcode::Sync, 0},
        {2, IcuControlOpcode::Notify, 0}}};

    auto result = run_barrier_cycle(controllers, barrier, records, {0, 0, 0});
    require(!result.barrier.completed && barrier.arrived(0)
        && barrier.arrived(1),
        "SYNC arrivals did not persist while waiting for NOTIFY");
    result = run_barrier_cycle(controllers, barrier, records, {0, 0, 1});
    require(!result.barrier.completed,
        "future NOTIFY was accepted before its Scheduler cycle");
    result = run_barrier_cycle(controllers, barrier, records, {0, 0, 2});
    require(result.barrier.completed && result.barrier.notifier == 2,
        "late NOTIFY did not release already-arrived participants");
}

void event_broadcast_contract()
{
    std::array<IcuQctlController, 2> controllers{};
    const IcuQueueControl wait{2, IcuControlOpcode::WaitEvent, 0x12345678u};

    IcuQctlEventManager manager;
    manager.configure(2, 2);
    manager.push(0x12345678u, {true, true});

    // An early broadcast is not buffered inside either Controller.
    for (auto& controller : controllers) {
        IcuQctlControllerInputs early{1, &wait, true, 0x12345678u};
        const auto outputs = controller.evaluate(early);
        require(outputs.scheduler_counter_ce && !outputs.event_fire,
            "Controller buffered an early DMA/Event broadcast");
        controller.commit(early, outputs);
    }
    manager.commit({false, false});
    require(manager.valid() && manager.tag() == 0x12345678u,
        "Event Manager failed to retain an early broadcast");

    for (auto& controller : controllers) {
        IcuQctlControllerInputs mismatch{2, &wait, true, 0x12345679u};
        auto outputs = controller.evaluate(mismatch);
        require(!outputs.scheduler_counter_ce && !outputs.event_fire,
            "mismatched event TAG released QCTL");
        controller.commit(mismatch, outputs);
        require(controller.state() == IcuQctlControllerState::EVENT_WAIT,
            "WAIT_EVENT did not enter EVENT_WAIT");
    }

    std::vector<bool> fires;
    for (auto& controller : controllers) {
        IcuQctlControllerInputs match{2, &wait, manager.valid(), manager.tag()};
        const auto outputs = controller.evaluate(match);
        require(outputs.event_ready && outputs.event_fire
            && outputs.retire_head && !outputs.scheduler_counter_ce,
            "matching 32-bit broadcast TAG did not release every waiter");
        fires.push_back(outputs.event_fire);
        controller.commit(match, outputs);
    }
    manager.commit(fires);
    require(!manager.valid(),
        "Event Manager did not retire after every selected Controller accepted");
}

void mem_packet_retirement_is_not_sram_commit()
{
    IcuQctlController controller;
    const IcuQueueControl packet{7, kQueueControlMemPacket, 3};
    IcuQctlControllerInputs inputs{7, &packet};
    auto outputs = controller.evaluate(inputs);
    require(outputs.mem_packet_valid && outputs.mem_packet_ordinal == 3
        && !outputs.retire_head && !outputs.scheduler_counter_ce,
        "MEM_PACKET request did not hold its ordinal at the boundary");
    controller.commit(inputs, outputs);
    require(controller.state() == IcuQctlControllerState::MEMPACKET_WAIT,
        "MEM_PACKET did not enter MEMPACKET_WAIT");

    const bool final_sram_write_commit = true;
    (void)final_sram_write_commit; // Event manager input, not packet retirement.
    outputs = controller.evaluate(inputs);
    require(outputs.mem_packet_valid && outputs.mem_packet_ordinal == 3
        && !outputs.retire_head,
        "final SRAM commit was confused with MEM_PACKET retirement");

    inputs.mem_packet_retired = true;
    outputs = controller.evaluate(inputs);
    require(outputs.retire_head && outputs.mem_packet_valid
        && outputs.mem_packet_ordinal == 3,
        "MEM_PACKET retirement did not preserve the request through handshake");
    controller.commit(inputs, outputs);
    require(controller.state() == IcuQctlControllerState::IDLE,
        "MEM_PACKET retirement did not return to IDLE");

    const IcuQueueControl second{7, kQueueControlMemPacket, 4};
    inputs = IcuQctlControllerInputs{7, &second};
    outputs = controller.evaluate(inputs);
    require(outputs.mem_packet_valid && outputs.mem_packet_ordinal == 4
        && !outputs.retire_head,
        "next MEM_PACKET ordinal was skipped or reordered");
    controller.commit(inputs, outputs);
    inputs.mem_packet_retired = true;
    outputs = controller.evaluate(inputs);
    require(outputs.mem_packet_ordinal == 4 && outputs.retire_head,
        "second MEM_PACKET did not retire in ordinal order");
    controller.commit(inputs, outputs);
}

void complete_runtime_path()
{
    IcuQctlRuntime runtime;
    runtime.configure(32, 3, 2, 2, 1, 2);
    runtime.configure_partition(0, 0, {
        {0, IcuControlOpcode::Sync, 0},
        {1, IcuControlOpcode::WaitEvent, 77},
        {1, kQueueControlMemPacket, 1}}, 100);
    runtime.configure_partition(1, 8, {
        {0, IcuControlOpcode::Sync, 0}}, 101);
    runtime.configure_partition(2, 16, {
        {0, IcuControlOpcode::Notify, 0}}, 102);
    runtime.configure_barrier({true, true, false});
    runtime.push_event(77, {true, false, false});
    for (unsigned cycle = 0; !runtime.all_primed() && cycle < 32; ++cycle)
        runtime.prime_cycle();
    require(runtime.all_primed(),
        "complete QCTL Runtime did not prime every Cache");

    std::vector<std::uint32_t> scheduler_cycles(3, 0);
    std::vector<bool> retired(3, false);
    bool observed_event = false;
    bool observed_packet_hold = false;
    bool packet_retired = false;
    for (unsigned physical = 0; physical < 24; ++physical) {
        const auto outputs = runtime.tick(scheduler_cycles, retired);
        retired.assign(3, false);
        for (std::size_t queue = 0; queue < scheduler_cycles.size(); ++queue)
            scheduler_cycles[queue] += outputs[queue].scheduler_counter_ce;
        observed_event = observed_event || outputs[0].event_fire;
        if (outputs[0].mem_packet_valid) {
            require(outputs[0].mem_packet_ordinal == 1,
                "complete QCTL Runtime changed MEM_PACKET ordinal");
            if (!packet_retired) {
                observed_packet_hold = true;
                retired[0] = true;
                packet_retired = true;
            }
        }
        if (outputs[0].done && outputs[1].done && outputs[2].done) break;
    }
    require(observed_event && runtime.event_manager().level() == 0,
        "complete QCTL Runtime did not consume Event Manager broadcast");
    require(observed_packet_hold && packet_retired,
        "complete QCTL Runtime did not hold/retire MEM_PACKET");
    require(scheduler_cycles[0] >= 1 && scheduler_cycles[1] >= 1
        && scheduler_cycles[2] >= 1,
        "complete QCTL Runtime failed to resume external Scheduler counters");
    runtime.reset();
    require(runtime.event_manager().level() == 0,
        "complete QCTL Runtime reset retained an Event-side buffer");
}

} // namespace

int main()
try {
    pool_cache_partition_and_offsets();
    scheduler_counter_enable_and_wait_states();
    arrival_release_barrier_rounds_and_reset();
    sync_first_notify_later();
    event_broadcast_contract();
    mem_packet_retirement_is_not_sram_commit();
    complete_runtime_path();
    std::cout << "QCTL runtime passed: Pool/Cache/Controller separation, "
                 "arrival-release barriers, 32-bit events, CE and MEM_PACKET\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
