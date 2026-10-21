#include "ftlpu/icu/common_barrier.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>

namespace {
using namespace ftlpu;
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
template <typename Function> void rejects(Function function)
{
    bool rejected = false;
    try { function(); } catch (const StaticScheduleError&) { rejected = true; }
    require(rejected, "invalid common barrier program was accepted");
}

const std::vector<IcuRawImemWord96> image{
    {{{0x00000027u, 0x00000000u, 0x00000000u}}},
    {{{0x02000021u, 0x00000180u, 0x00303200u}}},
    {{{0x00000400u, 0x00000000u, 0x00000000u}}},
};
std::vector<IcuRawImemWord96> make_long_image()
{
    std::vector<IcuRawImemWord96> result(2);
    result[0].lanes[0] = 0x17; // enabled packed-v1, one descriptor
    unsigned offset = 0;
    const auto put = [&](std::uint32_t value, unsigned width) {
        for (unsigned bit = 0; bit < width; ++bit, ++offset)
            if ((value >> bit) & 1u)
                result[1].lanes[offset / 32] |= 1u << (offset % 32);
    };
    put(0, 3); // no delta dictionary
    put(20, 32); put(100, 13); // first absolute cycle / SRAM row
    put(0, 1); // one-record run
    put(0, 1); put(1, 2); put(0, 1); put(2, 6); put(0, 1);
    put(6, 11); put(2, 8); put(1, 10); // seven reads, interval=3, row stride=1
    return result;
}

void controller_events()
{
    IcuCommonBarrierController control;
    control.configure(12, {{2, 100}, {3, 101}, {4, 102}},
        {{100, {7, 8}}, {101, {9}}, {102, {10}}});
    control.notify_mem_commit(12, 7); // Early, one member is insufficient.
    control.notify_mem_commit(12, 7);
    control.notify_mem_commit(11, 8); // Old generation cannot satisfy it.
    for (unsigned c = 0; c < 2; ++c) {
        require(control.begin_cycle(true), "early event incorrectly stopped computation");
        control.end_cycle();
    }
    for (unsigned p = 0; p < 4; ++p) {
        require(!control.begin_cycle(true), "incomplete aggregate released barrier");
        control.end_cycle();
    }
    require(control.logical_cycle() == 2 && control.physical_cycle() == 6,
        "waiting consumed logical computation budget");
    control.notify_mem_commit(12, 8);
    control.notify_mem_commit(12, 9); // Next barrier already ready.
    require(control.begin_cycle(true), "late aggregate did not resume at C");
    control.end_cycle();
    require(control.begin_cycle(true), "early-ready consecutive barrier added a wait cycle");
    control.end_cycle();
    require(!control.begin_cycle(true), "old aggregate was reused by repeated page load");
    control.end_cycle();
    control.notify_mem_commit(12, 10);
    require(control.begin_cycle(true), "repeated page instance did not release");
    control.end_cycle();
    require(control.done() && control.wait_cycles() == 5,
        "common barrier completion accounting is incorrect");
    rejects([&] { control.notify_mem_commit(12, 999); });
    rejects([&] { control.notify_mem_commit(13, 7); });

    IcuCommonBarrierController empty;
    empty.configure(1, {}, {});
    require(empty.done() && empty.begin_cycle(true), "empty preloaded table did not run");
    empty.end_cycle();
    rejects([] { IcuCommonBarrierController c; c.configure(1, {{2, 1}, {2, 2}}, {{1, {1}}, {2, {2}}}); });
    rejects([] { IcuCommonBarrierController c; c.configure(1, {{2, 1}}, {}); });
    rejects([] { IcuCommonBarrierController c; c.configure(1, {}, {{1, {1}}}); });
    rejects([] { IcuCommonBarrierController c; c.configure(1, {{2, 1}}, {{1, {1, 1}}}); });
    rejects([] { IcuCommonBarrierController c; c.configure(0, {}, {}); });
    rejects([] {
        IcuCommonBarrierController c;
        std::vector<IcuCommonBarrierEntry> table(hw::kCommonBarrierCapacity + 1);
        c.configure(1, table, {});
    });
    rejects([] {
        IcuCommonBarrierController c;
        std::vector<std::uint32_t> instances(
            IcuCommonBarrierController::kMaxCompletionInstances + 1);
        c.configure(1, {{1, 1}}, {{1, std::move(instances)}});
    });
}

using Decoded = std::tuple<std::size_t, std::size_t, std::size_t>;
std::vector<Decoded> decode_with_pause(unsigned pause_at)
{
    IcuMacroV1Decoder<MemInstruction, 96, 3> decoder(IcuMacroQueueKind::Mem, image);
    std::vector<Decoded> result;
    for (unsigned step = 0; !decoder.done() && step < 80; ++step) {
        if (step == pause_at) {
            const auto before = decoder.statistics().decoded_contexts;
            for (unsigned p = 0; p < 7; ++p) decoder.hold_cycle();
            require(decoder.statistics().decoded_contexts == before,
                "HOLD advanced parser, delta expansion or admission");
        }
        if (const auto context = decoder.tick(true))
            result.emplace_back(context->schedule.start_cycle,
                context->instruction.address, context->instruction.stream);
    }
    require(decoder.done(), "paused finite decoder did not complete");
    require(decoder.statistics().fetched_words == image.size(),
        "paused in-flight i-MEM response was lost or consumed twice");
    return result;
}

void physical_response_and_context_hold()
{
    const auto reference = decode_with_pause(999);
    for (unsigned stage = 0; stage < 32; ++stage)
        require(decode_with_pause(stage) == reference,
            "decoder HOLD changed the reconstructed instruction sequence");

    using Queue = DistributedIcuQueue<MemInstruction, 96, 32, 4, 2, 4>;
    Queue raw;
    raw.load_raw_macro(IcuMacroQueueKind::Mem, make_long_image());
    while (!raw.raw_macro_ready()) raw.prefetch_raw_macro();
    Queue native;
    native.push_nop(50);
    std::vector<std::pair<std::size_t, std::size_t>> issued;
    for (unsigned logical = 0; logical < 40; ++logical) {
        if (logical == 21 || logical == 25) {
            const auto active = raw.active_macro_count();
            require(active != 0, "test did not retain a live Macro across HOLD");
            for (unsigned wait = 0; wait < 9; ++wait) {
                raw.hold_compute_cycle();
                native.hold_compute_cycle();
            }
            require(raw.cycle() == logical && native.cycle() == logical
                && raw.active_macro_count() == active,
                "HOLD advanced a Macro context or Native loop/NOP state");
        }
        if (const auto instruction = raw.tick()) issued.emplace_back(logical, instruction->address);
        native.tick();
    }
    require(issued.size() == 7 && issued[0].first == 20
        && issued[1].first == 23 && issued.back().first == 38,
        "single Macro crossing barriers changed logical issue sequence");

    Queue overlapping;
    overlapping.load_raw_macro(IcuMacroQueueKind::Mem, image);
    while (!overlapping.raw_macro_ready()) overlapping.prefetch_raw_macro();
    for (unsigned logical = 0; logical < 3; ++logical) overlapping.tick();
    require(overlapping.active_macro_count() == 2,
        "test did not reach multiple simultaneously active contexts");
    for (unsigned p = 0; p < 11; ++p) overlapping.hold_compute_cycle();
    require(overlapping.active_macro_count() == 2 && overlapping.cycle() == 3,
        "HOLD changed one of several active contexts");
    require(overlapping.tick()->address == 100,
        "first predecoded context failed to resume at exact cycle C");
    for (unsigned logical = 4; logical < 7; ++logical) overlapping.tick();
    require(overlapping.tick()->address == 101 && overlapping.done(),
        "second predecoded context was lost or issued twice");

    Queue nativeFetch;
    for (unsigned n = 0; n < 20; ++n) nativeFetch.push_nop(1);
    nativeFetch.configure_all();
    nativeFetch.prefetch_only();
    require(nativeFetch.pending_fetch_count() != 0, "test had no outstanding Native i-MEM read");
    const auto pc = nativeFetch.fetch_pc();
    const auto cycle = nativeFetch.cycle();
    for (unsigned p = 0; p < 5; ++p) nativeFetch.hold_compute_cycle();
    require(nativeFetch.fetch_pc() == pc && nativeFetch.cycle() == cycle
        && nativeFetch.pending_fetch_count() == 0,
        "Native HOLD failed to capture a reserved response without advancing PC");
}

struct DispatchFixture {
    InstructionControlUnit icu{};
    std::array<MemArrayModel, hw::kHemispheres> mem{MemArrayModel{}, MemArrayModel{}};
    VxmSrSlice vxm{};
    std::array<SxmSlice, hw::kHemispheres> sxm{
        SxmSlice(SxmStreamPortMap::SameDirection(0, 0)),
        SxmSlice(SxmStreamPortMap::SameDirection(0, 0))};
    std::array<Mxm, hw::kMxmCount> mxm{};
    void tick() { icu.dispatch(mem, vxm, sxm, mxm); }
};

void independent_transport_and_shared_clock()
{
    auto f = std::make_unique<DispatchFixture>();
    f->icu.mem_iq(0).load_raw_macro(IcuMacroQueueKind::Mem, image);
    f->icu.prime_raw_macro_frontends();
    f->icu.enqueue_mem_nop(1, 40);
    f->icu.enqueue_vxm_nop(0, 40);
    f->icu.configure_common_barriers(5, {{0, 1}}, {{1, {0}}});
    f->icu.push_mem_transport_synchronized_raw(0,
        InstructionControlUnit::encode_mem_synchronized_packet(1, 77, 0, 1,
            MemInstruction::Write(30, 0)));
    f->tick();
    require(!f->icu.compute_enabled() && f->icu.mem_iq(0).cycle() == 0,
        "raw Macro escaped cycle-zero common barrier");
    const auto location = IcuLocation::Mem(Hemisphere::East, 0, 0);
    require(f->icu.mem_accepts_synchronized_notification(location, 77),
        "separate MEM transport did not accept RX while raw Macro was paused");
    f->icu.notify_mem_synchronized(location, 77);
    f->tick();
    require(f->icu.mem_transport_iq(0).synchronized_issued_count() == 0,
        "transport notification bypassed the physical D+1 arrival boundary");
    f->tick();
    require(f->icu.mem_transport_iq(0).synchronized_issued_count() == 1,
        "transport MEM write was blocked by raw Macro HOLD");
    require(f->icu.mem_iq(0).cycle() == 0 && f->icu.mem_iq(1).cycle() == 0
        && f->icu.vxm_iq(0).cycle() == 0,
        "Native and raw Macro did not hold the same logical boundary");
    f->icu.notify_common_mem_commit(5, 0);
    f->tick();
    require(f->icu.logical_cycle() == 1 && f->icu.physical_cycle() == 4,
        "common resume edge or logical/physical time is incorrect");
    require(f->icu.mem_iq(0).cycle() == 1 && f->icu.mem_iq(1).cycle() == 1
        && f->icu.vxm_iq(0).cycle() == 1,
        "Native/Macro phase changed across resume");

    auto conflict = std::make_unique<DispatchFixture>();
    conflict->icu.mem_iq(0).load_raw_macro(IcuMacroQueueKind::Mem, image);
    conflict->icu.prime_raw_macro_frontends();
    conflict->icu.configure_common_barriers(8, {}, {});
    conflict->icu.push_mem_transport_synchronized_raw(0,
        InstructionControlUnit::encode_mem_synchronized_packet(1, 88, 0, 1,
            MemInstruction::Write(30, 0)));
    for (unsigned c = 0; c < 2; ++c) conflict->tick();
    conflict->icu.notify_mem_synchronized(location, 88);
    conflict->tick(); // physical notification alignment cycle
    rejects([&] { conflict->tick(); });
}

void unsafe_boundary_is_rejected()
{
    auto system = std::make_unique<TspSliceSystem>();
    system->icu().configure_common_barriers(1, {{0, 1}}, {{1, {1}}});
    system->icu().notify_common_mem_commit(1, 1); // Even ready events require safety.
    system->stream_fabric(Hemisphere::East).initialize_cell(0, 0, 0,
        StreamId::East(0), StreamCell::Valid(1));
    rejects([&] { system->tick(TspSliceSystem::LogSinks{}); });

    auto internal = std::make_unique<TspSliceSystem>();
    internal->icu().configure_common_barriers(1, {{0, 1}}, {{1, {1}}});
    internal->icu().notify_common_mem_commit(1, 1);
    internal->vxm_unit().datapath().fabric().registers().initialize_cell(
        3, 0, 0, StreamId::East(0), StreamCell::Valid(1));
    require(!internal->common_barrier_resources_drained(),
        "resource safety missed data in VXM internal ordinary SR");
    rejects([&] { internal->tick(TspSliceSystem::LogSinks{}); });

    auto resume = std::make_unique<TspSliceSystem>();
    resume->icu().configure_common_barriers(1, {{0, 1}}, {{1, {1}}});
    resume->tick(TspSliceSystem::LogSinks{});
    resume->stream_fabric(Hemisphere::East).initialize_cell(0, 0, 0,
        StreamId::East(0), StreamCell::Valid(1));
    resume->icu().notify_common_mem_commit(1, 1);
    rejects([&] { resume->tick(TspSliceSystem::LogSinks{}); });
}

void transport_commits_visible_sram_before_resume()
{
    auto system = std::make_unique<TspSliceSystem>();
    auto& icu = system->icu();
    icu.mem_iq(0).load_raw_macro(IcuMacroQueueKind::Mem, image);
    icu.prime_raw_macro_frontends();
    icu.configure_common_barriers(40, {{0, 1}}, {{1, {400}}});
    icu.push_mem_transport_synchronized_raw(0,
        InstructionControlUnit::encode_mem_synchronized_packet(1, 90, 0, 1,
            MemInstruction::Write(30, StreamId::East(0))));
    system->tick(TspSliceSystem::LogSinks{}); // Enter HOLD, arm transport.
    icu.notify_mem_synchronized(IcuLocation::Mem(0), 90);
    system->tick(TspSliceSystem::LogSinks{}); // D+1 alignment, no SRAM write.
    const auto input_column = system->mem_array(Hemisphere::East).ports()
        .input_column(0, StreamDirection::East);
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            system->stream_fabric(Hemisphere::East).initialize_cell(
                input_column, tile, lane, StreamId::East(0),
                StreamCell::Valid(static_cast<std::uint8_t>(42 + tile),
                    lane + 1 == hw::kLanesPerTile, 400));
        system->tick(TspSliceSystem::LogSinks{});
        require(icu.logical_cycle() == 0,
            "page-ready was incorrectly inferred before complete SRAM visibility");
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            require(system->mem_array(Hemisphere::East).sram_lane_byte(
                        0, 0, tile, 30, lane) == 42 + tile,
                "HOLD blocked actual transport SR-to-SRAM commit");
    }
    require(system->common_barrier_resources_drained(),
        "last tile MEM commit did not drain the point-to-point transport path");
    icu.notify_common_mem_commit(40, 400);
    system->tick(TspSliceSystem::LogSinks{});
    require(icu.logical_cycle() == 1 && icu.common_barriers_done(),
        "visible page commit did not resume raw Macro at C");
}
} // namespace

int main()
try {
    controller_events();
    physical_response_and_context_hold();
    independent_transport_and_shared_clock();
    unsafe_boundary_is_rejected();
    transport_commits_visible_sram_before_resume();
    std::cout << "common barriers: event generations, finite frontends, mixed hold/resume, transport and unsafe boundary passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
