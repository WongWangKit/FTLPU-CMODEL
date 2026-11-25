#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/mxm/instruction.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using namespace ftlpu;

constexpr std::size_t kQueueCount = InstructionControlUnit::kMemQueues
    + 5 * InstructionControlUnit::kMxmQueues;
constexpr std::array<std::size_t, 7> kQueues{
    0, 1, 208, 212, 216, 220, 224};
constexpr std::array<std::string_view, 7> kFuNames{
    "mem", "mem_notifier", "mxm_load", "mxm_dequant",
    "mxm_compute", "mxm_param_load", "mxm_backend"};

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

Ddr4Config ddr_config()
{
    Ddr4Config config;
    config.read_latency_cycles = 2;
    config.read_latency_jitter_cycles = 0;
    config.peak_bandwidth_bytes_per_second = config.lpu_clock_hz * 32;
    return config;
}

void set_bits(std::vector<std::uint8_t>& bytes, std::size_t offset,
    std::uint64_t value, std::size_t width)
{
    for (std::size_t bit = 0; bit < width; ++bit)
        if ((value >> bit) & 1u)
            bytes[(offset + bit) / 8] |= std::uint8_t{1}
                << ((offset + bit) % 8);
        else
            bytes[(offset + bit) / 8] &= ~(std::uint8_t{1}
                << ((offset + bit) % 8));
}

std::vector<std::uint8_t> mem_context(std::uint32_t issue_cycle,
    std::uint32_t address, std::uint32_t stream)
{
    const std::array<std::uint32_t, 9> lanes{
        0x25, 0, 0, 0x02000021, 0x180, 0x00303200, 0x400, 0, 0};
    std::vector<std::uint8_t> bytes(36);
    for (std::size_t byte = 0; byte < bytes.size(); ++byte)
        bytes[byte] = static_cast<std::uint8_t>(
            lanes[byte / 4] >> (8 * (byte % 4)));
    set_bits(bytes, 135, issue_cycle, 32);
    set_bits(bytes, 167, address, 13);
    set_bits(bytes, 201, stream, 6);
    return bytes;
}

std::vector<std::uint8_t> mxm_context(std::uint32_t issue_cycle,
    std::uint64_t instruction, std::size_t instruction_bits,
    std::size_t operand_bits)
{
    std::vector<std::uint8_t> bytes(32);
    bytes[0] = 0x15;
    std::size_t cursor = 128;
    auto push = [&](std::uint64_t value, std::size_t width) {
        set_bits(bytes, cursor, value, width);
        cursor += width;
    };
    push(0, 3);
    push(issue_cycle, 32);
    push(0, operand_bits);
    push(0, 1);
    push(0, 1);
    push(instruction, instruction_bits);
    push(0, 2);
    push(0, 2);
    return bytes;
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    for (std::size_t byte = 0; byte < 4; ++byte)
        bytes.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
}

std::vector<std::uint8_t> qctl_records(
    std::initializer_list<IcuQueueControl> records)
{
    std::vector<std::uint8_t> bytes;
    for (const auto& record : records) {
        append_u32(bytes, record.cycle);
        append_u32(bytes, static_cast<std::uint32_t>(record.opcode));
        append_u32(bytes, record.operand);
    }
    return bytes;
}

void stage(C2cDmaSystem& system, std::uint64_t address,
    const std::vector<std::uint8_t>& bytes)
{
    for (std::size_t offset = 0; offset < bytes.size(); offset += 32) {
        C2cVector vector{};
        for (std::size_t byte = 0; byte < 32; ++byte)
            vector.payload[byte / hw::kLanesPerTile]
                [byte % hw::kLanesPerTile] = offset + byte < bytes.size()
                    ? bytes[offset + byte] : 0;
        system.ddr4().initialize_vector(address + offset, vector);
    }
}

struct IssueSample {
    bool valid{false};
    bool ready{true};
    bool fire{false};
    std::uint64_t instruction{0};
    std::uint32_t cycle{0};
};

template <typename Queue>
IssueSample mxm_issue(const Queue& queue)
{
    const auto& pins = queue.macro_issue_port_signals();
    return {pins.issue_valid_o, pins.issue_ready_i, pins.issue_fire,
        pins.issue_instruction_o
            ? isa::encode_mxm_command(*pins.issue_instruction_o) : 0,
        pins.issue_cycle_o};
}

IssueSample issue_sample(const InstructionControlUnit& icu,
    std::size_t selected)
{
    if (selected < 2) {
        const auto& pins = icu.mem_iq(selected).mem_issue_port_signals();
        return {pins.mem_issue_valid_o, pins.mem_issue_ready_i,
            pins.mem_issue_fire,
            pins.mem_issue_instruction_o
                ? isa::encode_mem_instruction(*pins.mem_issue_instruction_o) : 0,
            pins.mem_issue_cycle_o};
    }
    switch (selected) {
    case 2: return mxm_issue(icu.mxm_load_iq(0));
    case 3: return mxm_issue(icu.mxm_dequant_iq(0));
    case 4: return mxm_issue(icu.mxm_compute_iq(0));
    case 5: return mxm_issue(icu.mxm_param_load_iq(0));
    case 6: return mxm_issue(icu.mxm_backend_iq(0));
    default: throw std::out_of_range("invalid joint trace Queue selection");
    }
}

} // namespace

int main(int argc, char** argv)
try {
    require(argc == 2, "expected output directory");
    const std::filesystem::path output = argv[1];
    std::filesystem::create_directories(output);
    std::ofstream trace(output / "cmodel_qctl_runtime_issue_trace.csv");
    require(static_cast<bool>(trace), "cannot create joint CModel Trace");

    C2cDmaSystem system(ddr_config(), 8);
    static_assert(std::is_same_v<decltype(system.chip().mxm_unit(0)), MxmCore&>,
        "joint test must bind the new five-stream MxmCore production model");

    struct ImageData {
        ProgramImage image;
        std::vector<std::uint8_t> bytes;
    };
    std::vector<ImageData> data;
    std::uint64_t address = 0x90000;
    auto add_imem = [&](IcuMacroQueueKind kind, std::size_t queue,
                        std::vector<std::uint8_t> bytes) {
        ProgramImage image{address, kind, queue, bytes.size(),
            IcuQueueMode::Macro};
        data.push_back({image, std::move(bytes)});
        address += 0x100;
    };
    add_imem(IcuMacroQueueKind::Mem, 0, mem_context(7, 100, 1));
    add_imem(IcuMacroQueueKind::Mem, 1, mem_context(9, 200, 2));
    add_imem(IcuMacroQueueKind::MxmLoad, 0, mxm_context(7, 2, 2, 1));
    add_imem(IcuMacroQueueKind::MxmDequant, 0, mxm_context(8, 1, 1, 1));
    add_imem(IcuMacroQueueKind::MxmCompute, 0, mxm_context(9, 7, 7, 6));
    add_imem(IcuMacroQueueKind::MxmParamLoad, 0, mxm_context(10, 1, 1, 1));
    add_imem(IcuMacroQueueKind::MxmBackend, 0, mxm_context(11, 8, 13, 10));

    constexpr std::uint64_t generation = 1;
    auto add_qctl = [&](IcuMacroQueueKind kind, std::size_t queue,
                        std::vector<std::uint8_t> bytes) {
        ProgramImage image{};
        image.source_address = address;
        image.kind = kind;
        image.queue_index = queue;
        image.byte_count = bytes.size();
        image.storage = ProgramStorage::Qctl;
        image.generation = generation;
        data.push_back({image, std::move(bytes)});
        address += 0x100;
    };
    add_qctl(IcuMacroQueueKind::Mem, 0, qctl_records({
        {1, IcuControlOpcode::Sync, 0},
        {3, IcuControlOpcode::WaitEvent, 0x55},
        {5, kQueueControlMemPacket, 7}}));
    add_qctl(IcuMacroQueueKind::Mem, 1, qctl_records({
        {1, IcuControlOpcode::Notify, 0}}));
    const auto participant_records = qctl_records({
        {1, IcuControlOpcode::Sync, 0},
        {3, IcuControlOpcode::WaitEvent, 0x55}});
    add_qctl(IcuMacroQueueKind::MxmLoad, 0, participant_records);
    add_qctl(IcuMacroQueueKind::MxmDequant, 0, participant_records);
    add_qctl(IcuMacroQueueKind::MxmCompute, 0, participant_records);
    add_qctl(IcuMacroQueueKind::MxmParamLoad, 0, participant_records);
    add_qctl(IcuMacroQueueKind::MxmBackend, 0, participant_records);

    ProgramImage events{};
    events.source_address = address;
    events.kind = IcuMacroQueueKind::Mem;
    events.queue_index = 0;
    events.storage = ProgramStorage::QueueEvents;
    events.generation = generation;
    data.push_back({events, {}});

    std::vector<ProgramImage> images;
    for (const auto& entry : data) {
        stage(system, entry.image.source_address, entry.bytes);
        images.push_back(entry.image);
    }
    auto& loader = system.program_loader();
    loader.submit(images, system.cycle());
    for (std::size_t timeout = 0; !loader.ready_to_start(); ++timeout) {
        require(timeout < 20000, "joint Program Load/PRIME timed out");
        system.tick();
    }
    loader.start();
    loader.set_dependencies_ready(true);

    trace << "global_cycle,phase,queue,fu,logical_cycle,head_valid,head_cycle,opcode,operand,"
        "scheduler_ce,sync_arrive,sync_release,notify_valid,notify_ready,event_valid,event_wait,"
        "event_consume,mem_packet_valid,mem_packet_ordinal,mem_packet_retired,issue_valid,"
        "issue_ready,issue_fire,issue_instruction,issue_cycle,event_in_valid,event_in_tag,fault\n";

    std::array<std::size_t, 7> issue_count{};
    std::array<bool, 7> saw_pause{};
    std::array<bool, 6> saw_sync_arrive{};
    bool saw_notify = false;
    bool saw_event_wait = false;
    bool saw_event_consume = false;
    bool saw_mem_packet = false;
    bool saw_mem_packet_retire = false;
    std::size_t runtime_cycle = 0;
    bool runtime_started = false;
    for (std::size_t timeout = 0; timeout < 20000; ++timeout) {
        InstructionControlUnit::QctlRuntimeInputs inputs;
        inputs.event_targets.assign(kQueueCount, false);
        inputs.mem_packet_retired.assign(kQueueCount, false);
        if (runtime_started && runtime_cycle == 7) {
            inputs.event_valid = true;
            inputs.event_tag = 0x55;
            for (std::size_t index : std::array<std::size_t, 6>{0,208,212,216,220,224})
                inputs.event_targets[index] = true;
        }
        if (runtime_started && runtime_cycle == 13)
            inputs.mem_packet_retired[0] = true;
        system.chip().icu().drive_qctl_runtime_inputs(std::move(inputs));
        system.tick();
        auto& icu = system.chip().icu();
        if (!loader.started()) continue;
        runtime_started = true;
        const auto& view = icu.qctl_hardware_interface_view();
        require(!view.fault_o, "QCTL Runtime raised a fault");
        for (std::size_t selected = 0; selected < kQueues.size(); ++selected) {
            const auto queue = kQueues[selected];
            const auto& endpoint = view.endpoints.at(queue);
            const auto issue = issue_sample(icu, selected);
            saw_pause[selected] = saw_pause[selected]
                || !endpoint.scheduler_counter_ce_o;
            if (selected != 1)
                saw_sync_arrive[selected > 1 ? selected - 1 : 0] =
                    saw_sync_arrive[selected > 1 ? selected - 1 : 0]
                    || endpoint.sync_arrive_valid_o;
            saw_notify = saw_notify
                || (selected == 1 && endpoint.notify_valid_o
                    && endpoint.notify_ready_i);
            saw_event_wait = saw_event_wait || endpoint.event_wait_o;
            saw_event_consume = saw_event_consume || endpoint.event_consume_o;
            saw_mem_packet = saw_mem_packet || endpoint.mem_packet_valid_o;
            saw_mem_packet_retire = saw_mem_packet_retire
                || endpoint.mem_packet_retired_i;
            issue_count[selected] += issue.fire;
            trace << runtime_cycle << ",POST_POSEDGE," << queue << ','
                  << kFuNames[selected] << ',' << endpoint.scheduler_cycle_i
                  << ',' << endpoint.store_head_valid_i << ','
                  << endpoint.store_head_record_i.cycle << ','
                  << static_cast<unsigned>(endpoint.store_head_record_i.opcode)
                  << ',' << endpoint.store_head_record_i.operand << ','
                  << endpoint.scheduler_counter_ce_o << ','
                  << endpoint.sync_arrive_valid_o << ','
                  << endpoint.sync_release_valid_i << ','
                  << endpoint.notify_valid_o << ',' << endpoint.notify_ready_i
                  << ',' << endpoint.event_valid_i << ',' << endpoint.event_wait_o
                  << ',' << endpoint.event_consume_o << ','
                  << endpoint.mem_packet_valid_o << ','
                  << endpoint.mem_packet_ordinal_o << ','
                  << endpoint.mem_packet_retired_i << ',' << issue.valid << ','
                  << issue.ready << ',' << issue.fire << ',' << std::hex
                  << issue.instruction << std::dec << ',' << issue.cycle << ','
                  << view.event_in_valid_i << ',' << view.event_in_tag_i << ','
                  << endpoint.fault_o << '\n';
        }
        ++runtime_cycle;
        if (runtime_cycle > 28) break;
    }

    require(runtime_started, "joint runtime never started");
    require(saw_sync_arrive == std::array<bool, 6>{true,true,true,true,true,true},
        "not all MEM/MXM participants reached SYNC");
    require(saw_notify, "NOTIFY did not complete the Barrier rendezvous");
    require(saw_event_wait && saw_event_consume,
        "Event WAIT/Consume was not observed");
    require(saw_mem_packet && saw_mem_packet_retire,
        "MEM_PACKET Valid/Retire was not observed");
    require(std::ranges::all_of(saw_pause, [](bool value) { return value; }),
        "not every selected Scheduler paused under QCTL");
    // The retained MEM macro fixture expands to its two scheduled native
    // operations; each of the five new MXM streams expands to one command.
    require(issue_count == std::array<std::size_t, 7>{2,2,1,1,1,1,1},
        "MEM/MXM production issue count mismatch");
    // This transport-focused case deliberately supplies no activation/weight
    // stream payloads to the datapath.  Validate the five production ICU
    // Frontends/Schedulers here; datapath C1-C8 remains a retained baseline.
    const auto& icu = system.chip().icu();
    require(!icu.mem_iq(0).macro_decoder_fault()
            && !icu.mem_iq(1).macro_decoder_fault()
            && !icu.mxm_load_iq(0).macro_decoder_fault()
            && !icu.mxm_dequant_iq(0).macro_decoder_fault()
            && !icu.mxm_compute_iq(0).macro_decoder_fault()
            && !icu.mxm_param_load_iq(0).macro_decoder_fault()
            && !icu.mxm_backend_iq(0).macro_decoder_fault(),
        "a production MEM/MXM Macro Frontend faulted during joint test");
    std::cout << "QCTL_RUNTIME_ISSUE_CMODEL_PASS rows="
              << runtime_cycle * kQueues.size() << " issues=9\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "qctl_runtime_issue_joint_trace_test failed: "
              << error.what() << '\n';
    return 1;
}
