#include "ftlpu/system/c2c_dma_system.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ftlpu;

namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

std::string pack(const IcuQueueControl& record)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(8) << record.operand
           << std::setw(8) << static_cast<unsigned>(record.opcode)
           << std::setw(8) << record.cycle;
    return stream.str();
}

struct Action {
    enum class Kind { Partition, Record } kind{Kind::Partition};
    std::size_t queue{0};
    std::size_t base_or_address{0};
    std::size_t count{0};
    std::uint64_t generation{0};
    IcuQueueControl record{};
};

IcuQctlRuntime::LoadInputs drive(const Action& action)
{
    IcuQctlRuntime::LoadInputs input{};
    input.load_mode = true;
    if (action.kind == Action::Kind::Partition) {
        input.partition_valid = true;
        input.partition_queue = action.queue;
        input.partition_base = action.base_or_address;
        input.partition_count = action.count;
        input.partition_generation = action.generation;
    } else {
        input.load_valid = true;
        input.load_address = action.base_or_address;
        input.load_data = action.record;
    }
    return input;
}
}

int main(int argc, char** argv) try
{
    require(argc == 2, "expected output directory");
    const std::filesystem::path output = argv[1];
    std::filesystem::create_directories(output);
    auto system = std::make_unique<C2cDmaSystem>();
    auto& icu = system->chip().icu();
    icu.stage_program_issue_enabled(false);

    constexpr std::size_t queue = 16;
    constexpr std::uint64_t generation = 1;
    const IcuQueueControl first{3, IcuControlOpcode::WaitEvent, 0x16};
    const IcuQueueControl second{5, kQueueControlMemPacket, 0x103};
    const std::vector<Action> actions{
        {Action::Kind::Partition, queue, 0, 2, generation, {}},
        {Action::Kind::Record, queue, 0, 0, generation, first},
        {Action::Kind::Record, queue, 1, 0, generation, second},
    };

    std::ofstream trace(output / "phase3_r5_qctl_distributed_trace.csv");
    require(static_cast<bool>(trace), "cannot open distributed QCTL trace");
    trace << "cycle,phase,load_mode,partition_valid,partition_ready,partition_fire,"
        "partition_queue,partition_base,partition_count,partition_generation,"
        "load_valid,load_ready,load_fire,load_address,load_data,"
        "relay_packet_valid,relay_packet_ready,relay_packet_kind,relay_packet_queue,"
        "relay_packet_offset,east_r0_input_fire,east_r0_local_fire,east_r0_output_valid,"
        "east_r0_output_ready,east_r0_fifo_level,east_r1_input_fire,east_r1_local_fire,"
        "east_r1_fifo_level,endpoint_configure_valid,endpoint_store_loaded,"
        "endpoint_head_valid,endpoint_head_error,endpoint_configured,endpoint_primed,"
        "queue_active,load_complete,all_primed,fault,fault_code\n";

    std::size_t action_index = 0;
    std::optional<Action> held{};
    unsigned partition_fires = 0;
    unsigned record_fires = 0;
    unsigned configure_pulses = 0;
    unsigned stalls = 0;
    bool last_load_complete = false;
    bool last_all_primed = false;
    bool last_fault = false;
    constexpr unsigned rows = 12;
    for (unsigned cycle = 0; cycle < rows; ++cycle) {
        IcuQctlRuntime::LoadInputs input{};
        if (cycle == 0) {
            input.clear = true;
            input.load_mode = false;
        } else {
            if (!held && action_index < actions.size()) held = actions[action_index];
            if (held) input = drive(*held);
            else input.load_mode = false;
        }

        icu.drive_qctl_program_load(input);
        system->tick();
        const auto& view = icu.qctl_hardware_interface_view();
        const auto& manager = view.load_manager;
        const bool partition_fire = manager.partition_valid_i
            && manager.partition_ready_o;
        const bool load_fire = manager.load_valid_i && manager.load_ready_o;
        const bool accepted = held && ((held->kind == Action::Kind::Partition
            && partition_fire) || (held->kind == Action::Kind::Record && load_fire));
        if (held && !accepted) ++stalls;
        if (accepted) {
            if (held->kind == Action::Kind::Partition) ++partition_fires;
            else ++record_fires;
            ++action_index;
            held.reset();
        }
        configure_pulses += view.endpoint_configure_valid_o.at(queue);

        const auto& r0 = view.relays.at(0);
        const auto& r1 = view.relays.at(1);
        const auto& endpoint = view.endpoints.at(queue);
        last_load_complete = view.load_complete_o;
        last_all_primed = view.all_primed_o;
        last_fault = view.fault_o;
        trace << cycle << ",PRE_POSEDGE," << manager.load_mode_i << ','
            << manager.partition_valid_i << ',' << manager.partition_ready_o << ','
            << partition_fire << ',' << manager.partition_queue_i << ','
            << manager.partition_base_i << ',' << manager.partition_count_i << ','
            << manager.partition_generation_i << ',' << manager.load_valid_i << ','
            << manager.load_ready_o << ',' << load_fire << ','
            << manager.load_address_i << ',' << pack(manager.load_data_i) << ','
            << manager.relay_packet_valid_o << ',' << manager.relay_packet_ready_i << ','
            << manager.relay_packet_kind_o << ',' << manager.relay_packet_queue_o << ','
            << manager.relay_packet_offset_o << ',' << r0.chain_packet_fire << ','
            << r0.local_fire << ',' << r0.next_packet_valid_o << ','
            << r0.next_packet_ready_i << ',' << r0.fifo_level << ','
            << r1.chain_packet_fire << ',' << r1.local_fire << ',' << r1.fifo_level << ','
            << view.endpoint_configure_valid_o.at(queue) << ','
            << view.endpoint_store_loaded_o.at(queue) << ','
            << view.endpoint_head_valid_o.at(queue) << ','
            << view.endpoint_head_error_o.at(queue) << ',' << endpoint.configured_o << ','
            << endpoint.primed_o << ',' << view.queue_active_o.at(queue) << ','
            << view.load_complete_o << ',' << view.all_primed_o << ','
            << view.fault_o << ',' << static_cast<unsigned>(view.fault_code_o) << '\n';
    }

    const auto first_record = icu.qctl_committed_program_record(0);
    const auto second_record = icu.qctl_committed_program_record(1);
    const auto status = icu.qctl_queue_status(queue);
    const bool pass = action_index == actions.size() && !held
        && partition_fires == 1 && record_fires == 2 && configure_pulses == 1
        && first_record && second_record && pack(*first_record) == pack(first)
        && pack(*second_record) == pack(second) && status.primed
        && last_load_complete && last_all_primed && !last_fault;

    std::ofstream manifest(output / "phase3_r5_qctl_distributed_manifest.json");
    require(static_cast<bool>(manifest), "cannot open distributed QCTL manifest");
    manifest << "{\n  \"schema\": \"ftlpu.phase3.r5.qctl_distributed.v2\",\n"
        << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
        << "  \"phase\": \"PRE_POSEDGE\",\n  \"cycle_offset\": 0,\n"
        << "  \"trace\": \"phase3_r5_qctl_distributed_trace.csv\",\n"
        << "  \"rows\": " << rows << ",\n"
        << "  \"partition_fires\": " << partition_fires << ",\n"
        << "  \"record_fires\": " << record_fires << ",\n"
        << "  \"configure_pulses\": " << configure_pulses << ",\n"
        << "  \"stall_cycles\": " << stalls << ",\n"
        << "  \"queue\": " << queue << ",\n"
        << "  \"relay_hops\": 1,\n"
        << "  \"final_primed\": " << (status.primed ? "true" : "false") << ",\n"
        << "  \"fault\": " << (last_fault ? "true" : "false") << "\n}\n";
    if (!pass) std::cerr << "R5 distributed QCTL trace did not reach PASS"
        << " action_index=" << action_index << " held=" << held.has_value()
        << " partition_fires=" << partition_fires
        << " record_fires=" << record_fires
        << " configure_pulses=" << configure_pulses
        << " first_record=" << first_record.has_value()
        << " second_record=" << second_record.has_value()
        << " first_match=" << (first_record && pack(*first_record) == pack(first))
        << " second_match=" << (second_record && pack(*second_record) == pack(second))
        << " primed=" << status.primed
        << " load_complete=" << last_load_complete
        << " all_primed=" << last_all_primed
        << " fault=" << last_fault << '\n';
    return pass ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "R5 distributed QCTL trace: " << error.what() << '\n';
    return 1;
}
