#include "ftlpu/icu/qctl_runtime.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ftlpu;

namespace {

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

std::uint64_t bit_mask(const std::vector<bool>& bits)
{
    std::uint64_t value = 0;
    for (std::size_t bit = 0; bit < bits.size() && bit < 64; ++bit)
        if (bits[bit]) value |= std::uint64_t{1} << bit;
    return value;
}

std::string hex_value(std::uint64_t value, unsigned width = 1)
{
    std::ostringstream stream;
    stream << "0x" << std::hex << std::setfill('0') << std::setw(width)
           << value;
    return stream.str();
}

std::string record_hex(const IcuQueueControl& record)
{
    std::ostringstream stream;
    stream << "0x" << std::hex << std::setfill('0')
           << std::setw(8) << record.operand
           << "0000000" << (static_cast<unsigned>(record.opcode) & 0xf)
           << std::setw(8) << record.cycle;
    return stream.str();
}

template<class Value>
std::uint64_t packed_values(const std::vector<Value>& values, unsigned width)
{
    std::uint64_t result = 0;
    const auto mask = (std::uint64_t{1} << width) - 1;
    for (std::size_t index = 0; index < values.size(); ++index)
        result |= (static_cast<std::uint64_t>(values[index]) & mask)
            << (index * width);
    return result;
}

class TraceHarness {
public:
    explicit TraceHarness(std::ostream& trace, std::ostream& stimulus)
        : trace_(trace), stimulus_(stimulus)
    {
        runtime_.configure(16, 3, 2, 2, 1, 4);
        scheduler_cycles_.assign(3, 0);
        run_.assign(3, false);
        retired_.assign(3, false);
        participants_.assign(3, false);
        event_targets_.assign(3, false);
        expected_.assign(3, qctl_zero_record());
    }

    void begin_case(std::string id)
    {
        case_id_ = std::move(id);
        begin_case_clear();
    }

    // The RTL MEM_PACKET case changes case_id immediately after the retired
    // endpoint becomes done, then samples one final RUN row before asserting
    // Clear for the next case.  Preserve that real TB boundary explicitly;
    // this is a trace/testbench cadence, not a functional timing offset.
    void begin_case_after_runtime_idle(std::string id)
    {
        case_id_ = std::move(id);
        runtime_cycle();
        begin_case_clear();
    }

    void begin_case_clear()
    {
        run_.assign(3, false);
        retired_.assign(3, false);
        participants_.assign(3, false);
        runtime_.configure_barrier(participants_);
        event_valid_ = false;
        event_tag_ = 0;
        event_targets_.assign(3, false);
        expected_.assign(3, qctl_zero_record());
        load_mode_ = true;
        IcuQctlRuntime::LoadInputs clear{};
        clear.clear = true;
        clear.load_mode = true;
        load_action(clear, "clear_assert", -1, qctl_zero_record());
        // The RTL Scheduler counter is reset by the posedge while Clear is
        // asserted, so the following clear-deassert sample already observes
        // cycle zero.  Mirror that edge here instead of retaining the prior
        // case's external scheduler input for one extra trace row.
        scheduler_cycles_.assign(3, 0);
        load_action({}, "clear_deassert", -1, qctl_zero_record());
    }

    void configure_queue(std::size_t queue, std::size_t base,
        std::size_t count, std::uint64_t generation)
    {
        IcuQctlRuntime::LoadInputs input{};
        input.partition_valid = true;
        input.partition_queue = queue;
        input.partition_base = base;
        input.partition_count = count;
        input.partition_generation = generation;
        load_action(input, "partition", static_cast<int>(queue),
            qctl_zero_record());
        idle_load();
    }

    void write_record(std::size_t queue, std::size_t address,
        IcuQueueControl record, bool trailing_idle = true)
    {
        expected_.at(queue) = record;
        IcuQctlRuntime::LoadInputs input{};
        input.load_valid = true;
        input.load_address = address;
        input.load_data = record;
        load_action(input, "record_load", static_cast<int>(queue), record);
        if (trailing_idle) idle_load();
    }

    void set_participants(std::vector<bool> participants,
        bool emit_load_cycle = false)
    {
        participants_ = std::move(participants);
        runtime_.configure_barrier(participants_);
        if (emit_load_cycle) idle_load();
    }

    void finish_load()
    {
        load_mode_ = false;
        log_stimulus("load_complete", -1, qctl_zero_record());
        runtime_cycle();
    }

    bool pending_response_completes_prime() const
    {
        const auto& current = view();
        if (!current.pool.response_valid_o || current.pool.response_error_o)
            return false;
        bool pending = false;
        for (std::size_t queue = 0; queue < current.endpoints.size(); ++queue) {
            if (!current.endpoints[queue].configured_o
                || current.endpoints[queue].primed_o) continue;
            pending = true;
            if (current.pool.response_queue_o != queue) return false;
        }
        return pending;
    }

    void prime(bool defer_final_response_to_run = false)
    {
        for (unsigned step = 0;
             !runtime_.all_primed()
                 && !(defer_final_response_to_run
                      && pending_response_completes_prime())
                 && step < 64;
             ++step)
            runtime_cycle();
        require(runtime_.all_primed()
                || (defer_final_response_to_run
                    && pending_response_completes_prime()),
            "QCTL shared stimulus failed to prime");
    }

    void run_assert(int queue = -1)
    {
        run_.assign(3, true);
        log_stimulus("run_assert", queue,
            queue >= 0 ? expected_.at(static_cast<std::size_t>(queue))
                       : qctl_zero_record());
        runtime_cycle();
    }

    void idle_runtime() { runtime_cycle(); }

    // Advance only the posedge-owned Runtime state.  The RTL test can observe
    // a final Pool response at one negedge, consume it at the following
    // posedge, and then drive Clear for the next POST_COMB negedge sample.
    // Keeping this edge separate avoids inventing an extra trace row.
    void runtime_edge_without_trace()
    {
        runtime_.drive_event_input(event_valid_, event_tag_, event_targets_);
        const auto& outputs = runtime_.begin_cycle(scheduler_cycles_, run_);
        runtime_.set_mem_packet_retired_inputs(retired_);
        for (std::size_t queue = 0; queue < scheduler_cycles_.size(); ++queue)
            scheduler_cycles_[queue] += outputs[queue].scheduler_counter_ce;
        runtime_.complete_cycle(retired_);
    }

    const IcuQctlGlobalInterfaceView& view() const
    { return runtime_.hardware_interface_view(); }
    bool all_primed() const noexcept { return runtime_.all_primed(); }
    IcuQctlRuntime::QueueOutputs queue_status(std::size_t queue) const
    { return runtime_.queue_status(queue); }
    const std::vector<std::uint32_t>& scheduler_cycles() const
    { return scheduler_cycles_; }

    void wait_for_event_wait(std::size_t queue)
    {
        for (unsigned step = 0; step < 32; ++step) {
            if (view().endpoints.at(queue).event_wait_o) return;
            runtime_cycle();
        }
        throw std::runtime_error("WAIT_EVENT did not reach its boundary");
    }

    void event_in(std::uint32_t tag, std::vector<bool> targets)
    {
        event_valid_ = true;
        event_tag_ = tag;
        event_targets_ = std::move(targets);
        log_stimulus("event_in", 0, expected_[0]);
        runtime_cycle();
        event_valid_ = false;
        event_targets_.assign(3, false);
        runtime_cycle();
    }

    void wait_all_done()
    {
        // Query committed Cache state.  The hardware interface view is the
        // already-emitted POST_COMB sample and intentionally remains unchanged
        // after complete_cycle(); using it here would insert a spurious cycle.
        for (unsigned step = 0; !runtime_.all_done() && step < 64; ++step)
            runtime_cycle();
        require(runtime_.all_done(),
            "QCTL shared stimulus did not complete");
    }

    void retire_mem_packet(std::size_t queue)
    {
        for (unsigned step = 0; step < 32; ++step) {
            if (view().mem_packet_valid_o.at(queue)) break;
            runtime_cycle();
        }
        require(view().mem_packet_valid_o.at(queue),
            "MEM_PACKET request was not presented");
        // The cycle in which VALID becomes observable already follows the
        // first RTL posedge in repeat(2); one further full cycle reaches the
        // second posedge before retirement is driven at its negedge.
        runtime_cycle();
        require(view().mem_packet_valid_o.at(queue)
                && view().mem_packet_ordinal_o.at(queue) == 7
                && scheduler_cycles_.at(queue) == 1,
            "MEM_PACKET did not hold ordinal and Schedule Cycle");
        retired_[queue] = true;
        log_stimulus("mem_packet_retired", static_cast<int>(queue),
            expected_.at(queue));
        runtime_cycle();
        retired_[queue] = false;
    }

    void cycle_override(std::size_t queue, std::uint32_t value)
    {
        log_stimulus("cycle_override", static_cast<int>(queue),
            expected_.at(queue));
        runtime_cycle(false);
        scheduler_cycles_.at(queue) = value;
    }

    void clear_fault()
    {
        // The RTL final Clear is asserted while normal RUN mode remains
        // selected; Clear itself resets the QCTL owners.  It is not a LOAD
        // transaction and therefore must not raise load_mode.
        load_mode_ = false;
        IcuQctlRuntime::LoadInputs clear{};
        clear.clear = true;
        load_action(clear, "clear_assert", -1, qctl_zero_record());
        scheduler_cycles_.assign(3, 0);
        // The RTL TB samples one final negedge after Clear is removed.
        idle_load();
    }

    std::size_t stimulus_lines() const noexcept { return stimulus_lines_; }
    std::size_t trace_lines() const noexcept { return trace_lines_; }
    const std::vector<std::size_t>& case_stimulus_lines() const noexcept
    { return case_stimulus_lines_; }

private:
    void idle_load()
    {
        IcuQctlRuntime::LoadInputs input{};
        input.load_mode = load_mode_;
        runtime_.load_cycle(input);
        emit_trace();
        ++physical_cycle_;
    }

    void load_action(IcuQctlRuntime::LoadInputs input,
        const std::string& action, int queue, IcuQueueControl record)
    {
        input.load_mode = load_mode_;
        runtime_.load_cycle(input);
        log_stimulus(action, queue, record);
        emit_trace();
        ++physical_cycle_;
    }

    void runtime_cycle(bool advance_scheduler = true)
    {
        runtime_.drive_event_input(event_valid_, event_tag_, event_targets_);
        const auto& outputs = runtime_.begin_cycle(scheduler_cycles_, run_);
        runtime_.set_mem_packet_retired_inputs(retired_);
        emit_trace();
        if (advance_scheduler)
            for (std::size_t queue = 0; queue < scheduler_cycles_.size(); ++queue)
                scheduler_cycles_[queue] += outputs[queue].scheduler_counter_ce;
        runtime_.complete_cycle(retired_);
        ++physical_cycle_;
    }

    void log_stimulus(const std::string& action, int queue,
        const IcuQueueControl& record)
    {
        stimulus_ << "{\"case_id\":\"" << case_id_
                  << "\",\"cycle\":" << physical_cycle_
                  << ",\"action\":\"" << action << "\",\"queue\":"
                  << queue << ",\"record\":\"" << record_hex(record)
                  << "\",\"opcode\":"
                  << static_cast<unsigned>(record.opcode)
                  << ",\"operand\":" << record.operand << "}\n";
        ++stimulus_lines_;
        if (case_ids_.empty() || case_ids_.back() != case_id_) {
            case_ids_.push_back(case_id_);
            case_stimulus_lines_.push_back(0);
        }
        ++case_stimulus_lines_.back();
    }

    void emit(std::string line)
    {
        trace_ << line << '\n';
        ++trace_lines_;
    }

    void emit_trace()
    {
        const auto& view = runtime_.hardware_interface_view();
        std::size_t pool_request_fire_count = 0;
        for (std::size_t queue = 0;
             queue < view.pool.request_valid_i.size(); ++queue) {
            pool_request_fire_count += view.pool.request_valid_i[queue]
                && view.pool.request_ready_o[queue];
        }
        require(pool_request_fire_count <= 1,
            "QCTL shared Pool accepted more than one read in a cycle");
        const auto bools = [](const std::vector<bool>& values) {
            return hex_value(bit_mask(values));
        };
        emit("{\"case_id\":\"" + case_id_ + "\",\"cycle\":"
            + std::to_string(physical_cycle_)
            + ",\"phase\":\"POST_COMB\",\"boundary\":\"global_control\""
              ",\"rst_n\":1,\"clear\":" + std::to_string(view.clear_i)
            + ",\"run\":" + std::to_string(run_[0])
            + ",\"load_mode\":" + std::to_string(view.load_mode_i)
            + ",\"sync_participant_mask\":\"" + bools(participants_)
            + "\",\"mem_packet_retired_mask\":\"" + bools(retired_)
            + "\"}");
        emit("{\"case_id\":\"" + case_id_ + "\",\"cycle\":"
            + std::to_string(physical_cycle_)
            + ",\"phase\":\"POST_COMB\",\"boundary\":\"partition_config\""
              ",\"valid\":" + std::to_string(view.partition_valid_i)
            + ",\"ready\":" + std::to_string(view.partition_ready_o)
            + ",\"fire\":"
            + std::to_string(view.partition_valid_i && view.partition_ready_o)
            + ",\"queue\":" + std::to_string(view.partition_queue_i)
            + ",\"base\":" + std::to_string(view.partition_base_i)
            + ",\"count\":" + std::to_string(view.partition_count_i)
            + ",\"generation\":"
            + std::to_string(view.partition_generation_i) + "}");
        emit("{\"case_id\":\"" + case_id_ + "\",\"cycle\":"
            + std::to_string(physical_cycle_)
            + ",\"phase\":\"POST_COMB\",\"boundary\":\"record_load\""
              ",\"valid\":" + std::to_string(view.load_valid_i)
            + ",\"ready\":" + std::to_string(view.load_ready_o)
            + ",\"fire\":"
            + std::to_string(view.load_valid_i && view.load_ready_o)
            + ",\"address\":" + std::to_string(view.load_address_i)
            + ",\"data\":\"" + record_hex(view.load_data_i) + "\"}");
        emit("{\"case_id\":\"" + case_id_ + "\",\"cycle\":"
            + std::to_string(physical_cycle_)
            + ",\"phase\":\"POST_COMB\",\"boundary\":\"event_ingress\""
              ",\"valid\":" + std::to_string(view.event_in_valid_i)
            + ",\"ready\":" + std::to_string(view.event_in_ready_o)
            + ",\"fire\":"
            + std::to_string(view.event_in_valid_i && view.event_in_ready_o)
            + ",\"tag\":" + std::to_string(view.event_in_tag_i)
            + ",\"target_mask\":\"" + bools(view.event_in_target_mask_i)
            + "\",\"backpressure\":"
            + std::to_string(view.event_backpressure_o) + "}");
        emit("{\"case_id\":\"" + case_id_ + "\",\"cycle\":"
            + std::to_string(physical_cycle_)
            + ",\"phase\":\"POST_COMB\",\"boundary\":\"global_status\""
              ",\"queue_active\":\"" + bools(view.queue_active_o)
            + "\",\"all_primed\":" + std::to_string(view.all_primed_o)
            + ",\"all_done\":" + std::to_string(view.all_done_o)
            + ",\"fault\":" + std::to_string(view.fault_o)
            + ",\"fault_code\":"
            + std::to_string(static_cast<unsigned>(view.fault_code_o)) + "}");

        const auto& pool = view.pool;
        emit("{\"case_id\":\"" + case_id_ + "\",\"cycle\":"
            + std::to_string(physical_cycle_)
            + ",\"phase\":\"POST_COMB\",\"boundary\":\"pool_fabric\""
              ",\"request_valid_mask\":\"" + bools(pool.request_valid_i)
            + "\",\"request_ready_mask\":\"" + bools(pool.request_ready_o)
            + "\",\"request_index_vector\":\""
            + hex_value(packed_values(pool.request_index_i, 2))
            + "\",\"request_generation_vector\":\""
            + hex_value(packed_values(pool.request_generation_i, 8))
            + "\",\"response_valid\":" + std::to_string(pool.response_valid_o)
            + ",\"response_queue\":" + std::to_string(pool.response_queue_o)
            + ",\"response_data\":\"" + record_hex(pool.response_data_o)
            + "\",\"response_error\":"
            + std::to_string(pool.response_error_o) + "}");
        emit("{\"case_id\":\"" + case_id_ + "\",\"cycle\":"
            + std::to_string(physical_cycle_)
            + ",\"phase\":\"POST_COMB\",\"boundary\":\"event_buffer_fabric\""
              ",\"wait_mask\":\"" + bools(view.endpoint_event_wait_i)
            + "\",\"consume_mask\":\"" + bools(view.endpoint_event_consume_i)
            + "\",\"event_valid_mask\":\"" + bools(view.endpoint_event_valid_o)
            + "\",\"event_tag\":" + std::to_string(view.endpoint_event_tag_o)
            + "}");

        for (std::size_t queue = 0; queue < 3; ++queue) {
            const auto& endpoint = view.endpoints[queue];
            const auto response_valid =
                view.fabric.endpoint_pool_response_valid_o[queue];
            emit("{\"case_id\":\"" + case_id_ + "\",\"cycle\":"
                + std::to_string(physical_cycle_)
                + ",\"phase\":\"POST_COMB\",\"boundary\":\"endpoint\",\"queue\":"
                + std::to_string(queue) + ",\"instance\":\"qctl_endpoint_"
                + std::to_string(queue) + "\",\"configure_valid\":"
                + std::to_string(view.endpoint_configure_valid_o[queue])
                + ",\"configure_count\":"
                + std::to_string(view.endpoint_configure_count_o[queue])
                + ",\"configure_generation\":"
                + std::to_string(view.endpoint_configure_generation_o[queue])
                + ",\"pool_request_valid\":"
                + std::to_string(endpoint.pool_request_valid_o)
                + ",\"pool_request_ready\":"
                + std::to_string(endpoint.pool_request_ready_i)
                + ",\"pool_request_fire\":"
                + std::to_string(endpoint.pool_request_valid_o
                    && endpoint.pool_request_ready_i)
                + ",\"pool_request_index\":"
                + std::to_string(endpoint.pool_request_index_o)
                + ",\"pool_request_generation\":"
                + std::to_string(endpoint.pool_request_generation_o)
                + ",\"pool_response_valid\":" + std::to_string(response_valid)
                + ",\"pool_response_data\":\""
                + record_hex(view.fabric.endpoint_pool_response_data_o)
                + "\",\"pool_response_error\":"
                + std::to_string(view.fabric.endpoint_pool_response_error_o)
                + ",\"scheduler_ce\":"
                + std::to_string(endpoint.scheduler_counter_ce_o)
                + ",\"local_cycle\":" + std::to_string(scheduler_cycles_[queue])
                + ",\"event_valid\":" + std::to_string(endpoint.event_valid_i)
                + ",\"event_tag\":" + std::to_string(endpoint.event_tag_i)
                + ",\"event_wait\":" + std::to_string(endpoint.event_wait_o)
                + ",\"event_consume\":" + std::to_string(endpoint.event_consume_o)
                + ",\"sync_arrive\":" + std::to_string(endpoint.sync_arrive_valid_o)
                + ",\"sync_release\":" + std::to_string(endpoint.sync_release_valid_i)
                + ",\"notify_valid\":" + std::to_string(endpoint.notify_valid_o)
                + ",\"notify_ready\":" + std::to_string(endpoint.notify_ready_i)
                + ",\"notify_fire\":"
                + std::to_string(endpoint.notify_valid_o && endpoint.notify_ready_i)
                + ",\"mem_packet_valid\":"
                + std::to_string(endpoint.mem_packet_valid_o)
                + ",\"mem_packet_ordinal\":"
                + std::to_string(endpoint.mem_packet_ordinal_o)
                + ",\"mem_packet_retired\":"
                + std::to_string(endpoint.mem_packet_retired_i)
                + ",\"mem_packet_fire\":"
                + std::to_string(endpoint.mem_packet_valid_o
                    && endpoint.mem_packet_retired_i)
                + ",\"configured\":" + std::to_string(endpoint.configured_o)
                + ",\"primed\":" + std::to_string(endpoint.primed_o)
                + ",\"done\":" + std::to_string(endpoint.done_o)
                + ",\"fault\":" + std::to_string(endpoint.fault_o)
                + ",\"fault_code\":"
                + std::to_string(static_cast<unsigned>(endpoint.fault_code_o))
                + ",\"program_record\":\"" + record_hex(expected_[queue])
                + "\",\"opcode\":"
                + std::to_string(static_cast<unsigned>(expected_[queue].opcode))
                + ",\"issue_valid\":null,\"issue_ready\":null,\"issue_fire\":null}");
        }
    }

    IcuQctlRuntime runtime_{};
    std::vector<std::uint32_t> scheduler_cycles_{};
    std::vector<bool> run_{};
    std::vector<bool> retired_{};
    std::vector<bool> participants_{};
    std::vector<bool> event_targets_{};
    std::vector<IcuQueueControl> expected_{};
    bool event_valid_{false};
    std::uint32_t event_tag_{0};
    bool load_mode_{true};
    // The RTL trace starts at the first negedge after reset deassertion, where
    // simulation_cycle is already one.  Keep the same externally sampled
    // cycle key instead of introducing a scoreboard offset.
    std::uint64_t physical_cycle_{1};
    std::string case_id_{"reset"};
    std::ostream& trace_;
    std::ostream& stimulus_;
    std::size_t stimulus_lines_{0};
    std::size_t trace_lines_{0};
    std::vector<std::string> case_ids_{};
    std::vector<std::size_t> case_stimulus_lines_{};
};

} // namespace

int main(int argc, char** argv)
try {
    std::ostringstream trace_sink;
    std::ostringstream stimulus_sink;
    std::unique_ptr<std::ofstream> trace_file;
    std::unique_ptr<std::ofstream> stimulus_file;
    std::ostream* trace = &trace_sink;
    std::ostream* stimulus = &stimulus_sink;
    std::filesystem::path output_directory;
    if (argc > 1) {
        output_directory = argv[1];
        std::filesystem::create_directories(output_directory);
        trace_file = std::make_unique<std::ofstream>(
            output_directory / "cmodel_qctl_arch_trace.jsonl");
        stimulus_file = std::make_unique<std::ofstream>(
            output_directory / "shared_qctl_stimulus.jsonl");
        require(*trace_file && *stimulus_file,
            "cannot create QCTL Phase 2-B trace outputs");
        trace = trace_file.get();
        stimulus = stimulus_file.get();
    }

    TraceHarness h(*trace, *stimulus);

    h.begin_case("no_control_passthrough");
    h.configure_queue(0, 0, 0, 1);
    h.finish_load();
    h.prime(true);
    h.run_assert(0);
    h.idle_runtime();
    h.idle_runtime();
    require(h.scheduler_cycles()[0] == 3
            && h.view().endpoints[0].scheduler_counter_ce_o
            && !h.view().fault_o,
        "zero-record Queue did not pass three Schedule Cycles");

    h.begin_case("wait_event");
    h.configure_queue(0, 0, 1, 2);
    h.write_record(0, 0, {2, IcuControlOpcode::WaitEvent, 0x55});
    h.finish_load();
    h.prime(true);
    h.run_assert(0);
    h.wait_for_event_wait(0);
    // EVENT_WAIT became observable after the first relevant posedge; one
    // additional full cycle reaches the second posedge before EVENT is driven.
    h.idle_runtime();
    require(h.scheduler_cycles()[0] == 2,
        "WAIT_EVENT did not freeze its boundary cycle");
    h.event_in(0x55, {true, false, false});
    h.wait_all_done();
    require(!h.view().fault_o, "WAIT_EVENT completion raised a fault");

    h.begin_case("sync_notify");
    h.configure_queue(0, 0, 1, 3);
    h.configure_queue(1, 1, 1, 3);
    h.configure_queue(2, 2, 1, 3);
    h.write_record(0, 0, {1, IcuControlOpcode::Sync, 0});
    h.write_record(1, 1, {1, IcuControlOpcode::Sync, 0});
    h.write_record(2, 2, {0, IcuControlOpcode::Notify, 0}, false);
    h.set_participants({true, true, false}, true);
    h.finish_load();
    h.prime(true);
    h.run_assert();
    h.wait_all_done();
    require(!h.view().fault_o, "SYNC/NOTIFY rendezvous raised a fault");

    h.begin_case("mem_packet");
    h.configure_queue(0, 0, 1, 4);
    h.write_record(0, 0, {1, kQueueControlMemPacket, 7});
    h.finish_load();
    h.prime(true);
    h.run_assert(0);
    h.retire_mem_packet(0);
    h.wait_all_done();
    require(!h.view().fault_o, "MEM_PACKET retirement raised a fault");

    h.begin_case_after_runtime_idle("pool_cache_arbitration");
    h.configure_queue(0, 0, 2, 5);
    h.configure_queue(1, 2, 2, 5);
    h.write_record(0, 0, {20, IcuControlOpcode::WaitEvent, 0x10});
    h.write_record(0, 1, {21, IcuControlOpcode::WaitEvent, 0x11});
    h.write_record(1, 2, {20, IcuControlOpcode::WaitEvent, 0x20});
    h.write_record(1, 3, {21, IcuControlOpcode::WaitEvent, 0x21});
    h.finish_load();
    // RTL wait(all_primed_o) releases as soon as the final Pool response is
    // visible and consumes it at the following edge.  Model that posedge
    // separately so the next case's Clear row is the next POST_COMB sample.
    h.prime(true);
    h.runtime_edge_without_trace();
    require(h.all_primed() && !h.view().fault_o,
        "Pool/Cache arbitration failed to prime both Queues");

    h.begin_case("fault_reset");
    h.configure_queue(0, 0, 1, 6);
    h.write_record(0, 0, {0, IcuControlOpcode::WaitEvent, 0x66});
    h.finish_load();
    h.prime(true);
    h.cycle_override(0, 1);
    h.run_assert(0);
    const auto faulted = h.queue_status(0);
    require(faulted.fault && faulted.fault_code == 2,
        "missed boundary did not report sticky global local-fault code 0x12");
    h.clear_fault();

    require(h.stimulus_lines() == 46,
        "CModel shared QCTL Stimulus is not the RTL 46-action contract");
    const std::vector<std::size_t> expected_case_lines{5, 7, 10, 7, 9, 8};
    require(h.case_stimulus_lines() == expected_case_lines,
        "CModel QCTL per-case Stimulus counts differ from RTL");

    if (trace_file) trace_file->flush();
    if (stimulus_file) stimulus_file->flush();
    std::cout << "QCTL_PHASE2B_TRACE_PASS stimulus_rows="
              << h.stimulus_lines() << " trace_rows=" << h.trace_lines()
              << " cases=6 boundaries=10 phase=POST_COMB";
    if (!output_directory.empty())
        std::cout << " output=" << output_directory.string();
    std::cout << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
