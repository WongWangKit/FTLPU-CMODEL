#include "ftlpu/system/icu.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace ftlpu;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

std::vector<IcuRawImemWord128> make_single_context_image(
    std::uint32_t start_cycle,
    std::uint64_t native_instruction,
    std::size_t native_bits,
    std::size_t operand_bits)
{
    std::vector<IcuRawImemWord128> image(2);
    image[0].lanes[0] = 0x15u;
    std::size_t offset = 128;
    auto push = [&](std::uint64_t value, std::size_t width) {
        for (std::size_t bit = 0; bit < width; ++bit, ++offset) {
            if (((value >> bit) & 1u) == 0) continue;
            image[offset / 128].lanes[(offset % 128) / 32]
                |= std::uint32_t{1} << (offset % 32);
        }
    };
    push(0, 3);
    push(start_cycle, 32);
    push(0, operand_bits);
    push(0, 1);
    push(0, 1);
    push(native_instruction, native_bits);
    push(0, 2);
    push(0, 2);
    return image;
}

template <typename Queue>
void prime(Queue& queue, IcuMacroQueueKind kind,
    std::uint32_t start_cycle, std::uint64_t instruction,
    std::size_t native_bits, std::size_t operand_bits)
{
    queue.load_raw_macro(kind, make_single_context_image(start_cycle,
        instruction, native_bits, operand_bits));
    for (std::size_t cycle = 0; !queue.raw_macro_ready(); ++cycle) {
        require(cycle < 64, "MXM trace frontend did not prime");
        queue.prefetch_raw_macro();
    }
}

struct Sample {
    bool valid{};
    bool ready{};
    bool fire{};
    std::uint64_t instruction{};
    std::uint32_t issue_cycle{};
    bool conflict{};
    bool fault{};
};

struct FrontendSample {
    bool start_valid{};
    bool start_ready{};
    bool start_fire{};
    bool request_valid{};
    bool request_ready{};
    bool request_fire{};
    std::size_t request_address{};
    bool request_last{};
    bool response_valid{};
    bool response_ready{};
    bool response_fire{};
    bool response_last{};
    bool word_valid{};
    bool word_ready{};
    bool word_fire{};
    bool word_last{};
    bool decoder_word_valid{};
    bool decoder_word_ready{};
    bool decoder_word_fire{};
    bool decoder_word_last{};
    bool context_enqueue_valid{};
    bool context_enqueue_ready{};
    bool context_enqueue_fire{};
    bool context_dequeue_valid{};
    bool context_dequeue_ready{};
    bool context_dequeue_fire{};
    std::size_t word_fifo_level{};
    std::size_t context_fifo_level{};
    std::size_t active_count{};
    bool fetch_complete{};
    bool decoder_done{};
    bool frontend_ready{};
    bool fault{};
};

template <typename Queue>
FrontendSample advance_frontend(Queue& queue)
{
    queue.prefetch_raw_macro();
    const auto& fetch = queue.mxm_fetch_port_signals();
    const auto& context = queue.context_fifo_port_signals();
    return FrontendSample{
        fetch.start_valid_i,
        fetch.start_ready_o,
        fetch.start_fire,
        fetch.imem_read_req_valid_o,
        fetch.imem_read_req_ready_i,
        fetch.imem_read_req_fire,
        fetch.imem_read_req_address_o,
        fetch.imem_read_req_last_o,
        fetch.imem_read_rsp_valid_i,
        fetch.imem_read_rsp_ready_o,
        fetch.imem_read_rsp_fire,
        fetch.imem_read_rsp_last_i,
        fetch.word_valid_o,
        fetch.word_ready_i,
        fetch.word_fire,
        fetch.word_last_o,
        fetch.decoder_word_valid_i,
        fetch.decoder_word_ready_o,
        fetch.decoder_word_fire,
        fetch.decoder_word_last_i,
        context.enqueue_valid_i,
        context.enqueue_ready_o,
        context.enqueue_fire,
        context.dequeue_valid_o,
        context.dequeue_ready_i,
        context.dequeue_fire,
        queue.word_fifo_level(),
        queue.context_fifo_occupancy(),
        queue.active_macro_count(),
        queue.macro_fetch_complete(),
        queue.macro_decoder_done(),
        queue.raw_macro_ready(),
        queue.macro_decoder_fault()};
}

void write_frontend_sample(std::ofstream& trace,
    const FrontendSample& value)
{
    trace << ',' << value.start_valid << ',' << value.start_ready
          << ',' << value.start_fire << ',' << value.request_valid
          << ',' << value.request_ready << ',' << value.request_fire
          << ',' << value.request_address << ',' << value.request_last
          << ',' << value.response_valid << ',' << value.response_ready
          << ',' << value.response_fire << ',' << value.response_last
          << ',' << value.word_valid << ',' << value.word_ready
          << ',' << value.word_fire << ',' << value.word_last
          << ',' << value.decoder_word_valid
          << ',' << value.decoder_word_ready
          << ',' << value.decoder_word_fire
          << ',' << value.decoder_word_last
          << ',' << value.context_enqueue_valid
          << ',' << value.context_enqueue_ready
          << ',' << value.context_enqueue_fire
          << ',' << value.context_dequeue_valid
          << ',' << value.context_dequeue_ready
          << ',' << value.context_dequeue_fire
          << ',' << value.word_fifo_level
          << ',' << value.context_fifo_level
          << ',' << value.active_count
          << ',' << value.fetch_complete
          << ',' << value.decoder_done
          << ',' << value.frontend_ready
          << ',' << value.fault;
}

template <typename Queue>
Sample sample(Queue& queue)
{
    static_cast<void>(queue.dispatch_macro_issue(true));
    const auto& pins = queue.macro_issue_port_signals();
    return Sample{
        pins.issue_valid_o,
        pins.issue_ready_i,
        pins.issue_fire,
        pins.issue_instruction_o
            ? isa::encode_mxm_command(*pins.issue_instruction_o) : 0,
        pins.issue_cycle_o,
        pins.conflict_o,
        pins.fault_o};
}

void write_sample(std::ofstream& trace, const Sample& value)
{
    trace << ',' << value.valid << ',' << value.ready << ',' << value.fire
          << ',' << value.instruction << ',' << value.issue_cycle;
}

} // namespace

int main(int argc, char** argv)
try {
    require(argc == 2, "expected output directory");
    const std::filesystem::path output = argv[1];
    std::filesystem::create_directories(output);
    std::ofstream trace(output / "cmodel_trace.csv");
    require(static_cast<bool>(trace), "cannot open CModel MXM C7 trace");

    std::ofstream frontend_trace(output / "cmodel_frontend_trace.csv");
    require(static_cast<bool>(frontend_trace),
        "cannot open CModel MXM C7 frontend trace");

    InstructionControlUnit::MxmLoadIcu frontend_load;
    InstructionControlUnit::MxmComputeIcu frontend_compute;
    InstructionControlUnit::MxmDequantIcu frontend_dequant;
    InstructionControlUnit::MxmParamLoadIcu frontend_param;
    InstructionControlUnit::MxmBackendIcu frontend_backend;
    frontend_load.load_raw_macro(IcuMacroQueueKind::MxmLoad,
        make_single_context_image(3, 0x2, 2, 1));
    frontend_compute.load_raw_macro(IcuMacroQueueKind::MxmCompute,
        make_single_context_image(4, 0x7, 7, 6));
    frontend_dequant.load_raw_macro(IcuMacroQueueKind::MxmDequant,
        make_single_context_image(3, 0x1, 1, 1));
    frontend_param.load_raw_macro(IcuMacroQueueKind::MxmParamLoad,
        make_single_context_image(6, 0x1, 1, 1));
    frontend_backend.load_raw_macro(IcuMacroQueueKind::MxmBackend,
        make_single_context_image(7, 0x8, 13, 10));

    frontend_trace << "cycle,phase";
    constexpr std::array<std::string_view, 5> stream_names{
        "load", "compute", "dequant", "param_load", "backend"};
    constexpr std::array<std::string_view, 33> frontend_signals{
        "fetch_start_valid", "fetch_start_ready", "fetch_start_fire",
        "imem_request_valid", "imem_request_ready", "imem_request_fire",
        "imem_request_address", "imem_request_last",
        "imem_response_valid", "imem_response_ready", "imem_response_fire",
        "imem_response_last", "fetch_word_valid", "fetch_word_ready",
        "fetch_word_fire", "fetch_word_last", "decoder_word_valid",
        "decoder_word_ready", "decoder_word_fire", "decoder_word_last",
        "context_enqueue_valid", "context_enqueue_ready",
        "context_enqueue_fire", "context_dequeue_valid",
        "context_dequeue_ready", "context_dequeue_fire", "word_fifo_level",
        "context_fifo_level", "active_count", "fetch_complete",
        "decoder_done", "frontend_ready", "fault"};
    for (const auto stream : stream_names)
        for (const auto signal : frontend_signals)
            frontend_trace << ',' << stream << '_' << signal;
    frontend_trace << '\n';

    std::size_t frontend_rows = 0;
    std::size_t ready_tail = 0;
    for (; frontend_rows < 64 && ready_tail < 3; ++frontend_rows) {
        const std::array<FrontendSample, 5> samples{
            advance_frontend(frontend_load),
            advance_frontend(frontend_compute),
            advance_frontend(frontend_dequant),
            advance_frontend(frontend_param),
            advance_frontend(frontend_backend)};
        frontend_trace << frontend_rows << ",BOTH";
        for (const auto& value : samples)
            write_frontend_sample(frontend_trace, value);
        frontend_trace << '\n';
        const bool all_ready = frontend_load.raw_macro_ready()
            && frontend_compute.raw_macro_ready()
            && frontend_dequant.raw_macro_ready()
            && frontend_param.raw_macro_ready()
            && frontend_backend.raw_macro_ready();
        ready_tail = all_ready ? ready_tail + 1 : 0;
    }
    require(ready_tail == 3,
        "CModel MXM C7 frontend trace did not reach stable Ready");

    InstructionControlUnit::MxmLoadIcu load;
    InstructionControlUnit::MxmComputeIcu compute;
    InstructionControlUnit::MxmDequantIcu dequant;
    InstructionControlUnit::MxmParamLoadIcu param;
    InstructionControlUnit::MxmBackendIcu backend;
    prime(load, IcuMacroQueueKind::MxmLoad, 3, 0x2, 2, 1);
    prime(compute, IcuMacroQueueKind::MxmCompute, 4, 0x7, 7, 6);
    prime(dequant, IcuMacroQueueKind::MxmDequant, 3, 0x1, 1, 1);
    prime(param, IcuMacroQueueKind::MxmParamLoad, 6, 0x1, 1, 1);
    prime(backend, IcuMacroQueueKind::MxmBackend, 7, 0x8, 13, 10);

    trace << "cycle,phase";
    for (std::string_view stream : {
             "load", "compute", "dequant", "param_load", "backend"})
        trace << ',' << stream << "_issue_valid," << stream
              << "_issue_ready," << stream << "_issue_fire," << stream
              << "_issue_instruction," << stream << "_issue_cycle";
    trace << ",conflict,fault\n";

    std::array<std::size_t, 5> fires{};
    for (std::uint32_t cycle = 0; cycle <= 7; ++cycle) {
        const std::array<Sample, 5> samples{
            sample(load), sample(compute), sample(dequant),
            sample(param), sample(backend)};
        trace << cycle << ",PRE_POSEDGE";
        for (std::size_t stream = 0; stream < samples.size(); ++stream) {
            write_sample(trace, samples[stream]);
            fires[stream] += samples[stream].fire;
        }
        bool conflict = false;
        bool fault = false;
        for (const auto& value : samples) {
            conflict = conflict || value.conflict;
            fault = fault || value.fault;
        }
        trace << ',' << conflict << ',' << fault << '\n';
    }

    require(fires == std::array<std::size_t, 5>{1, 1, 1, 1, 1},
        "CModel MXM C7 trace did not contain one Fire per stream");
    std::cout << "MXM_C7_CMODEL_TRACE_PASS rows=8 signals_per_row=27"
              << " frontend_rows=" << frontend_rows
              << " frontend_signals_per_row=165\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "mxm_five_stream_cycle_trace_test failed: "
              << error.what() << '\n';
    return 1;
}
