#include "ftlpu/core/timing.hpp"
#include "ftlpu/mxm/datapath_clocked.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Backend = ftlpu::mxm_timing::BackendColumnPipe<8, 16>;
using Operation = Backend::Operation;
constexpr std::size_t kCycles = 54;

std::string bits_u64(std::uint64_t value, std::size_t width)
{
    std::string result(width, '0');
    for (std::size_t bit = 0; bit < width; ++bit)
        if ((value >> bit) & 1u) result[width - 1 - bit] = '1';
    return result;
}

template <typename T, std::size_t N>
std::string packed_bits(const std::array<T, N>& values, std::size_t width)
{
    std::string result;
    result.reserve(N * width);
    for (std::size_t index = N; index-- > 0;)
        result += bits_u64(static_cast<std::uint64_t>(values[index]), width);
    return result;
}

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path& path) : stream_(path)
    {
        if (!stream_) throw std::runtime_error("cannot open C6 CModel trace");
    }
    void begin(std::size_t cycle) noexcept { cycle_ = cycle; sequence_ = 0; }
    void next_sample() noexcept { sequence_ = 0; }
    void sample(std::string_view phase, std::string_view point,
        std::string_view signal, std::size_t width, const std::string& value,
        std::string_view type = "SIGNAL_SAMPLE")
    {
        stream_ << "{\"schema\":\"ftlpu.mxm.datapath.c6.trace.v1\"," 
                   "\"run_id\":\"mxm_c6_directed\"," 
                   "\"case_id\":\"c6_backend_column\"," 
                   "\"source\":\"CMODEL\",\"global_cycle\":" << cycle_
                << ",\"logical_cycle\":" << cycle_
                << ",\"transport_tick\":" << cycle_
                << ",\"phase\":\"" << phase
                << "\",\"sample_point\":\"" << point
                << "\",\"event_sequence\":" << sequence_++
                << ",\"module\":\"lpu_mxm_backend_column\"," 
                   "\"instance\":\"backend_column0\",\"signal\":\"" << signal
                << "\",\"width\":" << width << ",\"value\":\"" << value
                << "\",\"event_type\":\"" << type
                << "\",\"mapping_status\":\"DIRECT\"," 
                   "\"source_ref\":\"include/ftlpu/mxm/datapath_clocked.hpp\"}\n";
    }
private:
    std::ofstream stream_;
    std::size_t cycle_{};
    std::size_t sequence_{};
};

void set_command(std::size_t cycle, Backend::Inputs& input)
{
    input.slot_valid_i = true;
    switch (cycle) {
    case 2: input.operation_i = Operation::DirectOut; input.address_i = 2; break;
    case 3: input.operation_i = Operation::DirectOut; input.address_i = 3;
        input.output_bf16_i = true; break;
    case 10: input.operation_i = Operation::Store; input.address_i = 0; break;
    case 11: input.operation_i = Operation::Store; input.address_i = 1; break;
    case 20: input.operation_i = Operation::AccOut; input.address_i = 0; break;
    case 21: input.operation_i = Operation::AccOut; input.address_i = 1; break;
    case 30: input.operation_i = Operation::AccStore; input.address_i = 0; break;
    case 31: input.operation_i = Operation::AccStore; input.address_i = 1; break;
    case 40: input.operation_i = Operation::AccOut; input.address_i = 0; break;
    case 41: input.operation_i = Operation::AccOut; input.address_i = 1;
        input.output_bf16_i = true; break;
    default: input.slot_valid_i = false; break;
    }
}

void set_partial(std::size_t cycle, Backend::Inputs& input)
{
    const bool valid = cycle == 4 || cycle == 5 || cycle == 12 || cycle == 13
        || cycle == 22 || cycle == 23 || cycle == 32 || cycle == 33
        || cycle == 42 || cycle == 43;
    input.partial_valid_i = valid;
    input.partial_tag_i = static_cast<std::uint16_t>(0x800u + cycle);
    std::uint32_t value = 0;
    if (cycle == 4) value = 0x3f800000u;
    if (cycle == 5) value = 0x3f808001u;
    if (cycle == 12 || cycle == 22 || cycle == 23
        || cycle == 42 || cycle == 43) value = 0x3f800000u;
    if (cycle == 13 || cycle == 32 || cycle == 33) value = 0x40000000u;
    for (std::size_t lane = 0; lane < 8; ++lane)
        input.partial_i[lane] = value;
}

void drive_cycle(std::size_t cycle, Backend::Inputs& input)
{
    input = {};
    input.rst_ni = cycle >= 2;
    input.run_i = input.rst_ni && cycle != 48 && cycle != 49;
    if (!input.run_i) return;
    set_command(cycle, input);
    set_partial(cycle, input);
}

void trace_pre(TraceWriter& trace, const Backend::Inputs& i,
    const Backend::Outputs& o)
{
    trace.next_sample();
    trace.sample("LOW","PRE_POSEDGE","rst_ni",1,bits_u64(i.rst_ni,1));
    trace.sample("LOW","PRE_POSEDGE","run_i",1,bits_u64(i.run_i,1));
    trace.sample("LOW","PRE_POSEDGE","slot_valid_i",1,bits_u64(i.slot_valid_i,1));
    trace.sample("LOW","PRE_POSEDGE","operation_i",2,bits_u64(static_cast<unsigned>(i.operation_i),2));
    trace.sample("LOW","PRE_POSEDGE","output_bf16_i",1,bits_u64(i.output_bf16_i,1));
    trace.sample("LOW","PRE_POSEDGE","address_i",16,bits_u64(i.address_i,16));
    trace.sample("LOW","PRE_POSEDGE","partial_valid_i",1,bits_u64(i.partial_valid_i,1));
    trace.sample("LOW","PRE_POSEDGE","partial_i",256,packed_bits(i.partial_i,32));
    trace.sample("LOW","PRE_POSEDGE","partial_tag_i",16,bits_u64(i.partial_tag_i,16));
    trace.sample("LOW","PRE_POSEDGE","slot_valid_o",1,bits_u64(o.slot_valid_o,1));
    trace.sample("LOW","PRE_POSEDGE","slot_bf16_o",1,bits_u64(o.slot_bf16_o,1));
    trace.sample("LOW","PRE_POSEDGE","slot_operation_o",2,bits_u64(static_cast<unsigned>(o.slot_operation_o),2));
    trace.sample("LOW","PRE_POSEDGE","slot_address_o",16,bits_u64(o.slot_address_o,16));
    trace.sample("LOW","PRE_POSEDGE","bank_request_valid_o",2,packed_bits(o.bank_request_valid_o,1));
    trace.sample("LOW","PRE_POSEDGE","bank_write_o",2,packed_bits(o.bank_write_o,1));
    trace.sample("LOW","PRE_POSEDGE","bank_address_o",6,packed_bits(o.bank_address_o,3));
    trace.sample("LOW","PRE_POSEDGE","bank_write_o[0]",1,bits_u64(o.bank_write_o[0],1));
    trace.sample("LOW","PRE_POSEDGE","bank_write_o[1]",1,bits_u64(o.bank_write_o[1],1));
    trace.sample("LOW","PRE_POSEDGE","bank_write_data_o[0]",256,packed_bits(o.bank_write_data_o[0],32));
    trace.sample("LOW","PRE_POSEDGE","bank_write_data_o[1]",256,packed_bits(o.bank_write_data_o[1],32));
    trace.sample("LOW","PRE_POSEDGE","invalid_read_o",2,packed_bits(o.invalid_read_o,1),"STATE_SAMPLE");
    trace.sample("LOW","PRE_POSEDGE","accepted_read",1,bits_u64(o.accepted_read,1),"STATE_SAMPLE");
    trace.sample("LOW","PRE_POSEDGE","alu_valid_o",1,bits_u64(o.alu_valid_o,1));
    trace.sample("LOW","PRE_POSEDGE","operand_a_o",256,packed_bits(o.operand_a_o,32));
    trace.sample("LOW","PRE_POSEDGE","operand_b_o",256,packed_bits(o.operand_b_o,32));
    trace.sample("LOW","PRE_POSEDGE","alu_address_o",16,bits_u64(o.alu_address_o,16));
    trace.sample("LOW","PRE_POSEDGE","alu_tag_o",16,bits_u64(o.alu_tag_o,16));
}

void trace_post(TraceWriter& trace, const Backend::Outputs& o)
{
    trace.next_sample();
    trace.sample("HIGH","POST_POSEDGE","result_valid_o",1,bits_u64(o.result_valid_o,1));
    trace.sample("HIGH","POST_POSEDGE","result_bf16_o",1,bits_u64(o.result_bf16_o,1));
    trace.sample("HIGH","POST_POSEDGE","result_o",256,packed_bits(o.result_o,32));
    trace.sample("HIGH","POST_POSEDGE","result_address_o",16,bits_u64(o.result_address_o,16));
    trace.sample("HIGH","POST_POSEDGE","result_tag_o",16,bits_u64(o.result_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE","fault_o",1,bits_u64(o.fault_o,1));
    trace.sample("HIGH","POST_POSEDGE","context_valid_q",2,packed_bits(o.context_valid_q,1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","context_ok_q",2,packed_bits(o.context_ok_q,1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","bypass_valid_q",3,packed_bits(o.bypass_valid_q,1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","sum_valid",3,packed_bits(o.sum_valid_q,1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","pending_valid_q",5,packed_bits(o.pending_valid_q,1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","bank_capture_valid_q",2,packed_bits(o.bank_capture_valid_q,1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","bank_read_valid",2,packed_bits(o.bank_read_valid_o,1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","bank_read_address",6,packed_bits(o.bank_read_address_o,3),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","bank_read_valid[0]",1,bits_u64(o.bank_read_valid_o[0],1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","bank_read_valid[1]",1,bits_u64(o.bank_read_valid_o[1],1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","bank_read_data[0]",256,packed_bits(o.bank_read_data_o[0],32),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","bank_read_data[1]",256,packed_bits(o.bank_read_data_o[1],32),"STATE_SAMPLE");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto output_dir = argc > 1 ? std::filesystem::path(argv[1])
                                         : std::filesystem::path("mxm_c6_cycle");
        std::filesystem::create_directories(output_dir);
        TraceWriter trace(output_dir / "cmodel_trace.jsonl");
        Backend backend;
        ftlpu::timing::Clock clock;
        ftlpu::timing::SystemPhase phases;
        clock.register_module(backend);
        Backend::Inputs inputs{};
        phases.add_node("MXM C6 backend column", [&] { backend.low(); });
        for (std::size_t cycle = 0; cycle < kCycles; ++cycle) {
            drive_cycle(cycle, inputs);
            backend.drive(inputs);
            phases.low();
            trace.begin(cycle);
            trace_pre(trace, inputs, backend.outputs());
            clock.complete_low();
            clock.validate_all();
            clock.commit_all();
            phases.high();
            trace_post(trace, backend.outputs());
            clock.complete_high();
            clock.negedge_all();
        }
        if (clock.cycle() != kCycles) throw std::runtime_error("C6 clock count mismatch");
        std::cout << "MXM_C6_CMODEL_TRACE_PASS cycles=" << kCycles << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MXM_C6_CMODEL_TRACE_FAIL: " << error.what() << '\n';
        return 1;
    }
}
