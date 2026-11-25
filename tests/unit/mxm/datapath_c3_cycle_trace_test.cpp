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

using Supercell = ftlpu::mxm_timing::SupercellPipe<8>;
constexpr std::size_t kCycles = 40;

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
        if (!stream_) throw std::runtime_error("cannot open C3 CModel trace");
    }
    void begin(std::size_t cycle) noexcept { cycle_ = cycle; sequence_ = 0; }
    void next_sample() noexcept { sequence_ = 0; }
    void sample(std::string_view phase, std::string_view point,
        std::string_view signal, std::size_t width, const std::string& value,
        std::string_view type = "SIGNAL_SAMPLE")
    {
        stream_ << "{\"schema\":\"ftlpu.mxm.datapath.c3.trace.v1\"," 
                   "\"run_id\":\"mxm_c3_directed\"," 
                   "\"case_id\":\"c3_single_supercell\"," 
                   "\"source\":\"CMODEL\",\"global_cycle\":" << cycle_
                << ",\"logical_cycle\":" << cycle_
                << ",\"transport_tick\":" << cycle_
                << ",\"phase\":\"" << phase
                << "\",\"sample_point\":\"" << point
                << "\",\"event_sequence\":" << sequence_++
                << ",\"module\":\"lpu_mxm_supercell\"," 
                   "\"instance\":\"supercell0\",\"signal\":\"" << signal
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

bool activation_cycle(std::size_t cycle)
{
    return cycle == 18 || cycle == 19 || cycle == 20
        || cycle == 22 || cycle == 23 || cycle == 26;
}

void drive_cycle(std::size_t cycle, Supercell::Inputs& input)
{
    input = {};
    input.rst_ni = cycle >= 2;
    input.tag_i = static_cast<std::uint16_t>(0x600u + cycle);
    input.activation_valid_i = activation_cycle(cycle);
    input.weight_buffer_i = (cycle & 1u) != 0;
    for (std::size_t lane = 0; lane < 8; ++lane)
        input.activation_i[lane] = static_cast<std::uint16_t>(
            0x3f00u + ((cycle * 3u + lane) & 0x7fu));

    input.weight_write_valid_i = (cycle >= 2 && cycle <= 17) || cycle == 18;
    input.weight_write_buffer_i = cycle >= 10;
    for (std::size_t lane = 0; lane < 8; ++lane)
        input.weight_write_data_i[lane] = static_cast<std::uint16_t>(
            0x3e80u + cycle * 0x10u + lane);
}

void trace_pre(TraceWriter& trace, const Supercell::Inputs& i,
    const Supercell::Outputs& o)
{
    trace.next_sample();
    const auto bit = [](bool value) { return value ? std::string("1") : std::string("0"); };
    trace.sample("LOW","PRE_POSEDGE","rst_ni",1,bit(i.rst_ni));
    trace.sample("LOW","PRE_POSEDGE","activation_valid_i",1,bit(i.activation_valid_i));
    trace.sample("LOW","PRE_POSEDGE","activation_i",128,packed_bits(i.activation_i,16));
    trace.sample("LOW","PRE_POSEDGE","weight_buffer_i",1,bit(i.weight_buffer_i));
    trace.sample("LOW","PRE_POSEDGE","tag_i",16,bits_u64(i.tag_i,16));
    trace.sample("LOW","PRE_POSEDGE","weight_write_valid_i",1,bit(i.weight_write_valid_i));
    trace.sample("LOW","PRE_POSEDGE","weight_write_buffer_i",1,bit(i.weight_write_buffer_i));
    trace.sample("LOW","PRE_POSEDGE","weight_write_data_i",128,packed_bits(i.weight_write_data_i,16));
    trace.sample("LOW","PRE_POSEDGE","below_valid_i",8,packed_bits(i.below_valid_i,1));
    trace.sample("LOW","PRE_POSEDGE","below_partial_i",256,packed_bits(i.below_partial_i,32));
    trace.sample("LOW","PRE_POSEDGE","below_tag_i",128,packed_bits(i.below_tag_i,16));
    trace.sample("LOW","PRE_POSEDGE","weight_write_accepted_o",1,bit(o.weight_write_accepted_o));
    trace.sample("LOW","PRE_POSEDGE","weight_buffer_ready_o",2,packed_bits(o.weight_buffer_ready_o,1));
    trace.sample("LOW","PRE_POSEDGE","weight_buffer_busy_o",2,packed_bits(o.weight_buffer_busy_o,1));
    trace.sample("LOW","PRE_POSEDGE","input_weight_fault",8,packed_bits(o.input_weight_fault,1),"STATE_SAMPLE");
    trace.sample("LOW","PRE_POSEDGE","partial_protocol_fault",8,packed_bits(o.partial_protocol_fault,1),"STATE_SAMPLE");
}

void trace_post(TraceWriter& trace, const Supercell::Outputs& o)
{
    trace.next_sample();
    const auto bit = [](bool value) { return value ? std::string("1") : std::string("0"); };
    trace.sample("HIGH","POST_POSEDGE","weight_forward_valid_o",1,bit(o.weight_forward_valid_o));
    trace.sample("HIGH","POST_POSEDGE","weight_forward_buffer_o",1,bit(o.weight_forward_buffer_o));
    trace.sample("HIGH","POST_POSEDGE","weight_forward_data_o",128,packed_bits(o.weight_forward_data_o,16));
    trace.sample("HIGH","POST_POSEDGE","weight_buffer_ready_o",2,packed_bits(o.weight_buffer_ready_o,1));
    trace.sample("HIGH","POST_POSEDGE","activation_valid_o",1,bit(o.activation_valid_o));
    trace.sample("HIGH","POST_POSEDGE","activation_o",128,packed_bits(o.activation_o,16));
    trace.sample("HIGH","POST_POSEDGE","weight_buffer_o",1,bit(o.weight_buffer_o));
    trace.sample("HIGH","POST_POSEDGE","activation_tag_o",16,bits_u64(o.activation_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE","dot_valid_o",8,packed_bits(o.dot_valid_o,1));
    trace.sample("HIGH","POST_POSEDGE","dot_result_o",256,packed_bits(o.dot_result_o,32));
    trace.sample("HIGH","POST_POSEDGE","dot_tag_o",128,packed_bits(o.dot_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE","alignment_lost_o",8,packed_bits(o.alignment_lost_o,1));
    trace.sample("HIGH","POST_POSEDGE","partial_valid_o",8,packed_bits(o.partial_valid_o,1));
    trace.sample("HIGH","POST_POSEDGE","partial_o",256,packed_bits(o.partial_o,32));
    trace.sample("HIGH","POST_POSEDGE","partial_tag_o",128,packed_bits(o.partial_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE","fault_o",1,bit(o.fault_o));
    trace.sample("HIGH","POST_POSEDGE","weight_valid_q_0",8,packed_bits(o.weight_valid_q[0],1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","weight_valid_q_1",8,packed_bits(o.weight_valid_q[1],1),"STATE_SAMPLE");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto output_dir = argc > 1 ? std::filesystem::path(argv[1])
                                         : std::filesystem::path("mxm_c3_cycle");
        std::filesystem::create_directories(output_dir);
        TraceWriter trace(output_dir / "cmodel_trace.jsonl");
        Supercell supercell(true, false);
        ftlpu::timing::Clock clock;
        ftlpu::timing::SystemPhase phases;
        clock.register_module(supercell);
        Supercell::Inputs inputs{};
        phases.add_node("MXM C3 single supercell", [&] { supercell.low(); });
        for (std::size_t cycle = 0; cycle < kCycles; ++cycle) {
            drive_cycle(cycle, inputs);
            supercell.drive(inputs);
            phases.low();
            trace.begin(cycle);
            trace_pre(trace, inputs, supercell.outputs());
            clock.complete_low();
            clock.validate_all();
            clock.commit_all();
            phases.high();
            trace_post(trace, supercell.outputs());
            clock.complete_high();
            clock.negedge_all();
        }
        if (clock.cycle() != kCycles) throw std::runtime_error("C3 clock count mismatch");
        std::cout << "MXM_C3_CMODEL_TRACE_PASS cycles=" << kCycles << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MXM_C3_CMODEL_TRACE_FAIL: " << error.what() << '\n';
        return 1;
    }
}
