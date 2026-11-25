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

using ResultReturn = ftlpu::mxm_timing::ResultReturnPipe<4, 8>;
constexpr std::size_t kCycles = 36;

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

std::string packed_vectors(const std::array<ResultReturn::Fp32Vector, 4>& values)
{
    std::string result;
    for (std::size_t index = values.size(); index-- > 0;)
        result += packed_bits(values[index], 32);
    return result;
}

template <typename T>
std::string packed_matrix_scalar(const std::array<std::array<T, 4>, 4>& values,
    std::size_t width)
{
    std::string result;
    for (std::size_t row = 4; row-- > 0;)
        for (std::size_t column = 4; column-- > 0;)
            result += bits_u64(static_cast<std::uint64_t>(values[row][column]), width);
    return result;
}

std::string packed_matrix_vectors(
    const std::array<std::array<ResultReturn::Fp32Vector, 4>, 4>& values)
{
    std::string result;
    for (std::size_t row = 4; row-- > 0;)
        for (std::size_t column = 4; column-- > 0;)
            result += packed_bits(values[row][column], 32);
    return result;
}

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path& path) : stream_(path)
    {
        if (!stream_) throw std::runtime_error("cannot open C5 CModel trace");
    }
    void begin(std::size_t cycle) noexcept { cycle_ = cycle; sequence_ = 0; }
    void next_sample() noexcept { sequence_ = 0; }
    void sample(std::string_view phase, std::string_view point,
        std::string_view signal, std::size_t width, const std::string& value)
    {
        stream_ << "{\"schema\":\"ftlpu.mxm.datapath.c5.trace.v1\"," 
                   "\"run_id\":\"mxm_c5_directed\"," 
                   "\"case_id\":\"c5_result_return\"," 
                   "\"source\":\"CMODEL\",\"global_cycle\":" << cycle_
                << ",\"logical_cycle\":" << cycle_
                << ",\"transport_tick\":" << cycle_
                << ",\"phase\":\"" << phase
                << "\",\"sample_point\":\"" << point
                << "\",\"event_sequence\":" << sequence_++
                << ",\"module\":\"lpu_mxm_result_return\"," 
                   "\"instance\":\"result_return0\",\"signal\":\"" << signal
                << "\",\"width\":" << width << ",\"value\":\"" << value
                << "\",\"event_type\":\"SIGNAL_SAMPLE\"," 
                   "\"mapping_status\":\"DIRECT\"," 
                   "\"source_ref\":\"include/ftlpu/mxm/datapath_clocked.hpp\"}\n";
    }
private:
    std::ofstream stream_;
    std::size_t cycle_{};
    std::size_t sequence_{};
};

void drive_cycle(std::size_t cycle, ResultReturn::Inputs& input)
{
    input = {};
    input.rst_ni = cycle >= 2;
    input.run_i = input.rst_ni && cycle != 16 && cycle != 17;
    if (!input.run_i) return;
    const int epoch = cycle < 16 ? static_cast<int>(cycle) - 2
                                 : static_cast<int>(cycle) - 18;
    for (std::size_t column = 0; column < 4; ++column) {
        input.top_valid_i[column] = epoch >= 0
            && (epoch < 6 || ((epoch + static_cast<int>(column)) % 4 != 1));
        input.top_tag_i[column] = static_cast<std::uint16_t>(column * 1024u + epoch);
        for (std::size_t lane = 0; lane < 8; ++lane) {
            input.top_i[column][lane] = 0x80000000u
                ^ (static_cast<std::uint32_t>(column) << 24u)
                ^ (static_cast<std::uint32_t>(lane) << 16u)
                ^ static_cast<std::uint32_t>(epoch * 997 + static_cast<int>(lane * 37u + column));
        }
    }
}

void trace_pre(TraceWriter& trace, const ResultReturn::Inputs& i)
{
    trace.next_sample();
    trace.sample("LOW","PRE_POSEDGE","rst_ni",1,bits_u64(i.rst_ni,1));
    trace.sample("LOW","PRE_POSEDGE","run_i",1,bits_u64(i.run_i,1));
    trace.sample("LOW","PRE_POSEDGE","top_valid_i",4,packed_bits(i.top_valid_i,1));
    trace.sample("LOW","PRE_POSEDGE","top_i",1024,packed_vectors(i.top_i));
    trace.sample("LOW","PRE_POSEDGE","top_tag_i",64,packed_bits(i.top_tag_i,16));
}

void trace_post(TraceWriter& trace, const ResultReturn::Outputs& o)
{
    trace.next_sample();
    trace.sample("HIGH","POST_POSEDGE","left_valid_o",4,packed_bits(o.left_valid_o,1));
    trace.sample("HIGH","POST_POSEDGE","left_o",1024,packed_vectors(o.left_o));
    trace.sample("HIGH","POST_POSEDGE","left_tag_o",64,packed_bits(o.left_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE","down_valid_o",16,packed_matrix_scalar(o.down_valid_o,1));
    trace.sample("HIGH","POST_POSEDGE","down_o",4096,packed_matrix_vectors(o.down_o));
    trace.sample("HIGH","POST_POSEDGE","down_tag_o",256,packed_matrix_scalar(o.down_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE","left_link_valid_o",16,packed_matrix_scalar(o.left_link_valid_o,1));
    trace.sample("HIGH","POST_POSEDGE","left_link_o",4096,packed_matrix_vectors(o.left_link_o));
    trace.sample("HIGH","POST_POSEDGE","left_link_tag_o",256,packed_matrix_scalar(o.left_link_tag_o,16));
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto output_dir = argc > 1 ? std::filesystem::path(argv[1])
                                         : std::filesystem::path("mxm_c5_cycle");
        std::filesystem::create_directories(output_dir);
        TraceWriter trace(output_dir / "cmodel_trace.jsonl");
        ResultReturn result_return;
        ftlpu::timing::Clock clock;
        ftlpu::timing::SystemPhase phases;
        clock.register_module(result_return);
        ResultReturn::Inputs inputs{};
        phases.add_node("MXM C5 result return", [&] { result_return.low(); });
        for (std::size_t cycle = 0; cycle < kCycles; ++cycle) {
            drive_cycle(cycle, inputs);
            result_return.drive(inputs);
            phases.low();
            trace.begin(cycle);
            trace_pre(trace, inputs);
            clock.complete_low();
            clock.validate_all();
            clock.commit_all();
            phases.high();
            trace_post(trace, result_return.outputs());
            clock.complete_high();
            clock.negedge_all();
        }
        if (clock.cycle() != kCycles) throw std::runtime_error("C5 clock count mismatch");
        std::cout << "MXM_C5_CMODEL_TRACE_PASS cycles=" << kCycles << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MXM_C5_CMODEL_TRACE_FAIL: " << error.what() << '\n';
        return 1;
    }
}
