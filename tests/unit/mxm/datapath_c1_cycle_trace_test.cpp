#include "ftlpu/core/timing.hpp"
#include "ftlpu/mxm/datapath_clocked.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using ftlpu::mxm_timing::Bf16DotCsaPipe;
using ftlpu::mxm_timing::Fp32AddPipe;
using OutputPipe = ftlpu::mxm_timing::OutputPipe<3>;

constexpr std::size_t kCycles = 28;

std::string bits_u64(std::uint64_t value, std::size_t width)
{
    std::string result(width, '0');
    for (std::size_t bit = 0; bit < width; ++bit)
        if ((value >> bit) & 1u) result[width - 1 - bit] = '1';
    return result;
}

template <typename T, std::size_t N>
std::string packed_bits(const std::array<T, N>& values, std::size_t element_width)
{
    std::string result;
    result.reserve(N * element_width);
    for (std::size_t index = N; index-- > 0;)
        result += bits_u64(static_cast<std::uint64_t>(values[index]), element_width);
    return result;
}

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path& path)
        : stream_(path)
    {
        if (!stream_) throw std::runtime_error("cannot open C1 CModel trace");
    }

    void begin_cycle(std::size_t cycle) noexcept
    {
        cycle_ = cycle;
        sequence_ = 0;
    }

    void begin_sample() noexcept { sequence_ = 0; }

    void sample(std::string_view phase, std::string_view sample_point,
        std::string_view module, std::string_view instance,
        std::string_view signal, std::size_t width, const std::string& value,
        std::string_view event_type = "SIGNAL_SAMPLE")
    {
        stream_ << "{\"schema\":\"ftlpu.mxm.datapath.c1.trace.v1\","
                   "\"run_id\":\"mxm_c1_directed\","
                   "\"case_id\":\"c1_pipeline\",\"source\":\"CMODEL\","
                   "\"global_cycle\":" << cycle_
                << ",\"logical_cycle\":" << cycle_
                << ",\"transport_tick\":" << cycle_
                << ",\"phase\":\"" << phase
                << "\",\"sample_point\":\"" << sample_point
                << "\",\"event_sequence\":" << sequence_++
                << ",\"module\":\"" << module
                << "\",\"instance\":\"" << instance
                << "\",\"signal\":\"" << signal
                << "\",\"width\":" << width
                << ",\"value\":\"" << value
                << "\",\"event_type\":\"" << event_type
                << "\",\"mapping_status\":\"DIRECT\","
                   "\"source_ref\":\"include/ftlpu/mxm/datapath_clocked.hpp\"}\n";
    }

private:
    std::ofstream stream_;
    std::size_t cycle_{};
    std::size_t sequence_{};
};

bool active(std::size_t cycle)
{
    return cycle == 2 || cycle == 3 || cycle == 5 || cycle == 6
        || cycle == 8 || cycle == 9 || cycle == 17 || cycle == 18;
}

void drive_cycle(std::size_t cycle, Bf16DotCsaPipe::Inputs& dot,
    Fp32AddPipe::Inputs& add, OutputPipe::Inputs& output)
{
    const bool rst_ni = !(
        cycle < 2 || cycle == 14 || cycle == 15);
    const bool valid = active(cycle);
    dot = {};
    add = {};
    output = {};
    dot.rst_ni = add.rst_ni = output.rst_ni = rst_ni;
    dot.valid_i = add.valid_i = output.valid_i = valid;
    dot.tag_i = static_cast<std::uint16_t>(0x100u + cycle);
    add.tag_i = static_cast<std::uint16_t>(0x200u + cycle);
    output.meta_i = static_cast<std::uint32_t>(0x30000000u + cycle);
    output.bf16_i = (cycle & 1u) != 0;

    static constexpr std::array<std::array<std::uint16_t, 8>, 4> activations{{
        {{0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80}},
        {{0x3f80, 0xbf80, 0x4000, 0xc000, 0x3f00, 0xbf00, 0x4040, 0xc040}},
        {{0x0000, 0x8000, 0x0001, 0x8001, 0x3f80, 0x4000, 0x4040, 0x4080}},
        {{0x7f80, 0xff80, 0x7fc1, 0x3f80, 0xbf80, 0x0000, 0x4000, 0xc000}},
    }};
    static constexpr std::array<std::array<std::uint16_t, 8>, 4> weights{{
        {{0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80}},
        {{0x4000, 0x4000, 0x3f00, 0x3f00, 0x4040, 0x4040, 0x3f80, 0x3f80}},
        {{0x3f80, 0x3f80, 0x3f80, 0x3f80, 0xbf80, 0xbf80, 0xbf80, 0xbf80}},
        {{0x0000, 0x0000, 0x3f80, 0x7f80, 0x7f80, 0x3f80, 0x4000, 0x4000}},
    }};
    const auto pattern = cycle % activations.size();
    dot.activation_i = activations[pattern];
    dot.weight_i = weights[pattern];

    static constexpr std::array<std::uint32_t, 8> lhs{{
        0x3f800000, 0xbf800000, 0x7f800000, 0x00000001,
        0x7f7fffff, 0x40490fdb, 0x80000000, 0x3f000000,
    }};
    static constexpr std::array<std::uint32_t, 8> rhs{{
        0x40000000, 0x3f800000, 0xff800000, 0x00000001,
        0x7f7fffff, 0xc02df854, 0x00000000, 0x3f000000,
    }};
    add.lhs_i = lhs[cycle % lhs.size()];
    add.rhs_i = rhs[cycle % rhs.size()];

    output.data_i = {
        static_cast<std::uint32_t>(0x3f800000u + cycle),
        static_cast<std::uint32_t>((cycle & 1u) ? 0x3f808000u : 0xbf800000u),
        static_cast<std::uint32_t>((cycle % 3u) ? 0x7f7fffffu : 0x7fc00001u),
    };
}

void trace_pre(TraceWriter& trace, const Bf16DotCsaPipe::Inputs& dot,
    const Fp32AddPipe::Inputs& add, const OutputPipe::Inputs& output)
{
    trace.begin_sample();
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "rst_ni", 1, dot.rst_ni ? "1" : "0");
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "valid_i", 1, dot.valid_i ? "1" : "0");
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "activation_i", 128, packed_bits(dot.activation_i, 16));
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "weight_i", 128, packed_bits(dot.weight_i, 16));
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "tag_i", 16, bits_u64(dot.tag_i, 16));

    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "rst_ni", 1, add.rst_ni ? "1" : "0");
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "valid_i", 1, add.valid_i ? "1" : "0");
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "lhs_i", 32, bits_u64(add.lhs_i, 32));
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "rhs_i", 32, bits_u64(add.rhs_i, 32));
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "tag_i", 16, bits_u64(add.tag_i, 16));

    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "rst_ni", 1, output.rst_ni ? "1" : "0");
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "valid_i", 1, output.valid_i ? "1" : "0");
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "bf16_i", 1, output.bf16_i ? "1" : "0");
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "data_i", 96, packed_bits(output.data_i, 32));
    trace.sample("LOW", "PRE_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "meta_i", 32, bits_u64(output.meta_i, 32));
}

void trace_post(TraceWriter& trace, const Bf16DotCsaPipe::Outputs& dot,
    const Fp32AddPipe::Outputs& add, const OutputPipe::Outputs& output)
{
    trace.begin_sample();
    std::uint16_t valid_q = 0;
    for (std::size_t stage = 0; stage < dot.stage_valid_q.size(); ++stage)
        if (dot.stage_valid_q[stage]) valid_q |= std::uint16_t{1} << stage;
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "valid_q", 9, bits_u64(valid_q, 9), "STATE_SAMPLE");
    for (std::size_t stage = 0; stage < dot.stage_tag_q.size(); ++stage) {
        trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
            "tag_q_" + std::to_string(stage), 16,
            bits_u64(dot.stage_tag_q[stage], 16), "STATE_SAMPLE");
    }
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "valid_o", 1, dot.valid_o ? "1" : "0");
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "result_o", 32, bits_u64(dot.result_o, 32));
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "tag_o", 16, bits_u64(dot.tag_o, 16));
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_bf16_dot_csa", "dot0",
        "alignment_lost_o", 1, dot.alignment_lost_o ? "1" : "0");

    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "a1_valid_q", 1, add.a1_valid_q ? "1" : "0", "STATE_SAMPLE");
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "a2_valid_q", 1, add.a2_valid_q ? "1" : "0", "STATE_SAMPLE");
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "a1_tag_q", 16, bits_u64(add.a1_tag_q, 16), "STATE_SAMPLE");
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "a2_tag_q", 16, bits_u64(add.a2_tag_q, 16), "STATE_SAMPLE");
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "valid_o", 1, add.valid_o ? "1" : "0");
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "result_o", 32, bits_u64(add.result_o, 32));
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_fp32_add_pipe", "add0",
        "tag_o", 16, bits_u64(add.tag_o, 16));

    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "valid_o", 1, output.valid_o ? "1" : "0");
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "bf16_o", 1, output.bf16_o ? "1" : "0");
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "data_o", 96, packed_bits(output.data_o, 32));
    trace.sample("HIGH", "POST_POSEDGE", "lpu_mxm_output_pipe", "output0",
        "meta_o", 32, bits_u64(output.meta_o, 32));
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto output_dir = argc > 1
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("mxm_c1_cycle");
        std::filesystem::create_directories(output_dir);
        TraceWriter trace(output_dir / "cmodel_trace.jsonl");

        Bf16DotCsaPipe dot;
        Fp32AddPipe add;
        OutputPipe output;
        ftlpu::timing::Clock clock;
        ftlpu::timing::SystemPhase phases;
        clock.register_module(dot);
        clock.register_module(add);
        clock.register_module(output);
        phases.add_node("MXM C1 dot", [&] { dot.low(); });
        phases.add_node("MXM C1 add", [&] { add.low(); });
        phases.add_node("MXM C1 output", [&] { output.low(); });

        Bf16DotCsaPipe::Inputs dot_inputs{};
        Fp32AddPipe::Inputs add_inputs{};
        OutputPipe::Inputs output_inputs{};
        for (std::size_t cycle = 0; cycle < kCycles; ++cycle) {
            drive_cycle(cycle, dot_inputs, add_inputs, output_inputs);
            dot.drive(dot_inputs);
            add.drive(add_inputs);
            output.drive(output_inputs);
            phases.low();
            trace.begin_cycle(cycle);
            trace_pre(trace, dot_inputs, add_inputs, output_inputs);
            clock.complete_low();
            clock.validate_all();
            clock.commit_all();
            phases.high();
            trace_post(trace, dot.outputs(), add.outputs(), output.outputs());
            clock.complete_high();
            clock.negedge_all();
        }
        if (clock.cycle() != kCycles || phases.low_count() != kCycles
            || phases.high_count() != kCycles)
            throw std::runtime_error("C1 harness did not use the unified clock");
        std::cout << "MXM_C1_CMODEL_TRACE_PASS cycles=" << kCycles << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MXM_C1_CMODEL_TRACE_FAIL: " << error.what() << '\n';
        return 1;
    }
}
