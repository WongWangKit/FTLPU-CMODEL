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

using Dequant = ftlpu::mxm_timing::WeightDequantPipe<8, 32>;
using WeightBuffer = ftlpu::mxm_timing::WeightBufferPipe<8, 3>;
constexpr std::size_t kCycles = 30;

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

std::string packed_stream_valid(const Dequant::StreamValid& values)
{
    std::string result;
    result.reserve(32 * 8);
    for (std::size_t stream = 32; stream-- > 0;)
        for (std::size_t lane = 8; lane-- > 0;)
            result.push_back(values[stream][lane] ? '1' : '0');
    return result;
}

std::string packed_stream_data(const Dequant::StreamData& values)
{
    std::string result;
    result.reserve(32 * 8 * 8);
    for (std::size_t stream = 32; stream-- > 0;)
        for (std::size_t lane = 8; lane-- > 0;)
            result += bits_u64(values[stream][lane], 8);
    return result;
}

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path& path) : stream_(path)
    {
        if (!stream_) throw std::runtime_error("cannot open C2 CModel trace");
    }
    void begin(std::size_t cycle) noexcept { cycle_ = cycle; sequence_ = 0; }
    void next_sample() noexcept { sequence_ = 0; }
    void sample(std::string_view phase, std::string_view point,
        std::string_view module, std::string_view instance,
        std::string_view signal, std::size_t width, const std::string& value,
        std::string_view type = "SIGNAL_SAMPLE")
    {
        stream_ << "{\"schema\":\"ftlpu.mxm.datapath.c2.trace.v1\","
                   "\"run_id\":\"mxm_c2_directed\","
                   "\"case_id\":\"c2_dequant_weight_buffer\","
                   "\"source\":\"CMODEL\",\"global_cycle\":" << cycle_
                << ",\"logical_cycle\":" << cycle_
                << ",\"transport_tick\":" << cycle_
                << ",\"phase\":\"" << phase
                << "\",\"sample_point\":\"" << point
                << "\",\"event_sequence\":" << sequence_++
                << ",\"module\":\"" << module
                << "\",\"instance\":\"" << instance
                << "\",\"signal\":\"" << signal
                << "\",\"width\":" << width
                << ",\"value\":\"" << value
                << "\",\"event_type\":\"" << type
                << "\",\"mapping_status\":\"DIRECT\","
                   "\"source_ref\":\"include/ftlpu/mxm/datapath_clocked.hpp\"}\n";
    }
private:
    std::ofstream stream_;
    std::size_t cycle_{};
    std::size_t sequence_{};
};

void clear_streams(Dequant::Inputs& input)
{
    input.stream_valid_i = {};
    input.stream_i = {};
}

void set_parameters(Dequant::Inputs& input, std::size_t cycle)
{
    for (std::size_t lane = 0; lane < 8; ++lane) {
        const auto scale = static_cast<std::uint16_t>(
            (cycle & 1u) ? 0x3f00u + lane : 0x3f80u + lane);
        const auto zero = static_cast<std::int8_t>(
            static_cast<int>(lane) - static_cast<int>(cycle & 3u));
        input.stream_valid_i[Dequant::kScaleLoStream][lane] = true;
        input.stream_valid_i[Dequant::kScaleHiStream][lane] = true;
        input.stream_valid_i[Dequant::kZeroStream][lane] = true;
        input.stream_i[Dequant::kScaleLoStream][lane]
            = static_cast<std::uint8_t>(scale);
        input.stream_i[Dequant::kScaleHiStream][lane]
            = static_cast<std::uint8_t>(scale >> 8u);
        input.stream_i[Dequant::kZeroStream][lane]
            = static_cast<std::uint8_t>(zero);
    }
}

void set_quantized(Dequant::Inputs& input, std::size_t cycle)
{
    for (std::size_t lane = 0; lane < 8; ++lane) {
        input.stream_valid_i[Dequant::kQStream][lane] = true;
        input.stream_i[Dequant::kQStream][lane] = static_cast<std::uint8_t>(
            static_cast<int>(cycle * 7 + lane * 13) - 96);
    }
}

void drive_cycle(std::size_t cycle, Dequant::Inputs& dequant,
    WeightBuffer::Inputs& weight)
{
    dequant = {};
    weight = {};
    const bool rst_ni = cycle >= 2;
    dequant.rst_ni = weight.rst_ni = rst_ni;
    dequant.run_i = cycle != 22;
    clear_streams(dequant);

    if (cycle == 2 || cycle == 3 || cycle == 18) {
        dequant.param_write_i = true;
        dequant.param_write_buffer_i = cycle == 3;
        set_parameters(dequant, cycle);
    }
    const bool request = (cycle >= 4 && cycle <= 11)
        || cycle == 13 || cycle == 19 || cycle == 20 || cycle == 23;
    if (request) {
        dequant.request_i = true;
        dequant.param_buffer_i = (cycle & 1u) != 0;
        dequant.buffer_i = ((cycle / 2u) & 1u) != 0;
        set_quantized(dequant, cycle);
    }

    weight.activation_valid_i = cycle == 16 || cycle == 17;
    weight.weight_buffer_i = cycle == 17;
    weight.tag_i = static_cast<std::uint16_t>(0x500u + cycle);
    for (std::size_t lane = 0; lane < 8; ++lane)
        weight.activation_i[lane] = static_cast<std::uint16_t>(0x3f80u + lane);
}

void trace_pre(TraceWriter& trace, const Dequant::Inputs& di,
    const Dequant::Outputs& d, const WeightBuffer::Inputs& wi,
    const WeightBuffer::Outputs& w)
{
    trace.next_sample();
    const auto bit = [](bool value) { return value ? std::string("1") : std::string("0"); };
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","rst_ni",1,bit(di.rst_ni));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","run_i",1,bit(di.run_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","request_i",1,bit(di.request_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","buffer_i",1,bit(di.buffer_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","param_write_i",1,bit(di.param_write_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","param_write_buffer_i",1,bit(di.param_write_buffer_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","param_buffer_i",1,bit(di.param_buffer_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","stream_valid_i",256,packed_stream_valid(di.stream_valid_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","stream_i",2048,packed_stream_data(di.stream_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","consume_o",256,packed_stream_valid(d.consume_o));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","accepted_o",1,bit(d.accepted_o));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","param_accepted_o",1,bit(d.param_accepted_o));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","param_ready_o",2,bits_u64((d.param_ready_o[1]?2u:0u)|(d.param_ready_o[0]?1u:0u),2));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","fault_o",1,bit(d.fault_o));

    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","rst_ni",1,bit(wi.rst_ni));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","activation_valid_i",1,bit(wi.activation_valid_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","activation_i",128,packed_bits(wi.activation_i,16));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_buffer_i",1,bit(wi.weight_buffer_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","tag_i",16,bits_u64(wi.tag_i,16));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_write_valid_i",1,bit(wi.weight_write_valid_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_write_buffer_i",1,bit(wi.weight_write_buffer_i));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_write_data_i",128,packed_bits(wi.weight_write_data_i,16));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_write_accepted_o",1,bit(w.weight_write_accepted_o));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_buffer_ready_o",2,bits_u64((w.weight_buffer_ready_o[1]?2u:0u)|(w.weight_buffer_ready_o[0]?1u:0u),2));
    trace.sample("LOW","PRE_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_buffer_busy_o",2,bits_u64((w.weight_buffer_busy_o[1]?2u:0u)|(w.weight_buffer_busy_o[0]?1u:0u),2));
}

void trace_post(TraceWriter& trace, const Dequant::Outputs& d,
    const WeightBuffer::Outputs& w)
{
    trace.next_sample();
    const auto bit = [](bool value) { return value ? std::string("1") : std::string("0"); };
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","valid_q",2,bits_u64((d.valid_q[1]?2u:0u)|(d.valid_q[0]?1u:0u),2),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","buffer_q",2,bits_u64((d.buffer_q[1]?2u:0u)|(d.buffer_q[0]?1u:0u),2),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","valid_o",1,bit(d.valid_o));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","buffer_o",1,bit(d.buffer_o));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","weight_bf16_o",128,packed_bits(d.weight_bf16_o,16));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","param_ready_o",2,bits_u64((d.param_ready_o[1]?2u:0u)|(d.param_ready_o[0]?1u:0u),2));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_weight_dequant_pipe","dequant0","fault_o",1,bit(d.fault_o));

    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_forward_valid_o",1,bit(w.weight_forward_valid_o));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_forward_buffer_o",1,bit(w.weight_forward_buffer_o));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_forward_data_o",128,packed_bits(w.weight_forward_data_o,16));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_buffer_ready_o",2,bits_u64((w.weight_buffer_ready_o[1]?2u:0u)|(w.weight_buffer_ready_o[0]?1u:0u),2));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","activation_valid_o",1,bit(w.activation_valid_o));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","activation_o",128,packed_bits(w.activation_o,16));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_buffer_o",1,bit(w.weight_buffer_o));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","activation_tag_o",16,bits_u64(w.activation_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","fault_o",1,bit(w.fault_o));
    for (std::size_t bank=0;bank<2;++bank) for(std::size_t group=0;group<3;++group) {
        const auto suffix=std::to_string(bank)+"_"+std::to_string(group);
        trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","weight_valid_q_"+suffix,1,bit(w.weight_valid_q[bank][group]),"STATE_SAMPLE");
        trace.sample("HIGH","POST_POSEDGE","lpu_mxm_supercell","weight_buffer0","weights_q_"+suffix,128,packed_bits(w.weights_q[bank][group],16),"STATE_SAMPLE");
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto output_dir=argc>1?std::filesystem::path(argv[1]):std::filesystem::path("mxm_c2_cycle");
        std::filesystem::create_directories(output_dir);
        TraceWriter trace(output_dir/"cmodel_trace.jsonl");
        Dequant dequant;
        WeightBuffer weight_buffer;
        ftlpu::timing::Clock clock;
        ftlpu::timing::SystemPhase phases;
        clock.register_module(dequant);
        clock.register_module(weight_buffer);
        Dequant::Inputs dequant_inputs{};
        WeightBuffer::Inputs weight_inputs{};
        phases.add_node("MXM C2 dequant",[&]{dequant.low();});
        phases.add_node("MXM C2 weight buffer",[&]{
            const auto d=dequant.outputs();
            weight_inputs.weight_write_valid_i=d.valid_o;
            weight_inputs.weight_write_buffer_i=d.buffer_o;
            weight_inputs.weight_write_data_i=d.weight_bf16_o;
            weight_buffer.drive(weight_inputs);
            weight_buffer.low();
        });
        for(std::size_t cycle=0;cycle<kCycles;++cycle) {
            drive_cycle(cycle,dequant_inputs,weight_inputs);
            dequant.drive(dequant_inputs);
            phases.low();
            trace.begin(cycle);
            trace_pre(trace,dequant_inputs,dequant.outputs(),weight_inputs,weight_buffer.outputs());
            clock.complete_low();clock.validate_all();clock.commit_all();
            phases.high();
            trace_post(trace,dequant.outputs(),weight_buffer.outputs());
            clock.complete_high();clock.negedge_all();
        }
        if(clock.cycle()!=kCycles) throw std::runtime_error("C2 clock count mismatch");
        std::cout<<"MXM_C2_CMODEL_TRACE_PASS cycles="<<kCycles<<'\n';
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"MXM_C2_CMODEL_TRACE_FAIL: "<<error.what()<<'\n';
        return 1;
    }
}
