#include "ftlpu/core/timing.hpp"
#include "ftlpu/mxm/datapath_clocked.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Supercell = ftlpu::mxm_timing::SupercellPipe<8>;
constexpr std::size_t kRows = 4;
constexpr std::size_t kColumns = 4;
constexpr std::size_t kCells = kRows * kColumns;
constexpr std::size_t kCycles = 110;

std::size_t cell_index(std::size_t row, std::size_t column)
{
    return row * kColumns + column;
}

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
        if (!stream_) throw std::runtime_error("cannot open C4 CModel trace");
    }
    void begin(std::size_t cycle) noexcept { cycle_ = cycle; sequence_ = 0; }
    void next_sample() noexcept { sequence_ = 0; }
    void sample(std::string_view phase, std::string_view point,
        std::string_view instance, std::string_view signal, std::size_t width,
        const std::string& value, std::string_view type = "SIGNAL_SAMPLE")
    {
        stream_ << "{\"schema\":\"ftlpu.mxm.datapath.c4.trace.v1\"," 
                   "\"run_id\":\"mxm_c4_directed\"," 
                   "\"case_id\":\"c4_chain_array_4x4\"," 
                   "\"source\":\"CMODEL\",\"global_cycle\":" << cycle_
                << ",\"logical_cycle\":" << cycle_
                << ",\"transport_tick\":" << cycle_
                << ",\"phase\":\"" << phase
                << "\",\"sample_point\":\"" << point
                << "\",\"event_sequence\":" << sequence_++
                << ",\"module\":\"lpu_mxm_supercell\",\"instance\":\""
                << instance << "\",\"signal\":\"" << signal
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

struct RowDrive {
    bool activation_valid{};
    Supercell::Bf16Vector activation{};
    bool activation_buffer{};
    std::uint16_t tag{};
    bool weight_valid{};
    bool weight_buffer{};
    Supercell::Bf16Vector weight{};
};

std::array<RowDrive, kRows> row_drive(std::size_t cycle)
{
    std::array<RowDrive, kRows> drive{};
    const bool loading = cycle >= 2 && cycle <= 65;
    for (std::size_t row = 0; row < kRows; ++row) {
        if (loading) {
            drive[row].weight_valid = true;
            drive[row].weight_buffer = ((cycle - 2u) & 1u) != 0;
            const auto base = drive[row].weight_buffer ? 0x4000u : 0x3f80u;
            for (std::size_t lane = 0; lane < 8; ++lane)
                drive[row].weight[lane] = static_cast<std::uint16_t>(
                    base + row * 0x10u + lane);
        }
    }
    constexpr std::array<std::size_t, 4> bases{70, 71, 73, 76};
    for (std::size_t wave = 0; wave < bases.size(); ++wave) {
        for (std::size_t row = 0; row < kRows; ++row) {
            if (cycle != bases[wave] + row * 3u) continue;
            drive[row].activation_valid = true;
            drive[row].activation_buffer = (wave & 1u) != 0;
            drive[row].tag = static_cast<std::uint16_t>(0x700u + wave);
            for (std::size_t lane = 0; lane < 8; ++lane)
                drive[row].activation[lane] = static_cast<std::uint16_t>(
                    0x3f00u + row * 0x20u + wave * 4u + lane);
        }
    }
    return drive;
}

void build_inputs(std::size_t cycle,
    const std::array<std::unique_ptr<Supercell>, kCells>& cells,
    std::array<Supercell::Inputs, kCells>& inputs)
{
    const auto external = row_drive(cycle);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kColumns; ++column) {
            const auto index = cell_index(row, column);
            auto& input = inputs[index];
            input = {};
            input.rst_ni = cycle >= 2;
            if (column == 0) {
                input.activation_valid_i = external[row].activation_valid;
                input.activation_i = external[row].activation;
                input.weight_buffer_i = external[row].activation_buffer;
                input.tag_i = external[row].tag;
                input.weight_write_valid_i = external[row].weight_valid;
                input.weight_write_buffer_i = external[row].weight_buffer;
                input.weight_write_data_i = external[row].weight;
            } else {
                const auto left = cells[index - 1]->outputs();
                input.activation_valid_i = left.activation_valid_o;
                input.activation_i = left.activation_o;
                input.weight_buffer_i = left.weight_buffer_o;
                input.tag_i = left.activation_tag_o;
                input.weight_write_valid_i = left.weight_forward_valid_o;
                input.weight_write_buffer_i = left.weight_forward_buffer_o;
                input.weight_write_data_i = left.weight_forward_data_o;
            }
            if (row > 0) {
                const auto below = cells[cell_index(row - 1, column)]->outputs();
                input.below_valid_i = below.partial_valid_o;
                input.below_partial_i = below.partial_o;
                input.below_tag_i = below.partial_tag_o;
            }
        }
    }
}

std::string instance_name(std::size_t index)
{
    return "r" + std::to_string(index / kColumns)
        + "c" + std::to_string(index % kColumns);
}

void trace_pre(TraceWriter& trace, std::string_view instance,
    const Supercell::Inputs& i, const Supercell::Outputs& o)
{
    const auto bit = [](bool value) { return value ? std::string("1") : std::string("0"); };
    trace.sample("LOW","PRE_POSEDGE",instance,"rst_ni",1,bit(i.rst_ni));
    trace.sample("LOW","PRE_POSEDGE",instance,"activation_valid_i",1,bit(i.activation_valid_i));
    trace.sample("LOW","PRE_POSEDGE",instance,"activation_i",128,packed_bits(i.activation_i,16));
    trace.sample("LOW","PRE_POSEDGE",instance,"weight_buffer_i",1,bit(i.weight_buffer_i));
    trace.sample("LOW","PRE_POSEDGE",instance,"tag_i",16,bits_u64(i.tag_i,16));
    trace.sample("LOW","PRE_POSEDGE",instance,"weight_write_valid_i",1,bit(i.weight_write_valid_i));
    trace.sample("LOW","PRE_POSEDGE",instance,"weight_write_buffer_i",1,bit(i.weight_write_buffer_i));
    trace.sample("LOW","PRE_POSEDGE",instance,"weight_write_data_i",128,packed_bits(i.weight_write_data_i,16));
    trace.sample("LOW","PRE_POSEDGE",instance,"below_valid_i",8,packed_bits(i.below_valid_i,1));
    trace.sample("LOW","PRE_POSEDGE",instance,"below_partial_i",256,packed_bits(i.below_partial_i,32));
    trace.sample("LOW","PRE_POSEDGE",instance,"below_tag_i",128,packed_bits(i.below_tag_i,16));
    trace.sample("LOW","PRE_POSEDGE",instance,"weight_write_accepted_o",1,bit(o.weight_write_accepted_o));
    trace.sample("LOW","PRE_POSEDGE",instance,"weight_buffer_ready_o",2,packed_bits(o.weight_buffer_ready_o,1));
    trace.sample("LOW","PRE_POSEDGE",instance,"weight_buffer_busy_o",2,packed_bits(o.weight_buffer_busy_o,1));
    trace.sample("LOW","PRE_POSEDGE",instance,"input_weight_fault",8,packed_bits(o.input_weight_fault,1),"STATE_SAMPLE");
    trace.sample("LOW","PRE_POSEDGE",instance,"partial_protocol_fault",8,packed_bits(o.partial_protocol_fault,1),"STATE_SAMPLE");
}

void trace_post(TraceWriter& trace, std::string_view instance,
    const Supercell::Outputs& o)
{
    const auto bit = [](bool value) { return value ? std::string("1") : std::string("0"); };
    trace.sample("HIGH","POST_POSEDGE",instance,"weight_forward_valid_o",1,bit(o.weight_forward_valid_o));
    trace.sample("HIGH","POST_POSEDGE",instance,"weight_forward_buffer_o",1,bit(o.weight_forward_buffer_o));
    trace.sample("HIGH","POST_POSEDGE",instance,"weight_forward_data_o",128,packed_bits(o.weight_forward_data_o,16));
    trace.sample("HIGH","POST_POSEDGE",instance,"weight_buffer_ready_o",2,packed_bits(o.weight_buffer_ready_o,1));
    trace.sample("HIGH","POST_POSEDGE",instance,"activation_valid_o",1,bit(o.activation_valid_o));
    trace.sample("HIGH","POST_POSEDGE",instance,"activation_o",128,packed_bits(o.activation_o,16));
    trace.sample("HIGH","POST_POSEDGE",instance,"weight_buffer_o",1,bit(o.weight_buffer_o));
    trace.sample("HIGH","POST_POSEDGE",instance,"activation_tag_o",16,bits_u64(o.activation_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE",instance,"dot_valid_o",8,packed_bits(o.dot_valid_o,1));
    trace.sample("HIGH","POST_POSEDGE",instance,"dot_result_o",256,packed_bits(o.dot_result_o,32));
    trace.sample("HIGH","POST_POSEDGE",instance,"dot_tag_o",128,packed_bits(o.dot_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE",instance,"alignment_lost_o",8,packed_bits(o.alignment_lost_o,1));
    trace.sample("HIGH","POST_POSEDGE",instance,"partial_valid_o",8,packed_bits(o.partial_valid_o,1));
    trace.sample("HIGH","POST_POSEDGE",instance,"partial_o",256,packed_bits(o.partial_o,32));
    trace.sample("HIGH","POST_POSEDGE",instance,"partial_tag_o",128,packed_bits(o.partial_tag_o,16));
    trace.sample("HIGH","POST_POSEDGE",instance,"fault_o",1,bit(o.fault_o));
    trace.sample("HIGH","POST_POSEDGE",instance,"weight_valid_q_0",8,packed_bits(o.weight_valid_q[0],1),"STATE_SAMPLE");
    trace.sample("HIGH","POST_POSEDGE",instance,"weight_valid_q_1",8,packed_bits(o.weight_valid_q[1],1),"STATE_SAMPLE");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto output_dir = argc > 1 ? std::filesystem::path(argv[1])
                                         : std::filesystem::path("mxm_c4_cycle");
        std::filesystem::create_directories(output_dir);
        TraceWriter trace(output_dir / "cmodel_trace.jsonl");
        std::array<std::unique_ptr<Supercell>, kCells> cells{};
        std::array<Supercell::Inputs, kCells> inputs{};
        ftlpu::timing::Clock clock;
        ftlpu::timing::SystemPhase phases;
        for (std::size_t row = 0; row < kRows; ++row) {
            for (std::size_t column = 0; column < kColumns; ++column) {
                const auto index = cell_index(row, column);
                cells[index] = std::make_unique<Supercell>(
                    row == 0, column + 1 == kColumns);
                clock.register_module(*cells[index]);
            }
        }
        phases.add_node("MXM C4 4x4 supercell array", [&] {
            for (auto& cell : cells) cell->low();
        });
        for (std::size_t cycle = 0; cycle < kCycles; ++cycle) {
            build_inputs(cycle, cells, inputs);
            for (std::size_t index = 0; index < kCells; ++index)
                cells[index]->drive(inputs[index]);
            phases.low();
            trace.begin(cycle);
            trace.next_sample();
            for (std::size_t index = 0; index < kCells; ++index)
                trace_pre(trace, instance_name(index), inputs[index], cells[index]->outputs());
            clock.complete_low();
            clock.validate_all();
            clock.commit_all();
            phases.high();
            trace.next_sample();
            for (std::size_t index = 0; index < kCells; ++index)
                trace_post(trace, instance_name(index), cells[index]->outputs());
            clock.complete_high();
            clock.negedge_all();
        }
        if (clock.cycle() != kCycles) throw std::runtime_error("C4 clock count mismatch");
        std::cout << "MXM_C4_CMODEL_TRACE_PASS rows=" << kRows
                  << " cols=" << kColumns << " cycles=" << kCycles << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MXM_C4_CMODEL_TRACE_FAIL: " << error.what() << '\n';
        return 1;
    }
}
