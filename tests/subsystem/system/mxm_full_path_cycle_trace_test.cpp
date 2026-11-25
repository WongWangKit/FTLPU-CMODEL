#include "ftlpu/system/c2c_dma_system.hpp"

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
#include <vector>

namespace {

using namespace ftlpu;

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

std::vector<std::uint8_t> context_bytes(std::uint32_t start_cycle,
    std::uint64_t native_instruction, std::size_t native_bits,
    std::size_t operand_bits, std::size_t inner_count = 1)
{
    std::array<std::uint32_t, 8> lanes{};
    lanes[0] = 0x15u;
    std::size_t offset = 128;
    auto push = [&](std::uint64_t value, std::size_t width) {
        for (std::size_t bit = 0; bit < width; ++bit, ++offset)
            lanes[offset / 32] |= std::uint32_t((value >> bit) & 1u)
                << (offset % 32);
    };
    push(0, 3);
    push(start_cycle, 32);
    push(0, operand_bits);
    push(0, 1);
    push(0, 1);
    push(native_instruction, native_bits);
    push(inner_count == 1 ? 0 : 1, 2);
    push(0, 2);
    if (inner_count != 1) {
        push(inner_count - 1, 11);
        push(0, 8);
        push(0, 10);
    }

    std::vector<std::uint8_t> bytes(32);
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::uint8_t>(
            lanes[index / 4] >> (8 * (index % 4)));
    return bytes;
}

void stage(C2cDmaSystem& system, std::uint64_t address,
    const std::vector<std::uint8_t>& bytes)
{
    C2cVector vector{};
    for (std::size_t byte = 0; byte < bytes.size(); ++byte)
        vector.payload[byte / hw::kLanesPerTile]
            [byte % hw::kLanesPerTile] = bytes[byte];
    system.ddr4().initialize_vector(address, vector);
}

void drive_data(MxmCore& core)
{
    auto inputs = core.inputs();
    inputs.rst_ni = true;
    for (std::size_t row = 0; row < MxmCore::kRows; ++row) {
        inputs.activation_valid_i[row] = true;
        inputs.activation_i[row].fill(0x3f80);
        for (std::size_t lane = 0; lane < MxmCore::kTerms; ++lane) {
            inputs.weight_stream_valid_i[row][MxmCore::kWeightQStream][lane] = true;
            inputs.weight_stream_valid_i[row][MxmCore::kWeightScaleLoStream][lane] = true;
            inputs.weight_stream_valid_i[row][MxmCore::kWeightScaleHiStream][lane] = true;
            inputs.weight_stream_valid_i[row][MxmCore::kWeightZeroStream][lane] = true;
            inputs.weight_stream_i[row][MxmCore::kWeightQStream][lane] = 1;
            inputs.weight_stream_i[row][MxmCore::kWeightScaleLoStream][lane] = 0x80;
            inputs.weight_stream_i[row][MxmCore::kWeightScaleHiStream][lane] = 0x3f;
            inputs.weight_stream_i[row][MxmCore::kWeightZeroStream][lane] = 0;
        }
    }
    core.drive(std::move(inputs));
}

template <std::size_t N>
std::uint64_t bool_bits(const std::array<bool, N>& values)
{
    static_assert(N <= 64);
    std::uint64_t bits = 0;
    for (std::size_t index = 0; index < N; ++index)
        bits |= std::uint64_t(values[index]) << index;
    return bits;
}

template <std::size_t R, std::size_t C>
std::uint64_t bool_bits(const std::array<std::array<bool, C>, R>& values)
{
    static_assert(R * C <= 64);
    std::uint64_t bits = 0;
    for (std::size_t row = 0; row < R; ++row)
        for (std::size_t column = 0; column < C; ++column)
            bits |= std::uint64_t(values[row][column]) << (row * C + column);
    return bits;
}

template <typename Values>
std::string fp32_hex(const Values& values)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (auto outer = values.rbegin(); outer != values.rend(); ++outer)
        for (auto lane = outer->rbegin(); lane != outer->rend(); ++lane)
            stream << std::setw(8) << *lane;
    return stream.str();
}

template <std::size_t N>
std::string u16_hex(const std::array<std::uint16_t, N>& values)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (auto value = values.rbegin(); value != values.rend(); ++value)
        stream << std::setw(4) << *value;
    return stream.str();
}

std::uint64_t operation_bits(
    const std::array<MxmBackendOperation, MxmCore::kColumns>& values)
{
    std::uint64_t result = 0;
    for (std::size_t column = 0; column < values.size(); ++column)
        result |= std::uint64_t(values[column]) << (2 * column);
    return result;
}

std::uint64_t bank_bool_bits(
    const std::array<std::array<bool, 2>, MxmCore::kColumns>& values)
{
    return bool_bits(values);
}

void write_hex(std::ofstream& trace, std::uint64_t value)
{
    trace << ',' << std::hex << value << std::dec;
}

} // namespace

int main(int argc, char** argv)
try {
    require(argc == 2, "expected output directory");
    const std::filesystem::path output = argv[1];
    std::filesystem::create_directories(output);
    std::ofstream trace(output / "cmodel_trace.csv");
    require(static_cast<bool>(trace), "cannot open CModel C8 trace");

    C2cDmaSystem system(ddr_config(), 8);
    struct ImageSpec {
        IcuMacroQueueKind kind;
        std::uint32_t start;
        std::uint64_t native;
        std::size_t native_bits;
        std::size_t operand_bits;
        std::size_t inner_count;
    };
    const std::array<ImageSpec, 5> specs{{
        {IcuMacroQueueKind::MxmLoad, 1, 2, 2, 1, 1},
        {IcuMacroQueueKind::MxmCompute, 35, 1, 7, 6, 1},
        {IcuMacroQueueKind::MxmDequant, 1, 1, 1, 1, 32},
        {IcuMacroQueueKind::MxmParamLoad, 0, 1, 1, 1, 1},
        {IcuMacroQueueKind::MxmBackend, 58, 4, 13, 10, 1},
    }};
    std::vector<ProgramImage> images;
    for (std::size_t index = 0; index < specs.size(); ++index) {
        const auto bytes = context_bytes(specs[index].start,
            specs[index].native, specs[index].native_bits,
            specs[index].operand_bits, specs[index].inner_count);
        const auto source = std::uint64_t{0xa0000} + index * 0x100;
        stage(system, source, bytes);
        images.push_back(ProgramImage{source, specs[index].kind, 0,
            bytes.size(), IcuQueueMode::Macro});
    }
    auto& loader = system.program_loader();
    loader.submit(images, system.cycle());
    loader.start();
    loader.set_dependencies_ready(true);

    trace << "cycle,phase,compute_fire,load_fire,param_fire,dequant_fire,backend_fire,"
        "compute_queue_level,load_queue_level,param_queue_level,dequant_queue_level,backend_queue_level,"
        "compute_row_valid,load_row_valid,param_row_valid,dequant_row_valid,"
        "cell_weight_valid,cell_activation_valid,partial_valid,partial_data,partial_tag,"
        "returned_valid,returned_data,returned_tag,backend_issue_valid,"
        "backend_issue_operation,backend_issue_address,backend_slot_valid,alu_valid,"
        "bank_request_valid,bank_write,result_valid,result_bf16,"
        "result_data,result_address,result_tag,frontend_fault,backend_fault,fault\n";

    std::array<std::size_t, 5> command_fires{};
    std::size_t partials = 0;
    std::size_t returned = 0;
    std::size_t results = 0;
    bool recorded_last = false;
    for (std::size_t physical = 0; physical < 6000 && !recorded_last; ++physical) {
        drive_data(system.chip().mxm_unit(0));
        system.tick();
        auto& core = system.chip().mxm_unit(0);
        const auto& icu = system.chip().icu();
        const auto queue_cycle = icu.mxm_load_iq(0).cycle();
        if (queue_cycle == 0) continue;
        const auto logical = queue_cycle - 1;
        if (logical > 90) break;

        const auto& pre = core.pre_posedge_outputs();
        const auto& post = core.outputs();
        command_fires[0] += pre.load_launch_fire;
        command_fires[1] += pre.compute_launch_fire;
        command_fires[2] += pre.dequant_launch_fire;
        command_fires[3] += pre.param_load_launch_fire;
        command_fires[4] += pre.backend_launch_fire;
        for (bool valid : post.partial_valid_o) partials += valid;
        for (bool valid : post.returned_valid_o) returned += valid;
        for (bool valid : post.result_valid_o) results += valid;

        trace << logical << ",BOTH";
        write_hex(trace, pre.compute_launch_fire);
        write_hex(trace, pre.load_launch_fire);
        write_hex(trace, pre.param_load_launch_fire);
        write_hex(trace, pre.dequant_launch_fire);
        write_hex(trace, pre.backend_launch_fire);
        write_hex(trace, core.compute_queue_level());
        write_hex(trace, core.load_queue_level());
        write_hex(trace, core.param_load_queue_level());
        write_hex(trace, core.dequant_queue_level());
        write_hex(trace, core.backend_queue_level());
        write_hex(trace, bool_bits(post.compute_row_valid_o));
        write_hex(trace, bool_bits(post.load_row_valid_o));
        write_hex(trace, bool_bits(post.param_load_row_valid_o));
        write_hex(trace, bool_bits(post.dequant_row_valid_o));
        write_hex(trace, bool_bits(post.cell_weight_valid_o));
        write_hex(trace, bool_bits(post.cell_activation_valid_o));
        write_hex(trace, bool_bits(post.partial_valid_o));
        trace << ',' << fp32_hex(post.partial_o)
              << ',' << u16_hex(post.partial_tag_o);
        write_hex(trace, bool_bits(post.returned_valid_o));
        trace << ',' << fp32_hex(post.returned_o)
              << ',' << u16_hex(post.returned_tag_o);
        write_hex(trace, bool_bits(post.backend_issue_valid_o));
        write_hex(trace, operation_bits(post.backend_issue_operation_o));
        trace << ',' << u16_hex(post.backend_issue_address_o);
        write_hex(trace, bool_bits(post.backend_slot_valid_o));
        write_hex(trace, bool_bits(post.alu_valid_o));
        write_hex(trace, bank_bool_bits(post.bank_request_valid_o));
        write_hex(trace, bank_bool_bits(post.bank_write_o));
        write_hex(trace, bool_bits(post.result_valid_o));
        write_hex(trace, bool_bits(post.result_bf16_o));
        trace << ',' << fp32_hex(post.result_o)
              << ',' << u16_hex(post.result_address_o)
              << ',' << u16_hex(post.result_tag_o);
        write_hex(trace, post.frontend_fault_o);
        write_hex(trace, post.backend_fault_o);
        write_hex(trace, post.fault_o);
        trace << '\n';
        recorded_last = logical == 90;
    }

    if (!recorded_last) {
        const auto& icu = system.chip().icu();
        std::cerr << "C8 progress loader_state="
                  << static_cast<unsigned>(loader.state())
                  << " started=" << loader.started()
                  << " issue_enabled=" << icu.program_issue_enabled()
                  << " queue_cycles=" << icu.mxm_load_iq(0).cycle() << ','
                  << icu.mxm_compute_iq(0).cycle() << ','
                  << icu.mxm_dequant_iq(0).cycle() << ','
                  << icu.mxm_param_load_iq(0).cycle() << ','
                  << icu.mxm_backend_iq(0).cycle()
                  << " ready=" << icu.mxm_load_iq(0).raw_macro_ready()
                  << icu.mxm_compute_iq(0).raw_macro_ready()
                  << icu.mxm_dequant_iq(0).raw_macro_ready()
                  << icu.mxm_param_load_iq(0).raw_macro_ready()
                  << icu.mxm_backend_iq(0).raw_macro_ready()
                  << " faults=" << icu.mxm_load_iq(0).macro_decoder_fault()
                  << icu.mxm_compute_iq(0).macro_decoder_fault()
                  << icu.mxm_dequant_iq(0).macro_decoder_fault()
                  << icu.mxm_param_load_iq(0).macro_decoder_fault()
                  << icu.mxm_backend_iq(0).macro_decoder_fault() << '\n';
    }
    require(recorded_last, "CModel C8 trace did not reach logical cycle 90");
    require(command_fires == std::array<std::size_t, 5>{1, 1, 32, 1, 1},
        "CModel C8 command Fire counts differ from the legal schedule");
    require(partials == MxmCore::kColumns
            && returned == MxmCore::kRows
            && results == MxmCore::kColumns,
        "CModel C8 did not produce one complete matrix result wave");
    require(!system.chip().mxm_unit(0).outputs().fault_o,
        "CModel C8 ended with a fault");
    std::cout << "MXM_C8_CMODEL_TRACE_PASS rows=91 commands=36 partials="
              << partials << " returned=" << returned
              << " results=" << results << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "mxm_full_path_cycle_trace_test failed: "
              << error.what() << '\n';
    return 1;
}
