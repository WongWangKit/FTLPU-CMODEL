#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "smollm2_layer_phases.hpp"
#include "smollm2_norm_residual_harness.hpp"
#include "system_gantt_trace.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ftlpu::test::smollm2_layer::kHidden;
using ftlpu::test::smollm2_layer::kPrefillLength;
using ftlpu::test::smollm2_layer::PhaseResult;
namespace phases = ftlpu::test::smollm2_norm_residual;

std::vector<float> make_prefill_input()
{
    auto input = std::vector<float>(kPrefillLength * kHidden);
    for (std::size_t token = 0; token < kPrefillLength; ++token) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto raw = static_cast<int>(
                (token * 7 + hidden * 5) % 29) - 14;
            input[token * kHidden + hidden] =
                ftlpu::Bf16::from_float(
                    static_cast<float>(raw) * 0.046875f).to_float();
        }
    }
    return input;
}

std::vector<float> make_decode_input()
{
    auto input = std::vector<float>(kHidden);
    for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
        const auto raw = static_cast<int>(
            (kPrefillLength * 7 + hidden * 5) % 29) - 14;
        input[hidden] = ftlpu::Bf16::from_float(
            static_cast<float>(raw) * 0.046875f).to_float();
    }
    return input;
}

void merge_trace(
    const std::filesystem::path& output_path,
    const std::vector<std::pair<std::filesystem::path, std::size_t>>& phases)
{
    auto output = std::ofstream(output_path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write combined layer trace");
    output << "start,end,resource,detail\n";
    for (const auto& [path, offset] : phases) {
        auto input = std::ifstream(path);
        if (!input) {
            throw std::runtime_error(
                "cannot read phase trace: " + path.string());
        }
        auto line = std::string {};
        std::getline(input, line);
        while (std::getline(input, line)) {
            const auto first = line.find(',');
            const auto second = line.find(',', first + 1);
            if (first == std::string::npos
                || second == std::string::npos) {
                throw std::runtime_error("invalid phase trace row");
            }
            const auto start = std::stoull(line.substr(0, first));
            const auto end = std::stoull(
                line.substr(first + 1, second - first - 1));
            output << start + offset << ',' << end + offset
                   << line.substr(second) << '\n';
        }
    }
}

} // namespace

int main() try
{
    const auto* configured_trace = std::getenv("FTLPU_SCHEDULE_TRACE");
    const auto trace_enabled = configured_trace != nullptr
        && std::string {configured_trace}.empty() == false;
    const auto combined_trace = trace_enabled
        ? std::filesystem::path {configured_trace}
        : std::filesystem::path {};
    const auto trace_directory = trace_enabled
        ? combined_trace.parent_path() : std::filesystem::path {};
    if (trace_enabled && !trace_directory.empty()) {
        std::filesystem::create_directories(trace_directory);
    }
    auto trace_paths = std::array<std::filesystem::path, 12> {};
    if (trace_enabled) {
        for (std::size_t phase = 0; phase < trace_paths.size(); ++phase) {
            trace_paths[phase] = trace_directory
                / ("smollm2_layer_phase_" + std::to_string(phase) + ".csv");
        }
    }

    auto system = ftlpu::TspSliceSystem {};
    auto chip_timing = integration_timing::SystemGanttTrace {};
    const auto collect_chip_timing =
        integration_timing::SystemGanttTrace::enabled();
    if (collect_chip_timing) chip_timing.attach(system);
    const auto prefill_input = make_prefill_input();
    const auto prefill_norm1 = phases::run_rmsnorm(
        system, prefill_input, trace_paths[0], "prefill norm1");
    const auto prefill_attention =
        ftlpu::test::smollm2_layer::run_prefill_attention(
            system, prefill_norm1.output, trace_paths[1]);
    const auto prefill_attention_residual = phases::run_residual(
        system, prefill_input, prefill_attention.output,
        trace_paths[2], "prefill attention residual");
    const auto prefill_norm2 = phases::run_rmsnorm(
        system, prefill_attention_residual.output,
        trace_paths[3], "prefill norm2");
    const auto prefill_ffn =
        ftlpu::test::smollm2_layer::run_prefill_ffn(
            system, prefill_norm2.output, trace_paths[4]);
    const auto prefill_output = phases::run_residual(
        system, prefill_attention_residual.output,
        prefill_ffn.output, trace_paths[5], "prefill FFN residual");

    const auto decode_input = make_decode_input();
    const auto decode_norm1 = phases::run_rmsnorm(
        system, decode_input, trace_paths[6], "decode norm1");
    const auto decode_attention =
        ftlpu::test::smollm2_layer::run_decode_attention(
            system, decode_norm1.output, trace_paths[7], {},
            prefill_attention.key_cache,
            prefill_attention.value_cache);
    const auto decode_attention_residual = phases::run_residual(
        system, decode_input, decode_attention.output,
        trace_paths[8], "decode attention residual");
    const auto decode_norm2 = phases::run_rmsnorm(
        system, decode_attention_residual.output,
        trace_paths[9], "decode norm2");
    const auto decode_ffn =
        ftlpu::test::smollm2_layer::run_decode_ffn(
            system, decode_norm2.output, trace_paths[10]);
    const auto decode_output = phases::run_residual(
        system, decode_attention_residual.output,
        decode_ffn.output, trace_paths[11], "decode FFN residual");

    const std::array<std::size_t, 12> cycles {
        prefill_norm1.cycles, prefill_attention.cycles,
        prefill_attention_residual.cycles, prefill_norm2.cycles,
        prefill_ffn.cycles, prefill_output.cycles,
        decode_norm1.cycles, decode_attention.cycles,
        decode_attention_residual.cycles, decode_norm2.cycles,
        decode_ffn.cycles, decode_output.cycles,
    };
    auto total_cycles = std::size_t {0};
    auto trace_phases = std::vector<
        std::pair<std::filesystem::path, std::size_t>> {};
    for (std::size_t phase = 0; phase < cycles.size(); ++phase) {
        if (trace_enabled) {
            trace_phases.emplace_back(trace_paths[phase], total_cycles);
        }
        total_cycles += cycles[phase];
    }
    if (trace_enabled) merge_trace(combined_trace, trace_phases);
    if (collect_chip_timing) {
        constexpr std::array<const char*, 12> labels {
            "Prefill RMSNorm 1", "Prefill Attention",
            "Prefill Attention Residual", "Prefill RMSNorm 2",
            "Prefill FFN", "Prefill FFN Residual",
            "Decode RMSNorm 1", "Decode Attention",
            "Decode Attention Residual", "Decode RMSNorm 2",
            "Decode FFN", "Decode FFN Residual",
        };
        auto start = std::size_t {0};
        for (std::size_t phase = 0; phase < cycles.size(); ++phase) {
            chip_timing.phase(start, start + cycles[phase], labels[phase]);
            start += cycles[phase];
        }
        chip_timing.detach(system);
        chip_timing.write(
            "smollm2_full_chip",
            "SmolLM2 full-chip Prefill + Decode timing");
    }

    auto checksum = 0.0;
    for (const auto value : decode_output.output) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("final decode output is not finite");
        }
        checksum += value;
    }
    std::cout
        << "SmolLM2 full layer prefill+decode passed: "
        << "prefill=[128,576], decode=[1,576], "
        << "new fixed-chain RMSNorm/residual/attention/SwiGLU/FFN; cycles="
        << total_cycles << ", checksum=" << checksum;
    if (trace_enabled) {
        std::cout << ", trace=" << combined_trace.string();
    }
    std::cout << '\n';
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "SmolLM2 full layer prefill+decode failed: "
              << error.what() << '\n';
    return 1;
}
