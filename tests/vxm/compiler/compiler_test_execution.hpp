#pragma once

#include "ftlpu/vxm/compiler/compiler.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

namespace vxm_compiler_test::execution {

template<typename Fn>
std::vector<ftlpu::VxmLutEntry> make_table(
    float minimum, float width, std::size_t count, Fn fn)
{
    auto entries = std::vector<ftlpu::VxmLutEntry>{};
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto x0 = minimum + static_cast<float>(index) * width;
        const auto y0 = fn(x0);
        entries.push_back(ftlpu::VxmLutEntry::from_float(
            (fn(x0 + width) - y0) / width, y0));
    }
    return entries;
}

void configure_luts(ftlpu::VxmSlice& slice)
{
    constexpr std::size_t entries = 256;
    constexpr float ln2 = 0.6931471805599453f;
    slice.configure_special_lut(
        ftlpu::VxmSpecialAluOpcode::Exp,
        {-ln2 / 2.0f, ln2 / entries},
        make_table(
            -ln2 / 2.0f, ln2 / entries, entries,
            [](float value) { return std::exp(value); }));
    slice.configure_special_lut(
        ftlpu::VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f / entries},
        make_table(
            1.0f, 1.0f / entries, entries,
            [](float value) { return 1.0f / value; }));
    slice.configure_special_lut(
        ftlpu::VxmSpecialAluOpcode::Rsqrt,
        {1.0f, 3.0f / entries},
        make_table(
            1.0f, 3.0f / entries, entries,
            [](float value) { return 1.0f / std::sqrt(value); }));
}

std::vector<const ftlpu::vxm::compiler::VxmScheduledPhase*>
phases_producing(
    const ftlpu::vxm::compiler::VxmCompiledProgram& program,
    ftlpu::vxm::compiler::ValueId value)
{
    auto phases =
        std::vector<const ftlpu::vxm::compiler::VxmScheduledPhase*>{};
    for (const auto& phase : program.schedule.phases) {
        if (std::any_of(
                phase.outputs.begin(), phase.outputs.end(),
                [value](const auto& output) {
                    return output.value == value;
                })) {
            phases.push_back(&phase);
        }
    }
    return phases;
}

void assert_full_stage_coverage(
    const ftlpu::vxm::compiler::VxmScheduledPhase& phase)
{
    auto covered = std::array<bool, ftlpu::VxmLane::kAluCount>{};
    for (const auto& instruction : phase.instructions) {
        covered.at(instruction.stage) = true;
    }
    assert(std::all_of(
        covered.begin(), covered.end(),
        [](bool present) { return present; }));
}

void assert_compact_control_is_shared(
    const ftlpu::vxm::compiler::VxmCompiledProgram& program)
{
    for (const auto& phase : program.phases) {
        for (const auto& command : phase.config_commands) {
            assert(command.stage
                   < ftlpu::VxmSuperlaneInstructionControl::kStageCount);
        }
    }
}

std::filesystem::path output_path(const char* name)
{
    const auto source = std::filesystem::path(__FILE__);
    if (source.is_absolute()) {
        const auto directory = source.parent_path() / "results";
        std::filesystem::create_directories(directory);
        return directory / name;
    }
    const auto directory = std::filesystem::current_path()
        / "tests" / "vxm" / "compiler" / "results";
    std::filesystem::create_directories(directory);
    return directory / name;
}

inline void check()
{
    using namespace ftlpu;
    using namespace ftlpu::vxm::compiler;

    constexpr std::size_t rows = 8;
    constexpr std::size_t length = 16;
    constexpr auto sram_timing = VxmSramTiming{1, 1};

    auto softmax = compile_kernel(make_softmax_kernel(rows, length));
    const auto maximum = softmax.kernel.named("maximum").output;
    const auto denominator = softmax.kernel.named("denominator").output;
    const auto output = softmax.kernel.named("output").output;

    const auto maximum_phases = phases_producing(softmax, maximum);
    assert(maximum_phases.size() == 1);
    assert(maximum_phases.front()->chain_depth == VxmChainDepth::Two);
    assert(maximum_phases.front()->parallel_chain_count == 8);
    assert_full_stage_coverage(*maximum_phases.front());

    const auto denominator_phases =
        phases_producing(softmax, denominator);
    assert(denominator_phases.size() == 2);
    for (const auto* phase : denominator_phases) {
        assert(phase->chain_depth == VxmChainDepth::Four);
        assert(phase->parallel_chain_count == 4);
        assert_full_stage_coverage(*phase);
    }
    const auto output_phases = phases_producing(softmax, output);
    assert(output_phases.size() == 2);
    for (const auto* phase : output_phases) {
        assert(phase->parallel_chain_count == 4);
        assert_full_stage_coverage(*phase);
    }
    assert_compact_control_is_shared(softmax);

    auto rmsnorm =
        compile_kernel(make_rmsnorm_kernel(rows, length, 1.0e-5f));
    const auto square_sum = rmsnorm.kernel.named("square_sum").output;
    const auto square_phases = phases_producing(rmsnorm, square_sum);
    assert(square_phases.size() == 1);
    assert(square_phases.front()->chain_depth == VxmChainDepth::Two);
    assert(square_phases.front()->parallel_chain_count == 8);
    assert_full_stage_coverage(*square_phases.front());
    assert_compact_control_is_shared(rmsnorm);

    auto swiglu = compile_kernel(make_swiglu_kernel(2, length));
    assert(swiglu.schedule.phases.size() == 1);
    assert(swiglu.schedule.phases.front().chain_depth
           == VxmChainDepth::Eight);
    assert(swiglu.schedule.phases.front().parallel_chain_count == 2);
    assert_full_stage_coverage(swiglu.schedule.phases.front());
    assert_compact_control_is_shared(swiglu);

    const auto input = softmax.kernel.named("x").output;
    auto values = VxmHostValueStore{};
    auto references = std::vector<std::vector<std::vector<float>>>(
        VxmSlice::kTileCount,
        std::vector<std::vector<float>>(
            VxmSuperlane::kLaneCount,
            std::vector<float>(rows * length)));
    for (std::size_t tile = 0; tile < VxmSlice::kTileCount; ++tile) {
        for (std::size_t lane = 0;
             lane < VxmSuperlane::kLaneCount; ++lane) {
            auto data = std::vector<float>(rows * length);
            for (std::size_t row = 0; row < rows; ++row) {
                auto maximum_value = -1.0e30f;
                for (std::size_t element = 0;
                     element < length; ++element) {
                    const auto index = row * length + element;
                    data[index] = 1.5f * std::sin(
                        0.031f * static_cast<float>(
                            index + lane * 3 + tile * 5));
                    maximum_value =
                        std::max(maximum_value, data[index]);
                }
                auto sum = 0.0f;
                for (std::size_t element = 0;
                     element < length; ++element) {
                    sum += std::exp(
                        data[row * length + element] - maximum_value);
                }
                for (std::size_t element = 0;
                     element < length; ++element) {
                    const auto index = row * length + element;
                    references[tile][lane][index] =
                        std::exp(data[index] - maximum_value) / sum;
                }
            }
            values.set(input, tile, lane, std::move(data));
        }
    }

    auto slice = VxmSlice{};
    configure_luts(slice);
    const auto result = VxmSliceCModelAdapter{sram_timing}.run(
        slice, softmax, values);

    // The first denominator wave immediately reuses ReduceMax through SRAM,
    // so exactly one round-trip cycle remains exposed. Later waves have
    // enough independent work to hide the same SRAM latency. The last two
    // phases each need one separate local-scalar register load cycle.
    assert(result.phase_shifts
           == std::vector<std::size_t>({0, 1, 1, 1, 1, 2, 3}));
    assert(result.phase_sram_waits
           == std::vector<std::size_t>({0, 1, 0, 0, 0, 0, 0}));
    assert(result.phase_scalar_load_waits
           == std::vector<std::size_t>({0, 0, 0, 0, 0, 1, 1}));

    const auto maximum_output = std::find_if(
        result.outputs.begin(), result.outputs.end(),
        [maximum](const auto& event) {
            return event.superlane == 0
                && event.value == maximum
                && event.element_index == 0;
        });
    const auto maximum_read = std::find_if(
        result.requests.begin(), result.requests.end(),
        [maximum](const auto& request) {
            return request.superlane == 0
                && request.phase_id == 1
                && request.value == maximum
                && request.element_index == 0;
        });
    assert(maximum_output != result.outputs.end());
    assert(maximum_read != result.requests.end());
    assert(maximum_output->sram_visible_cycle
           == maximum_output->cycle + sram_timing.write_latency);
    assert(maximum_read->issue_cycle
           >= maximum_output->sram_visible_cycle);
    assert(maximum_read->required_cycle
           == maximum_read->issue_cycle + sram_timing.read_latency);

    auto maximum_error = 0.0f;
    for (std::size_t tile = 0; tile < VxmSlice::kTileCount; ++tile) {
        for (std::size_t lane = 0;
             lane < VxmSuperlane::kLaneCount; ++lane) {
            const auto& actual = values.get(output, tile, lane);
            assert(actual.size() == rows * length);
            for (std::size_t index = 0;
                 index < rows * length; ++index) {
                maximum_error = std::max(
                    maximum_error,
                    std::fabs(actual[index]
                              - references[tile][lane][index]));
            }
        }
    }
    assert(maximum_error < 5.0e-3f);

    write_cmodel_summary_report(
        output_path("parallel_lowering_summary.txt"),
        softmax, result);
    write_cmodel_detailed_report(
        output_path("parallel_lowering_detailed.txt"),
        softmax, result);
    write_cmodel_gantt(
        output_path("parallel_lowering_gantt.html"),
        softmax, result);

    std::cout
        << "VXM parallel lowering passed: depth2=8 chains, "
        << "depth4=4 chains, depth8=2 chains; max_abs_error="
        << maximum_error << '\n'
        << "Gantt: "
        << output_path("parallel_lowering_gantt.html").string()
        << '\n';
}

} // namespace vxm_compiler_test::execution
