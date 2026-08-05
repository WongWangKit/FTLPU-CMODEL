#include "ftlpu/vxm/superlane.hpp"
#include "hardware_test_output.hpp"
#include "hardware_timing_report.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using ftlpu::VxmAluOpcode;
using ftlpu::VxmAluPrecision;
using ftlpu::VxmCastTarget;
using ftlpu::VxmChainDepth;
using ftlpu::VxmLane;
using ftlpu::VxmLaneAluInstruction;
using ftlpu::VxmLaneOperand;
using ftlpu::VxmLutEntry;
using ftlpu::VxmSpecialAluOpcode;
using ftlpu::VxmSuperlane;

constexpr std::size_t kElements = 128;
constexpr std::size_t kRmsTokens = 8;
constexpr std::size_t kSoftmaxTokens = 8;
constexpr std::size_t kSwigluTokens = 2;

struct ErrorStats {
    float maximum_relative{0.0f};
    double mean_relative{0.0};
    std::size_t count{0};

    void add(float actual, float expected)
    {
        const auto denominator = std::max(std::fabs(expected), 1.0e-5f);
        const auto relative = std::fabs(actual - expected) / denominator;
        maximum_relative = std::max(maximum_relative, relative);
        mean_relative += relative;
        ++count;
    }

    void finish()
    {
        if (count != 0) mean_relative /= static_cast<double>(count);
    }
};

struct OperatorResult {
    std::string name{};
    std::size_t tokens{0};
    std::size_t elements{0};
    std::size_t cycles{0};
    std::size_t feedback_values{0};
    ErrorStats error{};
    double total_active_utilization{0.0};
    double total_useful_utilization{0.0};
    std::array<double, 3> depth_active_utilization{};
    std::array<double, 3> depth_useful_utilization{};
};

OperatorResult make_result(
    std::string name, std::size_t tokens, std::size_t elements,
    std::size_t cycles, std::size_t feedback_values,
    ErrorStats error, const VxmLane::Statistics& statistics)
{
    auto result = OperatorResult{
        std::move(name), tokens, elements, cycles,
        feedback_values, error};
    result.total_active_utilization = statistics.active_utilization();
    result.total_useful_utilization = statistics.useful_utilization();
    constexpr std::array<VxmChainDepth, 3> kDepths{
        VxmChainDepth::Two,
        VxmChainDepth::Four,
        VxmChainDepth::Eight};
    for (std::size_t index = 0; index < kDepths.size(); ++index) {
        result.depth_active_utilization[index] =
            statistics.for_depth(kDepths[index]).active_utilization();
        result.depth_useful_utilization[index] =
            statistics.for_depth(kDepths[index]).useful_utilization();
    }
    return result;
}

template <typename Fn>
std::vector<VxmLutEntry> make_table(
    float input_min, float segment_width, std::size_t count, Fn fn)
{
    auto entries = std::vector<VxmLutEntry>{};
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto x0 =
            input_min + static_cast<float>(index) * segment_width;
        const auto y0 = fn(x0);
        entries.push_back(VxmLutEntry::from_float(
            (fn(x0 + segment_width) - y0) / segment_width, y0));
    }
    return entries;
}

void configure_luts(VxmSuperlane& superlane)
{
    constexpr std::size_t kEntries = 256;
    constexpr float kLn2 = 0.6931471805599453f;
    superlane.configure_special_lut(
        VxmSpecialAluOpcode::Exp,
        {-kLn2 / 2.0f, kLn2 / static_cast<float>(kEntries)},
        make_table(
            -kLn2 / 2.0f, kLn2 / static_cast<float>(kEntries),
            kEntries, [](float x) { return std::exp(x); }));
    superlane.configure_special_lut(
        VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f / static_cast<float>(kEntries)},
        make_table(
            1.0f, 1.0f / static_cast<float>(kEntries), kEntries,
            [](float x) { return 1.0f / x; }));
    superlane.configure_special_lut(
        VxmSpecialAluOpcode::Rsqrt,
        {1.0f, 3.0f / static_cast<float>(kEntries)},
        make_table(
            1.0f, 3.0f / static_cast<float>(kEntries), kEntries,
            [](float x) { return 1.0f / std::sqrt(x); }));
}

void put_fp16(
    VxmSuperlane::StreamMatrix& streams, std::size_t physical_head,
    bool rhs_port, float value)
{
    const auto bytes = VxmLane::pack_float16(value);
    const auto base =
        VxmLane::fixed_input_group_for_stage(physical_head, rhs_port)
        * VxmLane::kStreamGroupBytes;
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        for (std::size_t byte = 0;
             byte < VxmLane::kStreamGroupBytes; ++byte) {
            streams[lane][base + byte] = bytes[byte];
        }
    }
}

float output_float(const VxmSuperlane::Output& output)
{
    return VxmLane::unpack_float32(output.byte_values[0]);
}

void record_tick(
    VxmSuperlane& superlane,
    std::vector<vxm_hardware_test::TimingCycle>& trace,
    std::string event)
{
    superlane.tick();
    trace.push_back(vxm_hardware_test::capture_cycle(
        superlane.cycle() - 1, superlane.lane(0),
        superlane.outputs().size(), std::move(event)));
}

VxmLaneAluInstruction basic(
    VxmAluOpcode opcode, VxmLaneOperand lhs,
    VxmLaneOperand rhs = VxmLaneOperand::Imm(0.0f),
    std::size_t repeat = 1)
{
    auto instruction = VxmLaneAluInstruction{opcode, lhs, rhs};
    instruction.repeat_count = repeat;
    return instruction;
}

VxmLaneAluInstruction special(
    VxmSpecialAluOpcode opcode, VxmLaneOperand lhs,
    std::size_t repeat = 1)
{
    auto instruction = VxmLaneAluInstruction{opcode, lhs};
    instruction.repeat_count = repeat;
    return instruction;
}

VxmLaneAluInstruction accumulator(
    VxmAluOpcode opcode, bool reset, bool emit,
    std::size_t repeat, std::optional<std::size_t> output_stream = {})
{
    auto instruction = basic(
        opcode, VxmLaneOperand::Previous(), VxmLaneOperand::Acc(), repeat);
    instruction.precision = VxmAluPrecision::Float32;
    instruction.accumulator_reset = reset;
    instruction.accumulator_write = true;
    instruction.accumulator_emit = emit;
    instruction.output_stream = output_stream;
    instruction.output_type = VxmCastTarget::Float32;
    return instruction;
}

template <std::size_t Tokens>
using Tensor = std::array<std::array<float, kElements>, Tokens>;

template <std::size_t Tokens>
Tensor<Tokens> make_input(float offset, float token_scale)
{
    auto result = Tensor<Tokens>{};
    for (std::size_t token = 0; token < Tokens; ++token) {
        for (std::size_t element = 0; element < kElements; ++element) {
            const auto wave =
                std::sin(static_cast<float>(element) * 0.071f
                         + static_cast<float>(token) * 0.37f);
            result[token][element] = VxmLane::unpack_float16(
                VxmLane::pack_float16(
                    offset + token_scale * static_cast<float>(token) + wave));
        }
    }
    return result;
}

template <std::size_t Tokens>
void enqueue_depth2_reduction(
    VxmSuperlane& superlane, VxmAluOpcode head_opcode,
    VxmAluOpcode reduce_opcode, bool square_inputs,
    bool emit_to_stream)
{
    for (std::size_t stage = 0; stage < 8; ++stage) {
        if (stage % 2 == 0) {
            const auto rhs = square_inputs
                ? VxmLaneOperand::StreamFloat16()
                : VxmLaneOperand::Imm(0.0f);
            auto head = basic(
                head_opcode, VxmLaneOperand::StreamFloat16(),
                rhs, kElements);
            if (head_opcode == VxmAluOpcode::Multiply) {
                head.precision = VxmAluPrecision::Float32;
            }
            superlane.enqueue_instruction_for_depth(
                VxmChainDepth::Two, stage, head);
            continue;
        }
        superlane.enqueue_instruction_for_depth(
            VxmChainDepth::Two, stage,
            accumulator(reduce_opcode, true, false, 1));
        superlane.enqueue_instruction_for_depth(
            VxmChainDepth::Two, stage,
            accumulator(reduce_opcode, false, false, kElements - 2));
        const auto output = emit_to_stream
            ? std::optional<std::size_t>{
                VxmLane::fixed_output_stream_for_block(stage / 2)}
            : std::optional<std::size_t>{};
        superlane.enqueue_instruction_for_depth(
            VxmChainDepth::Two, stage,
            accumulator(reduce_opcode, false, true, 1, output));
    }
}

template <std::size_t Tokens>
void feed_depth2(
    VxmSuperlane& superlane, const Tensor<Tokens>& input,
    bool duplicate_rhs,
    std::vector<vxm_hardware_test::TimingCycle>& trace,
    std::string_view event)
{
    static_assert(Tokens == 8);
    constexpr std::array<std::size_t, 8> kHeads{
        0, 2, 4, 6, 8, 10, 12, 14};
    for (std::size_t element = 0; element < kElements; ++element) {
        auto streams = VxmSuperlane::StreamMatrix{};
        for (std::size_t token = 0; token < Tokens; ++token) {
            put_fp16(streams, kHeads[token], false, input[token][element]);
            if (duplicate_rhs) {
                put_fp16(streams, kHeads[token], true, input[token][element]);
            }
        }
        superlane.set_stream_inputs(streams);
        record_tick(superlane, trace, std::string{event});
    }
}

template <std::size_t Tokens>
void collect_block_scalar_outputs(
    const VxmSuperlane& superlane,
    std::array<float, Tokens>& values,
    std::array<std::size_t, 8>& stream_positions)
{
    for (const auto& output : superlane.outputs()) {
        const auto block = output.stream / VxmLane::kStreamGroupBytes;
        assert(block < stream_positions.size());
        const auto position = block;
        assert(position < Tokens);
        assert(stream_positions[block]++ == 0);
        values[position] = output_float(output);
    }
}

OperatorResult run_rmsnorm()
{
    constexpr float kEpsilon = 1.0e-5f;
    auto x = make_input<kRmsTokens>(1.25f, 0.08f);
    auto gamma = make_input<kRmsTokens>(0.85f, 0.01f);
    auto expected = Tensor<kRmsTokens>{};
    for (std::size_t token = 0; token < kRmsTokens; ++token) {
        double square_sum = 0.0;
        for (const auto value : x[token]) square_sum += value * value;
        const auto inverse = static_cast<float>(
            1.0 / std::sqrt(square_sum / kElements + kEpsilon));
        for (std::size_t element = 0; element < kElements; ++element) {
            expected[token][element] =
                x[token][element] * gamma[token][element] * inverse;
        }
    }

    auto superlane = VxmSuperlane{};
    configure_luts(superlane);
    superlane.set_chain_depth(VxmChainDepth::Two);
    auto trace = std::vector<vxm_hardware_test::TimingCycle>{};
    enqueue_depth2_reduction<kRmsTokens>(
        superlane, VxmAluOpcode::Multiply, VxmAluOpcode::Add,
        true, false);

    // These depth-4 configurations are prefetched while the square-sum
    // phase runs. Eight old C1/C3/C5/C7 results become two inputs per new
    // depth-4 chain; no Stream Register round trip is used.
    for (const auto stage : {std::size_t{0}, std::size_t{4}}) {
        auto scale = basic(
            VxmAluOpcode::Multiply, VxmLaneOperand::Feedback(),
            VxmLaneOperand::Imm(1.0f / static_cast<float>(kElements)), 2);
        scale.precision = VxmAluPrecision::Float32;
        superlane.enqueue_instruction_for_depth(
            VxmChainDepth::Four, stage, scale);
    }
    for (const auto stage : {std::size_t{2}, std::size_t{6}}) {
        superlane.enqueue_instruction_for_depth(
            VxmChainDepth::Four, stage,
            basic(VxmAluOpcode::Bypass, VxmLaneOperand::Previous(),
                  VxmLaneOperand::Imm(0.0f), 2));
    }

    record_tick(superlane, trace, "RMSNorm config decode");

    // Reduction tails originally used all three FIFO entries. The initial
    // decode frees one entry, allowing the next phase to be prefetched.
    for (const auto stage : {std::size_t{1}, std::size_t{5}}) {
        auto add_epsilon = basic(
            VxmAluOpcode::Add, VxmLaneOperand::Previous(),
            VxmLaneOperand::Imm(kEpsilon), 2);
        add_epsilon.precision = VxmAluPrecision::Float32;
        superlane.enqueue_instruction_for_depth(
            VxmChainDepth::Four, stage, add_epsilon);
    }
    for (const auto stage : {std::size_t{3}, std::size_t{7}}) {
        auto rsqrt = special(
            VxmSpecialAluOpcode::Rsqrt,
            VxmLaneOperand::Previous(), 2);
        rsqrt.output_stream =
            VxmLane::fixed_output_stream_for_block(stage / 2);
        rsqrt.output_type = VxmCastTarget::Float32;
        superlane.enqueue_instruction_for_depth(
            VxmChainDepth::Four, stage, rsqrt);
    }

    feed_depth2(
        superlane, x, true, trace, "RMSNorm square-sum input");
    record_tick(superlane, trace, "RMSNorm square-sum drain");
    superlane.request_chain_depth_transition(VxmChainDepth::Four);
    record_tick(superlane, trace, "RMSNorm 2-to-4 feedback transition");
    const auto feedback_values =
        superlane.lane(0).last_feedback_capture_count();
    assert(feedback_values == kRmsTokens);

    auto inverse_rms = std::array<float, kRmsTokens>{};
    auto scalar_positions = std::array<std::size_t, 8>{};
    while (!superlane.idle()) {
        record_tick(superlane, trace, "RMSNorm rsqrt pipeline");
        for (const auto& output : superlane.outputs()) {
            const auto block =
                output.stream / VxmLane::kStreamGroupBytes;
            assert(block == 1 || block == 3
                   || block == 5 || block == 7);
            const auto chain = (block - 1) / 2;
            const auto position =
                chain * 2 + scalar_positions[block]++;
            assert(position < inverse_rms.size());
            inverse_rms[position] = output_float(output);
        }
    }
    for (const auto block : {std::size_t{1}, std::size_t{3},
                             std::size_t{5}, std::size_t{7}}) {
        assert(scalar_positions[block] == 2);
    }

    superlane.set_chain_depth(VxmChainDepth::Two);
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        for (std::size_t token = 0; token < kRmsTokens; ++token) {
            superlane.lane(lane).load_local_scalar(
                token * 2 + 1, inverse_rms[token]);
        }
    }
    for (std::size_t stage = 0; stage < 8; ++stage) {
        if (stage % 2 == 0) {
            auto multiply_gamma = basic(
                VxmAluOpcode::Multiply,
                VxmLaneOperand::StreamFloat16(),
                VxmLaneOperand::StreamFloat16(), kElements);
            multiply_gamma.precision = VxmAluPrecision::Float32;
            superlane.enqueue_instruction(stage, multiply_gamma);
        } else {
            auto normalize = basic(
                VxmAluOpcode::Multiply, VxmLaneOperand::Previous(),
                VxmLaneOperand::Acc(), kElements);
            normalize.precision = VxmAluPrecision::Float32;
            normalize.output_stream =
                VxmLane::fixed_output_stream_for_block(stage / 2);
            normalize.output_type = VxmCastTarget::Float32;
            superlane.enqueue_instruction(stage, normalize);
        }
    }
    record_tick(superlane, trace, "RMSNorm output config decode");

    auto actual = Tensor<kRmsTokens>{};
    auto positions = std::array<std::size_t, kRmsTokens>{};
    constexpr std::array<std::size_t, kRmsTokens> kHeads{
        0, 2, 4, 6, 8, 10, 12, 14};
    for (std::size_t element = 0; element < kElements; ++element) {
        auto streams = VxmSuperlane::StreamMatrix{};
        for (std::size_t token = 0; token < kRmsTokens; ++token) {
            put_fp16(streams, kHeads[token], false, x[token][element]);
            put_fp16(streams, kHeads[token], true, gamma[token][element]);
        }
        superlane.set_stream_inputs(streams);
        record_tick(superlane, trace, "RMSNorm normalize input");
        for (const auto& output : superlane.outputs()) {
            const auto token =
                output.stream / VxmLane::kStreamGroupBytes;
            actual[token][positions[token]++] = output_float(output);
        }
    }
    while (!superlane.idle()) {
        record_tick(superlane, trace, "RMSNorm output drain");
        for (const auto& output : superlane.outputs()) {
            const auto token =
                output.stream / VxmLane::kStreamGroupBytes;
            actual[token][positions[token]++] = output_float(output);
        }
    }
    for (const auto position : positions) assert(position == kElements);

    auto error = ErrorStats{};
    for (std::size_t token = 0; token < kRmsTokens; ++token) {
        for (std::size_t element = 0; element < kElements; ++element) {
            error.add(actual[token][element], expected[token][element]);
        }
    }
    error.finish();
    assert(error.maximum_relative < 0.03f);
    vxm_hardware_test::write_timing_reports(
        "rmsnorm_operator", "RMSNorm hardware throughput", trace);
    return make_result(
        "RMSNorm", kRmsTokens, kElements, superlane.cycle(),
        feedback_values, error, superlane.lane(0).statistics());
}

void enqueue_softmax_sum_and_reciprocal(VxmSuperlane& superlane)
{
    for (std::size_t stage = 0; stage < 8; ++stage) {
        switch (stage % 4) {
        case 0: {
            auto subtract = basic(
                VxmAluOpcode::Subtract,
                VxmLaneOperand::StreamFloat16(),
                VxmLaneOperand::StreamFloat16(), kElements);
            subtract.precision = VxmAluPrecision::Float32;
            superlane.enqueue_instruction(stage, subtract);
            superlane.enqueue_instruction(
                stage, basic(
                    VxmAluOpcode::Bypass, VxmLaneOperand::Feedback()));
            break;
        }
        case 1:
            superlane.enqueue_instruction(
                stage, special(
                    VxmSpecialAluOpcode::Exp,
                    VxmLaneOperand::Previous(), kElements));
            superlane.enqueue_instruction(
                stage, basic(
                    VxmAluOpcode::Bypass, VxmLaneOperand::Previous()));
            break;
        case 2:
            superlane.enqueue_instruction(
                stage, basic(
                    VxmAluOpcode::Bypass, VxmLaneOperand::Previous(),
                    VxmLaneOperand::Imm(0.0f), kElements));
            superlane.enqueue_instruction(
                stage, basic(
                    VxmAluOpcode::Bypass, VxmLaneOperand::Previous()));
            break;
        case 3:
            superlane.enqueue_instruction(
                stage, accumulator(VxmAluOpcode::Add, true, false, 1));
            superlane.enqueue_instruction(
                stage, accumulator(
                    VxmAluOpcode::Add, false, false, kElements - 2));
            superlane.enqueue_instruction(
                stage, accumulator(VxmAluOpcode::Add, false, true, 1));
            break;
        }
    }
}

void enqueue_softmax_reciprocal_tails(VxmSuperlane& superlane)
{
    for (const auto stage : {std::size_t{3}, std::size_t{7}}) {
        auto reciprocal = special(
            VxmSpecialAluOpcode::Reciprocal,
            VxmLaneOperand::Previous());
        reciprocal.output_stream =
            VxmLane::fixed_output_stream_for_block(stage / 2);
        reciprocal.output_type = VxmCastTarget::Float32;
        superlane.enqueue_instruction(stage, reciprocal);
    }
}

OperatorResult run_softmax()
{
    auto x = make_input<kSoftmaxTokens>(-0.4f, 0.11f);
    auto expected = Tensor<kSoftmaxTokens>{};
    for (std::size_t token = 0; token < kSoftmaxTokens; ++token) {
        const auto maximum =
            *std::max_element(x[token].begin(), x[token].end());
        double sum = 0.0;
        for (const auto value : x[token]) sum += std::exp(value - maximum);
        for (std::size_t element = 0; element < kElements; ++element) {
            expected[token][element] =
                static_cast<float>(std::exp(x[token][element] - maximum) / sum);
        }
    }

    auto superlane = VxmSuperlane{};
    configure_luts(superlane);
    auto trace = std::vector<vxm_hardware_test::TimingCycle>{};
    superlane.set_chain_depth(VxmChainDepth::Two);
    enqueue_depth2_reduction<kSoftmaxTokens>(
        superlane, VxmAluOpcode::Bypass, VxmAluOpcode::Max,
        false, true);
    record_tick(superlane, trace, "Softmax max config decode");
    auto maximum = std::array<float, kSoftmaxTokens>{};
    auto maximum_positions = std::array<std::size_t, 8>{};
    feed_depth2(
        superlane, x, false, trace, "Softmax max input");
    while (!superlane.idle()) {
        record_tick(superlane, trace, "Softmax max drain");
        collect_block_scalar_outputs(
            superlane, maximum, maximum_positions);
    }
    for (const auto count : maximum_positions) assert(count == 1);

    auto inverse_sum = std::array<float, kSoftmaxTokens>{};
    std::size_t feedback_values = 0;
    constexpr std::array<std::size_t, 4> kHeads{0, 4, 8, 12};
    for (std::size_t wave = 0; wave < 2; ++wave) {
        superlane.set_chain_depth(VxmChainDepth::Four);
        enqueue_softmax_sum_and_reciprocal(superlane);
        record_tick(superlane, trace, "Softmax sum config decode");
        enqueue_softmax_reciprocal_tails(superlane);

        for (std::size_t element = 0; element < kElements; ++element) {
            auto streams = VxmSuperlane::StreamMatrix{};
            for (std::size_t chain = 0; chain < kHeads.size(); ++chain) {
                const auto token = wave * kHeads.size() + chain;
                put_fp16(
                    streams, kHeads[chain], false, x[token][element]);
                put_fp16(
                    streams, kHeads[chain], true, maximum[token]);
            }
            superlane.set_stream_inputs(streams);
            record_tick(superlane, trace, "Softmax exp-sum input");
            feedback_values +=
                superlane.lane(0).last_feedback_capture_count();
        }

        auto wave_positions = std::array<std::size_t, 8>{};
        while (!superlane.idle()) {
            record_tick(superlane, trace, "Softmax sum feedback/reciprocal");
            feedback_values +=
                superlane.lane(0).last_feedback_capture_count();
            for (const auto& output : superlane.outputs()) {
                const auto block =
                    output.stream / VxmLane::kStreamGroupBytes;
                assert(block == 1 || block == 3
                       || block == 5 || block == 7);
                const auto chain = (block - 1) / 2;
                inverse_sum[wave * 4 + chain] = output_float(output);
                ++wave_positions[chain];
            }
        }
        for (std::size_t chain = 0; chain < 4; ++chain) {
            assert(wave_positions[chain] == 1);
        }
    }
    assert(feedback_values == kSoftmaxTokens);

    auto actual = Tensor<kSoftmaxTokens>{};
    auto positions = std::array<std::size_t, kSoftmaxTokens>{};
    for (std::size_t wave = 0; wave < 2; ++wave) {
        superlane.set_chain_depth(VxmChainDepth::Four);
        for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
            for (std::size_t chain = 0; chain < 4; ++chain) {
                const auto token = wave * 4 + chain;
                superlane.lane(lane).load_local_scalar(
                    kHeads[chain] + 3, inverse_sum[token]);
            }
        }
        for (std::size_t stage = 0; stage < 8; ++stage) {
            switch (stage % 4) {
            case 0: {
                auto subtract = basic(
                    VxmAluOpcode::Subtract,
                    VxmLaneOperand::StreamFloat16(),
                    VxmLaneOperand::StreamFloat16(), kElements);
                subtract.precision = VxmAluPrecision::Float32;
                superlane.enqueue_instruction(stage, subtract);
                break;
            }
            case 1:
                superlane.enqueue_instruction(
                    stage, special(
                        VxmSpecialAluOpcode::Exp,
                        VxmLaneOperand::Previous(), kElements));
                break;
            case 2:
                superlane.enqueue_instruction(
                    stage, basic(
                        VxmAluOpcode::Bypass,
                        VxmLaneOperand::Previous(),
                        VxmLaneOperand::Imm(0.0f), kElements));
                break;
            case 3: {
                auto multiply = basic(
                    VxmAluOpcode::Multiply,
                    VxmLaneOperand::Previous(),
                    VxmLaneOperand::Acc(), kElements);
                multiply.precision = VxmAluPrecision::Float32;
                multiply.output_stream =
                    VxmLane::fixed_output_stream_for_block(stage / 2);
                multiply.output_type = VxmCastTarget::Float32;
                superlane.enqueue_instruction(stage, multiply);
                break;
            }
            }
        }
        record_tick(superlane, trace, "Softmax output config decode");
        for (std::size_t element = 0; element < kElements; ++element) {
            auto streams = VxmSuperlane::StreamMatrix{};
            for (std::size_t chain = 0; chain < 4; ++chain) {
                const auto token = wave * 4 + chain;
                put_fp16(
                    streams, kHeads[chain], false, x[token][element]);
                put_fp16(
                    streams, kHeads[chain], true, maximum[token]);
            }
            superlane.set_stream_inputs(streams);
            record_tick(superlane, trace, "Softmax normalize input");
            for (const auto& output : superlane.outputs()) {
                const auto block =
                    output.stream / VxmLane::kStreamGroupBytes;
                const auto chain = (block - 1) / 2;
                const auto token = wave * 4 + chain;
                actual[token][positions[token]++] = output_float(output);
            }
        }
        while (!superlane.idle()) {
            record_tick(superlane, trace, "Softmax output drain");
            for (const auto& output : superlane.outputs()) {
                const auto block =
                    output.stream / VxmLane::kStreamGroupBytes;
                const auto chain = (block - 1) / 2;
                const auto token = wave * 4 + chain;
                actual[token][positions[token]++] = output_float(output);
            }
        }
    }
    for (const auto position : positions) assert(position == kElements);

    auto error = ErrorStats{};
    for (std::size_t token = 0; token < kSoftmaxTokens; ++token) {
        for (std::size_t element = 0; element < kElements; ++element) {
            error.add(actual[token][element], expected[token][element]);
        }
    }
    error.finish();
    assert(error.maximum_relative < 0.03f);
    vxm_hardware_test::write_timing_reports(
        "softmax_operator", "Softmax hardware throughput", trace);
    return make_result(
        "Softmax", kSoftmaxTokens, kElements, superlane.cycle(),
        feedback_values, error, superlane.lane(0).statistics());
}

OperatorResult run_swiglu()
{
    auto gate = make_input<kSwigluTokens>(-0.7f, 0.24f);
    auto up = make_input<kSwigluTokens>(0.8f, 0.07f);
    auto expected = Tensor<kSwigluTokens>{};
    for (std::size_t token = 0; token < kSwigluTokens; ++token) {
        for (std::size_t element = 0; element < kElements; ++element) {
            const auto value = gate[token][element];
            expected[token][element] =
                value / (1.0f + std::exp(-value)) * up[token][element];
        }
    }

    auto superlane = VxmSuperlane{};
    configure_luts(superlane);
    superlane.set_chain_depth(VxmChainDepth::Eight);
    auto trace = std::vector<vxm_hardware_test::TimingCycle>{};
    for (std::size_t stage = 0; stage < 8; ++stage) {
        auto instruction = VxmLaneAluInstruction{};
        switch (stage) {
        case 0:
            instruction = basic(
                VxmAluOpcode::Negate,
                VxmLaneOperand::StreamFloat16(),
                VxmLaneOperand::StreamFloat16(), kElements);
            break;
        case 1:
            instruction = special(
                VxmSpecialAluOpcode::Exp,
                VxmLaneOperand::Previous(), kElements);
            break;
        case 2:
            instruction = basic(
                VxmAluOpcode::Add, VxmLaneOperand::Previous(),
                VxmLaneOperand::Imm(1.0f), kElements);
            break;
        case 3:
            instruction = special(
                VxmSpecialAluOpcode::Reciprocal,
                VxmLaneOperand::Previous(), kElements);
            break;
        case 4:
            instruction = basic(
                VxmAluOpcode::Multiply, VxmLaneOperand::Previous(),
                VxmLaneOperand::Original(), kElements);
            break;
        case 5:
            instruction = basic(
                VxmAluOpcode::Multiply, VxmLaneOperand::Previous(),
                VxmLaneOperand::Aux(), kElements);
            break;
        case 6:
            instruction = basic(
                VxmAluOpcode::Bypass, VxmLaneOperand::Previous(),
                VxmLaneOperand::Imm(0.0f), kElements);
            break;
        case 7:
            instruction = basic(
                VxmAluOpcode::Bypass, VxmLaneOperand::Previous(),
                VxmLaneOperand::Imm(0.0f), kElements);
            instruction.output_stream =
                VxmLane::fixed_output_stream_for_block(stage / 2);
            instruction.output_type = VxmCastTarget::Float32;
            break;
        }
        superlane.enqueue_instruction(stage, instruction);
    }
    record_tick(superlane, trace, "SwiGLU config decode");

    auto actual = Tensor<kSwigluTokens>{};
    auto positions = std::array<std::size_t, kSwigluTokens>{};
    constexpr std::array<std::size_t, 2> kHeads{0, 8};
    for (std::size_t element = 0; element < kElements; ++element) {
        auto streams = VxmSuperlane::StreamMatrix{};
        for (std::size_t token = 0; token < kSwigluTokens; ++token) {
            put_fp16(streams, kHeads[token], false, gate[token][element]);
            put_fp16(streams, kHeads[token], true, up[token][element]);
        }
        superlane.set_stream_inputs(streams);
        record_tick(superlane, trace, "SwiGLU input");
        for (const auto& output : superlane.outputs()) {
            const auto token = output.stream < 8 ? 0U : 1U;
            actual[token][positions[token]++] = output_float(output);
        }
    }
    while (!superlane.idle()) {
        record_tick(superlane, trace, "SwiGLU output drain");
        for (const auto& output : superlane.outputs()) {
            const auto token = output.stream < 8 ? 0U : 1U;
            actual[token][positions[token]++] = output_float(output);
        }
    }
    for (const auto position : positions) assert(position == kElements);

    auto error = ErrorStats{};
    for (std::size_t token = 0; token < kSwigluTokens; ++token) {
        for (std::size_t element = 0; element < kElements; ++element) {
            error.add(actual[token][element], expected[token][element]);
        }
    }
    error.finish();
    assert(error.maximum_relative < 0.04f);
    vxm_hardware_test::write_timing_reports(
        "swiglu_operator", "SwiGLU hardware throughput", trace);
    return make_result(
        "SwiGLU", kSwigluTokens, kElements, superlane.cycle(),
        0, error, superlane.lane(0).statistics());
}

void write_summary(const std::vector<OperatorResult>& results)
{
    const auto path =
        vxm_hardware_test::results_directory()
        / "operator_throughput_results.txt";
    auto file = std::ofstream{path, std::ios::trunc};
    assert(file);
    file << "VXM operator hardware throughput test\n"
         << "mapping=one token per physical chain\n"
         << "feedback=hardware tail-to-head path (not Stream reinjection)\n\n"
         << std::fixed << std::setprecision(6);
    for (const auto& result : results) {
        file << result.name
             << ": PASS"
             << " tokens=" << result.tokens
             << " elements_per_token=" << result.elements
             << " cycles=" << result.cycles
             << " feedback_values=" << result.feedback_values
             << " max_relative_error="
             << result.error.maximum_relative
             << " mean_relative_error="
             << result.error.mean_relative
             << " active_utilization="
             << result.total_active_utilization
             << " useful_utilization="
             << result.total_useful_utilization << '\n'
             << "  depth2_active="
             << result.depth_active_utilization[0]
             << " depth2_useful="
             << result.depth_useful_utilization[0]
             << " depth4_active="
             << result.depth_active_utilization[1]
             << " depth4_useful="
             << result.depth_useful_utilization[1]
             << " depth8_active="
             << result.depth_active_utilization[2]
             << " depth8_useful="
             << result.depth_useful_utilization[2] << '\n';
    }
}

} // namespace

int main()
{
    auto results = std::vector<OperatorResult>{};
    results.push_back(run_rmsnorm());
    results.push_back(run_softmax());
    results.push_back(run_swiglu());
    write_summary(results);
    vxm_hardware_test::write_pass_result(
        "operator_throughput_test_results.txt",
        "operator_throughput_test");
    return 0;
}
