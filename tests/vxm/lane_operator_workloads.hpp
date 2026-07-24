#pragma once

#include "lane_test_driver.hpp"
#include "operator_error_metrics.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kNormTokens = 128;
constexpr std::size_t kSoftmaxTokens = 128;
constexpr std::size_t kSwigluTokens = 128;
constexpr std::size_t kVectorLength = 128;

template <typename Fn>
std::vector<ftlpu::VxmLutEntry> table(float min, float width,
                                      std::size_t count, Fn fn)
{
    auto result = std::vector<ftlpu::VxmLutEntry>{};
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto x0 = min + static_cast<float>(i) * width;
        const auto y0 = fn(x0);
        result.push_back(ftlpu::VxmLutEntry::from_float(
            (fn(x0 + width) - y0) / width, y0));
    }
    return result;
}

void configure_luts(ftlpu::VxmLane& lane)
{
    constexpr std::size_t exp_entries = 256;
    constexpr std::size_t reciprocal_entries = 256;
    constexpr std::size_t rsqrt_entries = 512;
    constexpr float ln2 = 0.6931471805599453f;

    lane.special_alu().configure_lut(ftlpu::VxmSpecialAluOpcode::Exp,
        {-ln2 / 2.0f, ln2 / exp_entries},
        table(-ln2 / 2.0f, ln2 / exp_entries, exp_entries,
            [](float x) { return std::exp(x); }));
    lane.special_alu().configure_lut(ftlpu::VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f / reciprocal_entries},
        table(1.0f, 1.0f / reciprocal_entries, reciprocal_entries,
            [](float x) { return 1.0f / x; }));
    lane.special_alu().configure_lut(ftlpu::VxmSpecialAluOpcode::Rsqrt,
        {1.0f, 3.0f / rsqrt_entries},
        table(1.0f, 3.0f / rsqrt_entries, rsqrt_entries,
            [](float x) { return 1.0f / std::sqrt(x); }));
}

void put_float(ftlpu::VxmLane::StreamBytes& streams,
               std::size_t group, float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto base = group * ftlpu::VxmLane::kStreamGroupBytes;
    for (std::size_t byte = 0; byte < 4; ++byte) {
        streams[base + byte] = static_cast<std::uint8_t>(bits >> (8 * byte));
    }
}

float output_float(const ftlpu::VxmLane::Output& output)
{
    return ftlpu::VxmLane::unpack_float32(output.bytes);
}

ftlpu::VxmLaneAluInstruction basic(
    ftlpu::VxmAluOpcode opcode,
    ftlpu::VxmLaneOperand lhs = ftlpu::VxmLaneOperand::Previous(),
    ftlpu::VxmLaneOperand rhs = ftlpu::VxmLaneOperand::Imm(0.0f))
{
    auto instruction = ftlpu::VxmLaneAluInstruction{opcode, lhs, rhs};
    instruction.precision = ftlpu::VxmAluPrecision::Float32;
    instruction.output_type = ftlpu::VxmCastTarget::Float32;
    return instruction;
}

ftlpu::VxmLaneAluInstruction special(ftlpu::VxmSpecialAluOpcode opcode)
{
    auto instruction = ftlpu::VxmLaneAluInstruction{
        opcode, ftlpu::VxmLaneOperand::Previous()};
    instruction.output_type = ftlpu::VxmCastTarget::Float32;
    return instruction;
}

ftlpu::VxmLaneAluInstruction repeated(
    ftlpu::VxmLaneAluInstruction instruction, std::size_t count)
{
    instruction.repeat_count = count;
    return instruction;
}

bool near(float actual, float expected, float absolute, float relative)
{
    return std::fabs(actual - expected)
        <= absolute + relative * std::fabs(expected);
}

struct WorkloadResult {
    ftlpu::VxmLane::Statistics statistics{};
    VxmOperatorErrorMetrics error{};
    double host_milliseconds{0.0};
};

void validate_statistics(const ftlpu::VxmLane::Statistics& statistics)
{
    assert(statistics.timeline.size() == statistics.cycles);
    std::uint64_t active = 0;
    std::uint64_t useful = 0;
    for (const auto& cycle : statistics.timeline) {
        assert(cycle.useful_slots() <= cycle.active_slots());
        active += cycle.active_slots();
        useful += cycle.useful_slots();
    }
    assert(active == statistics.executed_slots);
    assert(useful == statistics.useful_slots);

    std::uint64_t grouped_cycles = 0;
    std::uint64_t grouped_active = 0;
    std::uint64_t grouped_useful = 0;
    for (const auto depth : {ftlpu::VxmChainDepth::Two,
                             ftlpu::VxmChainDepth::Four,
                             ftlpu::VxmChainDepth::Eight}) {
        const auto& group = statistics.for_depth(depth);
        grouped_cycles += group.cycles;
        grouped_active += group.executed_slots;
        grouped_useful += group.useful_slots;
    }
    assert(grouped_cycles == statistics.cycles);
    assert(grouped_active == statistics.executed_slots);
    assert(grouped_useful == statistics.useful_slots);
}

void print_statistics(std::ostream& os, const std::string& name,
                      const WorkloadResult& result)
{
    const auto print_group = [&os](const char* label,
                                   const ftlpu::VxmLane::Utilization& group) {
        os << "  " << std::setw(5) << label
           << " cycles=" << std::setw(5) << group.cycles
           << " active=" << std::fixed << std::setprecision(2)
           << 100.0 * group.active_utilization() << "%"
           << " useful=" << 100.0 * group.useful_utilization() << "%"
           << " peak_active=" << 100.0 * group.peak_active_utilization() << "%"
           << " peak_useful=" << 100.0 * group.peak_useful_utilization() << "%\n";
    };

    os << name << " host_runtime=" << std::fixed << std::setprecision(3)
       << result.host_milliseconds << " ms"
       << " cycle_records=" << result.statistics.timeline.size() << '\n';
    print_group("2", result.statistics.for_depth(ftlpu::VxmChainDepth::Two));
    print_group("4", result.statistics.for_depth(ftlpu::VxmChainDepth::Four));
    print_group("8", result.statistics.for_depth(ftlpu::VxmChainDepth::Eight));

    print_group("Total", result.statistics.total());
    print_vxm_operator_error(os, result.error);
}

void print_cycle_trace(std::ostream& os, const std::string& name,
                       const ftlpu::VxmLane::Statistics& statistics)
{
    os << "\n[" << name << " cycle trace]\n"
       << "legend: U=useful B=bypass S=stalled .=idle\n"
       << "cycle depth ALU0..ALU7 active useful\n";
    for (const auto& activity : statistics.timeline) {
        os << activity.cycle << ' '
           << static_cast<std::size_t>(activity.chain_depth) << ' ';
        for (std::size_t stage = 0; stage < ftlpu::VxmLane::kAluCount; ++stage) {
            const auto state = activity.states[stage];
            os << (state == ftlpu::VxmLaneAluTraceState::Executed
                ? (activity.useful[stage] ? 'U' : 'B')
                : state == ftlpu::VxmLaneAluTraceState::Stalled ? 'S' : '.');
        }
        os << ' ' << activity.active_slots()
           << ' ' << activity.useful_slots() << '\n';
    }
}

template <typename Lane, typename Consume>
void drain(Lane& lane, Consume consume)
{
    while (!lane.idle()) {
        lane.tick();
        consume();
    }
}

WorkloadResult test_rmsnorm()
{
    constexpr float epsilon = 1.0e-5f;
    auto lane = VxmLaneTestDriver{};
    configure_luts(lane);
    auto x = std::vector<std::vector<float>>(
        kNormTokens, std::vector<float>(kVectorLength));
    auto gamma = std::vector<float>(kVectorLength);
    auto actual = std::vector<std::vector<float>>(kNormTokens);
    for (std::size_t i = 0; i < kVectorLength; ++i) {
        gamma[i] = 0.75f + 0.01f * static_cast<float>(i % 17);
    }
    auto error = VxmOperatorErrorMetrics{};
    for (std::size_t token = 0; token < kNormTokens; ++token) {
        for (std::size_t i = 0; i < kVectorLength; ++i) {
            x[token][i] = 1.5f * std::sin(
                0.017f * static_cast<float>((token + 1) * (i + 3)))
                + 0.02f * static_cast<float>(token % 5);
        }
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t batch = 0; batch < kNormTokens; batch += 4) {
        auto sums = std::array<float, 4>{};
        auto inv_rms = std::array<float, 4>{};

        // Phase 1: four independent 2-stage square/accumulate chains.
        lane.set_chain_depth(ftlpu::VxmChainDepth::Two);
        for (std::size_t row = 0; row < 4; ++row) {
            const auto head = row * 2;
            const auto tail = head + 1;
            lane.enqueue_instruction(head, repeated(
                basic(ftlpu::VxmAluOpcode::Multiply,
                    ftlpu::VxmLaneOperand::StreamFloat32(),
                    ftlpu::VxmLaneOperand::StreamFloat32()), kVectorLength));
            auto first = basic(ftlpu::VxmAluOpcode::Add,
                ftlpu::VxmLaneOperand::Previous(), ftlpu::VxmLaneOperand::Acc());
            first.accumulator_reset = true;
            first.accumulator_write = true;
            first.accumulator_emit = false;
            lane.enqueue_instruction(tail, first);
            auto middle = first;
            middle.accumulator_reset = false;
            middle.repeat_count = kVectorLength - 2;
            lane.enqueue_instruction(tail, middle);
            auto last = middle;
            last.repeat_count = 1;
            last.accumulator_emit = true;
            last.output_stream = lane.fixed_output_stream_for_stage(tail);
            lane.enqueue_instruction(tail, last);
        }
        lane.tick(); // one Superlane-control decode cycle before Data
        const auto consume_sum = [&] {
            for (const auto& output : lane.outputs()) {
                sums[output.stream / 4] = output_float(output);
            }
        };
        for (std::size_t i = 0; i < kVectorLength; ++i) {
            auto streams = ftlpu::VxmLane::StreamBytes{};
            for (std::size_t row = 0; row < 4; ++row) {
                const auto value = x[batch + row][i];
                put_float(streams, row * 2, value);
                put_float(streams, row * 2 + 1, value);
            }
            lane.set_stream_inputs(streams);
            lane.tick();
            consume_sum();
        }
        drain(lane, consume_sum);

        // Phase 2: two 4-stage scalar chains, Scale -> Add epsilon -> Rsqrt.
        lane.set_chain_depth(ftlpu::VxmChainDepth::Four);
        for (std::size_t pair = 0; pair < 2; ++pair) {
            for (std::size_t chain = 0; chain < 2; ++chain) {
                const auto base = chain * 4;
                lane.enqueue_instruction(base, basic(ftlpu::VxmAluOpcode::Multiply,
                    ftlpu::VxmLaneOperand::StreamFloat32(),
                    ftlpu::VxmLaneOperand::Imm(1.0f / kVectorLength)));
                lane.enqueue_instruction(base + 1, basic(ftlpu::VxmAluOpcode::Add,
                    ftlpu::VxmLaneOperand::Previous(),
                    ftlpu::VxmLaneOperand::Imm(epsilon)));
                lane.enqueue_instruction(base + 2,
                    basic(ftlpu::VxmAluOpcode::Bypass));
                auto rsqrt = special(ftlpu::VxmSpecialAluOpcode::Rsqrt);
                rsqrt.output_stream = lane.fixed_output_stream_for_stage(base + 3);
                lane.enqueue_instruction(base + 3, rsqrt);
            }
        }
        lane.tick(); // one Superlane-control decode cycle before Data
        auto scalar_output_index = std::array<std::size_t, 2>{};
        const auto consume_scalar = [&] {
            for (const auto& output : lane.outputs()) {
                const auto chain = output.stream == 4 ? 0u : 1u;
                const auto pair = scalar_output_index[chain]++;
                inv_rms[pair * 2 + chain] = output_float(output);
            }
        };
        for (std::size_t pair = 0; pair < 2; ++pair) {
            auto streams = ftlpu::VxmLane::StreamBytes{};
            put_float(streams, 0, sums[pair * 2]);
            put_float(streams, 4, sums[pair * 2 + 1]);
            lane.set_stream_inputs(streams);
            lane.tick();
            consume_scalar();
        }
        drain(lane, consume_scalar);
        assert(scalar_output_index[0] == 2);
        assert(scalar_output_index[1] == 2);

        // Phase 3: four 2-stage chains.  The row scalar is explicitly loaded
        // into the small C1/C3-local register before the element stream starts.
        lane.set_chain_depth(ftlpu::VxmChainDepth::Two);
        for (std::size_t row = 0; row < 4; ++row) {
            lane.load_local_scalar(row * 2 + 1, inv_rms[row]);
        }
        for (std::size_t row = 0; row < 4; ++row) {
            const auto head = row * 2;
            const auto tail = head + 1;
            lane.enqueue_instruction(head, repeated(
                basic(ftlpu::VxmAluOpcode::Multiply,
                    ftlpu::VxmLaneOperand::StreamFloat32(),
                    ftlpu::VxmLaneOperand::StreamFloat32()), kVectorLength));
            auto normalize = basic(ftlpu::VxmAluOpcode::Multiply,
                ftlpu::VxmLaneOperand::Previous(), ftlpu::VxmLaneOperand::Acc());
            normalize.output_stream = lane.fixed_output_stream_for_stage(tail);
            normalize.repeat_count = kVectorLength;
            lane.enqueue_instruction(tail, normalize);
        }
        lane.tick(); // one Superlane-control decode cycle before Data
        const auto consume_output = [&] {
            for (const auto& output : lane.outputs()) {
                const auto row = output.stream / 4;
                actual[batch + row].push_back(output_float(output));
            }
        };
        for (std::size_t i = 0; i < kVectorLength; ++i) {
            auto streams = ftlpu::VxmLane::StreamBytes{};
            for (std::size_t row = 0; row < 4; ++row) {
                put_float(streams, row * 2, x[batch + row][i]);
                put_float(streams, row * 2 + 1, gamma[i]);
            }
            lane.set_stream_inputs(streams);
            lane.tick();
            consume_output();
        }
        drain(lane, consume_output);
    }

    for (std::size_t token = 0; token < kNormTokens; ++token) {
        assert(actual[token].size() == kVectorLength);
        float square_sum = 0.0f;
        for (const auto value : x[token]) square_sum += value * value;
        const auto scale = 1.0f / std::sqrt(square_sum / kVectorLength + epsilon);
        for (std::size_t i = 0; i < kVectorLength; ++i) {
            const auto expected = x[token][i] * gamma[i] * scale;
            error.observe(actual[token][i], expected);
            assert(near(actual[token][i], expected,
                        2.5e-3f, 5.0e-3f));
        }
    }
    const auto end = std::chrono::steady_clock::now();
    auto result = WorkloadResult{};
    result.statistics = lane.statistics();
    result.error = error;
    result.host_milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    validate_statistics(result.statistics);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Two).cycles != 0);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Four).cycles != 0);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Eight).cycles == 0);
    return result;
}

WorkloadResult test_softmax()
{
    auto lane = VxmLaneTestDriver{};
    configure_luts(lane);
    auto x = std::vector<std::vector<float>>(
        kSoftmaxTokens, std::vector<float>(kVectorLength));
    auto actual = std::vector<std::vector<float>>(kSoftmaxTokens);
    auto error = VxmOperatorErrorMetrics{};
    for (std::size_t token = 0; token < kSoftmaxTokens; ++token) {
        for (std::size_t i = 0; i < kVectorLength; ++i) {
            x[token][i] = 2.0f * std::cos(
                0.023f * static_cast<float>((token + 2) * (i + 1)))
                + 0.03f * static_cast<float>(token % 7);
        }
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t batch = 0; batch < kSoftmaxTokens; batch += 4) {
        auto maxima = std::array<float, 4>{};

        // Phase 1: four 2-stage row-maximum chains.
        lane.set_chain_depth(ftlpu::VxmChainDepth::Two);
        for (std::size_t row = 0; row < 4; ++row) {
            const auto head = row * 2;
            const auto tail = head + 1;
            lane.enqueue_instruction(head, repeated(
                basic(ftlpu::VxmAluOpcode::Bypass,
                    ftlpu::VxmLaneOperand::StreamFloat32()), kVectorLength));
            auto first = basic(ftlpu::VxmAluOpcode::Max,
                ftlpu::VxmLaneOperand::Previous(), ftlpu::VxmLaneOperand::Acc());
            first.accumulator_reset = true;
            first.accumulator_write = true;
            first.accumulator_emit = false;
            lane.enqueue_instruction(tail, first);
            auto middle = first;
            middle.accumulator_reset = false;
            middle.repeat_count = kVectorLength - 2;
            lane.enqueue_instruction(tail, middle);
            auto last = middle;
            last.repeat_count = 1;
            last.accumulator_emit = true;
            last.output_stream = lane.fixed_output_stream_for_stage(tail);
            lane.enqueue_instruction(tail, last);
        }
        lane.tick(); // one Superlane-control decode cycle before Data
        const auto consume_max = [&] {
            for (const auto& output : lane.outputs()) {
                maxima[output.stream / 4] = output_float(output);
            }
        };
        for (std::size_t i = 0; i < kVectorLength; ++i) {
            auto streams = ftlpu::VxmLane::StreamBytes{};
            for (std::size_t row = 0; row < 4; ++row) {
                put_float(streams, row * 2, x[batch + row][i]);
            }
            lane.set_stream_inputs(streams);
            lane.tick();
            consume_max();
        }
        drain(lane, consume_max);

        // Phases 2/3/4 use the two 4-stage chains, two rows at a time.
        lane.set_chain_depth(ftlpu::VxmChainDepth::Four);
        for (std::size_t pair = 0; pair < 2; ++pair) {
            auto sums = std::array<float, 2>{};
            auto inverse_sums = std::array<float, 2>{};

            // Subtract -> Exp -> Bypass -> local Sum.
            for (std::size_t chain = 0; chain < 2; ++chain) {
                const auto base = chain * 4;
                lane.enqueue_instruction(base, repeated(
                    basic(ftlpu::VxmAluOpcode::Subtract,
                        ftlpu::VxmLaneOperand::StreamFloat32(),
                        ftlpu::VxmLaneOperand::StreamFloat32()), kVectorLength));
                lane.enqueue_instruction(base + 1, repeated(
                    special(ftlpu::VxmSpecialAluOpcode::Exp), kVectorLength));
                lane.enqueue_instruction(base + 2, repeated(
                    basic(ftlpu::VxmAluOpcode::Bypass), kVectorLength));
                auto first = basic(ftlpu::VxmAluOpcode::Add,
                    ftlpu::VxmLaneOperand::Previous(), ftlpu::VxmLaneOperand::Acc());
                first.accumulator_reset = true;
                first.accumulator_write = true;
                first.accumulator_emit = false;
                lane.enqueue_instruction(base + 3, first);
                auto middle = first;
                middle.accumulator_reset = false;
                middle.repeat_count = kVectorLength - 2;
                lane.enqueue_instruction(base + 3, middle);
                auto last = middle;
                last.repeat_count = 1;
                last.accumulator_emit = true;
                last.output_stream = lane.fixed_output_stream_for_stage(base + 3);
                lane.enqueue_instruction(base + 3, last);
            }
            lane.tick(); // one Superlane-control decode cycle before Data
            const auto consume_sum = [&] {
                for (const auto& output : lane.outputs()) {
                    sums[output.stream == 4 ? 0 : 1] = output_float(output);
                }
            };
            for (std::size_t i = 0; i < kVectorLength; ++i) {
                auto streams = ftlpu::VxmLane::StreamBytes{};
                for (std::size_t chain = 0; chain < 2; ++chain) {
                    const auto row = pair * 2 + chain;
                    put_float(streams, chain * 4, x[batch + row][i]);
                    put_float(streams, chain * 4 + 1, maxima[row]);
                }
                lane.set_stream_inputs(streams);
                lane.tick();
                consume_sum();
            }
            drain(lane, consume_sum);

            // The completed row sums are scalars: Bypass -> Bypass -> Bypass
            // -> Reciprocal LUT.
            for (std::size_t chain = 0; chain < 2; ++chain) {
                const auto base = chain * 4;
                lane.enqueue_instruction(base, basic(ftlpu::VxmAluOpcode::Bypass,
                    ftlpu::VxmLaneOperand::StreamFloat32()));
                lane.enqueue_instruction(base + 1, basic(ftlpu::VxmAluOpcode::Bypass));
                lane.enqueue_instruction(base + 2, basic(ftlpu::VxmAluOpcode::Bypass));
                auto reciprocal = special(ftlpu::VxmSpecialAluOpcode::Reciprocal);
                reciprocal.output_stream = lane.fixed_output_stream_for_stage(base + 3);
                lane.enqueue_instruction(base + 3, reciprocal);
            }
            lane.tick(); // one Superlane-control decode cycle before Data
            auto streams = ftlpu::VxmLane::StreamBytes{};
            put_float(streams, 0, sums[0]);
            put_float(streams, 4, sums[1]);
            const auto consume_reciprocal = [&] {
                for (const auto& output : lane.outputs()) {
                    inverse_sums[output.stream == 4 ? 0 : 1] = output_float(output);
                }
            };
            lane.set_stream_inputs(streams);
            lane.tick();
            consume_reciprocal();
            drain(lane, consume_reciprocal);

            // Store each row scalar beside C3, then recompute Exp and normalize.
            lane.load_local_scalar(3, inverse_sums[0]);
            lane.load_local_scalar(7, inverse_sums[1]);
            for (std::size_t chain = 0; chain < 2; ++chain) {
                const auto base = chain * 4;
                lane.enqueue_instruction(base, repeated(
                    basic(ftlpu::VxmAluOpcode::Subtract,
                        ftlpu::VxmLaneOperand::StreamFloat32(),
                        ftlpu::VxmLaneOperand::StreamFloat32()), kVectorLength));
                lane.enqueue_instruction(base + 1, repeated(
                    special(ftlpu::VxmSpecialAluOpcode::Exp), kVectorLength));
                lane.enqueue_instruction(base + 2, repeated(
                    basic(ftlpu::VxmAluOpcode::Bypass), kVectorLength));
                auto normalize = basic(ftlpu::VxmAluOpcode::Multiply,
                    ftlpu::VxmLaneOperand::Previous(), ftlpu::VxmLaneOperand::Acc());
                normalize.output_stream = lane.fixed_output_stream_for_stage(base + 3);
                normalize.repeat_count = kVectorLength;
                lane.enqueue_instruction(base + 3, normalize);
            }
            lane.tick(); // one Superlane-control decode cycle before Data
            const auto consume_output = [&] {
                for (const auto& output : lane.outputs()) {
                    const auto chain = output.stream == 4 ? 0u : 1u;
                    actual[batch + pair * 2 + chain].push_back(output_float(output));
                }
            };
            for (std::size_t i = 0; i < kVectorLength; ++i) {
                auto element_streams = ftlpu::VxmLane::StreamBytes{};
                for (std::size_t chain = 0; chain < 2; ++chain) {
                    const auto row = pair * 2 + chain;
                    put_float(element_streams, chain * 4, x[batch + row][i]);
                    put_float(element_streams, chain * 4 + 1, maxima[row]);
                }
                lane.set_stream_inputs(element_streams);
                lane.tick();
                consume_output();
            }
            drain(lane, consume_output);
        }
    }

    for (std::size_t token = 0; token < kSoftmaxTokens; ++token) {
        assert(actual[token].size() == kVectorLength);
        const auto maximum = *std::max_element(x[token].begin(), x[token].end());
        float denominator = 0.0f;
        for (const auto value : x[token]) denominator += std::exp(value - maximum);
        float probability_sum = 0.0f;
        for (std::size_t i = 0; i < kVectorLength; ++i) {
            const auto expected = std::exp(x[token][i] - maximum) / denominator;
            probability_sum += actual[token][i];
            error.observe(actual[token][i], expected);
            assert(near(actual[token][i], expected, 2.0e-4f, 1.0e-2f));
        }
        assert(near(probability_sum, 1.0f, 2.0e-3f, 0.0f));
    }
    const auto end = std::chrono::steady_clock::now();
    auto result = WorkloadResult{};
    result.statistics = lane.statistics();
    result.error = error;
    result.host_milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    validate_statistics(result.statistics);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Two).cycles != 0);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Four).cycles != 0);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Eight).cycles == 0);
    return result;
}

WorkloadResult test_swiglu()
{
    constexpr std::size_t elements = kSwigluTokens * kVectorLength;
    auto lane = VxmLaneTestDriver{};
    configure_luts(lane);
    auto gate = std::vector<float>(elements);
    auto up = std::vector<float>(elements);
    auto actual = std::vector<float>{};
    actual.reserve(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        gate[i] = 2.0f * std::sin(0.019f * static_cast<float>(i + 1));
        up[i] = 1.25f * std::cos(0.013f * static_cast<float>(i + 5));
    }

    const auto start = std::chrono::steady_clock::now();
    lane.set_chain_depth(ftlpu::VxmChainDepth::Eight);
    lane.enqueue_instruction(0, repeated(
        basic(ftlpu::VxmAluOpcode::Negate,
            ftlpu::VxmLaneOperand::StreamFloat32(),
            ftlpu::VxmLaneOperand::StreamFloat32()), elements));
    lane.enqueue_instruction(1, repeated(
        special(ftlpu::VxmSpecialAluOpcode::Exp), elements));
    lane.enqueue_instruction(2, repeated(
        basic(ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Previous(), ftlpu::VxmLaneOperand::Imm(1.0f)),
        elements));
    lane.enqueue_instruction(3, repeated(
        special(ftlpu::VxmSpecialAluOpcode::Reciprocal), elements));
    lane.enqueue_instruction(4, repeated(
        basic(ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Previous(), ftlpu::VxmLaneOperand::Original()),
        elements));
    lane.enqueue_instruction(5, repeated(
        basic(ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Previous(), ftlpu::VxmLaneOperand::Aux()),
        elements));
    lane.enqueue_instruction(6, repeated(
        basic(ftlpu::VxmAluOpcode::Bypass), elements));
    auto output = basic(ftlpu::VxmAluOpcode::Bypass);
    output.output_stream = lane.fixed_output_stream_for_stage(7);
    output.repeat_count = elements;
    lane.enqueue_instruction(7, output);
    lane.tick(); // one Superlane-control decode cycle before Data
    const auto consume = [&] {
        for (const auto& output : lane.outputs()) actual.push_back(output_float(output));
    };
    for (std::size_t i = 0; i < elements; ++i) {
        auto streams = ftlpu::VxmLane::StreamBytes{};
        put_float(streams, 0, gate[i]);
        put_float(streams, 1, up[i]);
        lane.set_stream_inputs(streams);
        lane.tick();
        consume();
    }
    drain(lane, consume);

    assert(actual.size() == elements);
    auto error = VxmOperatorErrorMetrics{};
    for (std::size_t i = 0; i < elements; ++i) {
        const auto expected = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
        error.observe(actual[i], expected);
        assert(near(actual[i], expected, 2.5e-3f, 7.5e-3f));
    }
    const auto end = std::chrono::steady_clock::now();
    auto result = WorkloadResult{};
    result.statistics = lane.statistics();
    result.error = error;
    result.host_milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    validate_statistics(result.statistics);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Two).cycles == 0);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Four).cycles == 0);
    assert(result.statistics.for_depth(ftlpu::VxmChainDepth::Eight).cycles != 0);
    return result;
}

} // namespace

inline void run_lane_operator_workloads(std::ostream& summary,
                                        std::ostream* result_file = nullptr)
{
    const auto rmsnorm = test_rmsnorm();
    const auto softmax = test_softmax();
    const auto swiglu = test_swiglu();

    const auto print_summary = [&](std::ostream& os) {
        os << "VXM single-lane workloads: RMSNorm/Softmax "
           << kNormTokens << " tokens x " << kVectorLength
           << ", SwiGLU " << kSwigluTokens << " tokens x "
           << kVectorLength << "\n";
        print_statistics(os, "RMSNorm", rmsnorm);
        print_statistics(os, "Softmax", softmax);
        print_statistics(os, "SwiGLU", swiglu);
    };
    print_summary(summary);
    if (result_file != nullptr) {
        print_summary(*result_file);
        print_cycle_trace(*result_file, "RMSNorm", rmsnorm.statistics);
        print_cycle_trace(*result_file, "Softmax", softmax.statistics);
        print_cycle_trace(*result_file, "SwiGLU", swiglu.statistics);
    }
}
