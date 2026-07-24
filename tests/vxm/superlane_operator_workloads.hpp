#pragma once

#include "ftlpu/vxm/superlane.hpp"
#include "operator_error_metrics.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSuperlaneTokensPerLane = 8;
constexpr std::size_t kSuperlaneVectorLength = 128;

template <typename Fn>
std::vector<ftlpu::VxmLutEntry> superlane_table(
    float min, float width, std::size_t count, Fn fn)
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

void configure_superlane_luts(ftlpu::VxmSuperlane& superlane)
{
    constexpr std::size_t exp_entries = 256;
    constexpr std::size_t reciprocal_entries = 256;
    constexpr std::size_t rsqrt_entries = 512;
    constexpr float ln2 = 0.6931471805599453f;

    superlane.configure_special_lut(ftlpu::VxmSpecialAluOpcode::Exp,
        {-ln2 / 2.0f, ln2 / exp_entries},
        superlane_table(-ln2 / 2.0f, ln2 / exp_entries, exp_entries,
            [](float x) { return std::exp(x); }));
    superlane.configure_special_lut(ftlpu::VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f / reciprocal_entries},
        superlane_table(1.0f, 1.0f / reciprocal_entries, reciprocal_entries,
            [](float x) { return 1.0f / x; }));
    superlane.configure_special_lut(ftlpu::VxmSpecialAluOpcode::Rsqrt,
        {1.0f, 3.0f / rsqrt_entries},
        superlane_table(1.0f, 3.0f / rsqrt_entries, rsqrt_entries,
            [](float x) { return 1.0f / std::sqrt(x); }));
}

void put_superlane_float(ftlpu::VxmLane::StreamBytes& streams,
                         std::size_t group, float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto base = group * ftlpu::VxmLane::kStreamGroupBytes;
    for (std::size_t byte = 0; byte < 4; ++byte) {
        streams[base + byte] =
            static_cast<std::uint8_t>(bits >> (8 * byte));
    }
}

float superlane_output_float(const ftlpu::VxmSuperlane::Output& output,
                             std::size_t lane)
{
    return ftlpu::VxmLane::unpack_float32(output.byte_values[lane]);
}

ftlpu::VxmLaneAluInstruction superlane_basic(
    ftlpu::VxmAluOpcode opcode,
    ftlpu::VxmLaneOperand lhs = ftlpu::VxmLaneOperand::Previous(),
    ftlpu::VxmLaneOperand rhs = ftlpu::VxmLaneOperand::Imm(0.0f))
{
    auto instruction = ftlpu::VxmLaneAluInstruction{opcode, lhs, rhs};
    instruction.precision = ftlpu::VxmAluPrecision::Float32;
    instruction.output_type = ftlpu::VxmCastTarget::Float32;
    return instruction;
}

ftlpu::VxmLaneAluInstruction superlane_special(
    ftlpu::VxmSpecialAluOpcode opcode)
{
    auto instruction = ftlpu::VxmLaneAluInstruction{
        opcode, ftlpu::VxmLaneOperand::Previous()};
    instruction.output_type = ftlpu::VxmCastTarget::Float32;
    return instruction;
}

ftlpu::VxmLaneAluInstruction superlane_repeated(
    ftlpu::VxmLaneAluInstruction instruction, std::size_t count)
{
    instruction.repeat_count = count;
    return instruction;
}

bool superlane_near(float actual, float expected,
                    float absolute, float relative)
{
    return std::fabs(actual - expected)
        <= absolute + relative * std::fabs(expected);
}

template <typename Consume>
void drain_superlane(ftlpu::VxmSuperlane& superlane, Consume consume)
{
    while (!superlane.idle()) {
        superlane.tick();
        consume();
    }
}

template <typename Consume>
void drain_superlane_datapath(
    ftlpu::VxmSuperlane& superlane, Consume consume)
{
    while (!superlane.datapath_idle()) {
        superlane.tick();
        consume();
    }
}

void validate_superlane_lockstep(const ftlpu::VxmSuperlane& superlane)
{
    const auto& reference = superlane.lane(0).statistics();
    for (std::size_t lane = 1; lane < ftlpu::VxmSuperlane::kLaneCount;
         ++lane) {
        const auto& current = superlane.lane(lane).statistics();
        assert(current.cycles == reference.cycles);
        assert(current.executed_slots == reference.executed_slots);
        assert(current.useful_slots == reference.useful_slots);
        assert(current.stage_executions == reference.stage_executions);
        assert(current.timeline.size() == reference.timeline.size());
        for (std::size_t cycle = 0; cycle < reference.timeline.size();
             ++cycle) {
            assert(current.timeline[cycle].chain_depth
                   == reference.timeline[cycle].chain_depth);
            assert(current.timeline[cycle].states
                   == reference.timeline[cycle].states);
            assert(current.timeline[cycle].useful
                   == reference.timeline[cycle].useful);
        }
    }
}

struct SuperlaneWorkloadResult {
    std::string name{};
    ftlpu::VxmLane::Statistics statistics{};
    double host_milliseconds{0.0};
    VxmOperatorErrorMetrics error{};
};

SuperlaneWorkloadResult test_superlane_rmsnorm()
{
    constexpr float epsilon = 1.0e-5f;
    auto superlane = ftlpu::VxmSuperlane{};
    configure_superlane_luts(superlane);

    auto x = std::vector<std::vector<std::vector<float>>>(
        ftlpu::VxmSuperlane::kLaneCount,
        std::vector<std::vector<float>>(
            kSuperlaneTokensPerLane,
            std::vector<float>(kSuperlaneVectorLength)));
    auto gamma = std::vector<float>(kSuperlaneVectorLength);
    auto actual = std::vector<std::vector<std::vector<float>>>(
        ftlpu::VxmSuperlane::kLaneCount,
        std::vector<std::vector<float>>(kSuperlaneTokensPerLane));

    for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
        gamma[i] = 0.75f + 0.01f * static_cast<float>(i % 17);
    }
    for (std::size_t lane = 0; lane < ftlpu::VxmSuperlane::kLaneCount;
         ++lane) {
        for (std::size_t token = 0; token < kSuperlaneTokensPerLane;
             ++token) {
            for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
                x[lane][token][i] = 1.5f * std::sin(
                    0.017f * static_cast<float>(
                        (token + 1) * (i + 3) + lane * 7))
                    + 0.02f * static_cast<float>((token + lane) % 5);
            }
        }
    }

    const auto enqueue_square = [&](ftlpu::VxmChainDepth target_depth) {
        for (std::size_t row = 0; row < 4; ++row) {
            const auto head = row * 2;
            const auto tail = head + 1;
            superlane.enqueue_instruction_for_depth(target_depth, head,
                superlane_repeated(
                    superlane_basic(ftlpu::VxmAluOpcode::Multiply,
                        ftlpu::VxmLaneOperand::StreamFloat32(),
                        ftlpu::VxmLaneOperand::StreamFloat32()),
                    kSuperlaneVectorLength));
            auto first = superlane_basic(ftlpu::VxmAluOpcode::Add,
                ftlpu::VxmLaneOperand::Previous(),
                ftlpu::VxmLaneOperand::Acc());
            first.accumulator_reset = true;
            first.accumulator_write = true;
            first.accumulator_emit = false;
            superlane.enqueue_instruction_for_depth(
                target_depth, tail, first);
            auto middle = first;
            middle.accumulator_reset = false;
            middle.repeat_count = kSuperlaneVectorLength - 2;
            superlane.enqueue_instruction_for_depth(
                target_depth, tail, middle);
            auto last = middle;
            last.repeat_count = 1;
            last.accumulator_emit = true;
            last.output_stream =
                superlane.lane(0).fixed_output_stream_for_stage(tail);
            superlane.enqueue_instruction_for_depth(
                target_depth, tail, last);
        }
    };
    const auto enqueue_scalar_rsqrt =
        [&](ftlpu::VxmChainDepth target_depth) {
        for (std::size_t pair = 0; pair < 2; ++pair) {
            for (std::size_t chain = 0; chain < 2; ++chain) {
                const auto base = chain * 4;
                superlane.enqueue_instruction_for_depth(target_depth, base,
                    superlane_basic(ftlpu::VxmAluOpcode::Multiply,
                        ftlpu::VxmLaneOperand::StreamFloat32(),
                        ftlpu::VxmLaneOperand::Imm(
                            1.0f / kSuperlaneVectorLength)));
                superlane.enqueue_instruction_for_depth(
                    target_depth, base + 1,
                    superlane_basic(ftlpu::VxmAluOpcode::Add,
                        ftlpu::VxmLaneOperand::Previous(),
                        ftlpu::VxmLaneOperand::Imm(epsilon)));
                superlane.enqueue_instruction_for_depth(
                    target_depth, base + 2,
                    superlane_basic(ftlpu::VxmAluOpcode::Bypass));
                auto rsqrt =
                    superlane_special(ftlpu::VxmSpecialAluOpcode::Rsqrt);
                rsqrt.output_stream =
                    superlane.lane(0).fixed_output_stream_for_stage(base + 3);
                superlane.enqueue_instruction_for_depth(
                    target_depth, base + 3, rsqrt);
            }
        }
    };
    const auto enqueue_normalize = [&](ftlpu::VxmChainDepth target_depth) {
        for (std::size_t row = 0; row < 4; ++row) {
            const auto head = row * 2;
            const auto tail = head + 1;
            superlane.enqueue_instruction_for_depth(target_depth, head,
                superlane_repeated(
                    superlane_basic(ftlpu::VxmAluOpcode::Multiply,
                        ftlpu::VxmLaneOperand::StreamFloat32(),
                        ftlpu::VxmLaneOperand::StreamFloat32()),
                    kSuperlaneVectorLength));
            auto normalize = superlane_basic(ftlpu::VxmAluOpcode::Multiply,
                ftlpu::VxmLaneOperand::Previous(),
                ftlpu::VxmLaneOperand::Acc());
            normalize.output_stream =
                superlane.lane(0).fixed_output_stream_for_stage(tail);
            normalize.repeat_count = kSuperlaneVectorLength;
            superlane.enqueue_instruction_for_depth(
                target_depth, tail, normalize);
        }
    };

    const auto start = std::chrono::steady_clock::now();
    superlane.set_chain_depth(ftlpu::VxmChainDepth::Two);
    enqueue_square(ftlpu::VxmChainDepth::Two);
    superlane.tick(); // the only exposed RMSNorm decode-start cycle
    for (std::size_t batch = 0; batch < kSuperlaneTokensPerLane;
         batch += 4) {
        auto sums =
            std::array<std::array<float, 4>,
                       ftlpu::VxmSuperlane::kLaneCount>{};
        auto inv_rms =
            std::array<std::array<float, 4>,
                       ftlpu::VxmSuperlane::kLaneCount>{};

        const auto consume_sum = [&] {
            for (const auto& output : superlane.outputs()) {
                const auto row = output.stream / 4;
                for (std::size_t lane = 0;
                     lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                    sums[lane][row] =
                        superlane_output_float(output, lane);
                }
            }
        };
        for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
            auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
            for (std::size_t lane = 0;
                 lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                for (std::size_t row = 0; row < 4; ++row) {
                    const auto value = x[lane][batch + row][i];
                    put_superlane_float(streams[lane], row * 2, value);
                    put_superlane_float(streams[lane], row * 2 + 1, value);
                }
            }
            superlane.set_stream_inputs(streams);
            superlane.tick();
            consume_sum();
        }
        enqueue_scalar_rsqrt(ftlpu::VxmChainDepth::Four);
        drain_superlane_datapath(superlane, consume_sum);
        superlane.set_chain_depth(ftlpu::VxmChainDepth::Four);
        auto scalar_output_index =
            std::array<std::array<std::size_t, 2>,
                       ftlpu::VxmSuperlane::kLaneCount>{};
        const auto consume_scalar = [&] {
            for (const auto& output : superlane.outputs()) {
                const auto chain = output.stream == 4 ? 0u : 1u;
                for (std::size_t lane = 0;
                     lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                    const auto pair = scalar_output_index[lane][chain]++;
                    inv_rms[lane][pair * 2 + chain] =
                        superlane_output_float(output, lane);
                }
            }
        };
        for (std::size_t pair = 0; pair < 2; ++pair) {
            auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
            for (std::size_t lane = 0;
                 lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                put_superlane_float(
                    streams[lane], 0, sums[lane][pair * 2]);
                put_superlane_float(
                    streams[lane], 4, sums[lane][pair * 2 + 1]);
            }
            superlane.set_stream_inputs(streams);
            superlane.tick();
            consume_scalar();
        }
        enqueue_normalize(ftlpu::VxmChainDepth::Two);
        drain_superlane_datapath(superlane, consume_scalar);
        superlane.set_chain_depth(ftlpu::VxmChainDepth::Two);
        for (std::size_t lane = 0;
             lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
            for (std::size_t row = 0; row < 4; ++row) {
                superlane.lane(lane).load_local_scalar(
                    row * 2 + 1, inv_rms[lane][row]);
            }
        }
        const auto consume_output = [&] {
            for (const auto& output : superlane.outputs()) {
                const auto row = output.stream / 4;
                for (std::size_t lane = 0;
                     lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                    actual[lane][batch + row].push_back(
                        superlane_output_float(output, lane));
                }
            }
        };
        for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
            auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
            for (std::size_t lane = 0;
                 lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                for (std::size_t row = 0; row < 4; ++row) {
                    put_superlane_float(
                        streams[lane], row * 2,
                        x[lane][batch + row][i]);
                    put_superlane_float(
                        streams[lane], row * 2 + 1, gamma[i]);
                }
            }
            superlane.set_stream_inputs(streams);
            superlane.tick();
            consume_output();
        }
        const auto has_next_batch =
            batch + 4 < kSuperlaneTokensPerLane;
        if (has_next_batch) {
            enqueue_square(ftlpu::VxmChainDepth::Two);
            drain_superlane_datapath(superlane, consume_output);
        } else {
            drain_superlane(superlane, consume_output);
        }
    }

    auto error = VxmOperatorErrorMetrics{};
    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        for (std::size_t token = 0; token < kSuperlaneTokensPerLane;
             ++token) {
            assert(actual[lane][token].size() == kSuperlaneVectorLength);
            float square_sum = 0.0f;
            for (const auto value : x[lane][token]) {
                square_sum += value * value;
            }
            const auto scale = 1.0f / std::sqrt(
                square_sum / kSuperlaneVectorLength + epsilon);
            for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
                const auto expected =
                    x[lane][token][i] * gamma[i] * scale;
                error.observe(actual[lane][token][i], expected);
                assert(superlane_near(
                    actual[lane][token][i],
                    expected,
                    2.5e-3f, 5.0e-3f));
            }
        }
    }
    validate_superlane_lockstep(superlane);
    const auto end = std::chrono::steady_clock::now();
    return {"RMSNorm", superlane.lane(0).statistics(),
        std::chrono::duration<double, std::milli>(end - start).count(),
        error};
}

SuperlaneWorkloadResult test_superlane_softmax()
{
    auto superlane = ftlpu::VxmSuperlane{};
    configure_superlane_luts(superlane);
    auto x = std::vector<std::vector<std::vector<float>>>(
        ftlpu::VxmSuperlane::kLaneCount,
        std::vector<std::vector<float>>(
            kSuperlaneTokensPerLane,
            std::vector<float>(kSuperlaneVectorLength)));
    auto actual = std::vector<std::vector<std::vector<float>>>(
        ftlpu::VxmSuperlane::kLaneCount,
        std::vector<std::vector<float>>(kSuperlaneTokensPerLane));
    auto error = VxmOperatorErrorMetrics{};
    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        for (std::size_t token = 0; token < kSuperlaneTokensPerLane;
             ++token) {
            for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
                x[lane][token][i] = 2.0f * std::cos(
                    0.023f * static_cast<float>(
                        (token + 2) * (i + 1) + lane * 5))
                    + 0.03f * static_cast<float>((token + lane) % 7);
            }
        }
    }

    const auto enqueue_max = [&](ftlpu::VxmChainDepth target_depth) {
        for (std::size_t row = 0; row < 4; ++row) {
            const auto head = row * 2;
            const auto tail = head + 1;
            superlane.enqueue_instruction_for_depth(target_depth, head,
                superlane_repeated(
                    superlane_basic(ftlpu::VxmAluOpcode::Bypass,
                        ftlpu::VxmLaneOperand::StreamFloat32()),
                    kSuperlaneVectorLength));
            auto first = superlane_basic(ftlpu::VxmAluOpcode::Max,
                ftlpu::VxmLaneOperand::Previous(),
                ftlpu::VxmLaneOperand::Acc());
            first.accumulator_reset = true;
            first.accumulator_write = true;
            first.accumulator_emit = false;
            superlane.enqueue_instruction_for_depth(
                target_depth, tail, first);
            auto middle = first;
            middle.accumulator_reset = false;
            middle.repeat_count = kSuperlaneVectorLength - 2;
            superlane.enqueue_instruction_for_depth(
                target_depth, tail, middle);
            auto last = middle;
            last.repeat_count = 1;
            last.accumulator_emit = true;
            last.output_stream =
                superlane.lane(0).fixed_output_stream_for_stage(tail);
            superlane.enqueue_instruction_for_depth(
                target_depth, tail, last);
        }
    };
    const auto enqueue_exp_sum = [&](ftlpu::VxmChainDepth target_depth) {
        for (std::size_t chain = 0; chain < 2; ++chain) {
            const auto base = chain * 4;
            superlane.enqueue_instruction_for_depth(target_depth, base,
                superlane_repeated(
                    superlane_basic(ftlpu::VxmAluOpcode::Subtract,
                        ftlpu::VxmLaneOperand::StreamFloat32(),
                        ftlpu::VxmLaneOperand::StreamFloat32()),
                    kSuperlaneVectorLength));
            superlane.enqueue_instruction_for_depth(target_depth, base + 1,
                superlane_repeated(
                    superlane_special(ftlpu::VxmSpecialAluOpcode::Exp),
                    kSuperlaneVectorLength));
            superlane.enqueue_instruction_for_depth(target_depth, base + 2,
                superlane_repeated(
                    superlane_basic(ftlpu::VxmAluOpcode::Bypass),
                    kSuperlaneVectorLength));
            auto first = superlane_basic(ftlpu::VxmAluOpcode::Add,
                ftlpu::VxmLaneOperand::Previous(),
                ftlpu::VxmLaneOperand::Acc());
            first.accumulator_reset = true;
            first.accumulator_write = true;
            first.accumulator_emit = false;
            superlane.enqueue_instruction_for_depth(
                target_depth, base + 3, first);
            auto middle = first;
            middle.accumulator_reset = false;
            middle.repeat_count = kSuperlaneVectorLength - 2;
            superlane.enqueue_instruction_for_depth(
                target_depth, base + 3, middle);
            auto last = middle;
            last.repeat_count = 1;
            last.accumulator_emit = true;
            last.output_stream =
                superlane.lane(0).fixed_output_stream_for_stage(base + 3);
            superlane.enqueue_instruction_for_depth(
                target_depth, base + 3, last);
        }
    };
    const auto enqueue_reciprocal = [&](ftlpu::VxmChainDepth target_depth) {
        for (std::size_t chain = 0; chain < 2; ++chain) {
            const auto base = chain * 4;
            superlane.enqueue_instruction_for_depth(target_depth, base,
                superlane_basic(ftlpu::VxmAluOpcode::Bypass,
                    ftlpu::VxmLaneOperand::StreamFloat32()));
            superlane.enqueue_instruction_for_depth(target_depth, base + 1,
                superlane_basic(ftlpu::VxmAluOpcode::Bypass));
            superlane.enqueue_instruction_for_depth(target_depth, base + 2,
                superlane_basic(ftlpu::VxmAluOpcode::Bypass));
            auto reciprocal =
                superlane_special(ftlpu::VxmSpecialAluOpcode::Reciprocal);
            reciprocal.output_stream =
                superlane.lane(0).fixed_output_stream_for_stage(base + 3);
            superlane.enqueue_instruction_for_depth(
                target_depth, base + 3, reciprocal);
        }
    };
    const auto enqueue_softmax_normalize =
        [&](ftlpu::VxmChainDepth target_depth) {
        for (std::size_t chain = 0; chain < 2; ++chain) {
            const auto base = chain * 4;
            superlane.enqueue_instruction_for_depth(target_depth, base,
                superlane_repeated(
                    superlane_basic(ftlpu::VxmAluOpcode::Subtract,
                        ftlpu::VxmLaneOperand::StreamFloat32(),
                        ftlpu::VxmLaneOperand::StreamFloat32()),
                    kSuperlaneVectorLength));
            superlane.enqueue_instruction_for_depth(target_depth, base + 1,
                superlane_repeated(
                    superlane_special(ftlpu::VxmSpecialAluOpcode::Exp),
                    kSuperlaneVectorLength));
            superlane.enqueue_instruction_for_depth(target_depth, base + 2,
                superlane_repeated(
                    superlane_basic(ftlpu::VxmAluOpcode::Bypass),
                    kSuperlaneVectorLength));
            auto normalize =
                superlane_basic(ftlpu::VxmAluOpcode::Multiply,
                    ftlpu::VxmLaneOperand::Previous(),
                    ftlpu::VxmLaneOperand::Acc());
            normalize.output_stream =
                superlane.lane(0).fixed_output_stream_for_stage(base + 3);
            normalize.repeat_count = kSuperlaneVectorLength;
            superlane.enqueue_instruction_for_depth(
                target_depth, base + 3, normalize);
        }
    };

    const auto start = std::chrono::steady_clock::now();
    superlane.set_chain_depth(ftlpu::VxmChainDepth::Two);
    enqueue_max(ftlpu::VxmChainDepth::Two);
    superlane.tick(); // the only exposed Softmax decode-start cycle
    for (std::size_t batch = 0; batch < kSuperlaneTokensPerLane;
         batch += 4) {
        auto maxima =
            std::array<std::array<float, 4>,
                       ftlpu::VxmSuperlane::kLaneCount>{};
        const auto consume_max = [&] {
            for (const auto& output : superlane.outputs()) {
                const auto row = output.stream / 4;
                for (std::size_t lane = 0;
                     lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                    maxima[lane][row] =
                        superlane_output_float(output, lane);
                }
            }
        };
        for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
            auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
            for (std::size_t lane = 0;
                 lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                for (std::size_t row = 0; row < 4; ++row) {
                    put_superlane_float(
                        streams[lane], row * 2,
                        x[lane][batch + row][i]);
                }
            }
            superlane.set_stream_inputs(streams);
            superlane.tick();
            consume_max();
        }
        enqueue_exp_sum(ftlpu::VxmChainDepth::Four);
        drain_superlane_datapath(superlane, consume_max);
        superlane.set_chain_depth(ftlpu::VxmChainDepth::Four);
        for (std::size_t pair = 0; pair < 2; ++pair) {
            auto sums =
                std::array<std::array<float, 2>,
                           ftlpu::VxmSuperlane::kLaneCount>{};
            auto inverse_sums =
                std::array<std::array<float, 2>,
                           ftlpu::VxmSuperlane::kLaneCount>{};

            const auto consume_sum = [&] {
                for (const auto& output : superlane.outputs()) {
                    const auto chain = output.stream == 4 ? 0u : 1u;
                    for (std::size_t lane = 0;
                         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                        sums[lane][chain] =
                            superlane_output_float(output, lane);
                    }
                }
            };
            for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
                auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
                for (std::size_t lane = 0;
                     lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                    for (std::size_t chain = 0; chain < 2; ++chain) {
                        const auto row = pair * 2 + chain;
                        put_superlane_float(
                            streams[lane], chain * 4,
                            x[lane][batch + row][i]);
                        put_superlane_float(
                            streams[lane], chain * 4 + 1,
                            maxima[lane][row]);
                    }
                }
                superlane.set_stream_inputs(streams);
                superlane.tick();
                consume_sum();
            }
            enqueue_reciprocal(ftlpu::VxmChainDepth::Four);
            drain_superlane_datapath(superlane, consume_sum);
            auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
            for (std::size_t lane = 0;
                 lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                put_superlane_float(streams[lane], 0, sums[lane][0]);
                put_superlane_float(streams[lane], 4, sums[lane][1]);
            }
            const auto consume_reciprocal = [&] {
                for (const auto& output : superlane.outputs()) {
                    const auto chain = output.stream == 4 ? 0u : 1u;
                    for (std::size_t lane = 0;
                         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                        inverse_sums[lane][chain] =
                            superlane_output_float(output, lane);
                    }
                }
            };
            superlane.set_stream_inputs(streams);
            superlane.tick();
            consume_reciprocal();
            enqueue_softmax_normalize(ftlpu::VxmChainDepth::Four);
            drain_superlane_datapath(superlane, consume_reciprocal);

            for (std::size_t lane = 0;
                 lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                superlane.lane(lane).load_local_scalar(
                    3, inverse_sums[lane][0]);
                superlane.lane(lane).load_local_scalar(
                    7, inverse_sums[lane][1]);
            }
            const auto consume_output = [&] {
                for (const auto& output : superlane.outputs()) {
                    const auto chain = output.stream == 4 ? 0u : 1u;
                    for (std::size_t lane = 0;
                         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                        actual[lane][batch + pair * 2 + chain].push_back(
                            superlane_output_float(output, lane));
                    }
                }
            };
            for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
                auto element_streams =
                    ftlpu::VxmSuperlane::StreamMatrix{};
                for (std::size_t lane = 0;
                     lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                    for (std::size_t chain = 0; chain < 2; ++chain) {
                        const auto row = pair * 2 + chain;
                        put_superlane_float(
                            element_streams[lane], chain * 4,
                            x[lane][batch + row][i]);
                        put_superlane_float(
                            element_streams[lane], chain * 4 + 1,
                            maxima[lane][row]);
                    }
                }
                superlane.set_stream_inputs(element_streams);
                superlane.tick();
                consume_output();
            }
            const auto has_next_pair = pair + 1 < 2;
            const auto has_next_batch =
                batch + 4 < kSuperlaneTokensPerLane;
            if (has_next_pair) {
                enqueue_exp_sum(ftlpu::VxmChainDepth::Four);
                drain_superlane_datapath(superlane, consume_output);
            } else if (has_next_batch) {
                enqueue_max(ftlpu::VxmChainDepth::Two);
                drain_superlane_datapath(superlane, consume_output);
                superlane.set_chain_depth(ftlpu::VxmChainDepth::Two);
            } else {
                drain_superlane(superlane, consume_output);
            }
        }
    }

    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        for (std::size_t token = 0; token < kSuperlaneTokensPerLane;
             ++token) {
            assert(actual[lane][token].size() == kSuperlaneVectorLength);
            const auto maximum = *std::max_element(
                x[lane][token].begin(), x[lane][token].end());
            float denominator = 0.0f;
            for (const auto value : x[lane][token]) {
                denominator += std::exp(value - maximum);
            }
            float probability_sum = 0.0f;
            for (std::size_t i = 0; i < kSuperlaneVectorLength; ++i) {
                const auto expected =
                    std::exp(x[lane][token][i] - maximum) / denominator;
                probability_sum += actual[lane][token][i];
                error.observe(actual[lane][token][i], expected);
                assert(superlane_near(
                    actual[lane][token][i], expected,
                    2.0e-4f, 1.0e-2f));
            }
            assert(superlane_near(
                probability_sum, 1.0f, 2.0e-3f, 0.0f));
        }
    }
    validate_superlane_lockstep(superlane);
    const auto end = std::chrono::steady_clock::now();
    return {"Softmax", superlane.lane(0).statistics(),
        std::chrono::duration<double, std::milli>(end - start).count(),
        error};
}

SuperlaneWorkloadResult test_superlane_swiglu()
{
    constexpr std::size_t elements =
        kSuperlaneTokensPerLane * kSuperlaneVectorLength;
    auto superlane = ftlpu::VxmSuperlane{};
    configure_superlane_luts(superlane);
    auto gate = std::vector<std::vector<float>>(
        ftlpu::VxmSuperlane::kLaneCount,
        std::vector<float>(elements));
    auto up = gate;
    auto actual = std::vector<std::vector<float>>(
        ftlpu::VxmSuperlane::kLaneCount);
    auto error = VxmOperatorErrorMetrics{};
    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        actual[lane].reserve(elements);
        for (std::size_t i = 0; i < elements; ++i) {
            gate[lane][i] = 2.0f * std::sin(
                0.019f * static_cast<float>(i + 1)
                + 0.07f * static_cast<float>(lane));
            up[lane][i] = 1.25f * std::cos(
                0.013f * static_cast<float>(i + 5)
                + 0.05f * static_cast<float>(lane));
        }
    }

    const auto start = std::chrono::steady_clock::now();
    superlane.set_chain_depth(ftlpu::VxmChainDepth::Eight);
    superlane.enqueue_instruction(0, superlane_repeated(
        superlane_basic(ftlpu::VxmAluOpcode::Negate,
            ftlpu::VxmLaneOperand::StreamFloat32(),
            ftlpu::VxmLaneOperand::StreamFloat32()), elements));
    superlane.enqueue_instruction(1, superlane_repeated(
        superlane_special(ftlpu::VxmSpecialAluOpcode::Exp), elements));
    superlane.enqueue_instruction(2, superlane_repeated(
        superlane_basic(ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Imm(1.0f)), elements));
    superlane.enqueue_instruction(3, superlane_repeated(
        superlane_special(ftlpu::VxmSpecialAluOpcode::Reciprocal),
        elements));
    superlane.enqueue_instruction(4, superlane_repeated(
        superlane_basic(ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Original()), elements));
    superlane.enqueue_instruction(5, superlane_repeated(
        superlane_basic(ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Aux()), elements));
    superlane.enqueue_instruction(6, superlane_repeated(
        superlane_basic(ftlpu::VxmAluOpcode::Bypass), elements));
    auto output = superlane_basic(ftlpu::VxmAluOpcode::Bypass);
    output.output_stream =
        superlane.lane(0).fixed_output_stream_for_stage(7);
    output.repeat_count = elements;
    superlane.enqueue_instruction(7, output);
    superlane.tick(); // explicit one-cycle Superlane decoder

    const auto consume = [&] {
        for (const auto& result : superlane.outputs()) {
            for (std::size_t lane = 0;
                 lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
                actual[lane].push_back(
                    superlane_output_float(result, lane));
            }
        }
    };
    for (std::size_t i = 0; i < elements; ++i) {
        auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
        for (std::size_t lane = 0;
             lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
            put_superlane_float(streams[lane], 0, gate[lane][i]);
            put_superlane_float(streams[lane], 1, up[lane][i]);
        }
        superlane.set_stream_inputs(streams);
        superlane.tick();
        consume();
    }
    drain_superlane(superlane, consume);

    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        assert(actual[lane].size() == elements);
        for (std::size_t i = 0; i < elements; ++i) {
            const auto expected =
                gate[lane][i] / (1.0f + std::exp(-gate[lane][i]))
                * up[lane][i];
            error.observe(actual[lane][i], expected);
            assert(superlane_near(
                actual[lane][i], expected, 2.5e-3f, 7.5e-3f));
        }
    }
    validate_superlane_lockstep(superlane);
    const auto end = std::chrono::steady_clock::now();
    return {"SwiGLU", superlane.lane(0).statistics(),
        std::chrono::duration<double, std::milli>(end - start).count(),
        error};
}

char superlane_trace_state(
    const ftlpu::VxmLane::CycleActivity& activity, std::size_t stage)
{
    const auto state = activity.states[stage];
    if (state == ftlpu::VxmLaneAluTraceState::Stalled) return 'S';
    if (state != ftlpu::VxmLaneAluTraceState::Executed) return '.';
    return activity.useful[stage] ? 'U' : 'B';
}

void print_superlane_result(std::ostream& os,
                            const SuperlaneWorkloadResult& result)
{
    const auto total = result.statistics.total();
    os << result.name
       << " cycles=" << total.cycles
       << " active=" << std::fixed << std::setprecision(2)
       << 100.0 * total.active_utilization() << "%"
       << " useful=" << 100.0 * total.useful_utilization() << "%"
       << " host_runtime=" << std::setprecision(3)
       << result.host_milliseconds << " ms\n";
    print_vxm_operator_error(os, result.error);
}

void print_superlane_trace(std::ostream& os,
                           const SuperlaneWorkloadResult& result)
{
    os << "\n[" << result.name << " Lane 0 cycle trace]\n"
       << "legend: U=useful B=bypass S=stalled .=idle\n"
       << "cycle depth ALU0..ALU7 active useful\n";
    for (const auto& activity : result.statistics.timeline) {
        os << activity.cycle << ' '
           << static_cast<std::size_t>(activity.chain_depth) << ' ';
        for (std::size_t stage = 0;
             stage < ftlpu::VxmLane::kAluCount; ++stage) {
            os << superlane_trace_state(activity, stage);
        }
        os << ' ' << activity.active_slots()
           << ' ' << activity.useful_slots() << '\n';
    }
}

void write_superlane_gantt(
    const std::filesystem::path& path,
    const std::array<SuperlaneWorkloadResult, 3>& results)
{
    auto file = std::ofstream(path, std::ios::trunc);
    assert(file && "failed to create Superlane Gantt HTML");
    file << R"HTML(<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VXM Superlane 流水线甘特图</title>
<style>
:root{--ink:#182338;--muted:#64748b;--panel:#fff;--bg:#f4f7fb;--blue:#3178e8;--green:#16a974;--orange:#f2993a;--red:#e45b5b;--grid:#dce4ef}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei",system-ui,sans-serif}
.wrap{max-width:1480px;margin:auto;padding:28px}.head{display:flex;justify-content:space-between;gap:24px;align-items:flex-end}
h1{margin:0;font-size:30px}.sub{color:var(--muted);margin-top:8px}.card{background:var(--panel);border:1px solid #e3e9f2;border-radius:14px;box-shadow:0 8px 24px #23395d12}
.controls{margin-top:22px;padding:16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
select,button,input{font:inherit} select,button{border:1px solid #cdd8e7;border-radius:8px;background:#fff;padding:8px 12px}
button.primary{background:var(--blue);color:#fff;border-color:var(--blue)} label{color:var(--muted)}
#summary{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;margin:14px 0}
.metric{padding:15px}.metric b{display:block;font-size:24px;margin-top:5px}.metric span{color:var(--muted);font-size:13px}
.chart{padding:18px;overflow:hidden}.legend{display:flex;gap:18px;margin-bottom:12px;color:var(--muted)}
.key{display:inline-flex;align-items:center;gap:6px}.sw{width:14px;height:14px;border-radius:3px}
#canvasWrap{overflow-x:auto;border:1px solid var(--grid);border-radius:10px;background:#fff}
canvas{display:block}.foot{color:var(--muted);font-size:13px;margin-top:12px}
@media(max-width:800px){#summary{grid-template-columns:1fr 1fr}.head{display:block}}
</style>
</head>
<body><div class="wrap">
<div class="head"><div><h1>VXM Superlane 流水线甘特图</h1>
<div class="sub">16 条 Lane 锁步执行；图中展示 Lane 0，控制状态代表整个 Superlane。</div></div>
<div class="sub">U 有效计算 · B Bypass · S Stall · 灰色 Idle</div></div>
<div class="controls card">
<label>算子 <select id="workload"></select></label>
<button id="play" class="primary">▶ 播放</button>
<button id="reset">回到起点</button>
<label>当前周期 <input id="cycle" type="range" min="0" value="0" step="1"></label>
<output id="cycleValue">0</output>
<label>每格宽度 <input id="zoom" type="range" min="2" max="18" value="7"></label>
</div>
<div id="summary"></div>
<div class="chart card">
<div class="legend">
<span class="key"><i class="sw" style="background:#16a974"></i>Useful</span>
<span class="key"><i class="sw" style="background:#f2993a"></i>Bypass</span>
<span class="key"><i class="sw" style="background:#e45b5b"></i>Stall</span>
<span class="key"><i class="sw" style="background:#edf1f6"></i>Idle</span>
</div>
<div id="canvasWrap"><canvas id="gantt"></canvas></div>
<div class="foot">蓝色竖线为播放光标；顶部色带标出当前周期的 2/4/8 级流水链配置。</div>
</div></div>
<script>
const workloads = [
)HTML";

    for (std::size_t result_index = 0;
         result_index < results.size(); ++result_index) {
        const auto& result = results[result_index];
        const auto total = result.statistics.total();
        if (result_index != 0) file << ",\n";
        file << "{\"name\":\"" << result.name
             << "\",\"cycles\":" << result.statistics.cycles
             << ",\"active\":" << std::fixed << std::setprecision(4)
             << 100.0 * total.active_utilization()
             << ",\"useful\":"
             << 100.0 * total.useful_utilization()
             << ",\"host\":" << result.host_milliseconds
             << ",\"timeline\":[";
        for (std::size_t cycle = 0;
             cycle < result.statistics.timeline.size(); ++cycle) {
            if (cycle != 0) file << ',';
            const auto& activity = result.statistics.timeline[cycle];
            file << "{\"d\":"
                 << static_cast<std::size_t>(activity.chain_depth)
                 << ",\"s\":\"";
            for (std::size_t stage = 0;
                 stage < ftlpu::VxmLane::kAluCount; ++stage) {
                file << superlane_trace_state(activity, stage);
            }
            file << "\"}";
        }
        file << "]}";
    }

    file << R"HTML(
];
const sel=document.querySelector('#workload'),play=document.querySelector('#play'),reset=document.querySelector('#reset');
const slider=document.querySelector('#cycle'),cycleValue=document.querySelector('#cycleValue'),zoom=document.querySelector('#zoom');
const canvas=document.querySelector('#gantt'),wrap=document.querySelector('#canvasWrap'),ctx=canvas.getContext('2d');
workloads.forEach((w,i)=>sel.add(new Option(w.name,i)));
let index=0,cursor=0,running=false,last=0;
function metric(label,value){return `<div class="metric card"><span>${label}</span><b>${value}</b></div>`}
function selectWorkload(){
 index=Number(sel.value);cursor=0;running=false;play.textContent='▶ 播放';
 slider.max=Math.max(0,workloads[index].cycles-1);slider.value=0;updateSummary();draw();
}
function updateSummary(){
 const w=workloads[index];
 document.querySelector('#summary').innerHTML=
  metric('总周期',w.cycles)+metric('Active占用率',w.active.toFixed(2)+'%')+
  metric('Useful占用率',w.useful.toFixed(2)+'%')+metric('宿主机运行时间',w.host.toFixed(3)+' ms');
}
function draw(){
 const w=workloads[index],cw=Number(zoom.value),label=78,row=38,top=34;
 const visible=Math.max(1,Math.floor((wrap.clientWidth-label)/cw));
 let start=Math.max(0,Math.floor(cursor-visible/2)); start=Math.min(start,Math.max(0,w.cycles-visible));
 const count=Math.min(visible,w.cycles-start);
 canvas.width=Math.max(wrap.clientWidth,label+count*cw);canvas.height=top+8*row+28;
 ctx.font='13px Microsoft YaHei';ctx.textBaseline='middle';ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);
 for(let stage=0;stage<8;stage++){const y=top+stage*row;ctx.fillStyle='#182338';ctx.fillText('ALU '+stage,14,y+row/2);
  for(let j=0;j<count;j++){const state=w.timeline[start+j].s[stage];ctx.fillStyle=state==='U'?'#16a974':state==='B'?'#f2993a':state==='S'?'#e45b5b':'#edf1f6';
   ctx.fillRect(label+j*cw,y,cw-1,row-3);}}
 let runStart=0;
 while(runStart<count){const depth=w.timeline[start+runStart].d;let runEnd=runStart+1;
  while(runEnd<count&&w.timeline[start+runEnd].d===depth)runEnd++;
  ctx.fillStyle=depth===2?'#25b8cc':depth===4?'#3178e8':'#7957d7';
  ctx.fillRect(label+runStart*cw,5,(runEnd-runStart)*cw-1,20);
  if((runEnd-runStart)*cw>32){ctx.fillStyle='#fff';ctx.textAlign='center';ctx.fillText(depth+'级',label+(runStart+runEnd)*cw/2,15);ctx.textAlign='left';}
  runStart=runEnd;}
 const x=label+(cursor-start)*cw;ctx.strokeStyle='#175cd3';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();
 ctx.fillStyle='#64748b';ctx.fillText('周期 '+start,label,canvas.height-12);ctx.fillText('周期 '+(start+count-1),Math.max(label,canvas.width-92),canvas.height-12);
 slider.value=cursor;cycleValue.value=cursor;
}
function frame(ts){if(!running)return;if(ts-last>70){cursor++;last=ts;if(cursor>=workloads[index].cycles){cursor=0;}draw();}requestAnimationFrame(frame)}
play.onclick=()=>{running=!running;play.textContent=running?'⏸ 暂停':'▶ 播放';if(running)requestAnimationFrame(frame)};
reset.onclick=()=>{cursor=0;draw()};slider.oninput=()=>{cursor=Number(slider.value);draw()};
zoom.oninput=draw;sel.onchange=selectWorkload;window.onresize=draw;selectWorkload();
</script></body></html>
)HTML";
}

} // namespace

inline void run_superlane_operator_workloads(
    std::ostream& summary,
    const std::filesystem::path& result_path,
    const std::filesystem::path& gantt_path)
{
    const auto results = std::array<SuperlaneWorkloadResult, 3>{
        test_superlane_rmsnorm(),
        test_superlane_softmax(),
        test_superlane_swiglu()};

    summary << "VXM Superlane workloads: 16 lanes x "
            << kSuperlaneTokensPerLane << " tokens/lane x "
            << kSuperlaneVectorLength << " elements\n";
    auto result_file = std::ofstream(result_path, std::ios::trunc);
    assert(result_file && "failed to create Superlane result file");
    result_file << "VXM Superlane workloads: 16 lanes x "
                << kSuperlaneTokensPerLane << " tokens/lane x "
                << kSuperlaneVectorLength << " elements\n";
    for (const auto& result : results) {
        print_superlane_result(summary, result);
        print_superlane_result(result_file, result);
        print_superlane_trace(result_file, result);
    }
    write_superlane_gantt(gantt_path, results);
}
