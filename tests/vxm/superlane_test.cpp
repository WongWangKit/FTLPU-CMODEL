#include "ftlpu/vxm/superlane.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

std::int8_t expected_swiglu(
    std::int32_t gate,
    std::int32_t up,
    const ftlpu::VxmLane::SwigluParams& params)
{
    const auto gate_fp32 = static_cast<float>(gate) * params.gate_scale;
    const auto up_fp32 = static_cast<float>(up) * params.up_scale;
    const auto sigmoid = 1.0f / (1.0f + std::exp(-gate_fp32));
    const auto product = gate_fp32 * sigmoid * up_fp32;
    return ftlpu::VxmAlu::quantize_scalar(product, params.output_scale, params.output_zero_point);
}

ftlpu::VxmSuperlane::Int32Vector make_vector(std::int32_t base, std::int32_t stride)
{
    auto out = ftlpu::VxmSuperlane::Int32Vector {};
    for (std::size_t lane = 0; lane < out.size(); ++lane) {
        out[lane] = base + static_cast<std::int32_t>(lane) * stride;
    }
    return out;
}

} // namespace

int main()
{
    const ftlpu::VxmLane::SwigluParams params {
        0.25f,
        0.5f,
        0.125f,
        0,
    };

    auto superlane = ftlpu::VxmSuperlane {};
    const auto gates = make_vector(2, 1);
    const auto ups = make_vector(11, -1);
    superlane.load_swiglu_program(params);
    superlane.set_swiglu_inputs(gates, ups);
    for (std::size_t cycle = 0; cycle < 9; ++cycle) {
        superlane.tick();
        assert(!superlane.output().has_value());
    }
    superlane.tick();
    assert(superlane.output().has_value());
    for (std::size_t lane = 0; lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        assert(superlane.output()->values[lane] == expected_swiglu(gates[lane], ups[lane], params));
    }

    auto pipelined = ftlpu::VxmSuperlane {};
    constexpr std::size_t kTokens = 5;
    pipelined.load_pipelined_swiglu_program(params, kTokens);

    std::vector<ftlpu::VxmSuperlane::Int32Vector> token_gates {};
    std::vector<ftlpu::VxmSuperlane::Int32Vector> token_ups {};
    for (std::size_t token = 0; token < kTokens; ++token) {
        token_gates.push_back(make_vector(static_cast<std::int32_t>(2 + token * 3), 1));
        token_ups.push_back(make_vector(static_cast<std::int32_t>(11 - token), -1));
    }

    std::vector<ftlpu::VxmSuperlane::Int8Vector> outputs {};
    std::vector<std::size_t> output_cycles {};
    for (std::size_t cycle = 0; cycle < kTokens + 9; ++cycle) {
        if (cycle < kTokens) {
            pipelined.set_swiglu_inputs(token_gates[cycle], token_ups[cycle]);
        }

        pipelined.tick();
        if (cycle == 0) {
            assert(pipelined.lane(0).last_trace()[0].lhs.source == "stream[0..3]");
            assert(pipelined.lane(15).last_trace()[0].lhs.source == "stream[0..3]");
        }
        if (pipelined.output().has_value()) {
            outputs.push_back(pipelined.output()->values);
            output_cycles.push_back(cycle);
        }
    }

    assert(outputs.size() == kTokens);
    for (std::size_t token = 0; token < kTokens; ++token) {
        assert(output_cycles[token] == token + 9);
        for (std::size_t lane = 0; lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
            assert(outputs[token][lane] == expected_swiglu(token_gates[token][lane], token_ups[token][lane], params));
        }
    }

    return 0;
}
