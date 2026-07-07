#include "ftlpu/vxm/lane.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

bool nearly_equal(float a, float b, float eps = 1.0e-5f)
{
    return std::fabs(a - b) <= eps;
}

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

} // namespace

int main()
{
    const ftlpu::VxmLane::SwigluParams params {
        0.25f,
        0.5f,
        0.125f,
        0,
    };

    auto lane = ftlpu::VxmLane {};
    lane.load_swiglu_program(params);

    const auto gate = static_cast<std::int32_t>(2);
    const auto up = static_cast<std::int32_t>(11);
    const auto gate_streams = ftlpu::VxmLane::pack_int32(gate);
    const auto up_streams = ftlpu::VxmLane::pack_int32(up);
    assert(ftlpu::VxmLane::unpack_int32(gate_streams) == gate);
    assert(ftlpu::VxmLane::unpack_int32(ftlpu::VxmLane::pack_int32(-1024)) == -1024);

    lane.set_swiglu_input(gate_streams, up_streams);
    for (std::size_t cycle = 0; cycle < ftlpu::VxmLane::kAluStages - 1; ++cycle) {
        lane.tick();
        assert(!lane.output().has_value());
    }

    lane.tick();
    assert(lane.output().has_value());
    assert(lane.output()->value == expected_swiglu(gate, up, params));
    assert(lane.cycle() == ftlpu::VxmLane::kAluStages);

    auto pipelined = ftlpu::VxmLane {};
    pipelined.load_swiglu_program(params);
    pipelined.set_swiglu_input(ftlpu::VxmLane::pack_int32(0), ftlpu::VxmLane::pack_int32(4));
    pipelined.tick();
    pipelined.set_swiglu_input(ftlpu::VxmLane::pack_int32(4), ftlpu::VxmLane::pack_int32(8));
    for (std::size_t cycle = 1; cycle < ftlpu::VxmLane::kAluStages; ++cycle) {
        pipelined.tick();
    }
    assert(pipelined.output().has_value());
    assert(pipelined.output()->value == expected_swiglu(0, 4, params));
    pipelined.tick();
    assert(pipelined.output().has_value());
    assert(pipelined.output()->value == expected_swiglu(4, 8, params));

    auto trace_lane = ftlpu::VxmLane {};
    trace_lane.load_swiglu_program(params);
    trace_lane.set_swiglu_input(ftlpu::VxmLane::pack_int32(2), ftlpu::VxmLane::pack_int32(1));
    trace_lane.tick();
    assert(trace_lane.stage_value(1).has_value());
    assert(nearly_equal(*trace_lane.stage_value(1), -0.5f));
    trace_lane.tick();
    assert(trace_lane.stage_value(2).has_value());
    assert(nearly_equal(*trace_lane.stage_value(2), std::exp(-0.5f)));

    auto saturated = ftlpu::VxmLane {};
    saturated.load_swiglu_program(params);
    saturated.set_swiglu_input(ftlpu::VxmLane::pack_int32(64), ftlpu::VxmLane::pack_int32(1024));
    for (std::size_t cycle = 0; cycle < ftlpu::VxmLane::kAluStages; ++cycle) {
        saturated.tick();
    }
    assert(saturated.output().has_value());
    assert(saturated.output()->value == 127);

    return 0;
}
