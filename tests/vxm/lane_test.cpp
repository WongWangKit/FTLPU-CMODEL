#include "ftlpu/vxm/lane.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

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
    for (std::size_t alu = 0; alu < ftlpu::VxmLane::kAluCount; ++alu) {
        assert(lane.queue_depth(alu) == 1);
    }

    const auto gate = static_cast<std::int32_t>(2);
    const auto up = static_cast<std::int32_t>(11);
    const auto gate_streams = ftlpu::VxmLane::pack_int32(gate);
    const auto up_streams = ftlpu::VxmLane::pack_int32(up);
    assert(ftlpu::VxmLane::unpack_int32(gate_streams) == gate);
    assert(ftlpu::VxmLane::unpack_int32(ftlpu::VxmLane::pack_int32(-1024)) == -1024);

    lane.set_swiglu_input(gate_streams, up_streams);
    for (std::size_t cycle = 0; cycle < 9; ++cycle) {
        lane.tick();
        assert(!lane.output().has_value());
    }

    lane.tick();
    assert(lane.output().has_value());
    assert(lane.output()->value == expected_swiglu(gate, up, params));
    assert(lane.cycle() == 10);

    auto source_select = ftlpu::VxmLane {};
    source_select.enqueue_instruction(0, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::StreamInt32(0),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float32,
    });
    source_select.enqueue_instruction(1, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::StreamInt32(4),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float32,
    });
    source_select.enqueue_instruction(2, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Alu(0),
        ftlpu::VxmLaneOperand::Imm(0.25f),
    });
    source_select.enqueue_instruction(3, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Alu(1),
        ftlpu::VxmLaneOperand::Imm(0.5f),
    });
    source_select.enqueue_instruction(4, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::Alu(2),
        ftlpu::VxmLaneOperand::Alu(3),
    });
    source_select.set_swiglu_input(ftlpu::VxmLane::pack_int32(8), ftlpu::VxmLane::pack_int32(6));
    source_select.tick();
    assert(source_select.last_trace()[0].state == ftlpu::VxmLaneAluTraceState::Executed);
    assert(source_select.last_trace()[1].state == ftlpu::VxmLaneAluTraceState::Executed);
    assert(source_select.last_trace()[2].state == ftlpu::VxmLaneAluTraceState::Stalled);
    assert(source_select.last_trace()[0].lhs.source == "stream[0..3]");
    assert(source_select.last_trace()[0].lhs.value.has_value());
    assert(*source_select.last_trace()[0].lhs.value == 8.0f);
    assert(source_select.last_trace()[2].lhs.source == "alu0");
    assert(!source_select.last_trace()[2].lhs.value.has_value());
    assert(source_select.alu_output(0).has_value());
    assert(source_select.alu_output(1).has_value());
    assert(nearly_equal(*source_select.alu_output(0), 8.0f));
    assert(nearly_equal(*source_select.alu_output(1), 6.0f));
    assert(!source_select.alu_output(2).has_value());
    source_select.tick();
    assert(source_select.last_trace_cycle() == 1);
    assert(source_select.last_trace()[0].state == ftlpu::VxmLaneAluTraceState::Idle);
    assert(source_select.last_trace()[2].state == ftlpu::VxmLaneAluTraceState::Executed);
    assert(source_select.last_trace()[3].state == ftlpu::VxmLaneAluTraceState::Executed);
    assert(source_select.last_trace()[4].state == ftlpu::VxmLaneAluTraceState::Stalled);
    assert(source_select.alu_output(2).has_value());
    assert(source_select.alu_output(3).has_value());
    assert(nearly_equal(*source_select.alu_output(2), 2.0f));
    assert(nearly_equal(*source_select.alu_output(3), 3.0f));
    source_select.tick();
    assert(source_select.last_trace()[4].state == ftlpu::VxmLaneAluTraceState::Executed);
    assert(source_select.alu_output(4).has_value());
    assert(nearly_equal(*source_select.alu_output(4), 5.0f));

    auto fp16_output = ftlpu::VxmLane {};
    fp16_output.enqueue_instruction(0, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::StreamInt32(0),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float16,
        9,
    });
    fp16_output.set_swiglu_input(ftlpu::VxmLane::pack_int32(3), ftlpu::VxmLane::pack_int32(0));
    fp16_output.tick();
    assert(fp16_output.output().has_value());
    assert(fp16_output.output()->stream == 9);
    assert(fp16_output.output()->byte_count == 2);
    const auto half_three = ftlpu::VxmAlu::cast_scalar_to_float16_bits(3.0f);
    assert(fp16_output.output()->bytes[0] == static_cast<std::uint8_t>(half_three & 0xffu));
    assert(fp16_output.output()->bytes[1] == static_cast<std::uint8_t>((half_three >> 8) & 0xffu));

    auto trace_lane = ftlpu::VxmLane {};
    trace_lane.load_swiglu_program(params);
    trace_lane.set_swiglu_input(ftlpu::VxmLane::pack_int32(2), ftlpu::VxmLane::pack_int32(1));
    trace_lane.tick();
    assert(trace_lane.alu_output(0).has_value());
    assert(trace_lane.alu_output(1).has_value());
    assert(nearly_equal(*trace_lane.alu_output(0), 2.0f));
    assert(nearly_equal(*trace_lane.alu_output(1), 1.0f));
    trace_lane.tick();
    assert(trace_lane.alu_output(2).has_value());
    assert(trace_lane.alu_output(3).has_value());
    assert(nearly_equal(*trace_lane.alu_output(2), 0.5f));
    assert(nearly_equal(*trace_lane.alu_output(3), 0.5f));
    trace_lane.tick();
    assert(trace_lane.alu_output(5).has_value());
    assert(nearly_equal(*trace_lane.alu_output(5), -0.5f));
    trace_lane.tick();
    assert(trace_lane.alu_output(6).has_value());
    assert(nearly_equal(*trace_lane.alu_output(6), std::exp(-0.5f)));

    auto saturated = ftlpu::VxmLane {};
    saturated.load_swiglu_program(params);
    saturated.set_swiglu_input(ftlpu::VxmLane::pack_int32(64), ftlpu::VxmLane::pack_int32(1024));
    for (std::size_t cycle = 0; cycle < 10; ++cycle) {
        saturated.tick();
    }
    assert(saturated.output().has_value());
    assert(saturated.output()->value == 127);

    auto pipelined = ftlpu::VxmLane {};
    const std::vector<std::int32_t> pipeline_gates {2, 4, -3, 8, 1};
    const std::vector<std::int32_t> pipeline_ups {11, 7, 5, -9, 13};
    pipelined.load_pipelined_swiglu_program(params, pipeline_gates.size());
    for (std::size_t alu = 0; alu < ftlpu::VxmLane::kAluCount; ++alu) {
        assert(pipelined.queue_depth(alu) == pipeline_gates.size());
    }

    std::vector<std::int8_t> pipeline_outputs {};
    std::vector<std::size_t> pipeline_output_cycles {};
    for (std::size_t cycle = 0; cycle < pipeline_gates.size() + 9; ++cycle) {
        if (cycle < pipeline_gates.size()) {
            pipelined.set_swiglu_input(
                ftlpu::VxmLane::pack_int32(pipeline_gates[cycle]),
                ftlpu::VxmLane::pack_int32(pipeline_ups[cycle]));
        }

        pipelined.tick();
        if (pipelined.output().has_value()) {
            pipeline_outputs.push_back(pipelined.output()->value);
            pipeline_output_cycles.push_back(cycle);
        }
    }

    assert(pipeline_outputs.size() == pipeline_gates.size());
    for (std::size_t token = 0; token < pipeline_gates.size(); ++token) {
        assert(pipeline_output_cycles[token] == token + 9);
        assert(pipeline_outputs[token] == expected_swiglu(pipeline_gates[token], pipeline_ups[token], params));
    }

    return 0;
}
