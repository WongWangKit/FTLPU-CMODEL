#include "ftlpu/vxm/lane.hpp"
#include "ftlpu/vxm/superlane.hpp"
#include "hardware_test_output.hpp"

#include <array>
#include <cassert>
#include <cstddef>

int main()
{
    using namespace ftlpu;

    static_assert(VxmLane::kAluCount == 16);
    static_assert(VxmLane::kBlockCount == 8);
    static_assert(VxmLane::kStreamGroupBytes == 2);
    static_assert(VxmLane::kStreamGroupCount == 16);

    auto lane = VxmLane{};
    lane.set_chain_depth(VxmChainDepth::Two);

    auto config = VxmLane::Configs{};
    for (std::size_t head = 0; head < VxmLane::kAluCount; head += 2) {
        config[head] = VxmLaneAluInstruction{
            VxmAluOpcode::Add,
            VxmLaneOperand::StreamFloat16(),
            VxmLaneOperand::StreamFloat16()};
        auto tail = VxmLaneAluInstruction{
            VxmAluOpcode::Bypass,
            VxmLaneOperand::Previous()};
        tail.output_type = VxmCastTarget::Float16;
        tail.output_stream =
            VxmLane::fixed_output_stream_for_block(head / 2);
        config[head + 1] = tail;
    }

    auto streams = VxmLane::StreamBytes{};
    for (std::size_t head = 0; head < VxmLane::kAluCount; head += 2) {
        const auto chain = head / 2;
        const auto lhs = VxmLane::pack_float16(
            static_cast<float>(chain + 1));
        const auto rhs = VxmLane::pack_float16(
            static_cast<float>((chain + 1) * 10));
        const auto lhs_base =
            VxmLane::fixed_input_group_for_stage(head, false)
            * VxmLane::kStreamGroupBytes;
        const auto rhs_base =
            VxmLane::fixed_input_group_for_stage(head, true)
            * VxmLane::kStreamGroupBytes;
        for (std::size_t byte = 0;
             byte < VxmLane::kStreamGroupBytes; ++byte) {
            streams[lhs_base + byte] = lhs[byte];
            streams[rhs_base + byte] = rhs[byte];
        }
    }

    lane.set_stream_inputs(streams);
    lane.tick(config);
    assert(lane.outputs().empty());

    // A second vector enters while all eight tails process the first vector:
    // after fill, all 16 ALUs are active in the same cycle.
    lane.set_stream_inputs(streams);
    lane.tick(config);
    assert(lane.outputs().size() == 8);
    assert(lane.statistics().timeline.back().active_slots() == 16);

    for (std::size_t chain = 0; chain < 8; ++chain) {
        const auto& output = lane.outputs()[chain];
        assert(output.stream == chain * VxmLane::kStreamGroupBytes);
        assert(output.byte_count == 2);
        assert(VxmLane::unpack_float16(
                   {output.bytes[0], output.bytes[1]})
               == static_cast<float>((chain + 1) * 11));
    }

    lane.tick(config);
    assert(lane.outputs().size() == 8);

    // All eight fixed output blocks also own independent optional static
    // quantizers. With two-stage chains, eight INT8 values retire together
    // one cycle after the corresponding ALU tails.
    auto quantized = VxmLane{};
    quantized.set_chain_depth(VxmChainDepth::Two);
    auto quantized_config = config;
    for (std::size_t tail = 1; tail < VxmLane::kAluCount; tail += 2) {
        quantized_config[tail]->output_type = VxmCastTarget::Int8;
        quantized_config[tail]->output_scale = 1.0f;
    }
    quantized.set_stream_inputs(streams);
    quantized.tick(quantized_config);
    assert(quantized.outputs().empty());
    quantized.tick(quantized_config);
    assert(quantized.outputs().empty());
    quantized.tick(quantized_config);
    assert(quantized.outputs().size() == VxmLane::kBlockCount);
    for (std::size_t block = 0; block < VxmLane::kBlockCount; ++block) {
        assert(quantized.outputs()[block].stream
            == VxmLane::fixed_output_stream_for_block(block));
    }

    // Superlane-wide saturation: every lane executes all eight two-stage
    // chains, so the fixed output bank retires 8 x lane_count INT8 values in
    // one cycle. This is the physical peak used by the architecture.
    auto saturated = VxmSuperlane{};
    saturated.set_chain_depth(VxmChainDepth::Two);
    for (std::size_t head = 0;
         head < VxmSuperlaneInstructionControl::kStageCount;
         head += 2) {
        auto saturated_head = VxmLaneAluInstruction{
            VxmAluOpcode::Add,
            VxmLaneOperand::StreamFloat16(),
            VxmLaneOperand::StreamFloat16()};
        saturated_head.repeat_count = 2;
        saturated.enqueue_instruction(head, saturated_head);
        auto saturated_tail = VxmLaneAluInstruction{
            VxmAluOpcode::Bypass,
            VxmLaneOperand::Previous()};
        saturated_tail.output_stream =
            VxmLane::fixed_output_stream_for_block(head / 2);
        saturated_tail.output_type = VxmCastTarget::Int8;
        saturated_tail.output_scale = 1.0f;
        saturated_tail.repeat_count = 2;
        saturated.enqueue_instruction(head + 1, saturated_tail);
    }
    saturated.tick(); // shared Superlane decode start
    auto saturated_streams = VxmSuperlane::StreamMatrix{};
    for (std::size_t lane_index = 0;
         lane_index < VxmSuperlane::kLaneCount; ++lane_index) {
        saturated_streams[lane_index] = streams;
    }
    saturated.set_stream_inputs(saturated_streams);
    saturated.tick();
    saturated.set_stream_inputs(saturated_streams);
    saturated.tick();
    assert(saturated.outputs().empty());
    saturated.tick();
    assert(saturated.outputs().size() == VxmLane::kBlockCount);
    auto retired_values = std::size_t{0};
    for (const auto& output : saturated.outputs()) {
        assert(output.byte_count == 1);
        retired_values += output.values.size();
    }
    assert(retired_values
        == VxmLane::kBlockCount * VxmSuperlane::kLaneCount);
    vxm_hardware_test::write_pass_result(
        "fp16_16alu_test_results.txt", "fp16_16alu_test");
    return 0;
}
