#include "ftlpu/vxm/slice.hpp"
#include "hardware_test_output.hpp"
#include "hardware_timing_report.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void put_int32(ftlpu::VxmLane::StreamBytes& streams, std::size_t base,
               std::int32_t value)
{
    const auto bytes =
        ftlpu::VxmLane::pack_float16(static_cast<float>(value));
    for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
        streams[base + byte] = bytes[byte];
    }
}

void tick_with_scheduled_inputs(
    ftlpu::VxmSlice& slice,
    const ftlpu::VxmSlice::StreamMatrix& inputs)
{
    slice.prepare_cycle();
    for (std::size_t tile = 0;
         tile < ftlpu::VxmSlice::kTileCount;
         ++tile) {
        if (slice.required_streams_at(tile)
            && slice.input_buffer(tile).empty()) {
            slice.set_stream_inputs(tile, inputs);
        }
    }
    slice.tick();
}

void test_feedback_2_to_8()
{
    using namespace ftlpu;

    auto slice = VxmSlice{};
    slice.set_chain_depth(VxmChainDepth::Two);
    auto trace =
        std::vector<vxm_hardware_test::TimingCycle>{};
    const auto record =
        [&slice, &trace](std::string event) {
            trace.push_back(vxm_hardware_test::capture_cycle(
                slice.cycle() - 1,
                slice.superlane(0),
                slice.outputs_at(0).size(),
                std::move(event)));
        };

    // Two compact packets are sent on every logical Slice instruction
    // channel. The old depth-2 configuration produces four values per
    // physical 8-ALU block; the following depth-8 configuration consumes
    // those values through the fixed Feedback network.
    for (std::size_t stage = 0; stage < VxmSlice::kAluQueues; ++stage) {
        auto producer = stage % 2 == 0
            ? VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::StreamFloat16()}
            : VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous()};
        slice.issue_south(
            stage,
            VxmCompactInstructionCodec::encode(
                stage, VxmChainDepth::Two, producer));

        auto consumer = stage == 0
            ? VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Feedback()}
            : VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous()};
        consumer.repeat_count = 4;
        if (stage == 7) {
            consumer.output_stream = 6;
            consumer.output_type = VxmCastTarget::Float32;
        }
        slice.issue_south(
            stage,
            VxmCompactInstructionCodec::encode(
                stage, VxmChainDepth::Eight, consumer));
    }

    slice.tick();
    // Old packets reach Superlane 0 and enter decode.
    record("config decode");

    auto inputs = VxmSlice::StreamMatrix{};
    for (std::size_t lane = 0;
         lane < VxmSuperlane::kLaneCount; ++lane) {
        for (std::size_t head = 0;
             head < VxmLane::kAluCount; head += 2) {
            const auto value = static_cast<std::int32_t>(
                lane * 100 + (head / 2 + 1) * 10);
            const auto group =
                VxmLane::fixed_input_group_for_stage(head, false);
            put_int32(
                inputs[lane],
                group * VxmLane::kStreamGroupBytes,
                value);
        }
    }
    tick_with_scheduled_inputs(slice, inputs);
    // Old C0/C2/C4/C6 heads execute; new packets decode.
    record("input heads execute");

    slice.request_chain_depth_transition(
        0, VxmChainDepth::Eight);
    tick_with_scheduled_inputs(slice, inputs);
    // Old tails retire and are captured at the same edge.
    record("depth transition");

    assert(slice.instruction_at(0, 2));
    const auto propagated_feedback =
        VxmCompactInstructionCodec::decode(
            0, *slice.instruction_at(0, 2));
    assert(propagated_feedback.chain_depth == VxmChainDepth::Eight);
    assert(propagated_feedback.instruction.lhs.kind
           == VxmLaneOperandKind::Feedback);

    const auto& lane0 = slice.superlane(0).lane(0);
    assert(lane0.chain_depth() == VxmChainDepth::Eight);
    assert(lane0.last_feedback_capture_count() == 8);
    for (const auto tail : {1U, 3U, 5U, 7U}) {
        assert(lane0.last_trace()[tail].state
               == VxmLaneAluTraceState::Executed);
    }

    tick_with_scheduled_inputs(slice, inputs);
    // No bubble: new C0 consumes the first C1 result.
    record("feedback enters C0");
    assert(slice.superlane(0).lane(0).last_trace()[0].state
           == VxmLaneAluTraceState::Executed);

    for (std::size_t cycle = 0; cycle < 6; ++cycle) {
        tick_with_scheduled_inputs(slice, inputs);
        record("pipeline");
        assert(slice.outputs_at(0).empty());
    }
    for (std::size_t item = 0; item < 4; ++item) {
        tick_with_scheduled_inputs(slice, inputs);
        record("pipeline output");
        const auto& outputs = slice.outputs_at(0);
        assert(outputs.size() == 2);
        assert(outputs[0].stream == 6);
        assert(outputs[1].stream == 14);
        for (std::size_t lane = 0;
             lane < VxmSuperlane::kLaneCount; ++lane) {
            const auto expected =
                static_cast<float>(lane * 100 + (item + 1) * 10);
            assert(VxmLane::unpack_float32(
                       outputs[0].byte_values[lane])
                   == expected);
        }
    }
    vxm_hardware_test::write_timing_reports(
        "slice", "VXM Slice complete timing sequence", trace);
}

void test_missing_bundle_is_schedule_error()
{
    using namespace ftlpu;

    auto slice = VxmSlice{};
    slice.set_chain_depth(VxmChainDepth::Two);
    const auto stream_head = VxmLaneAluInstruction{
        VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamFloat16()};
    slice.issue_south(
        0,
        VxmCompactInstructionCodec::encode(
            0, VxmChainDepth::Two, stream_head));

    slice.tick(); // packet enters local decode
    auto schedule_error = false;
    try {
        slice.tick(); // compiler failed to make the Bundle ready for issue
    } catch (const std::logic_error&) {
        schedule_error = true;
    }
    assert(schedule_error);
}

}

int main()
{
    using namespace ftlpu;
    auto slice = VxmSlice{};
    slice.set_chain_depth(VxmChainDepth::Two);

    const auto head = VxmLaneAluInstruction{VxmAluOpcode::Add,
        VxmLaneOperand::StreamFloat16(),
        VxmLaneOperand::StreamFloat16()};
    slice.issue_south(0, VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Two, head));
    auto tail = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    tail.output_type = VxmCastTarget::Int8;
    tail.output_scale = 1.0f;
    tail.output_stream = 0;
    slice.issue_south(1, VxmCompactInstructionCodec::encode(
        1, VxmChainDepth::Two, tail));

    auto input = VxmSlice::StreamMatrix{};
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        put_int32(input[lane], 0, static_cast<std::int32_t>(lane));
        put_int32(input[lane], 2, 10);
    }
    slice.prepare_cycle();
    assert(slice.required_streams_at(0));
    for (std::size_t stream = 0; stream < 4; ++stream) {
        assert((*slice.required_streams_at(0))[stream]);
    }

    slice.tick();
    // Instruction reaches tile 0 and spends one cycle decoding.
    assert(!slice.output_at(0));
    tick_with_scheduled_inputs(slice, input);
    assert(!slice.output_at(0));
    tick_with_scheduled_inputs(slice, input);
    assert(!slice.output_at(0));

    // The compact instruction row propagates one Superlane per cycle; each
    // destination decodes it locally. The INT8 result is still in its fixed
    // one-cycle tail quantizer here.
    assert(slice.instruction_at(0, 3).has_value());
    assert(slice.cycle() == 3);

    tick_with_scheduled_inputs(slice, input);
    assert(slice.output_at(0));
    assert(slice.output_at(0)->stream == 0);
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        assert(slice.output_at(0)->values[lane] == static_cast<std::int8_t>(lane + 10));
    }

    assert(slice.cycle() == 4);

    // A repeat-count Current Config remains the source of stream requirements
    // after its one instruction packet has moved away from this tile.
    auto repeated = VxmSlice{};
    repeated.set_chain_depth(VxmChainDepth::Two);
    auto repeated_head = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamFloat16()};
    repeated_head.repeat_count = 2;
    repeated.issue_south(0, VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Two, repeated_head));
    auto repeated_tail = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    repeated_tail.output_type = VxmCastTarget::Float32;
    repeated_tail.output_stream = 0;
    repeated_tail.repeat_count = 2;
    repeated.issue_south(1, VxmCompactInstructionCodec::encode(
        1, VxmChainDepth::Two, repeated_tail));

    auto repeated_input = VxmSlice::StreamMatrix{};
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        put_int32(repeated_input[lane], 0, static_cast<std::int32_t>(lane));
    }
    repeated.tick();
    // Instruction reaches tile 0 and starts decoding.
    repeated.prepare_cycle();
    assert(repeated.required_streams_at(0));
    for (std::size_t stream = 0; stream < 2; ++stream) {
        assert((*repeated.required_streams_at(0))[stream]);
    }
    tick_with_scheduled_inputs(repeated, repeated_input);
    tick_with_scheduled_inputs(repeated, repeated_input);
    assert(repeated.output_at(0));
    tick_with_scheduled_inputs(repeated, repeated_input);
    assert(repeated.output_at(0));
    test_feedback_2_to_8();
    test_missing_bundle_is_schedule_error();
    vxm_hardware_test::write_pass_result(
        "slice_test_results.txt", "slice_test");
    return 0;
}
