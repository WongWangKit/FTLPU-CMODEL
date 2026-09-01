#include "ftlpu/vxm/lane.hpp"
#include "hardware_test_output.hpp"
#include "hardware_timing_report.hpp"
#include "lane_test_driver.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Fn>
bool throws(Fn fn)
{
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

void put_int32(ftlpu::VxmLane::StreamBytes& streams, std::size_t base,
               std::int32_t value)
{
    const auto bytes =
        ftlpu::VxmLane::pack_float16(static_cast<float>(value));
    for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
        streams[base + byte] = bytes[byte];
    }
}

ftlpu::VxmLaneAluInstruction output_bypass(std::size_t stream,
                                           ftlpu::VxmCastTarget type = ftlpu::VxmCastTarget::Float32)
{
    auto instruction = ftlpu::VxmLaneAluInstruction{
        ftlpu::VxmAluOpcode::Bypass, ftlpu::VxmLaneOperand::Previous()};
    instruction.output_stream = stream;
    instruction.output_type = type;
    return instruction;
}

float run_fixed_feedback_path(
    ftlpu::VxmChainDepth depth, std::size_t head)
{
    using namespace ftlpu;
    auto lane = VxmLaneTestDriver{};
    lane.set_chain_depth(depth);
    const auto length = static_cast<std::size_t>(depth);
    const auto tail = head + length - 1;

    for (std::size_t stage = head; stage <= tail; ++stage) {
        lane.enqueue_instruction(stage, stage == head
            ? VxmLaneAluInstruction{
                VxmAluOpcode::Add,
                VxmLaneOperand::StreamFloat16(),
                VxmLaneOperand::Imm(2.0f)}
            : VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous()});

        auto second = stage == head
            ? VxmLaneAluInstruction{
                VxmAluOpcode::Add,
                VxmLaneOperand::Feedback(),
                VxmLaneOperand::Imm(3.0f)}
            : VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous()};
        if (stage == tail) {
            second.output_stream =
                lane.fixed_output_stream_for_stage(stage);
            second.output_type = VxmCastTarget::Float32;
        }
        lane.enqueue_instruction(stage, second);
    }

    lane.tick(); // initial decode
    auto input = VxmLane::StreamBytes{};
    const auto input_base =
        VxmLane::fixed_input_group_for_stage(head, false)
        * VxmLane::kStreamGroupBytes;
    put_int32(input, input_base, 5);
    lane.set_stream_inputs(input);
    lane.tick();
    for (std::size_t stage = 1; stage < length; ++stage) {
        lane.tick();
    }
    assert(lane.last_trace()[tail].result
           && *lane.last_trace()[tail].result == 7.0f);

    lane.tick(); // selected tail writes the existing head input register
    assert(lane.last_trace()[head].result
           && *lane.last_trace()[head].result == 10.0f);
    for (std::size_t stage = 1; stage < length; ++stage) {
        lane.tick();
    }
    assert(lane.output());
    return VxmLane::unpack_float32(lane.output()->bytes);
}

float run_merged_feedback_path(
    ftlpu::VxmChainDepth depth,
    std::size_t producer_head,
    std::size_t consumer_head)
{
    using namespace ftlpu;
    auto lane = VxmLaneTestDriver{};
    lane.set_chain_depth(depth);
    const auto length = static_cast<std::size_t>(depth);
    const auto producer_tail = producer_head + length - 1;
    const auto consumer_tail = consumer_head + length - 1;

    for (std::size_t stage = producer_head;
         stage <= producer_tail; ++stage) {
        lane.enqueue_instruction(stage, stage == producer_head
            ? VxmLaneAluInstruction{
                VxmAluOpcode::Add,
                VxmLaneOperand::StreamFloat16(),
                VxmLaneOperand::Imm(2.0f)}
            : VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous()});
    }
    for (std::size_t stage = consumer_head;
         stage <= consumer_tail; ++stage) {
        auto instruction = stage == consumer_head
            ? VxmLaneAluInstruction{
                VxmAluOpcode::Add,
                VxmLaneOperand::Feedback(),
                VxmLaneOperand::Imm(3.0f)}
            : VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous()};
        if (stage == consumer_tail) {
            instruction.output_stream =
                lane.fixed_output_stream_for_stage(stage);
            instruction.output_type = VxmCastTarget::Float32;
        }
        lane.enqueue_instruction(stage, instruction);
    }

    lane.tick();
    auto input = VxmLane::StreamBytes{};
    const auto input_base =
        VxmLane::fixed_input_group_for_stage(producer_head, false)
        * VxmLane::kStreamGroupBytes;
    put_int32(input, input_base, 5);
    lane.set_stream_inputs(input);
    lane.tick();
    for (std::size_t stage = 1; stage < length; ++stage) {
        lane.tick();
    }
    assert(lane.last_trace()[producer_tail].result
           && *lane.last_trace()[producer_tail].result == 7.0f);
    lane.tick();
    assert(lane.last_trace()[consumer_head].result
           && *lane.last_trace()[consumer_head].result == 10.0f);
    for (std::size_t stage = 1; stage < length; ++stage) {
        lane.tick();
    }
    assert(lane.output());
    return VxmLane::unpack_float32(lane.output()->bytes);
}

}

int main()
{
    using namespace ftlpu;
    static_assert(VxmLane::kAluCount == 16);
    static_assert(VxmLane::kBlockCount == 8);
    static_assert(VxmLane::kInputStreams == 32);
    static_assert(VxmLane::fixed_input_group_for_stage(0, false) == 0);
    static_assert(VxmLane::fixed_input_group_for_stage(0, true) == 1);
    static_assert(VxmLane::fixed_input_group_for_stage(2, false) == 2);
    static_assert(VxmLane::fixed_input_group_for_stage(2, true) == 3);
    static_assert(VxmLane::fixed_input_group_for_stage(4, false) == 4);
    static_assert(VxmLane::fixed_input_group_for_stage(4, true) == 5);
    static_assert(VxmLane::fixed_input_group_for_stage(6, false) == 6);
    static_assert(VxmLane::fixed_input_group_for_stage(6, true) == 7);
    static_assert(VxmLane::has_fixed_feedback_path(1, 0));
    static_assert(VxmLane::has_fixed_feedback_path(3, 2));
    static_assert(VxmLane::has_fixed_feedback_path(5, 4));
    static_assert(VxmLane::has_fixed_feedback_path(7, 6));
    static_assert(VxmLane::has_fixed_feedback_path(3, 0));
    static_assert(VxmLane::has_fixed_feedback_path(7, 4));
    static_assert(VxmLane::has_fixed_feedback_path(7, 0));
    static_assert(VxmLane::has_fixed_feedback_path(5, 0));
    static_assert(!VxmLane::has_fixed_feedback_path(1, 6));

    auto topology = VxmLane{};
    topology.set_chain_depth(VxmChainDepth::Two);
    assert(topology.chain_count() == 8);
    assert(topology.is_chain_head(0) && topology.is_chain_tail(1));
    assert(topology.is_chain_head(6) && topology.is_chain_tail(7));
    topology.set_chain_depth(VxmChainDepth::Four);
    assert(topology.chain_count() == 4);
    assert(topology.is_chain_head(4) && topology.is_chain_tail(7));
    topology.set_chain_depth(VxmChainDepth::Eight);
    assert(topology.chain_count() == 2);
    assert(topology.is_chain_head(0) && topology.is_chain_tail(7));

    const auto explicit_group = VxmLaneOperand::StreamFloat16(1.0f, 3);
    assert(VxmLane::input_group_for_operand(0, false, explicit_group) == 3);
    assert(VxmLane::input_group_for_operand(8, false, explicit_group) == 11);
    const auto east_group = VxmLaneOperand::StreamFloat16(
        1.0f, 3, VxmStreamSource::East);
    const auto west_group = VxmLaneOperand::StreamFloat16(
        1.0f, 3, VxmStreamSource::West);
    assert(VxmLane::input_group_for_operand(0, false, east_group) == 3);
    assert(VxmLane::input_group_for_operand(8, false, east_group) == 3);
    assert(VxmLane::input_group_for_operand(0, false, west_group) == 11);
    assert(VxmLane::input_group_for_operand(8, false, west_group) == 11);

    auto selected_group = VxmLaneTestDriver{};
    selected_group.set_chain_depth(VxmChainDepth::Two);
    selected_group.enqueue_instruction(0, {VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamFloat16(1.0f, 3)});
    selected_group.enqueue_instruction(1, output_bypass(0));
    selected_group.tick();
    auto selected_streams = VxmLane::StreamBytes{};
    put_int32(selected_streams, 0, 2);
    put_int32(selected_streams,
        3 * VxmLane::kStreamGroupBytes, 9);
    selected_group.set_stream_inputs(selected_streams);
    selected_group.tick();
    selected_group.tick();
    assert(selected_group.output());
    assert(VxmLane::unpack_float32(
        selected_group.output()->bytes) == 9.0f);

    // RoPE-style direct fusion. The chain head captures both its own input
    // pair and the following FMA/FMS stream pair in the same row cycle. The
    // captured operands stay aligned while the two fully-pipelined
    // multipliers advance consecutive rows.
    auto fused_rope = VxmLaneTestDriver{};
    fused_rope.set_chain_depth(VxmChainDepth::Two);
    auto low_product = VxmLaneAluInstruction{
        VxmAluOpcode::Multiply,
        VxmLaneOperand::StreamFloat16(
            1.0f, 0, VxmStreamSource::East),
        VxmLaneOperand::StreamFloat16(
            1.0f, 4, VxmStreamSource::East)};
    low_product.repeat_count = 2;
    fused_rope.enqueue_instruction(0, low_product);
    auto low_rotate = VxmLaneAluInstruction{
        VxmAluOpcode::FusedMultiplySubtract,
        VxmLaneOperand::StreamFloat16(
            1.0f, 0, VxmStreamSource::West),
        VxmLaneOperand::StreamFloat16(
            1.0f, 5, VxmStreamSource::East)};
    low_rotate.output_stream = 0;
    low_rotate.output_type = VxmCastTarget::Float32;
    low_rotate.repeat_count = 2;
    fused_rope.enqueue_instruction(1, low_rotate);

    auto high_product = VxmLaneAluInstruction{
        VxmAluOpcode::Multiply,
        VxmLaneOperand::StreamFloat16(
            1.0f, 0, VxmStreamSource::West),
        VxmLaneOperand::StreamFloat16(
            1.0f, 4, VxmStreamSource::East)};
    high_product.repeat_count = 2;
    fused_rope.enqueue_instruction(2, high_product);
    auto high_rotate = VxmLaneAluInstruction{
        VxmAluOpcode::FusedMultiplyAdd,
        VxmLaneOperand::StreamFloat16(
            1.0f, 0, VxmStreamSource::East),
        VxmLaneOperand::StreamFloat16(
            1.0f, 5, VxmStreamSource::East)};
    high_rotate.output_stream = 2;
    high_rotate.output_type = VxmCastTarget::Float32;
    high_rotate.repeat_count = 2;
    fused_rope.enqueue_instruction(3, high_rotate);

    fused_rope.tick();
    for (int row = 0; row < 2; ++row) {
        auto streams = VxmLane::StreamBytes{};
        put_int32(streams, 0, row == 0 ? 2 : 5);
        put_int32(streams, 8 * VxmLane::kStreamGroupBytes,
                  row == 0 ? 3 : 7);
        put_int32(streams, 4 * VxmLane::kStreamGroupBytes,
                  row == 0 ? 4 : 2);
        put_int32(streams, 5 * VxmLane::kStreamGroupBytes,
                  row == 0 ? 1 : 3);
        fused_rope.set_stream_inputs(streams);
        fused_rope.tick();
    }
    fused_rope.tick();
    fused_rope.tick();
    assert(fused_rope.outputs().size() == 4);
    assert(VxmLane::unpack_float32(
        fused_rope.outputs()[0].bytes) == 5.0f);
    assert(VxmLane::unpack_float32(
        fused_rope.outputs()[1].bytes) == 5.0f);
    assert(VxmLane::unpack_float32(
        fused_rope.outputs()[2].bytes) == 11.0f);
    assert(VxmLane::unpack_float32(
        fused_rope.outputs()[3].bytes) == 11.0f);
    fused_rope.tick();
    assert(fused_rope.outputs().size() == 4);
    assert(VxmLane::unpack_float32(
        fused_rope.outputs()[0].bytes) == -11.0f);
    assert(VxmLane::unpack_float32(
        fused_rope.outputs()[2].bytes) == 29.0f);

    // One Current Config Register is held for several element cycles.  The
    // repeat counter advances only when that ALU actually executes.
    auto repeated_config = VxmLaneTestDriver{};
    repeated_config.set_chain_depth(VxmChainDepth::Two);
    auto repeated_head = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamFloat16()};
    repeated_head.repeat_count = 3;
    repeated_config.enqueue_instruction(0, repeated_head);
    auto repeated_tail = output_bypass(0);
    repeated_tail.repeat_count = 3;
    repeated_config.enqueue_instruction(1, repeated_tail);
    assert(repeated_config.config_entry_count(0) == 1);
    assert(repeated_config.config_entry_count(1) == 1);
    repeated_config.tick(); // explicit FIFO -> Decoder cycle
    assert(repeated_config.current_repeat_count(0) == 0);
    assert(repeated_config.instruction_control().decoding(0));
    for (int value = 1; value <= 3; ++value) {
        auto input = VxmLane::StreamBytes{};
        put_int32(input, 0, value);
        repeated_config.set_stream_inputs(input);
        repeated_config.tick();
        assert(repeated_config.current_repeat_count(0)
            == static_cast<std::size_t>(3 - value));
        assert(repeated_config.current_repeat_count(1)
            == static_cast<std::size_t>(value == 1 ? 3 : 4 - value));
    }
    repeated_config.tick();
    assert(repeated_config.idle());

    auto routes = VxmLaneTestDriver{};
    routes.set_chain_depth(VxmChainDepth::Two);
    assert(throws([&] { routes.enqueue_instruction(16,
        {VxmAluOpcode::Bypass, VxmLaneOperand::Imm(0)}); }));
    assert(throws([&] { routes.enqueue_instruction(0,
        {VxmSpecialAluOpcode::Exp, VxmLaneOperand::StreamFloat16()}); }));
    assert(throws([&] { routes.enqueue_instruction(1,
        {VxmAluOpcode::Add, VxmLaneOperand::StreamFloat16(), VxmLaneOperand::Imm(1)}); }));
    assert(throws([&] { routes.enqueue_instruction(1, output_bypass(4)); }));
    assert(throws([&] { routes.enqueue_instruction(1,
        {VxmAluOpcode::Bypass, VxmLaneOperand::Feedback()}); }));
    assert(throws([&] { routes.enqueue_instruction(0,
        {VxmAluOpcode::Add, VxmLaneOperand::StreamFloat16(),
         VxmLaneOperand::Feedback()}); }));
    routes.enqueue_instruction(0,
        {VxmAluOpcode::Add, VxmLaneOperand::StreamFloat16(), VxmLaneOperand::Imm(1)});
    // Control may already contain prefetched configurations while an empty
    // data path switches its 2/4/8 routing mode.
    routes.set_chain_depth(VxmChainDepth::Four);
    assert(routes.chain_depth() == VxmChainDepth::Four);

    // No dynamic backpressure: an invalid static schedule is diagnosed before
    // an upstream result can overwrite an occupied pipeline register.
    auto collision = VxmLaneTestDriver{};
    collision.set_chain_depth(VxmChainDepth::Two);
    collision.enqueue_instruction(0, {VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamFloat16()});
    collision.enqueue_instruction(0, {VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamFloat16()});
    collision.tick(); // decode first configuration before Data arrives
    auto collision_input = VxmLane::StreamBytes{};
    put_int32(collision_input, 0, 1);
    collision.set_stream_inputs(collision_input);
    collision.tick();
    collision.set_stream_inputs(collision_input);
    assert(throws([&] { collision.tick(); }));
    assert(collision.queue_depth(0) == 1);
    assert(collision.cycle() == 2);

    // Four independent 2-stage chains consume all eight fixed input Groups;
    // their tails also have fixed output bindings.
    auto parallel = VxmLaneTestDriver{};
    parallel.set_chain_depth(VxmChainDepth::Two);
    parallel.enqueue_instruction(0, {VxmAluOpcode::Add,
        VxmLaneOperand::StreamFloat16(), VxmLaneOperand::StreamFloat16()});
    parallel.enqueue_instruction(1, output_bypass(0));
    parallel.enqueue_instruction(2, {VxmAluOpcode::Multiply,
        VxmLaneOperand::StreamFloat16(), VxmLaneOperand::StreamFloat16()});
    parallel.enqueue_instruction(3, output_bypass(2));
    parallel.enqueue_instruction(4, {VxmAluOpcode::Subtract,
        VxmLaneOperand::StreamFloat16(), VxmLaneOperand::StreamFloat16()});
    parallel.enqueue_instruction(5, output_bypass(4));
    parallel.enqueue_instruction(6, {VxmAluOpcode::Max,
        VxmLaneOperand::StreamFloat16(), VxmLaneOperand::StreamFloat16()});
    parallel.enqueue_instruction(7, output_bypass(6));
    parallel.tick(); // decode all eight broadcast configurations
    auto streams = VxmLane::StreamBytes{};
    put_int32(streams, 0, 3); put_int32(streams, 2, 4);
    put_int32(streams, 4, 5); put_int32(streams, 6, 6);
    put_int32(streams, 8, 20); put_int32(streams, 10, 8);
    put_int32(streams, 12, 9); put_int32(streams, 14, 11);
    parallel.set_stream_inputs(streams);
    parallel.tick();
    assert(parallel.outputs().empty());
    parallel.tick();
    assert(parallel.outputs().size() == 6);
    assert(parallel.outputs()[0].stream == 0);
    assert(parallel.outputs()[1].stream == 4);
    assert(parallel.outputs()[2].stream == 6);
    assert(VxmLane::unpack_float32(parallel.outputs()[0].bytes) == 7.0f);
    assert(VxmLane::unpack_float32(parallel.outputs()[1].bytes) == 12.0f);
    assert(VxmLane::unpack_float32(parallel.outputs()[2].bytes) == 11.0f);
    // The Multiply chain has one extra internal pipeline cycle but retains
    // II=1, so its result follows one cycle later.
    parallel.tick();
    assert(parallel.outputs().size() == 2);
    assert(parallel.outputs()[0].stream == 2);
    assert(VxmLane::unpack_float32(parallel.outputs()[0].bytes) == 30.0f);

    // Prove C1/C3 use LUT results rather than host exp/divide.  Deliberately
    // non-mathematical tables make the final result distinguishable.
    auto lut_lane = VxmLaneTestDriver{};
    lut_lane.special_alu().configure_lut(VxmSpecialAluOpcode::Exp,
        {-0.5f, 1.0f}, {VxmLutEntry::from_float(0.0f, 2.0f)});
    lut_lane.special_alu().configure_lut(VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f}, {VxmLutEntry::from_float(0.0f, 0.25f)});
    lut_lane.set_chain_depth(VxmChainDepth::Eight);
    lut_lane.enqueue_instruction(0, {VxmAluOpcode::Negate,
        VxmLaneOperand::StreamFloat16(),
        VxmLaneOperand::StreamFloat16()});
    lut_lane.enqueue_instruction(1, {VxmSpecialAluOpcode::Exp,
        VxmLaneOperand::Previous()});
    lut_lane.enqueue_instruction(2, {VxmAluOpcode::Add,
        VxmLaneOperand::Previous(), VxmLaneOperand::Imm(1.0f)});
    lut_lane.enqueue_instruction(3, {VxmSpecialAluOpcode::Reciprocal,
        VxmLaneOperand::Previous()});
    lut_lane.enqueue_instruction(4, {VxmAluOpcode::Multiply,
        VxmLaneOperand::Previous(), VxmLaneOperand::Original()});
    lut_lane.enqueue_instruction(5, {VxmAluOpcode::Multiply,
        VxmLaneOperand::Previous(), VxmLaneOperand::Aux()});
    lut_lane.enqueue_instruction(6, {VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()});
    auto lut_output = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    lut_output.output_type = VxmCastTarget::Int8;
    lut_output.output_scale = 0.25f;
    lut_output.output_stream = 6;
    lut_lane.enqueue_instruction(7, lut_output);
    lut_lane.tick(); // decode configuration before the first input
    auto lut_streams = VxmLane::StreamBytes{};
    const auto gate = VxmLane::pack_float16(1.0f);
    const auto up = VxmLane::pack_float16(2.0f);
    for (std::size_t byte = 0; byte < VxmLane::kStreamGroupBytes; ++byte) {
        lut_streams[byte] = gate[byte];
        lut_streams[VxmLane::kStreamGroupBytes + byte] = up[byte];
    }
    lut_lane.set_stream_inputs(lut_streams);
    // The ALU chain takes 18 cycles; the optional FP16->INT8 tail adds one.
    for (std::size_t cycle = 0; cycle < 17; ++cycle) {
        lut_lane.tick();
        assert(!lut_lane.output());
    }
    lut_lane.tick();
    assert(!lut_lane.output());
    lut_lane.tick();
    assert(lut_lane.output());
    assert(lut_lane.output()->stream == 6);
    assert(lut_lane.output()->value == 1);

    // C1 uses ordinary Add plus a local feedback register; there is no
    // separate reduction operation or cross-lane data movement.
    auto accumulation = VxmLaneTestDriver{};
    accumulation.set_chain_depth(VxmChainDepth::Two);
    for (int token = 0; token < 3; ++token) {
        accumulation.enqueue_instruction(0, {VxmAluOpcode::Bypass,
            VxmLaneOperand::StreamFloat16()});
        auto add = VxmLaneAluInstruction{VxmAluOpcode::Add,
            VxmLaneOperand::Previous(), VxmLaneOperand::Acc()};
        add.precision = VxmAluPrecision::Float32;
        add.accumulator_reset = token == 0;
        add.accumulator_write = true;
        add.accumulator_emit = token == 2;
        if (token == 2) {
            add.output_stream = 0;
            add.output_type = VxmCastTarget::Float32;
        }
        accumulation.enqueue_instruction(1, add);
    }
    accumulation.tick(); // decode the first configuration set
    for (int value = 1; value <= 3; ++value) {
        auto input = VxmLane::StreamBytes{};
        put_int32(input, 0, value);
        accumulation.set_stream_inputs(input);
        accumulation.tick();
        assert(!accumulation.output());
    }
    accumulation.tick();
    assert(accumulation.output());
    assert(VxmLane::unpack_float32(accumulation.output()->bytes) == 6.0f);

    // Max uses the same feedback path and initializes to -infinity, which is
    // required for rows containing only negative values.
    auto local_max = VxmLaneTestDriver{};
    local_max.set_chain_depth(VxmChainDepth::Two);
    for (int token = 0; token < 3; ++token) {
        local_max.enqueue_instruction(0, {VxmAluOpcode::Bypass,
            VxmLaneOperand::StreamFloat16()});
        auto maximum = VxmLaneAluInstruction{VxmAluOpcode::Max,
            VxmLaneOperand::Previous(), VxmLaneOperand::Acc()};
        maximum.precision = VxmAluPrecision::Float32;
        maximum.accumulator_reset = token == 0;
        maximum.accumulator_write = true;
        maximum.accumulator_emit = token == 2;
        if (token == 2) {
            maximum.output_stream = 0;
            maximum.output_type = VxmCastTarget::Float32;
        }
        local_max.enqueue_instruction(1, maximum);
    }
    local_max.tick(); // decode the first configuration set
    for (const auto value : {-5, -2, -7}) {
        auto input = VxmLane::StreamBytes{};
        put_int32(input, 0, value);
        local_max.set_stream_inputs(input);
        local_max.tick();
    }
    local_max.tick();
    assert(local_max.output());
    assert(VxmLane::unpack_float32(local_max.output()->bytes) == -2.0f);

    // C7 feeds C0 directly at the clock boundary.  The value never enters a
    // Stream Register or a separate tail holding register: the already
    // decoded Feedback selector captures it in C0's normal input register.
    auto feedback = VxmLaneTestDriver{};
    feedback.set_chain_depth(VxmChainDepth::Eight);
    for (std::size_t stage = 0; stage < 8; ++stage) {
        feedback.enqueue_instruction(stage, stage == 0
            ? VxmLaneAluInstruction{
                VxmAluOpcode::Add,
                VxmLaneOperand::StreamFloat16(),
                VxmLaneOperand::Imm(2.0f)}
            : VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous()});

        auto second = stage == 0
            ? VxmLaneAluInstruction{
                VxmAluOpcode::Add,
                VxmLaneOperand::Feedback(),
                VxmLaneOperand::Imm(3.0f)}
            : VxmLaneAluInstruction{
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous()};
        if (stage == 7) {
            second.output_stream = 6;
            second.output_type = VxmCastTarget::Float32;
        }
        feedback.enqueue_instruction(stage, second);
    }

    auto feedback_packet_instruction = VxmLaneAluInstruction{
        VxmAluOpcode::Add,
        VxmLaneOperand::Feedback(),
        VxmLaneOperand::Imm(3.0f)};
    feedback_packet_instruction.repeat_count = 2;
    const auto feedback_packet = VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Eight, feedback_packet_instruction);
    const auto feedback_decoded =
        VxmCompactInstructionCodec::decode(0, feedback_packet);
    assert(feedback_decoded.instruction.lhs.kind
           == VxmLaneOperandKind::Feedback);
    assert(feedback_decoded.instruction.repeat_count == 2);

    feedback.tick(); // initial decode
    auto feedback_input = VxmLane::StreamBytes{};
    put_int32(feedback_input, 0, 5);
    feedback.set_stream_inputs(feedback_input);
    feedback.tick(); // first C0 computes 5 + 2
    for (std::size_t stage = 1; stage < 8; ++stage) {
        feedback.tick();
    }
    assert(feedback.last_trace()[7].result
           && *feedback.last_trace()[7].result == 7.0f);
    assert(feedback.last_trace()[0].state
           == VxmLaneAluTraceState::Stalled);

    feedback.tick(); // C0 immediately consumes C7 feedback: 7 + 3
    assert(feedback.last_trace()[0].result
           && *feedback.last_trace()[0].result == 10.0f);
    assert(!feedback.output());
    for (std::size_t stage = 1; stage < 8; ++stage) {
        feedback.tick();
    }
    assert(feedback.output());
    assert(feedback.output()->stream == 6);
    assert(VxmLane::unpack_float32(feedback.output()->bytes) == 10.0f);

    // Every same-configuration chain reuses the same Feedback opcode.  The
    // physical destination is inferred from its fixed tail/head placement.
    assert(run_fixed_feedback_path(VxmChainDepth::Two, 0) == 10.0f);
    assert(run_fixed_feedback_path(VxmChainDepth::Two, 2) == 10.0f);
    assert(run_fixed_feedback_path(VxmChainDepth::Two, 4) == 10.0f);
    assert(run_fixed_feedback_path(VxmChainDepth::Two, 6) == 10.0f);
    assert(run_fixed_feedback_path(VxmChainDepth::Four, 0) == 10.0f);
    assert(run_fixed_feedback_path(VxmChainDepth::Four, 4) == 10.0f);
    assert(run_fixed_feedback_path(VxmChainDepth::Eight, 0) == 10.0f);

    // Chain merging reuses the wider links from the same fixed network.
    assert(run_merged_feedback_path(
        VxmChainDepth::Two, 2, 0) == 10.0f);   // C3 -> C0
    assert(run_merged_feedback_path(
        VxmChainDepth::Two, 6, 4) == 10.0f);   // C7 -> C4
    assert(run_merged_feedback_path(
        VxmChainDepth::Four, 4, 0) == 10.0f);  // C7 -> C0

    // Four 2-stage chains produce together.  C1/C5 take the direct paths;
    // C3/C7 are held for one cycle and then drained to C0/C4.  No Stream
    // Register round trip or general FIFO is involved.
    auto multi_feedback = VxmLaneTestDriver{};
    multi_feedback.set_chain_depth(VxmChainDepth::Two);
    for (const auto head : {std::size_t{0}, std::size_t{2},
                            std::size_t{4}, std::size_t{6}}) {
        multi_feedback.enqueue_instruction(head, {
            VxmAluOpcode::Bypass,
            VxmLaneOperand::StreamFloat16()});
        multi_feedback.enqueue_instruction(head + 1, {
            VxmAluOpcode::Bypass,
            VxmLaneOperand::Previous()});
    }
    for (const auto head : {std::size_t{0}, std::size_t{4}}) {
        auto consume = VxmLaneAluInstruction{
            VxmAluOpcode::Bypass,
            VxmLaneOperand::Feedback()};
        consume.repeat_count = 2;
        multi_feedback.enqueue_instruction(head, consume);

        auto output = VxmLaneAluInstruction{
            VxmAluOpcode::Bypass,
            VxmLaneOperand::Previous()};
        output.repeat_count = 2;
        output.output_stream =
            multi_feedback.fixed_output_stream_for_stage(head + 1);
        output.output_type = VxmCastTarget::Float32;
        multi_feedback.enqueue_instruction(head + 1, output);
    }

    multi_feedback.tick();
    auto multi_streams = VxmLane::StreamBytes{};
    put_int32(multi_streams, 0, 10);
    put_int32(multi_streams, 4, 20);
    put_int32(multi_streams, 8, 30);
    put_int32(multi_streams, 12, 40);
    multi_feedback.set_stream_inputs(multi_streams);
    multi_feedback.tick();
    multi_feedback.tick(); // four tails complete together
    assert(multi_feedback.feedback_pending_count() == 4);
    assert(!multi_feedback.output());

    multi_feedback.tick(); // direct values execute; C3/C7 drain
    assert(multi_feedback.feedback_pending_count() == 0);
    assert(!multi_feedback.output());

    multi_feedback.tick(); // first output pair
    assert(multi_feedback.outputs().size() == 4);
    assert(VxmLane::unpack_float32(
               multi_feedback.outputs()[0].bytes) == 10.0f);
    assert(VxmLane::unpack_float32(
               multi_feedback.outputs()[1].bytes) == 30.0f);

    multi_feedback.tick(); // held output pair
    assert(multi_feedback.outputs().size() == 4);
    assert(VxmLane::unpack_float32(
               multi_feedback.outputs()[0].bytes) == 20.0f);
    assert(VxmLane::unpack_float32(
               multi_feedback.outputs()[1].bytes) == 40.0f);
    assert(multi_feedback.idle());

    // Direct 2->8 transition: C1 enters C0 immediately, while C3/C5/C7
    // occupy the three fixed holding registers and drain in that order.
    auto four_to_one = VxmLaneTestDriver{};
    four_to_one.set_chain_depth(VxmChainDepth::Two);
    auto lane_timing =
        std::vector<vxm_hardware_test::TimingCycle>{};
    const auto record_lane =
        [&four_to_one, &lane_timing](std::string event) {
            lane_timing.push_back(
                vxm_hardware_test::capture_cycle(
                    four_to_one.cycle() - 1,
                    four_to_one,
                    four_to_one.outputs().size(),
                    std::move(event)));
        };
    for (const auto head : {std::size_t{0}, std::size_t{2},
                            std::size_t{4}, std::size_t{6}}) {
        four_to_one.enqueue_instruction(head, {
            VxmAluOpcode::Bypass,
            VxmLaneOperand::StreamFloat16()});
        four_to_one.enqueue_instruction(head + 1, {
            VxmAluOpcode::Bypass,
            VxmLaneOperand::Previous()});
    }
    auto consume_four = VxmLaneAluInstruction{
        VxmAluOpcode::Bypass,
        VxmLaneOperand::Feedback()};
    consume_four.repeat_count = 4;
    four_to_one.enqueue_instruction(0, consume_four);
    auto output_four = VxmLaneAluInstruction{
        VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    output_four.repeat_count = 4;
    output_four.output_stream = 0;
    output_four.output_type = VxmCastTarget::Float32;
    four_to_one.enqueue_instruction(1, output_four);

    four_to_one.tick();
    record_lane("config decode");
    auto four_streams = VxmLane::StreamBytes{};
    put_int32(four_streams, 0, 10);
    put_int32(four_streams, 4, 20);
    put_int32(four_streams, 8, 30);
    put_int32(four_streams, 12, 40);
    four_to_one.set_stream_inputs(four_streams);
    four_to_one.tick();
    record_lane("input heads execute");
    four_to_one.tick();
    record_lane("tails capture");
    assert(four_to_one.feedback_pending_count() == 6);

    four_to_one.tick();
    record_lane("feedback drain");
    assert(four_to_one.feedback_pending_count() == 4);
    assert(!four_to_one.output());

    for (std::size_t expected = 0; expected < 4; ++expected) {
        four_to_one.tick();
        record_lane("pipeline output");
        assert(four_to_one.feedback_pending_count()
               == (expected < 2 ? 2 - 2 * expected : 0));
        assert(four_to_one.outputs().size() == 2);
        assert(VxmLane::unpack_float32(
                   four_to_one.outputs()[0].bytes)
               == static_cast<float>(10 * (expected + 1)));
    }
    assert(four_to_one.idle());
    vxm_hardware_test::write_timing_reports(
        "lane", "VXM Lane complete timing sequence", lane_timing);

    vxm_hardware_test::write_pass_result(
        "lane_test_results.txt", "lane_test");
    return 0;
}
