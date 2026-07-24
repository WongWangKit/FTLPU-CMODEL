#include "ftlpu/vxm/lane.hpp"
#include "lane_test_driver.hpp"
#include "lane_operator_workloads.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

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
    const auto bytes = ftlpu::VxmLane::pack_int32(value);
    for (std::size_t byte = 0; byte < 4; ++byte) streams[base + byte] = bytes[byte];
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

std::filesystem::path result_path()
{
    const auto source = std::filesystem::path(__FILE__);
    if (source.is_absolute()) {
        return source.parent_path() / "lane_test_results.txt";
    }
    const auto cwd = std::filesystem::current_path();
    if (std::filesystem::exists(cwd / source)) {
        return (cwd / source).parent_path() / "lane_test_results.txt";
    }
    if (std::filesystem::exists(cwd / "lane_test.cpp")) {
        return cwd / "lane_test_results.txt";
    }
    return cwd / "lane_test_results.txt";
}

}

int main()
{
    using namespace ftlpu;
    static_assert(VxmLane::kAluCount == 8);
    static_assert(VxmLane::kBlockCount == 4);
    static_assert(VxmLane::kInputStreams == 32);
    static_assert(VxmLane::fixed_input_group_for_stage(0, false) == 0);
    static_assert(VxmLane::fixed_input_group_for_stage(0, true) == 1);
    static_assert(VxmLane::fixed_input_group_for_stage(2, false) == 2);
    static_assert(VxmLane::fixed_input_group_for_stage(2, true) == 3);
    static_assert(VxmLane::fixed_input_group_for_stage(4, false) == 4);
    static_assert(VxmLane::fixed_input_group_for_stage(4, true) == 5);
    static_assert(VxmLane::fixed_input_group_for_stage(6, false) == 6);
    static_assert(VxmLane::fixed_input_group_for_stage(6, true) == 7);

    auto topology = VxmLane{};
    topology.set_chain_depth(VxmChainDepth::Two);
    assert(topology.chain_count() == 4);
    assert(topology.is_chain_head(0) && topology.is_chain_tail(1));
    assert(topology.is_chain_head(6) && topology.is_chain_tail(7));
    topology.set_chain_depth(VxmChainDepth::Four);
    assert(topology.chain_count() == 2);
    assert(topology.is_chain_head(4) && topology.is_chain_tail(7));
    topology.set_chain_depth(VxmChainDepth::Eight);
    assert(topology.chain_count() == 1);
    assert(topology.is_chain_head(0) && topology.is_chain_tail(7));

    // One Current Config Register is held for several element cycles.  The
    // repeat counter advances only when that ALU actually executes.
    auto repeated_config = VxmLaneTestDriver{};
    repeated_config.set_chain_depth(VxmChainDepth::Two);
    auto repeated_head = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamInt32()};
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
    assert(throws([&] { routes.enqueue_instruction(8,
        {VxmAluOpcode::Bypass, VxmLaneOperand::Imm(0)}); }));
    assert(throws([&] { routes.enqueue_instruction(0,
        {VxmSpecialAluOpcode::Exp, VxmLaneOperand::StreamFloat32()}); }));
    assert(throws([&] { routes.enqueue_instruction(1,
        {VxmAluOpcode::Add, VxmLaneOperand::StreamFloat32(), VxmLaneOperand::Imm(1)}); }));
    assert(throws([&] { routes.enqueue_instruction(1, output_bypass(4)); }));
    routes.enqueue_instruction(0,
        {VxmAluOpcode::Add, VxmLaneOperand::StreamInt32(), VxmLaneOperand::Imm(1)});
    // Control may already contain prefetched configurations while an empty
    // data path switches its 2/4/8 routing mode.
    routes.set_chain_depth(VxmChainDepth::Four);
    assert(routes.chain_depth() == VxmChainDepth::Four);

    // No dynamic backpressure: an invalid static schedule is diagnosed before
    // an upstream result can overwrite an occupied pipeline register.
    auto collision = VxmLaneTestDriver{};
    collision.set_chain_depth(VxmChainDepth::Two);
    collision.enqueue_instruction(0, {VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamInt32()});
    collision.enqueue_instruction(0, {VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamInt32()});
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
        VxmLaneOperand::StreamInt32(), VxmLaneOperand::StreamInt32()});
    parallel.enqueue_instruction(1, output_bypass(0));
    parallel.enqueue_instruction(2, {VxmAluOpcode::Multiply,
        VxmLaneOperand::StreamInt32(), VxmLaneOperand::StreamInt32()});
    parallel.enqueue_instruction(3, output_bypass(4));
    parallel.enqueue_instruction(4, {VxmAluOpcode::Subtract,
        VxmLaneOperand::StreamInt32(), VxmLaneOperand::StreamInt32()});
    parallel.enqueue_instruction(5, output_bypass(8));
    parallel.enqueue_instruction(6, {VxmAluOpcode::Max,
        VxmLaneOperand::StreamInt32(), VxmLaneOperand::StreamInt32()});
    parallel.enqueue_instruction(7, output_bypass(12));
    parallel.tick(); // decode all eight broadcast configurations
    auto streams = VxmLane::StreamBytes{};
    put_int32(streams, 0, 3); put_int32(streams, 4, 4);
    put_int32(streams, 8, 5); put_int32(streams, 12, 6);
    put_int32(streams, 16, 20); put_int32(streams, 20, 8);
    put_int32(streams, 24, 9); put_int32(streams, 28, 11);
    parallel.set_stream_inputs(streams);
    parallel.tick();
    assert(parallel.outputs().empty());
    parallel.tick();
    assert(parallel.outputs().size() == 3);
    assert(parallel.outputs()[0].stream == 0);
    assert(parallel.outputs()[1].stream == 8);
    assert(parallel.outputs()[2].stream == 12);
    assert(VxmLane::unpack_float32(parallel.outputs()[0].bytes) == 7.0f);
    assert(VxmLane::unpack_float32(parallel.outputs()[1].bytes) == 12.0f);
    assert(VxmLane::unpack_float32(parallel.outputs()[2].bytes) == 11.0f);
    // The Multiply chain has one extra internal pipeline cycle but retains
    // II=1, so its result follows one cycle later.
    parallel.tick();
    assert(parallel.outputs().size() == 1);
    assert(parallel.outputs()[0].stream == 4);
    assert(VxmLane::unpack_float32(parallel.outputs()[0].bytes) == 30.0f);

    // Prove C1/C3 use LUT results rather than host exp/divide.  Deliberately
    // non-mathematical tables make the final result distinguishable.
    auto lut_lane = VxmLaneTestDriver{};
    lut_lane.special_alu().configure_lut(VxmSpecialAluOpcode::Exp,
        {-0.5f, 1.0f}, {VxmLutEntry::from_float(0.0f, 2.0f)});
    lut_lane.special_alu().configure_lut(VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f}, {VxmLutEntry::from_float(0.0f, 0.25f)});
    const auto swiglu = VxmLane::SwigluParams{1.0f, 1.0f, 0.25f, 0};
    lut_lane.load_pipelined_swiglu_program(swiglu, 1);
    lut_lane.tick(); // decode configuration before the first input
    lut_lane.set_swiglu_input(VxmLane::pack_int32(1), VxmLane::pack_int32(2));
    // 1 + 5 + 1 + 5 + 2 + 2 + 1 + 1 = 18 cycles.
    for (std::size_t cycle = 0; cycle < 17; ++cycle) {
        lut_lane.tick();
        assert(!lut_lane.output());
    }
    lut_lane.tick();
    assert(lut_lane.output());
    assert(lut_lane.output()->stream == 12);
    assert(lut_lane.output()->value == 1);

    // C1 uses ordinary Add plus a local feedback register; there is no
    // separate reduction operation or cross-lane data movement.
    auto accumulation = VxmLaneTestDriver{};
    accumulation.set_chain_depth(VxmChainDepth::Two);
    for (int token = 0; token < 3; ++token) {
        accumulation.enqueue_instruction(0, {VxmAluOpcode::Bypass,
            VxmLaneOperand::StreamInt32()});
        auto add = VxmLaneAluInstruction{VxmAluOpcode::Add,
            VxmLaneOperand::Previous(), VxmLaneOperand::Acc()};
        add.precision = VxmAluPrecision::Float32;
        add.accumulator_reset = token == 0;
        add.accumulator_write = true;
        add.accumulator_emit = token == 2;
        if (token == 2) add.output_stream = 0;
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
            VxmLaneOperand::StreamInt32()});
        auto maximum = VxmLaneAluInstruction{VxmAluOpcode::Max,
            VxmLaneOperand::Previous(), VxmLaneOperand::Acc()};
        maximum.precision = VxmAluPrecision::Float32;
        maximum.accumulator_reset = token == 0;
        maximum.accumulator_write = true;
        maximum.accumulator_emit = token == 2;
        if (token == 2) maximum.output_stream = 0;
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
    const auto output_path = result_path();
    auto result_file = std::ofstream(output_path, std::ios::trunc);
    assert(result_file && "failed to create VXM lane result file");
    run_lane_operator_workloads(std::cout, &result_file);
    result_file.close();
    std::cout << "Full lane results: " << output_path.string() << '\n';
    return 0;
}
