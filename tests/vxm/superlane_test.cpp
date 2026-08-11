#include "ftlpu/vxm/superlane.hpp"
#include "hardware_test_output.hpp"
#include "hardware_timing_report.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

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

} // namespace

int main()
{
    auto superlane = ftlpu::VxmSuperlane{};
    constexpr std::size_t entries = 64;
    superlane.configure_special_lut(ftlpu::VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f / entries},
        table(1.0f, 1.0f / entries, entries,
              [](float x) { return 1.0f / x; }));

    // Lanes share the LUT, while instruction FIFO/decode/config state is owned
    // only once by the Superlane.
    for (std::size_t lane = 0; lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        assert(&superlane.lane(lane).special_alu() == &superlane.special_alu());
    }
    const auto low = superlane.lane(0).special_alu().make_lookup(
        ftlpu::VxmSpecialAluOpcode::Reciprocal, 1.1f);
    const auto high = superlane.lane(1).special_alu().make_lookup(
        ftlpu::VxmSpecialAluOpcode::Reciprocal, 1.8f);
    assert(low.index != high.index);

    // The local Superlane FIFO has exactly three queued entries.  Overflow is
    // a compiler scheduling error; the C Model reports it instead of stalling.
    auto fifo_limit = ftlpu::VxmSuperlane{};
    fifo_limit.set_chain_depth(ftlpu::VxmChainDepth::Two);
    for (int immediate = 0; immediate < 3; ++immediate) {
        fifo_limit.enqueue_instruction(0, {
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::StreamFloat16(),
            ftlpu::VxmLaneOperand::Imm(static_cast<float>(immediate))});
    }
    assert(fifo_limit.instruction_control().fifo_entry_count(0) == 3);
    bool overflow_reported = false;
    try {
        fifo_limit.enqueue_instruction(0, {
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::StreamFloat16(),
            ftlpu::VxmLaneOperand::Imm(3.0f)});
    } catch (const std::overflow_error&) {
        overflow_reported = true;
    }
    assert(overflow_reported);

    // Once the first configuration has paid the decode-start cycle, following
    // configurations are decoded while Current executes.  Even one-element
    // runs A/B/C must execute in three consecutive cycles without a bubble.
    auto hidden_decode = ftlpu::VxmSuperlane{};
    hidden_decode.set_chain_depth(ftlpu::VxmChainDepth::Two);
    for (int immediate = 0; immediate < 3; ++immediate) {
        hidden_decode.enqueue_instruction(0, {
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::StreamFloat16(),
            ftlpu::VxmLaneOperand::Imm(static_cast<float>(immediate))});
    }
    auto hidden_tail = ftlpu::VxmLaneAluInstruction{
        ftlpu::VxmAluOpcode::Bypass,
        ftlpu::VxmLaneOperand::Previous()};
    hidden_tail.output_type = ftlpu::VxmCastTarget::Float32;
    hidden_tail.output_stream = 0;
    hidden_tail.repeat_count = 3;
    hidden_decode.enqueue_instruction(1, hidden_tail);
    hidden_decode.tick(); // the only exposed decode-start cycle

    for (int item = 0; item < 3; ++item) {
        auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
        for (std::size_t lane = 0;
             lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
            const auto bytes = ftlpu::VxmLane::pack_float16(10.0f);
            for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
                streams[lane][byte] = bytes[byte];
            }
        }
        hidden_decode.set_stream_inputs(streams);
        hidden_decode.tick();
        assert(hidden_decode.lane(0).last_trace()[0].state
            == ftlpu::VxmLaneAluTraceState::Executed);
    }
    hidden_decode.tick();
    assert(hidden_decode.idle());

    // The Superlane owns one shared Current Config and repeat counter.  The
    // 16 lanes execute it in lockstep; the counter is consumed once per cycle,
    // not once per lane.
    superlane.set_chain_depth(ftlpu::VxmChainDepth::Two);
    auto head = ftlpu::VxmLaneAluInstruction{ftlpu::VxmAluOpcode::Bypass,
        ftlpu::VxmLaneOperand::StreamFloat16()};
    head.repeat_count = 3;
    superlane.enqueue_instruction(0, head);
    auto tail = ftlpu::VxmLaneAluInstruction{ftlpu::VxmAluOpcode::Bypass,
        ftlpu::VxmLaneOperand::Previous()};
    tail.output_type = ftlpu::VxmCastTarget::Float32;
    tail.output_stream = 0;
    tail.repeat_count = 3;
    superlane.enqueue_instruction(1, tail);
    assert(superlane.config_entry_count(0) == 1);
    assert(superlane.config_entry_count(1) == 1);
    superlane.tick(); // explicit FIFO -> Decoder cycle
    assert(superlane.remaining_in_current(0) == 0);
    assert(superlane.instruction_control().decoding(0));

    for (std::size_t cycle = 0; cycle < 3; ++cycle) {
        auto streams = ftlpu::VxmSuperlane::StreamMatrix{};
        for (std::size_t lane = 0; lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
            const auto bytes = ftlpu::VxmLane::pack_float16(
                static_cast<float>(cycle * 16 + lane));
            for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
                streams[lane][byte] = bytes[byte];
            }
            assert(superlane.remaining_executions(0) == 3 - cycle);
        }
        superlane.set_stream_inputs(streams);
        superlane.tick();
        assert(superlane.remaining_in_current(0) == 2 - cycle);
        assert(superlane.remaining_in_current(1)
            == (cycle == 0 ? 3 : 3 - cycle));
    }
    superlane.tick();
    assert(superlane.idle());

    // One shared Feedback configuration selects the fixed C7 -> C0 wire in
    // all 16 lanes, while each lane carries its own numerical result.
    auto feedback = ftlpu::VxmSuperlane{};
    feedback.set_chain_depth(ftlpu::VxmChainDepth::Eight);
    for (std::size_t stage = 0;
         stage < 8; ++stage) {
        feedback.enqueue_instruction(stage, stage == 0
            ? ftlpu::VxmLaneAluInstruction{
                ftlpu::VxmAluOpcode::Add,
                ftlpu::VxmLaneOperand::StreamFloat16(),
                ftlpu::VxmLaneOperand::Imm(2.0f)}
            : ftlpu::VxmLaneAluInstruction{
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::Previous()});

        auto second = stage == 0
            ? ftlpu::VxmLaneAluInstruction{
                ftlpu::VxmAluOpcode::Add,
                ftlpu::VxmLaneOperand::Feedback(),
                ftlpu::VxmLaneOperand::Imm(3.0f)}
            : ftlpu::VxmLaneAluInstruction{
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::Previous()};
        if (stage == 7) {
            second.output_stream = 6;
            second.output_type = ftlpu::VxmCastTarget::Float32;
        }
        feedback.enqueue_instruction(stage, second);
    }

    feedback.tick(); // initial decode
    auto feedback_streams = ftlpu::VxmSuperlane::StreamMatrix{};
    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        const auto bytes = ftlpu::VxmLane::pack_float16(
            static_cast<float>(lane));
        for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
            feedback_streams[lane][byte] = bytes[byte];
        }
    }
    feedback.set_stream_inputs(feedback_streams);
    feedback.tick();
    for (std::size_t stage = 1;
         stage < 8; ++stage) {
        feedback.tick();
    }
    assert(feedback.lane(0).last_trace()[7].result);
    feedback.tick(); // every lane consumes its own C7 result at C0
    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        assert(feedback.lane(lane).last_trace()[0].result);
        assert(*feedback.lane(lane).last_trace()[0].result
               == static_cast<float>(lane + 5));
    }
    for (std::size_t stage = 1;
         stage < 8; ++stage) {
        feedback.tick();
    }
    assert(feedback.outputs().size() == 2);
    assert(feedback.outputs()[0].stream == 6);
    assert(feedback.outputs()[1].stream == 14);
    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        assert(ftlpu::VxmLane::unpack_float32(
                   feedback.outputs()[0].byte_values[lane])
               == static_cast<float>(lane + 5));
    }

    // A real 2->8 boundary: the new depth-8 Feedback config is decoded while
    // the old depth-2 data is in flight.  The transition is requested before
    // C1/C3/C5/C7 retire, so their results are captured at the same edge and
    // C0 executes the first result on the immediately following cycle.
    auto no_bubble = ftlpu::VxmSuperlane{};
    no_bubble.set_chain_depth(ftlpu::VxmChainDepth::Two);
    auto superlane_timing =
        std::vector<vxm_hardware_test::TimingCycle>{};
    const auto record_superlane =
        [&no_bubble, &superlane_timing](std::string event) {
            superlane_timing.push_back(
                vxm_hardware_test::capture_cycle(
                    no_bubble.cycle() - 1,
                    no_bubble,
                    no_bubble.outputs().size(),
                    std::move(event)));
        };
    for (std::size_t stage = 0;
         stage < 8; ++stage) {
        auto old_instruction = stage % 2 == 0
            ? ftlpu::VxmLaneAluInstruction{
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::StreamFloat16()}
            : ftlpu::VxmLaneAluInstruction{
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::Previous()};
        no_bubble.enqueue_instruction_for_depth(
            ftlpu::VxmChainDepth::Two,
            stage, old_instruction);

        auto new_instruction = stage == 0
            ? ftlpu::VxmLaneAluInstruction{
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::Feedback()}
            : ftlpu::VxmLaneAluInstruction{
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::Previous()};
        new_instruction.repeat_count = 4;
        if (stage == 7) {
            new_instruction.output_stream = 6;
            new_instruction.output_type =
                ftlpu::VxmCastTarget::Float32;
        }
        no_bubble.enqueue_instruction_for_depth(
            ftlpu::VxmChainDepth::Eight,
            stage, new_instruction);
    }

    no_bubble.tick(); // old configs start decoding
    record_superlane("config decode");
    auto transition_streams =
        ftlpu::VxmSuperlane::StreamMatrix{};
    for (std::size_t lane = 0;
         lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
        for (std::size_t group = 0; group < 4; ++group) {
            const auto value = static_cast<std::int32_t>(
                lane * 100 + (group + 1) * 10);
            const auto bytes =
                ftlpu::VxmLane::pack_float16(static_cast<float>(value));
            const auto base = group * 4;
            for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
                transition_streams[lane][base + byte] = bytes[byte];
            }
        }
    }
    no_bubble.set_stream_inputs(transition_streams);
    no_bubble.tick(); // four old depth-2 heads execute
    record_superlane("input heads execute");

    no_bubble.request_chain_depth_transition(
        ftlpu::VxmChainDepth::Eight);
    assert(no_bubble.chain_depth_transition_pending());
    no_bubble.tick(); // old tails retire and new depth commits
    record_superlane("depth transition");
    assert(!no_bubble.chain_depth_transition_pending());
    assert(no_bubble.lane(0).chain_depth()
           == ftlpu::VxmChainDepth::Eight);
    assert(no_bubble.lane(0).last_feedback_capture_count() == 8);
    assert(no_bubble.lane(0).feedback_pending_count() == 6);
    assert(no_bubble.lane(0).last_trace()[1].state
           == ftlpu::VxmLaneAluTraceState::Executed);
    assert(no_bubble.lane(0).last_trace()[3].state
           == ftlpu::VxmLaneAluTraceState::Executed);
    assert(no_bubble.lane(0).last_trace()[5].state
           == ftlpu::VxmLaneAluTraceState::Executed);
    assert(no_bubble.lane(0).last_trace()[7].state
           == ftlpu::VxmLaneAluTraceState::Executed);

    no_bubble.tick(); // no empty cycle: new C0 consumes C1 result
    record_superlane("feedback enters C0");
    assert(no_bubble.lane(0).last_trace()[0].state
           == ftlpu::VxmLaneAluTraceState::Executed);
    assert(no_bubble.lane(0).feedback_pending_count() == 4);

    for (std::size_t cycle = 0; cycle < 6; ++cycle) {
        no_bubble.tick();
        record_superlane("pipeline");
        assert(no_bubble.outputs().empty());
    }
    for (std::size_t item = 0; item < 4; ++item) {
        no_bubble.tick();
        record_superlane("pipeline output");
        assert(no_bubble.outputs().size() == 2);
        assert(no_bubble.outputs()[0].stream == 6);
        assert(no_bubble.outputs()[1].stream == 14);
        for (std::size_t lane = 0;
             lane < ftlpu::VxmSuperlane::kLaneCount; ++lane) {
            const auto expected =
                static_cast<float>(lane * 100 + (item + 1) * 10);
            assert(ftlpu::VxmLane::unpack_float32(
                       no_bubble.outputs()[0].byte_values[lane])
                   == expected);
        }
    }
    assert(no_bubble.idle());
    vxm_hardware_test::write_timing_reports(
        "superlane", "VXM Superlane complete timing sequence",
        superlane_timing);

    vxm_hardware_test::write_pass_result(
        "superlane_test_results.txt", "superlane_test");
    return 0;
}
