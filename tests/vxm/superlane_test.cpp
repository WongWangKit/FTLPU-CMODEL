#include "ftlpu/vxm/superlane.hpp"
#include "superlane_operator_workloads.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
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

std::filesystem::path test_output_path(const char* filename)
{
    const auto source = std::filesystem::path(__FILE__);
    if (source.is_absolute()) return source.parent_path() / filename;
    const auto cwd = std::filesystem::current_path();
    if (std::filesystem::exists(cwd / source)) {
        return (cwd / source).parent_path() / filename;
    }
    if (std::filesystem::exists(cwd / "superlane_test.cpp")) {
        return cwd / filename;
    }
    return cwd / filename;
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
            ftlpu::VxmLaneOperand::StreamInt32(),
            ftlpu::VxmLaneOperand::Imm(static_cast<float>(immediate))});
    }
    assert(fifo_limit.instruction_control().fifo_entry_count(0) == 3);
    bool overflow_reported = false;
    try {
        fifo_limit.enqueue_instruction(0, {
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::StreamInt32(),
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
            ftlpu::VxmLaneOperand::StreamInt32(),
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
            const auto bytes = ftlpu::VxmLane::pack_int32(10);
            for (std::size_t byte = 0; byte < 4; ++byte) {
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
        ftlpu::VxmLaneOperand::StreamInt32()};
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
            const auto bytes = ftlpu::VxmLane::pack_int32(
                static_cast<std::int32_t>(cycle * 16 + lane));
            for (std::size_t byte = 0; byte < 4; ++byte) {
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

    const auto result_path =
        test_output_path("superlane_test_results.txt");
    const auto gantt_path =
        test_output_path("superlane_pipeline_gantt.html");
    run_superlane_operator_workloads(
        std::cout, result_path, gantt_path);
    std::cout << "Full Superlane results: "
              << result_path.string() << '\n'
              << "Interactive pipeline Gantt: "
              << gantt_path.string() << '\n';
    return 0;
}
