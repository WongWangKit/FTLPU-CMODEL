#include "ftlpu/core/timing.hpp"
#include "ftlpu/mxm/core.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

namespace {

using ftlpu::MxmCore;

[[noreturn]] void fail_check(const char* expression, int line)
{
    throw std::runtime_error(
        "check failed at line " + std::to_string(line) + ": " + expression);
}

// Keep directed-test failures in the console.  The MSVC Debug CRT turns the
// standard assert() failure path into a modal abort dialog, which blocks
// unattended regressions.
#undef assert
#define assert(expression) \
    ((expression) ? static_cast<void>(0) : fail_check(#expression, __LINE__))

// The harness also verifies that heap-backed Core state keeps the same
// externally visible Current/Next contract.
struct Harness {
    MxmCore core{};
    ftlpu::timing::Clock clock{};
    ftlpu::timing::SystemPhase phases{};
    MxmCore::Inputs inputs{};

    Harness()
    {
        clock.register_module(core);
        phases.add_node("MXM new core LOW", [this] { core.low(); },
            [this] { core.high(); });
    }

    MxmCore::Outputs tick()
    {
        core.drive(inputs);
        phases.low();
        const auto sampled = core.outputs();
        clock.complete_low();
        clock.validate_all();
        clock.commit_all();
        phases.high();
        clock.complete_high();
        clock.negedge_all();
        return sampled;
    }
};

void set_weight_streams(MxmCore::Inputs& inputs, std::int8_t quantized = 1)
{
    for (std::size_t row = 0; row < MxmCore::kRows; ++row) {
        for (std::size_t lane = 0; lane < MxmCore::kTerms; ++lane) {
            inputs.weight_stream_valid_i[row][MxmCore::kWeightQStream][lane] = true;
            inputs.weight_stream_valid_i[row][MxmCore::kWeightScaleLoStream][lane] = true;
            inputs.weight_stream_valid_i[row][MxmCore::kWeightScaleHiStream][lane] = true;
            inputs.weight_stream_valid_i[row][MxmCore::kWeightZeroStream][lane] = true;
            inputs.weight_stream_i[row][MxmCore::kWeightQStream][lane]
                = static_cast<std::uint8_t>(quantized);
            inputs.weight_stream_i[row][MxmCore::kWeightScaleLoStream][lane] = 0x80;
            inputs.weight_stream_i[row][MxmCore::kWeightScaleHiStream][lane] = 0x3f;
            inputs.weight_stream_i[row][MxmCore::kWeightZeroStream][lane] = 0;
        }
    }
}

void set_unit_activations(MxmCore::Inputs& inputs)
{
    for (std::size_t row = 0; row < MxmCore::kRows; ++row) {
        inputs.activation_valid_i[row] = true;
        inputs.activation_i[row].fill(0x3f80); // BF16 1.0.
    }
}

void verify_global_clock_and_queue_hold()
{
    Harness harness;
    harness.inputs.rst_ni = false;
    harness.tick();
    assert(harness.clock.phase_count() == 4);
    assert(harness.clock.cycle() == 1);
    assert(harness.phases.low_count() == 1);
    assert(harness.phases.high_count() == 1);

    harness.inputs.rst_ni = true;
    harness.inputs.run_i = false;
    harness.inputs.compute_enqueue_valid_i = true;
    harness.inputs.compute_instruction_i = {7, true};
    const auto enqueue = harness.tick();
    assert(enqueue.compute_enqueue_ready_o);
    assert(enqueue.compute_enqueue_fire);
    assert(!enqueue.compute_launch_ready_o);
    assert(harness.core.compute_queue_level() == 1);

    harness.inputs.compute_enqueue_valid_i = false;
    harness.inputs.run_i = true;
    harness.inputs.compute_launch_i = true;
    const auto launch = harness.tick();
    assert(launch.compute_launch_ready_o);
    assert(launch.compute_launch_fire);
    assert(!launch.compute_row_valid_o[0]); // Next is not visible before commit.
    assert(harness.core.compute_queue_level() == 0);
    assert(harness.core.outputs().compute_row_valid_o[0]);
    assert(harness.core.outputs().compute_row_tag_o[0] == 0);

    harness.inputs.compute_launch_i = false;
    for (std::size_t cycle = 0; cycle < 2; ++cycle) {
        const auto out = harness.tick();
        assert(!out.compute_row_valid_o[1]);
    }
    const auto row_one = harness.tick();
    // INTERVAL=3 means the third relay edge commits row1.  The LOW sample
    // still sees the old Current; HIGH observes the newly committed row.
    assert(!row_one.compute_row_valid_o[1]);
    assert(harness.core.outputs().compute_row_valid_o[1]);
    assert(harness.core.outputs().compute_row_instruction_o[1].activation_stream == 7);
    assert(harness.core.outputs().compute_row_instruction_o[1].weight_buffer);
}

void verify_queue_full_pop_push()
{
    Harness harness;
    harness.inputs.rst_ni = false;
    harness.tick();
    harness.inputs.rst_ni = true;
    harness.inputs.run_i = false;
    harness.inputs.compute_enqueue_valid_i = true;
    for (std::size_t index = 0; index < MxmCore::kQueueDepth; ++index) {
        harness.inputs.compute_instruction_i.activation_stream
            = static_cast<std::uint8_t>(index);
        const auto out = harness.tick();
        assert(out.compute_enqueue_fire);
    }
    assert(harness.core.compute_queue_level() == MxmCore::kQueueDepth);
    const auto blocked = harness.tick();
    assert(!blocked.compute_enqueue_ready_o && !blocked.compute_enqueue_fire);

    harness.inputs.run_i = true;
    harness.inputs.compute_launch_i = true;
    harness.inputs.compute_instruction_i.activation_stream = 63;
    const auto replace = harness.tick();
    assert(replace.compute_launch_fire && replace.compute_enqueue_fire);
    assert(harness.core.compute_queue_level() == MxmCore::kQueueDepth);
}

void reset_and_start(Harness& harness)
{
    harness.inputs.rst_ni = false;
    harness.tick();
    harness.inputs.rst_ni = true;
    harness.inputs.run_i = false;
    set_weight_streams(harness.inputs);
    set_unit_activations(harness.inputs);
}

void enqueue_parameter_command(Harness& harness)
{
    harness.inputs.param_load_enqueue_valid_i = true;
    harness.inputs.param_load_instruction_i.param_buffer = false;
    assert(harness.tick().param_load_enqueue_fire);
    harness.inputs.param_load_enqueue_valid_i = false;
    harness.inputs.run_i = true;
    harness.inputs.param_load_launch_i = true;
    assert(harness.tick().param_load_launch_fire);
    harness.inputs.param_load_launch_i = false;
    // The parameter wave is spaced three cycles per row.  Keep the fixed
    // byte streams valid until every row has accepted its command.
    for (std::size_t cycle = 0; cycle < 12; ++cycle)
        harness.tick();
}

void load_weight_matrix(Harness& harness)
{
    harness.inputs.run_i = false;
    harness.inputs.load_enqueue_valid_i = true;
    harness.inputs.load_instruction_i = {false, false};
    assert(harness.tick().load_enqueue_fire);
    harness.inputs.load_enqueue_valid_i = false;

    harness.inputs.dequant_enqueue_valid_i = true;
    harness.inputs.dequant_instruction_i.param_buffer = false;
    for (std::size_t index = 0; index < MxmCore::kQueueDepth; ++index)
        assert(harness.tick().dequant_enqueue_fire);
    harness.inputs.dequant_enqueue_valid_i = false;

    harness.inputs.run_i = true;
    harness.inputs.load_launch_i = true;
    harness.inputs.dequant_launch_i = true;
    for (std::size_t beat = 0; beat < MxmCore::kMatrixN; ++beat) {
        harness.inputs.load_launch_i = beat == 0;
        harness.inputs.dequant_enqueue_valid_i = beat < MxmCore::kQueueDepth;
        const auto out = harness.tick();
        if (beat == 0) assert(out.load_launch_fire);
        assert(out.dequant_launch_fire);
        if (beat < MxmCore::kQueueDepth) assert(out.dequant_enqueue_fire);
    }
    harness.inputs.load_launch_i = false;
    harness.inputs.dequant_launch_i = false;
    harness.inputs.dequant_enqueue_valid_i = false;

    // Drain D1/D2/D3 and the registered horizontal weight chain for the last
    // row and last column.
    for (std::size_t cycle = 0; cycle < 20; ++cycle) {
        const auto out = harness.tick();
        assert(!out.frontend_fault_o);
    }
}

struct RunStats {
    std::size_t partials{};
    std::size_t results{};
};

RunStats run_compute_with_backend(Harness& harness,
    MxmCore::BackendOperation operation, std::uint16_t address)
{
    // Drive only real valid commands; payload bits outside valid cycles are
    // intentionally don't-care, as in the RTL interface contract.
    // Keep run asserted: RTL uses run_i in the Supercell reset tree, so
    // deasserting it here would intentionally invalidate loaded weights.
    harness.inputs.run_i = true;
    harness.inputs.compute_enqueue_valid_i = true;
    harness.inputs.compute_instruction_i = {0, false};
    assert(harness.tick().compute_enqueue_fire);
    harness.inputs.compute_enqueue_valid_i = false;
    harness.inputs.backend_enqueue_valid_i = true;
    harness.inputs.backend_instruction_i = {false, address, operation};
    assert(harness.tick().backend_enqueue_fire);
    harness.inputs.backend_enqueue_valid_i = false;
    harness.inputs.run_i = true;

    RunStats stats{};
    harness.inputs.compute_launch_i = true;
    assert(harness.tick().compute_launch_fire);
    harness.inputs.compute_launch_i = false;
    for (std::size_t offset = 1; offset <= 40; ++offset) {
        harness.inputs.backend_launch_i = offset == 23;
        const auto out = harness.tick();
        if (offset == 23) assert(out.backend_launch_fire);
        for (bool valid : out.partial_valid_o) stats.partials += valid ? 1 : 0;
        for (std::size_t column = 0; column < MxmCore::kColumns; ++column) {
            if (!out.result_valid_o[column]) continue;
            ++stats.results;
            assert(!out.result_bf16_o[column]);
            for (const auto word : out.result_o[column]) assert(word != 0);
        }
        assert(!out.frontend_fault_o);
        if (out.backend_fault_o) {
            std::cerr << "backend fault offset=" << offset << " returned=";
            for (const auto value : out.returned_valid_o) std::cerr << value;
            std::cerr << " slot=";
            for (const auto value : out.backend_slot_valid_o) std::cerr << value;
            std::cerr << " issue=";
            for (const auto value : out.backend_issue_valid_o) std::cerr << value;
            std::cerr << '\n';
            throw std::runtime_error("unexpected backend fault");
        }
    }
    harness.inputs.backend_launch_i = false;
    return stats;
}

void verify_new_core_pipeline_and_accumulator()
{
    Harness harness;
    reset_and_start(harness);
    enqueue_parameter_command(harness);
    load_weight_matrix(harness);

    const auto stored = run_compute_with_backend(
        harness, MxmCore::BackendOperation::Store, 0);
    assert(stored.partials == MxmCore::kColumns);
    assert(stored.results == 0);
    for (std::size_t column = 0; column < MxmCore::kColumns; ++column)
        assert(harness.core.accumulator_initialized(column, 0));

    const auto accumulated = run_compute_with_backend(
        harness, MxmCore::BackendOperation::AccOut, 0);
    assert(accumulated.partials == MxmCore::kColumns);
    assert(accumulated.results == MxmCore::kColumns);
}

} // namespace

int main()
{
#if defined(_MSC_VER)
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        verify_global_clock_and_queue_hold();
        verify_queue_full_pop_push();
        verify_new_core_pipeline_and_accumulator();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mxm_core_clocked_test: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "mxm_core_clocked_test: unknown failure\n";
        return 1;
    }
}
