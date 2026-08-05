#include "pipelined_edge_ffn_harness.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

void require_same_datapath_timeline(
    const full_ffn_test::FullFfnResult& direct,
    const full_ffn_test::FullFfnResult& replay)
{
    if (direct.timeline.size() != replay.timeline.size()) {
        throw std::runtime_error("ICU replay changed the timeline length");
    }

    for (std::size_t index = 0; index < direct.timeline.size(); ++index) {
        const auto& lhs = direct.timeline[index];
        const auto& rhs = replay.timeline[index];
        const auto same = lhs.cycle == rhs.cycle
            && lhs.phase == rhs.phase
            && lhs.events == rhs.events
            && lhs.gate_mxm_cells == rhs.gate_mxm_cells
            && lhs.up_mxm_cells == rhs.up_mxm_cells
            && lhs.vxm_alu_slots == rhs.vxm_alu_slots
            && lhs.quantized_outputs == rhs.quantized_outputs
            && lhs.mem_reads == rhs.mem_reads
            && lhs.mem_writes == rhs.mem_writes
            && lhs.sxm_transpose_tiles == rhs.sxm_transpose_tiles
            && lhs.sxm_permute_tiles == rhs.sxm_permute_tiles
            && lhs.gate_up_weight_load_tiles
                == rhs.gate_up_weight_load_tiles
            && lhs.down_weight_load_tiles == rhs.down_weight_load_tiles
            && lhs.down_mxm_cells == rhs.down_mxm_cells
            && lhs.down_acc_outputs == rhs.down_acc_outputs;
        if (!same) {
            auto message = std::ostringstream{};
            message << "ICU replay changed datapath activity at cycle "
                    << index;
            throw std::runtime_error(message.str());
        }
    }
}

} // namespace

int main()
try {
    auto compiler_run = full_ffn_test::PipelinedEdgeFfnHarness{};
    const auto direct = compiler_run.run();
    const auto& program = compiler_run.compiled_icu_program();
    if (!direct.correctness_passed || !direct.functionality_passed
        || program.functional_events() == 0
        || program.active_queues() == 0) {
        throw std::runtime_error(
            "FFN compiler pass did not produce a valid ICU program");
    }

    auto icu_run = full_ffn_test::PipelinedEdgeFfnHarness{program};
    const auto result = icu_run.run();
    if (!result.correctness_passed || !result.functionality_passed) {
        throw std::runtime_error("ICU-driven FFN result failed validation");
    }
    if (result.cycles != direct.cycles
        || result.down_values_checked != direct.down_values_checked
        || result.front_compute_issues != direct.front_compute_issues
        || result.down_compute_issues != direct.down_compute_issues) {
        throw std::runtime_error(
            "ICU-driven FFN changed the direct schedule or workload result");
    }
    require_same_datapath_timeline(direct, result);

    auto source = std::filesystem::path(__FILE__);
    auto output = source.is_absolute()
        ? source.parent_path()
        : std::filesystem::current_path() / source.parent_path();
    if (!std::filesystem::exists(output)) {
        output = std::filesystem::current_path() / "tests" / "integration"
            / "icu_driven_transformer_ffn";
    }
    full_ffn_test::write_reports(result, output / "results");

    std::cout << "PASS icu_driven_edge_ffn"
              << " cycles=" << result.cycles
              << " active_queues=" << program.active_queues()
              << " control_events=" << program.functional_events()
              << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ICU-driven FFN failed: " << error.what() << '\n';
    return 1;
}
