#pragma once

#include "ftlpu/system/tsp_slice_system.hpp"
#include "system_timing_report.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

namespace integration_timing {

// One common, read-only trace collector for black-box system tests.  Tests
// load data/instructions and tick TspSliceSystem as before; this class only
// observes the post-cycle system snapshot and never drives a module directly.
class SystemGanttTrace {
public:
    enum class Detail { Chip, Superlane, Full };

    // The CMake option provides the normal default.  FTLPU_GANTT can
    // override it per process without rebuilding: 1/on/true enables,
    // while 0/off/false disables collection.
    static bool enabled() noexcept
    {
        if (const auto* configured = std::getenv("FTLPU_GANTT")) {
            const auto value = std::string_view {configured};
            if (value.empty() || value == "0" || value == "off"
                || value == "OFF" || value == "false"
                || value == "FALSE") {
                return false;
            }
            return true;
        }
#if defined(FTLPU_ENABLE_GANTT) && FTLPU_ENABLE_GANTT
        return true;
#else
        return false;
#endif
    }

    static Detail detail() noexcept
    {
        static const auto configured_detail = [] {
            if (const auto* configured = std::getenv("FTLPU_GANTT")) {
                const auto value = std::string_view {configured};
                if (value == "full" || value == "FULL") {
                    return Detail::Full;
                }
                if (value == "superlane" || value == "SUPERLANE") {
                    return Detail::Superlane;
                }
            }
            return Detail::Chip;
        }();
        return configured_detail;
    }

    void attach(ftlpu::TspSliceSystem& system)
    {
        system.set_timing_observer(
            [this](const auto& snapshot) { capture(snapshot); });
    }

    void detach(ftlpu::TspSliceSystem& system) noexcept
    {
        system.clear_timing_observer();
    }

    void capture(const ftlpu::TspSliceSystem& system)
    {
        capture(system.system_timing_snapshot());
    }

    void capture(const ftlpu::TspSliceSystem::SystemTimingSnapshot& snapshot)
    {
        // Individual black-box phases intentionally reset their local CModel
        // cycle.  Detect that boundary and retain one monotonically increasing
        // chip-level trace without modifying the phase implementations.
        if (last_local_cycle_.has_value()
            && snapshot.cycle <= *last_local_cycle_) {
            cycle_offset_ = last_absolute_cycle_ + 1;
        }
        const auto cycle = cycle_offset_ + snapshot.cycle;
        last_local_cycle_ = snapshot.cycle;
        last_absolute_cycle_ = cycle;
        capture_icu(snapshot.icu, cycle);
        capture_mem(snapshot, cycle);
        capture_mxm(snapshot, cycle);
        capture_sxm(snapshot, cycle);
        capture_vxm(snapshot.vxm, cycle);
    }

    void phase(
        std::size_t first_cycle, std::size_t end_cycle,
        std::string label)
    {
        report_.add(
            "Protocol phase", first_cycle, end_cycle,
            std::move(label), "compute");
    }

    void write(std::string_view prefix, std::string_view title) const
    {
        auto format = Report::OutputFormat::Both;
        if (const auto* configured = std::getenv("FTLPU_GANTT_FORMAT")) {
            const auto value = std::string_view {configured};
            if (value == "csv" || value == "CSV") {
                format = Report::OutputFormat::Csv;
            }
            else if (value == "html" || value == "HTML") {
                format = Report::OutputFormat::Html;
            }
        }
        report_.write(prefix, title, format);
    }

    static std::string prefix_from_name(std::string_view name)
    {
        auto result = std::string {};
        result.reserve(name.size());
        auto underscore = false;
        for (const auto ch : name) {
            const auto byte = static_cast<unsigned char>(ch);
            if (std::isalnum(byte)) {
                result += static_cast<char>(std::tolower(byte));
                underscore = false;
            }
            else if (!result.empty() && !underscore) {
                result += '_';
                underscore = true;
            }
        }
        while (!result.empty() && result.back() == '_') result.pop_back();
        return result.empty() ? "system" : result;
    }

private:
    static std::string hemisphere_name(std::size_t index)
    {
        return index == ftlpu::hemisphere_index(ftlpu::Hemisphere::East)
            ? "East" : "West";
    }

    static double utilization(std::size_t active, std::size_t capacity)
    {
        if (capacity == 0) return 0.0;
        return std::clamp(
            static_cast<double>(active) / static_cast<double>(capacity),
            0.0, 1.0);
    }

    void capture_icu(
        const ftlpu::InstructionControlUnit::TimingSnapshot& timing,
        std::size_t cycle)
    {
        if (timing.total_issues() == 0) return;
        if (detail() == Detail::Chip) {
            report_.add_or_extend(
                "ICU dispatch", cycle, cycle + 1,
                "instruction dispatch", "config");
            return;
        }
        auto label = std::string {"issues="}
            + std::to_string(timing.total_issues());
        if (timing.mem_issues != 0) {
            label += " MEM=" + std::to_string(timing.mem_issues);
        }
        if (timing.mxm_load_issues != 0) {
            label += " MXM-load=" + std::to_string(timing.mxm_load_issues);
        }
        if (timing.mxm_compute_issues != 0) {
            label += " MXM-compute=" + std::to_string(timing.mxm_compute_issues);
        }
        if (timing.sxm_transpose_issues != 0
            || timing.sxm_permute_issues != 0) {
            label += " SXM=" + std::to_string(
                timing.sxm_transpose_issues + timing.sxm_permute_issues);
        }
        if (timing.vxm_issues != 0) {
            label += " VXM=" + std::to_string(timing.vxm_issues);
        }
        report_.add_or_extend(
            "ICU dispatch", cycle, cycle + 1,
            std::move(label), "config",
            utilization(timing.total_issues(), 16));
    }

    void capture_mem(
        const ftlpu::TspSliceSystem::SystemTimingSnapshot& snapshot,
        std::size_t cycle)
    {
        for (std::size_t hemisphere = 0;
             hemisphere < snapshot.mems.size(); ++hemisphere) {
            const auto& timing = snapshot.mems[hemisphere];
            const auto name = hemisphere_name(hemisphere);
            if (timing.reads != 0) {
                if (detail() == Detail::Chip) {
                    report_.add_or_extend(
                        "MEM " + name + " Read", cycle, cycle + 1,
                        "read active", "read");
                }
                else {
                report_.add_or_extend(
                    "MEM " + name + " Read", cycle, cycle + 1,
                    "transfers=" + std::to_string(timing.reads), "read",
                    utilization(
                        timing.reads,
                        ftlpu::hw::kMemSliceColumns
                            * ftlpu::hw::kTileRows));
                }
            }
            if (timing.writes != 0) {
                if (detail() == Detail::Chip) {
                    report_.add_or_extend(
                        "MEM " + name + " Write", cycle, cycle + 1,
                        "write active", "write");
                }
                else {
                report_.add_or_extend(
                    "MEM " + name + " Write", cycle, cycle + 1,
                    "transfers=" + std::to_string(timing.writes), "write",
                    utilization(
                        timing.writes,
                        ftlpu::hw::kMemSliceColumns
                            * ftlpu::hw::kTileRows));
                }
            }
        }
    }

    void capture_mxm(
        const ftlpu::TspSliceSystem::SystemTimingSnapshot& snapshot,
        std::size_t cycle)
    {
        constexpr auto capacity = ftlpu::hw::kMxmSupercellsPerPlane
            * ftlpu::hw::kMxmSupercellsPerPlane;
        for (std::size_t mxm = 0; mxm < snapshot.mxms.size(); ++mxm) {
            const auto& timing = snapshot.mxms[mxm];
            if (timing.compute_issues == 0
                && timing.computing_cells == 0 && timing.outputs == 0) {
                continue;
            }
            if (detail() == Detail::Chip) {
                report_.add_or_extend(
                    "MXM " + std::to_string(mxm), cycle, cycle + 1,
                    "array active", "compute");
                continue;
            }
            auto label = std::string {"active cells="}
                + std::to_string(timing.computing_cells);
            if (timing.compute_issues != 0) {
                label += " inputs=" + std::to_string(timing.compute_issues);
            }
            if (timing.outputs != 0) {
                label += " outputs=" + std::to_string(timing.outputs);
            }
            report_.add_or_extend(
                "MXM " + std::to_string(mxm), cycle, cycle + 1,
                std::move(label), "compute",
                utilization(timing.computing_cells, capacity));
            if (timing.deskew_writes != 0
                || timing.deskew_vectors != 0) {
                report_.add_or_extend(
                    "MXM " + std::to_string(mxm) + " Lane Deskew",
                    cycle, cycle + 1,
                    "lane writes=" + std::to_string(timing.deskew_writes)
                        + " flat vectors="
                        + std::to_string(timing.deskew_vectors),
                    "compute");
            }
        }
    }

    void capture_sxm(
        const ftlpu::TspSliceSystem::SystemTimingSnapshot& snapshot,
        std::size_t cycle)
    {
        for (std::size_t hemisphere = 0;
             hemisphere < snapshot.sxms.size(); ++hemisphere) {
            const auto& timing = snapshot.sxms[hemisphere];
            const auto name = hemisphere_name(hemisphere);
            const auto transpose = timing.captured_rows
                + timing.transpose_rows;
            if (transpose != 0 || timing.transpose_bank_loads != 0) {
                if (detail() == Detail::Chip) {
                    report_.add_or_extend(
                        "SXM " + name + " Transpose", cycle, cycle + 1,
                        "transpose active", "compute");
                }
                else {
                report_.add_or_extend(
                    "SXM " + name + " Transpose", cycle, cycle + 1,
                    "capture=" + std::to_string(timing.captured_rows)
                        + " bank-load="
                        + std::to_string(timing.transpose_bank_loads)
                        + " output="
                        + std::to_string(timing.transpose_rows),
                    "compute", utilization(
                        transpose + timing.transpose_bank_loads,
                        3 * ftlpu::hw::kTileRows));
                }
            }
            if (timing.permute_rows != 0) {
                if (detail() == Detail::Chip) {
                    report_.add_or_extend(
                        "SXM " + name + " Permute", cycle, cycle + 1,
                        "permute active", "compute");
                }
                else {
                report_.add_or_extend(
                    "SXM " + name + " Permute", cycle, cycle + 1,
                    "output rows=" + std::to_string(timing.permute_rows),
                    "compute", utilization(
                        timing.permute_rows, ftlpu::hw::kTileRows));
                }
            }
        }
    }

    void capture_vxm(const ftlpu::TspSliceSystem::VxmTimingSnapshot& snapshot)
    {
        capture_vxm(snapshot, snapshot.cycle);
    }

    void capture_vxm(
        const ftlpu::TspSliceSystem::VxmTimingSnapshot& snapshot,
        std::size_t cycle)
    {
        auto total_executed = std::size_t {0};
        auto total_stalled = std::size_t {0};
        auto total_outputs = std::size_t {0};
        for (std::size_t superlane = 0;
             superlane < snapshot.superlanes.size(); ++superlane) {
            const auto& timing = snapshot.superlanes[superlane];
            const auto executed = static_cast<std::size_t>(std::count(
                timing.alu_states.begin(), timing.alu_states.end(),
                ftlpu::VxmLaneAluTraceState::Executed));
            const auto stalled = static_cast<std::size_t>(std::count(
                timing.alu_states.begin(), timing.alu_states.end(),
                ftlpu::VxmLaneAluTraceState::Stalled));
            total_executed += executed;
            total_stalled += stalled;
            total_outputs += timing.outputs;
            if (executed == 0 && stalled == 0 && timing.outputs == 0) continue;
            if (detail() == Detail::Chip) continue;
            auto label = std::string {"depth="}
                + std::to_string(static_cast<std::size_t>(timing.depth))
                + " active ALUs " + std::to_string(executed);
            if (stalled != 0) {
                label += " stalled=" + std::to_string(stalled);
            }
            if (timing.outputs != 0) {
                label += " outputs=" + std::to_string(timing.outputs);
            }
            report_.add_or_extend(
                "VXM Superlane " + std::to_string(superlane),
                cycle, cycle + 1,
                std::move(label), executed == 0 ? "config" : "compute",
                utilization(executed, ftlpu::VxmLane::kAluCount));
            if (detail() == Detail::Full) {
                for (std::size_t alu = 0;
                     alu < timing.alu_states.size(); ++alu) {
                    const auto state = timing.alu_states[alu];
                    if (state == ftlpu::VxmLaneAluTraceState::Idle) continue;
                    report_.add_or_extend(
                        "VXM SL" + std::to_string(superlane)
                            + " ALU" + std::to_string(alu),
                        cycle, cycle + 1,
                        state == ftlpu::VxmLaneAluTraceState::Executed
                            ? "executed" : "stalled",
                        state == ftlpu::VxmLaneAluTraceState::Executed
                            ? "compute" : "config");
                }
            }
        }
        if (detail() == Detail::Chip
            && (total_executed != 0 || total_stalled != 0
                || total_outputs != 0)) {
            auto label = std::string {"active ALUs="}
                + std::to_string(total_executed);
            if (total_stalled != 0) {
                label += " stalled=" + std::to_string(total_stalled);
            }
            if (total_outputs != 0) {
                label += " outputs=" + std::to_string(total_outputs);
            }
            report_.add_or_extend(
                "VXM", cycle, cycle + 1, "vector array active",
                total_executed == 0 ? "config" : "compute",
                1.0);
        }
    }

    Report report_{};
    std::optional<std::size_t> last_local_cycle_{};
    std::size_t cycle_offset_{0};
    std::size_t last_absolute_cycle_{0};
};

} // namespace integration_timing
