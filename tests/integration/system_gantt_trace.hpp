#pragma once

#include "ftlpu/system/tsp_slice_system.hpp"
#include "system_timing_report.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>

namespace integration_timing {

// One common, read-only trace collector for black-box system tests.  Tests
// load data/instructions and tick TspSliceSystem as before; this class only
// observes the post-cycle system snapshot and never drives a module directly.
class SystemGanttTrace {
public:
    void capture(const ftlpu::TspSliceSystem& system)
    {
        capture(system.system_timing_snapshot());
    }

    void capture(const ftlpu::TspSliceSystem::SystemTimingSnapshot& snapshot)
    {
        const auto cycle = snapshot.cycle;
        capture_icu(snapshot.icu, cycle);
        capture_mem(snapshot, cycle);
        capture_mxm(snapshot, cycle);
        capture_sxm(snapshot, cycle);
        capture_vxm(snapshot.vxm);
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
        report_.write(prefix, title);
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
                report_.add_or_extend(
                    "MEM " + name + " Read", cycle, cycle + 1,
                    "transfers=" + std::to_string(timing.reads), "read",
                    utilization(
                        timing.reads,
                        ftlpu::hw::kMemSliceColumns
                            * ftlpu::hw::kTileRows));
            }
            if (timing.writes != 0) {
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
            if (transpose != 0) {
                report_.add_or_extend(
                    "SXM " + name + " Transpose", cycle, cycle + 1,
                    "capture=" + std::to_string(timing.captured_rows)
                        + " pipeline=" + std::to_string(timing.transpose_rows),
                    "compute", utilization(
                        transpose, 2 * ftlpu::hw::kTileRows));
            }
            if (timing.permute_rows != 0) {
                report_.add_or_extend(
                    "SXM " + name + " Permute", cycle, cycle + 1,
                    "output rows=" + std::to_string(timing.permute_rows),
                    "compute", utilization(
                        timing.permute_rows, ftlpu::hw::kTileRows));
            }
        }
    }

    void capture_vxm(const ftlpu::TspSliceSystem::VxmTimingSnapshot& snapshot)
    {
        for (std::size_t superlane = 0;
             superlane < snapshot.superlanes.size(); ++superlane) {
            const auto& timing = snapshot.superlanes[superlane];
            const auto executed = static_cast<std::size_t>(std::count(
                timing.alu_states.begin(), timing.alu_states.end(),
                ftlpu::VxmLaneAluTraceState::Executed));
            const auto stalled = static_cast<std::size_t>(std::count(
                timing.alu_states.begin(), timing.alu_states.end(),
                ftlpu::VxmLaneAluTraceState::Stalled));
            if (executed == 0 && stalled == 0 && timing.outputs == 0) continue;
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
                snapshot.cycle, snapshot.cycle + 1,
                std::move(label), executed == 0 ? "config" : "compute",
                utilization(executed, ftlpu::VxmLane::kAluCount));
        }
    }

    Report report_{};
};

} // namespace integration_timing
