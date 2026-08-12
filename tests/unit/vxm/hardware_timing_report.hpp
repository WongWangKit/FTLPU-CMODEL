#pragma once

#include "ftlpu/vxm/superlane.hpp"
#include "hardware_test_output.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vxm_hardware_test {

struct TimingCycle {
    using AluStates = std::array<
        ftlpu::VxmLaneAluTraceState, ftlpu::VxmLane::kAluCount>;

    std::size_t cycle{0};
    ftlpu::VxmChainDepth depth{ftlpu::VxmChainDepth::Two};
    std::vector<AluStates> superlanes{};
    std::size_t pending_feedback{0};
    std::size_t outputs{0};
    std::size_t lockstep_lanes{1};
    std::string event{};
};

inline TimingCycle capture_cycle(
    std::size_t cycle, const ftlpu::VxmLane& lane,
    std::size_t outputs, std::string event)
{
    auto result = TimingCycle{};
    result.cycle = cycle;
    result.depth = lane.chain_depth();
    result.pending_feedback = lane.feedback_pending_count();
    result.outputs = outputs;
    result.lockstep_lanes = 1;
    result.event = std::move(event);
    result.superlanes.emplace_back();
    for (std::size_t stage = 0;
         stage < ftlpu::VxmLane::kAluCount; ++stage) {
        result.superlanes.back()[stage] = lane.last_trace()[stage].state;
    }
    return result;
}

inline TimingCycle capture_cycle(
    std::size_t cycle, const ftlpu::VxmSuperlane& superlane,
    std::size_t outputs, std::string event)
{
    auto result = TimingCycle{};
    result.cycle = cycle;
    result.depth = superlane.lane(0).chain_depth();
    result.pending_feedback =
        superlane.lane(0).feedback_pending_count();
    result.outputs = outputs;
    result.event = std::move(event);
    result.lockstep_lanes = ftlpu::VxmSuperlane::kLaneCount;
    result.superlanes.emplace_back();
    for (std::size_t stage = 0;
         stage < ftlpu::VxmLane::kAluCount; ++stage) {
        result.superlanes.back()[stage] =
            superlane.lane(0).last_trace()[stage].state;
    }
    return result;
}

inline char state_code(ftlpu::VxmLaneAluTraceState state)
{
    switch (state) {
    case ftlpu::VxmLaneAluTraceState::Executed: return 'E';
    case ftlpu::VxmLaneAluTraceState::Stalled: return 'S';
    case ftlpu::VxmLaneAluTraceState::Idle: return '.';
    }
    return '?';
}

inline void write_cycle_details(
    std::string_view prefix, std::string_view title,
    const std::vector<TimingCycle>& trace)
{
    const auto path = results_directory()
        / (std::string{prefix} + "_cycle_details.txt");
    auto file = std::ofstream{path, std::ios::trunc};
    if (!file) {
        throw std::runtime_error(
            "cannot create VXM hardware cycle details");
    }
    file << "VXM hardware cycle details\n"
         << "test=" << title << '\n'
         << "state: E=executed S=stalled .=idle\n\n";
    for (const auto& cycle : trace) {
        file << "cycle=" << cycle.cycle
             << " event=" << cycle.event
             << " depth=" << static_cast<std::size_t>(cycle.depth)
             << " pending_feedback=" << cycle.pending_feedback
             << " outputs=" << cycle.outputs
             << " lockstep_lanes=" << cycle.lockstep_lanes;
        for (std::size_t superlane = 0;
             superlane < cycle.superlanes.size(); ++superlane) {
            file << " superlane" << superlane << '=';
            for (const auto state : cycle.superlanes[superlane]) {
                file << state_code(state);
            }
        }
        file << '\n';
    }
}

inline void write_timing_gantt(
    std::string_view prefix, std::string_view title,
    const std::vector<TimingCycle>& trace)
{
    const auto path = results_directory()
        / (std::string{prefix} + "_timing_gantt.html");
    auto file = std::ofstream{path, std::ios::trunc};
    if (!file) {
        throw std::runtime_error(
            "cannot create VXM hardware timing Gantt");
    }
    const auto event_style = [](const TimingCycle& cycle) {
        if (cycle.outputs != 0) return std::string_view{"output"};
        if (cycle.event.find("config") != std::string::npos
            || cycle.event.find("decode") != std::string::npos) {
            return std::string_view{"config"};
        }
        if (cycle.event.find("transition") != std::string::npos
            || cycle.event.find("feedback") != std::string::npos) {
            return std::string_view{"transition"};
        }
        if (cycle.event.find("input") != std::string::npos
            || cycle.event.find("head") != std::string::npos) {
            return std::string_view{"data"};
        }
        return std::string_view{"idle"};
    };
    const auto write_header = [&file, &trace]() {
        file << "<tr><th class=\"label\">Item</th>";
        for (std::size_t index = 0; index < trace.size(); ++index) {
            const auto show = index % 5 == 0 || index + 1 == trace.size();
            file << "<th class=\"tick\" title=\"Cycle "
                 << trace[index].cycle << "\">";
            if (show) file << trace[index].cycle;
            file << "</th>";
        }
        file << "</tr>";
    };

    file << R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>VXM hardware timing Gantt</title>
<style>
:root{--tick:14px;--label:150px}
body{font:13px system-ui;margin:18px;background:#f4f7fb;color:#172033}
h1{font-size:20px;margin:0 0 4px}p{margin:4px 0 12px;color:#657188}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 12px}
.key:before{content:"";display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;background:var(--c)}
.wrap{overflow:auto;border-radius:7px;box-shadow:0 2px 10px #ccd5e2;margin-bottom:12px;background:white}
table{border-collapse:collapse;table-layout:fixed;width:max-content;background:white}
th,td{border:1px solid #d7deea;text-align:center;height:20px}
.label{box-sizing:border-box;width:var(--label);min-width:var(--label);max-width:var(--label);padding:3px 7px;text-align:left;position:sticky;left:0;z-index:2;background:white}
.tick,.cell{box-sizing:border-box;width:var(--tick);min-width:var(--tick);max-width:var(--tick);padding:1px 0;font-size:8px;overflow:visible}
.band{padding:2px 5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:left;font-size:11px}
.e{background:#31b889;color:#fff}.s{background:#e36a6a;color:#fff}.idle{background:#edf1f6;color:#506075}
.config{background:#ef9f32;color:#fff}.data{background:#4185d7;color:#fff}
.transition{background:#8b67d4;color:#fff}.output{background:#9b62d1;color:#fff}
details{margin:9px 0}summary{cursor:pointer;font-weight:650;color:#33445f;padding:4px 2px}
.phase-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:5px 12px;margin:8px 0 14px}
.phase{background:white;border-left:5px solid var(--c);padding:5px 8px;border-radius:3px;box-shadow:0 1px 3px #d9e0ea}
</style></head><body><h1>)HTML"
         << title
         << R"HTML(</h1>
<p>Compact cycle view. SIMD lanes inside one Superlane are lockstep and therefore aggregated; hover a phase or cell for full details.</p>
<div class="legend">
 <span class="key" style="--c:#31b889">Executed</span>
 <span class="key" style="--c:#e36a6a">Stalled</span>
 <span class="key" style="--c:#ef9f32">Config</span>
 <span class="key" style="--c:#4185d7">Input</span>
 <span class="key" style="--c:#8b67d4">Feedback/transition</span>
 <span class="key" style="--c:#9b62d1">Output</span>
</div><div class="phase-list">)HTML";

    for (std::size_t first = 0; first < trace.size();) {
        auto last = first + 1;
        while (last < trace.size()
               && trace[last].event == trace[first].event
               && event_style(trace[last]) == event_style(trace[first])) {
            ++last;
        }
        file << "<div class=\"phase\" style=\"--c:";
        const auto style = event_style(trace[first]);
        if (style == "config") file << "#ef9f32";
        else if (style == "data") file << "#4185d7";
        else if (style == "transition") file << "#8b67d4";
        else if (style == "output") file << "#9b62d1";
        else file << "#aeb8c8";
        file << "\"><b>C" << trace[first].cycle;
        if (last - first > 1) file << "-C" << trace[last - 1].cycle;
        file << "</b> " << trace[first].event << "</div>";
        first = last;
    }
    file << "</div><div class=\"wrap\"><table>";
    write_header();
    file << "<tr><td class=\"label\">Phase</td>";
    for (std::size_t first = 0; first < trace.size();) {
        auto last = first + 1;
        while (last < trace.size()
               && trace[last].event == trace[first].event
               && event_style(trace[last]) == event_style(trace[first])) {
            ++last;
        }
        file << "<td colspan=\"" << (last - first)
             << "\" class=\"band " << event_style(trace[first])
             << "\" title=\"C" << trace[first].cycle << "-C"
             << trace[last - 1].cycle << ": " << trace[first].event
             << "\">";
        if (last - first >= 6) file << trace[first].event;
        file << "</td>";
        first = last;
    }
    file << "</tr><tr><td class=\"label\">Chain depth</td>";
    for (const auto& cycle : trace) {
        file << "<td class=\"cell idle\" title=\"Depth "
             << static_cast<std::size_t>(cycle.depth) << "\">"
             << static_cast<std::size_t>(cycle.depth) << "</td>";
    }
    file << "</tr><tr><td class=\"label\">Outputs</td>";
    for (const auto& cycle : trace) {
        file << "<td class=\"cell "
             << (cycle.outputs == 0 ? "idle" : "output")
             << "\" title=\"Outputs: " << cycle.outputs << "\">";
        if (cycle.outputs != 0) file << cycle.outputs;
        file << "</td>";
    }
    file << "</tr><tr><td class=\"label\">Superlane 0 VXM ALUs</td>";
    for (const auto& cycle : trace) {
        auto executed = std::size_t {0};
        auto stalled = std::size_t {0};
        if (!cycle.superlanes.empty()) {
            for (const auto state : cycle.superlanes.front()) {
                executed += state == ftlpu::VxmLaneAluTraceState::Executed;
                stalled += state == ftlpu::VxmLaneAluTraceState::Stalled;
            }
        }
        const auto* style = executed != 0 ? "e" : stalled != 0 ? "s" : "idle";
        file << "<td class=\"cell " << style
             << "\" title=\"Superlane 0: " << executed
             << " executed, " << stalled << " stalled, "
             << cycle.lockstep_lanes << " lockstep SIMD lanes\">";
        if (executed != 0) file << executed;
        else if (stalled != 0) file << 'S';
        file << "</td>";
    }
    file << "</tr></table></div>";

    std::size_t superlane_count = 0;
    for (const auto& cycle : trace) {
        superlane_count = std::max(
            superlane_count, cycle.superlanes.size());
    }
    for (std::size_t superlane = 0;
         superlane < superlane_count; ++superlane) {
        const auto lockstep_lanes = trace.empty()
            ? std::size_t {1} : trace.front().lockstep_lanes;
        file << "<details" << (superlane == 0 ? " open" : "")
             << "><summary>Superlane " << superlane
             << " - C0-C15, " << lockstep_lanes
             << " lockstep SIMD lane" << (lockstep_lanes == 1 ? "" : "s")
             << "</summary><div class=\"wrap\"><table>";
        write_header();
        for (std::size_t stage = 0;
             stage < ftlpu::VxmLane::kAluCount; ++stage) {
            file << "<tr><td class=\"label\">C" << stage << "</td>";
            for (const auto& cycle : trace) {
                const auto state = superlane < cycle.superlanes.size()
                    ? cycle.superlanes[superlane][stage]
                    : ftlpu::VxmLaneAluTraceState::Idle;
                const auto code = state_code(state);
                const auto* style =
                    code == 'E' ? "e" : code == 'S' ? "s" : "idle";
                file << "<td class=\"cell " << style
                     << "\" title=\"Superlane " << superlane
                     << " C" << stage
                     << " cycle " << cycle.cycle << ": "
                     << (code == 'E' ? "executed"
                         : code == 'S' ? "stalled" : "idle")
                     << "\">";
                if (code != '.') file << code;
                file << "</td>";
            }
            file << "</tr>";
        }
        file << "</table></div></details>";
    }
    file << "</body></html>";
}

inline void write_timing_reports(
    std::string_view prefix, std::string_view title,
    const std::vector<TimingCycle>& trace)
{
    write_cycle_details(prefix, title, trace);
    write_timing_gantt(prefix, title, trace);
}

} // namespace vxm_hardware_test
