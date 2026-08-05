#pragma once

#include "ftlpu/vxm/lane.hpp"
#include "hardware_test_output.hpp"

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
    std::size_t cycle{0};
    ftlpu::VxmChainDepth depth{ftlpu::VxmChainDepth::Two};
    std::array<ftlpu::VxmLaneAluTraceState,
               ftlpu::VxmLane::kAluCount> states{};
    std::size_t pending_feedback{0};
    std::size_t outputs{0};
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
    result.event = std::move(event);
    for (std::size_t stage = 0;
         stage < ftlpu::VxmLane::kAluCount; ++stage) {
        result.states[stage] = lane.last_trace()[stage].state;
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
             << " alu=";
        for (const auto state : cycle.states) {
            file << state_code(state);
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
    file << R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>VXM hardware timing Gantt</title>
<style>
body{font:14px system-ui;margin:24px;background:#f4f7fb;color:#172033}
h1{font-size:21px;margin-bottom:6px}p{color:#657188}
.wrap{overflow:auto;border-radius:8px;box-shadow:0 2px 12px #ccd5e2}
table{border-collapse:collapse;background:white}
th,td{border:1px solid #d7deea;padding:5px 8px;text-align:center;min-width:34px}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:white}
.e{background:#31b889;color:white}.s{background:#e36a6a;color:white}
.i{background:#edf1f6}.config{background:#ef9f32;color:white}
.data{background:#4185d7;color:white}.transition{background:#8b67d4;color:white}
.output{background:#9b62d1;color:white}
</style></head><body><h1>)HTML"
         << title
         << R"HTML(</h1>
<p>Complete recorded test sequence. E=executed, S=stalled, .=idle.</p>
<div class="wrap"><table><tr><th>Item</th>)HTML";
    for (const auto& cycle : trace) {
        file << "<th>C" << cycle.cycle << "</th>";
    }
    file << "</tr><tr><td>Event</td>";
    for (const auto& cycle : trace) {
        auto style = std::string_view{"i"};
        if (cycle.outputs != 0) {
            style = "output";
        } else if (cycle.event.find("config") != std::string::npos
            || cycle.event.find("decode") != std::string::npos) {
            style = "config";
        } else if (cycle.event.find("input") != std::string::npos
                   || cycle.event.find("head") != std::string::npos) {
            style = "data";
        } else if (cycle.event.find("transition") != std::string::npos
                   || cycle.event.find("feedback") != std::string::npos) {
            style = "transition";
        }
        file << "<td class=\"" << style << "\">"
             << cycle.event << "</td>";
    }
    file << "</tr><tr><td>Chain depth</td>";
    for (const auto& cycle : trace) {
        file << "<td>" << static_cast<std::size_t>(cycle.depth)
             << "</td>";
    }
    file << "</tr><tr><td>Pending feedback</td>";
    for (const auto& cycle : trace) {
        file << "<td>" << cycle.pending_feedback << "</td>";
    }
    file << "</tr><tr><td>Outputs</td>";
    for (const auto& cycle : trace) {
        file << "<td>" << cycle.outputs << "</td>";
    }
    file << "</tr>";
    for (std::size_t stage = 0;
         stage < ftlpu::VxmLane::kAluCount; ++stage) {
        file << "<tr><td>C" << stage << "</td>";
        for (const auto& cycle : trace) {
            const auto code = state_code(cycle.states[stage]);
            const auto* style =
                code == 'E' ? "e" : code == 'S' ? "s" : "i";
            file << "<td class=\"" << style << "\">"
                 << code << "</td>";
        }
        file << "</tr>";
    }
    file << "</table></div></body></html>";
}

inline void write_timing_reports(
    std::string_view prefix, std::string_view title,
    const std::vector<TimingCycle>& trace)
{
    write_cycle_details(prefix, title, trace);
    write_timing_gantt(prefix, title, trace);
}

} // namespace vxm_hardware_test
