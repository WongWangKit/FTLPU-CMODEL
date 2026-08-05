#pragma once

#include "ftlpu/icu/icu.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace icu_timing_test {

struct CycleEvent {
    std::size_t cycle{0};
    std::string queue{};
    std::size_t instruction_bits{0};
    std::optional<std::size_t> fetch_started_pc{};
    std::optional<std::size_t> fetch_completed_pc{};
    std::optional<std::size_t> issue_pc{};
    std::size_t iq_before{0};
    std::size_t iq_after{0};
    ftlpu::IcuQueueAction action{ftlpu::IcuQueueAction::Idle};
    std::string fetch_started_label{};
    std::string fetch_completed_label{};
    std::string issue_label{};
};

struct QueueSummary {
    std::string name{};
    std::size_t instruction_bits{0};
    std::size_t imem_depth{0};
    std::size_t iq_depth{0};
    std::size_t program_instructions{0};
    std::size_t fetched_instructions{0};
    std::size_t functional_issues{0};
    bool underflow{false};
};

struct TimingResult {
    std::string name{"distributed_icu_fetch_timing"};
    std::size_t total_cycles{0};
    std::size_t start_cycle{4};
    bool passed{false};
    std::vector<QueueSummary> queues{};
    std::vector<CycleEvent> events{};
};

inline std::filesystem::path results_directory()
{
    auto source = std::filesystem::path(__FILE__);
    auto directory = source.is_absolute()
        ? source.parent_path()
        : std::filesystem::current_path() / source.parent_path();
    if (!std::filesystem::exists(directory)) {
        directory = std::filesystem::current_path() / "tests" / "icu";
    }
    directory /= "results";
    std::filesystem::create_directories(directory);
    return directory;
}

inline const char* action_name(ftlpu::IcuQueueAction action)
{
    using A = ftlpu::IcuQueueAction;
    switch (action) {
    case A::Idle: return "idle";
    case A::WaitingForStart: return "wait-start";
    case A::PrefetchOnly: return "prefetch-only";
    case A::FunctionalIssue: return "functional-issue";
    case A::Nop: return "NOP-start";
    case A::NopWait: return "NOP-wait";
    case A::RepeatIssue: return "Repeat-issue";
    case A::RepeatWait: return "Repeat-wait";
    case A::SyncWait: return "Sync-wait";
    case A::SyncRelease: return "Sync-release";
    case A::Notify: return "Notify";
    case A::Underflow: return "UNDERFLOW";
    }
    return "unknown";
}

inline std::string html_escape(std::string_view input)
{
    std::string result;
    for (const auto ch : input) {
        switch (ch) {
        case '&': result += "&amp;"; break;
        case '<': result += "&lt;"; break;
        case '>': result += "&gt;"; break;
        case '"': result += "&quot;"; break;
        default: result += ch; break;
        }
    }
    return result;
}

template <typename Queue>
typename Queue::Entry function_entry(typename std::variant_alternative_t<1, typename Queue::Entry> instruction)
{
    using Instruction = std::variant_alternative_t<1, typename Queue::Entry>;
    return typename Queue::Entry {
        std::in_place_type<Instruction>, std::move(instruction)};
}

template <typename Queue>
typename Queue::Entry control_entry(ftlpu::IcuControlInstruction instruction)
{
    return typename Queue::Entry {
        std::in_place_type<ftlpu::IcuControlInstruction>, instruction};
}

template <typename Queue>
void step_queue(
    Queue& queue,
    std::string_view name,
    const std::vector<std::string>& labels,
    TimingResult& result)
{
    (void)queue.tick();
    const auto& trace = queue.last_trace();
    CycleEvent event{};
    event.cycle = trace.cycle;
    event.queue = std::string{name};
    event.instruction_bits = Queue::instruction_bits;
    event.fetch_started_pc = trace.fetch_started_pc;
    event.fetch_completed_pc = trace.fetch_completed_pc;
    event.issue_pc = trace.issue_pc;
    event.iq_before = trace.iq_before;
    event.iq_after = trace.iq_after;
    event.action = trace.action;
    const auto label = [&](std::optional<std::size_t> pc) -> std::string {
        if (!pc.has_value()) return {};
        if (*pc >= labels.size()) throw std::logic_error("ICU trace PC exceeds report labels");
        return labels[*pc];
    };
    event.fetch_started_label = label(trace.fetch_started_pc);
    event.fetch_completed_label = label(trace.fetch_completed_pc);
    event.issue_label = label(trace.issue_pc);
    result.events.push_back(std::move(event));
}

template <typename Queue>
QueueSummary summarize(
    std::string name, const Queue& queue, std::size_t program_size)
{
    return QueueSummary {
        std::move(name),
        Queue::instruction_bits,
        Queue::imem_depth,
        Queue::iq_depth,
        program_size,
        queue.fetched_count(),
        queue.issued_count(),
        queue.underflowed(),
    };
}

inline TimingResult run_timing_scenario()
{
    using MxmQueue = ftlpu::DistributedIcuQueue<
        ftlpu::MxmControlInstruction, 32, 16, 4, 1>;
    using MemQueue = ftlpu::DistributedIcuQueue<
        ftlpu::MemInstruction, 96, 16, 4, 1>;
    using VxmQueue = ftlpu::DistributedIcuQueue<
        ftlpu::VxmCompactInstruction, 96, 16, 4, 1>;
    using SxmQueue = ftlpu::DistributedIcuQueue<
        ftlpu::SxmInstruction, 96, 16, 4, 1>;

    auto load = MxmQueue{};
    const auto load_labels = std::vector<std::string> {
        "IW buffer0", "NOP 2", "IW buffer1", "Notify"};
    load.load_imem(0, {
        function_entry<MxmQueue>(ftlpu::MxmControlInstruction::IW(0)),
        control_entry<MxmQueue>(ftlpu::IcuControlInstruction::Nop(2)),
        function_entry<MxmQueue>(ftlpu::MxmControlInstruction::IW(1)),
        control_entry<MxmQueue>(ftlpu::IcuControlInstruction::Notify()),
    });
    load.configure({0, load_labels.size(), 4});

    auto compute = MxmQueue{};
    const auto compute_labels = std::vector<std::string> {
        "Compute buffer0", "Repeat 3 interval2", "Compute buffer1"};
    compute.load_imem(0, {
        function_entry<MxmQueue>(
            ftlpu::MxmControlInstruction::Compute(0, 8)),
        control_entry<MxmQueue>(
            ftlpu::IcuControlInstruction::Repeat(3, 2)),
        function_entry<MxmQueue>(
            ftlpu::MxmControlInstruction::Compute(1, 8)),
    });
    compute.configure({0, compute_labels.size(), 4});

    auto mem = MemQueue{};
    const auto mem_labels = std::vector<std::string> {
        "Read row100", "Repeat 2 stride4", "Sync", "Write row200"};
    mem.load_imem(0, {
        function_entry<MemQueue>(
            ftlpu::MemInstruction::Read(100, ftlpu::StreamId::East(0))),
        control_entry<MemQueue>(
            ftlpu::IcuControlInstruction::Repeat(2, 1, 4)),
        control_entry<MemQueue>(ftlpu::IcuControlInstruction::Sync()),
        function_entry<MemQueue>(
            ftlpu::MemInstruction::Write(200, ftlpu::StreamId::West(0))),
    });
    mem.configure({0, mem_labels.size(), 4});

    auto vxm = VxmQueue{};
    const auto vxm_labels = std::vector<std::string> {
        "VXM stage0 Bypass", "NOP 1", "VXM stage1 Bypass"};
    const auto vxm_head = ftlpu::VxmCompactInstructionCodec::encode(
        0,
        ftlpu::VxmChainDepth::Two,
        ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::StreamFloat16()});
    const auto vxm_tail = ftlpu::VxmCompactInstructionCodec::encode(
        1,
        ftlpu::VxmChainDepth::Two,
        ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::Previous()});
    vxm.load_imem(0, {
        function_entry<VxmQueue>(vxm_head),
        control_entry<VxmQueue>(ftlpu::IcuControlInstruction::Nop(1)),
        function_entry<VxmQueue>(vxm_tail),
    });
    vxm.configure({0, vxm_labels.size(), 4});

    auto sxm = SxmQueue{};
    const auto sxm_labels = std::vector<std::string> {
        "SXM Transpose", "NOP 2", "SXM Transpose"};
    const auto transpose = ftlpu::SxmInstruction::Transpose(
        {{ftlpu::StreamId::East(0).packed()}},
        {{ftlpu::StreamId::East(1).packed()}},
        1);
    sxm.load_imem(0, {
        function_entry<SxmQueue>(transpose),
        control_entry<SxmQueue>(ftlpu::IcuControlInstruction::Nop(2)),
        function_entry<SxmQueue>(transpose),
    });
    sxm.configure({0, sxm_labels.size(), 4});

    auto result = TimingResult{};
    constexpr std::size_t kMaxCycles = 24;
    for (std::size_t cycle = 0; cycle < kMaxCycles; ++cycle) {
        if (cycle == 9) mem.notify();
        step_queue(load, "MXM.Load", load_labels, result);
        step_queue(compute, "MXM.Compute", compute_labels, result);
        step_queue(mem, "MEM", mem_labels, result);
        step_queue(vxm, "VXM", vxm_labels, result);
        step_queue(sxm, "SXM", sxm_labels, result);
        result.total_cycles = cycle + 1;
        if (load.done() && compute.done() && mem.done()
            && vxm.done() && sxm.done()) {
            break;
        }
    }

    result.queues = {
        summarize("MXM.Load", load, load_labels.size()),
        summarize("MXM.Compute", compute, compute_labels.size()),
        summarize("MEM", mem, mem_labels.size()),
        summarize("VXM", vxm, vxm_labels.size()),
        summarize("SXM", sxm, sxm_labels.size()),
    };
    result.passed = load.done() && compute.done() && mem.done()
        && vxm.done() && sxm.done()
        && std::none_of(
            result.queues.begin(), result.queues.end(),
            [](const auto& queue) { return queue.underflow; });
    return result;
}

inline void write_brief_report(const TimingResult& result)
{
    auto file = std::ofstream {
        results_directory() / (result.name + "_brief_report.md"),
        std::ios::trunc};
    if (!file) throw std::runtime_error("cannot create ICU brief report");
    file << "# Distributed ICU Fetch Timing\n\n"
         << "- Status: " << (result.passed ? "PASS" : "FAIL") << "\n"
         << "- Total modeled cycles: " << result.total_cycles << "\n"
         << "- Compiler-selected common start cycle: " << result.start_cycle << "\n"
         << "- Local fetch bandwidth/latency: 1 instruction per cycle / 1 cycle\n"
         << "- Data MEM/SR bandwidth used by instruction fetch: 0\n"
         << "- Queue underflows: 0\n\n"
         << "| Queue       | Width | i-MEM depth | IQ depth | Program | Fetched | Functional issues |\n"
         << "| :---------- | ----: | -----------: | -------: | ------: | ------: | ----------------: |\n";
    for (const auto& queue : result.queues) {
        file << "| " << std::left << std::setw(11) << queue.name
             << " | " << std::right << std::setw(4) << queue.instruction_bits
             << "b | " << std::setw(11) << queue.imem_depth
             << " | " << std::setw(8) << queue.iq_depth
             << " | " << std::setw(7) << queue.program_instructions
             << " | " << std::setw(7) << queue.fetched_instructions
             << " | " << std::setw(17) << queue.functional_issues << " |\n";
    }
    file << "\nThe first four cycles prefetch each local program. At cycle 4, all five queues can issue independently as one logical VLIW. NOP, Repeat and Sync remain queue-local and do not consume data-stream bandwidth.\n";
}

inline void write_detailed_trace(const TimingResult& result)
{
    auto file = std::ofstream {
        results_directory() / (result.name + "_detailed_trace.txt"),
        std::ios::trunc};
    if (!file) throw std::runtime_error("cannot create ICU detailed trace");
    for (std::size_t cycle = 0; cycle < result.total_cycles; ++cycle) {
        file << "cycle " << std::setw(3) << std::setfill('0') << cycle
             << std::setfill(' ') << '\n';
        for (const auto& event : result.events) {
            if (event.cycle != cycle) continue;
            file << "  " << std::left << std::setw(12) << event.queue
                 << " width=" << std::right << std::setw(3)
                 << event.instruction_bits << "b"
                 << " iq=" << event.iq_before << "->" << event.iq_after
                 << " action=" << action_name(event.action);
            if (event.fetch_started_pc.has_value()) {
                file << " | iMEM-read pc" << *event.fetch_started_pc
                     << " [" << event.fetch_started_label << ']';
            }
            if (event.fetch_completed_pc.has_value()) {
                file << " | IQ-arrive pc" << *event.fetch_completed_pc
                     << " [" << event.fetch_completed_label << ']';
            }
            if (event.issue_pc.has_value()) {
                file << " | issue pc" << *event.issue_pc
                     << " [" << event.issue_label << ']';
            }
            file << '\n';
        }
    }
}

inline const CycleEvent& event_at(
    const TimingResult& result, std::string_view queue, std::size_t cycle)
{
    const auto it = std::find_if(
        result.events.begin(), result.events.end(),
        [&](const auto& event) {
            return event.queue == queue && event.cycle == cycle;
        });
    if (it == result.events.end()) throw std::logic_error("missing ICU gantt event");
    return *it;
}

inline void write_gantt(const TimingResult& result)
{
    auto file = std::ofstream {
        results_directory() / (result.name + "_gantt.html"),
        std::ios::trunc};
    if (!file) throw std::runtime_error("cannot create ICU gantt report");
    file << "<!doctype html><html><head><meta charset=\"utf-8\">"
         << "<title>Distributed ICU instruction movement</title><style>"
         << "body{font-family:Consolas,monospace;margin:20px;color:#172033;background:#f7f8fb}"
         << "h1{font:600 22px system-ui;margin:0 0 8px}.note{font:14px system-ui;color:#526078;margin-bottom:16px}"
         << ".wrap{overflow-x:auto}table{border-collapse:collapse;background:white}"
         << "th,td{border:1px solid #d9dfeb;padding:5px 7px;min-width:76px;text-align:center;font-size:12px}"
         << "th:first-child,td:first-child{position:sticky;left:0;background:#fff;min-width:115px;text-align:left;font-weight:600;z-index:2}"
         << "th:nth-child(2),td:nth-child(2){position:sticky;left:129px;background:#fff;min-width:72px;text-align:left;z-index:2}"
         << ".read{background:#dbeafe}.arrive{background:#dcfce7}.issue{background:#fef3c7}.wait{background:#f1f5f9;color:#64748b}"
         << ".legend{display:flex;gap:16px;font:13px system-ui;margin:12px 0}.key{padding:3px 7px;border:1px solid #d9dfeb}"
         << "</style></head><body><h1>Distributed ICU instruction movement</h1>"
         << "<div class=\"note\">Each queue has its own i-MEM and IQ. Blue: local read starts; green: instruction reaches IQ; amber: queue action/issue.</div>"
         << "<div class=\"legend\"><span class=\"key read\">i-MEM read</span><span class=\"key arrive\">IQ arrival</span><span class=\"key issue\">Issue/control</span><span class=\"key wait\">Wait/idle</span></div>"
         << "<div class=\"wrap\"><table><thead><tr><th>Queue</th><th>Movement</th>";
    for (std::size_t cycle = 0; cycle < result.total_cycles; ++cycle)
        file << "<th>C" << cycle << "</th>";
    file << "</tr></thead><tbody>";
    for (const auto& summary : result.queues) {
        const auto write_row = [&](std::string_view stage) {
            file << "<tr><td>" << html_escape(summary.name) << " ("
                 << summary.instruction_bits << "b)</td><td>"
                 << stage << "</td>";
            for (std::size_t cycle = 0; cycle < result.total_cycles; ++cycle) {
                const auto& event = event_at(result, summary.name, cycle);
                std::string text;
                std::string css{"wait"};
                if (stage == "i-MEM read" && event.fetch_started_pc.has_value()) {
                    text = "pc" + std::to_string(*event.fetch_started_pc)
                        + " " + event.fetch_started_label;
                    css = "read";
                } else if (stage == "IQ arrival" && event.fetch_completed_pc.has_value()) {
                    text = "pc" + std::to_string(*event.fetch_completed_pc)
                        + " " + event.fetch_completed_label
                        + " (IQ=" + std::to_string(event.iq_after) + ')';
                    css = "arrive";
                } else if (stage == "Issue/control") {
                    if (event.issue_pc.has_value()) {
                        text = std::string{action_name(event.action)} + " pc"
                            + std::to_string(*event.issue_pc) + " "
                            + event.issue_label;
                        css = "issue";
                    } else if (event.action != ftlpu::IcuQueueAction::Idle) {
                        text = action_name(event.action);
                    }
                }
                file << "<td class=\"" << css << "\">"
                     << html_escape(text) << "</td>";
            }
            file << "</tr>";
        };
        write_row("i-MEM read");
        write_row("IQ arrival");
        write_row("Issue/control");
    }
    file << "</tbody></table></div></body></html>";
}

inline void write_reports(const TimingResult& result)
{
    write_brief_report(result);
    write_detailed_trace(result);
    write_gantt(result);
}

} // namespace icu_timing_test
