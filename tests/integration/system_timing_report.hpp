#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace integration_timing {

struct Event {
    std::string resource;
    std::string label;
    std::string kind;
    std::size_t first_cycle{0};
    std::size_t end_cycle{0};
    double utilization{1.0};
};

class Report {
public:
    enum class OutputFormat { Csv, Html, Both };

    void ensure_resource(std::string resource)
    {
        if (std::find(resources_.begin(), resources_.end(), resource)
            == resources_.end()) {
            resources_.push_back(std::move(resource));
        }
    }

    void add(
        std::string resource, std::size_t first_cycle,
        std::size_t end_cycle, std::string label,
        std::string kind, double utilization = 1.0)
    {
        if (end_cycle <= first_cycle) {
            throw std::invalid_argument(
                "system timing event must occupy at least one cycle");
        }
        ensure_resource(resource);
        events_.push_back(Event {
            std::move(resource), std::move(label), std::move(kind),
            first_cycle, end_cycle,
            std::clamp(utilization, 0.0, 1.0)});
    }

    void add_or_extend(
        std::string resource, std::size_t first_cycle,
        std::size_t end_cycle, std::string label,
        std::string kind, double utilization = 1.0)
    {
        // A trace can contain millions of cycles but only a small, fixed set
        // of hardware resources.  Keep the last event index per resource so
        // extending a run is O(1), rather than reverse-scanning the complete
        // event vector on every captured cycle.
        if (const auto found = last_event_.find(resource);
            found != last_event_.end()) {
            auto& existing = events_[found->second];
            if (existing.label == label
                && existing.kind == kind
                && existing.utilization == utilization
                && existing.end_cycle == first_cycle) {
                existing.end_cycle = end_cycle;
                return;
            }
        }
        add(
            std::move(resource), first_cycle, end_cycle,
            std::move(label), std::move(kind), utilization);
        last_event_[events_.back().resource] = events_.size() - 1;
    }

    void write(
        std::string_view prefix, std::string_view title,
        OutputFormat format = OutputFormat::Both) const
    {
        if (events_.empty()) return;
        const auto directory = results_directory();
        if (format != OutputFormat::Html) {
            write_csv(directory / (std::string {prefix} + "_schedule.csv"));
        }
        if (format != OutputFormat::Csv) {
            write_html(
                directory / (std::string {prefix} + "_timing_gantt.html"),
                title);
        }
    }

private:
    static std::filesystem::path results_directory()
    {
        auto cursor = std::filesystem::current_path();
        while (true) {
            const auto integration = cursor / "tests" / "integration";
            if (std::filesystem::exists(
                    integration / "system_timing_report.hpp")) {
                auto results = integration / "results";
                std::filesystem::create_directories(results);
                return results;
            }
            const auto parent = cursor.parent_path();
            if (parent == cursor || parent.empty()) break;
            cursor = parent;
        }

        const auto source = std::filesystem::path(__FILE__);
        const auto integration = source.is_absolute()
            ? source.parent_path()
            : std::filesystem::current_path() / source.parent_path();
        auto results = integration / "results";
        std::filesystem::create_directories(results);
        return results;
    }

    static std::string csv_field(std::string_view value)
    {
        auto result = std::string {"\""};
        for (const auto ch : value) {
            result += ch;
            if (ch == '"') result += '"';
        }
        result += '"';
        return result;
    }

    static std::string html_field(std::string_view value)
    {
        auto result = std::string {};
        result.reserve(value.size());
        for (const auto ch : value) {
            switch (ch) {
            case '&': result += "&amp;"; break;
            case '<': result += "&lt;"; break;
            case '>': result += "&gt;"; break;
            case '"': result += "&quot;"; break;
            case '\'': result += "&#39;"; break;
            default: result += ch; break;
            }
        }
        return result;
    }

    static std::size_t tick_step(std::size_t cycle_count)
    {
        const auto target = std::max<std::size_t>(1, cycle_count / 6);
        constexpr std::size_t steps[] {
            1, 2, 5, 10, 20, 25, 50, 100, 150, 200, 250, 500, 1000};
        for (const auto step : steps) {
            if (step >= target) return step;
        }
        return ((target + 999) / 1000) * 1000;
    }

    static std::string event_class(const Event& event)
    {
        if (event.resource == "Protocol phase") return "phase";
        if (event.resource == "ICU dispatch") return "icu";
        if (event.resource.rfind("VXM Superlane", 0) == 0) return "vxm";
        if (event.resource.rfind("MXM ", 0) == 0) return "mxm";
        if (event.resource.rfind("SXM ", 0) == 0) return "sxm";
        if (event.resource == "Local scalar register") return "scalar";
        if (event.resource == "MEM dual-port") return "overlap";
        if (event.resource.find("MEM") != std::string::npos) {
            return event.resource.find("Read") != std::string::npos
                ? "read" : "mem";
        }
        if (event.resource.find("Stream Register") != std::string::npos) {
            return "stream";
        }
        return event.kind;
    }

    static double event_opacity(const Event& event)
    {
        if (event.utilization < 1.0) {
            return std::clamp(0.18 + 0.82 * event.utilization, 0.18, 1.0);
        }
        if (event.resource.rfind("VXM Superlane", 0) != 0) return 0.82;
        const auto marker = event.label.find("active ALUs ");
        if (marker == std::string::npos) return 0.35;
        const auto first = marker + std::string_view {"active ALUs "}.size();
        std::size_t active = 0;
        for (auto index = first;
             index < event.label.size() && event.label[index] >= '0'
             && event.label[index] <= '9'; ++index) {
            active = active * 10 + static_cast<std::size_t>(event.label[index] - '0');
        }
        return std::clamp(0.18 + 0.82 * static_cast<double>(active) / 16.0,
                          0.18, 1.0);
    }

    void write_csv(const std::filesystem::path& path) const
    {
        auto file = std::ofstream {path, std::ios::trunc};
        if (!file) {
            throw std::runtime_error(
                "cannot create system timing CSV report");
        }
        file << "resource,start_cycle,end_cycle,kind,utilization,detail\n";
        for (const auto& event : events_) {
            file << csv_field(event.resource) << ','
                 << event.first_cycle << ',' << event.end_cycle << ','
                 << csv_field(event.kind) << ',' << event.utilization << ','
                 << csv_field(event.label) << '\n';
        }
    }

    void write_html(
        const std::filesystem::path& path,
        std::string_view title) const
    {
        auto file = std::ofstream {path, std::ios::trunc};
        if (!file) {
            throw std::runtime_error(
                "cannot create system timing Gantt report");
        }
        const auto first_cycle = std::min_element(
            events_.begin(), events_.end(),
            [](const Event& lhs, const Event& rhs) {
                return lhs.first_cycle < rhs.first_cycle;
            })->first_cycle;
        const auto end_cycle = std::max_element(
            events_.begin(), events_.end(),
            [](const Event& lhs, const Event& rhs) {
                return lhs.end_cycle < rhs.end_cycle;
            })->end_cycle;

        const auto cycle_count = end_cycle - first_cycle;
        const auto cycle_width = std::clamp(
            900.0 / static_cast<double>(cycle_count), 1.0, 4.0);
        constexpr double label_width = 190.0;
        constexpr double row_height = 25.0;
        constexpr double top_height = 36.0;
        const auto plot_width = cycle_width * static_cast<double>(cycle_count);
        const auto svg_width = label_width + plot_width + 18.0;
        const auto svg_height = top_height
            + row_height * static_cast<double>(resources_.size()) + 10.0;
        const auto step = tick_step(cycle_count);

        file << R"HTML(<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>System timing Gantt</title><style>
body{font:13px system-ui;margin:18px;background:#f4f7fb;color:#172033}
h1{font-size:20px;margin:0 0 4px}p{margin:4px 0 10px;color:#657188}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 12px}
.key:before{content:"";display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;background:var(--c)}
.wrap{overflow:auto;border-radius:7px;box-shadow:0 2px 10px #ccd5e2;background:white;padding:8px}
svg{display:block;background:#fff}.row-label{font-size:13px;font-weight:650;fill:#172033}
.tick-label{font-size:12px;font-weight:650;fill:#23314d}.grid{stroke:#dce3ee;stroke-width:1}
.phase{fill:#8067c8}.icu{fill:#9b82dc}.vxm{fill:#e45678}.mem{fill:#38a56a}
.mxm{fill:#378bd2}.sxm{fill:#159da5}
.stream{fill:#4b91d1}.scalar{fill:#9a69d4}.overlap{fill:#2c9ca1}
.config{fill:#e8a02b}.read{fill:#4389cf}.compute{fill:#3aab78}.write{fill:#9564ce}
rect.event{shape-rendering:crispEdges}rect.event:hover{stroke:#172033;stroke-width:1.5;opacity:1!important}
</style></head><body><h1>)HTML"
             << html_field(title)
             << R"HTML(</h1><p>System-level black-box schedule. Resources are rows; horizontal distance is hardware cycles. Hover a block for exact bounds.</p>
<div class="legend">
 <span class="key" style="--c:#8067c8">Protocol</span>
 <span class="key" style="--c:#9b82dc">ICU/config</span>
 <span class="key" style="--c:#4b91d1">MEM/Stream read</span>
 <span class="key" style="--c:#378bd2">MXM</span>
 <span class="key" style="--c:#159da5">SXM</span>
 <span class="key" style="--c:#e45678">VXM Superlane</span>
 <span class="key" style="--c:#9a69d4">Local scalar</span>
 <span class="key" style="--c:#38a56a">MEM write</span>
</div><div class="wrap"><svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="System timing Gantt" viewBox="0 0 )HTML"
             << svg_width << ' ' << svg_height << "\" width=\""
             << svg_width << "\" height=\"" << svg_height << "\">";

        const auto draw_tick = [&](std::size_t cycle) {
            const auto x = label_width
                + static_cast<double>(cycle - first_cycle) * cycle_width;
            file << "<line class=\"grid\" x1=\"" << x << "\" y1=\"25\" x2=\""
                 << x << "\" y2=\"" << (svg_height - 6.0) << "\"/>"
                 << "<text class=\"tick-label\" x=\"" << (x + 3.0)
                 << "\" y=\"18\">C" << cycle << "</text>";
        };
        draw_tick(first_cycle);
        const auto first_aligned = ((first_cycle / step) + 1) * step;
        for (auto cycle = first_aligned;
             cycle <= end_cycle; cycle += step) {
            draw_tick(cycle);
        }

        for (std::size_t row = 0; row < resources_.size(); ++row) {
            const auto y = top_height + static_cast<double>(row) * row_height;
            const auto& resource = resources_[row];
            file << "<line class=\"grid\" x1=\"0\" y1=\"" << y
                 << "\" x2=\"" << (label_width + plot_width)
                 << "\" y2=\"" << y << "\"/>"
                 << "<text class=\"row-label\" x=\"6\" y=\"" << (y + 17.0)
                 << "\">" << html_field(resource) << "</text>";
            for (const auto& event : events_) {
                if (event.resource != resource) continue;
                const auto start = std::max(event.first_cycle, first_cycle);
                const auto finish = std::min(event.end_cycle, end_cycle);
                if (finish <= start) continue;
                const auto x = label_width
                    + static_cast<double>(start - first_cycle) * cycle_width;
                const auto width = std::max(
                    1.0, static_cast<double>(finish - start) * cycle_width);
                file << "<rect class=\"event " << event_class(event)
                     << "\" x=\"" << x << "\" y=\"" << (y + 3.0)
                     << "\" width=\"" << width << "\" height=\"19\" opacity=\""
                     << event_opacity(event) << "\"><title>C"
                     << event.first_cycle << "-C" << (event.end_cycle - 1)
                     << ": " << html_field(event.label) << "</title></rect>";
            }
        }
        const auto bottom = top_height
            + static_cast<double>(resources_.size()) * row_height;
        file << "<line class=\"grid\" x1=\"0\" y1=\"" << bottom
             << "\" x2=\"" << (label_width + plot_width)
             << "\" y2=\"" << bottom << "\"/></svg></div></body></html>";
    }

    std::vector<Event> events_{};
    std::vector<std::string> resources_{};
    std::unordered_map<std::string, std::size_t> last_event_{};
};

} // namespace integration_timing
