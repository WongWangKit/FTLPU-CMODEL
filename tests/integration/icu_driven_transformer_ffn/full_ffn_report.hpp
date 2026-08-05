#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace full_ffn_test {

struct CycleRecord {
    std::size_t cycle{0};
    std::string phase{};
    std::string events{};
    std::size_t gate_mxm_cells{0};
    std::size_t up_mxm_cells{0};
    std::size_t vxm_alu_slots{0};
    std::size_t quantized_outputs{0};
    std::size_t mem_reads{0};
    std::size_t mem_writes{0};
    std::size_t sxm_transpose_tiles{0};
    std::size_t sxm_permute_tiles{0};
    std::size_t gate_up_weight_load_tiles{0};
    std::size_t down_weight_load_tiles{0};
    std::size_t down_mxm_cells{0};
    std::size_t down_acc_outputs{0};
    std::size_t icu_imem_reads{0};
    std::size_t icu_iq_arrivals{0};
    std::size_t icu_dispatches{0};
    std::size_t icu_waiting_queues{0};
};

struct Utilization {
    double active{0.0};
    double time{0.0};
    double whole{0.0};
};

struct FullFfnResult {
    std::string name{};
    std::string description{};
    std::size_t tokens{0};
    std::size_t hidden{0};
    std::size_t intermediate{0};
    std::size_t output_features{0};
    std::size_t chain_depth{0};
    std::size_t chains_per_lane{0};
    std::size_t weight_reuse_per_buffer{0};
    std::size_t configured_quantizer_channels{0};
    std::size_t quantizer_channels{0};
    std::size_t vxm_peak_alu_slots{0};
    std::size_t quantizer_peak_outputs{0};
    std::size_t cycles{0};
    std::size_t swiglu_values_checked{0};
    std::size_t layout_values_checked{0};
    std::size_t down_values_checked{0};
    std::size_t activation_mem_writes{0};
    std::size_t activation_mem_reads{0};
    std::size_t sxm_transpose_instructions{0};
    std::size_t sxm_permute_instructions{0};
    std::size_t down_pair_merges{0};
    std::size_t down_weight_reuses{0};
    std::size_t front_compute_issues{0};
    std::size_t front_compute_bubbles{0};
    std::size_t down_compute_issues{0};
    std::size_t down_compute_bubbles{0};
    std::size_t down_input_wait_cycles{0};
    std::size_t gate_up_compute_load_overlap_cycles{0};
    std::size_t down_compute_load_overlap_cycles{0};
    std::size_t mxm_vxm_overlap_cycles{0};
    std::size_t sxm_down_mxm_overlap_cycles{0};
    double vxm_overlap_percentage{0.0};
    double down_mxm_overlap_percentage{0.0};
    std::string down_accumulator_format{"INT32"};
    bool correctness_passed{false};
    bool functionality_passed{false};
    bool icu_driven{false};
    std::size_t icu_prefetch_cycles{0};
    std::size_t icu_active_queues{0};
    std::size_t icu_functional_events{0};
    std::size_t icu_programmed_instructions{0};
    std::size_t icu_fetched_instructions{0};
    std::size_t icu_functional_issues{0};
    std::size_t icu_underflowed_queues{0};
    Utilization total_mxm{};
    Utilization gate_up_mxm{};
    Utilization vxm{};
    Utilization vxm_total{};
    Utilization quantizer{};
    Utilization quantizer_physical{};
    Utilization mem{};
    Utilization sxm{};
    Utilization down_mxm{};
    std::vector<CycleRecord> timeline{};
};

inline std::filesystem::path results_directory()
{
    auto source = std::filesystem::path(__FILE__);
    auto directory = source.is_absolute()
        ? source.parent_path()
        : std::filesystem::current_path() / source.parent_path();
    if (!std::filesystem::exists(directory)) {
        directory = std::filesystem::current_path() / "tests" / "integration"
            / "icu_driven_transformer_ffn";
    }
    directory /= "results";
    std::filesystem::create_directories(directory);
    return directory;
}

inline std::string html_escape(std::string_view input)
{
    auto output = std::string{};
    for (const auto value : input) {
        switch (value) {
        case '&': output += "&amp;"; break;
        case '<': output += "&lt;"; break;
        case '>': output += "&gt;"; break;
        case '"': output += "&quot;"; break;
        default: output += value; break;
        }
    }
    return output;
}

inline void write_utilization_row(
    std::ofstream& file, std::string_view resource,
    const Utilization& utilization)
{
    const auto percentage = [](double value) {
        auto text = std::ostringstream{};
        text << std::fixed << std::setprecision(2)
             << value * 100.0 << '%';
        return text.str();
    };
    file << "| " << std::left << std::setw(20) << resource
         << " | " << std::right << std::setw(7)
         << percentage(utilization.active)
         << " | " << std::setw(7) << percentage(utilization.time)
         << " | " << std::setw(7) << percentage(utilization.whole)
         << " |\n";
}

inline void write_brief_report(
    const FullFfnResult& result,
    std::filesystem::path directory = results_directory())
{
    std::filesystem::create_directories(directory);
    auto file = std::ofstream{
        directory / (result.name + "_brief_report.md"),
        std::ios::trunc};
    if (!file) throw std::runtime_error("cannot create full-FFN brief report");
    file << "# Pipelined Edge Transformer FFN Datapath\n\n"
         << result.description << "\n\n"
         << "- Status: "
         << (result.correctness_passed && result.functionality_passed
                 ? "PASS" : "FAIL") << "\n"
         << "- Shape: tokens=`" << result.tokens << "`, hidden=`"
         << result.hidden << "`, SwiGLU intermediate=`"
         << result.intermediate << "`, output=`"
         << result.output_features << "`\n"
         << "- VXM chain depth/chains per lane: "
         << result.chain_depth << "/" << result.chains_per_lane << "\n"
         << "- Weight-buffer reuse rounds before block switch: "
         << result.weight_reuse_per_buffer << "\n"
         << "- Flow: `Gate/Up MXM -> ACC/cast -> SR -> lane-banked VXM Input "
            "Buffer -> VXM SwiGLU -> FP16-to-INT8 -> SR/MEM "
            "-> SXM byte transpose/permute -> Down MXM -> FP16 cast`\n"
         << "- Total cycles: " << result.cycles << "\n"
         << "- Checked values (SwiGLU/layout/Down): "
         << result.swiglu_values_checked << "/"
         << result.layout_values_checked << "/"
         << result.down_values_checked << "\n"
         << "- Activation MEM vector writes/reads: "
         << result.activation_mem_writes << "/"
         << result.activation_mem_reads << "\n"
         << "- SXM Transpose/Permute instructions: "
         << result.sxm_transpose_instructions << "/"
         << result.sxm_permute_instructions << "\n"
         << "- Down " << result.down_accumulator_format
         << " element merges: " << result.down_pair_merges << "\n"
         << "- Down weight-buffer compute uses: "
         << result.down_weight_reuses << "\n"
         << "- Front MXM Compute issues/bubbles: "
         << result.front_compute_issues << "/"
         << result.front_compute_bubbles << "\n"
         << "- Down MXM Compute issues/bubbles: "
         << result.down_compute_issues << "/"
         << result.down_compute_bubbles << "\n"
         << "- Down inter-burst input-wait cycles: "
         << result.down_input_wait_cycles << "\n"
         << "- VXM peak ALU slots (active/all configured Superlanes): "
         << result.vxm_peak_alu_slots << "/"
         << ftlpu::hw::kTileRows * ftlpu::hw::kLanesPerTile * 16 << "\n"
         << "- VXM quantized outputs/cycle (peak/configured/physical): "
         << result.quantizer_peak_outputs << "/"
         << result.configured_quantizer_channels << "/"
         << result.quantizer_channels << "\n"
         << "- MXM compute/background weight-load overlap cycles: "
         << result.gate_up_compute_load_overlap_cycles << "\n"
         << "- Gate/Up MXM with VXM overlap: "
         << result.mxm_vxm_overlap_cycles << " cycles ("
         << std::fixed << std::setprecision(2)
         << result.vxm_overlap_percentage * 100.0
         << "% of VXM-active cycles)\n"
         << "- SXM with Down MXM overlap: "
         << result.sxm_down_mxm_overlap_cycles << " cycles ("
         << result.down_mxm_overlap_percentage * 100.0
         << "% of Down-MXM-active cycles)\n\n"
         << (result.icu_driven
             ? "- ICU control: distributed local i-MEM replay\n"
               "- ICU prefetch cycles before system cycle 0: "
                   + std::to_string(result.icu_prefetch_cycles) + "\n"
               "- ICU active queues: "
                   + std::to_string(result.icu_active_queues) + "\n"
               "- ICU functional events/program words/fetched words/issues: "
                   + std::to_string(result.icu_functional_events) + "/"
                   + std::to_string(result.icu_programmed_instructions) + "/"
                   + std::to_string(result.icu_fetched_instructions) + "/"
                   + std::to_string(result.icu_functional_issues) + "\n"
               "- ICU underflowed queues: "
                   + std::to_string(result.icu_underflowed_queues) + "\n\n"
             : std::string{})
         << "| Resource             |  Active |    Time |   Whole |\n"
         << "| :------------------- | ------: | ------: | ------: |\n";
    write_utilization_row(file, "**MXM total**", result.total_mxm);
    write_utilization_row(file, "Front MXMs (2)", result.gate_up_mxm);
    write_utilization_row(file, "Down MXMs (2)", result.down_mxm);
    write_utilization_row(
        file,
        "VXM (all " + std::to_string(ftlpu::hw::kTileRows) + " SL)",
        result.vxm);
    write_utilization_row(file, "VXM configured total", result.vxm_total);
    write_utilization_row(file, "VXM q configured", result.quantizer);
    write_utilization_row(file, "VXM q physical", result.quantizer_physical);
    write_utilization_row(file, "MEM ports", result.mem);
    write_utilization_row(file, "SXM INT8 plane", result.sxm);
    file << "\n`Utilization over whole test = Utilization while active x "
            "Time utilization`. Time utilization counts cycles with at least "
            "one operation on that resource.\n\n"
            "`SXM INT8 single-byte plane` means each INT8 element occupies one "
            "byte plane in the transpose storage; FP16 uses two byte planes "
            "(low/high byte). It is the SXM data-width mode, not a separate "
            "hardware block. The total and phase MXM rows use all four "
            "configured MXMs as capacity. The VXM total row measures every "
            "configured Superlane; the chain count and depth above come from "
            "the active VXM program, so `VXM q configured` measures only "
            "configured chain outputs; "
            "`VXM q physical` measures all 8 fixed quantizers per lane.\n";
}

inline void write_detailed_trace(
    const FullFfnResult& result,
    std::filesystem::path directory = results_directory())
{
    std::filesystem::create_directories(directory);
    auto file = std::ofstream{
        directory / (result.name + "_detailed_trace.txt"),
        std::ios::trunc};
    if (!file) throw std::runtime_error("cannot create full-FFN trace");
    file << "Full FFN per-cycle trace\n"
         << "shape=tokens:" << result.tokens << " hidden:" << result.hidden
         << " intermediate:" << result.intermediate
         << " output:" << result.output_features << "\n"
         << "vxm=superlanes:" << ftlpu::hw::kTileRows
         << " lanes_per_superlane:" << ftlpu::hw::kLanesPerTile
         << " chain_depth:" << result.chain_depth
         << " chains_per_lane:" << result.chains_per_lane << "\n"
         << "flow=Gate/Up_MXM,VXM_SwiGLU,StaticQuant,SR_MEM,"
            "SXM_Transpose_Permute,Down_MXM,FP16_Cast\n"
         << "fields: cycle phase events gate_cells up_cells vxm_slots "
            "quant_outputs mem_reads mem_writes sxm_transpose_tiles "
            "sxm_permute_tiles front_load_tiles down_load_tiles down_cells "
            "down_outputs icu_imem_reads icu_iq_arrivals icu_dispatches "
            "icu_waiting_queues\n\n";
    for (const auto& cycle : result.timeline) {
        file << "cycle=" << cycle.cycle
             << " phase=" << cycle.phase
             << " events=" << (cycle.events.empty() ? "-" : cycle.events)
             << " gate_cells=" << cycle.gate_mxm_cells
             << " up_cells=" << cycle.up_mxm_cells
             << " vxm_slots=" << cycle.vxm_alu_slots
             << " quant_outputs=" << cycle.quantized_outputs
             << " mem_reads=" << cycle.mem_reads
             << " mem_writes=" << cycle.mem_writes
             << " sxm_transpose_tiles=" << cycle.sxm_transpose_tiles
             << " sxm_permute_tiles=" << cycle.sxm_permute_tiles
             << " front_load_tiles=" << cycle.gate_up_weight_load_tiles
             << " down_load_tiles=" << cycle.down_weight_load_tiles
             << " down_cells=" << cycle.down_mxm_cells
             << " down_outputs=" << cycle.down_acc_outputs
             << " icu_imem_reads=" << cycle.icu_imem_reads
             << " icu_iq_arrivals=" << cycle.icu_iq_arrivals
             << " icu_dispatches=" << cycle.icu_dispatches
             << " icu_waiting_queues=" << cycle.icu_waiting_queues
             << '\n';
    }
}

inline void write_gantt(
    const FullFfnResult& result,
    std::filesystem::path directory = results_directory())
{
    std::filesystem::create_directories(directory);
    auto file = std::ofstream{
        directory / (result.name + "_gantt.html"),
        std::ios::trunc};
    if (!file) throw std::runtime_error("cannot create full-FFN Gantt");
    constexpr std::size_t label_width = 180;
    constexpr std::size_t row_height = 23;
    constexpr std::size_t pixels_per_cycle = 1;
    const auto plot_width = std::max<std::size_t>(1, result.timeline.size());
    const auto max_mxm_cells = ftlpu::hw::kMxmSupercellsPerPlane
        * ftlpu::hw::kMxmSupercellsPerPlane;
    const std::vector<std::string> labels{
        "Protocol phase", "ICU dispatch", "Gate/Up weight load", "Gate MXM", "Up MXM", "VXM ALUs",
        "Static quantizer (8/lane)", "MEM", "SXM Transpose", "SXM Permute",
        "Down weight load", "Down MXM/ACC"};
    file << "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>Pipelined Edge FFN Gantt</title><style>"
            "body{font:13px system-ui;margin:20px;background:#f5f7fb;color:#172033}"
            ".wrap{overflow:auto;background:#fff;border:1px solid #d9e0ea}"
            "svg{display:block}.label{font-weight:600}.grid{stroke:#d9e0ea}"
            ".phase{fill:#7655c7}.mxm{fill:#2f7fd3}.vxm{fill:#d44f72}"
            ".quant{fill:#e87f31}.mem{fill:#4e9b72}.sxm{fill:#19a0a8}"
            ".load{fill:#d4a022}.icu{fill:#6b5bd2}</style></head><body><h1>Pipelined Edge Transformer FFN</h1>"
         << "<p>" << result.tokens << " x " << result.hidden << " x "
         << result.intermediate << ", cycles=" << result.cycles
         << ". Hover a bar for cycle details.</p><div class=\"wrap\"><svg width=\""
         << label_width + plot_width + 30 << "\" height=\""
         << 42 + labels.size() * row_height << "\">";
    for (std::size_t row = 0; row < labels.size(); ++row) {
        const auto y = 25 + row * row_height;
        file << "<text class=\"label\" x=\"6\" y=\"" << y + 15
             << "\">" << labels[row] << "</text><line class=\"grid\" x1=\""
             << label_width << "\" y1=\"" << y + row_height
             << "\" x2=\"" << label_width + plot_width
             << "\" y2=\"" << y + row_height << "\"/>";
    }
    const auto tick_spacing = std::max<std::size_t>(
        50, ((result.timeline.size() / 8 + 49) / 50) * 50);
    for (std::size_t cycle = 0;
         cycle <= result.timeline.size(); cycle += tick_spacing) {
        const auto x = label_width + cycle;
        file << "<line class=\"grid\" x1=\"" << x << "\" y1=\"20\" x2=\""
             << x << "\" y2=\"" << 25 + labels.size() * row_height
             << "\"/><text x=\"" << x + 2 << "\" y=\"14\">C"
             << cycle << "</text>";
    }
    for (std::size_t i = 0; i < result.timeline.size(); ++i) {
        const auto& c = result.timeline[i];
        const auto x = label_width + i * pixels_per_cycle;
        const auto rect = [&](std::size_t row, const char* css, double opacity) {
            if (opacity <= 0.0) return;
            file << "<rect class=\"" << css << "\" x=\"" << x
                 << "\" y=\"" << 27 + row * row_height << "\" width=\"1\" height=\"19\" opacity=\""
                 << std::min(1.0, opacity) << "\"><title>C" << c.cycle << ' '
                 << html_escape(c.phase) << ' ' << html_escape(c.events)
                 << "</title></rect>";
        };
        rect(0, "phase", c.phase.empty() ? 0.0 : 0.65);
        rect(1, "icu", static_cast<double>(c.icu_dispatches) /
            std::max<std::size_t>(1, result.icu_active_queues));
        rect(2, "load", static_cast<double>(c.gate_up_weight_load_tiles) /
            (2 * ftlpu::hw::kTileRows));
        rect(3, "mxm", static_cast<double>(c.gate_mxm_cells) / max_mxm_cells);
        rect(4, "mxm", static_cast<double>(c.up_mxm_cells) / max_mxm_cells);
        rect(5, "vxm", static_cast<double>(c.vxm_alu_slots) /
            std::max<std::size_t>(1,
                ftlpu::hw::kTileRows * ftlpu::hw::kLanesPerTile * 16));
        rect(6, "quant", static_cast<double>(c.quantized_outputs) /
            std::max<std::size_t>(1, result.quantizer_channels));
        rect(7, "mem", std::min(1.0, static_cast<double>(c.mem_reads + c.mem_writes) /
            ftlpu::hw::kLanesPerTile));
        rect(8, "sxm", static_cast<double>(c.sxm_transpose_tiles) /
            ftlpu::hw::kTileRows);
        rect(9, "sxm", static_cast<double>(c.sxm_permute_tiles) /
            ftlpu::hw::kTileRows);
        rect(10, "load", static_cast<double>(c.down_weight_load_tiles) /
            ftlpu::hw::kTileRows);
        rect(11, "mxm", std::max(
            static_cast<double>(c.down_mxm_cells) / max_mxm_cells,
            c.down_acc_outputs == 0 ? 0.0 : 0.15));
    }
    file << "</svg></div></body></html>";
}

inline void write_reports(
    const FullFfnResult& result,
    std::filesystem::path directory = results_directory())
{
    write_brief_report(result, directory);
    write_detailed_trace(result, directory);
    write_gantt(result, directory);
}

} // namespace full_ffn_test
