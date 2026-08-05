#pragma once

#include "ftlpu/vxm/compiler/cmodel_adapter.hpp"
#include "ftlpu/vxm/compiler/print.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>

namespace ftlpu::vxm::compiler {

inline void write_cmodel_summary_report(
    const std::filesystem::path& path,
    const VxmCompiledProgram& program,
    const VxmCModelRunResult& result)
{
    auto file = std::ofstream{path};
    if (!file) {
        throw std::runtime_error(
            "cannot open VXM C Model summary report");
    }
    auto requests_by_value = std::map<ValueId, std::size_t>{};
    auto outputs_by_value = std::map<ValueId, std::size_t>{};
    for (const auto& request : result.requests) {
        ++requests_by_value[request.value];
    }
    for (const auto& output : result.outputs) {
        ++outputs_by_value[output.value];
    }

    file << "VXM C Model summary\n"
         << "kernel=" << program.kernel.name << '\n'
         << "compact_instruction_bits="
         << VxmCompactInstructionCodec::kEncodedBits << '\n'
         << "phases=" << program.phases.size() << '\n'
         << "superlanes=" << VxmSlice::kTileCount << '\n'
         << "lanes_per_superlane=" << VxmSuperlane::kLaneCount << '\n'
         << "sram_read_latency=" << result.sram_read_latency << '\n'
         << "sram_write_latency=" << result.sram_write_latency << '\n'
         << "cycles=" << result.cycles << '\n'
         << "sram_read_requests=" << result.requests.size() << '\n'
         << "output_events=" << result.outputs.size() << '\n';
    file << "phase_plan:\n";
    for (std::size_t index = 0;
         index < program.phases.size(); ++index) {
        const auto& phase = program.phases[index];
        const auto shift = result.phase_shifts.at(index);
        file << "  " << phase.phase_id << ' ' << phase.name
             << " depth=" << static_cast<std::size_t>(phase.chain_depth)
             << " scheduled=[" << phase.data_start_cycle
             << ',' << phase.end_cycle << ")"
             << " actual=[" << phase.data_start_cycle + shift
             << ',' << phase.end_cycle + shift << ")"
             << " dependency_shift=" << shift
             << " added_sram_wait="
             << result.phase_sram_waits.at(index)
             << " added_scalar_load_wait="
             << result.phase_scalar_load_waits.at(index) << '\n';
    }
    file << "requests_by_value:\n";
    for (const auto& [value, count] : requests_by_value) {
        file << "  v" << value << '=' << count << '\n';
    }
    file << "outputs_by_value:\n";
    for (const auto& [value, count] : outputs_by_value) {
        file << "  v" << value << '=' << count
             << (std::find(
                     program.kernel.outputs.begin(),
                     program.kernel.outputs.end(), value)
                     != program.kernel.outputs.end()
                     ? " final" : " intermediate")
             << '\n';
    }
}

inline char trace_state_code(VxmLaneAluTraceState state)
{
    switch (state) {
    case VxmLaneAluTraceState::Idle: return '.';
    case VxmLaneAluTraceState::Stalled: return 'S';
    case VxmLaneAluTraceState::Executed: return 'E';
    }
    return '?';
}

inline void write_cmodel_detailed_report(
    const std::filesystem::path& path,
    const VxmCompiledProgram& program,
    const VxmCModelRunResult& result)
{
    auto file = std::ofstream{path};
    if (!file) {
        throw std::runtime_error(
            "cannot open VXM C Model detailed report");
    }
    file << "VXM C Model detailed cycle report\n"
         << "kernel=" << program.kernel.name
         << " cycles=" << result.cycles
         << " sram_read_latency=" << result.sram_read_latency
         << " sram_write_latency=" << result.sram_write_latency
         << "\n\n";
    file << "ACTUAL PHASE TIMING\n";
    for (std::size_t index = 0;
         index < program.phases.size(); ++index) {
        const auto& phase = program.phases[index];
        const auto shift = result.phase_shifts.at(index);
        file << "  phase=" << phase.phase_id
             << " actual=[" << phase.data_start_cycle + shift
             << ',' << phase.end_cycle + shift << ")"
             << " cumulative_shift=" << shift
             << " added_sram_wait="
             << result.phase_sram_waits.at(index)
             << " added_scalar_load_wait="
             << result.phase_scalar_load_waits.at(index) << '\n';
    }
    file << '\n';
    file << "COMPACT CONFIGURATION\n";
    print_compiled_program(file, program);
    file << "CYCLE TIMELINE\n"
         << "state: E=executed S=stalled .=idle\n\n";

    for (const auto& record : result.timeline) {
        file << "cycle " << record.cycle << '\n';
        for (const auto& config : record.configs) {
            const auto decoded = VxmCompactInstructionCodec::decode(
                config.stage, config.packet);
            file << "  CONFIG south ALU" << config.stage
                 << " phase=" << config.phase_id
                 << " packet={0x" << std::hex << config.packet.control
                 << ",0x" << config.packet.immediate_bits << std::dec
                 << "} "
                 << VxmLane::operation_name(
                        decoded.instruction.operation)
                 << " repeat=" << decoded.instruction.repeat_count
                 << '\n';
        }
        for (std::size_t tile = 0;
             tile < VxmSlice::kTileCount
             && tile <= record.cycle; ++tile) {
            const auto source_cycle = record.cycle - tile;
            if (source_cycle >= result.timeline.size()) continue;
            for (const auto& config :
                 result.timeline[source_cycle].configs) {
                file << "  CONFIG_ARRIVE superlane=" << tile
                     << " ALU" << config.stage
                     << " phase=" << config.phase_id
                     << " packet={0x" << std::hex
                     << config.packet.control << ",0x"
                     << config.packet.immediate_bits << std::dec
                     << "}\n";
            }
        }
        for (const auto& depth : record.depth_changes) {
            file << "  DEPTH superlane=" << depth.superlane
                 << " phase=" << depth.phase_id
                 << " depth=" << static_cast<std::size_t>(depth.depth)
                 << (depth.feedback_transition
                         ? " feedback_transition" : "")
                 << '\n';
        }
        for (const auto& request : record.requests) {
            file << "  REQUEST superlane=" << request.superlane
                 << " phase=" << request.phase_id
                 << " v" << request.value
                 << " stream=" << request.stream_base
                 << " element=" << request.element_index
                 << " required_cycle=" << request.required_cycle
                 << (request.hold ? " hold" : "") << '\n';
        }
        for (const auto& input : record.inputs) {
            file << "  INPUT superlane=" << input.superlane
                 << " phase=" << input.phase_id
                 << " v" << input.value
                 << " stream=" << input.stream_base
                 << " element=" << input.element_index
                 << (input.held ? " held" : "") << '\n';
        }
        for (const auto& scalar : record.scalar_loads) {
            file << "  SCALAR superlane=" << scalar.superlane
                 << " phase=" << scalar.phase_id
                 << " v" << scalar.value
                 << " stream=" << scalar.source_stream
                 << " -> ALU" << scalar.destination_stage << '\n';
        }
        for (const auto& output : record.outputs) {
            file << "  OUTPUT superlane=" << output.superlane
                 << " phase=" << output.phase_id
                 << " v" << output.value
                 << " stream=" << output.stream_base
                 << " element=" << output.element_index
                 << " sram_visible_cycle="
                 << output.sram_visible_cycle << '\n';
        }
        for (std::size_t tile = 0;
             tile < VxmSlice::kTileCount; ++tile) {
            const auto& states = record.alu_states[tile];
            if (std::all_of(
                    states.begin(), states.end(),
                    [](auto state) {
                        return state == VxmLaneAluTraceState::Idle;
                    })) {
                continue;
            }
            file << "  ALU superlane=" << tile
                 << " depth="
                 << static_cast<std::size_t>(record.depths[tile])
                 << " lane0=";
            for (const auto state : states) {
                file << trace_state_code(state);
            }
            file << '\n';
        }
        file << '\n';
    }
}

inline void write_cmodel_gantt(
    const std::filesystem::path& path,
    const VxmCompiledProgram& program,
    const VxmCModelRunResult& result)
{
    auto file = std::ofstream{path};
    if (!file) {
        throw std::runtime_error("cannot open VXM C Model Gantt report");
    }
    file << R"HTML(<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VXM C Model Gantt</title>
<style>
:root{color-scheme:light dark;--bg:#f4f7fb;--panel:#fff;--text:#172033;--muted:#68748a;--line:#d8dfeb;--idle:#edf1f6;--exec:#1f9d76;--stall:#df5b5b;--config:#ef9f32;--input:#4185d7;--output:#9b62d1}
@media(prefers-color-scheme:dark){:root{--bg:#10151f;--panel:#171e2b;--text:#e8edf6;--muted:#a4afc1;--line:#344054;--idle:#273142;--exec:#36c596;--stall:#f27777;--config:#f4ad45;--input:#65a5ef;--output:#ba86e8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}header{padding:18px 22px 10px}h1{font-size:20px;margin:0 0 5px}p{margin:0;color:var(--muted)}.controls{display:flex;gap:18px;flex-wrap:wrap;padding:10px 22px}.controls label{display:flex;gap:8px;align-items:center}.legend{display:flex;gap:14px;flex-wrap:wrap;padding:4px 22px 12px;color:var(--muted)}.swatch{width:12px;height:12px;display:inline-block;border-radius:3px;margin-right:5px}.wrap{margin:0 14px 14px;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:auto}.section{padding:10px 14px 4px;font-weight:600}.canvas-wrap{overflow:auto}canvas{display:block}.hover{padding:8px 14px 12px;color:var(--muted);min-height:34px}
</style></head><body>
<header><h1>VXM完整执行甘特图</h1><p>上方显示配置数量的Superlane Phase波前，下方显示所选Superlane的配置、数据、ALU和输出。</p></header>
<div class="controls">
<label>Superlane <select id="tile"></select></label>
<label>起始周期 <input id="start" type="range" min="0" value="0"></label>
<label>显示周期 <input id="window" type="range" min="20" value="120"></label>
</div>
<div class="legend">
<span><i class="swatch" style="background:var(--exec)"></i>ALU执行</span>
<span><i class="swatch" style="background:var(--stall)"></i>Stall</span>
<span><i class="swatch" style="background:var(--config)"></i>配置</span>
<span><i class="swatch" style="background:var(--input)"></i>输入</span>
<span><i class="swatch" style="background:var(--output)"></i>输出</span>
</div>
<div class="wrap"><div class="section">Superlane Phase波前</div><div class="canvas-wrap"><canvas id="overview"></canvas></div></div>
<div class="wrap"><div class="section">所选Superlane逐周期执行</div><div class="canvas-wrap"><canvas id="detail"></canvas></div><div id="hover" class="hover">移动鼠标查看周期。</div></div>
<script>
const DATA=)HTML";
    file << "{\"cycles\":" << result.cycles
         << ",\"superlanes\":" << VxmSlice::kTileCount
         << ",\"lanesPerSuperlane\":" << VxmSuperlane::kLaneCount
         << ",\"readLatency\":" << result.sram_read_latency
         << ",\"writeLatency\":" << result.sram_write_latency
         << ",\"phases\":[";
    for (std::size_t index = 0; index < program.phases.size(); ++index) {
        if (index) file << ',';
        const auto& phase = program.phases[index];
        const auto shift = result.phase_shifts.at(index);
        file << "{\"id\":" << phase.phase_id
             << ",\"name\":\"" << phase.name << "\""
             << ",\"start\":" << phase.data_start_cycle + shift
             << ",\"end\":" << phase.end_cycle + shift << '}';
    }
    file << "],\"records\":[";
    for (std::size_t index = 0; index < result.timeline.size(); ++index) {
        if (index) file << ',';
        const auto& record = result.timeline[index];
        file << "{\"c\":" << record.cycle << ",\"cfg\":[";
        for (std::size_t item = 0; item < record.configs.size(); ++item) {
            if (item) file << ',';
            file << '[' << record.configs[item].phase_id
                 << ',' << record.configs[item].stage << ']';
        }
        file << "],\"in\":[";
        for (std::size_t item = 0; item < record.inputs.size(); ++item) {
            if (item) file << ',';
            const auto& input = record.inputs[item];
            file << '[' << input.superlane << ',' << input.value
                 << ',' << input.stream_base << ','
                 << input.element_index << ']';
        }
        file << "],\"out\":[";
        for (std::size_t item = 0; item < record.outputs.size(); ++item) {
            if (item) file << ',';
            const auto& output = record.outputs[item];
            file << '[' << output.superlane << ',' << output.value
                 << ',' << output.stream_base << ','
                 << output.element_index << ']';
        }
        file << "],\"s\":[";
        for (std::size_t tile = 0;
             tile < VxmSlice::kTileCount; ++tile) {
            if (tile) file << ',';
            file << '"';
            for (const auto state : record.alu_states[tile]) {
                file << trace_state_code(state);
            }
            file << '"';
        }
        file << "]}";
    }
    file << R"HTML(]};
const COLORS=['#5277d3','#22a087','#a86ac7','#d88735','#d85c77','#5d9eb8'];
const tileSel=document.getElementById('tile'),startCtl=document.getElementById('start'),windowCtl=document.getElementById('window');
for(let i=0;i<DATA.superlanes;i++){const o=document.createElement('option');o.value=i;o.textContent=i;tileSel.appendChild(o)}
startCtl.max=Math.max(0,DATA.cycles-20);windowCtl.max=DATA.cycles;windowCtl.value=Math.min(120,DATA.cycles);
const overview=document.getElementById('overview'),detail=document.getElementById('detail'),hover=document.getElementById('hover');
const LEFT=105,CELL=8,ROW=20;
function range(){const start=+startCtl.value,count=Math.min(+windowCtl.value,DATA.cycles-start);return{start,count}}
function setup(canvas,rows,count){canvas.width=Math.max(760,LEFT+count*CELL+20);canvas.height=34+rows*ROW;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.font='12px system-ui';ctx.textBaseline='middle';return ctx}
function grid(ctx,rows,start,count){ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--line');ctx.lineWidth=1;for(let x=0;x<=count;x+=10){const px=LEFT+x*CELL;ctx.beginPath();ctx.moveTo(px,24);ctx.lineTo(px,28+rows*ROW);ctx.stroke();ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--muted');ctx.fillText(start+x,px+2,12)}}
function phaseAt(tile,cycle){for(const p of DATA.phases){if(cycle>=p.start+tile&&cycle<p.end+tile)return p}return null}
function drawOverview(){const{start,count}=range(),ctx=setup(overview,DATA.superlanes,count);grid(ctx,DATA.superlanes,start,count);for(let t=0;t<DATA.superlanes;t++){ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--text');ctx.fillText('Superlane '+t,8,34+t*ROW+ROW/2);for(let j=0;j<count;j++){const p=phaseAt(t,start+j);ctx.fillStyle=p?COLORS[p.id%COLORS.length]:getComputedStyle(document.documentElement).getPropertyValue('--idle');ctx.fillRect(LEFT+j*CELL,34+t*ROW,CELL-1,ROW-3)}}}
function hasEvent(list,tile){return list.some(x=>x[0]===tile)}
function drawDetail(){const{start,count}=range(),tile=+tileSel.value,labels=['Config','Input',...Array.from({length:16},(_,i)=>'ALU'+i),'Output'],ctx=setup(detail,labels.length,count);grid(ctx,labels.length,start,count);labels.forEach((l,i)=>{ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--text');ctx.fillText(l,8,34+i*ROW+ROW/2)});for(let j=0;j<count;j++){const cycle=start+j,r=DATA.records[cycle],x=LEFT+j*CELL,source=cycle-tile,cfg=source>=0?DATA.records[source].cfg:[];ctx.fillStyle=cfg.length?getComputedStyle(document.documentElement).getPropertyValue('--config'):getComputedStyle(document.documentElement).getPropertyValue('--idle');ctx.fillRect(x,34,CELL-1,ROW-3);ctx.fillStyle=hasEvent(r.in,tile)?getComputedStyle(document.documentElement).getPropertyValue('--input'):getComputedStyle(document.documentElement).getPropertyValue('--idle');ctx.fillRect(x,34+ROW,CELL-1,ROW-3);for(let a=0;a<16;a++){const s=r.s[tile][a];ctx.fillStyle=s==='E'?getComputedStyle(document.documentElement).getPropertyValue('--exec'):s==='S'?getComputedStyle(document.documentElement).getPropertyValue('--stall'):getComputedStyle(document.documentElement).getPropertyValue('--idle');ctx.fillRect(x,34+(a+2)*ROW,CELL-1,ROW-3)}ctx.fillStyle=hasEvent(r.out,tile)?getComputedStyle(document.documentElement).getPropertyValue('--output'):getComputedStyle(document.documentElement).getPropertyValue('--idle');ctx.fillRect(x,34+18*ROW,CELL-1,ROW-3)}}
function draw(){drawOverview();drawDetail()}
[tileSel,startCtl,windowCtl].forEach(e=>e.addEventListener('input',draw));
detail.addEventListener('mousemove',e=>{const rect=detail.getBoundingClientRect(),x=(e.clientX-rect.left)*(detail.width/rect.width),cycle=Math.floor((x-LEFT)/CELL)+(+startCtl.value);if(cycle<0||cycle>=DATA.cycles)return;const r=DATA.records[cycle],tile=+tileSel.value,p=phaseAt(tile,cycle);hover.textContent=`cycle ${cycle} · Superlane ${tile} · ${p?p.name:'idle'} · ALU ${r.s[tile]} · input ${r.in.filter(x=>x[0]===tile).length} · output ${r.out.filter(x=>x[0]===tile).length}`});
draw();
</script></body></html>)HTML";
}

} // namespace ftlpu::vxm::compiler
