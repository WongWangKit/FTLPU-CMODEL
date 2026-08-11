# FTLPU Pipeline Viewer

The pipeline viewer is a dependency-free Canvas waveform workspace for runtime
schedule traces. Open `index.html` in a browser and load the CSV emitted by
`write_schedule_trace_csv()`.

When serving the repository over HTTP, a trace can also be opened directly:

```text
http://127.0.0.1:8765/tools/pipeline_viewer/?trace=/docs/traces/smollm2_block8_dequant_ffn_schedule.csv
```

Supported input columns:

```csv
start,end,resource,detail
0,32,"MEM.E.Read","slice=0 addr=0 stream=E0"
32,64,"MXM.E0.Compute","Compute buffer=0 act=E0 out=W0"
```

Controls:

- Mouse wheel: zoom around the pointer.
- Drag: pan the visible cycle window.
- Shift + wheel: horizontal pan.
- Ctrl/Alt + wheel: vertical resource scrolling.
- A/B buttons followed by a click: place measurement cursors.
- Overview click: recenter the visible window.
- Overview drag or wheel: pan horizontally; hold Shift for fine control.
- Focus the overview and use Left/Right; hold Shift for smaller steps.
- Search and unit selector: filter resources and instruction details.

The viewer automatically enables level-of-detail sampling in wide views and
returns to exact per-event drawing as you zoom in. The status strip reports the
LOD draw count; hit testing and event inspection always use the full trace.

Runtime tests already generate traces such as:

```text
build-ftlpu-vs2026/compiler/ftlpu_lower/
  smollm2_135m_ffn_seq128_pipeline/ffn.runtime.csv
  smollm2_attention_pipeline/attention.runtime.csv
  smollm2_decoder_layer_binary_runtime/decoder_layer.runtime.csv
```

The viewer renders only visible rows and cycle intervals, so long schedules do
not require an ultra-wide SVG. Per-resource time indexes and a cached, bounded
overview keep full-layer traces with hundreds of thousands of events
interactive. Existing SVG/PNG renderers remain useful for documentation
snapshots.
