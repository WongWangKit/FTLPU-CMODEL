# Documentation layout

- `FTLPU.drawio` is the editable architecture source.
- `figures/` contains rendered PNG and SVG documentation figures.
- `traces/` contains generated schedule CSV files for Pipeline Viewer.
- Markdown design notes remain directly under `docs/`.
- `c2c_shared_streams.md` and `c2c_shared_streams.zh-CN.md` describe the
  default C2C RX -> ordinary SR -> MEM Write path.
- `icu_vxm_stream_nd.md` and `icu_vxm_stream_nd.zh-CN.md` describe the merged
  VXM packet and N-D launch domain.
- `icu_sxm_tile_program.md` and `icu_sxm_tile_program.zh-CN.md` describe the
  coarse transpose/permute tile program.
- `icu_mem_slice_program.md` and `icu_mem_slice_program.zh-CN.md` describe the
  multi-operation program executed by one physical MEM slice ICU.

Generated files keep stable names so they can be regenerated in place. For
example, serve the repository and open:

```text
http://127.0.0.1:8765/tools/pipeline_viewer/?trace=/docs/traces/smollm2_attention_schedule.csv
```
