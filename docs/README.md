Current MEM/MXM/SXM ISA: [packed Macro](icu_macro_scheduler.zh-CN.md).

Current VXM ISA: [96-bit sequential configurations](icu_vxm_sequential.md). The STREAM_ND documents below are historical and are not supported by the current decoder.

# Documentation layout

- [Native MEM/MXM/VXM RTL alignment](rtl_alignment.md) documents the native
  entry point, current instructions, cycle contracts and port-level RTL comparisons.

- [MXM scheduling contract](mxm_scheduling_contract.md), [instruction encoding](mxm_instructions.md),
  and [weight ingress](mxm_weight_ingress.md) are the shared VMODEL references.
  The linked [Supercell](mxm_supercell.md) and [accumulator](mxm_accumulator.md) notes cover the datapath.
  Run their RTL commands in VMODEL and the CModel comparisons from [RTL alignment](rtl_alignment.md).

- [MEM instruction encoding](mem_instructions.md) and the [MEM scheduling contract](mem_scheduling_contract.md)
  describe the native 32-bit word, 1/3-cycle propagation, valid management and compiler timing.
- The [VMODEL hardware architecture](vmodel_architecture.md) explains the RTL module boundaries,
  [VXM data paths and instructions](vmodel_architecture.md#vxm-module-boundaries), and the [C3 Log extension](vmodel_architecture.md#c3-log-extension).
- [VXM RTL verification](vxm_rtl_verification.md), [Lane and Superlane chains](vxm_chain_verification.md),
  and [physical ALU positions](vxm_alu_position_verification.md) describe the RTL test scope and commands.
- Open the [MEM data wavefront](visualizations/mem-stream-edge.html) and [MEM/MXM timing view](visualizations/mem-mxm-phase-3d.html)
  locally in a browser. These diagrams show scheduling examples.

- `FTLPU.drawio` is the editable architecture source.
- `figures/` contains rendered PNG and SVG documentation figures.
- `traces/` contains generated schedule CSV files for Pipeline Viewer.
- Markdown design notes remain directly under `docs/`.
- `c2c_shared_streams.md` and `c2c_shared_streams.zh-CN.md` describe the
  default C2C RX -> ordinary SR -> MEM Write path.
- `icu_vxm_stream_nd.md` and `icu_vxm_stream_nd.zh-CN.md` are historical records of the merged
  VXM packet and N-D launch domain.
- `icu_sxm_tile_program.md` and `icu_sxm_tile_program.zh-CN.md` are historical records of the
  coarse transpose/permute tile program.
- `icu_mem_slice_program.md` and `icu_mem_slice_program.zh-CN.md` are historical records of the
  multi-operation program executed by one physical MEM slice ICU.
- [SXM instruction status](sxm_instruction_status.md) describes the opt-in typed
  matrix reference, unchanged legacy ABI, and the remaining RTL/CModel boundaries.
- [SXM/MEM regression](sxm_mem_regression.md) contains repository-local commands,
  the validated small-regression results, and publication scope.
- [SXM cycle-23 fix](sxm_mem_transpose_timing.md) records the legacy ICU scheduling fix.
- [MEM Main/R1 timing](mem_read_timing.md) describes unchanged Main defaults and explicit R1 use.

Generated files keep stable names so they can be regenerated in place. For
example, serve the repository and open:

```text
http://127.0.0.1:8765/tools/pipeline_viewer/?trace=/docs/traces/smollm2_attention_schedule.csv
```
