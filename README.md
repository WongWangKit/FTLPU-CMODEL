# FTLPU-CMODEL

[English](README.md) | [简体中文](README.zh-CN.md)

FTLPU-CMODEL is a cycle-oriented C++20 model of an FTLPU/TSP-style processor.
It models stream-register timing, per-queue ICU control, two mirrored MEM
hemispheres, four MXMs, a central VXM, and one SXM per hemisphere.

The project is inspired by public Groq LPU/TSP material, but it is not a
bit-accurate implementation of private Groq hardware or ISA. Its purpose is to
provide a concrete target for dataflow scheduling and future compiler work.

## Architecture Snapshot

| Block | Current model |
| --- | --- |
| Vector shape | 4 tiles/superlanes x 8 lanes = 32 elements |
| Streams | 32 eastward + 32 westward streams, one byte per register |
| MEM | 52 slices per hemisphere, two bank queues per slice, 208 ICU queues total |
| SRAM | Two 256 KiB single-port banks per slice, 26 MiB per hemisphere, 52 MiB total |
| Accumulators | JSON-configured complete 32x32 FP32 block count per MXM (256 blocks / 1 MiB by default) |
| MXM | Four 32 x 32 FP16 GEMM arrays, two per hemisphere |
| MXM weights | Two peer buffers per supercell, selected by `IW`/`Compute` |
| MXM decode | Selectable `Linear1x16` or activation-stationary `Native4x4` |
| VXM | One central slice; two mirrored 8-stage chains (16 physical ALUs) per lane |
| SXM | One four-tile slice per hemisphere for Transpose/Permute |
| ICU | 208 MEM, 12 MXM, 8 VXM compact-control, 4 SXM, and 6 C2C/DMA queues |

The fixed full-chip topology is:

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W <-> VXM <-> MEM.E <-> SXM.E <-> MXM0/MXM1
```

Each hemisphere uses local stream-register columns `sreg0..sreg15`.
`sreg0` is next to VXM, MEM occupies the thirteen groups between
`sreg0..sreg13`, C2C attaches at `sreg13`, and SXM spans `sreg14` to the MXM
boundary at `sreg15`.

Stream reads are broadcast-capable: multiple functional units may consume the
same register value in one cycle. A consumed value no longer propagates
passively, and multiple producers still cannot write different values to the
same stream register.

## Execution Model

Whole-system workloads follow an offline-only contract:

1. Initialize external input data in MEM before cycle 0.
2. Generate the complete instruction timeline and enqueue every ICU queue.
3. Start the clock and call only `TspSliceSystem::tick()`.
4. Read final MEM state and compare it with a software golden model.

MEM, MXM, VXM, and SXM instructions must meet their stream operands in the same
cycle. Each MEM slice has independent bank-0 and bank-1 queues. A bank is
single-port and performs at most one read or write per tile per cycle; the two
banks may operate concurrently. Queue-local `NOP N` and `Repeat n,d` commands
encode delays and regular instruction trains. MEM repeats may also apply a
signed address stride.

## Validated Workloads

| Test | Workload | Validation |
| --- | --- | --- |
| `w8a16_projection_test` | `[128,576] x [576,1536]` W8A16 projection | 196,608 FP32 outputs |
| `w8a16_swiglu_test` | gate/up projection plus SwiGLU | 196,608 FP16 outputs |
| `dual_hemisphere_w8a16_swiglu_test` | full gate/up, SwiGLU, and down FFN | `[128,576]` final FP16 output |
| `rmsnorm_test` | `[32,32]` FP16 RMSNorm | all stored FP16 outputs |
| `smollm2_attention_test` | ICU-driven Q/K RoPE, QK, softmax, and P x V hardware tile | bit-checked RoPE plus `8 x 32` attention output |
| `mxm_decode_layout_comparison_test` | Same `K=128, N=32` GEMV in both decode layouts | bit-identical BF16 outputs and cycle comparison |
| `smollm2_decode_ffn_test` | Native 4 x 4 weight-streaming decode FFN | `[1,576]` final BF16 output |
| `sxm_mem_transpose_test` | continuous MEM -> SXM -> MEM FP16 transpose | four 32 x 32 matrices |

The full FFN uses all four MXMs and currently schedules 90,817 cycles. Its final
gate/up reduction streams accumulator results directly into the shared VXM
SwiGLU pipeline. The standalone attention executable validates one physical
`8-token x 4-head x 8-dimension` tile. The layer-phase harness repeats that
tile mapping over a `[128,576]` API shape.

MXM Decode instructions carry an explicit layout bit. `Linear1x16` loads four
independent 8-element activation vectors per tile from eight streams and walks
one partial sum through all 16 supercells. `Native4x4` loads one 8-element
vector per tile from two BF16 streams, broadcasts it across that physical row,
and computes four vertical reduction chains in parallel. Its 32 INT8 weight
streams map to four 8-stream physical columns; column `c` reaches the MXM
boundary `c` cycles after column 0, producing a seven-cycle diagonal wave.
The layout comparison test keeps both implementations numerically locked to
the same golden GEMV. The SmolLM2 decode FFN selects `Native4x4`; the current
decode attention resident layout intentionally remains on `Linear1x16`.

## Build

The hardware target is defined once in
[`config/ftlpu-lpu32.json`](config/ftlpu-lpu32.json). CMake validates this file
and generates the compile-time constants used by fixed-size CModel structures.
Select another compatible target at configure time with
`-DFTLPU_HARDWARE_CONFIG=<target.json>`. FTLPU-SOFTWARE uses the same cache
variable and JSON file, so changing physical geometry requires reconfiguring
and rebuilding both projects.

The current Windows build is tested with Visual Studio 2026 Community:

```powershell
cmake -S . -B build-vs2026 `
  -G "Visual Studio 18 2026" `
  -A x64
cmake --build build-vs2026 --config Release
ctest --test-dir build-vs2026 -C Release --output-on-failure
```

For another CMake generator:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the two largest regressions directly:

```powershell
build-vs2026\Release\dual_hemisphere_w8a16_swiglu_test.exe
build-vs2026\Release\smollm2_attention_test.exe
```

Whole-system logging is disabled by default because per-cycle traces are
expensive. Small tests and demos can provide `TspSliceSystem::LogSinks` for
separate ICU, MEM, MXM, VXM, SXM, and system logs.
The no-log path skips MEM/VXM/SXM trace construction. SRAM geometry comes from
the hardware target JSON. The default target uses 8192 bank-local rows at
32 bytes per row, giving 256 KiB per bank and 512 KiB per MEM slice. Backing
storage is allocated lazily in 4 KiB pages.

## Schedule Diagrams

- [W8A16 projection pipeline](docs/figures/w8a16_projection_pipeline.svg)
- [Full FFN pipeline](docs/figures/w8a16_swiglu_pipeline.svg)
- [Full FFN detailed ICU schedule](docs/figures/w8a16_swiglu_schedule_detail.svg)
- [SmolLM2 attention pipeline](docs/figures/smollm2_attention_pipeline.svg)
- [Attention optimization comparison](docs/figures/smollm2_attention_pipeline_optimization.svg)
- [Attention detailed ICU schedule](docs/figures/smollm2_attention_schedule_detail.svg)

Regenerate the FFN schedule:

```powershell
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\w8a16_swiglu\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
build-vs2026\Release\dual_hemisphere_w8a16_swiglu_test.exe
python scripts\render_swiglu_schedule_trace.py `
  logs\w8a16_swiglu\schedule.csv `
  docs\figures\w8a16_swiglu_schedule_detail.svg
```

Regenerate the attention schedule:

```powershell
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\smollm2_attention\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
build-vs2026\Release\smollm2_attention_test.exe
python scripts\render_schedule_trace.py `
  logs\smollm2_attention\schedule.csv `
  docs\figures\smollm2_attention_schedule_detail.svg
```

Accumulator bars in detailed diagrams use purple for partial sums retained in
an MXM-local accumulator and red for final `stream+clear` operations.

## Repository Layout

- `include/ftlpu/core/`: hardware constants, streams, FP16, and ISA codec.
- `include/ftlpu/mem/`: homogeneous SRAM slices and MEM instruction pipelines.
- `include/ftlpu/mxm/`: supercells, arrays, control slices, GEMM datapath, and
  MXM-local accumulators.
- `include/ftlpu/vxm/`: ALU, lane, superlane, and central VXM slice.
- `include/ftlpu/sxm/`: Shift/Distribute/Transpose/Permute models.
- `include/ftlpu/system/`: ICU, stream topology, and full-chip integration.
- `tests/`: unit and offline whole-system numerical regressions.
- `examples/`: small trace-oriented demos.
- `scripts/`: schedule visualization tools.
- `docs/`: architecture notes, optimization studies, and diagrams.

## Documentation

- [JSON hardware configuration guide (Chinese)](docs/hardware_configuration.zh-CN.md)
- [Architecture reference](docs/architecture.md)
- [中文架构说明](docs/architecture.zh-CN.md)
- [MEM ICU N-D stream descriptor](docs/icu_mem_stream_nd.md)
- [MXM ICU N-D stream descriptor](docs/icu_mxm_stream_nd.md)
- [Attention pipeline optimization study](docs/attention_pipeline_optimization.md)
- [Editable topology diagram](docs/FTLPU.drawio)
