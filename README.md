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
| MEM | 44 slices per hemisphere, 88 ICU queues total |
| SRAM | 256 KiB per slice, 11 MiB per hemisphere, 22 MiB total |
| Accumulators | Two four-slice FP32 accumulator groups per hemisphere |
| MXM | Four 32 x 32 FP16 GEMM arrays, two per hemisphere |
| MXM weights | Two peer buffers per supercell, selected by `IW`/`Compute` |
| VXM | One central slice, 16 independently controlled ALUs per lane |
| SXM | One four-tile slice per hemisphere for Transpose/Permute |
| ICU | 88 MEM, 4 MXM load, 4 MXM compute, 16 VXM, and 4 SXM queues |

The fixed full-chip topology is:

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W <-> VXM <-> MEM.E <-> SXM.E <-> MXM0/MXM1
```

Each hemisphere uses local stream-register columns `sreg0..sreg12`.
`sreg0` is next to VXM, MEM occupies the eleven groups between
`sreg0..sreg11`, and SXM connects `sreg11` to the MXM boundary at `sreg12`.

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
cycle. Queue-local `NOP N` and `Repeat n,d` commands encode delays and regular
instruction trains. MEM repeats may also apply a signed address stride.

## Validated Workloads

| Test | Workload | Validation |
| --- | --- | --- |
| `w8a16_projection_test` | `[128,576] x [576,1536]` W8A16 projection | 196,608 FP32 outputs |
| `w8a16_swiglu_test` | gate/up projection plus SwiGLU | 196,608 FP16 outputs |
| `dual_hemisphere_w8a16_swiglu_test` | full gate/up, SwiGLU, and down FFN | `[128,576]` final FP16 output |
| `rmsnorm_test` | `[32,32]` FP16 RMSNorm | all stored FP16 outputs |
| `smollm2_attention_test` | Q/K/V, RoPE, QK, softmax, P x V, and `o_proj` | `[128,576]` attention output |
| `sxm_mem_transpose_test` | continuous MEM -> SXM -> MEM FP16 transpose | four 32 x 32 matrices |

The full FFN uses all four MXMs and currently schedules 90,817 cycles. Its final
gate/up reduction streams accumulator results directly into the shared VXM
SwiGLU pipeline. The complete SmolLM2 attention workload uses sequence length
128, hidden size 576, 9 query heads, 3 KV heads, and head dimension 64; its
validated schedule is 81,273 cycles.

## Build

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

## Schedule Diagrams

- [W8A16 projection pipeline](docs/w8a16_projection_pipeline.svg)
- [Full FFN pipeline](docs/w8a16_swiglu_pipeline.svg)
- [Full FFN detailed ICU schedule](docs/w8a16_swiglu_schedule_detail.svg)
- [SmolLM2 attention pipeline](docs/smollm2_attention_pipeline.svg)
- [Attention optimization comparison](docs/smollm2_attention_pipeline_optimization.svg)
- [Attention detailed ICU schedule](docs/smollm2_attention_schedule_detail.svg)

Regenerate the FFN schedule:

```powershell
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\w8a16_swiglu\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
build-vs2026\Release\dual_hemisphere_w8a16_swiglu_test.exe
python scripts\render_swiglu_schedule_trace.py `
  logs\w8a16_swiglu\schedule.csv `
  docs\w8a16_swiglu_schedule_detail.svg
```

Regenerate the attention schedule:

```powershell
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\smollm2_attention\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
build-vs2026\Release\smollm2_attention_test.exe
python scripts\render_schedule_trace.py `
  logs\smollm2_attention\schedule.csv `
  docs\smollm2_attention_schedule_detail.svg
```

Accumulator bars in detailed diagrams use purple for partial sums retained in
SRAM and red for final `stream+clear` operations.

## Repository Layout

- `include/ftlpu/core/`: hardware constants, streams, FP16, and ISA codec.
- `include/ftlpu/mem/`: SRAM, MEM instruction pipelines, and accumulators.
- `include/ftlpu/mxm/`: supercells, arrays, control slices, and GEMM datapath.
- `include/ftlpu/vxm/`: ALU, lane, superlane, and central VXM slice.
- `include/ftlpu/sxm/`: Shift/Distribute/Transpose/Permute models.
- `include/ftlpu/system/`: ICU, stream topology, and full-chip integration.
- `tests/`: unit and offline whole-system numerical regressions.
- `examples/`: small trace-oriented demos.
- `scripts/`: schedule visualization tools.
- `docs/`: architecture notes, optimization studies, and diagrams.

## Documentation

- [Architecture reference](docs/architecture.md)
- [中文架构说明](docs/architecture.zh-CN.md)
- [Attention pipeline optimization study](docs/attention_pipeline_optimization.md)
- [Editable topology diagram](docs/FTLPU.drawio)
