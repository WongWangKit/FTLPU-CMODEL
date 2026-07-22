# FTLPU-CMODEL

[English](README.md) | [简体中文](README.zh-CN.md)

FTLPU-CMODEL is a cycle-oriented C++20 model for experimenting with an
FTLPU/TSP-style dataflow: MEM streams move through stream registers, MXM consumes
streamed FP16 vectors for matrix multiply, VXM executes 16-ALU lane programs,
SXM transposes or permutes eastward vectors, and an ICU dispatches instructions
into each functional block.

The model is inspired by public Groq TSP/LPU descriptions, but it is not a
bit-accurate Groq implementation. The current goal is to build a useful compiler
and scheduling playground where instructions, stream timing, and functional-unit
handoffs can be tested cycle by cycle.

## Current Status

The repository currently models:

- `MEM`: two mirrored hemispheres with 44 slice columns each (88 total), 4 tile
  rows, 8 lanes per tile, 32 east streams and
  32 west streams per lane. Each stream register is one byte wide. Each
  tile-local SRAM has two banks, and each bank is 4096 x 8 bytes. The eastmost
  two four-slice groups nearest MXM retain normal SRAM Read/Write behavior and
  additionally support `Accumulate(address, stream)`: each group forms one
  FP32 value and adds a streamed FP32 value. The instruction selects whether
  the sum is written back in place or emitted on the same four west streams.
- `MXM`: four units, two beyond each hemisphere. Each unit is a 4 x 4 array of 8 x 8
  supercells. Each supercell has two peer weight buffers; `IW` selects which
  buffer receives the right-shifting weight stream and `Compute` selects both
  the weight buffer and output stream base for a 32 x 32 FP16 GEMM datapath
  with FP32 accumulation. Stored INT8 weights use symmetric per-output-column
  scales and are dequantized and cast to FP16 by VXM before `IW`; each MXM
  consumes 16 byte streams for eight FP16 columns.
- `VXM`: one central VXM slice with 4 superlanes/tiles and 8 lanes per
  superlane. Every lane has 16 ALUs with independent ICU queues. ALU operands
  can come from INT8/FP16/FP32 streams, immediates, or prior-cycle ALU outputs;
  results can be retained or emitted to a selected stream and hemisphere.
  Offline programs synthesize dequant as eight multiply/cast pairs and SwiGLU
  as a pipelined multiply, sigmoid decomposition, and FP16 cast.
- `SXM`: one data-movement slice per hemisphere between MEM and MXM. Each consumes and
  produces the 32 east streams, supports 8 x 8 transpose and 32-lane
  `Permute`, and passively forwards data when its queues do not issue. Its
  FP16 weight path captures low/high byte planes together and emits eight
  transposed FP16 columns on 16 interleaved byte streams for MXM `IW`.
- `ICU`: per-queue instruction dispatch with `NOP N` and `Repeat n,d`, including
  MEM address stride support. It owns 88 MEM queues, 4 MXM load queues, 4 MXM
  compute queues, 16 VXM ALU queues, and independent Transpose/Permute queues
  for both SXMs.
- `TspSliceSystem`: full mirrored topology with VXM in the center. Each side has
  `MEM <-> SXM <-> MXM0/1` in local orientation and 13 stream-register columns.
  Global MXM IDs 0..1 and MEM queues 0..43 select East; MXM IDs 2..3 and MEM
  queues 44..87 select West. VXM instructions independently select their input
  and output hemispheres.
- A compact model ISA codec for MEM, MXM, VXM ALU, and ICU queue commands.

`w8a16_projection_test` is the canonical whole-system test. It computes
`[128,576] x [576,1536]` with per-output-column symmetric W8 quantization. Before
cycle 0, the test may initialize SRAM and result storage and enqueue the complete
program into ICU. During execution it calls only `TspSliceSystem::tick()`.
ICU-scheduled MEM reads feed INT8 weights west to VXM, VXM converts them to FP16,
ICU `IW` loads both MXMs, ICU MEM reads stream FP16 activations east, and ICU
`Compute` pulses produce FP32 partial sums. ICU MEM `Accumulate` instructions
combine all 18 K tiles in slices 36..43. MXM0 and MXM1 use separate four-slice
read-modify-write groups, so their compute/output windows remain parallel.

`w8a16_swiglu_test` extends that flow to a complete
`X[128,576] -> gate/up[128,1536] -> SwiGLU[128,1536]` workload. MXM0 and MXM1
produce gate/up partial sums, the two MEM accumulator groups combine all 18 K
tiles, ICU MEM Reads stream both FP32 operands west, and an ICU-scheduled VXM
ALU pipeline writes FP16 results back to MEM. The final 196,608 values are checked.

`rmsnorm_test` is a full-system `[32,32]` FP16 RMSNorm regression. VXM squares
`x`; MXM0, loaded with FP16 ones, and the MEM accumulator produce replicated
FP32 row sums. VXM then calculates `1 / sqrt(mean(x^2) + epsilon)`, explicitly
delays `x` and `gamma` through the pipeline, and writes the FP16 result back to
MEM. Every output is checked against an FP16-aware scalar golden reference.

`dual_hemisphere_w8a16_swiglu_test` is the full-chip regression with the same
`X[128,576]` and gate/up `[576,1536]` dimensions. Adjacent 32-column output
blocks alternate between hemispheres: East MXM0/1 and West MXM2/3 work in the
same compute windows. The central VXM interleaves East/West Swish instructions
and writes each FP16 half back to its own MEM hemisphere. All projection
accumulators and 196,608 SwiGLU outputs are checked against software golden data.

`smollm2_attention_test` executes the SmolLM2 attention path through `o_proj`.
It treats the input
as an already normalized FP16 `X[128,576]`, applies symmetric per-output-column
W8 quantization to Q/K/V weights, and computes `Q[128,576]` plus
`K/V[128,192]` across all four MXMs. Each head keeps its low and high 32-value
halves in the two accumulator groups of one hemisphere. On the final K tile,
`Accumulate(..., Stream)` sends both FP32 halves directly to VXM. ICU-scheduled
VXM ALUs apply non-interleaved RoPE to Q/K, cast V to FP16, and write all outputs
back to ordinary MEM. East and West use separate VXM ALU groups, so all four
MXMs remain synchronized through the final reduction. RoPE Q is stored directly
in an IW-ready layout: its two
32-value reduction blocks occupy 32 MEM slices. For GQA score computation,
three Q heads are assigned to each KV head's hemisphere. Q bypasses SXM. Each
hemisphere uses both local MXMs for independent query blocks; K is replicated
into the released RoPE-table slices so each MXM consumes a separate stream
pair. Each MXM uses its two weight buffers for the two 32-value Q reductions
and writes its complete score tile into its own accumulator group. Q token
blocks become WS columns, giving VXM one fixed query per lane while keys advance
in time.
Three ICU-scheduled VXM passes compute scaled max, exponent/sum, and normalized
FP16 probabilities. Softmax P3 writes the result directly into a packed
16-stream layout; SXM immediately converts it into the persistent diagonal MEM
layout used by the later 2-stream P replay. V projection writes its FP16 result directly into two
16-stream packed MEM layouts; SXM captures one 8 x 8 FP16 block per beat and
sends each transposed/permuted output directly to the explicit
`IW(buffer, column)` destination. SXM also transposes each 32 x 32
probability block immediately after softmax, so P x V only replays one FP16
activation vector per cycle on two streams for `softmax(QK^T) x V`.
A final
W8A16 `576 x 576` projection produces `o_proj[128,576]`. Q/K/V, softmax,
all GQA contexts, and the final FP16 output are checked against scalar golden
models. Causal masking and KV-cache placement remain subsequent stages.

Whole-system tests follow a strict offline contract. Before cycle 0 they may
initialize MEM SRAM and enqueue a complete ICU
program. After ticking starts, they cannot access MEM, MXM, VXM, or SXM control
interfaces directly. `TspSliceSystem` intentionally exposes only explicit
initialization/result methods, `icu()`, `tick()`, and `cycle()`.

## Repository Layout

- `include/ftlpu/core/`: hardware parameters, stream words, topology helpers,
  instruction pipeline primitives, and instruction encoding.
- `include/ftlpu/mem/`: MEM slice and full tile-array model.
- `include/ftlpu/mxm/`: MXM supercell, array, control slice, wrapper, and
  system-owned datapath state.
- `include/ftlpu/vxm/`: VXM ALU, lane, superlane, and slice models.
- `include/ftlpu/sxm/`: SXM shift, distribute, permute, transpose, and integrated
  stream-facing slice models.
- `include/ftlpu/system/`: ICU and whole-slice system integration.
- `tests/core/`, `tests/mem/`, `tests/mxm/`, `tests/vxm/`: subsystem tests.
- `tests/integration/`: cross-unit tests for MEM/MXM/VXM/ICU flows.
- `examples/`: small trace-oriented demos.
- `docs/architecture.md`: detailed project notes.

## Build

On Windows with Visual Studio generator:

```powershell
cmake -S . -B build-vs2026
cmake --build build-vs2026 --config Release
ctest --test-dir build-vs2026 -C Release --output-on-failure
```

With a single-config generator:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Common Tests

Run the offline ICU W8A16 projection:

```powershell
ctest --test-dir build-vs2026 -C Release -R w8a16_projection_test --output-on-failure
```

Run the complete W8A16 SwiGLU workload:

```powershell
ctest --test-dir build-vs2026 -C Release -R w8a16_swiglu_test --output-on-failure
ctest --test-dir build-vs2026 -C Release -R rmsnorm_test --output-on-failure
```

Run the VXM tests:

```powershell
ctest --test-dir build-vs2026 -C Release -R "vxm_dequant|vxm_swish|vxm_lane|vxm_superlane|vxm_slice" --output-on-failure
```

## Logs and Diagrams

Long whole-system workloads keep logging disabled by default so file I/O does
not dominate simulation time. Unit-level trace demos remain under `examples/`.
The current offline projection schedule is shown in
[docs/w8a16_projection_pipeline.svg](docs/w8a16_projection_pipeline.svg).
The complete dual-hemisphere gate/up projection and VXM SwiGLU schedule is shown
in [docs/w8a16_swiglu_pipeline.svg](docs/w8a16_swiglu_pipeline.svg).
The end-to-end SmolLM2 attention schedule, including QK, three-pass softmax,
P x V, and o_proj, is shown in
[docs/smollm2_attention_pipeline.svg](docs/smollm2_attention_pipeline.svg).
Measured bottlenecks, validated cycle reductions, and the next pipeline-overlap
steps are documented in
[docs/attention_pipeline_optimization.md](docs/attention_pipeline_optimization.md).
The baseline/validated schedule comparison is available as
[docs/smollm2_attention_pipeline_optimization.svg](docs/smollm2_attention_pipeline_optimization.svg).
Exact ICU queue windows for projection, QK/softmax, P x V, and o_proj are shown
in [docs/smollm2_attention_schedule_detail.svg](docs/smollm2_attention_schedule_detail.svg).

Regenerate the detailed schedule from the actual offline instruction queues:

```powershell
New-Item -ItemType Directory -Force logs\smollm2_attention | Out-Null
$env:FTLPU_SCHEDULE_TRACE = "$PWD\logs\smollm2_attention\schedule.csv"
$env:FTLPU_SCHEDULE_TRACE_ONLY = "1"
.\build-vs2026\Release\smollm2_attention_test.exe
python scripts\render_schedule_trace.py `
  logs\smollm2_attention\schedule.csv `
  docs\smollm2_attention_schedule_detail.svg
```

The renderer also accepts repeated `--window START:END:TITLE` arguments for
zooming into any custom cycle range. SVG bars contain hover text with the exact
queue, cycle interval, MEM slice/address/stream, and repeat information.

## Demo Executables

After building, demos are available under the build output directory. Examples:

```powershell
.\build-vs2026\Release\tile_array_trace_demo.exe tile_array_trace.log
.\build-vs2026\Release\vector_roundtrip_demo.exe vector_roundtrip.log
.\build-vs2026\Release\mxm_control_trace_demo.exe mxm_control_trace.log
.\build-vs2026\Release\mem_mxm_trace_demo.exe mem_mxm_mem.log mem_mxm_mxm.log
```

## More Documentation

See [docs/architecture.md](docs/architecture.md) for the current architecture,
timing model, instruction queues, offline workload contract, data layout, and
known limitations.
