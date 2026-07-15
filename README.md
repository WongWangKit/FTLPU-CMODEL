# FTLPU-CMODEL

FTLPU-CMODEL is a cycle-oriented C++17 model for experimenting with an
FTLPU/TSP-style dataflow: MEM streams move through stream registers, MXM consumes
streamed int8 vectors for matrix multiply, VXM executes ALU instruction queues,
and an ICU dispatches instructions into each functional block.

The model is inspired by public Groq TSP/LPU descriptions, but it is not a
bit-accurate Groq implementation. The current goal is to build a useful compiler
and scheduling playground where instructions, stream timing, and functional-unit
handoffs can be tested cycle by cycle.

## Current Status

The repository currently models:

- Lightweight vector timing: one logical stream value is a 320-byte vector
  spanning 20 tiles x 16 lanes. Instructions execute on all 20 tile rows in
  the same cycle; the model does not simulate south-to-north instruction
  propagation or per-tile data skew. East/west stream-register boundaries
  remain cycle accurate and move a vector by exactly one boundary per cycle.

- `MEM`: 44 slice columns, 20 tile rows, 16 lanes per tile, 32 east streams and
  32 west streams per lane. Each stream register is one byte wide. Each
  tile-local SRAM has two banks, and each bank is 4096 x 16 bytes.
- `MXM`: two east-side MXM units. Each unit is a 20 x 20 array of 16 x 16
  supercells. Each supercell has two peer weight buffers; `IW` selects which
  buffer receives the right-shifting weight stream and `Compute` selects both
  the weight buffer and output stream base for a 320 x 320 int8 GEMM datapath
  with int32 accumulation.
- `VXM`: one west-side VXM slice with 20 superlanes/tiles. Each superlane has
  16 lanes, and each lane has 16 ALU issue queues. ALU outputs can write int8
  or fp16 bytes onto streams.
- `ICU`: per-queue instruction dispatch with `NOP N` and `Repeat n,d`, including
  MEM address stride support.
- `TspSliceSystem`: fixed local topology with VXM on the west side of MEM and
  two MXMs on the east side of MEM.
- A compact model ISA codec for MEM, MXM, VXM ALU, and ICU queue commands.

The largest integration test models an FFN-like path:

1. Load gate/up weights from MEM into two MXMs.
2. Stream activations from MEM into both MXMs.
3. Route gate/up int32 outputs west through MEM streams into VXM.
4. Execute SwiGLU using ALU instructions.
5. Store the int8 hidden result back to MEM.
6. Load down-projection weights into the two MXMs.
7. Stream the hidden result through MXM.
8. Route the two int32 partial results into VXM for add + quant.
9. Store the final int8 result back to MEM and compare against golden data.

`mem_dual_mxm_swiglu_offline_icu_test` is the canonical FFN integration test.
All MEM, MXM, and VXM instructions are generated offline and loaded into the ICU
before cycle 0. MXM output is controlled by the `Compute` instruction stream.
The runtime loop only advances clocks and bridges data. This is the shape
intended for a future compiler backend.

An MXM activation still traverses 20 supercell columns. A 320-byte activation
enters on one `Compute` pulse, advances one supercell column per cycle, and
produces its complete 320-element int32 output on the twentieth pipeline cycle.

`attention_projection_test` is the first single-head attention-oriented test. It
initializes `seq_len=160`, `hidden=320` input and Wq/Wk matrices in MEM, loads
Wq and Wk into the two MXMs, streams X through both MXMs, sends Q/K int32 results
to VXM, requantizes to int8 with ALU `Multiply` + `Cast(Int8)`, stores Q/K int8
streams back to MEM, then loads Q into MXM0 and streams K to compute sampled
`K * Q^T` scores against golden data. Raw scores are not staged to MEM: the MXM
west-stream score output feeds VXM directly for softmax pass 1, which scales to
fp32 and stores that intermediate in MEM. Passes 2 and 3 reload fp32
intermediates through MEM streams to compute `exp(x - max)`, divide by the row
sum, scale, `Cast(Int8)`, and store the final attention probabilities in MEM.
The test writes an ICU dispatch trace to
`build-vs2026/logs/attention_projection/icu.log`.

## Repository Layout

- `include/ftlpu/core/`: hardware parameters, stream words, topology helpers,
  instruction pipeline primitives, and instruction encoding.
- `include/ftlpu/mem/`: MEM slice and full tile-array model.
- `include/ftlpu/mxm/`: MXM supercell, array, control slice, wrapper, and
  system-owned datapath state.
- `include/ftlpu/vxm/`: VXM ALU, lane, superlane, and slice models.
- `include/ftlpu/system/`: ICU and whole-slice system integration.
- `tests/core/`, `tests/mem/`, `tests/mxm/`, `tests/vxm/`: subsystem tests.
- `tests/integration/`: cross-unit tests for MEM/MXM/VXM/ICU flows.
- `examples/`: small trace-oriented demos.
- `docs/architecture.md`: detailed project notes.

## Build

On Windows with the Visual Studio 2026 installation used by this repository:

```powershell
$cmake = "D:\Apps\Microsoft Visual Studio\2026\community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = "D:\Apps\Microsoft Visual Studio\2026\community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $cmake -S . -B build-vs2026 -G "Visual Studio 18 2026" -A x64 -DCMAKE_GENERATOR_INSTANCE="D:/Apps/Microsoft Visual Studio/2026/community"
& $cmake --build build-vs2026 --config Release
& $ctest --test-dir build-vs2026 -C Release --output-on-failure
```

Use the CMake bundled with Visual Studio 2026. Older standalone CMake versions
may not recognize the `Visual Studio 18 2026` generator.

With a single-config generator:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Common Tests

Run the offline ICU FFN test:

```powershell
& $cmake --build build-vs2026 --config Release
& $ctest --test-dir build-vs2026 -C Release -R mem_dual_mxm_swiglu_offline_icu --output-on-failure
```

Debug builds retain assertions and are useful for subsystem tests, but the
large 320-wide integration workloads should normally be run in Release mode.

Run the VXM tests:

```powershell
& $ctest --test-dir build-vs2026 -C Debug -R "vxm_alu|vxm_lane|vxm_superlane|vxm_slice" --output-on-failure
```

Run the attention projection test:

```powershell
& $ctest --test-dir build-vs2026 -C Release -R attention_projection_test --output-on-failure
```

## Logs and Diagrams

FFN integration tests skip log generation by default so the long-running
workloads are not dominated by file I/O. Enable logs when debugging:

```powershell
$env:FTLPU_FFN_LOG = "1"
& $ctest --test-dir build-vs2026 -C Release -R mem_dual_mxm_swiglu_offline_icu --output-on-failure
Remove-Item Env:\FTLPU_FFN_LOG
```

When enabled, integration tests write logs under the build directory:

- `build-vs2026/logs/mem_mxm/`
- `build-vs2026/logs/mem_dual_mxm_swiglu_offline_icu/`
- `build-vs2026/logs/mem_dual_mxm_swiglu_early_compute_icu/`

The FFN tests generate four functional-unit logs:

- `icu.log`
- `mem.log`
- `mxm.log`
- `vxm.log`

They also generate a pipeline diagram:

- `build-vs2026/logs/mem_dual_mxm_swiglu_offline_icu/pipeline.svg`

The diagram separates `MEM W read`, `MEM A read`, `MEM write`, `MXM0 load`,
`MXM0 compute`, `MXM1 load`, `MXM1 compute`, and `VXM` rows. There is no
separate `LW` phase; `IW` fills a selected buffer and `Compute` names the buffer
to consume plus the output stream base.

## Demo Executables

After building, demos are available under the build output directory. Examples:

```powershell
.\build-vs2026\Release\tile_array_trace_demo.exe tile_array_trace.log
.\build-vs2026\Release\vector_roundtrip_demo.exe vector_roundtrip.log
.\build-vs2026\Release\mxm_control_trace_demo.exe mxm_control_trace.log
.\build-vs2026\Release\mem_mxm_trace_demo.exe mem_mxm_mem.log mem_mxm_mxm.log
.\build-vs2026\Release\vxm_lane_trace_demo.exe vxm_lane_trace.log
```

## More Documentation

See [docs/architecture.md](docs/architecture.md) for the current architecture,
timing model, instruction queues, FFN workload, data layout, generated logs, and
known limitations.
