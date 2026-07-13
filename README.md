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

- `MEM`: 44 slice columns, 20 tile rows, 16 lanes per tile, 32 east streams and
  32 west streams per lane. Each stream register is one byte wide. Each
  tile-local SRAM has two banks, and each bank is 4096 x 16 bytes.
- `MXM`: two east-side MXM units. Each unit is a 20 x 20 array of 16 x 16
  supercells. Each supercell has two peer weight buffers; `IW` selects which
  buffer to fill and `Compute` selects which buffer to use for a 320 x 320 int8
  GEMM datapath with int32 accumulation.
- `VXM`: one west-side VXM slice with 20 superlanes/tiles. Each superlane has
  16 lanes, and each lane has 16 ALU issue queues.
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
All MEM, MXM, MXM-output, and VXM instructions are generated offline and loaded
into the ICU before cycle 0. The runtime loop only advances clocks and bridges
data. This is the shape intended for a future compiler backend.

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

On Windows with Visual Studio generator:

```powershell
cmake -S . -B build-vs2019
cmake --build build-vs2019 --config Debug
ctest --test-dir build-vs2019 -C Debug --output-on-failure
```

With a single-config generator:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Common Tests

Run the offline ICU FFN test:

```powershell
ctest --test-dir build-vs2019 -C Debug -R mem_dual_mxm_swiglu_offline_icu --output-on-failure
```

Run the VXM tests:

```powershell
ctest --test-dir build-vs2019 -C Debug -R "vxm_alu|vxm_lane|vxm_superlane|vxm_slice" --output-on-failure
```

## Logs and Diagrams

Integration tests write logs under the build directory:

- `build-vs2019/logs/mem_mxm/`
- `build-vs2019/logs/mem_dual_mxm_swiglu_offline_icu/`
- `build-vs2019/logs/mem_dual_mxm_swiglu_early_compute_icu/`

The FFN tests generate four functional-unit logs:

- `icu.log`
- `mem.log`
- `mxm.log`
- `vxm.log`

They also generate a pipeline diagram:

- `build-vs2019/logs/mem_dual_mxm_swiglu_offline_icu/pipeline.svg`

The diagram separates `MEM W read`, `MEM A read`, `MEM write`, `MXM0 load`,
`MXM0 compute`, `MXM1 load`, `MXM1 compute`, and `VXM` rows. There is no
separate `LW` phase; `IW` fills a selected buffer and `Compute` names the buffer
to consume.

## Demo Executables

After building, demos are available under the build output directory. Examples:

```powershell
.\build-vs2019\Debug\tile_array_trace_demo.exe tile_array_trace.log
.\build-vs2019\Debug\vector_roundtrip_demo.exe vector_roundtrip.log
.\build-vs2019\Debug\mxm_control_trace_demo.exe mxm_control_trace.log
.\build-vs2019\Debug\mem_mxm_trace_demo.exe mem_mxm_mem.log mem_mxm_mxm.log
.\build-vs2019\Debug\vxm_lane_trace_demo.exe vxm_lane_trace.log
```

## More Documentation

See [docs/architecture.md](docs/architecture.md) for the current architecture,
timing model, instruction queues, FFN workload, data layout, generated logs, and
known limitations.
