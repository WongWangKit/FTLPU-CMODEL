# FTLPU-CMODEL Architecture Notes

This document describes the current C++ model in this repository. It is written
as an implementation guide for future compiler and scheduler work.

## Design Goal

The model is a cycle-oriented playground for an FTLPU/TSP-like architecture. It
focuses on three ideas:

- Streams are explicit. Data moves through named eastward and westward streams,
  with byte-wide stream registers.
- Instructions are explicit. MEM, MXM, and VXM work is driven by instruction
  queues rather than direct calls into individual tiles.
- Timing matters. Tests check whether instructions and data arrive in the same
  cycle at the functional unit that consumes them.

This is not a bit-accurate Groq LPU/TSP model. The public material is used as
architectural inspiration, while the exact instruction encoding and timing rules
here are local to this CModel.

## Hardware Shape

The modeled slice uses the constants in `include/ftlpu/core/hardware_params.hpp`.

Current topology:

- 20 tile rows.
- 44 MEM slice columns.
- 4 MEM slices per stream-register group.
- 11 MEM slice groups.
- 12 stream-register columns, including both boundaries.
- 16 lanes per tile/superlane.
- 64 streams per lane: 32 eastward and 32 westward.
- 1 byte per stream register.
- One modeled SRAM block per MEM slice column.
- One modeled hemisphere: 44 SRAM blocks.
- Public two-hemisphere total: 88 SRAM blocks.
- 320-byte physical vector width.
- 8192 vector words per SRAM block.

The SRAM arrays in `TileArrayModel` are byte-addressable for simplicity, while
the public SRAM block math is still tracked in the hardware parameters.

## Stream Direction Conventions

The model treats streams as lane-local channels:

- `E0..E31`: eastward streams.
- `W0..W31`: westward streams.
- Combined stream IDs `0..31` mean east streams.
- Combined stream IDs `32..63` mean west streams.

For MEM instructions, the `stream` field uses the combined `0..63` numbering.
For VXM operands, `StreamInt32(base)` consumes four consecutive byte streams and
packs them little-endian into one int32 operand.

## MEM Model

The MEM tile-array model is in `include/ftlpu/mem/tile_array.hpp`.

Important behavior:

- There are 44 independent MEM instruction queues, one per slice column.
- Instructions enter at tile 0 and move north one tile per cycle.
- A `Read(address, stream)` reads 16 bytes from the tile SRAM, one byte per lane,
  and writes them into the selected stream at the stream register adjacent to
  that MEM slice.
- A `Write(address, stream)` consumes 16 bytes from the selected stream, one byte
  per lane, and writes them into SRAM.
- `Gather` and `Scatter` are represented in the instruction enum, but the current
  tests focus on `Read` and `Write`.

Per cycle and per tile, one MEM instruction can move 16 bytes across the 16 lanes
for a single stream ID.

## MXM Model

The MXM model lives under `include/ftlpu/mxm/`.

Current components:

- `MxmSupercell`: one 16 x 16 int8 weight block.
- `MxmArray`: a 20 x 20 grid of supercells.
- `MxmControlSlice`: south-to-north control pipeline for `IW`, `Compute`, and
  `Output`.
- `MxmGemmEngine`: execution helper for 320 x 320 int8 GEMM with int32
  accumulation.
- `Mxm`: wrapper containing an array and its control slice.

The system contains two MXMs on the east side of MEM.

### Weight Loading

`IW(column_block)` loads one 16 x 16 supercell weight matrix per tile row. The
weight bytes arrive from MEM at the east handoff stream register. Each MXM uses
16 streams per cycle:

- MXM 0 consumes streams `E0..E15`.
- MXM 1 consumes streams `E16..E31`.

For a full 320 x 320 weight matrix, there are 20 column blocks, so loading one
MXM plane takes 20 `IW` cycles after data reaches the MXM boundary.

### Compute

`Compute` is a one-cycle pulse. The ICU emits one `Compute` instruction per cycle
for the active compute window.

The GEMM engine models activation flow:

- Activations enter from MEM into tile rows with a one-cycle south-to-north skew.
- Each active supercell consumes one 16-byte activation vector.
- The supercell computes 16 dot products against its 16 local weight columns.
- Activations move east across supercell columns.
- Partial sums accumulate into int32 result columns.

The engine is currently driven by integration tests rather than being fully
embedded inside `TspSliceSystem::tick()`. This is a known integration boundary:
the ICU supplies `Compute` and `Output` pulses, but the test still owns the
`MxmGemmEngine` object that turns those pulses into numeric results.

## VXM Model

The VXM model lives under `include/ftlpu/vxm/`.

Hierarchy:

- `VxmAlu`: vector helper for supported ALU ops.
- `VxmLane`: one lane with 16 ALUs and one instruction queue per ALU.
- `VxmSuperlane`: 16 lanes.
- `VxmSlice`: 20 superlanes/tiles with south-to-north instruction flow.

Supported ALU opcodes currently include:

- `Pass`
- `Add`
- `Subtract`
- `Multiply`
- `Divide`
- `Negate`
- `Abs`
- `Min`
- `Max`
- `Clamp`
- `Square`
- `Sqrt`
- `Exp`
- `Log`
- `Relu`
- `Cast`

Quantization and dequantization are not special ALU opcodes. They are modeled
with primitive ALU operations:

- Cast int32 stream data to fp32.
- Multiply by a scale to dequantize.
- Multiply by reciprocal output scale.
- Add zero point if needed.
- Cast to int8.

SwiGLU is built from primitive ops:

```text
gate_i32 -> cast fp32 -> multiply gate_scale
up_i32   -> cast fp32 -> multiply up_scale
sigmoid(gate) = 1 / (1 + exp(-gate))
hidden = gate * sigmoid(gate) * up
hidden_i8 = cast_int8(hidden / output_scale + zero_point)
```

## ICU Model

The ICU is implemented in `include/ftlpu/system/icu.hpp`.

Queue counts:

- 44 MEM queues, one per MEM slice column.
- 2 MXM queues, one per MXM.
- 2 MXM output queues, one per MXM output-control path.
- 16 VXM queues, one per ALU.

The ICU supports per-queue commands:

- `Instruction`: dispatch one MEM/MXM/VXM instruction.
- `NOP N`: delay only the current queue by `N` cycles.
- `Repeat n,d`: repeat the previous instruction `n` times, with `d` cycles
  between dispatches.
- MEM `Repeat` additionally supports an address stride.

This is important for compiler work: a compiler can build each queue as a static
timeline and use NOP/Repeat to compact long regular schedules.

## Instruction Encoding

The model ISA encoder is in `include/ftlpu/core/instruction_codec.hpp`.

Current encoding:

- MEM instruction: 32 bits.
- MXM control instruction: 32 bits.
- VXM ALU instruction: 3 x 32-bit words.
- ICU queue command: 32 bits.

This is a compact FTLPU CModel encoding, not a Groq hardware binary encoding.
It deliberately rejects model-only metadata such as VXM operand scale and output
zero point, because those should be synthesized with explicit ALU instructions.

The codec is covered by `tests/core/instruction_codec_test.cpp`.

## Whole-System Topology

`TspSliceSystem` fixes the current local topology:

```text
VXM <-> west MEM edge ... 44 MEM slices ... east MEM edge <-> MXM0/MXM1
```

Important paths:

- MEM east streams feed MXM weight and activation inputs.
- MXM int32 outputs are written back into MEM west streams near the east edge.
- Those west streams travel through MEM to the west edge.
- VXM consumes selected streams according to its ALU operands.
- VXM output streams are injected back into MEM and can be written into SRAM.

The system tick handles ICU dispatch, MEM tick, VXM stream bridge, and MXM
control slices. The current GEMM numeric engine is still test-driven, as noted
above.

## FFN Integration Tests

The main integration file is:

```text
tests/integration/mem_dual_mxm_swiglu_test.cpp
```

It builds a 160 x 320 activation matrix and three 640 x 320 weight matrices:

- gate projection weights
- up projection weights
- down projection weights

The test computes:

```text
hidden = swiglu(A * W_gate, A * W_up)
out = quant((hidden_left * W_down_left) + (hidden_right * W_down_right))
```

Shapes:

- Activation: `160 x 320`, int8.
- Gate weight: effectively `320 x 640`, loaded in two 320-column passes.
- Up weight: effectively `320 x 640`, loaded in two 320-column passes.
- Hidden: `160 x 640`, int8.
- Down weight: `640 x 320`, split across two MXMs.
- Final output: `160 x 320`, int8.

The generated data uses non-trivial symmetric quantization scales so the output
is not mostly zero. The test compares SRAM contents against golden reference
data computed in the test.

### Online-Style Test

`mem_dual_mxm_swiglu_test` schedules some work from the test harness while the
simulation runs. It is useful for debugging phase timing and generating detailed
logs.

### Offline ICU Test

`mem_dual_mxm_swiglu_offline_icu_test` uses the same source file with
`FTLPU_OFFLINE_ICU_FFN_TEST` enabled.

This test builds an `OfflineIcuProgram` before cycle 0. The program contains all
MEM, MXM, MXM output, and VXM ALU instructions. It then loads those instructions
into the ICU queues once and starts ticking.

This is the intended shape for a future compiler:

```text
high-level workload -> compiler/scheduler -> OfflineIcuProgram -> ICU queues
```

The current offline test still has a small runtime bridge for the standalone
`MxmGemmEngine`, because numeric GEMM execution is not yet fully embedded in the
system tick.

## Logs

The integration tests write logs under the build directory.

Online FFN:

```text
build-vs2019/logs/mem_dual_mxm_swiglu/
```

Offline ICU FFN:

```text
build-vs2019/logs/mem_dual_mxm_swiglu_offline_icu/
```

Each directory contains:

- `icu.log`: queue depths and dispatched instructions.
- `mem.log`: MEM stream/register/SRAM activity.
- `mxm.log`: MXM load/compute state.
- `vxm.log`: VXM ALU state for tile 0.
- `pipeline.svg`: phase timeline with separate MEM read, MEM write, MXM, and
  VXM rows.

The offline ICU log prints the key phase starts, for example:

```text
offline ICU FFN program loaded before cycle 0
  load0=0 gemm0=38 load1=239 gemm1=277 down_load=478 down_gemm=516
```

## Data Layout Summary

The FFN test stages data directly into SRAM before cycle 0. After that, movement
between units is through ICU-controlled MEM Read/Write instructions and streams.

Activation:

- Stored in MEM column `32`.
- Address for row `r`, lane `l`: `r * 16 + l`.
- Tile selects the 16-element block of the K dimension.

Weights:

- Staged across MEM columns `0..31`.
- MXM0 reads weight streams from columns `0..15`.
- MXM1 reads weight streams from columns `16..31`.
- Each pass reads 20 column blocks.

SwiGLU hidden output:

- Pass 0 stored in MEM column `40`.
- Pass 1 stored in MEM column `41`.

Final output:

- Stored in MEM column `42`.

## Known Limitations

- The model is not bit-accurate to any private Groq ISA.
- `Gather` and `Scatter` are not the focus of current tests.
- `MxmGemmEngine` is still owned by tests rather than fully integrated into
  `TspSliceSystem::tick()`.
- The current VXM op list is intentionally small and contains only the ops needed
  for the current FFN/SwiGLU experiments plus simple scalar primitives.
- The compiler does not exist yet. `OfflineIcuProgram` is a prototype container
  for what the compiler should emit.

## Suggested Next Steps

Near-term engineering work:

- Move MXM GEMM execution fully into `TspSliceSystem`.
- Turn `OfflineIcuProgram` into a reusable module instead of a test-local helper.
- Add a textual or binary program dump/load format using the ISA codec.
- Add a simple compiler pass that emits MEM Read/Write, MXM IW/Compute/Output,
  and VXM ALU timelines for the FFN workload.
- Add resource-conflict diagnostics for stream-register and queue collisions.
