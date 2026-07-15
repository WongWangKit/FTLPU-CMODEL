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
- One modeled hemisphere: 44 MEM/SRAM slice columns.
- Public two-hemisphere total: 88 MEM/SRAM slices.
- Each slice column has 20 tile-local SRAM blocks.
- Each tile-local SRAM block has two banks.
- Each bank is 4096 words x 16 bytes.
- A complete slice is 20 x 2 x 4096 x 16 bytes = 2.5 MiB.

Software-visible MEM addresses follow the public-style layout:

```text
[39:24] TSP chip
[23]    hemisphere, East=1 and West=0
[22:17] slice number 0..43
[16]    bank
[15:4]  4096-word offset within the selected bank
[3:0]   byte offset within the 16-byte SRAM word
```

The current model instantiates one hemisphere. A MEM slice receives one
instruction stream; for `Read` and `Write`, the encoded hardware address field
is the 13-bit slice-local word address copied from software address bits
`[16:4]`. `set_sram_byte` and `sram_byte` keep byte-level access for tests and
initialization, while MEM `Read` and `Write` require 16-byte alignment and move
one full SRAM word per cycle.

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
- Each MEM slice column is modeled as a single instruction port. A slice cannot
  issue a `Read` and a `Write` in the same cycle; schedules that need both must
  place them on different cycles.
- A `Read(address, stream)` reads one aligned 16-byte SRAM word from the bank
  selected by address bit 16, one byte per lane, and writes those bytes into the
  selected stream at the stream register adjacent to that MEM slice.
- A `Write(address, stream)` consumes 16 bytes from the selected stream, one byte
  per lane, and writes one aligned SRAM word into the bank selected by address
  bit 16.
- `Gather` and `Scatter` are represented in the instruction enum, but the current
  tests focus on `Read` and `Write`.

Per cycle and per tile, one MEM instruction can move 16 bytes across the 16 lanes
for a single stream ID.

## MXM Model

The MXM model lives under `include/ftlpu/mxm/`.

Current components:

- `MxmSupercell`: one 16 x 16 int8 weight block.
- `MxmArray`: a 20 x 20 grid of supercells.
- `MxmControlSlice`: south-to-north control pipeline for `IW` and `Compute`.
- `Mxm`: wrapper containing the array, its control slice, and the datapath
  state for activation flow, int32 accumulation, and output stream injection.

The system contains two MXMs on the east side of MEM.

### Weight Loading

Each supercell has two peer weight buffers. `IW(buffer)` injects one 16 x 16
weight block into the west side of the selected row buffer. On every valid `IW`
cycle for that row, the selected buffer shifts one column to the east and the
new block enters column 0. To end with column 0..19 in the expected order, MEM
reads weight column blocks in reverse order: column block 19 first, then 18,
down to 0. There is no separate `LW` commit instruction:
`Compute(buffer, activation_stream, output_stream)` directly selects which
buffer supplies weights for that activation token.

The weight bytes for `IW` arrive from MEM at the east handoff stream register.
Each MXM uses 16 streams per cycle:

- MXM 0 consumes streams `E0..E15`.
- MXM 1 consumes streams `E16..E31`.

For a full 320 x 320 weight matrix, there are 20 column blocks. The current
tests issue 20 continuous `IW` pulses into one buffer while the other buffer can
still be used by in-flight compute. The two weight buffers have independent row
shift state, so loading one buffer does not disturb the other.

### Compute

`Compute(buffer, activation_stream, output_stream)` is a one-cycle pulse. The
ICU emits one `Compute` instruction per cycle for the active compute window.
The selected buffer id and output stream base are carried with the activation
event as it moves across the MXM, so later `IW` commands can overwrite the
other buffer without changing in-flight work.

The system-owned MXM runtime models activation flow:

- Activations enter from MEM into tile rows with a one-cycle south-to-north skew.
- Each active supercell consumes one 16-byte activation vector.
- The supercell computes 16 dot products against its 16 local weight columns.
- Activations move east across supercell columns.
- Partial sums accumulate into int32 result columns.

The runtime is owned by `TspSliceSystem`. ICU `Compute` pulses consume MEM east
handoff streams through the MXM datapath. When the contribution from tile row 19
completes a result column block, the MXM automatically injects the int32 result
bytes onto the MEM west streams named by the `Compute` instruction.

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

VXM can also cast fp32 values to fp16 stream output. The fp16 path writes the
IEEE half-precision bit pattern as two little-endian byte streams, while the
legacy int8 path still writes one byte stream.

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
- 2 MXM load queues, one per MXM.
- 2 MXM compute queues, one per MXM.
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

The MEM instruction address field is not the full software address. It encodes
only the slice-local SRAM word address: bit 12 is the bank and bits 11:0 are the
4096-word offset. The low software byte offset must be zero for `Read`/`Write`.

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
control slices. MXM compute and output are owned by the `Mxm` datapath inside
`TspSliceSystem`; tests no longer use a separate software GEMM engine.

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

### Offline ICU Test

`mem_dual_mxm_swiglu_offline_icu_test` uses the same source file with
`FTLPU_OFFLINE_ICU_FFN_TEST` enabled.

This test builds an `OfflineIcuProgram` before cycle 0. The program contains all
MEM, MXM, and VXM ALU instructions. MXM result stream placement is encoded in
the `Compute` instructions. It then loads those instructions into the ICU queues
once and starts ticking.

This is the intended shape for a future compiler:

```text
high-level workload -> compiler/scheduler -> OfflineIcuProgram -> ICU queues
```

### Attention Softmax

`attention_projection_test` implements stable softmax as three ICU-scheduled
VXM passes. MXM score output flows directly west through the MEM stream
registers into VXM; it is not first written to SRAM. Each VXM lane represents
one query and consecutive cycles represent key positions, so reductions happen
over time without a physical transpose:

```text
pass 1: Cast -> Multiply -> Max(self feedback)
pass 2: Subtract(max) -> Exp -> Add(self feedback)
pass 3: Divide(sum) -> Multiply(127) -> Cast(Int8)
```

The first `Max` instruction uses negative infinity as its seed and the first
`Add` uses zero. Their remaining 159 instructions read the same ALU's previous
cycle output. Scaled scores, exponentials, final maxima, and final sums are
moved through MEM by ICU `Read`/`Write` instructions. Test-side code only reads
the completed SRAM state and computes post-run golden values; it does not feed
maxima or sums into VXM.

The current offline test only loads the ICU queues, initializes external MEM
contents, ticks the system datapaths, and checks final MEM contents.

The test also exercises the MXM two-buffer path. `IW(buffer)` fills one selected
buffer through the per-row right-shift path while
`Compute(buffer, activation_stream, output_stream)` continues to use the other.
MXM control has separate load and compute queues, so the scheduler can overlap
next-weight transfer with current compute. The second gate/up pass and the down
pass use this ping-pong schedule:

- MXM0 starts `IW` immediately when the previous GEMM starts. Shared activation
  traffic uses alternate stream IDs during the first few rows so it does not
  collide with `E0..E15`.
- MXM1 starts `IW` immediately after MXM0's 20-cycle `IW` window. The current
  FFN schedule uses `MXM0: cycles 38..57` and `MXM1: cycles 58..77` for the
  second gate/up pass.
- In the early-compute variant, the second gate/up GEMM starts as soon as the
  compute queue is free. The down weights are loaded in two segments: MXM1 can
  start early after stream conflicts clear, while MXM0 waits until the `E0`
  activation stream is no longer needed. Because MEM is single-port, the down
  GEMM still waits until the second SwiGLU writeback has finished before reading
  hidden activations from the same MEM slice.
- `Compute` names the buffer to consume and the MEM west stream base for int32
  result output; no `LW`, active-weight commit, or separate MXM output
  instruction is issued.

`mxm.log` includes per-phase and total MXM performance counters:

```text
offline_gate_up_p0_mxm0 perf cycles=160 active_cycles=160 ...
offline_total_mxm0 perf cycles=614 active_cycles=480 ...
```

The per-phase utilization uses the tile0 compute window. The total utilization
also uses tile0 scheduling time and includes weight-load, writeback, and wait
cycles, so it is the number to watch when changing the scheduler.

## Logs

The FFN integration tests skip log generation by default. Set
`FTLPU_FFN_LOG=1` when trace files or the pipeline diagram are needed.
When enabled, the integration tests write logs under the build directory.

Offline ICU FFN:

```text
build-vs2019/logs/mem_dual_mxm_swiglu_offline_icu/
```

Each directory contains:

- `icu.log`: queue depths and dispatched instructions.
- `mem.log`: MEM stream/register/SRAM activity.
- `mxm.log`: MXM load/compute state.
- `vxm.log`: VXM ALU state for tile 0.
- `pipeline.svg`: phase timeline with separate MEM weight-read, MEM
  activation-read, MEM write, MXM load, MXM compute, and VXM rows.

The offline ICU log prints the key phase starts, for example:

```text
offline ICU FFN program loaded before cycle 0
  schedule=baseline ...
  load0.mxm0_iw=... gemm0=... gemm0_output=...
```

The early-compute variant currently prints:

```text
offline ICU FFN program loaded before cycle 0
  schedule=early_mxm_compute p1 compute starts as soon as the compute queue is free
  load0.mxm0_iw=18 load0.mxm1_iw=18 load0.done=38 gemm0=38 gemm0_output=38 load1.mxm1_iw=58 load1.mxm0_iw=38 load1.done=78 gemm1=198 gemm1_output=236 down_load.mxm1_iw=256 down_load.mxm0_iw=367 down_load.done=387 down_gemm=450 down_output=450
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
- Each pass reads 20 column blocks in reverse order, because `IW` shifts each
  row's selected weight buffer eastward.

SwiGLU hidden output:

- Pass 0 stored in MEM column `40`.
- Pass 1 stored in MEM column `41`.

Final output:

- Stored in MEM column `42`.

## Known Limitations

- The model is not bit-accurate to any private Groq ISA.
- `Gather` and `Scatter` are not the focus of current tests.
- The current VXM op list is intentionally small and contains only the ops needed
  for the current FFN/SwiGLU experiments plus simple scalar primitives.
- The compiler does not exist yet. `OfflineIcuProgram` is a prototype container
  for what the compiler should emit.

## Suggested Next Steps

Near-term engineering work:

- Tighten the MXM runtime timing against more detailed hardware pipeline
  assumptions.
- Turn `OfflineIcuProgram` into a reusable module instead of a test-local helper.
- Add a textual or binary program dump/load format using the ISA codec.
- Add a simple compiler pass that emits MEM Read/Write, MXM IW/Compute, and VXM
  ALU timelines for the FFN workload.
- Add resource-conflict diagnostics for stream-register and queue collisions.
