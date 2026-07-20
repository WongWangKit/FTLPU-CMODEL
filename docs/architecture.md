# FTLPU-CMODEL Architecture Notes

[English](architecture.md) | [简体中文](architecture.zh-CN.md)

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

- 4 tile rows.
- 44 MEM slice columns.
- 4 MEM slices per stream-register group.
- 11 MEM slice groups.
- Groups 9 and 10, nearest MXM, keep their normal MEM/SRAM slices and each
  provide one FP32 partial-sum accumulator port.
- 13 system stream-register columns. MEM spans `sreg0..sreg11`; SXM sits
  between MEM's east boundary `sreg11` and the MXM boundary `sreg12`.
- 8 lanes per tile/superlane.
- 64 streams per lane: 32 eastward and 32 westward.
- 1 byte per stream register.
- Two modeled hemispheres: 44 MEM/SRAM slice columns each, 88 total.
- Each slice column has 4 tile-local SRAM blocks.
- Each tile-local SRAM block has two banks.
- Each bank is 4096 words x 8 bytes.
- A complete slice is 4 x 2 x 4096 x 8 bytes = 256 KiB.

Software-visible MEM addresses follow the public-style layout:

```text
[39:24] TSP chip
[23]    hemisphere, East=1 and West=0
[22:17] slice number 0..43
[16]    bank
[15:4]  4096-word offset within the selected bank
[3:0]   byte offset within the 16-byte SRAM word
```

The current model instantiates both hemispheres. A MEM slice receives one
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
VXM ALU instructions independently name stream operands and an optional output
stream. Offline programs combine multiple queues to consume eight W8 streams
and emit 16 FP16 byte streams in parallel.

## MEM Model

The MEM tile-array model is in `include/ftlpu/mem/tile_array.hpp`.

Important behavior:

- There are 88 independent MEM instruction queues, one per slice column in each hemisphere.
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
- `Accumulate(address, stream)` is issued on MEM queue 36 or 40. The selected
  four-slice group reads one FP32 value per lane, consumes four consecutive
  west streams, adds the values, and writes the result back in place.
- `Gather` and `Scatter` are represented in the instruction enum, but the current
  tests focus on `Read` and `Write`.

Per cycle and per tile, one MEM instruction can move 8 bytes across the 8 lanes
for a single stream ID.

### Eastmost Accumulator Groups

The two MEM groups nearest MXM remain fully SRAM-backed: slices 36..39 and
40..43 still support ordinary `Read` and `Write`. An `Accumulate` instruction
temporarily treats one group as the four byte planes of an FP32 SRAM vector.
Each group has an independent single-port read-modify-write path and occupies
its own four slices for that tile and cycle. Consumed streams do not continue
west. There are no persistent routes or separate accumulator result storage.

For a projection larger than one 32 x 32 MXM tile, the accumulator adds the
partial vector from every K tile. The W8A16 projection regression uses 18 K
tiles for `[128,576] x [576,1536]`, producing and checking 196,608 FP32 values.

## MXM Model

The MXM model lives under `include/ftlpu/mxm/`.

Current components:

- `MxmSupercell`: one 8 x 8 FP16 weight block.
- `MxmArray`: a 4 x 4 grid of supercells.
- `MxmControlSlice`: south-to-north control pipeline for `IW` and `Compute`.
- `Mxm`: wrapper containing the array, its control slice, and the datapath
  state for activation flow, FP32 accumulation, and output stream injection.

The system contains two MXMs per hemisphere, four in total.

### Weight Loading

Each supercell has two peer weight buffers. `IW(buffer)` injects one 8 x 8
weight block into the west side of the selected row buffer. On every valid `IW`
cycle for that row, the selected buffer shifts one column to the east and the
new block enters column 0. To end with column 0..3 in the expected order, MEM
reads weight column blocks in reverse order: column block 3 first, then 2,
down to 0. There is no separate `LW` commit instruction:
`Compute(buffer, activation_stream, output_stream)` directly selects which
buffer supplies weights for that activation token.

Weights are symmetrically quantized to INT8 with one scale per output column:
`scale[n] = max_k(abs(W[k][n])) / 127`, with zero point 0. Eight VXM multiply
ALU instructions consume the W8 streams and apply the column scales; eight
cast instructions then produce two little-endian byte streams per FP16 value.
The FP16 weight bytes for `IW`
then arrive from MEM at the east handoff stream register. Each MXM owns 16
physical streams, which carry eight FP16 weight columns per cycle.
Each MXM uses 16 streams per cycle:

- MXM 0 consumes streams `E0..E15`.
- MXM 1 consumes streams `E16..E31`.

For a full 32 x 32 weight matrix, there are 4 column blocks. The current
tests issue 4 continuous `IW` pulses into one buffer while the other buffer can
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
- Each active supercell consumes eight FP16 activations from two byte streams.
- The supercell computes eight dot products against its eight local FP16 weight columns.
- Activations move east across supercell columns.
- Partial sums accumulate into FP32 result columns.

The runtime is owned by `TspSliceSystem`. ICU `Compute` pulses consume MEM east
handoff streams through the MXM datapath. When the contribution from tile row 3
completes a result column block, the MXM automatically injects the FP32 result
bytes onto the MEM west streams named by the `Compute` instruction.

## VXM Model

The VXM model lives under `include/ftlpu/vxm/`.

Hierarchy:

- `VxmAlu`: scalar ALU behavior shared by every lane.
- `VxmLane`: one lane with 16 ALUs and 16 independent issue queues.
- `VxmSuperlane`: 8 lanes.
- `VxmSlice`: 4 superlanes/tiles with south-to-north instruction flow.

Each ALU instruction selects two operands from streams, immediates, or
prior-cycle ALU outputs and may emit INT8, FP16, or FP32 bytes to a selected
stream and hemisphere. The ICU exposes one queue per ALU. Test-side offline
program generation expands W8 dequant into eight parallel multiply operations
followed by eight FP16 casts, and expands SwiGLU into a pipelined ALU graph.
There is no fused runtime Dequant or Swish instruction.

## ICU Model

The ICU is implemented in `include/ftlpu/system/icu.hpp`.

Queue counts:

- 88 MEM queues, 44 per hemisphere.
- 4 MXM load queues, one per MXM.
- 4 MXM compute queues, one per MXM.
- 4 SXM queues: one Transpose and one Permute queue per hemisphere.
- 16 VXM queues, one for each ALU in a lane.

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
- VXM ALU instruction: 4 x 32-bit words (operation/operands/output, two FP32
  immediate words, and hemisphere routing).
- ICU queue command: 32 bits.

The MEM instruction address field is not the full software address. It encodes
only the slice-local SRAM word address: bit 12 is the bank and bits 11:0 are the
4096-word offset. The low software byte offset must be zero for `Read`/`Write`.

This is a compact FTLPU CModel encoding, not a Groq hardware binary encoding.

The codec is covered by `tests/core/instruction_codec_test.cpp`.

## Whole-System Topology

`TspSliceSystem` fixes the full mirrored topology:

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W(44) <-> VXM <-> MEM.E(44) <-> SXM.E <-> MXM0/MXM1
```

Important paths:

- Each hemisphere uses the same local `sreg0..sreg12` orientation: `sreg0` is
  adjacent to VXM and `sreg12` is adjacent to that hemisphere's MXMs.
- Global MEM queues 0..43 and MXM IDs 0..1 address East; MEM queues 44..87 and
  MXM IDs 2..3 address West.

- MEM east streams cross SXM before feeding MXM weight and activation inputs.
- SXM accepts only `E0..E31`. With no issued SXM instruction, east data moves
  from `sreg11` to `sreg12` as an ordinary one-cycle register hop.
- The symmetric west register hop moves MXM output from `sreg12` to `sreg11`;
  SXM does not transform west streams.
- MXM int32 outputs are written into west streams at `sreg12`.
- Those west streams travel through MEM to the west edge.
- VXM consumes operands from the stream range and hemisphere selected by each ALU instruction.
- VXM outputs select a destination hemisphere, allowing same-side or cross-center routing.

The system tick handles ICU dispatch, MEM tick, SXM evaluation, VXM stream
bridge, and MXM control slices. MXM compute and output are owned by the `Mxm`
datapath inside `TspSliceSystem`; tests no longer use a separate software GEMM
engine.

## Offline Whole-System Workload

Whole-system integration tests use an offline-only control contract:

1. Before cycle 0, initialize external data in MEM SRAM.
2. Generate the complete instruction timeline and enqueue it into ICU.
3. Start the clock and call only `TspSliceSystem::tick()`.
4. After execution, read final MEM state for verification.

Runtime test code cannot obtain MEM, MXM, VXM, or SXM objects from
`TspSliceSystem`. The public whole-system surface is deliberately limited to
explicit initialization/result methods, `icu()`, `tick()`, and `cycle()`.
Unit tests may still instantiate an individual functional unit directly.

The intended compiler flow is:

```text
workload -> placement/scheduling -> per-queue ICU program -> system ticks
```

Every queue has an independent timeline. A scheduler aligns data and
instructions by inserting queue-local `NOP N` commands and uses
`Repeat n,d` for regular instruction trains. MEM repeats may advance the
slice-local SRAM address by a signed stride.

### W8A16 Projection Regression

`tests/integration/w8a16_projection_test.cpp` is the canonical whole-system
workload. It computes:

```text
A[128,576] fp16 x W[576,1536] int8 -> C[128,1536] fp32
```

Weights use symmetric per-output-column quantization. The test initializes W8
weights and FP16 activations in SRAM, then preloads all ICU queues. At runtime:

- MEM Read sends eight W8 streams west from eight slice groups.
- Eight VXM multiply/cast ALU pairs apply the output-column scales and emit 16
  east byte streams containing eight FP16 values.
- Four consecutive reverse-order `IW` pulses load one 32-column MXM weight
  tile; MXM0 and MXM1 cover adjacent output tiles.
- MEM Read emits FP16 activation vectors on east streams.
- Repeated `Compute` pulses select the weight buffer, activation stream base,
  and west output stream base.
- ICU MEM `Accumulate` instructions consume MXM FP32 output bytes at `sreg11`
  and update the four-slice SRAM result in place across all 18 K tiles.

Both MXMs load weights and compute in parallel. MXM0 accumulates through group9
and MXM1 through group10; the group9 ACC instruction is issued one cycle later
because its input is one westward register hop farther away. The final
comparison reconstructs 196,608 FP32 values from SRAM and checks them against
an FP16-aware scalar golden model.

### W8A16 SwiGLU Regression

`tests/integration/w8a16_swiglu_test.cpp` executes two complete projections:

```text
X[128,576] fp16 x Wgate/Wup[576,1536] int8
    -> gate/up[128,1536] fp32 -> SwiGLU[128,1536] fp16
```

MXM0 carries gate while MXM1 carries up. Their partial sums are accumulated in
MEM groups 9 and 10 across all 18 K tiles. After projection completes, ICU MEM
Read instructions emit gate on `W0..W3` and up on `W4..W7`; one Swish
instruction consumes each pair at the west edge. The one-cycle result is
injected on `E0..E1` and written to slices 29 and 30. The test independently
checks both accumulated projections and every final FP16 result.

### Initialization And Result Access

The allowed non-ICU APIs are intentionally narrow:

- `initialize_mem_sram_lane_byte(...)`: external data initialization only.
- `read_mem_sram_lane_byte(...)`: inspect final SRAM state.

No public whole-system method can issue a MEM/MXM/VXM/SXM instruction, inject a
runtime stream, or tick only part of the datapath.

## Logs

Long whole-system regressions run without logs by default because per-cycle
stream traces dominate simulation time. `TspSliceSystem::LogSinks` can route
ICU, MEM, MXM, VXM, SXM, and system traces when developing a smaller workload.
Unit-level trace demos remain under `examples/`.

## Data Layout Summary

The W8A16 projection stores weights in one MEM slice from each of eight groups:
`0,4,8,...,28`. Their different westward distances are reflected in the
offline MEM issue cycles so all eight values meet their VXM instructions
together.

FP16 activations are duplicated across slices `32..35`. Two byte streams feed
MXM0 and two feed MXM1. MXM0 FP32 results occupy slices `36..39`, while MXM1
results occupy slices `40..43`; each group stores four byte planes at
`row * 48 + output_block`.

Weight column blocks are read in reverse order because each `IW` shifts the
selected weight buffer east and inserts the new block at column 0. Activation
rows and output columns are tiled by the 32-element MXM dimension.

## Known Limitations

- The model is not bit-accurate to any private Groq ISA.
- `Gather` and `Scatter` are not the focus of current tests.
- VXM currently provides 16 ALUs per lane; higher-level functions are offline instruction graphs.
- Softmax still needs an explicit datapath and instruction definition.
- The compiler does not exist yet. `OfflineIcuProgram` is a prototype container
  for what the compiler should emit.

## Suggested Next Steps

Near-term engineering work:

- Tighten the MXM runtime timing against more detailed hardware pipeline
  assumptions.
- Turn `OfflineIcuProgram` into a reusable module instead of a test-local helper.
- Add a textual or binary program dump/load format using the ISA codec.
- Add a simple compiler pass that emits MEM Read/Write, MXM IW/Compute, and VXM
  function-unit timelines for the projection workload.
- Add resource-conflict diagnostics for stream-register and queue collisions.
