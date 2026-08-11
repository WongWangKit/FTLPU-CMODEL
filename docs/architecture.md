# FTLPU-CMODEL Architecture Reference

This document describes the current implementation on branch `4x8_fp16`.
Public Groq LPU/TSP material is architectural inspiration only; field widths,
timing rules, and functional behavior here belong to this CModel.

## 1. Current Configuration

The central vector shape is:

```text
4 tiles/superlanes x 8 lanes = 32 elements
```

| Property | Value |
| --- | ---: |
| Hemispheres | 2 |
| MEM slices | 52 per hemisphere, 104 total |
| MEM groups | 13 per hemisphere, 4 slices per group |
| Stream-register columns | 16 per hemisphere (`sreg0..sreg15`) |
| Streams per lane | 32 eastward + 32 westward |
| Stream-register width | 1 byte |
| SRAM capacity | 2 MiB per slice, 208 MiB full chip |
| MXM units | 4 total, 2 per hemisphere |
| MXM array | 32 x 32 FP16/BF16 multiply with FP32 accumulation |
| VXM | 1 central slice, 16 ALUs per lane |
| SXM | 1 four-tile slice per hemisphere |

One MEM slice owns a `65536 x 32-byte` SRAM block. Each row spans all four
tiles; one tile accesses its local 8-byte segment when the instruction wave
reaches that tile. The model has 104 homogeneous slices across two hemispheres,
for `104 x 2 MiB = 208 MiB` total SRAM. There is no logical bank subdivision.

## 2. Full-Chip Topology

```text
MXM2/MXM3 <-> SXM.W <-> C2C.W <-> MEM.W(52) <-> VXM <-> MEM.E(52) <-> C2C.E <-> SXM.E <-> MXM0/MXM1
```

Both hemispheres use the same local orientation:

- `sreg0` is adjacent to VXM.
- Thirteen MEM groups occupy the boundaries `sreg0..sreg13`.
- Each hemisphere has independent C2C TX/RX endpoints attached at the MEM boundary `sreg13`.
- One SR hop separates C2C from SXM: `sreg14` is the C2C/SXM boundary.
- SXM connects `sreg14` to the MXM boundary `sreg15`.
- East streams move from VXM toward MXM.
- West streams move from MXM toward VXM.

Global MEM queues `0..51` and MXMs `0..1` select East. MEM queues `52..103`
and MXMs `2..3` select West.

The shared stream fabric is double-buffered by cycle: functional units read
current state and stage next state, then the system commits once. A value written
this cycle is visible next cycle.

### Broadcast Consumption

Multiple functional units may consume the same physical stream-register cell
in one cycle and observe identical data and vector tags. Consumption is
idempotent and means "at least one consumer":

- any consumption suppresses passive propagation of that value;
- multiple readers are legal;
- multiple producers targeting the same next-state cell remain illegal.

Broadcast consumption lets two local MXMs share one activation stream pair. It
does not allow weight and activation data to occupy the same stream ID.

## 3. Clock and Control Flow

MEM, MXM control, and SXM Transpose instructions enter tile 0 at the south edge
and advance north by one tile per cycle. Workloads must align data and control
at every tile. Tests do not directly manipulate an in-flight tile.

VXM follows the same offline-scheduling contract. An issued ALU instruction
must find every stream or prior-ALU operand in that cycle. A missing operand is
a static schedule error; the model does not insert an implicit queue stall.

`TspSliceSystem::tick()` performs one complete system cycle:

1. ICU dispatches the next command from every queue.
2. MEM, SXM, VXM, and MXM read current stream-register state.
3. Functional units consume operands and stage outputs.
4. Unconsumed values stage passive links.
5. The shared stream fabric commits next state.

Whole-system tests initialize data and ICU queues before cycle 0. After clocking
starts, they call only `tick()` until the offline schedule completes.

## 4. MEM

### Organization

Each hemisphere has 52 MEM slice columns and one instruction queue per slice.
Four adjacent slices form a group between two stream-register boundaries. All
52 slices are homogeneous SRAM; no MEM group has accumulator behavior.

A MEM instruction is a single-port operation for its slice. A slice cannot read
and write in the same cycle, even at different addresses.

### Instructions

- `Read(address, stream)` reads the tile-local 8-byte SRAM segment and writes it
  to one stream ID.
- `Write(address, stream)` consumes one 8-byte stream segment and stores it.
- `Gather` and `Scatter` are encoded but intentionally reject execution because
  the address-stream datapath is not modeled yet.

Each instruction wave eventually visits all four tiles, so a complete wave
moves one 32-byte physical vector row as four skewed 8-byte segments.

### Addressing

The public-style software address layout used as reference is:

```text
[39:27] chip
[26]    hemisphere
[25:20] slice
[19:4]  row offset within the slice
[3:0]   software byte offset
```

`MemInstruction::address` stores only the 16-bit slice-local row field
corresponding to software bits `[19:4]`, giving rows `0..65535`. The test
initialization/result APIs expose tile and lane byte selection separately.

## 5. MXM

Each `Mxm` contains:

- a `4 x 4` array of supercells;
- one raw `8 x 8` 16-bit weight block per supercell;
- two peer weight buffers per supercell;
- a south-to-north control slice;
- activation-flow and FP32 output state.

### Weight Loading

`IW(buffer, column)` writes one explicit supercell-column block into the selected
buffer. The two-bit `column` field selects columns `0..3`; load order does not
implicitly shift the final layout. There is no `LW` instruction.

The default full-supercell form consumes eight INT8 east streams:

```text
8 INT8 values per lane x 1 byte stream = 8 streams
```

A separate Dequant instruction queue supplies one BF16 scale immediate per
load. Dequant and IW issue together and propagate south-to-north in lockstep.
At each tile the load path computes `BF16(INT8 * BF16(scale))` before writing
the selected weight buffer. A missing or misaligned Dequant pulse is an
execution error.

Local MXM0 reserves `E0..E15`; local MXM1 reserves `E16..E31`. The default
INT8 load uses the first eight streams in each half. Four continuous IW pulses
fill one 32-column weight tile. The peer buffer may still supply in-flight
Compute work while IW fills the inactive buffer.

`IWColumn(buffer, column, inner_column)` is the narrow form. It consumes two
east streams in Direct16 compatibility mode, or one INT8 stream by default,
and writes one of the eight columns inside the selected `8 x 8` supercell:

```text
default: 1 INT8 value per lane x 1 byte stream = 1 stream
Direct16: 1 16-bit value per lane x 2 byte streams = 2 streams
```

Default INT8 narrow load uses `E0` for local MXM0 or `E16` for local MXM1;
Direct16 uses `E0..E1` or `E16..E17`. On an empty buffer, eight narrow
pulses, one for each `inner_column` in `0..7`, make that supercell valid for
Compute. A narrow write to an already complete buffer updates the selected
column without invalidating its other columns.

W8 weights may use symmetric per-output-column scales:

```text
scale[n] = max_k(abs(W[k,n])) / 127
```

Strict per-output-column scaling uses eight `IWColumn`/Dequant pairs. A full
`8 x 8` IW broadcasts one scale immediate across all eight loaded columns.
`IWDirect16` and `IWColumnDirect16` retain the compatibility path that stores
incoming FP16/BF16 bits unchanged.

### Compute

`Compute(buffer, activation_stream_base, output_stream_base, ..., data_format,
compute_mode)` is a one-cycle control pulse. The selected buffer, stream bases,
FP16/BF16 format, and compute mode travel with the activation wave.

| Mode | Activation input per tile | Result per MXM pulse | MACs per supercell |
|---|---:|---:|---:|
| `Vector` | `1 x 8`, 2 byte streams | `1 x 32` | 64 |
| `Block8` | `8 x 8`, 16 byte streams | `8 x 32` | 512 |

Both modes interpret the raw weight bits using the Compute format. `Vector`
preserves the original behavior: consecutive pulses inject consecutive rows,
and a Stream destination emits one FP32 row over four west byte streams.

`Block8` treats each stream pair as one activation row and each lane as one K
element. One pulse computes eight output rows, advances the row counter by
eight, and selects one logical `8 x 32` FP32 accumulator row. Each supercell
column updates its `8 x 8` segment at that same wide-row address.
Block8 is accumulator-only. A Block8 `AccumulatorRead` emits one `8 x 8`
column segment per cycle over all 32 west byte streams; the four column
segments drain the full `8 x 32` row in four cycles. Up to four consecutive
Block8 pulses form one `32 x 32` output tile.

Activations move east across the four supercell columns while partial sums move
north and accumulate as FP32. There is no separate MXM output command or
software output queue.

The system owns all MXM runtime state; whole-system tests do not use a separate
GEMM engine or runtime helper.

## 6. VXM

The central VXM contains four superlanes, eight lanes per superlane, and 16 ALUs
per lane. Each ALU has an independent ICU queue. The same ALU instruction is
applied across the physical lanes.

Supported opcodes are:

```text
Pass Add Subtract Multiply Divide Negate Abs Min Max Clamp
Square Sqrt Exp Log Relu Cast
```

Operands may come from INT8, FP16, BF16, INT32, or FP32 streams; FP32
immediates; or prior-cycle ALU outputs. Results may remain in an ALU register
or be emitted to a selected stream and destination hemisphere. Cast can emit
FP16 or BF16 as two little-endian byte streams.

Quantization and dequantization are instruction graphs, not dedicated opcodes.
For example, W8 dequant is synthesized with Multiply and Cast. SwiGLU is
synthesized from arithmetic, Exp, pipeline-delay Pass operations, and Cast.
RMSNorm uses ALU feedback for `sum(x^2)` and keeps inverse RMS resident while
`x` and `gamma` stream through.

## 7. SXM

There are two independent SXMs, one per hemisphere. SXM transforms only east
streams; west streams take the symmetric register hop without transformation.

Each SXM has four tile rows. A Transpose instruction advances south to north one
tile per cycle so each tile captures its matching diagonal wavefront. FP16 low
and high bytes are two planes. Tile-local Transpose exchanges rows and columns
of an `8 x 8` block. Transpose and Permute have one physical width only:
16 byte streams in and 16 byte streams out. One beat therefore carries all
eight FP16 rows; the former two-stream serial mode is not supported.

Transpose output is registered for one cycle before Permute may consume it.
Permute rearranges complete blocks across four superlanes/32 lanes. The current
implementation uses one transpose buffer; same-destination blocks can pipeline
at `II=4`.

Each hemisphere has two ICU queues for SXM: one Transpose queue and one Permute
queue. With no issued SXM operation, east streams pass from `sreg13` to
`sreg14` as an ordinary one-cycle link.

## 8. ICU and ISA

The ICU owns 136 independent queues:

| Queue class | Count |
| --- | ---: |
| MEM | 104 |
| MXM load | 4 |
| MXM Dequant | 4 |
| MXM compute | 4 |
| VXM ALU | 16 |
| SXM Transpose/Permute | 4 |

Implemented queue commands are:

- `NOP N`: delay that queue by `N` cycles.
- `Repeat n,d`: repeat the previous instruction `n` times at interval `d`.
- MEM Repeat may apply a signed address stride.

`Sync`, `Notify`, `Ifetch`, and power configuration are not implemented.

The compact model codec currently covers:

- 32-bit MEM instructions;
- 32-bit MXM control instructions;
- 16-bit MXM Dequant BF16 scale immediates;
- VXM ALU instructions encoded as four 32-bit words;
- 32-bit ICU NOP/Repeat commands.

SXM instructions use a fixed 13 x 32-bit packet encoding. The packet carries
the header, 16 source and destination stream selectors, the intra-tile lane
map, and the full cross-tile permute map.
Reserved bits and field ranges are validated by
`tests/unit/core/instruction_codec_test.cpp`.

## 9. Scheduling Patterns

### W8A16 Weight and Activation Coexistence

Raw INT8 weights travel east from MEM to the MXM-local dequantizer. The scale
comes from the independent Dequant queue, so VXM is not used by the default
weight-load path.

| Active operation | INT8 IW streams | Available peer half |
| --- | --- | --- |
| Load local MXM0 | `E0..E7` | `E16..E31` |
| Load local MXM1 | `E16..E23` | `E0..E15` |
| No IW | none | normally `E0..E1` |

Both local MXMs may load in one cycle using 16 total east streams. Activation
may use an unoccupied range subject to the normal stream-collision rules.
Direct16 compatibility loads still occupy a full 16-stream half.

### Ping-Pong Weights

For projection reductions, Compute uses buffer `k mod 2` while MEM, Dequant,
and IW prepare reduction `k+1` in the other buffer. SRAM slices, stream IDs,
and the MXM Dequant/load queues are explicit scheduling resources.

### Accumulator Lifetime

Each MXM owns two mode-specific FP32 accumulator SRAMs:

| Mode | SRAM geometry | Row width | Capacity |
|---|---:|---:|---:|
| `Vector` | `(block_count * 32) x 32 FP32` | 128 bytes | 384 KiB at 96 blocks |
| `Block8` | `(block_count * 4) x (8 x 32 FP32)` | 1024 bytes | 384 KiB at 96 blocks |

`FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT` is a CMake cache parameter and defaults to
96 complete 32x32 partial-sum blocks. Vector Compute writes one eight-column
segment of a narrow row. Block8 Compute writes the same eight-column segment
for all eight output rows, and the four column segments share one logical wide
address. The Compute/AccumulatorRead mode bit selects the SRAM. A Block8 read
occupies all 32 west streams, so its output window must be scheduled against
the local peer MXM. Address reuse is legal only after the final result has been
emitted and cleared.

### Single-Port MEM

Different addresses do not remove a slice conflict. Read and Write windows must
be disjoint whenever they reserve the same physical slice.

## 10. Validated Whole-System Workloads

### W8A16 Projection

`w8a16_projection_test` computes:

```text
A[128,576] fp16 x W[576,1536] int8 -> C[128,1536] fp32
```

Weights use symmetric per-output-column W8 scales. VXM dequantizes weights,
MXM0/1 compute adjacent output blocks, and their MXM-local accumulators sum
18 K tiles. All 196,608 outputs are compared against an FP16-aware scalar
golden model.

### Block8 + MXM Dequant FFN

`smollm2_block8_dequant_ffn_test` uses the same SmolLM2 dimensions:

```text
X[128,576] BF16
  -> gate/up[128,1536]
  -> BF16 SwiGLU[128,1536]
  -> down[128,576] FP32
```

Weights enter every MXM as eight INT8 streams. The independent Dequant queue
supplies one BF16 scale per eight-column group, and the converted BF16 values
are written directly into the weight buffers. Gate, up, and down all use
Block8 Compute and the configurable `(block_count * 4) x (8 x 32 FP32)` wide accumulator.

Four consecutive Block8 pulses cover one physical 32-row tile. The 128 input
rows are scheduled as four such groups, with enough separation for the last
column wave to leave the internal partial-sum rows before they are reused.
Gate/up accumulator results are checked exactly and materialized as FP32 VXM
inputs at an explicit test phase boundary. The six-stage VXM SwiGLU pipeline
accepts one 32-element vector per cycle, casts the result to BF16, and writes
the 16-stream packed activation layout into both hemispheres. Down consumes
that VXM output through the full 48-reduction projection. Down results use a
scaled FP32 tolerance because the hardware accumulates tile partial sums in a
different association than the scalar golden loop.

Accumulator state is encoded on the owning `MXM.*.Compute` event rather than
drawn as a separate functional-unit lane. Partial reductions are purple; the
final reduction is dark purple because the completed sum is ready but still
retained in SRAM. Red is reserved for a Compute or AccumulatorRead operation
that explicitly emits to stream and clears the slot.

The generated schedule artifacts are
[`smollm2_block8_dequant_ffn_schedule.csv`](traces/smollm2_block8_dequant_ffn_schedule.csv)
and
[`smollm2_block8_dequant_ffn_schedule_detail.svg`](figures/smollm2_block8_dequant_ffn_schedule_detail.svg).

### RMSNorm

`rmsnorm_test` computes `[32,32]` FP16 RMSNorm entirely through MEM and VXM.
ALU0 squares one hidden column per cycle; ALU1 recurrently accumulates
`sum(x^2)` independently in all 32 physical lanes. VXM then computes inverse
RMS and keeps it resident in an ALU. MEM schedules each `x` vector one cycle
before its matching `gamma`, so two multiply ALUs consume them directly without
Pass stages. No MXM or MEM accumulator is used.

### SmolLM2 Attention

`smollm2_attention_test` validates:

```text
Q/K/V projection -> Q/K RoPE -> QK score -> scaled three-pass softmax
-> P x V -> o_proj[128,576]
```

The configuration is sequence length 128, hidden size 576, 9 query heads,
3 KV heads, and head dimension 64. QK, P x V, and o_proj use independent work
across all four MXMs where stream and accumulator resources allow. SXM prepares
packed/transpose layouts for attention replay. The complete numerical golden
check passes at 94,761 scheduled cycles.

See [attention_pipeline_optimization.md](attention_pipeline_optimization.md) for
phase timing, measured MXM utilization, and remaining overlap opportunities.

### Decoder-Layer Decode

`decoder_layer_decode_test` starts from a materialized 128-token K/V cache and
executes token 128 through one tile-scale decoder layer:

```text
RMSNorm -> Q/K/V -> RoPE -> append K/V
        -> QK -> softmax -> P x V -> O -> residual
        -> RMSNorm -> Gate/Up -> SwiGLU -> Down -> residual
```

The test uses hidden/head dimension 32 and intermediate dimension 64 so every
matrix edge is an exact MXM tile. All GEMV, QK, and P x V work is issued through
the system MEM/MXM paths. Decode Q is installed as one weight column with the
two-stream `IWColumn` form, then all 129 cached K rows stream through the MXM.
The nonlinear composition follows the FP16 boundaries already validated by the
dedicated RMSNorm, VXM, attention, and SwiGLU tests. Scaling to the SmolLM2
576/1536/9-head shape repeats these tile mappings across four MXMs.

Decode control supports two explicitly encoded layouts:

- `Linear1x16` retains the original 16-supercell serial reduction. Each tile
  loads four independent activation vectors from eight BF16 streams, and one
  wave produces eight outputs after 16 cell stages.
- `Native4x4` treats the physical array as four columns of four cells. Each
  tile loads one 8-element activation vector from two BF16 streams and
  broadcasts it across its row. Weight column `c` is scheduled `c` cycles
  after column 0, so `(tile,column)` executes at `launch + tile + column`.
  Four 8-output vertical reductions complete together after seven stages.

`mxm_decode_layout_comparison_test` executes the same `K=128, N=32` GEMV in
both layouts and requires bit-identical BF16 results. The full decode FFN uses
`Native4x4`; decode attention keeps `Linear1x16` until its RoPE and cache
resident layouts are migrated as a unit.

### Multi-executable boundaries

`TspSliceSystem::reset_execution_state()` establishes a clean cycle-zero
boundary between command binaries. It resets ICU queues, MEM instruction and
stream state, MXM datapaths and accumulators, VXM, SXM, and the system cycle,
while preserving MEM SRAM contents. This distinction is required for
model-level execution: decoder activations can remain resident in SRAM while
each layer starts with no stale pipeline or stream state.

## 11. Logs and Diagrams

Long regressions disable logging by default because per-cycle stream dumps
dominate wall time. `TspSliceSystem::LogSinks` can independently capture ICU,
MEM, MXM, VXM, SXM, and system logs.
With all sinks null, MEM transfer traces, VXM operand strings, and SXM event
strings are not constructed. SRAM uses sparse 4 KiB backing pages while
preserving the complete architectural address space.

Schedule CSV export is controlled by:

- `FTLPU_SCHEDULE_TRACE=<path>`
- `FTLPU_SCHEDULE_TRACE_ONLY=1`
- `FTLPU_SCHEDULE_REPORT=1`

Detailed diagrams are generated by `scripts/render_schedule_trace.py` and
`scripts/render_swiglu_schedule_trace.py`. Accumulator colors are:

- purple: retain partial sum in the MXM accumulator;
- red: emit final sum to stream and clear the slot.

## 12. Known Limitations

- The model is not bit-accurate to private Groq hardware or ISA.
- Gather/Scatter lack the address-stream execution datapath.
- Offline schedules are still constructed inside integration tests; there is
  no standalone compiler or reusable program file format.
- Resource allocation uses workload-specific SRAM slices and stream IDs rather
  than a general allocator.
- The simulator scans substantial inactive state every cycle and remains slow
  for long workloads.

## 13. Next Engineering Steps

1. Extract a reusable offline program and resource-calendar scheduler.
2. Add SRAM/stream lifetime allocation and conflict diagnostics.
3. Add program serialization using the existing ISA codec.
4. Pipeline remaining attention phases using data-ready events instead of
   global phase barriers.
5. Add simulator fast-forward for queue NOP spans and globally idle intervals.
