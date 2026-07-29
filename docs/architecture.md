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
| Stream-register columns | 15 per hemisphere (`sreg0..sreg14`) |
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
MXM2/MXM3 <-> SXM.W <-> MEM.W(52) <-> VXM <-> MEM.E(52) <-> SXM.E <-> MXM0/MXM1
```

Both hemispheres use the same local orientation:

- `sreg0` is adjacent to VXM.
- Thirteen MEM groups occupy the boundaries `sreg0..sreg13`.
- SXM connects the MEM boundary `sreg13` to the MXM boundary `sreg14`.
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

The original full-supercell form consumes 16 east streams:

```text
8 16-bit values x 2 byte streams = 16 streams
```

Local MXM0 uses `E0..E15`; local MXM1 uses `E16..E31`. Four continuous IW
pulses fill one 32-column weight tile. The peer buffer may still supply
in-flight Compute work while IW fills the inactive buffer.

`IWColumn(buffer, column, inner_column)` is the narrow form. It consumes two
east streams and writes one of the eight columns inside the selected `8 x 8`
supercell:

```text
1 16-bit value per lane x 2 byte streams = 2 streams
```

Local MXM0 uses `E0..E1`; local MXM1 uses `E16..E17`. On an empty buffer,
eight narrow pulses, one for each `inner_column` in `0..7`, make that
supercell valid for Compute. A narrow write to an already complete buffer
updates the selected column without invalidating its other columns.

W8 weights use symmetric per-output-column scales:

```text
scale[n] = max_k(abs(W[k,n])) / 127
```

VXM multiplies INT8 values by the corresponding scale and casts them to FP16
or BF16 before IW. IW stores the two incoming bytes unchanged and therefore
does not carry a data-format field.

### Compute

`Compute(buffer, activation_stream_base, output_stream_base, ..., data_format)`
is a one-cycle control pulse. Consecutive pulses inject consecutive activation
vectors. The selected buffer, stream bases, and FP16/BF16 format travel with
the activation wave.

Each supercell interprets both the activation and raw weight bits using the
Compute format, then dots eight elements against eight weight columns.
Activations move east across the four supercell columns while partial sums move
north and accumulate as FP32. Completed outputs are automatically written to
four consecutive west byte streams selected by the Compute instruction. There
is no separate MXM output command or software output queue.

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

The ICU owns 132 independent queues:

| Queue class | Count |
| --- | ---: |
| MEM | 104 |
| MXM load | 4 |
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
- VXM ALU instructions encoded as four 32-bit words;
- 32-bit ICU NOP/Repeat commands.

SXM instructions use a fixed 13 x 32-bit packet encoding. The packet carries
the header, 16 source and destination stream selectors, the intra-tile lane
map, and the full cross-tile permute map.
Reserved bits and field ranges are validated by
`tests/core/instruction_codec_test.cpp`.

## 9. Scheduling Patterns

### W8A16 Weight and Activation Coexistence

Raw INT8 weights travel west from MEM to VXM. Dequantized FP16 weights then
travel east to MXM, which is the point where they may conflict with activation.

| Active operation | FP16 IW streams | Shared activation |
| --- | --- | --- |
| Load local MXM0 | `E0..E15` | `E16..E17` |
| Load local MXM1 | `E16..E31` | `E0..E1` |
| No IW | none | normally `E0..E1` |

Loading both MXMs in one cycle would occupy all 32 east streams. Offline FFN
schedules therefore load one inactive buffer at a time and place activation in
the opposite half. Both local MXMs broadcast-consume the same FP16 activation
pair.

### Ping-Pong Weights

For projection reductions, Compute uses buffer `k mod 2` while VXM and IW
prepare reduction `k+1` in the other buffer. SRAM slices, stream IDs, VXM ALUs,
and MXM load queues are all explicit scheduling resources.

### Accumulator Lifetime

Each MXM owns a 1 MiB FP32 accumulator with 8192 rows of 32 values. Non-final
`Compute` instructions retain their sums in the selected MXM accumulator.
Final `Compute` or `AccumulatorRead` instructions emit the selected segment to
streams and may clear it. Address reuse is legal only after the final result has
been emitted and cleared.

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

### Full FFN

`dual_hemisphere_w8a16_swiglu_test` computes:

```text
X[128,576]
  -> gate/up[128,1536]
  -> SwiGLU[128,1536]
  -> down[128,576]
```

All four MXMs participate. Non-final gate/up reductions run both hemispheres in
parallel. During the final reduction, East and West 32-row blocks alternate
through the single shared VXM SwiGLU pipeline, so one hemisphere's MXMs are idle
for each alternating block. This is a deliberate throughput tradeoff: it removes
the standalone accumulator readback/SwiGLU phase and reduces the validated
schedule from 93,642 to 90,817 cycles.

SwiGLU results are stored in both MEM hemispheres. Down projection reads local
copies, uses all four MXMs, accumulates 48 K tiles, casts the final sums to FP16,
and verifies all 73,728 output values.

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
