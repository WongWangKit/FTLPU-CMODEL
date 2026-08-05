# FTLPU-CMODEL Architecture Reference

This document describes the current transformer-datapath implementation.
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
| MEM slices | 44 per hemisphere, 88 total |
| MEM groups | 11 per hemisphere, 4 slices per group |
| Stream-register columns | 13 per hemisphere (`sreg0..sreg12`) |
| Streams per lane | 32 eastward + 32 westward |
| Stream-register width | 1 byte |
| SRAM capacity | 2.5 MiB per slice, 220 MiB full chip |
| MXM units | 4 total, 2 per hemisphere |
| MXM array | 32 x 32 systolic K-block output, shared ACC per hemisphere |
| VXM | 1 central slice, 16 physical ALUs per lane |
| SXM | 1 four-tile slice per hemisphere |

One MEM slice owns two SRAM banks. Under the 4-tile/8-lane transformer
configuration, each bank contains `40960 x 32-byte` vector rows, so one slice
retains 2.5 MiB rather than shrinking when the physical vector width changes.
Each row spans all four tiles; one tile accesses its local 8-byte segment when
the instruction wave reaches that tile.

## 2. Full-Chip Topology

```text
MXM2/MXM3 <-> SXM.W <-> MEM.W(44) <-> VXM <-> MEM.E(44) <-> SXM.E <-> MXM0/MXM1
```

Both hemispheres use the same local orientation:

- `sreg0` is adjacent to VXM.
- Eleven MEM groups occupy the boundaries `sreg0..sreg11`.
- SXM connects the MEM boundary `sreg11` to the MXM boundary `sreg12`.
- East streams move from VXM toward MXM.
- West streams move from MXM toward VXM.

Global MEM queues `0..43` and MXMs `0..1` select East. MEM queues `44..87`
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

The normal MXM map gives each local MXM one fixed 16-stream window. Compute
reads activation from the first byte group in that window. Full-width weight
loading owns the window for that cycle; background loading uses only its fixed
upper half so activation and weight data do not occupy the same stream ID.

## 3. Clock and Control Flow

MEM, MXM control, and SXM Transpose instructions enter tile 0 at the south edge
and advance north by one tile per cycle. Workloads must align data and control
at every tile. Tests do not directly manipulate an in-flight tile.

`TspSliceSystem::tick()` performs one complete system cycle:

1. ICU dispatches the next command from every queue.
2. MEM, SXM, VXM, and MXM read current stream-register state.
3. Functional units consume operands and stage outputs.
4. Unconsumed values stage passive links.
5. The shared stream fabric commits next state.

Whole-system tests may construct the static schedule on the host. The compiler
splits it into one program per independently scheduled ICU endpoint and writes
each program into the local i-MEM beside that endpoint's IQ. A program
descriptor supplies `base_pc`, instruction count, and `start_cycle`.

The modeled runtime path is:

```text
per-function local i-MEM -> one-word/cycle prefetch -> finite local IQ
                          -> narrow module control pipeline
```

The i-MEM and IQ widths and depths are parameterized independently for MEM,
MXM, VXM, and SXM. Runtime instruction fetch does not use data SRAM or the data
Stream Register fabric. External loading of the local i-MEM banks is outside
the current execution model.

## 4. MEM

### Organization

Each hemisphere has 44 MEM slice columns and one instruction queue per slice.
Four adjacent slices form a group between two stream-register boundaries.

A MEM instruction is a single-port operation for its slice. A slice cannot read
and write in the same cycle, even at different addresses. MXM partial sums use
ordinary four-slice Read/Write traffic; MEM does not contain an arithmetic unit.

### Instructions

- `Read(address, stream)` reads the tile-local 8-byte SRAM segment and writes it
  to one stream ID.
- `Write(address, stream)` consumes one 8-byte stream segment and stores it.
- `Gather` and `Scatter` are encoded but intentionally reject execution because
  the address-stream datapath is not modeled yet.

Each instruction wave eventually visits all four tiles, so a complete wave
moves one 32-byte physical vector row as four skewed 8-byte segments.

### Addressing

Addresses use separate semantic fields rather than the former
`address + lane` convention:

```text
global byte address = hemisphere | MEM slice | bank | row | byte offset
MEM instruction     = bank | row
```

The address type names `MemGlobalAddress24`, `MemSliceByteAddress17`, and
`MemLocalWordAddress13` retain historical suffixes for source compatibility;
the actual field widths are derived from the active hardware configuration.
The fields use:

- `ceil(log2(hemisphere_count))` bits for hemisphere;
- `ceil(log2(mem_slice_count))` bits for MEM slice;
- `ceil(log2(bank_count))` bits for bank;
- `ceil(log2(rows_per_bank))` bits for row;
- `ceil(log2(row_bytes))` bits for byte offset.

For the 4x8 transformer configuration this means 1 hemisphere bit, 6 slice
bits, 1 bank bit, 16 row bits, and 5 byte-offset bits. Only values inside the
configured counts are legal even where a bit field has unused encodings.
`MemInstruction::address` carries only `bank | row`; tile and lane select the
physical segment and byte within the 32-byte vector row.

`MemLocalWordAddress13::advance_words()` linearizes `(bank,row)` while
incrementing. It therefore crosses from the last row of one bank to row zero of
the next bank, but never crosses a MEM-slice boundary.

### DMA

`DmaDescriptor` identifies direction, purpose, host buffer and byte offset,
global MEM base address, and physical-vector count. `DmaEngine` moves at most
one physical vector per cycle. Every beat accesses one hemisphere, one MEM
slice, one bank, and one vector row. A multi-beat descriptor uses the local
word-address increment rule above, so it may span both banks of one slice.

Transfer IDs are non-zero and allocated monotonically by
`DmaEngine::enqueue()`. Pending and active requests therefore cannot reuse an
ID. Completions are exposed as a consumable queue through
`completion_ready()` and `pop_completion()`; `completion_history()` is retained
separately for debugging.

## 5. MXM

Each `Mxm` contains:

- a `4 x 4` array of supercells;
- one `8 x 8` FP16 weight block per supercell;
- two peer weight buffers per supercell;
- a south-to-north control slice;
- activation and systolic partial-sum pipelines.

The two MXMs in a hemisphere feed one shared accumulator slice at the MEM
boundary. Each output lane has two fixed int32 adder paths. Independent mode
assigns one path to each MXM. Merge mode fixes ACC0 to `MXM0 + MXM1` and ACC1
to `ACC0 + old`; there is no accumulator crossbar or per-MXM result bank.

### Weight Loading

Each local MXM owns a fixed 16-stream window: local MXM0 uses `0..15` and local
MXM1 uses `16..31` in the hemisphere's input direction. `IW` has three modes:

- `Full` consumes the complete 16-stream window and writes the whole weight
  row in one load phase.
- `BackgroundLowerHalf` consumes the fixed upper eight streams and writes
  weight columns `0..7`.
- `BackgroundUpperHalf` consumes the same fixed upper eight streams and writes
  weight columns `8..15`.

The two background pulses make the inactive buffer valid. The first half
invalidates a previously complete buffer, so Compute cannot observe mixed old
and new weights. Smaller evaluation configurations preserve the same rule with
two equal fixed halves.

W8 weights use symmetric per-output-column scales:

```text
scale[n] = max_k(abs(W[k,n])) / 127
```

### Compute

`Compute(buffer, output_stream_base)` is a one-cycle control pulse. Activation
has no stream-select field: it always comes from the first byte group in this
MXM's local 16-stream window. Consecutive pulses inject consecutive activation
vectors. The selected weight buffer and output stream base travel with the
activation wave.

Each supercell is a weight-stationary scalar-MAC array. One physical MAC row
consumes one activation component and broadcasts it across every output
column. A complete activation row may enter every cycle, but component `k` is
delayed by `k` cycles. In steady state, MAC row `k` therefore processes token
row `t-k` while its partial sum advances to the next MAC row. Supercell latency
is the block K dimension and initiation interval is one cycle.

The Compute control and activation for the next outer supercell row are delayed
by the complete MAC-array latency so they meet the matching northbound partial
sum. Activation vectors still advance east by one supercell column per cycle.
A completed K block enters the shared hemisphere accumulator. A final result
is multiplied combinationally by the compiler-configured activation-scale x
weight-scale product, then passes through the one-cycle FP16 Cast and is
written to one fixed two-byte stream group for the VXM Input Buffer. A
non-final multi-context result is written as four int32 byte streams for
compiler-scheduled MEM storage; partial sums are never dequantized.

The system owns all MXM runtime state; whole-system tests do not use a separate
GEMM engine or runtime helper.

## 6. VXM

The central VXM contains one superlane per tile row and one lane per tile lane.
Each lane has 16 physical ALU stages. Eight shared compact-instruction channels
feed the stages, with local decode supporting chain depths of 2, 4, or 8.

Supported opcodes are:

```text
Pass Add Subtract Multiply Divide Negate Abs Min Max Clamp
Square Sqrt Exp Log Relu Cast
```

The physical stream operand is FP16. Other operands select previous, original,
auxiliary, accumulator, immediate, or feedback values. A 96-bit compact packet
carries the operation, operands, precision, output type, chain depth, repeat
count, and accumulator controls. Output stream IDs use the central 64-stream
interface; compact packets do not carry hemisphere fields.

An output instruction may instead select the optional static quantization
boundary. Every output lane has a fixed one-cycle FP16-to-INT8 pipeline with no
ready/backpressure signal. The compiler supplies the static scale and
zero-point and schedules the resulting single byte stream through SR into MEM.
Dynamic per-token scale calculation and replay are intentionally not modeled.

## 7. SXM

There are two independent SXMs, one per hemisphere. SXM transforms only east
streams; west streams take the symmetric register hop without transformation.

Each SXM has four tile rows. A Transpose instruction advances south to north one
tile per cycle so each tile captures its matching diagonal wavefront. FP16 low
and high bytes use two planes; statically quantized INT8 read from MEM uses the
explicit single-byte-plane mode. Tile-local Transpose exchanges rows and
columns of one parameterized lane-count square block.

Transpose output is registered for one cycle before Permute may consume it.
Permute rearranges complete blocks across four superlanes/32 lanes. The current
implementation uses one transpose buffer; same-destination blocks can pipeline
at `II=4`.

Each hemisphere has three independently scheduled SXM ICU queues: Stream,
Transpose, and Permute. This permits the physically independent transpose and
permute paths to issue in the same logical VLIW cycle. With no issued SXM
operation, east streams pass from `sreg11` to `sreg12` as an ordinary one-cycle
link.

## 8. ICU and ISA

The ICU owns 114 independent queues:

| Queue class | Count |
| --- | ---: |
| MEM | 88 |
| MXM load | 4 |
| MXM compute | 4 |
| MXM activation dequantize | 4 |
| VXM ALU | 8 |
| SXM Stream | 2 |
| SXM Transpose | 2 |
| SXM Permute | 2 |

Every functional slice has one program-ordered IQ whose entry type is:

```cpp
template <typename FuncInstruction>
using IqEntry =
    std::variant<IcuControlInstruction, FuncInstruction>;
```

The ICU examines only the head of each IQ every cycle. ICU control instructions
execute locally; functional instructions dispatch to that slice. Consequently,
an ordered sequence such as `Read, Sync, Write` issues Read, blocks with Sync at
the head until a notification token arrives, and only then dispatches Write.
There is no separate control program and no `Dispatch` pseudo-instruction.

Each MXM has three IQs because load, compute, and activation dequantize are
independently schedulable and may overlap. Other queue counts follow physical
functional paths: one per MEM slice, one per VXM shared compact-instruction
channel, and three per hemisphere SXM.

Implemented queue commands are:

- `NOP N`: delay that queue by `N` cycles.
- `Repeat n,d`: repeat the previous instruction `n` times at interval `d`.
- MEM Repeat may apply a signed address stride.
- `Sync`: remain at the queue head until a notification token is available.
- `Notify`: emit a barrier notification locally.

The compact model codec currently covers:

- 32-bit MEM instructions;
- 32-bit MXM control instructions;
- 32-bit ICU control instructions;
- SXM instruction packets.

VXM compact instructions use their VXM-local 96-bit codec and are not encoded
as legacy core program packets.

Reserved bits and field ranges are validated by
`tests/core/instruction_codec_test.cpp`.

## 9. Scheduling Patterns

### Accumulator Lifetime

A single live result context may remain in the shared ACC operand/result
registers. Multiple token contexts spill int32 partials with ordinary MEM
`Write` instructions and restore them with ordinary `Read` instructions. The
compiler selects local start/accumulate/finalize or MEM
start/accumulate/finalize in the MXM Compute control.

### Single-Port MEM

Different addresses do not remove a slice conflict. Read and Write windows
must be disjoint whenever they reserve the same physical slice.

## 10. Logs and Diagrams

Long regressions disable logging by default because per-cycle stream dumps
dominate wall time. `TspSliceSystem::LogSinks` can independently capture ICU,
MEM, MXM, VXM, SXM, and system logs.

## 11. Known Limitations

- The model is not bit-accurate to private Groq hardware or ISA.
- Gather/Scatter lack the address-stream execution datapath.
- Offline schedules are still constructed inside integration tests; there is
  no standalone compiler or serialized program-file container yet.
- External loading of the distributed local i-MEM banks is not modeled yet.
- Resource allocation uses workload-specific SRAM slices and stream IDs rather
  than a general allocator.
- The simulator scans substantial inactive state every cycle and remains slow
  for long workloads.

## 13. Next Engineering Steps

1. Extract a reusable offline program and resource-calendar scheduler.
2. Add SRAM/stream lifetime allocation and conflict diagnostics.
3. Add ProgramImage serialization using the existing ISA codec.
4. Pipeline remaining attention phases using data-ready events instead of
   global phase barriers.
5. Add simulator fast-forward for queue NOP spans and globally idle intervals.
