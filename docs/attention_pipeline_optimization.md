# Attention Pipeline Optimization Study

This note analyzes the offline ICU schedule in `smollm2_attention_test` for:

- sequence length 128;
- hidden size 576;
- 9 query heads, 3 KV heads, and head dimension 64;
- W8A16 Q/K/V and output projections;
- FP16 RoPE, three-pass softmax, GQA context, and output projection.

All cycle counts below come from the cycle-accurate CModel. The optimized
schedule still passes the complete numerical golden check.

## Measured Result

| Phase | Baseline cycles | Validated cycles | Change |
|---|---:|---:|---:|
| Q projection | 26,900 | 18,060 | -32.9% |
| K projection | 10,760 | 7,224 | -32.9% |
| V projection | 10,760 | 7,224 | -32.9% |
| GQA Q copy | 425 | 188 | -55.8% |
| K replica for dual MXM QK | 0 | 452 | new |
| QK + softmax | 37,440 | 19,132 | -48.9% |
| P replay layout | 0 | 1,582 | explicit stage |
| P x V | 31,392 | 10,856 | -65.4% |
| Context route | 824 | 16 | -98.1% |
| o_proj | 48,412 | 18,166 | -62.5% |
| **Total** | **166,913** | **81,273** | **-51.3%** |

P3 writes packed x16 directly, removing the former two-slice output and VXM
repacking pass. SXM converts P to its persistent replay layout immediately after
softmax, so P x V contains only the 2-stream replay. V packing is not a separate
phase: V projection writes the packed MEM layout as its architectural output.

### Current SXM and IW dataflow

- V projection writes its two 32-dimension halves directly into two 16-stream
  MEM layouts. SXM later captures one FP16 8 x 8 block per beat and its
  single-buffer transpose/permute path writes directly to `IW(buffer, column)`.
  Transpose instructions advance through the four tiles one cycle at a time,
  and Permute reads a completed block no earlier than the following cycle.
  The 2-bit column ID removes reverse-load semantics.
- Softmax P3 writes P blocks into 16 streams directly. SXM transposes them after
  softmax and writes ordinary MEM. P x V then replays one FP16 query vector per
  cycle on two streams without invoking SXM for P.
- Context stays in its producer hemisphere. During a four-MXM o_proj wave, one
  stored FP16 copy feeds source-local MXM0 directly. The second copy enters
  VXM, which distributes it to source-local MXM1 and both remote MXMs. No
  cross-hemisphere MEM copy is materialized.

## Validated Scheduling Changes

### Tight MXM block issue

An MXM activation block contains 32 vectors. The next block cannot start after
only 32 cycles in the current model because both the four-row control wavefront
and the four-column activation wavefront must drain. The safe issue interval is:

```text
32 activation cycles + 3 control-skew cycles + 3 datapath-tail cycles = 38 cycles
```

The original schedule used 64 cycles. Projection, QK, and o_proj now use the
validated 38-cycle interval where their downstream stream paths permit it.

P x V still retains its original interval. Compressing it produced an E0
stream-register collision because the next probability wavefront reached the
passive SR chain before the previous SXM skewed output had drained.

### Parallel projection reductions

East and west hemisphere MEM/MXM resources are independent. Both non-final and
final Q/K/V reduction blocks now compute at the same cycle. Final Q/K
reductions route East RoPE through VXM ALU0..5 and West RoPE through ALU8..13;
V casting similarly uses separate ALU pairs. The final token-block interval
remains 64 cycles because a shorter interval overlaps packed Q writes on MEM
slices 32/33 with the next activation read.

### Explicit invalid-schedule detection

MXM now rejects a Compute block whose row cursor exceeds the physical 32 rows.
Previously this condition indexed past the accumulator arrays and terminated
the process with an access violation. Invalid schedules now produce a useful
exception.

### Per-MXM utilization monitoring

The system samples every MXM on every tick. The validated workload reports:

| MXM | Array utilization | Active-cycle density |
|---|---:|---:|
| East MXM0 | 46.62% | 84.21% |
| East MXM1 | 46.62% | 84.21% |
| West MXM0 | 32.76% | 84.21% |
| West MXM1 | 32.76% | 84.21% |

Combined array utilization is about 39.69%. Active-cycle density is already
high, so the largest remaining gains come from eliminating idle regions and
balancing work across hemispheres, not from changing the supercell MAC loop.

## Current State and Remaining Overlaps

### 1. Remaining: extend four-MXM scheduling to Q/K/V projection

QK now uses all four MXMs whenever both hemispheres have assigned query work.
RoPE writes each Q head's two 32-value reduction blocks directly into IW-ready
SRAM layouts spanning 32 MEM slices. Each local MXM receives an independent
query block and holds its two reduction blocks in weight buffers 0 and 1.
The current schedule replicates K into the released RoPE-table slices, so local
MXM0 reads original K and local MXM1 reads the replica. The stream fabric now
supports same-cycle broadcast consumption, so a future schedule can remove this
replica when both MXMs consume the same register column and stream. The two
arrays accumulate independent complete score tiles in their separate MEM
accumulator groups.

The recurrent softmax max/sum state and its temporary SRAM streams remain
serial, so VXM drains the completed score tiles one at a time. This preserves
the exact softmax semantics while removing the MXM-side QK serialization.

o_proj now groups adjacent output-column pairs into one wave: East MXM0/1
compute the even pair while West MXM0/1 compute the odd pair. Nine output pairs
therefore execute in five waves. The final wave has only the East pair, which
explains the remaining East/West utilization difference.

Projection still prepares one reduction at a time and can apply the same
technique after its weight and activation SRAM lifetimes are separated.

Required work:

- extend the independent-work scheduler to projection;
- allocate weight SRAM slices that do not conflict with simultaneous activation
  reads or result writes;
- keep a buffer live until the final column wavefront has left the MXM.

This would remove most of the repeated 42-cycle weight-preparation prefix from
Q/K/V projection reduction blocks. The equivalent o_proj change is already
complete.

### 2. Remaining: partition shared VXM work by hemisphere

Projection post-processing is now partitioned: RoPE needs six ALUs per
hemisphere, so East uses ALU0..5 and West uses ALU8..13. This removes the
32-cycle final-reduction hemisphere skew. Softmax still uses ALU0..6 and its
recurrent max/sum state is serialized. Dequant uses all 16 ALUs for two cycles;
it can remain serialized because its short bursts can normally be hidden behind
MXM compute.

Remaining work can enable:

- an East Q head and West Q head to run QK/softmax concurrently;
- overlapping softmax with other VXM work when stream and SRAM lifetimes permit.

### 3. Completed: reuse V weights across grouped-query heads

Three query heads share one KV head. P x V is now ordered as:

```text
KV head -> key block -> load V once -> three query heads -> query blocks
```

This reduces V matrix loads from 72 to 24. Each load consumes packed V x16 from
MEM and uses the SXM single-buffer transpose/permute path to write the four explicit
IW columns directly.

### 4. Remaining: pipeline the three softmax passes

Pass 3 now writes FP16 probabilities directly into the packed x16 layout, and
the old VXM repacking loop is gone. P transpose currently starts after the full
softmax phase; per-work readiness can overlap it with later softmax work.

Pass 1, pass 2, and pass 3 use different ALU sets but currently reuse the same
west input and east output stream IDs. Allocate separate stream ranges:

```text
pass 1: W0-W3  -> E8-E11
pass 2: W4-W7  -> E12-E15
pass 3: W8-W11 -> E16-E17
```

The scaled-score, exponent, and probability SRAM slices are already distinct.
After stream remapping, pass 3 of block N can overlap pass 2 of block N+1, and
both can overlap QK/pass 1 for a later block if its hemisphere and ALU bank are
available.

### 5. Remaining: start consumers as soon as producer data is complete

The single global `phase_start` acts as a full-chip barrier. Replace it with
data-ready events:

- start QK for a query head as soon as that Q head and its shared K head exist;
- project V while earlier heads execute QK and softmax;
- perform a required GQA Q copy immediately after that Q head is written;
- start P x V for a head as soon as its probability blocks and V head exist;
- start the first o_proj reductions while later attention heads still produce
  context data.

This is a list-scheduling problem over independent ICU queues rather than a
sequence of global phases.

### 6. Completed: run o_proj on all four MXMs

Each full wave assigns one output-column pair to each hemisphere, so all four
MXMs issue Compute together. Context remains in the hemisphere that produced
it. One existing MEM copy feeds local MXM0 directly; the other is read on
`W16..W17` and VXM distributes it to local MXM1 plus both remote MXMs. Weight
dequant uses `W8..W15`, keeping it disjoint from fixed MXM result streams
`W0..W7`. No remote context MEM copy is materialized.

### 7. Completed: tile-pipelined SXM transpose buffer

Each hemisphere has one transpose buffer. A Transpose instruction advances
from tile0 through tile3, one tile per cycle, and each tile captures only when
its local instruction slot is active. Permute remains a global cross-lane
operation and can consume a completed transpose block after a one-cycle
register boundary. Permute releases each tile before that tile captures the
next block in the same cycle, so blocks using the same 16-stream destination
run at `II=4` with one buffer. Changing the Permute destination, such as MXM0
`E0..E15` to MXM1 `E16..E31`, first drains the three-cycle northbound tail.

### 8. Completed: safe accumulator lifetimes

Context accumulator addresses include the query head. During o_proj, each
hemisphere has only one active output pair, and its two local MXMs use the two
physical accumulator groups. The final reduction emits each sum with
`stream+clear` before that hemisphere reuses the row for its next pair.

## SRAM and Stream Allocation

The current fixed `kWeightSlices` list is shared by every phase. Tight scheduling
exposed a real single-port conflict between an o_proj result write on slice 28
and a next-weight read from the same slice. The compiler should allocate SRAM
by lifetime and concurrent access:

- keep weight reads away from activation reads during Compute;
- keep next-buffer weight reads away from final result-write slices;
- reserve separate stream ranges for simultaneously active VXM passes;
- model every MEM slice as a one-operation-per-cycle resource.

This bank-aware allocation should be part of offline instruction generation,
not a collection of additional fixed delay constants.

## Recommended Implementation Order

1. Add a resource-calendar/list scheduler and remove the global `phase_start` barrier.
2. Overlap completed P-block transposes with the remaining QK/softmax tail.
3. Add bank-aware MEM and stream-ID allocation.
4. Pipeline softmax passes with separate stream ranges.
5. Partition remaining shared VXM ALUs by hemisphere.
6. Coalesce adjacent same-route SXM blocks at `II=4`; insert a three-cycle
   drain only when the Permute destination changes.

## CModel Runtime Optimization

Hardware cycle optimization and simulator wall-time optimization are separate.
After schedule work, the CModel can be accelerated by storing NOP spans as
`next_issue_cycle` values, fast-forwarding globally idle intervals, and scanning
only active SR streams and units. These changes preserve cycle results while
avoiding a full 44-slice, 64-stream state scan on every empty tick.

## Exact Schedule Visualization

`smollm2_attention_test` can export the actual offline ICU schedule as CSV by
setting `FTLPU_SCHEDULE_TRACE`. `FTLPU_SCHEDULE_TRACE_ONLY=1` stops after queue
construction, so regenerating a diagram does not run the full numerical model.
The standard detailed view is
[`smollm2_attention_schedule_detail.svg`](smollm2_attention_schedule_detail.svg).

The SVG expands eight representative windows:

- cycles 0..204: the first Q projection reduction;
- cycles 3348..3432: Q RoPE FP32 rotation, FP16 cast, and MEM writeback;
- cycles 32320..32490: V projection FP16 cast and direct packed-x16 writes;
- cycles 33148..33792: four-MXM direct Q IW and QK score production;
- cycles 37440..37880: pipelined softmax P1 max, P2 exp/sum, and P3 normalize/cast;
- cycles 52288..52692: post-softmax packed P blocks launched at `II=4`, followed
  by the final three-cycle SXM tail;
- cycles 52692..52980: direct V `IW(column)`, including the three-cycle SXM
  drain before changing the Permute destination from MXM0 to MXM1;
- cycles 63108..63420: the first o_proj reduction window.

Accumulator lanes are placed immediately after their hemisphere's MXMs. Purple
bars retain a partial sum in SRAM; red bars send the final sum to the stream and
clear the accumulator.

Each bar is generated from an actual scheduled instruction rather than a
hand-authored phase estimate. Hovering a bar exposes its exact queue and operand
details. `scripts/render_schedule_trace.py` accepts custom windows for debugging
other overlaps or collisions.
