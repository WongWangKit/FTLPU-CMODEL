# FTLPU VMODEL architecture

> Hardware reference from FTLPU-VMODEL. For CModel interfaces and validation, see [RTL alignment](rtl_alignment.md).

This RTL derives its architectural constants, instruction encodings, and
cycle contracts from the sibling `FTLPU-CMODEL` repository's
`refactor/cmodel-architecture` branch.

## Fixed configuration

| Item | RTL value |
| --- | ---: |
| Tile rows / lanes per tile | 4 / 8 |
| Physical vector | 32 bytes |
| Hemispheres | 2 |
| Byte streams | 32 east + 32 west |
| MEM slices | 52 per hemisphere, 104 total |
| MEM groups / stream-register columns | 13 / 15 |
| SRAM geometry per MEM slice | 65,536 x 32 bytes |
| MXM | two 32 x 32 arrays per hemisphere, four total |
| VXM | one four-row VXM, 8 heterogeneous local ALU queues plus 1 global configuration queue |
| SXM | one per hemisphere |
| ICU address slots / physical queues | 138 / 131 |

The physical topology is:

```text
MXM.W[0:1] <-> SXM.W <-> MEM.W(52) <-> VXM <-> MEM.E(52) <-> SXM.E <-> MXM.E[0:1]
```

## ICU queue map

The schedule-loading interface is uniformly 416 bits so that the largest,
13-word SXM packet fits without a side channel. Physical VXM FIFO memories use
only their architectural 5/6/7/15-bit payload widths.

| Queue indices | Consumer | Payload bits |
| --- | --- | ---: |
| 0..103 | MEM | 47 |
| 104..105 | MXM weight load, one queue per hemisphere | 48 |
| 106..107 | MXM BF16 dequant scale, one queue per hemisphere | 16 |
| 108..109 | MXM compute / accumulator read, one queue per hemisphere | 48 |
| 110..111 | Reserved | - |
| 112 | VXM local Q0 | 6 |
| 113 | VXM local Q1 | 5 |
| 114 | VXM local Q2 | 7 |
| 115 | VXM local Q3 | 5 |
| 116 | VXM local Q4 | 7 |
| 117 | VXM local Q5 | 5 |
| 118 | VXM local Q6 | 7 |
| 119 | VXM local Q7 | 5 |
| 120 | VXM global configuration | 15 |
| 121..127 | Reserved | - |
| 128..129 | SXM transpose | 416 |
| 130..131 | SXM permute | 416 |
| 132..133 | secondary MXM weight load, one queue per hemisphere | 48 |
| 134..135 | secondary MXM BF16 dequant scale, one queue per hemisphere | 16 |
| 136..137 | secondary MXM compute / accumulator read, one queue per hemisphere | 48 |

Schedules may be preloaded while `run_i=0` or refilled while execution is
active. Each queue independently accepts a simultaneous enqueue/dequeue so
long schedules do not require layer-sized instruction SRAMs. `NOP` and
`Repeat` use the same 32-bit encoding. MEM queues additionally apply the signed
12-bit Repeat stride to SRAM row address bits `[30:15]`.

## Implemented RTL boundary

- Architectural constants and MEM/MXM/VXM/SXM instruction decoders.
- 131 independent physical ICU queues in a stable 138-slot address map,
  including runtime refill, NOP, and Repeat timing. VXM slots 121..127 are
  unimplemented reservations.
- South-to-north four-tile control pipelines.
- A byte-stream register stage with aggregate broadcast consumption and
  conflicting-producer detection.
- A synthesizable tile-local MEM SRAM slice supporting Read, Write, and
  ReadWrite; Gather/Scatter deliberately fault because the C model does not
  implement their address-stream datapath.
- Four-tile MEM-column composition and MXM/VXM control-wave blocks.
- Two 52-column MEM hemispheres with 14 MEM boundaries, passive one-hop
  East/West propagation, broadcast consumption, and collision detection.
- One physical SXM per hemisphere, including four tile-local FP16-byte
  Transpose banks, northbound capture control, complete-block Permute maps,
  registered passive boundary-13/14 bypass, and sticky encoding/collision
  faults.
- Two physical MXMs per hemisphere split into control-wave, Direct16/INT8
  weight buffers, vector/Block8 compute and accumulation, FP32 result emission,
  and stream-bridge modules. Local MXM 0 and 1 own distinct weight-stream
  windows while sharing activation streams.
- A top-level schedule-loading interface wired through ICU, MEM control waves,
  SRAM slices, and the MEM stream fabric.
- A C-model vector generator plus VCS end-to-end comparison for mirrored
  `MEM Read -> passive SR hops -> MEM Write` transfers.
- A second C-model generator and VCS comparison for a 16-stream, four-tile
  `MEM Read -> SXM Transpose -> identity block Permute -> MEM Write` transfer.
- A full 32x32 FP16 wavefront regression with four overlapping Transpose
  captures, MEM Repeat address strides, seven distinct cross-tile Permute maps,
  and comparison of all 1,024 output elements against C-model SRAM state.
- One four-Tile VXM between the two hemispheres, with global-direction input
  selection, opposite-side result routing, and passive boundary crossing.
- Eight heterogeneous compact local-control waves plus one independently
  committed global configuration wave.
- A position-parameterized 32-bit ALU interface with FP16, BF16, and FP32
  Basic and Special arithmetic.
- FP16 Bypass, Add, Subtract, Multiply, Negate, Max, Exp,
  Reciprocal, and Rsqrt units. Basic operations have one-cycle latency except
  Multiply at two cycles; special operations use five arithmetic stages plus
  a variable single-port-SRAM arbitration wait.
- FP32 Bypass, Add, Subtract, Multiply, Negate, and Max with the same Basic
  timing contract.
- FP32 Exp, Reciprocal, and Rsqrt with FP16 LUT coefficients widened before
  FP32 address/interpolation arithmetic.
- BF16 Basic and Special operations with FP32 internal arithmetic and RNE
  narrowing at each ALU result; the Special LUT remains FP16.
- C-model/VCS `MEM -> MXM -> MEM` comparisons using non-symmetric 32x32 BF16
  and FP16 permutation matrices. Each format loads and computes both weight
  buffers with distinct permutations and signed activation sets. Together the
  regressions cover sixteen Direct16 IW waves, sixteen four-tile accumulation
  sequences, sixteen staggered output column blocks, and all 128 FP32 results
  while detecting format, buffer-selection, and lane/column/block errors.
- A C-model/VCS Column Direct16 regression loads all 32 weight columns through
  two East streams, verifies delayed cell validity, and checks a non-symmetric
  32x32 BF16 permutation across all FP32 outputs.
- A C-model/VCS INT8 dequant regression pairs the load and dequant queues,
  checks positive and negative quantized weights with four BF16 scales, and
  compares all 32 FP32 outputs.
- A module-boundary MXM negative regression checks that missing/mispaired
  dequant instructions and overlapping compute waves assert sticky
  faults without consuming streams.
- A C-model/VCS Block8 regression consumes sixteen BF16 activation streams,
  computes eight rows by 32 columns, accumulates four tile contributions in a
  dedicated wide accumulator, checks all 256 BF16 stream results, and then
  checks all 256 FP32 values during retained read, read-clear, and
  zero-after-clear after an SRAM-destination compute.
- A C-model/VCS accumulator regression covers SRAM-destination compute,
  four-bank FP32 accumulation, continuous row progression with row stride 3,
  a new-wave row reset, retained AccumulatorRead, read-clear, and
  zero-after-clear across all 32 columns.
- A three-phase C-model/VCS SmolLM2 FFN regression carries actual RTL
  intermediates through SRAM from dual-MXM INT8-dequant Block8 gate/up, through
  the six-stage BF16 SwiGLU VXM program, into the INT8-dequant Block8 down
  projection, comparing gate/up, SwiGLU, and final FP32 images independently.

### MXM module boundaries

The compiler-facing timing, input and bank-scheduling obligations are collected
in [the MXM scheduling contract](mxm_scheduling_contract.md), including the
distinction between hardware guarantees and compiler padding policies.

The standalone parameterized BF16 Supercell is implemented separately from
the legacy MXM below. It uses two weight register buffers, a 64-bit common-
exponent CSA dot-product pipeline and three-stage FP32 upward addition.
Activation broadcasts to all local output groups simultaneously and advances
right through one link register per cell; local group outputs are aligned.
Weight tails also cross one register per cell with data/valid/buffer ID together,
so each cell can start replacing its retired buffer on the following edge.
Continuous 32-activation tiles now overlap 32-beat loads of four distinct weight
tiles without compute bubbles; a 40-activation test also keeps compute continuous
while weight inputs idle. MATRIX and COLUMN use the same registered weight
shift chain for MATRIX_N beats or one beat. Column addressing and selective
group writes are removed; the input sequence determines placement. See
[`mxm_supercell.md`](mxm_supercell.md) for ports, numerical rules, cycle
conventions and executable regressions. The new `lpu_mxm_frontend` now connects
the physical array to count-free queue-specific instructions: 7-bit compute
(6-bit activation stream, 1-bit weight buffer), 2-bit load (buffer, mode; byte sources fixed in configuration),
and a default 13-bit independent backend word (BF16 format, 10-bit explicit
address, 2-bit operation). Backend width is 3+clog2(DEPTH), with even DEPTH in
2..65536. Every compute/backend launch describes one beat; repeated equal words
require repeated valid launches. There is no encoded length, repeat loop,
per-instruction compute index or backend base+index address generator.
The compiler retains groups of at least 32 continuous compute beats and even
complete in-place accumulation passes, including the specified bank phase.
Both frontend streams relay through two intermediate registers plus the next
local register per Superlane: three-cycle latency with II=1. Valid bubbles and
16-bit per-beat transaction tags travel with the word; no forwarding timers remain.
MATRIX retains its fixed MATRIX_N-beat INT8 loading controller (32 by default),
COLUMN one beat; consecutive COLUMN launches are supported. A third independent
queue in `lpu_mxm_backend` relays one beat per cycle, one cycle between Superlanes.
It controls DIRECT_OUT/STORE/ACC_STORE/ACC_OUT, two 1RW 512x256-bit banks per
output position at default depth, actual bank/dependency detection and five-cycle
read-to-write latency. `lpu_mxm_core` connects the array and backend through a static
down/left `lpu_mxm_result_return` fabric. In a square grid, top column c
descends to bottom-indexed row c, turns left, and enters that Superlane's
local accumulator. Downward and leftward streams have separate registers
at crossing cells. Each path has ROWS registers (four by default), retaining
the one-cycle inter-Superlane phase without added alignment waits. Both output opcodes
select FP32 or BF16 RNE using the highest backend bit (default bit12). A shared registered output stage
gives all four combinations the same six-cycle launch-to-output latency.
Commands launch two cycles before left-edge returned data availability: SRAM
is prefetched, and the fresh returned bus directly enters A1 with SRAM B. Three wide
new-partial waiting registers have been removed; left-edge availability to
output is four cycles, and top-array availability to output is eight by default. STORE uses the existing three-stage passthrough so all writes
share the accumulation writeback phase and retain five-cycle read-to-write timing.
BF16 occupies the low half of each 32-bit core lane with a format sideband;
final stream packing/routing and the legacy `lpu_top` migration remain separate. See
[`mxm_instructions.md`](mxm_instructions.md).

New frontend modules are `lpu_mxm_isa_pkg`, `lpu_mxm_command_queue`,
`lpu_mxm_instruction_wave`, `lpu_mxm_control`, `lpu_mxm_dequant_control`,
`lpu_mxm_weight_dequant_pipe`, and `lpu_mxm_frontend`;
new backend modules are `lpu_mxm_backend_isa_pkg`, `lpu_mxm_accumulator_bank`,
`lpu_mxm_backend_column`, `lpu_mxm_output_pipe`, `lpu_mxm_backend`,
`lpu_mxm_result_return`, and `lpu_mxm_core`.
The new frontend dequantizes INT8, FP8 E4M3FN and packed FP4 E2M1 at each row
ingress. The five-bit parameter-load queue selects scale mode, zero mode and the
destination parameter buffer. The three-bit dequant-execute queue selects the
weight format and source parameter buffer. D1 snapshots the operands and the
existing weight register captures BF16 at D5. The five queue payload widths are
7/2/5/3/13 by default. All four frontend
streams retain the three-cycle row interval. See [mxm_weight_ingress.md](mxm_weight_ingress.md).
No Block8 mode, per-cycle vertical instruction shift or result-feedback path
exists in the new frontend. The following table describes the
legacy compatibility datapath only; its old control is explicitly isolated
under `rtl/mxm/legacy/` because `lpu_top` and existing C-model schedules still
use it. Shared control pipelines used by MEM/SXM/VXM are not removed.

| Module | Hardware responsibility |
| --- | --- |
| `lpu_mxm_legacy_control` | Compatibility-only per-cycle load, dequant, and compute propagation for the old slice. |
| `lpu_mxm_dequantizer` | Converts signed INT8 weights times a BF16 scale into BF16 with round-to-nearest-even. |
| `lpu_mxm_weight_buffer` | Consumes full-cell or per-column Direct16/INT8 East streams and owns two buffers of four-by-four 8x8 weight cells plus inner-column validity. |
| `lpu_mxm_dot_bank` | Implements the 32 parallel eight-element FP32 dot products for one active physical tile. |
| `lpu_mxm_compute` | Selects the active tile, runs one or eight dot-bank rows, accumulates four tile contributions, and emits Vector or Block8 transactions. |
| `lpu_mxm_accumulator` | Architectural Vector reference backend used by the default regression build. |
| `lpu_mxm_block_accumulator` | Architectural Block8 reference backend used by the default regression build. |
| `lpu_mxm_shared_accumulator` | Selects the cycle-compatible reference backends or the physical Vector-only SRAM backend. |
| `lpu_mxm_accumulator_sram` | Composes two 512 x 128 single-port macros into one 512 x 256 row bank. |
| `lpu_mxm_slice` | Composes control, weight storage, compute, passive stream routing, collision detection, and sticky faults. |

The functional MXM subset supports full-supercell and Column Direct16 IW,
full-supercell and Column INT8-to-BF16 dequant IW, Vector and Block8 Compute,
FP16/BF16 input, Stream or SRAM destination, nonzero row stride, retain/clear,
Vector and Block8 AccumulatorRead, FP32 accumulator stream output, and BF16
Block8 compute stream output. Overlapping compute waves assert `mxm_fault_o`. This keeps
unsupported behavior explicit while preserving the current 48-bit C-model
encoding.

The RTL is structurally synthesizable: state uses bounded registers/SRAM-style
arrays, loops have static bounds, and no simulation-only timing constructs are
present under `rtl/`. The MEM tile has two explicit implementations selected
by `USE_SRAM_MACRO`. The default behavioral path preserves the C-model's
zero-latency reads. The physical path uses a synchronous 1RW abstraction and
banks a 64-bit x 65,536-row tile across 32 ARM `sram_64_2048` macros. A MEM
Read returns one cycle after issue; ReadWrite performs the read first and the
write on the following cycle because the macro is single-port. Design Compiler
L-2016.03 maps the surrounding logic to TSMC28 cells and preserves all 32 macro
instances; `scripts/dc_mem_macro.sh` checks that count before and after compile.

Design Compiler L-2016.03 maps the physical Vector accumulator to sixteen ARM
`sram_128_512` macros: eight independently selected 256-bit row banks, with two
128-bit macros per bank. One DesignWare FP32 adder is reused across the eight
lanes. At TT 0.9 V/25 C the isolated target has 522,018.7 um2 cell area,
including 491,773.8 um2 of SRAM macro area, and a 2.37 ns critical path under a
10 ns clock constraint. `scripts/dc_acc_macro.sh` checks the macro count before
and after mapping and writes all reports under `build/dc_acc_macro/`.
Accumulator capacity is configured at `lpu_top` with
`MXM_ACCUMULATOR_BLOCK_COUNT`, where one block is one complete 32x32 FP32
partial-sum tile. The default 32 blocks derive 1024 Vector rows or 128 Block8
rows, and both layouts therefore contain 128 KiB per MXM. The 13-bit instruction
address limits the parameter to 1..256 blocks. The default reference build
represents the Vector accumulator as four `(block_count * 32)` x 256-bit
segment banks and Block8 as four `(block_count * 4)` x 2048-bit banks. The
macro build instead uses one shared 128 KiB physical store for Vector data,
performs a post-reset zero sweep, and serializes its read-modify-write
operation. Block8 is intentionally unsupported in this backend. The full macro
script additionally defines `FTLPU_DISABLE_BLOCK8`, reducing the MXM compute
array from eight dot rows to one and making Block8 instructions fault rather
than allocating unused adders.

### VXM module boundaries

| Module | Hardware responsibility |
| --- | --- |
| `lpu_vxm_control` | Advances eight heterogeneous local words and the global configuration through four Tile rows, one row per cycle. |
| `lpu_vxm_input_converter` | Performs supported chain-head conversions among FP16, BF16, and FP32 while preserving a stable 32-bit container. |
| `lpu_vxm_alu` | Validates format/opcode, dispatches Basic versus position-specific Special execution, and detects result collisions. |
| `lpu_vxm_float_unpack` | Combinationally extracts the selected format's sign, exponent, fraction, and class; arithmetic fields apply DAZ while raw-bit operations bypass the module. |
| `lpu_vxm_basic_opcode_decode` | Decodes the six Basic opcodes in parallel with operand unpack and produces one-hot operation enables. |
| `lpu_vxm_basic_alu` | Executes FP16/BF16/FP32 Bypass/Add/Subtract/Multiply/Negate/Max with one/two-cycle timing. ADD/SUBTRACT/MAX share one comparator front end, and all formats share the segmented adder and multiplier. |
| `lpu_vxm_segmented_addsub` | Implements one 27-bit shared significand add/subtract datapath as carry-connected 14-bit and 13-bit groups, plus a carry output. BF16/FP16 isolate the high group; per-format operand masking selects 11/14/27 active bits. |
| `lpu_vxm_grouped_lzc` | Encodes leading zeros from one format-masked 27-bit subtraction result using four 7/7/7/6-bit groups (low to high); only the near-subtraction path enables it. |
| `lpu_vxm_shared_float_adder` | Consumes the common unpackers' fields and Zero/Inf/NaN flags without reclassifying raw bits; selects FP16/BF16/FP32, isolates special/idle operands, uses 7-bit or 10-bit signed working exponents, and shares the 14+13 add/subtract hardware. Near subtraction uses the grouped LZC; far subtraction needs at most one left shift. The variable shifter remains behavioral before RNE/FTZ packing. |
| `lpu_vxm_mul8x8` | Defines one operand-isolated 8x8 unsigned multiplier boundary while leaving its internal implementation to synthesis. |
| `lpu_vxm_significand_multiplier` | Builds a shared 24x24 significand multiplier from nine gated 8x8 blocks; BF16 enables one block, FP16 up to four, and FP32 up to nine. |
| `lpu_vxm_shared_float_multiplier` | Left-aligns FP16/BF16/FP32 significands, drives the shared block multiplier, performs FP32-style normalization/RNE, and packs the selected result format. |
| `lpu_vxm_shared_float_compare` | Consumes the common unpacked fields and implements shared magnitude/ordered relations: FP16/BF16 use the low 15-bit layer and FP32 adds a high 16-bit layer. MAX result semantics remain in the Basic ALU. |
| `lpu_vxm_lut_storage` | Provides the legacy/standalone configurable LUT used by isolated ALU tests. |
| `lpu_vxm_lut_sram` | Implements one private single-read function/Lane SRAM with 64 rows; each row holds a complete coefficient pair or Log entry (26/32/50 bits). |
| `lpu_vxm_tile_lut` | Implements one private Tile/Superlane LUT set: one single-read SRAM for every special-function/Lane pair, with one-cycle Stage-tagged return and no cross-Superlane read sharing. |
| `lpu_vxm_special_alu` | Performs range reduction, waits for its tagged external LUT response, interpolates `k*dx+b`, and restores the result exponent. |
| `lpu_vxm_fp16_pkg` | Implements FP16 classification, FTZ, RNE arithmetic, division, square-root, reciprocal, and rsqrt helpers. |
| `lpu_vxm_math_pkg` | Retains reusable format-conversion and FP32 helpers shared by the remaining RTL. |
| `lpu_vxm_stream_bridge` | Arbitrates external East traffic, unconsumed cross-hemisphere West traffic, and active VXM result producers. |
| `lpu_vxm_slice` | Composes four execution Tiles into the single VXM and applies the global direction bit to its boundary-input MUXes and result routes. |
| `lpu_vxm_global_config` | Holds shadow/current global VXM state and commits it only at a safe configuration boundary. |

The ALU uses a 32-bit data container and a 2-bit format selector. FP16, BF16,
and FP32 Basic and Special operations share the interface. The reserved format
does not issue and raises `unsupported_format_o`; unsupported position/opcode
combinations raise `illegal_opcode_o`. Arithmetic uses deterministic
RNE/DAZ/FTZ handling and a canonical quiet NaN. Bypass preserves the selected
format's raw bits, while Negate changes only its sign bit.

Basic multiplication uses one explicitly segmented significand multiplier for
all three formats. The 24-bit operands are divided into three 8-bit chunks.
Format gating plus zero-chunk operand isolation activates only `P22` for BF16,
up to `P11/P12/P21/P22` for FP16, and up to all nine blocks for FP32. Each 8x8
leaf still uses the synthesizable `*` operator, so the target synthesis flow
chooses its internal Array/Booth/library implementation.

Basic comparison is also explicitly hierarchical. One 15-bit comparator
covers the complete signless FP16/BF16 encoding and the low portion of FP32;
FP32 first compares `[30:15]` and consults the shared low layer only on a tie.
The resulting magnitude order selects the larger operand for floating-point
addition/subtraction alignment and also drives signed MAX selection, avoiding
three independently inferred comparators in each physical ALU.

The Special path uses function ID 0 for Exp, 1 for Reciprocal, 2 for
Rsqrt, and 3 for Log. Every Tile/Superlane owns an independent LUT set;
each of its eight Lanes has one single-read SRAM per function. With Log,
there are 32 SRAMs per Superlane and 128 per four-Superlane Slice. All SRAMs
have 64 rows: Exp uses 26-bit rows, Reciprocal/Rsqrt use 32-bit rows, and Log
uses 50-bit rows. Programmable FP16 `input_min` and `segment_width` state
is replicated across each function's eight Lane SRAMs.
Superlanes can access the same function and Lane concurrently with unrelated
addresses. Each private SRAM returns its physical-Stage tag in one cycle;
there is no cross-Superlane ownership selection or collision. Two Stages
within one Superlane requesting the same function/Lane in one cycle still
constitute a protocol fault that the compiler schedule must prevent.

The compute unit clamps the derived address, consumes the tagged `{k,b}`
response, and evaluates `k*dx+b` before exponent/sign restoration. LUT-free
special cases retain the base pipeline latency; a lookup adds only its
arbitration wait. In BF16 and FP32 modes, `input_min`, `segment_width`, `k`, and `b`
are widened from FP16 and all address/interpolation arithmetic remains FP32;
BF16 narrows the final ALU result with RNE. The
default SRAM depth is 64 and is parameterized. One Slice LUT programming port
broadcasts identical contents into all four private Superlane sets and all
eight Lane replicas of the selected function. Only programming is broadcast;
read addresses, SRAM storage, and response tags are independent.

The active ICU-to-Slice control path separates one 15-bit global queue from
eight physical local FIFOs with 6/5/7/5/7/5/7/5-bit payloads.  The local words
and the committed global word advance through four Tile rows. Each Tile stores
its resident configuration. Queues 121..127 are no longer instantiated. The
former 16 x 128-bit control path is not active. Compact local decode, fixed
stream operand selection, 16-stage ALU chaining, and FP16/BF16/FP32 result
serialization are connected at Tile level; the Slice connects all four Tiles
to the two boundary hemispheres.

The 15-bit global configuration begins with a 1-bit `flow_direction`, followed
by 2-bit `chain_length`, `active_width`, and `compute_dtype` fields and
independent 2-bit read-width and dtype fields for LHS and RHS. The direction
bit selects the left or right input boundary and routes results to the opposite
hemisphere. Chain-head conversion is selected from source and compute dtypes
without a redundant conversion-enable bit.

`lhs_read_bits` and `rhs_read_bits` encode 16-bit one-beat or 32-bit two-beat
collection independently. FP16/BF16 use one beat. FP32 uses little-endian
phase order: low 16 bits
first and high 16 bits second over the same fixed stream pair. The assembled
operand and all internal/feedback/accumulator tokens remain 32 bits. FP32 tail
traffic is serialized over the corresponding result pair in the same order.
The collector retains a complete operand through the following execute cycle,
giving repeated FP32 traffic a low/high/execute cadence without overwriting a
resident operand.

VXM unit tests are separated by boundary. `lpu_vxm_icu_map_tb` checks all
eight heterogeneous local FIFOs plus the global FIFO in the physical ICU, and
`lpu_vxm_control_tb` checks four-Tile local/global propagation.
The primary ALU acceptance tests are organized by physical position under
`sim/tb/vxm/alu_positions`. Each of the 16 tops runs the shared execution-stage
checker, so the position-specific compact decoder, legal operation set, source
MUX, chain role, data format, and arithmetic result are verified together.
`lpu_vxm_alu_tb` remains the lower-level combined arithmetic-core regression,
while `lpu_vxm_16_alu_tb` runs all physical positions concurrently.

The full C-model `TspSliceSystem` currently constructs its MEM region through
`TileArrayModel::LegacyLocalLinear()`: a MEM Read injects at the slice group's
input boundary. The standalone `MemArrayModel` defaults to downstream-boundary
placement. RTL follows the full-system mapping because existing workloads and
their `read_latency()` schedules use it. This distinction is captured in code
comments and in the generated East/West regression.

The remaining MXM modes are subsequent implementation stages. VXM subnormal
values intentionally use the C-model FTZ policy. The Special ALU and
programmable LUT storage remain separate modules; Tile integration encodes the
requesting Stage, while Slice integration instantiates private per-Superlane,
per-Lane SRAMs with tagged one-cycle response routing.
SXM ShiftSelect and Distribute are
intentionally still control/ISA-only. Their interfaces should consume the
already stable ISA and control-wave blocks; no new encoding should be
introduced without first updating the C model codec tests.

For an isolated MEM/SXM beat in the current C-model topology, the generator
uses `capture_cycle - (14 - group)` for East reads and
`capture_cycle + (14 - group)` for West writes. This accounts for the
next-state commit at the MEM boundary before SXM can observe or MEM can consume
the value. The generator runs and verifies the C model before publishing any
RTL vectors.

## Cycle rules

1. Instructions dispatch from ICU queues on a rising edge.
2. Control pulses enter tile 0 and advance one tile per rising edge.
3. Stream stages read current inputs and publish next-state outputs on the next
   rising edge.
4. Multiple consumers must OR their consume bits before driving a stream stage;
   this preserves the C model's broadcast semantics.
5. Two different producers targeting the same next-state cell assert
   `conflict_o`.
6. Transpose control advances through four tiles; Permute is a single physical
   instruction and may read a captured tile only after one full cycle.
7. When a wavefront reuses one Transpose tile in the same cycle, Permute reads
   and releases the old bank contents before the new Transpose capture commits.
8. Passive traffic crosses the SXM boundary-14 register in one cycle. MXM
   Direct16/activation consumers and returning FP32 results are scheduled
   against this physical hop.

For small simulations, `lpu_top.ACTIVE_MEM_COLUMNS` can elaborate a prefix of
the 52 physical columns in each hemisphere while preserving architectural queue
numbering. Its synthesis/default value is 52.

### C3 Log extension

Log is available at physical stages C3/C7/C11/C15 in every chain mode. Q3
and Q7 grow from five to six physical instruction bits. The canonical Log
word is `6'b100110` (`0x26`): bit 5 extends the opcode and bits 4:3 must be
zero. All old five-bit words keep their meanings. Noncanonical extension
words are illegal. The local decoder exports the four-bit semantic opcode
`VXM_LOCAL_LOG=8`; Basic opcodes and operand-source encodings are unchanged.
ICU queues 115 and 119 and their control waves use the package width helper,
so the extension is retained through the physical FIFO and mirrored decode.

The Slice has four LUT functions. Bank 3 is Log; each Superlane has
eight additional independent single-read Lane SRAMs. A Log row is exactly
50 bits: `{base_uq0_28[27:0], inverse_uq1_21[21:0]}`. The 64 anchors are
`c=1+address/64`, `base=RNE(ln(c)*2^28)`, and
`inverse=RNE((1/c)*2^21)`. Program bank 3 with input_min `0x3c00` and width
`0x2400` (1/64). `lut_write_log_i[49:0]` programs a complete row atomically,
using the existing write-valid/bank/address; k/b are ignored for bank 3.
Legacy banks retain their 16+16 programming interface and physical row widths.
The Log response has its own 50-bit container and the same Tile/Stage tag.

The common unpackers supply DAZ-effective sign/exponent/fraction and
Zero/Inf/NaN flags. Log performs no second source classification. Positive
normal inputs use the high six fraction bits as address and the remaining
bits as the exact residual. A missing table reports a fault and returns NaN;
special values and the near-one polynomial bypass the SRAM.

BF16 activates 14 base fraction bits and 9 inverse fraction bits; FP16 uses
15 and 12; FP32 uses all 28 and 21. Narrow coefficients use RNE, not simple
truncation. One shared multiply demand point reuses C3's first Reciprocal
Newton multiplier. A phase-selected Horner adder evaluates one/two/four
Taylor terms. Narrow modes bypass unused phases. The base remains fixed
point through a signed 48-bit Q40 exponent/base/polynomial sum, with one RNE
normalization before final format packing. It is not first rounded to FP32.

For FP32/FP16 inputs in [63/64,65/64], and BF16 inputs in
[127/128,129/128], use the same polynomial with `u=x-1`, base=0 and e=0.
This avoids cancellation near one. NaN/negative nonzero map to canonical NaN,
DAZ-effective zero maps to -Inf, and +Inf maps to +Inf. Special values cross
the same phases. The single-outstanding contract remains; without LUT wait,
the accepted edge plus seven following edges gives the existing eight-cycle
external Special latency. Adding Log does not make the ALU II=1.

`sim/vectors/vxm_log_lut.hex` and `vxm_log_cases.hex` are regenerated by
`python3 sim/cmodel/generate_log_vectors.py`. Run the full low-format
numerical sweep with `python3 sim/cmodel/check_log_low_precision.py` and the
cycle-level VCS regression with `bash scripts/vcs_vxm_log.sh`. The test checks
all C3-class decode positions, chain modes, the 50-bit SRAM read path, native
format results, special values, eight-cycle timing, metadata and reset.
