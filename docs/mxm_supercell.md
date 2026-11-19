# Parameterized BF16 Supercell

The compiler/RTL timing and input obligations are defined in
[MXM scheduling contract](mxm_scheduling_contract.md). This document describes
the Supercell datapath, arithmetic and implementation-level timing.

The new `lpu_mxm_supercell` is standalone and is also instantiated by the new
`lpu_mxm_frontend` array. The legacy `lpu_mxm_slice` and `lpu_top` are not yet
migrated to the new accumulator/stream backend. Defaults live in
`rtl/mxm/lpu_mxm_config_pkg.sv`: four-by-four Supercells, eight input Lanes,
eight output groups, BF16 operands, and FP32 result/accumulator storage.
The existing accumulator capacity stays 32 FP32 elements by 1024 addresses.

In the complete core, top results no longer connect directly to the backend.
`lpu_mxm_result_return` provides separate downward and leftward registered
streams at positions r>=c. Column c turns at bottom-indexed row c and exits
left into that Superlane's accumulator. A square ROWS-by-ROWS grid adds ROWS
cycles to every path, preserving the one-cycle inter-Superlane result phase.
The compute Supercell and its upward arithmetic are unchanged. These return
registers are in a separate coordinate-indexed fabric so standalone arithmetic
tests retain their existing interfaces; physical co-placement remains a layout task.

## Dataflow and interfaces

Each valid input is a vector of `TERMS` BF16 activations, a weight-buffer bit,
and a `TAG_WIDTH`-bit transaction label. Group `n` computes
`dot[n] = sum_k activation[k] * weight[buffer][n][k]`.
The input vector, buffer selection, label and validity broadcast to all
`OUTPUTS` groups on the same acceptance edge. Each group's M1 captures its
products using its own weights on that edge. There is no group-to-group
activation shift chain or extra activation-loading stage. Every local dot
group for a valid configured vector becomes valid together; merged groups
also become valid together when all lower partials satisfy the protocol.
An independent single link register forwards the vector, buffer selection,
label and validity to the right neighbor each edge, regardless of `OUTPUTS`.
The default horizontal chain of four cells takes four link cycles to traverse.
Different cells retain their one-edge phase offsets; no horizontal output
alignment is required. Four link cycles do not mean four-cycle dot latency.
All vectors and outputs are packed with element/group zero in the low bits.

Every group has its own `TERMS` multipliers. An 8x8 cell therefore has 64
8x8-bit significand multipliers. It does not replicate these for Block8 mode.
There are two private weight shift-register sets and parallel compute taps.
The matrix/one-column load instructions and their control implementation are
specified in [mxm_instructions.md](mxm_instructions.md). Both use the same shift
input: MATRIX supplies MATRIX_N beats and COLUMN supplies one. There is no
addressed-group write, mode input or column-address port on the Supercell.
No simultaneous same-buffer compute/write is permitted on the M1 capture edge.
An accepted write supplies `TERMS` BF16 weights to group zero of the selected
buffer and shifts its previous contents toward group `OUTPUTS-1`. The other
buffer holds. There is no group-address input. Weight data is not reset;
per-group validity is reset.

On that shift edge, the old tail, valid and buffer ID are captured together in
one link register. `weight_forward_{valid,buffer,data}_o` is consumed by the
right neighbor on the NEXT edge, matching activation's one-edge cell link.
Data transfers only when the tail is valid. `LAST_COLUMN=1` disables forwarding
and discards the displaced tail. A single cell fills after `OUTPUTS` accepted
shifts of that buffer; a chain of C cells requires `C*OUTPUTS` source shifts,
with the last cell finishing C-1 clock edges after the last source shift.
Send the rightmost cell's block first, and within a block send group
`OUTPUTS-1` first and group zero last. The source load duration remains 32
beats by default. Compute may start on the left before the last cell finishes:
activation arrives at each cell with the SAME one-edge offset.

Each local chain holds when its own write-valid is low; in-flight link slots
continue to the next cell. Buffer IDs can be interleaved: each
buffer advances only on its own accepted writes. The outgoing buffer ID
identifies the selected private chain, so the displaced weights and ID always
match because data/ID/valid are registered as one transaction; bubbles travel
as invalid link slots. Refill of an already valid chain also forwards old
tails, but their replacement now advances one cell per edge rather than
modifying all neighboring cells on the same edge.

Continuous default ping-pong loading uses source beats E1-32/B0, E33-64/B1,
E65-96/new B0, E97-128/new B1. SC c last uses old B0 at E64+c, starts its
replacement at E65+c, completes it at E96+c, and starts computing with new B0
at E97+c. Thus 32 activations per tile hide the 32-beat weight load with no
compute bubbles. The four-cell completion drain is overlapped by the matching
activation links, not charged as a three-cycle compute/load guard.

`weight_buffer_ready_o` means all groups have been written since reset.
It is not a version/atomic-load marker: the compiler must finish each intended
reload before using the bank. Writes to the bank selected by the current
valid activation input are rejected on its M1 capture edge, set sticky
`fault_o`, and leave the weights unchanged. The inactive bank can load during
computation. All groups capture their weights together: after that edge,
arithmetic stages and the outgoing link register do not extend local bank
ownership. A bubble or a different input bank therefore releases the bank
on the next edge; a continuous stream on the same bank keeps it busy.
The compiler schedules each cell's shifted write/compute phases; consecutive
32-beat ping-pong blocks satisfy local ownership automatically. There is
no reverse ready/backpressure link; a receiving-cell conflict faults locally.
MATRIX and COLUMN share the registered tail chain. A following COLUMN shifts
the selected buffer once; it does not discard a preceding MATRIX tail. Already
valid downstream cells may receive displaced old weights, while empty cells
wait until a valid tail reaches them. Source order, not an instruction address,
determines the final contents.

The bottom cell sets `BOTTOM=1` and adds +0 through the same three-register
adder as every other cell. Other cells require each `below_valid_i[n]` to
match the local dot-valid and its transaction label. Misalignment sets
`fault_o` and suppresses that addition; nothing stalls or retries. Labels are
metadata/checks, not dynamic scheduling. `valid` describes occupied pipeline
slots and propagates bubbles. Data outputs are meaningful only when valid.

The local dot result is exposed separately from the upward merged partial.
This cell has no accumulator SRAM, feedback accumulator, result-address
generator, instruction FIFO or output arbitration. Those belong to the full
MXM. Each vertical cell receives a different K-segment; horizontal neighbors
reuse the same activations with different output-column weights.
The agreed two-bank accumulator organization, address mapping and five-cycle
read-to-write contract are recorded in [mxm_accumulator.md](mxm_accumulator.md).

## Numerical contract

* BF16 inputs use DAZ, preserving the input zero sign for classification.
* Complete 16-bit significand products and signed 11-bit exponents are kept
  until reduction. A product is not first rounded or range-clipped to FP32.
* Choose the maximum normalized exponent of the finite nonzero products.
  For eight terms, place its leading-bit weight at bit 59. The common exponent
  is carried separately from the signed 64-bit data words: these are not FP64.
* The default width budget is one sign bit, three growth bits and 60 aligned
  product bits. For another `TERMS`, growth is `ceil(log2(TERMS))` and the
  aligned region is `63-growth`. At least 16 aligned bits are required.
* Align unsigned magnitudes by logical right shift, discarding out-of-window
  bits toward zero, and only then apply two's-complement sign. No jamming bit
  is inserted into the sum. `alignment_lost_o` reports whether any nonzero
  alignment bits were discarded; it is not a rounding correction.
* CSA levels and the final 64-bit carry-propagate add do not round. Normalize
  the signed sum and apply one RNE conversion to FP32. This is correct RNE of
  the retained-window sum, not guaranteed correct rounding of an exact BF16
  dot product across arbitrary exponent spans. Severe cancellation can expose
  discarded terms. A finite exact-zero local dot returns +0.
* NaNs, DAZ-effective zero times infinity, or opposing infinite products
  produce canonical `0x7fc00000`. Other infinite products produce signed Inf.
* FP32 dot outputs and cross-cell addition preserve gradual underflow. This
  is deliberately separate from BF16 input DAZ. There is no BF16 output
  conversion in this cell; its future FTZ policy does not clear FP32 partials.
* Cross-cell arithmetic is FP32 RNE, including signed zeros, subnormals,
  Inf/NaN and overflow. Intermediate FP32 partials retain the previously
  accepted range/rounding limitations between Supercells.

Weight dequantization occurs once at each Superlane ingress; this
standalone cell's write/forward ports and both buffers remain BF16. The D5
normalization/RNE result is captured directly by the cell's existing weight
register, avoiding a separate dequant output register and an extra write cycle.
Scale and zero values are loaded into independent parameter ping-pong
registers. D1 snapshots the selected parameters; subsequent stages never reread those
registers. Parameter-load and per-beat dequant-execute commands have separate
queues, with the same three-cycle row spacing as weight-load commands.
See [mxm_weight_ingress.md](mxm_weight_ingress.md) for byte packing and load timing.

## Registered timing

Every stage has II=1 and fixed latency, including zero, cancellation and
special values. Stage boundaries are actual arithmetic registers, not a
combinational calculation followed by an artificial output delay.

| Stage | Work |
|---|---|
| M1 | BF16 classification, sign/exponent calculation, 8x8 significand multiply |
| M2 | Normalize the complete product, retaining extended exponent |
| R1 | Maximum exponent selection |
| R2 | Common alignment, magnitude truncation and signed conversion |
| R3 | First two CSA levels, 8 -> 6 -> 4 for eight terms |
| R4 | Next two CSA levels, 4 -> 3 -> 2 |
| R5 | Final 64-bit carry-propagate addition |
| R6 | Sign/magnitude, exact leading-zero detection and normalization |
| R7 | RNE, FP32 packing, range and special-case selection |
| A1 | FP32 special-value detection, magnitude ordering, alignment and sticky |
| A2 | FP32 significand addition/subtraction and exact leading-zero detection |
| A3 | Normalization, RNE, range and special-case selection |

The R1 maximum-exponent selector uses a balanced comparator tree (three levels
for eight terms), padded with an inactive sentinel for non-power-of-two sizes.
R6 and FP32 A2 use explicit hierarchical half-word leading-one detection.
No visible stage or rounding rule changes. The cross-cell adder still uses exact
leading-zero detection, not LZA.
Its A2 path includes add/subtract followed by detection. Timing/area at 2 ns or
3 ns has not been synthesized or closed; LZA and further register rebalancing remain
possible optimizations without changing the numerical contract.

If upstream registers launch a vector at edge E0, every group's M1 captures
at E1, all local dot outputs are valid after E9, and all bottom/legally aligned
merged outputs after E12. There is no group-dependent offset. Equivalently,
if **acceptance** is the sampling edge of M1, dot-valid is acceptance+8 and
partial-valid is acceptance+11. This
distinction avoids a one-edge counting ambiguity in scoreboards.

For four horizontal bottom cells (no extra link stages), the same vector has:

| Cell | M1 acceptance / activation link output | Local dot | Merged partial |
|---|---|---|---|
| 0 | E1 | E9 | E12 |
| 1 | E2 | E10 | E13 |
| 2 | E3 | E11 | E14 |
| 3 | E4 | E12 | E15 |

Each table entry applies simultaneously to every output group in that cell.
The link output becomes visible after its edge and is sampled by the next
cell on the following edge. Valid bubbles propagate with exactly the same
per-cell timing; consecutive vectors remain II=1.

Two vertically adjacent cells use a three-edge activation/instruction skew:
the upper cell starts three edges later, so its dot result coincides with
the lower merged partial. No additional inter-cell link register is assumed.
An extra physical link stage must be reflected in the compiler schedule.
For non-default term counts, compressor levels are split over the same two
registered stages; functionality and latency are preserved, not timing closure.

## Verification

Run `python scripts/test_mxm_supercell.py` with Icarus Verilog on PATH or
under `build/iverilog/bin`. `--compile-only` uses pyslang on PATH/import path
or under `build/tools`. `bash scripts/vcs_mxm_supercell.sh` is the VCS runner
for the twelve default-size benches (including output conversion, compact ISA, weight-shift, control, frontend,
backend, result return and core).

The Python integer reference models DAZ, the retained alignment window and
exact integer-to-FP32 rounding without host floating-point arithmetic or a
CSA-tree imitation. It generates 2048 directed/random dot cases for each of
1/3/8/16 terms and 4096 FP32-add cases. Benches check fixed latency, bubbles,
II=1, tags, reset, exponent-window boundaries, range overflow/underflow,
subnormals, RNE ties, cancellation and Inf/NaN.

The integration bench connects a bottom cell, an upper cell with three-edge
skew and a right neighbor. It checks 8x8, 3x3, 1x1 and 3x5 configurations,
different activation segments/weights, simultaneous group output phases, horizontal transit,
serial weight transfer, reverse-group/block ordering, interleaved load-buffer
IDs and bubbles, final-cell forwarding suppression, concurrent inactive-bank
loading, token-local compute-bank switches, missing/mistagged
lower partials, capture-edge bank write rejection, release after capture,
and invalid weights after reset. A separate four-cell horizontal bench sends
58 vectors with bubbles and alternating buffer IDs. It checks every link's
data/control and one-edge transit, rectangular dot/partial valid masks and
same-tag outputs, all four cells' numerical results, and result counts through
pipeline drain. Both integration benches use public DUT ports only.
This is RTL functional verification, not target-model accuracy validation or
physical timing verification.
