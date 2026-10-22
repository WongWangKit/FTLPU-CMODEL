# FTLPU-VMODEL

SystemVerilog implementation and verification workspace for the FTLPU LPU.
The architecture is aligned with the sibling `FTLPU-CMODEL` repository's
`refactor/cmodel-architecture` branch.

## Current status

The repository now contains the first functional RTL baseline rather than an
empty shell:

- exact C-model hardware constants and ISA field layouts;
- 131 independent, runtime-refillable ICU queues in a 138-slot address map,
  with NOP, Repeat, interval,
  and MEM row stride;
- 4-tile northbound control-wave pipelines;
- byte-stream next-state stages with broadcast consumption;
- tile-local, four-tile, and dual-hemisphere MEM blocks with Read, Write, and
  ReadWrite;
- 14-boundary East/West MEM stream fabrics with passive propagation;
- physical SXM Transpose banks and block-Permute datapaths in both hemispheres;
- shared VXM boundary bridge, compact control, and standalone FP16 ALUs;
- four MXM datapaths with Direct16/INT8 weight buffers, BF16/FP16 Vector and
  Block8 compute, narrow and wide accumulator SRAMs, and stream output/readback;
- top-level `MXM_ACCUMULATOR_BLOCK_COUNT` sizing for complete 32x32 FP32
  partial-sum blocks (32 blocks / 128 KiB per MXM by default);
- C-model-generated golden vectors and VCS end-to-end comparison;
- smoke and functional architecture testbenches.

See [docs/architecture.md](docs/architecture.md) for the queue map, topology,
cycle rules, and remaining datapath work.

## Layout

```text
rtl/core/         Constants, ISA decoders, stream/control primitives
rtl/icu/          Generic queue and 138-slot/131-physical-queue ICU
rtl/mem/          MEM tile, four-tile column, and dual-hemisphere fabric
rtl/mxm/          MXM control, weight storage, vector compute, and routing
rtl/sxm/          Transpose/Permute control and byte datapath
rtl/vxm/          VXM control, operand orchestration, ALU, math, and routing
rtl/lpu_top.sv    Static-schedule loading and architectural dispatch top
sim/tb/           Smoke and functional architecture tests
sim/cmodel/       C-model golden-vector generators
sim/vectors/      Checked-in differential-test vectors
scripts/          Linux and PowerShell lint/simulation entry points
docs/             Architecture and implementation boundary
```

## Prerequisites

Install at least one supported open-source tool:

- [Verilator](https://verilator.org/) for lint
- [Icarus Verilog](https://steveicarus.github.io/iverilog/) for simulation

## Quick start

Linux/macOS:

```bash
./scripts/lint.sh
./scripts/smoke.sh
```

PowerShell:

```powershell
./scripts/lint.ps1
./scripts/smoke.ps1
```

VCS (the local O-2018.09 installation needs the linker compatibility flag
already included by the script):

```bash
./scripts/vcs.sh
```

Regenerate only the C-model golden vectors:

```bash
./scripts/generate_cmodel_vectors.sh
```

Run only the C-model-generated RTL SmolLM2 FFN regression:

```bash
./scripts/smollm2_ffn.sh
```

Run the full-size C-model SmolLM2 prefill FFN reference (all build products
remain under this repository's `build/` directory):

```bash
./scripts/run_smollm2_ffn_reference.sh
```

Run the TSMC28/ARM SRAM-macro mapping for one complete 64-bit x 65,536-row
MEM tile slice:

```bash
./scripts/dc_mem_macro.sh
```

This target explicitly banks 32 `sram_64_2048` single-port macros, links the
ARM macro `.db` with the TSMC28 standard-cell library, and fails if DC does not
preserve exactly 32 macro instances. Reports and the mapped netlist stay under
`build/dc_mem_macro/`. The default simulation path retains the original
zero-latency behavioral MEM for C-model cycle compatibility; setting
`USE_SRAM_MACRO=1` selects the synchronous physical-memory path, where Read is
one cycle and ReadWrite is serialized as read then write.

Run the 128 KiB Vector-accumulator macro target with:

```bash
./scripts/dc_acc_macro.sh
```

It banks sixteen ARM `sram_128_512` macros as eight 256-bit row banks and uses
one cycle-shared DesignWare FP32 adder. `FTLPU_USE_SRAM128X512_MACRO` selects
this physical backend. The physical backend intentionally rejects Block8;
the default architectural simulation backend keeps Vector and Block8 support
for C-model regression compatibility. `scripts/dc_full_macro.sh` is the
full-size 52-column MEM+ACC synthesis target and also defines
`FTLPU_DISABLE_BLOCK8` so only one Vector dot row is elaborated.

The unit-level architecture test verifies ICU NOP/Repeat timing, signed MEM
repeat stride, and an SRAM Read/Write round trip. The VCS system regression
also verifies mirrored transfers across passive SR hops and a 16-stream,
four-tile `MEM -> SXM Transpose -> Permute -> MEM` round trip. Its complete
416-bit packets, schedule, initialization, and expected SRAM image are emitted
by the sibling C model. A second SXM regression performs a complete 32x32 FP16
transpose using four overlapping input beats and seven non-identity wavefront
Permute maps across all four physical tiles.

The replacement VXM ALU interface uses a 32-bit data container and a 2-bit
format field. FP16, BF16, and FP32 Bypass, Add, Subtract, Multiply, Negate, and
Max are implemented. All three formats support Exp, Reciprocal, and Rsqrt
through the C-model-compatible LUT path. Basic timing is one cycle except
Multiply at two. Special operations use a five-stage arithmetic pipeline plus
any wait introduced by single-port SRAM arbitration. Each Tile owns three
separately programmable FP16 `{k,b}` SRAMs. BF16 and FP32 Special execution widen LUT
range configuration and coefficients, then perform address arithmetic and
`k*dx+b` interpolation in FP32. BF16 rounds back to 16 bits at each ALU result.
Floating-point arithmetic paths use the project's deterministic RNE/DAZ/FTZ
policy and canonicalize NaNs. Bypass preserves every source bit, and Negate
changes only the active format's sign bit. FP16, BF16, and FP32 Add/Subtract share one
27-bit segmented significand datapath plus carry behind a front format MUX.
FP16 uses a 7-bit signed working exponent; BF16/FP32 use 10-bit signed working
exponents. Operand isolation uses one 14-bit low adder group for FP16/BF16 and
enables a 13-bit high group for FP32. Near subtraction uses one grouped 27-bit
leading-zero encoder; far subtraction bypasses it with a zero/one-bit shift.
FP16/BF16 share the complete low-15-bit magnitude comparator and FP32 reuses
that comparator below an added high-16-bit comparison. ADD, SUBTRACT, and MAX
consume the same per-ALU comparison result. MAX propagates either input NaN as
the selected format's canonical quiet NaN, treats DAZ inputs as signed zero,
returns positive zero for every effective-zero pair, and selects LHS for equal
nonzero operands. Basic multiplication is shared by
all three formats through nine operand-isolated 8x8 significand blocks: BF16 uses
one block, FP16 uses at most four, and FP32 uses at most nine. The 8x8 leaf
implementation remains synthesis-selected. Their former carry-propagating
addition tree has been replaced by a shared 3:2 carry-save compressor tree.
BF16 bypasses compression, FP16 adds P22 only at its second and final CSA level,
and FP32 follows the complete `9 -> 6 -> 4 -> 3 -> 2` tree before the one shared
carry-propagating adder. Multiplication consumes the common unpacked format
fields directly instead of widening FP16/BF16 through FP32. Its shared signed
exponent path generates only `E0 = Ea + Eb - Bias` and `E1 = E0 + 1` while the
significand array is active; normalization or an RNE carry selects E1 through a
two-input MUX. The former sequential exponent corrections are removed, and no
E2 candidate is instantiated.
Multiply's two-cycle interface now corresponds to two real hardware stages.
The common Basic front end performs operand unpack/DAZ/classification once and
resolves the Multiply special-result class (`NaN`, `Inf`, or `Zero`) before the
multiplier boundary; the multiplier does not inspect raw floating encodings.
Stage 1 performs the segmented significand multiply, CSA compression, final
carry-propagating addition, and parallel E0/E1 generation, then registers the
48-bit product plus format/sign/special metadata. Stage 2 normalizes, performs
RNE, selects E0/E1, packs the native result, and writes the Basic ALU output
register. Consecutive Multiply requests remain accepted every cycle.

The global VXM word is 15 bits: one flow-direction bit selects the input
hemisphere and the opposite output boundary, while the remaining fields include
the two-bit compute dtype introduced in place of the old precision flag. A
separate chain-head converter supports native FP16/BF16/FP32, FP16/BF16-to-FP32
widening, and FP16/FP32-to-BF16 conversion. Unsupported cross-format
combinations are reported explicitly.

The first replacement datapath layer now consists of a physical-stage local
decoder and operand MUX. It derives Head/Internal/Tail from chain length 2/4/8,
decodes each queue's heterogeneous 5/6/7-bit word, converts only fixed chain-
head stream groups, and carries a 32-bit `{value, original, auxiliary}` token
through internal stages. Missing operand data stalls `operands_valid`; illegal
source/opcode encodings and unsupported conversions are separate faults. The
unit test covers Stream, Immediate, Previous, Original/Auxiliary, Accumulator,
and Feedback paths as well as the same physical stage changing role with chain
length. A vectorized FP16 boundary module also assembles each stage block's
hard-wired LHS/RHS byte-stream pairs for all eight lanes; no stream identity is
stored in the compact instruction. `lpu_vxm_datapath_stage` combines that
boundary with eight lane-local MUXes and presents one physical stage's parallel
ALU request interface.

At each Tile chain head, a fixed pair of byte streams supplies 16 bits per
lane per cycle. An FP16/BF16 source is captured in one beat; an FP32 source is
assembled from low-16 and high-16 beats by independent LHS/RHS phase state.
The complete 32-bit token stays native throughout the ALU chain. FP32 tail
results use the same pair in two output beats. A completed input is consumed
by the ALU on the following cycle, so repeated FP32 inputs follow the intended
`low-half / high-half / execute` three-cycle cadence.

`lpu_vxm_execution_stage` now binds one lane's operand datapath directly to
its Basic/LUT ALU. It exposes one request-accept handshake and returns
`{result, original, auxiliary}` with metadata held across the ALU's different
latencies. The initial binding permits one operation in flight per physical
ALU, which prevents a short Basic result from overtaking a five-cycle LUT
result. A Special lookup receives a tagged, one-cycle response from the
Lane-local SRAM set shared by its adjacent Tile pair.

`lpu_vxm_tile_execution` performs the first full Tile assembly: 16 physical
stages by eight lockstep lanes, resident mirrored Q0-Q7 configuration for
stages `q`/`q+8`, and registered `{value, original, auxiliary}` forwarding
between non-tail stages. Loading a configuration and starting an execution are
separate handshakes, so Repeat reuses the stored heterogeneous 5--7 bit local
words rather than rewriting them. A global four-bit phase word (`first`,
`last`, accumulator enable, output enable) accompanies each execution token.
On the last execution, one end marker is injected into lane 0 at one
representative chain head, follows the same registered path as its data, and
produces `config_done` at that chain's tail. The other lanes/chains do not
duplicate it, and the ALU does not interpret this marker.
`lpu_vxm_instruction_controller` owns the one shared repeat/interval counter,
loads all eight local words and the global word once, and retires the resident
configuration only after that returned marker. The Tile owns
per-stage/lane C1/C3
accumulators, fixed odd-tail feedback registers, atomic two-byte output
registers with backpressure. Each adjacent Tile pair owns 24 single-read LUT
SRAMs: one for every `{Exp/Reciprocal/Rsqrt, Lane}` combination. The four-Tile
Slice therefore owns two identical sets, or 48 SRAMs. Adjacent Tiles use the
set in alternating wave phases; simultaneous ownership of the same SRAM is a
schedule fault rather than a request that is serialized.
One chain-length-4 Tile test checks ordinary forwarding; a
second checks feedback consumption, resident-config accumulator repetition,
output holding, and concurrent Exp/Reciprocal/Rsqrt traffic across all eight
lanes. A controller test checks configuration-load backpressure, repeat phase
generation, interval timing, and final-marker retirement.

The scalable execution regression lives under `sim/tb/vxm`: one parameterized
checker is instantiated for all 16 physical ALU positions. Every position
executes all supported Basic/special operations crossed with all
architecturally reachable operand sources, plus missing-input stalls,
compact-decode faults, LUT access, and result-token metadata checks. New
coverage belongs in the checker rather than in per-ALU/per-operation wrapper
files.
Chain-length-4 roles, in-flight reset recovery, unsupported dtype/chain/opcode
cases, special-value behavior, and a nonzero LUT address are included in the
same checker.

The active VXM control path uses eight heterogeneous local instruction waves
and one independent global configuration queue. `lpu_vxm_control` advances the
local and global words through four rows one cycle at a time. Each row retains
the most recently received global word. `lpu_vxm_slice` instantiates one VXM
made from four execution Tiles, with each Tile placed between the corresponding
left- and right-hemisphere boundaries. The 15-bit global word contains one
`flow_direction` bit. Each Tile latches that bit with its configuration; it
controls both the 2:1 boundary-input MUX and the opposite-boundary result route.
Accepted chain-head operands generate exact stream consume pulses; unrelated
West traffic still crosses through `lpu_vxm_stream_bridge`. A configuration
wave currently schedules one execution per Tile.
Connecting decoded ICU Repeat commands to the shared instruction controller is
the remaining control-path integration step; it does not require another Tile
interface change.

The SmolLM2 FFN system regression keeps one RTL instance and executes three
separately scheduled phases over real intermediate SRAM data:
`INT8 dequant Block8 gate/up -> FP32-to-BF16 SwiGLU -> INT8 dequant Block8
down`. The checked-in regression uses `X[8,32]` so it remains practical for
routine VCS runs while preserving the C-model instruction sequence and all 32
physical lanes. `scripts/run_smollm2_ffn_reference.sh` separately runs the
full `X[128,576]`, intermediate-1536 C-model workload.

The MXM datapath supports full-supercell and per-column Direct16 IW followed by
non-overlapping Vector or Block8 Compute using FP16 or BF16
weights/activations. Its Vector C-model differential regression executes a non-symmetric
32x32 permutation matrix through `MEM -> MXM -> MEM` once per weight buffer in
both BF16 and FP16, checking 128 FP32 results. The two passes use different
permutations and opposite output signs so buffer-selection errors cannot alias.
A per-column C-model regression assembles all 32 weight columns through East
streams 0 and 1, checks cell validity only after eight inner-column writes, and
executes another non-symmetric 32x32 permutation. INT8 IW is paired with the
independent dequant queue, consumes eight full-cell weight streams, multiplies
signed bytes by a BF16 scale, and stores BF16 weights. Its C-model regression
uses distinct positive and negative quantized values/scales across all four
supercells. A module-boundary regression also checks missing or incorrectly
paired dequant instructions and overlapping compute waves. A Block8 C-model
regression consumes 16 activation streams, computes eight independent rows,
checks all 256 BF16 stream results, then checks Block8 SRAM destination and
all 256 FP32 results during retained read, read-clear, and zero-after-clear.
The Vector accumulator C-model regression checks continuous row progression with
non-unit row stride, accumulation after a new wave, retained AccumulatorRead,
read-clear, and zero-after-clear across all 32 columns. Overlapping compute
waves remain intentionally unsupported. SXM
ShiftSelect and Distribute remain intentionally unimplemented.

## Conventions

- Use SystemVerilog (`.sv` / `.svh`) for design and verification.
- Use active-low asynchronous reset named `rst_ni`.
- Use `_i`, `_o`, and `_io` suffixes for ports.
- Keep synthesizable RTL under `rtl/` and verification-only code under `sim/`.
- Add source files to `sim/filelist.f` in dependency order.
- Do not commit generated simulation output or waveforms.
- Keep compiler and EDA output under `build/`, and point tool scratch variables
  at `build/tmp/`; project workflows must not place generated artifacts outside
  the repository.
