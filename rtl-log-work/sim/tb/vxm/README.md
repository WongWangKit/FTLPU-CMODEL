# VXM RTL test organization

Keep scalable VXM tests in this directory instead of adding one top-level file
for every physical ALU, source, opcode, and data format.

- `lpu_vxm_alu_stage_checker.sv` is the reusable position-aware checker. Add
  new opcode, source, dtype, latency, metadata, and fault cases here.
- `lpu_vxm_significand_multiplier_tb.sv` drives the segmented multiplier only
  through its ports and checks BF16/FP16/FP32 block gating, zero-chunk operand
  isolation, disabled behavior, 900 deterministic random operands, and the
  complete 48-bit product after the format-aware CSA tree.
- `lpu_vxm_segmented_addsub_tb.sv` drives the 14+13 significand add/subtract
  datapath only through its ports and checks BF16/FP16/FP32 width boundaries,
  carry/borrow between groups, randomized arithmetic, and disabled isolation.
- `lpu_vxm_grouped_lzc_tb.sv` checks every active bit position of the shared
  27-bit leading-zero encoder through its ports, including format masking,
  zero detection, disabled behavior, and randomized input patterns.
- `lpu_vxm_shared_left_shift_tb.sv` checks the single format-masked 27-bit
  normalization shifter through its ports, including active-width truncation,
  all one-hot bit/shift combinations, and crossing the 14+13 boundary.
- `lpu_vxm_basic_frontend_tb.sv` checks FP16/BF16/FP32 unpack and classification,
  arithmetic DAZ field generation, reserved-format rejection, and one-hot
  decoding of every Basic opcode.
- `../lpu_vxm_input_converter_tb.sv` checks native-format raw pass-through
  and the existing cross-format numeric conversions. Arithmetic ALUs apply
  DAZ/NaN policy after native pass-through. `../lpu_vxm_execution_stage_tb.sv`
  checks raw BYPASS/NEGATE payloads through the public instruction and stream
  interfaces. Run only these two checks with `bash scripts/vcs_raw_ops.sh`;
  logs stay under `build/vcs/raw_ops_*/`.
- `lpu_vxm_shared_float_compare_tb.sv` drives the two unpackers into the shared
  comparator and checks FP16/BF16 low-part sharing, FP32 high/low hierarchy,
  signed ordering, NaN unordered behavior, signed zero, DAZ, disabled operation,
  and reserved-format behavior.
- `lpu_vxm_addsub_max_instruction_tb.sv` drives Q0 compact ADD/SUB/MAX/MUL
  instructions and both stream operands into the execution-stage ports. An
  independent exact-integer oracle checks all three formats, RNE,
  DAZ/FTZ, special values, one-cycle versus two-cycle latency, metadata, and
  deterministic random operands. ADD/SUB/MAX and MUL print separate reports;
  no DUT hierarchy is referenced.
  Run only this regression on a VCS server with
  `bash scripts/vcs_addsub_max.sh`; compile and simulation logs are kept in
  `build/vcs/vxm_addsub_max_instruction/`.
- `lpu_vxm_tile_pair_lut_tb.sv` programs only through module ports and checks
  eight Lane-parallel reads, back-to-back ownership by adjacent Tiles,
  simultaneous access to different functions, Stage-tagged returns, and
  explicit same-function/Lane collision reporting. It also checks that the
  EXP banks store one packed UQ1.25 base in a 26-bit row, RECIP banks store
  UQ1.15 coefficient pairs in 32-bit rows for both RECIP and RSQRT. The test
  also checks each bank's upper-container faults.
- `lpu_vxm_16_alu_tb.sv` is the regression top. It instantiates the checker for
  physical stages 0 through 15 and finishes only after all stages pass.
- `lpu_vxm_instruction_controller_tb.sv` verifies that one resident global +
  eight-local configuration is loaded once, executed by one shared Repeat
  counter, and retired only by the returned final marker.
- `lpu_vxm_tile_execution_tb.sv` verifies the assembled 16-stage/eight-lane
  Tile, resident configuration interface, chain-length-4 forwarding, and the
  single returned completion marker.
- `lpu_vxm_tile_fp32_phase_tb.sv` drives both FP32 input halves only through
  Tile ports, checks low/high phase consumption, 32-bit chain forwarding, and
  two-beat FP32 output serialization.
- `lpu_vxm_tile_special_paths_tb.sv` verifies tile-owned chain feedback,
  C1/C3 accumulator reset/update, FP16/BF16 packing/backpressure, and
  concurrent FP16/BF16/FP32 Exp/Reciprocal/Rsqrt traffic through the
  Lane-independent LUT SRAM interface.
- `lpu_vxm_slice_tb.sv` verifies the four physical Tile rows of the single VXM,
  both left-to-right and right-to-left flow settings, the one-cycle-per-row
  instruction wave, overlapping low/high FP32 port beats aligned with that
  wave, per-cell stream consumption, two-beat FP32 results, and the absence of boundary
  conflicts. Internal Slice signals are observation points only; all stimulus
  enters through the Slice instruction and stream ports.
- `lpu_vxm_slice_fp32_special_tb.sv` programs the compact LUT only through
  Slice ports, then verifies FP32 Exp/Reciprocal/Rsqrt across all four rows and
  both flow directions with low/high input and output phases.
- `lpu_vxm_slice_bf16_special_tb.sv` programs the same FP16 LUT through Slice
  ports, then verifies single-beat BF16 Exp/Reciprocal/Rsqrt across all four
  rows and both flow directions.
- Small tests in the parent `sim/tb` directory remain focused unit/smoke tests
  for one module or one historical regression.

The checker derives the local queue and special-ALU kind from the physical
stage. It must test only architecturally reachable sources for that position;
chain length is changed when a stage can legally act as both Head and Internal.
Supported operations are crossed with those legal source encodings using
fixed, independently known FP16/BF16/FP32 results. Distinct-value directed cases remain
alongside the cross to detect incorrect MUX selection.

EXP uses the same 64-entry UQ1.25 base table for all formats. FP16/BF16 test
the linear `B + B*delta` path, while FP32 additionally checks the registered
cubic Horner path. All stimulus still enters
through public instruction, data, configuration, and LUT programming ports.
The LUT coefficient multiply is one shared inferred 26x24 unsigned operator:
BF16 uses UQ1.9 x 8 active bits, FP16 uses UQ1.12 x 11 active bits, and FP32
uses UQ1.25 x 24 active bits. Narrow operands are zero-extended; the exact
multiplier/compressor implementation is intentionally left for DC analysis.
RECIP stores each `k,b` pair as two UQ1.15 values. Its initial `b-k*dx`
interpolation is entirely fixed point: BF16 activates UQ1.9 x 1 residual bit,
FP16 activates UQ1.12 x 4 bits, and FP32 activates UQ1.15 x 17 bits. One
maximum-width 16x17 unsigned multiplier is shared by zero extension, followed
by one signed 40-bit aligned subtraction and RNE. FP16/BF16 bypass iteration;
only FP32 performs one registered `y0*(2-m*y0)` Newton refinement using two
explicit 24x24-capable multiplier stages. Other special-function results cross
operand-isolated bypass registers, keeping a uniform eight-cycle external
latency across formats and functions.

RSQRT divides its 64-entry table into two 32-entry halves selected by the
normalized exponent parity. Five fraction bits select a segment and leave
2/5/18 residual bits for BF16/FP16/FP32. Its shared fixed interpolator activates
UQ1.9 x 2, UQ1.12 x 5, or UQ1.15 x 18 inside one maximum 16x18 multiplier and
one signed 40-bit subtraction. FP32 then executes a three-stage Newton path:
`y0*y0`, fused `1.5-0.5*m*y0^2` with one RNE, and the final `y0*correction`.
FP16/BF16 cross equal-length operand-isolated bypass registers.
`sim/cmodel/vxm_rsqrt_precision.cpp` regenerates the 64 secant coefficient
pairs mathematically and scans all 16,777,216 normalized FP32 mantissas. With
UQ1.15 storage it measures maximum linear relative errors of 0.321% (BF16),
0.0431% (FP16), and 1.19e-4 (FP32); one FP32 Newton step reduces the maximum
relative error to 1.47e-7 with a two-ULP worst-case bound. This is the declared
inference accuracy target, not a correctly-rounded libm contract. Run the
repeatable check with `bash scripts/check_rsqrt_precision.sh`.

The suite also covers every stage's chain-length-4 role, reset with Multiply
in flight, reserved chain/dtype/source/opcode encodings, and special-function
zero/Inf/NaN/negative/FTZ cases. Reciprocal additionally accesses a nonzero
LUT address so the test does not validate only entry zero. Its directed
FP16/BF16/FP32 cases distinguish the unity and fractional exponent candidates,
check scaling across input exponents, and cover the FTZ exponent boundary.

Log regression: `lpu_vxm_log_tb.sv` uses the 50-bit bank-3 Lane SRAM and
independently generated numerical vectors. `lpu_vxm_log_reference_pkg.sv`
also permits constant evaluation of production arithmetic helpers. Generated
`lpu_vxm_log_constants.sv` contains 2,335 reference checks. See the C3 Log
extension section in `docs/architecture.md` and `scripts/vcs_vxm_log.sh`.

C3 throughput regression: `lpu_vxm_c3_pipeline_tb.sv` drives public local
instruction and operand ports through the real decoder, common unpackers,
execution wrapper and Lane SRAM. It checks independent mathematical goldens,
fixed latency, II=1, ordering and public token metadata across C3/C7/C11/C15.
Run `bash scripts/vcs_vxm_c3_pipeline.sh` or
`bash scripts/verilator_vxm_c3_pipeline.sh` from the repository root.
The current C3 implementation replaces the separate interpolators/Newton
multipliers described above with four spatial shared multiply stages; see
the authoritative allocation table in `docs/architecture.md`.
