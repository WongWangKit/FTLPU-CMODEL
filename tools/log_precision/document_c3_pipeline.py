from pathlib import Path
R=Path(__file__).resolve().parents[2]/'rtl-log-work'
p=R/'docs/architecture.md';s=p.read_text()
s=s.replace('''truncation. One shared multiply demand point reuses C3's first Reciprocal
Newton multiplier. A phase-selected Horner adder evaluates one/two/four
Taylor terms. Narrow modes bypass unused phases.''','''truncation. Reciprocal, Rsqrt and Log share a spatial pipeline: one 22x18
fixed-point multiplier followed by three 24x24 significand multipliers.
Log evaluates its one/two/four Taylor terms in expanded Horner form; a
request never returns to an earlier multiplier. Multiplication by 1/2 or
1/4 is an exponent adjustment. Narrow modes bypass unused stages.''')
s=s.replace('''the same phases. The single-outstanding contract remains; without LUT wait,
the accepted edge plus seven following edges gives the existing eight-cycle
external Special latency. Adding Log does not make the ALU II=1.''','''the same pipeline. C3 accepts one independent instruction/data element each
cycle (II=1), including mixed functions and formats. With the integrated
one-cycle LUT response, every C3 result appears seven edges after its accepted
edge (eight edges inclusive). Missing responses fault and return NaN at that
same latency; this interface has no LUT-response backpressure or retry.

C3 Basic operations are also aligned to this eight-edge latency to prevent
short operations overtaking special functions. Their original arithmetic is
retained; a two-edge Basic mode removes Multiply/short-operation collisions,
then six registers align results. Other physical positions retain their prior
timing. The execution stage pipelines Original/Auxiliary/end-marker tags;
the Tile pipelines Repeat/accumulator-write tags and tracks in-flight requests
separately from ready. Accumulator dependencies still wait for retirement.

Pipeline arithmetic allocation:

| Stage | Reciprocal FP32 | Rsqrt FP32 | Log FP32 |
|---|---|---|---|
| M0 22x18 | k*delta, seed=b-product | k*delta, seed=b-product | inverse*delta -> u |
| A0 | delay | delay | h0=1/3-u/4 |
| M1 24x24 | m*seed; correction=2-product | seed*seed | h1=-1/2+u*h0 |
| M2 24x24 | seed*correction | fused 1.5-0.5*m*square | h2=1+u*h1 |
| M3 24x24 | delay | seed*correction | u*h2 |

M2's Rsqrt correction consumes the full 48-bit product before its single RNE;
it does not use an already rounded FP32 product. Log's exponent-times-ln2
constant path and Q40 restoration adder remain separate from these four
variable multipliers. BF16 Log uses only M0; FP16 Log uses M0 and M3.
FP16/BF16 Reciprocal and Rsqrt use only M0. Inactive arithmetic operands are
isolated and payload registers hold on bubbles. This is scalar II=1, not
multiple packed low-precision elements per lane per cycle.

This compute-interface throughput does not remove the existing shared-SRAM
schedule restrictions, Tile issue constraints or 16-bit stream beat limits.
Two clients requesting the same function/lane SRAM in one cycle still collide;
FP32 stream transfer still needs two 16-bit beats. No claim is made that an
entire Slice can inject one FP32 vector per cycle.''')
s+='''

### Instruction-driven C3 throughput regression

Run `bash scripts/vcs_vxm_c3_pipeline.sh` or
`bash scripts/verilator_vxm_c3_pipeline.sh`. The generator creates 240,753
instruction/operand/golden-result records with a deterministic seed. Python
double-precision math and independent RNE/DAZ/FTZ conversion provide expected
results; the black-box scoreboard never calls RTL arithmetic helpers or reads
internal signals. All positive normal FP16/BF16 inputs are covered for each
of the three special functions; FP32 uses random values, exponent extremes,
LUT boundaries and near-one boundaries. Six Basic opcodes are interleaved.

Each of C3/C7/C11/C15 is tested separately with a private Lane SRAM fixture,
so these tests do not imply four simultaneous accesses to a shared SRAM are
legal. Checks cover acceptance every cycle, exact eight-edge latency, result
order, Original/Auxiliary/end-marker tags, dtype/chain changes, bubbles, reset
with requests in flight, illegal instructions/dtypes, missing configuration,
dropped responses and recovery. Limits are 1 ULP for FP16/BF16 and 2 ULP for
FP32 special functions, exact bits for directed Basic arithmetic. The longest
uninterrupted issue burst is 240,753 elements. Physical frequency and area
still require synthesis and timing analysis.
'''
p.write_text(s)
p=R/'sim/tb/vxm/README.md';s=p.read_text()
s+='''
C3 throughput regression: `lpu_vxm_c3_pipeline_tb.sv` drives public local
instruction and operand ports through the real decoder, common unpackers,
execution wrapper and Lane SRAM. It checks independent mathematical goldens,
fixed latency, II=1, ordering and public token metadata across C3/C7/C11/C15.
Run `bash scripts/vcs_vxm_c3_pipeline.sh` or
`bash scripts/verilator_vxm_c3_pipeline.sh` from the repository root.
The current C3 implementation replaces the separate interpolators/Newton
multipliers described above with four spatial shared multiply stages; see
the authoritative allocation table in `docs/architecture.md`.
'''
p.write_text(s)
# The generated large vector file is reproducible and need not be versioned.
p=R/'.gitignore';s=p.read_text()+'''\n# Regenerated instruction-level C3 black-box vectors
sim/vectors/vxm_c3_pipeline_cases.hex
''';p.write_text(s)
for name in ['sim/filelist.f']:
 p=R/name;s=p.read_text()+'\nsim/tb/vxm/lpu_vxm_c3_pipeline_tb.sv\n';p.write_text(s)
for name in ['sim/filelist.f','sim/filelist_vxm_log.f']:
 p=R/name;p.write_text('\n'.join(x.strip() for x in p.read_text().splitlines())+'\n')
p=R/'sim/filelist_vxm_c3_pipeline.f'
s=(R/'sim/filelist_vxm_log.f').read_text()
s='\n'.join(x for x in s.splitlines() if x.startswith('rtl/'))+'\nsim/tb/vxm/lpu_vxm_c3_pipeline_tb.sv\n'
p.write_text(s)
for name in ['scripts/lint.sh','scripts/lint.ps1']:
 p=R/name;s=p.read_text()
 if name.endswith('.sh'):
  s+='\nverilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal --timing \\\n+  -f sim/filelist_vxm_c3_pipeline.f --top-module lpu_vxm_c3_pipeline_tb\n'
 else:
  s+='\nverilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal --timing -f sim/filelist_vxm_c3_pipeline.f --top-module lpu_vxm_c3_pipeline_tb\n'
 p.write_text(s)
print('Documented C3 throughput and black-box test contract.')
