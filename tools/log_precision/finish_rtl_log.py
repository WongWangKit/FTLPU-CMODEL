from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]/'rtl-log-work'

for path in (ROOT/'scripts').glob('*.tcl'):
    s=path.read_text()
    if 'rtl/vxm/lpu_vxm_special_alu.sv' in s:
        s=s.replace('rtl/vxm/lpu_vxm_special_alu.sv',
          'rtl/vxm/lpu_vxm_log_math_pkg.sv \\\n  rtl/vxm/lpu_vxm_log.sv \\\n  rtl/vxm/lpu_vxm_special_alu.sv')
        path.write_text(s,newline='\n')

p=ROOT/'sim/tb/lpu_vxm_icu_map_tb.sv';s=p.read_text()
s=s.replace('    $display("VXM physical', '''    // Q3/Q7 must retain extension bit 5 through the physical ICU FIFOs.
    @(negedge clk);run=1'b0;
    for (integer q=115;q<=119;q=q+4) begin
      @(negedge clk);enqueue_queue=8'(q);
      enqueue_payload='0;
      enqueue_payload[5:0]=lpu_pkg::VXM_LOG_INSTRUCTION;
      enqueue_valid=1'b1;#1ps;
      if(!enqueue_ready) $fatal(1,"Log ICU queue not ready");
      @(negedge clk);enqueue_valid=1'b0;
    end
    @(negedge clk);run=1'b1;
    @(posedge clk);#1ns;
    if(!issue_valid[115]||!issue_valid[119]||
       issue_payload[115*416+:6]!==lpu_pkg::VXM_LOG_INSTRUCTION||
       issue_payload[119*416+:6]!==lpu_pkg::VXM_LOG_INSTRUCTION)
      $fatal(1,"Log extension bit was truncated by ICU");

    $display("VXM physical''')
p.write_text(s,newline='\n')
p=ROOT/'rtl/icu/lpu_icu.sv';s=p.read_text().replace('        logic [LOCAL_WIDTH-1:0] local_issue_payload;',
  '        // Q3/Q7 include bit 5 of the backward-compatible Log extension.\n        logic [LOCAL_WIDTH-1:0] local_issue_payload;');p.write_text(s,newline='\n')

for name in ('lpu_vxm_slice.sv','lpu_vxm_tile_pair_lut.sv'):
    p=ROOT/'rtl/vxm'/name;s=p.read_text()
    s=s.replace('share 24 SRAMs','share 32 SRAMs').replace('3 x 8 SRAMs by default','3 x 8 SRAMs by default; 4 x 8 with Log')
    p.write_text(s,newline='\n')
p=ROOT/'rtl/vxm/lpu_vxm_alu.sv';s=p.read_text().replace('C3 with Reciprocal/Rsqrt.','C3 with Reciprocal/Rsqrt/Log.');p.write_text(s,newline='\n')

# Place self-contained vector regeneration and low-format checks beside the
# project's existing numerical references.
src=Path(__file__).with_name('generate_rtl_log_vectors.py').read_text()
src=src.replace("ROOT=Path(__file__).resolve().parents[2]/'rtl-log-work'", "ROOT=Path(__file__).resolve().parents[2]")
src=src.replace("(ROOT/'sim/vectors/vxm_log_count.svh').write_text(f'`define VXM_LOG_CASE_COUNT {len(vectors)}\\n')",
'''tb=ROOT/'sim/tb/vxm/lpu_vxm_log_tb.sv'
import re
tb.write_text(re.sub(r'CASE_COUNT = [0-9]+;', f'CASE_COUNT = {len(vectors)};', tb.read_text()))''')
(ROOT/'sim/cmodel/generate_log_vectors.py').write_text(src,newline='\n')
(ROOT/'sim/cmodel/check_log_low_precision.py').write_text(Path(__file__).with_name('check_low_rtl_flow.py').read_text(),newline='\n')
unused=ROOT/'sim/vectors/vxm_log_count.svh'
if unused.exists():unused.unlink()

p=ROOT/'scripts/lint.ps1';s=p.read_text().replace("'lpu_vxm_alu_tb',", "'lpu_vxm_alu_tb', 'lpu_vxm_log_tb',");p.write_text(s,newline='\n')
p=ROOT/'scripts/lint.sh';s=p.read_text()+'''\nverilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \\\n  -f sim/filelist_vxm_log.f --top-module lpu_vxm_log_tb\n''';p.write_text(s,newline='\n')

(ROOT/'scripts/vcs_vxm_log.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
mkdir -p build/vcs/vxm_log
python3 sim/cmodel/generate_log_vectors.py
vcs -full64 -sverilog -timescale=1ns/1ps -top lpu_vxm_log_tb \\
  -f sim/filelist_vxm_log.f -Mdir=build/vcs/vxm_log/csrc \\
  -o build/vcs/vxm_log/simv -l build/vcs/vxm_log/compile.log
build/vcs/vxm_log/simv -no_save -l build/vcs/vxm_log/run.log
''',newline='\n')

doc='''
### C3 Log extension

Log is available at physical stages C3/C7/C11/C15 in every chain mode. Q3
and Q7 grow from five to six physical instruction bits. The canonical Log
word is `6'b100110` (`0x26`): bit 5 extends the opcode and bits 4:3 must be
zero. All old five-bit words keep their meanings. Noncanonical extension
words are illegal. The local decoder exports the four-bit semantic opcode
`VXM_LOCAL_LOG=8`; Basic opcodes and operand-source encodings are unchanged.
ICU queues 115 and 119 and their control waves use the package width helper,
so the extension is retained through the physical FIFO and mirrored decode.

The Slice has four LUT functions. Bank 3 is Log; each adjacent Tile pair has
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
'''
p=ROOT/'docs/architecture.md';p.write_text(p.read_text()+doc,newline='\n')
p=ROOT/'sim/tb/vxm/README.md';p.write_text(p.read_text()+'''\nLog regression: `lpu_vxm_log_tb.sv` uses the 50-bit bank-3 Lane SRAM and
independently generated numerical vectors. `lpu_vxm_log_reference_pkg.sv`
also permits constant evaluation of production arithmetic helpers. Generated
`lpu_vxm_log_constants.sv` contains 2,335 reference checks. See the C3 Log
extension section in `docs/architecture.md` and `scripts/vcs_vxm_log.sh`.
''',newline='\n')
print('Updated ICU extension tests, synthesis lists, numerical references and docs.')
