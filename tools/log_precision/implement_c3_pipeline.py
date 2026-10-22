"""Apply the C3 throughput change to the reviewable RTL staging tree."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'rtl-log-work'
target=Path(r'C:\Users\zhaodong.feng\Desktop\LPU\FTLPU-VMODEL-develop')
# Preserve the current target hashes; the final synchronizer rejects concurrent edits.
manifest={str(p.relative_to(target)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
          for folder in ('rtl','sim','scripts','docs') for p in (target/folder).rglob('*') if p.is_file()}
(ROOT/'build-log/c3_pipeline_baseline.json').write_text(json.dumps(manifest,indent=2))

# Separate the common significand product from FP32 rounding. Existing callers
# retain exactly the same multiply function and numerical behavior.
p=R/'rtl/vxm/lpu_vxm_math_pkg.sv';s=p.read_text()
start=s.index('function automatic logic [31:0] fp32_multiply_rne(')
end=s.index('endfunction',start)+len('endfunction')
old=s[start:end]
raw=old.replace('fp32_multiply_rne','fp32_product_rne').replace('input logic [31:0] rhs\n','input logic [31:0] rhs,\n  input logic [47:0] raw_product\n')
raw=raw.replace('product = lhs_significand * rhs_significand;','product = raw_product;')
wrapper='''
function automatic logic [31:0] fp32_multiply_rne(
  input logic [31:0] lhs, input logic [31:0] rhs
);
  logic [23:0] a,b;
  logic [47:0] product;
  begin
    a={1'b1,lhs[22:0]}; b={1'b1,rhs[22:0]};
    product=a*b;
    fp32_multiply_rne=fp32_product_rne(lhs,rhs,product);
  end
endfunction'''
p.write_text(s[:start]+raw+wrapper+s[end:])

# Keep the EXP implementation behind the original public interface. Only C3
# elaborates the new pipeline; it does not elaborate the legacy arithmetic.
p=R/'rtl/vxm/lpu_vxm_special_alu.sv';s=p.read_text()
header=s[:s.index('  import lpu_pkg::*;')]
ports=['clk_i','rst_ni','valid_i','data_format_i','opcode_i','operand_format_valid_i',
 'operand_sign_i','operand_exponent_i','operand_fraction_i','operand_zero_i',
 'operand_inf_i','operand_nan_i','lut_configured_i','lut_input_min_i',
 'lut_segment_width_i','lut_read_valid_o','lut_read_bank_o','lut_read_address_o',
 'lut_read_valid_i','lut_read_k_i','lut_read_b_i','lut_read_log_i','ready_o',
 'valid_o','result_o','illegal_opcode_o','lut_fault_o']
params=['LUT_BANK_COUNT','LUT_ENTRY_COUNT','LUT_BANK_WIDTH','LUT_ADDRESS_WIDTH']
def instance(module,name,extra=[]):
 return module+' #(\n'+',\n'.join('    .'+x+'('+x+')' for x in extra+params)+'\n  ) '+name+' (\n'+',\n'.join('    .'+x for x in ports)+'\n  );\n'
public=header+'  generate if (SPECIAL_KIND == lpu_pkg::VXM_SPECIAL_RECIP_RSQRT) begin : g_c3\n'
public+=instance('lpu_vxm_c3_pipeline','u_pipeline')
public+='  end else begin : g_legacy\n'+instance('lpu_vxm_special_alu_legacy','u_legacy',['SPECIAL_KIND'])+'  end endgenerate\nendmodule\n\n'
p.write_text(public+s.replace('module lpu_vxm_special_alu #(','module lpu_vxm_special_alu_legacy #(',1))

# Expose the fused Rsqrt rounding tail as a pure no-multiply helper.
s=(R/'rtl/vxm/lpu_vxm_fp32_rsqrt_correction.sv').read_text()
fn=s[s.index('  function automatic'):s.index('  logic [23:0] m_significand;')]
body=s[s.index('    exponent_sum ='):s.index('\n  end\nendmodule')]
body=body.replace('correction_o','c3_rsqrt_correction').replace('    range_fault_o = enable_i &&','    invalid =')
pkg='''// Helpers consume an externally computed full product; no hidden multiplier.
package lpu_vxm_c3_math_pkg;
  import lpu_vxm_math_pkg::*;
'''+fn+'''
  function automatic logic [31:0] c3_rsqrt_correction(
    input logic [31:0] m_i,y_squared_i,
    input logic [47:0] significand_product
  );
    logic [63:0] product_q55,one_point_five_q55;
    logic signed [64:0] difference;
    logic invalid;
    integer exponent_sum,product_shift;
    begin
'''+body+'''
      if(invalid) c3_rsqrt_correction=32'h7fc00000;
    end
  endfunction
endpackage
'''
(R/'rtl/vxm/lpu_vxm_c3_math_pkg.sv').write_text(pkg)
print('Prepared shared product helpers and C3 binding.')
