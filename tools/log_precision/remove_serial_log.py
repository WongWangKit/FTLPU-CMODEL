from pathlib import Path
R=Path(__file__).resolve().parents[2]/'rtl-log-work'
p=R/'rtl/vxm/lpu_vxm_special_alu.sv';s=p.read_text()
start=s.index('  logic log_ready, log_valid')
end=s.index('  logic [15:0] sanitized_operand16;',start)
s=s[:start]+s[end:]
s=s.replace('.enable_i((add_valid_q && add_reciprocal_newton_q) || log_multiply_enable)',
            '.enable_i(add_valid_q && add_reciprocal_newton_q)')
s=s.replace('.lhs_i(log_multiply_enable ? log_multiply_lhs : add_reciprocal_m_q)', '.lhs_i(add_reciprocal_m_q)')
s=s.replace('.rhs_i(log_multiply_enable ? log_multiply_rhs : add_result_q)', '.rhs_i(add_result_q)')
s=s.replace('bypass3_valid_q) && log_ready;', 'bypass3_valid_q);')
start=s.index('    if (log_lut_request) begin')
end=s.index('\n    end',start)+len('\n    end')
s=s[:start]+s[end:]
s=s.replace('(rsqrt_square_valid_q && rsqrt_correction_fault) || log_fault;', '(rsqrt_square_valid_q && rsqrt_correction_fault);')
s=s.replace('rsqrt_newton_valid_q || bypass3_valid_q || log_valid;', 'rsqrt_newton_valid_q || bypass3_valid_q;')
s=s.replace('''      if (log_valid)
        result_o <= log_result;
      else if (rsqrt_newton_valid_q)''','''      if (rsqrt_newton_valid_q)''')
s=s.replace('''        (opcode_i == VXM_LOCAL_SPECIAL1) ||
        (opcode_i == VXM_LOCAL_LOG)));''','''        (opcode_i == VXM_LOCAL_SPECIAL1))));''')
# Wrapper is the sole public C3 entry; legacy remains for EXP only.
s=s.replace('''  import lpu_pkg::*;
  import lpu_vxm_fp16_pkg::*;''','''  // C3 is bound to the spatial pipeline above; this retained path serves EXP.
  import lpu_pkg::*;
  import lpu_vxm_fp16_pkg::*;''',1)
p.write_text(s)
for name in ['sim/filelist.f','sim/filelist_vxm_log.f','sim/filelist_vxm_c3_pipeline.f',
             'scripts/dc_synth.tcl','scripts/dc_full_macro_synth.tcl']:
 p=R/name;s=p.read_text();s='\n'.join(x for x in s.splitlines() if 'rtl/vxm/lpu_vxm_log.sv' not in x)+'\n';p.write_text(s)
print('Removed serial Log binding from the active implementation.')
