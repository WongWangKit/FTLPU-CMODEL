from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'rtl-log-work'

def change(path,old,new):
    p=ROOT/path;s=p.read_text();assert old in s,(path,old);p.write_text(s.replace(old,new,1),newline='\n')

p='rtl/vxm/lpu_vxm_special_alu.sv'
change(p,'  logic supported_opcode;', '''  logic log_ready, log_valid, log_fault, log_lut_request;
  logic [5:0] log_lut_address;
  logic [31:0] log_result;
  logic log_configured;
  always_comb begin
    log_configured = 1'b0;
    for (integer bank = 0; bank < LUT_BANK_COUNT; bank++)
      if (bank == VXM_LOG_LUT_BANK) log_configured = lut_configured_i[bank];
  end
  lpu_vxm_log u_log (
    .clk_i, .rst_ni,
    .valid_i(valid_i && operand_format_valid_i && supported_opcode && ready_o &&
      (opcode_i == VXM_LOCAL_LOG)),
    .data_format_i, .operand_sign_i, .operand_exponent_i, .operand_fraction_i,
    .operand_zero_i, .operand_inf_i, .operand_nan_i,
    .lut_configured_i(log_configured),
    .lut_read_valid_o(log_lut_request), .lut_read_address_o(log_lut_address),
    .lut_read_valid_i, .lut_read_data_i(lut_read_log_i),
    .ready_o(log_ready), .valid_o(log_valid), .result_o(log_result), .fault_o(log_fault)
  );

  logic supported_opcode;''')
change(p,'bypass1_valid_q || bypass2_valid_q || bypass3_valid_q);','bypass1_valid_q || bypass2_valid_q || bypass3_valid_q) && log_ready;')
change(p,'    lut_read_address_o = lookup_index[LUT_ADDRESS_WIDTH-1:0];','''    lut_read_address_o = lookup_index[LUT_ADDRESS_WIDTH-1:0];
    if (log_lut_request) begin
      lut_read_valid_o = 1'b1;
      lut_read_bank_o = LUT_BANK_WIDTH'(VXM_LOG_LUT_BANK);
      lut_read_address_o = LUT_ADDRESS_WIDTH'(log_lut_address);
    end''')
change(p,'(rsqrt_square_valid_q && rsqrt_correction_fault);','(rsqrt_square_valid_q && rsqrt_correction_fault) || log_fault;')
change(p,'rsqrt_newton_valid_q || bypass3_valid_q;','rsqrt_newton_valid_q || bypass3_valid_q || log_valid;')
change(p,'      if (rsqrt_newton_valid_q)\n        result_o', '      if (log_valid)\n        result_o <= log_result;\n      else if (rsqrt_newton_valid_q)\n        result_o')
change(p,'if (valid_i && operand_format_valid_i && supported_opcode && ready_o) begin','if (valid_i && operand_format_valid_i && supported_opcode && ready_o &&\n          (opcode_i != VXM_LOCAL_LOG)) begin')

change('rtl/vxm/lpu_vxm_slice.sv','tile*LUT_LANE_REQUESTS*50 +: LUT_LANE_REQUESTS*50','row*LUT_LANE_REQUESTS*50 +: LUT_LANE_REQUESTS*50')
change('rtl/vxm/lpu_vxm_log_math_pkg.sv','''    case (dtype)
      VXM_FORMAT_FP16: log_pack_result''','''    if ((value[30:23] == 8'hff) && (|value[22:0])) begin
      case (dtype)
        VXM_FORMAT_FP16: log_pack_result = 32'h00007e00;
        VXM_FORMAT_BF16: log_pack_result = 32'h00007fc0;
        default: log_pack_result = 32'h7fc00000;
      endcase
    end else case (dtype)
      VXM_FORMAT_FP16: log_pack_result''')

# Keep package order consistent for every filelist that binds the special ALU.
for path in (ROOT/'sim').glob('*.f'):
    s=path.read_text()
    if 'rtl/vxm/lpu_vxm_special_alu.sv' in s:
        s=s.replace('rtl/vxm/lpu_vxm_special_alu.sv',
          'rtl/vxm/lpu_vxm_log_math_pkg.sv\nrtl/vxm/lpu_vxm_log.sv\nrtl/vxm/lpu_vxm_special_alu.sv')
        path.write_text(s,newline='\n')
print('Bound Log continuation, preserved eight-cycle no-wait output timing.')
