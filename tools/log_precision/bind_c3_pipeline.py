from pathlib import Path
R=Path(__file__).resolve().parents[2]/'rtl-log-work'
for name in ['sim/filelist.f','sim/filelist_vxm_log.f','scripts/dc_synth.tcl','scripts/dc_full_macro_synth.tcl']:
 p=R/name;s=p.read_text();needle='rtl/vxm/lpu_vxm_special_alu.sv'
 suffix=' \\' if name.endswith('.tcl') else ''
 s=s.replace(needle,'rtl/vxm/lpu_vxm_c3_math_pkg.sv'+suffix+'\n  rtl/vxm/lpu_vxm_c3_pipeline.sv'+suffix+'\n  '+needle)
 p.write_text(s)

# Optional uniform two-edge Basic pipeline, used only by C3. It prevents a
# short Basic result from colliding with the previous cycle's Multiply.
p=R/'rtl/vxm/lpu_vxm_basic_alu.sv';s=p.read_text().replace('module lpu_vxm_basic_alu (','module lpu_vxm_basic_alu #(parameter bit UNIFORM_TWO_CYCLE=0) (',1)
s=s.replace('  logic ', '  logic ',1)
at=s.index('  import ')
s=s[:at]+'''  logic short_valid_q;
  logic [31:0] short_result_q;
'''+s[at:]
s=s.replace('ready_o = !valid_i || !opcode_valid || op_multiply ||','ready_o = UNIFORM_TWO_CYCLE || !valid_i || !opcode_valid || op_multiply ||')
s=s.replace("      valid_o <= 1'b0;\n      result_o <= 32'b0;", "      short_valid_q <= 1'b0;short_result_q <= 0;\n      valid_o <= 1'b0;\n      result_o <= 32'b0;",1)
needle='''      if (valid_i && opcode_valid && !op_multiply) begin'''
start=s.index(needle)
s=s[:start]+'''      short_valid_q <= valid_i && opcode_valid && !op_multiply;
      if(valid_i && opcode_valid && !op_multiply) short_result_q <= one_cycle_result;
      if(UNIFORM_TWO_CYCLE) begin
        if(short_valid_q) begin
          valid_o <= 1'b1;result_o <= short_result_q;
        end
      end else '''+s[start:]
p.write_text(s)

# C3 pads all Basic outputs to eight edges. Other stage timing is unchanged.
p=R/'rtl/vxm/lpu_vxm_alu.sv';s=p.read_text()
s=s.replace('  logic basic_ready;', '''  localparam bit C3_PIPELINED=SPECIAL_KIND==lpu_pkg::VXM_SPECIAL_RECIP_RSQRT;
  logic basic_ready;
  logic basic_raw_valid;
  logic [31:0] basic_raw_result;
  logic [5:0] basic_delay_valid_q;
  logic [31:0] basic_delay_result_q[0:5];''')
s=s.replace('  lpu_vxm_basic_alu u_basic (','  lpu_vxm_basic_alu #(.UNIFORM_TWO_CYCLE(C3_PIPELINED)) u_basic (')
s=s.replace('.valid_o(basic_valid)', '.valid_o(basic_raw_valid)').replace('.result_o(basic_result)', '.result_o(basic_raw_result)')
pos=s.index('  lpu_vxm_special_alu #(')
s=s[:pos]+'''  generate if(C3_PIPELINED) begin:g_basic_align
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if(!rst_ni) begin
        basic_delay_valid_q<='0;
        for(integer i=0;i<6;i++) basic_delay_result_q[i]<=0;
      end else begin
        basic_delay_valid_q<={basic_delay_valid_q[4:0],basic_raw_valid};
        if(basic_raw_valid) basic_delay_result_q[0]<=basic_raw_result;
        for(integer i=1;i<6;i++)
          if(basic_delay_valid_q[i-1]) basic_delay_result_q[i]<=basic_delay_result_q[i-1];
      end
    end
    assign basic_valid=basic_delay_valid_q[5];
    assign basic_result=basic_delay_result_q[5];
  end else begin:g_basic_direct
    assign basic_valid=basic_raw_valid;
    assign basic_result=basic_raw_result;
  end endgenerate

'''+s[pos:]
p.write_text(s)

# Replace the C3 metadata slot with a matching eight-edge tag pipeline.
p=R/'rtl/vxm/lpu_vxm_execution_stage.sv';s=p.read_text()
start=s.index('  // One transaction is kept')
end=s.index('  lpu_vxm_datapath_mux #(',start)
s=s[:start]+'''  localparam bit C3_PIPELINED=SPECIAL_KIND==lpu_pkg::VXM_SPECIAL_RECIP_RSQRT;
  logic metadata_valid_q;
  logic [CONTAINER_WIDTH-1:0] metadata_original_q,metadata_auxiliary_q;
  logic metadata_end_marker_q;
  logic [7:0] tag_valid_q,tag_end_q;
  logic [CONTAINER_WIDTH-1:0] tag_original_q[0:7],tag_auxiliary_q[0:7];

'''+s[end:]
s=s.replace('input_ready_o = !metadata_valid_q || alu_output_valid;',
 'input_ready_o = alu_input_ready && (C3_PIPELINED || !metadata_valid_q || alu_output_valid);')
start=s.index('  always_ff @(posedge clk_i or negedge rst_ni)')
end=s.index('  lpu_vxm_alu #(',start)
old=s[start:end]
s=s[:start]+'''  generate if(C3_PIPELINED) begin:g_tags
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if(!rst_ni) begin
        tag_valid_q<=0;tag_end_q<=0;
        for(integer i=0;i<8;i++) begin tag_original_q[i]<=0;tag_auxiliary_q[i]<=0;end
      end else begin
        tag_valid_q<={tag_valid_q[6:0],request_accepted_o};
        tag_end_q<={tag_end_q[6:0],request_accepted_o && request_end_marker_i};
        if(request_accepted_o) begin tag_original_q[0]<=request_original;tag_auxiliary_q[0]<=request_auxiliary;end
        for(integer i=1;i<8;i++) if(tag_valid_q[i-1]) begin
          tag_original_q[i]<=tag_original_q[i-1];tag_auxiliary_q[i]<=tag_auxiliary_q[i-1];
        end
      end
    end
    assign metadata_valid_q=tag_valid_q[7];
    assign metadata_original_q=tag_original_q[7];
    assign metadata_auxiliary_q=tag_auxiliary_q[7];
    assign metadata_end_marker_q=tag_end_q[7];
  end else begin:g_single_tag
'''+old+'''  end endgenerate

'''+s[end:]
# Avoid a combinational cycle: ALU ready may depend on its valid input. C3 is
# always ready; legacy metadata admission retains its original independence.
s=s.replace('alu_input_valid = operands_valid && input_ready_o;',
 'alu_input_valid = operands_valid && (C3_PIPELINED || !metadata_valid_q || alu_output_valid);')
p.write_text(s)
print('Bound C3 pipeline, uniform Basic timing, and eight-edge metadata.')
