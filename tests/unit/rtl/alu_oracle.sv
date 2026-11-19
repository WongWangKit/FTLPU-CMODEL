`timescale 1ns/1ps
module alu_oracle;
  parameter integer KIND=2;
  parameter integer BASIC=0;
  reg clk=0,rst_n=0,valid=0;
  reg [1:0] fmt=0;reg [3:0] op=0;reg [31:0] a=0,b=0;
  wire ready,out_valid,illegal,unsupported,collision,fault;
  wire [31:0] result;
  wire lut_request;wire [1:0] lut_bank;wire [5:0] lut_address;
  reg response=0;reg [15:0] k=0,offset=0;reg [49:0] log_row=0;
  reg [63:0] table_rows[0:255];reg [15:0] config_min[0:3],config_width[0:3];
  wire [63:0] minima={config_min[3],config_min[2],config_min[1],config_min[0]};
  wire [63:0] widths={config_width[3],config_width[2],config_width[1],config_width[0]};
  wire operand_format_valid,operand_sign,operand_zero,operand_inf,operand_nan;
  wire [7:0] operand_exponent;wire [22:0] operand_fraction;
  lpu_vxm_float_unpack unpack(.data_format_i(fmt),.value_i(a),.format_valid_o(operand_format_valid),
    .sign_o(operand_sign),.exponent_o(operand_exponent),.fraction_o(operand_fraction),
    .effective_zero_o(operand_zero),.is_inf_o(operand_inf),.is_nan_o(operand_nan));
  generate if(BASIC)begin:basic_oracle
  lpu_vxm_alu #(.SPECIAL_KIND(KIND),.LUT_BANK_COUNT(4)) dut(
    .clk_i(clk),.rst_ni(rst_n),.input_valid_i(valid),.data_format_i(fmt),.opcode_i(op),.lhs_i(a),.rhs_i(b),
    .lut_configured_i(4'b1111),.lut_input_min_i(minima),.lut_segment_width_i(widths),
    .lut_read_valid_o(lut_request),.lut_read_bank_o(lut_bank),.lut_read_address_o(lut_address),
    .lut_read_valid_i(response),.lut_read_k_i(k),.lut_read_b_i(offset),.lut_read_log_i(log_row),
    .input_ready_o(ready),.output_valid_o(out_valid),.result_o(result),.illegal_opcode_o(illegal),
    .unsupported_format_o(unsupported),.result_collision_o(collision),.lut_fault_o(fault));
  end else begin:special_oracle
  assign unsupported=0;assign collision=0;
  lpu_vxm_special_alu #(.SPECIAL_KIND(KIND),.LUT_BANK_COUNT(4)) dut(
    .clk_i(clk),.rst_ni(rst_n),.valid_i(valid),.data_format_i(fmt),.opcode_i(op),
    .operand_format_valid_i(operand_format_valid),.operand_sign_i(operand_sign),.operand_exponent_i(operand_exponent),
    .operand_fraction_i(operand_fraction),.operand_zero_i(operand_zero),.operand_inf_i(operand_inf),.operand_nan_i(operand_nan),
    .lut_configured_i(4'b1111),.lut_input_min_i(minima),.lut_segment_width_i(widths),
    .lut_read_valid_o(lut_request),.lut_read_bank_o(lut_bank),.lut_read_address_o(lut_address),
    .lut_read_valid_i(response),.lut_read_k_i(k),.lut_read_b_i(offset),.lut_read_log_i(log_row),
    .ready_o(ready),.valid_o(out_valid),.result_o(result),.illegal_opcode_o(illegal),.lut_fault_o(fault));
  end endgenerate
  always @(posedge clk) begin
    response<=lut_request;
    if(lut_request)begin
      if(lut_bank==0)begin k<={3'b0,table_rows[lut_bank*64+lut_address][25:13]};offset<={3'b0,table_rows[lut_bank*64+lut_address][12:0]};end
      else begin k<=table_rows[lut_bank*64+lut_address][31:16];offset<=table_rows[lut_bank*64+lut_address][15:0];end
      log_row<=table_rows[lut_bank*64+lut_address][49:0];
    end
  end
  integer fd,fo,rc,t=0,v,format_value,operation;string input_path,output_path,lut_path;
  initial begin
    if(!$value$plusargs("INPUT=%s",input_path)||!$value$plusargs("OUTPUT=%s",output_path)||!$value$plusargs("LUT=%s",lut_path))$fatal(1,"paths");
    $readmemh(lut_path,table_rows);
    config_min[0]=16'hb800;config_width[0]=16'h2400;
    config_min[1]=16'h3c00;config_width[1]=16'h2400;
    config_min[2]=16'h3c00;config_width[2]=16'h2800;
    config_min[3]=16'h3c00;config_width[3]=16'h2400;
    rst_n=1;#1;rst_n=0;#2;clk=1;#1;clk=0;rst_n=1;
    fd=$fopen(input_path,"r");fo=$fopen(output_path,"w");
    if (!fd || !fo) $fatal(1,"cannot open test trace file");
    while(!$feof(fd))begin
      rc=$fscanf(fd,"%d %d %d %h %h\n",v,format_value,operation,a,b);
      if(rc==5)begin
        valid=v;fmt=format_value;op=operation;
        #1;
        if(valid&&!ready)$fatal(1,"oracle request not ready E%0d",t);
        #1;clk=1;#1;
        $fwrite(fo,"%0d %0d %08x %0d\n",t,out_valid,out_valid?result:0,illegal||unsupported||collision||fault);
        clk=0;#1;t=t+1;
      end
    end
    $fclose(fd);$fclose(fo);$display("ALU_ORACLE PASS cycles=%0d",t);$finish;
  end
endmodule
