`timescale 1ns/1ps
// Black-box scoreboard: only instruction/data/configuration ports and public
// outputs are observed. Golden results come from Python math, not RTL helpers.
module lpu_vxm_c3_pipeline_tb #(parameter integer PHYSICAL_STAGE=3);
  import lpu_pkg::*;
  `include "sim/vectors/vxm_c3_pipeline_count.svh"
  logic clk=0,rst_n=0;
  always #5 clk=~clk;
  logic instruction_valid=0;
  logic [6:0] instruction=0;
  logic [1:0] dtype=0,chain=VXM_CHAIN_LENGTH_4;
  logic [31:0] lhs=0,rhs=0,token_original=0,token_auxiliary=0;
  logic end_marker=0;
  wire ready,accepted,result_valid,execution_fault,result_end;
  wire [31:0] result,original,auxiliary;
  logic config_valid=0,write_valid=0;
  logic [1:0] function_id=0;
  logic [5:0] write_address=0;
  logic [15:0] write_k=0,write_b=0;
  logic [49:0] write_log=0;
  wire [3:0] configured;
  wire [63:0] input_min,segment_width;
  wire request_valid;
  wire [1:0] request_bank;
  wire [5:0] request_address;
  logic [7:0] requests;
  logic [47:0] addresses;
  logic [31:0] stages;
  wire [7:0] responses;
  wire [31:0] response_stages;
  wire [127:0] response_k,response_b;
  wire [399:0] response_log;
  logic response_valid;
  logic [15:0] selected_k,selected_b;
  logic [49:0] selected_log;
  logic drop_response=0,expect_fault=0;
  wire collision,lut_fault;
  logic [107:0] cases[0:CASE_COUNT-1];
  logic [31:0] coefficient_rows[0:127];
  logic [49:0] log_rows[0:63];
  logic [31:0] expected_now;
  logic [2:0] tolerance_now;
  integer case_index_now;
  logic [31:0] expected_queue[0:31],original_queue[0:31],auxiliary_queue[0:31];
  logic [1:0] format_queue[0:31];
  logic [2:0] tolerance_queue[0:31];
  logic end_queue[0:31];
  integer cycle_queue[0:31],index_queue[0:31];
  integer head=0,tail=0,cycle=0,checked=0,issued=0,max_ulp[0:2];
  integer consecutive=0,longest=0;
  logic fault_seen=0;

  always_comb begin
    requests=0;addresses=0;stages=0;
    if(request_valid) begin
      requests[request_bank]=1;
      addresses[request_bank*6+:6]=request_address;
      stages[request_bank*4+:4]=4'(PHYSICAL_STAGE);
    end
    response_valid=0;selected_k=0;selected_b=0;selected_log=0;
    for(integer bank=0;bank<4;bank++) if(responses[bank]) begin
      response_valid=!drop_response;
      selected_k=response_k[bank*16+:16];selected_b=response_b[bank*16+:16];
      selected_log=response_log[bank*50+:50];
    end
  end
  lpu_vxm_tile_pair_lut #(.FUNCTION_COUNT(4),.LANES(1)) u_lut (
    .clk_i(clk),.rst_ni(rst_n),.config_valid_i(config_valid),.config_function_i(function_id),
    .config_input_min_i(16'h3c00),.config_segment_width_i(16'h2400),
    .write_valid_i(write_valid),.write_function_i(function_id),.write_address_i(write_address),
    .write_k_i(write_k),.write_b_i(write_b),.write_log_i(write_log),
    .request_valid_i(requests),.request_address_i(addresses),.request_stage_i(stages),
    .response_valid_o(responses),.response_stage_o(response_stages),
    .response_k_o(response_k),.response_b_o(response_b),.response_log_o(response_log),
    .configured_o(configured),.input_min_o(input_min),.segment_width_o(segment_width),
    .collision_o(collision),.fault_o(lut_fault)
  );
  lpu_vxm_execution_stage #(.LOCAL_QUEUE(PHYSICAL_STAGE%8),.PHYSICAL_STAGE(PHYSICAL_STAGE),
    .SPECIAL_KIND(VXM_SPECIAL_RECIP_RSQRT),.LUT_BANK_COUNT(4)) dut (
    .clk_i(clk),.rst_ni(rst_n),.instruction_valid_i(instruction_valid),.instruction_i(instruction),
    .chain_length_i(chain),.compute_dtype_i(dtype),.lhs_dtype_i(dtype),.rhs_dtype_i(dtype),
    .head_lhs_valid_i(1'b0),.head_lhs_data_i(32'b0),.head_rhs_valid_i(1'b0),.head_rhs_data_i(32'b0),
    .previous_valid_i(instruction_valid),.previous_value_i(lhs),
    .previous_original_i(token_original),.previous_auxiliary_i(token_auxiliary),
    .feedback_valid_i(1'b0),.feedback_value_i(32'b0),.feedback_original_i(32'b0),.feedback_auxiliary_i(32'b0),
    .immediate_valid_i(1'b1),.immediate_data_i(rhs),
    .accumulator_valid_i(1'b0),.accumulator_data_i(32'b0),.request_end_marker_i(end_marker),
    .lut_configured_i(configured),.lut_input_min_i(input_min),.lut_segment_width_i(segment_width),
    .lut_read_valid_o(request_valid),.lut_read_bank_o(request_bank),.lut_read_address_o(request_address),
    .lut_read_valid_i(response_valid),.lut_read_k_i(selected_k),.lut_read_b_i(selected_b),.lut_read_log_i(selected_log),
    .input_ready_o(ready),.request_accepted_o(accepted),.result_valid_o(result_valid),
    .result_value_o(result),.result_original_o(original),.result_auxiliary_o(auxiliary),
    .result_end_marker_o(result_end),.chain_head_o(),.chain_tail_o(),.fault_o(execution_fault)
  );

  function automatic integer ulp_distance(input logic [31:0] got,ref_value,input logic [1:0] fmt);
    logic [31:0] g,r,gm,rm,exp_mask,frac_mask,sign_mask;
    begin
      g=fmt==VXM_FORMAT_FP32?got:{16'b0,got[15:0]};
      r=fmt==VXM_FORMAT_FP32?ref_value:{16'b0,ref_value[15:0]};
      exp_mask=fmt==VXM_FORMAT_FP32?32'h7f800000:(fmt==VXM_FORMAT_FP16?32'h7c00:32'h7f80);
      frac_mask=fmt==VXM_FORMAT_FP32?32'h007fffff:(fmt==VXM_FORMAT_FP16?32'h03ff:32'h007f);
      sign_mask=fmt==VXM_FORMAT_FP32?32'h80000000:32'h8000;
      gm=g&~sign_mask;rm=r&~sign_mask;
      if((r&exp_mask)==exp_mask) ulp_distance=g==r?0:32'h7fffffff;
      else if((g&exp_mask)==exp_mask) ulp_distance=32'h7fffffff;
      else if((g&sign_mask)!=(r&sign_mask)) ulp_distance=gm==0 && rm==0?0:32'h7fffffff;
      else ulp_distance=integer'(gm>rm?gm-rm:rm-gm);
      if(fmt!=VXM_FORMAT_FP32 && got[31:16]!=0) ulp_distance=32'h7fffffff;
    end
  endfunction

  // Sample acceptance at the edge, then observe registered outputs after NBA.
  always @(posedge clk) begin:scoreboard
    integer slot,error;
    if(!rst_n) begin head=0;tail=0;cycle=0;consecutive=0;end
    else begin
      cycle++;
      if(instruction_valid && !expect_fault) begin
        if(!ready || !accepted) $fatal(1,"II=1 violated case %0d cycle %0d",case_index_now,cycle);
        if(tail-head>=32) $fatal(1,"Scoreboard overflow");
        slot=tail%32;expected_queue[slot]=expected_now;tolerance_queue[slot]=tolerance_now;
        original_queue[slot]=token_original;auxiliary_queue[slot]=token_auxiliary;
        format_queue[slot]=dtype;end_queue[slot]=end_marker;
        cycle_queue[slot]=cycle;index_queue[slot]=case_index_now;tail++;issued++;
        consecutive++;if(consecutive>longest)longest=consecutive;
      end else consecutive=0;
      #1;
      if(execution_fault)fault_seen=1;
      if((execution_fault && !expect_fault)||lut_fault||collision) $fatal(1,"Unexpected public fault at cycle %0d",cycle);
      if(result_valid && !expect_fault) begin
        if(head==tail) $fatal(1,"Unrequested or duplicate result");
        slot=head%32;
        if(cycle-cycle_queue[slot]!=7) $fatal(1,"Latency changed: %0d",cycle-cycle_queue[slot]);
        error=ulp_distance(result,expected_queue[slot],format_queue[slot]);
        if(error>tolerance_queue[slot])
          $fatal(1,"case %0d fmt %0d got=%h expected=%h ULP=%0d",index_queue[slot],format_queue[slot],result,expected_queue[slot],error);
        if(error>max_ulp[format_queue[slot]])max_ulp[format_queue[slot]]=error;
        if(original!==original_queue[slot] || auxiliary!==auxiliary_queue[slot] || result_end!==end_queue[slot])
          $fatal(1,"Public token metadata mismatch case %0d",index_queue[slot]);
        head++;checked++;
      end
      if(head<tail && cycle-cycle_queue[head%32]>=7)
        $fatal(1,"Missing or late result case %0d",index_queue[head%32]);
    end
  end

  task automatic drive_case(input integer index);
    begin
      @(negedge clk);
      dtype=cases[index][107:106];instruction=cases[index][105:99];
      // Basic operations select the public immediate RHS source.
      if(instruction[5:0]<6) instruction[4:3]=2'b10;
      lhs=cases[index][98:67];rhs=cases[index][66:35];
      expected_now=cases[index][34:3];tolerance_now=cases[index][2:0];
      token_original=32'h13579bdf^32'(index);token_auxiliary=32'h2468ace0+32'(index);
      end_marker=(index%13)==0;chain=2'(index%3);case_index_now=index;instruction_valid=1;
    end
  endtask
  task automatic drain;
    begin
      @(negedge clk);instruction_valid=0;
      repeat(10)@(negedge clk);
      if(head!=tail) $fatal(1,"Outstanding scoreboard entries after drain");
    end
  endtask
  task automatic expect_numeric_fault(input bit missing_response,input logic [6:0] selected_instruction=7'h26);
    begin
      @(negedge clk);
      expect_fault=1;fault_seen=0;drop_response=missing_response;
      dtype=VXM_FORMAT_FP32;instruction=selected_instruction;lhs=32'h40000000;instruction_valid=1;
      #1;if(!accepted)$fatal(1,"Faulting Log request was not accepted");
      @(posedge clk);#2;
      @(negedge clk);instruction_valid=0;
      repeat(7)begin @(posedge clk);#2;end
      if(!result_valid || result!=32'h7fc00000 || !fault_seen)
        $fatal(1,"Missing LUT/configuration did not report fault and NaN");
      @(negedge clk);drop_response=0;
      repeat(2)@(negedge clk);
    end
  endtask
  initial begin
    for(integer i=0;i<3;i++)max_ulp[i]=0;
    $readmemh("sim/vectors/vxm_c3_pipeline_cases.hex",cases);
    $readmemh("sim/vectors/vxm_c3_recip_rsqrt_lut.hex",coefficient_rows);
    $readmemh("sim/vectors/vxm_log_lut.hex",log_rows);
    repeat(3)@(negedge clk);rst_n=1;
    for(integer bank=1;bank<4;bank++) begin
      config_valid=1;function_id=2'(bank);@(negedge clk);config_valid=0;
      for(integer addr=0;addr<64;addr++) begin
        write_valid=1;write_address=6'(addr);
        if(bank==3)write_log=log_rows[addr];
        else begin write_k=coefficient_rows[(bank-1)*64+addr][31:16];write_b=coefficient_rows[(bank-1)*64+addr][15:0];end
        @(negedge clk);
      end
      write_valid=0;
    end
    // Every cycle has an instruction: mixed functions, dtypes and Basic ops.
    for(integer i=0;i<CASE_COUNT;i++)drive_case(i);
    drain();
    // Bubbles must preserve ordering and exact latency.
    for(integer i=0;i<512;i++) begin
      drive_case(i);
      if(i%5==0) begin @(negedge clk);instruction_valid=0;end
    end
    drain();
    if(checked!=CASE_COUNT+512 || longest<CASE_COUNT) $fatal(1,"Throughput coverage incomplete");
    // Reset while seven requests are still in flight; none may escape reset.
    for(integer i=0;i<7;i++)drive_case(i);
    @(negedge clk);instruction_valid=0;rst_n=0;
    repeat(2)@(negedge clk);rst_n=1;
    repeat(10) begin @(negedge clk);if(result_valid)$fatal(1,"Stale result survived reset");end
    // Unconfigured numeric lookup must fault; special values bypass the LUT.
    expect_numeric_fault(0);
    config_valid=1;function_id=3;write_valid=1;write_address=0;write_log=log_rows[0];
    @(negedge clk);config_valid=0;write_valid=0;
    expect_numeric_fault(1);
    // Verify recovery after a dropped response with a normal numeric request.
    @(negedge clk);expect_fault=0;instruction=7'h26;dtype=VXM_FORMAT_FP32;
    lhs=32'h40000000;expected_now=32'h3f317218;tolerance_now=2;case_index_now=-1;instruction_valid=1;
    drain();
    // An invalid Rsqrt Newton correction must retain its public range fault.
    expect_fault=1;config_valid=1;function_id=2;write_valid=1;write_address=32;write_k=0;write_b=0;
    @(negedge clk);config_valid=0;write_valid=0;
    expect_numeric_fault(0,7'h07);
    // Illegal instruction is rejected through the decoder's public fault.
    expect_fault=1;instruction=7'h27;instruction_valid=1;#1;
    if(accepted || !execution_fault)$fatal(1,"Illegal instruction accepted");
    @(negedge clk);instruction_valid=0;
    instruction=7'h26;dtype=2'b11;instruction_valid=1;#1;
    if(accepted || !execution_fault)$fatal(1,"Unsupported dtype accepted");
    @(negedge clk);instruction_valid=0;
    $display("C3_PIPELINE TEST_PASS stage=%0d checked=%0d longest_II1_burst=%0d max_ULP=%0d/%0d/%0d",
      PHYSICAL_STAGE,checked,longest,max_ulp[0],max_ulp[1],max_ulp[2]);
    $finish;
  end
  initial begin #10000000;$fatal(1,"C3 black-box timeout");end
endmodule
