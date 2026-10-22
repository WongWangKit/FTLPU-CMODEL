`timescale 1ns/1ps

module lpu_vxm_log_tb;
  localparam integer CASE_COUNT = 2335;
  import lpu_pkg::*;
  import lpu_vxm_log_reference_pkg::*;
  logic clk=0,rst_n=0;
  always #5 clk=~clk;
  logic instruction_valid=0;
  logic [6:0] instruction={1'b0,VXM_LOG_INSTRUCTION};
  logic [1:0] dtype=0,chain=VXM_CHAIN_LENGTH_4;
  logic [31:0] operand=0;
  wire accepted,ready,result_valid,execution_fault;
  wire [31:0] result,original,auxiliary;
  wire result_end;
  wire request_valid,response_valid;
  wire [1:0] request_bank;
  wire [5:0] request_address;
  logic config_valid=0,write_valid=0;
  logic [5:0] write_address=0;
  logic [49:0] write_log=0;
  wire [3:0] configured;
  wire [63:0] input_min,segment_width;
  logic [7:0] requests;
  logic [47:0] addresses;
  logic [31:0] stages;
  wire [7:0] responses;
  wire [31:0] response_stages;
  wire [127:0] response_k,response_b;
  wire [399:0] response_log;
  wire collision,lut_fault;
  logic [49:0] rows[0:63];
  logic [65:0] cases[0:CASE_COUNT-1];
  wire [15:0] decode_faults;
  wire [63:0] decoded_opcodes;
  logic end_marker=0;

  always_comb begin
    requests='0;addresses='0;stages='0;
    if(request_valid) begin
      requests[request_bank]=1;
      addresses[request_bank*6+:6]=request_address;
      stages[request_bank*4+:4]=4'd3;
    end
  end
  assign response_valid=responses[3];

  lpu_vxm_tile_pair_lut #(.FUNCTION_COUNT(4),.LANES(1)) u_lut (
    .clk_i(clk),.rst_ni(rst_n),
    .config_valid_i(config_valid),.config_function_i(2'd3),
    .config_input_min_i(16'h3c00),.config_segment_width_i(16'h2400),
    .write_valid_i(write_valid),.write_function_i(2'd3),
    .write_address_i(write_address),.write_k_i(16'b0),.write_b_i(16'b0),.write_log_i(write_log),
    .request_valid_i(requests),.request_address_i(addresses),.request_stage_i(stages),
    .response_valid_o(responses),.response_stage_o(response_stages),
    .response_k_o(response_k),.response_b_o(response_b),.response_log_o(response_log),
    .configured_o(configured),.input_min_o(input_min),.segment_width_o(segment_width),
    .collision_o(collision),.fault_o(lut_fault)
  );
  lpu_vxm_execution_stage #(.LOCAL_QUEUE(3),.PHYSICAL_STAGE(3),
    .SPECIAL_KIND(VXM_SPECIAL_RECIP_RSQRT),.LUT_BANK_COUNT(4)) dut (
    .clk_i(clk),.rst_ni(rst_n),.instruction_valid_i(instruction_valid),.instruction_i(instruction),
    .chain_length_i(chain),.compute_dtype_i(dtype),.lhs_dtype_i(dtype),.rhs_dtype_i(dtype),
    .head_lhs_valid_i(1'b0),.head_lhs_data_i(32'b0),
    .head_rhs_valid_i(1'b0),.head_rhs_data_i(32'b0),
    .previous_valid_i(instruction_valid),.previous_value_i(operand),
    .previous_original_i(32'h12345678),.previous_auxiliary_i(32'h87654321),
    .feedback_valid_i(1'b0),.feedback_value_i(32'b0),
    .feedback_original_i(32'b0),.feedback_auxiliary_i(32'b0),
    .immediate_valid_i(1'b0),.immediate_data_i(32'b0),
    .accumulator_valid_i(1'b0),.accumulator_data_i(32'b0),
    .request_end_marker_i(end_marker),
    .lut_configured_i(configured),.lut_input_min_i(input_min),.lut_segment_width_i(segment_width),
    .lut_read_valid_o(request_valid),.lut_read_bank_o(request_bank),.lut_read_address_o(request_address),
    .lut_read_valid_i(response_valid),.lut_read_k_i(response_k[3*16+:16]),
    .lut_read_b_i(response_b[3*16+:16]),.lut_read_log_i(response_log[3*50+:50]),
    .input_ready_o(ready),.request_accepted_o(accepted),.result_valid_o(result_valid),
    .result_value_o(result),.result_original_o(original),.result_auxiliary_o(auxiliary),
    .result_end_marker_o(result_end),.chain_head_o(),.chain_tail_o(),.fault_o(execution_fault)
  );
  for(genvar s=0;s<16;s++) begin:g_decode
    lpu_vxm_local_decoder #(.LOCAL_QUEUE(s%8),.PHYSICAL_STAGE(s)) u_decode (
      .instruction_i(instruction),.chain_length_i(chain),
      .opcode_o(decoded_opcodes[s*4+:4]),.lhs_source_o(),.rhs_source_o(),
      .chain_head_o(),.chain_tail_o(),.illegal_instruction_o(decode_faults[s])
    );
  end
  task automatic execute_case(input integer index);
    integer elapsed;
    begin
      @(negedge clk);
      dtype=cases[index][65:64];operand=cases[index][63:32];
      chain=2'(index%3);instruction={1'b0,VXM_LOG_INSTRUCTION};
      end_marker=index[0];instruction_valid=1;
      #1;
      if(!accepted||!ready) $fatal(1,"Log request %0d was not accepted",index);
      for(integer s=0;s<16;s++)
        if(decode_faults[s] !== ((s%4)!=3)) $fatal(1,"Log placement mismatch at stage %0d",s);
      @(posedge clk);#1;
      if(result_valid) $fatal(1,"Log returned before the special pipeline");
      @(negedge clk);instruction_valid=0;
      elapsed=0;
      while(!result_valid&&elapsed<12) begin
        @(posedge clk);#1;elapsed++;
        if(execution_fault||lut_fault||collision) $fatal(1,"Log transaction fault");
      end
      if(!result_valid||elapsed!=7) $fatal(1,"Log eight-cycle latency mismatch: %0d",elapsed);
      if(!log_within_ulp(result,cases[index][31:0],dtype))
        $fatal(1,"Log mismatch case %0d dtype %0d x=%h got=%h expected=%h",index,dtype,operand,result,cases[index][31:0]);
      if(original!=32'h12345678||auxiliary!=32'h87654321||result_end!==end_marker)
        $fatal(1,"Log token metadata mismatch");
    end
  endtask
  initial begin
    $readmemh("sim/vectors/vxm_log_lut.hex",rows);
    $readmemh("sim/vectors/vxm_log_cases.hex",cases);
    repeat(3) @(posedge clk);
    @(negedge clk);rst_n=1;config_valid=1;
    @(negedge clk);config_valid=0;
    for(integer i=0;i<64;i++) begin
      write_valid=1;write_address=6'(i);write_log=rows[i];@(negedge clk);
    end
    write_valid=0;
    for(integer i=0;i<CASE_COUNT;i++) execute_case(i);
    // An extended noncanonical instruction must not be accepted.
    @(negedge clk);instruction=7'h27;instruction_valid=1;#1;
    if(accepted||!execution_fault) $fatal(1,"Illegal Log extension accepted");
    instruction_valid=0;
    @(negedge clk);rst_n=0;
    repeat(2) @(posedge clk);#1;
    if(result_valid) $fatal(1,"Log result survived reset");
    $display("VXM_LOG TEST_PASS");$finish;
  end
  initial begin #1000000;$fatal(1,"Log test timeout");end
endmodule
