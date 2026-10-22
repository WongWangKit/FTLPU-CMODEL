`timescale 1ns/1ps

module lpu_vxm_execution_stage_tb;
  import lpu_pkg::*;

  logic clk;
  logic rst_n;
  logic instruction_valid;
  logic [6:0] instruction;
  logic [1:0] compute_dtype;
  logic head_lhs_valid;
  logic [31:0] head_lhs_data;
  logic head_rhs_valid;
  logic [31:0] head_rhs_data;
  wire input_ready;
  wire request_accepted;
  wire result_valid;
  wire [31:0] result_value;
  wire [31:0] result_original;
  wire [31:0] result_auxiliary;
  wire chain_head;
  wire chain_tail;
  wire fault;

  always #5 clk = ~clk;

  lpu_vxm_execution_stage #(
    .LOCAL_QUEUE(0),
    .PHYSICAL_STAGE(0),
    .SPECIAL_KIND(VXM_SPECIAL_NONE)
  ) dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .instruction_valid_i(instruction_valid),
    .instruction_i(instruction),
    .chain_length_i(VXM_CHAIN_LENGTH_8),
    .compute_dtype_i(compute_dtype),
    .lhs_dtype_i(compute_dtype),
    .rhs_dtype_i(compute_dtype),
    .head_lhs_valid_i(head_lhs_valid),
    .head_lhs_data_i(head_lhs_data),
    .head_rhs_valid_i(head_rhs_valid),
    .head_rhs_data_i(head_rhs_data),
    .previous_valid_i(1'b0),
    .previous_value_i('0),
    .previous_original_i('0),
    .previous_auxiliary_i('0),
    .feedback_valid_i(1'b0),
    .feedback_value_i('0),
    .feedback_original_i('0),
    .feedback_auxiliary_i('0),
    .immediate_valid_i(1'b0),
    .immediate_data_i('0),
    .accumulator_valid_i(1'b0),
    .accumulator_data_i('0),
    .request_end_marker_i(1'b0),
    .lut_configured_i('0),
    .lut_input_min_i('0),
    .lut_segment_width_i('0),
    .lut_read_valid_o(),
    .lut_read_bank_o(),
    .lut_read_address_o(),
    .lut_read_valid_i(1'b0),
    .lut_read_k_i('0),
    .lut_read_b_i('0),
    .input_ready_o(input_ready),
    .request_accepted_o(request_accepted),
    .result_valid_o(result_valid),
    .result_value_o(result_value),
    .result_original_o(result_original),
    .result_auxiliary_o(result_auxiliary),
    .result_end_marker_o(),
    .chain_head_o(chain_head),
    .chain_tail_o(chain_tail),
    .fault_o(fault)
  );

  task automatic check_condition(input logic condition, input string message);
    if (!condition) $fatal(1, "%s", message);
  endtask

  task automatic check_raw_instruction(
    input logic [1:0] format,
    input logic [2:0] opcode,
    input logic [31:0] raw_lhs,
    input logic [31:0] expected_result
  );
    logic [31:0] expected_token;
    begin
      expected_token = format == VXM_FORMAT_FP32 ? raw_lhs :
        {16'b0, raw_lhs[15:0]};
      @(negedge clk);
      compute_dtype = format;
      instruction = {4'b0, opcode}; // Q0: two head streams
      head_lhs_data = raw_lhs;
      head_rhs_data = 32'h00000001; // raw auxiliary token is preserved too
      head_rhs_valid = 1'b1;
      instruction_valid = 1'b1;
      #1;
      check_condition(input_ready && request_accepted && !fault,
                      "raw instruction was not accepted");
      @(posedge clk);
      #1;
      if (result_valid !== 1'b1 || result_value !== expected_result ||
          result_original !== expected_token ||
          result_auxiliary !== 32'h00000001 || fault !== 1'b0)
        $fatal(1,
          "raw op=%0d format=%0d lhs=%h result=%h expected=%h token=%h expected_token=%h",
          opcode, format, raw_lhs, result_value, expected_result,
          result_original, expected_token);
      @(negedge clk);
      instruction_valid = 1'b0;
      @(posedge clk);
      #1;
      check_condition(!result_valid && !fault,
                      "raw instruction result did not retire");
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    instruction_valid = 1'b0;
    instruction = '0;
    compute_dtype = VXM_FORMAT_FP16;
    head_lhs_valid = 1'b1;
    head_lhs_data = 32'h00003c00;
    head_rhs_valid = 1'b1;
    head_rhs_data = 32'h00004000;

    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    // Compact Add + selected head operands -> ALU -> result token.
    @(negedge clk);
    instruction = 7'b0000001;
    instruction_valid = 1'b1;
    #1;
    check_condition(input_ready && request_accepted,
                    "Add request was not accepted");
    @(posedge clk);
    #1;
    check_condition(result_valid && result_value == 32'h00004200,
                    "unified Add stage expected FP16 3.0");
    check_condition(result_original == 32'h00003c00 &&
                    result_auxiliary == 32'h00004000,
                    "Add result token metadata did not match its operands");
    check_condition(chain_head && !chain_tail && !fault,
                    "stage-0 chain role or fault output is wrong");

    @(negedge clk);
    instruction_valid = 1'b0;
    @(posedge clk);
    #1;
    check_condition(!result_valid, "Add result_valid lasted too long");

    // Multiply takes an extra cycle, while metadata stays attached.
    @(negedge clk);
    instruction = 7'b0000011;
    head_lhs_data = 32'h00003e00;
    instruction_valid = 1'b1;
    @(posedge clk);
    #1;
    check_condition(!result_valid, "Multiply completed too early");
    @(negedge clk);
    instruction_valid = 1'b0;
    @(posedge clk);
    #1;
    check_condition(result_valid && result_value == 32'h00004200,
                    "unified Multiply stage expected FP16 3.0");
    check_condition(result_original == 32'h00003e00 &&
                    result_auxiliary == 32'h00004000,
                    "Multiply result metadata was not latency-aligned");
    check_condition(!fault, "unexpected execution-stage fault");

    @(negedge clk);
    head_rhs_valid = 1'b0;
    instruction = 7'b0000001;
    instruction_valid = 1'b1;
    #1;
    check_condition(!request_accepted && !fault,
                    "missing data must stall without issuing or faulting");

    // These use the compact instruction and head-stream ports, not the ALU
    // inputs. NEGATE flips only the sign; BYPASS keeps every source payload bit.
    check_raw_instruction(VXM_FORMAT_FP16, VXM_LOCAL_NEGATE,
                          32'h00000001, 32'h00008001);
    check_raw_instruction(VXM_FORMAT_FP16, VXM_LOCAL_NEGATE,
                          32'h00007e01, 32'h0000fe01);
    check_raw_instruction(VXM_FORMAT_FP16, VXM_LOCAL_NEGATE,
                          32'h00000000, 32'h00008000);
    check_raw_instruction(VXM_FORMAT_FP16, VXM_LOCAL_BYPASS,
                          32'h00000001, 32'h00000001);
    check_raw_instruction(VXM_FORMAT_FP16, VXM_LOCAL_BYPASS,
                          32'h00007e01, 32'h00007e01);
    check_raw_instruction(VXM_FORMAT_BF16, VXM_LOCAL_NEGATE,
                          32'h00000001, 32'h00008001);
    check_raw_instruction(VXM_FORMAT_BF16, VXM_LOCAL_NEGATE,
                          32'h00007fc1, 32'h0000ffc1);
    check_raw_instruction(VXM_FORMAT_BF16, VXM_LOCAL_NEGATE,
                          32'h00000000, 32'h00008000);
    check_raw_instruction(VXM_FORMAT_BF16, VXM_LOCAL_BYPASS,
                          32'h00000001, 32'h00000001);
    check_raw_instruction(VXM_FORMAT_BF16, VXM_LOCAL_BYPASS,
                          32'h00007fc1, 32'h00007fc1);
    check_raw_instruction(VXM_FORMAT_FP32, VXM_LOCAL_NEGATE,
                          32'h00000001, 32'h80000001);
    check_raw_instruction(VXM_FORMAT_FP32, VXM_LOCAL_NEGATE,
                          32'h7fc00001, 32'hffc00001);
    check_raw_instruction(VXM_FORMAT_FP32, VXM_LOCAL_NEGATE,
                          32'h00000000, 32'h80000000);
    check_raw_instruction(VXM_FORMAT_FP32, VXM_LOCAL_BYPASS,
                          32'h00000001, 32'h00000001);
    check_raw_instruction(VXM_FORMAT_FP32, VXM_LOCAL_BYPASS,
                          32'h7fc00001, 32'h7fc00001);

    $display("LPU_VXM_EXECUTION_STAGE_TB_PASS");
    $finish;
  end
endmodule
