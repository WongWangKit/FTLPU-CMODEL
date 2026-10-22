`timescale 1ns/1ps

module lpu_vxm_datapath_mux_tb;
  import lpu_pkg::*;

  logic instruction_valid;
  logic [1:0] chain_length;
  logic [1:0] compute_dtype;
  logic [1:0] lhs_dtype;
  logic [1:0] rhs_dtype;
  logic head_lhs_valid;
  logic [31:0] head_lhs_data;
  logic head_rhs_valid;
  logic [31:0] head_rhs_data;
  logic previous_valid;
  logic [31:0] previous_value;
  logic [31:0] previous_original;
  logic [31:0] previous_auxiliary;
  logic feedback_valid;
  logic [31:0] feedback_value;
  logic [31:0] feedback_original;
  logic [31:0] feedback_auxiliary;
  logic immediate_valid;
  logic [31:0] immediate_data;
  logic accumulator_valid;
  logic [31:0] accumulator_data;

  logic [6:0] q0_instruction;
  wire q0_valid;
  wire [2:0] q0_opcode;
  wire [31:0] q0_lhs;
  wire [31:0] q0_rhs;
  wire [31:0] q0_original;
  wire [31:0] q0_auxiliary;
  wire q0_head;
  wire q0_tail;
  wire q0_decode_fault;
  wire q0_conversion_fault;

  logic [6:0] q2_instruction;
  wire q2_valid;
  wire [2:0] q2_opcode;
  wire [31:0] q2_lhs;
  wire [31:0] q2_rhs;
  wire [31:0] q2_original;
  wire [31:0] q2_auxiliary;
  wire q2_head;
  wire q2_tail;
  wire q2_decode_fault;
  wire q2_conversion_fault;

  logic [6:0] q3_instruction;
  wire q3_valid;
  wire [31:0] q3_lhs;
  wire [31:0] q3_rhs;
  wire q3_head;
  wire q3_tail;
  wire q3_decode_fault;

  lpu_vxm_datapath_mux #(.LOCAL_QUEUE(0), .PHYSICAL_STAGE(0)) u_q0 (
    .instruction_valid_i(instruction_valid),
    .instruction_i(q0_instruction),
    .chain_length_i(chain_length),
    .compute_dtype_i(compute_dtype),
    .lhs_dtype_i(lhs_dtype),
    .rhs_dtype_i(rhs_dtype),
    .head_lhs_valid_i(head_lhs_valid),
    .head_lhs_data_i(head_lhs_data),
    .head_rhs_valid_i(head_rhs_valid),
    .head_rhs_data_i(head_rhs_data),
    .previous_valid_i(previous_valid),
    .previous_value_i(previous_value),
    .previous_original_i(previous_original),
    .previous_auxiliary_i(previous_auxiliary),
    .feedback_valid_i(feedback_valid),
    .feedback_value_i(feedback_value),
    .feedback_original_i(feedback_original),
    .feedback_auxiliary_i(feedback_auxiliary),
    .immediate_valid_i(immediate_valid),
    .immediate_data_i(immediate_data),
    .accumulator_valid_i(accumulator_valid),
    .accumulator_data_i(accumulator_data),
    .operands_valid_o(q0_valid),
    .opcode_o(q0_opcode),
    .lhs_o(q0_lhs),
    .rhs_o(q0_rhs),
    .token_original_o(q0_original),
    .token_auxiliary_o(q0_auxiliary),
    .chain_head_o(q0_head),
    .chain_tail_o(q0_tail),
    .decode_fault_o(q0_decode_fault),
    .conversion_fault_o(q0_conversion_fault)
  );

  lpu_vxm_datapath_mux #(.LOCAL_QUEUE(2), .PHYSICAL_STAGE(2)) u_q2 (
    .instruction_valid_i(instruction_valid),
    .instruction_i(q2_instruction),
    .chain_length_i(chain_length),
    .compute_dtype_i(compute_dtype),
    .lhs_dtype_i(lhs_dtype),
    .rhs_dtype_i(rhs_dtype),
    .head_lhs_valid_i(head_lhs_valid),
    .head_lhs_data_i(head_lhs_data),
    .head_rhs_valid_i(head_rhs_valid),
    .head_rhs_data_i(head_rhs_data),
    .previous_valid_i(previous_valid),
    .previous_value_i(previous_value),
    .previous_original_i(previous_original),
    .previous_auxiliary_i(previous_auxiliary),
    .feedback_valid_i(feedback_valid),
    .feedback_value_i(feedback_value),
    .feedback_original_i(feedback_original),
    .feedback_auxiliary_i(feedback_auxiliary),
    .immediate_valid_i(immediate_valid),
    .immediate_data_i(immediate_data),
    .accumulator_valid_i(accumulator_valid),
    .accumulator_data_i(accumulator_data),
    .operands_valid_o(q2_valid),
    .opcode_o(q2_opcode),
    .lhs_o(q2_lhs),
    .rhs_o(q2_rhs),
    .token_original_o(q2_original),
    .token_auxiliary_o(q2_auxiliary),
    .chain_head_o(q2_head),
    .chain_tail_o(q2_tail),
    .decode_fault_o(q2_decode_fault),
    .conversion_fault_o(q2_conversion_fault)
  );

  lpu_vxm_datapath_mux #(.LOCAL_QUEUE(3), .PHYSICAL_STAGE(3)) u_q3 (
    .instruction_valid_i(instruction_valid),
    .instruction_i(q3_instruction),
    .chain_length_i(chain_length),
    .compute_dtype_i(compute_dtype),
    .lhs_dtype_i(lhs_dtype),
    .rhs_dtype_i(rhs_dtype),
    .head_lhs_valid_i(head_lhs_valid),
    .head_lhs_data_i(head_lhs_data),
    .head_rhs_valid_i(head_rhs_valid),
    .head_rhs_data_i(head_rhs_data),
    .previous_valid_i(previous_valid),
    .previous_value_i(previous_value),
    .previous_original_i(previous_original),
    .previous_auxiliary_i(previous_auxiliary),
    .feedback_valid_i(feedback_valid),
    .feedback_value_i(feedback_value),
    .feedback_original_i(feedback_original),
    .feedback_auxiliary_i(feedback_auxiliary),
    .immediate_valid_i(immediate_valid),
    .immediate_data_i(immediate_data),
    .accumulator_valid_i(accumulator_valid),
    .accumulator_data_i(accumulator_data),
    .operands_valid_o(q3_valid),
    .opcode_o(),
    .lhs_o(q3_lhs),
    .rhs_o(q3_rhs),
    .token_original_o(),
    .token_auxiliary_o(),
    .chain_head_o(q3_head),
    .chain_tail_o(q3_tail),
    .decode_fault_o(q3_decode_fault),
    .conversion_fault_o()
  );

  task automatic check_condition(input logic condition, input string message);
    if (!condition) $fatal(1, "%s", message);
  endtask

  initial begin
    instruction_valid = 1'b1;
    chain_length = VXM_CHAIN_LENGTH_8;
    compute_dtype = VXM_FORMAT_FP32;
    lhs_dtype = VXM_FORMAT_FP16;
    rhs_dtype = VXM_FORMAT_FP16;
    head_lhs_valid = 1'b1;
    head_lhs_data = 32'h00003e00; // FP16 1.5
    head_rhs_valid = 1'b1;
    head_rhs_data = 32'h00004000; // FP16 2.0
    previous_valid = 1'b1;
    previous_value = 32'h40a00000;
    previous_original = 32'h3f800000;
    previous_auxiliary = 32'h40000000;
    feedback_valid = 1'b1;
    feedback_value = 32'h41000000;
    feedback_original = 32'h40400000;
    feedback_auxiliary = 32'h40800000;
    immediate_valid = 1'b1;
    immediate_data = 32'h40e00000;
    accumulator_valid = 1'b1;
    accumulator_data = 32'h41100000;
    q0_instruction = 7'b0000001; // Add, Stream, Stream.
    q2_instruction = 7'b0000001;
    q3_instruction = 7'b0000001;
    #1;

    check_condition(q0_valid && q0_head && !q0_tail, "Q0 must be a chain-8 head");
    check_condition(q0_opcode == VXM_LOCAL_ADD, "Q0 opcode decode failed");
    check_condition(q0_lhs == 32'h3fc00000 && q0_rhs == 32'h40000000,
            "head FP16 operands were not widened to FP32");
    check_condition(q0_original == q0_lhs && q0_auxiliary == q0_rhs,
            "new head token did not capture original/auxiliary operands");

    // Stage 2 becomes a head at chain length 2. Q2 encodes independent
    // two-bit LHS/RHS head selectors.
    chain_length = VXM_CHAIN_LENGTH_2;
    q2_instruction = 7'b0001010; // Subtract, LHS Immediate, RHS Stream.
    #1;
    check_condition(q2_valid && q2_head && !q2_tail, "stage 2 must be a chain-2 head");
    check_condition(q2_opcode == VXM_LOCAL_SUBTRACT, "Q2 opcode decode failed");
    check_condition(q2_lhs == immediate_data && q2_rhs == 32'h40000000,
            "Q2 head source selection failed");
    check_condition(q2_original == immediate_data && q2_auxiliary == q2_rhs,
            "Q2 head token metadata failed");

    // The same physical stage is internal at chain length 4. Its LHS is
    // fixed Previous and RHS code 01 means Auxiliary.
    chain_length = VXM_CHAIN_LENGTH_4;
    q2_instruction = 7'b0100001; // Add, internal LHS=0, RHS Auxiliary.
    #1;
    check_condition(q2_valid && !q2_head && !q2_tail, "stage 2 must be chain-4 internal");
    check_condition(q2_lhs == previous_value && q2_rhs == previous_auxiliary,
            "internal Previous/Auxiliary selection failed");
    check_condition(q2_original == previous_original &&
            q2_auxiliary == previous_auxiliary,
            "internal token metadata was not forwarded");

    // Q3 is a C3 position: RHS code 11 enables its local accumulator.
    q3_instruction = 7'b0011001; // Add, RHS Accumulator.
    #1;
    check_condition(q3_valid && !q3_head && q3_tail, "stage 3 must be chain-4 tail");
    check_condition(q3_lhs == previous_value && q3_rhs == accumulator_data,
            "C3 accumulator source selection failed");

    // Feedback supplies the whole source token. A fresh RHS stream can still
    // be used by the operation, while token metadata remains from feedback.
    chain_length = VXM_CHAIN_LENGTH_8;
    q0_instruction = 7'b0010001; // Add, LHS Feedback, RHS Stream.
    #1;
    check_condition(q0_valid && q0_lhs == feedback_value &&
            q0_rhs == 32'h40000000, "feedback operand selection failed");
    check_condition(q0_original == feedback_original &&
            q0_auxiliary == feedback_auxiliary,
            "feedback token metadata was not preserved");

    // Missing data stalls the operation; it is not a configuration fault.
    head_rhs_valid = 1'b0;
    #1;
    check_condition(!q0_valid && !q0_decode_fault && !q0_conversion_fault,
            "missing data must stall without faulting");
    head_rhs_valid = 1'b1;

    lhs_dtype = VXM_FORMAT_BF16;
    head_lhs_data = 32'h00003fc0;
    q0_instruction = 7'b0000001;
    #1;
    check_condition(q0_valid && !q0_conversion_fault &&
            q0_lhs == 32'h3fc00000,
            "BF16-to-FP32 chain-head widening failed");

    compute_dtype = VXM_FORMAT_FP16;
    #1;
    check_condition(!q0_valid && q0_conversion_fault,
            "unsupported BF16-to-FP16 conversion must be reported");

    compute_dtype = VXM_FORMAT_FP32;
    lhs_dtype = VXM_FORMAT_FP16;
    head_lhs_data = 32'h00003e00;
    q0_instruction = 7'b0011001; // Illegal head LHS source 3.
    #1;
    check_condition(!q0_valid && q0_decode_fault,
            "illegal compact source encoding was not reported");

    $display("LPU_VXM_DATAPATH_MUX_TB_PASS");
    $finish;
  end
endmodule
