`timescale 1ns/1ps

// Reusable exhaustive checker for one of the 16 physical VXM ALU positions.
// Keep position-specific cases here so adding a dtype or source does not
// multiply the number of top-level test files.
module lpu_vxm_alu_stage_checker #(
  parameter integer PHYSICAL_STAGE = 0,
  parameter integer LOCAL_QUEUE = PHYSICAL_STAGE % 8,
  parameter integer SPECIAL_KIND =
    ((LOCAL_QUEUE == 1) || (LOCAL_QUEUE == 5)) ?
      lpu_pkg::VXM_SPECIAL_EXP :
    (((LOCAL_QUEUE == 3) || (LOCAL_QUEUE == 7)) ?
      lpu_pkg::VXM_SPECIAL_RECIP_RSQRT : lpu_pkg::VXM_SPECIAL_NONE)
) (
  input  logic clk_i,
  input  logic rst_ni,
  output logic done_o
);
  import lpu_pkg::*;

  logic instruction_valid;
  logic [6:0] instruction;
  logic [1:0] chain_length;
  logic [1:0] compute_dtype;
  logic [1:0] lhs_dtype;
  logic [1:0] rhs_dtype;
  logic local_reset_asserted;
  wire local_rst_n = rst_ni && !local_reset_asserted;
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

  logic lut_config_valid;
  logic [1:0] lut_config_bank;
  logic [15:0] lut_config_min;
  logic [15:0] lut_config_width;
  logic lut_write_valid;
  logic [1:0] lut_write_bank;
  logic [5:0] lut_write_address;
  logic [15:0] lut_write_k;
  logic [15:0] lut_write_b;
  wire [2:0] lut_configured;
  wire [47:0] lut_input_min;
  wire [47:0] lut_segment_width;
  wire lut_read_valid;
  wire [1:0] lut_read_bank;
  wire [5:0] lut_read_address;
  wire lut_response_valid;
  wire [15:0] lut_response_k;
  wire [15:0] lut_response_b;
  wire lut_storage_fault;

  wire input_ready;
  wire request_accepted;
  wire result_valid;
  wire [31:0] result_value;
  wire [31:0] result_original;
  wire [31:0] result_auxiliary;
  wire chain_head;
  wire chain_tail;
  wire execution_fault;
  logic unexpected_fault_seen_q;

  always_ff @(posedge clk_i or negedge local_rst_n) begin
    if (!local_rst_n)
      unexpected_fault_seen_q <= 1'b0;
    else if (execution_fault || lut_storage_fault)
      unexpected_fault_seen_q <= 1'b1;
  end

  lpu_vxm_lut_storage u_lut (
    .clk_i,
    .rst_ni(local_rst_n),
    .config_valid_i(lut_config_valid),
    .config_bank_i(lut_config_bank),
    .config_input_min_i(lut_config_min),
    .config_segment_width_i(lut_config_width),
    .write_valid_i(lut_write_valid),
    .write_bank_i(lut_write_bank),
    .write_address_i(lut_write_address),
    .write_k_i(lut_write_k),
    .write_b_i(lut_write_b),
    .read_valid_i(lut_read_valid),
    .read_bank_i(lut_read_bank),
    .read_address_i(lut_read_address),
    .read_valid_o(lut_response_valid),
    .read_k_o(lut_response_k),
    .read_b_o(lut_response_b),
    .configured_o(lut_configured),
    .input_min_o(lut_input_min),
    .segment_width_o(lut_segment_width),
    .fault_o(lut_storage_fault)
  );

  lpu_vxm_execution_stage #(
    .LOCAL_QUEUE(LOCAL_QUEUE),
    .PHYSICAL_STAGE(PHYSICAL_STAGE),
    .SPECIAL_KIND(SPECIAL_KIND)
  ) dut (
    .clk_i,
    .rst_ni(local_rst_n),
    .instruction_valid_i(instruction_valid),
    .instruction_i(instruction),
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
    .request_end_marker_i(1'b0),
    .lut_configured_i(lut_configured),
    .lut_input_min_i(lut_input_min),
    .lut_segment_width_i(lut_segment_width),
    .lut_read_valid_o(lut_read_valid),
    .lut_read_bank_o(lut_read_bank),
    .lut_read_address_o(lut_read_address),
    .lut_read_valid_i(lut_response_valid),
    .lut_read_k_i(lut_response_k),
    .lut_read_b_i(lut_response_b),
    .input_ready_o(input_ready),
    .request_accepted_o(request_accepted),
    .result_valid_o(result_valid),
    .result_value_o(result_value),
    .result_original_o(result_original),
    .result_auxiliary_o(result_auxiliary),
    .result_end_marker_o(),
    .chain_head_o(chain_head),
    .chain_tail_o(chain_tail),
    .fault_o(execution_fault)
  );

  function automatic logic [6:0] encode_instruction(
    input logic [2:0] opcode,
    input logic [1:0] lhs_code,
    input logic [1:0] rhs_code
  );
    logic [6:0] encoded;
    begin
      encoded = '0;
      encoded[2:0] = opcode;
      if (LOCAL_QUEUE == 0) begin
        encoded[4:3] = lhs_code;
        encoded[5] = rhs_code[0];
      end else if ((LOCAL_QUEUE % 2) == 0) begin
        encoded[4:3] = lhs_code;
        encoded[6:5] = rhs_code;
      end else begin
        encoded[4:3] = rhs_code;
      end
      encode_instruction = encoded;
    end
  endfunction

  function automatic logic expected_chain_head(input logic [1:0] chain);
    case (chain)
      VXM_CHAIN_LENGTH_2:
        expected_chain_head = ((PHYSICAL_STAGE % 2) == 0);
      VXM_CHAIN_LENGTH_4:
        expected_chain_head = ((PHYSICAL_STAGE % 4) == 0);
      VXM_CHAIN_LENGTH_8:
        expected_chain_head = ((PHYSICAL_STAGE % 8) == 0);
      default: expected_chain_head = 1'b0;
    endcase
  endfunction

  function automatic logic expected_chain_tail(input logic [1:0] chain);
    case (chain)
      VXM_CHAIN_LENGTH_2:
        expected_chain_tail = ((PHYSICAL_STAGE % 2) == 1);
      VXM_CHAIN_LENGTH_4:
        expected_chain_tail = ((PHYSICAL_STAGE % 4) == 3);
      VXM_CHAIN_LENGTH_8:
        expected_chain_tail = ((PHYSICAL_STAGE % 8) == 7);
      default: expected_chain_tail = 1'b0;
    endcase
  endfunction

  task automatic fail(input string message);
    $fatal(1, "VXM ALU%0d/Q%0d: %s", PHYSICAL_STAGE, LOCAL_QUEUE,
           message);
  endtask

  task automatic set_default_sources;
    begin
      head_lhs_valid = 1'b1;
      head_lhs_data = 32'h00003c00;       // 1.0
      head_rhs_valid = 1'b1;
      head_rhs_data = 32'h00004000;       // 2.0
      previous_valid = 1'b1;
      previous_value = 32'h00003c00;      // 1.0
      previous_original = 32'h00004000;   // 2.0
      previous_auxiliary = 32'h00004200;  // 3.0
      feedback_valid = 1'b1;
      feedback_value = 32'h00004200;      // 3.0
      feedback_original = 32'h00003800;   // 0.5, metadata
      feedback_auxiliary = 32'h00003c00;  // 1.0, metadata
      immediate_valid = 1'b1;
      immediate_data = 32'h00004400;      // 4.0
      accumulator_valid = 1'b1;
      accumulator_data = 32'h00004500;    // 5.0
    end
  endtask

  task automatic program_lut_bank(
    input logic [1:0] bank,
    input logic [15:0] input_min,
    input logic [15:0] segment_width,
    input logic [15:0] k,
    input logic [15:0] b
  );
    begin
      @(negedge clk_i);
      lut_config_bank = bank;
      lut_config_min = input_min;
      lut_config_width = segment_width;
      lut_write_bank = bank;
      lut_write_address = '0;
      lut_write_k = k;
      lut_write_b = b;
      lut_config_valid = 1'b1;
      lut_write_valid = 1'b1;
      @(posedge clk_i);
      #1;
      if (lut_storage_fault)
        fail("LUT configuration fault");
      @(negedge clk_i);
      lut_config_valid = 1'b0;
      lut_write_valid = 1'b0;
    end
  endtask

  task automatic write_lut_entry(
    input logic [1:0] bank,
    input logic [5:0] address,
    input logic [15:0] k,
    input logic [15:0] b
  );
    begin
      @(negedge clk_i);
      lut_write_bank = bank;
      lut_write_address = address;
      lut_write_k = k;
      lut_write_b = b;
      lut_write_valid = 1'b1;
      @(posedge clk_i);
      #1;
      if (lut_storage_fault)
        fail("LUT entry write fault");
      @(negedge clk_i);
      lut_write_valid = 1'b0;
    end
  endtask

  task automatic issue_and_expect(
    input logic [1:0] selected_chain,
    input logic [2:0] opcode,
    input logic [1:0] lhs_code,
    input logic [1:0] rhs_code,
    input logic [15:0] expected_result,
    input logic [31:0] expected_original,
    input logic [31:0] expected_auxiliary
  );
    integer timeout;
    logic observed;
    begin
      @(negedge clk_i);
      chain_length = selected_chain;
      instruction = encode_instruction(opcode, lhs_code, rhs_code);
      instruction_valid = 1'b1;
      #1;
      if (!input_ready || !request_accepted)
        fail("legal request was not accepted");
      if ((chain_head !== expected_chain_head(selected_chain)) ||
          (chain_tail !== expected_chain_tail(selected_chain)))
        fail("Head/Internal/Tail role decode mismatch");

      @(posedge clk_i);
      #1;
      observed = result_valid;
      @(negedge clk_i);
      instruction_valid = 1'b0;

      timeout = 0;
      while (!observed && (timeout < 12)) begin
        @(posedge clk_i);
        #1;
        observed = result_valid;
        timeout = timeout + 1;
      end
      if (!observed)
        fail("operation timed out");
      if (result_value !== {16'b0, expected_result})
        fail("operation result mismatch");
      if ((result_original !== expected_original) ||
          (result_auxiliary !== expected_auxiliary))
        fail("result token metadata mismatch");
      if (execution_fault || lut_storage_fault || unexpected_fault_seen_q)
        fail("unexpected fault during legal request");
    end
  endtask

  task automatic test_basic_operations;
    begin
      set_default_sources();
      if ((LOCAL_QUEUE % 2) == 0) begin
        // Every even stage is a chain head at chain length 2.
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_BYPASS,
                         2'd0, 2'd0, 16'h3c00,
                         32'h00003c00, 32'h00004000);
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                         2'd0, 2'd0, 16'h4200,
                         32'h00003c00, 32'h00004000);
        head_lhs_data = 32'h00004200;
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_SUBTRACT,
                         2'd0, 2'd0, 16'h3c00,
                         32'h00004200, 32'h00004000);
        head_lhs_data = 32'h00003e00;
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_MULTIPLY,
                         2'd0, 2'd0, 16'h4200,
                         32'h00003e00, 32'h00004000);
        head_lhs_data = 32'h00003c00;
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_NEGATE,
                         2'd0, 2'd0, 16'hbc00,
                         32'h00003c00, 32'h00004000);
        head_lhs_data = 32'h0000c000;
        head_rhs_data = 32'h00003e00;
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_MAX,
                         2'd0, 2'd0, 16'h3e00,
                         32'h0000c000, 32'h00003e00);
      end else begin
        // Odd stages are internal for every supported even chain length.
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_BYPASS,
                         2'd0, 2'd0, 16'h3c00,
                         previous_original, previous_auxiliary);
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                         2'd0, 2'd0, 16'h4200,
                         previous_original, previous_auxiliary);
        previous_value = 32'h00004200;
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_SUBTRACT,
                         2'd0, 2'd0, 16'h3c00,
                         previous_original, previous_auxiliary);
        previous_value = 32'h00003e00;
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_MULTIPLY,
                         2'd0, 2'd0, 16'h4200,
                         previous_original, previous_auxiliary);
        previous_value = 32'h00003c00;
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_NEGATE,
                         2'd0, 2'd0, 16'hbc00,
                         previous_original, previous_auxiliary);
        previous_value = 32'h0000c000;
        previous_original = 32'h00003e00;
        issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_MAX,
                         2'd0, 2'd0, 16'h3e00,
                         previous_original, previous_auxiliary);
      end
    end
  endtask

  task automatic issue_formatted_basic_and_expect(
    input logic [1:0] format,
    input logic [2:0] opcode,
    input logic [31:0] lhs_value,
    input logic [31:0] rhs_value,
    input logic [31:0] expected
  );
    logic observed;
    integer timeout;
    logic [31:0] expected_original;
    begin
      set_default_sources();
      compute_dtype = format;
      lhs_dtype = format;
      rhs_dtype = format;
      head_lhs_data = lhs_value;
      head_rhs_data = rhs_value;
      previous_value = lhs_value;
      previous_original = rhs_value;
      previous_auxiliary = rhs_value;
      expected_original = ((PHYSICAL_STAGE % 2) == 0) ?
        lhs_value : rhs_value;

      @(negedge clk_i);
      chain_length = VXM_CHAIN_LENGTH_2;
      instruction = encode_instruction(opcode, 2'd0, 2'd0);
      instruction_valid = 1'b1;
      #1;
      if (!input_ready || !request_accepted || execution_fault)
        fail("formatted Basic request was not accepted through module ports");
      @(posedge clk_i);
      #1;
      observed = result_valid;
      @(negedge clk_i);
      instruction_valid = 1'b0;
      timeout = 0;
      while (!observed && (timeout < (SPECIAL_KIND==VXM_SPECIAL_RECIP_RSQRT?8:4))) begin
        @(posedge clk_i);
        #1;
        observed = result_valid;
        timeout = timeout + 1;
      end
      if (!observed || result_value !== expected)
        fail("formatted Basic result mismatch");
      if(timeout!=(SPECIAL_KIND==VXM_SPECIAL_RECIP_RSQRT?7:(opcode==VXM_LOCAL_MULTIPLY?1:0)))
        fail("formatted Basic latency mismatch");
      if (result_original !== expected_original ||
          result_auxiliary !== rhs_value)
        fail("formatted Basic token metadata mismatch");
      compute_dtype = VXM_FORMAT_FP16;
      lhs_dtype = VXM_FORMAT_FP16;
      rhs_dtype = VXM_FORMAT_FP16;
    end
  endtask

  task automatic test_fp32_basic_path;
    begin
      issue_formatted_basic_and_expect(VXM_FORMAT_FP32, VXM_LOCAL_BYPASS,
        32'h3fc00000, 32'h40000000, 32'h3fc00000);
      issue_formatted_basic_and_expect(VXM_FORMAT_FP32, VXM_LOCAL_ADD,
        32'h3fc00000, 32'h40000000, 32'h40600000);
      issue_formatted_basic_and_expect(VXM_FORMAT_FP32, VXM_LOCAL_SUBTRACT,
        32'h40400000, 32'h40000000, 32'h3f800000);
      issue_formatted_basic_and_expect(VXM_FORMAT_FP32, VXM_LOCAL_MULTIPLY,
        32'h3fc00000, 32'h40000000, 32'h40400000);
      issue_formatted_basic_and_expect(VXM_FORMAT_FP32, VXM_LOCAL_NEGATE,
        32'h3f800000, 32'h40000000, 32'hbf800000);
      issue_formatted_basic_and_expect(VXM_FORMAT_FP32, VXM_LOCAL_MAX,
        32'hc0000000, 32'h3fc00000, 32'h3fc00000);
    end
  endtask

  task automatic test_bf16_basic_path;
    begin
      issue_formatted_basic_and_expect(VXM_FORMAT_BF16, VXM_LOCAL_BYPASS,
        32'h00003fc0, 32'h00004000, 32'h00003fc0);
      issue_formatted_basic_and_expect(VXM_FORMAT_BF16, VXM_LOCAL_ADD,
        32'h00003fc0, 32'h00004000, 32'h00004060);
      issue_formatted_basic_and_expect(VXM_FORMAT_BF16, VXM_LOCAL_SUBTRACT,
        32'h00004040, 32'h00004000, 32'h00003f80);
      issue_formatted_basic_and_expect(VXM_FORMAT_BF16, VXM_LOCAL_MULTIPLY,
        32'h00003fc0, 32'h00004000, 32'h00004040);
      issue_formatted_basic_and_expect(VXM_FORMAT_BF16, VXM_LOCAL_NEGATE,
        32'h00003f80, 32'h00004000, 32'h0000bf80);
      issue_formatted_basic_and_expect(VXM_FORMAT_BF16, VXM_LOCAL_MAX,
        32'h0000c000, 32'h00003fc0, 32'h00003fc0);
    end
  endtask

  task automatic issue_fp32_special_and_expect(
    input logic [2:0] opcode,
    input logic [31:0] operand,
    input logic [31:0] expected
  );
    logic observed;
    integer timeout;
    begin
      set_default_sources();
      compute_dtype = VXM_FORMAT_FP32;
      lhs_dtype = VXM_FORMAT_FP32;
      rhs_dtype = VXM_FORMAT_FP32;
      previous_value = operand;
      previous_original = 32'h3fc00000;
      previous_auxiliary = 32'h40000000;

      @(negedge clk_i);
      chain_length = VXM_CHAIN_LENGTH_8;
      instruction = encode_instruction(opcode, 2'd0, 2'd0);
      instruction_valid = 1'b1;
      #1;
      if (!input_ready || !request_accepted || execution_fault)
        fail("FP32 Special request was not accepted through module ports");
      @(posedge clk_i);
      #1;
      observed = result_valid;
      @(negedge clk_i);
      instruction_valid = 1'b0;
      timeout = 0;
      while (!observed && (timeout < 12)) begin
        @(posedge clk_i);
        #1;
        observed = result_valid;
        timeout = timeout + 1;
      end
      if (!observed || result_value !== expected)
        fail("FP32 Special result mismatch");
      if (result_original !== 32'h3fc00000 ||
          result_auxiliary !== 32'h40000000)
        fail("FP32 Special token metadata mismatch");
      if (execution_fault || lut_storage_fault || unexpected_fault_seen_q)
        fail("unexpected fault during FP32 Special request");
      compute_dtype = VXM_FORMAT_FP16;
      lhs_dtype = VXM_FORMAT_FP16;
      rhs_dtype = VXM_FORMAT_FP16;
    end
  endtask

  task automatic test_fp32_special_operations;
    begin
      if (SPECIAL_KIND == VXM_SPECIAL_EXP) begin
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h00000000, 32'h3f800000);
        // With B=1 and delta=0.125, exercise all three Horner terms.
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h3e000000, 32'h3f910aab);
      end else if (SPECIAL_KIND == VXM_SPECIAL_RECIP_RSQRT) begin
        // Exact power-of-two input selects the unity exponent candidate.
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h3f800000, 32'h3f800000);
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h40000000, 32'h3f000000);
        // Bank 1 entry 32 starts from UQ1.15 0.75; the expected value includes
        // the optional FP32 Newton correction y*(2-m*y).
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h3fc00000, 32'h3f280000);
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h3fc10000, 32'h3f275ff8);
        // The same row with e=1 selects the fractional exponent candidate.
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h40400000, 32'h3ea80000);
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h00800000, 32'h7e800000);
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL0, 32'h7f000000, 32'h00000000);
        issue_fp32_special_and_expect(
          VXM_LOCAL_SPECIAL1, 32'h40800000, 32'h3f000000);
      end
    end
  endtask

  task automatic issue_bf16_special_and_expect(
    input logic [2:0] opcode,
    input logic [15:0] operand,
    input logic [15:0] expected
  );
    logic observed;
    integer timeout;
    begin
      set_default_sources();
      compute_dtype = VXM_FORMAT_BF16;
      lhs_dtype = VXM_FORMAT_BF16;
      rhs_dtype = VXM_FORMAT_BF16;
      previous_value = {16'b0, operand};
      previous_original = 32'h00003fc0;
      previous_auxiliary = 32'h00004000;

      @(negedge clk_i);
      chain_length = VXM_CHAIN_LENGTH_8;
      instruction = encode_instruction(opcode, 2'd0, 2'd0);
      instruction_valid = 1'b1;
      #1;
      if (!input_ready || !request_accepted || execution_fault)
        fail("BF16 Special request was not accepted through module ports");
      @(posedge clk_i);
      #1;
      observed = result_valid;
      @(negedge clk_i);
      instruction_valid = 1'b0;
      timeout = 0;
      while (!observed && (timeout < 12)) begin
        @(posedge clk_i);
        #1;
        observed = result_valid;
        timeout = timeout + 1;
      end
      if (!observed || result_value !== {16'b0, expected})
        fail("BF16 Special result mismatch");
      if (result_original !== 32'h00003fc0 ||
          result_auxiliary !== 32'h00004000)
        fail("BF16 Special token metadata mismatch");
      if (execution_fault || lut_storage_fault || unexpected_fault_seen_q)
        fail("unexpected fault during BF16 Special request");
      compute_dtype = VXM_FORMAT_FP16;
      lhs_dtype = VXM_FORMAT_FP16;
      rhs_dtype = VXM_FORMAT_FP16;
    end
  endtask

  task automatic test_bf16_special_operations;
    begin
      if (SPECIAL_KIND == VXM_SPECIAL_EXP) begin
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h0000, 16'h3f80);
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h3e00, 16'h3f90);
      end else if (SPECIAL_KIND == VXM_SPECIAL_RECIP_RSQRT) begin
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h3f80, 16'h3f80);
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h4000, 16'h3f00);
        // The UQ1.15 coefficient 0.75 is widened before interpolation and the
        // result is rounded only when it leaves this BF16 ALU.
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h3fc0, 16'h3f40);
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h3fc1, 16'h3f3e);
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h4040, 16'h3ec0);
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h0080, 16'h7e80);
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL0, 16'h7f00, 16'h0000);
        issue_bf16_special_and_expect(
          VXM_LOCAL_SPECIAL1, 16'h4080, 16'h3f00);
      end
    end
  endtask

  task automatic test_head_sources;
    begin
      // All even physical positions become heads with chain length 2.
      set_default_sources();
      issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                       2'd0, 2'd0, 16'h4200,
                       32'h00003c00, 32'h00004000); // Stream/Stream
      issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                       2'd0, 2'd1, 16'h4500,
                       32'h00003c00, 32'h00004400); // Stream/Immediate
      issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                       2'd1, 2'd0, 16'h4600,
                       32'h00004400, 32'h00004000); // Immediate/Stream
      issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                       2'd1, 2'd1, 16'h4800,
                       32'h00004400, 32'h00004400); // Immediate/Immediate
      issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                       2'd2, 2'd0, 16'h4500,
                       feedback_original, feedback_auxiliary); // Feedback/Stream
      issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                       2'd2, 2'd1, 16'h4700,
                       feedback_original, feedback_auxiliary); // Feedback/Immediate
    end
  endtask

  task automatic test_internal_sources;
    begin
      set_default_sources();
      issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_ADD,
                       2'd0, 2'd0, 16'h4200,
                       previous_original, previous_auxiliary); // Original
      issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_ADD,
                       2'd0, 2'd1, 16'h4400,
                       previous_original, previous_auxiliary); // Auxiliary
      issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_ADD,
                       2'd0, 2'd2, 16'h4500,
                       previous_original, previous_auxiliary); // Immediate
      if ((LOCAL_QUEUE % 2) == 1)
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_ADD,
                         2'd0, 2'd3, 16'h4600,
                         previous_original, previous_auxiliary); // Accumulator
    end
  endtask

  task automatic test_chain_length4;
    begin
      set_default_sources();
      if ((PHYSICAL_STAGE % 4) == 0)
        issue_and_expect(VXM_CHAIN_LENGTH_4, VXM_LOCAL_ADD,
                         2'd0, 2'd0, 16'h4200,
                         head_lhs_data, head_rhs_data);
      else
        issue_and_expect(VXM_CHAIN_LENGTH_4, VXM_LOCAL_ADD,
                         2'd0, 2'd0, 16'h4200,
                         previous_original, previous_auxiliary);
    end
  endtask

  task automatic set_uniform_operation_operands(
    input logic [2:0] opcode,
    output logic [15:0] expected
  );
    logic [15:0] a;
    logic [15:0] b;
    begin
      a = 16'h3c00;
      b = 16'h3c00;
      expected = 16'h0000;
      case (opcode)
        VXM_LOCAL_BYPASS:   expected = 16'h3c00; // 1
        VXM_LOCAL_ADD:      expected = 16'h4000; // 1 + 1 = 2
        VXM_LOCAL_SUBTRACT: expected = 16'h0000; // 1 - 1 = 0
        VXM_LOCAL_MULTIPLY: begin
          a = 16'h4000;
          b = 16'h4000;
          expected = 16'h4400;                     // 2 * 2 = 4
        end
        VXM_LOCAL_NEGATE:   expected = 16'hbc00; // -1
        VXM_LOCAL_MAX:      expected = 16'h3c00; // max(1,1)
        VXM_LOCAL_SPECIAL0: begin
          if (SPECIAL_KIND == VXM_SPECIAL_EXP) begin
            a = 16'h0000;
            expected = 16'h3c00;                   // exp(0) = 1
          end else begin
            a = 16'h4000;
            expected = 16'h3800;                   // reciprocal(2)
          end
        end
        VXM_LOCAL_SPECIAL1: begin
          a = 16'h4400;
          expected = 16'h3800;                     // rsqrt(4)
        end
        default: expected = 16'h7e00;
      endcase

      set_default_sources();
      head_lhs_data = {16'b0, a};
      head_rhs_data = {16'b0, b};
      previous_value = {16'b0, a};
      previous_original = {16'b0, b};
      previous_auxiliary = {16'b0, b};
      feedback_value = {16'b0, a};
      immediate_data = {16'b0, b};
      accumulator_data = {16'b0, b};
    end
  endtask

  task automatic test_basic_operation_source_crosses;
    integer operation;
    integer lhs_code;
    integer rhs_code;
    logic [15:0] expected;
    begin
      // Cross every Basic operation with every legal source encoding. The
      // dedicated source tests above use distinct values; this cross uses
      // controlled equal operands so expected FP16 values remain independent
      // constants instead of reusing the DUT arithmetic implementation.
      for (operation = VXM_LOCAL_BYPASS;
           operation <= VXM_LOCAL_MAX; operation = operation + 1) begin
        if ((LOCAL_QUEUE % 2) == 0) begin
          for (lhs_code = 0; lhs_code <= 2; lhs_code = lhs_code + 1) begin
            for (rhs_code = 0; rhs_code <= 1; rhs_code = rhs_code + 1) begin
              set_uniform_operation_operands(operation[2:0], expected);
              issue_and_expect(VXM_CHAIN_LENGTH_2, operation[2:0],
                               lhs_code[1:0], rhs_code[1:0], expected,
                               lhs_code == 2 ? feedback_original :
                                 {16'b0, (operation == VXM_LOCAL_MULTIPLY) ?
                                   16'h4000 : 16'h3c00},
                               lhs_code == 2 ? feedback_auxiliary :
                                 {16'b0, (operation == VXM_LOCAL_MULTIPLY) ?
                                   16'h4000 : 16'h3c00});
            end
          end
          if (LOCAL_QUEUE != 0) begin
            for (rhs_code = 0; rhs_code <= 2; rhs_code = rhs_code + 1) begin
              set_uniform_operation_operands(operation[2:0], expected);
              issue_and_expect(VXM_CHAIN_LENGTH_8, operation[2:0],
                               2'd0, rhs_code[1:0], expected,
                               previous_original, previous_auxiliary);
            end
          end
        end else begin
          for (rhs_code = 0; rhs_code <= 3; rhs_code = rhs_code + 1) begin
            set_uniform_operation_operands(operation[2:0], expected);
            issue_and_expect(VXM_CHAIN_LENGTH_8, operation[2:0],
                             2'd0, rhs_code[1:0], expected,
                             previous_original, previous_auxiliary);
          end
        end
      end
    end
  endtask

  task automatic test_special_operations;
    integer rhs_code;
    logic [15:0] expected;
    begin
      if (SPECIAL_KIND == VXM_SPECIAL_EXP) begin
        for (rhs_code = 0; rhs_code <= 3; rhs_code = rhs_code + 1) begin
          set_uniform_operation_operands(VXM_LOCAL_SPECIAL0, expected);
          issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                           2'd0, rhs_code[1:0], expected,
                           previous_original, previous_auxiliary);
        end
      end else if (SPECIAL_KIND == VXM_SPECIAL_RECIP_RSQRT) begin
        for (rhs_code = 0; rhs_code <= 3; rhs_code = rhs_code + 1) begin
          set_uniform_operation_operands(VXM_LOCAL_SPECIAL0, expected);
          issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                           2'd0, rhs_code[1:0], expected,
                           previous_original, previous_auxiliary);
          set_uniform_operation_operands(VXM_LOCAL_SPECIAL1, expected);
          issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL1,
                           2'd0, rhs_code[1:0], expected,
                           previous_original, previous_auxiliary);
        end
      end
    end
  endtask

  task automatic test_special_edge_cases;
    begin
      if (SPECIAL_KIND == VXM_SPECIAL_EXP) begin
        set_default_sources();
        previous_value = 32'h00007c00;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h7c00,
                         previous_original, previous_auxiliary); // exp(+Inf)
        previous_value = 32'h0000fc00;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h0000,
                         previous_original, previous_auxiliary); // exp(-Inf)
        previous_value = 32'h00007d01;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h7e00,
                         previous_original, previous_auxiliary); // exp(NaN)
        previous_value = 32'h00000001;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h3c00,
                         previous_original, previous_auxiliary); // FTZ -> exp(0)
        previous_value = 32'h00003000;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h3c80,
                         previous_original, previous_auxiliary); // linear B+B*delta
      end else if (SPECIAL_KIND == VXM_SPECIAL_RECIP_RSQRT) begin
        set_default_sources();
        previous_value = 32'h00000000;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h7c00,
                         previous_original, previous_auxiliary); // recip(+0)
        previous_value = 32'h00008000;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'hfc00,
                         previous_original, previous_auxiliary); // recip(-0)
        previous_value = 32'h00007c00;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h0000,
                         previous_original, previous_auxiliary); // recip(+Inf)
        previous_value = 32'h0000fc00;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h8000,
                         previous_original, previous_auxiliary); // recip(-Inf)
        previous_value = 32'h0000c000;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'hb800,
                         previous_original, previous_auxiliary); // recip(-2)
        previous_value = 32'h00003c00;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h3c00,
                         previous_original, previous_auxiliary); // unity candidate
        previous_value = 32'h00003e00;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h3a00,
                         previous_original, previous_auxiliary); // LUT address 32
        previous_value = 32'h00004200;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h3600,
                         previous_original, previous_auxiliary); // fraction candidate
        previous_value = 32'h00000400;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h7400,
                         previous_original, previous_auxiliary); // upper exponent edge
        previous_value = 32'h00007800;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h0000,
                         previous_original, previous_auxiliary); // FTZ lower edge
        previous_value = 32'h00003e08;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h39f0,
                         previous_original, previous_auxiliary); // b-a*dx
        previous_value = 32'h00007d01;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h7e00,
                         previous_original, previous_auxiliary); // recip(NaN)
        previous_value = 32'h00000001;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL0,
                         2'd0, 2'd0, 16'h7c00,
                         previous_original, previous_auxiliary); // FTZ -> recip(0)

        previous_value = 32'h00000000;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL1,
                         2'd0, 2'd0, 16'h7c00,
                         previous_original, previous_auxiliary); // rsqrt(0)
        previous_value = 32'h00007c00;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL1,
                         2'd0, 2'd0, 16'h0000,
                         previous_original, previous_auxiliary); // rsqrt(+Inf)
        previous_value = 32'h0000c000;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL1,
                         2'd0, 2'd0, 16'h7e00,
                         previous_original, previous_auxiliary); // rsqrt(negative)
        previous_value = 32'h00007d01;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL1,
                         2'd0, 2'd0, 16'h7e00,
                         previous_original, previous_auxiliary); // rsqrt(NaN)
        previous_value = 32'h00000001;
        issue_and_expect(VXM_CHAIN_LENGTH_8, VXM_LOCAL_SPECIAL1,
                         2'd0, 2'd0, 16'h7c00,
                         previous_original, previous_auxiliary); // FTZ -> rsqrt(0)
      end
    end
  endtask

  task automatic test_reset_in_flight;
    begin
      set_default_sources();
      @(negedge clk_i);
      chain_length = VXM_CHAIN_LENGTH_2;
      instruction = encode_instruction(VXM_LOCAL_MULTIPLY, 2'd0, 2'd0);
      instruction_valid = 1'b1;
      #1;
      if (!request_accepted)
        fail("pre-reset Multiply was not accepted");
      @(posedge clk_i);
      #1;
      if (result_valid)
        fail("Multiply completed before reset could be applied");
      @(negedge clk_i);
      instruction_valid = 1'b0;
      local_reset_asserted = 1'b1;
      #1;
      if (result_valid)
        fail("in-flight result survived asynchronous reset");
      @(negedge clk_i);
      local_reset_asserted = 1'b0;

      set_default_sources();
      issue_and_expect(VXM_CHAIN_LENGTH_2, VXM_LOCAL_ADD,
                       2'd0, 2'd0, 16'h4200,
                       (LOCAL_QUEUE % 2) == 0 ? head_lhs_data :
                         previous_original,
                       (LOCAL_QUEUE % 2) == 0 ? head_rhs_data :
                         previous_auxiliary);
    end
  endtask

  task automatic expect_rejected_fault(
    input logic [6:0] rejected_instruction,
    input logic [1:0] rejected_chain,
    input logic [1:0] rejected_compute_dtype,
    input logic [1:0] rejected_lhs_dtype,
    input logic [1:0] rejected_rhs_dtype
  );
    begin
      @(negedge clk_i);
      instruction = rejected_instruction;
      chain_length = rejected_chain;
      compute_dtype = rejected_compute_dtype;
      lhs_dtype = rejected_lhs_dtype;
      rhs_dtype = rejected_rhs_dtype;
      instruction_valid = 1'b1;
      #1;
      if (!execution_fault || request_accepted)
        fail("invalid configuration was not rejected");
      @(posedge clk_i);
      #1;
      if (result_valid)
        fail("rejected configuration produced a result");
      @(negedge clk_i);
      instruction_valid = 1'b0;
      compute_dtype = VXM_FORMAT_FP16;
      lhs_dtype = VXM_FORMAT_FP16;
      rhs_dtype = VXM_FORMAT_FP16;
    end
  endtask

  task automatic test_stall_and_illegal;
    logic [6:0] rejected_instruction;
    begin
      set_default_sources();
      @(negedge clk_i);
      chain_length = VXM_CHAIN_LENGTH_2;
      instruction = encode_instruction(VXM_LOCAL_ADD, 2'd0, 2'd0);
      if ((LOCAL_QUEUE % 2) == 0)
        head_lhs_valid = 1'b0;
      else
        previous_valid = 1'b0;
      instruction_valid = 1'b1;
      #1;
      if (request_accepted || execution_fault)
        fail("missing selected input did not produce a clean stall");
      @(negedge clk_i);
      instruction_valid = 1'b0;

      set_default_sources();
      if (unexpected_fault_seen_q)
        fail("a legal-operation fault pulse was observed before fault testing");

      if ((LOCAL_QUEUE % 2) == 0)
        rejected_instruction =
          encode_instruction(VXM_LOCAL_ADD, 2'd3, 2'd0);
      else begin
        rejected_instruction =
          encode_instruction(VXM_LOCAL_ADD, 2'd0, 2'd0);
        rejected_instruction[5] = 1'b1;
      end
      expect_rejected_fault(rejected_instruction, VXM_CHAIN_LENGTH_2,
                            VXM_FORMAT_FP16, VXM_FORMAT_FP16,
                            VXM_FORMAT_FP16);

      // Reserved chain-length encoding.
      expect_rejected_fault(
        encode_instruction(VXM_LOCAL_ADD, 2'd0, 2'd0),
        VXM_CHAIN_LENGTH_INVALID, VXM_FORMAT_FP16,
        VXM_FORMAT_FP16, VXM_FORMAT_FP16);

      // The fourth data-format encoding remains reserved.
      expect_rejected_fault(
        encode_instruction(VXM_LOCAL_ADD, 2'd0, 2'd0),
        VXM_CHAIN_LENGTH_2, VXM_FORMAT_RESERVED,
        VXM_FORMAT_FP16, VXM_FORMAT_FP16);

      // BF16-to-FP16 is intentionally not an implicit conversion. Such a
      // chain-head request must fault in the converter before reaching ALU.
      if ((LOCAL_QUEUE % 2) == 0)
        expect_rejected_fault(
          encode_instruction(VXM_LOCAL_ADD, 2'd0, 2'd0),
          VXM_CHAIN_LENGTH_2, VXM_FORMAT_FP16,
          VXM_FORMAT_BF16, VXM_FORMAT_FP16);
      if ((LOCAL_QUEUE % 2) == 0)
        expect_rejected_fault(
          encode_instruction(VXM_LOCAL_ADD, 2'd0, 2'd0),
          VXM_CHAIN_LENGTH_2, VXM_FORMAT_FP16,
          VXM_FORMAT_FP16, VXM_FORMAT_BF16);

      // C0/C2 internal positions do not provide an Accumulator RHS source.
      if (((LOCAL_QUEUE % 2) == 0) && (LOCAL_QUEUE != 0))
        expect_rejected_fault(
          encode_instruction(VXM_LOCAL_ADD, 2'd0, 2'd3),
          VXM_CHAIN_LENGTH_8, VXM_FORMAT_FP16,
          VXM_FORMAT_FP16, VXM_FORMAT_FP16);

      // Position-specific illegal operation encodings.
      if ((LOCAL_QUEUE % 2) == 0)
        begin
          expect_rejected_fault(
            encode_instruction(VXM_LOCAL_SPECIAL0, 2'd0, 2'd0),
            VXM_CHAIN_LENGTH_2, VXM_FORMAT_FP16,
            VXM_FORMAT_FP16, VXM_FORMAT_FP16);
          expect_rejected_fault(
            encode_instruction(VXM_LOCAL_SPECIAL1, 2'd0, 2'd0),
            VXM_CHAIN_LENGTH_2, VXM_FORMAT_FP16,
            VXM_FORMAT_FP16, VXM_FORMAT_FP16);
        end
      else if ((LOCAL_QUEUE == 1) || (LOCAL_QUEUE == 5))
        expect_rejected_fault(
          encode_instruction(VXM_LOCAL_SPECIAL1, 2'd0, 2'd0),
          VXM_CHAIN_LENGTH_2, VXM_FORMAT_FP16,
          VXM_FORMAT_FP16, VXM_FORMAT_FP16);
    end
  endtask

  initial begin
    done_o = 1'b0;
    instruction_valid = 1'b0;
    instruction = '0;
    chain_length = VXM_CHAIN_LENGTH_2;
    compute_dtype = VXM_FORMAT_FP16;
    lhs_dtype = VXM_FORMAT_FP16;
    rhs_dtype = VXM_FORMAT_FP16;
    local_reset_asserted = 1'b0;
    lut_config_valid = 1'b0;
    lut_config_bank = '0;
    lut_config_min = '0;
    lut_config_width = 16'h3c00;
    lut_write_valid = 1'b0;
    lut_write_bank = '0;
    lut_write_address = '0;
    lut_write_k = '0;
    lut_write_b = '0;
    set_default_sources();

    wait (rst_ni);
    // Program the exact piecewise-linear points used by special tests.
    program_lut_bank(2'd0, 16'h0000, 16'h3c00, 16'h1000, 16'h0000);
    program_lut_bank(2'd1, 16'h3c00, 16'h3800, 16'h0000, 16'h8000);
    program_lut_bank(2'd2, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);
    write_lut_entry(2'd1, 6'd32, 16'h8000, 16'h6000);

    test_basic_operations();
    test_fp32_basic_path();
    test_bf16_basic_path();
    if ((LOCAL_QUEUE % 2) == 0)
      test_head_sources();
    if ((LOCAL_QUEUE % 2) == 1 || LOCAL_QUEUE != 0)
      test_internal_sources();
    test_chain_length4();
    test_basic_operation_source_crosses();
    test_special_operations();
    test_fp32_special_operations();
    test_bf16_special_operations();
    test_special_edge_cases();
    test_reset_in_flight();
    test_stall_and_illegal();

    done_o = 1'b1;
    $display("VXM_ALU_STAGE_PASS physical=%0d queue=%0d",
             PHYSICAL_STAGE, LOCAL_QUEUE);
  end
endmodule
