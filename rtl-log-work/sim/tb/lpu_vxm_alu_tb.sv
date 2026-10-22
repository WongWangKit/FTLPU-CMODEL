`timescale 1ns/1ps

module lpu_vxm_alu_tb;
  import lpu_pkg::*;

  logic clk;
  logic rst_n;

  logic basic_input_valid;
  logic [1:0] basic_format;
  logic [2:0] basic_opcode;
  logic [31:0] basic_lhs;
  logic [31:0] basic_rhs;
  wire basic_ready;
  wire basic_output_valid;
  wire [31:0] basic_result;
  wire basic_illegal;
  wire basic_unsupported;
  wire basic_collision;

  logic exp_lut_config_valid;
  logic [1:0] exp_lut_config_bank;
  logic [15:0] exp_lut_config_min;
  logic [15:0] exp_lut_config_width;
  logic exp_lut_write_valid;
  logic [1:0] exp_lut_write_bank;
  logic [5:0] exp_lut_write_address;
  logic [15:0] exp_lut_write_k;
  logic [15:0] exp_lut_write_b;
  wire [2:0] exp_lut_configured;
  wire [47:0] exp_lut_input_min;
  wire [47:0] exp_lut_segment_width;
  wire exp_lut_request_valid;
  wire [1:0] exp_lut_request_bank;
  wire [5:0] exp_lut_request_address;
  wire exp_lut_response_valid;
  wire [15:0] exp_lut_response_k;
  wire [15:0] exp_lut_response_b;
  wire exp_lut_storage_fault;

  logic exp_input_valid;
  logic [1:0] exp_format;
  logic [2:0] exp_opcode;
  logic [31:0] exp_lhs;
  wire exp_output_valid;
  wire [31:0] exp_result;
  wire exp_illegal;
  wire exp_unsupported;
  wire exp_lut_fault;

  logic recip_lut_config_valid;
  logic [1:0] recip_lut_config_bank;
  logic [15:0] recip_lut_config_min;
  logic [15:0] recip_lut_config_width;
  logic recip_lut_write_valid;
  logic [1:0] recip_lut_write_bank;
  logic [5:0] recip_lut_write_address;
  logic [15:0] recip_lut_write_k;
  logic [15:0] recip_lut_write_b;
  wire [2:0] recip_lut_configured;
  wire [47:0] recip_lut_input_min;
  wire [47:0] recip_lut_segment_width;
  wire recip_lut_request_valid;
  wire [1:0] recip_lut_request_bank;
  wire [5:0] recip_lut_request_address;
  wire recip_lut_response_valid;
  wire [15:0] recip_lut_response_k;
  wire [15:0] recip_lut_response_b;
  wire recip_lut_storage_fault;

  logic recip_input_valid;
  logic [1:0] recip_format;
  logic [2:0] recip_opcode;
  logic [31:0] recip_lhs;
  wire recip_output_valid;
  wire [31:0] recip_result;
  wire recip_illegal;
  wire recip_unsupported;
  wire recip_lut_fault;
  logic special_fault_seen;

  always #5 clk = ~clk;

  lpu_vxm_alu #(
    .SPECIAL_KIND(VXM_SPECIAL_NONE)
  ) u_basic (
    .clk_i(clk),
    .rst_ni(rst_n),
    .input_valid_i(basic_input_valid),
    .data_format_i(basic_format),
    .opcode_i(basic_opcode),
    .lhs_i(basic_lhs),
    .rhs_i(basic_rhs),
    .lut_configured_i('0),
    .lut_input_min_i('0),
    .lut_segment_width_i('0),
    .lut_read_valid_o(),
    .lut_read_bank_o(),
    .lut_read_address_o(),
    .lut_read_valid_i(1'b0),
    .lut_read_k_i('0),
    .lut_read_b_i('0),
    .input_ready_o(basic_ready),
    .output_valid_o(basic_output_valid),
    .result_o(basic_result),
    .illegal_opcode_o(basic_illegal),
    .unsupported_format_o(basic_unsupported),
    .result_collision_o(basic_collision),
    .lut_fault_o()
  );

  lpu_vxm_lut_storage u_exp_lut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .config_valid_i(exp_lut_config_valid),
    .config_bank_i(exp_lut_config_bank),
    .config_input_min_i(exp_lut_config_min),
    .config_segment_width_i(exp_lut_config_width),
    .write_valid_i(exp_lut_write_valid),
    .write_bank_i(exp_lut_write_bank),
    .write_address_i(exp_lut_write_address),
    .write_k_i(exp_lut_write_k),
    .write_b_i(exp_lut_write_b),
    .read_valid_i(exp_lut_request_valid),
    .read_bank_i(exp_lut_request_bank),
    .read_address_i(exp_lut_request_address),
    .read_valid_o(exp_lut_response_valid),
    .read_k_o(exp_lut_response_k),
    .read_b_o(exp_lut_response_b),
    .configured_o(exp_lut_configured),
    .input_min_o(exp_lut_input_min),
    .segment_width_o(exp_lut_segment_width),
    .fault_o(exp_lut_storage_fault)
  );

  lpu_vxm_alu #(
    .SPECIAL_KIND(VXM_SPECIAL_EXP)
  ) u_exp (
    .clk_i(clk),
    .rst_ni(rst_n),
    .input_valid_i(exp_input_valid),
    .data_format_i(exp_format),
    .opcode_i(exp_opcode),
    .lhs_i(exp_lhs),
    .rhs_i(32'b0),
    .lut_configured_i(exp_lut_configured),
    .lut_input_min_i(exp_lut_input_min),
    .lut_segment_width_i(exp_lut_segment_width),
    .lut_read_valid_o(exp_lut_request_valid),
    .lut_read_bank_o(exp_lut_request_bank),
    .lut_read_address_o(exp_lut_request_address),
    .lut_read_valid_i(exp_lut_response_valid),
    .lut_read_k_i(exp_lut_response_k),
    .lut_read_b_i(exp_lut_response_b),
    .input_ready_o(),
    .output_valid_o(exp_output_valid),
    .result_o(exp_result),
    .illegal_opcode_o(exp_illegal),
    .unsupported_format_o(exp_unsupported),
    .result_collision_o(),
    .lut_fault_o(exp_lut_fault)
  );

  lpu_vxm_lut_storage u_recip_lut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .config_valid_i(recip_lut_config_valid),
    .config_bank_i(recip_lut_config_bank),
    .config_input_min_i(recip_lut_config_min),
    .config_segment_width_i(recip_lut_config_width),
    .write_valid_i(recip_lut_write_valid),
    .write_bank_i(recip_lut_write_bank),
    .write_address_i(recip_lut_write_address),
    .write_k_i(recip_lut_write_k),
    .write_b_i(recip_lut_write_b),
    .read_valid_i(recip_lut_request_valid),
    .read_bank_i(recip_lut_request_bank),
    .read_address_i(recip_lut_request_address),
    .read_valid_o(recip_lut_response_valid),
    .read_k_o(recip_lut_response_k),
    .read_b_o(recip_lut_response_b),
    .configured_o(recip_lut_configured),
    .input_min_o(recip_lut_input_min),
    .segment_width_o(recip_lut_segment_width),
    .fault_o(recip_lut_storage_fault)
  );

  lpu_vxm_alu #(
    .SPECIAL_KIND(VXM_SPECIAL_RECIP_RSQRT)
  ) u_recip (
    .clk_i(clk),
    .rst_ni(rst_n),
    .input_valid_i(recip_input_valid),
    .data_format_i(recip_format),
    .opcode_i(recip_opcode),
    .lhs_i(recip_lhs),
    .rhs_i(32'b0),
    .lut_configured_i(recip_lut_configured),
    .lut_input_min_i(recip_lut_input_min),
    .lut_segment_width_i(recip_lut_segment_width),
    .lut_read_valid_o(recip_lut_request_valid),
    .lut_read_bank_o(recip_lut_request_bank),
    .lut_read_address_o(recip_lut_request_address),
    .lut_read_valid_i(recip_lut_response_valid),
    .lut_read_k_i(recip_lut_response_k),
    .lut_read_b_i(recip_lut_response_b),
    .input_ready_o(),
    .output_valid_o(recip_output_valid),
    .result_o(recip_result),
    .illegal_opcode_o(recip_illegal),
    .unsupported_format_o(recip_unsupported),
    .result_collision_o(),
    .lut_fault_o(recip_lut_fault)
  );

  task automatic program_lut_entry_zero(
    input logic select_recip,
    input logic [1:0] bank,
    input logic [15:0] input_min,
    input logic [15:0] segment_width,
    input logic [15:0] k,
    input logic [15:0] b
  );
    begin
      @(negedge clk);
      if (select_recip) begin
        recip_lut_config_bank = bank;
        recip_lut_config_min = input_min;
        recip_lut_config_width = segment_width;
        recip_lut_config_valid = 1'b1;
        recip_lut_write_bank = bank;
        recip_lut_write_address = '0;
        recip_lut_write_k = k;
        recip_lut_write_b = b;
        recip_lut_write_valid = 1'b1;
      end else begin
        exp_lut_config_bank = bank;
        exp_lut_config_min = input_min;
        exp_lut_config_width = segment_width;
        exp_lut_config_valid = 1'b1;
        exp_lut_write_bank = bank;
        exp_lut_write_address = '0;
        exp_lut_write_k = k;
        exp_lut_write_b = b;
        exp_lut_write_valid = 1'b1;
      end
      @(posedge clk);
      #1;
      @(negedge clk);
      exp_lut_config_valid = 1'b0;
      exp_lut_write_valid = 1'b0;
      recip_lut_config_valid = 1'b0;
      recip_lut_write_valid = 1'b0;
    end
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      special_fault_seen <= 1'b0;
    else
      special_fault_seen <= special_fault_seen | exp_lut_fault |
        recip_lut_fault | exp_lut_storage_fault | recip_lut_storage_fault;
  end

  task automatic issue_basic_one_cycle(
    input logic [2:0] operation,
    input logic [15:0] lhs,
    input logic [15:0] rhs,
    input logic [15:0] expected
  );
    begin
      @(negedge clk);
      basic_opcode = operation;
      basic_lhs = {16'b0, lhs};
      basic_rhs = {16'b0, rhs};
      basic_input_valid = 1'b1;
      @(posedge clk);
      #1;
      if (!basic_output_valid || basic_result !== {16'b0, expected})
        $fatal(1, "basic opcode %0d expected %h got valid=%b result=%h",
               operation, expected, basic_output_valid, basic_result);
      @(negedge clk);
      basic_input_valid = 1'b0;
    end
  endtask

  task automatic issue_basic_fp16_multiply(
    input logic [15:0] lhs,
    input logic [15:0] rhs,
    input logic [15:0] expected
  );
    begin
      @(negedge clk);
      basic_format = VXM_FORMAT_FP16;
      basic_opcode = VXM_LOCAL_MULTIPLY;
      basic_lhs = {16'b0, lhs};
      basic_rhs = {16'b0, rhs};
      basic_input_valid = 1'b1;
      #1;
      if (!basic_ready || basic_unsupported || basic_illegal)
        $fatal(1, "FP16 Multiply was not accepted");
      @(posedge clk);
      #1;
      if (basic_output_valid)
        $fatal(1, "FP16 Multiply returned in one cycle");
      @(negedge clk);
      basic_input_valid = 1'b0;
      @(posedge clk);
      #1;
      if (!basic_output_valid || basic_result !== {16'b0, expected})
        $fatal(1,
          "FP16 Multiply expected %h got valid=%b result=%h",
          expected, basic_output_valid, basic_result);
    end
  endtask

  task automatic check_back_to_back_fp16_multiply;
    begin
      // Request A enters the internal product register.
      @(negedge clk);
      basic_format = VXM_FORMAT_FP16;
      basic_opcode = VXM_LOCAL_MULTIPLY;
      basic_lhs = 32'h00003e00;
      basic_rhs = 32'h00004000;
      basic_input_valid = 1'b1;
      #1;
      if (!basic_ready)
        $fatal(1, "back-to-back Multiply A was not accepted");
      @(posedge clk);
      #1;
      if (basic_output_valid)
        $fatal(1, "back-to-back Multiply A returned early");

      // Request B enters stage 1 while A completes stage 2.
      @(negedge clk);
      basic_lhs = 32'h00003e00;
      basic_rhs = 32'h00003e00;
      #1;
      if (!basic_ready)
        $fatal(1, "back-to-back Multiply B was not accepted");
      @(posedge clk);
      #1;
      if (!basic_output_valid || basic_result !== 32'h00004200)
        $fatal(1, "back-to-back Multiply A result mismatch: %h",
          basic_result);

      @(negedge clk);
      basic_input_valid = 1'b0;
      @(posedge clk);
      #1;
      if (!basic_output_valid || basic_result !== 32'h00004080)
        $fatal(1, "back-to-back Multiply B result mismatch: %h",
          basic_result);
    end
  endtask

  task automatic issue_basic_fp32(
    input logic [2:0] operation,
    input logic [31:0] lhs,
    input logic [31:0] rhs,
    input logic [31:0] expected,
    input integer latency
  );
    begin
      @(negedge clk);
      basic_format = VXM_FORMAT_FP32;
      basic_opcode = operation;
      basic_lhs = lhs;
      basic_rhs = rhs;
      basic_input_valid = 1'b1;
      #1;
      if (!basic_ready || basic_unsupported || basic_illegal)
        $fatal(1, "FP32 basic opcode %0d was not accepted", operation);
      for (integer cycle = 1; cycle <= latency; cycle++) begin
        @(posedge clk);
        #1;
        if ((cycle < latency) && basic_output_valid)
          $fatal(1, "FP32 opcode %0d returned early", operation);
        if (cycle == 1) begin
          @(negedge clk);
          basic_input_valid = 1'b0;
        end
      end
      if (!basic_output_valid || basic_result !== expected)
        $fatal(1,
          "FP32 opcode %0d expected %h got valid=%b result=%h",
          operation, expected, basic_output_valid, basic_result);
      basic_format = VXM_FORMAT_FP16;
    end
  endtask

  task automatic issue_basic_bf16(
    input logic [2:0] operation,
    input logic [15:0] lhs,
    input logic [15:0] rhs,
    input logic [15:0] expected,
    input integer latency
  );
    begin
      @(negedge clk);
      basic_format = VXM_FORMAT_BF16;
      basic_opcode = operation;
      basic_lhs = {16'b0, lhs};
      basic_rhs = {16'b0, rhs};
      basic_input_valid = 1'b1;
      #1;
      if (!basic_ready || basic_unsupported || basic_illegal)
        $fatal(1, "BF16 basic opcode %0d was not accepted", operation);
      for (integer cycle = 1; cycle <= latency; cycle++) begin
        @(posedge clk);
        #1;
        if ((cycle < latency) && basic_output_valid)
          $fatal(1, "BF16 opcode %0d returned early", operation);
        if (cycle == 1) begin
          @(negedge clk);
          basic_input_valid = 1'b0;
        end
      end
      if (!basic_output_valid || basic_result !== {16'b0, expected})
        $fatal(1,
          "BF16 opcode %0d expected %h got valid=%b result=%h",
          operation, expected, basic_output_valid, basic_result);
      basic_format = VXM_FORMAT_FP16;
    end
  endtask

  // Drives nonzero garbage in the unused upper half of the public 32-bit
  // container. The selected 16-bit format must isolate it before arithmetic.
  task automatic issue_basic_16_with_upper_noise(
    input logic [1:0] format,
    input logic [2:0] operation,
    input logic [31:0] lhs,
    input logic [31:0] rhs,
    input logic [15:0] expected
  );
    begin
      @(negedge clk);
      basic_format = format;
      basic_opcode = operation;
      basic_lhs = lhs;
      basic_rhs = rhs;
      basic_input_valid = 1'b1;
      @(posedge clk);
      #1;
      if (!basic_output_valid || basic_result !== {16'b0, expected})
        $fatal(1,
          "16-bit format MUX failed: format=%0d expected=%h got=%h",
          format, expected, basic_result);
      @(negedge clk);
      basic_input_valid = 1'b0;
      basic_format = VXM_FORMAT_FP16;
    end
  endtask

  task automatic issue_special_and_expect(
    input logic select_recip,
    input logic [2:0] operation,
    input logic [15:0] operand,
    input logic [15:0] expected
  );
    begin
      @(negedge clk);
      if (select_recip) begin
        recip_opcode = operation;
        recip_lhs = {16'b0, operand};
        recip_input_valid = 1'b1;
      end else begin
        exp_opcode = operation;
        exp_lhs = {16'b0, operand};
        exp_input_valid = 1'b1;
      end
      @(posedge clk);
      #1;
      if ((select_recip && recip_output_valid) ||
          (!select_recip && exp_output_valid))
        $fatal(1, "special result returned before eight pipeline stages");
      @(negedge clk);
      recip_input_valid = 1'b0;
      exp_input_valid = 1'b0;
      repeat (6) begin
        @(posedge clk);
        #1;
        if ((select_recip && recip_output_valid) ||
            (!select_recip && exp_output_valid))
          $fatal(1, "special result returned early");
      end
      @(posedge clk);
      #1;
      if (select_recip) begin
        if (!recip_output_valid || recip_result !== {16'b0, expected})
          $fatal(1, "special opcode %0d expected %h got %h",
                 operation, expected, recip_result);
      end else begin
        if (!exp_output_valid || exp_result !== {16'b0, expected})
          $fatal(1, "exp expected %h got %h", expected, exp_result);
      end
    end
  endtask

  task automatic issue_special_fp32_and_expect(
    input logic select_recip,
    input logic [2:0] operation,
    input logic [31:0] operand,
    input logic [31:0] expected
  );
    integer timeout;
    logic observed;
    begin
      @(negedge clk);
      if (select_recip) begin
        recip_format = VXM_FORMAT_FP32;
        recip_opcode = operation;
        recip_lhs = operand;
        recip_input_valid = 1'b1;
      end else begin
        exp_format = VXM_FORMAT_FP32;
        exp_opcode = operation;
        exp_lhs = operand;
        exp_input_valid = 1'b1;
      end
      #1;
      if ((select_recip && recip_unsupported) ||
          (!select_recip && exp_unsupported))
        $fatal(1, "FP32 special request was rejected");
      @(posedge clk);
      #1;
      if ((select_recip && recip_output_valid) ||
          (!select_recip && exp_output_valid))
        $fatal(1, "FP32 special result returned before eight stages");
      @(negedge clk);
      recip_input_valid = 1'b0;
      exp_input_valid = 1'b0;
      timeout = 0;
      observed = 1'b0;
      while (!observed && (timeout < 12)) begin
        @(posedge clk);
        #1;
        observed = select_recip ? recip_output_valid : exp_output_valid;
        timeout = timeout + 1;
      end
      if (select_recip) begin
        if (!observed || recip_result !== expected)
          $fatal(1, "FP32 special opcode %0d expected %h got %h",
                 operation, expected, recip_result);
      end else if (!observed || exp_result !== expected) begin
        $fatal(1, "FP32 exp expected %h got %h", expected, exp_result);
      end
      exp_format = VXM_FORMAT_FP16;
      recip_format = VXM_FORMAT_FP16;
    end
  endtask

  task automatic issue_special_bf16_and_expect(
    input logic select_recip,
    input logic [2:0] operation,
    input logic [15:0] operand,
    input logic [15:0] expected
  );
    begin
      @(negedge clk);
      if (select_recip) begin
        recip_format = VXM_FORMAT_BF16;
        recip_opcode = operation;
        recip_lhs = {16'b0, operand};
        recip_input_valid = 1'b1;
      end else begin
        exp_format = VXM_FORMAT_BF16;
        exp_opcode = operation;
        exp_lhs = {16'b0, operand};
        exp_input_valid = 1'b1;
      end
      #1;
      if ((select_recip && (recip_unsupported || recip_illegal)) ||
          (!select_recip && (exp_unsupported || exp_illegal)))
        $fatal(1, "BF16 special request was rejected");
      @(posedge clk);
      #1;
      if ((select_recip && recip_output_valid) ||
          (!select_recip && exp_output_valid))
        $fatal(1, "BF16 special result returned before eight stages");
      @(negedge clk);
      recip_input_valid = 1'b0;
      exp_input_valid = 1'b0;
      repeat (6) begin
        @(posedge clk);
        #1;
        if ((select_recip && recip_output_valid) ||
            (!select_recip && exp_output_valid))
          $fatal(1, "BF16 special result returned early");
      end
      @(posedge clk);
      #1;
      if (select_recip) begin
        if (!recip_output_valid || recip_result !== {16'b0, expected})
          $fatal(1, "BF16 special opcode %0d expected %h got %h",
                 operation, expected, recip_result);
      end else if (!exp_output_valid || exp_result !== {16'b0, expected}) begin
        $fatal(1, "BF16 exp expected %h got %h", expected, exp_result);
      end
      exp_format = VXM_FORMAT_FP16;
      recip_format = VXM_FORMAT_FP16;
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    basic_input_valid = 1'b0;
    basic_format = VXM_FORMAT_FP16;
    basic_opcode = VXM_LOCAL_BYPASS;
    basic_lhs = '0;
    basic_rhs = '0;
    exp_input_valid = 1'b0;
    exp_format = VXM_FORMAT_FP16;
    exp_opcode = VXM_LOCAL_SPECIAL0;
    exp_lhs = '0;
    recip_input_valid = 1'b0;
    recip_format = VXM_FORMAT_FP16;
    recip_opcode = VXM_LOCAL_SPECIAL0;
    recip_lhs = '0;
    exp_lut_config_valid = 1'b0;
    exp_lut_config_bank = '0;
    exp_lut_config_min = '0;
    exp_lut_config_width = 16'h3c00;
    exp_lut_write_valid = 1'b0;
    exp_lut_write_bank = '0;
    exp_lut_write_address = '0;
    exp_lut_write_k = '0;
    exp_lut_write_b = '0;
    recip_lut_config_valid = 1'b0;
    recip_lut_config_bank = '0;
    recip_lut_config_min = '0;
    recip_lut_config_width = 16'h3c00;
    recip_lut_write_valid = 1'b0;
    recip_lut_write_bank = '0;
    recip_lut_write_address = '0;
    recip_lut_write_k = '0;
    recip_lut_write_b = '0;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    // Minimal programmed tables for the exact points checked below. The
    // memory is external to both special execution units.
    program_lut_entry_zero(1'b0, 2'd0, 16'h0000, 16'h3c00,
                           16'h1000, 16'h0000);
    program_lut_entry_zero(1'b1, 2'd1, 16'h3c00, 16'h3c00,
                           16'h0000, 16'h8000);
    program_lut_entry_zero(1'b1, 2'd2, 16'h3c00, 16'h3c00,
                           16'h0000, 16'h8000);

    issue_basic_one_cycle(VXM_LOCAL_BYPASS, 16'h3c00, 16'h0000,
                          16'h3c00);
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h3c00, 16'h4000,
                          16'h4200);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h4200, 16'h4000,
                          16'h3c00);
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h4000, 16'hbc00,
                          16'h3c00);
    // FP16 alignment uses a zero-extended exponent in the shared subtractor.
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h3c00, 16'h0400,
                          16'h3c00);
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h7bff, 16'h0400,
                          16'h7bff);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h0400, 16'h3c00,
                          16'hbc00);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h3c00, 16'h4000,
                          16'hbc00);
    // Near subtraction exercises full leading-zero normalization; the
    // two-exponent-gap case takes the one-bit far-path bypass.
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h3c01, 16'h3c00,
                          16'h1400);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h3c00, 16'h3bff,
                          16'h1000);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h3c00, 16'h3400,
                          16'h3a00);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h0401, 16'h0400,
                          16'h0000); // A nonzero subnormal result is FTZ.
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h0400, 16'h0401,
                          16'h8000); // FTZ preserves the arithmetic sign.
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h7bff, 16'h7bff,
                          16'h7c00);
    issue_basic_16_with_upper_noise(VXM_FORMAT_FP16, VXM_LOCAL_ADD,
      32'hdead3c00, 32'hbeef4000, 16'h4200);
    issue_basic_one_cycle(VXM_LOCAL_NEGATE, 16'h3c00, 16'h0000,
                          16'hbc00);
    issue_basic_one_cycle(VXM_LOCAL_MAX, 16'hc000, 16'h3e00,
                          16'h3e00);

    // Multiply has one extra registered stage. These cases select E0, select
    // E1 after product normalization, and select E1 after an RNE carry.
    issue_basic_fp16_multiply(16'h3e00, 16'h4000, 16'h4200);
    issue_basic_fp16_multiply(16'h3e00, 16'h3e00, 16'h4080);
    issue_basic_fp16_multiply(16'h3c01, 16'h3ffe, 16'h4000);
    issue_basic_fp16_multiply(16'h7bff, 16'h4000, 16'h7c00);
    issue_basic_fp16_multiply(16'h0400, 16'h3800, 16'h0000);
    issue_basic_fp16_multiply(16'h7c00, 16'h0000, 16'h7e00);
    issue_basic_fp16_multiply(16'hbc00, 16'h4000, 16'hc000);
    check_back_to_back_fp16_multiply();

    // BYPASS and NEGATE are raw-bit operations; arithmetic DAZ does not alter
    // their payloads.
    issue_basic_one_cycle(VXM_LOCAL_BYPASS, 16'h0001, 16'h0000,
                          16'h0001);
    issue_basic_one_cycle(VXM_LOCAL_NEGATE, 16'h0001, 16'h0000,
                          16'h8001);
    issue_basic_one_cycle(VXM_LOCAL_BYPASS, 16'h7e01, 16'h0000,
                          16'h7e01);
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h0001, 16'h3c00,
                          16'h3c00);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h0001, 16'h3c00,
                          16'hbc00);
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h8000, 16'h8000,
                          16'h8000);
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h0000, 16'h8000,
                          16'h0000);
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h7c00, 16'hfc00,
                          16'h7e00);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h7c00, 16'h7c00,
                          16'h7e00);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h3c00, 16'h7c00,
                          16'hfc00);
    issue_basic_one_cycle(VXM_LOCAL_SUBTRACT, 16'h8000, 16'h0000,
                          16'h8000);
    issue_basic_one_cycle(VXM_LOCAL_ADD, 16'h3c00, 16'h7e01,
                          16'h7e00);
    issue_basic_one_cycle(VXM_LOCAL_MAX, 16'h0000, 16'h8000,
                          16'h0000);
    issue_basic_one_cycle(VXM_LOCAL_MAX, 16'h3c00, 16'h7e01,
                          16'h7e00);

    // FP32 requests use the same public ALU ports and preserve the existing
    // one-cycle/two-cycle Basic timing contract.
    issue_basic_fp32(VXM_LOCAL_BYPASS, 32'h3fc00000, 32'h00000000,
                     32'h3fc00000, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h3fc00000, 32'h40000000,
                     32'h40600000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h40400000, 32'h40000000,
                     32'h3f800000, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h40000000, 32'hbf800000,
                     32'h3f800000, 1);
    // Wide exponent gap, with each input order, still selects one subtractor.
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h3f800000, 32'h00800000,
                     32'h3f800000, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h7f7fffff, 32'h00800000,
                     32'h7f7fffff, 1); // Exponent distance 253.
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h00800000, 32'h3f800000,
                     32'hbf800000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h3f800000, 32'h40000000,
                     32'hbf800000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h3f800001, 32'h3f800000,
                     32'h34000000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h3f800000, 32'h3f7fffff,
                     32'h33800000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h3f800000, 32'h3e800000,
                     32'h3f400000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h00800001, 32'h00800000,
                     32'h00000000, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h7f7fffff, 32'h7f7fffff,
                     32'h7f800000, 1);
    issue_basic_fp32(VXM_LOCAL_MULTIPLY, 32'h3fc00000, 32'h40000000,
                     32'h40400000, 2);
    issue_basic_fp32(VXM_LOCAL_MULTIPLY, 32'h3fc00000, 32'h3fc00000,
                     32'h40100000, 2);
    issue_basic_fp32(VXM_LOCAL_MULTIPLY, 32'h3f800001, 32'h3ffffffe,
                     32'h40000000, 2);
    issue_basic_fp32(VXM_LOCAL_MULTIPLY, 32'hff7fffff, 32'h40000000,
                     32'hff800000, 2);
    issue_basic_fp32(VXM_LOCAL_NEGATE, 32'h3f800000, 32'h00000000,
                     32'hbf800000, 1);
    issue_basic_fp32(VXM_LOCAL_MAX, 32'hc0000000, 32'h3fc00000,
                     32'h3fc00000, 1);
    // Raw-bit operations preserve FP32 subnormal and NaN payload bits.
    issue_basic_fp32(VXM_LOCAL_BYPASS, 32'h00000001, 32'h00000000,
                     32'h00000001, 1);
    issue_basic_fp32(VXM_LOCAL_NEGATE, 32'h00000001, 32'h00000000,
                     32'h80000001, 1);
    issue_basic_fp32(VXM_LOCAL_BYPASS, 32'h7fc00001, 32'h00000000,
                     32'h7fc00001, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h00000001, 32'h3f800000,
                     32'h3f800000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h00000001, 32'h3f800000,
                     32'hbf800000, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h80000000, 32'h80000000,
                     32'h80000000, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h00000000, 32'h80000000,
                     32'h00000000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h7f800000, 32'h7f800000,
                     32'h7fc00000, 1);
    issue_basic_fp32(VXM_LOCAL_SUBTRACT, 32'h3f800000, 32'h7f800000,
                     32'hff800000, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h3f800000, 32'h7fc00001,
                     32'h7fc00000, 1);
    issue_basic_fp32(VXM_LOCAL_MAX, 32'h00000000, 32'h80000000,
                     32'h00000000, 1);
    issue_basic_fp32(VXM_LOCAL_ADD, 32'h7f800000, 32'hff800000,
                     32'h7fc00000, 1);
    issue_basic_fp32(VXM_LOCAL_MULTIPLY, 32'h7f800000, 32'h00000000,
                     32'h7fc00000, 2);
    issue_basic_fp32(VXM_LOCAL_MULTIPLY, 32'h00800000, 32'h3f000000,
                     32'h00000000, 2);
    issue_basic_fp32(VXM_LOCAL_MAX, 32'h7fc00001, 32'h3f800000,
                     32'h7fc00000, 1);
    issue_basic_fp32(VXM_LOCAL_MAX, 32'h40000000, 32'h7fc00001,
                     32'h7fc00000, 1);

    // BF16 occupies one 16-bit stream beat. ADD/SUB use the native BF16
    // active width of the shared significand datapath and round directly.
    issue_basic_bf16(VXM_LOCAL_BYPASS, 16'h3fc0, 16'h0000,
                     16'h3fc0, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h3fc0, 16'h4000,
                     16'h4060, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h4040, 16'h4000,
                     16'h3f80, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h4000, 16'hbf80,
                     16'h3f80, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h3f80, 16'h0080,
                     16'h3f80, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h7f7f, 16'h0080,
                     16'h7f7f, 1); // Exponent distance 253.
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h0080, 16'h3f80,
                     16'hbf80, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h3f80, 16'h4000,
                     16'hbf80, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h3f81, 16'h3f80,
                     16'h3c00, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h3f80, 16'h3f7f,
                     16'h3b80, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h3f80, 16'h3e80,
                     16'h3f40, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h0081, 16'h0080,
                     16'h0000, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h7f7f, 16'h7f7f,
                     16'h7f80, 1);
    issue_basic_16_with_upper_noise(VXM_FORMAT_BF16, VXM_LOCAL_ADD,
      32'hdead3f80, 32'hbeef4000, 16'h4040);
    issue_basic_bf16(VXM_LOCAL_MULTIPLY, 16'h3fc0, 16'h4000,
                     16'h4040, 2);
    issue_basic_bf16(VXM_LOCAL_MULTIPLY, 16'h3fc0, 16'h3fc0,
                     16'h4010, 2);
    issue_basic_bf16(VXM_LOCAL_MULTIPLY, 16'h3f81, 16'h3ffe,
                     16'h4000, 2);
    issue_basic_bf16(VXM_LOCAL_MULTIPLY, 16'hff7f, 16'h4000,
                     16'hff80, 2);
    issue_basic_bf16(VXM_LOCAL_MULTIPLY, 16'h0080, 16'h3f00,
                     16'h0000, 2);
    issue_basic_bf16(VXM_LOCAL_NEGATE, 16'h3f80, 16'h0000,
                     16'hbf80, 1);
    issue_basic_bf16(VXM_LOCAL_MAX, 16'hc000, 16'h3fc0,
                     16'h3fc0, 1);
    issue_basic_bf16(VXM_LOCAL_BYPASS, 16'h0001, 16'h0000,
                     16'h0001, 1);
    issue_basic_bf16(VXM_LOCAL_NEGATE, 16'h0001, 16'h0000,
                     16'h8001, 1);
    issue_basic_bf16(VXM_LOCAL_BYPASS, 16'h7fc1, 16'h0000,
                     16'h7fc1, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h0001, 16'h3f80,
                     16'h3f80, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h0001, 16'h3f80,
                     16'hbf80, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h8000, 16'h8000,
                     16'h8000, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h0000, 16'h8000,
                     16'h0000, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h7f80, 16'h7f80,
                     16'h7fc0, 1);
    issue_basic_bf16(VXM_LOCAL_SUBTRACT, 16'h3f80, 16'h7f80,
                     16'hff80, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h3f80, 16'h7fc1,
                     16'h7fc0, 1);
    issue_basic_bf16(VXM_LOCAL_MAX, 16'h0000, 16'h8000,
                     16'h0000, 1);
    issue_basic_bf16(VXM_LOCAL_MAX, 16'h3f80, 16'h7fc1,
                     16'h7fc0, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h7f80, 16'hff80,
                     16'h7fc0, 1);
    // Halfway cases verify RNE on the per-ALU BF16 result boundary.
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h3f80, 16'h3b80,
                     16'h3f80, 1);
    issue_basic_bf16(VXM_LOCAL_ADD, 16'h3f81, 16'h3b80,
                     16'h3f82, 1);
    issue_basic_bf16(VXM_LOCAL_MULTIPLY, 16'h7f80, 16'h0000,
                     16'h7fc0, 2);

    @(negedge clk);
    basic_format = VXM_FORMAT_RESERVED;
    basic_input_valid = 1'b1;
    #1;
    if (!basic_unsupported || basic_illegal)
      $fatal(1, "reserved format must report only unsupported_format");
    @(negedge clk);
    basic_input_valid = 1'b0;
    basic_format = VXM_FORMAT_FP16;

    @(negedge clk);
    basic_opcode = VXM_LOCAL_SPECIAL0;
    basic_input_valid = 1'b1;
    #1;
    if (!basic_illegal || basic_unsupported)
      $fatal(1, "special opcode on a basic-only position must be illegal");
    @(negedge clk);
    basic_input_valid = 1'b0;

    issue_special_and_expect(1'b0, VXM_LOCAL_SPECIAL0,
                             16'h0000, 16'h3c00); // exp(0) = 1
    issue_special_and_expect(1'b1, VXM_LOCAL_SPECIAL0,
                             16'h4000, 16'h3800); // reciprocal(2) = 0.5
    issue_special_and_expect(1'b1, VXM_LOCAL_SPECIAL1,
                             16'h4400, 16'h3800); // rsqrt(4) = 0.5

    // FP32 EXP uses the packed UQ1.25 base and cubic Horner continuation;
    // RECIP/RSQRT retain their format-specific coefficient paths.
    issue_special_fp32_and_expect(1'b0, VXM_LOCAL_SPECIAL0,
                                  32'h00000000, 32'h3f800000);
    issue_special_fp32_and_expect(1'b1, VXM_LOCAL_SPECIAL0,
                                  32'h40000000, 32'h3f000000);
    issue_special_fp32_and_expect(1'b1, VXM_LOCAL_SPECIAL1,
                                  32'h40800000, 32'h3f000000);
    issue_special_fp32_and_expect(1'b0, VXM_LOCAL_SPECIAL0,
                                  32'h7f800000, 32'h7f800000);
    issue_special_fp32_and_expect(1'b0, VXM_LOCAL_SPECIAL0,
                                  32'hff800000, 32'h00000000);
    issue_special_fp32_and_expect(1'b1, VXM_LOCAL_SPECIAL0,
                                  32'h00000000, 32'h7f800000);
    issue_special_fp32_and_expect(1'b1, VXM_LOCAL_SPECIAL1,
                                  32'hbf800000, 32'h7fc00000);

    // BF16 special operations reuse FP32 interpolation arithmetic and round
    // only at the output. EXP uses its packed UQ1.25 base in linear mode.
    issue_special_bf16_and_expect(1'b0, VXM_LOCAL_SPECIAL0,
                                  16'h0000, 16'h3f80);
    issue_special_bf16_and_expect(1'b1, VXM_LOCAL_SPECIAL0,
                                  16'h4000, 16'h3f00);
    issue_special_bf16_and_expect(1'b1, VXM_LOCAL_SPECIAL1,
                                  16'h4080, 16'h3f00);
    issue_special_bf16_and_expect(1'b0, VXM_LOCAL_SPECIAL0,
                                  16'h7f80, 16'h7f80);
    issue_special_bf16_and_expect(1'b1, VXM_LOCAL_SPECIAL0,
                                  16'h0000, 16'h7f80);
    issue_special_bf16_and_expect(1'b1, VXM_LOCAL_SPECIAL1,
                                  16'hbf80, 16'h7fc0);

    if (special_fault_seen)
      $fatal(1, "unexpected LUT protocol/configuration fault");

    $display("LPU_VXM_ALU_TB_PASS");
    $finish;
  end
endmodule
