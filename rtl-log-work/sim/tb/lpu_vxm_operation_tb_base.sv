`timescale 1ns/1ps

module lpu_vxm_operation_tb_base #(
  parameter integer TEST_ID = 0,
  parameter integer SPECIAL_KIND = lpu_pkg::VXM_SPECIAL_NONE,
  parameter logic [2:0] OPCODE = lpu_pkg::VXM_LOCAL_BYPASS,
  parameter logic [15:0] LHS = 16'h0000,
  parameter logic [15:0] RHS = 16'h0000,
  parameter logic [15:0] EXPECTED = 16'h0000,
  parameter integer LATENCY = 1,
  parameter logic [1:0] LUT_BANK = 2'd0,
  parameter logic [15:0] LUT_INPUT_MIN = 16'h0000,
  parameter logic [15:0] LUT_SEGMENT_WIDTH = 16'h3c00,
  parameter logic [15:0] LUT_K = 16'h0000,
  parameter logic [15:0] LUT_B = 16'h0000
);
  import lpu_pkg::*;

  logic clk;
  logic rst_n;
  logic input_valid;

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
  wire lut_request_valid;
  wire [1:0] lut_request_bank;
  wire [5:0] lut_request_address;
  wire lut_response_valid;
  wire [15:0] lut_response_k;
  wire [15:0] lut_response_b;
  wire lut_storage_fault;

  wire input_ready;
  wire output_valid;
  wire [31:0] result;
  wire illegal_opcode;
  wire unsupported_format;
  wire result_collision;
  wire lut_fault;
  logic fault_seen;

  always #5 clk = ~clk;

  lpu_vxm_lut_storage u_lut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .config_valid_i(lut_config_valid),
    .config_bank_i(lut_config_bank),
    .config_input_min_i(lut_config_min),
    .config_segment_width_i(lut_config_width),
    .write_valid_i(lut_write_valid),
    .write_bank_i(lut_write_bank),
    .write_address_i(lut_write_address),
    .write_k_i(lut_write_k),
    .write_b_i(lut_write_b),
    .read_valid_i(lut_request_valid),
    .read_bank_i(lut_request_bank),
    .read_address_i(lut_request_address),
    .read_valid_o(lut_response_valid),
    .read_k_o(lut_response_k),
    .read_b_o(lut_response_b),
    .configured_o(lut_configured),
    .input_min_o(lut_input_min),
    .segment_width_o(lut_segment_width),
    .fault_o(lut_storage_fault)
  );

  lpu_vxm_alu #(
    .SPECIAL_KIND(SPECIAL_KIND)
  ) dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .input_valid_i(input_valid),
    .data_format_i(VXM_FORMAT_FP16),
    .opcode_i(OPCODE),
    .lhs_i({16'b0, LHS}),
    .rhs_i({16'b0, RHS}),
    .lut_configured_i(lut_configured),
    .lut_input_min_i(lut_input_min),
    .lut_segment_width_i(lut_segment_width),
    .lut_read_valid_o(lut_request_valid),
    .lut_read_bank_o(lut_request_bank),
    .lut_read_address_o(lut_request_address),
    .lut_read_valid_i(lut_response_valid),
    .lut_read_k_i(lut_response_k),
    .lut_read_b_i(lut_response_b),
    .input_ready_o(input_ready),
    .output_valid_o(output_valid),
    .result_o(result),
    .illegal_opcode_o(illegal_opcode),
    .unsupported_format_o(unsupported_format),
    .result_collision_o(result_collision),
    .lut_fault_o(lut_fault)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fault_seen <= 1'b0;
    else
      fault_seen <= fault_seen | illegal_opcode | unsupported_format |
        result_collision | lut_fault | lut_storage_fault;
  end

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    input_valid = 1'b0;
    lut_config_valid = 1'b0;
    lut_config_bank = LUT_BANK;
    lut_config_min = LUT_INPUT_MIN;
    lut_config_width = LUT_SEGMENT_WIDTH;
    lut_write_valid = 1'b0;
    lut_write_bank = LUT_BANK;
    lut_write_address = '0;
    lut_write_k = LUT_K;
    lut_write_b = LUT_B;

    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    if (SPECIAL_KIND != VXM_SPECIAL_NONE) begin
      lut_config_valid = 1'b1;
      lut_write_valid = 1'b1;
      @(posedge clk);
      #1;
      @(negedge clk);
      lut_config_valid = 1'b0;
      lut_write_valid = 1'b0;
    end

    @(negedge clk);
    input_valid = 1'b1;
    #1;
    if (!input_ready)
      $fatal(1, "VXM operation test %0d was not ready", TEST_ID);

    for (integer cycle = 1; cycle <= LATENCY; cycle++) begin
      @(posedge clk);
      #1;
      if ((cycle < LATENCY) && output_valid)
        $fatal(1, "VXM operation test %0d returned at cycle %0d, expected %0d",
               TEST_ID, cycle, LATENCY);
      if (cycle == 1) begin
        @(negedge clk);
        input_valid = 1'b0;
      end
    end

    if (!output_valid || result !== {16'b0, EXPECTED})
      $fatal(1, "VXM operation test %0d expected %h, got valid=%b result=%h",
             TEST_ID, EXPECTED, output_valid, result);
    if (fault_seen)
      $fatal(1, "VXM operation test %0d observed a fault", TEST_ID);

    $display("LPU_VXM_OPERATION_TB_PASS id=%0d opcode=%0d",
             TEST_ID, OPCODE);
    $finish;
  end
endmodule
