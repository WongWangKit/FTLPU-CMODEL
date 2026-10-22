`timescale 1ns/1ps

module lpu_vxm_rsqrt_linear_fixed_tb;
  import lpu_pkg::*;

  logic        enable;
  logic [1:0]  data_format;
  logic [15:0] k_uq1_15;
  logic [15:0] b_uq1_15;
  logic [17:0] residual;
  wire  [31:0] y0;
  wire         range_fault;

  lpu_vxm_rsqrt_linear_fixed dut (
    .enable_i(enable),
    .data_format_i(data_format),
    .k_uq1_15_i(k_uq1_15),
    .b_uq1_15_i(b_uq1_15),
    .residual_i(residual),
    .y0_o(y0),
    .range_fault_o(range_fault)
  );

  task automatic check(
    input logic [1:0] format,
    input logic [17:0] residual_in,
    input logic [31:0] expected
  );
    begin
      data_format = format;
      residual = residual_in;
      #1;
      if (range_fault || (y0 !== expected))
        $fatal(1,
          "RSQRT fixed interpolation format=%0d residual=%h result=%h/%h fault=%b",
          format, residual_in, y0, expected, range_fault);
    end
  endtask

  initial begin
    enable = 1'b1;
    k_uq1_15 = 16'h8000; // 1.0; parity scaling is table-generator owned.
    b_uq1_15 = 16'h6000; // 0.75
    residual = 18'b0;
    data_format = VXM_FORMAT_FP16;

    check(VXM_FORMAT_BF16, 18'd0,     32'h3f400000);
    check(VXM_FORMAT_FP16, 18'd0,     32'h00003a00);
    check(VXM_FORMAT_FP32, 18'd0,     32'h3f400000);

    // All three residuals encode dx=1/128 after format-specific scaling.
    check(VXM_FORMAT_BF16, 18'd1,     32'h3f3e0000);
    check(VXM_FORMAT_FP16, 18'd8,     32'h000039f0);
    check(VXM_FORMAT_FP32, 18'd65536, 32'h3f3e0000);

    enable = 1'b0;
    #1;
    if ((y0 !== 32'b0) || range_fault)
      $fatal(1, "disabled RSQRT interpolation toggled output");

    $display("LPU_VXM_RSQRT_LINEAR_FIXED_TB_PASS");
    $finish;
  end
endmodule
