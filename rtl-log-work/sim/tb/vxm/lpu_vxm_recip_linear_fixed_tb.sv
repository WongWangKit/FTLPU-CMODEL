`timescale 1ns/1ps

module lpu_vxm_recip_linear_fixed_tb;
  import lpu_pkg::*;

  logic        enable;
  logic [1:0]  data_format;
  logic [15:0] k_uq1_15;
  logic [15:0] b_uq1_15;
  logic [16:0] residual;
  wire  [31:0] y0;
  wire         range_fault;

  lpu_vxm_recip_linear_fixed dut (
    .enable_i(enable),
    .data_format_i(data_format),
    .k_uq1_15_i(k_uq1_15),
    .b_uq1_15_i(b_uq1_15),
    .residual_i(residual),
    .y0_o(y0),
    .range_fault_o(range_fault)
  );

  task automatic check(
    input logic [1:0]  format,
    input logic [16:0] residual_in,
    input logic [31:0] expected
  );
    begin
      data_format = format;
      residual = residual_in;
      #1;
      if (range_fault || (y0 !== expected))
        $fatal(1,
          "RECIP fixed interpolation format=%0d residual=%h result=%h/%h fault=%b",
          format, residual_in, y0, expected, range_fault);
    end
  endtask

  initial begin
    enable = 1'b1;
    k_uq1_15 = 16'h8000; // 1.0
    b_uq1_15 = 16'h6000; // 0.75
    residual = 17'b0;
    data_format = VXM_FORMAT_FP16;

    // dx=0: the result is exactly the table intercept in every format.
    check(VXM_FORMAT_BF16, 17'd0,     32'h3f400000);
    check(VXM_FORMAT_FP16, 17'd0,     32'h00003a00);
    check(VXM_FORMAT_FP32, 17'd0,     32'h3f400000);

    // The three residual encodings below all represent dx=1/128.
    // Therefore y0=0.75-1/128=0.7421875 in each active format.
    check(VXM_FORMAT_BF16, 17'd1,     32'h3f3e0000);
    check(VXM_FORMAT_FP16, 17'd8,     32'h000039f0);
    check(VXM_FORMAT_FP32, 17'd65536, 32'h3f3e0000);

    enable = 1'b0;
    #1;
    if ((y0 !== 32'b0) || range_fault)
      $fatal(1, "disabled RECIP fixed interpolation toggled output");

    enable = 1'b1;
    data_format = 2'b11;
    #1;
    if (!range_fault || (y0 !== 32'b0))
      $fatal(1, "reserved RECIP format did not report a fault");

    $display("LPU_VXM_RECIP_LINEAR_FIXED_TB_PASS");
    $finish;
  end
endmodule
