`timescale 1ns/1ps

module lpu_vxm_special_coefficient_multiplier_tb;
  import lpu_pkg::*;

  logic        enable;
  logic [1:0]  data_format;
  logic [25:0] coefficient;
  logic [31:0] value;
  wire  [31:0] coefficient_value;
  wire  [31:0] product;

  lpu_vxm_special_coefficient_multiplier dut (
    .enable_i(enable),
    .data_format_i(data_format),
    .coefficient_i(coefficient),
    .offset_i(coefficient),
    .value_i(value),
    .coefficient_value_o(coefficient_value),
    .offset_value_o(),
    .product_o(product)
  );

  task automatic check(
    input logic [1:0] format,
    input logic [25:0] coefficient_in,
    input logic [31:0] value_in,
    input logic [31:0] expected_coefficient,
    input logic [31:0] expected_product
  );
    begin
      data_format = format;
      coefficient = coefficient_in;
      value = value_in;
      #1;
      if ((coefficient_value !== expected_coefficient) ||
          (product !== expected_product))
        $fatal(1,
          "special coefficient multiply format=%0d coefficient=%h/%h product=%h/%h",
          format, coefficient_value, expected_coefficient,
          product, expected_product);
    end
  endtask

  initial begin
    enable = 1'b1;
    data_format = VXM_FORMAT_FP16;
    coefficient = 26'h2000000; // UQ1.25 1.0
    value = 32'b0;

    // All stimulus and observations use only public module ports.
    check(VXM_FORMAT_BF16, 26'h2000000, 32'h3e000000,
          32'h3f800000, 32'h3e000000); // 1.0 * 0.125
    check(VXM_FORMAT_FP16, 26'h2000000, 32'h00003000,
          32'h00003c00, 32'h00003000); // 1.0 * 0.125
    check(VXM_FORMAT_FP32, 26'h2000000, 32'h3e000000,
          32'h3f800000, 32'h3e000000); // 1.0 * 0.125
    check(VXM_FORMAT_BF16, 26'h1800000, 32'h3f000000,
          32'h3f400000, 32'h3ec00000); // 0.75 * 0.5
    check(VXM_FORMAT_FP16, 26'h1800000, 32'h00003800,
          32'h00003a00, 32'h00003600); // 0.75 * 0.5
    check(VXM_FORMAT_FP32, 26'h1800000, 32'h3f000000,
          32'h3f400000, 32'h3ec00000); // 0.75 * 0.5

    // 1.0+2^-10 is a BF16 UQ1.9 halfway case (ties to even), but remains
    // representable in the wider FP16 UQ1.12 and FP32 UQ1.25 modes.
    check(VXM_FORMAT_BF16, 26'h2008000, 32'h3f000000,
          32'h3f800000, 32'h3f000000);
    check(VXM_FORMAT_FP16, 26'h2008000, 32'h00003800,
          32'h00003c01, 32'h00003801);
    check(VXM_FORMAT_FP32, 26'h2008000, 32'h3f000000,
          32'h3f802000, 32'h3f002000);

    enable = 1'b0;
    #1;
    if ((coefficient_value !== 32'b0) || (product !== 32'b0))
      $fatal(1, "disabled special coefficient multiplier toggled output");

    $display("LPU_VXM_SPECIAL_COEFFICIENT_MULTIPLIER_TB_PASS");
    $finish;
  end
endmodule
