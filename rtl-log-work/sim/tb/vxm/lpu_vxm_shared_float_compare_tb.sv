`timescale 1ns/1ps

module lpu_vxm_shared_float_compare_tb;
  import lpu_pkg::*;

  logic        enable;
  logic [1:0]  data_format;
  logic [31:0] lhs;
  logic [31:0] rhs;

  wire lhs_sign;
  wire [7:0] lhs_exponent;
  wire [22:0] lhs_fraction;
  wire lhs_zero;
  wire lhs_nan;
  wire rhs_sign;
  wire [7:0] rhs_exponent;
  wire [22:0] rhs_fraction;
  wire rhs_zero;
  wire rhs_nan;

  wire magnitude_gt;
  wire magnitude_equal;
  wire ordered_gt;
  wire ordered_equal;
  wire unordered;

  lpu_vxm_float_unpack u_lhs_unpack (
    .data_format_i(data_format),
    .value_i(lhs),
    .format_valid_o(),
    .sign_o(lhs_sign),
    .exponent_o(lhs_exponent),
    .fraction_o(lhs_fraction),
    .effective_zero_o(lhs_zero),
    .was_subnormal_o(),
    .is_normal_o(),
    .is_inf_o(),
    .is_nan_o(lhs_nan)
  );

  lpu_vxm_float_unpack u_rhs_unpack (
    .data_format_i(data_format),
    .value_i(rhs),
    .format_valid_o(),
    .sign_o(rhs_sign),
    .exponent_o(rhs_exponent),
    .fraction_o(rhs_fraction),
    .effective_zero_o(rhs_zero),
    .was_subnormal_o(),
    .is_normal_o(),
    .is_inf_o(),
    .is_nan_o(rhs_nan)
  );

  lpu_vxm_shared_float_compare dut (
    .enable_i(enable),
    .data_format_i(data_format),
    .lhs_sign_i(lhs_sign),
    .lhs_exponent_i(lhs_exponent),
    .lhs_fraction_i(lhs_fraction),
    .lhs_zero_i(lhs_zero),
    .lhs_nan_i(lhs_nan),
    .rhs_sign_i(rhs_sign),
    .rhs_exponent_i(rhs_exponent),
    .rhs_fraction_i(rhs_fraction),
    .rhs_zero_i(rhs_zero),
    .rhs_nan_i(rhs_nan),
    .magnitude_gt_o(magnitude_gt),
    .magnitude_equal_o(magnitude_equal),
    .ordered_gt_o(ordered_gt),
    .ordered_equal_o(ordered_equal),
    .unordered_o(unordered)
  );

  task automatic check_compare(
    input logic [1:0]  format,
    input logic [31:0] lhs_value,
    input logic [31:0] rhs_value,
    input logic [4:0]  expected
  );
    begin
      enable = 1'b1;
      data_format = format;
      lhs = lhs_value;
      rhs = rhs_value;
      #1;
      if ({magnitude_gt, magnitude_equal, ordered_gt,
           ordered_equal, unordered} !== expected)
        $fatal(1,
          "format=%0d lhs=%h rhs=%h compare=%b expected=%b",
          format, lhs_value, rhs_value,
          {magnitude_gt, magnitude_equal, ordered_gt,
           ordered_equal, unordered}, expected);
    end
  endtask

  initial begin
    enable = 1'b0;
    data_format = VXM_FORMAT_FP16;
    lhs = '0;
    rhs = '0;
    #1;
    if ({magnitude_gt, magnitude_equal, ordered_gt,
         ordered_equal, unordered} !== 5'b0)
      $fatal(1, "disabled comparator produced a relation");

    // expected order: {magnitude_gt, magnitude_equal, ordered_gt,
    //                  ordered_equal, unordered}
    // FP16 and BF16 use the complete shared low-15-bit comparator.
    check_compare(VXM_FORMAT_FP16, 32'h0000c000, 32'h00003e00,
                  5'b10000); // |-2|>|1.5|, but -2 < 1.5.
    check_compare(VXM_FORMAT_FP16, 32'h00003c01, 32'h00003c02,
                  5'b00000);
    check_compare(VXM_FORMAT_BF16, 32'h0000c000, 32'h00003fc0,
                  5'b10000);

    // FP32 first compares the high 16 bits and reuses the low 15-bit leaf
    // when the high portions match.
    check_compare(VXM_FORMAT_FP32, 32'h3f800000, 32'h40000000,
                  5'b00000);
    check_compare(VXM_FORMAT_FP32, 32'h3f800001, 32'h3f800002,
                  5'b00000);
    check_compare(VXM_FORMAT_FP32, 32'hbf800002, 32'hbf800001,
                  5'b10000);

    check_compare(VXM_FORMAT_FP32, 32'h40000000, 32'h3f800000,
                  5'b10100); // +2 > +1.
    check_compare(VXM_FORMAT_FP32, 32'hbf800000, 32'hc0000000,
                  5'b00100); // -1 > -2.

    // DAZ reaches the comparator through the common unpackers.
    check_compare(VXM_FORMAT_FP16, 32'h00000001, 32'h00003c00,
                  5'b00000);

    // Signed zeros are ordered equal; MAX applies its +0 selection outside
    // the comparator. NaNs are unordered and suppress all ordered relations.
    check_compare(VXM_FORMAT_FP32, 32'h80000000, 32'h00000000,
                  5'b01010);
    check_compare(VXM_FORMAT_FP16, 32'h00007e01, 32'h00003c00,
                  5'b00001);
    check_compare(VXM_FORMAT_BF16, 32'h00003f80, 32'h00007fc1,
                  5'b00001);

    check_compare(VXM_FORMAT_RESERVED, 32'hffffffff, 32'h12345678,
                  5'b00000);

    $display("LPU_VXM_SHARED_FLOAT_COMPARE_TB_PASS");
    $finish;
  end
endmodule
