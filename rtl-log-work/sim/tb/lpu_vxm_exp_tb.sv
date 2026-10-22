`timescale 1ns/1ps

module lpu_vxm_exp_tb;
  lpu_vxm_operation_tb_base #(
    .TEST_ID(6),
    .SPECIAL_KIND(lpu_pkg::VXM_SPECIAL_EXP),
    .OPCODE(lpu_pkg::VXM_LOCAL_SPECIAL0),
    .LHS(16'h0000),
    .EXPECTED(16'h3c00),
    .LATENCY(5),
    .LUT_BANK(2'd0),
    .LUT_INPUT_MIN(16'hb800),
    .LUT_SEGMENT_WIDTH(16'h3c00),
    .LUT_K(16'h3c00),
    .LUT_B(16'h3800)
  ) test();
endmodule
