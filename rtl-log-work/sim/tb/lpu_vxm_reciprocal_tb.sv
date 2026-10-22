`timescale 1ns/1ps

module lpu_vxm_reciprocal_tb;
  lpu_vxm_operation_tb_base #(
    .TEST_ID(7),
    .SPECIAL_KIND(lpu_pkg::VXM_SPECIAL_RECIP_RSQRT),
    .OPCODE(lpu_pkg::VXM_LOCAL_SPECIAL0),
    .LHS(16'h4000),
    .EXPECTED(16'h3800),
    .LATENCY(5),
    .LUT_BANK(2'd1),
    .LUT_INPUT_MIN(16'h3c00),
    .LUT_SEGMENT_WIDTH(16'h3c00),
    .LUT_K(16'h0000),
    .LUT_B(16'h3c00)
  ) test();
endmodule
