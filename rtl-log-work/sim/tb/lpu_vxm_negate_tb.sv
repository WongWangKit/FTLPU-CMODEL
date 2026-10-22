`timescale 1ns/1ps

module lpu_vxm_negate_tb;
  lpu_vxm_operation_tb_base #(
    .TEST_ID(4),
    .OPCODE(lpu_pkg::VXM_LOCAL_NEGATE),
    .LHS(16'h3c00),
    .EXPECTED(16'hbc00),
    .LATENCY(1)
  ) test();
endmodule
