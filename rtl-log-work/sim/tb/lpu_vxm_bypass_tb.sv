`timescale 1ns/1ps

module lpu_vxm_bypass_tb;
  lpu_vxm_operation_tb_base #(
    .TEST_ID(0),
    .OPCODE(lpu_pkg::VXM_LOCAL_BYPASS),
    .LHS(16'h3c00),
    .EXPECTED(16'h3c00),
    .LATENCY(1)
  ) test();
endmodule
