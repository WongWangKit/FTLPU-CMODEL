`timescale 1ns/1ps

module lpu_vxm_max_tb;
  lpu_vxm_operation_tb_base #(
    .TEST_ID(5),
    .OPCODE(lpu_pkg::VXM_LOCAL_MAX),
    .LHS(16'hc000),
    .RHS(16'h3e00),
    .EXPECTED(16'h3e00),
    .LATENCY(1)
  ) test();
endmodule
