`timescale 1ns/1ps

module lpu_vxm_add_tb;
  lpu_vxm_operation_tb_base #(
    .TEST_ID(1),
    .OPCODE(lpu_pkg::VXM_LOCAL_ADD),
    .LHS(16'h3c00),
    .RHS(16'h4000),
    .EXPECTED(16'h4200),
    .LATENCY(1)
  ) test();
endmodule
