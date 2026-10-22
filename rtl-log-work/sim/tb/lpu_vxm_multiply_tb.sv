`timescale 1ns/1ps

module lpu_vxm_multiply_tb;
  lpu_vxm_operation_tb_base #(
    .TEST_ID(3),
    .OPCODE(lpu_pkg::VXM_LOCAL_MULTIPLY),
    .LHS(16'h3e00),
    .RHS(16'h4000),
    .EXPECTED(16'h4200),
    .LATENCY(2)
  ) test();
endmodule
