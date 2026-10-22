`timescale 1ns/1ps

module lpu_vxm_subtract_tb;
  lpu_vxm_operation_tb_base #(
    .TEST_ID(2),
    .OPCODE(lpu_pkg::VXM_LOCAL_SUBTRACT),
    .LHS(16'h4200),
    .RHS(16'h4000),
    .EXPECTED(16'h3c00),
    .LATENCY(1)
  ) test();
endmodule
