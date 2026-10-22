`timescale 1ns/1ps

// Runs the shared port-only execution-stage checker for one physical ALU.
module lpu_vxm_alu_position_tb_base #(
  parameter integer PHYSICAL_STAGE = 0
);
  logic clk;
  logic rst_n;
  wire done;

  always #5 clk = ~clk;

  lpu_vxm_alu_stage_checker #(
    .PHYSICAL_STAGE(PHYSICAL_STAGE)
  ) checker (
    .clk_i(clk),
    .rst_ni(rst_n),
    .done_o(done)
  );

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    wait (done);
    $display("LPU_VXM_ALU_POSITION_TB_PASS physical=%0d queue=%0d",
             PHYSICAL_STAGE, PHYSICAL_STAGE % 8);
    $finish;
  end

  initial begin
    #100000;
    $fatal(1, "VXM ALU%0d position regression timed out", PHYSICAL_STAGE);
  end
endmodule
