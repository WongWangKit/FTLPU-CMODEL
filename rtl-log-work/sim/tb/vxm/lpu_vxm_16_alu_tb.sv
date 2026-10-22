`timescale 1ns/1ps

module lpu_vxm_16_alu_tb;
  logic clk;
  logic rst_n;
  wire [15:0] done;

  always #5 clk = ~clk;

  genvar stage;
  generate
    for (stage = 0; stage < 16; stage = stage + 1) begin : g_stage
      lpu_vxm_alu_stage_checker #(.PHYSICAL_STAGE(stage)) u_checker (
        .clk_i(clk),
        .rst_ni(rst_n),
        .done_o(done[stage])
      );
    end
  endgenerate

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    wait (&done);
    $display("LPU_VXM_16_ALU_TB_PASS");
    $finish;
  end

  initial begin
    #100000;
    $fatal(1, "VXM 16-ALU regression timed out, done=%h", done);
  end
endmodule
