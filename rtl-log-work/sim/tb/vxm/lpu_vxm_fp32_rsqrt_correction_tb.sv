`timescale 1ns/1ps

module lpu_vxm_fp32_rsqrt_correction_tb;
  logic        enable;
  logic [31:0] m;
  logic [31:0] y_squared;
  wire  [31:0] correction;
  wire         range_fault;

  lpu_vxm_fp32_rsqrt_correction dut (
    .enable_i(enable),
    .m_i(m),
    .y_squared_i(y_squared),
    .correction_o(correction),
    .range_fault_o(range_fault)
  );

  task automatic check(
    input logic [31:0] m_in,
    input logic [31:0] y_squared_in,
    input logic [31:0] expected
  );
    begin
      m = m_in;
      y_squared = y_squared_in;
      #1;
      if (range_fault || (correction !== expected))
        $fatal(1,
          "RSQRT correction m=%h y2=%h result=%h/%h fault=%b",
          m_in, y_squared_in, correction, expected, range_fault);
    end
  endtask

  initial begin
    enable = 1'b1;
    m = 32'h3f800000;
    y_squared = 32'h3f800000;

    check(32'h3f800000, 32'h3f800000, 32'h3f800000); // 1.5-0.5=1
    check(32'h40000000, 32'h3f000000, 32'h3f800000); // 1.5-0.5=1
    check(32'h3f800000, 32'h3e800000, 32'h3fb00000); // 1.5-0.125

    enable = 1'b0;
    #1;
    if ((correction !== 32'b0) || range_fault)
      $fatal(1, "disabled RSQRT correction toggled output");

    $display("LPU_VXM_FP32_RSQRT_CORRECTION_TB_PASS");
    $finish;
  end
endmodule
