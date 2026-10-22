// Port-only checks of the 14+13 shared significand add/subtract datapath.
module lpu_vxm_segmented_addsub_tb;
  logic enable;
  logic subtract;
  logic high_enable;
  logic [26:0] active_mask;
  logic [26:0] lhs;
  logic [26:0] rhs;
  wire [27:0] result;

  lpu_vxm_segmented_addsub dut (
    .enable_i(enable),
    .subtract_i(subtract),
    .high_enable_i(high_enable),
    .active_mask_i(active_mask),
    .lhs_i(lhs),
    .rhs_i(rhs),
    .result_o(result)
  );

  task automatic check(
    input integer width,
    input logic do_subtract,
    input logic [26:0] lhs_value,
    input logic [26:0] rhs_value
  );
    logic [26:0] mask_value;
    logic [26:0] raw_a;
    logic [26:0] raw_b;
    logic [26:0] a;
    logic [26:0] b;
    logic [27:0] expected;
    begin
      case (width)
        11: mask_value = 27'h00007ff;
        14: mask_value = 27'h0003fff;
        27: mask_value = 27'h7ffffff;
        default: $fatal(1, "Unsupported significand width %0d", width);
      endcase
      raw_a = lhs_value;
      raw_b = rhs_value;
      a = raw_a & mask_value;
      b = raw_b & mask_value;
      // The caller orders magnitudes before a subtract operation.
      if (do_subtract && (a < b)) begin
        raw_a = rhs_value;
        raw_b = lhs_value;
        a = raw_a & mask_value;
        b = raw_b & mask_value;
      end
      enable = 1'b1;
      subtract = do_subtract;
      high_enable = (width == 27);
      active_mask = mask_value;
      lhs = raw_a;
      rhs = raw_b;
      if (do_subtract)
        expected = {1'b0, a} - {1'b0, b} + (28'd1 << width);
      else
        expected = {1'b0, a} + {1'b0, b};
      #1;
      if (result !== expected)
        $fatal(1,
          "SEGMENTED_ADDSUB width=%0d sub=%0b a=%h b=%h got=%h expected=%h",
          width, do_subtract, a, b, result, expected);
    end
  endtask

  initial begin
    enable = 1'b0;
    subtract = 1'b0;
    high_enable = 1'b0;
    active_mask = '0;
    lhs = '0;
    rhs = '0;
    #1;
    if (result !== 28'b0)
      $fatal(1, "Disabled add/subtract datapath is not isolated");

    // Carry-out at each format boundary and carry/borrow across the 14-bit
    // boundary when FP32 enables the high group.
    check(11, 1'b0, 27'h00007ff, 27'd1);
    check(14, 1'b0, 27'h0003fff, 27'd1);
    check(27, 1'b0, 27'h7ffffff, 27'd1);
    check(27, 1'b0, 27'h0003fff, 27'd1);
    check(27, 1'b1, 27'h0004000, 27'd1);
    check(14, 1'b1, 27'h0002000, 27'h0000001);
    check(11, 1'b1, 27'h0000400, 27'h0000001);

    for (integer sample = 0; sample < 300; sample++) begin
      check(11, 1'b0, $urandom, $urandom);
      check(11, 1'b1, $urandom, $urandom);
      check(14, 1'b0, $urandom, $urandom);
      check(14, 1'b1, $urandom, $urandom);
      check(27, 1'b0, $urandom, $urandom);
      check(27, 1'b1, $urandom, $urandom);
    end

    enable = 1'b0;
    subtract = 1'b1;
    high_enable = 1'b1;
    active_mask = 27'h7ffffff;
    lhs = 27'h7ffffff;
    rhs = 27'h7ffffff;
    #1;
    if (result !== 28'b0)
      $fatal(1, "Disabled add/subtract datapath leaked a result");

    $display("LPU_VXM_SEGMENTED_ADDSUB_TB_PASS");
    $finish;
  end
endmodule
