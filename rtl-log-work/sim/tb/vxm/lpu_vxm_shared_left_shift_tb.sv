// Port-only checks of the FP16/BF16/FP32 shared normalization shifter.
module lpu_vxm_shared_left_shift_tb;
  logic enable;
  logic [26:0] active_mask;
  logic [26:0] value;
  logic [4:0] shift;
  wire [26:0] result;

  lpu_vxm_shared_left_shift dut (
    .enable_i(enable),
    .active_mask_i(active_mask),
    .value_i(value),
    .shift_i(shift),
    .result_o(result)
  );

  task automatic check(
    input integer width,
    input logic enabled,
    input logic [26:0] input_value,
    input logic [4:0] input_shift
  );
    logic [26:0] mask_value;
    logic [26:0] expected;
    begin
      case (width)
        11: mask_value = 27'h00007ff;
        14: mask_value = 27'h0003fff;
        27: mask_value = 27'h7ffffff;
        default: $fatal(1, "Unsupported left-shift width %0d", width);
      endcase
      expected = enabled ?
        (((input_value & mask_value) << input_shift) & mask_value) : 27'b0;
      enable = enabled;
      active_mask = mask_value;
      value = input_value;
      shift = input_shift;
      #1;
      if (result !== expected)
        $fatal(1,
          "SHARED_LEFT_SHIFT width=%0d en=%0b value=%h shift=%0d got=%h expected=%h",
          width, enabled, input_value, input_shift, result, expected);
    end
  endtask

  initial begin
    enable = 1'b0;
    active_mask = 27'b0;
    value = 27'b0;
    shift = 5'b0;
    check(11, 1'b0, 27'h7ffffff, 5'd1);
    check(14, 1'b1, 27'h7ffc000, 5'd0);
    check(11, 1'b1, 27'h7fffffe, 5'd1);
    check(14, 1'b1, 27'h0002000, 5'd1);
    check(27, 1'b1, 27'h0002000, 5'd1); // Crosses the 14+13 boundary.

    for (integer bit_index = 0; bit_index < 27; bit_index++) begin
      for (integer shift_index = 0; shift_index < 27; shift_index++) begin
        if (bit_index < 11)
          check(11, 1'b1, 27'd1 << bit_index, shift_index[4:0]);
        if (bit_index < 14)
          check(14, 1'b1, 27'd1 << bit_index, shift_index[4:0]);
        check(27, 1'b1, 27'd1 << bit_index, shift_index[4:0]);
      end
    end
    for (integer sample = 0; sample < 300; sample++) begin
      check(11, 1'b1, $urandom, $urandom_range(0, 26));
      check(14, 1'b1, $urandom, $urandom_range(0, 26));
      check(27, 1'b1, $urandom, $urandom_range(0, 26));
    end

    $display("LPU_VXM_SHARED_LEFT_SHIFT_TB_PASS");
    $finish;
  end
endmodule
