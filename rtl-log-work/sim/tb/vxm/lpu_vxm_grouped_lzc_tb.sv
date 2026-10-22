// Port-only checks of the shared four-group leading-zero encoder.
module lpu_vxm_grouped_lzc_tb;
  logic enable;
  logic [5:0] active_width;
  logic [26:0] value;
  wire [4:0] shift;
  wire zero;

  lpu_vxm_grouped_lzc dut (
    .enable_i(enable),
    .active_width_i(active_width),
    .value_i(value),
    .shift_o(shift),
    .zero_o(zero)
  );

  task automatic check(
    input integer width,
    input logic enabled,
    input logic [26:0] input_value
  );
    logic [26:0] mask_value;
    logic [26:0] effective_value;
    logic [4:0] expected_shift;
    logic expected_zero;
    logic found;
    begin
      case (width)
        11: mask_value = 27'h00007ff;
        14: mask_value = 27'h0003fff;
        27: mask_value = 27'h7ffffff;
        default: $fatal(1, "Unsupported LZC width %0d", width);
      endcase
      effective_value = enabled ? (input_value & mask_value) : 27'b0;
      expected_zero = effective_value == 27'b0;
      expected_shift = 5'd0;
      found = 1'b0;
      for (integer bit_index = width - 1; bit_index >= 0; bit_index--)
        if (!found && effective_value[bit_index]) begin
          expected_shift = width - 1 - bit_index;
          found = 1'b1;
        end

      enable = enabled;
      active_width = width;
      value = input_value;
      #1;
      if ({zero, shift} !== {expected_zero, expected_shift})
        $fatal(1,
          "GROUPED_LZC width=%0d en=%0b value=%h got=%b/%0d expected=%b/%0d",
          width, enabled, input_value, zero, shift,
          expected_zero, expected_shift);
    end
  endtask

  initial begin
    enable = 1'b0;
    active_width = 6'd11;
    value = 27'b0;
    check(11, 1'b0, 27'h7ffffff);
    check(11, 1'b1, 27'h7fff800);
    check(14, 1'b1, 27'h7ffc000);
    check(27, 1'b1, 27'b0);

    for (integer bit_index = 0; bit_index < 27; bit_index++) begin
      if (bit_index < 11)
        check(11, 1'b1, 27'd1 << bit_index);
      if (bit_index < 14)
        check(14, 1'b1, 27'd1 << bit_index);
      check(27, 1'b1, 27'd1 << bit_index);
    end
    for (integer sample = 0; sample < 300; sample++) begin
      check(11, 1'b1, $urandom);
      check(14, 1'b1, $urandom);
      check(27, 1'b1, $urandom);
    end

    $display("LPU_VXM_GROUPED_LZC_TB_PASS");
    $finish;
  end
endmodule
