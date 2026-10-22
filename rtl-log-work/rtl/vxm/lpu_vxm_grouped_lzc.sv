// Shared 27-bit leading-zero encoder for the near-subtraction path. The four
// groups are 7/7/7/6 bits from low to high, preserving the 14+13 adder boundary.
module lpu_vxm_grouped_lzc (
  input  logic        enable_i,
  input  logic [5:0]  active_width_i,
  input  logic [26:0] value_i,
  output logic [4:0]  shift_o,
  output logic        zero_o
);
  logic [26:0] active_mask;
  logic [26:0] value_masked;
  logic [4:0] full_count;
  logic [4:0] format_offset;

  function automatic logic [2:0] clz7(input logic [6:0] value);
    casez (value)
      7'b1??????: clz7 = 3'd0;
      7'b01?????: clz7 = 3'd1;
      7'b001????: clz7 = 3'd2;
      7'b0001???: clz7 = 3'd3;
      7'b00001??: clz7 = 3'd4;
      7'b000001?: clz7 = 3'd5;
      7'b0000001: clz7 = 3'd6;
      default:    clz7 = 3'd7;
    endcase
  endfunction

  always_comb begin
    active_mask = 27'b0;
    format_offset = 5'd0;
    case (active_width_i)
      6'd11: begin
        active_mask = 27'h00007ff;
        format_offset = 5'd16;
      end
      6'd14: begin
        active_mask = 27'h0003fff;
        format_offset = 5'd13;
      end
      6'd27: begin
        active_mask = 27'h7ffffff;
        format_offset = 5'd0;
      end
      default: begin end
    endcase
    value_masked = enable_i ? (value_i & active_mask) : 27'b0;
    zero_o = value_masked == 27'b0;

    // Each group is reduced in parallel, then the highest nonzero group is
    // selected. The top six-bit group is padded at its LSB for clz7.
    full_count = 5'd27;
    if (|value_masked[26:21])
      full_count = {2'b0, clz7({value_masked[26:21], 1'b0})};
    else if (|value_masked[20:14])
      full_count = 5'd6 + {2'b0, clz7(value_masked[20:14])};
    else if (|value_masked[13:7])
      full_count = 5'd13 + {2'b0, clz7(value_masked[13:7])};
    else if (|value_masked[6:0])
      full_count = 5'd20 + {2'b0, clz7(value_masked[6:0])};

    shift_o = zero_o ? 5'd0 : full_count - format_offset;
  end
endmodule
