// Fixed-point RECIP LUT interpolation for y0 = b - k*dx.
// A 64-entry table consumes the top six fraction bits as its address. The
// remaining fraction bits are a right-aligned residual:
//   BF16: 1 bit / 2^7, FP16: 4 bits / 2^10, FP32: 17 bits / 2^23.
// k and b are physically UQ1.15. Narrow formats use RNE-reduced UQ1.9 and
// UQ1.12 values. One 16x17 unsigned multiply is shared by zero extension;
// the aligned subtraction uses one signed 40-bit datapath.
module lpu_vxm_recip_linear_fixed (
  input  logic        enable_i,
  input  logic [1:0]  data_format_i,
  input  logic [15:0] k_uq1_15_i,
  input  logic [15:0] b_uq1_15_i,
  input  logic [16:0] residual_i,
  output logic [31:0] y0_o,
  output logic        range_fault_o
);
  import lpu_pkg::*;
  import lpu_vxm_math_pkg::*;

  function automatic logic [15:0] round_shift_uq1_15(
    input logic [15:0] value,
    input integer shift
  );
    logic [15:0] retained;
    logic [15:0] discarded;
    logic [15:0] mask;
    logic [15:0] halfway;
    begin
      if (shift == 0) begin
        round_shift_uq1_15 = value;
      end else begin
        retained = value >> shift;
        mask = (16'b1 << shift) - 1'b1;
        discarded = value & mask;
        halfway = 16'b1 << (shift-1);
        if ((discarded > halfway) ||
            ((discarded == halfway) && retained[0]))
          retained = retained + 1'b1;
        round_shift_uq1_15 = retained;
      end
    end
  endfunction

  function automatic logic [31:0] fixed_to_fp32(
    input logic [39:0] magnitude,
    input integer fraction_bits
  );
    integer leading_bit;
    integer exponent;
    logic [39:0] normalized;
    begin
      leading_bit = -1;
      normalized = 40'b0;
      for (integer bit_index = 0; bit_index < 40; bit_index++)
        if (magnitude[bit_index])
          leading_bit = bit_index;
      if (leading_bit < 0) begin
        fixed_to_fp32 = 32'b0;
      end else begin
        exponent = 127 + leading_bit - fraction_bits;
        if (leading_bit <= 23)
          normalized = magnitude << (23-leading_bit);
        else
          normalized = magnitude >> (leading_bit-23);
        fixed_to_fp32 = {1'b0, exponent[7:0], normalized[22:0]};
      end
    end
  endfunction

  logic [15:0] active_k;
  logic [15:0] active_b;
  logic [16:0] active_residual;
  logic [32:0] product;
  logic [38:0] b_aligned;
  logic signed [39:0] b_extended;
  logic signed [39:0] product_extended;
  logic signed [39:0] difference;
  logic [39:0] retained;
  logic [39:0] remainder;
  logic [39:0] remainder_mask;
  logic [39:0] halfway;
  logic [31:0] y0_fp32;
  integer coefficient_fraction_bits;
  integer input_fraction_bits;
  integer coefficient_shift;

  always_comb begin
    active_k = 16'b0;
    active_b = 16'b0;
    active_residual = 17'b0;
    coefficient_fraction_bits = 0;
    input_fraction_bits = 0;
    coefficient_shift = 0;
    range_fault_o = 1'b0;

    case (data_format_i)
      VXM_FORMAT_BF16: begin
        coefficient_fraction_bits = 9;
        input_fraction_bits = 7;
        coefficient_shift = 6;
        active_residual[0] = residual_i[0];
      end
      VXM_FORMAT_FP16: begin
        coefficient_fraction_bits = 12;
        input_fraction_bits = 10;
        coefficient_shift = 3;
        active_residual[3:0] = residual_i[3:0];
      end
      VXM_FORMAT_FP32: begin
        coefficient_fraction_bits = 15;
        input_fraction_bits = 23;
        active_residual = residual_i;
      end
      default: range_fault_o = enable_i;
    endcase

    if (enable_i) begin
      active_k = round_shift_uq1_15(k_uq1_15_i, coefficient_shift);
      active_b = round_shift_uq1_15(b_uq1_15_i, coefficient_shift);
    end else begin
      active_residual = 17'b0;
    end

    // Explicit extensions preserve the full 16x17 product. In narrow modes
    // the unused high inputs are constant zero and removable by synthesis.
    product = {17'b0, active_k} * {16'b0, active_residual};
    b_aligned = {23'b0, active_b} << input_fraction_bits;
    b_extended = $signed({1'b0, b_aligned});
    product_extended = $signed({7'b0, product});
    difference = b_extended - product_extended;

    retained = 40'b0;
    remainder = 40'b0;
    remainder_mask = 40'b0;
    halfway = 40'b0;
    if (difference < 0) begin
      range_fault_o = enable_i;
    end else if (enable_i) begin
      retained = $unsigned(difference) >> input_fraction_bits;
      remainder_mask = (40'b1 << input_fraction_bits) - 1'b1;
      remainder = $unsigned(difference) & remainder_mask;
      halfway = 40'b1 << (input_fraction_bits-1);
      if ((remainder > halfway) ||
          ((remainder == halfway) && retained[0]))
        retained = retained + 1'b1;
    end

    y0_fp32 = fixed_to_fp32(retained, coefficient_fraction_bits);
    y0_o = y0_fp32;
    if (data_format_i == VXM_FORMAT_FP16)
      y0_o = {16'b0, fp32_to_fp16(y0_fp32)};
    if (!enable_i || range_fault_o)
      y0_o = 32'b0;
  end
endmodule
