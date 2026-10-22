// Shared unsigned coefficient multiplier for LUT-based special functions.
//
// The default EXP instance is a physical 26x24 multiply. Parameters also
// permit narrower coefficient formats while retaining one physical datapath.
// Narrow formats round the stored coefficient to their selected internal
// precision, right-align the
// active value significand, and zero-fill every unused high bit:
//   BF16 : UQ1.9  (10 bits) x  8-bit significand
//   FP16 : UQ1.12 (13 bits) x 11-bit significand
//   FP32 : UQ1.25 (26 bits) x 24-bit significand
// This deliberately leaves the multiplier architecture to synthesis/DC.
module lpu_vxm_special_coefficient_multiplier #(
  parameter integer COEFFICIENT_WIDTH = 26,
  parameter integer STORED_FRACTION_BITS = 25,
  parameter integer BF16_FRACTION_BITS = 9,
  parameter integer FP16_FRACTION_BITS = 12,
  parameter integer FP32_FRACTION_BITS = 25
) (
  input  logic        enable_i,
  input  logic [1:0]  data_format_i,
  input  logic [25:0] coefficient_i,
  input  logic [25:0] offset_i,
  // FP16 is carried in [15:0]. BF16 and FP32 are carried as FP32 values.
  input  logic [31:0] value_i,
  output logic [31:0] coefficient_value_o,
  output logic [31:0] offset_value_o,
  output logic [31:0] product_o
);
  import lpu_pkg::*;
  import lpu_vxm_math_pkg::*;

  function automatic logic [25:0] round_shift_uq1_25(
    input logic [25:0] value,
    input integer shift
  );
    logic [25:0] retained;
    logic [25:0] discarded;
    logic [25:0] mask;
    logic [25:0] halfway;
    begin
      if (shift == 0) begin
        round_shift_uq1_25 = value;
      end else begin
        retained = value >> shift;
        mask = (26'b1 << shift) - 1'b1;
        discarded = value & mask;
        halfway = 26'b1 << (shift-1);
        if ((discarded > halfway) ||
            ((discarded == halfway) && retained[0]))
          retained = retained + 1'b1;
        round_shift_uq1_25 = retained;
      end
    end
  endfunction

  // Convert magnitude*2^binary_exponent to FP32 with one RNE operation.
  function automatic logic [31:0] scaled_unsigned_to_fp32(
    input logic [49:0] magnitude,
    input integer binary_exponent,
    input logic sign
  );
    integer leading_bit;
    integer shift;
    integer exponent;
    logic [49:0] retained;
    logic [49:0] discarded;
    logic [49:0] mask;
    logic [49:0] halfway;
    begin
      leading_bit = -1;
      retained = 50'b0;
      discarded = 50'b0;
      mask = 50'b0;
      halfway = 50'b0;
      for (integer bit_index = 0; bit_index < 50; bit_index++)
        if (magnitude[bit_index])
          leading_bit = bit_index;

      if (leading_bit < 0) begin
        scaled_unsigned_to_fp32 = {sign, 31'b0};
      end else begin
        exponent = 127 + binary_exponent + leading_bit;
        if (leading_bit > 23) begin
          shift = leading_bit - 23;
          retained = magnitude >> shift;
          mask = (50'b1 << shift) - 1'b1;
          discarded = magnitude & mask;
          halfway = 50'b1 << (shift-1);
          if ((discarded > halfway) ||
              ((discarded == halfway) && retained[0]))
            retained = retained + 1'b1;
          if (retained[24]) begin
            retained = retained >> 1;
            exponent = exponent + 1;
          end
        end else begin
          retained = magnitude << (23-leading_bit);
        end

        if (exponent <= 0)
          scaled_unsigned_to_fp32 = {sign, 31'b0};
        else if (exponent >= 255)
          scaled_unsigned_to_fp32 = {sign, 8'hff, 23'b0};
        else
          scaled_unsigned_to_fp32 =
            {sign, exponent[7:0], retained[22:0]};
      end
    end
  endfunction

  logic [25:0] active_coefficient;
  logic [25:0] active_offset;
  logic [COEFFICIENT_WIDTH-1:0] multiplier_coefficient;
  logic [23:0] active_significand;
  logic [COEFFICIENT_WIDTH+24-1:0] narrow_unsigned_product;
  logic [49:0] unsigned_product;
  logic [15:0] value_bf16;
  logic value_sign;
  integer coefficient_fraction_bits;
  integer value_fraction_bits;
  integer value_unbiased_exponent;
  integer product_binary_exponent;

  always_comb begin
    active_coefficient = 26'b0;
    active_offset = 26'b0;
    active_significand = 24'b0;
    value_bf16 = 16'b0;
    value_sign = 1'b0;
    coefficient_fraction_bits = 0;
    value_fraction_bits = 0;
    value_unbiased_exponent = 0;

    case (data_format_i)
      VXM_FORMAT_BF16: begin
        active_coefficient = round_shift_uq1_25(
          coefficient_i,
          STORED_FRACTION_BITS-BF16_FRACTION_BITS);
        active_offset = round_shift_uq1_25(
          offset_i,
          STORED_FRACTION_BITS-BF16_FRACTION_BITS);
        coefficient_fraction_bits = BF16_FRACTION_BITS;
        value_bf16 = fp32_to_bf16_ftz(value_i);
        value_sign = value_bf16[15];
        value_fraction_bits = 7;
        if (value_bf16[14:7] != 0) begin
          active_significand[7:0] = {1'b1, value_bf16[6:0]};
          value_unbiased_exponent = value_bf16[14:7] - 127;
        end
      end
      VXM_FORMAT_FP16: begin
        active_coefficient = round_shift_uq1_25(
          coefficient_i,
          STORED_FRACTION_BITS-FP16_FRACTION_BITS);
        active_offset = round_shift_uq1_25(
          offset_i,
          STORED_FRACTION_BITS-FP16_FRACTION_BITS);
        coefficient_fraction_bits = FP16_FRACTION_BITS;
        value_sign = value_i[15];
        value_fraction_bits = 10;
        if (value_i[14:10] != 0) begin
          active_significand[10:0] = {1'b1, value_i[9:0]};
          value_unbiased_exponent = value_i[14:10] - 15;
        end
      end
      VXM_FORMAT_FP32: begin
        active_coefficient = coefficient_i;
        active_offset = offset_i;
        coefficient_fraction_bits = FP32_FRACTION_BITS;
        value_sign = value_i[31];
        value_fraction_bits = 23;
        if (value_i[30:23] != 0) begin
          active_significand = {1'b1, value_i[22:0]};
          value_unbiased_exponent = value_i[30:23] - 127;
        end
      end
      default: begin end
    endcase

    if (!enable_i) begin
      active_coefficient = 26'b0;
      active_offset = 26'b0;
      active_significand = 24'b0;
    end
    // This is the only physical multiply described by this module. DC may
    // choose its implementation after the format MUXes and zero extension.
    multiplier_coefficient =
      active_coefficient[COEFFICIENT_WIDTH-1:0];
    // Explicit context extension preserves the full W+24 product in both
    // simulation and synthesis; the constant-zero halves are removable.
    narrow_unsigned_product =
      {{24{1'b0}}, multiplier_coefficient} *
      {{COEFFICIENT_WIDTH{1'b0}}, active_significand};
    unsigned_product = {{(50-(COEFFICIENT_WIDTH+24)){1'b0}},
                        narrow_unsigned_product};
    product_binary_exponent = value_unbiased_exponent -
      value_fraction_bits - coefficient_fraction_bits;

    coefficient_value_o = scaled_unsigned_to_fp32(
      {{24{1'b0}}, active_coefficient},
      -coefficient_fraction_bits, 1'b0);
    offset_value_o = scaled_unsigned_to_fp32(
      {{24{1'b0}}, active_offset},
      -coefficient_fraction_bits, 1'b0);
    product_o = scaled_unsigned_to_fp32(
      unsigned_product, product_binary_exponent, value_sign);

    if (data_format_i == VXM_FORMAT_FP16) begin
      coefficient_value_o = {16'b0, fp32_to_fp16(coefficient_value_o)};
      offset_value_o = {16'b0, fp32_to_fp16(offset_value_o)};
      product_o = {16'b0, fp32_to_fp16(product_o)};
    end
  end

  initial begin
    if ((COEFFICIENT_WIDTH < 1) || (COEFFICIENT_WIDTH > 26))
      $error("special coefficient width must be in [1,26]");
    if ((BF16_FRACTION_BITS > STORED_FRACTION_BITS) ||
        (FP16_FRACTION_BITS > STORED_FRACTION_BITS) ||
        (FP32_FRACTION_BITS != STORED_FRACTION_BITS))
      $error("invalid special coefficient fractional precision");
  end
endmodule
