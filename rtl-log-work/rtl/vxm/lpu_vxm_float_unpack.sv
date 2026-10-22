// Common combinational floating-point format unpacker.
//
// Arithmetic consumers see subnormal operands as signed zero (DAZ).  The raw
// input remains outside this module so bit-preserving operations such as
// BYPASS and NEGATE do not inherit the arithmetic DAZ/NaN policy.
module lpu_vxm_float_unpack (
  input  logic [1:0]  data_format_i,
  input  logic [31:0] value_i,

  output logic        format_valid_o,
  output logic        sign_o,
  output logic [7:0]  exponent_o,
  output logic [22:0] fraction_o,
  output logic        effective_zero_o,
  output logic        was_subnormal_o,
  output logic        is_normal_o,
  output logic        is_inf_o,
  output logic        is_nan_o
);
  import lpu_pkg::*;

  logic [7:0]  raw_exponent;
  logic [22:0] raw_fraction;
  logic        use_exp_high;
  logic        use_frac_mid;
  logic        use_frac_high;
  logic        exponent_zero;
  logic        exponent_ones;
  logic        fraction_zero;

  always_comb begin
    format_valid_o = 1'b0;
    sign_o = 1'b0;
    raw_exponent = 8'b0;
    raw_fraction = 23'b0;
    use_exp_high = 1'b0;
    use_frac_mid = 1'b0;
    use_frac_high = 1'b0;

    case (data_format_i)
      VXM_FORMAT_FP16: begin
        format_valid_o = 1'b1;
        sign_o = value_i[15];
        raw_exponent = {3'b000, value_i[14:10]};
        raw_fraction = {13'b0, value_i[9:0]};
        use_frac_mid = 1'b1;
      end

      VXM_FORMAT_BF16: begin
        format_valid_o = 1'b1;
        sign_o = value_i[15];
        raw_exponent = value_i[14:7];
        raw_fraction = {16'b0, value_i[6:0]};
        use_exp_high = 1'b1;
      end

      VXM_FORMAT_FP32: begin
        format_valid_o = 1'b1;
        sign_o = value_i[31];
        raw_exponent = value_i[30:23];
        raw_fraction = value_i[22:0];
        use_exp_high = 1'b1;
        use_frac_mid = 1'b1;
        use_frac_high = 1'b1;
      end

      default: begin end
    endcase

    // Reuse the low-bit reductions for every format. Only formats with
    // wider fields include the additional groups in classification.
    exponent_zero = (~|raw_exponent[4:0]) &&
      (!use_exp_high || ~|raw_exponent[7:5]);
    exponent_ones = (&raw_exponent[4:0]) &&
      (!use_exp_high || &raw_exponent[7:5]);
    fraction_zero = (~|raw_fraction[6:0]) &&
      (!use_frac_mid || ~|raw_fraction[9:7]) &&
      (!use_frac_high || ~|raw_fraction[22:10]);

    effective_zero_o = format_valid_o && exponent_zero;
    was_subnormal_o = format_valid_o && exponent_zero && !fraction_zero;
    is_normal_o = format_valid_o && !exponent_zero && !exponent_ones;
    is_inf_o = format_valid_o && exponent_ones && fraction_zero;
    is_nan_o = format_valid_o && exponent_ones && !fraction_zero;

    // DAZ is applied only to the numerical fields exported to arithmetic
    // consumers. sign_o is deliberately retained for signed-zero semantics.
    exponent_o = raw_exponent;
    fraction_o = raw_fraction;
    if (effective_zero_o) begin
      exponent_o = 8'b0;
      fraction_o = 23'b0;
    end
  end
endmodule
