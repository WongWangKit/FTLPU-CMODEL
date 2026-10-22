module lpu_vxm_shared_float_multiplier (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        enable_i,
  input  logic [1:0]  data_format_i,
  input  logic        lhs_sign_i,
  input  logic [7:0]  lhs_exponent_i,
  input  logic [22:0] lhs_fraction_i,
  input  logic        rhs_sign_i,
  input  logic [7:0]  rhs_exponent_i,
  input  logic [22:0] rhs_fraction_i,
  input  logic        special_nan_i,
  input  logic        special_inf_i,
  input  logic        special_zero_i,
  output logic        valid_o,
  output logic [31:0] result_o,
  output logic [8:0]  active_blocks_o
);
  import lpu_pkg::*;

  localparam logic [15:0] FP16_CANONICAL_NAN = 16'h7e00;
  localparam logic [15:0] BF16_CANONICAL_NAN = 16'h7fc0;
  localparam logic [31:0] FP32_CANONICAL_NAN = 32'h7fc00000;

  // Stage 1 combinational values: unpack fields enter the significand array
  // and exponent candidate logic in parallel.
  logic [23:0] stage_lhs_significand;
  logic [23:0] stage_rhs_significand;
  logic [47:0] stage_significand_product;
  logic        stage_format_valid;
  logic        stage_finite_multiply;
  logic        stage_result_sign;
  logic        stage_special_nan;
  logic        stage_special_inf;
  logic        stage_special_zero;
  logic signed [9:0] stage_exponent_lhs;
  logic signed [9:0] stage_exponent_rhs;
  logic signed [9:0] stage_exponent_bias;
  logic signed [9:0] stage_exponent_e0;
  logic signed [9:0] stage_exponent_e1;

  // Stage 1 registers.  The complete product is stored rather than the two
  // carry-save operands, minimizing the per-ALU register count.
  logic        product_valid_q;
  logic [1:0]  product_format_q;
  logic [47:0] product_q;
  logic        product_sign_q;
  logic        product_special_nan_q;
  logic        product_special_inf_q;
  logic        product_special_zero_q;
  logic signed [9:0] product_exponent_e0_q;
  logic signed [9:0] product_exponent_e1_q;

  // Stage 2 combinational values: normalization, RNE, exponent selection and
  // format-native result packing.
  logic [23:0] retained_significand;
  logic [24:0] rounded_significand;
  logic        normalize_shift;
  logic        guard_bit;
  logic        round_bit;
  logic        sticky_bit;
  logic        round_increment;
  logic        rounding_carry;
  logic signed [9:0] selected_exponent;
  logic signed [9:0] exponent_limit;

  always_comb begin
    stage_format_valid = 1'b1;
    stage_lhs_significand = 24'b0;
    stage_rhs_significand = 24'b0;
    stage_exponent_lhs = 10'sd0;
    stage_exponent_rhs = 10'sd0;
    stage_exponent_bias = 10'sd0;

    // Encoded exponents are unsigned and are zero-extended before the shared
    // signed E0 calculation.  Significands are left-aligned for the 3x3 array.
    case (data_format_i)
      VXM_FORMAT_FP16: begin
        stage_lhs_significand = {
          1'b1, lhs_fraction_i[9:0], 13'b0};
        stage_rhs_significand = {
          1'b1, rhs_fraction_i[9:0], 13'b0};
        stage_exponent_lhs = $signed({5'b0, lhs_exponent_i[4:0]});
        stage_exponent_rhs = $signed({5'b0, rhs_exponent_i[4:0]});
        stage_exponent_bias = 10'sd15;
      end
      VXM_FORMAT_BF16: begin
        stage_lhs_significand = {
          1'b1, lhs_fraction_i[6:0], 16'b0};
        stage_rhs_significand = {
          1'b1, rhs_fraction_i[6:0], 16'b0};
        stage_exponent_lhs = $signed({2'b0, lhs_exponent_i});
        stage_exponent_rhs = $signed({2'b0, rhs_exponent_i});
        stage_exponent_bias = 10'sd127;
      end
      VXM_FORMAT_FP32: begin
        stage_lhs_significand = {1'b1, lhs_fraction_i};
        stage_rhs_significand = {1'b1, rhs_fraction_i};
        stage_exponent_lhs = $signed({2'b0, lhs_exponent_i});
        stage_exponent_rhs = $signed({2'b0, rhs_exponent_i});
        stage_exponent_bias = 10'sd127;
      end
      default: begin
        stage_format_valid = 1'b0;
      end
    endcase

    stage_result_sign = lhs_sign_i ^ rhs_sign_i;
    stage_special_nan = special_nan_i;
    stage_special_inf = special_inf_i;
    stage_special_zero = special_zero_i;
    stage_finite_multiply = enable_i && stage_format_valid &&
      !stage_special_nan && !stage_special_inf && !stage_special_zero;

    // E0 and E1 settle in parallel with the longer significand path.
    stage_exponent_e0 = stage_exponent_lhs + stage_exponent_rhs -
      stage_exponent_bias;
    stage_exponent_e1 = stage_exponent_e0 + 10'sd1;
  end

  lpu_vxm_significand_multiplier u_significand_multiplier (
    .enable_i(stage_finite_multiply),
    .data_format_i,
    .lhs_significand_i(stage_lhs_significand),
    .rhs_significand_i(stage_rhs_significand),
    .product_o(stage_significand_product),
    .active_blocks_o
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      product_valid_q <= 1'b0;
      product_format_q <= VXM_FORMAT_FP16;
      product_q <= 48'b0;
      product_sign_q <= 1'b0;
      product_special_nan_q <= 1'b0;
      product_special_inf_q <= 1'b0;
      product_special_zero_q <= 1'b0;
      product_exponent_e0_q <= 10'sd0;
      product_exponent_e1_q <= 10'sd0;
    end else begin
      product_valid_q <= enable_i && stage_format_valid;
      if (enable_i && stage_format_valid) begin
        product_format_q <= data_format_i;
        product_q <= stage_significand_product;
        product_sign_q <= stage_result_sign;
        product_special_nan_q <= stage_special_nan;
        product_special_inf_q <= stage_special_inf;
        product_special_zero_q <= stage_special_zero;
        product_exponent_e0_q <= stage_exponent_e0;
        product_exponent_e1_q <= stage_exponent_e1;
      end
    end
  end

  always_comb begin
    retained_significand = 24'b0;
    exponent_limit = 10'sd0;
    guard_bit = 1'b0;
    round_bit = 1'b0;
    sticky_bit = 1'b0;
    normalize_shift = product_q[47];

    case (product_format_q)
      VXM_FORMAT_FP16: begin
        exponent_limit = 10'sd31;
        if (normalize_shift) begin
          retained_significand[10:0] = product_q[47:37];
          guard_bit = product_q[36];
          round_bit = product_q[35];
          sticky_bit = |product_q[34:26];
        end else begin
          retained_significand[10:0] = product_q[46:36];
          guard_bit = product_q[35];
          round_bit = product_q[34];
          sticky_bit = |product_q[33:26];
        end
      end
      VXM_FORMAT_BF16: begin
        exponent_limit = 10'sd255;
        if (normalize_shift) begin
          retained_significand[7:0] = product_q[47:40];
          guard_bit = product_q[39];
          round_bit = product_q[38];
          sticky_bit = |product_q[37:32];
        end else begin
          retained_significand[7:0] = product_q[46:39];
          guard_bit = product_q[38];
          round_bit = product_q[37];
          sticky_bit = |product_q[36:32];
        end
      end
      VXM_FORMAT_FP32: begin
        exponent_limit = 10'sd255;
        if (normalize_shift) begin
          retained_significand = product_q[47:24];
          guard_bit = product_q[23];
          round_bit = product_q[22];
          sticky_bit = |product_q[21:0];
        end else begin
          retained_significand = product_q[46:23];
          guard_bit = product_q[22];
          round_bit = product_q[21];
          sticky_bit = |product_q[20:0];
        end
      end
      default: begin end
    endcase

    round_increment = guard_bit &&
      (round_bit || sticky_bit || retained_significand[0]);
    rounded_significand = {1'b0, retained_significand} + round_increment;
    case (product_format_q)
      VXM_FORMAT_FP16: rounding_carry = rounded_significand[11];
      VXM_FORMAT_BF16: rounding_carry = rounded_significand[8];
      VXM_FORMAT_FP32: rounding_carry = rounded_significand[24];
      default:         rounding_carry = 1'b0;
    endcase
    selected_exponent = (normalize_shift || rounding_carry) ?
      product_exponent_e1_q : product_exponent_e0_q;

    valid_o = product_valid_q;
    result_o = 32'b0;
    if (product_valid_q) begin
      if (product_special_nan_q) begin
        case (product_format_q)
          VXM_FORMAT_FP16: result_o[15:0] = FP16_CANONICAL_NAN;
          VXM_FORMAT_BF16: result_o[15:0] = BF16_CANONICAL_NAN;
          VXM_FORMAT_FP32: result_o = FP32_CANONICAL_NAN;
          default: begin end
        endcase
      end else if (product_special_inf_q) begin
        case (product_format_q)
          VXM_FORMAT_FP16:
            result_o[15:0] = {product_sign_q, 5'h1f, 10'b0};
          VXM_FORMAT_BF16:
            result_o[15:0] = {product_sign_q, 8'hff, 7'b0};
          VXM_FORMAT_FP32:
            result_o = {product_sign_q, 8'hff, 23'b0};
          default: begin end
        endcase
      end else if (product_special_zero_q ||
                   (selected_exponent <= 10'sd0)) begin
        case (product_format_q)
          VXM_FORMAT_FP16,
          VXM_FORMAT_BF16: result_o[15:0] = {product_sign_q, 15'b0};
          VXM_FORMAT_FP32: result_o = {product_sign_q, 31'b0};
          default: begin end
        endcase
      end else if (selected_exponent >= exponent_limit) begin
        case (product_format_q)
          VXM_FORMAT_FP16:
            result_o[15:0] = {product_sign_q, 5'h1f, 10'b0};
          VXM_FORMAT_BF16:
            result_o[15:0] = {product_sign_q, 8'hff, 7'b0};
          VXM_FORMAT_FP32:
            result_o = {product_sign_q, 8'hff, 23'b0};
          default: begin end
        endcase
      end else begin
        case (product_format_q)
          VXM_FORMAT_FP16: result_o[15:0] = {
            product_sign_q, selected_exponent[4:0],
            rounded_significand[9:0]};
          VXM_FORMAT_BF16: result_o[15:0] = {
            product_sign_q, selected_exponent[7:0],
            rounded_significand[6:0]};
          VXM_FORMAT_FP32: result_o = {
            product_sign_q, selected_exponent[7:0],
            rounded_significand[22:0]};
          default: begin end
        endcase
      end
    end
  end
endmodule
