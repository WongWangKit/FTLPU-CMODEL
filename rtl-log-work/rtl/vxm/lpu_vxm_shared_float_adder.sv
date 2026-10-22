// One format-selected FP16/BF16/FP32 add/subtract cone.
//
// The format MUX and operand isolation are deliberately before the shared
// significand datapath. Near subtraction uses one grouped leading-zero
// encoder and one format-masked variable left shifter.
module lpu_vxm_shared_float_adder (
  input  logic        enable_i,
  input  logic [1:0]  data_format_i,
  input  logic        subtract_i,
  // The common Basic-ALU unpackers own format classification and DAZ.
  input  logic        lhs_sign_i,
  input  logic [7:0]  lhs_exponent_i,
  input  logic [22:0] lhs_fraction_i,
  input  logic        lhs_zero_i,
  input  logic        lhs_inf_i,
  input  logic        lhs_nan_i,
  input  logic        rhs_sign_i,
  input  logic [7:0]  rhs_exponent_i,
  input  logic [22:0] rhs_fraction_i,
  input  logic        rhs_zero_i,
  input  logic        rhs_inf_i,
  input  logic        rhs_nan_i,
  input  logic        lhs_magnitude_ge_i,
  output logic [31:0] result_o
);
  import lpu_pkg::*;

  localparam logic [15:0] FP16_CANONICAL_NAN = 16'h7e00;
  localparam logic [15:0] BF16_CANONICAL_NAN = 16'h7fc0;
  localparam logic [31:0] FP32_CANONICAL_NAN = 32'h7fc00000;

  logic lhs_sign;
  logic rhs_sign;
  logic rhs_effective_sign;
  logic lhs_zero;
  logic rhs_zero;
  logic lhs_inf;
  logic rhs_inf;
  logic lhs_nan;
  logic rhs_nan;
  logic signed [6:0] fp16_lhs_exponent;
  logic signed [6:0] fp16_rhs_exponent;
  logic signed [9:0] wide_lhs_exponent;
  logic signed [9:0] wide_rhs_exponent;
  logic [7:0] lhs_alignment_exponent;
  logic [7:0] rhs_alignment_exponent;
  logic [7:0] big_alignment_exponent;
  logic [7:0] small_alignment_exponent;
  logic [22:0] lhs_fraction;
  logic [22:0] rhs_fraction;
  logic [5:0] significand_width;
  logic [26:0] active_mask;
  logic high_group_enable;
  logic [31:0] lhs_effective;
  logic [31:0] rhs_effective;
  logic [26:0] significand_lhs;
  logic [26:0] significand_rhs;
  logic [26:0] significand_big;
  logic [26:0] significand_small;
  logic [26:0] aligned_small;
  logic [27:0] arithmetic_result;
  logic sign_big;
  logic sign_small;
  logic magnitude_subtract;
  logic normal_arithmetic_enable;
  logic [7:0] exponent_distance;
  logic near_subtract;
  logic [4:0] near_leading_zeros;
  logic near_result_zero;
  logic [4:0] subtract_shift;
  logic [26:0] subtract_normalized;

  logic signed [6:0] fp16_exponent_work;
  logic signed [9:0] wide_exponent_work;
  logic [13:0] fp16_normalized;
  logic [10:0] bf16_normalized;
  logic [26:0] fp32_normalized;
  logic [11:0] fp16_rounded;
  logic [8:0] bf16_rounded;
  logic [24:0] fp32_rounded;
  logic [31:0] normal_result;
  logic [31:0] special_result;
  logic special_result_valid;

  function automatic logic [26:0] shift_right_sticky(
    input logic [26:0] value,
    input logic [7:0] distance,
    input logic [5:0] width
  );
    logic [26:0] shifted;
    logic sticky;
    begin
      shifted = 27'b0;
      sticky = 1'b0;
      for (integer bit_index = 0; bit_index < 27; bit_index++)
        if ((bit_index < width) && (bit_index < distance))
          sticky = sticky | value[bit_index];
      if (distance < width)
        shifted = value >> distance;
      shifted[0] = shifted[0] | sticky;
      shift_right_sticky = shifted;
    end
  endfunction

  // The single shared 14+13 significand add/subtract datapath. The active
  // mask is 11 bits for BF16, 14 for FP16 and 27 for FP32.
  lpu_vxm_segmented_addsub u_significand_addsub (
    .enable_i(normal_arithmetic_enable),
    .subtract_i(magnitude_subtract),
    .high_enable_i(high_group_enable),
    .active_mask_i(active_mask),
    .lhs_i(significand_big),
    .rhs_i(aligned_small),
    .result_o(arithmetic_result)
  );

  // Only close-magnitude subtraction can need more than one left shift.
  assign near_subtract = normal_arithmetic_enable && magnitude_subtract &&
    (exponent_distance <= 8'd1);
  lpu_vxm_grouped_lzc u_near_lzc (
    .enable_i(near_subtract),
    .active_width_i(significand_width),
    .value_i(arithmetic_result[26:0]),
    .shift_o(near_leading_zeros),
    .zero_o(near_result_zero)
  );

  // The same 27-bit shift network serves all three formats. Unused high
  // bits are zeroed before the shift; only the active result bits are used.
  lpu_vxm_shared_left_shift u_subtract_left_shift (
    .enable_i(normal_arithmetic_enable && magnitude_subtract),
    .active_mask_i(active_mask),
    .value_i(arithmetic_result[26:0]),
    .shift_i(subtract_shift),
    .result_o(subtract_normalized)
  );

  // Classification is owned by the two common unpackers. Zero (including
  // DAZ), Inf and NaN bypass the normal significand datapath entirely.
  assign normal_arithmetic_enable = enable_i &&
    ((data_format_i == VXM_FORMAT_FP16) ||
     (data_format_i == VXM_FORMAT_BF16) ||
     (data_format_i == VXM_FORMAT_FP32)) &&
    !lhs_nan_i && !rhs_nan_i && !lhs_inf_i && !rhs_inf_i &&
    !lhs_zero_i && !rhs_zero_i;

  always_comb begin
    // Front format MUX. Defaults also isolate the arithmetic cone when idle.
    lhs_sign = 1'b0;
    rhs_sign = 1'b0;
    fp16_lhs_exponent = 7'sd0;
    fp16_rhs_exponent = 7'sd0;
    wide_lhs_exponent = 10'sd0;
    wide_rhs_exponent = 10'sd0;
    lhs_alignment_exponent = 8'b0;
    rhs_alignment_exponent = 8'b0;
    lhs_fraction = 23'b0;
    rhs_fraction = 23'b0;
    significand_width = 6'd0;
    active_mask = 27'b0;
    high_group_enable = 1'b0;
    lhs_effective = 32'b0;
    rhs_effective = 32'b0;
    lhs_zero = 1'b1;
    rhs_zero = 1'b1;
    lhs_inf = 1'b0;
    rhs_inf = 1'b0;
    lhs_nan = 1'b0;
    rhs_nan = 1'b0;

    if (enable_i) begin
      lhs_sign = lhs_sign_i;
      rhs_sign = rhs_sign_i;
      lhs_zero = lhs_zero_i;
      rhs_zero = rhs_zero_i;
      lhs_inf = lhs_inf_i;
      rhs_inf = rhs_inf_i;
      lhs_nan = lhs_nan_i;
      rhs_nan = rhs_nan_i;
      case (data_format_i)
        VXM_FORMAT_FP16: begin
          lhs_effective[15:0] = {
            lhs_sign_i, lhs_exponent_i[4:0], lhs_fraction_i[9:0]};
          rhs_effective[15:0] = {
            rhs_sign_i, rhs_exponent_i[4:0], rhs_fraction_i[9:0]};
        end
        VXM_FORMAT_BF16: begin
          lhs_effective[15:0] = {
            lhs_sign_i, lhs_exponent_i, lhs_fraction_i[6:0]};
          rhs_effective[15:0] = {
            rhs_sign_i, rhs_exponent_i, rhs_fraction_i[6:0]};
        end
        VXM_FORMAT_FP32: begin
          lhs_effective = {lhs_sign_i, lhs_exponent_i, lhs_fraction_i};
          rhs_effective = {rhs_sign_i, rhs_exponent_i, rhs_fraction_i};
        end
        default: begin end
      endcase
    end

    if (normal_arithmetic_enable) begin
      lhs_fraction = lhs_fraction_i;
      rhs_fraction = rhs_fraction_i;
      case (data_format_i)
        VXM_FORMAT_FP16: begin
          fp16_lhs_exponent = $signed({2'b0, lhs_exponent_i[4:0]});
          fp16_rhs_exponent = $signed({2'b0, rhs_exponent_i[4:0]});
          lhs_alignment_exponent = {3'b0, lhs_exponent_i[4:0]};
          rhs_alignment_exponent = {3'b0, rhs_exponent_i[4:0]};
          significand_width = 6'd14;
          active_mask = 27'h0003fff;
        end
        VXM_FORMAT_BF16: begin
          wide_lhs_exponent = $signed({2'b0, lhs_exponent_i});
          wide_rhs_exponent = $signed({2'b0, rhs_exponent_i});
          lhs_alignment_exponent = lhs_exponent_i;
          rhs_alignment_exponent = rhs_exponent_i;
          significand_width = 6'd11;
          active_mask = 27'h00007ff;
        end
        VXM_FORMAT_FP32: begin
          wide_lhs_exponent = $signed({2'b0, lhs_exponent_i});
          wide_rhs_exponent = $signed({2'b0, rhs_exponent_i});
          lhs_alignment_exponent = lhs_exponent_i;
          rhs_alignment_exponent = rhs_exponent_i;
          significand_width = 6'd27;
          active_mask = 27'h7ffffff;
          high_group_enable = 1'b1;
        end
        default: begin end
      endcase
    end

    rhs_effective_sign = rhs_sign ^ subtract_i;
    if (data_format_i == VXM_FORMAT_FP32)
      rhs_effective[31] = rhs_effective_sign;
    else
      rhs_effective[15] = rhs_effective_sign;

    significand_lhs = 27'b0;
    significand_rhs = 27'b0;
    if (normal_arithmetic_enable)
      case (data_format_i)
        VXM_FORMAT_FP16: begin
          significand_lhs[13:0] = {1'b1, lhs_fraction[9:0], 3'b0};
          significand_rhs[13:0] = {1'b1, rhs_fraction[9:0], 3'b0};
        end
        VXM_FORMAT_BF16: begin
          significand_lhs[10:0] = {1'b1, lhs_fraction[6:0], 3'b0};
          significand_rhs[10:0] = {1'b1, rhs_fraction[6:0], 3'b0};
        end
        VXM_FORMAT_FP32: begin
          significand_lhs[26:0] = {1'b1, lhs_fraction, 3'b0};
          significand_rhs[26:0] = {1'b1, rhs_fraction, 3'b0};
        end
        default: begin end
      endcase

    if (lhs_magnitude_ge_i) begin
      significand_big = significand_lhs;
      significand_small = significand_rhs;
      sign_big = lhs_sign;
      sign_small = rhs_effective_sign;
      big_alignment_exponent = lhs_alignment_exponent;
      small_alignment_exponent = rhs_alignment_exponent;
    end else begin
      significand_big = significand_rhs;
      significand_small = significand_lhs;
      sign_big = rhs_effective_sign;
      sign_small = lhs_sign;
      big_alignment_exponent = rhs_alignment_exponent;
      small_alignment_exponent = lhs_alignment_exponent;
    end
    // One unsigned subtractor for alignment. FP16 zero-extends its five-bit
    // exponent; the magnitude comparator guarantees a nonnegative distance.
    exponent_distance = big_alignment_exponent - small_alignment_exponent;
    aligned_small = shift_right_sticky(
      significand_small, exponent_distance, significand_width);
    magnitude_subtract = sign_big != sign_small;

    subtract_shift = 5'd0;
    if (magnitude_subtract) begin
      if (near_subtract) begin
        if (!near_result_zero)
          subtract_shift = near_leading_zeros;
      end else begin
        // At an exponent distance of two or more, the difference cannot
        // lose more than one leading bit of a normalized significand.
        case (data_format_i)
          VXM_FORMAT_FP16: subtract_shift = {4'b0, !arithmetic_result[13]};
          VXM_FORMAT_BF16: subtract_shift = {4'b0, !arithmetic_result[10]};
          VXM_FORMAT_FP32: subtract_shift = {4'b0, !arithmetic_result[26]};
          default: begin end
        endcase
      end
    end

    normal_result = 32'b0;
    fp16_exponent_work = lhs_magnitude_ge_i ?
      fp16_lhs_exponent : fp16_rhs_exponent;
    wide_exponent_work = lhs_magnitude_ge_i ?
      wide_lhs_exponent : wide_rhs_exponent;
    fp16_normalized = arithmetic_result[13:0];
    bf16_normalized = arithmetic_result[10:0];
    fp32_normalized = arithmetic_result[26:0];
    fp16_rounded = 12'b0;
    bf16_rounded = 9'b0;
    fp32_rounded = 25'b0;

    case (data_format_i)
      VXM_FORMAT_FP16: begin
        if (!magnitude_subtract && arithmetic_result[14]) begin
          fp16_normalized = arithmetic_result[14:1];
          fp16_normalized[0] = fp16_normalized[0] | arithmetic_result[0];
          fp16_exponent_work = fp16_exponent_work + 7'sd1;
        end else if (magnitude_subtract) begin
          fp16_normalized = subtract_normalized[13:0];
          fp16_exponent_work = fp16_exponent_work -
            $signed({2'b0, subtract_shift});
        end
        // FTZ keeps the arithmetic sign for a nonzero value that underflows;
        // exact cancellation remains canonical +0.
        if (fp16_normalized != 0)
          normal_result[15:0] = {sign_big, 15'b0};
        if ((fp16_normalized != 0) && (fp16_exponent_work > 0)) begin
          fp16_rounded = {1'b0, fp16_normalized[13:3]};
          if (fp16_normalized[2] &&
              (fp16_normalized[1] || fp16_normalized[0] ||
               fp16_normalized[3]))
            fp16_rounded = fp16_rounded + 12'd1;
          if (fp16_rounded[11]) begin
            fp16_rounded = fp16_rounded >> 1;
            fp16_exponent_work = fp16_exponent_work + 7'sd1;
          end
          if (fp16_exponent_work >= $signed(7'd31))
            normal_result[15:0] = {sign_big, 5'h1f, 10'b0};
          else
            normal_result[15:0] = {sign_big,
              fp16_exponent_work[4:0], fp16_rounded[9:0]};
        end
      end
      VXM_FORMAT_BF16: begin
        if (!magnitude_subtract && arithmetic_result[11]) begin
          bf16_normalized = arithmetic_result[11:1];
          bf16_normalized[0] = bf16_normalized[0] | arithmetic_result[0];
          wide_exponent_work = wide_exponent_work + 10'sd1;
        end else if (magnitude_subtract) begin
          bf16_normalized = subtract_normalized[10:0];
          wide_exponent_work = wide_exponent_work -
            $signed({5'b0, subtract_shift});
        end
        if (bf16_normalized != 0)
          normal_result[15:0] = {sign_big, 15'b0};
        if ((bf16_normalized != 0) && (wide_exponent_work > 0)) begin
          bf16_rounded = {1'b0, bf16_normalized[10:3]};
          if (bf16_normalized[2] &&
              (bf16_normalized[1] || bf16_normalized[0] ||
               bf16_normalized[3]))
            bf16_rounded = bf16_rounded + 9'd1;
          if (bf16_rounded[8]) begin
            bf16_rounded = bf16_rounded >> 1;
            wide_exponent_work = wide_exponent_work + 10'sd1;
          end
          if (wide_exponent_work >= $signed(10'd255))
            normal_result[15:0] = {sign_big, 8'hff, 7'b0};
          else
            normal_result[15:0] = {sign_big,
              wide_exponent_work[7:0], bf16_rounded[6:0]};
        end
      end
      VXM_FORMAT_FP32: begin
        if (!magnitude_subtract && arithmetic_result[27]) begin
          fp32_normalized = arithmetic_result[27:1];
          fp32_normalized[0] = fp32_normalized[0] | arithmetic_result[0];
          wide_exponent_work = wide_exponent_work + 10'sd1;
        end else if (magnitude_subtract) begin
          fp32_normalized = subtract_normalized[26:0];
          wide_exponent_work = wide_exponent_work -
            $signed({5'b0, subtract_shift});
        end
        if (fp32_normalized != 0)
          normal_result = {sign_big, 31'b0};
        if ((fp32_normalized != 0) && (wide_exponent_work > 0)) begin
          fp32_rounded = {1'b0, fp32_normalized[26:3]};
          if (fp32_normalized[2] &&
              (fp32_normalized[1] || fp32_normalized[0] ||
               fp32_normalized[3]))
            fp32_rounded = fp32_rounded + 25'd1;
          if (fp32_rounded[24]) begin
            fp32_rounded = fp32_rounded >> 1;
            wide_exponent_work = wide_exponent_work + 10'sd1;
          end
          if (wide_exponent_work >= $signed(10'd255))
            normal_result = {sign_big, 8'hff, 23'b0};
          else
            normal_result = {sign_big,
              wide_exponent_work[7:0], fp32_rounded[22:0]};
        end
      end
      default: normal_result = 32'b0;
    endcase

    // Post-compute result MUX. Special cases bypass the normal result, while
    // the front isolation above prevents idle operands from toggling it.
    special_result = 32'b0;
    special_result_valid = 1'b1;
    if (lhs_nan || rhs_nan) begin
      case (data_format_i)
        VXM_FORMAT_FP16: special_result[15:0] = FP16_CANONICAL_NAN;
        VXM_FORMAT_BF16: special_result[15:0] = BF16_CANONICAL_NAN;
        default: special_result = FP32_CANONICAL_NAN;
      endcase
    end else if (lhs_inf && rhs_inf && (lhs_sign != rhs_effective_sign)) begin
      case (data_format_i)
        VXM_FORMAT_FP16: special_result[15:0] = FP16_CANONICAL_NAN;
        VXM_FORMAT_BF16: special_result[15:0] = BF16_CANONICAL_NAN;
        default: special_result = FP32_CANONICAL_NAN;
      endcase
    end else if (lhs_inf)
      special_result = lhs_effective;
    else if (rhs_inf)
      special_result = rhs_effective;
    else if (lhs_zero && rhs_zero) begin
      if (data_format_i == VXM_FORMAT_FP32)
        special_result = {(lhs_sign & rhs_effective_sign), 31'b0};
      else
        special_result[15:0] = {(lhs_sign & rhs_effective_sign), 15'b0};
    end else if (lhs_zero)
      special_result = rhs_effective;
    else if (rhs_zero)
      special_result = lhs_effective;
    else
      special_result_valid = 1'b0;

    result_o = !enable_i ? 32'b0 :
      (special_result_valid ? special_result : normal_result);
  end
endmodule
