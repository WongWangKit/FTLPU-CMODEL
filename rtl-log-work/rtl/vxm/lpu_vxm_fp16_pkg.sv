package lpu_vxm_fp16_pkg;
  localparam logic [15:0] FP16_CANONICAL_NAN = 16'h7e00;
  localparam logic [15:0] FP16_ONE           = 16'h3c00;

  function automatic logic fp16_is_nan(input logic [15:0] value);
    fp16_is_nan = (value[14:10] == 5'h1f) && (value[9:0] != 0);
  endfunction

  function automatic logic fp16_is_inf(input logic [15:0] value);
    fp16_is_inf = (value[14:10] == 5'h1f) && (value[9:0] == 0);
  endfunction

  function automatic logic fp16_is_zero_or_subnormal(
    input logic [15:0] value
  );
    fp16_is_zero_or_subnormal = value[14:10] == 0;
  endfunction

  // Match VxmDataFormat::round_fp16_ftz(): canonicalize NaNs and flush every
  // input/output subnormal to a signed zero.
  function automatic logic [15:0] fp16_sanitize_ftz(
    input logic [15:0] value
  );
    begin
      if (fp16_is_nan(value))
        fp16_sanitize_ftz = {value[15], FP16_CANONICAL_NAN[14:0]};
      else if (fp16_is_zero_or_subnormal(value))
        fp16_sanitize_ftz = {value[15], 15'b0};
      else
        fp16_sanitize_ftz = value;
    end
  endfunction

  function automatic logic [13:0] shift_right_sticky14(
    input logic [13:0] value,
    input integer distance
  );
    logic [13:0] shifted;
    logic sticky;
    begin
      sticky = 1'b0;
      for (integer bit_index = 0; bit_index < 14; bit_index++)
        if (bit_index < distance)
          sticky = sticky | value[bit_index];
      shifted = distance >= 14 ? 14'b0 : value >> distance;
      shifted[0] = shifted[0] | sticky;
      shift_right_sticky14 = shifted;
    end
  endfunction

  function automatic logic [15:0] fp16_round_normal_ftz(
    input logic sign,
    input integer exponent,
    input logic [13:0] extended
  );
    logic [11:0] rounded;
    integer final_exponent;
    begin
      final_exponent = exponent;
      if ((extended == 0) || (final_exponent <= 0)) begin
        fp16_round_normal_ftz = {sign, 15'b0};
      end else begin
        rounded = {1'b0, extended[13:3]};
        if (extended[2] &&
            (extended[1] || extended[0] || extended[3]))
          rounded = rounded + 1'b1;
        if (rounded[11]) begin
          rounded = rounded >> 1;
          final_exponent = final_exponent + 1;
        end
        if (final_exponent >= 31)
          fp16_round_normal_ftz = {sign, 5'h1f, 10'b0};
        else
          fp16_round_normal_ftz = {
            sign, final_exponent[4:0], rounded[9:0]};
      end
    end
  endfunction

  function automatic logic [15:0] fp16_add_rne_ftz_ordered(
    input logic [15:0] lhs_in,
    input logic [15:0] rhs_in,
    input logic        lhs_magnitude_ge
  );
    logic [15:0] lhs;
    logic [15:0] rhs;
    logic sign_big;
    logic sign_small;
    logic [4:0] exponent_big;
    logic [4:0] exponent_small;
    logic [13:0] significand_big;
    logic [13:0] significand_small;
    logic [13:0] aligned_small;
    logic [14:0] sum;
    logic [13:0] normalized;
    integer exponent;
    integer distance;
    begin
      lhs = fp16_sanitize_ftz(lhs_in);
      rhs = fp16_sanitize_ftz(rhs_in);
      if (fp16_is_nan(lhs) || fp16_is_nan(rhs)) begin
        fp16_add_rne_ftz_ordered = FP16_CANONICAL_NAN;
      end else if (fp16_is_inf(lhs) || fp16_is_inf(rhs)) begin
        if (fp16_is_inf(lhs) && fp16_is_inf(rhs) &&
            (lhs[15] != rhs[15]))
          fp16_add_rne_ftz_ordered = FP16_CANONICAL_NAN;
        else
          fp16_add_rne_ftz_ordered = fp16_is_inf(lhs) ? lhs : rhs;
      end else if (lhs[14:0] == 0 && rhs[14:0] == 0) begin
        fp16_add_rne_ftz_ordered = {lhs[15] & rhs[15], 15'b0};
      end else if (lhs[14:0] == 0) begin
        fp16_add_rne_ftz_ordered = rhs;
      end else if (rhs[14:0] == 0) begin
        fp16_add_rne_ftz_ordered = lhs;
      end else begin
        if (lhs_magnitude_ge) begin
          sign_big = lhs[15];
          sign_small = rhs[15];
          exponent_big = lhs[14:10];
          exponent_small = rhs[14:10];
          significand_big = {1'b1, lhs[9:0], 3'b0};
          significand_small = {1'b1, rhs[9:0], 3'b0};
        end else begin
          sign_big = rhs[15];
          sign_small = lhs[15];
          exponent_big = rhs[14:10];
          exponent_small = lhs[14:10];
          significand_big = {1'b1, rhs[9:0], 3'b0};
          significand_small = {1'b1, lhs[9:0], 3'b0};
        end
        exponent = exponent_big;
        distance = exponent_big - exponent_small;
        aligned_small = shift_right_sticky14(
          significand_small, distance);
        normalized = '0;
        if (sign_big == sign_small) begin
          sum = {1'b0, significand_big} + {1'b0, aligned_small};
          if (sum[14]) begin
            normalized = sum[14:1];
            normalized[0] = normalized[0] | sum[0];
            exponent = exponent + 1;
          end else
            normalized = sum[13:0];
        end else begin
          normalized = significand_big - aligned_small;
          for (integer normalize_step = 0;
               normalize_step < 13; normalize_step++) begin
            if (!normalized[13] && (normalized != 0) && (exponent > 1)) begin
              normalized = normalized << 1;
              exponent = exponent - 1;
            end
          end
        end
        if (normalized == 0)
          fp16_add_rne_ftz_ordered = 16'b0;
        else if (!normalized[13] && (exponent <= 1))
          fp16_add_rne_ftz_ordered = {sign_big, 15'b0};
        else
          fp16_add_rne_ftz_ordered = fp16_round_normal_ftz(
            sign_big, exponent, normalized);
      end
    end
  endfunction

  // Compatibility wrapper for callers which do not already own a magnitude
  // comparator. Basic ALU bypasses this wrapper and shares its comparator
  // between ADD, SUBTRACT and MAX.
  function automatic logic [15:0] fp16_add_rne_ftz(
    input logic [15:0] lhs_in,
    input logic [15:0] rhs_in
  );
    logic [15:0] lhs;
    logic [15:0] rhs;
    begin
      lhs = fp16_sanitize_ftz(lhs_in);
      rhs = fp16_sanitize_ftz(rhs_in);
      fp16_add_rne_ftz = fp16_add_rne_ftz_ordered(
        lhs, rhs, lhs[14:0] >= rhs[14:0]);
    end
  endfunction

  function automatic logic [15:0] fp16_sub_rne_ftz(
    input logic [15:0] lhs,
    input logic [15:0] rhs
  );
    fp16_sub_rne_ftz = fp16_add_rne_ftz(lhs, {~rhs[15], rhs[14:0]});
  endfunction

  function automatic logic [15:0] fp16_multiply_rne_ftz(
    input logic [15:0] lhs_in,
    input logic [15:0] rhs_in
  );
    logic [15:0] lhs;
    logic [15:0] rhs;
    logic sign;
    logic [21:0] product;
    logic [13:0] extended;
    integer exponent;
    begin
      lhs = fp16_sanitize_ftz(lhs_in);
      rhs = fp16_sanitize_ftz(rhs_in);
      sign = lhs[15] ^ rhs[15];
      if (fp16_is_nan(lhs) || fp16_is_nan(rhs))
        fp16_multiply_rne_ftz = FP16_CANONICAL_NAN;
      else if ((fp16_is_inf(lhs) && (rhs[14:0] == 0)) ||
               (fp16_is_inf(rhs) && (lhs[14:0] == 0)))
        fp16_multiply_rne_ftz = FP16_CANONICAL_NAN;
      else if (fp16_is_inf(lhs) || fp16_is_inf(rhs))
        fp16_multiply_rne_ftz = {sign, 5'h1f, 10'b0};
      else if ((lhs[14:0] == 0) || (rhs[14:0] == 0))
        fp16_multiply_rne_ftz = {sign, 15'b0};
      else begin
        product = {1'b1, lhs[9:0]} * {1'b1, rhs[9:0]};
        exponent = lhs[14:10] + rhs[14:10] - 15;
        if (product[21]) begin
          extended = product[21:8];
          extended[0] = extended[0] | (|product[7:0]);
          exponent = exponent + 1;
        end else begin
          extended = product[20:7];
          extended[0] = extended[0] | (|product[6:0]);
        end
        fp16_multiply_rne_ftz = fp16_round_normal_ftz(
          sign, exponent, extended);
      end
    end
  endfunction

  function automatic logic [15:0] fp16_divide_rne_ftz(
    input logic [15:0] lhs_in,
    input logic [15:0] rhs_in
  );
    logic [15:0] lhs;
    logic [15:0] rhs;
    logic sign;
    logic [24:0] numerator;
    logic [14:0] quotient;
    logic [10:0] remainder;
    logic [13:0] extended;
    integer exponent;
    begin
      lhs = fp16_sanitize_ftz(lhs_in);
      rhs = fp16_sanitize_ftz(rhs_in);
      sign = lhs[15] ^ rhs[15];
      if (fp16_is_nan(lhs) || fp16_is_nan(rhs) ||
          ((lhs[14:0] == 0) && (rhs[14:0] == 0)) ||
          (fp16_is_inf(lhs) && fp16_is_inf(rhs)))
        fp16_divide_rne_ftz = FP16_CANONICAL_NAN;
      else if (rhs[14:0] == 0)
        fp16_divide_rne_ftz = {sign, 5'h1f, 10'b0};
      else if (lhs[14:0] == 0)
        fp16_divide_rne_ftz = {sign, 15'b0};
      else if (fp16_is_inf(lhs))
        fp16_divide_rne_ftz = {sign, 5'h1f, 10'b0};
      else if (fp16_is_inf(rhs))
        fp16_divide_rne_ftz = {sign, 15'b0};
      else begin
        exponent = lhs[14:10] - rhs[14:10] + 15;
        if ({1'b1, lhs[9:0]} < {1'b1, rhs[9:0]}) begin
          numerator = {1'b1, lhs[9:0], 14'b0};
          exponent = exponent - 1;
        end else
          numerator = {2'b01, lhs[9:0], 13'b0};
        quotient = numerator / {1'b1, rhs[9:0]};
        remainder = numerator % {1'b1, rhs[9:0]};
        extended = quotient[13:0];
        extended[0] = extended[0] | (remainder != 0);
        fp16_divide_rne_ftz = fp16_round_normal_ftz(
          sign, exponent, extended);
      end
    end
  endfunction

  function automatic logic [15:0] fp16_max_ftz(
    input logic [15:0] lhs_in,
    input logic [15:0] rhs_in
  );
    logic [15:0] lhs;
    logic [15:0] rhs;
    logic lhs_less;
    begin
      lhs = fp16_sanitize_ftz(lhs_in);
      rhs = fp16_sanitize_ftz(rhs_in);
      // Match std::max(lhs, rhs): equal values and an unordered RHS return LHS.
      if (fp16_is_nan(lhs))
        fp16_max_ftz = FP16_CANONICAL_NAN;
      else if (fp16_is_nan(rhs))
        fp16_max_ftz = lhs;
      else begin
        if ((lhs[14:0] == 0) && (rhs[14:0] == 0))
          lhs_less = 1'b0;
        else if (lhs[15] != rhs[15])
          lhs_less = lhs[15];
        else if (lhs[15])
          lhs_less = lhs[14:0] > rhs[14:0];
        else
          lhs_less = lhs[14:0] < rhs[14:0];
        fp16_max_ftz = lhs_less ? rhs : lhs;
      end
    end
  endfunction

  function automatic logic [15:0] fp16_sqrt_rne_ftz(
    input logic [15:0] value_in
  );
    logic [15:0] value;
    logic [11:0] adjusted_significand;
    logic [27:0] radicand;
    logic [27:0] root;
    logic [27:0] trial;
    logic [55:0] root_square;
    logic [55:0] trial_square;
    logic [13:0] extended;
    integer unbiased_exponent;
    integer output_exponent;
    begin
      value = fp16_sanitize_ftz(value_in);
      if (fp16_is_nan(value) || (value[15] && value[14:0] != 0))
        fp16_sqrt_rne_ftz = FP16_CANONICAL_NAN;
      else if (fp16_is_inf(value) || (value[14:0] == 0))
        fp16_sqrt_rne_ftz = value;
      else begin
        unbiased_exponent = value[14:10] - 15;
        adjusted_significand = {1'b0, 1'b1, value[9:0]};
        if ((unbiased_exponent % 2) != 0) begin
          adjusted_significand = adjusted_significand << 1;
          unbiased_exponent = unbiased_exponent - 1;
        end
        radicand = {adjusted_significand, 16'b0};
        root = '0;
        for (integer root_bit = 13; root_bit >= 0; root_bit--) begin
          trial = root | (28'b1 << root_bit);
          trial_square = {28'b0, trial} * trial;
          if (trial_square <= {28'b0, radicand})
            root = trial;
        end
        extended = root[13:0];
        root_square = {28'b0, root} * root;
        if (root_square != {28'b0, radicand})
          extended[0] = 1'b1;
        output_exponent = unbiased_exponent / 2 + 15;
        fp16_sqrt_rne_ftz = fp16_round_normal_ftz(
          1'b0, output_exponent, extended);
      end
    end
  endfunction

  function automatic logic [15:0] fp16_reciprocal_rne_ftz(
    input logic [15:0] value
  );
    fp16_reciprocal_rne_ftz = fp16_divide_rne_ftz(FP16_ONE, value);
  endfunction

  function automatic logic [15:0] fp16_rsqrt_rne_ftz(
    input logic [15:0] value
  );
    logic [15:0] sanitized;
    begin
      sanitized = fp16_sanitize_ftz(value);
      if (fp16_is_nan(sanitized) ||
          (sanitized[15] && sanitized[14:0] != 0))
        fp16_rsqrt_rne_ftz = FP16_CANONICAL_NAN;
      else if (sanitized[14:0] == 0)
        fp16_rsqrt_rne_ftz = 16'h7c00;
      else if (fp16_is_inf(sanitized))
        fp16_rsqrt_rne_ftz = 16'b0;
      else
        fp16_rsqrt_rne_ftz = fp16_reciprocal_rne_ftz(
          fp16_sqrt_rne_ftz(sanitized));
    end
  endfunction

  function automatic integer fp16_to_uint_floor(
    input logic [15:0] value_in
  );
    logic [15:0] value;
    integer exponent;
    integer significand;
    begin
      value = fp16_sanitize_ftz(value_in);
      exponent = value[14:10] - 15;
      significand = 1024 + value[9:0];
      if (value[15] || (value[14:0] == 0) || (exponent < 0))
        fp16_to_uint_floor = 0;
      else if (fp16_is_inf(value) || (exponent >= 20))
        fp16_to_uint_floor = 32'h7fffffff;
      else if (exponent >= 10)
        fp16_to_uint_floor = significand << (exponent - 10);
      else
        fp16_to_uint_floor = significand >> (10 - exponent);
    end
  endfunction

  function automatic integer fp16_to_sint_rne(
    input logic [15:0] value_in
  );
    logic [15:0] value;
    integer exponent;
    integer significand;
    integer magnitude;
    integer remainder;
    integer halfway;
    integer shift;
    begin
      value = fp16_sanitize_ftz(value_in);
      exponent = value[14:10] - 15;
      significand = 1024 + value[9:0];
      magnitude = 0;
      if (fp16_is_inf(value) || fp16_is_nan(value)) begin
        magnitude = 32'h7fffffff;
      end else if (value[14:0] == 0 || exponent < -1) begin
        magnitude = 0;
      end else if (exponent == -1) begin
        // Exactly 0.5 rounds to the even integer zero.
        magnitude = significand > 1024 ? 1 : 0;
      end else if (exponent >= 10) begin
        magnitude = significand << (exponent - 10);
      end else begin
        shift = 10 - exponent;
        magnitude = significand >> shift;
        remainder = significand & ((1 << shift) - 1);
        halfway = 1 << (shift - 1);
        if ((remainder > halfway) ||
            ((remainder == halfway) && magnitude[0]))
          magnitude = magnitude + 1;
      end
      fp16_to_sint_rne = value[15] ? -magnitude : magnitude;
    end
  endfunction

  function automatic logic [15:0] uint_to_fp16_rne_ftz(
    input integer value
  );
    logic [31:0] magnitude;
    logic [13:0] extended;
    integer leading_bit;
    integer exponent;
    integer shift;
    logic sticky;
    begin
      magnitude = value < 0 ? 0 : value;
      leading_bit = -1;
      for (integer bit_index = 0; bit_index < 31; bit_index++)
        if (magnitude[bit_index])
          leading_bit = bit_index;
      if (leading_bit < 0) begin
        uint_to_fp16_rne_ftz = 16'b0;
      end else begin
        exponent = leading_bit + 15;
        if (leading_bit <= 13)
          extended = magnitude << (13 - leading_bit);
        else begin
          shift = leading_bit - 13;
          extended = magnitude >> shift;
          sticky = 1'b0;
          for (integer sticky_bit = 0; sticky_bit < 31; sticky_bit++)
            if (sticky_bit < shift)
              sticky = sticky | magnitude[sticky_bit];
          extended[0] = extended[0] | sticky;
        end
        uint_to_fp16_rne_ftz = fp16_round_normal_ftz(
          1'b0, exponent, extended);
      end
    end
  endfunction

  function automatic logic [15:0] fp16_scale_pow2_ftz(
    input logic [15:0] value_in,
    input integer exponent_delta
  );
    logic [15:0] value;
    integer exponent;
    begin
      value = fp16_sanitize_ftz(value_in);
      exponent = value[14:10] + exponent_delta;
      if (fp16_is_nan(value) || fp16_is_inf(value) ||
          (value[14:0] == 0))
        fp16_scale_pow2_ftz = value;
      else if (exponent <= 0)
        fp16_scale_pow2_ftz = {value[15], 15'b0};
      else if (exponent >= 31)
        fp16_scale_pow2_ftz = {value[15], 5'h1f, 10'b0};
      else
        fp16_scale_pow2_ftz = {
          value[15], exponent[4:0], value[9:0]};
    end
  endfunction
endpackage
