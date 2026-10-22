package lpu_vxm_math_pkg;
function automatic signed [31:0] fp32_to_sint(input logic [31:0] bits);
  logic signed [31:0] value;
  logic [23:0] mantissa;
  integer exponent;
  begin
    exponent = bits[30:23] - 127;
    mantissa = {1'b1, bits[22:0]};
    if (bits[30:23] == 0)
      value = '0;
    else if (exponent < 0)
      value = '0;
    else if (exponent <= 23)
      value = $signed({1'b0, mantissa >> (23-exponent)});
    else if (exponent < 31)
      value = $signed({1'b0, mantissa}) <<< (exponent-23);
    else
      value = 32'sh7fffffff;
    fp32_to_sint = bits[31] ? -value : value;
  end
endfunction

function automatic logic fp32_is_integral(input logic [31:0] bits);
  integer exponent;
  logic [22:0] fraction_mask;
  begin
    exponent = bits[30:23] - 127;
    if (bits[30:23] == 0)
      fp32_is_integral = (bits[22:0] == 0);
    else if (bits[30:23] == 8'hff)
      fp32_is_integral = 1'b0;
    else if (exponent < 0)
      fp32_is_integral = 1'b0;
    else if (exponent >= 23)
      fp32_is_integral = (exponent < 31);
    else begin
      fraction_mask = (23'h7fffff >> exponent);
      fp32_is_integral = ((bits[22:0] & fraction_mask) == 0);
    end
  end
endfunction

function automatic signed [7:0] saturate_int8(
  input logic signed [31:0] value
);
  begin
    if (value > 127)
      saturate_int8 = 8'sd127;
    else if (value < -128)
      saturate_int8 = -8'sd128;
    else
      saturate_int8 = value[7:0];
  end
endfunction


function automatic logic [31:0] fp16_to_fp32(input logic [15:0] half);
  logic [31:0] sign;
  logic [4:0] exponent;
  logic [10:0] mantissa;
  integer normalized_exponent;
  begin
    sign = {half[15], 31'b0};
    exponent = half[14:10];
    mantissa = {1'b0, half[9:0]};
    if (exponent == 0) begin
      if (mantissa == 0)
        fp16_to_fp32 = sign;
      else begin
        normalized_exponent = -14;
        for (integer normalize_step = 0;
             normalize_step < 10; normalize_step++)
          if (!mantissa[10]) begin
            mantissa = mantissa << 1;
            normalized_exponent = normalized_exponent - 1;
          end
        mantissa[10] = 1'b0;
        fp16_to_fp32 = sign |
          ((normalized_exponent + 127) << 23) |
          ({21'b0, mantissa} << 13);
      end
    end else if (exponent == 5'h1f)
      fp16_to_fp32 = sign | 32'h7f800000 | ({21'b0, mantissa} << 13);
    else
      fp16_to_fp32 = sign | ((exponent + 112) << 23) |
        ({21'b0, mantissa} << 13);
  end
endfunction

function automatic logic [15:0] fp32_to_fp16(input logic [31:0] value);
  logic [15:0] sign;
  logic [7:0] source_exponent;
  logic [22:0] mantissa;
  logic [23:0] normalized;
  logic [31:0] remainder;
  logic [31:0] halfway;
  logic [10:0] half_mantissa;
  logic [15:0] half;
  integer exponent;
  integer shift;
  begin
    sign = {value[31], 15'b0};
    source_exponent = value[30:23];
    mantissa = value[22:0];
    exponent = source_exponent - 127 + 15;
    if (source_exponent == 8'hff) begin
      half_mantissa = (mantissa == 0) ? 0 : ((mantissa >> 13) | 1);
      fp32_to_fp16 = sign | 16'h7c00 | half_mantissa[9:0];
    end else if (exponent >= 31)
      fp32_to_fp16 = sign | 16'h7c00;
    else if (exponent <= 0) begin
      if (exponent < -10)
        fp32_to_fp16 = sign;
      else begin
        normalized = {1'b1, mantissa};
        shift = 14 - exponent;
        half_mantissa = normalized >> shift;
        remainder = normalized & ((32'b1 << shift) - 1);
        halfway = 32'b1 << (shift - 1);
        if ((remainder > halfway) ||
            ((remainder == halfway) && half_mantissa[0]))
          half_mantissa = half_mantissa + 1'b1;
        fp32_to_fp16 = sign | half_mantissa[9:0];
      end
    end else begin
      half = sign | (exponent << 10) | (mantissa >> 13);
      remainder = mantissa & 23'h001fff;
      if ((remainder > 32'h00001000) ||
          ((remainder == 32'h00001000) && half[0]))
        half = half + 1'b1;
      fp32_to_fp16 = half;
    end
  end
endfunction

function automatic logic [15:0] fp32_to_bf16(input logic [31:0] value);
  logic [31:0] rounded;
  begin
    if ((value[30:23] == 8'hff) && (value[22:0] != 0)) begin
      fp32_to_bf16 = value[31:16];
      if (fp32_to_bf16[6:0] == 0)
        fp32_to_bf16[0] = 1'b1;
    end else begin
      rounded = value + 32'h00007fff + value[16];
      fp32_to_bf16 = rounded[31:16];
    end
  end
endfunction

function automatic logic bf16_is_nan(input logic [15:0] value);
  bf16_is_nan = (value[14:7] == 8'hff) && (value[6:0] != 0);
endfunction

function automatic logic bf16_is_inf(input logic [15:0] value);
  bf16_is_inf = (value[14:7] == 8'hff) && (value[6:0] == 0);
endfunction

function automatic logic [15:0] bf16_sanitize_ftz(
  input logic [15:0] value
);
  begin
    if (bf16_is_nan(value))
      bf16_sanitize_ftz = {value[15], 8'hff, 7'h40};
    else if (value[14:7] == 0)
      bf16_sanitize_ftz = {value[15], 15'b0};
    else
      bf16_sanitize_ftz = value;
  end
endfunction

function automatic logic [31:0] bf16_to_fp32(input logic [15:0] value);
  bf16_to_fp32 = {bf16_sanitize_ftz(value), 16'b0};
endfunction

function automatic logic [15:0] fp32_to_bf16_ftz(
  input logic [31:0] value
);
  logic [15:0] rounded;
  begin
    rounded = fp32_to_bf16(value);
    fp32_to_bf16_ftz = bf16_sanitize_ftz(rounded);
  end
endfunction

function automatic logic [26:0] fp32_shift_right_sticky(
  input logic [26:0] value,
  input integer distance
);
  logic [26:0] shifted;
  logic sticky;
  begin
    sticky = 1'b0;
    for (integer bit_index = 0; bit_index < 27; bit_index++)
      if (bit_index < distance)
        sticky = sticky | value[bit_index];
    if (distance >= 27)
      shifted = '0;
    else
      shifted = value >> distance;
    shifted[0] = shifted[0] | sticky;
    fp32_shift_right_sticky = shifted;
  end
endfunction

function automatic logic [31:0] fp32_add_rne_ordered(
  input logic [31:0] lhs,
  input logic [31:0] rhs,
  input logic        lhs_magnitude_ge
);
  logic sign_big;
  logic sign_small;
  logic [7:0] exponent_big;
  logic [7:0] exponent_small;
  logic [26:0] significand_big;
  logic [26:0] significand_small;
  logic [26:0] aligned_small;
  logic [27:0] sum;
  logic [26:0] normalized;
  logic [24:0] rounded;
  logic [23:0] final_significand;
  integer exponent;
  integer distance;
  begin
    if (lhs[30:0] == 0 && rhs[30:0] == 0)
      fp32_add_rne_ordered = {lhs[31] & rhs[31], 31'b0};
    else if (lhs[30:0] == 0)
      fp32_add_rne_ordered = rhs;
    else if (rhs[30:0] == 0)
      fp32_add_rne_ordered = lhs;
    else begin
      if (lhs_magnitude_ge) begin
        sign_big = lhs[31];
        sign_small = rhs[31];
        exponent_big = lhs[30:23];
        exponent_small = rhs[30:23];
        significand_big = {1'b1, lhs[22:0], 3'b0};
        significand_small = {1'b1, rhs[22:0], 3'b0};
      end else begin
        sign_big = rhs[31];
        sign_small = lhs[31];
        exponent_big = rhs[30:23];
        exponent_small = lhs[30:23];
        significand_big = {1'b1, rhs[22:0], 3'b0};
        significand_small = {1'b1, lhs[22:0], 3'b0};
      end
      exponent = exponent_big;
      distance = exponent_big - exponent_small;
      aligned_small = fp32_shift_right_sticky(significand_small, distance);
      if (sign_big == sign_small) begin
        sum = {1'b0, significand_big} + {1'b0, aligned_small};
        if (sum[27]) begin
          normalized = sum[27:1];
          normalized[0] = normalized[0] | sum[0];
          exponent = exponent + 1;
        end else
          normalized = sum[26:0];
      end else begin
        normalized = significand_big - aligned_small;
        if (normalized == 0) begin
          fp32_add_rne_ordered = 32'b0;
          normalized = '0;
        end else begin
          for (integer normalize_step = 0;
               normalize_step < 26; normalize_step++)
            if (!normalized[26] && (exponent > 1)) begin
              normalized = normalized << 1;
              exponent = exponent - 1;
            end
        end
      end
      if (normalized != 0) begin
        if ((exponent == 1) && !normalized[26])
          fp32_add_rne_ordered = 32'h7fc00001;
        else begin
          rounded = {1'b0, normalized[26:3]};
          if (normalized[2] &&
              (normalized[1] || normalized[0] || normalized[3]))
            rounded = rounded + 1'b1;
          if (rounded[24]) begin
            final_significand = rounded[24:1];
            exponent = exponent + 1;
          end else
            final_significand = rounded[23:0];
          if (exponent >= 255)
            fp32_add_rne_ordered = {sign_big, 8'hff, 23'b0};
          else
            fp32_add_rne_ordered = {
              sign_big, exponent[7:0], final_significand[22:0]};
        end
      end
    end
  end
endfunction

// Compatibility wrapper for users without an external shared comparator.
function automatic logic [31:0] fp32_add_rne(
  input logic [31:0] lhs,
  input logic [31:0] rhs
);
  fp32_add_rne = fp32_add_rne_ordered(
    lhs, rhs, lhs[30:0] >= rhs[30:0]);
endfunction

function automatic logic [31:0] fp32_product_rne(
  input logic [31:0] lhs,
  input logic [31:0] rhs,
  input logic [47:0] raw_product
);
  logic sign;
  logic [23:0] lhs_significand;
  logic [23:0] rhs_significand;
  logic [47:0] product;
  logic [23:0] significand;
  logic guard_bit;
  logic round_bit;
  logic sticky_bit;
  logic [24:0] rounded;
  integer exponent;
  begin
    sign = lhs[31] ^ rhs[31];
    if ((lhs[30:0] == 0) || (rhs[30:0] == 0))
      fp32_product_rne = {sign, 31'b0};
    else begin
      lhs_significand = {1'b1, lhs[22:0]};
      rhs_significand = {1'b1, rhs[22:0]};
      product = raw_product;
      exponent = lhs[30:23] + rhs[30:23] - 127;
      if (product[47]) begin
        significand = product[47:24];
        guard_bit = product[23];
        round_bit = product[22];
        sticky_bit = |product[21:0];
        exponent = exponent + 1;
      end else begin
        significand = product[46:23];
        guard_bit = product[22];
        round_bit = product[21];
        sticky_bit = |product[20:0];
      end
      rounded = {1'b0, significand};
      if (guard_bit && (round_bit || sticky_bit || significand[0]))
        rounded = rounded + 1'b1;
      if (rounded[24]) begin
        significand = rounded[24:1];
        exponent = exponent + 1;
      end else
        significand = rounded[23:0];
      if (exponent <= 0)
        fp32_product_rne = 32'h7fc00001;
      else if (exponent >= 255)
        fp32_product_rne = {sign, 8'hff, 23'b0};
      else
        fp32_product_rne = {
          sign, exponent[7:0], significand[22:0]};
    end
  end
endfunction
function automatic logic [31:0] fp32_multiply_rne(
  input logic [31:0] lhs, input logic [31:0] rhs
);
  logic [23:0] a,b;
  logic [47:0] product;
  begin
    a={1'b1,lhs[22:0]}; b={1'b1,rhs[22:0]};
    product=a*b;
    fp32_multiply_rne=fp32_product_rne(lhs,rhs,product);
  end
endfunction

function automatic logic [31:0] fp32_divide_rne(
  input logic [31:0] lhs,
  input logic [31:0] rhs
);
  logic sign;
  logic [23:0] lhs_significand;
  logic [23:0] rhs_significand;
  logic [50:0] scaled_numerator;
  logic [50:0] quotient;
  logic [50:0] remainder;
  logic [26:0] normalized;
  logic [24:0] rounded;
  logic [23:0] final_significand;
  integer exponent;
  begin
    sign = lhs[31] ^ rhs[31];
    lhs_significand = {1'b1, lhs[22:0]};
    rhs_significand = {1'b1, rhs[22:0]};
    scaled_numerator = '0;
    quotient = '0;
    remainder = '0;
    normalized = '0;
    rounded = '0;
    final_significand = '0;
    exponent = lhs[30:23] - rhs[30:23] + 127;

    if (rhs[30:0] == 0)
      fp32_divide_rne = 32'h7fc00001;
    else if (lhs[30:0] == 0)
      fp32_divide_rne = {sign, 31'b0};
    else begin
      if (lhs_significand < rhs_significand) begin
        scaled_numerator = {lhs_significand, 27'b0};
        exponent = exponent - 1;
      end else
        scaled_numerator = {1'b0, lhs_significand, 26'b0};
      quotient = scaled_numerator / rhs_significand;
      remainder = scaled_numerator % rhs_significand;
      normalized = quotient[26:0];
      normalized[0] = normalized[0] | (remainder != 0);
      rounded = {1'b0, normalized[26:3]};
      if (normalized[2] &&
          (normalized[1] || normalized[0] || normalized[3]))
        rounded = rounded + 1'b1;
      if (rounded[24]) begin
        final_significand = rounded[24:1];
        exponent = exponent + 1;
      end else
        final_significand = rounded[23:0];

      if (exponent <= 0)
        fp32_divide_rne = {sign, 31'b0};
      else if (exponent >= 255)
        fp32_divide_rne = {sign, 8'hff, 23'b0};
      else
        fp32_divide_rne = {
          sign, exponent[7:0], final_significand[22:0]};
    end
  end
endfunction

function automatic logic [31:0] fp32_reciprocal_factorial(
  input integer term
);
  begin
    case (term)
      0, 1: fp32_reciprocal_factorial = 32'h3f800000;
      2:  fp32_reciprocal_factorial = 32'h3f000000;
      3:  fp32_reciprocal_factorial = 32'h3e2aaaab;
      4:  fp32_reciprocal_factorial = 32'h3d2aaaab;
      5:  fp32_reciprocal_factorial = 32'h3c088889;
      6:  fp32_reciprocal_factorial = 32'h3ab60b61;
      7:  fp32_reciprocal_factorial = 32'h39500d01;
      8:  fp32_reciprocal_factorial = 32'h37d00d01;
      9:  fp32_reciprocal_factorial = 32'h3638ef1d;
      10: fp32_reciprocal_factorial = 32'h3493f27e;
      11: fp32_reciprocal_factorial = 32'h32d7322b;
      12: fp32_reciprocal_factorial = 32'h310f76c7;
      13: fp32_reciprocal_factorial = 32'h2f309231;
      14: fp32_reciprocal_factorial = 32'h2d49cba5;
      15: fp32_reciprocal_factorial = 32'h2b573f9f;
      16: fp32_reciprocal_factorial = 32'h29573f9f;
      17: fp32_reciprocal_factorial = 32'h274a963c;
      18: fp32_reciprocal_factorial = 32'h253413c3;
      19: fp32_reciprocal_factorial = 32'h2317a4da;
      20: fp32_reciprocal_factorial = 32'h20f2a15d;
      default: fp32_reciprocal_factorial = 32'b0;
    endcase
  end
endfunction

function automatic logic [31:0] fp32_exp_approx(
  input logic [31:0] value
);
  logic negative;
  logic [31:0] magnitude;
  logic [31:0] polynomial;
  begin
    negative = value[31];
    magnitude = {1'b0, value[30:0]};
    if ((magnitude[30:23] > 8'd130) ||
        ((magnitude[30:23] == 8'd130) &&
         (magnitude[22:0] != 0)))
      magnitude = 32'h41000000;

    polynomial = fp32_reciprocal_factorial(20);
    for (integer term = 19; term >= 0; term--)
      polynomial = fp32_add_rne(
        fp32_multiply_rne(polynomial, magnitude),
        fp32_reciprocal_factorial(term));

    if (negative)
      fp32_exp_approx = fp32_divide_rne(32'h3f800000, polynomial);
    else
      fp32_exp_approx = polynomial;
  end
endfunction

endpackage
