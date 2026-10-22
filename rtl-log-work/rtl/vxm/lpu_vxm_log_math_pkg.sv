// Log-specific representation helpers. Arithmetic inputs are already classified
// by the common ALU unpackers; these helpers never classify source operands.
package lpu_vxm_log_math_pkg;
  import lpu_pkg::*;
  import lpu_vxm_math_pkg::*;
  import lpu_vxm_fp16_pkg::*;

  localparam logic signed [47:0] LOG_LN2_Q40 = 48'sh00b17217f7d2;

  function automatic logic [31:0] log_round_format(
    input logic [31:0] value, input logic [1:0] dtype
  );
    case (dtype)
      VXM_FORMAT_FP16: log_round_format =
        fp16_to_fp32(fp16_sanitize_ftz(fp32_to_fp16(value)));
      VXM_FORMAT_BF16: log_round_format =
        bf16_to_fp32(fp32_to_bf16_ftz(value));
      default: log_round_format = value;
    endcase
  endfunction

  // One signed 48-bit, 40-fraction-bit restoration accumulator, with RNE
  // normalization into FP32. Small near-one results bypass this fixed scale.
  function automatic logic [31:0] log_fixed_to_fp32(
    input logic signed [47:0] value, input integer fraction_bits
  );
    logic sign;
    logic [47:0] magnitude, retained, mask, remainder, halfway;
    logic [24:0] significand;
    integer leading, shift, exponent;
    begin
      sign = value[47];
      magnitude = sign ? $unsigned(-value) : $unsigned(value);
      retained = 0; mask = 0; remainder = 0; halfway = 0;
      significand = 0; leading = -1; shift = 0; exponent = 0;
      for (integer bit_index = 0; bit_index < 48; bit_index++)
        if (magnitude[bit_index]) leading = bit_index;
      if (leading < 0) log_fixed_to_fp32 = 32'b0;
      else begin
        exponent = 127 + leading - fraction_bits;
        if (leading > 23) begin
          shift = leading - 23;
          retained = magnitude >> shift;
          mask = (48'b1 << shift) - 1'b1;
          remainder = magnitude & mask;
          halfway = 48'b1 << (shift-1);
          significand = {1'b0, retained[23:0]};
          if ((remainder > halfway) ||
              ((remainder == halfway) && significand[0]))
            significand = significand + 1'b1;
          if (significand[24]) begin
            significand = significand >> 1;
            exponent = exponent + 1;
          end
        end else begin
          retained = magnitude << (23-leading);
          significand = {1'b0, retained[23:0]};
        end
        log_fixed_to_fp32 = {sign, exponent[7:0], significand[22:0]};
      end
    end
  endfunction

  function automatic logic signed [47:0] log_fp32_to_q40(input logic [31:0] value);
    logic [47:0] magnitude, remainder, halfway, mask;
    logic [23:0] significand;
    integer shift;
    begin
      significand = {1'b1, value[22:0]};
      magnitude = 0; remainder = 0; halfway = 0; mask = 0;
      shift = integer'(value[30:23]) - 127 + 17;
      if (value[30:23] == 0) magnitude = 0;
      else if (shift >= 0) magnitude = {24'b0, significand} << shift;
      else if (shift > -48) begin
        magnitude = {24'b0, significand} >> (-shift);
        mask = (48'b1 << (-shift)) - 1'b1;
        remainder = {24'b0, significand} & mask;
        halfway = 48'b1 << (-shift-1);
        if ((remainder > halfway) ||
            ((remainder == halfway) && magnitude[0]))
          magnitude = magnitude + 1'b1;
      end
      log_fp32_to_q40 = value[31] ? -$signed(magnitude) : $signed(magnitude);
    end
  endfunction

  function automatic logic [27:0] log_active_base(
    input logic [27:0] stored, input logic [1:0] dtype
  );
    integer shift;
    logic [28:0] retained;
    logic [27:0] mask, discarded, halfway;
    begin
      shift = dtype == VXM_FORMAT_BF16 ? 14 :
        (dtype == VXM_FORMAT_FP16 ? 13 : 0);
      retained = {1'b0, stored}; mask = 0; discarded = 0; halfway = 0;
      if (shift != 0) begin
        retained = {1'b0, stored} >> shift;
        mask = (28'b1 << shift) - 1'b1;
        discarded = stored & mask;
        halfway = 28'b1 << (shift-1);
        if ((discarded > halfway) || ((discarded == halfway) && retained[0]))
          retained = retained + 1'b1;
        retained = retained << shift;
      end
      log_active_base = retained[27:0];
    end
  endfunction

  function automatic logic [21:0] log_active_inverse(
    input logic [21:0] stored, input logic [1:0] dtype
  );
    integer shift;
    logic [22:0] retained;
    logic [21:0] mask, discarded, halfway;
    begin
      shift = dtype == VXM_FORMAT_BF16 ? 12 :
        (dtype == VXM_FORMAT_FP16 ? 9 : 0);
      retained = {1'b0, stored}; mask = 0; discarded = 0; halfway = 0;
      if (shift != 0) begin
        retained = {1'b0, stored} >> shift;
        mask = (22'b1 << shift) - 1'b1;
        discarded = stored & mask;
        halfway = 22'b1 << (shift-1);
        if ((discarded > halfway) || ((discarded == halfway) && retained[0]))
          retained = retained + 1'b1;
        retained = retained << shift;
      end
      log_active_inverse = retained[21:0];
    end
  endfunction

  function automatic logic [31:0] log_pack_result(
    input logic [31:0] value, input logic [1:0] dtype
  );
    if ((value[30:23] == 8'hff) && (|value[22:0])) begin
      case (dtype)
        VXM_FORMAT_FP16: log_pack_result = 32'h00007e00;
        VXM_FORMAT_BF16: log_pack_result = 32'h00007fc0;
        default: log_pack_result = 32'h7fc00000;
      endcase
    end else case (dtype)
      VXM_FORMAT_FP16: log_pack_result =
        {16'b0, fp16_sanitize_ftz(fp32_to_fp16(value))};
      VXM_FORMAT_BF16: log_pack_result = {16'b0, fp32_to_bf16_ftz(value)};
      default: log_pack_result = value;
    endcase
  endfunction
endpackage
