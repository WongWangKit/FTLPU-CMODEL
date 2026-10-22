module lpu_vxm_special_alu #(
  parameter integer SPECIAL_KIND = lpu_pkg::VXM_SPECIAL_NONE,
  parameter integer LUT_BANK_COUNT = 3,
  parameter integer LUT_ENTRY_COUNT = 64,
  parameter integer LUT_BANK_WIDTH =
    LUT_BANK_COUNT <= 1 ? 1 : $clog2(LUT_BANK_COUNT),
  parameter integer LUT_ADDRESS_WIDTH =
    LUT_ENTRY_COUNT <= 1 ? 1 : $clog2(LUT_ENTRY_COUNT)
) (
  input  logic                         clk_i,
  input  logic                         rst_ni,
  input  logic                         valid_i,
  input  logic [1:0]                   data_format_i,
  input  logic [3:0]                   opcode_i,
  input  logic                         operand_format_valid_i,
  input  logic                         operand_sign_i,
  input  logic [7:0]                   operand_exponent_i,
  input  logic [22:0]                  operand_fraction_i,
  input  logic                         operand_zero_i,
  input  logic                         operand_inf_i,
  input  logic                         operand_nan_i,

  // EXP stores one packed UQ1.25 base as {k[12:0], b[12:0]}.
  // RECIP and RSQRT both store unsigned UQ1.15 k/b values.
  // BF16 and FP32 execute their approximation arithmetic in FP32.
  input  logic [LUT_BANK_COUNT-1:0]    lut_configured_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_input_min_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_segment_width_i,
  output logic                         lut_read_valid_o,
  output logic [LUT_BANK_WIDTH-1:0]    lut_read_bank_o,
  output logic [LUT_ADDRESS_WIDTH-1:0] lut_read_address_o,
  input  logic                         lut_read_valid_i,
  input  logic [15:0]                  lut_read_k_i,
  input  logic [15:0]                  lut_read_b_i,
  input  logic [49:0]                  lut_read_log_i,

  output logic                         ready_o,
  output logic                         valid_o,
  output logic [31:0]                  result_o,
  output logic                         illegal_opcode_o,
  output logic                         lut_fault_o
);
  generate if (SPECIAL_KIND == lpu_pkg::VXM_SPECIAL_RECIP_RSQRT) begin : g_c3
lpu_vxm_c3_pipeline #(
    .LUT_BANK_COUNT(LUT_BANK_COUNT),
    .LUT_ENTRY_COUNT(LUT_ENTRY_COUNT),
    .LUT_BANK_WIDTH(LUT_BANK_WIDTH),
    .LUT_ADDRESS_WIDTH(LUT_ADDRESS_WIDTH)
  ) u_pipeline (
    .clk_i,
    .rst_ni,
    .valid_i,
    .data_format_i,
    .opcode_i,
    .operand_format_valid_i,
    .operand_sign_i,
    .operand_exponent_i,
    .operand_fraction_i,
    .operand_zero_i,
    .operand_inf_i,
    .operand_nan_i,
    .lut_configured_i,
    .lut_input_min_i,
    .lut_segment_width_i,
    .lut_read_valid_o,
    .lut_read_bank_o,
    .lut_read_address_o,
    .lut_read_valid_i,
    .lut_read_k_i,
    .lut_read_b_i,
    .lut_read_log_i,
    .ready_o,
    .valid_o,
    .result_o,
    .illegal_opcode_o,
    .lut_fault_o
  );
  end else begin : g_legacy
lpu_vxm_special_alu_legacy #(
    .SPECIAL_KIND(SPECIAL_KIND),
    .LUT_BANK_COUNT(LUT_BANK_COUNT),
    .LUT_ENTRY_COUNT(LUT_ENTRY_COUNT),
    .LUT_BANK_WIDTH(LUT_BANK_WIDTH),
    .LUT_ADDRESS_WIDTH(LUT_ADDRESS_WIDTH)
  ) u_legacy (
    .clk_i,
    .rst_ni,
    .valid_i,
    .data_format_i,
    .opcode_i,
    .operand_format_valid_i,
    .operand_sign_i,
    .operand_exponent_i,
    .operand_fraction_i,
    .operand_zero_i,
    .operand_inf_i,
    .operand_nan_i,
    .lut_configured_i,
    .lut_input_min_i,
    .lut_segment_width_i,
    .lut_read_valid_o,
    .lut_read_bank_o,
    .lut_read_address_o,
    .lut_read_valid_i,
    .lut_read_k_i,
    .lut_read_b_i,
    .lut_read_log_i,
    .ready_o,
    .valid_o,
    .result_o,
    .illegal_opcode_o,
    .lut_fault_o
  );
  end endgenerate
endmodule

module lpu_vxm_special_alu_legacy #(
  parameter integer SPECIAL_KIND = lpu_pkg::VXM_SPECIAL_NONE,
  parameter integer LUT_BANK_COUNT = 3,
  parameter integer LUT_ENTRY_COUNT = 64,
  parameter integer LUT_BANK_WIDTH =
    LUT_BANK_COUNT <= 1 ? 1 : $clog2(LUT_BANK_COUNT),
  parameter integer LUT_ADDRESS_WIDTH =
    LUT_ENTRY_COUNT <= 1 ? 1 : $clog2(LUT_ENTRY_COUNT)
) (
  input  logic                         clk_i,
  input  logic                         rst_ni,
  input  logic                         valid_i,
  input  logic [1:0]                   data_format_i,
  input  logic [3:0]                   opcode_i,
  input  logic                         operand_format_valid_i,
  input  logic                         operand_sign_i,
  input  logic [7:0]                   operand_exponent_i,
  input  logic [22:0]                  operand_fraction_i,
  input  logic                         operand_zero_i,
  input  logic                         operand_inf_i,
  input  logic                         operand_nan_i,

  // EXP stores one packed UQ1.25 base as {k[12:0], b[12:0]}.
  // RECIP and RSQRT both store unsigned UQ1.15 k/b values.
  // BF16 and FP32 execute their approximation arithmetic in FP32.
  input  logic [LUT_BANK_COUNT-1:0]    lut_configured_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_input_min_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_segment_width_i,
  output logic                         lut_read_valid_o,
  output logic [LUT_BANK_WIDTH-1:0]    lut_read_bank_o,
  output logic [LUT_ADDRESS_WIDTH-1:0] lut_read_address_o,
  input  logic                         lut_read_valid_i,
  input  logic [15:0]                  lut_read_k_i,
  input  logic [15:0]                  lut_read_b_i,
  input  logic [49:0]                  lut_read_log_i,

  output logic                         ready_o,
  output logic                         valid_o,
  output logic [31:0]                  result_o,
  output logic                         illegal_opcode_o,
  output logic                         lut_fault_o
);
  // C3 is bound to the spatial pipeline above; this retained path serves EXP.
  import lpu_pkg::*;
  import lpu_vxm_fp16_pkg::*;
  import lpu_vxm_math_pkg::*;

  localparam logic [15:0] FP16_INV_LN2 = 16'h3dc5;
  localparam logic [15:0] FP16_LN2     = 16'h398c;
  localparam logic [31:0] FP32_INV_LN2 = 32'h3fb8aa3b;
  localparam logic [31:0] FP32_LN2     = 32'h3f317218;
  localparam logic [31:0] FP32_ONE     = 32'h3f800000;
  localparam logic [31:0] FP32_TWO     = 32'h40000000;
  localparam logic [31:0] FP32_HALF    = 32'h3f000000;
  localparam logic [31:0] FP32_ONE_SIXTH = 32'h3e2aaaab;
  localparam logic [31:0] FP32_CANONICAL_NAN = 32'h7fc00000;

  function automatic logic fp32_is_nan_local(input logic [31:0] value);
    fp32_is_nan_local =
      (value[30:23] == 8'hff) && (value[22:0] != 0);
  endfunction

  function automatic logic fp32_is_inf_local(input logic [31:0] value);
    fp32_is_inf_local =
      (value[30:23] == 8'hff) && (value[22:0] == 0);
  endfunction

  function automatic logic [31:0] fp32_sanitize_ftz_local(
    input logic [31:0] value
  );
    begin
      if (fp32_is_nan_local(value))
        fp32_sanitize_ftz_local = {value[31], 8'hff, 1'b1, 22'b0};
      else if (value[30:23] == 0)
        fp32_sanitize_ftz_local = {value[31], 31'b0};
      else
        fp32_sanitize_ftz_local = value;
    end
  endfunction

  function automatic signed [31:0] fp32_to_sint_rne_local(
    input logic [31:0] value
  );
    logic [23:0] significand;
    logic [31:0] magnitude;
    logic [31:0] remainder;
    logic [31:0] halfway;
    integer exponent;
    integer shift;
    begin
      significand = {1'b1, value[22:0]};
      magnitude = 0;
      remainder = 0;
      halfway = 0;
      exponent = value[30:23];
      exponent = exponent - 127;
      shift = 0;
      if ((value[30:23] == 0) || (exponent < -1))
        magnitude = 0;
      else if (exponent == -1)
        magnitude = value[22:0] == 0 ? 0 : 1;
      else if (exponent >= 31)
        magnitude = 32'h7fffffff;
      else if (exponent >= 23)
        magnitude = {8'b0, significand} << (exponent-23);
      else begin
        shift = 23-exponent;
        magnitude = significand >> shift;
        remainder = significand & ((32'b1 << shift)-1'b1);
        halfway = 32'b1 << (shift-1);
        if ((remainder > halfway) ||
            ((remainder == halfway) && magnitude[0]))
          magnitude = magnitude + 1'b1;
      end
      fp32_to_sint_rne_local = value[31] ?
        -$signed(magnitude) : $signed(magnitude);
    end
  endfunction

  function automatic logic [31:0] uint_to_fp32_exact_local(
    input integer value
  );
    logic [31:0] magnitude;
    logic [31:0] normalized;
    logic [7:0] exponent;
    integer most_significant_bit;
    begin
      magnitude = value;
      normalized = 0;
      exponent = 0;
      most_significant_bit = 0;
      if (value <= 0)
        uint_to_fp32_exact_local = 32'b0;
      else begin
        for (integer bit_index = 0; bit_index < 31; bit_index++)
          if (magnitude[bit_index])
            most_significant_bit = bit_index;
        exponent = 127 + most_significant_bit;
        if (most_significant_bit <= 23)
          normalized = magnitude << (23-most_significant_bit);
        else
          normalized = magnitude >> (most_significant_bit-23);
        uint_to_fp32_exact_local =
          {1'b0, exponent, normalized[22:0]};
      end
    end
  endfunction

  function automatic logic [31:0] fp32_scale_pow2_ftz_local(
    input logic [31:0] value_in,
    input integer scale
  );
    logic [31:0] value;
    integer exponent;
    begin
      value = fp32_sanitize_ftz_local(value_in);
      exponent = value[30:23];
      exponent = exponent + scale;
      if (fp32_is_nan_local(value) || fp32_is_inf_local(value) ||
          (value[30:0] == 0))
        fp32_scale_pow2_ftz_local = value;
      else if (exponent <= 0)
        fp32_scale_pow2_ftz_local = {value[31], 31'b0};
      else if (exponent >= 255)
        fp32_scale_pow2_ftz_local = {value[31], 8'hff, 23'b0};
      else
        fp32_scale_pow2_ftz_local =
          {value[31], exponent[7:0], value[22:0]};
    end
  endfunction

  // Convert the packed 26-bit EXP table value B/2^25 to FP32. The table
  // contains only positive values, but the converter is general over the
  // full UQ1.25 magnitude range and performs RNE when low bits are lost.
  function automatic logic [31:0] uq1_25_to_fp32_local(
    input logic [25:0] value
  );
    integer leading_bit;
    integer exponent;
    integer shift;
    logic [26:0] rounded_significand;
    logic [25:0] retained;
    logic [25:0] remainder_mask;
    logic [25:0] remainder;
    logic [25:0] halfway;
    begin
      leading_bit = -1;
      exponent = 0;
      shift = 0;
      rounded_significand = 27'b0;
      retained = 26'b0;
      remainder_mask = 26'b0;
      remainder = 26'b0;
      halfway = 26'b0;
      for (integer bit_index = 0; bit_index < 26; bit_index++)
        if (value[bit_index])
          leading_bit = bit_index;
      if (leading_bit < 0) begin
        uq1_25_to_fp32_local = 32'b0;
      end else begin
        exponent = 127 + leading_bit - 25;
        if (leading_bit <= 23) begin
          rounded_significand = {1'b0, value} << (23-leading_bit);
        end else begin
          shift = leading_bit - 23;
          retained = value >> shift;
          remainder_mask = (26'b1 << shift) - 1'b1;
          remainder = value & remainder_mask;
          halfway = 26'b1 << (shift-1);
          rounded_significand = {1'b0, retained};
          if ((remainder > halfway) ||
              ((remainder == halfway) && rounded_significand[0]))
            rounded_significand = rounded_significand + 1'b1;
          if (rounded_significand[24]) begin
            rounded_significand = rounded_significand >> 1;
            exponent = exponent + 1;
          end
        end
        uq1_25_to_fp32_local =
          {1'b0, exponent[7:0], rounded_significand[22:0]};
      end
    end
  endfunction

  logic [31:0] reciprocal_newton_stage1_product;
  logic supported_opcode;
  logic [15:0] sanitized_operand16;
  logic [15:0] sanitized_operand_bf16;
  logic [31:0] working_operand32;
  logic request_lookup;
  logic request_reciprocal;
  logic request_rsqrt;
  logic request_uq_coefficient;
  logic request_subtract_slope;
  logic request_exp_base;
  logic request_exp_cubic;
  logic request_reciprocal_newton;
  logic request_rsqrt_newton;
  logic [LUT_BANK_WIDTH-1:0] request_bank;
  logic [31:0] request_local_input;
  logic [31:0] request_direct_result;
  logic request_multiplier_sign;
  integer request_result_exponent;
  logic request_reciprocal_dual_exponent;
  logic signed [9:0] request_reciprocal_unity_exponent;
  logic signed [9:0] request_reciprocal_fraction_exponent;
  logic request_configured;
  logic [31:0] request_input_min;
  logic [31:0] request_segment_width;

  logic stage0_valid_q;
  logic stage0_lookup_q;
  logic [1:0] stage0_format_q;
  logic [LUT_BANK_WIDTH-1:0] stage0_bank_q;
  logic [31:0] stage0_local_input_q;
  logic [31:0] stage0_direct_result_q;
  logic stage0_multiplier_sign_q;
  logic signed [7:0] stage0_result_exponent_q;
  logic [31:0] stage0_input_min_q;
  logic [31:0] stage0_segment_width_q;
  logic stage0_reciprocal_q;
  logic stage0_rsqrt_q;
  logic stage0_uq_coefficient_q;
  logic stage0_subtract_slope_q;
  logic stage0_exp_base_q;
  logic stage0_exp_cubic_q;
  logic stage0_reciprocal_newton_q;
  logic stage0_rsqrt_newton_q;
  logic stage0_reciprocal_dual_exponent_q;
  logic signed [9:0] stage0_reciprocal_unity_exponent_q;
  logic signed [9:0] stage0_reciprocal_fraction_exponent_q;

  logic [31:0] lookup_position;
  logic [15:0] lookup_index_fp16;
  logic [31:0] lookup_index_fp32;
  logic [31:0] lookup_x0;
  logic [31:0] lookup_dx;
  integer lookup_index;

  logic read_meta_valid_q;
  logic read_meta_lookup_q;
  logic [1:0] read_meta_format_q;
  logic [31:0] read_meta_dx_q;
  logic [31:0] read_meta_direct_result_q;
  logic read_meta_multiplier_sign_q;
  logic signed [7:0] read_meta_result_exponent_q;
  logic read_meta_uq_coefficient_q;
  logic read_meta_subtract_slope_q;
  logic read_meta_exp_base_q;
  logic read_meta_exp_cubic_q;
  logic read_meta_reciprocal_newton_q;
  logic [31:0] read_meta_reciprocal_m_q;
  logic [16:0] read_meta_reciprocal_residual_q;
  logic read_meta_rsqrt_q;
  logic read_meta_rsqrt_newton_q;
  logic [17:0] read_meta_rsqrt_residual_q;
  logic read_meta_reciprocal_dual_exponent_q;
  logic signed [9:0] read_meta_reciprocal_unity_exponent_q;
  logic signed [9:0] read_meta_reciprocal_fraction_exponent_q;

  logic multiply_valid_q;
  logic multiply_lookup_q;
  logic [1:0] multiply_format_q;
  logic [31:0] multiply_product_q;
  logic [31:0] multiply_b_q;
  logic [31:0] multiply_direct_result_q;
  logic multiply_multiplier_sign_q;
  logic signed [7:0] multiply_result_exponent_q;
  logic multiply_subtract_slope_q;
  logic multiply_exp_cubic_q;
  logic [31:0] multiply_exp_dx_q;
  logic [25:0] multiply_exp_base_uq_q;
  logic multiply_reciprocal_newton_q;
  logic multiply_reciprocal_fixed_q;
  logic multiply_rsqrt_newton_q;
  logic [31:0] multiply_reciprocal_m_q;
  logic multiply_reciprocal_dual_exponent_q;
  logic signed [9:0] multiply_reciprocal_unity_exponent_q;
  logic signed [9:0] multiply_reciprocal_fraction_exponent_q;

  logic add_valid_q;
  logic [1:0] add_format_q;
  logic [31:0] add_result_q;
  logic add_multiplier_sign_q;
  logic signed [7:0] add_result_exponent_q;
  logic add_exp_cubic_q;
  logic [31:0] add_exp_dx_q;
  logic [25:0] add_exp_base_uq_q;
  logic add_reciprocal_newton_q;
  logic add_rsqrt_newton_q;
  logic [31:0] add_reciprocal_m_q;
  logic add_reciprocal_dual_exponent_q;
  logic signed [9:0] add_reciprocal_unity_exponent_q;
  logic signed [9:0] add_reciprocal_fraction_exponent_q;
  logic [31:0] restored_result;
  logic [31:0] restored_wide;
  logic [31:0] decoded_lut_k;
  logic [31:0] decoded_lut_b;
  logic [25:0] decoded_exp_base_uq;

  logic exp_horner1_valid_q;
  logic [31:0] exp_horner1_result_q;
  logic [31:0] exp_horner1_dx_q;
  logic [25:0] exp_horner1_base_uq_q;
  logic signed [7:0] exp_horner1_exponent_q;
  logic exp_horner2_valid_q;
  logic [31:0] exp_horner2_result_q;
  logic [25:0] exp_horner2_base_uq_q;
  logic signed [7:0] exp_horner2_exponent_q;
  logic exp_base_valid_q;
  logic [31:0] exp_base_result_q;
  logic signed [7:0] exp_base_exponent_q;
  logic [31:0] exp_restored_result;

  logic reciprocal_multiply_valid_q;
  logic [31:0] reciprocal_xy_q;
  logic [31:0] reciprocal_seed_q;
  logic reciprocal_multiply_sign_q;
  logic signed [9:0] reciprocal_multiply_unity_exponent_q;
  logic signed [9:0] reciprocal_multiply_fraction_exponent_q;
  logic reciprocal_correction_valid_q;
  logic [31:0] reciprocal_correction_q;
  logic [31:0] reciprocal_correction_seed_q;
  logic reciprocal_correction_sign_q;
  logic signed [9:0] reciprocal_correction_unity_exponent_q;
  logic signed [9:0] reciprocal_correction_fraction_exponent_q;
  logic reciprocal_newton_valid_q;
  logic [31:0] reciprocal_newton_result_q;
  logic reciprocal_newton_sign_q;
  logic signed [9:0] reciprocal_newton_unity_exponent_q;
  logic signed [9:0] reciprocal_newton_fraction_exponent_q;
  logic [31:0] reciprocal_newton_restored_result;
  logic rsqrt_square_valid_q;
  logic [31:0] rsqrt_y_squared_q;
  logic [31:0] rsqrt_square_seed_q;
  logic [31:0] rsqrt_square_m_q;
  logic signed [7:0] rsqrt_square_exponent_q;
  logic rsqrt_correction_valid_q;
  logic [31:0] rsqrt_correction_q;
  logic [31:0] rsqrt_correction_seed_q;
  logic signed [7:0] rsqrt_correction_exponent_q;
  logic rsqrt_newton_valid_q;
  logic [31:0] rsqrt_newton_result_q;
  logic signed [7:0] rsqrt_newton_exponent_q;
  logic [31:0] rsqrt_newton_restored_result;
  logic special_coefficient_multiply_enable;
  logic [1:0] special_coefficient_multiply_format;
  logic [25:0] special_coefficient_multiply_uq;
  logic [31:0] special_coefficient_multiply_value;
  logic [31:0] special_coefficient_value;
  logic [31:0] special_coefficient_product;
  logic reciprocal_linear_enable;
  logic [31:0] reciprocal_linear_y0;
  logic reciprocal_linear_fault;
  logic rsqrt_linear_enable;
  logic [31:0] rsqrt_linear_y0;
  logic rsqrt_linear_fault;
  logic [31:0] reciprocal_newton_stage2_product;
  logic [31:0] rsqrt_square_product;
  logic [31:0] rsqrt_correction_value;
  logic rsqrt_correction_fault;
  logic [31:0] rsqrt_final_product;
  logic bypass1_valid_q;
  logic [31:0] bypass1_result_q;
  logic bypass2_valid_q;
  logic [31:0] bypass2_result_q;
  logic bypass3_valid_q;
  logic [31:0] bypass3_result_q;
  logic [16:0] lookup_reciprocal_residual;
  logic [17:0] lookup_rsqrt_residual;

  // Low-precision linear EXP and the final FP32 cubic scale never overlap:
  // one ALU accepts only one outstanding special request. They therefore
  // share one 26x24 unsigned coefficient multiplier. Format-dependent
  // operands are zero-extended inside the multiplier.
  always_comb begin
    special_coefficient_multiply_enable = 1'b0;
    special_coefficient_multiply_format = read_meta_format_q;
    special_coefficient_multiply_uq = decoded_exp_base_uq;
    special_coefficient_multiply_value = read_meta_dx_q;
    if (read_meta_valid_q && read_meta_lookup_q && lut_read_valid_i &&
        read_meta_exp_base_q && !read_meta_exp_cubic_q) begin
      special_coefficient_multiply_enable = 1'b1;
    end
    if (exp_horner2_valid_q) begin
      special_coefficient_multiply_enable = 1'b1;
      special_coefficient_multiply_format = VXM_FORMAT_FP32;
      special_coefficient_multiply_uq = exp_horner2_base_uq_q;
      special_coefficient_multiply_value = exp_horner2_result_q;
    end
  end

  lpu_vxm_special_coefficient_multiplier u_exp_coefficient_multiplier (
    .enable_i(special_coefficient_multiply_enable),
    .data_format_i(special_coefficient_multiply_format),
    .coefficient_i(special_coefficient_multiply_uq),
    .offset_i(special_coefficient_multiply_uq),
    .value_i(special_coefficient_multiply_value),
    .coefficient_value_o(special_coefficient_value),
    .offset_value_o(),
    .product_o(special_coefficient_product)
  );

  always_comb begin
    reciprocal_linear_enable = read_meta_valid_q && read_meta_lookup_q &&
      lut_read_valid_i && read_meta_uq_coefficient_q && !read_meta_rsqrt_q;
    rsqrt_linear_enable = read_meta_valid_q && read_meta_lookup_q &&
      lut_read_valid_i && read_meta_rsqrt_q;
  end

  lpu_vxm_recip_linear_fixed u_recip_linear_fixed (
    .enable_i(reciprocal_linear_enable),
    .data_format_i(read_meta_format_q),
    .k_uq1_15_i(lut_read_k_i),
    .b_uq1_15_i(lut_read_b_i),
    .residual_i(read_meta_reciprocal_residual_q),
    .y0_o(reciprocal_linear_y0),
    .range_fault_o(reciprocal_linear_fault)
  );

  lpu_vxm_rsqrt_linear_fixed u_rsqrt_linear_fixed (
    .enable_i(rsqrt_linear_enable),
    .data_format_i(read_meta_format_q),
    .k_uq1_15_i(lut_read_k_i),
    .b_uq1_15_i(lut_read_b_i),
    .residual_i(read_meta_rsqrt_residual_q),
    .y0_o(rsqrt_linear_y0),
    .range_fault_o(rsqrt_linear_fault)
  );

  lpu_vxm_fp32_newton_multiplier u_recip_newton_multiply_stage1 (
    .enable_i(add_valid_q && add_reciprocal_newton_q),
    .lhs_i(add_reciprocal_m_q),
    .rhs_i(add_result_q),
    .product_o(reciprocal_newton_stage1_product)
  );

  lpu_vxm_fp32_newton_multiplier u_recip_newton_multiply_stage2 (
    .enable_i(reciprocal_correction_valid_q),
    .lhs_i(reciprocal_correction_seed_q),
    .rhs_i(reciprocal_correction_q),
    .product_o(reciprocal_newton_stage2_product)
  );

  lpu_vxm_fp32_newton_multiplier u_rsqrt_newton_square (
    .enable_i(add_valid_q && add_rsqrt_newton_q),
    .lhs_i(add_result_q),
    .rhs_i(add_result_q),
    .product_o(rsqrt_square_product)
  );

  lpu_vxm_fp32_rsqrt_correction u_rsqrt_newton_correction (
    .enable_i(rsqrt_square_valid_q),
    .m_i(rsqrt_square_m_q),
    .y_squared_i(rsqrt_y_squared_q),
    .correction_o(rsqrt_correction_value),
    .range_fault_o(rsqrt_correction_fault)
  );

  lpu_vxm_fp32_newton_multiplier u_rsqrt_newton_final_multiply (
    .enable_i(rsqrt_correction_valid_q),
    .lhs_i(rsqrt_correction_seed_q),
    .rhs_i(rsqrt_correction_q),
    .product_o(rsqrt_final_product)
  );

  always_comb begin
    // One request per physical ALU may be outstanding. A lookup can remain
    // in read_meta_valid_q for an arbitrary number of arbitration cycles.
    ready_o = !(stage0_valid_q || read_meta_valid_q ||
                multiply_valid_q || add_valid_q ||
                exp_horner1_valid_q || exp_horner2_valid_q ||
                exp_base_valid_q || reciprocal_multiply_valid_q ||
                reciprocal_correction_valid_q || reciprocal_newton_valid_q ||
                rsqrt_square_valid_q || rsqrt_correction_valid_q ||
                rsqrt_newton_valid_q ||
                bypass1_valid_q || bypass2_valid_q || bypass3_valid_q);
  end

  always_comb begin
    exp_restored_result = fp32_scale_pow2_ftz_local(
      exp_base_result_q, exp_base_exponent_q);
    rsqrt_newton_restored_result = fp32_scale_pow2_ftz_local(
      rsqrt_newton_result_q, rsqrt_newton_exponent_q);
  end

  always_comb begin
    logic signed [9:0] selected_exponent;

    selected_exponent = reciprocal_newton_result_q[30:23] >= 8'd127 ?
      reciprocal_newton_unity_exponent_q :
      reciprocal_newton_fraction_exponent_q;
    if ((reciprocal_newton_result_q[30:0] == 0) ||
        (selected_exponent <= 10'sd0))
      reciprocal_newton_restored_result = 32'b0;
    else if (selected_exponent >= 10'sd255)
      reciprocal_newton_restored_result = 32'h7f800000;
    else
      reciprocal_newton_restored_result =
        {1'b0, selected_exponent[7:0],
         reciprocal_newton_result_q[22:0]};
    if (reciprocal_newton_sign_q)
      reciprocal_newton_restored_result[31] = 1'b1;
  end

  always_comb begin
    integer normalized_exponent;
    integer rounded_exponent;
    integer exponent_magnitude;
    logic [15:0] exponent_fp16;
    logic [15:0] exponent_times_ln2_fp16;
    logic [31:0] exponent_fp32;
    logic [31:0] exponent_times_ln2_fp32;
    logic [15:0] selected_input_min;
    logic [15:0] selected_segment_width;

    supported_opcode =
      ((SPECIAL_KIND == VXM_SPECIAL_EXP) &&
       (opcode_i == VXM_LOCAL_SPECIAL0)) ||
      ((SPECIAL_KIND == VXM_SPECIAL_RECIP_RSQRT) &&
       ((opcode_i == VXM_LOCAL_SPECIAL0) ||
        (opcode_i == VXM_LOCAL_SPECIAL1)));
    illegal_opcode_o = valid_i &&
      (!operand_format_valid_i || !supported_opcode);
    // Reconstruct the DAZ-effective operand from the single shared unpack
    // front end.  No special operation reclassifies the raw input bits.
    sanitized_operand16 = {
      operand_sign_i, operand_exponent_i[4:0], operand_fraction_i[9:0]};
    sanitized_operand_bf16 = {
      operand_sign_i, operand_exponent_i[7:0], operand_fraction_i[6:0]};
    working_operand32 = data_format_i == VXM_FORMAT_BF16 ?
      bf16_to_fp32(sanitized_operand_bf16) :
      {operand_sign_i, operand_exponent_i, operand_fraction_i};
    request_lookup = 1'b0;
    request_reciprocal = 1'b0;
    request_rsqrt = 1'b0;
    request_uq_coefficient = 1'b0;
    request_subtract_slope = 1'b0;
    request_exp_base = 1'b0;
    request_exp_cubic = 1'b0;
    request_reciprocal_newton = 1'b0;
    request_rsqrt_newton = 1'b0;
    request_bank = '0;
    request_local_input = 32'b0;
    request_direct_result = data_format_i == VXM_FORMAT_FP16 ?
      {16'b0, FP16_CANONICAL_NAN} : FP32_CANONICAL_NAN;
    request_multiplier_sign = 1'b0;
    request_result_exponent = 0;
    request_reciprocal_dual_exponent = 1'b0;
    request_reciprocal_unity_exponent = 0;
    request_reciprocal_fraction_exponent = 0;
    normalized_exponent = 0;
    rounded_exponent = 0;
    exponent_magnitude = 0;
    exponent_fp16 = 16'b0;
    exponent_times_ln2_fp16 = 16'b0;
    exponent_fp32 = 32'b0;
    exponent_times_ln2_fp32 = 32'b0;

    if ((SPECIAL_KIND == VXM_SPECIAL_EXP) &&
        (opcode_i == VXM_LOCAL_SPECIAL0)) begin
      request_bank = 0;
      request_exp_base = 1'b1;
      request_exp_cubic = data_format_i == VXM_FORMAT_FP32;
      if (data_format_i != VXM_FORMAT_FP16) begin
        if (operand_nan_i)
          request_direct_result = FP32_CANONICAL_NAN;
        else if (operand_inf_i)
          request_direct_result = operand_sign_i ?
            32'b0 : 32'h7f800000;
        else begin
          request_lookup = 1'b1;
          rounded_exponent = fp32_to_sint_rne_local(fp32_multiply_rne(
            working_operand32, FP32_INV_LN2));
          if (rounded_exponent > 127)
            rounded_exponent = 127;
          else if (rounded_exponent < -127)
            rounded_exponent = -127;
          exponent_magnitude = rounded_exponent < 0 ?
            -rounded_exponent : rounded_exponent;
          exponent_fp32 = uint_to_fp32_exact_local(exponent_magnitude);
          if (rounded_exponent < 0)
            exponent_fp32[31] = 1'b1;
          exponent_times_ln2_fp32 = fp32_multiply_rne(
            exponent_fp32, FP32_LN2);
          request_local_input = fp32_add_rne(
            working_operand32,
            {~exponent_times_ln2_fp32[31],
             exponent_times_ln2_fp32[30:0]});
          request_result_exponent = rounded_exponent;
        end
      end else begin
        if (operand_nan_i)
          request_direct_result[15:0] = FP16_CANONICAL_NAN;
        else if (operand_inf_i)
          request_direct_result[15:0] = operand_sign_i ?
            16'b0 : 16'h7c00;
        else begin
          request_lookup = 1'b1;
          rounded_exponent = fp16_to_sint_rne(fp16_multiply_rne_ftz(
            sanitized_operand16, FP16_INV_LN2));
          if (rounded_exponent > 31)
            rounded_exponent = 31;
          else if (rounded_exponent < -31)
            rounded_exponent = -31;
          exponent_magnitude = rounded_exponent < 0 ?
            -rounded_exponent : rounded_exponent;
          exponent_fp16 = uint_to_fp16_rne_ftz(exponent_magnitude);
          if (rounded_exponent < 0)
            exponent_fp16[15] = 1'b1;
          exponent_times_ln2_fp16 = fp16_multiply_rne_ftz(
            exponent_fp16, FP16_LN2);
          request_local_input[15:0] = fp16_sub_rne_ftz(
            sanitized_operand16, exponent_times_ln2_fp16);
          request_result_exponent = rounded_exponent;
        end
      end
    end else if ((SPECIAL_KIND == VXM_SPECIAL_RECIP_RSQRT) &&
                 (opcode_i == VXM_LOCAL_SPECIAL0)) begin
      request_bank = 1;
      request_reciprocal = 1'b1;
      request_uq_coefficient = 1'b1;
      request_subtract_slope = 1'b1;
      request_reciprocal_newton = data_format_i == VXM_FORMAT_FP32;
      if (data_format_i != VXM_FORMAT_FP16) begin
        if (operand_nan_i)
          request_direct_result = FP32_CANONICAL_NAN;
        else if (operand_zero_i)
          request_direct_result =
            {operand_sign_i, 8'hff, 23'b0};
        else if (operand_inf_i)
          request_direct_result = {operand_sign_i, 31'b0};
        else begin
          request_lookup = 1'b1;
          request_local_input =
            {1'b0, 8'd127, working_operand32[22:0]};
          request_multiplier_sign = operand_sign_i;
          // For x = 1.f * 2^e, the interpolated reciprocal is either
          // exactly 1.x or lies in [0.5, 1). Generate both biased output
          // exponents now; the rounded LUT result selects between them.
          request_reciprocal_dual_exponent = 1'b1;
          request_reciprocal_unity_exponent =
            $signed({2'b00, ~operand_exponent_i}) - 10'sd1;
          request_reciprocal_fraction_exponent =
            $signed({2'b00, ~operand_exponent_i}) - 10'sd2;
        end
      end else begin
        if (operand_nan_i)
          request_direct_result[15:0] = FP16_CANONICAL_NAN;
        else if (operand_zero_i)
          request_direct_result[15:0] =
            {operand_sign_i, 5'h1f, 10'b0};
        else if (operand_inf_i)
          request_direct_result[15:0] = {operand_sign_i, 15'b0};
        else begin
          request_lookup = 1'b1;
          request_local_input[15:0] =
            {1'b0, 5'd15, sanitized_operand16[9:0]};
          request_multiplier_sign = operand_sign_i;
          request_reciprocal_dual_exponent = 1'b1;
          request_reciprocal_unity_exponent =
            $signed({5'b0, ~operand_exponent_i[4:0]}) - 10'sd1;
          request_reciprocal_fraction_exponent =
            $signed({5'b0, ~operand_exponent_i[4:0]}) - 10'sd2;
        end
      end
    end else if ((SPECIAL_KIND == VXM_SPECIAL_RECIP_RSQRT) &&
                 (opcode_i == VXM_LOCAL_SPECIAL1)) begin
      request_bank = 2;
      request_rsqrt = 1'b1;
      request_uq_coefficient = 1'b1;
      request_subtract_slope = 1'b1;
      request_rsqrt_newton = data_format_i == VXM_FORMAT_FP32;
      if (data_format_i != VXM_FORMAT_FP16) begin
        if (operand_nan_i || (operand_sign_i && !operand_zero_i))
          request_direct_result = FP32_CANONICAL_NAN;
        else if (operand_zero_i)
          request_direct_result = 32'h7f800000;
        else if (operand_inf_i)
          request_direct_result = 32'b0;
        else begin
          request_lookup = 1'b1;
          normalized_exponent = operand_exponent_i;
          normalized_exponent = normalized_exponent - 127;
          request_local_input =
            {1'b0, 8'd127, working_operand32[22:0]};
          if ((normalized_exponent % 2) != 0) begin
            request_local_input[30:23] = 8'd128;
            normalized_exponent = normalized_exponent - 1;
          end
          request_result_exponent = -(normalized_exponent / 2);
        end
      end else begin
        if (operand_nan_i || (operand_sign_i && !operand_zero_i))
          request_direct_result[15:0] = FP16_CANONICAL_NAN;
        else if (operand_zero_i)
          request_direct_result[15:0] = 16'h7c00;
        else if (operand_inf_i)
          request_direct_result[15:0] = 16'b0;
        else begin
          request_lookup = 1'b1;
          normalized_exponent = operand_exponent_i[4:0] - 15;
          request_local_input[15:0] =
            {1'b0, 5'd15, sanitized_operand16[9:0]};
          if ((normalized_exponent % 2) != 0) begin
            request_local_input[14:10] = 5'd16;
            normalized_exponent = normalized_exponent - 1;
          end
          request_result_exponent = -(normalized_exponent / 2);
        end
      end
    end

    request_configured = 1'b0;
    selected_input_min = 16'b0;
    selected_segment_width = FP16_ONE;
    for (integer bank = 0; bank < LUT_BANK_COUNT; bank++) begin
      if (request_bank == bank) begin
        request_configured = lut_configured_i[bank];
        selected_input_min = lut_input_min_i[bank*16 +: 16];
        selected_segment_width =
          lut_segment_width_i[bank*16 +: 16];
      end
    end
    if (data_format_i != VXM_FORMAT_FP16) begin
      request_input_min = fp16_to_fp32(selected_input_min);
      request_segment_width = fp16_to_fp32(selected_segment_width);
    end else begin
      request_input_min = {16'b0, selected_input_min};
      request_segment_width = {16'b0, selected_segment_width};
    end
  end

  always_comb begin
    lookup_reciprocal_residual = 17'b0;
    lookup_rsqrt_residual = 18'b0;
    if (stage0_reciprocal_q &&
        (stage0_format_q != VXM_FORMAT_FP16)) begin
      // A normalized reciprocal input is 1.f. With 64 uniform segments,
      // the six high fraction bits are the SRAM address and the remaining
      // 17 bits are the exact local dx. BF16 naturally has zeros in the
      // lower FP32 fraction positions after widening.
      lookup_index = stage0_local_input_q[22:17];
      lookup_position = 32'b0;
      lookup_index_fp16 = 16'b0;
      lookup_index_fp32 = 32'b0;
      lookup_x0 = 32'b0;
      lookup_dx = 32'b0;
      if (stage0_format_q == VXM_FORMAT_BF16)
        lookup_reciprocal_residual[0] = stage0_local_input_q[16];
      else
        lookup_reciprocal_residual = stage0_local_input_q[16:0];
    end else if (stage0_reciprocal_q) begin
      lookup_index = stage0_local_input_q[9:4];
      lookup_position = 32'b0;
      lookup_index_fp16 = 16'b0;
      lookup_index_fp32 = 32'b0;
      lookup_x0 = 32'b0;
      lookup_dx = 32'b0;
      lookup_reciprocal_residual[3:0] = stage0_local_input_q[3:0];
    end else if (stage0_rsqrt_q &&
                 (stage0_format_q != VXM_FORMAT_FP16)) begin
      // One address bit selects [1,2) versus [2,4); five fraction bits
      // select one of 32 segments in that parity half of the table.
      lookup_index = {
        (stage0_local_input_q[30:23] == 8'd128),
        stage0_local_input_q[22:18]};
      lookup_position = 32'b0;
      lookup_index_fp16 = 16'b0;
      lookup_index_fp32 = 32'b0;
      lookup_x0 = 32'b0;
      lookup_dx = 32'b0;
      if (stage0_format_q == VXM_FORMAT_BF16)
        lookup_rsqrt_residual[1:0] = stage0_local_input_q[17:16];
      else
        lookup_rsqrt_residual = stage0_local_input_q[17:0];
    end else if (stage0_rsqrt_q) begin
      lookup_index = {
        (stage0_local_input_q[14:10] == 5'd16),
        stage0_local_input_q[9:5]};
      lookup_position = 32'b0;
      lookup_index_fp16 = 16'b0;
      lookup_index_fp32 = 32'b0;
      lookup_x0 = 32'b0;
      lookup_dx = 32'b0;
      lookup_rsqrt_residual[4:0] = stage0_local_input_q[4:0];
    end else if (stage0_format_q != VXM_FORMAT_FP16) begin
      lookup_position = fp32_divide_rne(
        fp32_add_rne(stage0_local_input_q,
          {~stage0_input_min_q[31], stage0_input_min_q[30:0]}),
        stage0_segment_width_q);
      lookup_index = fp32_to_sint(lookup_position);
      if (lookup_position[31])
        lookup_index = 0;
      else if (lookup_index >= LUT_ENTRY_COUNT)
        lookup_index = LUT_ENTRY_COUNT - 1;
      lookup_index_fp32 = uint_to_fp32_exact_local(lookup_index);
      lookup_x0 = fp32_add_rne(
        stage0_input_min_q,
        fp32_multiply_rne(lookup_index_fp32,
                          stage0_segment_width_q));
      lookup_dx = fp32_add_rne(
        stage0_local_input_q, {~lookup_x0[31], lookup_x0[30:0]});
      lookup_index_fp16 = 16'b0;
    end else begin
      lookup_position = 32'b0;
      lookup_position[15:0] = fp16_divide_rne_ftz(
        fp16_sub_rne_ftz(stage0_local_input_q[15:0],
                         stage0_input_min_q[15:0]),
        stage0_segment_width_q[15:0]);
      lookup_index = fp16_to_uint_floor(lookup_position[15:0]);
      if (lookup_position[15])
        lookup_index = 0;
      else if (lookup_index >= LUT_ENTRY_COUNT)
        lookup_index = LUT_ENTRY_COUNT - 1;
      lookup_index_fp16 = uint_to_fp16_rne_ftz(lookup_index);
      lookup_index_fp32 = 32'b0;
      lookup_x0 = 32'b0;
      lookup_x0[15:0] = fp16_add_rne_ftz(
        stage0_input_min_q[15:0],
        fp16_multiply_rne_ftz(lookup_index_fp16,
                              stage0_segment_width_q[15:0]));
      lookup_dx = 32'b0;
      lookup_dx[15:0] = fp16_sub_rne_ftz(
        stage0_local_input_q[15:0], lookup_x0[15:0]);
    end

    lut_read_valid_o = stage0_valid_q && stage0_lookup_q;
    lut_read_bank_o = stage0_bank_q;
    lut_read_address_o = lookup_index[LUT_ADDRESS_WIDTH-1:0];

  end

  always_comb begin
    logic signed [9:0] reciprocal_selected_exponent;

    restored_wide = 32'b0;
    reciprocal_selected_exponent = 10'sd0;
    if (add_reciprocal_dual_exponent_q &&
        (add_format_q != VXM_FORMAT_FP16)) begin
      // The fixed RECIP interpolator has already applied RNE. A 1.x seed
      // selects the unity candidate; a 0.5..1 seed selects the fraction
      // candidate. BF16 is rounded after exponent insertion, so a mantissa
      // carry advances the fraction candidate into unity naturally.
      reciprocal_selected_exponent = add_result_q[30:23] >= 8'd127 ?
        add_reciprocal_unity_exponent_q :
        add_reciprocal_fraction_exponent_q;
      if ((add_result_q[30:0] == 0) ||
          (reciprocal_selected_exponent <= 10'sd0))
        restored_wide = 32'b0;
      else if (reciprocal_selected_exponent >= 10'sd255)
        restored_wide = 32'h7f800000;
      else
        restored_wide =
          {1'b0, reciprocal_selected_exponent[7:0], add_result_q[22:0]};
      if (add_multiplier_sign_q)
        restored_wide[31] = 1'b1;
      restored_result = add_format_q == VXM_FORMAT_BF16 ?
        {16'b0, fp32_to_bf16_ftz(restored_wide)} : restored_wide;
    end else if (add_reciprocal_dual_exponent_q) begin
      reciprocal_selected_exponent = add_result_q[14:10] >= 5'd15 ?
        add_reciprocal_unity_exponent_q :
        add_reciprocal_fraction_exponent_q;
      restored_result = 32'b0;
      if ((add_result_q[14:0] == 0) ||
          (reciprocal_selected_exponent <= 10'sd0))
        restored_result[15:0] = 16'b0;
      else if (reciprocal_selected_exponent >= 10'sd31)
        restored_result[15:0] = 16'h7c00;
      else
        restored_result[15:0] =
          {1'b0, reciprocal_selected_exponent[4:0], add_result_q[9:0]};
      if (add_multiplier_sign_q)
        restored_result[15] = 1'b1;
    end else if (add_format_q != VXM_FORMAT_FP16) begin
      restored_wide = fp32_scale_pow2_ftz_local(
        add_result_q, add_result_exponent_q);
      if (add_multiplier_sign_q)
        restored_wide[31] = ~restored_wide[31];
      restored_result = add_format_q == VXM_FORMAT_BF16 ?
        {16'b0, fp32_to_bf16_ftz(restored_wide)} : restored_wide;
    end else begin
      restored_result = 32'b0;
      restored_result[15:0] = fp16_scale_pow2_ftz(
        add_result_q[15:0], add_result_exponent_q);
      if (add_multiplier_sign_q)
        restored_result[15] = ~restored_result[15];
    end
  end

  always_comb begin
    decoded_lut_k = 32'b0;
    decoded_lut_b = 32'b0;
    decoded_exp_base_uq = {lut_read_k_i[12:0], lut_read_b_i[12:0]};
    if (read_meta_exp_base_q) begin
      if (read_meta_format_q == VXM_FORMAT_FP16) begin
        decoded_lut_b[15:0] = fp32_to_fp16(
          uq1_25_to_fp32_local(decoded_exp_base_uq));
        // Linear FP16 mode evaluates B + B*delta.
        decoded_lut_k = decoded_lut_b;
      end else begin
        decoded_lut_b = uq1_25_to_fp32_local(decoded_exp_base_uq);
        // BF16 linear mode uses B as its slope. FP32 cubic mode overrides
        // the multiplier input below but carries this value to the final
        // scale multiplication.
        decoded_lut_k = decoded_lut_b;
      end
    end else if (read_meta_format_q == VXM_FORMAT_FP16) begin
      decoded_lut_k = {16'b0, lut_read_k_i};
      decoded_lut_b = {16'b0, lut_read_b_i};
    end else begin
      decoded_lut_k = fp16_to_fp32(lut_read_k_i);
      decoded_lut_b = fp16_to_fp32(lut_read_b_i);
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stage0_valid_q <= 1'b0;
      stage0_lookup_q <= 1'b0;
      stage0_format_q <= VXM_FORMAT_FP16;
      stage0_bank_q <= '0;
      stage0_local_input_q <= 32'b0;
      stage0_direct_result_q <= 32'b0;
      stage0_multiplier_sign_q <= 1'b0;
      stage0_result_exponent_q <= '0;
      stage0_input_min_q <= 32'b0;
      stage0_segment_width_q <= {16'b0, FP16_ONE};
      stage0_reciprocal_q <= 1'b0;
      stage0_rsqrt_q <= 1'b0;
      stage0_uq_coefficient_q <= 1'b0;
      stage0_subtract_slope_q <= 1'b0;
      stage0_exp_base_q <= 1'b0;
      stage0_exp_cubic_q <= 1'b0;
      stage0_reciprocal_newton_q <= 1'b0;
      stage0_rsqrt_newton_q <= 1'b0;
      stage0_reciprocal_dual_exponent_q <= 1'b0;
      stage0_reciprocal_unity_exponent_q <= '0;
      stage0_reciprocal_fraction_exponent_q <= '0;
      read_meta_valid_q <= 1'b0;
      read_meta_lookup_q <= 1'b0;
      read_meta_format_q <= VXM_FORMAT_FP16;
      read_meta_dx_q <= 32'b0;
      read_meta_direct_result_q <= 32'b0;
      read_meta_multiplier_sign_q <= 1'b0;
      read_meta_result_exponent_q <= '0;
      read_meta_uq_coefficient_q <= 1'b0;
      read_meta_subtract_slope_q <= 1'b0;
      read_meta_exp_base_q <= 1'b0;
      read_meta_exp_cubic_q <= 1'b0;
      read_meta_reciprocal_newton_q <= 1'b0;
      read_meta_reciprocal_m_q <= 32'b0;
      read_meta_reciprocal_residual_q <= 17'b0;
      read_meta_rsqrt_q <= 1'b0;
      read_meta_rsqrt_newton_q <= 1'b0;
      read_meta_rsqrt_residual_q <= 18'b0;
      read_meta_reciprocal_dual_exponent_q <= 1'b0;
      read_meta_reciprocal_unity_exponent_q <= '0;
      read_meta_reciprocal_fraction_exponent_q <= '0;
      multiply_valid_q <= 1'b0;
      multiply_lookup_q <= 1'b0;
      multiply_format_q <= VXM_FORMAT_FP16;
      multiply_product_q <= 32'b0;
      multiply_b_q <= 32'b0;
      multiply_direct_result_q <= 32'b0;
      multiply_multiplier_sign_q <= 1'b0;
      multiply_result_exponent_q <= '0;
      multiply_subtract_slope_q <= 1'b0;
      multiply_exp_cubic_q <= 1'b0;
      multiply_exp_dx_q <= 32'b0;
      multiply_exp_base_uq_q <= 26'b0;
      multiply_reciprocal_newton_q <= 1'b0;
      multiply_reciprocal_fixed_q <= 1'b0;
      multiply_rsqrt_newton_q <= 1'b0;
      multiply_reciprocal_m_q <= 32'b0;
      multiply_reciprocal_dual_exponent_q <= 1'b0;
      multiply_reciprocal_unity_exponent_q <= '0;
      multiply_reciprocal_fraction_exponent_q <= '0;
      add_valid_q <= 1'b0;
      add_format_q <= VXM_FORMAT_FP16;
      add_result_q <= 32'b0;
      add_multiplier_sign_q <= 1'b0;
      add_result_exponent_q <= '0;
      add_exp_cubic_q <= 1'b0;
      add_exp_dx_q <= 32'b0;
      add_exp_base_uq_q <= 26'b0;
      add_reciprocal_newton_q <= 1'b0;
      add_rsqrt_newton_q <= 1'b0;
      add_reciprocal_m_q <= 32'b0;
      add_reciprocal_dual_exponent_q <= 1'b0;
      add_reciprocal_unity_exponent_q <= '0;
      add_reciprocal_fraction_exponent_q <= '0;
      exp_horner1_valid_q <= 1'b0;
      exp_horner1_result_q <= 32'b0;
      exp_horner1_dx_q <= 32'b0;
      exp_horner1_base_uq_q <= 26'b0;
      exp_horner1_exponent_q <= '0;
      exp_horner2_valid_q <= 1'b0;
      exp_horner2_result_q <= 32'b0;
      exp_horner2_base_uq_q <= 26'b0;
      exp_horner2_exponent_q <= '0;
      exp_base_valid_q <= 1'b0;
      exp_base_result_q <= 32'b0;
      exp_base_exponent_q <= '0;
      reciprocal_multiply_valid_q <= 1'b0;
      reciprocal_xy_q <= 32'b0;
      reciprocal_seed_q <= 32'b0;
      reciprocal_multiply_sign_q <= 1'b0;
      reciprocal_multiply_unity_exponent_q <= '0;
      reciprocal_multiply_fraction_exponent_q <= '0;
      reciprocal_correction_valid_q <= 1'b0;
      reciprocal_correction_q <= 32'b0;
      reciprocal_correction_seed_q <= 32'b0;
      reciprocal_correction_sign_q <= 1'b0;
      reciprocal_correction_unity_exponent_q <= '0;
      reciprocal_correction_fraction_exponent_q <= '0;
      reciprocal_newton_valid_q <= 1'b0;
      reciprocal_newton_result_q <= 32'b0;
      reciprocal_newton_sign_q <= 1'b0;
      reciprocal_newton_unity_exponent_q <= '0;
      reciprocal_newton_fraction_exponent_q <= '0;
      rsqrt_square_valid_q <= 1'b0;
      rsqrt_y_squared_q <= 32'b0;
      rsqrt_square_seed_q <= 32'b0;
      rsqrt_square_m_q <= 32'b0;
      rsqrt_square_exponent_q <= '0;
      rsqrt_correction_valid_q <= 1'b0;
      rsqrt_correction_q <= 32'b0;
      rsqrt_correction_seed_q <= 32'b0;
      rsqrt_correction_exponent_q <= '0;
      rsqrt_newton_valid_q <= 1'b0;
      rsqrt_newton_result_q <= 32'b0;
      rsqrt_newton_exponent_q <= '0;
      bypass1_valid_q <= 1'b0;
      bypass1_result_q <= 32'b0;
      bypass2_valid_q <= 1'b0;
      bypass2_result_q <= 32'b0;
      bypass3_valid_q <= 1'b0;
      bypass3_result_q <= 32'b0;
      valid_o <= 1'b0;
      result_o <= 32'b0;
      lut_fault_o <= 1'b0;
    end else begin
      lut_fault_o <=
        (reciprocal_linear_enable && reciprocal_linear_fault) ||
        (rsqrt_linear_enable && rsqrt_linear_fault) ||
        (rsqrt_square_valid_q && rsqrt_correction_fault);

      valid_o <= exp_base_valid_q || reciprocal_newton_valid_q ||
                 rsqrt_newton_valid_q || bypass3_valid_q;
      if (rsqrt_newton_valid_q)
        result_o <= rsqrt_newton_restored_result;
      else if (reciprocal_newton_valid_q)
        result_o <= reciprocal_newton_restored_result;
      else if (exp_base_valid_q)
        result_o <= exp_restored_result;
      else if (bypass3_valid_q)
        result_o <= bypass3_result_q;

      // Low-precision and non-iterative special operations cross three
      // operand-isolated bypass registers so every Lane keeps the same
      // externally visible latency as FP32 EXP/RECIP/RSQRT.
      bypass3_valid_q <= bypass2_valid_q;
      if (bypass2_valid_q)
        bypass3_result_q <= bypass2_result_q;
      bypass2_valid_q <= bypass1_valid_q;
      if (bypass1_valid_q)
        bypass2_result_q <= bypass1_result_q;
      bypass1_valid_q <= add_valid_q && !add_exp_cubic_q &&
        !add_reciprocal_newton_q && !add_rsqrt_newton_q;
      if (add_valid_q && !add_exp_cubic_q &&
          !add_reciprocal_newton_q && !add_rsqrt_newton_q)
        bypass1_result_q <= restored_result;

      // FP32 RSQRT Newton pipeline:
      //   y_squared = y0*y0
      //   correction = 1.5 - 0.5*m*y_squared (fused, one RNE)
      //   y1 = y0*correction
      rsqrt_newton_valid_q <= rsqrt_correction_valid_q;
      if (rsqrt_correction_valid_q) begin
        rsqrt_newton_result_q <= rsqrt_final_product;
        rsqrt_newton_exponent_q <= rsqrt_correction_exponent_q;
      end

      rsqrt_correction_valid_q <= rsqrt_square_valid_q;
      if (rsqrt_square_valid_q) begin
        rsqrt_correction_q <= rsqrt_correction_value;
        rsqrt_correction_seed_q <= rsqrt_square_seed_q;
        rsqrt_correction_exponent_q <= rsqrt_square_exponent_q;
      end

      rsqrt_square_valid_q <= add_valid_q && add_rsqrt_newton_q;
      if (add_valid_q && add_rsqrt_newton_q) begin
        rsqrt_y_squared_q <= rsqrt_square_product;
        rsqrt_square_seed_q <= add_result_q;
        rsqrt_square_m_q <= add_reciprocal_m_q;
        rsqrt_square_exponent_q <= add_result_exponent_q;
      end

      // One FP32 reciprocal Newton step: y1 = y0 * (2 - m*y0).
      reciprocal_newton_valid_q <= reciprocal_correction_valid_q;
      if (reciprocal_correction_valid_q) begin
        reciprocal_newton_result_q <= reciprocal_newton_stage2_product;
        reciprocal_newton_sign_q <= reciprocal_correction_sign_q;
        reciprocal_newton_unity_exponent_q <=
          reciprocal_correction_unity_exponent_q;
        reciprocal_newton_fraction_exponent_q <=
          reciprocal_correction_fraction_exponent_q;
      end

      reciprocal_correction_valid_q <= reciprocal_multiply_valid_q;
      if (reciprocal_multiply_valid_q) begin
        reciprocal_correction_q <= fp32_add_rne(
          FP32_TWO, {~reciprocal_xy_q[31], reciprocal_xy_q[30:0]});
        reciprocal_correction_seed_q <= reciprocal_seed_q;
        reciprocal_correction_sign_q <= reciprocal_multiply_sign_q;
        reciprocal_correction_unity_exponent_q <=
          reciprocal_multiply_unity_exponent_q;
        reciprocal_correction_fraction_exponent_q <=
          reciprocal_multiply_fraction_exponent_q;
      end

      reciprocal_multiply_valid_q <=
        add_valid_q && add_reciprocal_newton_q;
      if (add_valid_q && add_reciprocal_newton_q) begin
        reciprocal_xy_q <= reciprocal_newton_stage1_product;
        reciprocal_seed_q <= add_result_q;
        reciprocal_multiply_sign_q <= add_multiplier_sign_q;
        reciprocal_multiply_unity_exponent_q <=
          add_reciprocal_unity_exponent_q;
        reciprocal_multiply_fraction_exponent_q <=
          add_reciprocal_fraction_exponent_q;
      end

      // FP32 EXP uses a fully registered cubic Horner continuation after
      // the common LUT multiply/add stages. Each stage is kept explicit so
      // it can later map to one fused multiply-add macro without changing
      // the surrounding protocol.
      exp_base_valid_q <= exp_horner2_valid_q;
      if (exp_horner2_valid_q) begin
        exp_base_result_q <= special_coefficient_product;
        exp_base_exponent_q <= exp_horner2_exponent_q;
      end

      exp_horner2_valid_q <= exp_horner1_valid_q;
      if (exp_horner1_valid_q) begin
        exp_horner2_result_q <= fp32_add_rne(
          fp32_multiply_rne(exp_horner1_result_q,
                            exp_horner1_dx_q),
          FP32_ONE);
        exp_horner2_base_uq_q <= exp_horner1_base_uq_q;
        exp_horner2_exponent_q <= exp_horner1_exponent_q;
      end

      exp_horner1_valid_q <= add_valid_q && add_exp_cubic_q;
      if (add_valid_q && add_exp_cubic_q) begin
        exp_horner1_result_q <= fp32_add_rne(
          fp32_multiply_rne(add_result_q, add_exp_dx_q),
          FP32_ONE);
        exp_horner1_dx_q <= add_exp_dx_q;
        exp_horner1_base_uq_q <= add_exp_base_uq_q;
        exp_horner1_exponent_q <= add_result_exponent_q;
      end

      add_valid_q <= multiply_valid_q;
      add_format_q <= multiply_format_q;
      if (multiply_reciprocal_fixed_q) begin
        // RECIP's fixed-point interpolation has already completed b-k*dx.
        // Carry y0 through the common exponent-restoration stage without
        // rebuilding the operation in the floating-point add datapath.
        add_result_q <= multiply_b_q;
      end else if (multiply_lookup_q) begin
        if (multiply_format_q != VXM_FORMAT_FP16) begin
          if (multiply_subtract_slope_q)
            add_result_q <= fp32_add_rne(
              multiply_b_q,
              {~multiply_product_q[31], multiply_product_q[30:0]});
          else
            add_result_q <= fp32_add_rne(
              multiply_product_q, multiply_b_q);
        end else begin
          if (multiply_subtract_slope_q)
            add_result_q <= {16'b0, fp16_sub_rne_ftz(
              multiply_b_q[15:0], multiply_product_q[15:0])};
          else
            add_result_q <= {16'b0, fp16_add_rne_ftz(
              multiply_product_q[15:0], multiply_b_q[15:0])};
        end
      end else
        add_result_q <= multiply_direct_result_q;
      add_multiplier_sign_q <= multiply_multiplier_sign_q;
      add_result_exponent_q <= multiply_result_exponent_q;
      add_exp_cubic_q <= multiply_exp_cubic_q;
      add_exp_dx_q <= multiply_exp_dx_q;
      add_exp_base_uq_q <= multiply_exp_base_uq_q;
      add_reciprocal_newton_q <= multiply_reciprocal_newton_q;
      add_rsqrt_newton_q <= multiply_rsqrt_newton_q;
      add_reciprocal_m_q <= multiply_reciprocal_m_q;
      add_reciprocal_dual_exponent_q <=
        multiply_reciprocal_dual_exponent_q;
      add_reciprocal_unity_exponent_q <=
        multiply_reciprocal_unity_exponent_q;
      add_reciprocal_fraction_exponent_q <=
        multiply_reciprocal_fraction_exponent_q;

      multiply_valid_q <= read_meta_valid_q &&
        (!read_meta_lookup_q || lut_read_valid_i);
      multiply_lookup_q <= read_meta_lookup_q;
      multiply_format_q <= read_meta_format_q;
      if (read_meta_format_q != VXM_FORMAT_FP16) begin
        if (read_meta_exp_cubic_q) begin
          multiply_product_q <= fp32_multiply_rne(
            FP32_ONE_SIXTH, read_meta_dx_q);
          multiply_b_q <= FP32_HALF;
        end else if (read_meta_exp_base_q) begin
          multiply_product_q <= special_coefficient_product;
          multiply_b_q <= special_coefficient_value;
        end else if (read_meta_uq_coefficient_q) begin
          multiply_product_q <= 32'b0;
          multiply_b_q <= read_meta_rsqrt_q ?
            rsqrt_linear_y0 : reciprocal_linear_y0;
        end else begin
          multiply_product_q <= fp32_multiply_rne(
            decoded_lut_k, read_meta_dx_q);
          multiply_b_q <= decoded_lut_b;
        end
      end else begin
        if (read_meta_exp_base_q) begin
          multiply_product_q <= special_coefficient_product;
          multiply_b_q <= special_coefficient_value;
        end else if (read_meta_uq_coefficient_q) begin
          multiply_product_q <= 32'b0;
          multiply_b_q <= read_meta_rsqrt_q ?
            rsqrt_linear_y0 : reciprocal_linear_y0;
        end else begin
          multiply_product_q <= {16'b0, fp16_multiply_rne_ftz(
            decoded_lut_k[15:0], read_meta_dx_q[15:0])};
          multiply_b_q <= decoded_lut_b;
        end
      end
      multiply_direct_result_q <= read_meta_direct_result_q;
      multiply_multiplier_sign_q <= read_meta_multiplier_sign_q;
      multiply_result_exponent_q <= read_meta_result_exponent_q;
      multiply_subtract_slope_q <= read_meta_subtract_slope_q;
      multiply_exp_cubic_q <= read_meta_exp_cubic_q;
      multiply_exp_dx_q <= read_meta_dx_q;
      multiply_exp_base_uq_q <= decoded_exp_base_uq;
      multiply_reciprocal_newton_q <= read_meta_reciprocal_newton_q;
      multiply_reciprocal_fixed_q <= read_meta_uq_coefficient_q;
      multiply_rsqrt_newton_q <= read_meta_rsqrt_newton_q;
      multiply_reciprocal_m_q <= read_meta_reciprocal_m_q;
      multiply_reciprocal_dual_exponent_q <=
        read_meta_reciprocal_dual_exponent_q;
      multiply_reciprocal_unity_exponent_q <=
        read_meta_reciprocal_unity_exponent_q;
      multiply_reciprocal_fraction_exponent_q <=
        read_meta_reciprocal_fraction_exponent_q;
      // Direct results advance immediately. LUT results hold their metadata
      // until the one-cycle Lane SRAM service returns the tagged response.
      if (read_meta_valid_q &&
          (!read_meta_lookup_q || lut_read_valid_i))
        read_meta_valid_q <= 1'b0;

      if (stage0_valid_q) begin
        read_meta_valid_q <= 1'b1;
        read_meta_lookup_q <= stage0_lookup_q;
        read_meta_format_q <= stage0_format_q;
        read_meta_dx_q <= lookup_dx;
        read_meta_direct_result_q <= stage0_direct_result_q;
        read_meta_multiplier_sign_q <= stage0_multiplier_sign_q;
        read_meta_result_exponent_q <= stage0_result_exponent_q;
        read_meta_uq_coefficient_q <= stage0_uq_coefficient_q;
        read_meta_subtract_slope_q <= stage0_subtract_slope_q;
        read_meta_exp_base_q <= stage0_exp_base_q;
        read_meta_exp_cubic_q <= stage0_exp_cubic_q;
        read_meta_reciprocal_newton_q <= stage0_reciprocal_newton_q;
        read_meta_reciprocal_m_q <= stage0_local_input_q;
        read_meta_reciprocal_residual_q <= lookup_reciprocal_residual;
        read_meta_rsqrt_q <= stage0_rsqrt_q;
        read_meta_rsqrt_newton_q <= stage0_rsqrt_newton_q;
        read_meta_rsqrt_residual_q <= lookup_rsqrt_residual;
        read_meta_reciprocal_dual_exponent_q <=
          stage0_reciprocal_dual_exponent_q;
        read_meta_reciprocal_unity_exponent_q <=
          stage0_reciprocal_unity_exponent_q;
        read_meta_reciprocal_fraction_exponent_q <=
          stage0_reciprocal_fraction_exponent_q;
        stage0_valid_q <= 1'b0;
      end

      if (valid_i && operand_format_valid_i && supported_opcode && ready_o &&
          (opcode_i != VXM_LOCAL_LOG)) begin
        stage0_valid_q <= 1'b1;
        stage0_lookup_q <= request_lookup && request_configured;
        stage0_format_q <= data_format_i;
        stage0_bank_q <= request_bank;
        stage0_local_input_q <= request_local_input;
        stage0_direct_result_q <= request_lookup && !request_configured ?
          (data_format_i == VXM_FORMAT_FP16 ?
            {16'b0, FP16_CANONICAL_NAN} : FP32_CANONICAL_NAN) :
          request_direct_result;
        stage0_multiplier_sign_q <= request_multiplier_sign;
        stage0_result_exponent_q <= request_result_exponent;
        stage0_input_min_q <= request_input_min;
        stage0_segment_width_q <= request_segment_width;
        stage0_reciprocal_q <= request_reciprocal;
        stage0_rsqrt_q <= request_rsqrt;
        stage0_uq_coefficient_q <= request_uq_coefficient &&
          request_lookup && request_configured;
        stage0_subtract_slope_q <= request_subtract_slope;
        stage0_exp_base_q <= request_exp_base;
        stage0_exp_cubic_q <= request_exp_cubic && request_lookup &&
          request_configured;
        stage0_reciprocal_newton_q <= request_reciprocal_newton &&
          request_lookup && request_configured;
        stage0_rsqrt_newton_q <= request_rsqrt_newton &&
          request_lookup && request_configured;
        stage0_reciprocal_dual_exponent_q <=
          request_reciprocal_dual_exponent && request_lookup &&
          request_configured;
        stage0_reciprocal_unity_exponent_q <=
          request_reciprocal_unity_exponent;
        stage0_reciprocal_fraction_exponent_q <=
          request_reciprocal_fraction_exponent;
        if (request_lookup && !request_configured)
          lut_fault_o <= 1'b1;
      end
    end
  end
endmodule
