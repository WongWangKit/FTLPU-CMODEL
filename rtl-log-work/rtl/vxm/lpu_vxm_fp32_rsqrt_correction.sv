// Fused FP32 RSQRT Newton correction:
//   correction = 1.5 - 0.5*m*y_squared
// m and y_squared are positive finite normal FP32 values on this internal
// path. Their 24x24 significand product remains at full 48-bit precision,
// is aligned with 1.5 in Q55, subtracted, and rounded to FP32 only once.
module lpu_vxm_fp32_rsqrt_correction (
  input  logic        enable_i,
  input  logic [31:0] m_i,
  input  logic [31:0] y_squared_i,
  output logic [31:0] correction_o,
  output logic        range_fault_o
);
  function automatic logic [31:0] q55_to_fp32_rne(
    input logic [63:0] magnitude
  );
    integer leading_bit;
    integer shift;
    integer exponent;
    logic [63:0] retained;
    logic [63:0] remainder;
    logic [63:0] remainder_mask;
    logic [63:0] halfway;
    begin
      leading_bit = -1;
      retained = 64'b0;
      remainder = 64'b0;
      remainder_mask = 64'b0;
      halfway = 64'b0;
      for (integer bit_index = 0; bit_index < 64; bit_index++)
        if (magnitude[bit_index])
          leading_bit = bit_index;

      if (leading_bit < 0) begin
        q55_to_fp32_rne = 32'b0;
      end else begin
        exponent = 127 + leading_bit - 55;
        if (leading_bit <= 23) begin
          retained = magnitude << (23-leading_bit);
        end else begin
          shift = leading_bit - 23;
          retained = magnitude >> shift;
          remainder_mask = (64'b1 << shift) - 1'b1;
          remainder = magnitude & remainder_mask;
          halfway = 64'b1 << (shift-1);
          if ((remainder > halfway) ||
              ((remainder == halfway) && retained[0]))
            retained = retained + 1'b1;
          if (retained[24]) begin
            retained = retained >> 1;
            exponent = exponent + 1;
          end
        end
        if (exponent <= 0)
          q55_to_fp32_rne = 32'b0;
        else if (exponent >= 255)
          q55_to_fp32_rne = 32'h7f800000;
        else
          q55_to_fp32_rne =
            {1'b0, exponent[7:0], retained[22:0]};
      end
    end
  endfunction

  logic [23:0] m_significand;
  logic [23:0] y_squared_significand;
  logic [47:0] significand_product;
  logic [63:0] product_q55;
  logic [63:0] one_point_five_q55;
  logic signed [64:0] difference;
  integer product_shift;
  integer exponent_sum;

  always_comb begin
    m_significand = {1'b1, m_i[22:0]};
    y_squared_significand = {1'b1, y_squared_i[22:0]};
    significand_product = m_significand * y_squared_significand;
    exponent_sum = $signed({1'b0, m_i[30:23]}) - 127 +
                   $signed({1'b0, y_squared_i[30:23]}) - 127;
    // Product has 46 fractional significand bits. The additional -1 is
    // the exact factor 0.5 in the Newton equation; Q55 adds 55 places.
    product_shift = exponent_sum + 8;
    product_q55 = 64'b0;
    if (product_shift >= 0 && product_shift <= 16)
      product_q55 = {16'b0, significand_product} << product_shift;
    else if (product_shift < 0 && product_shift >= -47)
      product_q55 = {16'b0, significand_product} >> (-product_shift);

    one_point_five_q55 = 64'd3 << 54;
    difference = $signed({1'b0, one_point_five_q55}) -
                 $signed({1'b0, product_q55});
    range_fault_o = enable_i &&
      ((m_i[30:23] == 0) || (m_i[30:23] == 8'hff) ||
       (y_squared_i[30:23] == 0) ||
       (y_squared_i[30:23] == 8'hff) || (difference <= 0));
    correction_o = 32'b0;
    if (enable_i && !range_fault_o)
      correction_o = q55_to_fp32_rne($unsigned(difference[63:0]));
  end
endmodule
