module lpu_mxm_dequantizer (
  input  logic [8*8*8-1:0] quantized_i,
  input  logic [15:0] scale_bf16_i,
  output logic [8*8*16-1:0] weight_bf16_o
);
  import lpu_vxm_math_pkg::*;

  function automatic logic [31:0] int8_to_fp32(
    input logic [7:0] quantized
  );
    logic [8:0] magnitude;
    logic [23:0] significand;
    logic [7:0] exponent;
    integer leading_bit;
    begin
      if (quantized == 8'h00) begin
        int8_to_fp32 = 32'h00000000;
      end else begin
        magnitude = quantized[7]
          ? (9'd256 - {1'b0, quantized})
          : {1'b0, quantized};
        leading_bit = 0;
        for (integer bit_index = 0; bit_index < 8; bit_index++)
          if (magnitude[bit_index])
            leading_bit = bit_index;
        significand = {15'b0, magnitude};
        significand = significand << (23-leading_bit);
        exponent = 127 + leading_bit;
        int8_to_fp32 = {
          quantized[7],
          exponent,
          significand[22:0]
        };
      end
    end
  endfunction

  always_comb begin
    for (integer value = 0; value < 64; value++) begin
      logic [31:0] quantized_fp32;
      logic [31:0] scaled_fp32;
      quantized_fp32 = int8_to_fp32(quantized_i[value*8 +: 8]);
      scaled_fp32 = fp32_multiply_rne(
        quantized_fp32, {scale_bf16_i, 16'b0});
      weight_bf16_o[value*16 +: 16] = fp32_to_bf16(scaled_fp32);
    end
  end
endmodule
