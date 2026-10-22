module lpu_mxm_dot_bank (
  input  logic format_bf16_i,
  input  logic [8*16-1:0] activation_bits_i,
  input  logic [4*8*8*16-1:0] weight_bits_i,
  output logic [32*32-1:0] partial_values_o
);
  import lpu_vxm_math_pkg::*;

  localparam integer BLOCKS = 4;
  localparam integer LANES = 8;

  function automatic integer weight_index(
    input integer block,
    input integer lane,
    input integer column
  );
    weight_index = ((block*LANES+lane)*LANES+column)*16;
  endfunction

  function automatic logic [31:0] decode_16(
    input logic [15:0] bits,
    input logic bf16
  );
    decode_16 = bf16 ? {bits, 16'b0} : fp16_to_fp32(bits);
  endfunction

  always_comb begin
    partial_values_o = '0;
    for (integer block = 0; block < BLOCKS; block++) begin
      for (integer column = 0; column < LANES; column++) begin
        logic [31:0] partial;
        partial = '0;
        for (integer lane = 0; lane < LANES; lane++) begin
          partial = fp32_add_rne(
            partial,
            fp32_multiply_rne(
              decode_16(
                activation_bits_i[lane*16 +: 16], format_bf16_i),
              decode_16(
                weight_bits_i[
                  weight_index(block, lane, column) +: 16],
                format_bf16_i)));
        end
        partial_values_o[(block*LANES+column)*32 +: 32] = partial;
      end
    end
  end
endmodule
