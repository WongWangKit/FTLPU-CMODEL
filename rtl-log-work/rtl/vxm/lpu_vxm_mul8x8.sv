module lpu_vxm_mul8x8 (
  input  logic        enable_i,
  input  logic [7:0]  lhs_i,
  input  logic [7:0]  rhs_i,
  output logic [15:0] product_o
);
  logic [7:0] gated_lhs;
  logic [7:0] gated_rhs;

  // Keep the sub-multiplier boundary explicit, but leave its internal
  // Array/Booth/Wallace/Dadda implementation to synthesis for now. Operand
  // isolation prevents a disabled block from switching internally.
  always_comb begin
    gated_lhs = enable_i ? lhs_i : 8'b0;
    gated_rhs = enable_i ? rhs_i : 8'b0;
    product_o = gated_lhs * gated_rhs;
  end
endmodule
