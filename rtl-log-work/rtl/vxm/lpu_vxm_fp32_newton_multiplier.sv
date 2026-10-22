// One combinational FP32 Newton multiply stage. Finite operands use one
// inferred unsigned 24x24 significand multiply inside fp32_multiply_rne.
// Two instances keep the two Newton products on separate dedicated 24x24
// significand datapaths:
//   stage 1: m*y0
//   stage 2: y0*(2-m*y0)
// The current special-ALU issue controller remains single-outstanding; its
// admission policy can be relaxed separately. The module boundary keeps the
// two physical demand points explicit for DC.
module lpu_vxm_fp32_newton_multiplier (
  input  logic        enable_i,
  input  logic [31:0] lhs_i,
  input  logic [31:0] rhs_i,
  output logic [31:0] product_o
);
  import lpu_vxm_math_pkg::*;

  always_comb begin
    product_o = 32'b0;
    if (enable_i)
      product_o = fp32_multiply_rne(lhs_i, rhs_i);
  end
endmodule
