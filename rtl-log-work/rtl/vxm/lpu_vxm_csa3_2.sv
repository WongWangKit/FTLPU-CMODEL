// Three-to-two carry-save compressor.  carry_o is already shifted left by
// one bit, so the represented value is sum_o + carry_o = a_i + b_i + c_i.
// No horizontal carry propagation occurs inside this module.
module lpu_vxm_csa3_2 #(
  parameter integer WIDTH = 48
) (
  input  logic [WIDTH-1:0] a_i,
  input  logic [WIDTH-1:0] b_i,
  input  logic [WIDTH-1:0] c_i,
  output logic [WIDTH-1:0] sum_o,
  output logic [WIDTH-1:0] carry_o
);
  logic [WIDTH-1:0] carry_unshifted;

  always_comb begin
    sum_o = a_i ^ b_i ^ c_i;
    carry_unshifted = (a_i & b_i) | (a_i & c_i) | (b_i & c_i);
    carry_o = carry_unshifted << 1;
  end
endmodule
