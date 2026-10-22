// Parallel one-hot decode for the six Basic VXM operations.
module lpu_vxm_basic_opcode_decode (
  input  logic [2:0] opcode_i,
  output logic       opcode_valid_o,
  output logic       bypass_o,
  output logic       add_o,
  output logic       subtract_o,
  output logic       multiply_o,
  output logic       negate_o,
  output logic       max_o
);
  import lpu_pkg::*;

  always_comb begin
    opcode_valid_o = 1'b1;
    bypass_o = 1'b0;
    add_o = 1'b0;
    subtract_o = 1'b0;
    multiply_o = 1'b0;
    negate_o = 1'b0;
    max_o = 1'b0;

    case (opcode_i)
      VXM_LOCAL_BYPASS:   bypass_o = 1'b1;
      VXM_LOCAL_ADD:      add_o = 1'b1;
      VXM_LOCAL_SUBTRACT: subtract_o = 1'b1;
      VXM_LOCAL_MULTIPLY: multiply_o = 1'b1;
      VXM_LOCAL_NEGATE:   negate_o = 1'b1;
      VXM_LOCAL_MAX:      max_o = 1'b1;
      default: opcode_valid_o = 1'b0;
    endcase
  end
endmodule
