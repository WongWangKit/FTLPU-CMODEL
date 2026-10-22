// One logical left shifter shared by FP16, BF16 and FP32 normalization.
// The format mask isolates unused high input bits and truncates the result.
module lpu_vxm_shared_left_shift (
  input  logic        enable_i,
  input  logic [26:0] active_mask_i,
  input  logic [26:0] value_i,
  input  logic [4:0]  shift_i,
  output logic [26:0] result_o
);
  logic [26:0] isolated_value;
  logic [26:0] shifted_value;

  assign isolated_value = enable_i ? (value_i & active_mask_i) : 27'b0;
  assign shifted_value = isolated_value << shift_i;
  assign result_o = shifted_value & active_mask_i;
endmodule
