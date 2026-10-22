// One shared 27-bit significand add/subtract datapath, split at the FP16
// boundary: 14 low bits and 13 high bits. The extra result bit is the carry,
// not a 28th full-adder bit. BF16/FP16 isolate the high group completely.
module lpu_vxm_segmented_addsub (
  input  logic        enable_i,
  input  logic        subtract_i,
  input  logic        high_enable_i,
  input  logic [26:0] active_mask_i,
  input  logic [26:0] lhs_i,
  input  logic [26:0] rhs_i,
  output logic [27:0] result_o
);
  logic [26:0] lhs_gated;
  logic [26:0] rhs_gated;
  logic [26:0] rhs_addend;
  logic [14:0] low_sum;
  logic [13:0] high_sum;

  always_comb begin
    lhs_gated = enable_i ? (lhs_i & active_mask_i) : 27'b0;
    rhs_gated = enable_i ? (rhs_i & active_mask_i) : 27'b0;
    rhs_addend = (enable_i && subtract_i) ?
      (rhs_gated ^ active_mask_i) : rhs_gated;

    low_sum = 15'b0;
    high_sum = 14'b0;
    result_o = 28'b0;
    if (enable_i) begin
      low_sum = {1'b0, lhs_gated[13:0]} +
        {1'b0, rhs_addend[13:0]} + {14'b0, subtract_i};
      result_o[13:0] = low_sum[13:0];
      if (high_enable_i) begin
        high_sum = {1'b0, lhs_gated[26:14]} +
          {1'b0, rhs_addend[26:14]} + {13'b0, low_sum[14]};
        result_o[27:14] = high_sum;
      end else begin
        // For FP16 this is the carry above bit 13. For BF16, the carry
        // above bit 10 appears in result_o[11] because bits 11:13 are masked.
        result_o[14] = low_sum[14];
      end
    end
  end
endmodule
