module lpu_vxm_basic_alu #(parameter bit UNIFORM_TWO_CYCLE=0) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [1:0]  data_format_i,
  input  logic [2:0]  opcode_i,
  input  logic [31:0] lhs_i,
  input  logic [31:0] rhs_i,
  input  logic        lhs_format_valid_i,
  input  logic        lhs_sign_i,
  input  logic [7:0]  lhs_exponent_i,
  input  logic [22:0] lhs_fraction_i,
  input  logic        lhs_zero_i,
  input  logic        lhs_inf_i,
  input  logic        lhs_nan_i,
  input  logic        rhs_format_valid_i,
  input  logic        rhs_sign_i,
  input  logic [7:0]  rhs_exponent_i,
  input  logic [22:0] rhs_fraction_i,
  input  logic        rhs_zero_i,
  input  logic        rhs_inf_i,
  input  logic        rhs_nan_i,
  output logic        ready_o,
  output logic        valid_o,
  output logic [31:0] result_o,
  output logic        illegal_opcode_o,
  output logic        result_collision_o
);
  logic short_valid_q;
  logic [31:0] short_result_q;
  import lpu_pkg::*;

  localparam logic [15:0] FP16_CANONICAL_NAN = 16'h7e00;
  localparam logic [15:0] BF16_CANONICAL_NAN = 16'h7fc0;
  localparam logic [31:0] FP32_CANONICAL_NAN = 32'h7fc00000;

  logic        shared_multiply_valid;
  logic [31:0] one_cycle_result;
  logic [31:0] format_lhs_value;
  logic [31:0] arithmetic_lhs_value;
  logic [31:0] arithmetic_rhs_value;
  logic [31:0] max_result;
  logic [31:0] shared_addsub_result;
  logic [31:0] shared_multiply_result;
  logic        compare_enable;
  logic        compare_magnitude_gt;
  logic        compare_magnitude_equal;
  logic        compare_lhs_magnitude_ge;
  logic        compare_ordered_gt;
  logic        compare_ordered_equal;
  logic        opcode_valid;
  logic        op_bypass;
  logic        op_add;
  logic        op_subtract;
  logic        op_multiply;
  logic        op_negate;
  logic        op_max;
  logic        multiply_special_nan;
  logic        multiply_special_inf;
  logic        multiply_special_zero;

  // Operation decode proceeds in parallel with the shared unpack front end in
  // lpu_vxm_alu.  This block consumes classification metadata; only the raw
  // value-preserving BYPASS/NEGATE paths use lhs_i directly.
  lpu_vxm_basic_opcode_decode u_opcode_decode (
    .opcode_i,
    .opcode_valid_o(opcode_valid),
    .bypass_o(op_bypass),
    .add_o(op_add),
    .subtract_o(op_subtract),
    .multiply_o(op_multiply),
    .negate_o(op_negate),
    .max_o(op_max)
  );

  always_comb begin
    compare_enable = valid_i && (op_add || op_subtract || op_max) &&
      lhs_format_valid_i && rhs_format_valid_i;
    compare_lhs_magnitude_ge = compare_magnitude_gt ||
      compare_magnitude_equal;

    // The shared unpack front end owns operand classification.  Resolve the
    // Multiply-specific special-result class here where opcode and operand
    // metadata meet; the multiplier receives no raw Zero/Inf/NaN fields.
    multiply_special_nan = lhs_nan_i || rhs_nan_i ||
      (lhs_inf_i && rhs_zero_i) || (rhs_inf_i && lhs_zero_i);
    multiply_special_inf = !multiply_special_nan &&
      (lhs_inf_i || rhs_inf_i);
    multiply_special_zero = !multiply_special_nan &&
      !multiply_special_inf && (lhs_zero_i || rhs_zero_i);

    // Rebuild only the selected format's effective arithmetic encoding.
    // The unpackers have already changed a subnormal fraction to zero, while
    // BYPASS/NEGATE continue to use lhs_i directly below.
    arithmetic_lhs_value = 32'b0;
    arithmetic_rhs_value = 32'b0;
    case (data_format_i)
      VXM_FORMAT_FP16: begin
        arithmetic_lhs_value[15:0] = {
          lhs_sign_i, lhs_exponent_i[4:0], lhs_fraction_i[9:0]};
        arithmetic_rhs_value[15:0] = {
          rhs_sign_i, rhs_exponent_i[4:0], rhs_fraction_i[9:0]};
      end
      VXM_FORMAT_BF16: begin
        arithmetic_lhs_value[15:0] = {
          lhs_sign_i, lhs_exponent_i[7:0], lhs_fraction_i[6:0]};
        arithmetic_rhs_value[15:0] = {
          rhs_sign_i, rhs_exponent_i[7:0], rhs_fraction_i[6:0]};
      end
      VXM_FORMAT_FP32: begin
        arithmetic_lhs_value = {lhs_sign_i, lhs_exponent_i, lhs_fraction_i};
        arithmetic_rhs_value = {rhs_sign_i, rhs_exponent_i, rhs_fraction_i};
      end
      default: begin end
    endcase

    // MAX.NaN + DAZ policy:
    //   * classification is taken directly from the shared unpack front end;
    //   * any NaN returns the selected format's canonical quiet NaN;
    //   * any pair of effective zeros (including DAZ inputs) returns +0;
    //   * equal nonzero values select LHS deterministically;
    //   * infinities need no special datapath and follow ordered comparison.
    max_result = 32'b0;
    if (valid_i && op_max && (lhs_nan_i || rhs_nan_i)) begin
      case (data_format_i)
        VXM_FORMAT_FP16: max_result[15:0] = FP16_CANONICAL_NAN;
        VXM_FORMAT_BF16: max_result[15:0] = BF16_CANONICAL_NAN;
        VXM_FORMAT_FP32: max_result = FP32_CANONICAL_NAN;
        default: max_result = 32'b0;
      endcase
    end else if (valid_i && op_max && lhs_zero_i && rhs_zero_i) begin
      max_result = 32'b0;
    end else if (valid_i && op_max &&
                 (compare_ordered_gt || compare_ordered_equal)) begin
      max_result = arithmetic_lhs_value;
    end else if (valid_i && op_max) begin
      max_result = arithmetic_rhs_value;
    end
  end

  // FP16 and BF16 use the common low 15-bit magnitude comparator. FP32
  // extends it with a high 16-bit comparison. Its ordered result is shared
  // by ADD, SUBTRACT and MAX in this ALU.
  lpu_vxm_shared_float_compare u_shared_compare (
    .enable_i(compare_enable),
    .data_format_i,
    .lhs_sign_i(lhs_sign_i),
    .lhs_exponent_i(lhs_exponent_i),
    .lhs_fraction_i(lhs_fraction_i),
    .lhs_zero_i(lhs_zero_i),
    .lhs_nan_i(lhs_nan_i),
    .rhs_sign_i(rhs_sign_i),
    .rhs_exponent_i(rhs_exponent_i),
    .rhs_fraction_i(rhs_fraction_i),
    .rhs_zero_i(rhs_zero_i),
    .rhs_nan_i(rhs_nan_i),
    .magnitude_gt_o(compare_magnitude_gt),
    .magnitude_equal_o(compare_magnitude_equal),
    .ordered_gt_o(compare_ordered_gt),
    .ordered_equal_o(compare_ordered_equal),
    .unordered_o()
  );

  // ADD and SUBTRACT share one format-selected significand datapath.
  // The format MUX is inside this module and gates every inactive bit before
  // the four 8-bit arithmetic slices.
  lpu_vxm_shared_float_adder u_shared_addsub (
    .enable_i(valid_i && (op_add || op_subtract)),
    .data_format_i,
    .subtract_i(op_subtract),
    .lhs_sign_i(lhs_sign_i),
    .lhs_exponent_i(lhs_exponent_i),
    .lhs_fraction_i(lhs_fraction_i),
    .lhs_zero_i(lhs_zero_i),
    .lhs_inf_i(lhs_inf_i),
    .lhs_nan_i(lhs_nan_i),
    .rhs_sign_i(rhs_sign_i),
    .rhs_exponent_i(rhs_exponent_i),
    .rhs_fraction_i(rhs_fraction_i),
    .rhs_zero_i(rhs_zero_i),
    .rhs_inf_i(rhs_inf_i),
    .rhs_nan_i(rhs_nan_i),
    .lhs_magnitude_ge_i(compare_lhs_magnitude_ge),
    .result_o(shared_addsub_result)
  );

  always_comb begin
    ready_o = UNIFORM_TWO_CYCLE || !valid_i || !opcode_valid || op_multiply ||
      !shared_multiply_valid;
    illegal_opcode_o = valid_i && !opcode_valid;

    // BYPASS and NEGATE are raw-bit operations. They deliberately do not use
    // the comparator's DAZ/canonical-NaN result.
    format_lhs_value = 32'b0;
    case (data_format_i)
      VXM_FORMAT_FP16,
      VXM_FORMAT_BF16:
        format_lhs_value = {16'b0, lhs_i[15:0]};
      VXM_FORMAT_FP32:
        format_lhs_value = lhs_i;
      default: format_lhs_value = 32'b0;
    endcase

    one_cycle_result = 32'b0;
    if (op_bypass)
      one_cycle_result = format_lhs_value;
    else if (op_add || op_subtract)
      one_cycle_result = shared_addsub_result;
    else if (op_negate) begin
      one_cycle_result = format_lhs_value;
      if (data_format_i == VXM_FORMAT_FP32)
        one_cycle_result[31] = ~one_cycle_result[31];
      else
        one_cycle_result[15] = ~one_cycle_result[15];
    end else if (op_max)
      one_cycle_result = max_result;
  end

  // All three formats share the configurable significand multiplier. Format
  // controls select how many 8x8 blocks are active. Its internal product
  // register forms the first real Multiply pipeline boundary; this ALU's
  // result register forms the second.
  lpu_vxm_shared_float_multiplier u_shared_multiplier (
    .clk_i,
    .rst_ni,
    .enable_i(valid_i && op_multiply),
    .data_format_i,
    .lhs_sign_i(lhs_sign_i),
    .lhs_exponent_i(lhs_exponent_i),
    .lhs_fraction_i(lhs_fraction_i),
    .rhs_sign_i(rhs_sign_i),
    .rhs_exponent_i(rhs_exponent_i),
    .rhs_fraction_i(rhs_fraction_i),
    .special_nan_i(multiply_special_nan),
    .special_inf_i(multiply_special_inf),
    .special_zero_i(multiply_special_zero),
    .valid_o(shared_multiply_valid),
    .result_o(shared_multiply_result),
    .active_blocks_o()
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      short_valid_q <= 1'b0;short_result_q <= 0;
      valid_o <= 1'b0;
      result_o <= 32'b0;
      result_collision_o <= 1'b0;
    end else begin
      valid_o <= 1'b0;
      result_collision_o <= 1'b0;

      if (shared_multiply_valid) begin
        valid_o <= 1'b1;
        result_o <= shared_multiply_result;
      end

      short_valid_q <= valid_i && opcode_valid && !op_multiply;
      if(valid_i && opcode_valid && !op_multiply) short_result_q <= one_cycle_result;
      if(UNIFORM_TWO_CYCLE) begin
        if(short_valid_q) begin
          valid_o <= 1'b1;result_o <= short_result_q;
        end
      end else       if (valid_i && opcode_valid && !op_multiply) begin
        if (shared_multiply_valid) begin
          // The older multiply owns this cycle. The caller must retain the
          // new request while ready_o is low.
          result_collision_o <= 1'b1;
        end else begin
          valid_o <= 1'b1;
          result_o <= one_cycle_result;
        end
      end
    end
  end
endmodule
