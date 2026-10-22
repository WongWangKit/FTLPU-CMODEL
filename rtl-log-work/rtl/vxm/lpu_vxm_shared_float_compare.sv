// Shared FP16/BF16/FP32 magnitude and ordered comparator.
//
// Operand format decode, classification and DAZ are performed once by the
// Basic-ALU front end. This block only compares the resulting fields. FP16
// and BF16 use the complete low 15-bit comparator. FP32 adds a high 16-bit
// comparison and consults the same low comparator only when the high parts
// are equal.
module lpu_vxm_shared_float_compare (
  input  logic        enable_i,
  input  logic [1:0]  data_format_i,

  input  logic        lhs_sign_i,
  input  logic [7:0]  lhs_exponent_i,
  input  logic [22:0] lhs_fraction_i,
  input  logic        lhs_zero_i,
  input  logic        lhs_nan_i,

  input  logic        rhs_sign_i,
  input  logic [7:0]  rhs_exponent_i,
  input  logic [22:0] rhs_fraction_i,
  input  logic        rhs_zero_i,
  input  logic        rhs_nan_i,

  output logic magnitude_gt_o,
  output logic magnitude_equal_o,
  output logic ordered_gt_o,
  output logic ordered_equal_o,
  output logic unordered_o
);
  import lpu_pkg::*;

  logic [14:0] lhs_low;
  logic [14:0] rhs_low;
  logic [15:0] lhs_high;
  logic [15:0] rhs_high;
  logic low_gt;
  logic low_equal;
  logic high_gt;
  logic high_equal;
  logic both_zero;

  always_comb begin
    lhs_low = 15'b0;
    rhs_low = 15'b0;
    lhs_high = 16'b0;
    rhs_high = 16'b0;

    magnitude_gt_o = 1'b0;
    magnitude_equal_o = 1'b0;
    ordered_gt_o = 1'b0;
    ordered_equal_o = 1'b0;
    unordered_o = 1'b0;

    if (enable_i) begin
      case (data_format_i)
        VXM_FORMAT_FP16: begin
          lhs_low = {lhs_exponent_i[4:0], lhs_fraction_i[9:0]};
          rhs_low = {rhs_exponent_i[4:0], rhs_fraction_i[9:0]};
        end

        VXM_FORMAT_BF16: begin
          lhs_low = {lhs_exponent_i[7:0], lhs_fraction_i[6:0]};
          rhs_low = {rhs_exponent_i[7:0], rhs_fraction_i[6:0]};
        end

        VXM_FORMAT_FP32: begin
          lhs_high = {lhs_exponent_i[7:0], lhs_fraction_i[22:15]};
          rhs_high = {rhs_exponent_i[7:0], rhs_fraction_i[22:15]};
          lhs_low = lhs_fraction_i[14:0];
          rhs_low = rhs_fraction_i[14:0];
        end

        default: begin end
      endcase
    end

    // These operators describe the two intended physical comparator leaves;
    // synthesis remains free to optimize each leaf for the target library.
    low_gt = lhs_low > rhs_low;
    low_equal = lhs_low == rhs_low;
    high_gt = lhs_high > rhs_high;
    high_equal = lhs_high == rhs_high;
    both_zero = lhs_zero_i && rhs_zero_i;

    if (enable_i &&
        ((data_format_i == VXM_FORMAT_FP16) ||
         (data_format_i == VXM_FORMAT_BF16) ||
         (data_format_i == VXM_FORMAT_FP32))) begin
      unordered_o = lhs_nan_i || rhs_nan_i;

      if (!unordered_o) begin
        if (data_format_i == VXM_FORMAT_FP32) begin
          magnitude_gt_o = high_gt || (high_equal && low_gt);
          magnitude_equal_o = high_equal && low_equal;
        end else begin
          magnitude_gt_o = low_gt;
          magnitude_equal_o = low_equal;
        end

        if (both_zero) begin
          ordered_equal_o = 1'b1;
        end else if (lhs_sign_i != rhs_sign_i) begin
          ordered_gt_o = !lhs_sign_i;
        end else if (!lhs_sign_i) begin
          ordered_gt_o = magnitude_gt_o;
          ordered_equal_o = magnitude_equal_o;
        end else begin
          ordered_gt_o = !magnitude_gt_o && !magnitude_equal_o;
          ordered_equal_o = magnitude_equal_o;
        end
      end
    end
  end
endmodule
