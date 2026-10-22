module lpu_vxm_significand_multiplier (
  input  logic        enable_i,
  input  logic [1:0]  data_format_i,
  input  logic [23:0] lhs_significand_i,
  input  logic [23:0] rhs_significand_i,
  output logic [47:0] product_o,
  output logic [8:0]  active_blocks_o
);
  import lpu_pkg::*;

  logic [7:0] lhs_chunk [0:2];
  logic [7:0] rhs_chunk [0:2];
  logic [15:0] block_product [0:8];
  logic [47:0] shifted_product [0:8];
  logic [47:0] fp32_p22;
  logic [47:0] level1_sum [0:2];
  logic [47:0] level1_carry [0:2];
  logic [47:0] level2_sum [0:1];
  logic [47:0] level2_carry [0:1];
  logic [47:0] level3_sum;
  logic [47:0] level3_carry;
  logic [47:0] level4_sum;
  logic [47:0] level4_carry;
  logic [47:0] level2_main_third;
  logic [47:0] fp32_level3_a;
  logic [47:0] fp32_level3_b;
  logic [47:0] fp32_level3_c;
  logic [47:0] fp32_level4_c;
  logic [47:0] final_add_lhs;
  logic [47:0] final_add_rhs;
  logic [47:0] final_add_result;

  // Significands are left-aligned before entering this module:
  //   BF16 = {hidden+fraction[6:0], 16'b0}
  //   FP16 = {hidden+fraction[9:0], 13'b0}
  //   FP32 = {hidden+fraction[22:0]}
  // Thus BF16 needs block 8 only, FP16 needs blocks 4/5/7/8, and FP32
  // permits every block. Zero-valued chunks are additionally isolated.
  always_comb begin
    for (integer chunk = 0; chunk < 3; chunk++) begin
      lhs_chunk[chunk] = lhs_significand_i[chunk*8 +: 8];
      rhs_chunk[chunk] = rhs_significand_i[chunk*8 +: 8];
    end

    active_blocks_o = '0;
    for (integer lhs_index = 0; lhs_index < 3; lhs_index++) begin
      for (integer rhs_index = 0; rhs_index < 3; rhs_index++) begin
        integer block;
        logic format_enables_block;
        block = lhs_index*3 + rhs_index;
        case (data_format_i)
          VXM_FORMAT_BF16:
            format_enables_block =
              (lhs_index == 2) && (rhs_index == 2);
          VXM_FORMAT_FP16:
            format_enables_block =
              (lhs_index >= 1) && (rhs_index >= 1);
          VXM_FORMAT_FP32:
            format_enables_block = 1'b1;
          default:
            format_enables_block = 1'b0;
        endcase
        active_blocks_o[block] = enable_i && format_enables_block &&
          (|lhs_chunk[lhs_index]) && (|rhs_chunk[rhs_index]);
      end
    end
  end

  generate
    for (genvar lhs_block = 0; lhs_block < 3; lhs_block++) begin : g_lhs
      for (genvar rhs_block = 0; rhs_block < 3; rhs_block++) begin : g_rhs
        localparam integer BLOCK = lhs_block*3 + rhs_block;
        localparam integer SHIFT = (lhs_block+rhs_block)*8;
        lpu_vxm_mul8x8 u_mul8x8 (
          .enable_i(active_blocks_o[BLOCK]),
          .lhs_i(lhs_significand_i[lhs_block*8 +: 8]),
          .rhs_i(rhs_significand_i[rhs_block*8 +: 8]),
          .product_o(block_product[BLOCK])
        );
        always_comb begin
          shifted_product[BLOCK] =
            {{32{1'b0}}, block_product[BLOCK]} << SHIFT;
        end
      end
    end
  endgenerate

  // P22 is needed by BF16 and FP16 without first passing through the FP32-only
  // level-1 compressor.  Isolating this copy prevents that compressor from
  // switching in the narrow modes.
  always_comb begin
    fp32_p22 = data_format_i == VXM_FORMAT_FP32 ?
      shifted_product[8] : 48'b0;
    level2_main_third = data_format_i == VXM_FORMAT_FP16 ?
      shifted_product[8] : level1_sum[1];
    fp32_level3_a = data_format_i == VXM_FORMAT_FP32 ?
      level2_sum[0] : 48'b0;
    fp32_level3_b = data_format_i == VXM_FORMAT_FP32 ?
      level2_carry[0] : 48'b0;
    fp32_level3_c = data_format_i == VXM_FORMAT_FP32 ?
      level2_sum[1] : 48'b0;
    fp32_level4_c = data_format_i == VXM_FORMAT_FP32 ?
      level2_carry[1] : 48'b0;

    // A single carry-propagate adder is shared by FP16 and FP32.  FP16's
    // carry-save operands are temporarily moved from [47:26] to [21:0], so
    // its active carry path is only 22 bits.  The result is wired back to the
    // established left-aligned representation after the addition.
    final_add_lhs = 48'b0;
    final_add_rhs = 48'b0;
    case (data_format_i)
      VXM_FORMAT_FP16: begin
        final_add_lhs[21:0] = level2_sum[0][47:26];
        final_add_rhs[21:0] = level2_carry[0][47:26];
      end
      VXM_FORMAT_FP32: begin
        final_add_lhs = level4_sum;
        final_add_rhs = level4_carry;
      end
      default: begin end
    endcase
    final_add_result = final_add_lhs + final_add_rhs;

    case (data_format_i)
      VXM_FORMAT_BF16:
        product_o = shifted_product[8];
      VXM_FORMAT_FP16:
        product_o = {final_add_result[21:0], 26'b0};
      VXM_FORMAT_FP32:
        product_o = final_add_result;
      default:
        product_o = 48'b0;
    endcase
  end

  // Level 1: group the three narrow FP16 products together.  P22 deliberately
  // bypasses this level in FP16, while its FP32 copy participates in the
  // complete 9 -> 6 compression.
  lpu_vxm_csa3_2 #(.WIDTH(48)) u_level1_fp16_narrow (
    .a_i(shifted_product[4]),
    .b_i(shifted_product[5]),
    .c_i(shifted_product[7]),
    .sum_o(level1_sum[0]),
    .carry_o(level1_carry[0])
  );

  lpu_vxm_csa3_2 #(.WIDTH(48)) u_level1_fp32_p22 (
    .a_i(fp32_p22),
    .b_i(shifted_product[0]),
    .c_i(shifted_product[1]),
    .sum_o(level1_sum[1]),
    .carry_o(level1_carry[1])
  );

  lpu_vxm_csa3_2 #(.WIDTH(48)) u_level1_fp32_remaining (
    .a_i(shifted_product[2]),
    .b_i(shifted_product[3]),
    .c_i(shifted_product[6]),
    .sum_o(level1_sum[2]),
    .carry_o(level1_carry[2])
  );

  // Level 2 completes FP16's four-to-two compression.  FP32 continues with
  // four carry-save operands after this level.
  lpu_vxm_csa3_2 #(.WIDTH(48)) u_level2_main (
    .a_i(level1_sum[0]),
    .b_i(level1_carry[0]),
    .c_i(level2_main_third),
    .sum_o(level2_sum[0]),
    .carry_o(level2_carry[0])
  );

  lpu_vxm_csa3_2 #(.WIDTH(48)) u_level2_fp32 (
    .a_i(level1_carry[1]),
    .b_i(level1_sum[2]),
    .c_i(level1_carry[2]),
    .sum_o(level2_sum[1]),
    .carry_o(level2_carry[1])
  );

  // FP32-only levels finish 4 -> 3 -> 2 compression.  Their inactive inputs
  // are zero in BF16/FP16 modes, so they do not switch with narrow operands.
  lpu_vxm_csa3_2 #(.WIDTH(48)) u_level3_fp32 (
    .a_i(fp32_level3_a),
    .b_i(fp32_level3_b),
    .c_i(fp32_level3_c),
    .sum_o(level3_sum),
    .carry_o(level3_carry)
  );

  lpu_vxm_csa3_2 #(.WIDTH(48)) u_level4_fp32 (
    .a_i(level3_sum),
    .b_i(level3_carry),
    .c_i(fp32_level4_c),
    .sum_o(level4_sum),
    .carry_o(level4_carry)
  );
endmodule
