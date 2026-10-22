`timescale 1ns/1ps

module lpu_vxm_significand_multiplier_tb;
  import lpu_pkg::*;

  logic        enable;
  logic [1:0]  data_format;
  logic [23:0] lhs_significand;
  logic [23:0] rhs_significand;
  wire [47:0] product;
  wire [8:0]  active_blocks;
  logic [31:0] random_state;

  lpu_vxm_significand_multiplier dut (
    .enable_i(enable),
    .data_format_i(data_format),
    .lhs_significand_i(lhs_significand),
    .rhs_significand_i(rhs_significand),
    .product_o(product),
    .active_blocks_o(active_blocks)
  );

  function automatic logic [31:0] next_random(input logic [31:0] state);
    logic [31:0] value;
    begin
      value = state;
      value = value ^ (value << 13);
      value = value ^ (value >> 17);
      value = value ^ (value << 5);
      next_random = value;
    end
  endfunction

  function automatic logic [8:0] expected_active_blocks(
    input logic        request_enable,
    input logic [1:0]  format,
    input logic [23:0] lhs,
    input logic [23:0] rhs
  );
    integer lhs_index;
    integer rhs_index;
    integer block;
    logic format_enables_block;
    begin
      expected_active_blocks = '0;
      for (lhs_index = 0; lhs_index < 3; lhs_index++) begin
        for (rhs_index = 0; rhs_index < 3; rhs_index++) begin
          block = lhs_index*3 + rhs_index;
          case (format)
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
          expected_active_blocks[block] = request_enable &&
            format_enables_block &&
            (|lhs[lhs_index*8 +: 8]) &&
            (|rhs[rhs_index*8 +: 8]);
        end
      end
    end
  endfunction

  task automatic check_product(
    input logic        request_enable,
    input logic [1:0]  format,
    input logic [23:0] lhs,
    input logic [23:0] rhs,
    input logic [8:0]  expected_blocks,
    input logic [47:0] expected_product
  );
    begin
      enable = request_enable;
      data_format = format;
      lhs_significand = lhs;
      rhs_significand = rhs;
      #1;
      if (active_blocks !== expected_blocks)
        $fatal(1,
          "format=%0d active blocks %b expected %b",
          format, active_blocks, expected_blocks);
      if (product !== expected_product)
        $fatal(1,
          "format=%0d product %h expected %h",
          format, product, expected_product);
    end
  endtask

  task automatic check_calculated_product(
    input logic        request_enable,
    input logic [1:0]  format,
    input logic [23:0] lhs,
    input logic [23:0] rhs
  );
    logic [23:0] effective_lhs;
    logic [23:0] effective_rhs;
    logic [47:0] extended_lhs;
    logic [47:0] extended_rhs;
    logic [47:0] expected_product;
    begin
      effective_lhs = 24'b0;
      effective_rhs = 24'b0;
      case (format)
        VXM_FORMAT_BF16: begin
          effective_lhs = {lhs[23:16], 16'b0};
          effective_rhs = {rhs[23:16], 16'b0};
        end
        VXM_FORMAT_FP16: begin
          effective_lhs = {lhs[23:13], 13'b0};
          effective_rhs = {rhs[23:13], 13'b0};
        end
        VXM_FORMAT_FP32: begin
          effective_lhs = lhs;
          effective_rhs = rhs;
        end
        default: begin end
      endcase
      extended_lhs = {24'b0, effective_lhs};
      extended_rhs = {24'b0, effective_rhs};
      expected_product = request_enable ?
        extended_lhs * extended_rhs : 48'b0;
      check_product(request_enable, format, lhs, rhs,
        expected_active_blocks(request_enable, format, lhs, rhs),
        expected_product);
    end
  endtask

  initial begin
    enable = 1'b0;
    data_format = VXM_FORMAT_FP16;
    lhs_significand = '0;
    rhs_significand = '0;
    random_state = 32'h9283a6d1;
    #1;

    // BF16 has only its upper eight significand bits populated, so only P22
    // may switch and the result bypasses the partial-product addition tree.
    check_product(1'b1, VXM_FORMAT_BF16,
                  24'hab0000, 24'hcd0000, 9'b100000000,
                  48'hab0000 * 48'hcd0000);

    // FP16 has eleven populated bits in [23:13]. Blocks P11/P12/P21/P22
    // implement the equivalent 11x11 multiplication.
    check_product(1'b1, VXM_FORMAT_FP16,
                  24'habc000, 24'hd6a000, 9'b110110000,
                  48'habc000 * 48'hd6a000);

    // A general FP32 significand activates all nine 8x8 blocks.
    check_product(1'b1, VXM_FORMAT_FP32,
                  24'habcdef, 24'hd5a37b, 9'b111111111,
                  48'habcdef * 48'hd5a37b);

    // Chunk-zero detection also isolates unused blocks for an FP32 value
    // whose low sixteen significand bits happen to be zero.
    check_product(1'b1, VXM_FORMAT_FP32,
                  24'hab0000, 24'hcd0000, 9'b100000000,
                  48'hab0000 * 48'hcd0000);

    check_product(1'b0, VXM_FORMAT_FP32,
                  24'habcdef, 24'hd5a37b, 9'b000000000, 48'b0);
    check_product(1'b1, VXM_FORMAT_RESERVED,
                  24'habcdef, 24'hd5a37b, 9'b000000000, 48'b0);

    // Exact port-level regression.  Operands use the same left-aligned
    // significand convention as the floating-point wrapper.  The expected
    // product is calculated independently of the DUT's block/CSA structure.
    for (integer sample = 0; sample < 300; sample++) begin
      logic [23:0] lhs_random;
      logic [23:0] rhs_random;
      random_state = next_random(random_state);
      lhs_random = {1'b1, random_state[9:0], 13'b0};
      random_state = next_random(random_state);
      rhs_random = {1'b1, random_state[9:0], 13'b0};
      check_calculated_product(1'b1, VXM_FORMAT_FP16,
        lhs_random, rhs_random);

      random_state = next_random(random_state);
      lhs_random = {1'b1, random_state[6:0], 16'b0};
      random_state = next_random(random_state);
      rhs_random = {1'b1, random_state[6:0], 16'b0};
      check_calculated_product(1'b1, VXM_FORMAT_BF16,
        lhs_random, rhs_random);

      random_state = next_random(random_state);
      lhs_random = {1'b1, random_state[22:0]};
      random_state = next_random(random_state);
      rhs_random = {1'b1, random_state[22:0]};
      check_calculated_product(1'b1, VXM_FORMAT_FP32,
        lhs_random, rhs_random);
    end

    $display("LPU_VXM_SIGNIFICAND_MULTIPLIER_TB_PASS random_checks=900");
    $finish;
  end
endmodule
