`timescale 1ns/1ps

// Port-only Basic arithmetic regression. A real Q0 local instruction and two stream
// operands enter the execution stage; only its public handshake, result,
// metadata and fault ports are observed. The reference uses independent exact
// integer arithmetic, not any DUT helper or hierarchical signal.
module lpu_vxm_addsub_max_instruction_tb;
  import lpu_pkg::*;

  logic clk;
  logic rst_n;
  logic instruction_valid;
  logic [VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0] instruction;
  logic [1:0] dtype;
  logic [31:0] lhs_data;
  logic [31:0] rhs_data;
  wire input_ready;
  wire request_accepted;
  wire result_valid;
  wire [31:0] result_value;
  wire [31:0] result_original;
  wire [31:0] result_auxiliary;
  wire result_end_marker;
  wire chain_head;
  wire chain_tail;
  wire fault;
  integer addsub_max_checks;
  integer multiply_checks;
  logic [31:0] random_state;

  always #5 clk = ~clk;

  lpu_vxm_execution_stage #(
    .LOCAL_QUEUE(0),
    .PHYSICAL_STAGE(0),
    .SPECIAL_KIND(VXM_SPECIAL_NONE)
  ) dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .instruction_valid_i(instruction_valid),
    .instruction_i(instruction),
    .chain_length_i(VXM_CHAIN_LENGTH_2),
    .compute_dtype_i(dtype),
    .lhs_dtype_i(dtype),
    .rhs_dtype_i(dtype),
    .head_lhs_valid_i(1'b1),
    .head_lhs_data_i(lhs_data),
    .head_rhs_valid_i(1'b1),
    .head_rhs_data_i(rhs_data),
    .previous_valid_i(1'b0),
    .previous_value_i('0),
    .previous_original_i('0),
    .previous_auxiliary_i('0),
    .feedback_valid_i(1'b0),
    .feedback_value_i('0),
    .feedback_original_i('0),
    .feedback_auxiliary_i('0),
    .immediate_valid_i(1'b0),
    .immediate_data_i('0),
    .accumulator_valid_i(1'b0),
    .accumulator_data_i('0),
    .request_end_marker_i(1'b1),
    .lut_configured_i('0),
    .lut_input_min_i('0),
    .lut_segment_width_i('0),
    .lut_read_valid_o(),
    .lut_read_bank_o(),
    .lut_read_address_o(),
    .lut_read_valid_i(1'b0),
    .lut_read_k_i('0),
    .lut_read_b_i('0),
    .input_ready_o(input_ready),
    .request_accepted_o(request_accepted),
    .result_valid_o(result_valid),
    .result_value_o(result_value),
    .result_original_o(result_original),
    .result_auxiliary_o(result_auxiliary),
    .result_end_marker_o(result_end_marker),
    .chain_head_o(chain_head),
    .chain_tail_o(chain_tail),
    .fault_o(fault)
  );

  function automatic logic [31:0] next_random(input logic [31:0] state);
    logic [31:0] x;
    begin
      x = state ^ (state << 13);
      x = x ^ (x >> 17);
      next_random = x ^ (x << 5);
    end
  endfunction

  function automatic logic [31:0] pack_value(
    input integer exp_bits,
    input integer frac_bits,
    input logic sign_bit,
    input integer exponent,
    input logic [31:0] fraction
  );
    logic [31:0] sign_field;
    begin
      sign_field = 32'b0;
      sign_field[exp_bits+frac_bits] = sign_bit;
      pack_value = sign_field | (32'(exponent) << frac_bits) |
        (fraction & ((32'h1 << frac_bits) - 1));
    end
  endfunction

  function automatic logic [31:0] expected_stream_token(
    input logic [1:0] format,
    input logic [31:0] raw_value
  );
    begin
      expected_stream_token = format == VXM_FORMAT_FP32 ? raw_value :
        {16'b0, raw_value[15:0]};
    end
  endfunction

  function automatic logic [31:0] reference_multiply(
    input logic [1:0] format,
    input logic [31:0] lhs_raw,
    input logic [31:0] rhs_raw
  );
    integer exp_bits, frac_bits, exp_all_ones, bias;
    integer a_exp, b_exp, out_exp, shift_count;
    logic [31:0] frac_mask, a_frac, b_frac, canonical_nan;
    logic a_sign, b_sign, out_sign;
    logic a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
    logic normalize_shift;
    logic [63:0] a_significand, b_significand;
    logic [63:0] exact_product, retained, remainder;
    logic [63:0] remainder_mask, halfway;
    begin
      exp_bits = (format == VXM_FORMAT_FP16) ? 5 : 8;
      frac_bits = (format == VXM_FORMAT_FP16) ? 10 :
        ((format == VXM_FORMAT_BF16) ? 7 : 23);
      exp_all_ones = (1 << exp_bits) - 1;
      bias = (format == VXM_FORMAT_FP16) ? 15 : 127;
      frac_mask = (32'h1 << frac_bits) - 1;
      canonical_nan = (format == VXM_FORMAT_FP16) ? 32'h00007e00 :
        ((format == VXM_FORMAT_BF16) ? 32'h00007fc0 : 32'h7fc00000);
      a_sign = lhs_raw[exp_bits+frac_bits];
      b_sign = rhs_raw[exp_bits+frac_bits];
      out_sign = a_sign ^ b_sign;
      a_exp = int'((lhs_raw >> frac_bits) & 32'(exp_all_ones));
      b_exp = int'((rhs_raw >> frac_bits) & 32'(exp_all_ones));
      a_frac = lhs_raw & frac_mask;
      b_frac = rhs_raw & frac_mask;
      a_nan = (a_exp == exp_all_ones) && (a_frac != 0);
      b_nan = (b_exp == exp_all_ones) && (b_frac != 0);
      a_inf = (a_exp == exp_all_ones) && (a_frac == 0);
      b_inf = (b_exp == exp_all_ones) && (b_frac == 0);
      a_zero = (a_exp == 0); // project DAZ policy includes subnormals
      b_zero = (b_exp == 0);
      reference_multiply = 32'b0;

      if (a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero)) begin
        reference_multiply = canonical_nan;
      end else if (a_inf || b_inf) begin
        reference_multiply = pack_value(
          exp_bits, frac_bits, out_sign, exp_all_ones, 0);
      end else if (a_zero || b_zero) begin
        reference_multiply = pack_value(
          exp_bits, frac_bits, out_sign, 0, 0);
      end else begin
        a_significand = (64'h1 << frac_bits) | a_frac;
        b_significand = (64'h1 << frac_bits) | b_frac;
        exact_product = a_significand * b_significand;
        normalize_shift = exact_product[2*frac_bits+1];
        shift_count = frac_bits + (normalize_shift ? 1 : 0);
        retained = exact_product >> shift_count;
        remainder_mask = (64'h1 << shift_count) - 1;
        remainder = exact_product & remainder_mask;
        halfway = 64'h1 << (shift_count-1);
        if ((remainder > halfway) ||
            ((remainder == halfway) && retained[0]))
          retained = retained + 64'd1;

        out_exp = a_exp + b_exp - bias +
          (normalize_shift ? 1 : 0);
        if (retained[frac_bits+1]) begin
          retained = retained >> 1;
          out_exp = out_exp + 1;
        end

        if (out_exp <= 0)
          reference_multiply = pack_value(
            exp_bits, frac_bits, out_sign, 0, 0);
        else if (out_exp >= exp_all_ones)
          reference_multiply = pack_value(
            exp_bits, frac_bits, out_sign, exp_all_ones, 0);
        else
          reference_multiply = pack_value(
            exp_bits, frac_bits, out_sign, out_exp,
            retained[31:0] & frac_mask);
      end
    end
  endfunction

  function automatic logic [31:0] reference_result(
    input logic [1:0] format,
    input logic [2:0] opcode,
    input logic [31:0] lhs_raw,
    input logic [31:0] rhs_raw
  );
    integer exp_bits, frac_bits, exp_all_ones;
    integer a_exp, b_exp, out_exp, msb, drop;
    logic [31:0] frac_mask, a_frac, b_frac, a_bits, b_bits;
    logic a_sign, b_sign, b_effective_sign, out_sign;
    logic a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
    logic [31:0] canonical_nan, retained;
    logic [319:0] a_mag, b_mag, exact_mag, aligned;
    logic guard_bit, round_bit, sticky_bit, choose_a;
    begin
      exp_bits = (format == VXM_FORMAT_FP16) ? 5 : 8;
      frac_bits = (format == VXM_FORMAT_FP16) ? 10 :
        ((format == VXM_FORMAT_BF16) ? 7 : 23);
      exp_all_ones = (1 << exp_bits) - 1;
      frac_mask = (32'h1 << frac_bits) - 1;
      canonical_nan = (format == VXM_FORMAT_FP16) ? 32'h00007e00 :
        ((format == VXM_FORMAT_BF16) ? 32'h00007fc0 : 32'h7fc00000);
      a_sign = lhs_raw[exp_bits+frac_bits];
      b_sign = rhs_raw[exp_bits+frac_bits];
      a_exp = int'((lhs_raw >> frac_bits) & 32'(exp_all_ones));
      b_exp = int'((rhs_raw >> frac_bits) & 32'(exp_all_ones));
      a_frac = lhs_raw & frac_mask;
      b_frac = rhs_raw & frac_mask;
      a_nan = (a_exp == exp_all_ones) && (a_frac != 0);
      b_nan = (b_exp == exp_all_ones) && (b_frac != 0);
      a_inf = (a_exp == exp_all_ones) && (a_frac == 0);
      b_inf = (b_exp == exp_all_ones) && (b_frac == 0);

      // DAZ is applied before arithmetic/MAX; its zero keeps the input sign.
      if (a_exp == 0) a_frac = 0;
      if (b_exp == 0) b_frac = 0;
      a_zero = (a_exp == 0);
      b_zero = (b_exp == 0);
      a_bits = pack_value(exp_bits, frac_bits, a_sign, a_exp, a_frac);
      b_bits = pack_value(exp_bits, frac_bits, b_sign, b_exp, b_frac);
      reference_result = 32'b0;

      if (opcode == VXM_LOCAL_MAX) begin
        if (a_nan || b_nan) begin
          reference_result = canonical_nan;
        end else if (a_zero && b_zero) begin
          reference_result = 32'b0;
        end else begin
          choose_a = 1'b0;
          if (a_sign != b_sign)
            choose_a = !a_sign;
          else if (!a_sign)
            choose_a = (a_bits & ~(32'h1 << (exp_bits+frac_bits))) >=
              (b_bits & ~(32'h1 << (exp_bits+frac_bits)));
          else
            choose_a = (a_bits & ~(32'h1 << (exp_bits+frac_bits))) <=
              (b_bits & ~(32'h1 << (exp_bits+frac_bits)));
          reference_result = choose_a ? a_bits : b_bits;
        end
      end else begin
        b_effective_sign = b_sign ^ (opcode == VXM_LOCAL_SUBTRACT);
        if (a_nan || b_nan ||
            (a_inf && b_inf && (a_sign != b_effective_sign))) begin
          reference_result = canonical_nan;
        end else if (a_inf) begin
          reference_result = a_bits;
        end else if (b_inf) begin
          reference_result = pack_value(exp_bits, frac_bits,
            b_effective_sign, b_exp, b_frac);
        end else if (a_zero && b_zero) begin
          reference_result = pack_value(exp_bits, frac_bits,
            a_sign && b_effective_sign, 0, 0);
        end else if (a_zero) begin
          reference_result = pack_value(exp_bits, frac_bits,
            b_effective_sign, b_exp, b_frac);
        end else if (b_zero) begin
          reference_result = a_bits;
        end else begin
          // Exact common fixed-point scale: sig * 2^(encoded_exp-1).
          // No guard/round/sticky truncation is used in this oracle.
          a_mag = 320'(32'h1 << frac_bits | a_frac) << (a_exp - 1);
          b_mag = 320'(32'h1 << frac_bits | b_frac) << (b_exp - 1);
          if (a_sign == b_effective_sign) begin
            exact_mag = a_mag + b_mag;
            out_sign = a_sign;
          end else if (a_mag >= b_mag) begin
            exact_mag = a_mag - b_mag;
            out_sign = a_sign;
          end else begin
            exact_mag = b_mag - a_mag;
            out_sign = b_effective_sign;
          end

          if (exact_mag == 0) begin
            reference_result = 32'b0;  // exact cancellation: +0
          end else begin
            msb = -1;
            for (integer bit_index = 319; bit_index >= 0; bit_index--)
              if ((msb < 0) && exact_mag[bit_index]) msb = bit_index;
            out_exp = msb - frac_bits + 1;
            if (out_exp <= 0) begin
              reference_result = pack_value(exp_bits, frac_bits,
                out_sign, 0, 0);  // project FTZ policy
            end else if (out_exp >= exp_all_ones) begin
              reference_result = pack_value(exp_bits, frac_bits,
                out_sign, exp_all_ones, 0);
            end else begin
              drop = msb - frac_bits;
              guard_bit = 1'b0;
              round_bit = 1'b0;
              sticky_bit = 1'b0;
              aligned = exact_mag;
              if (drop > 0) begin
                aligned = exact_mag >> drop;
                guard_bit = exact_mag[drop-1];
                if (drop > 1) round_bit = exact_mag[drop-2];
                for (integer bit_index = 0; bit_index < drop-2; bit_index++)
                  sticky_bit = sticky_bit | exact_mag[bit_index];
              end else if (drop < 0) begin
                aligned = exact_mag << (-drop);
              end
              retained = aligned[31:0];
              if (guard_bit && (round_bit || sticky_bit || retained[0]))
                retained = retained + 32'd1;
              if (retained[frac_bits+1]) begin
                retained = retained >> 1;
                out_exp = out_exp + 1;
              end
              if (out_exp >= exp_all_ones)
                reference_result = pack_value(exp_bits, frac_bits,
                  out_sign, exp_all_ones, 0);
              else
                reference_result = pack_value(exp_bits, frac_bits,
                  out_sign, out_exp, retained & frac_mask);
            end
          end
        end
      end
    end
  endfunction

  task automatic check_case(
    input string label_text,
    input logic [1:0] format,
    input logic [2:0] opcode,
    input logic [31:0] lhs_value,
    input logic [31:0] rhs_value
  );
    logic [31:0] expected;
    integer expected_latency;
    begin
      expected = opcode == VXM_LOCAL_MULTIPLY ?
        reference_multiply(format, lhs_value, rhs_value) :
        reference_result(format, opcode, lhs_value, rhs_value);
      expected_latency = opcode == VXM_LOCAL_MULTIPLY ? 2 : 1;
      @(negedge clk);
      dtype = format;
      instruction = VXM_LOCAL_MAX_INSTRUCTION_WIDTH'(opcode); // Q0: stream/stream
      lhs_data = lhs_value;
      rhs_data = rhs_value;
      instruction_valid = 1'b1;
      #1;
      if (input_ready !== 1'b1 || request_accepted !== 1'b1 ||
          chain_head !== 1'b1 || chain_tail !== 1'b0 || fault !== 1'b0)
        $fatal(1, "%s: request/chain/fault ports invalid", label_text);
      @(posedge clk);
      #1;
      if (expected_latency == 1) begin
        if (result_valid !== 1'b1 || result_value !== expected ||
            result_original !== expected_stream_token(format, lhs_value) ||
            result_auxiliary !== expected_stream_token(format, rhs_value) ||
            result_end_marker !== 1'b1 || fault !== 1'b0)
          $fatal(1,
            "%s format=%0d op=%0d lhs=%h rhs=%h got=%h expected=%h valid=%b fault=%b",
            label_text, format, opcode, lhs_value, rhs_value,
            result_value, expected, result_valid, fault);
      end else if (result_valid !== 1'b0 || fault !== 1'b0) begin
        $fatal(1, "%s: Multiply returned before its second cycle", label_text);
      end
      @(negedge clk);
      instruction_valid = 1'b0;
      if (expected_latency == 2) begin
        @(posedge clk);
        #1;
        if (result_valid !== 1'b1 || result_value !== expected ||
            result_original !== expected_stream_token(format, lhs_value) ||
            result_auxiliary !== expected_stream_token(format, rhs_value) ||
            result_end_marker !== 1'b1 || fault !== 1'b0)
          $fatal(1,
            "%s format=%0d MUL lhs=%h rhs=%h got=%h expected=%h valid=%b fault=%b",
            label_text, format, lhs_value, rhs_value,
            result_value, expected, result_valid, fault);
        @(negedge clk);
      end

      if (opcode == VXM_LOCAL_MULTIPLY)
        multiply_checks = multiply_checks + 1;
      else
        addsub_max_checks = addsub_max_checks + 1;
      @(posedge clk);
      #1;
      if (result_valid !== 1'b0 || fault !== 1'b0)
        $fatal(1, "%s: stale result or fault after request", label_text);
    end
  endtask

  task automatic check_all_ops(
    input string label_text,
    input logic [1:0] format,
    input logic [31:0] lhs_value,
    input logic [31:0] rhs_value
  );
    begin
      check_case(label_text, format, VXM_LOCAL_ADD, lhs_value, rhs_value);
      check_case(label_text, format, VXM_LOCAL_SUBTRACT, lhs_value, rhs_value);
      check_case(label_text, format, VXM_LOCAL_MAX, lhs_value, rhs_value);
      check_case(label_text, format, VXM_LOCAL_MULTIPLY, lhs_value, rhs_value);
    end
  endtask

  task automatic check_oracle(
    input logic [1:0] format,
    input logic [2:0] opcode,
    input logic [31:0] lhs_value,
    input logic [31:0] rhs_value,
    input logic [31:0] known_result
  );
    logic [31:0] calculated;
    begin
      calculated = opcode == VXM_LOCAL_MULTIPLY ?
        reference_multiply(format, lhs_value, rhs_value) :
        reference_result(format, opcode, lhs_value, rhs_value);
      if (calculated !== known_result)
        $fatal(1, "reference oracle error: format=%0d opcode=%0d calculated=%h known=%h",
          format, opcode, calculated, known_result);
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    instruction_valid = 1'b0;
    instruction = '0;
    dtype = VXM_FORMAT_FP16;
    lhs_data = '0;
    rhs_data = '0;
    addsub_max_checks = 0;
    multiply_checks = 0;
    random_state = 32'h73a42db9;
    repeat (3) @(negedge clk);
    rst_n = 1'b1;

    // Hand-calculated anchors keep the independent oracle honest.
    check_oracle(VXM_FORMAT_FP16, VXM_LOCAL_ADD,
      32'h00003c00, 32'h00003c00, 32'h00004000);
    check_oracle(VXM_FORMAT_FP16, VXM_LOCAL_SUBTRACT,
      32'h00003c00, 32'h00003c00, 32'h00000000);
    check_oracle(VXM_FORMAT_FP16, VXM_LOCAL_MAX,
      32'h0000c000, 32'h00003e00, 32'h00003e00);
    check_oracle(VXM_FORMAT_FP16, VXM_LOCAL_ADD,
      32'h00003c00, 32'h00001000, 32'h00003c00); // even RNE tie
    check_oracle(VXM_FORMAT_FP16, VXM_LOCAL_ADD,
      32'h00003c01, 32'h00001000, 32'h00003c02); // odd RNE tie
    check_oracle(VXM_FORMAT_BF16, VXM_LOCAL_ADD,
      32'h00003f80, 32'h00003b80, 32'h00003f80);
    check_oracle(VXM_FORMAT_FP32, VXM_LOCAL_ADD,
      32'h3f800000, 32'h33800000, 32'h3f800000);
    check_oracle(VXM_FORMAT_FP32, VXM_LOCAL_SUBTRACT,
      32'h00800001, 32'h00800000, 32'h00000000); // FTZ
    check_oracle(VXM_FORMAT_FP16, VXM_LOCAL_MULTIPLY,
      32'h00003e00, 32'h00003e00, 32'h00004080); // 1.5 * 1.5
    check_oracle(VXM_FORMAT_FP16, VXM_LOCAL_MULTIPLY,
      32'h00003c01, 32'h00003ffe, 32'h00004000); // RNE carry
    check_oracle(VXM_FORMAT_BF16, VXM_LOCAL_MULTIPLY,
      32'h00003f81, 32'h00003ffe, 32'h00004000); // RNE carry
    check_oracle(VXM_FORMAT_FP32, VXM_LOCAL_MULTIPLY,
      32'h3f800001, 32'h3ffffffe, 32'h40000000); // RNE carry

    // Drive the hand-known Multiply anchors through the same public compact
    // instruction and stream-data interface used by the full regression.
    check_case("fp16-mul-exact", VXM_FORMAT_FP16, VXM_LOCAL_MULTIPLY,
      32'h00003e00, 32'h00003e00);
    check_case("fp16-mul-rne-carry", VXM_FORMAT_FP16, VXM_LOCAL_MULTIPLY,
      32'h00003c01, 32'h00003ffe);
    check_case("bf16-mul-rne-carry", VXM_FORMAT_BF16, VXM_LOCAL_MULTIPLY,
      32'h00003f81, 32'h00003ffe);
    check_case("fp32-mul-rne-carry", VXM_FORMAT_FP32, VXM_LOCAL_MULTIPLY,
      32'h3f800001, 32'h3ffffffe);

    // FP16: exact cancellation, near/far subtraction, S generation at
    // exponent gaps 3/4+, RNE ties on both even and odd retained LSBs,
    // overflow, FTZ, signed zeros, Inf, NaN, and input DAZ.
    check_all_ops("fp16-signs", VXM_FORMAT_FP16, 32'h0000c000, 32'h00003e00);
    check_all_ops("fp16-equal", VXM_FORMAT_FP16, 32'h00003c00, 32'h00003c00);
    check_all_ops("fp16-near", VXM_FORMAT_FP16, 32'h00003c00, 32'h00003bff);
    check_all_ops("fp16-gap1", VXM_FORMAT_FP16, 32'h00003c00, 32'h00003801);
    check_all_ops("fp16-gap2", VXM_FORMAT_FP16, 32'h00003c00, 32'h00003401);
    check_all_ops("fp16-gap3", VXM_FORMAT_FP16, 32'h00003c00, 32'h00003001);
    check_all_ops("fp16-gap4", VXM_FORMAT_FP16, 32'h00003c00, 32'h00002c01);
    check_all_ops("fp16-huge-gap", VXM_FORMAT_FP16, 32'h00007bff, 32'h00000401);
    check_all_ops("fp16-tie-even", VXM_FORMAT_FP16, 32'h00003c00, 32'h00001000);
    check_all_ops("fp16-tie-odd", VXM_FORMAT_FP16, 32'h00003c01, 32'h00001000);
    check_all_ops("fp16-overflow", VXM_FORMAT_FP16, 32'h00007bff, 32'h00007bff);
    check_all_ops("fp16-ftz", VXM_FORMAT_FP16, 32'h00000401, 32'h00000400);
    check_all_ops("fp16-zero", VXM_FORMAT_FP16, 32'h00008000, 32'h00000000);
    check_all_ops("fp16-infinity", VXM_FORMAT_FP16, 32'h00007c00, 32'h0000fc00);
    check_all_ops("fp16-nan", VXM_FORMAT_FP16, 32'h00007e01, 32'h00003c00);
    check_all_ops("fp16-daz", VXM_FORMAT_FP16, 32'h00000001, 32'h0000bc00);

    check_all_ops("bf16-signs", VXM_FORMAT_BF16, 32'h0000c000, 32'h00003fc0);
    check_all_ops("bf16-equal", VXM_FORMAT_BF16, 32'h00003f80, 32'h00003f80);
    check_all_ops("bf16-near", VXM_FORMAT_BF16, 32'h00003f80, 32'h00003f7f);
    check_all_ops("bf16-gap3", VXM_FORMAT_BF16, 32'h00003f80, 32'h00003e01);
    check_all_ops("bf16-gap4", VXM_FORMAT_BF16, 32'h00003f80, 32'h00003d81);
    check_all_ops("bf16-tie-even", VXM_FORMAT_BF16, 32'h00003f80, 32'h00003b80);
    check_all_ops("bf16-tie-odd", VXM_FORMAT_BF16, 32'h00003f81, 32'h00003b80);
    check_all_ops("bf16-overflow", VXM_FORMAT_BF16, 32'h00007f7f, 32'h00007f7f);
    check_all_ops("bf16-ftz", VXM_FORMAT_BF16, 32'h00000081, 32'h00000080);
    check_all_ops("bf16-zero", VXM_FORMAT_BF16, 32'h00008000, 32'h00000000);
    check_all_ops("bf16-infinity", VXM_FORMAT_BF16, 32'h00007f80, 32'h0000ff80);
    check_all_ops("bf16-nan", VXM_FORMAT_BF16, 32'h00003f80, 32'h00007fc1);
    check_all_ops("bf16-daz", VXM_FORMAT_BF16, 32'h00000001, 32'h0000bf80);

    check_all_ops("fp32-signs", VXM_FORMAT_FP32, 32'hc0000000, 32'h3fc00000);
    check_all_ops("fp32-equal", VXM_FORMAT_FP32, 32'h3f800000, 32'h3f800000);
    check_all_ops("fp32-near", VXM_FORMAT_FP32, 32'h3f800000, 32'h3f7fffff);
    check_all_ops("fp32-gap3", VXM_FORMAT_FP32, 32'h3f800000, 32'h3e000001);
    check_all_ops("fp32-gap4", VXM_FORMAT_FP32, 32'h3f800000, 32'h3d800001);
    check_all_ops("fp32-tie-even", VXM_FORMAT_FP32, 32'h3f800000, 32'h33800000);
    check_all_ops("fp32-tie-odd", VXM_FORMAT_FP32, 32'h3f800001, 32'h33800000);
    check_all_ops("fp32-overflow", VXM_FORMAT_FP32, 32'h7f7fffff, 32'h7f7fffff);
    check_all_ops("fp32-ftz", VXM_FORMAT_FP32, 32'h00800001, 32'h00800000);
    check_all_ops("fp32-zero", VXM_FORMAT_FP32, 32'h80000000, 32'h00000000);
    check_all_ops("fp32-infinity", VXM_FORMAT_FP32, 32'h7f800000, 32'hff800000);
    check_all_ops("fp32-nan", VXM_FORMAT_FP32, 32'h7fc00001, 32'h3f800000);
    check_all_ops("fp32-daz", VXM_FORMAT_FP32, 32'h00000001, 32'hbf800000);

    // Symmetric MAX selection and exceptional-value matrices in every
    // format. Running ADD/SUB/MUL on the same pairs also covers sign handling.
    for (integer format_index = 0; format_index < 3; format_index++) begin
      logic [1:0] selected_format;
      integer exp_bits, frac_bits, bias, exp_max;
      logic [31:0] plus_one, plus_two, minus_one, minus_two;
      logic [31:0] plus_zero, minus_zero, plus_inf, minus_inf, nan_value;
      logic [31:0] quiet_nan, negative_nan;
      logic [31:0] smallest_subnormal, negative_smallest_subnormal;
      selected_format = (format_index == 0) ? VXM_FORMAT_FP16 :
        ((format_index == 1) ? VXM_FORMAT_BF16 : VXM_FORMAT_FP32);
      exp_bits = (selected_format == VXM_FORMAT_FP16) ? 5 : 8;
      frac_bits = (selected_format == VXM_FORMAT_FP16) ? 10 :
        ((selected_format == VXM_FORMAT_BF16) ? 7 : 23);
      bias = (selected_format == VXM_FORMAT_FP16) ? 15 : 127;
      exp_max = (1 << exp_bits) - 1;
      plus_one = pack_value(exp_bits, frac_bits, 1'b0, bias, 0);
      plus_two = pack_value(exp_bits, frac_bits, 1'b0, bias+1, 0);
      minus_one = pack_value(exp_bits, frac_bits, 1'b1, bias, 0);
      minus_two = pack_value(exp_bits, frac_bits, 1'b1, bias+1, 0);
      plus_zero = pack_value(exp_bits, frac_bits, 1'b0, 0, 0);
      minus_zero = pack_value(exp_bits, frac_bits, 1'b1, 0, 0);
      plus_inf = pack_value(exp_bits, frac_bits, 1'b0, exp_max, 0);
      minus_inf = pack_value(exp_bits, frac_bits, 1'b1, exp_max, 0);
      nan_value = pack_value(exp_bits, frac_bits, 1'b0, exp_max, 1);
      quiet_nan = pack_value(exp_bits, frac_bits, 1'b0, exp_max,
        32'h1 << (frac_bits-1));
      negative_nan = pack_value(exp_bits, frac_bits, 1'b1, exp_max, 1);
      smallest_subnormal = pack_value(exp_bits, frac_bits, 1'b0, 0, 1);
      negative_smallest_subnormal = pack_value(
        exp_bits, frac_bits, 1'b1, 0, 1);
      check_all_ops("positive-order", selected_format, plus_one, plus_two);
      check_all_ops("positive-order-reverse", selected_format, plus_two, plus_one);
      check_all_ops("negative-order", selected_format, minus_one, minus_two);
      check_all_ops("negative-order-reverse", selected_format, minus_two, minus_one);
      check_all_ops("opposite-sign-reverse", selected_format, plus_one, minus_two);
      check_all_ops("same-negative", selected_format, minus_one, minus_one);
      check_all_ops("signed-zero-reverse", selected_format, plus_zero, minus_zero);
      check_all_ops("negative-zero-vs-negative", selected_format, minus_zero, minus_one);
      check_all_ops("zero-vs-positive", selected_format, minus_zero, plus_one);
      check_all_ops("positive-inf", selected_format, plus_one, plus_inf);
      check_all_ops("negative-inf", selected_format, minus_inf, minus_one);
      check_all_ops("opposite-infinities", selected_format, minus_inf, plus_inf);
      check_all_ops("nan-on-right", selected_format, plus_two, nan_value);
      check_all_ops("nan-on-left", selected_format, nan_value, minus_two);
      check_all_ops("two-nans", selected_format, nan_value, nan_value);
      check_case("max-quiet-nan", selected_format, VXM_LOCAL_MAX,
        quiet_nan, plus_one);
      check_case("max-negative-nan", selected_format, VXM_LOCAL_MAX,
        minus_one, negative_nan);
      check_case("max-daz-positive-zero", selected_format, VXM_LOCAL_MAX,
        smallest_subnormal, minus_zero);
      check_case("max-daz-two-subnormals", selected_format, VXM_LOCAL_MAX,
        smallest_subnormal, negative_smallest_subnormal);
    end

    // Deterministic random regression. Raw patterns deliberately include
    // exceptional exponents and both signs; every pair runs all four opcodes.
    for (integer format_index = 0; format_index < 3; format_index++) begin
      logic [1:0] selected_format;
      selected_format = (format_index == 0) ? VXM_FORMAT_FP16 :
        ((format_index == 1) ? VXM_FORMAT_BF16 : VXM_FORMAT_FP32);
      for (integer sample = 0; sample < 300; sample++) begin
        logic [31:0] a, b;
        random_state = next_random(random_state);
        a = random_state;
        random_state = next_random(random_state);
        b = random_state;
        if (selected_format != VXM_FORMAT_FP32) begin
          a = {16'b0, a[15:0]};
          b = {16'b0, b[15:0]};
        end
        // Force adjacent/equal magnitudes regularly to exercise cancellation.
        if ((sample % 10) == 0) b = a;
        if ((sample % 10) == 1) b = a ^ 32'h00000001;
        check_all_ops("random", selected_format, a, b);
      end
    end

    $display("LPU_VXM_ADDSUB_MAX_INSTRUCTION_TB_PASS checks=%0d",
      addsub_max_checks);
    $display("LPU_VXM_MULTIPLY_INSTRUCTION_TB_PASS checks=%0d",
      multiply_checks);
    $finish;
  end
endmodule
