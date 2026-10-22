module lpu_vxm_basic_frontend_tb;
  import lpu_pkg::*;

  logic [1:0]  data_format;
  logic [31:0] value;
  wire         format_valid;
  wire         sign;
  wire [7:0]   exponent;
  wire [22:0]  fraction;
  wire         effective_zero;
  wire         was_subnormal;
  wire         is_normal;
  wire         is_inf;
  wire         is_nan;

  logic [2:0] opcode;
  wire        opcode_valid;
  wire        op_bypass;
  wire        op_add;
  wire        op_subtract;
  wire        op_multiply;
  wire        op_negate;
  wire        op_max;

  lpu_vxm_float_unpack u_unpack (
    .data_format_i(data_format),
    .value_i(value),
    .format_valid_o(format_valid),
    .sign_o(sign),
    .exponent_o(exponent),
    .fraction_o(fraction),
    .effective_zero_o(effective_zero),
    .was_subnormal_o(was_subnormal),
    .is_normal_o(is_normal),
    .is_inf_o(is_inf),
    .is_nan_o(is_nan)
  );

  lpu_vxm_basic_opcode_decode u_decode (
    .opcode_i(opcode),
    .opcode_valid_o(opcode_valid),
    .bypass_o(op_bypass),
    .add_o(op_add),
    .subtract_o(op_subtract),
    .multiply_o(op_multiply),
    .negate_o(op_negate),
    .max_o(op_max)
  );

  task automatic check_unpack(
    input logic [1:0]  check_format,
    input logic [31:0] check_value,
    input logic         expected_valid,
    input logic         expected_sign,
    input logic [7:0]   expected_exponent,
    input logic [22:0]  expected_fraction,
    input logic [4:0]   expected_class
  );
    begin
      data_format = check_format;
      value = check_value;
      #1;
      if ({effective_zero, was_subnormal, is_normal, is_inf, is_nan} !==
          expected_class || format_valid !== expected_valid ||
          sign !== expected_sign || exponent !== expected_exponent ||
          fraction !== expected_fraction)
        $fatal(1,
          "unpack mismatch format=%0d value=%h valid=%b sign=%b exp=%h frac=%h class=%b",
          check_format, check_value, format_valid, sign, exponent, fraction,
          {effective_zero, was_subnormal, is_normal, is_inf, is_nan});
    end
  endtask

  initial begin
    data_format = VXM_FORMAT_FP16;
    value = '0;
    opcode = VXM_LOCAL_BYPASS;

    // class order: {effective_zero, was_subnormal, normal, inf, nan}
    check_unpack(VXM_FORMAT_FP16, 32'h0000bc01, 1'b1, 1'b1,
                 8'h0f, 23'h000001, 5'b00100);
    check_unpack(VXM_FORMAT_FP16, 32'h00008001, 1'b1, 1'b1,
                 8'h00, 23'h000000, 5'b11000);
    check_unpack(VXM_FORMAT_FP16, 32'h00007c00, 1'b1, 1'b0,
                 8'h1f, 23'h000000, 5'b00010);
    check_unpack(VXM_FORMAT_FP16, 32'h00007e01, 1'b1, 1'b0,
                 8'h1f, 23'h000201, 5'b00001);
    // FP16's upper input bits and padded exponent bits are not classified.
    check_unpack(VXM_FORMAT_FP16, 32'hffff7c00, 1'b1, 1'b0,
                 8'h1f, 23'h000000, 5'b00010);
    check_unpack(VXM_FORMAT_FP16, 32'h00007e00, 1'b1, 1'b0,
                 8'h1f, 23'h000200, 5'b00001);
    check_unpack(VXM_FORMAT_FP16, 32'h00000200, 1'b1, 1'b0,
                 8'h00, 23'h000000, 5'b11000);

    check_unpack(VXM_FORMAT_BF16, 32'h00003fc1, 1'b1, 1'b0,
                 8'h7f, 23'h000041, 5'b00100);
    check_unpack(VXM_FORMAT_BF16, 32'h00008001, 1'b1, 1'b1,
                 8'h00, 23'h000000, 5'b11000);
    check_unpack(VXM_FORMAT_BF16, 32'h00007f80, 1'b1, 1'b0,
                 8'hff, 23'h000000, 5'b00010);
    check_unpack(VXM_FORMAT_BF16, 32'h00007fc1, 1'b1, 1'b0,
                 8'hff, 23'h000041, 5'b00001);
    // Low five exponent bits alone must not classify a wide exponent.
    check_unpack(VXM_FORMAT_BF16, 32'h00000f80, 1'b1, 1'b0,
                 8'h1f, 23'h000000, 5'b00100);
    check_unpack(VXM_FORMAT_BF16, 32'h00001000, 1'b1, 1'b0,
                 8'h20, 23'h000000, 5'b00100);
    check_unpack(VXM_FORMAT_BF16, 32'h00007fff, 1'b1, 1'b0,
                 8'hff, 23'h00007f, 5'b00001);

    check_unpack(VXM_FORMAT_FP32, 32'hbf800001, 1'b1, 1'b1,
                 8'h7f, 23'h000001, 5'b00100);
    check_unpack(VXM_FORMAT_FP32, 32'h80000001, 1'b1, 1'b1,
                 8'h00, 23'h000000, 5'b11000);
    check_unpack(VXM_FORMAT_FP32, 32'h7f800000, 1'b1, 1'b0,
                 8'hff, 23'h000000, 5'b00010);
    check_unpack(VXM_FORMAT_FP32, 32'h7fc00001, 1'b1, 1'b0,
                 8'hff, 23'h400001, 5'b00001);
    check_unpack(VXM_FORMAT_FP32, 32'h0f800000, 1'b1, 1'b0,
                 8'h1f, 23'h000000, 5'b00100);
    check_unpack(VXM_FORMAT_FP32, 32'h10000000, 1'b1, 1'b0,
                 8'h20, 23'h000000, 5'b00100);
    check_unpack(VXM_FORMAT_FP32, 32'h7f800400, 1'b1, 1'b0,
                 8'hff, 23'h000400, 5'b00001);
    check_unpack(VXM_FORMAT_FP32, 32'h00400000, 1'b1, 1'b0,
                 8'h00, 23'h000000, 5'b11000);

    check_unpack(VXM_FORMAT_RESERVED, 32'hffffffff, 1'b0, 1'b0,
                 8'h00, 23'h000000, 5'b00000);

    for (integer operation = 0; operation < 8; operation++) begin
      opcode = operation[2:0];
      #1;
      if (operation < 6) begin
        if (!opcode_valid ||
            {op_max, op_negate, op_multiply, op_subtract, op_add, op_bypass}
              !== (6'b000001 << operation))
          $fatal(1, "opcode %0d did not produce exactly one Basic enable",
                 operation);
      end else if (opcode_valid || op_bypass || op_add || op_subtract ||
                   op_multiply || op_negate || op_max)
        $fatal(1, "special opcode %0d was accepted by Basic decoder",
               operation);
    end

    $display("LPU_VXM_BASIC_FRONTEND_TB_PASS");
    $finish;
  end
endmodule
