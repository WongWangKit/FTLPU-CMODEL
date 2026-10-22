`timescale 1ns/1ps

module lpu_vxm_input_converter_tb;
  import lpu_pkg::*;

  logic valid;
  logic [1:0] source_dtype;
  logic [1:0] compute_dtype;
  logic [31:0] raw_data;
  wire output_valid;
  wire [31:0] converted_data;
  wire conversion_active;
  wire unsupported_conversion;

  lpu_vxm_input_converter dut (
    .valid_i(valid),
    .source_dtype_i(source_dtype),
    .compute_dtype_i(compute_dtype),
    .raw_data_i(raw_data),
    .valid_o(output_valid),
    .converted_data_o(converted_data),
    .conversion_active_o(conversion_active),
    .unsupported_conversion_o(unsupported_conversion)
  );

  task automatic expect_conversion(
    input logic [1:0] source,
    input logic [1:0] compute,
    input logic [31:0] input_data,
    input logic expected_valid,
    input logic [31:0] expected_data,
    input logic expected_active,
    input logic expected_unsupported
  );
    begin
      valid = 1'b1;
      source_dtype = source;
      compute_dtype = compute;
      raw_data = input_data;
      #1;
      if ((output_valid !== expected_valid) ||
          (converted_data !== expected_data) ||
          (conversion_active !== expected_active) ||
          (unsupported_conversion !== expected_unsupported)) begin
        $fatal(1,
          "converter src=%0d compute=%0d expected v/data/active/unsupported=%b/%h/%b/%b got=%b/%h/%b/%b",
          source, compute, expected_valid, expected_data, expected_active,
          expected_unsupported, output_valid, converted_data,
          conversion_active, unsupported_conversion);
      end
    end
  endtask

  initial begin
    valid = 1'b0;
    source_dtype = VXM_FORMAT_FP16;
    compute_dtype = VXM_FORMAT_FP32;
    raw_data = '0;
    #1;

    // FP16 1.5 -> FP32 1.5.
    expect_conversion(VXM_FORMAT_FP16, VXM_FORMAT_FP32,
                      32'h00003e00, 1'b1, 32'h3fc00000, 1'b1, 1'b0);
    // A native format passes its low-half payload through unchanged.
    expect_conversion(VXM_FORMAT_FP16, VXM_FORMAT_FP16,
                      32'hffff3c00, 1'b1, 32'h00003c00, 1'b0, 1'b0);
    // FP16 subnormal input is flushed before widening.
    expect_conversion(VXM_FORMAT_FP16, VXM_FORMAT_FP32,
                      32'h00000001, 1'b1, 32'h00000000, 1'b1, 1'b0);
    // A native FP32 input uses the optional-converter bypass.
    expect_conversion(VXM_FORMAT_FP32, VXM_FORMAT_FP32,
                      32'hc0200000, 1'b1, 32'hc0200000, 1'b0, 1'b0);
    // Native BF16 is one 16-bit beat and widens by placing its payload in
    // the upper half of an FP32 value.
    expect_conversion(VXM_FORMAT_BF16, VXM_FORMAT_FP32,
                      32'h00003fc0, 1'b1, 32'h3fc00000, 1'b1, 1'b0);
    expect_conversion(VXM_FORMAT_BF16, VXM_FORMAT_BF16,
                      32'hffff3fc0, 1'b1, 32'h00003fc0, 1'b0, 1'b0);
    expect_conversion(VXM_FORMAT_FP16, VXM_FORMAT_BF16,
                      32'h00003e00, 1'b1, 32'h00003fc0, 1'b1, 1'b0);
    expect_conversion(VXM_FORMAT_FP32, VXM_FORMAT_BF16,
                      32'h3fc00000, 1'b1, 32'h00003fc0, 1'b1, 1'b0);
    // FP32-to-BF16 uses round-to-nearest-even at exact halfway values.
    expect_conversion(VXM_FORMAT_FP32, VXM_FORMAT_BF16,
                      32'h3f808000, 1'b1, 32'h00003f80, 1'b1, 1'b0);
    expect_conversion(VXM_FORMAT_FP32, VXM_FORMAT_BF16,
                      32'h3f818000, 1'b1, 32'h00003f82, 1'b1, 1'b0);
    // Native BF16 retains its raw payload; numeric ALUs apply DAZ/NaN policy.
    expect_conversion(VXM_FORMAT_BF16, VXM_FORMAT_BF16,
                      32'h00000001, 1'b1, 32'h00000001, 1'b0, 1'b0);
    expect_conversion(VXM_FORMAT_BF16, VXM_FORMAT_BF16,
                      32'h00007f81, 1'b1, 32'h00007f81, 1'b0, 1'b0);

    // Raw-bit operations see unmodified subnormal and NaN payloads.
    expect_conversion(VXM_FORMAT_FP16, VXM_FORMAT_FP16,
                      32'hffff0001, 1'b1, 32'h00000001, 1'b0, 1'b0);
    expect_conversion(VXM_FORMAT_FP16, VXM_FORMAT_FP16,
                      32'hffff7e01, 1'b1, 32'h00007e01, 1'b0, 1'b0);
    expect_conversion(VXM_FORMAT_BF16, VXM_FORMAT_BF16,
                      32'hffff0001, 1'b1, 32'h00000001, 1'b0, 1'b0);
    expect_conversion(VXM_FORMAT_BF16, VXM_FORMAT_BF16,
                      32'hffff7fc1, 1'b1, 32'h00007fc1, 1'b0, 1'b0);
    expect_conversion(VXM_FORMAT_RESERVED, VXM_FORMAT_FP32,
                      32'h00000000, 1'b0, 32'h00000000, 1'b0, 1'b1);

    valid = 1'b0;
    #1;
    if (output_valid || unsupported_conversion)
      $fatal(1, "invalid converter input generated an output or a fault");

    $display("LPU_VXM_INPUT_CONVERTER_TB_PASS");
    $finish;
  end
endmodule
