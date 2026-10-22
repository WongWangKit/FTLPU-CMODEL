module lpu_vxm_input_converter #(
  // One stable container carries narrow and wide formats. Current and future
  // VXM datapaths therefore keep the same physical module interface.
  parameter integer CONTAINER_WIDTH = 32
) (
  input  logic                       valid_i,
  input  logic [1:0]                 source_dtype_i,
  input  logic [1:0]                 compute_dtype_i,
  input  logic [CONTAINER_WIDTH-1:0] raw_data_i,

  output logic                       valid_o,
  output logic [CONTAINER_WIDTH-1:0] converted_data_o,
  output logic                       conversion_active_o,
  output logic                       unsupported_conversion_o
);
  import lpu_pkg::*;
  import lpu_vxm_math_pkg::*;
  import lpu_vxm_fp16_pkg::*;

  logic [15:0] fp16_input;

  always_comb begin
    fp16_input = fp16_sanitize_ftz(raw_data_i[15:0]);
    valid_o = 1'b0;
    converted_data_o = '0;
    conversion_active_o = 1'b0;
    unsupported_conversion_o = 1'b0;

    if (valid_i) begin
      case (source_dtype_i)
        VXM_FORMAT_FP16: begin
          case (compute_dtype_i)
            VXM_FORMAT_FP16: begin
              // No conversion: leave payload handling to the selected ALU.
              converted_data_o[15:0] = raw_data_i[15:0];
              valid_o = 1'b1;
            end
            VXM_FORMAT_FP32: begin
              converted_data_o[31:0] = fp16_to_fp32(fp16_input);
              conversion_active_o = 1'b1;
              valid_o = 1'b1;
            end
            VXM_FORMAT_BF16: begin
              converted_data_o[15:0] = fp32_to_bf16_ftz(
                fp16_to_fp32(fp16_input));
              conversion_active_o = 1'b1;
              valid_o = 1'b1;
            end
            default: unsupported_conversion_o = 1'b1;
          endcase
        end
        VXM_FORMAT_BF16: begin
          case (compute_dtype_i)
            VXM_FORMAT_BF16: begin
              converted_data_o[15:0] = raw_data_i[15:0];
              valid_o = 1'b1;
            end
            VXM_FORMAT_FP32: begin
              converted_data_o[31:0] = bf16_to_fp32(raw_data_i[15:0]);
              conversion_active_o = 1'b1;
              valid_o = 1'b1;
            end
            default: unsupported_conversion_o = 1'b1;
          endcase
        end
        VXM_FORMAT_FP32: begin
          if (compute_dtype_i == VXM_FORMAT_FP32) begin
            converted_data_o[31:0] = raw_data_i[31:0];
            valid_o = 1'b1;
          end else if (compute_dtype_i == VXM_FORMAT_BF16) begin
            converted_data_o[15:0] = fp32_to_bf16_ftz(raw_data_i[31:0]);
            conversion_active_o = 1'b1;
            valid_o = 1'b1;
          end else begin
            unsupported_conversion_o = 1'b1;
          end
        end
        default: unsupported_conversion_o = 1'b1;
      endcase
    end
  end
endmodule
