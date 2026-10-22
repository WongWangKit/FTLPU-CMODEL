module lpu_vxm_datapath_mux #(
  parameter integer LOCAL_QUEUE = 0,
  parameter integer PHYSICAL_STAGE = LOCAL_QUEUE,
  parameter integer CONTAINER_WIDTH = 32
) (
  input  logic instruction_valid_i,
  input  logic [lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    instruction_i,
  input  logic [1:0] chain_length_i,
  input  logic [1:0] compute_dtype_i,
  input  logic [1:0] lhs_dtype_i,
  input  logic [1:0] rhs_dtype_i,

  // Stream identities are fixed by physical stage and port. This module
  // receives the already selected fixed group and only chooses its use.
  input  logic                       head_lhs_valid_i,
  input  logic [CONTAINER_WIDTH-1:0] head_lhs_data_i,
  input  logic                       head_rhs_valid_i,
  input  logic [CONTAINER_WIDTH-1:0] head_rhs_data_i,

  input  logic                       previous_valid_i,
  input  logic [CONTAINER_WIDTH-1:0] previous_value_i,
  input  logic [CONTAINER_WIDTH-1:0] previous_original_i,
  input  logic [CONTAINER_WIDTH-1:0] previous_auxiliary_i,
  input  logic                       feedback_valid_i,
  input  logic [CONTAINER_WIDTH-1:0] feedback_value_i,
  input  logic [CONTAINER_WIDTH-1:0] feedback_original_i,
  input  logic [CONTAINER_WIDTH-1:0] feedback_auxiliary_i,
  input  logic                       immediate_valid_i,
  input  logic [CONTAINER_WIDTH-1:0] immediate_data_i,
  input  logic                       accumulator_valid_i,
  input  logic [CONTAINER_WIDTH-1:0] accumulator_data_i,

  output logic                       operands_valid_o,
  output logic [3:0]                 opcode_o,
  output logic [CONTAINER_WIDTH-1:0] lhs_o,
  output logic [CONTAINER_WIDTH-1:0] rhs_o,
  output logic [CONTAINER_WIDTH-1:0] token_original_o,
  output logic [CONTAINER_WIDTH-1:0] token_auxiliary_o,
  output logic                       chain_head_o,
  output logic                       chain_tail_o,
  output logic                       decode_fault_o,
  output logic                       conversion_fault_o
);
  import lpu_pkg::*;

  logic [2:0] lhs_source;
  logic [2:0] rhs_source;
  logic decoder_illegal;
  logic lhs_stream_valid;
  logic rhs_stream_valid;
  logic [CONTAINER_WIDTH-1:0] lhs_stream_data;
  logic [CONTAINER_WIDTH-1:0] rhs_stream_data;
  logic lhs_conversion_fault;
  logic rhs_conversion_fault;
  logic lhs_valid;
  logic rhs_valid;
  logic source_token_valid;

  lpu_vxm_local_decoder #(
    .LOCAL_QUEUE(LOCAL_QUEUE),
    .PHYSICAL_STAGE(PHYSICAL_STAGE)
  ) u_decoder (
    .instruction_i,
    .chain_length_i,
    .opcode_o,
    .lhs_source_o(lhs_source),
    .rhs_source_o(rhs_source),
    .chain_head_o,
    .chain_tail_o,
    .illegal_instruction_o(decoder_illegal)
  );

  // Conversion is confined to chain heads. Native FP16/BF16/FP32 payloads
  // pass unchanged; the selected numeric ALU applies DAZ/NaN policy.
  // FP16/BF16-to-FP32 widening, and FP16/FP32-to-BF16 conversion are
  // implemented.
  lpu_vxm_input_converter #(.CONTAINER_WIDTH(CONTAINER_WIDTH)) u_lhs_converter (
    .valid_i(head_lhs_valid_i && chain_head_o &&
      (lhs_source == VXM_SOURCE_STREAM)),
    .source_dtype_i(lhs_dtype_i),
    .compute_dtype_i,
    .raw_data_i(head_lhs_data_i),
    .valid_o(lhs_stream_valid),
    .converted_data_o(lhs_stream_data),
    .conversion_active_o(),
    .unsupported_conversion_o(lhs_conversion_fault)
  );

  lpu_vxm_input_converter #(.CONTAINER_WIDTH(CONTAINER_WIDTH)) u_rhs_converter (
    .valid_i(head_rhs_valid_i && chain_head_o &&
      (rhs_source == VXM_SOURCE_STREAM)),
    .source_dtype_i(rhs_dtype_i),
    .compute_dtype_i,
    .raw_data_i(head_rhs_data_i),
    .valid_o(rhs_stream_valid),
    .converted_data_o(rhs_stream_data),
    .conversion_active_o(),
    .unsupported_conversion_o(rhs_conversion_fault)
  );

  always_comb begin
    lhs_o = '0;
    rhs_o = '0;
    token_original_o = '0;
    token_auxiliary_o = '0;
    lhs_valid = 1'b0;
    rhs_valid = 1'b0;
    source_token_valid = 1'b0;

    if (chain_head_o) begin
      if (lhs_source == VXM_SOURCE_FEEDBACK) begin
        source_token_valid = feedback_valid_i;
        token_original_o = feedback_original_i;
        token_auxiliary_o = feedback_auxiliary_i;
      end else begin
        // A newly created token remembers both selected head operands.
        source_token_valid = 1'b1;
        case (lhs_source)
          VXM_SOURCE_STREAM: begin
            token_original_o = lhs_stream_data;
            source_token_valid = source_token_valid && lhs_stream_valid;
          end
          VXM_SOURCE_IMMEDIATE: begin
            token_original_o = immediate_data_i;
            source_token_valid = source_token_valid && immediate_valid_i;
          end
          default: source_token_valid = 1'b0;
        endcase
        case (rhs_source)
          VXM_SOURCE_STREAM: begin
            token_auxiliary_o = rhs_stream_data;
            source_token_valid = source_token_valid && rhs_stream_valid;
          end
          VXM_SOURCE_IMMEDIATE: begin
            token_auxiliary_o = immediate_data_i;
            source_token_valid = source_token_valid && immediate_valid_i;
          end
          default: source_token_valid = 1'b0;
        endcase
      end
    end else begin
      source_token_valid = previous_valid_i;
      token_original_o = previous_original_i;
      token_auxiliary_o = previous_auxiliary_i;
    end

    case (lhs_source)
      VXM_SOURCE_PREVIOUS: begin
        lhs_o = previous_value_i;
        lhs_valid = previous_valid_i;
      end
      VXM_SOURCE_STREAM: begin
        lhs_o = lhs_stream_data;
        lhs_valid = lhs_stream_valid;
      end
      VXM_SOURCE_IMMEDIATE: begin
        lhs_o = immediate_data_i;
        lhs_valid = immediate_valid_i;
      end
      VXM_SOURCE_FEEDBACK: begin
        lhs_o = feedback_value_i;
        lhs_valid = feedback_valid_i;
      end
      default: lhs_valid = 1'b0;
    endcase

    case (rhs_source)
      VXM_SOURCE_STREAM: begin
        rhs_o = rhs_stream_data;
        rhs_valid = rhs_stream_valid;
      end
      VXM_SOURCE_ORIGINAL: begin
        rhs_o = previous_original_i;
        rhs_valid = previous_valid_i;
      end
      VXM_SOURCE_AUXILIARY: begin
        rhs_o = previous_auxiliary_i;
        rhs_valid = previous_valid_i;
      end
      VXM_SOURCE_IMMEDIATE: begin
        rhs_o = immediate_data_i;
        rhs_valid = immediate_valid_i;
      end
      VXM_SOURCE_ACCUMULATOR: begin
        rhs_o = accumulator_data_i;
        rhs_valid = accumulator_valid_i;
      end
      default: rhs_valid = 1'b0;
    endcase

    decode_fault_o = instruction_valid_i && decoder_illegal;
    conversion_fault_o = instruction_valid_i &&
      (lhs_conversion_fault || rhs_conversion_fault);
    operands_valid_o = instruction_valid_i && !decoder_illegal &&
      !conversion_fault_o && source_token_valid && lhs_valid && rhs_valid;
  end
endmodule
