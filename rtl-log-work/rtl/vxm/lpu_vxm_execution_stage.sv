module lpu_vxm_execution_stage #(
  parameter integer LOCAL_QUEUE = 0,
  parameter integer PHYSICAL_STAGE = LOCAL_QUEUE,
  parameter integer SPECIAL_KIND = lpu_pkg::VXM_SPECIAL_NONE,
  parameter integer CONTAINER_WIDTH = 32,
  parameter integer LUT_BANK_COUNT = 3,
  parameter integer LUT_ENTRY_COUNT = 64,
  parameter integer LUT_BANK_WIDTH =
    LUT_BANK_COUNT <= 1 ? 1 : $clog2(LUT_BANK_COUNT),
  parameter integer LUT_ADDRESS_WIDTH =
    LUT_ENTRY_COUNT <= 1 ? 1 : $clog2(LUT_ENTRY_COUNT)
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic instruction_valid_i,
  input  logic [lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    instruction_i,
  input  logic [1:0] chain_length_i,
  input  logic [1:0] compute_dtype_i,
  input  logic [1:0] lhs_dtype_i,
  input  logic [1:0] rhs_dtype_i,

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
  // Opaque control metadata. The ALU never interprets Repeat state; its
  // wrapper merely keeps this bit aligned with the accepted request.
  input  logic                       request_end_marker_i,

  input  logic [LUT_BANK_COUNT-1:0]    lut_configured_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_input_min_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_segment_width_i,
  output logic                         lut_read_valid_o,
  output logic [LUT_BANK_WIDTH-1:0]    lut_read_bank_o,
  output logic [LUT_ADDRESS_WIDTH-1:0] lut_read_address_o,
  input  logic                         lut_read_valid_i,
  input  logic [15:0]                  lut_read_k_i,
  input  logic [15:0]                  lut_read_b_i,
  input  logic [49:0]                  lut_read_log_i,

  output logic input_ready_o,
  output logic request_accepted_o,
  output logic result_valid_o,
  output logic [CONTAINER_WIDTH-1:0] result_value_o,
  output logic [CONTAINER_WIDTH-1:0] result_original_o,
  output logic [CONTAINER_WIDTH-1:0] result_auxiliary_o,
  output logic result_end_marker_o,
  output logic chain_head_o,
  output logic chain_tail_o,
  output logic fault_o
);
  logic operands_valid;
  logic [3:0] opcode;
  logic [CONTAINER_WIDTH-1:0] lhs;
  logic [CONTAINER_WIDTH-1:0] rhs;
  logic [CONTAINER_WIDTH-1:0] request_original;
  logic [CONTAINER_WIDTH-1:0] request_auxiliary;
  logic decode_fault;
  logic conversion_fault;

  logic alu_input_valid;
  logic alu_input_ready;
  logic alu_output_valid;
  logic [31:0] alu_result;
  logic alu_illegal_opcode;
  logic alu_unsupported_format;
  logic alu_result_collision;
  logic alu_lut_fault;

  localparam bit C3_PIPELINED=SPECIAL_KIND==lpu_pkg::VXM_SPECIAL_RECIP_RSQRT;
  logic metadata_valid_q;
  logic [CONTAINER_WIDTH-1:0] metadata_original_q,metadata_auxiliary_q;
  logic metadata_end_marker_q;
  logic [7:0] tag_valid_q,tag_end_q;
  logic [CONTAINER_WIDTH-1:0] tag_original_q[0:7],tag_auxiliary_q[0:7];

  lpu_vxm_datapath_mux #(
    .LOCAL_QUEUE(LOCAL_QUEUE),
    .PHYSICAL_STAGE(PHYSICAL_STAGE),
    .CONTAINER_WIDTH(CONTAINER_WIDTH)
  ) u_operand_datapath (
    .instruction_valid_i,
    .instruction_i,
    .chain_length_i,
    .compute_dtype_i,
    .lhs_dtype_i,
    .rhs_dtype_i,
    .head_lhs_valid_i,
    .head_lhs_data_i,
    .head_rhs_valid_i,
    .head_rhs_data_i,
    .previous_valid_i,
    .previous_value_i,
    .previous_original_i,
    .previous_auxiliary_i,
    .feedback_valid_i,
    .feedback_value_i,
    .feedback_original_i,
    .feedback_auxiliary_i,
    .immediate_valid_i,
    .immediate_data_i,
    .accumulator_valid_i,
    .accumulator_data_i,
    .operands_valid_o(operands_valid),
    .opcode_o(opcode),
    .lhs_o(lhs),
    .rhs_o(rhs),
    .token_original_o(request_original),
    .token_auxiliary_o(request_auxiliary),
    .chain_head_o,
    .chain_tail_o,
    .decode_fault_o(decode_fault),
    .conversion_fault_o(conversion_fault)
  );

  always_comb begin
    // The finishing result owns the old metadata during this cycle; a new
    // request may be accepted at the same edge and replace it for the next.
    input_ready_o = alu_input_ready && (C3_PIPELINED || !metadata_valid_q || alu_output_valid);
    alu_input_valid = operands_valid && (C3_PIPELINED || !metadata_valid_q || alu_output_valid);
    request_accepted_o = alu_input_valid && alu_input_ready &&
      !alu_illegal_opcode && !alu_unsupported_format;
    result_valid_o = alu_output_valid && metadata_valid_q;
    result_value_o = alu_result[CONTAINER_WIDTH-1:0];
    result_original_o = metadata_original_q;
    result_auxiliary_o = metadata_auxiliary_q;
    result_end_marker_o = metadata_end_marker_q;
    fault_o = decode_fault || conversion_fault || alu_illegal_opcode ||
      alu_unsupported_format || alu_result_collision || alu_lut_fault ||
      (alu_output_valid && !metadata_valid_q);
  end

  generate if(C3_PIPELINED) begin:g_tags
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if(!rst_ni) begin
        tag_valid_q<=0;tag_end_q<=0;
        for(integer i=0;i<8;i++) begin tag_original_q[i]<=0;tag_auxiliary_q[i]<=0;end
      end else begin
        tag_valid_q<={tag_valid_q[6:0],request_accepted_o};
        tag_end_q<={tag_end_q[6:0],request_accepted_o && request_end_marker_i};
        if(request_accepted_o) begin tag_original_q[0]<=request_original;tag_auxiliary_q[0]<=request_auxiliary;end
        for(integer i=1;i<8;i++) if(tag_valid_q[i-1]) begin
          tag_original_q[i]<=tag_original_q[i-1];tag_auxiliary_q[i]<=tag_auxiliary_q[i-1];
        end
      end
    end
    assign metadata_valid_q=tag_valid_q[7];
    assign metadata_original_q=tag_original_q[7];
    assign metadata_auxiliary_q=tag_auxiliary_q[7];
    assign metadata_end_marker_q=tag_end_q[7];
  end else begin:g_single_tag
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      metadata_valid_q <= 1'b0;
      metadata_original_q <= '0;
      metadata_auxiliary_q <= '0;
      metadata_end_marker_q <= 1'b0;
    end else begin
      if (alu_output_valid)
        metadata_valid_q <= 1'b0;
      if (request_accepted_o) begin
        metadata_valid_q <= 1'b1;
        metadata_original_q <= request_original;
        metadata_auxiliary_q <= request_auxiliary;
        metadata_end_marker_q <= request_end_marker_i;
      end
    end
  end

  end endgenerate

  lpu_vxm_alu #(
    .SPECIAL_KIND(SPECIAL_KIND),
    .LUT_BANK_COUNT(LUT_BANK_COUNT),
    .LUT_ENTRY_COUNT(LUT_ENTRY_COUNT),
    .LUT_BANK_WIDTH(LUT_BANK_WIDTH),
    .LUT_ADDRESS_WIDTH(LUT_ADDRESS_WIDTH)
  ) u_alu (
    .clk_i,
    .rst_ni,
    .input_valid_i(alu_input_valid),
    .data_format_i(compute_dtype_i),
    .opcode_i(opcode),
    .lhs_i(lhs[31:0]),
    .rhs_i(rhs[31:0]),
    .lut_configured_i,
    .lut_input_min_i,
    .lut_segment_width_i,
    .lut_read_valid_o,
    .lut_read_bank_o,
    .lut_read_address_o,
    .lut_read_valid_i,
    .lut_read_k_i,
    .lut_read_b_i,
    .lut_read_log_i,
    .input_ready_o(alu_input_ready),
    .output_valid_o(alu_output_valid),
    .result_o(alu_result),
    .illegal_opcode_o(alu_illegal_opcode),
    .unsupported_format_o(alu_unsupported_format),
    .result_collision_o(alu_result_collision),
    .lut_fault_o(alu_lut_fault)
  );

  initial begin
    if (CONTAINER_WIDTH != 32)
      $error("Current VXM ALU binding requires a 32-bit data container");
  end
endmodule
