module lpu_vxm_datapath_stage #(
  parameter integer LOCAL_QUEUE = 0,
  parameter integer PHYSICAL_STAGE = LOCAL_QUEUE,
  parameter integer LANES = lpu_pkg::LANES_PER_TILE,
  parameter integer CONTAINER_WIDTH = 32
) (
  input  logic instruction_valid_i,
  input  logic [lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    instruction_i,
  input  logic [1:0] chain_length_i,
  input  logic [1:0] compute_dtype_i,
  input  logic [1:0] lhs_dtype_i,
  input  logic [1:0] rhs_dtype_i,

  input  logic [lpu_pkg::STREAMS_PER_DIRECTION-1:0] stream_valid_i,
  input  logic [lpu_pkg::STREAMS_PER_DIRECTION*LANES*8-1:0]
    stream_data_i,

  input  logic [LANES-1:0] previous_valid_i,
  input  logic [LANES*CONTAINER_WIDTH-1:0] previous_value_i,
  input  logic [LANES*CONTAINER_WIDTH-1:0] previous_original_i,
  input  logic [LANES*CONTAINER_WIDTH-1:0] previous_auxiliary_i,
  input  logic [LANES-1:0] feedback_valid_i,
  input  logic [LANES*CONTAINER_WIDTH-1:0] feedback_value_i,
  input  logic [LANES*CONTAINER_WIDTH-1:0] feedback_original_i,
  input  logic [LANES*CONTAINER_WIDTH-1:0] feedback_auxiliary_i,
  input  logic                       immediate_valid_i,
  input  logic [CONTAINER_WIDTH-1:0] immediate_data_i,
  input  logic [LANES-1:0] accumulator_valid_i,
  input  logic [LANES*CONTAINER_WIDTH-1:0] accumulator_data_i,

  output logic [LANES-1:0] operands_valid_o,
  output logic [3:0] opcode_o,
  output logic [LANES*CONTAINER_WIDTH-1:0] lhs_o,
  output logic [LANES*CONTAINER_WIDTH-1:0] rhs_o,
  output logic [LANES*CONTAINER_WIDTH-1:0] token_original_o,
  output logic [LANES*CONTAINER_WIDTH-1:0] token_auxiliary_o,
  output logic chain_head_o,
  output logic chain_tail_o,
  output logic decode_fault_o,
  output logic conversion_fault_o
);
  logic fixed_lhs_valid;
  logic fixed_rhs_valid;
  logic [LANES*CONTAINER_WIDTH-1:0] fixed_lhs_data;
  logic [LANES*CONTAINER_WIDTH-1:0] fixed_rhs_data;
  logic [LANES*4-1:0] lane_opcodes;
  logic [LANES-1:0] lane_chain_head;
  logic [LANES-1:0] lane_chain_tail;
  logic [LANES-1:0] lane_decode_fault;
  logic [LANES-1:0] lane_conversion_fault;

  lpu_vxm_fp16_stream_groups #(
    .PHYSICAL_STAGE(PHYSICAL_STAGE),
    .LANES(LANES),
    .CONTAINER_WIDTH(CONTAINER_WIDTH)
  ) u_fixed_stream_groups (
    .stream_valid_i,
    .stream_data_i,
    .lhs_valid_o(fixed_lhs_valid),
    .lhs_data_o(fixed_lhs_data),
    .rhs_valid_o(fixed_rhs_valid),
    .rhs_data_o(fixed_rhs_data)
  );

  genvar lane;
  generate
    for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane_mux
      lpu_vxm_datapath_mux #(
        .LOCAL_QUEUE(LOCAL_QUEUE),
        .PHYSICAL_STAGE(PHYSICAL_STAGE),
        .CONTAINER_WIDTH(CONTAINER_WIDTH)
      ) u_mux (
        .instruction_valid_i,
        .instruction_i,
        .chain_length_i,
        .compute_dtype_i,
        .lhs_dtype_i,
        .rhs_dtype_i,
        .head_lhs_valid_i(fixed_lhs_valid),
        .head_lhs_data_i(
          fixed_lhs_data[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .head_rhs_valid_i(fixed_rhs_valid),
        .head_rhs_data_i(
          fixed_rhs_data[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .previous_valid_i(previous_valid_i[lane]),
        .previous_value_i(
          previous_value_i[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .previous_original_i(
          previous_original_i[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .previous_auxiliary_i(
          previous_auxiliary_i[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .feedback_valid_i(feedback_valid_i[lane]),
        .feedback_value_i(
          feedback_value_i[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .feedback_original_i(
          feedback_original_i[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .feedback_auxiliary_i(
          feedback_auxiliary_i[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .immediate_valid_i,
        .immediate_data_i,
        .accumulator_valid_i(accumulator_valid_i[lane]),
        .accumulator_data_i(
          accumulator_data_i[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .operands_valid_o(operands_valid_o[lane]),
        .opcode_o(lane_opcodes[lane*4 +: 4]),
        .lhs_o(lhs_o[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .rhs_o(rhs_o[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .token_original_o(
          token_original_o[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .token_auxiliary_o(
          token_auxiliary_o[lane*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
        .chain_head_o(lane_chain_head[lane]),
        .chain_tail_o(lane_chain_tail[lane]),
        .decode_fault_o(lane_decode_fault[lane]),
        .conversion_fault_o(lane_conversion_fault[lane])
      );
    end
  endgenerate

  always_comb begin
    opcode_o = lane_opcodes[3:0];
    chain_head_o = lane_chain_head[0];
    chain_tail_o = lane_chain_tail[0];
    decode_fault_o = |lane_decode_fault;
    conversion_fault_o = |lane_conversion_fault;
  end
endmodule
