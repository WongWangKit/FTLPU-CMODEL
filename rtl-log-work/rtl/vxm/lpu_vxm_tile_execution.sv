module lpu_vxm_tile_execution #(
  parameter integer STAGES = lpu_pkg::VXM_ALU_COUNT,
  parameter integer LANES = lpu_pkg::LANES_PER_TILE,
  parameter integer CONTAINER_WIDTH = 32,
  parameter integer LUT_BANK_COUNT = 3,
  parameter integer LUT_ENTRY_COUNT = 64,
  parameter integer LUT_BANK_WIDTH =
    LUT_BANK_COUNT <= 1 ? 1 : $clog2(LUT_BANK_COUNT),
  parameter integer LUT_ADDRESS_WIDTH =
    LUT_ENTRY_COUNT <= 1 ? 1 : $clog2(LUT_ENTRY_COUNT),
  parameter integer LUT_STAGE_WIDTH =
    STAGES <= 1 ? 1 : $clog2(STAGES),
  parameter integer EXTERNAL_LUT = 0
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic local_config_load_i,
  output logic local_config_ready_o,
  input  logic [lpu_pkg::VXM_LOCAL_QUEUE_COUNT-1:0]
    local_config_active_i,
  input  logic [lpu_pkg::VXM_LOCAL_QUEUE_COUNT*
               lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    local_config_instruction_i,
  // Execute is repeated while the resident local configuration stays fixed.
  // The final bit is opaque metadata until it returns as config_done.
  input  logic execute_valid_i,
  output logic execute_ready_o,
  input  logic [lpu_pkg::VXM_REPEAT_CONTROL_WIDTH-1:0]
    repeat_control_i,
  output logic config_done_o,
  input  logic global_config_valid_i,
  input  logic [lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH-1:0] global_config_i,

  input  logic [lpu_pkg::STREAMS_PER_DIRECTION-1:0] stream_valid_i,
  input  logic [lpu_pkg::STREAMS_PER_DIRECTION*LANES*8-1:0]
    stream_data_i,
  output logic [lpu_pkg::STREAMS_PER_DIRECTION-1:0] stream_consumed_o,
  input  logic [lpu_pkg::VXM_LOCAL_QUEUE_COUNT-1:0] immediate_valid_i,
  input  logic [lpu_pkg::VXM_LOCAL_QUEUE_COUNT*CONTAINER_WIDTH-1:0]
    immediate_data_i,

  // Function 0/1/2 selects EXP/Reciprocal/Rsqrt. Standalone Tiles instantiate
  // one SRAM per function/Lane; Slice integration supplies a pair-shared set.
  input  logic                         lut_config_valid_i,
  input  logic [LUT_BANK_WIDTH-1:0]    lut_config_bank_i,
  input  logic [15:0]                  lut_config_input_min_i,
  input  logic [15:0]                  lut_config_segment_width_i,
  input  logic                         lut_write_valid_i,
  input  logic [LUT_BANK_WIDTH-1:0]    lut_write_bank_i,
  input  logic [LUT_ADDRESS_WIDTH-1:0] lut_write_address_i,
  input  logic [15:0]                  lut_write_k_i,
  input  logic [15:0]                  lut_write_b_i,
  input  logic [49:0]                  lut_write_log_i,

  input  logic [LUT_BANK_COUNT-1:0]    lut_shared_configured_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_shared_input_min_i,
  input  logic [LUT_BANK_COUNT*16-1:0] lut_shared_segment_width_i,
  output logic [LUT_BANK_COUNT*LANES-1:0]
    lut_lane_request_valid_o,
  output logic [LUT_BANK_COUNT*LANES*LUT_ADDRESS_WIDTH-1:0]
    lut_lane_request_address_o,
  output logic [LUT_BANK_COUNT*LANES*LUT_STAGE_WIDTH-1:0]
    lut_lane_request_stage_o,
  input  logic [LUT_BANK_COUNT*LANES-1:0]
    lut_lane_response_valid_i,
  input  logic [LUT_BANK_COUNT*LANES*LUT_STAGE_WIDTH-1:0]
    lut_lane_response_stage_i,
  input  logic [LUT_BANK_COUNT*LANES*16-1:0]
    lut_lane_response_k_i,
  input  logic [LUT_BANK_COUNT*LANES*16-1:0]
    lut_lane_response_b_i,
  input  logic [LUT_BANK_COUNT*LANES*50-1:0]
    lut_lane_response_log_i,

  // Every physical two-ALU block owns one fixed pair of byte streams for its
  // result. FP16/BF16 use one beat; FP32 uses low-16 then high-16 beats.
  input  logic [lpu_pkg::STREAMS_PER_DIRECTION-1:0] output_ready_i,
  output logic [lpu_pkg::STREAMS_PER_DIRECTION-1:0] output_valid_o,
  output logic [lpu_pkg::STREAMS_PER_DIRECTION*LANES*8-1:0]
    output_data_o,

  // Raw results remain visible for debug. Only decoded chain tails assert
  // tail_valid; architected output traffic uses output_valid/output_data.
  output logic [STAGES*LANES-1:0] tail_valid_o,
  output logic [STAGES*LANES*CONTAINER_WIDTH-1:0] tail_value_o,
  output logic [STAGES*LANES*CONTAINER_WIDTH-1:0] tail_original_o,
  output logic [STAGES*LANES*CONTAINER_WIDTH-1:0] tail_auxiliary_o,
  output logic [STAGES*LANES-1:0] accumulator_state_valid_o,
  output logic [STAGES*LANES*CONTAINER_WIDTH-1:0]
    accumulator_state_data_o,
  output logic [STAGES*LANES-1:0] feedback_state_valid_o,
  output logic [STAGES*LANES*CONTAINER_WIDTH-1:0]
    feedback_state_value_o,
  output logic [STAGES-1:0] instruction_pending_o,
  output logic idle_o,
  output logic fault_o
);
  import lpu_pkg::*;

  localparam integer STREAM_BLOCKS = STAGES/2;

  vxm_global_config_t active_config;
  vxm_global_config_t incoming_config;
  logic config_loaded_q;
  logic [VXM_GLOBAL_CONFIG_WIDTH-1:0] global_config_q;
  logic [STAGES-1:0] active_stage_q;
  logic [STAGES-1:0] pending_valid_q;
  logic [STAGES*VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    pending_instruction_q;
  logic [STAGES*VXM_REPEAT_CONTROL_WIDTH-1:0] pending_repeat_q;
  logic [STAGES*VXM_REPEAT_CONTROL_WIDTH-1:0] single_inflight_repeat_q;
  logic [STAGES-1:0] single_inflight_accumulator_write_q;
  logic [STAGES*VXM_REPEAT_CONTROL_WIDTH-1:0] inflight_repeat_q;
  logic [STAGES-1:0] inflight_accumulator_write_q;
  logic [3:0] inflight_count_q[0:STAGES-1];
  logic [STAGES-1:0] stage_issue_allowed;
  logic c3_pipeline_idle;
  logic pending_end_marker_q;
  logic [STAGES*LANES-1:0] input_token_valid_q;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] input_token_value_q;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] input_token_original_q;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] input_token_auxiliary_q;
  // Exactly one physical sideband route is live: lane 0 of one selected
  // chain. Other lanes remain data-lockstep and do not duplicate the marker.
  logic [STAGES-1:0] input_token_end_marker_q;
  logic [STAGES*LANES-1:0] accumulator_valid_q;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] accumulator_data_q;
  logic [STAGES*LANES-1:0] feedback_valid_q;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] feedback_value_q;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] feedback_original_q;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] feedback_auxiliary_q;
  logic [STREAM_BLOCKS-1:0] output_pending_q;
  logic [STREAM_BLOCKS-1:0] output_phase_q;
  logic [STREAM_BLOCKS-1:0] output_fp32_q;
  logic [STREAM_BLOCKS*LANES*CONTAINER_WIDTH-1:0] output_value_q;
  logic [STREAM_BLOCKS-1:0] lhs_input_phase_q;
  logic [STREAM_BLOCKS-1:0] rhs_input_phase_q;
  logic [STREAM_BLOCKS-1:0] lhs_input_ready_q;
  logic [STREAM_BLOCKS-1:0] rhs_input_ready_q;
  logic [STREAM_BLOCKS*LANES*CONTAINER_WIDTH-1:0]
    lhs_input_data_q;
  logic [STREAM_BLOCKS*LANES*CONTAINER_WIDTH-1:0]
    rhs_input_data_q;
  logic [STREAM_BLOCKS-1:0] lhs_input_capture;
  logic [STREAM_BLOCKS-1:0] rhs_input_capture;

  logic [STAGES*LANES-1:0] feedback_to_stage_valid;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] feedback_to_stage_value;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] feedback_to_stage_original;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] feedback_to_stage_auxiliary;
  logic [STAGES*LANES-1:0] accumulator_to_stage_valid;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] accumulator_to_stage_data;

  logic [STAGES*LANES-1:0] lut_request_valid;
  logic [STAGES*LANES*LUT_BANK_WIDTH-1:0] lut_request_bank;
  logic [STAGES*LANES*LUT_ADDRESS_WIDTH-1:0] lut_request_address;
  logic [STAGES*LANES-1:0] lut_response_valid;
  logic [STAGES*LANES*16-1:0] lut_response_k;
  logic [STAGES*LANES*16-1:0] lut_response_b;
  logic [STAGES*LANES*50-1:0] lut_response_log;
  logic [LUT_BANK_COUNT*LANES*50-1:0] selected_lut_response_log;
  logic [LUT_BANK_COUNT-1:0] lut_configured;
  logic [LUT_BANK_COUNT*16-1:0] lut_input_min;
  logic [LUT_BANK_COUNT*16-1:0] lut_segment_width;
  logic lut_storage_fault;
  logic lut_lane_protocol_fault;
  logic [LUT_BANK_COUNT*LANES-1:0] selected_lut_response_valid;
  logic [LUT_BANK_COUNT*LANES*LUT_STAGE_WIDTH-1:0]
    selected_lut_response_stage;
  logic [LUT_BANK_COUNT*LANES*16-1:0] selected_lut_response_k;
  logic [LUT_BANK_COUNT*LANES*16-1:0] selected_lut_response_b;
  logic internal_lut_collision;

  logic [STAGES*LANES-1:0] lane_input_ready;
  logic [STAGES*LANES-1:0] lane_request_accepted;
  logic [STAGES*LANES-1:0] lane_result_valid;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] lane_result_value;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] lane_result_original;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] lane_result_auxiliary;
  logic [STAGES*LANES-1:0] lane_result_end_marker;
  logic [STAGES*LANES-1:0] lane_fault;
  logic [STAGES*LANES-1:0] lane_chain_head;
  logic [STAGES*LANES-1:0] lane_chain_tail;
  logic [STAGES-1:0] stage_chain_head;
  logic [STAGES-1:0] stage_chain_tail;
  logic [STAGES-1:0] stage_accept;
  logic [STAGES-1:0] stage_result_any;
  logic [STAGES-1:0] stage_result_all;
  logic [STAGES-1:0] stage_result_end_marker;
  logic [STAGES-1:0] stage_emit;
  integer completion_tail_stage;
  integer completion_head_stage;

  always_comb begin
    incoming_config = vxm_global_config_t'(global_config_i);
  end

  function automatic logic chain_head_for(
    input integer physical_stage,
    input logic [1:0] chain_length
  );
    case (chain_length)
      VXM_CHAIN_LENGTH_2:
        chain_head_for = ((physical_stage % 2) == 0);
      VXM_CHAIN_LENGTH_4:
        chain_head_for = ((physical_stage % 4) == 0);
      VXM_CHAIN_LENGTH_8:
        chain_head_for = ((physical_stage % 8) == 0);
      default: chain_head_for = 1'b0;
    endcase
  endfunction

  function automatic logic chain_tail_for(
    input integer physical_stage,
    input logic [1:0] chain_length
  );
    case (chain_length)
      VXM_CHAIN_LENGTH_2:
        chain_tail_for = ((physical_stage % 2) == 1);
      VXM_CHAIN_LENGTH_4:
        chain_tail_for = ((physical_stage % 4) == 3);
      VXM_CHAIN_LENGTH_8:
        chain_tail_for = ((physical_stage % 8) == 7);
      default: chain_tail_for = 1'b0;
    endcase
  endfunction

  function automatic logic accumulator_write_for(
    input integer physical_stage,
    input logic [VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0] instruction,
    input logic [VXM_REPEAT_CONTROL_WIDTH-1:0] repeat_word
  );
    begin
      accumulator_write_for =
        repeat_word[VXM_REPEAT_ACCUMULATOR_BIT] &&
        (((physical_stage % 4) == 1) ||
         ((physical_stage % 4) == 3)) &&
        ((instruction[2:0] == VXM_LOCAL_ADD) ||
         (instruction[2:0] == VXM_LOCAL_MAX)) &&
        (instruction[4:3] == 2'b11);
    end
  endfunction

  function automatic logic feedback_slot_consumed(
    input integer tail_stage,
    input logic [1:0] chain_length
  );
    integer head_stage;
    begin
      case (chain_length)
        VXM_CHAIN_LENGTH_2: head_stage = tail_stage - 1;
        VXM_CHAIN_LENGTH_4: head_stage = tail_stage - 3;
        VXM_CHAIN_LENGTH_8: head_stage = tail_stage - 7;
        default: head_stage = -1;
      endcase
      feedback_slot_consumed = 1'b0;
      if (head_stage >= 0)
        feedback_slot_consumed = stage_accept[head_stage] &&
          (pending_instruction_q[
            head_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+3 +: 2] == 2'b10);
    end
  endfunction

  always_comb begin
    integer feedback_source_stage;
    active_config = vxm_global_config_t'(global_config_q);
    completion_tail_stage = -1;
    completion_head_stage = -1;
    for (integer find_tail = 0; find_tail < STAGES; find_tail++) begin
      if (active_stage_q[find_tail] &&
          chain_tail_for(find_tail, active_config.chain_length))
        completion_tail_stage = find_tail;
    end
    if (completion_tail_stage >= 0) begin
      case (active_config.chain_length)
        VXM_CHAIN_LENGTH_2:
          completion_head_stage = completion_tail_stage - 1;
        VXM_CHAIN_LENGTH_4:
          completion_head_stage = completion_tail_stage - 3;
        VXM_CHAIN_LENGTH_8:
          completion_head_stage = completion_tail_stage - 7;
        default: completion_head_stage = -1;
      endcase
    end
    instruction_pending_o = pending_valid_q;
    execute_ready_o = config_loaded_q && !(|pending_valid_q) &&
      (completion_head_stage >= 0) && !local_config_load_i;
    config_done_o = 1'b0;
    output_valid_o = '0;
    output_data_o = '0;
    for (integer output_block = 0;
         output_block < STREAM_BLOCKS; output_block++) begin
      if (output_pending_q[output_block]) begin
        output_valid_o[output_block*2] = 1'b1;
        output_valid_o[output_block*2+1] = 1'b1;
        for (integer output_lane = 0; output_lane < LANES; output_lane++) begin
          if (output_fp32_q[output_block] && output_phase_q[output_block]) begin
            output_data_o[
              ((output_block*2)*LANES+output_lane)*8 +: 8] =
              output_value_q[
                (output_block*LANES+output_lane)*CONTAINER_WIDTH+16 +: 8];
            output_data_o[
              ((output_block*2+1)*LANES+output_lane)*8 +: 8] =
              output_value_q[
                (output_block*LANES+output_lane)*CONTAINER_WIDTH+24 +: 8];
          end else begin
            output_data_o[
              ((output_block*2)*LANES+output_lane)*8 +: 8] =
              output_value_q[
                (output_block*LANES+output_lane)*CONTAINER_WIDTH +: 8];
            output_data_o[
              ((output_block*2+1)*LANES+output_lane)*8 +: 8] =
              output_value_q[
                (output_block*LANES+output_lane)*CONTAINER_WIDTH+8 +: 8];
          end
        end
      end
    end
    accumulator_state_valid_o = accumulator_valid_q;
    accumulator_state_data_o = accumulator_data_q;
    feedback_state_valid_o = feedback_valid_q;
    feedback_state_value_o = feedback_value_q;
    tail_valid_o = '0;
    tail_value_o = lane_result_value;
    tail_original_o = lane_result_original;
    tail_auxiliary_o = lane_result_auxiliary;
    feedback_to_stage_valid = '0;
    feedback_to_stage_value = '0;
    feedback_to_stage_original = '0;
    feedback_to_stage_auxiliary = '0;
    accumulator_to_stage_valid = accumulator_valid_q;
    accumulator_to_stage_data = accumulator_data_q;
    for (integer stage = 0; stage < STAGES; stage++) begin
      stage_chain_head[stage] = chain_head_for(stage, active_config.chain_length);
      stage_chain_tail[stage] = chain_tail_for(stage, active_config.chain_length);
      stage_accept[stage] = pending_valid_q[stage] &&
        (&lane_request_accepted[stage*LANES +: LANES]);
      stage_result_any[stage] =
        |lane_result_valid[stage*LANES +: LANES];
      stage_result_all[stage] =
        &lane_result_valid[stage*LANES +: LANES];
      stage_result_end_marker[stage] =
        lane_result_end_marker[stage*LANES];
      stage_emit[stage] =
        !inflight_accumulator_write_q[stage] ||
        inflight_repeat_q[stage*VXM_REPEAT_CONTROL_WIDTH +
          VXM_REPEAT_LAST_ITERATION_BIT];
      if (stage_chain_tail[stage] && stage_emit[stage])
        tail_valid_o[stage*LANES +: LANES] =
          lane_result_valid[stage*LANES +: LANES];
      if ((stage == completion_tail_stage) && stage_result_all[stage] &&
          stage_result_end_marker[stage])
        config_done_o = 1'b1;

      // Accumulator reset supplies the operation's neutral value directly to
      // the ALU on the initializing request; the completed result becomes the
      // first stored accumulator value.
      if (pending_valid_q[stage] && accumulator_write_for(
            stage,
            pending_instruction_q[
              stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
                VXM_LOCAL_MAX_INSTRUCTION_WIDTH],
            pending_repeat_q[
              stage*VXM_REPEAT_CONTROL_WIDTH +:
                VXM_REPEAT_CONTROL_WIDTH]) &&
          pending_repeat_q[stage*VXM_REPEAT_CONTROL_WIDTH +
            VXM_REPEAT_FIRST_ITERATION_BIT]) begin
        accumulator_to_stage_valid[stage*LANES +: LANES] = '1;
        for (integer reset_lane = 0; reset_lane < LANES; reset_lane++) begin
          accumulator_to_stage_data[
            (stage*LANES+reset_lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH] =
            pending_instruction_q[
              stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +: 3] == VXM_LOCAL_MAX ?
              (active_config.compute_dtype == VXM_FORMAT_FP32 ?
                32'hff800000 :
                (active_config.compute_dtype == VXM_FORMAT_BF16 ?
                  32'h0000ff80 : 32'h0000fc00)) : 32'h00000000;
        end
      end

      feedback_source_stage = -1;
      if (stage_chain_head[stage]) begin
        case (active_config.chain_length)
          VXM_CHAIN_LENGTH_2: feedback_source_stage = stage + 1;
          VXM_CHAIN_LENGTH_4: feedback_source_stage = stage + 3;
          VXM_CHAIN_LENGTH_8: feedback_source_stage = stage + 7;
          default: feedback_source_stage = -1;
        endcase
      end
      if ((feedback_source_stage >= 0) &&
          (feedback_source_stage < STAGES)) begin
        feedback_to_stage_valid[stage*LANES +: LANES] =
          feedback_valid_q[feedback_source_stage*LANES +: LANES];
        for (integer feedback_lane = 0;
             feedback_lane < LANES; feedback_lane++) begin
          feedback_to_stage_value[
            (stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
              CONTAINER_WIDTH] = feedback_value_q[
            (feedback_source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
              CONTAINER_WIDTH];
          feedback_to_stage_original[
            (stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
              CONTAINER_WIDTH] = feedback_original_q[
            (feedback_source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
              CONTAINER_WIDTH];
          feedback_to_stage_auxiliary[
            (stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
              CONTAINER_WIDTH] = feedback_auxiliary_q[
            (feedback_source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
              CONTAINER_WIDTH];
        end
      end
    end
    idle_o = c3_pipeline_idle && !(|pending_valid_q) && !(|input_token_valid_q) &&
      (&lane_input_ready) && !(|lane_result_valid) && !(|output_pending_q);
    local_config_ready_o = idle_o;
  end

  // A fixed stream pair carries 16 bits per lane. FP16/BF16 consume one beat;
  // FP32 consumes low-half then high-half beats under the per-operand phase
  // state. LHS and RHS collect independently for mixed read widths.
  always_comb begin
    stream_consumed_o = '0;
    lhs_input_capture = '0;
    rhs_input_capture = '0;
    for (integer block = 0; block < STREAM_BLOCKS; block++) begin
      integer head_stage;
      logic head_pending;
      logic lhs_uses_stream;
      logic rhs_uses_stream;
      head_stage = block*2;
      head_pending = pending_valid_q[head_stage] &&
        chain_head_for(head_stage, active_config.chain_length);
      lhs_uses_stream = pending_instruction_q[
        head_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+3 +: 2] == 2'b00;
      if ((head_stage % VXM_LOCAL_QUEUE_COUNT) == 0)
        rhs_uses_stream = !pending_instruction_q[
          head_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+5];
      else
        rhs_uses_stream = pending_instruction_q[
          head_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+5 +: 2] == 2'b00;

      lhs_input_capture[block] = head_pending && lhs_uses_stream &&
        !lhs_input_ready_q[block] &&
        stream_valid_i[block*4] && stream_valid_i[block*4+1];
      rhs_input_capture[block] = head_pending && rhs_uses_stream &&
        !rhs_input_ready_q[block] &&
        stream_valid_i[block*4+2] && stream_valid_i[block*4+3];

      if (lhs_input_capture[block]) begin
        stream_consumed_o[block*4] = 1'b1;
        stream_consumed_o[block*4+1] = 1'b1;
      end
      if (rhs_input_capture[block]) begin
        stream_consumed_o[block*4+2] = 1'b1;
        stream_consumed_o[block*4+3] = 1'b1;
      end
    end
  end

  // Collapse the physical Stage requests into one request per function/Lane.
  // A legal schedule never asks the same function/Lane from two Stages in the
  // same cycle. Different functions and different Lanes remain independent.
  always_comb begin
    lut_lane_request_valid_o = '0;
    lut_lane_request_address_o = '0;
    lut_lane_request_stage_o = '0;
    lut_response_valid = '0;
    lut_response_k = '0;
    lut_response_b = '0;
    lut_response_log = '0;
    lut_lane_protocol_fault = 1'b0;

    for (integer function_id = 0;
         function_id < LUT_BANK_COUNT; function_id++) begin
      for (integer lane_id = 0; lane_id < LANES; lane_id++) begin
        integer lane_function_index;
        lane_function_index = function_id*LANES + lane_id;
        for (integer stage_id = 0; stage_id < STAGES; stage_id++) begin
          integer client_index;
          client_index = stage_id*LANES + lane_id;
          if (lut_request_valid[client_index] &&
              (lut_request_bank[
                client_index*LUT_BANK_WIDTH +: LUT_BANK_WIDTH] ==
               function_id)) begin
            if (lut_lane_request_valid_o[lane_function_index])
              lut_lane_protocol_fault = 1'b1;
            else begin
              lut_lane_request_valid_o[lane_function_index] = 1'b1;
              lut_lane_request_address_o[
                lane_function_index*LUT_ADDRESS_WIDTH +:
                  LUT_ADDRESS_WIDTH] = lut_request_address[
                client_index*LUT_ADDRESS_WIDTH +: LUT_ADDRESS_WIDTH];
              lut_lane_request_stage_o[
                lane_function_index*LUT_STAGE_WIDTH +:
                  LUT_STAGE_WIDTH] = stage_id[LUT_STAGE_WIDTH-1:0];
            end
          end
        end

        if (selected_lut_response_valid[lane_function_index]) begin
          integer response_stage;
          integer response_client;
          response_stage = selected_lut_response_stage[
            lane_function_index*LUT_STAGE_WIDTH +: LUT_STAGE_WIDTH];
          response_client = response_stage*LANES + lane_id;
          if (response_stage >= STAGES)
            lut_lane_protocol_fault = 1'b1;
          else if (lut_response_valid[response_client])
            lut_lane_protocol_fault = 1'b1;
          else begin
            lut_response_valid[response_client] = 1'b1;
            lut_response_k[response_client*16 +: 16] =
              selected_lut_response_k[
                lane_function_index*16 +: 16];
            lut_response_log[response_client*50 +: 50] =
              selected_lut_response_log[lane_function_index*50 +: 50];
            lut_response_b[response_client*16 +: 16] =
              selected_lut_response_b[
                lane_function_index*16 +: 16];
          end
        end
      end
    end

    for (integer client_index = 0;
         client_index < STAGES*LANES; client_index++) begin
      if (lut_request_valid[client_index] &&
          (lut_request_bank[
            client_index*LUT_BANK_WIDTH +: LUT_BANK_WIDTH] >=
           LUT_BANK_COUNT))
        lut_lane_protocol_fault = 1'b1;
    end
  end

  generate
    if (EXTERNAL_LUT != 0) begin : g_external_lut
      assign lut_configured = lut_shared_configured_i;
      assign lut_input_min = lut_shared_input_min_i;
      assign lut_segment_width = lut_shared_segment_width_i;
      assign selected_lut_response_valid = lut_lane_response_valid_i;
      assign selected_lut_response_stage = lut_lane_response_stage_i;
      assign selected_lut_response_k = lut_lane_response_k_i;
      assign selected_lut_response_b = lut_lane_response_b_i;
      assign selected_lut_response_log = lut_lane_response_log_i;
      assign lut_storage_fault = 1'b0;
      assign internal_lut_collision = 1'b0;
    end else begin : g_internal_lut
      localparam integer LANE_FUNCTIONS = LUT_BANK_COUNT*LANES;
      logic [2*LANE_FUNCTIONS-1:0] pair_response_valid;
      logic [2*LANE_FUNCTIONS*LUT_STAGE_WIDTH-1:0]
        pair_response_stage;
      logic [2*LANE_FUNCTIONS*16-1:0] pair_response_k;
      logic [2*LANE_FUNCTIONS*16-1:0] pair_response_b;
      logic [2*LANE_FUNCTIONS*50-1:0] pair_response_log;

      lpu_vxm_tile_pair_lut #(
        .FUNCTION_COUNT(LUT_BANK_COUNT),
        .LANES(LANES),
        .ENTRY_COUNT(LUT_ENTRY_COUNT),
        .STAGE_WIDTH(LUT_STAGE_WIDTH),
        .FUNCTION_WIDTH(LUT_BANK_WIDTH),
        .ADDRESS_WIDTH(LUT_ADDRESS_WIDTH)
      ) u_standalone_lut (
        .clk_i,
        .rst_ni,
        .config_valid_i(lut_config_valid_i),
        .config_function_i(lut_config_bank_i),
        .config_input_min_i(lut_config_input_min_i),
        .config_segment_width_i(lut_config_segment_width_i),
        .write_valid_i(lut_write_valid_i),
        .write_function_i(lut_write_bank_i),
        .write_address_i(lut_write_address_i),
        .write_k_i(lut_write_k_i),
        .write_b_i(lut_write_b_i),
        .write_log_i(lut_write_log_i),
        .request_valid_i({{LANE_FUNCTIONS{1'b0}},
                          lut_lane_request_valid_o}),
        .request_address_i({{LANE_FUNCTIONS*LUT_ADDRESS_WIDTH{1'b0}},
                            lut_lane_request_address_o}),
        .request_stage_i({{LANE_FUNCTIONS*LUT_STAGE_WIDTH{1'b0}},
                          lut_lane_request_stage_o}),
        .response_valid_o(pair_response_valid),
        .response_stage_o(pair_response_stage),
        .response_k_o(pair_response_k),
        .response_b_o(pair_response_b),
        .response_log_o(pair_response_log),
        .configured_o(lut_configured),
        .input_min_o(lut_input_min),
        .segment_width_o(lut_segment_width),
        .collision_o(internal_lut_collision),
        .fault_o(lut_storage_fault)
      );

      assign selected_lut_response_valid =
        pair_response_valid[0 +: LANE_FUNCTIONS];
      assign selected_lut_response_stage = pair_response_stage[
        0 +: LANE_FUNCTIONS*LUT_STAGE_WIDTH];
      assign selected_lut_response_k =
        pair_response_k[0 +: LANE_FUNCTIONS*16];
      assign selected_lut_response_b =
        pair_response_b[0 +: LANE_FUNCTIONS*16];
      assign selected_lut_response_log =
        pair_response_log[0 +: LANE_FUNCTIONS*50];
    end
  endgenerate

  genvar stage;
  genvar lane;
  generate
    for (stage = 0; stage < STAGES; stage = stage + 1) begin : g_stage
      localparam integer LOCAL_QUEUE = stage % VXM_LOCAL_QUEUE_COUNT;
      localparam integer SPECIAL_KIND =
        ((LOCAL_QUEUE == 1) || (LOCAL_QUEUE == 5)) ? VXM_SPECIAL_EXP :
        (((LOCAL_QUEUE == 3) || (LOCAL_QUEUE == 7)) ?
          VXM_SPECIAL_RECIP_RSQRT : VXM_SPECIAL_NONE);
      localparam integer BLOCK = stage / 2;

      wire fixed_lhs_valid = lhs_input_ready_q[BLOCK];
      wire fixed_rhs_valid = rhs_input_ready_q[BLOCK];
      wire feedback_group_valid =
        &feedback_to_stage_valid[stage*LANES +: LANES];
      wire accumulator_group_valid =
        &accumulator_to_stage_valid[stage*LANES +: LANES];

      for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
        wire [CONTAINER_WIDTH-1:0] fixed_lhs_data = lhs_input_data_q[
          (BLOCK*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH];
        wire [CONTAINER_WIDTH-1:0] fixed_rhs_data = rhs_input_data_q[
          (BLOCK*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH];

        lpu_vxm_execution_stage #(
          .LOCAL_QUEUE(LOCAL_QUEUE),
          .PHYSICAL_STAGE(stage),
          .SPECIAL_KIND(SPECIAL_KIND),
          .CONTAINER_WIDTH(CONTAINER_WIDTH),
          .LUT_BANK_COUNT(LUT_BANK_COUNT),
          .LUT_ENTRY_COUNT(LUT_ENTRY_COUNT),
          .LUT_BANK_WIDTH(LUT_BANK_WIDTH),
          .LUT_ADDRESS_WIDTH(LUT_ADDRESS_WIDTH)
        ) u_execution_stage (
          .clk_i,
          .rst_ni,
          // Global configuration is sampled with local_config_load_i. Its
          // input valid is a load qualifier, not an execute-time dependency.
          .instruction_valid_i(
            pending_valid_q[stage] && config_loaded_q && stage_issue_allowed[stage]),
          .instruction_i(pending_instruction_q[
            stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
              VXM_LOCAL_MAX_INSTRUCTION_WIDTH]),
          .chain_length_i(active_config.chain_length),
          .compute_dtype_i(active_config.compute_dtype),
          .lhs_dtype_i(active_config.lhs_dtype),
          .rhs_dtype_i(active_config.rhs_dtype),
          .head_lhs_valid_i(fixed_lhs_valid),
          .head_lhs_data_i(fixed_lhs_data),
          .head_rhs_valid_i(fixed_rhs_valid),
          .head_rhs_data_i(fixed_rhs_data),
          .previous_valid_i(input_token_valid_q[stage*LANES+lane]),
          .previous_value_i(input_token_value_q[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .previous_original_i(input_token_original_q[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .previous_auxiliary_i(input_token_auxiliary_q[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .feedback_valid_i(feedback_group_valid),
          .feedback_value_i(feedback_to_stage_value[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .feedback_original_i(feedback_to_stage_original[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .feedback_auxiliary_i(feedback_to_stage_auxiliary[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .immediate_valid_i(immediate_valid_i[LOCAL_QUEUE]),
          .immediate_data_i(immediate_data_i[
            LOCAL_QUEUE*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .accumulator_valid_i(accumulator_group_valid),
          .accumulator_data_i(accumulator_to_stage_data[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .request_end_marker_i((lane == 0) ?
            (stage_chain_head[stage] ?
              ((stage == completion_head_stage) ?
                pending_end_marker_q : 1'b0) :
              input_token_end_marker_q[stage]) : 1'b0),
          .lut_configured_i(lut_configured),
          .lut_input_min_i(lut_input_min),
          .lut_segment_width_i(lut_segment_width),
          .lut_read_valid_o(lut_request_valid[stage*LANES+lane]),
          .lut_read_bank_o(lut_request_bank[
            (stage*LANES+lane)*LUT_BANK_WIDTH +: LUT_BANK_WIDTH]),
          .lut_read_address_o(lut_request_address[
            (stage*LANES+lane)*LUT_ADDRESS_WIDTH +: LUT_ADDRESS_WIDTH]),
          .lut_read_valid_i(lut_response_valid[stage*LANES+lane]),
          .lut_read_k_i(lut_response_k[(stage*LANES+lane)*16 +: 16]),
          .lut_read_b_i(lut_response_b[(stage*LANES+lane)*16 +: 16]),
          .lut_read_log_i(lut_response_log[(stage*LANES+lane)*50 +: 50]),
          .input_ready_o(lane_input_ready[stage*LANES+lane]),
          .request_accepted_o(lane_request_accepted[stage*LANES+lane]),
          .result_valid_o(lane_result_valid[stage*LANES+lane]),
          .result_value_o(lane_result_value[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .result_original_o(lane_result_original[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .result_auxiliary_o(lane_result_auxiliary[
            (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH]),
          .result_end_marker_o(
            lane_result_end_marker[stage*LANES+lane]),
          .chain_head_o(lane_chain_head[stage*LANES+lane]),
          .chain_tail_o(lane_chain_tail[stage*LANES+lane]),
          .fault_o(lane_fault[stage*LANES+lane])
        );
      end
    end
  endgenerate

  always_comb begin
    c3_pipeline_idle=1'b1;
    for(integer busy_stage=0;busy_stage<STAGES;busy_stage++)
      if(inflight_count_q[busy_stage]!=0)c3_pipeline_idle=1'b0;
  end

  // Track busy independently from ready: an II=1 ALU stays ready while
  // requests are in flight, so ready can no longer serve as an idle flag.
  for(genvar tag_stage=0;tag_stage<STAGES;tag_stage++) begin:g_retirement_tags
    if(tag_stage%4==3) begin:g_c3
      logic [VXM_REPEAT_CONTROL_WIDTH-1:0] repeat_q[0:7];
      logic [7:0] accumulator_q,valid_q;
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if(!rst_ni) begin
          accumulator_q<=0;valid_q<=0;
          for(integer i=0;i<8;i++)repeat_q[i]<=0;
        end else begin
          valid_q<={valid_q[6:0],stage_accept[tag_stage]};
          accumulator_q<={accumulator_q[6:0],stage_accept[tag_stage] &&
            accumulator_write_for(tag_stage,
              pending_instruction_q[tag_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+:VXM_LOCAL_MAX_INSTRUCTION_WIDTH],
              pending_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH])};
          if(stage_accept[tag_stage]) repeat_q[0]<=pending_repeat_q[
            tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH];
          for(integer i=1;i<8;i++) if(valid_q[i-1])repeat_q[i]<=repeat_q[i-1];
        end
      end
      assign inflight_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH]=repeat_q[7];
      assign inflight_accumulator_write_q[tag_stage]=accumulator_q[7];
    end else begin:g_single
      assign inflight_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH]=
        single_inflight_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH];
      assign inflight_accumulator_write_q[tag_stage]=single_inflight_accumulator_write_q[tag_stage];
    end
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if(!rst_ni)inflight_count_q[tag_stage]<=0;
      else case({stage_accept[tag_stage],stage_result_all[tag_stage]})
        2'b10:inflight_count_q[tag_stage]<=inflight_count_q[tag_stage]+1'b1;
        2'b01:inflight_count_q[tag_stage]<=inflight_count_q[tag_stage]-1'b1;
        default:begin end
      endcase
    end
    // Accumulator read-after-write dependencies still serialize. Independent
    // special functions do not wait for older independent elements.
    assign stage_issue_allowed[tag_stage]=inflight_count_q[tag_stage]==0 ||
      !accumulator_write_for(tag_stage,
        pending_instruction_q[tag_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+:VXM_LOCAL_MAX_INSTRUCTION_WIDTH],
        pending_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH]);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      config_loaded_q <= 1'b0;
      global_config_q <= '0;
      active_stage_q <= '0;
      pending_valid_q <= '0;
      pending_instruction_q <= '0;
      pending_repeat_q <= '0;
      single_inflight_repeat_q <= '0;
      single_inflight_accumulator_write_q <= '0;
      pending_end_marker_q <= '0;
      input_token_valid_q <= '0;
      input_token_value_q <= '0;
      input_token_original_q <= '0;
      input_token_auxiliary_q <= '0;
      input_token_end_marker_q <= '0;
      accumulator_valid_q <= '0;
      accumulator_data_q <= '0;
      feedback_valid_q <= '0;
      feedback_value_q <= '0;
      feedback_original_q <= '0;
      feedback_auxiliary_q <= '0;
      output_pending_q <= '0;
      output_phase_q <= '0;
      output_fp32_q <= '0;
      output_value_q <= '0;
      lhs_input_phase_q <= '0;
      rhs_input_phase_q <= '0;
      lhs_input_ready_q <= '0;
      rhs_input_ready_q <= '0;
      lhs_input_data_q <= '0;
      rhs_input_data_q <= '0;
      fault_o <= 1'b0;
    end else begin
      // The two byte streams form one atomic 16-bit beat. FP32 advances from
      // its low half to its high half; FP16/BF16 retire after the first beat.
      for (integer output_block = 0; output_block < STREAM_BLOCKS;
           output_block++) begin
        if (output_pending_q[output_block] &&
            output_ready_i[output_block*2] &&
            output_ready_i[output_block*2+1]) begin
          if (output_fp32_q[output_block] && !output_phase_q[output_block])
            output_phase_q[output_block] <= 1'b1;
          else begin
            output_pending_q[output_block] <= 1'b0;
            output_phase_q[output_block] <= 1'b0;
          end
        end
      end

      // Capture a single 16-bit beat or assemble two FP32 beats.  The low
      // half always arrives first. A complete operand remains resident until
      // its chain-head instruction is accepted.
      for (integer input_block = 0;
           input_block < STREAM_BLOCKS; input_block++) begin
        if (stage_accept[input_block*2]) begin
          lhs_input_ready_q[input_block] <= 1'b0;
          rhs_input_ready_q[input_block] <= 1'b0;
          lhs_input_phase_q[input_block] <= 1'b0;
          rhs_input_phase_q[input_block] <= 1'b0;
        end

        if (lhs_input_capture[input_block]) begin
          for (integer input_lane = 0; input_lane < LANES; input_lane++) begin
            if ((active_config.lhs_read_bits == VXM_READ_BITS_32) &&
                lhs_input_phase_q[input_block])
              lhs_input_data_q[
                (input_block*LANES+input_lane)*CONTAINER_WIDTH+16 +: 16] <= {
                stream_data_i[((input_block*4+1)*LANES+input_lane)*8 +: 8],
                stream_data_i[((input_block*4)*LANES+input_lane)*8 +: 8]};
            else begin
              lhs_input_data_q[
                (input_block*LANES+input_lane)*CONTAINER_WIDTH +: 16] <= {
                stream_data_i[((input_block*4+1)*LANES+input_lane)*8 +: 8],
                stream_data_i[((input_block*4)*LANES+input_lane)*8 +: 8]};
              lhs_input_data_q[
                (input_block*LANES+input_lane)*CONTAINER_WIDTH+16 +: 16] <=
                16'b0;
            end
          end
          if (active_config.lhs_read_bits == VXM_READ_BITS_32) begin
            if (lhs_input_phase_q[input_block]) begin
              lhs_input_ready_q[input_block] <= 1'b1;
              lhs_input_phase_q[input_block] <= 1'b0;
            end else begin
              lhs_input_phase_q[input_block] <= 1'b1;
            end
          end else if (active_config.lhs_read_bits == VXM_READ_BITS_16) begin
            lhs_input_ready_q[input_block] <= 1'b1;
          end else begin
            fault_o <= 1'b1;
          end
        end

        if (rhs_input_capture[input_block]) begin
          for (integer input_lane = 0; input_lane < LANES; input_lane++) begin
            if ((active_config.rhs_read_bits == VXM_READ_BITS_32) &&
                rhs_input_phase_q[input_block])
              rhs_input_data_q[
                (input_block*LANES+input_lane)*CONTAINER_WIDTH+16 +: 16] <= {
                stream_data_i[((input_block*4+3)*LANES+input_lane)*8 +: 8],
                stream_data_i[((input_block*4+2)*LANES+input_lane)*8 +: 8]};
            else begin
              rhs_input_data_q[
                (input_block*LANES+input_lane)*CONTAINER_WIDTH +: 16] <= {
                stream_data_i[((input_block*4+3)*LANES+input_lane)*8 +: 8],
                stream_data_i[((input_block*4+2)*LANES+input_lane)*8 +: 8]};
              rhs_input_data_q[
                (input_block*LANES+input_lane)*CONTAINER_WIDTH+16 +: 16] <=
                16'b0;
            end
          end
          if (active_config.rhs_read_bits == VXM_READ_BITS_32) begin
            if (rhs_input_phase_q[input_block]) begin
              rhs_input_ready_q[input_block] <= 1'b1;
              rhs_input_phase_q[input_block] <= 1'b0;
            end else begin
              rhs_input_phase_q[input_block] <= 1'b1;
            end
          end else if (active_config.rhs_read_bits == VXM_READ_BITS_16) begin
            rhs_input_ready_q[input_block] <= 1'b1;
          end else begin
            fault_o <= 1'b1;
          end
        end
      end

      // A non-head consumes the token held directly before its ALU.
      for (integer clear_stage = 0; clear_stage < STAGES; clear_stage++) begin
        if (stage_accept[clear_stage] && !stage_chain_head[clear_stage]) begin
          input_token_valid_q[clear_stage*LANES +: LANES] <= '0;
          input_token_end_marker_q[clear_stage] <= 1'b0;
        end
        if (stage_result_all[clear_stage]) begin
          single_inflight_repeat_q[
            clear_stage*VXM_REPEAT_CONTROL_WIDTH +:
              VXM_REPEAT_CONTROL_WIDTH] <= '0;
          single_inflight_accumulator_write_q[clear_stage] <= 1'b0;
        end
        if (stage_accept[clear_stage]) begin
          pending_valid_q[clear_stage] <= 1'b0;
          if (clear_stage == completion_head_stage)
            pending_end_marker_q <= 1'b0;
          single_inflight_repeat_q[
            clear_stage*VXM_REPEAT_CONTROL_WIDTH +:
              VXM_REPEAT_CONTROL_WIDTH] <= pending_repeat_q[
            clear_stage*VXM_REPEAT_CONTROL_WIDTH +:
              VXM_REPEAT_CONTROL_WIDTH];
          single_inflight_accumulator_write_q[clear_stage] <=
            accumulator_write_for(
              clear_stage,
              pending_instruction_q[
                clear_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
                  VXM_LOCAL_MAX_INSTRUCTION_WIDTH],
              pending_repeat_q[
                clear_stage*VXM_REPEAT_CONTROL_WIDTH +:
                  VXM_REPEAT_CONTROL_WIDTH]);
        end

        // A Feedback head consumes the fixed tail slot selected by the
        // current chain length. A same-edge new tail result below has
        // priority and replaces the consumed value.
        if (stage_accept[clear_stage] && stage_chain_head[clear_stage] &&
            (pending_instruction_q[
              clear_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+3 +: 2] ==
             2'b10)) begin
          case (active_config.chain_length)
            VXM_CHAIN_LENGTH_2:
              feedback_valid_q[(clear_stage+1)*LANES +: LANES] <= '0;
            VXM_CHAIN_LENGTH_4:
              feedback_valid_q[(clear_stage+3)*LANES +: LANES] <= '0;
            VXM_CHAIN_LENGTH_8:
              feedback_valid_q[(clear_stage+7)*LANES +: LANES] <= '0;
            default: fault_o <= 1'b1;
          endcase
        end
        if ((|lane_request_accepted[clear_stage*LANES +: LANES]) &&
            !(&lane_request_accepted[clear_stage*LANES +: LANES]))
          fault_o <= 1'b1;
        if ((lane_chain_head[clear_stage*LANES +: LANES] !=
             {LANES{stage_chain_head[clear_stage]}}) ||
            (lane_chain_tail[clear_stage*LANES +: LANES] !=
             {LANES{stage_chain_tail[clear_stage]}}))
          fault_o <= 1'b1;
        // Only Lane 0 carries the shared end marker. Shifting the complete
        // Lane vector avoids a zero-width part select when LANES=1.
        if (stage_result_any[clear_stage] &&
            (|(lane_result_end_marker[
              clear_stage*LANES +: LANES] >> 1)))
          fault_o <= 1'b1;
      end

      // Results are lockstep. Non-tail tokens enter the following physical
      // stage; writes have priority over same-edge consumption.
      for (integer source_stage = 0;
           source_stage < STAGES; source_stage++) begin
        if (stage_result_any[source_stage] &&
            !stage_result_all[source_stage])
          fault_o <= 1'b1;
        if (stage_result_all[source_stage] && stage_emit[source_stage] &&
            !stage_chain_tail[source_stage] &&
            (source_stage + 1 < STAGES)) begin
          if ((|input_token_valid_q[(source_stage+1)*LANES +: LANES]) &&
              !(stage_accept[source_stage+1] &&
                !stage_chain_head[source_stage+1]))
            fault_o <= 1'b1;
          input_token_valid_q[(source_stage+1)*LANES +: LANES] <= '1;
          input_token_end_marker_q[source_stage+1] <=
            lane_result_end_marker[source_stage*LANES];
          for (integer result_lane = 0;
               result_lane < LANES; result_lane++) begin
            input_token_value_q[
              ((source_stage+1)*LANES+result_lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH] <= lane_result_value[
              (source_stage*LANES+result_lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH];
            input_token_original_q[
              ((source_stage+1)*LANES+result_lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH] <= lane_result_original[
              (source_stage*LANES+result_lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH];
            input_token_auxiliary_q[
              ((source_stage+1)*LANES+result_lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH] <= lane_result_auxiliary[
              (source_stage*LANES+result_lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH];
          end
        end

        if (stage_result_all[source_stage] &&
            inflight_accumulator_write_q[source_stage]) begin
          accumulator_valid_q[source_stage*LANES +: LANES] <= '1;
          for (integer accumulator_lane = 0;
               accumulator_lane < LANES; accumulator_lane++) begin
            accumulator_data_q[
              (source_stage*LANES+accumulator_lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH] <= lane_result_value[
              (source_stage*LANES+accumulator_lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH];
          end
        end

        if (stage_result_all[source_stage] && stage_emit[source_stage] &&
            stage_chain_tail[source_stage]) begin
          if (inflight_repeat_q[
                source_stage*VXM_REPEAT_CONTROL_WIDTH +
                  VXM_REPEAT_OUTPUT_ENABLE_BIT] &&
              inflight_repeat_q[
                source_stage*VXM_REPEAT_CONTROL_WIDTH +
                  VXM_REPEAT_LAST_ITERATION_BIT]) begin
            if (output_pending_q[source_stage/2] &&
                !(output_ready_i[(source_stage/2)*2] &&
                  output_ready_i[(source_stage/2)*2+1] &&
                  (!output_fp32_q[source_stage/2] ||
                   output_phase_q[source_stage/2])))
              fault_o <= 1'b1;
            output_pending_q[source_stage/2] <= 1'b1;
            output_phase_q[source_stage/2] <= 1'b0;
            output_fp32_q[source_stage/2] <=
              active_config.compute_dtype == VXM_FORMAT_FP32;
            for (integer output_lane = 0;
                 output_lane < LANES; output_lane++) begin
              output_value_q[
                ((source_stage/2)*LANES+output_lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH] <=
                lane_result_value[
                  (source_stage*LANES+output_lane)*CONTAINER_WIDTH +:
                    CONTAINER_WIDTH];
            end
          end else begin
            if ((|feedback_valid_q[source_stage*LANES +: LANES]) &&
                !feedback_slot_consumed(source_stage, active_config.chain_length))
              fault_o <= 1'b1;
            feedback_valid_q[source_stage*LANES +: LANES] <= '1;
            for (integer feedback_lane = 0;
                 feedback_lane < LANES; feedback_lane++) begin
              feedback_value_q[
                (source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH] <= lane_result_value[
                (source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH];
              feedback_original_q[
                (source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH] <= lane_result_original[
                (source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH];
              feedback_auxiliary_q[
                (source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH] <= lane_result_auxiliary[
                (source_stage*LANES+feedback_lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH];
            end
          end
        end
      end

      // Configuration is resident: the eight logical queue words are loaded
      // once and mirrored into the two eight-stage physical halves. Repeat
      // execution below only creates tokens; it does not rewrite instructions.
      if (local_config_load_i && local_config_ready_o) begin
        config_loaded_q <= 1'b1;
        global_config_q <= global_config_i;
        lhs_input_phase_q <= '0;
        rhs_input_phase_q <= '0;
        lhs_input_ready_q <= '0;
        rhs_input_ready_q <= '0;
        if (((incoming_config.lhs_dtype == VXM_FORMAT_FP32) !=
             (incoming_config.lhs_read_bits == VXM_READ_BITS_32)) ||
            ((incoming_config.rhs_dtype == VXM_FORMAT_FP32) !=
             (incoming_config.rhs_read_bits == VXM_READ_BITS_32)) ||
            ((incoming_config.lhs_dtype == VXM_FORMAT_FP16) &&
             (incoming_config.lhs_read_bits != VXM_READ_BITS_16)) ||
            ((incoming_config.rhs_dtype == VXM_FORMAT_FP16) &&
             (incoming_config.rhs_read_bits != VXM_READ_BITS_16)) ||
            ((incoming_config.lhs_dtype == VXM_FORMAT_BF16) &&
             (incoming_config.lhs_read_bits != VXM_READ_BITS_16)) ||
            ((incoming_config.rhs_dtype == VXM_FORMAT_BF16) &&
             (incoming_config.rhs_read_bits != VXM_READ_BITS_16)) ||
            ((incoming_config.compute_dtype != VXM_FORMAT_FP16) &&
             (incoming_config.compute_dtype != VXM_FORMAT_BF16) &&
             (incoming_config.compute_dtype != VXM_FORMAT_FP32)))
          fault_o <= 1'b1;
        for (integer queue = 0;
             queue < VXM_LOCAL_QUEUE_COUNT; queue++) begin
          active_stage_q[queue] <= local_config_active_i[queue];
          active_stage_q[queue+VXM_LOCAL_QUEUE_COUNT] <=
            local_config_active_i[queue];
          pending_instruction_q[
            queue*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
              VXM_LOCAL_MAX_INSTRUCTION_WIDTH] <=
            local_config_instruction_i[
              queue*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
                VXM_LOCAL_MAX_INSTRUCTION_WIDTH];
          pending_instruction_q[
            (queue+VXM_LOCAL_QUEUE_COUNT)*
              VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
              VXM_LOCAL_MAX_INSTRUCTION_WIDTH] <=
            local_config_instruction_i[
              queue*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
                VXM_LOCAL_MAX_INSTRUCTION_WIDTH];
        end
      end

      // One execute token starts every configured ALU in lockstep. Only the
      // final iteration injects the single logical end marker, at the head of
      // one representative active chain. It follows the same registered path
      // as that chain's data and returns as config_done_o at its tail.
      if (execute_valid_i && execute_ready_o) begin
        pending_valid_q <= active_stage_q;
        pending_end_marker_q <=
          repeat_control_i[VXM_REPEAT_LAST_ITERATION_BIT];
        for (integer execute_stage = 0;
             execute_stage < STAGES; execute_stage++) begin
          pending_repeat_q[
            execute_stage*VXM_REPEAT_CONTROL_WIDTH +:
              VXM_REPEAT_CONTROL_WIDTH] <= repeat_control_i;
        end
      end

      if (|lane_fault)
        fault_o <= 1'b1;
      if (lut_storage_fault)
        fault_o <= 1'b1;
      if (lut_lane_protocol_fault || internal_lut_collision)
        fault_o <= 1'b1;
      if (local_config_load_i && local_config_ready_o &&
          !global_config_valid_i)
        fault_o <= 1'b1;
    end
  end

  initial begin
    // LANES=1 is the independently verifiable Lane building block; LANES=8
    // is the production Superlane with shared control and lockstep datapaths.
    if ((STAGES != 16) ||
        ((LANES != 1) && (LANES != lpu_pkg::LANES_PER_TILE)) ||
        (CONTAINER_WIDTH != 32))
      $error("VXM execution requires 16 stages, 1 or 8 lanes, and 32-bit containers");
  end
endmodule
