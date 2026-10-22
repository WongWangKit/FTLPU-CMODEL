`timescale 1ns/1ps

// Black-box regression for one Superlane: eight independent Lane datapaths
// share one instruction/configuration wave and must execute in lockstep.
module lpu_vxm_superlane_lockstep_tb;
  import lpu_pkg::*;

  localparam integer STAGES = 16;
  localparam integer LANES = 8;
  localparam integer CONTAINER_WIDTH = 32;

  logic clk;
  logic rst_n;
  logic local_config_load;
  wire local_config_ready;
  logic [VXM_LOCAL_QUEUE_COUNT-1:0] local_config_active;
  logic [VXM_LOCAL_QUEUE_COUNT*VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    local_config_instruction;
  logic execute_valid;
  wire execute_ready;
  logic [VXM_REPEAT_CONTROL_WIDTH-1:0] repeat_control;
  vxm_repeat_control_t repeat_fields;
  wire config_done;
  logic [VXM_GLOBAL_CONFIG_WIDTH-1:0] global_config;
  vxm_global_config_t config_fields;
  logic [STREAMS_PER_DIRECTION-1:0] stream_valid;
  logic [STREAMS_PER_DIRECTION*LANES*8-1:0] stream_data;
  logic [VXM_LOCAL_QUEUE_COUNT-1:0] immediate_valid;
  logic [VXM_LOCAL_QUEUE_COUNT*CONTAINER_WIDTH-1:0] immediate_data;
  wire [STAGES*LANES-1:0] tail_valid;
  wire [STAGES*LANES*CONTAINER_WIDTH-1:0] tail_value;
  wire [STAGES*LANES*CONTAINER_WIDTH-1:0] tail_original;
  wire [STAGES*LANES*CONTAINER_WIDTH-1:0] tail_auxiliary;
  wire [STAGES-1:0] instruction_pending;
  wire idle;
  wire fault;

  logic [STAGES*LANES-1:0] tail_seen;
  logic done_seen;

  always #5 clk = ~clk;

  lpu_vxm_tile_execution #(
    .LANES(LANES)
  ) dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .local_config_load_i(local_config_load),
    .local_config_ready_o(local_config_ready),
    .local_config_active_i(local_config_active),
    .local_config_instruction_i(local_config_instruction),
    .execute_valid_i(execute_valid),
    .execute_ready_o(execute_ready),
    .repeat_control_i(repeat_control),
    .config_done_o(config_done),
    .global_config_valid_i(1'b1),
    .global_config_i(global_config),
    .stream_valid_i(stream_valid),
    .stream_data_i(stream_data),
    .stream_consumed_o(),
    .immediate_valid_i(immediate_valid),
    .immediate_data_i(immediate_data),
    .lut_config_valid_i(1'b0),
    .lut_config_bank_i('0),
    .lut_config_input_min_i('0),
    .lut_config_segment_width_i('0),
    .lut_write_valid_i(1'b0),
    .lut_write_bank_i('0),
    .lut_write_address_i('0),
    .lut_write_k_i('0),
    .lut_write_b_i('0),
    .lut_shared_configured_i('0),
    .lut_shared_input_min_i('0),
    .lut_shared_segment_width_i('0),
    .lut_lane_request_valid_o(),
    .lut_lane_request_address_o(),
    .lut_lane_request_stage_o(),
    .lut_lane_response_valid_i('0),
    .lut_lane_response_stage_i('0),
    .lut_lane_response_k_i('0),
    .lut_lane_response_b_i('0),
    .output_ready_i('1),
    .output_valid_o(),
    .output_data_o(),
    .tail_valid_o(tail_valid),
    .tail_value_o(tail_value),
    .tail_original_o(tail_original),
    .tail_auxiliary_o(tail_auxiliary),
    .accumulator_state_valid_o(),
    .accumulator_state_data_o(),
    .feedback_state_valid_o(),
    .feedback_state_value_o(),
    .instruction_pending_o(instruction_pending),
    .idle_o(idle),
    .fault_o(fault)
  );

  function automatic logic [15:0] fp16_integer(input integer value);
    begin
      case (value)
        1:  fp16_integer = 16'h3c00;
        2:  fp16_integer = 16'h4000;
        3:  fp16_integer = 16'h4200;
        4:  fp16_integer = 16'h4400;
        5:  fp16_integer = 16'h4500;
        6:  fp16_integer = 16'h4600;
        7:  fp16_integer = 16'h4700;
        8:  fp16_integer = 16'h4800;
        9:  fp16_integer = 16'h4880;
        10: fp16_integer = 16'h4900;
        11: fp16_integer = 16'h4980;
        12: fp16_integer = 16'h4a00;
        13: fp16_integer = 16'h4a80;
        14: fp16_integer = 16'h4b00;
        15: fp16_integer = 16'h4b80;
        16: fp16_integer = 16'h4c00;
        17: fp16_integer = 16'h4c40;
        18: fp16_integer = 16'h4c80;
        19: fp16_integer = 16'h4cc0;
        default: fp16_integer = 16'h0000;
      endcase
    end
  endfunction

  function automatic logic stage_is_chain4_tail(input integer stage);
    stage_is_chain4_tail = ((stage % 4) == 3);
  endfunction

  task automatic configure_chain4_adds;
    logic [VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0] instruction;
    begin
      local_config_instruction = '0;
      for (integer queue = 0; queue < VXM_LOCAL_QUEUE_COUNT; queue++) begin
        instruction = '0;
        instruction[2:0] = VXM_LOCAL_ADD;
        case (queue % 4)
          0: begin end                         // Stream + Stream.
          1: instruction[4:3] = 2'd0;         // Previous + Original.
          2: instruction[6:5] = 2'd1;         // Previous + Auxiliary.
          3: instruction[4:3] = 2'd2;         // Previous + Immediate.
        endcase
        local_config_instruction[
          queue*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
            VXM_LOCAL_MAX_INSTRUCTION_WIDTH] = instruction;
      end
    end
  endtask

  task automatic drive_lane_distinct_streams;
    logic [15:0] lhs;
    logic [15:0] rhs;
    begin
      stream_data = '0;
      for (integer block = 0; block < 8; block++) begin
        for (integer lane = 0; lane < LANES; lane++) begin
          lhs = fp16_integer(lane + 1);
          rhs = fp16_integer(1);
          stream_data[((block*4+0)*LANES+lane)*8 +: 8] = lhs[7:0];
          stream_data[((block*4+1)*LANES+lane)*8 +: 8] = lhs[15:8];
          stream_data[((block*4+2)*LANES+lane)*8 +: 8] = rhs[7:0];
          stream_data[((block*4+3)*LANES+lane)*8 +: 8] = rhs[15:8];
        end
      end
    end
  endtask

  task automatic check_results(input integer selected_stage);
    logic [31:0] expected_value;
    logic [31:0] expected_original;
    begin
      for (integer stage = 0; stage < STAGES; stage++) begin
        if (stage_is_chain4_tail(stage) && stage==selected_stage) begin
          for (integer lane = 0; lane < LANES; lane++) begin
            // Per Lane: (lhs + 1) + lhs + 1 + 1 = 2*lhs + 3.
            expected_value = {16'h0000, fp16_integer(2*lane + 5)};
            expected_original = {16'h0000, fp16_integer(lane + 1)};
            if (tail_value[(stage*LANES+lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH] !== expected_value)
              $fatal(1, "Superlane value mismatch stage=%0d lane=%0d",
                     stage, lane);
            if (tail_original[(stage*LANES+lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH] !== expected_original)
              $fatal(1, "Superlane Original crossed lanes stage=%0d lane=%0d",
                     stage, lane);
            if (tail_auxiliary[(stage*LANES+lane)*CONTAINER_WIDTH +:
                  CONTAINER_WIDTH] !== 32'h00003c00)
              $fatal(1, "Superlane Auxiliary mismatch stage=%0d lane=%0d",
                     stage, lane);
          end
        end
      end
    end
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tail_seen <= '0;
      done_seen <= 1'b0;
    end else begin
      tail_seen <= tail_seen | tail_valid;
      done_seen <= done_seen | config_done;
      if (fault)
        $fatal(1, "Superlane DUT reported a fault");
      for (integer stage = 0; stage < STAGES; stage++) begin
        if(|tail_valid[stage*LANES+:LANES])check_results(stage);
        if (!stage_is_chain4_tail(stage) &&
            (|tail_valid[stage*LANES +: LANES]))
          $fatal(1, "non-tail stage asserted tail_valid stage=%0d", stage);
        // A shared instruction wave must never retire only part of the lanes.
        if ((|tail_valid[stage*LANES +: LANES]) &&
            (tail_valid[stage*LANES +: LANES] !== {LANES{1'b1}}))
          $fatal(1, "Superlane lost lockstep stage=%0d valid=%b",
                 stage, tail_valid[stage*LANES +: LANES]);
      end
    end
  end

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    local_config_load = 1'b0;
    local_config_active = '1;
    local_config_instruction = '0;
    execute_valid = 1'b0;
    repeat_fields = '0;
    repeat_fields.output_enable = 1'b1;
    repeat_fields.first_iteration = 1'b1;
    repeat_fields.last_iteration = 1'b1;
    repeat_control = repeat_fields;
    config_fields = '0;
    config_fields.chain_length = VXM_CHAIN_LENGTH_4;
    config_fields.compute_dtype = VXM_FORMAT_FP16;
    config_fields.lhs_dtype = VXM_FORMAT_FP16;
    config_fields.rhs_dtype = VXM_FORMAT_FP16;
    global_config = config_fields;
    stream_valid = '1;
    immediate_valid = '1;
    immediate_data = '0;
    for (integer queue = 0; queue < VXM_LOCAL_QUEUE_COUNT; queue++)
      immediate_data[queue*CONTAINER_WIDTH +: CONTAINER_WIDTH] =
        32'h00003c00;
    configure_chain4_adds();
    drive_lane_distinct_streams();

    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    wait (idle && local_config_ready);
    @(negedge clk);
    local_config_load = 1'b1;
    @(posedge clk);
    #1;
    @(negedge clk);
    local_config_load = 1'b0;
    wait (execute_ready);
    @(negedge clk);
    execute_valid = 1'b1;
    @(posedge clk);
    #1;
    @(negedge clk);
    execute_valid = 1'b0;

    wait (done_seen && idle);
    for (integer stage = 0; stage < STAGES; stage++) begin
      if (stage_is_chain4_tail(stage)) begin
        if (tail_seen[stage*LANES +: LANES] !== {LANES{1'b1}})
          $fatal(1, "Superlane tail missing lanes stage=%0d seen=%b",
                 stage, tail_seen[stage*LANES +: LANES]);
      end else if (|tail_seen[stage*LANES +: LANES]) begin
        $fatal(1, "Superlane non-tail observation stage=%0d", stage);
      end
    end
    if (|instruction_pending)
      $fatal(1, "Superlane left pending instructions");
    // Payload was checked on each public tail_valid pulse.

    $display("LPU_VXM_SUPERLANE_LOCKSTEP_TB_PASS");
    $finish;
  end

  initial begin
    #100000;
    $fatal(1, "Superlane lockstep regression timed out");
  end
endmodule
