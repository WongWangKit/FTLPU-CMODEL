`timescale 1ns/1ps

module lpu_vxm_tile_special_paths_tb;
  import lpu_pkg::*;

  localparam integer STAGES = 16;
  localparam integer LANES = 8;
  localparam integer CONTAINER_WIDTH = 32;
  localparam integer LUT_BANK_COUNT = 3;
  localparam integer LUT_ENTRY_COUNT = 64;
  localparam integer LUT_BANK_WIDTH = 2;
  localparam integer LUT_ADDRESS_WIDTH = 6;

  logic clk;
  logic rst_n;
  logic local_config_load;
  wire local_config_ready;
  logic [VXM_LOCAL_QUEUE_COUNT-1:0] local_config_active;
  logic [VXM_LOCAL_QUEUE_COUNT*VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    local_config_instruction;
  logic execute_valid;
  wire execute_ready;
  wire config_done;
  logic [VXM_REPEAT_CONTROL_WIDTH-1:0] repeat_control;
  vxm_repeat_control_t repeat_fields;
  logic global_config_valid;
  logic [VXM_GLOBAL_CONFIG_WIDTH-1:0] global_config;
  vxm_global_config_t config_fields;
  logic [STREAMS_PER_DIRECTION-1:0] stream_valid;
  logic [STREAMS_PER_DIRECTION*LANES*8-1:0] stream_data;
  wire [STREAMS_PER_DIRECTION-1:0] stream_consumed;
  logic [VXM_LOCAL_QUEUE_COUNT-1:0] immediate_valid;
  logic [VXM_LOCAL_QUEUE_COUNT*CONTAINER_WIDTH-1:0] immediate_data;
  logic lut_config_valid;
  logic [LUT_BANK_WIDTH-1:0] lut_config_bank;
  logic [15:0] lut_config_input_min;
  logic [15:0] lut_config_segment_width;
  logic lut_write_valid;
  logic [LUT_BANK_WIDTH-1:0] lut_write_bank;
  logic [LUT_ADDRESS_WIDTH-1:0] lut_write_address;
  logic [15:0] lut_write_k;
  logic [15:0] lut_write_b;
  logic [STREAMS_PER_DIRECTION-1:0] output_ready;

  wire [STREAMS_PER_DIRECTION-1:0] output_valid;
  wire [STREAMS_PER_DIRECTION*LANES*8-1:0] output_data;
  wire [STAGES*LANES-1:0] tail_valid;
  wire [STAGES*LANES*CONTAINER_WIDTH-1:0] tail_value;
  wire [STAGES*LANES-1:0] accumulator_state_valid;
  wire [STAGES*LANES*CONTAINER_WIDTH-1:0] accumulator_state_data;
  wire [STAGES*LANES-1:0] feedback_state_valid;
  wire [STAGES*LANES*CONTAINER_WIDTH-1:0] feedback_state_value;
  wire [STAGES-1:0] instruction_pending;
  wire idle;
  wire fault;
  integer config_done_count;

  always #5 clk = ~clk;

  lpu_vxm_tile_execution dut (
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
    .global_config_valid_i(global_config_valid),
    .global_config_i(global_config),
    .stream_valid_i(stream_valid),
    .stream_data_i(stream_data),
    .stream_consumed_o(stream_consumed),
    .immediate_valid_i(immediate_valid),
    .immediate_data_i(immediate_data),
    .lut_config_valid_i(lut_config_valid),
    .lut_config_bank_i(lut_config_bank),
    .lut_config_input_min_i(lut_config_input_min),
    .lut_config_segment_width_i(lut_config_segment_width),
    .lut_write_valid_i(lut_write_valid),
    .lut_write_bank_i(lut_write_bank),
    .lut_write_address_i(lut_write_address),
    .lut_write_k_i(lut_write_k),
    .lut_write_b_i(lut_write_b),
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
    .output_ready_i(output_ready),
    .output_valid_o(output_valid),
    .output_data_o(output_data),
    .tail_valid_o(tail_valid),
    .tail_value_o(tail_value),
    .tail_original_o(),
    .tail_auxiliary_o(),
    .accumulator_state_valid_o(accumulator_state_valid),
    .accumulator_state_data_o(accumulator_state_data),
    .feedback_state_valid_o(feedback_state_valid),
    .feedback_state_value_o(feedback_state_value),
    .instruction_pending_o(instruction_pending),
    .idle_o(idle),
    .fault_o(fault)
  );

  task automatic reset_tile(input logic [1:0] chain_length);
    begin
      @(negedge clk);
      rst_n = 1'b0;
      local_config_load = 1'b0;
      local_config_active = '1;
      local_config_instruction = '0;
      execute_valid = 1'b0;
      repeat_fields = '0;
      repeat_fields.first_iteration = 1'b1;
      repeat_fields.last_iteration = 1'b1;
      repeat_control = repeat_fields;
      immediate_valid = '0;
      immediate_data = '0;
      stream_valid = '1;
      stream_data = '0;
      output_ready = '0;
      lut_config_valid = 1'b0;
      lut_config_bank = '0;
      lut_config_input_min = '0;
      lut_config_segment_width = '0;
      lut_write_valid = 1'b0;
      lut_write_bank = '0;
      lut_write_address = '0;
      lut_write_k = '0;
      lut_write_b = '0;
      config_fields = '0;
      config_fields.chain_length = chain_length;
      config_fields.compute_dtype = VXM_FORMAT_FP16;
      config_fields.lhs_dtype = VXM_FORMAT_FP16;
      config_fields.rhs_dtype = VXM_FORMAT_FP16;
      global_config = config_fields;
      global_config_valid = 1'b1;
      repeat (3) @(posedge clk);
      @(negedge clk);
      rst_n = 1'b1;
    end
  endtask

  task automatic set_stream_block(
    input integer block,
    input logic [15:0] lhs,
    input logic [15:0] rhs
  );
    begin
      for (integer lane = 0; lane < LANES; lane++) begin
        stream_data[((block*4)*LANES+lane)*8 +: 8] = lhs[7:0];
        stream_data[((block*4+1)*LANES+lane)*8 +: 8] = lhs[15:8];
        stream_data[((block*4+2)*LANES+lane)*8 +: 8] = rhs[7:0];
        stream_data[((block*4+3)*LANES+lane)*8 +: 8] = rhs[15:8];
      end
    end
  endtask

  task automatic set_stream_block_fp32_half(
    input integer block,
    input logic [31:0] lhs,
    input logic [31:0] rhs,
    input logic high_half
  );
    logic [15:0] lhs_half;
    logic [15:0] rhs_half;
    begin
      lhs_half = high_half ? lhs[31:16] : lhs[15:0];
      rhs_half = high_half ? rhs[31:16] : rhs[15:0];
      for (integer lane = 0; lane < LANES; lane++) begin
        stream_data[((block*4)*LANES+lane)*8 +: 8] = lhs_half[7:0];
        stream_data[((block*4+1)*LANES+lane)*8 +: 8] = lhs_half[15:8];
        stream_data[((block*4+2)*LANES+lane)*8 +: 8] = rhs_half[7:0];
        stream_data[((block*4+3)*LANES+lane)*8 +: 8] = rhs_half[15:8];
      end
    end
  endtask

  task automatic drive_fp32_special_input_half(input logic high_half);
    begin
      stream_data = '0;
      for (integer block = 0; block < 8; block++) begin
        case (block % 4)
          0, 2: set_stream_block_fp32_half(
            block, 32'h00000000, 32'h00000000, high_half);
          1: set_stream_block_fp32_half(
            block, 32'h40000000, 32'h00000000, high_half);
          default: set_stream_block_fp32_half(
            block, 32'h40800000, 32'h00000000, high_half);
        endcase
      end
      stream_valid = '1;
      #1;
      if (stream_consumed !== {STREAMS_PER_DIRECTION{1'b1}})
        $fatal(1, "FP32 Special Tile did not consume the complete phase");
    end
  endtask

  task automatic set_instruction(
    input integer queue,
    input logic [VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0] instruction
  );
    begin
      local_config_instruction[
        queue*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
          VXM_LOCAL_MAX_INSTRUCTION_WIDTH] = instruction;
    end
  endtask

  task automatic set_repeat_control(
    input logic output_enable,
    input logic accumulator_enable,
    input logic first_iteration,
    input logic last_iteration
  );
    begin
      repeat_fields = '0;
      repeat_fields.output_enable = output_enable;
      repeat_fields.accumulator_enable = accumulator_enable;
      repeat_fields.first_iteration = first_iteration;
      repeat_fields.last_iteration = last_iteration;
      repeat_control = repeat_fields;
    end
  endtask

  task automatic load_config;
    begin
      wait (idle);
      @(negedge clk);
      local_config_load = 1'b1;
      @(posedge clk);
      @(negedge clk);
      local_config_load = 1'b0;
    end
  endtask

  task automatic execute_once;
    begin
      wait (execute_ready);
      @(negedge clk);
      execute_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      execute_valid = 1'b0;
    end
  endtask

  task automatic issue_all;
    begin
      load_config();
      execute_once();
    end
  endtask

  task automatic release_outputs;
    begin
      @(negedge clk);
      output_ready = '1;
      @(posedge clk);
      @(negedge clk);
      output_ready = '0;
      wait (!(|output_valid));
    end
  endtask

  task automatic check_output_word(
    input logic [STREAMS_PER_DIRECTION-1:0] expected_mask,
    input logic [15:0] expected_word
  );
    logic [15:0] observed;
    begin
      wait ((output_valid & expected_mask) == expected_mask);
      #1;
      if (|(output_valid & ~expected_mask))
        $fatal(1, "unexpected VXM output stream valid mask %h expected %h",
               output_valid, expected_mask);
      for (integer block = 0; block < 8; block++) begin
        if (expected_mask[block*2]) begin
          for (integer lane = 0; lane < LANES; lane++) begin
            observed[7:0] = output_data[
              ((block*2)*LANES+lane)*8 +: 8];
            observed[15:8] = output_data[
              ((block*2+1)*LANES+lane)*8 +: 8];
            if (observed !== expected_word)
              $fatal(1,
                "block %0d lane %0d output %h expected %h",
                block, lane, observed, expected_word);
          end
        end
      end
      // Backpressure must retain both valid and payload, not merely pulse it.
      repeat (3) begin
        @(posedge clk);
        #1;
        if ((output_valid & expected_mask) != expected_mask)
          $fatal(1, "VXM output did not remain valid under backpressure");
        for (integer block = 0; block < 8; block++) begin
          if (expected_mask[block*2]) begin
            for (integer lane = 0; lane < LANES; lane++) begin
              observed[7:0] = output_data[
                ((block*2)*LANES+lane)*8 +: 8];
              observed[15:8] = output_data[
                ((block*2+1)*LANES+lane)*8 +: 8];
              if (observed !== expected_word)
                $fatal(1, "VXM output changed while stalled");
            end
          end
        end
      end
    end
  endtask

  task automatic check_fp32_special_outputs(
    input logic [STREAMS_PER_DIRECTION-1:0] expected_mask
  );
    logic [15:0] observed;
    logic [15:0] expected_high;
    begin
      wait ((output_valid & expected_mask) == expected_mask);
      #1;
      if (|(output_valid & ~expected_mask))
        $fatal(1, "unexpected FP32 Special output mask %h", output_valid);
      // All selected reference results have a zero low half. Hold the first
      // phase under backpressure to verify phase and payload retention.
      repeat (2) begin
        for (integer block = 0; block < 8; block++) begin
          for (integer lane = 0; lane < LANES; lane++) begin
            observed[7:0] = output_data[((block*2)*LANES+lane)*8 +: 8];
            observed[15:8] =
              output_data[((block*2+1)*LANES+lane)*8 +: 8];
            if (observed !== 16'h0000)
              $fatal(1, "FP32 Special low phase mismatch");
          end
        end
        @(posedge clk);
        #1;
      end

      @(negedge clk);
      output_ready = expected_mask;
      @(posedge clk);
      #1;
      @(negedge clk);
      output_ready = '0;
      #1;
      for (integer block = 0; block < 8; block++) begin
        expected_high = ((block % 4) == 0 || (block % 4) == 2) ?
          16'h3f80 : 16'h3f00;
        for (integer lane = 0; lane < LANES; lane++) begin
          observed[7:0] = output_data[((block*2)*LANES+lane)*8 +: 8];
          observed[15:8] =
            output_data[((block*2+1)*LANES+lane)*8 +: 8];
          if (!output_valid[block*2] || !output_valid[block*2+1] ||
              observed !== expected_high)
            $fatal(1,
              "FP32 Special block %0d lane %0d high phase %h expected %h",
              block, lane, observed, expected_high);
        end
      end
    end
  endtask

  task automatic check_bf16_special_outputs(
    input logic [STREAMS_PER_DIRECTION-1:0] expected_mask
  );
    logic [15:0] observed;
    logic [15:0] expected_word;
    begin
      wait ((output_valid & expected_mask) == expected_mask);
      #1;
      if (|(output_valid & ~expected_mask))
        $fatal(1, "unexpected BF16 Special output mask %h", output_valid);
      // Hold the single BF16 output beat under backpressure. This checks the
      // public Tile output contract without driving internal state.
      repeat (2) begin
        for (integer block = 0; block < 8; block++) begin
          expected_word = ((block % 4) == 0 || (block % 4) == 2) ?
            16'h3f80 : 16'h3f00;
          for (integer lane = 0; lane < LANES; lane++) begin
            observed[7:0] =
              output_data[((block*2)*LANES+lane)*8 +: 8];
            observed[15:8] =
              output_data[((block*2+1)*LANES+lane)*8 +: 8];
            if (observed !== expected_word)
              $fatal(1,
                "BF16 Special block %0d lane %0d got %h expected %h",
                block, lane, observed, expected_word);
          end
        end
        @(posedge clk);
        #1;
      end
    end
  endtask

  task automatic program_lut(
    input logic [LUT_BANK_WIDTH-1:0] bank,
    input logic [15:0] input_min,
    input logic [15:0] segment_width,
    input logic [15:0] k,
    input logic [15:0] b
  );
    begin
      @(negedge clk);
      lut_config_valid = 1'b1;
      lut_config_bank = bank;
      lut_config_input_min = input_min;
      lut_config_segment_width = segment_width;
      lut_write_valid = 1'b1;
      lut_write_bank = bank;
      lut_write_address = '0;
      lut_write_k = k;
      lut_write_b = b;
      @(posedge clk);
      @(negedge clk);
      lut_config_valid = 1'b0;
      lut_write_valid = 1'b0;
    end
  endtask

  always @(posedge clk) begin
    #1;
    if (rst_n && fault)
      $fatal(1, "VXM tile special-path test observed fault");
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      config_done_count <= 0;
    else if (config_done)
      config_done_count <= config_done_count + 1;
  end

  initial begin
    logic [STAGES*LANES-1:0] state_mask;
    logic [STREAMS_PER_DIRECTION-1:0] output_mask;
    logic [15:0] observed;
    logic [15:0] expected;

    clk = 1'b0;
    rst_n = 1'b0;
    global_config_valid = 1'b0;
    global_config = '0;
    local_config_load = 1'b0;
    local_config_active = '1;
    local_config_instruction = '0;
    execute_valid = 1'b0;
    repeat_fields = '0;
    repeat_control = '0;
    stream_valid = '0;
    stream_data = '0;
    immediate_valid = '0;
    immediate_data = '0;
    output_ready = '0;
    lut_config_valid = 1'b0;
    lut_config_bank = '0;
    lut_config_input_min = '0;
    lut_config_segment_width = '0;
    lut_write_valid = 1'b0;
    lut_write_bank = '0;
    lut_write_address = '0;
    lut_write_k = '0;
    lut_write_b = '0;

    // Chain-length-4 feedback: each C3 tail returns to its own C0 head.
    reset_tile(VXM_CHAIN_LENGTH_4);
    for (integer block = 0; block < 8; block++)
      set_stream_block(block, 16'h3c00, 16'h4000); // 1.0, 2.0
    local_config_instruction = '0;
    set_repeat_control(1'b0, 1'b0, 1'b1, 1'b1);
    set_instruction(3, 7'h01); // Add Previous + Original.
    set_instruction(7, 7'h01);
    issue_all();
    state_mask = '0;
    state_mask[3*LANES +: LANES] = '1;
    state_mask[7*LANES +: LANES] = '1;
    state_mask[11*LANES +: LANES] = '1;
    state_mask[15*LANES +: LANES] = '1;
    wait ((feedback_state_valid & state_mask) == state_mask);
    #1;
    for (integer stage = 3; stage < STAGES; stage += 4)
      for (integer lane = 0; lane < LANES; lane++)
        if (feedback_state_value[
              (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH] !==
            32'h00004000)
          $fatal(1, "stage %0d lane %0d feedback capture mismatch",
                 stage, lane);

    local_config_instruction = '0;
    immediate_valid = '0;
    immediate_data = '0;
    immediate_valid[0] = 1'b1;
    immediate_valid[4] = 1'b1;
    immediate_data[0*CONTAINER_WIDTH +: CONTAINER_WIDTH] = 32'h00003c00;
    immediate_data[4*CONTAINER_WIDTH +: CONTAINER_WIDTH] = 32'h00003c00;
    set_repeat_control(1'b1, 1'b0, 1'b1, 1'b1);
    set_instruction(0, 7'h31); // Add Feedback + Immediate.
    set_instruction(4, 7'h31);
    set_instruction(1, 7'h09); // Add retained Auxiliary.
    set_instruction(5, 7'h09);
    set_instruction(3, 7'h01); // Add retained Original, output.
    set_instruction(7, 7'h01);
    issue_all();
    output_mask = 32'h0000cccc;
    // Feedback value 2 + immediate 1 + auxiliary 2 + original 1 = 6.
    check_output_word(output_mask, 16'h4600);
    if (|(feedback_state_valid & state_mask))
      $fatal(1, "consumed chain-length-4 feedback remained valid");
    release_outputs();
    wait (idle);

    // Chain-length-8 feedback exercises the longest fixed C7-to-C0 path in
    // each independent physical eight-stage half.
    reset_tile(VXM_CHAIN_LENGTH_8);
    for (integer block = 0; block < 8; block++)
      set_stream_block(block, 16'h3c00, 16'h4000);
    local_config_instruction = '0;
    set_repeat_control(1'b0, 1'b0, 1'b1, 1'b1);
    set_instruction(7, 7'h01); // Tail Add Previous + Original.
    issue_all();
    state_mask = '0;
    state_mask[7*LANES +: LANES] = '1;
    state_mask[15*LANES +: LANES] = '1;
    wait ((feedback_state_valid & state_mask) == state_mask);
    local_config_instruction = '0;
    immediate_valid = '0;
    immediate_data = '0;
    immediate_valid[0] = 1'b1;
    immediate_data[0 +: CONTAINER_WIDTH] = 32'h00003c00;
    set_repeat_control(1'b1, 1'b0, 1'b1, 1'b1);
    set_instruction(0, 7'h31); // Add Feedback + Immediate.
    set_instruction(1, 7'h09); // Add retained Auxiliary.
    set_instruction(7, 7'h01); // Add retained Original, output.
    issue_all();
    output_mask = 32'h0000c0c0;
    check_output_word(output_mask, 16'h4600);
    if (|(feedback_state_valid & state_mask))
      $fatal(1, "consumed chain-length-8 feedback remained valid");
    release_outputs();
    wait (idle);

    // Chain-length-2 C1/C3 accumulator recurrence and fixed output packing.
    reset_tile(VXM_CHAIN_LENGTH_2);
    for (integer block = 0; block < 8; block++)
      set_stream_block(block, 16'h3c00, 16'h0000); // 1.0.
    local_config_instruction = '0;
    set_repeat_control(1'b1, 1'b1, 1'b1, 1'b0);
    for (integer queue = 1; queue < 8; queue += 2)
      set_instruction(queue, 7'h19); // First repeat: reset/write, no emit.
    issue_all();
    state_mask = '0;
    for (integer stage = 1; stage < STAGES; stage += 2)
      state_mask[stage*LANES +: LANES] = '1;
    wait ((accumulator_state_valid & state_mask) == state_mask);
    #1;
    for (integer stage = 1; stage < STAGES; stage += 2)
      for (integer lane = 0; lane < LANES; lane++)
        if (accumulator_state_data[
              (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH] !==
            32'h00003c00)
          $fatal(1, "stage %0d lane %0d accumulator reset mismatch",
                 stage, lane);
    if (|feedback_state_valid)
      $fatal(1, "non-emitting accumulator write leaked into feedback");
    if (config_done_count != 0)
      $fatal(1, "non-final repeat returned an end marker");

    set_repeat_control(1'b1, 1'b1, 1'b0, 1'b1);
    for (integer queue = 1; queue < 8; queue += 2)
      set_instruction(queue, 7'h19); // Last repeat: write/emit/output.
    execute_once();
    output_mask = 32'h0000ffff;
    check_output_word(output_mask, 16'h4000); // 1.0 + saved 1.0.
    wait (config_done_count == 1);
    for (integer stage = 1; stage < STAGES; stage += 2)
      for (integer lane = 0; lane < LANES; lane++)
        if (accumulator_state_data[
              (stage*LANES+lane)*CONTAINER_WIDTH +: CONTAINER_WIDTH] !==
            32'h00004000)
          $fatal(1, "stage %0d lane %0d accumulator update mismatch",
                 stage, lane);
    release_outputs();
    wait (idle);

    // Concurrent tile-level LUT traffic: Q1/Q5 Exp, Q3 Reciprocal and
    // Q7 Rsqrt, all across eight lockstep lanes and both mirrored stages.
    reset_tile(VXM_CHAIN_LENGTH_2);
    program_lut(2'd0, 16'h0000, 16'h3c00, 16'h1000, 16'h0000);
    program_lut(2'd1, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);
    program_lut(2'd2, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);
    set_stream_block(0, 16'h0000, 16'h0000); // Exp(0) = 1.
    set_stream_block(1, 16'h4000, 16'h0000); // Reciprocal(2) = 0.5.
    set_stream_block(2, 16'h0000, 16'h0000); // Exp(0) = 1.
    set_stream_block(3, 16'h4400, 16'h0000); // Rsqrt(4) = 0.5.
    set_stream_block(4, 16'h0000, 16'h0000);
    set_stream_block(5, 16'h4000, 16'h0000);
    set_stream_block(6, 16'h0000, 16'h0000);
    set_stream_block(7, 16'h4400, 16'h0000);
    local_config_instruction = '0;
    set_repeat_control(1'b1, 1'b0, 1'b1, 1'b1);
    set_instruction(1, 7'h06);
    set_instruction(3, 7'h06);
    set_instruction(5, 7'h06);
    set_instruction(7, 7'h07);
    issue_all();
    output_mask = 32'h0000ffff;
    wait ((output_valid & output_mask) == output_mask);
    #1;
    for (integer block = 0; block < 8; block++) begin
      expected = ((block % 4) == 0 || (block % 4) == 2) ?
        16'h3c00 : 16'h3800;
      for (integer lane = 0; lane < LANES; lane++) begin
        observed[7:0] = output_data[((block*2)*LANES+lane)*8 +: 8];
        observed[15:8] = output_data[((block*2+1)*LANES+lane)*8 +: 8];
        if (observed !== expected)
          $fatal(1,
            "special block %0d lane %0d output %h expected %h",
            block, lane, observed, expected);
      end
    end
    release_outputs();
    wait (idle);

    // Native FP32 Special path. Data enters only through the Tile stream
    // ports in low/high phases; the existing FP16 LUT coefficients are
    // widened internally before FP32 interpolation.
    reset_tile(VXM_CHAIN_LENGTH_2);
    config_fields.compute_dtype = VXM_FORMAT_FP32;
    config_fields.lhs_read_bits = VXM_READ_BITS_32;
    config_fields.lhs_dtype = VXM_FORMAT_FP32;
    config_fields.rhs_read_bits = VXM_READ_BITS_32;
    config_fields.rhs_dtype = VXM_FORMAT_FP32;
    global_config = config_fields;
    stream_valid = '0;
    stream_data = '0;
    program_lut(2'd0, 16'h0000, 16'h3c00, 16'h1000, 16'h0000);
    program_lut(2'd1, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);
    program_lut(2'd2, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);
    local_config_instruction = '0;
    set_repeat_control(1'b1, 1'b0, 1'b1, 1'b1);
    set_instruction(1, 7'h06); // Exp(0).
    set_instruction(3, 7'h06); // Reciprocal(2).
    set_instruction(5, 7'h06); // Exp(0).
    set_instruction(7, 7'h07); // Rsqrt(4).
    load_config();
    execute_once();
    drive_fp32_special_input_half(1'b0);
    @(posedge clk);
    @(negedge clk);
    drive_fp32_special_input_half(1'b1);
    @(posedge clk);
    @(negedge clk);
    stream_valid = '0;
    stream_data = '0;
    output_mask = 32'h0000ffff;
    check_fp32_special_outputs(output_mask);
    release_outputs();
    wait (idle);

    // BF16 is a single 16-bit stream beat. Special functions widen both the
    // operand and compact FP16 LUT coefficients to FP32, then round the ALU
    // result back to BF16 before it reaches the Tile output ports.
    reset_tile(VXM_CHAIN_LENGTH_2);
    config_fields.compute_dtype = VXM_FORMAT_BF16;
    config_fields.lhs_dtype = VXM_FORMAT_BF16;
    config_fields.rhs_dtype = VXM_FORMAT_BF16;
    global_config = config_fields;
    program_lut(2'd0, 16'h0000, 16'h3c00, 16'h1000, 16'h0000);
    program_lut(2'd1, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);
    program_lut(2'd2, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);
    set_stream_block(0, 16'h0000, 16'h0000); // Exp(0) = 1.
    set_stream_block(1, 16'h4000, 16'h0000); // Reciprocal(2) = 0.5.
    set_stream_block(2, 16'h0000, 16'h0000);
    set_stream_block(3, 16'h4080, 16'h0000); // Rsqrt(4) = 0.5.
    set_stream_block(4, 16'h0000, 16'h0000);
    set_stream_block(5, 16'h4000, 16'h0000);
    set_stream_block(6, 16'h0000, 16'h0000);
    set_stream_block(7, 16'h4080, 16'h0000);
    local_config_instruction = '0;
    set_repeat_control(1'b1, 1'b0, 1'b1, 1'b1);
    set_instruction(1, 7'h06);
    set_instruction(3, 7'h06);
    set_instruction(5, 7'h06);
    set_instruction(7, 7'h07);
    issue_all();
    output_mask = 32'h0000ffff;
    check_bf16_special_outputs(output_mask);
    release_outputs();
    wait (idle);

    if (|instruction_pending)
      $fatal(1, "tile special-path test ended with pending instructions");
    $display("LPU_VXM_TILE_SPECIAL_PATHS_TB_PASS");
    $finish;
  end

  initial begin
    #100000;
    $fatal(1, "VXM tile special-path timeout pending=%h output=%h",
           instruction_pending, output_valid);
  end
endmodule
