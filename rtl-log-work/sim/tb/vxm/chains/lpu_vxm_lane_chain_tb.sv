`timescale 1ns/1ps

// Black-box regression for one Lane: 16 physical ALUs and one data item per
// stage. The nearest-neighbor wiring is fixed; only global chain_length is
// reloaded between runs.
module lpu_vxm_lane_chain_tb;
  import lpu_pkg::*;

  localparam integer STAGES = 16;
  localparam integer LANES = 1;
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
  logic global_config_valid;
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
  wire [STAGES*LANES-1:0] feedback_valid;
  wire [STAGES*LANES*CONTAINER_WIDTH-1:0] feedback_value;
  wire [STAGES-1:0] instruction_pending;
  wire idle;
  wire fault;

  logic clear_observation;
  logic [STAGES*LANES-1:0] tail_seen;
  logic [STAGES*LANES-1:0] expected_tail_mask;
  logic [STAGES*LANES*CONTAINER_WIDTH-1:0] captured_value,captured_original,captured_auxiliary;
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
    .global_config_valid_i(global_config_valid),
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
    .feedback_state_valid_o(feedback_valid),
    .feedback_state_value_o(feedback_value),
    .instruction_pending_o(instruction_pending),
    .idle_o(idle),
    .fault_o(fault)
  );

  function automatic logic [STAGES*LANES-1:0] tail_mask_for(
    input logic [1:0] chain_length
  );
    logic [STAGES*LANES-1:0] mask;
    integer step;
    integer tail_stage;
    begin
      mask = '0;
      case (chain_length)
        VXM_CHAIN_LENGTH_2: step = 2;
        VXM_CHAIN_LENGTH_4: step = 4;
        VXM_CHAIN_LENGTH_8: step = 8;
        default: step = STAGES + 1;
      endcase
      tail_stage = step - 1;
      while (tail_stage < STAGES) begin
        mask[tail_stage*LANES +: LANES] = '1;
        tail_stage = tail_stage + step;
      end
      tail_mask_for = mask;
    end
  endfunction

  task automatic set_repeat(
    input logic output_enable,
    input logic first_iteration,
    input logic last_iteration
  );
    begin
      repeat_fields = '0;
      repeat_fields.output_enable = output_enable;
      repeat_fields.first_iteration = first_iteration;
      repeat_fields.last_iteration = last_iteration;
      repeat_control = repeat_fields;
    end
  endtask

  task automatic set_stream_operands(
    input logic [15:0] lhs,
    input logic [15:0] rhs
  );
    begin
      stream_data = '0;
      // Each logical operand is carried by two adjacent 8-bit streams.
      // Every four-stream block contains LHS low/high then RHS low/high.
      for (integer block = 0; block < 8; block++) begin
        for (integer lane = 0; lane < LANES; lane++) begin
          stream_data[((block*4+0)*LANES+lane)*8 +: 8] = lhs[7:0];
          stream_data[((block*4+1)*LANES+lane)*8 +: 8] = lhs[15:8];
          stream_data[((block*4+2)*LANES+lane)*8 +: 8] = rhs[7:0];
          stream_data[((block*4+3)*LANES+lane)*8 +: 8] = rhs[15:8];
        end
      end
    end
  endtask

  task automatic set_immediates(input logic [15:0] value);
    begin
      immediate_valid = '1;
      immediate_data = '0;
      for (integer queue = 0; queue < VXM_LOCAL_QUEUE_COUNT; queue++)
        immediate_data[queue*CONTAINER_WIDTH +: CONTAINER_WIDTH] =
          {{(CONTAINER_WIDTH-16){1'b0}}, value};
    end
  endtask

  task automatic clear_seen;
    begin
      @(negedge clk);
      clear_observation = 1'b1;
      @(posedge clk);
      #1;
      @(negedge clk);
      clear_observation = 1'b0;
    end
  endtask

  task automatic load_config(input logic [1:0] chain_length);
    begin
      wait (idle && local_config_ready);
      config_fields.chain_length = chain_length;
      global_config = config_fields;
      @(negedge clk);
      local_config_load = 1'b1;
      @(posedge clk);
      #1;
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
      #1;
      @(negedge clk);
      execute_valid = 1'b0;
    end
  endtask

  task automatic check_tail_payload(
    input logic [CONTAINER_WIDTH-1:0] expected_value,
    input logic [CONTAINER_WIDTH-1:0] expected_original,
    input logic [CONTAINER_WIDTH-1:0] expected_auxiliary
  );
    begin
      for (integer stage = 0; stage < STAGES; stage++) begin
        for (integer lane = 0; lane < LANES; lane++) begin
          if (expected_tail_mask[stage*LANES+lane] &&
              (captured_value[(stage*LANES+lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH] !== expected_value))
            $fatal(1, "chain tail value mismatch stage=%0d lane=%0d",
                   stage, lane);
          if (expected_tail_mask[stage*LANES+lane] &&
              (captured_original[(stage*LANES+lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH] !== expected_original))
            $fatal(1, "chain tail original mismatch stage=%0d lane=%0d",
                   stage, lane);
          if (expected_tail_mask[stage*LANES+lane] &&
              (captured_auxiliary[(stage*LANES+lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH] !== expected_auxiliary))
            $fatal(1, "chain tail auxiliary mismatch stage=%0d lane=%0d",
                   stage, lane);
        end
      end
    end
  endtask

  task automatic run_output_chain(input logic [1:0] chain_length);
    begin
      local_config_instruction = '0; // BYPASS at every queue.
      set_stream_operands(16'h3c00, 16'h3c00);
      set_repeat(1'b1, 1'b1, 1'b1);
      expected_tail_mask = tail_mask_for(chain_length);
      clear_seen();
      load_config(chain_length);
      execute_once();
      wait ((tail_seen & expected_tail_mask) == expected_tail_mask);
      wait (done_seen && idle);
      if (tail_seen != expected_tail_mask)
        $fatal(1, "chain length %0d produced tail mask %h expected %h",
               chain_length, tail_seen, expected_tail_mask);
      if (|instruction_pending)
        $fatal(1, "chain length %0d left pending instructions",
               chain_length);
      check_tail_payload(32'h00003c00, 32'h00003c00, 32'h00003c00);
    end
  endtask

  task automatic run_arithmetic_chain4;
    logic [VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0] instruction;
    begin
      // Each four-stage chain must calculate:
      //   head:     stream 1.0 + stream 2.0 = 3.0
      //   internal: previous 3.0 + original 1.0 = 4.0
      //   internal: previous 4.0 + auxiliary 2.0 = 6.0
      //   tail:     previous 6.0 + immediate 1.0 = 7.0
      // Q0..Q3 and Q4..Q7 configure the four mirrored physical chains.
      local_config_instruction = '0;
      for (integer queue = 0; queue < VXM_LOCAL_QUEUE_COUNT; queue++) begin
        instruction = '0;
        instruction[2:0] = VXM_LOCAL_ADD;
        case (queue % 4)
          0: begin
            // At a length-4 chain head, zero source codes select both streams.
          end
          1: instruction[4:3] = 2'd0; // Original head operand.
          2: instruction[6:5] = 2'd1; // Auxiliary head operand.
          3: instruction[4:3] = 2'd2; // Queue-local immediate.
        endcase
        local_config_instruction[
          queue*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
            VXM_LOCAL_MAX_INSTRUCTION_WIDTH] = instruction;
      end
      set_stream_operands(16'h3c00, 16'h4000);
      set_immediates(16'h3c00);
      set_repeat(1'b1, 1'b1, 1'b1);
      expected_tail_mask = tail_mask_for(VXM_CHAIN_LENGTH_4);
      clear_seen();
      load_config(VXM_CHAIN_LENGTH_4);
      execute_once();
      wait ((tail_seen & expected_tail_mask) == expected_tail_mask);
      wait (done_seen && idle);
      if (tail_seen != expected_tail_mask)
        $fatal(1, "arithmetic chain4 produced wrong tail mask");
      if (|instruction_pending)
        $fatal(1, "arithmetic chain4 left pending instructions");
      check_tail_payload(32'h00004700, 32'h00003c00, 32'h00004000);
    end
  endtask

  task automatic run_feedback_chain4;
    logic [6:0] q0_feedback_bypass;
    logic [6:0] q4_feedback_bypass;
    begin
      expected_tail_mask = tail_mask_for(VXM_CHAIN_LENGTH_4);
      local_config_instruction = '0;
      set_stream_operands(16'h3c00, 16'h3c00);
      set_repeat(1'b0, 1'b1, 1'b0);
      clear_seen();
      load_config(VXM_CHAIN_LENGTH_4);
      execute_once();
      wait ((tail_seen & expected_tail_mask) == expected_tail_mask);
      wait (idle);
      if (feedback_valid != expected_tail_mask)
        $fatal(1, "chain4 feedback mask %h expected %h",
               feedback_valid, expected_tail_mask);
      for (integer bit_index = 0; bit_index < STAGES*LANES; bit_index++)
        if (expected_tail_mask[bit_index] &&
            feedback_value[bit_index*CONTAINER_WIDTH +:
              CONTAINER_WIDTH] !== 32'h00003c00)
          $fatal(1, "chain4 feedback value mismatch bit=%0d", bit_index);

      // Q0 and Q4 are the logical heads for the four mirrored length-4
      // chains. Select Feedback as LHS; internal BYPASS stages propagate it.
      q0_feedback_bypass = '0;
      q0_feedback_bypass[4:3] = 2'd2;
      q4_feedback_bypass = '0;
      q4_feedback_bypass[4:3] = 2'd2;
      local_config_instruction[
        0*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
          VXM_LOCAL_MAX_INSTRUCTION_WIDTH] = q0_feedback_bypass;
      local_config_instruction[
        4*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
          VXM_LOCAL_MAX_INSTRUCTION_WIDTH] = q4_feedback_bypass;
      set_repeat(1'b1, 1'b0, 1'b1);
      clear_seen();
      load_config(VXM_CHAIN_LENGTH_4);
      execute_once();
      wait ((tail_seen & expected_tail_mask) == expected_tail_mask);
      wait (done_seen && idle);
      if (tail_seen != expected_tail_mask)
        $fatal(1, "feedback execution produced wrong tail mask");
      check_tail_payload(32'h00003c00, 32'h00003c00, 32'h00003c00);
      if (|feedback_valid)
        $fatal(1, "feedback token was not consumed by the chain head");
    end
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tail_seen <= '0;
      done_seen <= 1'b0;
    end else if (clear_observation) begin
      tail_seen <= '0;
      done_seen <= 1'b0;
    end else begin
      tail_seen <= tail_seen | tail_valid;
      // Observe payload only when the public valid signal is asserted.
      for(integer i=0;i<STAGES*LANES;i++) if(tail_valid[i]) begin
        captured_value[i*CONTAINER_WIDTH+:CONTAINER_WIDTH]<=tail_value[i*CONTAINER_WIDTH+:CONTAINER_WIDTH];
        captured_original[i*CONTAINER_WIDTH+:CONTAINER_WIDTH]<=tail_original[i*CONTAINER_WIDTH+:CONTAINER_WIDTH];
        captured_auxiliary[i*CONTAINER_WIDTH+:CONTAINER_WIDTH]<=tail_auxiliary[i*CONTAINER_WIDTH+:CONTAINER_WIDTH];
      end
      done_seen <= done_seen | config_done;
      if (fault)
        $fatal(1, "single-Lane chain DUT reported a fault");
      if (|(tail_valid & ~expected_tail_mask))
        $fatal(1, "non-tail stage asserted tail_valid");
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
    repeat_control = '0;
    global_config_valid = 1'b1;
    config_fields = '0;
    config_fields.compute_dtype = VXM_FORMAT_FP16;
    config_fields.lhs_dtype = VXM_FORMAT_FP16;
    config_fields.rhs_dtype = VXM_FORMAT_FP16;
    global_config = config_fields;
    stream_valid = '1;
    stream_data = '0;
    immediate_valid = '0;
    immediate_data = '0;
    clear_observation = 1'b0;
    expected_tail_mask = '0;

    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    run_output_chain(VXM_CHAIN_LENGTH_2);
    run_output_chain(VXM_CHAIN_LENGTH_4);
    run_output_chain(VXM_CHAIN_LENGTH_8);
    run_arithmetic_chain4();
    run_feedback_chain4();

    $display("LPU_VXM_LANE_CHAIN_TB_PASS");
    $finish;
  end

  initial begin
    #100000;
    $fatal(1, "single-Lane VXM chain regression timed out seen=%h expected=%h",
           tail_seen, expected_tail_mask);
  end
endmodule
