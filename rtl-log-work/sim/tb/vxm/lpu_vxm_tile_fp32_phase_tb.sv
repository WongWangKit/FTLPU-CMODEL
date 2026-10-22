`timescale 1ns/1ps

module lpu_vxm_tile_fp32_phase_tb;
  import lpu_pkg::*;

  localparam integer LANES = LANES_PER_TILE;
  localparam integer STREAMS = STREAMS_PER_DIRECTION;
  localparam logic [STREAMS-1:0] INPUT_MASK = 32'h0f0f0f0f;
  localparam logic [7:0] TAIL_BLOCK_MASK = 8'haa;

  logic clk;
  logic rst_n;
  logic config_load;
  wire config_ready;
  logic [7:0] config_active;
  logic [8*VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0] config_instruction;
  logic execute_valid;
  wire execute_ready;
  logic [VXM_REPEAT_CONTROL_WIDTH-1:0] repeat_control;
  wire config_done;
  logic global_config_valid;
  logic [VXM_GLOBAL_CONFIG_WIDTH-1:0] global_config;
  vxm_global_config_t global_fields;
  logic [STREAMS-1:0] stream_valid;
  logic [STREAMS*LANES*8-1:0] stream_data;
  wire [STREAMS-1:0] stream_consumed;
  wire [STREAMS-1:0] output_valid;
  wire [STREAMS*LANES*8-1:0] output_data;
  wire idle;
  wire fault;

  logic [1:0] input_phase_seen;
  logic [7:0] output_low_seen;
  logic [7:0] output_high_seen;
  logic done_seen;

  always #5 clk = ~clk;

  lpu_vxm_tile_execution dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .local_config_load_i(config_load),
    .local_config_ready_o(config_ready),
    .local_config_active_i(config_active),
    .local_config_instruction_i(config_instruction),
    .execute_valid_i(execute_valid),
    .execute_ready_o(execute_ready),
    .repeat_control_i(repeat_control),
    .config_done_o(config_done),
    .global_config_valid_i(global_config_valid),
    .global_config_i(global_config),
    .stream_valid_i(stream_valid),
    .stream_data_i(stream_data),
    .stream_consumed_o(stream_consumed),
    .immediate_valid_i('0),
    .immediate_data_i('0),
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
    .output_valid_o(output_valid),
    .output_data_o(output_data),
    .tail_valid_o(),
    .tail_value_o(),
    .tail_original_o(),
    .tail_auxiliary_o(),
    .accumulator_state_valid_o(),
    .accumulator_state_data_o(),
    .feedback_state_valid_o(),
    .feedback_state_value_o(),
    .instruction_pending_o(),
    .idle_o(idle),
    .fault_o(fault)
  );

  task automatic drive_input_half(input logic high_half);
    begin
      stream_data = '0;
      for (integer block = 0; block < 8; block += 2) begin
        for (integer lane = 0; lane < LANES; lane++) begin
          // LHS=FP32 1.0 (3f80_0000), RHS=FP32 2.0 (4000_0000).
          stream_data[((block*4)*LANES+lane)*8 +: 8] =
            high_half ? 8'h80 : 8'h00;
          stream_data[((block*4+1)*LANES+lane)*8 +: 8] =
            high_half ? 8'h3f : 8'h00;
          stream_data[((block*4+2)*LANES+lane)*8 +: 8] = 8'h00;
          stream_data[((block*4+3)*LANES+lane)*8 +: 8] =
            high_half ? 8'h40 : 8'h00;
        end
      end
      stream_valid = INPUT_MASK;
    end
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      input_phase_seen <= '0;
      output_low_seen <= '0;
      output_high_seen <= '0;
      done_seen <= 1'b0;
    end else begin
      done_seen <= done_seen | config_done;
      if (stream_valid != '0) begin
        if (stream_consumed !== INPUT_MASK)
          $fatal(1, "FP32 Tile did not consume the complete input beat");
        if (!input_phase_seen[0])
          input_phase_seen[0] <= 1'b1;
        else
          input_phase_seen[1] <= 1'b1;
      end else if (stream_consumed != '0) begin
        $fatal(1, "FP32 Tile consumed data outside a port input beat");
      end

      for (integer block = 0; block < 8; block++) begin
        if (TAIL_BLOCK_MASK[block] &&
            output_valid[block*2] && output_valid[block*2+1]) begin
          if (!output_low_seen[block]) begin
            for (integer lane = 0; lane < LANES; lane++) begin
              if (output_data[((block*2)*LANES+lane)*8 +: 8] !== 8'h00 ||
                  output_data[((block*2+1)*LANES+lane)*8 +: 8] !== 8'h00)
                $fatal(1, "FP32 Tile block %0d low-half mismatch", block);
            end
            output_low_seen[block] <= 1'b1;
          end else begin
            for (integer lane = 0; lane < LANES; lane++) begin
              // Four chained Adds compute ((1+2)+1)+1 = FP32 6.0.
              if (output_data[((block*2)*LANES+lane)*8 +: 8] !== 8'hc0 ||
                  output_data[((block*2+1)*LANES+lane)*8 +: 8] !== 8'h40)
                $fatal(1, "FP32 Tile block %0d high-half mismatch", block);
            end
            output_high_seen[block] <= 1'b1;
          end
        end
      end

      if (fault)
        $fatal(1, "FP32 Tile reported a fault");
    end
  end

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    config_load = 1'b0;
    config_active = '1;
    config_instruction = '0;
    execute_valid = 1'b0;
    repeat_control = 4'b1011;
    global_config_valid = 1'b1;
    global_fields = '0;
    global_fields.chain_length = VXM_CHAIN_LENGTH_4;
    global_fields.compute_dtype = VXM_FORMAT_FP32;
    global_fields.lhs_read_bits = VXM_READ_BITS_32;
    global_fields.lhs_dtype = VXM_FORMAT_FP32;
    global_fields.rhs_read_bits = VXM_READ_BITS_32;
    global_fields.rhs_dtype = VXM_FORMAT_FP32;
    global_config = global_fields;
    stream_valid = '0;
    stream_data = '0;
    for (integer queue = 0; queue < VXM_LOCAL_QUEUE_COUNT; queue++)
      config_instruction[
        queue*VXM_LOCAL_MAX_INSTRUCTION_WIDTH +: 3] = VXM_LOCAL_ADD;

    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    config_load = 1'b1;
    @(posedge clk);
    @(negedge clk);
    config_load = 1'b0;
    global_config_valid = 1'b0;

    wait (execute_ready);
    @(negedge clk);
    execute_valid = 1'b1;
    @(posedge clk);
    @(negedge clk);
    execute_valid = 1'b0;
    drive_input_half(1'b0);
    @(posedge clk);
    @(negedge clk);
    drive_input_half(1'b1);
    @(posedge clk);
    @(negedge clk);
    stream_valid = '0;
    stream_data = '0;

    wait ((output_low_seen & TAIL_BLOCK_MASK) == TAIL_BLOCK_MASK);
    wait ((output_high_seen & TAIL_BLOCK_MASK) == TAIL_BLOCK_MASK);
    wait (done_seen);
    wait (idle);
    if (input_phase_seen != 2'b11)
      $fatal(1, "FP32 Tile did not observe both input phases");
    $display("LPU_VXM_TILE_FP32_PHASE_TB_PASS");
    $finish;
  end

  initial begin
    #30000;
    $fatal(1, "FP32 Tile phase regression timeout");
  end
endmodule
