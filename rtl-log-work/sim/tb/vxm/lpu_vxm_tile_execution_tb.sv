`timescale 1ns/1ps

module lpu_vxm_tile_execution_tb;
  import lpu_pkg::*;

  localparam integer STAGES = 16;
  localparam integer LANES = 8;
  localparam integer CONTAINER_WIDTH = 32;

  logic clk;
  logic rst_n;
  logic local_config_load;
  wire local_config_ready;
  logic [7:0] local_config_active;
  logic [8*7-1:0] local_config_instruction;
  logic execute_valid;
  wire execute_ready;
  logic [VXM_REPEAT_CONTROL_WIDTH-1:0] repeat_control;
  vxm_repeat_control_t repeat_fields;
  wire config_done;
  logic global_config_valid;
  logic [VXM_GLOBAL_CONFIG_WIDTH-1:0] global_config;
  vxm_global_config_t config_fields;
  logic [31:0] stream_valid;
  logic [32*LANES*8-1:0] stream_data;
  wire [STAGES*LANES-1:0] tail_valid;
  wire [STAGES*LANES*CONTAINER_WIDTH-1:0] tail_value;
  wire [STAGES-1:0] instruction_pending;
  wire idle;
  wire fault;
  logic [STAGES*LANES-1:0] tail_seen;
  logic [STAGES*LANES-1:0] expected_tail_mask;
  logic done_seen;

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
    .stream_consumed_o(),
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
    .output_valid_o(),
    .output_data_o(),
    .tail_valid_o(tail_valid),
    .tail_value_o(tail_value),
    .tail_original_o(),
    .tail_auxiliary_o(),
    .accumulator_state_valid_o(),
    .accumulator_state_data_o(),
    .feedback_state_valid_o(),
    .feedback_state_value_o(),
    .instruction_pending_o(instruction_pending),
    .idle_o(idle),
    .fault_o(fault)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tail_seen <= '0;
      done_seen <= 1'b0;
    end else begin
      tail_seen <= tail_seen | tail_valid;
      done_seen <= done_seen | config_done;
      if (fault)
        $fatal(1, "VXM tile assembly reported a fault");
      for (integer stage = 0; stage < STAGES; stage++) begin
        for (integer lane = 0; lane < LANES; lane++) begin
          if (tail_valid[stage*LANES+lane] &&
              (tail_value[(stage*LANES+lane)*CONTAINER_WIDTH +:
                CONTAINER_WIDTH] !== 32'h00003c00))
            $fatal(1, "tile stage %0d lane %0d tail mismatch", stage, lane);
        end
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
    repeat_fields.first_iteration = 1'b1;
    repeat_fields.last_iteration = 1'b1;
    repeat_control = repeat_fields;
    global_config_valid = 1'b1;
    config_fields = '0;
    config_fields.chain_length = VXM_CHAIN_LENGTH_4;
    config_fields.compute_dtype = VXM_FORMAT_FP16;
    config_fields.lhs_dtype = VXM_FORMAT_FP16;
    config_fields.rhs_dtype = VXM_FORMAT_FP16;
    global_config = config_fields;
    stream_valid = '1;
    stream_data = '0;
    expected_tail_mask = '0;
    expected_tail_mask[3*LANES +: LANES] = '1;
    expected_tail_mask[7*LANES +: LANES] = '1;
    expected_tail_mask[11*LANES +: LANES] = '1;
    expected_tail_mask[15*LANES +: LANES] = '1;

    // Every fixed FP16 group contains 1.0 for all lanes.
    for (integer stream = 0; stream < 32; stream++) begin
      for (integer lane = 0; lane < LANES; lane++) begin
        stream_data[(stream*LANES+lane)*8 +: 8] =
          (stream % 2) == 0 ? 8'h00 : 8'h3c;
      end
    end

    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    local_config_load = 1'b1;
    @(posedge clk);
    @(negedge clk);
    local_config_load = 1'b0;

    wait (execute_ready);
    @(negedge clk);
    execute_valid = 1'b1;
    @(posedge clk);
    @(negedge clk);
    execute_valid = 1'b0;

    wait ((tail_seen & expected_tail_mask) == expected_tail_mask);
    wait (done_seen);
    wait (idle);
    if (|instruction_pending)
      $fatal(1, "tile became idle with pending instructions");
    $display("LPU_VXM_TILE_EXECUTION_TB_PASS");
    $finish;
  end

  initial begin
    #20000;
    $fatal(1, "VXM tile assembly timeout pending=%h seen=%h",
           instruction_pending, tail_seen);
  end
endmodule
