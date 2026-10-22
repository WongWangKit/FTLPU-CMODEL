`timescale 1ns/1ps

module lpu_vxm_instruction_controller_tb;
  import lpu_pkg::*;

  localparam integer COUNT_WIDTH = 10;
  localparam integer INTERVAL_WIDTH = 8;

  logic clk;
  logic rst_n;
  logic command_valid;
  wire command_ready;
  logic [VXM_LOCAL_QUEUE_COUNT-1:0] command_local_active;
  logic [VXM_LOCAL_QUEUE_COUNT*VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    command_local_instruction;
  logic [VXM_GLOBAL_CONFIG_WIDTH-1:0] command_global_config;
  logic [COUNT_WIDTH-1:0] command_repeat_count;
  logic [INTERVAL_WIDTH-1:0] command_repeat_interval;
  logic command_accumulator_enable;
  logic command_output_enable;
  wire local_config_load;
  logic local_config_ready;
  wire [VXM_LOCAL_QUEUE_COUNT-1:0] local_config_active;
  wire [VXM_LOCAL_QUEUE_COUNT*VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    local_config_instruction;
  wire global_config_valid;
  wire [VXM_GLOBAL_CONFIG_WIDTH-1:0] global_config;
  wire execute_valid;
  logic execute_ready;
  wire [VXM_REPEAT_CONTROL_WIDTH-1:0] repeat_control;
  logic config_done;
  wire busy;
  wire done;
  wire fault;

  integer cycle_count;
  integer fire_count;
  integer previous_fire_cycle;

  always #5 clk = ~clk;

  lpu_vxm_instruction_controller #(
    .REPEAT_COUNT_WIDTH(COUNT_WIDTH),
    .REPEAT_INTERVAL_WIDTH(INTERVAL_WIDTH)
  ) dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .command_valid_i(command_valid),
    .command_ready_o(command_ready),
    .command_local_active_i(command_local_active),
    .command_local_instruction_i(command_local_instruction),
    .command_global_config_i(command_global_config),
    .command_repeat_count_i(command_repeat_count),
    .command_repeat_interval_i(command_repeat_interval),
    .command_accumulator_enable_i(command_accumulator_enable),
    .command_output_enable_i(command_output_enable),
    .local_config_load_o(local_config_load),
    .local_config_ready_i(local_config_ready),
    .local_config_active_o(local_config_active),
    .local_config_instruction_o(local_config_instruction),
    .global_config_valid_o(global_config_valid),
    .global_config_o(global_config),
    .execute_valid_o(execute_valid),
    .execute_ready_i(execute_ready),
    .repeat_control_o(repeat_control),
    .config_done_i(config_done),
    .busy_o(busy),
    .done_o(done),
    .fault_o(fault)
  );

  always @(posedge clk) begin
    if (!rst_n) begin
      cycle_count = 0;
      fire_count = 0;
      previous_fire_cycle = -100;
    end else begin
      cycle_count = cycle_count + 1;
      if (execute_valid && execute_ready) begin
        if ((fire_count > 0) &&
            ((cycle_count - previous_fire_cycle) < 2))
          $fatal(1, "repeat interval was shorter than two cycles");
        if (repeat_control[VXM_REPEAT_FIRST_ITERATION_BIT] !==
            (fire_count == 0))
          $fatal(1, "first-iteration phase mismatch at issue %0d", fire_count);
        if (repeat_control[VXM_REPEAT_LAST_ITERATION_BIT] !==
            (fire_count == 2))
          $fatal(1, "last-iteration phase mismatch at issue %0d", fire_count);
        if (!repeat_control[VXM_REPEAT_ACCUMULATOR_BIT] ||
            !repeat_control[VXM_REPEAT_OUTPUT_ENABLE_BIT])
          $fatal(1, "shared repeat policy bits were not preserved");
        previous_fire_cycle = cycle_count;
        fire_count = fire_count + 1;
      end
    end
  end

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    command_valid = 1'b0;
    command_local_active = 8'hbd;
    command_local_instruction = 56'h12_3456_789a_bcde;
    command_global_config = 15'h6a35;
    command_repeat_count = 10'd3;
    command_repeat_interval = 8'd2;
    command_accumulator_enable = 1'b1;
    command_output_enable = 1'b1;
    local_config_ready = 1'b0;
    execute_ready = 1'b1;
    config_done = 1'b0;

    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    command_valid = 1'b1;
    @(posedge clk);
    @(negedge clk);
    command_valid = 1'b0;

    wait (local_config_load);
    repeat (2) begin
      @(posedge clk);
      #1;
      if (!local_config_load || !busy || !global_config_valid)
        $fatal(1, "configuration load was not held during backpressure");
      if ((local_config_active !== command_local_active) ||
          (local_config_instruction !== command_local_instruction) ||
          (global_config !== command_global_config))
        $fatal(1, "resident configuration payload mismatch");
    end

    @(negedge clk);
    local_config_ready = 1'b1;
    @(posedge clk);
    @(negedge clk);
    local_config_ready = 1'b0;

    wait (fire_count == 3);
    @(posedge clk);
    #1;
    if (!busy || execute_valid || done)
      $fatal(1, "controller did not wait for the returned end marker");

    @(negedge clk);
    config_done = 1'b1;
    @(posedge clk);
    #1;
    if (!done || busy || !command_ready)
      $fatal(1, "returned end marker did not retire the configuration");
    @(negedge clk);
    config_done = 1'b0;
    @(posedge clk);
    #1;
    if (done)
      $fatal(1, "done must be a one-cycle pulse");

    // A zero execution count is illegal and must not start a configuration.
    @(negedge clk);
    command_repeat_count = '0;
    command_valid = 1'b1;
    @(posedge clk);
    @(negedge clk);
    command_valid = 1'b0;
    @(posedge clk);
    #1;
    if (!fault || busy)
      $fatal(1, "zero repeat count was not rejected");

    $display("LPU_VXM_INSTRUCTION_CONTROLLER_TB_PASS");
    $finish;
  end

  initial begin
    #20000;
    $fatal(1, "VXM instruction-controller timeout fires=%0d", fire_count);
  end
endmodule
