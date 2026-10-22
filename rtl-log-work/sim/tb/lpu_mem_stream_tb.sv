`timescale 1ns/1ps

module lpu_mem_stream_tb;
  timeunit 1ns;
  timeprecision 1ps;

  logic clk;
  logic rst_n;
  logic run;
  logic schedule_valid;
  logic schedule_ready;
  logic [7:0] schedule_queue;
  logic schedule_is_instruction;
  logic [31:0] schedule_command;
  logic [415:0] schedule_payload;
  logic host_write_valid;
  logic [6:0] host_column;
  logic [1:0] host_tile;
  logic [15:0] host_address;
  logic [63:0] host_write_data;
  logic [63:0] host_read_data;
  logic [137:0] queue_fault;
  logic [2*52*4-1:0] mem_fault;
  logic [1:0] stream_conflict;
  logic [63:0] cycle;
  logic [63:0] cmodel_vector [0:3];
  logic [63:0] cmodel_schedule [0:5];

  lpu_top #(
    .ICU_QUEUE_DEPTH(4),
    .MEM_DEPTH_ROWS(32),
    .ACTIVE_MEM_COLUMNS(12)
  ) dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .run_i(run),
    .schedule_valid_i(schedule_valid),
    .schedule_ready_o(schedule_ready),
    .schedule_queue_i(schedule_queue),
    .schedule_is_instruction_i(schedule_is_instruction),
    .schedule_command_i(schedule_command),
    .schedule_payload_i(schedule_payload),
    .host_mem_write_valid_i(host_write_valid),
    .host_mem_column_i(host_column),
    .host_mem_tile_i(host_tile),
    .host_mem_address_i(host_address),
    .host_mem_write_data_i(host_write_data),
    .host_mem_read_data_o(host_read_data),
    .mem_east_edge_valid_i(256'b0),
    .mem_east_edge_data_i(16384'b0),
    .mem_east_edge_valid_o(),
    .mem_east_edge_data_o(),
    .mem_west_edge_valid_i(256'b0),
    .mem_west_edge_data_i(16384'b0),
    .mem_west_edge_valid_o(),
    .mem_west_edge_data_o(),
    .dispatch_valid_o(),
    .dispatch_payload_o(),
    .queue_fault_o(queue_fault),
    .mem_fault_o(mem_fault),
    .mxm_fault_o(),
    .sxm_fault_o(),
    .vxm_fault_o(),
    .stream_conflict_o(stream_conflict),
    .cycle_o(cycle)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  task automatic host_write_tile(
    input logic [6:0] column,
    input logic [1:0] tile,
    input logic [15:0] address,
    input logic [63:0] data
  );
    begin
      @(negedge clk);
      host_column = column;
      host_tile = tile;
      host_address = address;
      host_write_data = data;
      host_write_valid = 1'b1;
      @(negedge clk);
      host_write_valid = 1'b0;
    end
  endtask

  task automatic enqueue_instruction(
    input logic [7:0] queue,
    input logic [46:0] instruction
  );
    begin
      @(negedge clk);
      schedule_queue = queue;
      schedule_is_instruction = 1'b1;
      schedule_payload = '0;
      schedule_payload[46:0] = instruction;
      schedule_valid = 1'b1;
      if (!schedule_ready)
        $fatal(1, "schedule queue %0d is not ready", queue);
      @(negedge clk);
      schedule_valid = 1'b0;
    end
  endtask

  task automatic enqueue_command(
    input logic [7:0] queue,
    input logic [31:0] command
  );
    begin
      @(negedge clk);
      schedule_queue = queue;
      schedule_is_instruction = 1'b0;
      schedule_command = command;
      schedule_valid = 1'b1;
      if (!schedule_ready)
        $fatal(1, "schedule queue %0d is not ready", queue);
      @(negedge clk);
      schedule_valid = 1'b0;
    end
  endtask

  task automatic enqueue_cmodel_record(input integer index);
    logic [63:0] record;
    begin
      record = cmodel_schedule[index];
      if (record[8])
        enqueue_instruction(record[7:0], record[55:9]);
      else
        enqueue_command(record[7:0], record[40:9]);
    end
  endtask

  initial begin
    $readmemh("sim/vectors/mem_stream_roundtrip.hex", cmodel_vector);
    $readmemh("sim/vectors/mem_stream_schedule.hex", cmodel_schedule);
    rst_n = 1'b0;
    run = 1'b0;
    schedule_valid = 1'b0;
    schedule_queue = '0;
    schedule_is_instruction = 1'b0;
    schedule_command = '0;
    schedule_payload = '0;
    host_write_valid = 1'b0;
    host_column = '0;
    host_tile = '0;
    host_address = '0;
    host_write_data = '0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    // C-model-style cycle-zero initialization of one complete 32-byte row.
    for (integer tile = 0; tile < 4; tile++)
      host_write_tile(7'd0, tile[1:0], 16'd3, cmodel_vector[tile]);

    // The full C-model system's compatibility mapping injects column 0 Read
    // data at boundary 0. It advances through boundaries 1 and 2, where
    // column 8 consumes it three cycles after the Read.
    enqueue_cmodel_record(0);
    enqueue_cmodel_record(1);
    enqueue_cmodel_record(2);

    @(negedge clk);
    run = 1'b1;
    repeat (16) @(posedge clk);
    @(negedge clk);
    run = 1'b0;

    if (queue_fault != '0)
      $fatal(1, "unexpected ICU queue fault: %h", queue_fault);
    if (mem_fault != '0)
      $fatal(1, "unexpected MEM fault");
    if (stream_conflict != '0)
      $fatal(1, "unexpected stream collision");

    // Offline result collection and comparison against the same C-model
    // vector row used to initialize the source slice.
    host_column = 7'd8;
    host_address = 16'd9;
    for (integer tile = 0; tile < 4; tile++) begin
      host_tile = tile[1:0];
      #1ns;
      if (host_read_data !== cmodel_vector[tile])
        $fatal(1, "tile %0d mismatch: got %h expected %h",
               tile, host_read_data, cmodel_vector[tile]);
    end

    // Exercise the mirrored direction: column 8 produces W6 at boundary 3,
    // it moves through boundary 2 to boundary 1, and column 0 consumes it.
    for (integer tile = 0; tile < 4; tile++)
      host_write_tile(7'd8, tile[1:0], 16'd4, cmodel_vector[tile]);

    enqueue_cmodel_record(3);
    enqueue_cmodel_record(4);
    enqueue_cmodel_record(5);

    @(negedge clk);
    run = 1'b1;
    repeat (16) @(posedge clk);
    @(negedge clk);
    run = 1'b0;

    if (queue_fault != '0 || mem_fault != '0 || stream_conflict != '0)
      $fatal(1, "fault during mirrored West transfer");

    host_column = 7'd0;
    host_address = 16'd10;
    for (integer tile = 0; tile < 4; tile++) begin
      host_tile = tile[1:0];
      #1ns;
      if (host_read_data !== cmodel_vector[tile])
        $fatal(1, "West tile %0d mismatch: got %h expected %h",
               tile, host_read_data, cmodel_vector[tile]);
    end

    $display("FTLPU-VMODEL East/West MEM-stream test passed in %0d cycles", cycle);
    $finish;
  end
endmodule
