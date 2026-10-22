`timescale 1ns/1ps

module lpu_sxm_wavefront_tb;
  timeunit 1ns;
  timeprecision 1ps;

  localparam integer VECTOR_WORDS = 256;
  localparam integer SCHEDULE_RECORDS = 107;

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
  logic [1:0] sxm_fault;
  logic [1:0] stream_conflict;
  logic [63:0] cycle;
  logic [63:0] cmodel_init [0:VECTOR_WORDS-1];
  logic [63:0] cmodel_golden [0:VECTOR_WORDS-1];
  logic [479:0] cmodel_schedule [0:SCHEDULE_RECORDS-1];

  lpu_top #(
    .ICU_QUEUE_DEPTH(16),
    .MEM_DEPTH_ROWS(64),
    .ACTIVE_MEM_COLUMNS(16)
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
    .sxm_fault_o(sxm_fault),
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

  task automatic enqueue_cmodel_record(input integer index);
    logic [479:0] record;
    begin
      record = cmodel_schedule[index];
      @(negedge clk);
      schedule_queue = record[7:0];
      schedule_is_instruction = record[8];
      schedule_command = record[40:9];
      schedule_payload = record[456:41];
      schedule_valid = 1'b1;
      #1ps;
      if (!schedule_ready)
        $fatal(1, "schedule queue %0d is not ready at record %0d",
               schedule_queue, index);
      @(negedge clk);
      schedule_valid = 1'b0;
    end
  endtask

  initial begin
    $readmemh("sim/vectors/sxm_wavefront_transpose_init.hex", cmodel_init);
    $readmemh("sim/vectors/sxm_wavefront_transpose_golden.hex", cmodel_golden);
    $readmemh(
      "sim/vectors/sxm_wavefront_transpose_schedule.hex", cmodel_schedule);

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

    // C-model memory layout: four block rows (addresses), four block columns
    // (tile rows), and sixteen byte-plane streams (MEM columns).
    for (integer beat = 0; beat < 4; beat++)
      for (integer column = 0; column < 16; column++)
        for (integer tile = 0; tile < 4; tile++)
          host_write_tile(
            column[6:0], tile[1:0], beat[15:0],
            cmodel_init[(beat*16+column)*4+tile]);

    // The schedule contains MEM Repeat+stride commands, four overlapping
    // Transpose captures, and seven distinct non-identity wavefront maps.
    for (integer record = 0; record < SCHEDULE_RECORDS; record++)
      enqueue_cmodel_record(record);

    @(negedge clk);
    run = 1'b1;
    repeat (56) @(posedge clk);
    @(negedge clk);

    if (queue_fault != '0)
      $fatal(1, "unexpected ICU queue fault: %h", queue_fault);
    if (mem_fault != '0)
      $fatal(1, "unexpected MEM fault: %h sxm=%b conflict=%b",
             mem_fault, sxm_fault, stream_conflict);
    if (sxm_fault != '0)
      $fatal(1, "unexpected SXM fault: %b", sxm_fault);
    if (stream_conflict != '0)
      $fatal(1, "unexpected stream collision: %b", stream_conflict);
    run = 1'b0;

    for (integer beat = 0; beat < 4; beat++) begin
      host_address = 16'd32 + beat;
      for (integer column = 0; column < 16; column++) begin
        host_column = column[6:0];
        for (integer tile = 0; tile < 4; tile++) begin
          host_tile = tile[1:0];
          #1ns;
          if (host_read_data !==
              cmodel_golden[(beat*16+column)*4+tile])
            $fatal(1,
              "wavefront mismatch beat=%0d column=%0d tile=%0d got=%h expected=%h",
              beat, column, tile, host_read_data,
              cmodel_golden[(beat*16+column)*4+tile]);
        end
      end
    end

    $display(
      "FTLPU-VMODEL C-model 32x32 FP16 wavefront transpose passed in %0d cycles",
      cycle);
    $finish;
  end
endmodule
