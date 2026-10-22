`timescale 1ns/1ps

module lpu_mxm_accumulator_tb;
  timeunit 1ns;
  timeprecision 1ps;

  localparam integer INIT_WORDS = 272;
  localparam integer GOLDEN_WORDS = 80;
  localparam integer SCHEDULE_RECORDS = 121;

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
  logic [1:0] mxm_fault;
  logic [1:0] sxm_fault;
  logic vxm_fault;
  logic [1:0] stream_conflict;
  logic [63:0] cycle;
  logic [63:0] cmodel_init [0:INIT_WORDS-1];
  logic [63:0] cmodel_golden [0:GOLDEN_WORDS-1];
  logic [479:0] cmodel_schedule [0:SCHEDULE_RECORDS-1];

  lpu_top #(
    .ICU_QUEUE_DEPTH(16),
    .MEM_DEPTH_ROWS(16),
    .ACTIVE_MEM_COLUMNS(18)
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
    .mxm_fault_o(mxm_fault),
    .sxm_fault_o(sxm_fault),
    .vxm_fault_o(vxm_fault),
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
    $readmemh("sim/vectors/mxm_accumulator_init.hex", cmodel_init);
    $readmemh("sim/vectors/mxm_accumulator_golden.hex", cmodel_golden);
    $readmemh("sim/vectors/mxm_accumulator_schedule.hex", cmodel_schedule);

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

    for (integer block = 0; block < 4; block++)
      for (integer column = 0; column < 16; column++)
        for (integer tile = 0; tile < 4; tile++)
          host_write_tile(
            column[6:0], tile[1:0], block[15:0],
            cmodel_init[(block*16+column)*4+tile]);

    for (integer input_set = 0; input_set < 2; input_set++)
      for (integer column = 16; column < 18; column++)
        for (integer tile = 0; tile < 4; tile++)
          host_write_tile(
            column[6:0], tile[1:0],
            (input_set == 0) ? 16'd4 : 16'd8,
            cmodel_init[256+input_set*8+(column-16)*4+tile]);

    for (integer record = 0; record < SCHEDULE_RECORDS; record++)
      enqueue_cmodel_record(record);

    @(negedge clk);
    run = 1'b1;
    repeat (260) @(posedge clk);
    @(negedge clk);

    if (queue_fault != '0 || mem_fault != '0 || mxm_fault != '0 ||
        sxm_fault != '0 || vxm_fault || stream_conflict != '0)
      $fatal(1,
        {"MXM accumulator fault queue=%h mem=%h mxm=%b sxm=%b ",
         "vxm=%b conflict=%b"},
        queue_fault, mem_fault, mxm_fault, sxm_fault, vxm_fault,
        stream_conflict);
    run = 1'b0;

    for (integer output_index = 0; output_index < 5; output_index++) begin
      case (output_index)
        0: host_address = 16'd5;
        1: host_address = 16'd6;
        2: host_address = 16'd7;
        3: host_address = 16'd9;
        default: host_address = 16'd10;
      endcase
      for (integer column = 0; column < 4; column++) begin
        host_column = column[6:0];
        for (integer tile = 0; tile < 4; tile++) begin
          host_tile = tile[1:0];
          #1ns;
          if (host_read_data !==
              cmodel_golden[output_index*16+column*4+tile])
            $fatal(1,
              {"MXM accumulator mismatch output=%0d column=%0d ",
               "tile=%0d got=%h expected=%h"},
              output_index, column, tile, host_read_data,
              cmodel_golden[output_index*16+column*4+tile]);
        end
      end
    end

    $display(
      "FTLPU-VMODEL C-model MXM accumulator SRAM/read-clear passed");
    $finish;
  end
endmodule
