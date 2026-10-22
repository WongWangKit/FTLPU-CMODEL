`timescale 1ns/1ps

module lpu_mxm_vector_test #(
  parameter INIT_FILE = "sim/vectors/mxm_vector_identity_init.hex",
  parameter GOLDEN_FILE = "sim/vectors/mxm_vector_identity_golden.hex",
  parameter SCHEDULE_FILE = "sim/vectors/mxm_vector_identity_schedule.hex",
  parameter bit FORMAT_BF16 = 1'b1
);
  timeunit 1ns;
  timeprecision 1ps;

  localparam integer INIT_WORDS = 528;
  localparam integer GOLDEN_WORDS = 32;
  localparam integer SCHEDULE_RECORDS = 134;

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
    $readmemh(INIT_FILE, cmodel_init);
    $readmemh(GOLDEN_FILE, cmodel_golden);
    $readmemh(SCHEDULE_FILE, cmodel_schedule);

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

    for (integer test_case = 0; test_case < 2; test_case++)
      for (integer block = 0; block < 4; block++)
        for (integer column = 0; column < 16; column++)
          for (integer tile = 0; tile < 4; tile++)
            host_write_tile(
              column[6:0], tile[1:0],
              (test_case*8+block),
              cmodel_init[
                test_case*256+(block*16+column)*4+tile]);

    for (integer test_case = 0; test_case < 2; test_case++)
      for (integer column = 16; column < 18; column++)
        for (integer tile = 0; tile < 4; tile++)
          host_write_tile(
            column[6:0], tile[1:0],
            (test_case == 0) ? 16'd4 : 16'd12,
            cmodel_init[512+test_case*8+(column-16)*4+tile]);

    for (integer record = 0; record < SCHEDULE_RECORDS; record++)
      enqueue_cmodel_record(record);

    @(negedge clk);
    run = 1'b1;
    repeat (120) @(posedge clk);
    @(negedge clk);

    if (queue_fault != '0 || mem_fault != '0 || mxm_fault != '0 ||
        sxm_fault != '0 || vxm_fault || stream_conflict != '0)
      $fatal(1,
        "MXM regression fault queue=%h mem=%h mxm=%b sxm=%b vxm=%b conflict=%b",
        queue_fault, mem_fault, mxm_fault, sxm_fault, vxm_fault,
        stream_conflict);
    run = 1'b0;

    for (integer test_case = 0; test_case < 2; test_case++) begin
      host_address = (test_case == 0) ? 16'd5 : 16'd13;
      for (integer column = 0; column < 4; column++) begin
        host_column = column[6:0];
        for (integer tile = 0; tile < 4; tile++) begin
          host_tile = tile[1:0];
          #1ns;
          if (host_read_data !==
              cmodel_golden[test_case*16+column*4+tile])
            $fatal(1,
              {"MXM double-buffer mismatch case=%0d column=%0d ",
               "tile=%0d got=%h expected=%h"},
              test_case, column, tile, host_read_data,
              cmodel_golden[test_case*16+column*4+tile]);
        end
      end
    end

    if (FORMAT_BF16)
      $display(
        {"FTLPU-VMODEL C-model MXM BF16 double-buffer permuted identity ",
         "passed in %0d cycles"},
        cycle);
    else
      $display(
        {"FTLPU-VMODEL C-model MXM FP16 double-buffer permuted identity ",
         "passed in %0d cycles"},
        cycle);
    $finish;
  end
endmodule

module lpu_mxm_vector_tb;
  lpu_mxm_vector_test u_test();
endmodule

module lpu_mxm_fp16_tb;
  lpu_mxm_vector_test #(
    .INIT_FILE("sim/vectors/mxm_vector_fp16_init.hex"),
    .GOLDEN_FILE("sim/vectors/mxm_vector_fp16_golden.hex"),
    .SCHEDULE_FILE("sim/vectors/mxm_vector_fp16_schedule.hex"),
    .FORMAT_BF16(1'b0)
  ) u_test();
endmodule
