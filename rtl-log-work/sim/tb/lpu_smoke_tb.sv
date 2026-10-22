`timescale 1ns/1ps

module lpu_smoke_tb;
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
  logic [63:0] cycle;

  lpu_top #(
    .ICU_QUEUE_DEPTH(4),
    .MEM_DEPTH_ROWS(32),
    .ACTIVE_MEM_COLUMNS(0)
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
    .host_mem_write_valid_i(1'b0),
    .host_mem_column_i(7'b0),
    .host_mem_tile_i(2'b0),
    .host_mem_address_i(16'b0),
    .host_mem_write_data_i(64'b0),
    .host_mem_read_data_o(),
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
    .queue_fault_o(),
    .mem_fault_o(),
    .mxm_fault_o(),
    .sxm_fault_o(),
    .vxm_fault_o(),
    .stream_conflict_o(),
    .cycle_o(cycle)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  initial begin
    rst_n = 1'b0;
    run = 1'b0;
    schedule_valid = 1'b0;
    schedule_queue = '0;
    schedule_is_instruction = 1'b0;
    schedule_command = '0;
    schedule_payload = '0;
    repeat (4) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    run = 1'b1;
    repeat (4) @(posedge clk);
    #1ns;
    if (cycle != 4)
      $fatal(1, "cycle counter mismatch: %0d", cycle);
    $display("FTLPU-VMODEL smoke test passed");
    $finish;
  end
endmodule
