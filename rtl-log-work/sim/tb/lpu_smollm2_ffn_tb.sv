`timescale 1ns/1ps

module lpu_smollm2_ffn_tb;
  timeunit 1ns;
  timeprecision 1ps;

  localparam integer INIT_WORDS = 448;
  localparam integer GOLDEN_WORDS = 448;
  localparam integer GATE_RECORDS = 296;
  localparam integer SWIGLU_RECORDS = 194;
  localparam integer DOWN_RECORDS = 150;

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
  logic [63:0] init_words [0:INIT_WORDS-1];
  logic [63:0] golden_words [0:GOLDEN_WORDS-1];
  logic [479:0] gate_schedule [0:GATE_RECORDS-1];
  logic [479:0] swiglu_schedule [0:SWIGLU_RECORDS-1];
  logic [479:0] down_schedule [0:DOWN_RECORDS-1];

  lpu_top #(
    .ICU_QUEUE_DEPTH(32),
    .MEM_DEPTH_ROWS(64),
    .ACTIVE_MEM_COLUMNS(52)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n), .run_i(run),
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
    .mem_east_edge_valid_i('0), .mem_east_edge_data_i('0),
    .mem_east_edge_valid_o(), .mem_east_edge_data_o(),
    .mem_west_edge_valid_i('0), .mem_west_edge_data_i('0),
    .mem_west_edge_valid_o(), .mem_west_edge_data_o(),
    .dispatch_valid_o(), .dispatch_payload_o(),
    .queue_fault_o(queue_fault), .mem_fault_o(mem_fault),
    .mxm_fault_o(mxm_fault), .sxm_fault_o(sxm_fault),
    .vxm_fault_o(vxm_fault), .stream_conflict_o(stream_conflict),
    .cycle_o(cycle)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

`ifdef FSDB_DUMP
  initial begin
    $fsdbDumpfile("smollm2_ffn.fsdb");
    $fsdbDumpvars(1, lpu_smollm2_ffn_tb);
    $fsdbDumpvars(2, dut.gen_mem[0]);
  end
`endif

  task automatic host_write_word(
    input integer column,
    input integer tile,
    input integer address,
    input logic [63:0] data
  );
    begin
      @(negedge clk);
      host_column = column[6:0];
      host_tile = tile[1:0];
      host_address = address[15:0];
      host_write_data = data;
      host_write_valid = 1'b1;
      @(negedge clk);
      host_write_valid = 1'b0;
    end
  endtask

  task automatic enqueue_word(input logic [479:0] record);
    begin
      @(negedge clk);
      schedule_queue = record[7:0];
      schedule_is_instruction = record[8];
      schedule_command = record[40:9];
      schedule_payload = record[456:41];
      schedule_valid = 1'b1;
      #1ps;
      if (!schedule_ready)
        $fatal(1, "FFN schedule queue %0d is full", schedule_queue);
      @(negedge clk);
      schedule_valid = 1'b0;
    end
  endtask

  task automatic phase_reset;
    begin
      @(negedge clk);
      run = 1'b0;
      rst_n = 1'b0;
      repeat (2) @(posedge clk);
      @(negedge clk);
      rst_n = 1'b1;
    end
  endtask

  task automatic run_phase(input integer cycles_to_run);
    begin
      @(negedge clk);
      run = 1'b1;
      repeat (cycles_to_run) @(posedge clk);
      @(negedge clk);
      run = 1'b0;
      if (queue_fault != '0 || mem_fault != '0 || mxm_fault != '0 ||
          sxm_fault != '0 || vxm_fault || stream_conflict != '0)
        $fatal(1,
          "FFN phase fault queue=%h mem=%h mxm=%b sxm=%b vxm=%b conflict=%b",
          queue_fault, mem_fault, mxm_fault, sxm_fault, vxm_fault,
          stream_conflict);
    end
  endtask

  task automatic expect_word(
    input integer column,
    input integer tile,
    input integer address,
    input integer golden_index
  );
    begin
      host_column = column[6:0];
      host_tile = tile[1:0];
      host_address = address[15:0];
      #1ns;
      if (host_read_data !== golden_words[golden_index])
        $fatal(1,
          "FFN mismatch address=%0d column=%0d tile=%0d got=%h expected=%h",
          address, column, tile, host_read_data, golden_words[golden_index]);
    end
  endtask

  initial begin
    $readmemh("sim/vectors/smollm2_ffn_init.hex", init_words);
    $readmemh("sim/vectors/smollm2_ffn_golden.hex", golden_words);
    $readmemh("sim/vectors/smollm2_ffn_gate_schedule.hex", gate_schedule);
    $readmemh("sim/vectors/smollm2_ffn_swiglu_schedule.hex", swiglu_schedule);
    $readmemh("sim/vectors/smollm2_ffn_down_schedule.hex", down_schedule);

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
          host_write_word(column, tile, block,
            init_words[(block*16+column)*4+tile]);
    for (integer block = 0; block < 4; block++)
      for (integer column = 0; column < 8; column++)
        for (integer tile = 0; tile < 4; tile++)
          host_write_word(column, tile, 4+block,
            init_words[256+(block*8+column)*4+tile]);
    for (integer column = 32; column < 48; column++)
      for (integer tile = 0; tile < 4; tile++)
        host_write_word(column, tile, 8,
          init_words[384+(column-32)*4+tile]);

    for (integer record = 0; record < GATE_RECORDS; record++)
      enqueue_word(gate_schedule[record]);
    run_phase(180);
    for (integer row = 0; row < 8; row++)
      for (integer column = 0; column < 8; column++)
        for (integer tile = 0; tile < 4; tile++)
          expect_word(column, tile, 20+row,
            (row*8+column)*4+tile);

    phase_reset();
    for (integer record = 0; record < SWIGLU_RECORDS; record++)
      enqueue_word(swiglu_schedule[record]);
    run_phase(64);
    for (integer column = 32; column < 48; column++)
      for (integer tile = 0; tile < 4; tile++)
        expect_word(column, tile, 0,
          256+(column-32)*4+tile);

    phase_reset();
    for (integer record = 0; record < DOWN_RECORDS; record++)
      enqueue_word(down_schedule[record]);
    run_phase(96);
    for (integer column = 0; column < 32; column++)
      for (integer tile = 0; tile < 4; tile++)
        expect_word(column, tile, 30,
          320+column*4+tile);

    $display(
      "FTLPU-VMODEL C-model SmolLM2 FFN passed: X[8,32], gate/up, BF16 SwiGLU, down");
    $finish;
  end
endmodule
