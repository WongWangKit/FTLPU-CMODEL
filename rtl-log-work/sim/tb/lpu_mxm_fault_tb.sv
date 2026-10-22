`timescale 1ns/1ps

module lpu_mxm_fault_tb;
  timeunit 1ns;
  timeprecision 1ps;

  logic clk;
  logic rst_n;
  logic run;

  logic [3:0] load_row_valid;
  logic [4*48-1:0] load_row_instruction;
  logic [3:0] dequant_row_valid;
  logic [4*16-1:0] dequant_row_instruction;
  logic [4*32-1:0] load_east_consumed;
  logic [2*4*4*8*8*16-1:0] weight_bits;
  logic [2*4*4-1:0] cell_valid;
  logic load_fault;

  logic [3:0] compute_row_valid;
  logic [4*48-1:0] compute_row_instruction;
  logic [4*32-1:0] compute_east_consumed;
  logic result_valid;
  logic [8*32*32-1:0] result_values;
  logic [12:0] result_address;
  logic [5:0] result_stream_base;
  logic result_stream_destination;
  logic result_clear;
  logic result_block_mode;
  logic compute_fault;

  logic [4*32-1:0] east_valid;
  logic [4*32*64-1:0] east_data;
  logic [47:0] candidate_instruction;

  lpu_mxm_weight_buffer u_weight_buffer (
    .clk_i(clk),
    .rst_ni(rst_n),
    .run_i(run),
    .load_row_valid_i(load_row_valid),
    .load_row_instruction_i(load_row_instruction),
    .dequant_row_valid_i(dequant_row_valid),
    .dequant_row_instruction_i(dequant_row_instruction),
    .east_valid_i(east_valid),
    .east_data_i(east_data),
    .east_consumed_o(load_east_consumed),
    .weight_bits_o(weight_bits),
    .cell_valid_o(cell_valid),
    .fault_o(load_fault)
  );

  lpu_mxm_compute u_compute (
    .clk_i(clk),
    .rst_ni(rst_n),
    .run_i(run),
    .compute_row_valid_i(compute_row_valid),
    .compute_row_instruction_i(compute_row_instruction),
    .east_valid_i(east_valid),
    .east_data_i(east_data),
    .weight_bits_i({(2*4*4*8*8*16){1'b0}}),
    .cell_valid_i({(2*4*4){1'b1}}),
    .east_consumed_o(compute_east_consumed),
    .result_valid_o(result_valid),
    .result_values_o(result_values),
    .result_address_o(result_address),
    .result_stream_base_o(result_stream_base),
    .result_stream_destination_o(result_stream_destination),
    .result_clear_o(result_clear),
    .result_block_mode_o(result_block_mode),
    .result_ready_i(1'b1),
    .fault_o(compute_fault)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  function automatic logic [47:0] supported_compute_instruction();
    logic [47:0] instruction;
    begin
      instruction = '0;
      instruction[1:0] = 2'd1;
      instruction[43:28] = 16'd1;
      instruction[44] = 1'b1;
      instruction[45] = 1'b1;
      supported_compute_instruction = instruction;
    end
  endfunction

  task automatic reset_run_state;
    begin
      @(negedge clk);
      run = 1'b0;
      load_row_valid = '0;
      dequant_row_valid = '0;
      compute_row_valid = '0;
      repeat (2) @(posedge clk);
      #1ps;
      if (load_fault || compute_fault)
        $fatal(1, "MXM fault did not clear while run was deasserted");
      @(negedge clk);
      run = 1'b1;
    end
  endtask

  task automatic expect_load_fault(
    input logic [47:0] instruction,
    input string name
  );
    begin
      reset_run_state();
      load_row_instruction = '0;
      load_row_instruction[0 +: 48] = instruction;
      load_row_valid = 4'b0001;
      @(posedge clk);
      #1ps;
      if (!load_fault)
        $fatal(1, "unsupported MXM load did not fault: %0s", name);
      if (load_east_consumed != '0)
        $fatal(1, "unsupported MXM load consumed data: %0s", name);
      @(negedge clk);
      load_row_valid = '0;
      repeat (2) @(posedge clk);
      #1ps;
      if (!load_fault)
        $fatal(1, "MXM load fault was not sticky: %0s", name);
    end
  endtask

  task automatic expect_compute_fault(
    input logic [47:0] instruction,
    input logic [3:0] row_valid,
    input string name
  );
    begin
      reset_run_state();
      compute_row_instruction = '0;
      for (integer tile = 0; tile < 4; tile++)
        compute_row_instruction[tile*48 +: 48] = instruction;
      compute_row_valid = row_valid;
      @(posedge clk);
      #1ps;
      if (!compute_fault)
        $fatal(1, "unsupported MXM compute did not fault: %0s", name);
      if (compute_east_consumed != '0)
        $fatal(1, "unsupported MXM compute consumed data: %0s", name);
      @(negedge clk);
      compute_row_valid = '0;
      repeat (2) @(posedge clk);
      #1ps;
      if (!compute_fault)
        $fatal(1, "MXM compute fault was not sticky: %0s", name);
    end
  endtask

  initial begin
    rst_n = 1'b0;
    run = 1'b0;
    load_row_valid = '0;
    load_row_instruction = '0;
    dequant_row_valid = '0;
    dequant_row_instruction = '0;
    compute_row_valid = '0;
    compute_row_instruction = '0;
    east_valid = '1;
    east_data = '0;
    candidate_instruction = '0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    reset_run_state();
    load_row_instruction = '0;
    load_row_instruction[9] = 1'b1;
    load_row_valid = 4'b0001;
    @(posedge clk);
    #1ps;
    if (load_fault || !cell_valid[0] || load_east_consumed[15:0] != '1)
      $fatal(1, "supported Direct16 load baseline failed");

    reset_run_state();
    candidate_instruction = supported_compute_instruction();
    compute_row_instruction = '0;
    compute_row_instruction[0 +: 48] = candidate_instruction;
    compute_row_valid = 4'b0001;
    @(posedge clk);
    #1ps;
    if (compute_fault || !compute_east_consumed[0] ||
        !compute_east_consumed[1])
      $fatal(1, "supported Vector Compute baseline failed");

    candidate_instruction = '0;
    candidate_instruction[9] = 1'b1;
    candidate_instruction[5] = 1'b1;
    reset_run_state();
    for (integer column = 0; column < 8; column++) begin
      candidate_instruction[8:6] = column[2:0];
      load_row_instruction = '0;
      load_row_instruction[0 +: 48] = candidate_instruction;
      load_row_valid = 4'b0001;
      #1ps;
      if (load_east_consumed[1:0] != 2'b11 ||
          load_east_consumed[15:2] != '0)
        $fatal(1, "Column Direct16 consumed the wrong streams");
      @(posedge clk);
      #1ps;
      if (load_fault || (cell_valid[0] != (column == 7)))
        $fatal(1, "Column Direct16 validity failed at column %0d", column);
      @(negedge clk);
    end
    load_row_valid = '0;

    candidate_instruction = '0;
    reset_run_state();
    east_data = '0;
    for (integer stream = 0; stream < 8; stream++)
      east_data[stream*64 +: 64] = 64'hfcfcfcfcfcfcfcfc;
    load_row_instruction = '0;
    load_row_instruction[0 +: 48] = candidate_instruction;
    dequant_row_instruction = '0;
    dequant_row_instruction[0 +: 16] = 16'h3f00;
    load_row_valid = 4'b0001;
    dequant_row_valid = 4'b0001;
    #1ps;
    if (load_east_consumed[7:0] != '1 ||
        load_east_consumed[15:8] != '0)
      $fatal(1, "INT8 dequant consumed the wrong streams");
    @(posedge clk);
    #1ps;
    if (load_fault || !cell_valid[0] || weight_bits[0 +: 16] != 16'hc000)
      $fatal(1, "supported INT8 dequant load baseline failed");

    candidate_instruction = '0;
    candidate_instruction[5] = 1'b1;
    reset_run_state();
    east_data = '0;
    east_data[0 +: 64] = 64'hfcfcfcfcfcfcfcfc;
    dequant_row_instruction = '0;
    dequant_row_instruction[0 +: 16] = 16'h3f00;
    for (integer column = 0; column < 8; column++) begin
      candidate_instruction[8:6] = column[2:0];
      load_row_instruction = '0;
      load_row_instruction[0 +: 48] = candidate_instruction;
      load_row_valid = 4'b0001;
      dequant_row_valid = 4'b0001;
      #1ps;
      if (!load_east_consumed[0] || load_east_consumed[15:1] != '0)
        $fatal(1, "Column INT8 dequant consumed the wrong streams");
      @(posedge clk);
      #1ps;
      if (load_fault || weight_bits[column*16 +: 16] != 16'hc000 ||
          (cell_valid[0] != (column == 7)))
        $fatal(1, "Column INT8 dequant failed at column %0d", column);
      @(negedge clk);
    end
    load_row_valid = '0;
    dequant_row_valid = '0;

    candidate_instruction = '0;
    expect_load_fault(candidate_instruction, "INT8 IW without Dequant");

    candidate_instruction = '0;
    candidate_instruction[9] = 1'b1;
    reset_run_state();
    load_row_instruction = '0;
    load_row_instruction[0 +: 48] = candidate_instruction;
    load_row_valid = 4'b0001;
    dequant_row_valid = 4'b0001;
    @(posedge clk);
    #1ps;
    if (!load_fault || load_east_consumed != '0)
      $fatal(1, "Direct16 IW accepted an accompanying Dequant");

    candidate_instruction = supported_compute_instruction();
    candidate_instruction[46] = 1'b1;
    reset_run_state();
    compute_row_instruction = '0;
    compute_row_instruction[0 +: 48] = candidate_instruction;
    compute_row_valid = 4'b0001;
    #1ps;
    if (compute_east_consumed[15:0] != '1 ||
        compute_east_consumed[31:16] != '0)
      $fatal(1, "supported Block8 consumed the wrong activation streams");
    @(posedge clk);
    #1ps;
    if (compute_fault || !result_block_mode)
      $fatal(1, "supported Block8 compute baseline failed");

    candidate_instruction = supported_compute_instruction();
    expect_compute_fault(candidate_instruction, 4'b0011, "overlapping waves");

    reset_run_state();
    $display("FTLPU-VMODEL MXM unsupported-mode fault test passed");
    $finish;
  end
endmodule
