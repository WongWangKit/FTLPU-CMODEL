`timescale 1ns/1ps

module lpu_arch_tb;
  timeunit 1ns;
  timeprecision 1ps;

  logic clk;
  logic rst_n;

  logic queue_run;
  logic enqueue_valid;
  logic enqueue_ready;
  logic enqueue_is_instruction;
  logic [31:0] enqueue_command;
  logic [415:0] enqueue_payload;
  logic issue_valid;
  logic [415:0] issue_payload;
  logic queue_fault;
  logic [3:0] queue_level;

  logic mem_instruction_valid;
  logic [46:0] mem_instruction;
  logic [63:0] mem_rx_valid;
  logic [4095:0] mem_rx_data;
  logic [63:0] mem_rx_consume;
  logic mem_tx_valid;
  logic [5:0] mem_tx_stream;
  logic [63:0] mem_tx_data;
  logic mem_tx_last;
  logic mem_fault;
  logic host_write_valid;
  logic [15:0] host_address;
  logic [63:0] host_write_data;
  logic [63:0] host_read_data;

  lpu_icu_queue #(
    .PAYLOAD_WIDTH(416),
    .DEPTH(8),
    .APPLY_MEM_STRIDE(1'b1)
  ) u_queue (
    .clk_i(clk),
    .rst_ni(rst_n),
    .run_i(queue_run),
    .enqueue_valid_i(enqueue_valid),
    .enqueue_ready_o(enqueue_ready),
    .enqueue_is_instruction_i(enqueue_is_instruction),
    .enqueue_command_i(enqueue_command),
    .enqueue_payload_i(enqueue_payload),
    .issue_valid_o(issue_valid),
    .issue_payload_o(issue_payload),
    .fault_o(queue_fault),
    .level_o(queue_level),
    .idle_o()
  );

  lpu_mem_tile_slice #(.DEPTH_ROWS(32)) u_mem (
    .clk_i(clk),
    .rst_ni(rst_n),
    .instruction_valid_i(mem_instruction_valid),
    .instruction_i(mem_instruction),
    .rx_valid_i(mem_rx_valid),
    .rx_data_i(mem_rx_data),
    .rx_consume_o(mem_rx_consume),
    .tx_valid_o(mem_tx_valid),
    .tx_stream_o(mem_tx_stream),
    .tx_data_o(mem_tx_data),
    .tx_last_o(mem_tx_last),
    .fault_o(mem_fault),
    .host_write_valid_i(host_write_valid),
    .host_address_i(host_address),
    .host_write_data_i(host_write_data),
    .host_read_data_o(host_read_data)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  task automatic enqueue_instruction(input logic [46:0] word);
    begin
      @(negedge clk);
      enqueue_valid = 1'b1;
      enqueue_is_instruction = 1'b1;
      enqueue_payload = '0;
      enqueue_payload[46:0] = word;
      @(negedge clk);
      enqueue_valid = 1'b0;
    end
  endtask

  task automatic enqueue_icu_command(input logic [31:0] command);
    begin
      @(negedge clk);
      enqueue_valid = 1'b1;
      enqueue_is_instruction = 1'b0;
      enqueue_command = command;
      @(negedge clk);
      enqueue_valid = 1'b0;
    end
  endtask

  task automatic expect_queue_issue(
    input logic expected_valid,
    input logic [15:0] expected_address
  );
    begin
      @(posedge clk);
      #1ns;
      if (issue_valid !== expected_valid)
        $fatal(1, "queue issue_valid mismatch");
      if (expected_valid && issue_payload[30:15] !== expected_address)
        $fatal(1, "queue address mismatch: got %0d expected %0d",
               issue_payload[30:15], expected_address);
    end
  endtask

  initial begin
    rst_n = 1'b0;
    queue_run = 1'b0;
    enqueue_valid = 1'b0;
    enqueue_is_instruction = 1'b0;
    enqueue_command = '0;
    enqueue_payload = '0;
    mem_instruction_valid = 1'b0;
    mem_instruction = '0;
    mem_rx_valid = '0;
    mem_rx_data = '0;
    host_write_valid = 1'b0;
    host_address = '0;
    host_write_data = '0;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    // Read row 10 on packed stream E3.
    enqueue_instruction((47'(10) << 15) | (47'(3) << 3));
    // NOP 2, then repeat twice at II=2 with row stride +3.
    enqueue_icu_command((32'(2) << 2) | 32'd1);
    enqueue_icu_command((32'(3) << 20) | (32'(2) << 12) |
                        (32'(2) << 2) | 32'd2);

    queue_run = 1'b1;
    expect_queue_issue(1'b1, 16'd10);
    expect_queue_issue(1'b0, '0);
    expect_queue_issue(1'b0, '0);
    expect_queue_issue(1'b0, '0);
    expect_queue_issue(1'b1, 16'd13);
    expect_queue_issue(1'b0, '0);
    expect_queue_issue(1'b1, 16'd16);
    if (queue_fault)
      $fatal(1, "ICU queue raised an unexpected fault");

    // Refill an empty queue while execution remains active.  Full workloads
    // stream schedules this way instead of sizing every queue for a layer.
    enqueue_instruction((47'(19) << 15) | (47'(3) << 3));
    expect_queue_issue(1'b1, 16'd19);
    if (queue_fault)
      $fatal(1, "streaming ICU refill raised an unexpected fault");

    // Host initialization followed by an architectural MEM Read.
    queue_run = 1'b0;
    @(negedge clk);
    host_address = 16'd5;
    host_write_data = 64'h8877_6655_4433_2211;
    host_write_valid = 1'b1;
    @(negedge clk);
    host_write_valid = 1'b0;
    mem_instruction = (47'(5) << 15) | (47'(7) << 3);
    mem_instruction_valid = 1'b1;
    @(posedge clk);
    #1ns;
    mem_instruction_valid = 1'b0;
    if (!mem_tx_valid || mem_tx_stream != 6'd7 ||
        mem_tx_data != 64'h8877_6655_4433_2211 || !mem_tx_last)
      $fatal(1, "MEM Read result mismatch");

    // Architectural MEM Write from packed stream W3 (selector 35).
    @(negedge clk);
    mem_rx_valid[35] = 1'b1;
    mem_rx_data[35*64 +: 64] = 64'h0123_4567_89ab_cdef;
    mem_instruction = (47'(6) << 15) | (47'(35) << 3) | 47'd1;
    mem_instruction_valid = 1'b1;
    @(posedge clk);
    #1ns;
    mem_instruction_valid = 1'b0;
    mem_rx_valid = '0;
    host_address = 16'd6;
    #1ns;
    if (host_read_data != 64'h0123_4567_89ab_cdef || mem_fault)
      $fatal(1, "MEM Write result mismatch");

    $display("FTLPU-VMODEL architecture test passed");
    $finish;
  end
endmodule
