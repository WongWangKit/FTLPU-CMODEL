`timescale 1ns/1ps

// Integration check for the only legal credit-return event: RX-ready pop.
module lpu_c2c_credit_rx_return_tb;
  localparam integer COLUMNS = 16, SUPERLANES = 4, STREAMS = 32, SLOTS = 2;
  localparam integer INJECT_BITS = 2*COLUMNS*SUPERLANES*SLOTS*STREAMS;
  logic clk_i = 1'b0, rst_ni;
  logic peer_vector_valid_i;
  logic [255:0] peer_vector_payload_i;
  logic rx_cmd_valid_i;
  logic [2:0] rx_cmd_stream_index_i;
  logic rx_cmd_pop_o, rx_ready_full_o, rx_ready_empty_o, credit_return_o;
  logic [1:0] rx_ready_count_o;
  logic [INJECT_BITS-1:0] srf_inject_valid_o;
  logic [INJECT_BITS*64-1:0] srf_inject_data_o;

  always #5 clk_i = ~clk_i;
  lpu_c2c_rx_srf_integration dut (.*);

  task automatic reset_rx;
    begin
      @(negedge clk_i);
      rst_ni=0; peer_vector_valid_i=0; peer_vector_payload_i='0;
      rx_cmd_valid_i=0; rx_cmd_stream_index_i=0;
      repeat(2) @(posedge clk_i);
      @(negedge clk_i); rst_ni=1;
    end
  endtask

  task automatic enqueue_vector(input [255:0] payload);
    begin
      @(negedge clk_i);
      peer_vector_valid_i=1; peer_vector_payload_i=payload;
      @(posedge clk_i); #1;
      peer_vector_valid_i=0;
    end
  endtask

  initial begin
    reset_rx();
    enqueue_vector(256'h1111);
    if (rx_ready_count_o != 2'd1 || credit_return_o || rx_cmd_pop_o)
      $fatal(1, "TEST_FAIL credit returned before RX-ready pop");
    @(negedge clk_i); rx_cmd_valid_i=1; rx_cmd_stream_index_i=3'd6;
    #1;
    if (!credit_return_o || !rx_cmd_pop_o)
      $fatal(1, "TEST_FAIL missing RX-ready pop credit return");
    @(posedge clk_i); #1;
    rx_cmd_valid_i=0;
    if (rx_ready_count_o != 0 || credit_return_o)
      $fatal(1, "TEST_FAIL RX credit return duration/count");
    $display("C2C_CREDIT_RX_POP_RETURN PASS");

    reset_rx();
    // A Receive command may wait first, but it cannot manufacture a return.
    @(negedge clk_i); rx_cmd_valid_i=1; rx_cmd_stream_index_i=3'd2;
    #1;
    if (credit_return_o || rx_cmd_pop_o)
      $fatal(1, "TEST_FAIL command-first premature credit return");
    enqueue_vector(256'h2222);
    // Pairing becomes visible after the enqueue edge and is sampled/pop'd at
    // the next edge; the return pulse therefore denotes that scheduled pop.
    if (rx_ready_count_o != 1 || !credit_return_o || !rx_cmd_pop_o)
      $fatal(1, "TEST_FAIL command-first enqueue/pair semantics");
    @(negedge clk_i); #1;
    if (!credit_return_o || !rx_cmd_pop_o)
      $fatal(1, "TEST_FAIL command-first missing paired credit return");
    @(posedge clk_i); #1;
    if (rx_ready_count_o != 0)
      $fatal(1, "TEST_FAIL command-first vector was not popped");
    $display("C2C_CREDIT_COMMAND_FIRST PASS");
    $display("C2C_CREDIT_RX_RETURN TEST_PASS");
    $finish;
  end
endmodule
