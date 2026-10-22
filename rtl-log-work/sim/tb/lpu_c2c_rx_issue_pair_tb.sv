`timescale 1ns/1ps

// Run the same queue/pairing checks at power-of-two and non-power-of-two depth.
module c2c_rx_issue_pair_check #(
  parameter integer DEPTH = 2
) (
  input logic clk,
  output logic done
);
  logic rst_n;
  logic enq_valid;
  logic [255:0] enq_payload;
  logic ready_valid, ready_pop, full, empty;
  logic [255:0] ready_payload;
  logic [$clog2(DEPTH+1)-1:0] count;
  logic cmd_valid, cmd_pop;
  logic [2:0] cmd_stream;
  logic replay_valid;
  logic [255:0] replay_payload;
  logic [4:0] replay_stream;

  // External command-head stimulus only, not an ICU or command FIFO DUT.
  // The test holds a command stable during early arrival until its pop edge.
  integer command_streams [0:3];
  logic [4:0] distractor_tx_stream;
  logic check_override;
  integer override_checks;

  // Independent linear queue scoreboard, deliberately not circular pointers.
  logic [255:0] expected_queue [0:DEPTH-1];
  integer expected_count;
  integer pairs;
  integer cycle_number;
  integer last_pair_cycle;
  integer i;

  c2c_rx_ready_fifo #(.DEPTH(DEPTH)) u_fifo (
    .clk_i(clk), .rst_ni(rst_n),
    .enq_valid_i(enq_valid), .enq_payload_i(enq_payload),
    .deq_pop_i(ready_pop), .deq_valid_o(ready_valid),
    .deq_payload_o(ready_payload), .full_o(full), .empty_o(empty),
    .count_o(count)
  );

  c2c_rx_issue_pair u_pair (
    .rx_cmd_valid_i(cmd_valid), .rx_cmd_stream_index_i(cmd_stream),
    .rx_cmd_pop_o(cmd_pop), .ready_valid_i(ready_valid),
    .ready_payload_i(ready_payload), .ready_pop_o(ready_pop),
    .replay_valid_o(replay_valid), .replay_payload_o(replay_payload),
    .replay_stream_idx_o(replay_stream)
  );

  function automatic [255:0] payload(input integer id);
    for (integer byte_index = 0; byte_index < 32; byte_index = byte_index + 1)
      payload[byte_index*8 +: 8] = 8'(id*37 + byte_index);
  endfunction

  task automatic fail(input string message);
    $fatal(1, "TEST_FAIL DEPTH=%0d cycle=%0d %s", DEPTH, cycle_number, message);
  endtask

  // Verify head status and the combinational transaction before AND after
  // every clock. When invalid, payload is not architecturally significant.
  task automatic check_outputs;
    logic expected_pair;
    begin
      expected_pair = cmd_valid && (expected_count != 0);
      if (count !== expected_count || empty !== (expected_count == 0) ||
          full !== (expected_count == DEPTH) ||
          ready_valid !== (expected_count != 0))
        fail("FIFO count/full/empty/head-valid mismatch");
      if (expected_count != 0 && ready_payload !== expected_queue[0])
        fail("FIFO payload head ordering mismatch");
      if (cmd_pop !== expected_pair || ready_pop !== expected_pair ||
          replay_valid !== expected_pair)
        fail("Command pop, vector pop and Replay valid must be atomic");
      if (replay_stream !== {2'b00, cmd_stream})
        fail("Receive selector was not zero-extended to five bits");
      if (expected_pair && replay_payload !== expected_queue[0])
        fail("Replay payload did not match the current FIFO head");
    end
  endtask

  // Inputs remain stable through the sampling edge. Enqueue into an empty
  // queue does not create a pre-edge pair; its new head is visible afterward.
  task automatic tick(
    input logic push, input integer vector_id,
    input logic command_valid, input integer receive_stream
  );
    logic pop_expected;
    begin
      @(negedge clk);
      enq_valid = push;
      enq_payload = payload(vector_id);
      cmd_valid = command_valid;
      cmd_stream = receive_stream[2:0];
      #1;
      check_outputs();
      pop_expected = command_valid && (expected_count != 0);
      if (push && expected_count == DEPTH && !pop_expected)
        fail("Test stimulus attempted an illegal FIFO overflow");
      if (check_override && pop_expected) begin
        if (distractor_tx_stream !== 5'd3 || replay_stream !== 5'd6 ||
            replay_stream === distractor_tx_stream)
          fail("TX source metadata affected RX destination");
        override_checks = override_checks + 1;
      end
      @(posedge clk);
      cycle_number = cycle_number + 1;
      if (pop_expected) begin
        for (integer j = 0; j < expected_count-1; j = j + 1)
          expected_queue[j] = expected_queue[j+1];
        expected_count = expected_count - 1;
        pairs = pairs + 1;
        last_pair_cycle = cycle_number;
      end
      if (push) begin
        expected_queue[expected_count] = payload(vector_id);
        expected_count = expected_count + 1;
      end
      #1;
      check_outputs();
    end
  endtask

  task automatic reset_case(input logic waiting_command);
    begin
      @(negedge clk);
      enq_valid = 1'b0;
      enq_payload = '0;
      cmd_valid = waiting_command;
      cmd_stream = 3'd7;
      rst_n = 1'b0;
      expected_count = 0;
      pairs = 0;
      last_pair_cycle = -1;
      #1;
      check_outputs();
      repeat (2) @(posedge clk);
      @(negedge clk);
      rst_n = 1'b1;
      #1;
      check_outputs();
    end
  endtask

  initial begin
    done = 1'b0;
    rst_n = 1'b0;
    enq_valid = 1'b0;
    enq_payload = '0;
    cmd_valid = 1'b0;
    cmd_stream = '0;
    expected_count = 0;
    pairs = 0;
    cycle_number = 0;
    last_pair_cycle = -1;
    check_override = 1'b0;
    override_checks = 0;
    distractor_tx_stream = 5'd3; // Intentionally not connected to either DUT.
    command_streams[0] = 3;
    command_streams[1] = 6;
    command_streams[2] = 1;
    command_streams[3] = 7;

    // Basic: V0 and Receive W6 produce one simultaneous pair/pop.
    reset_case(1'b0);
    tick(1, 0, 0, 0);
    tick(0, 0, 1, 6);
    tick(0, 0, 0, 0);
    if (pairs != 1) fail("Basic transaction was lost or duplicated");
    if (DEPTH == 2) $display("C2C_RX_PAIR_BASIC PASS");

    // Receive W3 waits at the external command head while the FIFO is empty.
    reset_case(1'b0);
    repeat (3) tick(0, 0, 1, 3);
    if (pairs != 0) fail("Early command was consumed without a vector");
    tick(1, 1, 1, 3);
    if (pairs != 0) fail("Empty enqueue incorrectly fell through");
    tick(0, 0, 1, 3);
    tick(0, 0, 0, 0);
    if (pairs != 1) fail("Early command paired more or less than once");
    if (DEPTH == 2) $display("C2C_RX_PAIR_COMMAND_FIRST PASS");

    // A complete vector waits unmodified until Receive W7 becomes available.
    reset_case(1'b0);
    tick(1, 2, 0, 0);
    repeat (3) tick(0, 0, 0, 0);
    if (pairs != 0 || expected_count != 1)
      fail("Early vector was not retained");
    tick(0, 0, 1, 7);
    tick(0, 0, 0, 0);
    if (pairs != 1) fail("Early vector was lost or duplicated");
    if (DEPTH == 2) $display("C2C_RX_PAIR_VECTOR_FIRST PASS");

    // Prime the FIFO, then stream four ordered pairs without a bubble. The
    // external command source advances to the next entry only after pop.
    reset_case(1'b0);
    for (i = 0; i < DEPTH; i = i + 1) tick(1, i, 0, 0);
    for (i = 0; i < 4; i = i + 1) begin
      if (ready_payload !== payload(i)) fail("Expected V0,V1,V2,V3 order");
      tick((i + DEPTH < 4), i + DEPTH, 1, command_streams[i]);
      if (pairs != i+1 || last_pair_cycle != cycle_number)
        fail("Expected one pair per cycle during II=1 window");
    end
    tick(0, 0, 0, 0);
    if (pairs != 4 || expected_count != 0)
      fail("Ordered pairing did not drain exactly four vectors");
    if (DEPTH == 2) begin
      $display("C2C_RX_PAIR_ORDER PASS");
      $display("C2C_RX_PAIR_II1 PASS");
    end

    // Receive W6 overrides the unconnected distractor TX source stream E3.
    reset_case(1'b0);
    tick(1, 20, 0, 0);
    check_override = 1'b1;
    tick(0, 0, 1, 6);
    check_override = 1'b0;
    if (override_checks != 1 || pairs != 1)
      fail("Stream independence check did not exercise a real pair");
    if (DEPTH == 2) $display("C2C_RX_PAIR_STREAM_OVERRIDE PASS");

    // Status, empty enqueue with waiting command, full push+pop, and repeated
    // pointer wrap. Occupancy stays DEPTH throughout the full streaming window.
    reset_case(1'b0);
    tick(0, 0, 1, 0);
    tick(1, 30, 1, 0);
    if (pairs != 0) fail("RX FIFO has unwanted empty fall-through");
    tick(0, 0, 1, 0);
    for (i = 0; i < DEPTH; i = i + 1) tick(1, 40+i, 0, 0);
    repeat (3) tick(0, 0, 0, 0);
    for (i = 0; i < 12; i = i + 1) begin
      tick(1, 40+DEPTH+i, 1, i%8);
      if (expected_count != DEPTH || full !== 1'b1)
        fail("Full simultaneous pop/push changed occupancy");
    end
    for (i = 0; i < DEPTH; i = i + 1) tick(0, 0, 1, i%8);
    tick(0, 0, 0, 0);
    if (pairs != 13+DEPTH || expected_count != 0)
      fail("Wrap/status test lost or duplicated a vector");
    if (DEPTH == 2) $display("C2C_RX_READY_FIFO PASS");

    // Reset nonempty storage while a Receive command is present. No old data
    // may pair after reset; the waiting command must pair only with Vnew.
    tick(1, 90, 0, 0);
    tick(1, 91, 0, 0);
    if (count != 2) fail("Reset test did not populate the FIFO");
    reset_case(1'b1);
    repeat (2) tick(0, 0, 1, 7);
    tick(1, 99, 1, 7);
    if (pairs != 0) fail("Stale vector survived reset");
    tick(0, 0, 1, 7);
    tick(0, 0, 0, 0);
    if (pairs != 1 || expected_count != 0)
      fail("Post-reset transaction failed");
    if (DEPTH == 2) $display("C2C_RX_PAIR_RESET PASS");
    if (DEPTH == 3) $display("C2C_RX_READY_FIFO_DEPTH3 PASS");
    done = 1'b1;
  end
endmodule

module lpu_c2c_rx_issue_pair_tb;
  logic clk = 1'b0;
  wire done2, done3;
  always #5 clk = ~clk;

  c2c_rx_issue_pair_check #(.DEPTH(2)) u_depth2 (.clk, .done(done2));
  c2c_rx_issue_pair_check #(.DEPTH(3)) u_depth3 (.clk, .done(done3));

  initial begin
    wait (done2 && done3);
    $display("========================================");
    $display("C2C_RX_ISSUE_PAIR TEST_PASS");
    $display("========================================");
    $finish;
  end

  initial begin
    #20000;
    $fatal(1, "TEST_FAIL RX pairing test timed out");
  end
endmodule
