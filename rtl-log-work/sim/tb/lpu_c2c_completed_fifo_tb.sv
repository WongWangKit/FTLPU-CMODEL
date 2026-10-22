`timescale 1ns/1ps

// Run the same independent reference-queue checks at two physical depths.
module c2c_completed_fifo_checker #(
  parameter integer DEPTH = 2
) (
  input logic clk_i,
  output logic done_o
);
  localparam integer COUNT_BITS = $clog2(DEPTH+1);
  logic rst_ni;
  logic enq_valid_i, deq_pop_i;
  logic [255:0] enq_payload_i, deq_payload_o;
  logic deq_valid_o, full_o, empty_o;
  logic [COUNT_BITS-1:0] count_o;
  // A shifting reference queue intentionally does not model DUT pointers.
  logic [255:0] expected [0:DEPTH-1];
  integer expected_count, cycle_id;

  c2c_completed_fifo #(.DEPTH(DEPTH)) dut (
    .clk_i, .rst_ni, .enq_valid_i, .enq_payload_i,
    .deq_pop_i, .deq_valid_o, .deq_payload_o,
    .full_o, .empty_o, .count_o
  );

  function automatic [255:0] make_entry(input integer id);
    for (integer b = 0; b < 32; b = b + 1)
      make_entry[b*8 +: 8] = 8'(id*67 + b*3 + 1);
  endfunction

  task automatic check_head_status;
    if (count_o !== COUNT_BITS'(expected_count) ||
        empty_o !== (expected_count == 0) ||
        full_o !== (expected_count == DEPTH) ||
        deq_valid_o !== (expected_count != 0))
      $fatal(1, "TEST_FAIL depth=%0d cycle=%0d count=%0d expected=%0d empty=%b full=%b valid=%b",
             DEPTH, cycle_id, count_o, expected_count, empty_o, full_o, deq_valid_o);
    if (expected_count != 0) begin
      if (deq_payload_o !== expected[0])
        $fatal(1, "TEST_FAIL depth=%0d cycle=%0d head got=%h expected=%h",
               DEPTH, cycle_id, deq_payload_o, expected[0]);
    end
  endtask

  task automatic step(
    input logic push, input logic pop, input integer id
  );
    logic accepted_pop, accepted_push;
    logic [255:0] new_entry;
    begin
      @(negedge clk_i);
      new_entry = make_entry(id);
      enq_valid_i = push;
      deq_pop_i = pop;
      enq_payload_i = new_entry;
      #1;
      // In particular, empty+push+pop must still be invalid BEFORE the edge.
      check_head_status();
      accepted_pop = pop && (expected_count != 0);
      accepted_push = push && ((expected_count < DEPTH) || accepted_pop);
      @(posedge clk_i);
      // The actual consumer sees this pre-NBA head, not the new memory data.
      check_head_status();
      if (accepted_pop) begin
        for (integer i = 0; i < expected_count-1; i = i + 1)
          expected[i] = expected[i+1];
        expected_count = expected_count - 1;
      end
      if (accepted_push) begin
        expected[expected_count] = new_entry;
        expected_count = expected_count + 1;
      end
      #1;
      cycle_id = cycle_id + 1;
      check_head_status();
    end
  endtask

  task automatic reset_fifo;
    @(negedge clk_i);
    rst_ni = 1'b0;
    // Neither request may update FIFO control while reset is active.
    enq_valid_i = 1'b1;
    deq_pop_i = 1'b1;
    enq_payload_i = make_entry(99);
    expected_count = 0;
    #1;
    check_head_status();
    if (dut.read_ptr_q !== '0 || dut.write_ptr_q !== '0)
      $fatal(1, "TEST_FAIL depth=%0d reset pointers", DEPTH);
    repeat (2) @(posedge clk_i);
    #1;
    check_head_status();
    @(negedge clk_i);
    enq_valid_i = 1'b0;
    deq_pop_i = 1'b0;
    rst_ni = 1'b1;
    #1;
    check_head_status();
  endtask

  initial begin
    done_o = 1'b0;
    rst_ni = 1'b0;
    enq_valid_i = 1'b0;
    deq_pop_i = 1'b0;
    enq_payload_i = '0;
    expected_count = 0;
    cycle_id = 0;
    reset_fifo();

    step(1, 0, 0);
    step(1, 0, 1);
    step(0, 1, 0);
    step(0, 1, 0);
    if (DEPTH == 2) $display("C2C_COMPLETED_FIFO_ORDER PASS");

    for (integer i = 0; i < DEPTH; i = i + 1) step(1, 0, 10+i);
    step(0, 0, 0);  // Stable head while full, without pop.
    step(1, 0, 99); // Rejected overflow must not overwrite any queued entry.
    for (integer i = 0; i < DEPTH; i = i + 1) step(0, 1, 0);
    step(0, 1, 0);  // Empty pop ignored.
    step(1, 1, 42); // Empty push+pop stores, but does not consume, entry 42.
    if (expected_count != 1) $fatal(1, "TEST_FAIL empty fall-through");
    step(0, 1, 0);
    if (DEPTH == 2) $display("C2C_COMPLETED_FIFO_STATUS PASS");

    for (integer i = 0; i < DEPTH; i = i + 1) step(1, 0, i);
    step(1, 1, DEPTH); // [V0,V1] -> [V1,V2] at depth 2, remaining full.
    if (expected_count != DEPTH) $fatal(1, "TEST_FAIL full pop+push count");
    for (integer i = 0; i < DEPTH; i = i + 1) step(0, 1, 0);
    if (DEPTH == 2) $display("C2C_COMPLETED_FIFO_SIMULTANEOUS PASS");

    step(1, 0, 0);
    for (integer i = 1; i <= 8; i = i + 1) begin
      step(1, 1, i);
      if (expected_count != 1) $fatal(1, "TEST_FAIL II1 occupancy");
    end
    step(0, 1, 0);
    if (DEPTH == 2) $display("C2C_COMPLETED_FIFO_II1 PASS");

    for (integer i = 0; i < DEPTH; i = i + 1) step(1, 0, i);
    reset_fifo();
    step(0, 0, 0);
    step(0, 1, 0);
    step(1, 0, 2);
    step(0, 1, 0);
    if (DEPTH == 2) $display("C2C_COMPLETED_FIFO_RESET PASS");
    else $display("C2C_COMPLETED_FIFO_DEPTH3 PASS");
    done_o = 1'b1;
  end
endmodule

module lpu_c2c_completed_fifo_tb;
  logic clk_i = 1'b0;
  logic depth2_done, depth3_done;
  always #5 clk_i = ~clk_i;

  c2c_completed_fifo_checker #(.DEPTH(2)) depth2 (
    .clk_i, .done_o(depth2_done)
  );
  c2c_completed_fifo_checker #(.DEPTH(3)) depth3 (
    .clk_i, .done_o(depth3_done)
  );

  initial begin
    wait (depth2_done && depth3_done);
    $display("C2C_COMPLETED_FIFO TEST_PASS");
    $finish;
  end

  initial begin
    #10000;
    $fatal(1, "TEST_FAIL simulation timeout");
  end
endmodule
