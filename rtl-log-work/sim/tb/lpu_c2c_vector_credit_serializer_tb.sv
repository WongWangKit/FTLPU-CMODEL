`timescale 1ns/1ps

module lpu_c2c_vector_credit_serializer_checker(output logic done_o);
  logic clk_i = 1'b0;
  logic rst_ni;
  logic tx_valid_i, credit_return_i, tx_pop_o, launch_valid_o;
  logic [255:0] tx_payload_i, launch_payload_o;
  logic [0:0] credit_count_o;
  logic serializer_busy_o, credit_error_o;
  integer launches;

  always #5 clk_i = ~clk_i;

  c2c_vector_credit_serializer #(
    .P_VECTOR_CREDITS(1), .D_LINK_SERIALIZATION_CYCLES(1)
  ) dut (
    .clk_i, .rst_ni, .tx_valid_i, .tx_payload_i, .tx_pop_o,
    .credit_return_i, .launch_valid_o, .launch_payload_o,
    .credit_count_o, .serializer_busy_o, .credit_error_o
  );

  task automatic drive(input logic valid, input [255:0] payload,
                       input logic credit_return);
    begin
      @(negedge clk_i);
      tx_valid_i = valid;
      tx_payload_i = payload;
      credit_return_i = credit_return;
    end
  endtask

  task automatic sample_launch(input logic expected_launch,
                               input logic [255:0] expected_payload,
                               input logic [0:0] expected_credit);
    begin
      #1;
      if (launch_valid_o !== expected_launch || tx_pop_o !== expected_launch)
        $fatal(1, "TEST_FAIL credit launch/pop mismatch");
      if (expected_launch && launch_payload_o !== expected_payload)
        $fatal(1, "TEST_FAIL credit payload mismatch");
      @(posedge clk_i);
      #1;
      if (credit_count_o !== expected_credit || credit_error_o)
        $fatal(1, "TEST_FAIL credit count/error mismatch");
      if (expected_launch) launches = launches + 1;
    end
  endtask

  initial begin
    done_o = 1'b0;
    rst_ni = 1'b0; tx_valid_i = 1'b0; tx_payload_i = '0;
    credit_return_i = 1'b0; launches = 0;
    repeat (2) @(posedge clk_i);
    @(negedge clk_i); rst_ni = 1'b1;

    drive(1'b1, 256'h01, 1'b0); sample_launch(1'b1, 256'h01, 1'd0);
    $display("C2C_CREDIT_BASIC PASS");

    // No credit is an ordinary FIFO-head wait, not a protocol error.
    drive(1'b1, 256'h02, 1'b0); sample_launch(1'b0, '0, 1'd0);
    $display("C2C_CREDIT_EXHAUST_WAIT PASS");

    drive(1'b1, 256'h02, 1'b1); sample_launch(1'b0, '0, 1'd1);
    drive(1'b1, 256'h02, 1'b0); sample_launch(1'b1, 256'h02, 1'd0);
    $display("C2C_CREDIT_RETURN PASS");

    // A legal same-edge launch and return leaves the count unchanged.
    @(negedge clk_i); rst_ni = 1'b0;
    @(posedge clk_i); @(negedge clk_i);
    rst_ni = 1'b1;
    tx_valid_i = 1'b0;
    credit_return_i = 1'b0;
    drive(1'b1, 256'h03, 1'b1); sample_launch(1'b1, 256'h03, 1'd1);
    $display("C2C_CREDIT_SIMULTANEOUS PASS");
    $display("C2C_CREDIT_LIFETIME PASS");
    $display("C2C_CREDIT_ZERO_GUARD PASS");
    if (launches != 3) $fatal(1, "TEST_FAIL unexpected launch count");
    done_o = 1'b1;
  end
endmodule

module lpu_c2c_serialization_delay_checker #(
  parameter integer D_LINK_SERIALIZATION_CYCLES = 3
) (output logic done_o);
  logic clk_i = 1'b0;
  logic rst_ni, tx_valid_i, credit_return_i, tx_pop_o, launch_valid_o;
  logic [255:0] tx_payload_i, launch_payload_o;
  logic [2:0] credit_count_o;
  logic serializer_busy_o, credit_error_o;
  integer vector_id, launch_count, edge_count, last_launch;

  always #5 clk_i = ~clk_i;
  c2c_vector_credit_serializer #(
    .P_VECTOR_CREDITS(4),
    .D_LINK_SERIALIZATION_CYCLES(D_LINK_SERIALIZATION_CYCLES)
  ) dut (.*);

  always @(negedge clk_i) begin
    tx_valid_i = (vector_id < 3);
    tx_payload_i = vector_id;
    credit_return_i = 1'b0;
  end
  always @(posedge clk_i) begin
    if (rst_ni && tx_pop_o) begin
      if (last_launch >= 0 &&
          edge_count != last_launch + D_LINK_SERIALIZATION_CYCLES)
        $fatal(1, "TEST_FAIL serializer spacing");
      last_launch = edge_count;
      vector_id = vector_id + 1;
      launch_count = launch_count + 1;
    end
    edge_count = edge_count + 1;
  end
  initial begin
    done_o=0; rst_ni=0; tx_valid_i=0; tx_payload_i='0; credit_return_i=0;
    vector_id=0; launch_count=0; edge_count=0; last_launch=-1;
    repeat(2) @(posedge clk_i);
    @(negedge clk_i); rst_ni=1;
    wait(launch_count == 3);
    #1;
    if (credit_count_o !== 3'd1 || credit_error_o)
      $fatal(1, "TEST_FAIL serialization credit accounting");
    if (D_LINK_SERIALIZATION_CYCLES == 3) begin
      $display("C2C_SERIALIZATION_DELAY PASS");
      $display("C2C_SERIALIZER_BUSY_GUARD PASS");
    end else begin
      $display("C2C_SERIALIZATION_II1 PASS");
    end
    done_o = 1'b1;
  end
endmodule

module lpu_c2c_vector_credit_serializer_tb;
  logic done_a, done_b, done_c;
  lpu_c2c_vector_credit_serializer_checker a(.done_o(done_a));
  lpu_c2c_serialization_delay_checker b(.done_o(done_b));
  lpu_c2c_serialization_delay_checker #(.D_LINK_SERIALIZATION_CYCLES(1))
    c(.done_o(done_c));
  initial begin
    wait (done_a && done_b && done_c);
    $display("C2C_VECTOR_CREDIT_SERIALIZER TEST_PASS");
    $finish;
  end
endmodule
