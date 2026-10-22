`timescale 1ns/1ps

// Stateless connection wrapper. Input tile t is [t*64 +: 64]; stream
// selection is external. Gather completion and the peer boundary are payload-only.
module lpu_c2c_tx_peer_path #(
  parameter integer FIFO_DEPTH = 2,
  parameter integer LINK_LATENCY = 1,
  parameter integer P_VECTOR_CREDITS = 4,
  parameter integer D_LINK_SERIALIZATION_CYCLES = 1
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic [255:0] tile_data_i,
  input  logic [3:0] tile_valid_i,
  output logic [3:0] tile_consume_o,
  input  logic credit_return_i,
  output logic peer_rx_valid_o,
  output logic [255:0] peer_rx_payload_o,
  output logic [$clog2(P_VECTOR_CREDITS+1)-1:0] credit_count_o,
  output logic serializer_busy_o,
  output logic credit_error_o
);
  logic gather_completed_valid;
  logic [255:0] gather_completed_payload;
  logic fifo_deq_valid, fifo_deq_pop;
  logic [255:0] fifo_deq_payload;
  logic fifo_full, fifo_empty, fifo_can_enqueue;
  logic [$clog2(FIFO_DEPTH+1)-1:0] fifo_count;

  c2c_tx_gather #(.SEGMENT_BITS(64)) u_gather (
    .clk_i, .rst_ni, .tile_data_i, .tile_valid_i,
    .tile_consume_o,
    .completed_valid_o(gather_completed_valid),
    .completed_payload_o(gather_completed_payload)
  );

  c2c_completed_fifo #(.DEPTH(FIFO_DEPTH)) u_fifo (
    .clk_i, .rst_ni,
    .enq_valid_i(gather_completed_valid),
    .enq_payload_i(gather_completed_payload),
    .deq_pop_i(fifo_deq_pop), .deq_valid_o(fifo_deq_valid),
    .deq_payload_o(fifo_deq_payload),
    .full_o(fifo_full), .empty_o(fifo_empty), .count_o(fifo_count)
  );

  // Credit and serialization only control FIFO-head launch.  They never
  // feed ready/stall/backpressure toward the ordinary SRF gather source.
  logic serializer_launch_valid;
  logic [255:0] serializer_launch_payload;

  c2c_vector_credit_serializer #(
    .P_VECTOR_CREDITS(P_VECTOR_CREDITS),
    .D_LINK_SERIALIZATION_CYCLES(D_LINK_SERIALIZATION_CYCLES)
  ) u_credit_serializer (
    .clk_i, .rst_ni,
    .tx_valid_i(fifo_deq_valid), .tx_payload_i(fifo_deq_payload),
    .tx_pop_o(fifo_deq_pop), .credit_return_i,
    .launch_valid_o(serializer_launch_valid),
    .launch_payload_o(serializer_launch_payload),
    .credit_count_o, .serializer_busy_o, .credit_error_o
  );

  // A full FIFO still has enqueue capacity when its head truly launches at
  // the same edge. Credit exhaustion alone is a normal internal wait.
  assign fifo_can_enqueue = !fifo_full || (fifo_deq_valid && fifo_deq_pop);

  c2c_peer_transport #(.LINK_LATENCY(LINK_LATENCY)) u_transport (
    .clk_i, .rst_ni,
    .tx_valid_i(serializer_launch_valid),
    .tx_payload_i(serializer_launch_payload),
    .rx_valid_o(peer_rx_valid_o), .rx_payload_o(peer_rx_payload_o)
  );

  // No gating of Gather and no hardware recovery. Legal schedules must not
  // overflow; report an illegal loss in simulation instead of hiding it.
`ifndef SYNTHESIS
  always @(posedge clk_i) begin
    if (rst_ni && gather_completed_valid && !fifo_can_enqueue)
      $fatal(1, "TEST_FAIL C2C completed FIFO enqueue capacity exceeded");
  end
`endif
endmodule
