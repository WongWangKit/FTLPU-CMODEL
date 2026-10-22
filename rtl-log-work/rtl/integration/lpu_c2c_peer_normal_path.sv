`timescale 1ns/1ps

// Test-only payload utility, not the architectural Receive/FIFO endpoint.
// Selected TX segments -> peer payload -> RX candidates. The testbench must
// supply rx_stream_idx_i with the peer payload and hold it through capture.
// No SRF attachment, command pairing, slot mapping, or wrapper state.
module lpu_c2c_peer_normal_path #(
  parameter integer FIFO_DEPTH = 2,
  parameter integer LINK_LATENCY = 1,
  parameter integer P_VECTOR_CREDITS = 4
) (
  input  logic clk_i,
  input  logic rst_ni,
  // Packed tile t uses data[t*64 +: 64]. TX has no stream metadata.
  input  logic [255:0] tx_tile_data_i,
  input  logic [3:0] tx_tile_valid_i,
  // Explicit test-only destination control; generic 5-bit Replay index.
  input  logic [4:0] rx_stream_idx_i,
  output logic [3:0] tx_tile_consume_o,
  output logic [3:0] rx_inject_valid_o,
  output logic [255:0] rx_inject_data_o,
  // Independent per-tile routing: stream_idx[t*5 +: 5].
  output logic [19:0] rx_inject_stream_idx_o
);
  logic peer_rx_valid;
  logic [255:0] peer_rx_payload;

  lpu_c2c_tx_peer_path #(
    .FIFO_DEPTH(FIFO_DEPTH), .LINK_LATENCY(LINK_LATENCY),
    .P_VECTOR_CREDITS(P_VECTOR_CREDITS)
  ) u_tx (
    .clk_i, .rst_ni,
    .tile_data_i(tx_tile_data_i), .tile_valid_i(tx_tile_valid_i),
    .tile_consume_o(tx_tile_consume_o),
    // This test-only utility has no RX-ready FIFO endpoint. It never invents
    // an architectural credit return; its test config supplies sufficient
    // static credits for each bounded schedule.
    .credit_return_i(1'b0),
    .peer_rx_valid_o(peer_rx_valid), .peer_rx_payload_o(peer_rx_payload)
  );

  c2c_rx_replay u_rx (
    .clk_i, .rst_ni,
    .vector_valid_i(peer_rx_valid), .vector_payload_i(peer_rx_payload),
    .vector_stream_idx_i(rx_stream_idx_i),
    .inject_valid_o(rx_inject_valid_o), .inject_data_o(rx_inject_data_o),
    .inject_stream_idx_o(rx_inject_stream_idx_o)
  );
endmodule
