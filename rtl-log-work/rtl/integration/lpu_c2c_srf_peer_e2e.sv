`timescale 1ns/1ps

// One-way peer attachment between independent source/destination hemispheres.
// This wrapper adds no state. Receive, not Send, selects the destination stream.
module lpu_c2c_srf_peer_e2e #(
  parameter integer COLUMNS = 16,
  parameter integer SUPERLANES = 4,
  parameter integer STREAMS = 32,
  parameter integer LOCAL_CONSUMERS = 2,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer C2C_CONSUMER_SLOT = 0,
  parameter integer C2C_PRODUCER_SLOT = 0,
  parameter integer TX_FIFO_DEPTH = 2,
  parameter integer RX_FIFO_DEPTH = 2,
  parameter integer LINK_LATENCY = 1,
  parameter integer P_VECTOR_CREDITS = 4,
  parameter integer D_LINK_SERIALIZATION_CYCLES = 1
) (
  input logic clk_i,
  input logic rst_ni,
  input logic tx_issue_valid_i,
  input logic [2:0] tx_stream_index_i,
  input logic rx_cmd_valid_i,
  input logic [2:0] rx_cmd_stream_index_i,
  output logic rx_cmd_pop_o,
  input logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] source_state_valid_i,
  input logic [2*COLUMNS*SUPERLANES*STREAMS*64-1:0] source_state_data_i,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_CONSUMERS*STREAMS-1:0]
    source_consume_o,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS-1:0]
    destination_inject_valid_o,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS*64-1:0]
    destination_inject_data_o,
  output logic rx_ready_full_o,
  output logic rx_ready_empty_o,
  output logic [$clog2(RX_FIFO_DEPTH+1)-1:0] rx_ready_count_o
);
  logic [255:0] tx_tile_data;
  logic [3:0] tx_tile_valid, tx_tile_consume;
  logic peer_rx_valid;
  logic [255:0] peer_rx_payload;
  logic rx_credit_return;

  lpu_c2c_tx_srf_adapter #(
    .COLUMNS(COLUMNS), .SUPERLANES(SUPERLANES), .STREAMS(STREAMS),
    .SEGMENT_BITS(64), .LOCAL_CONSUMERS(LOCAL_CONSUMERS),
    .C2C_CONSUMER_SLOT(C2C_CONSUMER_SLOT), .TX_COLUMN(13)
  ) u_tx_adapter (
    .clk_i, .rst_ni, .tx_issue_valid_i, .tx_stream_index_i,
    .srf_state_valid_i(source_state_valid_i), .srf_state_data_i(source_state_data_i),
    .tile_data_o(tx_tile_data), .tile_valid_o(tx_tile_valid), .tile_stream_idx_o(),
    .gather_tile_consume_i(tx_tile_consume),
    .srf_consume_o(source_consume_o)
  );
  lpu_c2c_tx_peer_path #(
    .FIFO_DEPTH(TX_FIFO_DEPTH), .LINK_LATENCY(LINK_LATENCY),
    .P_VECTOR_CREDITS(P_VECTOR_CREDITS),
    .D_LINK_SERIALIZATION_CYCLES(D_LINK_SERIALIZATION_CYCLES)
  ) u_tx (
    .clk_i, .rst_ni, .tile_data_i(tx_tile_data), .tile_valid_i(tx_tile_valid),
    .tile_consume_o(tx_tile_consume), .credit_return_i(rx_credit_return),
    .peer_rx_valid_o(peer_rx_valid), .peer_rx_payload_o(peer_rx_payload)
  );
  lpu_c2c_rx_srf_integration #(
    .FIFO_DEPTH(RX_FIFO_DEPTH), .COLUMNS(COLUMNS), .SUPERLANES(SUPERLANES),
    .STREAMS(STREAMS), .LOCAL_PRODUCERS(LOCAL_PRODUCERS),
    .C2C_PRODUCER_SLOT(C2C_PRODUCER_SLOT)
  ) u_rx (
    .clk_i, .rst_ni, .peer_vector_valid_i(peer_rx_valid),
    .peer_vector_payload_i(peer_rx_payload),
    .rx_cmd_valid_i, .rx_cmd_stream_index_i, .rx_cmd_pop_o,
    .rx_ready_full_o, .rx_ready_empty_o, .rx_ready_count_o,
    .credit_return_o(rx_credit_return),
    .srf_inject_valid_o(destination_inject_valid_o),
    .srf_inject_data_o(destination_inject_data_o)
  );
endmodule
