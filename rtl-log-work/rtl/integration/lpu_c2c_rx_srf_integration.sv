`timescale 1ns/1ps

// Thin RX connection wrapper. The source respects FIFO capacity; the external
// Receive-command queue holds its head until rx_cmd_pop_o. No TX metadata port.
module lpu_c2c_rx_srf_integration #(
  parameter integer FIFO_DEPTH = 2,
  parameter integer COLUMNS = 16,
  parameter integer SUPERLANES = 4,
  parameter integer STREAMS = 32,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer C2C_PRODUCER_SLOT = 0
) (
  input logic clk_i,
  input logic rst_ni,
  input logic peer_vector_valid_i,
  input logic [255:0] peer_vector_payload_i,
  input logic rx_cmd_valid_i,
  input logic [2:0] rx_cmd_stream_index_i,
  output logic rx_cmd_pop_o,
  output logic rx_ready_full_o,
  output logic rx_ready_empty_o,
  output logic [$clog2(FIFO_DEPTH+1)-1:0] rx_ready_count_o,
  output logic credit_return_o,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS-1:0]
    srf_inject_valid_o,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS*64-1:0]
    srf_inject_data_o
);
  logic ready_valid, ready_pop;
  logic [255:0] ready_payload;
  logic pair_valid;
  logic [255:0] pair_payload;
  logic [4:0] pair_stream;
  logic [3:0] replay_inject_valid;
  logic [255:0] replay_inject_data;
  logic [19:0] replay_inject_stream;

  c2c_rx_ready_fifo #(.DEPTH(FIFO_DEPTH)) u_ready_fifo (
    .clk_i, .rst_ni, .enq_valid_i(peer_vector_valid_i),
    .enq_payload_i(peer_vector_payload_i), .deq_pop_i(ready_pop),
    .deq_valid_o(ready_valid), .deq_payload_o(ready_payload),
    .full_o(rx_ready_full_o), .empty_o(rx_ready_empty_o),
    .count_o(rx_ready_count_o)
  );
  c2c_rx_issue_pair u_pair (
    .rx_cmd_valid_i, .rx_cmd_stream_index_i, .rx_cmd_pop_o,
    .ready_valid_i(ready_valid), .ready_payload_i(ready_payload),
    .ready_pop_o(ready_pop), .replay_valid_o(pair_valid),
    .replay_payload_o(pair_payload), .replay_stream_idx_o(pair_stream)
  );
  // A complete vector releases credit exactly when Receive pairing pops the
  // RX-ready FIFO; replay completion must not delay this return.
  assign credit_return_o = ready_pop;
  c2c_rx_replay u_replay (
    .clk_i, .rst_ni, .vector_valid_i(pair_valid),
    .vector_payload_i(pair_payload), .vector_stream_idx_i(pair_stream),
    .inject_valid_o(replay_inject_valid), .inject_data_o(replay_inject_data),
    .inject_stream_idx_o(replay_inject_stream)
  );
  lpu_c2c_rx_srf_adapter #(
    .COLUMNS(COLUMNS), .SUPERLANES(SUPERLANES), .STREAMS(STREAMS),
    .LOCAL_PRODUCERS(LOCAL_PRODUCERS), .C2C_PRODUCER_SLOT(C2C_PRODUCER_SLOT)
  ) u_adapter (
    .replay_inject_valid_i(replay_inject_valid),
    .replay_inject_data_i(replay_inject_data),
    .replay_inject_stream_idx_i(replay_inject_stream),
    .srf_inject_valid_o, .srf_inject_data_o
  );

`ifndef SYNTHESIS
  // Diagnostic only: do not gate/retry the source or implement link credit.
  always @(posedge clk_i) begin
    if (rst_ni && peer_vector_valid_i && rx_ready_full_o && !ready_pop)
      $fatal(1, "TEST_FAIL C2C RX-ready FIFO overflow in integration schedule");
  end
`endif
endmodule
