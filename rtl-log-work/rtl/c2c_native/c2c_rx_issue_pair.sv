`timescale 1ns/1ps

// Stateless FIFO-head rendezvous. Both upstream queues retain their heads
// until pop; their heads are consumed together at the Replay sampling edge.
// This is the proposed minimal Receive/ready-vector pairing contract.
module c2c_rx_issue_pair (
  input  logic rx_cmd_valid_i,
  input  logic [2:0] rx_cmd_stream_index_i,
  output logic rx_cmd_pop_o,
  input  logic ready_valid_i,
  input  logic [255:0] ready_payload_i,
  output logic ready_pop_o,
  output logic replay_valid_o,
  output logic [255:0] replay_payload_o,
  output logic [4:0] replay_stream_idx_o
);
  logic pair_fire;

  assign pair_fire = rx_cmd_valid_i && ready_valid_i;
  assign rx_cmd_pop_o = pair_fire;
  assign ready_pop_o = pair_fire;
  assign replay_valid_o = pair_fire;
  assign replay_payload_o = ready_payload_i;
  assign replay_stream_idx_o = {2'b00, rx_cmd_stream_index_i};
endmodule
