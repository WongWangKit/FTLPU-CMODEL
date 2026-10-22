`timescale 1ns/1ps

// Normal-path expansion of one complete vector into four diagonal SRF
// producer candidates. Collision handling and SRF selection are external.
module c2c_rx_replay (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic vector_valid_i,
  input  logic [255:0] vector_payload_i,
  input  logic [4:0] vector_stream_idx_i,
  output logic [3:0] inject_valid_o,
  // Packed tile t: data[t*64 +: 64], stream_idx[t*5 +: 5].
  output logic [255:0] inject_data_o,
  output logic [19:0] inject_stream_idx_o
);
  logic [2:0] deferred_valid_q;
  logic [260:0] deferred_entry_q [0:2];

  // If the input becomes visible after edge R, tile0 is visible in that
  // same emission cycle. Tile1/2/3 appear after edges R+1/R+2/R+3.
  // Suppress all candidates during reset, including the direct tile0 path.
  assign inject_valid_o = rst_ni ? {deferred_valid_q, vector_valid_i} : 4'b0000;
  assign inject_data_o[0 +: 64] = vector_payload_i[0 +: 64];
  assign inject_stream_idx_o[0 +: 5] = vector_stream_idx_i;

  generate
    for (genvar tile = 1; tile < 4; tile = tile + 1) begin : gen_delayed_tile
      assign inject_data_o[tile*64 +: 64] =
        deferred_entry_q[tile-1][tile*64 +: 64];
      assign inject_stream_idx_o[tile*5 +: 5] =
        deferred_entry_q[tile-1][260:256];
    end
  endgenerate

  // Valid always advances, including input bubbles. No ready or retention:
  // prior vectors keep emitting while the current tile0 candidate is invalid.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) deferred_valid_q <= '0;
    else deferred_valid_q <= {deferred_valid_q[1:0], vector_valid_i};
  end

  // Payload and routing metadata move together. Only valid state needs reset.
  always_ff @(posedge clk_i) begin
    if (rst_ni) begin
      if (vector_valid_i)
        deferred_entry_q[0] <= {vector_stream_idx_i, vector_payload_i};
      for (integer stage = 1; stage < 3; stage = stage + 1)
        if (deferred_valid_q[stage-1])
          deferred_entry_q[stage] <= deferred_entry_q[stage-1];
    end
  end
endmodule
