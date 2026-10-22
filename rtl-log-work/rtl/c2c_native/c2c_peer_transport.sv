`timescale 1ns/1ps

// Vector-level peer transport for the normal path, not a physical link.
// A valid input available after edge C is visible at RX after edge C+L.
// LINK_LATENCY must be >= 1; every stage advances each cycle, even bubbles.
module c2c_peer_transport #(
  parameter integer LINK_LATENCY = 1
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic tx_valid_i,
  input  logic [255:0] tx_payload_i,
  output logic rx_valid_o,
  output logic [255:0] rx_payload_o
);
  logic [LINK_LATENCY-1:0] valid_q;
  logic [255:0] entry_q [0:LINK_LATENCY-1];

  assign rx_valid_o = valid_q[LINK_LATENCY-1];
  assign rx_payload_o = entry_q[LINK_LATENCY-1];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      valid_q <= '0;
    end else begin
      valid_q[0] <= tx_valid_i;
      for (integer stage = 1; stage < LINK_LATENCY; stage = stage + 1)
        valid_q[stage] <= valid_q[stage-1];
    end
  end

  // Complete payloads travel atomically; invalid register values do not
  // matter. No payload reset and no input/output flow-control handshake.
  always_ff @(posedge clk_i) begin
    if (rst_ni) begin
      if (tx_valid_i) entry_q[0] <= tx_payload_i;
      for (integer stage = 1; stage < LINK_LATENCY; stage = stage + 1)
        if (valid_q[stage-1]) entry_q[stage] <= entry_q[stage-1];
    end
  end

`ifndef SYNTHESIS
  initial begin
    if (LINK_LATENCY < 1)
      $fatal(1, "TEST_FAIL LINK_LATENCY must be >= 1");
  end
`endif
endmodule
