`timescale 1ns/1ps

// Pure combinational attachment: Replay tile t -> West sreg13 superlane t.
// One shared valid covers the entire 64-bit segment. No payload/control state.
module lpu_c2c_rx_srf_adapter #(
  parameter integer COLUMNS = 16,
  parameter integer SUPERLANES = 4,
  parameter integer STREAMS = 32,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer C2C_PRODUCER_SLOT = 0
) (
  input logic [3:0] replay_inject_valid_i,
  input logic [255:0] replay_inject_data_i,
  input logic [19:0] replay_inject_stream_idx_i,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS-1:0]
    srf_inject_valid_o,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS*64-1:0]
    srf_inject_data_o
);
  localparam integer WEST_DIRECTION = 1;
  localparam integer RX_COLUMN = 13;
  integer tile, stream, inject_index;

  always @* begin
    srf_inject_valid_o = '0;
    srf_inject_data_o = '0;
    stream = 0;
    inject_index = 0;
    for (tile = 0; tile < 4; tile = tile + 1) begin
      if (replay_inject_valid_i[tile]) begin
        // The generic Replay port is five bits, but this attachment is W0..W7.
        // Illegal high bits are diagnosed, never treated as a legal alias.
`ifndef SYNTHESIS
        if (|replay_inject_stream_idx_i[tile*5+3 +: 2])
          $fatal(1, "TEST_FAIL C2C RX stream outside W0..W7 tile=%0d", tile);
`endif
        stream = replay_inject_stream_idx_i[tile*5 +: 3];
        inject_index = ((((WEST_DIRECTION*COLUMNS + RX_COLUMN)*SUPERLANES +
                          tile)*LOCAL_PRODUCERS + C2C_PRODUCER_SLOT)*STREAMS) +
                        stream;
        srf_inject_valid_o[inject_index] = 1'b1;
        srf_inject_data_o[inject_index*64 +: 64] =
          replay_inject_data_i[tile*64 +: 64];
      end
    end
  end

`ifndef SYNTHESIS
  initial begin
    if (COLUMNS <= RX_COLUMN || SUPERLANES != 4 || STREAMS < 8 ||
        LOCAL_PRODUCERS < 1 || C2C_PRODUCER_SLOT < 0 ||
        C2C_PRODUCER_SLOT >= LOCAL_PRODUCERS)
      $fatal(1, "TEST_FAIL Invalid C2C RX SRF attachment parameters");
  end
`endif
endmodule
