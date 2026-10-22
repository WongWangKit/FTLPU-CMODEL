`timescale 1ns/1ps

// Four-tile, normal-path diagonal gather. Stream selection is external.
module c2c_tx_gather #(
  parameter integer SEGMENT_BITS = 64
) (
  input  logic clk_i,
  input  logic rst_ni,
  // Packed tile t occupies [t*SEGMENT_BITS +: SEGMENT_BITS].
  input  logic [4*SEGMENT_BITS-1:0] tile_data_i,
  input  logic [3:0] tile_valid_i,
  output logic [3:0] tile_consume_o,
  output logic completed_valid_o,
  output logic [4*SEGMENT_BITS-1:0] completed_payload_o
);
  logic stage0_valid_q, stage1_valid_q, stage2_valid_q;
  logic [SEGMENT_BITS-1:0] stage0_payload_q;
  logic [2*SEGMENT_BITS-1:0] stage1_payload_q;
  logic [3*SEGMENT_BITS-1:0] stage2_payload_q;

  // Outputs are valid before the capture edge. Tile0 captured at edge C
  // reaches stage2 after edge C+2. Current tile3 completes it at edge C+3,
  // without an extra output register. After fill, one vector completes/edge.
  // A consume bit acknowledges a whole segment at that edge, not a stall
  // request. No retention, downstream ready, or recovery is provided.
  assign tile_consume_o = rst_ni ? {
    stage2_valid_q && tile_valid_i[3],
    stage1_valid_q && tile_valid_i[2],
    stage0_valid_q && tile_valid_i[1],
    tile_valid_i[0]
  } : 4'b0000;
  assign completed_valid_o = tile_consume_o[3];
  assign completed_payload_o = {
    tile_data_i[3*SEGMENT_BITS +: SEGMENT_BITS], stage2_payload_q
  };

  // Valid state advances every edge, including bubbles. Reset follows the
  // native SRF style: asynchronous assertion, active low.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stage0_valid_q <= 1'b0;
      stage1_valid_q <= 1'b0;
      stage2_valid_q <= 1'b0;
    end else begin
      stage0_valid_q <= tile_consume_o[0];
      stage1_valid_q <= tile_consume_o[1];
      stage2_valid_q <= tile_consume_o[2];
    end
  end

  // Payload registers need no reset: only their valid bits qualify them.
  // Source stream selection belongs to the TX SRF adapter, not the vector.
  always_ff @(posedge clk_i) begin
    if (tile_consume_o[0]) begin
      stage0_payload_q <= tile_data_i[0 +: SEGMENT_BITS];
    end
    if (tile_consume_o[1]) begin
      stage1_payload_q <= {
        tile_data_i[SEGMENT_BITS +: SEGMENT_BITS], stage0_payload_q
      };
    end
    if (tile_consume_o[2]) begin
      stage2_payload_q <= {
        tile_data_i[2*SEGMENT_BITS +: SEGMENT_BITS], stage1_payload_q
      };
    end
  end
endmodule
