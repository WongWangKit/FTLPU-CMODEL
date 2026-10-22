`timescale 1ns/1ps

// Stateless with respect to SRF payload: this adapter observes East sreg13,
// selects one C2C-attached stream per diagonal tile, and returns the matching
// segment consume.  Only the selector/valid delay line is local control state.
module lpu_c2c_tx_srf_adapter #(
  parameter integer COLUMNS = 16,
  parameter integer SUPERLANES = 4,
  parameter integer STREAMS = 32,
  parameter integer SEGMENT_BITS = 64,
  parameter integer LOCAL_CONSUMERS = 2,
  parameter integer C2C_CONSUMER_SLOT = 0,
  parameter integer TX_COLUMN = 13
) (
  input  logic clk_i,
  input  logic rst_ni,

  // One Send issue selects one of the C2C-attached East streams E0..E7.
  input  logic tx_issue_valid_i,
  input  logic [2:0] tx_stream_index_i,

  input  logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] srf_state_valid_i,
  input  logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0]
    srf_state_data_i,

  output logic [4*SEGMENT_BITS-1:0] tile_data_o,
  output logic [3:0] tile_valid_o,
  // Packed tile t selector occupies [t*5 +: 5].  It is retained as an
  // observable source-selection result; it is not vector metadata.
  output logic [19:0] tile_stream_idx_o,

  input  logic [3:0] gather_tile_consume_i,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_CONSUMERS*STREAMS-1:0]
    srf_consume_o
);
  localparam integer EAST_DIRECTION = 0;

  logic selector_valid_d1_q;
  logic selector_valid_d2_q;
  logic selector_valid_d3_q;
  logic [2:0] selector_stream_d1_q;
  logic [2:0] selector_stream_d2_q;
  logic [2:0] selector_stream_d3_q;

  integer tile;
  integer selected_stream;
  integer state_index;
  integer consume_index;
  logic selected_valid;

  // Advance every cycle.  A bubble is represented by a zero valid entry and
  // therefore cannot be compressed ahead of an earlier vector.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      selector_valid_d1_q <= 1'b0;
      selector_valid_d2_q <= 1'b0;
      selector_valid_d3_q <= 1'b0;
    end else begin
      selector_valid_d1_q <= tx_issue_valid_i;
      selector_valid_d2_q <= selector_valid_d1_q;
      selector_valid_d3_q <= selector_valid_d2_q;
    end
  end

  always_ff @(posedge clk_i) begin
    if (tx_issue_valid_i)
      selector_stream_d1_q <= tx_stream_index_i;
    if (selector_valid_d1_q)
      selector_stream_d2_q <= selector_stream_d1_q;
    if (selector_valid_d2_q)
      selector_stream_d3_q <= selector_stream_d2_q;
  end

  always @* begin
    tile_data_o = '0;
    tile_valid_o = '0;
    tile_stream_idx_o = '0;
    srf_consume_o = '0;

    // Tile t reads East / sreg13 / superlane t.  The same selected stream is
    // used for its SRF observation and for the returned C2C consume request.
    for (tile = 0; tile < 4; tile = tile + 1) begin
      selected_valid = 1'b0;
      selected_stream = 0;
      case (tile)
        0: begin
          selected_valid = tx_issue_valid_i;
          selected_stream = tx_stream_index_i;
        end
        1: begin
          selected_valid = selector_valid_d1_q;
          selected_stream = selector_stream_d1_q;
        end
        2: begin
          selected_valid = selector_valid_d2_q;
          selected_stream = selector_stream_d2_q;
        end
        default: begin
          selected_valid = selector_valid_d3_q;
          selected_stream = selector_stream_d3_q;
        end
      endcase

      tile_stream_idx_o[tile*5 +: 5] = {2'b00, selected_stream[2:0]};
      state_index = ((EAST_DIRECTION*COLUMNS + TX_COLUMN)*SUPERLANES + tile)*
                    STREAMS + selected_stream;
      tile_valid_o[tile] = selected_valid && srf_state_valid_i[state_index];
      if (tile_valid_o[tile]) begin
        tile_data_o[tile*SEGMENT_BITS +: SEGMENT_BITS] =
          srf_state_data_i[state_index*SEGMENT_BITS +: SEGMENT_BITS];
      end

      consume_index = ((((EAST_DIRECTION*COLUMNS + TX_COLUMN)*SUPERLANES +
                        tile)*LOCAL_CONSUMERS + C2C_CONSUMER_SLOT)*STREAMS) +
                      selected_stream;
      if (gather_tile_consume_i[tile] && tile_valid_o[tile])
        srf_consume_o[consume_index] = 1'b1;
    end
  end
endmodule
