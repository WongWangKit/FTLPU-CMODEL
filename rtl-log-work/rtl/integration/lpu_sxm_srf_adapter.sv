`timescale 1ns/1ps

// Stateless SXM/SRF segment adapter.  It translates only packed bus layout
// and the fixed SXM slot/boundary contract; it owns no payload or valid state.
module lpu_sxm_srf_adapter #(
  parameter integer COLUMNS = 16,
  parameter integer SUPERLANES = 4,
  parameter integer STREAMS = 32,
  parameter integer SEGMENT_BITS = 64,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer LOCAL_CONSUMERS = 2,
  parameter integer SXM_ACTIVE_STREAMS = 16,
  parameter integer SXM_CONSUMER_SLOT = 1,
  parameter integer SXM_PRODUCER_SLOT = 1
) (
  input logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] srf_state_valid_i,
  input logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] srf_state_data_i,
  input logic [SUPERLANES*SXM_ACTIVE_STREAMS*6-1:0] sxm_sr_read_req_i,
  output logic [SUPERLANES*SXM_ACTIVE_STREAMS-1:0] sxm_sr_read_valid_o,
  output logic [SUPERLANES*SXM_ACTIVE_STREAMS*SEGMENT_BITS-1:0] sxm_sr_read_data_o,
  input logic [SUPERLANES*SXM_ACTIVE_STREAMS-1:0] sxm_sr_consume_i,
  input logic [SUPERLANES*SXM_ACTIVE_STREAMS-1:0] sxm_sr_write_valid_i,
  input logic [SXM_ACTIVE_STREAMS*6-1:0] sxm_sr_write_sel_i,
  input logic [SUPERLANES*SXM_ACTIVE_STREAMS*SEGMENT_BITS-1:0] sxm_sr_write_data_i,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_CONSUMERS*STREAMS-1:0] srf_consume_o,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS-1:0] srf_inject_valid_o,
  output logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS*SEGMENT_BITS-1:0]
    srf_inject_data_o
);
  integer tile;
  integer stream_slot;
  integer direction;
  integer stream;
  integer column;
  integer sxm_index;
  integer state_index;
  integer consume_index;
  integer inject_index;
  logic [5:0] selector;
  logic sxm_east;

  always @* begin
    sxm_sr_read_valid_o = '0;
    sxm_sr_read_data_o = '0;
    srf_consume_o = '0;
    srf_inject_valid_o = '0;
    srf_inject_data_o = '0;

    // The native SXM selector follows the SR specification encoding
    // West=0/East=1. The Phase 1 SRF instance stores East in direction 0 and
    // West in direction 1, so the conversion is performed here at the
    // combinational integration boundary.
    // SXM reads sreg14 for East selectors and sreg15 for West selectors.
    // Each {tile,stream_slot} uses its own native selector.
    for (tile = 0; tile < SUPERLANES; tile = tile + 1) begin
      for (stream_slot = 0; stream_slot < SXM_ACTIVE_STREAMS;
           stream_slot = stream_slot + 1) begin
        sxm_index = tile*SXM_ACTIVE_STREAMS + stream_slot;
        selector = sxm_sr_read_req_i[sxm_index*6 +: 6];
        sxm_east = selector[5];
        direction = sxm_east ? 0 : 1;
        stream = selector[4:0];
        column = sxm_east ? 14 : 15;
        state_index = ((direction*COLUMNS + column)*SUPERLANES + tile)*
                      STREAMS + stream;
        sxm_sr_read_valid_o[sxm_index] = srf_state_valid_i[state_index];
        sxm_sr_read_data_o[sxm_index*SEGMENT_BITS +: SEGMENT_BITS] =
          srf_state_data_i[state_index*SEGMENT_BITS +: SEGMENT_BITS];

        if (sxm_sr_consume_i[sxm_index]) begin
          consume_index = ((((direction*COLUMNS + column)*SUPERLANES + tile)*
                           LOCAL_CONSUMERS + SXM_CONSUMER_SLOT)*STREAMS) + stream;
          srf_consume_o[consume_index] = 1'b1;
        end
      end
    end

    // SXM writes sreg15 for East destinations and sreg14 for West
    // destinations.  The write selector is slice-global per stream slot,
    // while each tile contributes its own 64-bit segment candidate.
    for (tile = 0; tile < SUPERLANES; tile = tile + 1) begin
      for (stream_slot = 0; stream_slot < SXM_ACTIVE_STREAMS;
           stream_slot = stream_slot + 1) begin
        sxm_index = tile*SXM_ACTIVE_STREAMS + stream_slot;
        selector = sxm_sr_write_sel_i[stream_slot*6 +: 6];
        sxm_east = selector[5];
        direction = sxm_east ? 0 : 1;
        stream = selector[4:0];
        column = sxm_east ? 15 : 14;
        if (sxm_sr_write_valid_i[sxm_index]) begin
          inject_index = ((((direction*COLUMNS + column)*SUPERLANES + tile)*
                          LOCAL_PRODUCERS + SXM_PRODUCER_SLOT)*STREAMS) + stream;
          srf_inject_valid_o[inject_index] = 1'b1;
          srf_inject_data_o[inject_index*SEGMENT_BITS +: SEGMENT_BITS] =
            sxm_sr_write_data_i[sxm_index*SEGMENT_BITS +: SEGMENT_BITS];
        end
      end
    end
  end
endmodule
