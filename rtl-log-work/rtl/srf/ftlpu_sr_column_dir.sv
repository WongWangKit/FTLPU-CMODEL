`timescale 1ns/1ps

module ftlpu_sr_column_dir #(
  parameter integer SUPERLANES      = 4,
  parameter integer STREAMS         = 32,
  parameter integer LANES           = 8,
  parameter integer DATA_BITS       = 8,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer LOCAL_CONSUMERS = 2
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic [SUPERLANES*STREAMS-1:0] column_valid_i,
  input  logic [SUPERLANES*STREAMS*LANES*DATA_BITS-1:0] column_data_i,
  output logic [SUPERLANES*STREAMS-1:0] column_valid_o,
  output logic [SUPERLANES*STREAMS*LANES*DATA_BITS-1:0] column_data_o,
  output logic [SUPERLANES*STREAMS-1:0] column_state_valid_o,
  output logic [SUPERLANES*STREAMS*LANES*DATA_BITS-1:0]
    column_state_data_o,
  input  logic [SUPERLANES*LOCAL_PRODUCERS*STREAMS-1:0] inject_valid_i,
  input  logic [SUPERLANES*LOCAL_PRODUCERS*STREAMS*LANES*DATA_BITS-1:0]
    inject_data_i,
  input  logic [SUPERLANES*LOCAL_CONSUMERS*STREAMS-1:0] consume_i,
  output logic [SUPERLANES-1:0] collision_o,
  output logic [SUPERLANES-1:0] invalid_consume_o
);
  localparam integer SEGMENT_BITS = LANES * DATA_BITS;
  localparam integer LEAF_DATA_BITS = STREAMS * SEGMENT_BITS;
  localparam integer LEAF_INJECT_VALID_BITS = LOCAL_PRODUCERS * STREAMS;
  localparam integer LEAF_INJECT_DATA_BITS =
    LOCAL_PRODUCERS * STREAMS * SEGMENT_BITS;
  localparam integer LEAF_CONSUME_BITS = LOCAL_CONSUMERS * STREAMS;

  generate
    for (genvar superlane = 0; superlane < SUPERLANES;
         superlane = superlane + 1) begin : gen_superlane
      ftlpu_sr_superlane_col_dir #(
        .STREAMS(STREAMS),
        .LANES(LANES),
        .DATA_BITS(DATA_BITS),
        .LOCAL_PRODUCERS(LOCAL_PRODUCERS),
        .LOCAL_CONSUMERS(LOCAL_CONSUMERS)
      ) u_leaf (
        .clk_i,
        .rst_ni,
        .upstream_valid_i(
          column_valid_i[superlane*STREAMS +: STREAMS]),
        .upstream_data_i(
          column_data_i[superlane*LEAF_DATA_BITS +: LEAF_DATA_BITS]),
        .state_valid_o(
          column_state_valid_o[superlane*STREAMS +: STREAMS]),
        .state_data_o(
          column_state_data_o[superlane*LEAF_DATA_BITS +: LEAF_DATA_BITS]),
        .consume_i(consume_i[
          superlane*LEAF_CONSUME_BITS +: LEAF_CONSUME_BITS]),
        .downstream_valid_o(
          column_valid_o[superlane*STREAMS +: STREAMS]),
        .downstream_data_o(
          column_data_o[superlane*LEAF_DATA_BITS +: LEAF_DATA_BITS]),
        .inject_valid_i(inject_valid_i[
          superlane*LEAF_INJECT_VALID_BITS +: LEAF_INJECT_VALID_BITS]),
        .inject_data_i(inject_data_i[
          superlane*LEAF_INJECT_DATA_BITS +: LEAF_INJECT_DATA_BITS]),
        .collision_o(collision_o[superlane]),
        .invalid_consume_o(invalid_consume_o[superlane])
      );
    end
  endgenerate
endmodule
