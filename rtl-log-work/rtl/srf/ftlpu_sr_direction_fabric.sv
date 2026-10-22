`timescale 1ns/1ps

module ftlpu_sr_direction_fabric #(
  parameter integer COLUMNS         = 16,
  parameter integer SUPERLANES      = 4,
  parameter integer STREAMS         = 32,
  parameter integer LANES           = 8,
  parameter integer DATA_BITS       = 8,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer LOCAL_CONSUMERS = 2,
  parameter integer DIRECTION       = 0
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic [SUPERLANES*STREAMS-1:0] stream_valid_i,
  input  logic [SUPERLANES*STREAMS*LANES*DATA_BITS-1:0] stream_data_i,
  output logic [SUPERLANES*STREAMS-1:0] stream_valid_o,
  output logic [SUPERLANES*STREAMS*LANES*DATA_BITS-1:0] stream_data_o,
  input  logic [COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS-1:0]
    inject_valid_i,
  input  logic [COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS*LANES*DATA_BITS-1:0]
    inject_data_i,
  input  logic [COLUMNS*SUPERLANES*LOCAL_CONSUMERS*STREAMS-1:0] consume_i,
  output logic [COLUMNS*SUPERLANES-1:0] collision_o,
  output logic [COLUMNS*SUPERLANES-1:0] invalid_consume_o,
  output logic [COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o,
  output logic [COLUMNS*SUPERLANES*STREAMS*LANES*DATA_BITS-1:0]
    state_data_o
);
  localparam integer COLUMN_VALID_BITS = SUPERLANES * STREAMS;
  localparam integer COLUMN_DATA_BITS =
    SUPERLANES * STREAMS * LANES * DATA_BITS;
  localparam integer COLUMN_INJECT_VALID_BITS =
    SUPERLANES * LOCAL_PRODUCERS * STREAMS;
  localparam integer COLUMN_INJECT_DATA_BITS =
    SUPERLANES * LOCAL_PRODUCERS * STREAMS * LANES * DATA_BITS;
  localparam integer COLUMN_CONSUME_BITS =
    SUPERLANES * LOCAL_CONSUMERS * STREAMS;

  logic [COLUMNS*COLUMN_VALID_BITS-1:0] column_valid_i;
  logic [COLUMNS*COLUMN_VALID_BITS-1:0] column_valid_o;
  logic [COLUMNS*COLUMN_DATA_BITS-1:0] column_data_i;
  logic [COLUMNS*COLUMN_DATA_BITS-1:0] column_data_o;

  generate
    for (genvar column = 0; column < COLUMNS; column = column + 1) begin : gen_column
      if (DIRECTION == 0) begin : gen_east
        if (column == 0) begin : gen_boundary
          assign column_valid_i[
            column*COLUMN_VALID_BITS +: COLUMN_VALID_BITS] = stream_valid_i;
          assign column_data_i[
            column*COLUMN_DATA_BITS +: COLUMN_DATA_BITS] = stream_data_i;
        end else begin : gen_link
          assign column_valid_i[
            column*COLUMN_VALID_BITS +: COLUMN_VALID_BITS] =
            column_valid_o[(column-1)*COLUMN_VALID_BITS +: COLUMN_VALID_BITS];
          assign column_data_i[
            column*COLUMN_DATA_BITS +: COLUMN_DATA_BITS] =
            column_data_o[(column-1)*COLUMN_DATA_BITS +: COLUMN_DATA_BITS];
        end
      end else begin : gen_west
        if (column == COLUMNS-1) begin : gen_boundary
          assign column_valid_i[
            column*COLUMN_VALID_BITS +: COLUMN_VALID_BITS] = stream_valid_i;
          assign column_data_i[
            column*COLUMN_DATA_BITS +: COLUMN_DATA_BITS] = stream_data_i;
        end else begin : gen_link
          assign column_valid_i[
            column*COLUMN_VALID_BITS +: COLUMN_VALID_BITS] =
            column_valid_o[(column+1)*COLUMN_VALID_BITS +: COLUMN_VALID_BITS];
          assign column_data_i[
            column*COLUMN_DATA_BITS +: COLUMN_DATA_BITS] =
            column_data_o[(column+1)*COLUMN_DATA_BITS +: COLUMN_DATA_BITS];
        end
      end

      ftlpu_sr_column_dir #(
        .SUPERLANES(SUPERLANES),
        .STREAMS(STREAMS),
        .LANES(LANES),
        .DATA_BITS(DATA_BITS),
        .LOCAL_PRODUCERS(LOCAL_PRODUCERS),
        .LOCAL_CONSUMERS(LOCAL_CONSUMERS)
      ) u_column (
        .clk_i,
        .rst_ni,
        .column_valid_i(
          column_valid_i[column*COLUMN_VALID_BITS +: COLUMN_VALID_BITS]),
        .column_data_i(
          column_data_i[column*COLUMN_DATA_BITS +: COLUMN_DATA_BITS]),
        .column_valid_o(
          column_valid_o[column*COLUMN_VALID_BITS +: COLUMN_VALID_BITS]),
        .column_data_o(
          column_data_o[column*COLUMN_DATA_BITS +: COLUMN_DATA_BITS]),
        .column_state_valid_o(
          state_valid_o[column*COLUMN_VALID_BITS +: COLUMN_VALID_BITS]),
        .column_state_data_o(
          state_data_o[column*COLUMN_DATA_BITS +: COLUMN_DATA_BITS]),
        .inject_valid_i(inject_valid_i[
          column*COLUMN_INJECT_VALID_BITS +: COLUMN_INJECT_VALID_BITS]),
        .inject_data_i(inject_data_i[
          column*COLUMN_INJECT_DATA_BITS +: COLUMN_INJECT_DATA_BITS]),
        .consume_i(consume_i[
          column*COLUMN_CONSUME_BITS +: COLUMN_CONSUME_BITS]),
        .collision_o(
          collision_o[column*SUPERLANES +: SUPERLANES]),
        .invalid_consume_o(
          invalid_consume_o[column*SUPERLANES +: SUPERLANES])
      );
    end
  endgenerate

  generate
    if (DIRECTION == 0) begin : gen_east_output
      assign stream_valid_o = column_valid_o[
        (COLUMNS-1)*COLUMN_VALID_BITS +: COLUMN_VALID_BITS];
      assign stream_data_o = column_data_o[
        (COLUMNS-1)*COLUMN_DATA_BITS +: COLUMN_DATA_BITS];
    end else begin : gen_west_output
      assign stream_valid_o = column_valid_o[0 +: COLUMN_VALID_BITS];
      assign stream_data_o = column_data_o[0 +: COLUMN_DATA_BITS];
    end
  endgenerate
endmodule
