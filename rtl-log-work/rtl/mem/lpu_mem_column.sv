module lpu_mem_column #(
  parameter integer DEPTH_ROWS = 65536,
  parameter bit USE_SRAM_MACRO = 1'b0
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic issue_valid_i,
  input  logic [46:0] issue_instruction_i,

  input  logic [4*64-1:0] tile_rx_valid_i,
  input  logic [4*64*64-1:0] tile_rx_data_i,
  output logic [4*64-1:0] tile_rx_consume_o,
  output logic [3:0] tile_tx_valid_o,
  output logic [4*6-1:0] tile_tx_stream_o,
  output logic [4*64-1:0] tile_tx_data_o,
  output logic [3:0] tile_tx_last_o,
  output logic [3:0] tile_fault_o,

  input  logic host_write_valid_i,
  input  logic [1:0] host_tile_i,
  input  logic [15:0] host_address_i,
  input  logic [63:0] host_write_data_i,
  output logic [63:0] host_read_data_o
);
  logic [3:0] instruction_valid;
  logic [4*47-1:0] instruction_payload;
  logic [4*64-1:0] host_read_data;

  lpu_control_pipeline #(.WIDTH(47), .ROWS(4)) u_control (
    .clk_i,
    .rst_ni,
    .issue_valid_i,
    .issue_payload_i(issue_instruction_i),
    .row_valid_o(instruction_valid),
    .row_payload_o(instruction_payload)
  );

  generate
    for (genvar tile_gen = 0; tile_gen < 4; tile_gen++) begin : gen_tile
      localparam integer TILE_INDEX = tile_gen;
      logic [63:0] rx_consume;
      logic tx_valid;
      logic [5:0] tx_stream;
      logic [63:0] tx_data;
      logic tx_last;
      logic fault;
      logic [63:0] host_read;

      lpu_mem_tile_slice #(
        .DEPTH_ROWS(DEPTH_ROWS),
        .USE_SRAM_MACRO(USE_SRAM_MACRO)
      ) u_tile (
        .clk_i,
        .rst_ni,
        .instruction_valid_i(instruction_valid[TILE_INDEX]),
        .instruction_i(instruction_payload[TILE_INDEX*47 +: 47]),
        .rx_valid_i(tile_rx_valid_i[TILE_INDEX*64 +: 64]),
        .rx_data_i(tile_rx_data_i[TILE_INDEX*64*64 +: 64*64]),
        .rx_consume_o(rx_consume),
        .tx_valid_o(tx_valid),
        .tx_stream_o(tx_stream),
        .tx_data_o(tx_data),
        .tx_last_o(tx_last),
        .fault_o(fault),
        .host_write_valid_i(host_write_valid_i && (host_tile_i == TILE_INDEX)),
        .host_address_i,
        .host_write_data_i,
        .host_read_data_o(host_read)
      );

      assign tile_rx_consume_o[TILE_INDEX*64 +: 64] = rx_consume;
      assign tile_tx_valid_o[TILE_INDEX] = tx_valid;
      assign tile_tx_stream_o[TILE_INDEX*6 +: 6] = tx_stream;
      assign tile_tx_data_o[TILE_INDEX*64 +: 64] = tx_data;
      assign tile_tx_last_o[TILE_INDEX] = tx_last;
      assign tile_fault_o[TILE_INDEX] = fault;
      assign host_read_data[TILE_INDEX*64 +: 64] = host_read;
    end
  endgenerate

  always_comb
    host_read_data_o = host_read_data[host_tile_i*64 +: 64];
endmodule
