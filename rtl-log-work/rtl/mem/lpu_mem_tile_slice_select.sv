module lpu_mem_tile_slice #(
  parameter integer DEPTH_ROWS = 65536,
  parameter integer LANES      = 8,
  parameter bit USE_SRAM_MACRO = 1'b0
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic instruction_valid_i,
  input  logic [46:0] instruction_i,
  input  logic [63:0] rx_valid_i,
  input  logic [64*LANES*8-1:0] rx_data_i,
  output logic [63:0] rx_consume_o,
  output logic tx_valid_o,
  output logic [5:0] tx_stream_o,
  output logic [LANES*8-1:0] tx_data_o,
  output logic tx_last_o,
  output logic fault_o,
  input  logic host_write_valid_i,
  input  logic [15:0] host_address_i,
  input  logic [LANES*8-1:0] host_write_data_i,
  output logic [LANES*8-1:0] host_read_data_o
);
  generate
    if (USE_SRAM_MACRO) begin : gen_sram
      lpu_mem_tile_slice_sram #(
        .DEPTH_ROWS(DEPTH_ROWS),
        .LANES(LANES)
      ) u_impl (.*);
    end else begin : gen_behavior
      lpu_mem_tile_slice_behavior #(
        .DEPTH_ROWS(DEPTH_ROWS),
        .LANES(LANES)
      ) u_impl (.*);
    end
  endgenerate
endmodule
