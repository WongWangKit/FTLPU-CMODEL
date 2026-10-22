module lpu_mem_tile_slice_behavior #(
  parameter integer DEPTH_ROWS = 65536,
  parameter integer LANES      = 8
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic        instruction_valid_i,
  input  logic [46:0] instruction_i,

  input  logic [63:0] rx_valid_i,
  input  logic [64*LANES*8-1:0] rx_data_i,
  output logic [63:0] rx_consume_o,

  output logic         tx_valid_o,
  output logic [5:0]   tx_stream_o,
  output logic [LANES*8-1:0] tx_data_o,
  output logic         tx_last_o,
  output logic         fault_o,

  input  logic         host_write_valid_i,
  input  logic [15:0]  host_address_i,
  input  logic [LANES*8-1:0] host_write_data_i,
  output logic [LANES*8-1:0] host_read_data_o
);
  localparam integer TILE_WORD_WIDTH = LANES * 8;

  logic [TILE_WORD_WIDTH-1:0] sram [0:DEPTH_ROWS-1];
  logic [2:0] opcode;
  logic [5:0] stream;
  logic [5:0] map_or_write_stream;
  logic [15:0] address;
  logic [15:0] write_address;
  logic instruction_legal;

  lpu_mem_instruction_decode u_decode (
    .instruction_i,
    .opcode_o(opcode),
    .stream_o(stream),
    .map_or_write_stream_o(map_or_write_stream),
    .address_o(address),
    .write_address_o(write_address),
    .legal_o(instruction_legal)
  );

  always_comb begin
    host_read_data_o = '0;
    if (host_address_i < DEPTH_ROWS)
      host_read_data_o = sram[host_address_i];

    rx_consume_o = '0;
    tx_valid_o  = 1'b0;
    tx_stream_o = stream;
    tx_data_o   = '0;
    tx_last_o   = 1'b0;

    if (instruction_valid_i && instruction_legal) begin
      if (opcode == 3'd1)
        rx_consume_o[stream] = rx_valid_i[stream];
      else if (opcode == 3'd2)
        rx_consume_o[map_or_write_stream] = rx_valid_i[map_or_write_stream];

      if ((opcode == 3'd0) && (address < DEPTH_ROWS)) begin
        tx_valid_o = 1'b1;
        tx_stream_o = stream;
        tx_data_o = sram[address];
        tx_last_o = 1'b1;
      end else if ((opcode == 3'd2) &&
                   (address < DEPTH_ROWS) &&
                   (write_address < DEPTH_ROWS) &&
                   rx_valid_i[map_or_write_stream]) begin
        tx_valid_o = 1'b1;
        tx_stream_o = stream;
        tx_data_o = sram[address];
        tx_last_o = 1'b1;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      fault_o     <= 1'b0;
    end else begin
      if (host_write_valid_i) begin
        if (host_address_i < DEPTH_ROWS)
          sram[host_address_i] <= host_write_data_i;
        else
          fault_o <= 1'b1;
      end

      if (instruction_valid_i) begin
        if (!instruction_legal || (address >= DEPTH_ROWS)) begin
          fault_o <= 1'b1;
        end else begin
          case (opcode)
            3'd0: begin end
            3'd1: begin
              if (rx_valid_i[stream])
                sram[address] <= rx_data_i[stream*TILE_WORD_WIDTH +: TILE_WORD_WIDTH];
              else
                fault_o <= 1'b1;
            end
            3'd2: begin
              if (write_address >= DEPTH_ROWS ||
                  !rx_valid_i[map_or_write_stream]) begin
                fault_o <= 1'b1;
              end else begin
                sram[write_address] <=
                  rx_data_i[map_or_write_stream*TILE_WORD_WIDTH +: TILE_WORD_WIDTH];
              end
            end
            default: fault_o <= 1'b1; // Gather/Scatter datapaths are not implemented.
          endcase
        end
      end
    end
  end
endmodule
