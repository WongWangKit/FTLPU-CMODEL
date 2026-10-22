module lpu_mem_tile_slice_sram #(
  parameter integer DEPTH_ROWS = 65536,
  parameter integer LANES      = 8
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
  localparam integer WORD_WIDTH = LANES * 8;

  logic [2:0] opcode;
  logic [5:0] stream;
  logic [5:0] map_or_write_stream;
  logic [15:0] address;
  logic [15:0] write_address;
  logic instruction_legal;

  logic memory_req_valid;
  logic memory_write;
  logic [15:0] memory_address;
  logic [WORD_WIDTH-1:0] memory_write_data;
  logic [WORD_WIDTH-1:0] memory_read_data;
  logic memory_read_valid;

  logic response_is_instruction_q;
  logic [5:0] response_stream_q;
  logic read_write_pending_q;
  logic [15:0] read_write_address_q;
  logic [WORD_WIDTH-1:0] read_write_data_q;

  lpu_mem_instruction_decode u_decode (
    .instruction_i,
    .opcode_o(opcode),
    .stream_o(stream),
    .map_or_write_stream_o(map_or_write_stream),
    .address_o(address),
    .write_address_o(write_address),
    .legal_o(instruction_legal)
  );

  lpu_sram_1rw_banked #(
    .WIDTH(WORD_WIDTH),
    .DEPTH(DEPTH_ROWS),
    .ADDR_WIDTH(16)
  ) u_memory (
    .clk_i,
    .rst_ni,
    .req_valid_i(memory_req_valid),
    .write_i(memory_write),
    .address_i(memory_address),
    .write_data_i(memory_write_data),
    .write_mask_i({WORD_WIDTH{1'b1}}),
    .read_valid_o(memory_read_valid),
    .read_data_o(memory_read_data)
  );

  always_comb begin
    memory_req_valid = 1'b1;
    memory_write = 1'b0;
    memory_address = host_address_i;
    memory_write_data = host_write_data_i;
    rx_consume_o = '0;

    if (read_write_pending_q) begin
      memory_write = 1'b1;
      memory_address = read_write_address_q;
      memory_write_data = read_write_data_q;
    end else if (host_write_valid_i) begin
      memory_write = 1'b1;
    end else if (instruction_valid_i && instruction_legal) begin
      case (opcode)
        3'd0: memory_address = address;
        3'd1: begin
          memory_req_valid = rx_valid_i[stream];
          memory_write = 1'b1;
          memory_address = address;
          memory_write_data =
            rx_data_i[stream*WORD_WIDTH +: WORD_WIDTH];
          rx_consume_o[stream] = rx_valid_i[stream];
        end
        3'd2: begin
          memory_req_valid = rx_valid_i[map_or_write_stream];
          memory_address = address;
          rx_consume_o[map_or_write_stream] =
            rx_valid_i[map_or_write_stream];
        end
        default: memory_req_valid = 1'b0;
      endcase
    end
  end

  always_comb begin
    tx_valid_o = memory_read_valid && response_is_instruction_q;
    tx_stream_o = response_stream_q;
    tx_data_o = memory_read_data;
    tx_last_o = tx_valid_o;
    host_read_data_o = response_is_instruction_q
      ? '0 : memory_read_data;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      response_is_instruction_q <= 1'b0;
      response_stream_q <= '0;
      read_write_pending_q <= 1'b0;
      read_write_address_q <= '0;
      read_write_data_q <= '0;
      fault_o <= 1'b0;
    end else begin
      if (memory_req_valid && !memory_write) begin
        response_is_instruction_q <= instruction_valid_i &&
          instruction_legal && !host_write_valid_i &&
          !read_write_pending_q &&
          ((opcode == 3'd0) || (opcode == 3'd2));
        response_stream_q <= stream;
      end

      if (read_write_pending_q) begin
        read_write_pending_q <= 1'b0;
        if (instruction_valid_i || host_write_valid_i)
          fault_o <= 1'b1;
      end else if (instruction_valid_i && instruction_legal &&
                   (opcode == 3'd2) &&
                   rx_valid_i[map_or_write_stream]) begin
        read_write_pending_q <= 1'b1;
        read_write_address_q <= write_address;
        read_write_data_q <=
          rx_data_i[map_or_write_stream*WORD_WIDTH +: WORD_WIDTH];
      end

      if (host_write_valid_i && (host_address_i >= DEPTH_ROWS))
        fault_o <= 1'b1;
      if (instruction_valid_i) begin
        if (!instruction_legal || address >= DEPTH_ROWS ||
            ((opcode == 3'd2) && write_address >= DEPTH_ROWS) ||
            ((opcode == 3'd1) && !rx_valid_i[stream]) ||
            ((opcode == 3'd2) && !rx_valid_i[map_or_write_stream]))
          fault_o <= 1'b1;
      end
    end
  end
endmodule
