module lpu_mxm_block_accumulator #(
  parameter integer ACCUMULATOR_BLOCK_COUNT =
    lpu_pkg::MXM_ACCUMULATOR_BLOCK_COUNT
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,
  input  logic write_valid_i,
  input  logic [8*32*32-1:0] write_values_i,
  input  logic [12:0] write_address_i,
  input  logic [5:0] write_stream_base_i,
  input  logic write_stream_destination_i,
  input  logic write_clear_i,
  output logic write_ready_o,
  input  logic [3:0] read_row_valid_i,
  input  logic [4*48-1:0] read_row_instruction_i,
  output logic [4*32-1:0] west_valid_o,
  output logic [4*32*64-1:0] west_data_o,
  output logic fault_o
);
  import lpu_vxm_math_pkg::*;

  localparam integer BLOCKS = 4;
  localparam integer ROWS = 8;
  localparam integer LANES = 8;
  localparam integer STREAMS = 32;
  localparam integer DEPTH = ACCUMULATOR_BLOCK_COUNT * 4;
  localparam integer SEGMENT_WIDTH = ROWS*LANES*32;

  logic [SEGMENT_WIDTH-1:0] segment_mem_q [0:BLOCKS-1][0:DEPTH-1];
  logic segment_valid_q [0:BLOCKS-1][0:DEPTH-1];

  logic pending_valid_q;
  logic [1:0] pending_block_q;
  logic [8*32*32-1:0] pending_values_q;
  logic [12:0] pending_address_q;
  logic [5:0] pending_stream_base_q;
  logic pending_stream_destination_q;
  logic pending_clear_q;

  logic [3:0] read_opcode_valid;
  logic read_overlapping;
  logic read_active;
  logic [1:0] read_block;
  logic [47:0] read_instruction;
  logic read_supported;
  logic operation_valid;
  logic operation_write;
  logic [1:0] operation_block;
  logic [12:0] operation_address;
  logic [5:0] operation_stream_base;
  logic operation_stream_destination;
  logic operation_clear;
  logic [SEGMENT_WIDTH-1:0] operation_values;
  logic [SEGMENT_WIDTH-1:0] stored_values;
  logic [SEGMENT_WIDTH-1:0] accumulated_values;

  function automatic logic accumulator_read_supported(
    input logic [47:0] instruction
  );
    accumulator_read_supported =
      (instruction[1:0] == 2'd2) &&
      (instruction[8:2] == '0) &&
      (instruction[14:9] == '0) &&
      (instruction[27:15] < DEPTH) &&
      (instruction[45:29] == '0) &&
      instruction[46] && !instruction[47];
  endfunction

  always_comb begin
    read_opcode_valid = '0;
    for (integer block = 0; block < BLOCKS; block++)
      read_opcode_valid[block] = read_row_valid_i[block] &&
        (read_row_instruction_i[block*48 +: 2] == 2'd2);
    read_overlapping =
      (read_opcode_valid & (read_opcode_valid - 1'b1)) != 0;
    read_active = 1'b0;
    read_block = '0;
    read_instruction = '0;
    for (integer block = 0; block < BLOCKS; block++) begin
      if (read_opcode_valid[block]) begin
        read_active = 1'b1;
        read_block = block;
        read_instruction = read_row_instruction_i[block*48 +: 48];
      end
    end
    read_supported = accumulator_read_supported(read_instruction);
  end

  always_comb begin
    write_ready_o = run_i && !pending_valid_q && !read_active;
    operation_valid = 1'b0;
    operation_write = 1'b0;
    operation_block = '0;
    operation_address = '0;
    operation_stream_base = '0;
    operation_stream_destination = 1'b0;
    operation_clear = 1'b0;
    operation_values = '0;

    if (pending_valid_q) begin
      operation_valid = run_i;
      operation_write = 1'b1;
      operation_block = pending_block_q;
      operation_address = pending_address_q;
      operation_stream_base = pending_stream_base_q;
      operation_stream_destination = pending_stream_destination_q;
      operation_clear = pending_clear_q;
      for (integer row = 0; row < ROWS; row++)
        for (integer lane = 0; lane < LANES; lane++)
          operation_values[(row*LANES+lane)*32 +: 32] =
            pending_values_q[
              (row*32+pending_block_q*LANES+lane)*32 +: 32];
    end else if (write_valid_i && write_ready_o) begin
      operation_valid = 1'b1;
      operation_write = 1'b1;
      operation_block = 2'd0;
      operation_address = write_address_i;
      operation_stream_base = write_stream_base_i;
      operation_stream_destination = write_stream_destination_i;
      operation_clear = write_clear_i;
      for (integer row = 0; row < ROWS; row++)
        for (integer lane = 0; lane < LANES; lane++)
          operation_values[(row*LANES+lane)*32 +: 32] =
            write_values_i[(row*32+lane)*32 +: 32];
    end else if (run_i && read_active && !read_overlapping &&
                 read_supported && !write_valid_i) begin
      operation_valid = 1'b1;
      operation_block = read_block;
      operation_address = read_instruction[27:15];
      operation_stream_base = read_instruction[14:9];
      operation_clear = read_instruction[28];
    end
  end

  always_comb begin
    stored_values = '0;
    if (operation_valid && operation_address < DEPTH &&
        segment_valid_q[operation_block][operation_address])
      stored_values = segment_mem_q[operation_block][operation_address];
    accumulated_values = stored_values;
    if (operation_valid && operation_write)
      for (integer value = 0; value < ROWS*LANES; value++)
        accumulated_values[value*32 +: 32] = fp32_add_rne(
          stored_values[value*32 +: 32],
          operation_values[value*32 +: 32]);
  end

  always_comb begin
    west_valid_o = '0;
    west_data_o = '0;
    if (operation_valid && operation_address < DEPTH &&
        (!operation_write || operation_stream_destination)) begin
      for (integer row = 0; row < ROWS; row++) begin
        for (integer lane = 0; lane < LANES; lane++) begin
          logic [31:0] value;
          logic [15:0] value_bf16;
          value = operation_write
            ? accumulated_values[(row*LANES+lane)*32 +: 32]
            : stored_values[(row*LANES+lane)*32 +: 32];
          value_bf16 = fp32_to_bf16(value);
          for (integer byte_index = 0; byte_index < 4; byte_index++) begin
            if (!operation_write || byte_index < 2) begin
              west_valid_o[
                operation_block*STREAMS+operation_stream_base+
                row*(operation_write ? 2 : 4)+byte_index] = 1'b1;
              west_data_o[
                (operation_block*STREAMS+operation_stream_base+
                 row*(operation_write ? 2 : 4)+byte_index)*64+
                lane*8 +: 8] = operation_write
                  ? value_bf16[byte_index*8 +: 8]
                  : value[byte_index*8 +: 8];
            end
          end
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      pending_valid_q <= 1'b0;
      pending_block_q <= '0;
      pending_values_q <= '0;
      pending_address_q <= '0;
      pending_stream_base_q <= '0;
      pending_stream_destination_q <= 1'b0;
      pending_clear_q <= 1'b0;
      fault_o <= 1'b0;
      for (integer block = 0; block < BLOCKS; block++)
        for (integer address = 0; address < DEPTH; address++)
          segment_valid_q[block][address] <= 1'b0;
    end else if (!run_i) begin
      pending_valid_q <= 1'b0;
      fault_o <= 1'b0;
    end else begin
      if (read_overlapping ||
          (read_active && !read_supported) ||
          (read_active && (pending_valid_q || write_valid_i)) ||
          (write_valid_i && (!write_ready_o || write_address_i >= DEPTH)))
        fault_o <= 1'b1;

      if (operation_valid && operation_address < DEPTH) begin
        if (operation_write) begin
          if (operation_stream_destination && operation_clear)
            segment_valid_q[operation_block][operation_address] <= 1'b0;
          else begin
            segment_mem_q[operation_block][operation_address] <=
              accumulated_values;
            segment_valid_q[operation_block][operation_address] <= 1'b1;
          end
        end else if (operation_clear)
          segment_valid_q[operation_block][operation_address] <= 1'b0;
      end

      if (pending_valid_q) begin
        if (pending_block_q == 2'd3)
          pending_valid_q <= 1'b0;
        else
          pending_block_q <= pending_block_q + 1'b1;
      end else if (write_valid_i && write_ready_o && write_address_i < DEPTH) begin
        pending_valid_q <= 1'b1;
        pending_block_q <= 2'd1;
        pending_values_q <= write_values_i;
        pending_address_q <= write_address_i;
        pending_stream_base_q <= write_stream_base_i;
        pending_stream_destination_q <= write_stream_destination_i;
        pending_clear_q <= write_clear_i;
      end
    end
  end
endmodule
