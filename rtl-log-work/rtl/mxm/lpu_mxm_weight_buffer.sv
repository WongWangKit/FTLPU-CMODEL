module lpu_mxm_weight_buffer #(
  parameter integer LOCAL_MXM_INDEX = 0
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,
  input  logic [3:0] load_row_valid_i,
  input  logic [4*48-1:0] load_row_instruction_i,
  input  logic [3:0] dequant_row_valid_i,
  input  logic [4*16-1:0] dequant_row_instruction_i,
  input  logic [4*32-1:0] east_valid_i,
  input  logic [4*32*64-1:0] east_data_i,
  output logic [4*32-1:0] east_consumed_o,
  output logic [2*4*4*8*8*16-1:0] weight_bits_o,
  output logic [2*4*4-1:0] cell_valid_o,
  output logic fault_o
);
  localparam integer TILES = 4;
  localparam integer BLOCKS = 4;
  localparam integer LANES = 8;
  localparam integer STREAMS = 32;

  logic [2*4*4*8-1:0] column_valid_q;
  logic [4*8*8*8-1:0] quantized_weights;
  logic [4*8*8*16-1:0] dequantized_weights;

  generate
    for (genvar tile_gen = 0; tile_gen < TILES; tile_gen++) begin : gen_dequant
      lpu_mxm_dequantizer u_dequantizer (
        .quantized_i(quantized_weights[tile_gen*8*8*8 +: 8*8*8]),
        .scale_bf16_i(dequant_row_instruction_i[tile_gen*16 +: 16]),
        .weight_bf16_o(
          dequantized_weights[tile_gen*8*8*16 +: 8*8*16])
      );
    end
  endgenerate

  function automatic integer weight_index(
    input integer buffer,
    input integer tile,
    input integer block,
    input integer lane,
    input integer column
  );
    weight_index =
      ((((buffer*TILES+tile)*BLOCKS+block)*LANES+lane)*LANES+column)*16;
  endfunction

  function automatic integer column_valid_index(
    input integer buffer,
    input integer tile,
    input integer block,
    input integer column
  );
    column_valid_index =
      (((buffer*TILES+tile)*BLOCKS+block)*LANES+column);
  endfunction

  function automatic logic instruction_supported(input logic [47:0] instruction);
    instruction_supported =
      (instruction[1:0] == 2'd0) &&
      (instruction[47:10] == '0) &&
      (instruction[5] || (instruction[8:6] == '0));
  endfunction

  function automatic integer input_stream_count(input logic [47:0] instruction);
    if (instruction[9])
      input_stream_count = instruction[5] ? 2 : 16;
    else
      input_stream_count = instruction[5] ? 1 : 8;
  endfunction

  always_comb begin
    east_consumed_o = '0;
    quantized_weights = '0;
    for (integer tile = 0; tile < TILES; tile++) begin
      logic [47:0] instruction;
      logic inputs_valid;
      logic dequant_matches;
      integer stream_count;
      integer stream_base;
      instruction = load_row_instruction_i[tile*48 +: 48];
      stream_count = input_stream_count(instruction);
      stream_base = LOCAL_MXM_INDEX * (instruction[9] ? 16 : 8);
      inputs_valid = 1'b1;
      for (integer stream = 0; stream < 16; stream++)
        if (stream < stream_count)
          inputs_valid = inputs_valid &&
            east_valid_i[tile*STREAMS+stream_base+stream];
      dequant_matches = instruction[9]
        ? !dequant_row_valid_i[tile]
        : dequant_row_valid_i[tile];
      if (run_i && load_row_valid_i[tile] &&
          instruction_supported(instruction) && inputs_valid &&
          dequant_matches)
        for (integer stream = 0; stream < 16; stream++)
          if (stream < stream_count)
            east_consumed_o[tile*STREAMS+stream_base+stream] = 1'b1;
      for (integer lane = 0; lane < LANES; lane++)
        for (integer column = 0; column < LANES; column++)
          quantized_weights[
            ((tile*LANES+lane)*LANES+column)*8 +: 8] =
              east_data_i[
                (tile*STREAMS+stream_base+
                 (instruction[5] ? 0 : column))*64+
                lane*8 +: 8];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      weight_bits_o <= '0;
      cell_valid_o <= '0;
      column_valid_q <= '0;
      fault_o <= 1'b0;
    end else if (!run_i) begin
      weight_bits_o <= '0;
      cell_valid_o <= '0;
      column_valid_q <= '0;
      fault_o <= 1'b0;
    end else begin
      for (integer tile = 0; tile < TILES; tile++) begin
        if (dequant_row_valid_i[tile] && !load_row_valid_i[tile])
          fault_o <= 1'b1;
        if (load_row_valid_i[tile]) begin
          logic [47:0] instruction;
          logic inputs_valid;
          integer buffer;
          integer block;
          integer selected_column;
          integer stream_count;
          integer stream_base;
          logic dequant_matches;
          instruction = load_row_instruction_i[tile*48 +: 48];
          buffer = instruction[2];
          block = instruction[4:3];
          selected_column = instruction[8:6];
          stream_count = input_stream_count(instruction);
          stream_base = LOCAL_MXM_INDEX * (instruction[9] ? 16 : 8);
          inputs_valid = 1'b1;
          for (integer stream = 0; stream < 16; stream++)
            if (stream < stream_count)
              inputs_valid = inputs_valid &&
                east_valid_i[tile*STREAMS+stream_base+stream];
          dequant_matches = instruction[9]
            ? !dequant_row_valid_i[tile]
            : dequant_row_valid_i[tile];
          if (!instruction_supported(instruction) || !inputs_valid ||
              !dequant_matches) begin
            fault_o <= 1'b1;
          end else if (instruction[5]) begin
            logic all_columns_valid;
            all_columns_valid = 1'b1;
            for (integer lane = 0; lane < LANES; lane++) begin
              if (instruction[9])
                  weight_bits_o[weight_index(
                    buffer, tile, block, lane, selected_column) +: 16] <= {
                    east_data_i[
                      (tile*STREAMS+stream_base+1)*64+lane*8 +: 8],
                    east_data_i[
                      (tile*STREAMS+stream_base+0)*64+lane*8 +: 8]};
              else
                weight_bits_o[weight_index(
                  buffer, tile, block, lane, selected_column) +: 16] <=
                    dequantized_weights[
                      ((tile*LANES+lane)*LANES+selected_column)*16 +: 16];
            end
            column_valid_q[column_valid_index(
              buffer, tile, block, selected_column)] <= 1'b1;
            for (integer column = 0; column < LANES; column++)
              if (column != selected_column)
                all_columns_valid = all_columns_valid &&
                  column_valid_q[column_valid_index(
                    buffer, tile, block, column)];
            if (all_columns_valid)
              cell_valid_o[(buffer*TILES+tile)*BLOCKS+block] <= 1'b1;
          end else begin
            for (integer lane = 0; lane < LANES; lane++) begin
              for (integer column = 0; column < LANES; column++) begin
                if (instruction[9])
                  weight_bits_o[weight_index(
                    buffer, tile, block, lane, column) +: 16] <= {
                      east_data_i[
                        (tile*STREAMS+stream_base+column*2+1)*64+
                        lane*8 +: 8],
                      east_data_i[
                        (tile*STREAMS+stream_base+column*2)*64+
                        lane*8 +: 8]};
                else
                  weight_bits_o[weight_index(
                    buffer, tile, block, lane, column) +: 16] <=
                      dequantized_weights[
                        ((tile*LANES+lane)*LANES+column)*16 +: 16];
              end
            end
            for (integer column = 0; column < LANES; column++)
              column_valid_q[column_valid_index(
                buffer, tile, block, column)] <= 1'b1;
            cell_valid_o[(buffer*TILES+tile)*BLOCKS+block] <= 1'b1;
          end
        end
      end
    end
  end
endmodule
