module lpu_mxm_compute #(
  parameter integer ACCUMULATOR_BLOCK_COUNT =
    lpu_pkg::MXM_ACCUMULATOR_BLOCK_COUNT
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,
  input  logic [3:0] compute_row_valid_i,
  input  logic [4*48-1:0] compute_row_instruction_i,
  input  logic [4*32-1:0] east_valid_i,
  input  logic [4*32*64-1:0] east_data_i,
  input  logic [2*4*4*8*8*16-1:0] weight_bits_i,
  input  logic [2*4*4-1:0] cell_valid_i,
  output logic [4*32-1:0] east_consumed_o,
  output logic result_valid_o,
  output logic [8*32*32-1:0] result_values_o,
  output logic [12:0] result_address_o,
  output logic [5:0] result_stream_base_o,
  output logic result_stream_destination_o,
  output logic result_clear_o,
  output logic result_block_mode_o,
  input  logic result_ready_i,
  output logic fault_o
);
  import lpu_vxm_math_pkg::*;

  localparam integer TILES = 4;
  localparam integer BLOCKS = 4;
  localparam integer LANES = 8;
  localparam integer STREAMS = 32;
`ifdef FTLPU_DISABLE_BLOCK8
  localparam integer DOT_ROWS = 1;
`else
  localparam integer DOT_ROWS = 8;
`endif
  localparam integer ACCUMULATOR_DEPTH = ACCUMULATOR_BLOCK_COUNT * 32;
  localparam integer BLOCK_ACCUMULATOR_DEPTH = ACCUMULATOR_BLOCK_COUNT * 4;

  logic [DOT_ROWS*32*32-1:0] sum_q;
  logic [1:0] buffer_active_prev_q;
  logic [4:0] next_row_q [0:1][0:TILES-1];

  logic [3:0] compute_opcode_valid;
  logic overlapping_waves;
  logic active_valid;
  logic [1:0] active_tile;
  logic [47:0] active_instruction;
  logic active_inputs_valid;
  logic active_cells_valid;
  logic active_transaction_valid;
  logic [DOT_ROWS*8*16-1:0] active_activation_bits;
  logic [4*8*8*16-1:0] active_weight_bits;
  logic [DOT_ROWS*32*32-1:0] partial_values;
  logic [8*32*32-1:0] total_values;
  logic [4:0] active_row;
  logic [28:0] active_address_wide;
  logic active_address_valid;

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

  function automatic logic instruction_supported(input logic [47:0] instruction);
    if (instruction[46])
`ifdef FTLPU_DISABLE_BLOCK8
      instruction_supported = 1'b0;
`else
      instruction_supported =
        (instruction[1:0] == 2'd1) &&
        (instruction[8:3] <= 6'd16) &&
        (instruction[14:9] <= 6'd16) &&
        (instruction[27:15] < BLOCK_ACCUMULATOR_DEPTH) &&
        (instruction[43:28] != 0);
`endif
    else
      instruction_supported =
        (instruction[1:0] == 2'd1) &&
        (instruction[8:3] <= 6'd30) &&
        (instruction[14:9] <= 6'd28) &&
        (instruction[43:28] != 0);
  endfunction

  always_comb begin
    compute_opcode_valid = '0;
    for (integer tile = 0; tile < TILES; tile++) begin
      compute_opcode_valid[tile] = compute_row_valid_i[tile] &&
        (compute_row_instruction_i[tile*48 +: 2] == 2'd1);
    end
  end

  assign overlapping_waves =
    (compute_opcode_valid & (compute_opcode_valid - 1'b1)) != 0;

  generate
    for (genvar row_gen = 0; row_gen < DOT_ROWS; row_gen++) begin : gen_dot_row
      lpu_mxm_dot_bank u_dot_bank (
        .format_bf16_i(active_instruction[45]),
        .activation_bits_i(
          active_activation_bits[row_gen*8*16 +: 8*16]),
        .weight_bits_i(active_weight_bits),
        .partial_values_o(partial_values[row_gen*32*32 +: 32*32])
      );
    end
  endgenerate

  always_comb begin
    active_valid = 1'b0;
    active_tile = '0;
    active_instruction = '0;
    active_inputs_valid = 1'b0;
    active_cells_valid = 1'b0;
    active_activation_bits = '0;
    active_weight_bits = '0;

    for (integer tile = 0; tile < TILES; tile++) begin
      if (compute_opcode_valid[tile]) begin
        active_valid = 1'b1;
        active_tile = tile;
        active_instruction = compute_row_instruction_i[tile*48 +: 48];

        active_inputs_valid = 1'b0;
        if (compute_row_instruction_i[tile*48+3 +: 6] <=
            (compute_row_instruction_i[tile*48+46] ? 6'd16 : 6'd30)) begin
          active_inputs_valid = 1'b1;
          for (integer output_row = 0; output_row < DOT_ROWS; output_row++) begin
            if (compute_row_instruction_i[tile*48+46] || output_row == 0) begin
              active_inputs_valid = active_inputs_valid &&
                east_valid_i[
                  tile*STREAMS+
                  compute_row_instruction_i[tile*48+3 +: 6]+
                  output_row*2] &&
                east_valid_i[
                  tile*STREAMS+
                  compute_row_instruction_i[tile*48+3 +: 6]+
                  output_row*2+1];
              for (integer lane = 0; lane < LANES; lane++) begin
                active_activation_bits[
                  (output_row*LANES+lane)*16 +: 16] = {
                    east_data_i[
                      (tile*STREAMS+
                       compute_row_instruction_i[tile*48+3 +: 6]+
                       output_row*2+1)*64+lane*8 +: 8],
                    east_data_i[
                      (tile*STREAMS+
                       compute_row_instruction_i[tile*48+3 +: 6]+
                       output_row*2)*64+lane*8 +: 8]};
              end
            end
          end
        end

        active_cells_valid = 1'b1;
        for (integer block = 0; block < BLOCKS; block++) begin
          active_cells_valid = active_cells_valid && cell_valid_i[
            (compute_row_instruction_i[tile*48+2]*TILES+tile)*BLOCKS+
            block];
          for (integer lane = 0; lane < LANES; lane++) begin
            for (integer column = 0; column < LANES; column++) begin
              active_weight_bits[
                ((block*LANES+lane)*LANES+column)*16 +: 16] =
                weight_bits_i[weight_index(
                  compute_row_instruction_i[tile*48+2],
                  tile,
                  block,
                  lane,
                  column) +: 16];
            end
          end
        end
      end
    end
  end

  always_comb begin
    active_row = '0;
    if (active_valid) begin
      active_row = next_row_q[active_instruction[2]][active_tile];
      if (active_tile == 0 &&
          !buffer_active_prev_q[active_instruction[2]] &&
          !active_instruction[46]) begin
        active_row = '0;
      end
    end
    active_address_wide = active_instruction[27:15] +
      (active_instruction[46] ? (active_row >> 3) : active_row) *
      active_instruction[43:28];
    active_address_valid = active_instruction[46]
      ? (active_address_wide < BLOCK_ACCUMULATOR_DEPTH)
      : (active_address_wide < ACCUMULATOR_DEPTH);
  end

  always_comb begin
    total_values = '0;
    for (integer value = 0; value < DOT_ROWS*BLOCKS*LANES; value++) begin
      if (active_tile == 0)
        total_values[value*32 +: 32] = partial_values[value*32 +: 32];
      else
        total_values[value*32 +: 32] = fp32_add_rne(
          sum_q[value*32 +: 32], partial_values[value*32 +: 32]);
    end
  end

  always_comb begin
    active_transaction_valid =
      run_i && active_valid && !overlapping_waves &&
      instruction_supported(active_instruction) &&
      active_inputs_valid && active_cells_valid &&
      active_address_valid &&
      !((active_tile == TILES-1) && !result_ready_i);

    east_consumed_o = '0;
    result_valid_o = 1'b0;
    result_values_o = total_values;
    result_address_o = active_address_wide[12:0];
    result_stream_base_o = active_instruction[14:9];
    result_stream_destination_o = active_instruction[44];
    result_clear_o = !active_instruction[47];
    result_block_mode_o = active_instruction[46];

    if (active_transaction_valid) begin
      for (integer stream_offset = 0; stream_offset < 16; stream_offset++)
        if (active_instruction[46] || stream_offset < 2)
          east_consumed_o[
            active_tile*STREAMS+active_instruction[8:3]+stream_offset] = 1'b1;

      if (active_tile == TILES-1) begin
        result_valid_o = 1'b1;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sum_q <= '0;
      buffer_active_prev_q <= '0;
      for (integer buffer = 0; buffer < 2; buffer++)
        for (integer tile = 0; tile < TILES; tile++)
          next_row_q[buffer][tile] <= '0;
      fault_o <= 1'b0;
    end else if (!run_i) begin
      sum_q <= '0;
      buffer_active_prev_q <= '0;
      for (integer buffer = 0; buffer < 2; buffer++)
        for (integer tile = 0; tile < TILES; tile++)
          next_row_q[buffer][tile] <= '0;
      fault_o <= 1'b0;
    end else begin
      buffer_active_prev_q <= '0;
      if (active_valid)
        buffer_active_prev_q[active_instruction[2]] <= 1'b1;

      if (overlapping_waves)
        fault_o <= 1'b1;

      if (active_valid && !overlapping_waves) begin
        if (!instruction_supported(active_instruction) ||
            !active_inputs_valid || !active_cells_valid ||
            !active_address_valid ||
            ((active_tile == TILES-1) && !result_ready_i)) begin
          fault_o <= 1'b1;
        end else begin
          if (active_tile == 0 &&
              !buffer_active_prev_q[active_instruction[2]] &&
              !active_instruction[46]) begin
            for (integer tile = 0; tile < TILES; tile++)
              next_row_q[active_instruction[2]][tile] <= '0;
          end
          next_row_q[active_instruction[2]][active_tile] <= active_row +
            (active_instruction[46] ? 5'd8 : 5'd1);
          sum_q <= total_values[0 +: DOT_ROWS*32*32];
        end
      end
    end
  end
endmodule
