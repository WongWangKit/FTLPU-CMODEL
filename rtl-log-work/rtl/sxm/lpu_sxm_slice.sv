module lpu_sxm_slice (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,

  input  logic         transpose_issue_valid_i,
  input  logic [415:0] transpose_issue_instruction_i,
  input  logic         permute_issue_valid_i,
  input  logic [415:0] permute_issue_instruction_i,

  // Boundary 13 is owned by MEM; boundary 14 is the outer SXM/MXM edge.
  input  logic [4*32-1:0]    east_inner_valid_i,
  input  logic [4*32*64-1:0] east_inner_data_i,
  output logic [4*32-1:0]    east_outer_valid_o,
  output logic [4*32*64-1:0] east_outer_data_o,
  input  logic [4*32-1:0]    west_outer_valid_i,
  input  logic [4*32*64-1:0] west_outer_data_i,
  output logic [4*32-1:0]    west_inner_valid_o,
  output logic [4*32*64-1:0] west_inner_data_o,

  output logic fault_o,
  output logic conflict_o
);
  localparam integer TILES   = 4;
  localparam integer ROWS    = 8;
  localparam integer PLANES  = 2;
  localparam integer STREAMS = 32;
  localparam integer LANES   = 8;

  logic [TILES-1:0] transpose_row_valid;
  logic [TILES*416-1:0] transpose_row_instruction;

  // C-model layout:
  //   block[plane][captured input lane][tile][input stream row]
  logic [7:0] transpose_bank_q
    [0:PLANES-1][0:ROWS-1][0:TILES-1][0:ROWS-1];
  logic [5:0] bank_destination_q [0:TILES-1][0:15];
  logic [TILES-1:0] tile_ready_q;

  logic         permute_valid_q;
  logic [415:0] permute_instruction_q;
  logic [TILES-1:0] capture_fire;
  logic [TILES-1:0] transpose_inputs_valid;
  logic [TILES-1:0] permute_destination_fire;
  logic [TILES-1:0] permute_source_consumed;
  logic [1:0] permute_source_tile [0:TILES-1];
  logic [31:0] map_seen;
  logic [4*32-1:0] east_consumed;
  logic [4*32-1:0] west_consumed;
  logic [4*32-1:0] east_pass_valid_q;
  logic [4*32*64-1:0] east_pass_data_q;
  logic [4*32-1:0] west_pass_valid_q;
  logic [4*32*64-1:0] west_pass_data_q;
  logic [4*32-1:0] east_pass_valid_d;
  logic [4*32*64-1:0] east_pass_data_d;
  logic [4*32-1:0] west_pass_valid_d;
  logic [4*32*64-1:0] west_pass_data_d;
  logic fault_d;
  logic conflict_d;

  lpu_sxm_control u_control (
    .clk_i,
    .rst_ni,
    .run_i,
    .transpose_valid_i(transpose_issue_valid_i),
    .transpose_instruction_i(transpose_issue_instruction_i),
    .permute_valid_i(permute_issue_valid_i),
    .permute_instruction_i(permute_issue_instruction_i),
    .transpose_row_valid_o(transpose_row_valid),
    .transpose_row_instruction_o(transpose_row_instruction),
    .permute_valid_o(permute_valid_q),
    .permute_instruction_o(permute_instruction_q)
  );

  always_comb begin
    east_outer_valid_o = east_pass_valid_q;
    east_outer_data_o = east_pass_data_q;
    west_inner_valid_o = west_pass_valid_q;
    west_inner_data_o = west_pass_data_q;
    east_pass_valid_d = '0;
    east_pass_data_d = '0;
    west_pass_valid_d = '0;
    west_pass_data_d = '0;
    east_consumed = '0;
    west_consumed = '0;
    capture_fire = '0;
    transpose_inputs_valid = '0;
    permute_destination_fire = '0;
    permute_source_consumed = '0;
    map_seen = '0;
    fault_d = 1'b0;
    conflict_d = 1'b0;

    for (integer destination_tile = 0;
         destination_tile < TILES; destination_tile++)
      permute_source_tile[destination_tile] = '0;

    // A Transpose instruction walks north through the four physical tiles.
    // A tile captures only when all sixteen selected streams are present.
    for (integer tile = 0; tile < TILES; tile++) begin
      transpose_inputs_valid[tile] = 1'b1;
      if (transpose_row_valid[tile]) begin
        if ((transpose_row_instruction[tile*416 +: 2] != 2'd2) ||
            (transpose_row_instruction[tile*416+6 +: 5] != 5'd16) ||
            (transpose_row_instruction[tile*416+11 +: 5] != 5'd16)) begin
          transpose_inputs_valid[tile] = 1'b0;
          fault_d = 1'b1;
        end

        for (integer stream_index = 0;
             stream_index < 16; stream_index++) begin
          if (transpose_row_instruction
                [tile*416+16+stream_index*6 +: 6] < 32) begin
            if (!east_inner_valid_i[
                  tile*STREAMS + transpose_row_instruction
                    [tile*416+16+stream_index*6 +: 6]])
              transpose_inputs_valid[tile] = 1'b0;
          end else begin
            if (!west_outer_valid_i[
                  tile*STREAMS + transpose_row_instruction
                    [tile*416+16+stream_index*6 +: 6] - 32])
              transpose_inputs_valid[tile] = 1'b0;
          end

          // Physical SXM instructions use one direction for each selector
          // list. Mixed lists cannot describe a single physical port.
          if (transpose_row_instruction
                [tile*416+16+stream_index*6+5] !=
              transpose_row_instruction[tile*416+16+5]) begin
            transpose_inputs_valid[tile] = 1'b0;
            fault_d = 1'b1;
          end
          if (transpose_row_instruction
                [tile*416+112+stream_index*6+5] !=
              transpose_row_instruction[tile*416+112+5]) begin
            transpose_inputs_valid[tile] = 1'b0;
            fault_d = 1'b1;
          end
        end

        if (transpose_inputs_valid[tile]) begin
          capture_fire[tile] = 1'b1;
          for (integer stream_index = 0;
               stream_index < 16; stream_index++) begin
            if (transpose_row_instruction
                  [tile*416+16+stream_index*6 +: 6] < 32)
              east_consumed[
                tile*STREAMS + transpose_row_instruction
                  [tile*416+16+stream_index*6 +: 6]] = 1'b1;
            else
              west_consumed[
                tile*STREAMS + transpose_row_instruction
                  [tile*416+16+stream_index*6 +: 6] - 32] = 1'b1;
          end
        end
      end
    end

    // Streams not captured by Transpose cross the physical boundary-14
    // register in one cycle. Instruction-produced Permute data below already
    // targets its selected architectural boundary directly.
    for (integer tile = 0; tile < TILES; tile++) begin
      for (integer stream = 0; stream < STREAMS; stream++) begin
        if (east_inner_valid_i[tile*STREAMS+stream] &&
            !east_consumed[tile*STREAMS+stream]) begin
          east_pass_valid_d[tile*STREAMS+stream] = 1'b1;
          east_pass_data_d[(tile*STREAMS+stream)*64 +: 64] =
            east_inner_data_i[(tile*STREAMS+stream)*64 +: 64];
        end
        if (west_outer_valid_i[tile*STREAMS+stream] &&
            !west_consumed[tile*STREAMS+stream]) begin
          west_pass_valid_d[tile*STREAMS+stream] = 1'b1;
          west_pass_data_d[(tile*STREAMS+stream)*64 +: 64] =
            west_outer_data_i[(tile*STREAMS+stream)*64 +: 64];
        end
      end
    end

    // Permute is a single physical instruction (not a four-row control
    // wave). Its map must be a 32-lane bijection made from complete 8-lane
    // tile blocks, exactly as enforced by SxmSlice in the C model.
    if (permute_valid_q) begin
      if ((permute_instruction_q[1:0] != 2'd3) ||
          (permute_instruction_q[6 +: 5] != 5'd16) ||
          (permute_instruction_q[11 +: 5] != 5'd16))
        fault_d = 1'b1;

      for (integer map_index = 0; map_index < 32; map_index++) begin
        if (map_seen[permute_instruction_q[240+map_index*5 +: 5]])
          fault_d = 1'b1;
        map_seen[permute_instruction_q[240+map_index*5 +: 5]] = 1'b1;
      end

      for (integer destination_tile = 0;
           destination_tile < TILES; destination_tile++) begin
        permute_source_tile[destination_tile] =
          permute_instruction_q[240+destination_tile*8*5+3 +: 2];

        for (integer lane = 0; lane < LANES; lane++) begin
          if (permute_instruction_q
                [240+(destination_tile*8+lane)*5+3 +: 2] !=
              permute_source_tile[destination_tile])
            fault_d = 1'b1;
        end

        if (tile_ready_q[permute_source_tile[destination_tile]]) begin
          permute_destination_fire[destination_tile] = 1'b1;
          for (integer stream_index = 0;
               stream_index < 16; stream_index++) begin
            if (bank_destination_q[permute_source_tile[destination_tile]]
                                  [stream_index] !=
                permute_instruction_q[16+stream_index*6 +: 6]) begin
              permute_destination_fire[destination_tile] = 1'b0;
              fault_d = 1'b1;
            end
          end
        end

        if (permute_destination_fire[destination_tile]) begin
          permute_source_consumed[
            permute_source_tile[destination_tile]] = 1'b1;

          for (integer row = 0; row < ROWS; row++) begin
            for (integer plane = 0; plane < PLANES; plane++) begin
              if (permute_instruction_q
                    [112+(row*2+plane)*6 +: 6] < 32) begin
                if (east_outer_valid_o[
                      destination_tile*STREAMS + permute_instruction_q
                        [112+(row*2+plane)*6 +: 6]])
                  conflict_d = 1'b1;
                else begin
                  east_outer_valid_o[
                    destination_tile*STREAMS + permute_instruction_q
                      [112+(row*2+plane)*6 +: 6]] = 1'b1;
                  for (integer lane = 0; lane < LANES; lane++)
                    east_outer_data_o[
                      (destination_tile*STREAMS + permute_instruction_q
                        [112+(row*2+plane)*6 +: 6])*64 + lane*8 +: 8] =
                      transpose_bank_q[plane][row]
                        [permute_source_tile[destination_tile]]
                        [permute_instruction_q
                          [240+(destination_tile*8+lane)*5 +: 5] & 5'h7];
                end
              end else begin
                if (west_inner_valid_o[
                      destination_tile*STREAMS + permute_instruction_q
                        [112+(row*2+plane)*6 +: 6] - 32])
                  conflict_d = 1'b1;
                else begin
                  west_inner_valid_o[
                    destination_tile*STREAMS + permute_instruction_q
                      [112+(row*2+plane)*6 +: 6] - 32] = 1'b1;
                  for (integer lane = 0; lane < LANES; lane++)
                    west_inner_data_o[
                      (destination_tile*STREAMS + permute_instruction_q
                        [112+(row*2+plane)*6 +: 6] - 32)*64 + lane*8 +: 8] =
                      transpose_bank_q[plane][row]
                        [permute_source_tile[destination_tile]]
                        [permute_instruction_q
                          [240+(destination_tile*8+lane)*5 +: 5] & 5'h7];
                end
              end
            end
          end
        end
      end
    end

    // CModel ordering is Permute(old bank) followed by Transpose capture.
    // Reusing a tile in the same cycle is legal only when Permute releases
    // that old tile; otherwise the capture would overwrite a full bank.
    for (integer tile = 0; tile < TILES; tile++) begin
      if (capture_fire[tile] && tile_ready_q[tile] &&
          !permute_source_consumed[tile]) begin
        capture_fire[tile] = 1'b0;
        fault_d = 1'b1;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      tile_ready_q <= '0;
      fault_o <= 1'b0;
      conflict_o <= 1'b0;
      east_pass_valid_q <= '0;
      east_pass_data_q <= '0;
      west_pass_valid_q <= '0;
      west_pass_data_q <= '0;
      for (integer plane = 0; plane < PLANES; plane++)
        for (integer lane = 0; lane < ROWS; lane++)
          for (integer tile = 0; tile < TILES; tile++)
            for (integer row = 0; row < ROWS; row++)
              transpose_bank_q[plane][lane][tile][row] <= '0;
      for (integer tile = 0; tile < TILES; tile++)
        for (integer stream_index = 0;
             stream_index < 16; stream_index++)
          bank_destination_q[tile][stream_index] <= '0;
    end else if (!run_i) begin
      tile_ready_q <= '0;
      fault_o <= 1'b0;
      conflict_o <= 1'b0;
      east_pass_valid_q <= '0;
      east_pass_data_q <= '0;
      west_pass_valid_q <= '0;
      west_pass_data_q <= '0;
    end else begin
      fault_o <= fault_o | fault_d;
      conflict_o <= conflict_o | conflict_d;
      east_pass_valid_q <= east_pass_valid_d;
      east_pass_data_q <= east_pass_data_d;
      west_pass_valid_q <= west_pass_valid_d;
      west_pass_data_q <= west_pass_data_d;

      for (integer tile = 0; tile < TILES; tile++) begin
        if (permute_source_consumed[tile])
          tile_ready_q[tile] <= 1'b0;

        if (capture_fire[tile]) begin
          tile_ready_q[tile] <= 1'b1;
          for (integer stream_index = 0;
               stream_index < 16; stream_index++)
            bank_destination_q[tile][stream_index] <=
              transpose_row_instruction
                [tile*416+112+stream_index*6 +: 6];

          for (integer row = 0; row < ROWS; row++) begin
            for (integer plane = 0; plane < PLANES; plane++) begin
              for (integer lane = 0; lane < LANES; lane++) begin
                if (transpose_row_instruction
                      [tile*416+16+(row*2+plane)*6 +: 6] < 32)
                  transpose_bank_q[plane][lane][tile][row] <=
                    east_inner_data_i[
                      (tile*STREAMS + transpose_row_instruction
                        [tile*416+16+(row*2+plane)*6 +: 6])*64 +
                      lane*8 +: 8];
                else
                  transpose_bank_q[plane][lane][tile][row] <=
                    west_outer_data_i[
                      (tile*STREAMS + transpose_row_instruction
                        [tile*416+16+(row*2+plane)*6 +: 6] - 32)*64 +
                      lane*8 +: 8];
              end
            end
          end
        end
      end
    end
  end
endmodule
