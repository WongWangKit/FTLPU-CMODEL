`timescale 1ns/1ps

// Four-tile Transpose control column.
//
// The wrapper is purely structural. Each leaf owns exactly one registered
// command/metadata hop; this module adds no command register, stall, retry,
// or replay behavior.
module sxm_transpose_control_column #(
    parameter integer P_TILE_ROWS = 4
) (
    input  wire                         clk_i,
    input  wire                         rst_ni,

    input  wire                         cmd_valid_i,
    input  wire [95:0]                  cmd_i,
    input  wire [5:0]                   dst_meta_i,

    input  wire [P_TILE_ROWS*16-1:0]    tile_src_valid_i,
    input  wire [P_TILE_ROWS*1024-1:0]  tile_src_data_i,
    input  wire [P_TILE_ROWS-1:0]       tile_buffer_available_i,

    output wire [P_TILE_ROWS*16-1:0]    tile_consume_o,
    output wire [P_TILE_ROWS-1:0]       tile_buffer_write_valid_o,
    output wire [P_TILE_ROWS*1024-1:0]  tile_buffer_write_data_o,
    output wire [P_TILE_ROWS*6-1:0]     tile_buffer_dst_meta_o,
    output wire [P_TILE_ROWS-1:0]       tile_fault_input_invalid_o,
    output wire [P_TILE_ROWS-1:0]       tile_fault_buffer_full_o,

    // Stage observation exposes the command presented to each tile in the
    // current cycle. Stage 0 is the external input; stages 1..3 are outputs
    // of the preceding leaf's registered north hop.
    output wire [P_TILE_ROWS-1:0]       stage_cmd_valid_o,
    output wire [P_TILE_ROWS*96-1:0]    stage_cmd_o,
    output wire [P_TILE_ROWS*6-1:0]     stage_dst_meta_o
);

    wire [P_TILE_ROWS-1:0]      stage_valid;
    wire [P_TILE_ROWS*96-1:0]   stage_cmd;
    wire [P_TILE_ROWS*6-1:0]    stage_meta;
    wire [P_TILE_ROWS-1:0]      north_valid;
    wire [P_TILE_ROWS*96-1:0]   north_cmd;
    wire [P_TILE_ROWS*6-1:0]    north_meta;

    assign stage_valid[0] = cmd_valid_i;
    assign stage_cmd[0 +: 96] = cmd_i;
    assign stage_meta[0 +: 6] = dst_meta_i;

    assign stage_cmd_valid_o = stage_valid;
    assign stage_cmd_o = stage_cmd;
    assign stage_dst_meta_o = stage_meta;

    genvar tile;
    generate
        for (tile = 0; tile < P_TILE_ROWS; tile = tile + 1) begin : g_tile
            sxm_transpose_superlane_leaf u_leaf (
                .clk_i(clk_i),
                .rst_ni(rst_ni),
                .cmd_valid_i(stage_valid[tile]),
                .cmd_i(stage_cmd[tile*96 +: 96]),
                .dst_meta_i(stage_meta[tile*6 +: 6]),
                .src_valid_i(tile_src_valid_i[tile*16 +: 16]),
                .src_data_i(tile_src_data_i[tile*1024 +: 1024]),
                .buffer_available_i(tile_buffer_available_i[tile]),
                .consume_o(tile_consume_o[tile*16 +: 16]),
                .buffer_write_valid_o(tile_buffer_write_valid_o[tile]),
                .buffer_write_data_o(
                    tile_buffer_write_data_o[tile*1024 +: 1024]),
                .buffer_dst_meta_o(tile_buffer_dst_meta_o[tile*6 +: 6]),
                .north_cmd_valid_o(north_valid[tile]),
                .north_cmd_o(north_cmd[tile*96 +: 96]),
                .north_dst_meta_o(north_meta[tile*6 +: 6]),
                .fault_input_invalid_o(tile_fault_input_invalid_o[tile]),
                .fault_buffer_full_o(tile_fault_buffer_full_o[tile])
            );

            if (tile < P_TILE_ROWS-1) begin : g_north_link
                assign stage_valid[tile+1] = north_valid[tile];
                assign stage_cmd[(tile+1)*96 +: 96] =
                    north_cmd[tile*96 +: 96];
                assign stage_meta[(tile+1)*6 +: 6] =
                    north_meta[tile*6 +: 6];
            end
        end
    endgenerate

endmodule
