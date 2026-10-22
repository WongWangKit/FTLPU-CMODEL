`timescale 1ns/1ps

// IMPLEMENTATION CHOICE / INTERNAL INTEGRATION INTERFACE:
// one logical bank composed of four independent single-port SRAM leaves and
// one independent command pipeline. This wrapper adds no storage beyond those
// five child blocks and performs no bank-global arbitration.
module mem_logical_bank_column #(
    parameter P_TILE_ROWS               = 4,
    parameter P_LANES_PER_TILE          = 8,
    parameter P_MEM_ELEMENT_BITS        = 8,
    parameter P_MEM_BANK_DEPTH_ROWS     = 32768,
    parameter P_MEM_CONTROL_HOP_CYCLES  = 1,
    parameter P_MEM_READ_TO_SR_CYCLES   = 1,
    parameter P_MEM_FAULT_BITS          = 2
) (
    input  wire                         clk_i,
    input  wire                         rst_ni,

    input  wire                         south_cmd_valid_i,
    input  wire [31:0]                  south_cmd_i,

    input  wire [P_TILE_ROWS-1:0]       stream_valid_i,
    input  wire [P_TILE_ROWS*P_LANES_PER_TILE*P_MEM_ELEMENT_BITS-1:0] stream_data_i,

    output wire [P_TILE_ROWS-1:0]       stream_consume_o,
    output wire [P_TILE_ROWS-1:0]       read_valid_o,
    output wire [P_TILE_ROWS*P_LANES_PER_TILE*P_MEM_ELEMENT_BITS-1:0] read_data_o,
    output wire [P_TILE_ROWS-1:0]       read_stream_dir_o,
    output wire [P_TILE_ROWS*5-1:0]     read_stream_idx_o,
    output wire [P_TILE_ROWS-1:0]       fault_valid_o,
    output wire [P_TILE_ROWS*P_MEM_FAULT_BITS-1:0] fault_code_o,

    // Current per-tile decoded command view for Slice/SR integration. These
    // are state-free passthroughs of the existing control-column outputs.
    output wire [P_TILE_ROWS-1:0]       current_cmd_valid_o,
    output wire [P_TILE_ROWS*3-1:0]     current_cmd_opcode_o,
    output wire [P_TILE_ROWS*15-1:0]    current_cmd_row_o,
    output wire [P_TILE_ROWS-1:0]       current_cmd_stream_dir_o,
    output wire [P_TILE_ROWS*5-1:0]     current_cmd_stream_idx_o,
    output wire [P_TILE_ROWS-1:0]       current_cmd_preserve_o,

    output wire                         north_cmd_valid_o,
    output wire [31:0]                  north_cmd_o,
    output wire                         pipeline_busy_o
);

    localparam P_SEGMENT_BITS = P_LANES_PER_TILE * P_MEM_ELEMENT_BITS;

    wire [P_TILE_ROWS-1:0]    leaf_cmd_valid;
    wire [P_TILE_ROWS*3-1:0]  leaf_cmd_opcode;
    wire [P_TILE_ROWS*15-1:0] leaf_cmd_row;
    wire [P_TILE_ROWS-1:0]    leaf_cmd_stream_dir;
    wire [P_TILE_ROWS*5-1:0]  leaf_cmd_stream_idx;
    wire [P_TILE_ROWS-1:0]    leaf_cmd_preserve;

    assign current_cmd_valid_o      = leaf_cmd_valid;
    assign current_cmd_opcode_o     = leaf_cmd_opcode;
    assign current_cmd_row_o        = leaf_cmd_row;
    assign current_cmd_stream_dir_o = leaf_cmd_stream_dir;
    assign current_cmd_stream_idx_o = leaf_cmd_stream_idx;
    assign current_cmd_preserve_o   = leaf_cmd_preserve;

    mem_bank_control_column #(
        .P_TILE_ROWS(P_TILE_ROWS),
        .P_MEM_CONTROL_HOP_CYCLES(P_MEM_CONTROL_HOP_CYCLES)
    ) u_control_column (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .south_cmd_valid_i(south_cmd_valid_i),
        .south_cmd_i(south_cmd_i),
        .north_cmd_valid_o(north_cmd_valid_o),
        .north_cmd_o(north_cmd_o),
        .leaf_cmd_valid_o(leaf_cmd_valid),
        .leaf_cmd_opcode_o(leaf_cmd_opcode),
        .leaf_cmd_row_o(leaf_cmd_row),
        .leaf_cmd_stream_dir_o(leaf_cmd_stream_dir),
        .leaf_cmd_stream_idx_o(leaf_cmd_stream_idx),
        .leaf_cmd_preserve_o(leaf_cmd_preserve),
        .pipeline_busy_o(pipeline_busy_o)
    );

    genvar tile;
    generate
        for (tile = 0; tile < P_TILE_ROWS; tile = tile + 1) begin : gen_leaf
            mem_bank_superlane_leaf #(
                .P_MEM_BANK_DEPTH_ROWS(P_MEM_BANK_DEPTH_ROWS),
                .P_LANES_PER_TILE(P_LANES_PER_TILE),
                .P_MEM_ELEMENT_BITS(P_MEM_ELEMENT_BITS),
                .P_MEM_READ_TO_SR_CYCLES(P_MEM_READ_TO_SR_CYCLES),
                .P_MEM_FAULT_BITS(P_MEM_FAULT_BITS)
            ) u_leaf (
                .clk_i(clk_i),
                .rst_ni(rst_ni),
                .cmd_valid_i(leaf_cmd_valid[tile]),
                .cmd_opcode_i(leaf_cmd_opcode[tile*3 +: 3]),
                .cmd_row_i(leaf_cmd_row[tile*15 +: 15]),
                .cmd_stream_dir_i(leaf_cmd_stream_dir[tile]),
                .cmd_stream_idx_i(leaf_cmd_stream_idx[tile*5 +: 5]),
                .cmd_preserve_i(leaf_cmd_preserve[tile]),
                .stream_valid_i(stream_valid_i[tile]),
                .stream_data_i(stream_data_i[tile*P_SEGMENT_BITS +: P_SEGMENT_BITS]),
                .stream_consume_o(stream_consume_o[tile]),
                .read_valid_o(read_valid_o[tile]),
                .read_data_o(read_data_o[tile*P_SEGMENT_BITS +: P_SEGMENT_BITS]),
                .read_stream_dir_o(read_stream_dir_o[tile]),
                .read_stream_idx_o(read_stream_idx_o[tile*5 +: 5]),
                .fault_valid_o(fault_valid_o[tile]),
                .fault_code_o(fault_code_o[tile*P_MEM_FAULT_BITS +: P_MEM_FAULT_BITS])
            );
        end
    endgenerate

endmodule
