`timescale 1ns/1ps

// One independent south-to-north command pipeline for a logical MEM bank.
// tile0 sees the current south command combinationally. Registered state is
// used only for tile1, tile2, tile3, and the north trace output.
module mem_bank_control_column #(
    parameter P_TILE_ROWS = 4,
    parameter P_MEM_CONTROL_HOP_CYCLES = 1
) (
    input  wire                         clk_i,
    input  wire                         rst_ni,

    input  wire                         south_cmd_valid_i,
    input  wire [31:0]                  south_cmd_i,

    output reg                          north_cmd_valid_o,
    output reg  [31:0]                  north_cmd_o,

    output wire [P_TILE_ROWS-1:0]       leaf_cmd_valid_o,
    output wire [P_TILE_ROWS*3-1:0]     leaf_cmd_opcode_o,
    output wire [P_TILE_ROWS*15-1:0]    leaf_cmd_row_o,
    output wire [P_TILE_ROWS-1:0]       leaf_cmd_stream_dir_o,
    output wire [P_TILE_ROWS*5-1:0]     leaf_cmd_stream_idx_o,
    output wire [P_TILE_ROWS-1:0]       leaf_cmd_preserve_o,

    output wire                         pipeline_busy_o
);

    reg        stage1_valid;
    reg        stage2_valid;
    reg        stage3_valid;
    reg [31:0] stage1_cmd;
    reg [31:0] stage2_cmd;
    reg [31:0] stage3_cmd;

    wire [P_TILE_ROWS*32-1:0] leaf_cmd_raw;

`ifndef SYNTHESIS
    initial begin
        if (P_TILE_ROWS != 4) begin
            $display("ERROR mem_bank_control_column supports P_TILE_ROWS=4 only");
            $finish;
        end
        if (P_MEM_CONTROL_HOP_CYCLES != 1) begin
            $display("ERROR mem_bank_control_column supports one-cycle hops only");
            $finish;
        end
    end
`endif

    // tile0 executes the south command in the current cycle; there is no
    // stage0 register and therefore no accidental extra cycle of latency.
    assign leaf_cmd_valid_o[0] = south_cmd_valid_i;
    assign leaf_cmd_valid_o[1] = stage1_valid;
    assign leaf_cmd_valid_o[2] = stage2_valid;
    assign leaf_cmd_valid_o[3] = stage3_valid;

    assign leaf_cmd_raw[0*32 +: 32] = south_cmd_i;
    assign leaf_cmd_raw[1*32 +: 32] = stage1_cmd;
    assign leaf_cmd_raw[2*32 +: 32] = stage2_cmd;
    assign leaf_cmd_raw[3*32 +: 32] = stage3_cmd;

    // Busy describes commands currently executing at tile0..tile3. The north
    // trace register is observational and does not extend functional busy.
    assign pipeline_busy_o = |leaf_cmd_valid_o;

    genvar tile;
    generate
        for (tile = 0; tile < P_TILE_ROWS; tile = tile + 1) begin : gen_decode
            wire [31:0] tile_cmd;
            assign tile_cmd = leaf_cmd_raw[tile*32 +: 32];
            assign leaf_cmd_opcode_o[tile*3 +: 3] = tile_cmd[2:0];
            assign leaf_cmd_stream_idx_o[tile*5 +: 5] = tile_cmd[7:3];
            assign leaf_cmd_stream_dir_o[tile] = tile_cmd[8];
            assign leaf_cmd_row_o[tile*15 +: 15] = tile_cmd[29:15];
            assign leaf_cmd_preserve_o[tile] = tile_cmd[31];
        end
    endgenerate

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            stage1_valid      <= 1'b0;
            stage2_valid      <= 1'b0;
            stage3_valid      <= 1'b0;
            north_cmd_valid_o <= 1'b0;
            stage1_cmd        <= 32'b0;
            stage2_cmd        <= 32'b0;
            stage3_cmd        <= 32'b0;
            north_cmd_o       <= 32'b0;
        end else begin
            // IMPLEMENTATION CHOICE: north trace is a registered copy of the
            // command presented to tile3 in this cycle. It does not feed back.
            north_cmd_valid_o <= stage3_valid;
            north_cmd_o       <= stage3_cmd;

            stage3_valid <= stage2_valid;
            stage3_cmd   <= stage2_cmd;
            stage2_valid <= stage1_valid;
            stage2_cmd   <= stage1_cmd;
            stage1_valid <= south_cmd_valid_i;
            stage1_cmd   <= south_cmd_i;
        end
    end

endmodule
