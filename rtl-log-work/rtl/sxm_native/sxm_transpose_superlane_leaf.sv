`timescale 1ns/1ps

// Tile-local SXM Transpose leaf.
//
// The 96-bit command is opaque at this level. Only the northbound command
// path is registered; transpose data, consume, write, and local fault outputs
// are combinational intents for the current cycle.
module sxm_transpose_superlane_leaf (
    input  wire          clk_i,
    input  wire          rst_ni,

    input  wire          cmd_valid_i,
    input  wire [95:0]   cmd_i,
    input  wire [5:0]    dst_meta_i,

    input  wire [15:0]   src_valid_i,
    input  wire [1023:0] src_data_i,
    input  wire          buffer_available_i,

    output wire [15:0]   consume_o,
    output wire          buffer_write_valid_o,
    output wire [1023:0] buffer_write_data_o,
    output wire [5:0]    buffer_dst_meta_o,

    output reg           north_cmd_valid_o,
    output reg  [95:0]   north_cmd_o,
    output reg  [5:0]    north_dst_meta_o,

    output wire          fault_input_invalid_o,
    output wire          fault_buffer_full_o
);

    wire all_src_valid;
    wire capture_success;

    reg [1023:0] transpose_data;
    integer out_row;
    integer out_lane;
    integer plane;

    assign all_src_valid = &src_valid_i;
    assign capture_success = cmd_valid_i && all_src_valid &&
                             buffer_available_i;

    // Faults are independent so an invalid input and unavailable buffer may
    // be reported together. A command never stalls at this leaf.
    assign fault_input_invalid_o = cmd_valid_i && !all_src_valid;
    assign fault_buffer_full_o = cmd_valid_i && !buffer_available_i;

    // Consume and result-buffer write are atomic across all 16 segments.
    assign consume_o = capture_success ? 16'hFFFF : 16'h0000;
    assign buffer_write_valid_o = capture_success;
    assign buffer_write_data_o = transpose_data;
    assign buffer_dst_meta_o = dst_meta_i;

    // Input byte:
    //   src_data_i[(stream*64) + (lane*8) +: 8]
    // Output byte:
    //   transpose_data[((2*out_row+plane)*64) + (out_lane*8) +: 8]
    //       = src_data_i[((2*out_lane+plane)*64) + (out_row*8) +: 8]
    always @* begin
        transpose_data = 1024'b0;
        for (out_row = 0; out_row < 8; out_row = out_row + 1) begin
            for (plane = 0; plane < 2; plane = plane + 1) begin
                for (out_lane = 0; out_lane < 8;
                     out_lane = out_lane + 1) begin
                    transpose_data[
                        ((2*out_row + plane)*64) + out_lane*8 +: 8] =
                    src_data_i[
                        ((2*out_lane + plane)*64) + out_row*8 +: 8];
                end
            end
        end
    end

    // One fixed-cycle north command hop. Capture failure does not affect
    // command progression. Reset behavior here only initializes this local
    // command-pipeline state deterministically.
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            north_cmd_valid_o <= 1'b0;
            north_cmd_o <= 96'b0;
            north_dst_meta_o <= 6'b0;
        end else begin
            north_cmd_valid_o <= cmd_valid_i;
            north_cmd_o <= cmd_i;
            north_dst_meta_o <= dst_meta_i;
        end
    end

endmodule
