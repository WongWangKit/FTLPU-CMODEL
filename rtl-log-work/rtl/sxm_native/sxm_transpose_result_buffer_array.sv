`timescale 1ns/1ps

// SXM Transpose result-buffer array.
//
// Each source tile owns one independent single-entry 128-byte buffer. The
// cycle-begin outputs expose old state. A same-cycle release makes that entry
// available to a Transpose write, which commits new state at the clock edge.
module sxm_transpose_result_buffer_array #(
    parameter integer P_TILE_ROWS     = 4,
    parameter integer P_DATA_BITS     = 1024,
    parameter integer P_DST_META_BITS = 6,
    parameter integer P_CYCLE_BITS    = 32
) (
    input  wire                                      clk_i,
    input  wire                                      rst_ni,

    input  wire [P_TILE_ROWS-1:0]                    write_valid_i,
    input  wire [P_TILE_ROWS*P_DATA_BITS-1:0]        write_data_i,
    input  wire [P_TILE_ROWS*P_DST_META_BITS-1:0]    write_dst_meta_i,

    input  wire [P_TILE_ROWS-1:0]                    buffer_release_i,

    output wire [P_TILE_ROWS-1:0]                    buffer_ready_o,
    output wire [P_TILE_ROWS-1:0]                    buffer_age_ok_o,
    output wire [P_TILE_ROWS-1:0]                    buffer_available_o,
    output wire [P_TILE_ROWS*P_DATA_BITS-1:0]        buffer_data_o,
    output wire [P_TILE_ROWS*P_DST_META_BITS-1:0]    buffer_dst_meta_o,
    output wire [P_TILE_ROWS*8-1:0]                  buffer_input_row_mask_o,
    output wire [P_TILE_ROWS*P_CYCLE_BITS-1:0]       buffer_ready_cycle_o
);

    reg [P_CYCLE_BITS-1:0] current_cycle_q;
    reg [P_TILE_ROWS-1:0] ready_q;
    reg [P_TILE_ROWS*P_DATA_BITS-1:0] data_q;
    reg [P_TILE_ROWS*P_DST_META_BITS-1:0] dst_meta_q;
    reg [P_TILE_ROWS*8-1:0] input_row_mask_q;
    reg [P_TILE_ROWS*P_CYCLE_BITS-1:0] ready_cycle_q;

    wire [P_TILE_ROWS-1:0] write_accept;

    assign buffer_ready_o = ready_q;
    assign buffer_data_o = data_q;
    assign buffer_dst_meta_o = dst_meta_q;
    assign buffer_input_row_mask_o = input_row_mask_q;
    assign buffer_ready_cycle_o = ready_cycle_q;

    // A ready entry is writable only when Permute releases the old entry in
    // this cycle. This enables read-old/write-new reuse without a second bank.
    assign buffer_available_o = (~ready_q) | buffer_release_i;
    assign write_accept = write_valid_i & buffer_available_o;

    genvar tile_gen;
    generate
        for (tile_gen = 0; tile_gen < P_TILE_ROWS;
             tile_gen = tile_gen + 1) begin : g_age
            assign buffer_age_ok_o[tile_gen] =
                ready_q[tile_gen] &&
                (ready_cycle_q[tile_gen*P_CYCLE_BITS +: P_CYCLE_BITS] <
                 current_cycle_q);
        end
    endgenerate

    integer tile;
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            current_cycle_q <= {P_CYCLE_BITS{1'b0}};
            ready_q <= {P_TILE_ROWS{1'b0}};
            data_q <= {P_TILE_ROWS*P_DATA_BITS{1'b0}};
            dst_meta_q <= {P_TILE_ROWS*P_DST_META_BITS{1'b0}};
            input_row_mask_q <= {P_TILE_ROWS*8{1'b0}};
            ready_cycle_q <= {P_TILE_ROWS*P_CYCLE_BITS{1'b0}};
        end else begin
            current_cycle_q <= current_cycle_q + {{P_CYCLE_BITS-1{1'b0}},
                                                   1'b1};
            for (tile = 0; tile < P_TILE_ROWS; tile = tile + 1) begin
                // Accepted write has priority over release so release+write
                // atomically replaces OLD with NEW at cycle commit.
                if (write_accept[tile]) begin
                    ready_q[tile] <= 1'b1;
                    data_q[tile*P_DATA_BITS +: P_DATA_BITS] <=
                        write_data_i[tile*P_DATA_BITS +: P_DATA_BITS];
                    dst_meta_q[tile*P_DST_META_BITS +: P_DST_META_BITS] <=
                        write_dst_meta_i[
                            tile*P_DST_META_BITS +: P_DST_META_BITS];
                    input_row_mask_q[tile*8 +: 8] <= 8'hFF;
                    ready_cycle_q[tile*P_CYCLE_BITS +: P_CYCLE_BITS] <=
                        current_cycle_q;
                end else if (buffer_release_i[tile]) begin
                    ready_q[tile] <= 1'b0;
                end
            end
        end
    end

endmodule
