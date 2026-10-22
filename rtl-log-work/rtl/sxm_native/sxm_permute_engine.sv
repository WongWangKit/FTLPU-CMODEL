`timescale 1ns/1ps

// Slice-global fixed-cyclic SXM Permute engine.
//
// This module is intentionally stateless and combinational. It accepts a
// decoded functional command, and routes each selected source buffer. A
// single-tile command fails closed when its source is unavailable. For the
// ALL selector, each destination is independently active only when its mapped
// source is ready and old enough. The supported profile is 4T x 16S.
module sxm_permute_engine #(
    parameter integer P_TILE_ROWS     = 4,
    parameter integer P_ACTIVE_STREAMS = 16,
    parameter integer P_SEGMENT_BITS   = 64,
    parameter integer P_BUFFER_BITS    = 1024
) (
    input  wire                             cmd_valid_i,
    input  wire [7:0]                       phase_id_i,
    input  wire [3:0]                       output_row_i,
    input  wire [2:0]                       output_tile_i,

    input  wire [P_TILE_ROWS-1:0]           buffer_ready_i,
    input  wire [P_TILE_ROWS-1:0]           buffer_age_ok_i,
    input  wire [P_TILE_ROWS*P_BUFFER_BITS-1:0] buffer_data_i,
    // Reserved for future slice-level metadata policy; AC6 deliberately does
    // not invent a metadata comparison rule.
    input  wire [P_TILE_ROWS*6-1:0]         buffer_dst_meta_i,

    output reg  [P_TILE_ROWS*2-1:0]         source_tile_sel_o,
    output reg  [P_TILE_ROWS*P_ACTIVE_STREAMS-1:0] dst_valid_o,
    output reg  [P_TILE_ROWS*P_BUFFER_BITS-1:0] dst_data_o,
    output reg  [P_TILE_ROWS-1:0]           buffer_release_o,

    output reg                              fault_phase_o,
    output reg                              fault_selector_o,
    output reg                              fault_buffer_not_ready_o
);

    reg phase_legal;
    reg row_legal;
    reg tile_legal;
    reg destination_active;
    reg source_available;
    reg stream_active;
    integer destination_tile;
    integer source_tile;
    integer stream;

    // Keep the metadata input as an explicit architectural connection while
    // documenting that it has no AC6 functional comparison semantics.
    wire unused_metadata;
    assign unused_metadata = ^buffer_dst_meta_i;

    always @* begin
        source_tile_sel_o = {P_TILE_ROWS*2{1'b0}};
        dst_valid_o = {P_TILE_ROWS*P_ACTIVE_STREAMS{1'b0}};
        dst_data_o = {P_TILE_ROWS*P_BUFFER_BITS{1'b0}};
        buffer_release_o = {P_TILE_ROWS{1'b0}};
        fault_phase_o = 1'b0;
        fault_selector_o = 1'b0;
        fault_buffer_not_ready_o = 1'b0;

        phase_legal = (phase_id_i < 8'd4);
        row_legal = (output_row_i <= 4'd8);
        tile_legal = (output_tile_i <= 3'd4);
        destination_active = 1'b0;
        source_available = 1'b0;
        stream_active = 1'b0;
        source_tile = 0;

        if (cmd_valid_i) begin
            fault_phase_o = !phase_legal;
            fault_selector_o = !row_legal || !tile_legal;

            if (phase_legal) begin
                // Fixed N=4 cyclic mapping. Illegal phases are not repaired
                // with modulo arithmetic.
                case (phase_id_i)
                    8'd0: source_tile_sel_o = 8'b01_10_11_00;
                    8'd1: source_tile_sel_o = 8'b10_11_00_01;
                    8'd2: source_tile_sel_o = 8'b11_00_01_10;
                    8'd3: source_tile_sel_o = 8'b00_01_10_11;
                    default: source_tile_sel_o = 8'b0;
                endcase
            end

            if (phase_legal && row_legal && tile_legal) begin
                for (destination_tile = 0;
                     destination_tile < P_TILE_ROWS;
                     destination_tile = destination_tile + 1) begin
                    source_tile = source_tile_sel_o[
                        destination_tile*2 +: 2];
                    source_available = buffer_ready_i[source_tile] &&
                                       buffer_age_ok_i[source_tile];

                    // ALL is a per-destination availability operation. This
                    // permits diagonal head/tail waves and continuous II=4
                    // traffic without turning absent sources into a command
                    // fault. A concrete tile selector remains fail-closed.
                    if (output_tile_i == 3'd4) begin
                        destination_active = source_available;
                    end else begin
                        destination_active =
                            (output_tile_i == destination_tile) &&
                            source_available;
                        if ((output_tile_i == destination_tile) &&
                            !source_available) begin
                            fault_buffer_not_ready_o = 1'b1;
                        end
                    end

                    for (stream = 0; stream < P_ACTIVE_STREAMS;
                         stream = stream + 1) begin
                        stream_active = (output_row_i == 4'd8) ||
                            (stream == (2*output_row_i)) ||
                            (stream == (2*output_row_i + 1));
                        if (destination_active && stream_active) begin
                            dst_valid_o[
                                destination_tile*P_ACTIVE_STREAMS +
                                stream] = 1'b1;
                            // Whole 64-bit segment routing preserves every
                            // lane index and byte order.
                            dst_data_o[
                                destination_tile*P_BUFFER_BITS +
                                stream*P_SEGMENT_BITS +:
                                P_SEGMENT_BITS] =
                            buffer_data_i[
                                source_tile*P_BUFFER_BITS +
                                stream*P_SEGMENT_BITS +:
                                P_SEGMENT_BITS];
                        end
                    end

                    if (destination_active &&
                        ((output_row_i == 4'd7) ||
                         (output_row_i == 4'd8))) begin
                        buffer_release_o[source_tile] = 1'b1;
                    end
                end
            end
        end
    end

endmodule
