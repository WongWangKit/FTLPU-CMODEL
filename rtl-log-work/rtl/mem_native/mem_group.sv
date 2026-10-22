`timescale 1ns/1ps

// Generic MEM group located between a left boundary (sreg[g]) and a right
// boundary (sreg[g+1]). Four slices share the same cycle-start boundary state.
module mem_group #(
    parameter P_SLICES                   = 4,
    parameter P_BANKS_PER_SLICE          = 2,
    parameter P_TILE_ROWS                = 4,
    parameter P_STREAMS_PER_DIRECTION    = 32,
    parameter P_SEGMENT_BITS             = 64,
    parameter P_SLICE_FAULT_CODE_BITS    = 3,
    parameter P_MEM_BANK_DEPTH_ROWS      = 32768
) (
    input  wire                         clk_i,
    input  wire                         rst_ni,

    // Entry ordering: issue = slice*2 + bank.
    input  wire [P_SLICES*P_BANKS_PER_SLICE-1:0] group_issue_valid_i,
    input  wire [P_SLICES*P_BANKS_PER_SLICE*32-1:0] group_issue_i,

    // Each boundary uses Slice SR cell ordering:
    // {direction,stream[4:0],tile[1:0]} (East 0..127, West 128..255).
    input  wire [255:0]                 left_boundary_state_valid_i,
    input  wire [16383:0]               left_boundary_state_data_i,
    input  wire [255:0]                 right_boundary_state_valid_i,
    input  wire [16383:0]               right_boundary_state_data_i,

    // Candidate ordering: candidate = slice*8 + bank*4 + tile.
    input  wire [P_SLICES*P_BANKS_PER_SLICE*P_TILE_ROWS-1:0]
                                            external_producer_collision_i,

    output reg  [255:0]                 left_boundary_consume_o,
    output reg  [255:0]                 right_boundary_consume_o,

    output wire [P_SLICES*P_BANKS_PER_SLICE*P_TILE_ROWS-1:0]
                                            left_boundary_inject_valid_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE*P_TILE_ROWS-1:0]
                                            right_boundary_inject_valid_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE*P_TILE_ROWS*P_SEGMENT_BITS-1:0]
                                            boundary_inject_data_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE*P_TILE_ROWS-1:0]
                                            boundary_inject_stream_dir_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE*P_TILE_ROWS*5-1:0]
                                            boundary_inject_stream_idx_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE*P_TILE_ROWS-1:0]
                                            internal_mem_collision_o,

    // Per {slice,bank} fault observation, preserving Slice fault bundles.
    output wire [P_SLICES*P_BANKS_PER_SLICE-1:0] slice_fault_valid_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE*P_SLICE_FAULT_CODE_BITS-1:0]
                                            slice_fault_code_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE-1:0] slice_fault_bank_id_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE-1:0] slice_fault_tile_valid_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE*2-1:0] slice_fault_tile_id_o,
    output wire [P_SLICES*P_BANKS_PER_SLICE*15-1:0] slice_fault_row_o,
    output wire                         group_fault_valid_o,

    output wire [P_SLICES-1:0]         slice_busy_o,
    output wire                         group_busy_o
);

    localparam P_CANDIDATES =
        P_SLICES * P_BANKS_PER_SLICE * P_TILE_ROWS;
    localparam P_ISSUES = P_SLICES * P_BANKS_PER_SLICE;

    // East Write observes left. West Write observes right. Unused direction
    // halves are deliberately not copied from the opposite boundary.
    wire [255:0]   slice_sr_state_valid;
    wire [16383:0] slice_sr_state_data;
    assign slice_sr_state_valid[127:0] =
        left_boundary_state_valid_i[127:0];
    assign slice_sr_state_valid[255:128] =
        right_boundary_state_valid_i[255:128];
    assign slice_sr_state_data[8191:0] =
        left_boundary_state_data_i[8191:0];
    assign slice_sr_state_data[16383:8192] =
        right_boundary_state_data_i[16383:8192];

    wire [P_SLICES*512-1:0] slice_consume;
    wire [P_CANDIDATES-1:0] slice_inject_valid;
    wire [P_CANDIDATES*P_SEGMENT_BITS-1:0] slice_inject_data;
    wire [P_CANDIDATES-1:0] slice_inject_stream_dir;
    wire [P_CANDIDATES*5-1:0] slice_inject_stream_idx;
    wire [P_CANDIDATES-1:0] slice_collision_feedback;

    genvar slice;
    generate
        for (slice = 0; slice < P_SLICES; slice = slice + 1) begin : gen_slice
            wire [1:0] bank_busy_unused;
            wire [1:0] north_valid_unused;
            wire [63:0] north_cmd_unused;

            mem_slice #(
                .P_MEM_BANK_DEPTH_ROWS(P_MEM_BANK_DEPTH_ROWS)
            ) u_slice (
                .clk_i(clk_i),
                .rst_ni(rst_ni),
                .bank_issue_valid_i(group_issue_valid_i[slice*2 +: 2]),
                .bank_issue_i(group_issue_i[slice*64 +: 64]),
                .sr_state_valid_i(slice_sr_state_valid),
                .sr_state_data_i(slice_sr_state_data),
                .producer_collision_i(slice_collision_feedback[slice*8 +: 8]),
                .sr_consume_o(slice_consume[slice*512 +: 512]),
                .sr_inject_valid_o(slice_inject_valid[slice*8 +: 8]),
                .sr_inject_data_o(slice_inject_data[slice*512 +: 512]),
                .sr_inject_stream_dir_o(slice_inject_stream_dir[slice*8 +: 8]),
                .sr_inject_stream_idx_o(slice_inject_stream_idx[slice*40 +: 40]),
                .mem_fault_valid_o(slice_fault_valid_o[slice*2 +: 2]),
                .mem_fault_code_o(slice_fault_code_o[slice*2*P_SLICE_FAULT_CODE_BITS +:
                                                      2*P_SLICE_FAULT_CODE_BITS]),
                .mem_fault_bank_id_o(slice_fault_bank_id_o[slice*2 +: 2]),
                .mem_fault_tile_valid_o(slice_fault_tile_valid_o[slice*2 +: 2]),
                .mem_fault_tile_id_o(slice_fault_tile_id_o[slice*4 +: 4]),
                .mem_fault_row_o(slice_fault_row_o[slice*30 +: 30]),
                .bank_north_cmd_valid_o(north_valid_unused),
                .bank_north_cmd_o(north_cmd_unused),
                .bank_pipeline_busy_o(bank_busy_unused),
                .slice_busy_o(slice_busy_o[slice])
            );
        end
    endgenerate

    assign boundary_inject_data_o = slice_inject_data;
    assign boundary_inject_stream_dir_o = slice_inject_stream_dir;
    assign boundary_inject_stream_idx_o = slice_inject_stream_idx;
    assign group_busy_o = |slice_busy_o;
    assign group_fault_valid_o = |slice_fault_valid_o;

    genvar candidate;
    generate
        for (candidate = 0; candidate < P_CANDIDATES;
             candidate = candidate + 1) begin : gen_candidate_boundary
            // East Read produces to right. West Read produces to left.
            assign left_boundary_inject_valid_o[candidate] =
                slice_inject_valid[candidate] &&
                slice_inject_stream_dir[candidate];
            assign right_boundary_inject_valid_o[candidate] =
                slice_inject_valid[candidate] &&
                !slice_inject_stream_dir[candidate];
        end
    endgenerate

    // Aggregate all consumer requests. Multiple consumers of the same SR cell
    // are legal and are merged with OR; no consume winner or conflict exists.
    integer consume_slice;
    integer consume_bank;
    integer consume_cell;
    integer consume_source;
    always @* begin
        left_boundary_consume_o = 256'b0;
        right_boundary_consume_o = 256'b0;
        for (consume_slice = 0; consume_slice < P_SLICES;
             consume_slice = consume_slice + 1) begin
            for (consume_bank = 0; consume_bank < P_BANKS_PER_SLICE;
                 consume_bank = consume_bank + 1) begin
                for (consume_cell = 0; consume_cell < 256;
                     consume_cell = consume_cell + 1) begin
                    consume_source = consume_slice*512 +
                                     consume_bank*256 + consume_cell;
                    if (slice_consume[consume_source]) begin
                        if (consume_cell < 128)
                            left_boundary_consume_o[consume_cell] = 1'b1;
                        else
                            right_boundary_consume_o[consume_cell] = 1'b1;
                    end
                end
            end
        end
    end

    // Compare all MEM Read candidates without choosing a producer. A target is
    // identified by boundary/direction, stream, and tile. Candidate identity
    // remains slice*8+bank*4+tile in all outputs and feedback.
    reg [P_CANDIDATES-1:0] internal_mem_collision;
    integer producer_a;
    integer producer_b;
    integer producer_a_tile;
    integer producer_b_tile;
    always @* begin
        internal_mem_collision = {P_CANDIDATES{1'b0}};
        for (producer_a = 0; producer_a < P_CANDIDATES;
             producer_a = producer_a + 1) begin
            producer_a_tile = producer_a % P_TILE_ROWS;
            if (slice_inject_valid[producer_a]) begin
                for (producer_b = 0; producer_b < P_CANDIDATES;
                     producer_b = producer_b + 1) begin
                    producer_b_tile = producer_b % P_TILE_ROWS;
                    if ((producer_a != producer_b) &&
                        slice_inject_valid[producer_b] &&
                        (slice_inject_stream_dir[producer_a] ==
                         slice_inject_stream_dir[producer_b]) &&
                        (slice_inject_stream_idx[producer_a*5 +: 5] ==
                         slice_inject_stream_idx[producer_b*5 +: 5]) &&
                        (producer_a_tile == producer_b_tile)) begin
                        internal_mem_collision[producer_a] = 1'b1;
                    end
                end
            end
        end
    end

    assign internal_mem_collision_o = internal_mem_collision;
    assign slice_collision_feedback = internal_mem_collision |
                                      external_producer_collision_i;

endmodule
