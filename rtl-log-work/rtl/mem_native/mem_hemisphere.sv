`timescale 1ns/1ps

// One MEM hemisphere: 52 slices arranged as 13 fixed groups between 14 MEM
// SR boundaries. This wrapper routes cycle-start state and candidates only; it
// does not implement SR propagation or next-state commit.
module mem_hemisphere #(
    parameter P_MEM_SLICES_PER_HEMI  = 52,
    parameter P_MEM_SLICES_PER_GROUP = 4,
    parameter P_MEM_BANK_DEPTH_ROWS  = 32768,
    parameter P_SLICE_FAULT_CODE_BITS = 3
) (
    input  wire clk_i,
    input  wire rst_ni,

    // issue = slice*2 + bank; slice0/bank0 occupies the lowest 32-bit slice.
    input  wire [P_MEM_SLICES_PER_HEMI*2-1:0] bank_issue_valid_i,
    input  wire [P_MEM_SLICES_PER_HEMI*2*32-1:0] bank_issue_i,

    // boundary b corresponds to sreg[b], b=0..P_GROUPS. Each boundary keeps
    // {direction,stream[4:0],tile[1:0]} Slice cell ordering.
    input  wire [((P_MEM_SLICES_PER_HEMI/P_MEM_SLICES_PER_GROUP)+1)*256-1:0]
        boundary_state_valid_i,
    input  wire [((P_MEM_SLICES_PER_HEMI/P_MEM_SLICES_PER_GROUP)+1)*16384-1:0]
        boundary_state_data_i,

    // External feedback ordering: boundary*P_PRODUCERS + global_producer.
    input  wire [((P_MEM_SLICES_PER_HEMI/P_MEM_SLICES_PER_GROUP)+1)*
                 P_MEM_SLICES_PER_HEMI*8-1:0]
        external_producer_collision_i,

    output reg  [((P_MEM_SLICES_PER_HEMI/P_MEM_SLICES_PER_GROUP)+1)*256-1:0]
        boundary_consume_o,

    // Global producer identity is the output index:
    // producer = group*32 + slice_local*8 + bank*4 + tile.
    output wire [P_MEM_SLICES_PER_HEMI*8-1:0] producer_valid_o,
    output wire [P_MEM_SLICES_PER_HEMI*8*64-1:0] producer_data_o,
    output wire [P_MEM_SLICES_PER_HEMI*8-1:0] producer_stream_dir_o,
    output wire [P_MEM_SLICES_PER_HEMI*8*5-1:0] producer_stream_idx_o,
    output wire [P_MEM_SLICES_PER_HEMI*8*4-1:0] producer_boundary_o,
    output wire [P_MEM_SLICES_PER_HEMI*8-1:0] internal_mem_collision_o,

    // Per global {slice,bank} fault observation.
    output wire [P_MEM_SLICES_PER_HEMI*2-1:0] bank_fault_valid_o,
    output wire [P_MEM_SLICES_PER_HEMI*2*P_SLICE_FAULT_CODE_BITS-1:0]
        bank_fault_code_o,
    output wire [P_MEM_SLICES_PER_HEMI*2-1:0] bank_fault_tile_valid_o,
    output wire [P_MEM_SLICES_PER_HEMI*2*2-1:0] bank_fault_tile_id_o,
    output wire [P_MEM_SLICES_PER_HEMI*2*15-1:0] bank_fault_row_o,

    output wire [(P_MEM_SLICES_PER_HEMI/P_MEM_SLICES_PER_GROUP)-1:0]
        group_fault_valid_o,
    output wire hemisphere_fault_valid_o,
    output wire [(P_MEM_SLICES_PER_HEMI/P_MEM_SLICES_PER_GROUP)-1:0]
        group_busy_o,
    output wire hemisphere_busy_o
);

    localparam P_MEM_GROUPS =
        P_MEM_SLICES_PER_HEMI / P_MEM_SLICES_PER_GROUP;
    localparam P_MEM_BOUNDARY_COUNT = P_MEM_GROUPS + 1;
    localparam P_GLOBAL_PRODUCERS = P_MEM_SLICES_PER_HEMI * 8;

    wire [P_GLOBAL_PRODUCERS-1:0] group_left_inject_valid;
    wire [P_GLOBAL_PRODUCERS-1:0] group_right_inject_valid;
    wire [P_GLOBAL_PRODUCERS*64-1:0] group_inject_data;
    wire [P_GLOBAL_PRODUCERS-1:0] group_inject_stream_dir;
    wire [P_GLOBAL_PRODUCERS*5-1:0] group_inject_stream_idx;
    wire [P_GLOBAL_PRODUCERS-1:0] group_external_collision;
    wire [P_MEM_GROUPS*256-1:0] group_left_consume;
    wire [P_MEM_GROUPS*256-1:0] group_right_consume;

`ifndef SYNTHESIS
    initial begin
        if (P_MEM_SLICES_PER_HEMI <= 0 ||
            P_MEM_SLICES_PER_GROUP <= 0 ||
            (P_MEM_SLICES_PER_HEMI % P_MEM_SLICES_PER_GROUP) != 0) begin
            $display("ERROR mem_hemisphere invalid slice/group topology");
            $finish;
        end
        if (P_MEM_SLICES_PER_GROUP != 4) begin
            $display("ERROR mem_hemisphere supports four slices per group only");
            $finish;
        end
        if (P_MEM_BANK_DEPTH_ROWS <= 0) begin
            $display("ERROR mem_hemisphere bank depth must be positive");
            $finish;
        end
    end
`endif

    genvar group;
    generate
        for (group = 0; group < P_MEM_GROUPS; group = group + 1) begin : gen_group
            wire [7:0] fault_bank_id_unused;

            mem_group #(
                .P_MEM_BANK_DEPTH_ROWS(P_MEM_BANK_DEPTH_ROWS)
            ) u_group (
                .clk_i(clk_i),
                .rst_ni(rst_ni),
                .group_issue_valid_i(bank_issue_valid_i[group*8 +: 8]),
                .group_issue_i(bank_issue_i[group*256 +: 256]),
                .left_boundary_state_valid_i(
                    boundary_state_valid_i[group*256 +: 256]),
                .left_boundary_state_data_i(
                    boundary_state_data_i[group*16384 +: 16384]),
                .right_boundary_state_valid_i(
                    boundary_state_valid_i[(group+1)*256 +: 256]),
                .right_boundary_state_data_i(
                    boundary_state_data_i[(group+1)*16384 +: 16384]),
                .external_producer_collision_i(
                    group_external_collision[group*32 +: 32]),
                .left_boundary_consume_o(
                    group_left_consume[group*256 +: 256]),
                .right_boundary_consume_o(
                    group_right_consume[group*256 +: 256]),
                .left_boundary_inject_valid_o(
                    group_left_inject_valid[group*32 +: 32]),
                .right_boundary_inject_valid_o(
                    group_right_inject_valid[group*32 +: 32]),
                .boundary_inject_data_o(
                    group_inject_data[group*2048 +: 2048]),
                .boundary_inject_stream_dir_o(
                    group_inject_stream_dir[group*32 +: 32]),
                .boundary_inject_stream_idx_o(
                    group_inject_stream_idx[group*160 +: 160]),
                .internal_mem_collision_o(
                    internal_mem_collision_o[group*32 +: 32]),
                .slice_fault_valid_o(bank_fault_valid_o[group*8 +: 8]),
                .slice_fault_code_o(
                    bank_fault_code_o[group*8*P_SLICE_FAULT_CODE_BITS +:
                                      8*P_SLICE_FAULT_CODE_BITS]),
                .slice_fault_bank_id_o(fault_bank_id_unused),
                .slice_fault_tile_valid_o(
                    bank_fault_tile_valid_o[group*8 +: 8]),
                .slice_fault_tile_id_o(
                    bank_fault_tile_id_o[group*16 +: 16]),
                .slice_fault_row_o(bank_fault_row_o[group*120 +: 120]),
                .group_fault_valid_o(group_fault_valid_o[group]),
                .slice_busy_o(),
                .group_busy_o(group_busy_o[group])
            );
        end
    endgenerate

    assign producer_valid_o = group_left_inject_valid |
                              group_right_inject_valid;
    assign producer_data_o = group_inject_data;
    assign producer_stream_dir_o = group_inject_stream_dir;
    assign producer_stream_idx_o = group_inject_stream_idx;
    assign hemisphere_busy_o = |group_busy_o;
    assign hemisphere_fault_valid_o = |group_fault_valid_o;

    // Preserve global producer identity while attaching its physical boundary.
    genvar producer_group;
    genvar local_producer;
    generate
        for (producer_group = 0; producer_group < P_MEM_GROUPS;
             producer_group = producer_group + 1) begin : gen_producer_group
            for (local_producer = 0; local_producer < 32;
                 local_producer = local_producer + 1) begin : gen_producer
                localparam GLOBAL_ID = producer_group*32 + local_producer;
                assign producer_boundary_o[GLOBAL_ID*4 +: 4] =
                    group_left_inject_valid[GLOBAL_ID]
                        ? producer_group[3:0]
                        : (producer_group + 1);
                assign group_external_collision[GLOBAL_ID] =
                    (group_left_inject_valid[GLOBAL_ID] &&
                     external_producer_collision_i[
                         producer_group*P_GLOBAL_PRODUCERS + GLOBAL_ID]) ||
                    (group_right_inject_valid[GLOBAL_ID] &&
                     external_producer_collision_i[
                         (producer_group+1)*P_GLOBAL_PRODUCERS + GLOBAL_ID]);
            end
        end
    endgenerate

    // Aggregate each Group's left/right consume onto the 14 physical MEM
    // boundaries. Shared boundaries OR both adjacent Group candidates.
    integer consume_group;
    integer consume_cell;
    always @* begin
        boundary_consume_o = {P_MEM_BOUNDARY_COUNT*256{1'b0}};
        for (consume_group = 0; consume_group < P_MEM_GROUPS;
             consume_group = consume_group + 1) begin
            for (consume_cell = 0; consume_cell < 256;
                 consume_cell = consume_cell + 1) begin
                if (group_left_consume[consume_group*256 + consume_cell])
                    boundary_consume_o[consume_group*256 + consume_cell] = 1'b1;
                if (group_right_consume[consume_group*256 + consume_cell])
                    boundary_consume_o[(consume_group+1)*256 + consume_cell] = 1'b1;
            end
        end
    end

endmodule
