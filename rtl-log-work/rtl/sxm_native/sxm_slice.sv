`timescale 1ns/1ps

// One functional SXM slice: Transpose control/capture, result buffering, and
// slice-global fixed Permute. SR collision detection remains owned by the
// future SR Fabric/integration layer; this native interface exports producer
// candidates only.
module sxm_slice #(
    parameter integer P_TILE_ROWS       = 4,
    parameter integer P_ACTIVE_STREAMS  = 16,
    parameter integer P_SELECTOR_BITS   = 6,
    parameter integer P_SEGMENT_BITS    = 64,
    parameter integer P_BUFFER_BITS     = 1024
) (
    input  wire clk_i,
    input  wire rst_ni,

    input  wire        transpose_cmd_valid_i,
    input  wire [95:0] transpose_cmd_i,
    input  wire        permute_cmd_valid_i,
    input  wire [95:0] permute_cmd_i,

    output reg  [P_TILE_ROWS*P_ACTIVE_STREAMS*P_SELECTOR_BITS-1:0]
                     sr_read_req_o,
    input  wire [P_TILE_ROWS*P_ACTIVE_STREAMS-1:0]
                     sr_read_valid_i,
    input  wire [P_TILE_ROWS*P_ACTIVE_STREAMS*P_SEGMENT_BITS-1:0]
                     sr_read_data_i,

    output wire [P_TILE_ROWS*P_ACTIVE_STREAMS-1:0] sr_consume_o,

    output wire [P_TILE_ROWS*P_ACTIVE_STREAMS-1:0] sr_write_valid_o,
    output reg  [P_ACTIVE_STREAMS*P_SELECTOR_BITS-1:0]
                     sr_write_sel_o,
    output wire [P_TILE_ROWS*P_ACTIVE_STREAMS*P_SEGMENT_BITS-1:0]
                     sr_write_data_o,

    output wire fault_valid_o,
    output wire [P_TILE_ROWS-1:0] transpose_input_invalid_o,
    output wire [P_TILE_ROWS-1:0] transpose_buffer_full_o,
    output wire permute_phase_fault_o,
    output wire permute_selector_fault_o,
    output wire permute_buffer_not_ready_o,
    output wire busy_o
);

    // THIS IS CURRENT  SXM ENTRY DECODE.
    // FINAL ISA ENCODING IS NOT FROZEN.
    wire [1:0] transpose_opcode;
    wire transpose_src_direction;
    wire [4:0] transpose_src_base;
    wire transpose_dst_direction;
    wire [4:0] transpose_dst_base;
    wire [3:0] transpose_input_row;
    wire transpose_reserved_nonzero;

    wire [1:0] permute_opcode;
    wire permute_src_direction;
    wire [4:0] permute_src_base;
    wire permute_dst_direction;
    wire [4:0] permute_dst_base;
    wire [3:0] permute_output_row;
    wire [2:0] permute_output_tile;
    wire [7:0] permute_phase_id;
    wire permute_reserved_nonzero;

    sxm_command_decode u_transpose_decode (
        .cmd_i(transpose_cmd_i),
        .opcode_o(transpose_opcode),
        .src_direction_o(transpose_src_direction),
        .src_base_o(transpose_src_base),
        .dst_direction_o(transpose_dst_direction),
        .dst_base_o(transpose_dst_base),
        .input_row_o(transpose_input_row),
        .reserved_nonzero_o(transpose_reserved_nonzero)
    );

    sxm_command_decode u_permute_decode (
        .cmd_i(permute_cmd_i),
        .opcode_o(permute_opcode),
        .src_direction_o(permute_src_direction),
        .src_base_o(permute_src_base),
        .dst_direction_o(permute_dst_direction),
        .dst_base_o(permute_dst_base),
        .output_row_o(permute_output_row),
        .output_tile_o(permute_output_tile),
        .phase_id_o(permute_phase_id),
        .reserved_nonzero_o(permute_reserved_nonzero)
    );

    // -entry command legality is enforced at the slice boundary. An
    // illegal command faults in its issue cycle and is removed before it can
    // enter the Transpose pipeline or reach the combinational Permute path.
    // This is fail-closed static-schedule behavior, not sticky recovery state.
    wire transpose_cmd_legal;
    wire permute_cmd_legal;
    wire effective_transpose_valid;
    wire effective_permute_valid;
    wire transpose_command_invalid;
    wire permute_command_invalid;

    assign transpose_cmd_legal = (transpose_opcode == 2'd0) &&
                                 (transpose_src_base <= 5'd16) &&
                                 (transpose_dst_base <= 5'd16) &&
                                 (transpose_input_row == 4'd8) &&
                                 !transpose_reserved_nonzero;
    assign permute_cmd_legal = (permute_opcode == 2'd1) &&
                               (permute_src_base <= 5'd16) &&
                               (permute_dst_base <= 5'd16) &&
                               !permute_reserved_nonzero;
    assign effective_transpose_valid = transpose_cmd_valid_i &&
                                       transpose_cmd_legal;
    assign effective_permute_valid = permute_cmd_valid_i &&
                                     permute_cmd_legal;
    assign transpose_command_invalid = transpose_cmd_valid_i &&
                                       !transpose_cmd_legal;
    assign permute_command_invalid = permute_cmd_valid_i &&
                                     !permute_cmd_legal;

    wire [P_TILE_ROWS-1:0] stage_cmd_valid;
    wire [P_TILE_ROWS*96-1:0] stage_cmd;
    wire [P_TILE_ROWS*6-1:0] stage_dst_meta;
    wire [P_TILE_ROWS-1:0] stage_src_direction;
    wire [P_TILE_ROWS*5-1:0] stage_src_base;

    wire [P_TILE_ROWS-1:0] transpose_buffer_write_valid;
    wire [P_TILE_ROWS*P_BUFFER_BITS-1:0] transpose_buffer_write_data;
    wire [P_TILE_ROWS*6-1:0] transpose_buffer_write_meta;
    wire [P_TILE_ROWS-1:0] result_buffer_available;

    sxm_transpose_control_column #(
        .P_TILE_ROWS(P_TILE_ROWS)
    ) u_transpose_control_column (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .cmd_valid_i(effective_transpose_valid),
        .cmd_i(transpose_cmd_i),
        .dst_meta_i({transpose_dst_direction, transpose_dst_base}),
        .tile_src_valid_i(sr_read_valid_i),
        .tile_src_data_i(sr_read_data_i),
        .tile_buffer_available_i(result_buffer_available),
        .tile_consume_o(sr_consume_o),
        .tile_buffer_write_valid_o(transpose_buffer_write_valid),
        .tile_buffer_write_data_o(transpose_buffer_write_data),
        .tile_buffer_dst_meta_o(transpose_buffer_write_meta),
        .tile_fault_input_invalid_o(transpose_input_invalid_o),
        .tile_fault_buffer_full_o(transpose_buffer_full_o),
        .stage_cmd_valid_o(stage_cmd_valid),
        .stage_cmd_o(stage_cmd),
        .stage_dst_meta_o(stage_dst_meta)
    );

    genvar stage;
    generate
        for (stage = 0; stage < P_TILE_ROWS; stage = stage + 1) begin : g_stage_decode
            sxm_command_decode u_stage_decode (
                .cmd_i(stage_cmd[stage*96 +: 96]),
                .src_direction_o(stage_src_direction[stage]),
                .src_base_o(stage_src_base[stage*5 +: 5])
            );
        end
    endgenerate

    integer read_tile;
    integer read_stream_slot;
    integer read_stream_index;
    always @* begin
        sr_read_req_o =
            {P_TILE_ROWS*P_ACTIVE_STREAMS*P_SELECTOR_BITS{1'b0}};
        for (read_tile = 0; read_tile < P_TILE_ROWS;
             read_tile = read_tile + 1) begin
            for (read_stream_slot = 0;
                 read_stream_slot < P_ACTIVE_STREAMS;
                 read_stream_slot = read_stream_slot + 1) begin
                read_stream_index =
                    stage_src_base[read_tile*5 +: 5] + read_stream_slot;
                if (stage_cmd_valid[read_tile] &&
                    (stage_src_base[read_tile*5 +: 5] <= 5'd16)) begin
                    sr_read_req_o[
                        (read_tile*P_ACTIVE_STREAMS + read_stream_slot)*
                        P_SELECTOR_BITS +: P_SELECTOR_BITS] =
                        {stage_src_direction[read_tile],
                         read_stream_index[4:0]};
                end
            end
        end
    end

    wire [P_TILE_ROWS-1:0] result_buffer_ready;
    wire [P_TILE_ROWS-1:0] result_buffer_age_ok;
    wire [P_TILE_ROWS*P_BUFFER_BITS-1:0] result_buffer_data;
    wire [P_TILE_ROWS*6-1:0] result_buffer_dst_meta;
    wire [P_TILE_ROWS*8-1:0] result_buffer_input_row_mask;
    wire [P_TILE_ROWS*32-1:0] result_buffer_ready_cycle;
    wire [P_TILE_ROWS-1:0] permute_buffer_release;

    sxm_transpose_result_buffer_array #(
        .P_TILE_ROWS(P_TILE_ROWS),
        .P_DATA_BITS(P_BUFFER_BITS),
        .P_DST_META_BITS(6),
        .P_CYCLE_BITS(32)
    ) u_result_buffer_array (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .write_valid_i(transpose_buffer_write_valid),
        .write_data_i(transpose_buffer_write_data),
        .write_dst_meta_i(transpose_buffer_write_meta),
        .buffer_release_i(permute_buffer_release),
        .buffer_ready_o(result_buffer_ready),
        .buffer_age_ok_o(result_buffer_age_ok),
        .buffer_available_o(result_buffer_available),
        .buffer_data_o(result_buffer_data),
        .buffer_dst_meta_o(result_buffer_dst_meta),
        .buffer_input_row_mask_o(result_buffer_input_row_mask),
        .buffer_ready_cycle_o(result_buffer_ready_cycle)
    );

    wire [P_TILE_ROWS*2-1:0] permute_source_tile_sel;

    sxm_permute_engine #(
        .P_TILE_ROWS(P_TILE_ROWS),
        .P_ACTIVE_STREAMS(P_ACTIVE_STREAMS),
        .P_SEGMENT_BITS(P_SEGMENT_BITS),
        .P_BUFFER_BITS(P_BUFFER_BITS)
    ) u_permute_engine (
        .cmd_valid_i(effective_permute_valid),
        .phase_id_i(permute_phase_id),
        .output_row_i(permute_output_row),
        .output_tile_i(permute_output_tile),
        .buffer_ready_i(result_buffer_ready),
        .buffer_age_ok_i(result_buffer_age_ok),
        .buffer_data_i(result_buffer_data),
        .buffer_dst_meta_i(result_buffer_dst_meta),
        .source_tile_sel_o(permute_source_tile_sel),
        .dst_valid_o(sr_write_valid_o),
        .dst_data_o(sr_write_data_o),
        .buffer_release_o(permute_buffer_release),
        .fault_phase_o(permute_phase_fault_o),
        .fault_selector_o(permute_selector_fault_o),
        .fault_buffer_not_ready_o(permute_buffer_not_ready_o)
    );

    integer write_stream_slot;
    integer write_stream_index;
    always @* begin
        sr_write_sel_o = {P_ACTIVE_STREAMS*P_SELECTOR_BITS{1'b0}};
        for (write_stream_slot = 0;
             write_stream_slot < P_ACTIVE_STREAMS;
             write_stream_slot = write_stream_slot + 1) begin
            write_stream_index = permute_dst_base + write_stream_slot;
            if (effective_permute_valid) begin
                sr_write_sel_o[write_stream_slot*P_SELECTOR_BITS +:
                               P_SELECTOR_BITS] =
                    {permute_dst_direction, write_stream_index[4:0]};
            end
        end
    end

    assign fault_valid_o = (|transpose_input_invalid_o) ||
                           (|transpose_buffer_full_o) ||
                           permute_phase_fault_o ||
                           permute_selector_fault_o ||
                           permute_buffer_not_ready_o ||
                           transpose_command_invalid ||
                           permute_command_invalid;

    assign busy_o = (|stage_cmd_valid) || (|result_buffer_ready);

endmodule
