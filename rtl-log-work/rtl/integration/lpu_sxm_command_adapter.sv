`timescale 1ns/1ps

// Stateless VMODEL-to-native SXM command translator.  It accepts only the
// native-compatible subset of the existing 416-bit format and fails closed
// for unsupported selector lists, maps, opcodes, or output-tile encodings.
module lpu_sxm_command_adapter (
  input  logic         vmodel_transpose_valid_i,
  input  logic [415:0] vmodel_transpose_instruction_i,
  input  logic         vmodel_permute_valid_i,
  input  logic [415:0] vmodel_permute_instruction_i,
  output logic         native_transpose_valid_o,
  output logic [95:0]  native_transpose_command_o,
  output logic         native_permute_valid_o,
  output logic [95:0]  native_permute_command_o,
  output logic         command_fault_o
);
  import lpu_pkg::*;

  logic src_legal;
  logic dst_legal;
  logic map_legal;
  logic [4:0] src_base;
  logic [4:0] dst_base;
  logic src_direction;
  logic dst_direction;
  logic [7:0] phase_id;
  logic [SXM_OUTPUT_TILE_WIDTH-1:0] output_tile;
  logic phase_candidate;
  logic phase_found;
  integer selector_index;
  integer map_index;
  integer destination_tile;
  integer expected_source_tile;
  integer phase_index;

  always_comb begin
    native_transpose_valid_o = 1'b0;
    native_transpose_command_o = '0;
    native_permute_valid_o = 1'b0;
    native_permute_command_o = '0;
    command_fault_o = 1'b0;

    // The selector lists have the same layout for Transpose and Permute.
    src_legal = 1'b1;
    dst_legal = 1'b1;
    src_base = vmodel_transpose_instruction_i[16 +: 5];
    dst_base = vmodel_transpose_instruction_i[112 +: 5];
    src_direction = vmodel_transpose_instruction_i[16+5];
    dst_direction = vmodel_transpose_instruction_i[112+5];
    for (selector_index = 0; selector_index < 16; selector_index = selector_index + 1) begin
      if ((vmodel_transpose_instruction_i[16+selector_index*6+5] != src_direction) ||
          (vmodel_transpose_instruction_i[16+selector_index*6 +: 5] !=
           src_base + selector_index))
        src_legal = 1'b0;
      if ((vmodel_transpose_instruction_i[112+selector_index*6+5] != dst_direction) ||
          (vmodel_transpose_instruction_i[112+selector_index*6 +: 5] !=
           dst_base + selector_index))
        dst_legal = 1'b0;
    end
    if (src_base > 5'd16 || dst_base > 5'd16)
      begin src_legal = 1'b0; dst_legal = 1'b0; end

    if (vmodel_transpose_valid_i) begin
      if ((vmodel_transpose_instruction_i[1:0] != SXM_TRANSPOSE) ||
          (vmodel_transpose_instruction_i[6 +: 5] != 5'd16) ||
          (vmodel_transpose_instruction_i[11 +: 5] != 5'd16) ||
          !src_legal || !dst_legal) begin
        command_fault_o = 1'b1;
      end else begin
        // VMODEL selector bit5 is East=0/West=1; current native SXM command
        // convention is West=0/East=1.  The native-to-SRF conversion remains
        // exclusively in lpu_sxm_srf_adapter.
        native_transpose_command_o[1:0] = 2'd0;
        native_transpose_command_o[2] = ~src_direction;
        native_transpose_command_o[7:3] = src_base;
        native_transpose_command_o[8] = ~dst_direction;
        native_transpose_command_o[13:9] = dst_base;
        native_transpose_command_o[17:14] = 4'd8;
        native_transpose_valid_o = 1'b1;
      end
    end

    src_legal = 1'b1;
    dst_legal = 1'b1;
    map_legal = 1'b1;
    src_base = vmodel_permute_instruction_i[16 +: 5];
    dst_base = vmodel_permute_instruction_i[112 +: 5];
    src_direction = vmodel_permute_instruction_i[16+5];
    dst_direction = vmodel_permute_instruction_i[112+5];
    phase_id = 8'd0;
    output_tile = vmodel_permute_instruction_i[
      SXM_OUTPUT_TILE_LSB +: SXM_OUTPUT_TILE_WIDTH];
    for (selector_index = 0; selector_index < 16; selector_index = selector_index + 1) begin
      if ((vmodel_permute_instruction_i[16+selector_index*6+5] != src_direction) ||
          (vmodel_permute_instruction_i[16+selector_index*6 +: 5] !=
           src_base + selector_index))
        src_legal = 1'b0;
      if ((vmodel_permute_instruction_i[112+selector_index*6+5] != dst_direction) ||
          (vmodel_permute_instruction_i[112+selector_index*6 +: 5] !=
           dst_base + selector_index))
        dst_legal = 1'b0;
    end
    if (src_base > 5'd16 || dst_base > 5'd16)
      begin src_legal = 1'b0; dst_legal = 1'b0; end

    // A native phase map routes complete 8-lane tile blocks with lane index
    // unchanged. Search all four legal phases; no approximate conversion.
    phase_found = 1'b0;
    phase_id = 8'd0;
    for (phase_index = 0; phase_index < 4; phase_index = phase_index + 1) begin
      phase_candidate = 1'b1;
      for (map_index = 0; map_index < 32; map_index = map_index + 1) begin
        destination_tile = map_index / 8;
        expected_source_tile = (phase_index + 4 - destination_tile) % 4;
        if (vmodel_permute_instruction_i[240+map_index*5 +: 5] !=
            expected_source_tile*8 + (map_index % 8))
          phase_candidate = 1'b0;
      end
      if (phase_candidate && !phase_found) begin
        phase_found = 1'b1;
        phase_id = phase_index;
      end
    end
    map_legal = phase_found;

    if (vmodel_permute_valid_i) begin
      if ((vmodel_permute_instruction_i[1:0] != SXM_PERMUTE) ||
          (vmodel_permute_instruction_i[6 +: 5] != 5'd16) ||
          (vmodel_permute_instruction_i[11 +: 5] != 5'd16) ||
          !src_legal || !dst_legal || !map_legal ||
          (output_tile > SXM_OUTPUT_TILE_ALL)) begin
        command_fault_o = 1'b1;
      end else begin
        native_permute_command_o[1:0] = 2'd1;
        native_permute_command_o[2] = ~src_direction;
        native_permute_command_o[7:3] = src_base;
        native_permute_command_o[8] = ~dst_direction;
        native_permute_command_o[13:9] = dst_base;
        native_permute_command_o[21:18] = 4'd8;
        native_permute_command_o[24:22] = output_tile;
        native_permute_command_o[32:25] = phase_id;
        native_permute_valid_o = 1'b1;
      end
    end
  end
endmodule
