`timescale 1ns/1ps

// THIS IS CURRENT  SXM ENTRY DECODE.
// FINAL ISA ENCODING IS NOT FROZEN.
//
// Keeping this stateless decode isolated prevents future ISA-field movement
// from changing the Transpose, Result Buffer, or Permute datapaths.
module sxm_command_decode (
    input  wire [95:0] cmd_i,

    output wire [1:0]  opcode_o,
    output wire        src_direction_o,
    output wire [4:0]  src_base_o,
    output wire        dst_direction_o,
    output wire [4:0]  dst_base_o,
    output wire [3:0]  input_row_o,
    output wire [3:0]  output_row_o,
    output wire [2:0]  output_tile_o,
    output wire [7:0]  phase_id_o,
    output wire        reserved_nonzero_o
);

    assign opcode_o = cmd_i[1:0];
    assign src_direction_o = cmd_i[2];
    assign src_base_o = cmd_i[7:3];
    assign dst_direction_o = cmd_i[8];
    assign dst_base_o = cmd_i[13:9];
    assign input_row_o = cmd_i[17:14];
    assign output_row_o = cmd_i[21:18];
    assign output_tile_o = cmd_i[24:22];
    assign phase_id_o = cmd_i[32:25];
    assign reserved_nonzero_o = |cmd_i[95:33];

endmodule
