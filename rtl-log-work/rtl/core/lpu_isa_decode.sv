module lpu_mem_instruction_decode (
  input  logic [46:0] instruction_i,
  output logic [2:0]  opcode_o,
  output logic [5:0]  stream_o,
  output logic [5:0]  map_or_write_stream_o,
  output logic [15:0] address_o,
  output logic [15:0] write_address_o,
  output logic        legal_o
);
  always_comb begin
    opcode_o              = instruction_i[2:0];
    stream_o              = instruction_i[8:3];
    map_or_write_stream_o = instruction_i[14:9];
    address_o             = instruction_i[30:15];
    write_address_o       = instruction_i[46:31];
    legal_o               = 1'b1;

    case (instruction_i[2:0])
      3'd0, 3'd1, 3'd3, 3'd4: legal_o = (instruction_i[46:31] == '0);
      3'd2: legal_o = (instruction_i[30:15] != instruction_i[46:31]);
      default: legal_o = 1'b0;
    endcase
  end
endmodule

module lpu_mxm_instruction_decode #(
  parameter integer ACCUMULATOR_BLOCK_COUNT =
    lpu_pkg::MXM_ACCUMULATOR_BLOCK_COUNT
) (
  input  logic [47:0] instruction_i,
  output logic [1:0]  opcode_o,
  output logic        weight_buffer_o,
  output logic [1:0]  weight_column_o,
  output logic        column_mode_o,
  output logic [2:0]  inner_column_o,
  output logic        weight_input_direct16_o,
  output logic [5:0]  activation_stream_base_o,
  output logic [5:0]  output_stream_base_o,
  output logic [12:0] accumulator_address_o,
  output logic [15:0] accumulator_row_stride_o,
  output logic        accumulator_destination_o,
  output logic        accumulator_clear_o,
  output logic        data_format_bf16_o,
  output logic        compute_mode_block8_o,
  output logic        legal_o
);
  localparam integer ACCUMULATOR_DEPTH = ACCUMULATOR_BLOCK_COUNT * 32;
  localparam integer BLOCK_ACCUMULATOR_DEPTH = ACCUMULATOR_BLOCK_COUNT * 4;

  always_comb begin
    opcode_o                    = instruction_i[1:0];
    weight_buffer_o             = instruction_i[2];
    weight_column_o             = instruction_i[4:3];
    column_mode_o               = instruction_i[5];
    inner_column_o              = instruction_i[8:6];
    weight_input_direct16_o     = instruction_i[9];
    activation_stream_base_o    = instruction_i[8:3];
    output_stream_base_o        = instruction_i[14:9];
    accumulator_address_o       = instruction_i[27:15];
    accumulator_row_stride_o    = instruction_i[43:28];
    accumulator_destination_o   = instruction_i[44];
    accumulator_clear_o         = instruction_i[1:0] == 2'd1
                                    ? !instruction_i[47]
                                    : instruction_i[28];
    data_format_bf16_o          = instruction_i[45];
    compute_mode_block8_o       = instruction_i[46];
    legal_o                     = 1'b1;

    case (instruction_i[1:0])
      2'd0: legal_o = (instruction_i[47:10] == '0) &&
                       (instruction_i[5] || (instruction_i[8:6] == '0));
      2'd1: legal_o =
        (instruction_i[8:3] <= (instruction_i[46] ? 6'd16 : 6'd30)) &&
        (instruction_i[14:9] <= (instruction_i[46] ? 6'd16 : 6'd28)) &&
        (instruction_i[43:28] != 16'd0) &&
        (instruction_i[27:15] < (instruction_i[46]
          ? BLOCK_ACCUMULATOR_DEPTH : ACCUMULATOR_DEPTH));
      2'd2: legal_o =
        (instruction_i[45:29] == '0) &&
        (instruction_i[47] == 1'b0) &&
        (instruction_i[8:2] == '0) &&
        (instruction_i[14:9] <= (instruction_i[46] ? 6'd0 : 6'd28)) &&
        (instruction_i[27:15] < (instruction_i[46]
          ? BLOCK_ACCUMULATOR_DEPTH : ACCUMULATOR_DEPTH));
      default: legal_o = 1'b0;
    endcase
  end
endmodule

module lpu_vxm_instruction_decode (
  input  logic [127:0] instruction_i,
  output logic [4:0]   opcode_o,
  output logic [2:0]   lhs_kind_o,
  output logic [5:0]   lhs_index_o,
  output logic [2:0]   rhs_kind_o,
  output logic [5:0]   rhs_index_o,
  output logic [31:0]  lhs_immediate_o,
  output logic [31:0]  rhs_immediate_o,
  output logic [1:0]   cast_target_o,
  output logic         output_valid_o,
  output logic [5:0]   output_stream_o,
  output logic         input_hemisphere_o,
  output logic         output_hemisphere_o,
  output logic         legal_o
);
  always_comb begin
    opcode_o            = instruction_i[4:0];
    lhs_kind_o          = instruction_i[7:5];
    lhs_index_o         = instruction_i[13:8];
    rhs_kind_o          = instruction_i[16:14];
    rhs_index_o         = instruction_i[22:17];
    cast_target_o       = instruction_i[24:23];
    output_valid_o      = instruction_i[25];
    output_stream_o     = instruction_i[31:26];
    lhs_immediate_o     = instruction_i[63:32];
    rhs_immediate_o     = instruction_i[95:64];
    input_hemisphere_o  = instruction_i[96];
    output_hemisphere_o = instruction_i[97];
    legal_o              = (instruction_i[127:98] == '0) &&
                           (instruction_i[4:0] <= 5'd15);
  end
endmodule

module lpu_sxm_instruction_header_decode (
  input  logic [415:0] instruction_i,
  output logic [1:0]   opcode_o,
  output logic [1:0]   shift_source_o,
  output logic [1:0]   shift_distance_o,
  output logic [4:0]   source_count_o,
  output logic [4:0]   destination_count_o,
  output logic         legal_o
);
  always_comb begin
    opcode_o            = instruction_i[1:0];
    shift_source_o      = instruction_i[3:2];
    shift_distance_o    = instruction_i[5:4];
    source_count_o      = instruction_i[10:6];
    destination_count_o = instruction_i[15:11];
    legal_o = (instruction_i[10:6] <= 5'd16) &&
              (instruction_i[15:11] <= 5'd16) &&
              (instruction_i[415:400] == '0);
  end
endmodule
