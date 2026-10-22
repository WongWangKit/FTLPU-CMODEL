module lpu_mxm_slice #(
  parameter integer LOCAL_MXM_INDEX = 0,
  parameter integer ACCUMULATOR_BLOCK_COUNT =
    lpu_pkg::MXM_ACCUMULATOR_BLOCK_COUNT
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,
  input  logic load_issue_valid_i,
  input  logic [47:0] load_issue_instruction_i,
  input  logic dequant_issue_valid_i,
  input  logic [15:0] dequant_issue_instruction_i,
  input  logic compute_issue_valid_i,
  input  logic [47:0] compute_issue_instruction_i,
  input  logic [4*32-1:0] east_from_sxm_valid_i,
  input  logic [4*32*64-1:0] east_from_sxm_data_i,
  input  logic [4*32-1:0] external_west_valid_i,
  input  logic [4*32*64-1:0] external_west_data_i,
  output logic [4*32-1:0] east_external_valid_o,
  output logic [4*32*64-1:0] east_external_data_o,
  output logic [4*32-1:0] west_to_sxm_valid_o,
  output logic [4*32*64-1:0] west_to_sxm_data_o,
  output logic fault_o,
  output logic conflict_o
);
  logic [3:0] load_row_valid;
  logic [4*48-1:0] load_row_instruction;
  logic [3:0] dequant_row_valid;
  logic [4*16-1:0] dequant_row_instruction;
  logic [3:0] compute_row_valid;
  logic [4*48-1:0] compute_row_instruction;
  logic [4*32-1:0] load_consumed;
  logic [4*32-1:0] compute_consumed;
  logic [2*4*4*8*8*16-1:0] weight_bits;
  logic [2*4*4-1:0] cell_valid;
  logic result_valid;
  logic [8*32*32-1:0] result_values;
  logic [12:0] result_address;
  logic [5:0] result_stream_base;
  logic result_stream_destination;
  logic result_clear;
  logic result_block_mode;
  logic result_ready;
  logic [4*32-1:0] accumulator_produced_valid;
  logic [4*32*64-1:0] accumulator_produced_data;
  logic load_fault;
  logic compute_fault;
  logic accumulator_fault;
  logic conflict_d;

  lpu_mxm_control u_control (
    .clk_i,
    .rst_ni,
    .load_valid_i(load_issue_valid_i),
    .load_instruction_i(load_issue_instruction_i),
    .dequant_valid_i(dequant_issue_valid_i),
    .dequant_instruction_i(dequant_issue_instruction_i),
    .compute_valid_i(compute_issue_valid_i),
    .compute_instruction_i(compute_issue_instruction_i),
    .load_row_valid_o(load_row_valid),
    .load_row_instruction_o(load_row_instruction),
    .dequant_row_valid_o(dequant_row_valid),
    .dequant_row_instruction_o(dequant_row_instruction),
    .compute_row_valid_o(compute_row_valid),
    .compute_row_instruction_o(compute_row_instruction)
  );

  lpu_mxm_weight_buffer #(
    .LOCAL_MXM_INDEX(LOCAL_MXM_INDEX)
  ) u_weight_buffer (
    .clk_i,
    .rst_ni,
    .run_i,
    .load_row_valid_i(load_row_valid),
    .load_row_instruction_i(load_row_instruction),
    .dequant_row_valid_i(dequant_row_valid),
    .dequant_row_instruction_i(dequant_row_instruction),
    .east_valid_i(east_from_sxm_valid_i),
    .east_data_i(east_from_sxm_data_i),
    .east_consumed_o(load_consumed),
    .weight_bits_o(weight_bits),
    .cell_valid_o(cell_valid),
    .fault_o(load_fault)
  );

  lpu_mxm_compute #(
    .ACCUMULATOR_BLOCK_COUNT(ACCUMULATOR_BLOCK_COUNT)
  ) u_compute (
    .clk_i,
    .rst_ni,
    .run_i,
    .compute_row_valid_i(compute_row_valid),
    .compute_row_instruction_i(compute_row_instruction),
    .east_valid_i(east_from_sxm_valid_i),
    .east_data_i(east_from_sxm_data_i),
    .weight_bits_i(weight_bits),
    .cell_valid_i(cell_valid),
    .east_consumed_o(compute_consumed),
    .result_valid_o(result_valid),
    .result_values_o(result_values),
    .result_address_o(result_address),
    .result_stream_base_o(result_stream_base),
    .result_stream_destination_o(result_stream_destination),
    .result_clear_o(result_clear),
    .result_block_mode_o(result_block_mode),
    .result_ready_i(result_ready),
    .fault_o(compute_fault)
  );

  lpu_mxm_shared_accumulator #(
    .ACCUMULATOR_BLOCK_COUNT(ACCUMULATOR_BLOCK_COUNT)
  ) u_accumulator (
    .clk_i,
    .rst_ni,
    .run_i,
    .write_valid_i(result_valid),
    .write_block_mode_i(result_block_mode),
    .write_values_i(result_values),
    .write_address_i(result_address),
    .write_stream_base_i(result_stream_base),
    .write_stream_destination_i(result_stream_destination),
    .write_clear_i(result_clear),
    .write_ready_o(result_ready),
    .read_row_valid_i(compute_row_valid),
    .read_row_instruction_i(compute_row_instruction),
    .west_valid_o(accumulator_produced_valid),
    .west_data_o(accumulator_produced_data),
    .fault_o(accumulator_fault)
  );

  always_comb begin
    east_external_valid_o = east_from_sxm_valid_i &
      ~(load_consumed | compute_consumed);
    east_external_data_o = east_from_sxm_data_i;
    west_to_sxm_valid_o = external_west_valid_i;
    west_to_sxm_data_o = external_west_data_i;
    conflict_d = 1'b0;
    if (|(load_consumed & compute_consumed))
      conflict_d = 1'b1;
    for (integer cell_index = 0; cell_index < 4*32; cell_index++) begin
      if (accumulator_produced_valid[cell_index]) begin
        if (west_to_sxm_valid_o[cell_index])
          conflict_d = 1'b1;
        else begin
          west_to_sxm_valid_o[cell_index] = 1'b1;
          west_to_sxm_data_o[cell_index*64 +: 64] =
            accumulator_produced_data[cell_index*64 +: 64];
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      conflict_o <= 1'b0;
    else if (!run_i)
      conflict_o <= 1'b0;
    else
      conflict_o <= conflict_o | conflict_d;
  end

  assign fault_o = load_fault | compute_fault | accumulator_fault;
endmodule
