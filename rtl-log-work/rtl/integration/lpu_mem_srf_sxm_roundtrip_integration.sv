`timescale 1ns/1ps
// Phase 4A-2 adds no data-path state: it is the explicit full-round-trip
// variant of the validated Phase 4A-1 shared-SRF integration.
module lpu_mem_srf_sxm_roundtrip_integration #(
  parameter integer MEM_SLICES=16, MEM_SLICES_PER_GROUP=4, MEM_DEPTH_ROWS=16,
  parameter integer COLUMNS=16, SUPERLANES=4, STREAMS=32, SEGMENT_BITS=64,
  parameter integer LOCAL_PRODUCERS=2, LOCAL_CONSUMERS=2
) (
  input logic clk_i,rst_ni,input logic [MEM_SLICES*2-1:0] vmodel_mem_issue_valid_i,
  input logic [MEM_SLICES*2*47-1:0] vmodel_mem_issue_instruction_i,
  output logic [MEM_SLICES*2-1:0] native_mem_issue_valid_o,mem_command_fault_o,
  output logic [MEM_SLICES*2*32-1:0] native_mem_issue_o,
  input logic vmodel_transpose_valid_i,input logic [415:0] vmodel_transpose_instruction_i,
  input logic vmodel_permute_valid_i,input logic [415:0] vmodel_permute_instruction_i,
  input logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i,
  input logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_i,
  output logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o,
  output logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] state_data_o,
  output logic [2*COLUMNS*SUPERLANES-1:0] srf_collision_o,srf_invalid_consume_o,
  output logic [MEM_SLICES*8-1:0] mem_producer_valid_o,mem_producer_direction_o,
  output logic [MEM_SLICES*8*SEGMENT_BITS-1:0] mem_producer_data_o,
  output logic [MEM_SLICES*8*5-1:0] mem_producer_stream_o,
  output logic [MEM_SLICES*8*4-1:0] mem_producer_boundary_o,
  output logic [(MEM_SLICES/MEM_SLICES_PER_GROUP+1)*256-1:0] mem_boundary_consume_o,
  output logic [MEM_SLICES*8-1:0] mem_internal_collision_o,
  output logic [MEM_SLICES*2-1:0] mem_bank_fault_valid_o,
  output logic mem_fault_valid_o,mem_busy_o,sxm_fault_valid_o,
  output logic [SUPERLANES-1:0] sxm_transpose_input_invalid_o,sxm_transpose_buffer_full_o,
  output logic sxm_permute_phase_fault_o,sxm_permute_selector_fault_o,
  output logic sxm_permute_buffer_not_ready_o,sxm_busy_o,sxm_command_fault_o
);
  lpu_mem_srf_sxm_turnaround_integration #(.MEM_SLICES(MEM_SLICES),.MEM_SLICES_PER_GROUP(MEM_SLICES_PER_GROUP),.MEM_DEPTH_ROWS(MEM_DEPTH_ROWS),.COLUMNS(COLUMNS),.SUPERLANES(SUPERLANES),.STREAMS(STREAMS),.SEGMENT_BITS(SEGMENT_BITS),.LOCAL_PRODUCERS(LOCAL_PRODUCERS),.LOCAL_CONSUMERS(LOCAL_CONSUMERS)) u_turnaround (.*);
endmodule
