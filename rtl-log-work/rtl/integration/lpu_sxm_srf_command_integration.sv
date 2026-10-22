`timescale 1ns/1ps

// Phase 3B command-path variant.  Phase 3A remains unchanged: this wrapper
// only translates VMODEL 416-bit SXM controls into native 96-bit controls.
module lpu_sxm_srf_command_integration #(
  parameter integer COLUMNS = 16,
  parameter integer SUPERLANES = 4,
  parameter integer STREAMS = 32,
  parameter integer SEGMENT_BITS = 64,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer LOCAL_CONSUMERS = 2
) (
  input logic clk_i,
  input logic rst_ni,
  input logic vmodel_transpose_valid_i,
  input logic [415:0] vmodel_transpose_instruction_i,
  input logic vmodel_permute_valid_i,
  input logic [415:0] vmodel_permute_instruction_i,
  input logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i,
  input logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_i,
  output logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_o,
  output logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_o,
  output logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o,
  output logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] state_data_o,
  output logic [2*COLUMNS*SUPERLANES-1:0] srf_collision_o,
  output logic [2*COLUMNS*SUPERLANES-1:0] srf_invalid_consume_o,
  output logic sxm_fault_valid_o,
  output logic [SUPERLANES-1:0] sxm_transpose_input_invalid_o,
  output logic [SUPERLANES-1:0] sxm_transpose_buffer_full_o,
  output logic sxm_permute_phase_fault_o,
  output logic sxm_permute_selector_fault_o,
  output logic sxm_permute_buffer_not_ready_o,
  output logic sxm_busy_o,
  output logic sxm_command_fault_o
);
  logic native_transpose_valid;
  logic [95:0] native_transpose_command;
  logic native_permute_valid;
  logic [95:0] native_permute_command;

  lpu_sxm_command_adapter u_command_adapter (
    .vmodel_transpose_valid_i, .vmodel_transpose_instruction_i,
    .vmodel_permute_valid_i, .vmodel_permute_instruction_i,
    .native_transpose_valid_o(native_transpose_valid),
    .native_transpose_command_o(native_transpose_command),
    .native_permute_valid_o(native_permute_valid),
    .native_permute_command_o(native_permute_command),
    .command_fault_o(sxm_command_fault_o)
  );

  lpu_sxm_srf_integration #(
    .COLUMNS(COLUMNS), .SUPERLANES(SUPERLANES), .STREAMS(STREAMS),
    .SEGMENT_BITS(SEGMENT_BITS), .LOCAL_PRODUCERS(LOCAL_PRODUCERS),
    .LOCAL_CONSUMERS(LOCAL_CONSUMERS)
  ) u_phase3a (
    .clk_i, .rst_ni,
    .transpose_cmd_valid_i(native_transpose_valid),
    .transpose_cmd_i(native_transpose_command),
    .permute_cmd_valid_i(native_permute_valid),
    .permute_cmd_i(native_permute_command),
    .boundary_valid_i, .boundary_data_i, .boundary_valid_o, .boundary_data_o,
    .state_valid_o, .state_data_o, .srf_collision_o, .srf_invalid_consume_o,
    .sxm_fault_valid_o, .sxm_transpose_input_invalid_o,
    .sxm_transpose_buffer_full_o, .sxm_permute_phase_fault_o,
    .sxm_permute_selector_fault_o, .sxm_permute_buffer_not_ready_o, .sxm_busy_o
  );
endmodule
