`timescale 1ns/1ps

// Phase 3A standalone SXM/SRF data-plane integration.  SRF slot0 is tied off;
// SXM owns fixed producer/consumer slot1.  Native 96-bit SXM commands are
// driven directly by the testbench in this phase.
module lpu_sxm_srf_integration #(
  parameter integer COLUMNS = 16,
  parameter integer SUPERLANES = 4,
  parameter integer STREAMS = 32,
  parameter integer SEGMENT_BITS = 64,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer LOCAL_CONSUMERS = 2
) (
  input logic clk_i,
  input logic rst_ni,
  input logic transpose_cmd_valid_i,
  input logic [95:0] transpose_cmd_i,
  input logic permute_cmd_valid_i,
  input logic [95:0] permute_cmd_i,
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
  output logic sxm_busy_o
);
  localparam integer SXM_ACTIVE_STREAMS = 16;
  logic [SUPERLANES*SXM_ACTIVE_STREAMS*6-1:0] sxm_sr_read_req;
  logic [SUPERLANES*SXM_ACTIVE_STREAMS-1:0] sxm_sr_read_valid;
  logic [SUPERLANES*SXM_ACTIVE_STREAMS*SEGMENT_BITS-1:0] sxm_sr_read_data;
  logic [SUPERLANES*SXM_ACTIVE_STREAMS-1:0] sxm_sr_consume;
  logic [SUPERLANES*SXM_ACTIVE_STREAMS-1:0] sxm_sr_write_valid;
  logic [SXM_ACTIVE_STREAMS*6-1:0] sxm_sr_write_sel;
  logic [SUPERLANES*SXM_ACTIVE_STREAMS*SEGMENT_BITS-1:0] sxm_sr_write_data;
  logic [2*COLUMNS*SUPERLANES*LOCAL_CONSUMERS*STREAMS-1:0] srf_consume;
  logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS-1:0] srf_inject_valid;
  logic [2*COLUMNS*SUPERLANES*LOCAL_PRODUCERS*STREAMS*SEGMENT_BITS-1:0]
    srf_inject_data;

  ftlpu_sr_hemisphere_fabric #(
    .COLUMNS(COLUMNS), .SUPERLANES(SUPERLANES), .STREAMS(STREAMS),
    .LANES(8), .DATA_BITS(8), .LOCAL_PRODUCERS(LOCAL_PRODUCERS),
    .LOCAL_CONSUMERS(LOCAL_CONSUMERS)
  ) u_srf (
    .clk_i, .rst_ni, .boundary_valid_i, .boundary_data_i,
    .boundary_valid_o, .boundary_data_o, .inject_valid_i(srf_inject_valid),
    .inject_data_i(srf_inject_data), .consume_i(srf_consume),
    .collision_o(srf_collision_o), .invalid_consume_o(srf_invalid_consume_o),
    .state_valid_o, .state_data_o
  );

  sxm_slice u_sxm (
    .clk_i, .rst_ni, .transpose_cmd_valid_i, .transpose_cmd_i,
    .permute_cmd_valid_i, .permute_cmd_i, .sr_read_req_o(sxm_sr_read_req),
    .sr_read_valid_i(sxm_sr_read_valid), .sr_read_data_i(sxm_sr_read_data),
    .sr_consume_o(sxm_sr_consume), .sr_write_valid_o(sxm_sr_write_valid),
    .sr_write_sel_o(sxm_sr_write_sel), .sr_write_data_o(sxm_sr_write_data),
    .fault_valid_o(sxm_fault_valid_o),
    .transpose_input_invalid_o(sxm_transpose_input_invalid_o),
    .transpose_buffer_full_o(sxm_transpose_buffer_full_o),
    .permute_phase_fault_o(sxm_permute_phase_fault_o),
    .permute_selector_fault_o(sxm_permute_selector_fault_o),
    .permute_buffer_not_ready_o(sxm_permute_buffer_not_ready_o),
    .busy_o(sxm_busy_o)
  );

  lpu_sxm_srf_adapter #(
    .COLUMNS(COLUMNS), .SUPERLANES(SUPERLANES), .STREAMS(STREAMS),
    .SEGMENT_BITS(SEGMENT_BITS), .LOCAL_PRODUCERS(LOCAL_PRODUCERS),
    .LOCAL_CONSUMERS(LOCAL_CONSUMERS), .SXM_ACTIVE_STREAMS(SXM_ACTIVE_STREAMS),
    .SXM_CONSUMER_SLOT(1), .SXM_PRODUCER_SLOT(1)
  ) u_adapter (
    .srf_state_valid_i(state_valid_o), .srf_state_data_i(state_data_o),
    .sxm_sr_read_req_i(sxm_sr_read_req), .sxm_sr_read_valid_o(sxm_sr_read_valid),
    .sxm_sr_read_data_o(sxm_sr_read_data), .sxm_sr_consume_i(sxm_sr_consume),
    .sxm_sr_write_valid_i(sxm_sr_write_valid), .sxm_sr_write_sel_i(sxm_sr_write_sel),
    .sxm_sr_write_data_i(sxm_sr_write_data), .srf_consume_o(srf_consume),
    .srf_inject_valid_o(srf_inject_valid), .srf_inject_data_o(srf_inject_data)
  );
endmodule
