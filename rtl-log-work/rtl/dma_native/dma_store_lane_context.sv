`timescale 1ns/1ps

// Single logical DMA Store-lane control context.
//
// This block accepts already-decoded descriptor semantics.  It intentionally
// does not decode a 192-bit descriptor slot, define an ICU codec, generate
// addresses, or issue external-storage requests.  The fixed P_LANE_INDEX
// identifies the logical DMA lane; it is distinct from an ordinary SRF stream
// selector even when a static schedule uses the same numeric value.
module dma_store_lane_context #(
  parameter integer P_LANE_INDEX       = 0,
  parameter integer P_VECTOR_COUNT_BITS = 20,
  parameter integer P_STRIDE_BITS       = 32
) (
  input  logic                            clk_i,
  input  logic                            rst_ni,

  // Decoded descriptor semantic fields. direction=1 denotes Store in the
  // current architectural convention; this generic holder retains it without
  // implementing either a Store or a Load data path.
  input  logic                            desc_valid_i,
  input  logic                            desc_direction_i,
  input  logic [63:0]                     desc_external_base_addr_i,
  input  logic [P_VECTOR_COUNT_BITS-1:0]  desc_vector_count_minus_1_i,
  input  logic [P_STRIDE_BITS-1:0]        desc_stride_bytes_i,
  output logic                            desc_accept_o,

  // Temporary decoded-context lifecycle hook.  This is not an external write
  // completion, gather completion, FIFO pop, or vector-sink acceptance.
  input  logic                            retire_i,

  output logic                            active_valid_o,
  output logic                            active_direction_o,
  output logic [2:0]                      active_lane_index_o,
  output logic [63:0]                     active_external_base_addr_o,
  output logic [P_VECTOR_COUNT_BITS-1:0]  active_vector_count_minus_1_o,
  output logic [P_STRIDE_BITS-1:0]        active_stride_bytes_o
);
  logic active_valid_q;
  logic active_direction_q;
  logic [63:0] active_external_base_addr_q;
  logic [P_VECTOR_COUNT_BITS-1:0] active_vector_count_minus_1_q;
  logic [P_STRIDE_BITS-1:0] active_stride_bytes_q;
  logic desc_fire;

  // Same-edge retirement may replace the old descriptor.  This preserves a
  // one-descriptor-per-cycle turnover without allowing overwrite while busy.
  assign desc_accept_o = !active_valid_q || retire_i;
  assign desc_fire = desc_valid_i && desc_accept_o;

  assign active_valid_o = active_valid_q;
  assign active_direction_o = active_direction_q;
  assign active_lane_index_o = P_LANE_INDEX[2:0];
  assign active_external_base_addr_o = active_external_base_addr_q;
  assign active_vector_count_minus_1_o = active_vector_count_minus_1_q;
  assign active_stride_bytes_o = active_stride_bytes_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      active_valid_q <= 1'b0;
      active_direction_q <= 1'b0;
      active_external_base_addr_q <= '0;
      active_vector_count_minus_1_q <= '0;
      active_stride_bytes_q <= '0;
    end else if (desc_fire) begin
      active_valid_q <= 1'b1;
      active_direction_q <= desc_direction_i;
      active_external_base_addr_q <= desc_external_base_addr_i;
      active_vector_count_minus_1_q <= desc_vector_count_minus_1_i;
      active_stride_bytes_q <= desc_stride_bytes_i;
    end else if (active_valid_q && retire_i) begin
      active_valid_q <= 1'b0;
    end
  end

`ifndef SYNTHESIS
  initial begin
    if ((P_LANE_INDEX < 0) || (P_LANE_INDEX > 7) ||
        (P_VECTOR_COUNT_BITS <= 0) || (P_STRIDE_BITS <= 0))
      $fatal(1, "TEST_FAIL invalid DMA Store lane-context parameters");
  end
`endif
endmodule
