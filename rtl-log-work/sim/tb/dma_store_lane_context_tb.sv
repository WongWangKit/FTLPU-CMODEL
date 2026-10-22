`timescale 1ns/1ps

module dma_store_lane_context_tb;
  localparam integer P_LANE_INDEX = 3;
  localparam integer P_VECTOR_COUNT_BITS = 20;
  localparam integer P_STRIDE_BITS = 32;

  logic clk_i;
  logic rst_ni;
  logic desc_valid_i;
  logic desc_direction_i;
  logic [63:0] desc_external_base_addr_i;
  logic [P_VECTOR_COUNT_BITS-1:0] desc_vector_count_minus_1_i;
  logic [P_STRIDE_BITS-1:0] desc_stride_bytes_i;
  logic desc_accept_o;
  logic retire_i;
  logic active_valid_o;
  logic active_direction_o;
  logic [2:0] active_lane_index_o;
  logic [63:0] active_external_base_addr_o;
  logic [P_VECTOR_COUNT_BITS-1:0] active_vector_count_minus_1_o;
  logic [P_STRIDE_BITS-1:0] active_stride_bytes_o;
  integer errors;

  dma_store_lane_context #(
    .P_LANE_INDEX(P_LANE_INDEX),
    .P_VECTOR_COUNT_BITS(P_VECTOR_COUNT_BITS),
    .P_STRIDE_BITS(P_STRIDE_BITS)
  ) dut (
    .clk_i,
    .rst_ni,
    .desc_valid_i,
    .desc_direction_i,
    .desc_external_base_addr_i,
    .desc_vector_count_minus_1_i,
    .desc_stride_bytes_i,
    .desc_accept_o,
    .retire_i,
    .active_valid_o,
    .active_direction_o,
    .active_lane_index_o,
    .active_external_base_addr_o,
    .active_vector_count_minus_1_o,
    .active_stride_bytes_o
  );

  always #5 clk_i = ~clk_i;

  task automatic drive_idle;
    begin
      desc_valid_i = 1'b0;
      desc_direction_i = 1'b0;
      desc_external_base_addr_i = '0;
      desc_vector_count_minus_1_i = '0;
      desc_stride_bytes_i = '0;
      retire_i = 1'b0;
    end
  endtask

  task automatic drive_store_desc(
    input logic [63:0] base_addr,
    input logic [P_VECTOR_COUNT_BITS-1:0] vector_count_minus_1,
    input logic [P_STRIDE_BITS-1:0] stride_bytes
  );
    begin
      desc_valid_i = 1'b1;
      desc_direction_i = 1'b1;
      desc_external_base_addr_i = base_addr;
      desc_vector_count_minus_1_i = vector_count_minus_1;
      desc_stride_bytes_i = stride_bytes;
    end
  endtask

  task automatic check_active(
    input logic [63:0] base_addr,
    input logic [P_VECTOR_COUNT_BITS-1:0] vector_count_minus_1,
    input logic [P_STRIDE_BITS-1:0] stride_bytes,
    input [8*24-1:0] label
  );
    begin
      if (!active_valid_o || !active_direction_o ||
          (active_lane_index_o != P_LANE_INDEX[2:0]) ||
          (active_external_base_addr_o != base_addr) ||
          (active_vector_count_minus_1_o != vector_count_minus_1) ||
          (active_stride_bytes_o != stride_bytes)) begin
        $display("ERROR %0s active descriptor mismatch", label);
        errors = errors + 1;
      end
    end
  endtask

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b0;
    errors = 0;
    drive_idle();

    // Reset does not expose a spurious active descriptor.
    #1;
    if (active_valid_o || !desc_accept_o ||
        (active_lane_index_o != P_LANE_INDEX[2:0])) begin
      $display("ERROR reset state mismatch");
      errors = errors + 1;
    end
    @(posedge clk_i);
    #1;
    $display("DMA_CONTEXT_RESET PASS");
    rst_ni = 1'b1;

    // Idle context accepts decoded Store semantics and captures them at edge.
    @(negedge clk_i);
    drive_store_desc(64'h0000_0000_0010_0020, 20'd6, 32'd32);
    if (!desc_accept_o) begin
      $display("ERROR idle context did not accept descriptor");
      errors = errors + 1;
    end
    @(posedge clk_i);
    #1;
    check_active(64'h0000_0000_0010_0020, 20'd6, 32'd32, "accept");
    $display("DMA_CONTEXT_ACCEPT PASS");

    // A busy context holds its state and deasserts accept; valid may wait.
    @(negedge clk_i);
    drive_store_desc(64'h0000_0000_00AA_0000, 20'd1, 32'd64);
    if (desc_accept_o) begin
      $display("ERROR busy context accepted overwrite");
      errors = errors + 1;
    end
    @(posedge clk_i);
    #1;
    check_active(64'h0000_0000_0010_0020, 20'd6, 32'd32, "hold");
    $display("DMA_CONTEXT_HOLD PASS");

    // retire_i is an explicit temporary lifecycle hook, not a DMA completion.
    @(negedge clk_i);
    drive_idle();
    retire_i = 1'b1;
    @(posedge clk_i);
    #1;
    if (active_valid_o || !desc_accept_o) begin
      $display("ERROR retirement did not clear active context");
      errors = errors + 1;
    end
    $display("DMA_CONTEXT_RETIRE PASS");

    // A new descriptor is accepted after retirement.
    @(negedge clk_i);
    drive_store_desc(64'h0000_0000_0020_0040, 20'd2, 32'd128);
    @(posedge clk_i);
    #1;
    check_active(64'h0000_0000_0020_0040, 20'd2, 32'd128, "reaccept");
    $display("DMA_CONTEXT_REACCEPT PASS");

    // Contract supports same-edge retirement of old state and replacement.
    @(negedge clk_i);
    drive_store_desc(64'h0000_0000_0030_0060, 20'd3, 32'd256);
    retire_i = 1'b1;
    if (!desc_accept_o) begin
      $display("ERROR retire-and-replace was not accepted");
      errors = errors + 1;
    end
    @(posedge clk_i);
    #1;
    check_active(64'h0000_0000_0030_0060, 20'd3, 32'd256,
                 "retire_replace");
    $display("DMA_CONTEXT_RETIRE_REPLACE PASS");

    @(negedge clk_i);
    drive_idle();
    if (errors == 0)
      $display("DMA_STORE_LANE_CONTEXT TEST_PASS");
    else
      $display("DMA_STORE_LANE_CONTEXT TEST_FAIL errors=%0d", errors);
    $finish;
  end
endmodule
