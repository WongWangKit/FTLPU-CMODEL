`timescale 1ns/1ps

// Completed-vector storage internal to C2C, not an SRF stall mechanism.
// DEPTH must be >= 2. No empty-to-output combinational fall-through.
module c2c_completed_fifo #(
  parameter integer DEPTH = 2
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic enq_valid_i,
  input  logic [255:0] enq_payload_i,
  input  logic deq_pop_i,
  output logic deq_valid_o,
  output logic [255:0] deq_payload_o,
  output logic full_o,
  output logic empty_o,
  output logic [$clog2(DEPTH+1)-1:0] count_o
);
  localparam integer PTR_BITS = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
  localparam integer COUNT_BITS = $clog2(DEPTH+1);

  // Each entry is one complete 256-bit payload; no TX stream metadata.
  logic [255:0] entry_mem [0:DEPTH-1];
  logic [PTR_BITS-1:0] read_ptr_q, write_ptr_q;
  logic [COUNT_BITS-1:0] count_q;
  logic enq_fire, deq_fire;

  function automatic [PTR_BITS-1:0] next_ptr(input [PTR_BITS-1:0] ptr);
    if (ptr == PTR_BITS'(DEPTH-1)) next_ptr = '0;
    else next_ptr = ptr + 1'b1;
  endfunction

  assign count_o = count_q;
  assign empty_o = (count_q == 0);
  assign full_o = (count_q == COUNT_BITS'(DEPTH));
  assign deq_valid_o = !empty_o;
  // Head is qualified only by deq_valid_o; invalid payload is unspecified.
  assign deq_payload_o = entry_mem[read_ptr_q];

  // An empty pop does not consume an entry arriving at this same edge.
  // A full pop frees a slot at this edge, so the paired push is accepted.
  assign deq_fire = rst_ni && deq_pop_i && !empty_o;
  assign enq_fire = rst_ni && enq_valid_i && (!full_o || deq_fire);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      read_ptr_q <= '0;
      write_ptr_q <= '0;
      count_q <= '0;
    end else begin
      if (deq_fire) read_ptr_q <= next_ptr(read_ptr_q);
      if (enq_fire) write_ptr_q <= next_ptr(write_ptr_q);
      case ({enq_fire, deq_fire})
        2'b10: count_q <= count_q + 1'b1;
        2'b01: count_q <= count_q - 1'b1;
        default: count_q <= count_q;
      endcase
    end
  end

  // Reset controls validity via count/pointers; the memory needs no reset.
  // Consumers sample the old head before the edge, including full pop+push.
  always_ff @(posedge clk_i) begin
    if (enq_fire)
      entry_mem[write_ptr_q] <= enq_payload_i;
  end
endmodule
