`timescale 1ns/1ps

// Complete received vectors only. Destination selection belongs to Receive.
// DEPTH >= 1. No empty-to-output combinational fall-through.
module c2c_rx_ready_fifo #(
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

  logic [255:0] payload_mem [0:DEPTH-1];
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
  assign deq_payload_o = payload_mem[read_ptr_q];

  // Sample the old head at the edge. A full pop permits a same-edge push;
  // an empty queue cannot pop the vector arriving at that edge.
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

  // Validity is carried by count/pointers; payload storage needs no reset.
  // Capacity must be respected by the normal-path source/static schedule.
  // This queue does not implement link credit or any SRF stall mechanism.
  always_ff @(posedge clk_i) begin
    if (enq_fire) payload_mem[write_ptr_q] <= enq_payload_i;
  end
endmodule
