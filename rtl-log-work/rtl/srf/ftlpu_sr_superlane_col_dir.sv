`timescale 1ns/1ps

module ftlpu_sr_superlane_col_dir #(
  parameter integer STREAMS        = 32,
  parameter integer LANES          = 8,
  parameter integer DATA_BITS      = 8,
  parameter integer LOCAL_PRODUCERS = 2,
  parameter integer LOCAL_CONSUMERS = 2
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic [STREAMS-1:0] upstream_valid_i,
  input  logic [STREAMS*LANES*DATA_BITS-1:0] upstream_data_i,

  output logic [STREAMS-1:0] state_valid_o,
  output logic [STREAMS*LANES*DATA_BITS-1:0] state_data_o,

  input  logic [LOCAL_CONSUMERS*STREAMS-1:0] consume_i,
  output logic [STREAMS-1:0] downstream_valid_o,
  output logic [STREAMS*LANES*DATA_BITS-1:0] downstream_data_o,

  input  logic [LOCAL_PRODUCERS*STREAMS-1:0] inject_valid_i,
  input  logic [LOCAL_PRODUCERS*STREAMS*LANES*DATA_BITS-1:0]
    inject_data_i,

  output logic collision_o,
  output logic invalid_consume_o
);
  localparam integer SEGMENT_BITS = LANES * DATA_BITS;

  logic [STREAMS-1:0] valid_state_q;
  logic [STREAMS*SEGMENT_BITS-1:0] data_state_q;
  logic [STREAMS-1:0] next_valid;
  logic [STREAMS*SEGMENT_BITS-1:0] next_data;
  logic [STREAMS-1:0] consume_any;

  integer stream;
  integer producer;
  integer consumer;
  integer candidate_count;
  logic [SEGMENT_BITS-1:0] candidate_data;

  // A stream segment is the atomic state item: all eight lanes share one
  // valid bit and are consumed or injected together. Multiple next-state
  // producers are an illegal static schedule. The leaf reports the collision
  // and commits an invalid segment; it does not select a winner or retry.
  always_comb begin
    next_valid = '0;
    next_data = '0;
    consume_any = '0;
    collision_o = 1'b0;
    invalid_consume_o = 1'b0;
    candidate_count = 0;
    candidate_data = '0;

    for (stream = 0; stream < STREAMS; stream = stream + 1) begin
      candidate_count = 0;
      candidate_data = '0;

      if (upstream_valid_i[stream]) begin
        candidate_count = 1;
        candidate_data =
          upstream_data_i[stream*SEGMENT_BITS +: SEGMENT_BITS];
      end

      for (producer = 0; producer < LOCAL_PRODUCERS;
           producer = producer + 1) begin
        if (inject_valid_i[producer*STREAMS + stream]) begin
          if (candidate_count == 0)
            candidate_data = inject_data_i[
              (producer*STREAMS + stream)*SEGMENT_BITS +: SEGMENT_BITS];
          candidate_count = candidate_count + 1;
        end
      end

      if (candidate_count == 1) begin
        next_valid[stream] = 1'b1;
        next_data[stream*SEGMENT_BITS +: SEGMENT_BITS] = candidate_data;
      end else if (candidate_count >= 2) begin
        collision_o = 1'b1;
      end

      for (consumer = 0; consumer < LOCAL_CONSUMERS;
           consumer = consumer + 1) begin
        if (consume_i[consumer*STREAMS + stream]) begin
          consume_any[stream] = 1'b1;
          if (!valid_state_q[stream])
            invalid_consume_o = 1'b1;
        end
      end
    end
  end

  // Only the leaf owns SRF payload and valid state. One rising edge advances
  // a segment by exactly one column; wrappers add no state or latency.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      valid_state_q <= '0;
      data_state_q <= '0;
    end else begin
      valid_state_q <= next_valid;
      data_state_q <= next_data;
    end
  end

  always_comb begin
    state_valid_o = valid_state_q;
    state_data_o = data_state_q;
    downstream_valid_o = valid_state_q & ~consume_any;
    downstream_data_o = data_state_q;
  end
endmodule
