module lpu_stream_stage #(
  parameter integer STREAMS = 32,
  parameter integer LANES   = 8,
  parameter integer BYTE_WIDTH = 8
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic [STREAMS-1:0] upstream_valid_i,
  input  logic [STREAMS*LANES*BYTE_WIDTH-1:0] upstream_data_i,
  input  logic [STREAMS-1:0] upstream_last_i,
  input  logic [STREAMS-1:0] consume_i,

  input  logic [STREAMS-1:0] producer_valid_i,
  input  logic [STREAMS*LANES*BYTE_WIDTH-1:0] producer_data_i,
  input  logic [STREAMS-1:0] producer_last_i,

  output logic [STREAMS-1:0] stage_valid_o,
  output logic [STREAMS*LANES*BYTE_WIDTH-1:0] stage_data_o,
  output logic [STREAMS-1:0] stage_last_o,
  output logic conflict_o
);
  localparam integer WORD_WIDTH = LANES * BYTE_WIDTH;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stage_valid_o <= '0;
      stage_data_o  <= '0;
      stage_last_o  <= '0;
      conflict_o    <= 1'b0;
    end else begin
      stage_valid_o <= '0;
      stage_last_o  <= '0;
      conflict_o    <= 1'b0;
      for (integer stream = 0; stream < STREAMS; stream++) begin
        if (producer_valid_i[stream]) begin
          stage_valid_o[stream] <= 1'b1;
          stage_data_o[stream*WORD_WIDTH +: WORD_WIDTH] <=
            producer_data_i[stream*WORD_WIDTH +: WORD_WIDTH];
          stage_last_o[stream] <= producer_last_i[stream];
          if (upstream_valid_i[stream] && !consume_i[stream] &&
              ((upstream_data_i[stream*WORD_WIDTH +: WORD_WIDTH] !=
                producer_data_i[stream*WORD_WIDTH +: WORD_WIDTH]) ||
               (upstream_last_i[stream] != producer_last_i[stream])))
            conflict_o <= 1'b1;
        end else if (upstream_valid_i[stream] && !consume_i[stream]) begin
          stage_valid_o[stream] <= 1'b1;
          stage_data_o[stream*WORD_WIDTH +: WORD_WIDTH] <=
            upstream_data_i[stream*WORD_WIDTH +: WORD_WIDTH];
          stage_last_o[stream] <= upstream_last_i[stream];
        end
      end
    end
  end
endmodule
