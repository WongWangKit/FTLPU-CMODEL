module lpu_vxm_fp16_stream_groups #(
  parameter integer PHYSICAL_STAGE = 0,
  parameter integer STREAMS = lpu_pkg::STREAMS_PER_DIRECTION,
  parameter integer LANES = lpu_pkg::LANES_PER_TILE,
  parameter integer CONTAINER_WIDTH = 32
) (
  input  logic [STREAMS-1:0] stream_valid_i,
  input  logic [STREAMS*LANES*8-1:0] stream_data_i,

  output logic                             lhs_valid_o,
  output logic [LANES*CONTAINER_WIDTH-1:0] lhs_data_o,
  output logic                             rhs_valid_o,
  output logic [LANES*CONTAINER_WIDTH-1:0] rhs_data_o
);
  // There are eight fixed two-stage blocks in one 16-stage lane. Each block
  // owns two FP16 input groups, and each group owns two adjacent byte-streams.
  localparam integer BLOCK = PHYSICAL_STAGE / 2;
  localparam integer LHS_BYTE0_STREAM = BLOCK * 4;
  localparam integer LHS_BYTE1_STREAM = BLOCK * 4 + 1;
  localparam integer RHS_BYTE0_STREAM = BLOCK * 4 + 2;
  localparam integer RHS_BYTE1_STREAM = BLOCK * 4 + 3;

  integer lane;
  always_comb begin
    lhs_valid_o = stream_valid_i[LHS_BYTE0_STREAM] &&
      stream_valid_i[LHS_BYTE1_STREAM];
    rhs_valid_o = stream_valid_i[RHS_BYTE0_STREAM] &&
      stream_valid_i[RHS_BYTE1_STREAM];
    lhs_data_o = '0;
    rhs_data_o = '0;
    for (lane = 0; lane < LANES; lane = lane + 1) begin
      lhs_data_o[lane*CONTAINER_WIDTH +: 16] = {
        stream_data_i[(LHS_BYTE1_STREAM*LANES+lane)*8 +: 8],
        stream_data_i[(LHS_BYTE0_STREAM*LANES+lane)*8 +: 8]
      };
      rhs_data_o[lane*CONTAINER_WIDTH +: 16] = {
        stream_data_i[(RHS_BYTE1_STREAM*LANES+lane)*8 +: 8],
        stream_data_i[(RHS_BYTE0_STREAM*LANES+lane)*8 +: 8]
      };
    end
  end

  initial begin
    if ((PHYSICAL_STAGE < 0) || (PHYSICAL_STAGE >= 16))
      $error("VXM physical stage must be in [0, 15]");
    if ((CONTAINER_WIDTH < 16) || (STREAMS < 32))
      $error("VXM FP16 stream grouping requires 32 streams and >=16 data bits");
  end
endmodule
