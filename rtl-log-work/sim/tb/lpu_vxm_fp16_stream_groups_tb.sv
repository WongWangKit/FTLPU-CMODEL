`timescale 1ns/1ps

module lpu_vxm_fp16_stream_groups_tb;
  localparam integer STREAMS = 32;
  localparam integer LANES = 8;

  logic [STREAMS-1:0] stream_valid;
  logic [STREAMS*LANES*8-1:0] stream_data;
  wire stage0_lhs_valid;
  wire stage0_rhs_valid;
  wire [LANES*32-1:0] stage0_lhs;
  wire [LANES*32-1:0] stage0_rhs;
  wire stage2_lhs_valid;
  wire stage2_rhs_valid;
  wire [LANES*32-1:0] stage2_lhs;
  wire [LANES*32-1:0] stage2_rhs;

  lpu_vxm_fp16_stream_groups #(.PHYSICAL_STAGE(0)) u_stage0 (
    .stream_valid_i(stream_valid),
    .stream_data_i(stream_data),
    .lhs_valid_o(stage0_lhs_valid),
    .lhs_data_o(stage0_lhs),
    .rhs_valid_o(stage0_rhs_valid),
    .rhs_data_o(stage0_rhs)
  );

  lpu_vxm_fp16_stream_groups #(.PHYSICAL_STAGE(2)) u_stage2 (
    .stream_valid_i(stream_valid),
    .stream_data_i(stream_data),
    .lhs_valid_o(stage2_lhs_valid),
    .lhs_data_o(stage2_lhs),
    .rhs_valid_o(stage2_rhs_valid),
    .rhs_data_o(stage2_rhs)
  );

  task automatic set_stream_byte(
    input integer stream,
    input integer lane,
    input logic [7:0] value
  );
    stream_data[(stream*LANES+lane)*8 +: 8] = value;
  endtask

  task automatic check_condition(input logic condition, input string message);
    if (!condition) $fatal(1, "%s", message);
  endtask

  integer lane;
  initial begin
    stream_valid = '0;
    stream_data = '0;
    for (lane = 0; lane < LANES; lane = lane + 1) begin
      // Stage 0/1 block: streams 0/1 form LHS; 2/3 form RHS.
      set_stream_byte(0, lane, 8'h00 + lane);
      set_stream_byte(1, lane, 8'h3c);
      set_stream_byte(2, lane, 8'h00 + lane);
      set_stream_byte(3, lane, 8'h40);
      // Stage 2/3 block: streams 4/5 and 6/7.
      set_stream_byte(4, lane, 8'h00 + lane);
      set_stream_byte(5, lane, 8'h42);
      set_stream_byte(6, lane, 8'h00 + lane);
      set_stream_byte(7, lane, 8'h44);
    end
    stream_valid[7:0] = 8'hff;
    #1;

    check_condition(stage0_lhs_valid && stage0_rhs_valid,
                    "stage 0 fixed groups were not valid");
    check_condition(stage2_lhs_valid && stage2_rhs_valid,
                    "stage 2 fixed groups were not valid");
    for (lane = 0; lane < LANES; lane = lane + 1) begin
      check_condition(stage0_lhs[lane*32 +: 32] == (32'h00003c00 + lane),
                      "stage 0 LHS byte assembly failed");
      check_condition(stage0_rhs[lane*32 +: 32] == (32'h00004000 + lane),
                      "stage 0 RHS byte assembly failed");
      check_condition(stage2_lhs[lane*32 +: 32] == (32'h00004200 + lane),
                      "stage 2 LHS byte assembly failed");
      check_condition(stage2_rhs[lane*32 +: 32] == (32'h00004400 + lane),
                      "stage 2 RHS byte assembly failed");
    end

    stream_valid[5] = 1'b0;
    #1;
    check_condition(!stage2_lhs_valid && stage2_rhs_valid,
                    "each FP16 group must require both byte streams");

    $display("LPU_VXM_FP16_STREAM_GROUPS_TB_PASS");
    $finish;
  end
endmodule
