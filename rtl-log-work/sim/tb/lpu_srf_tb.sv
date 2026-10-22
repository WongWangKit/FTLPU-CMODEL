`timescale 1ns/1ps

module lpu_srf_tb;
  timeunit 1ns;
  timeprecision 1ps;

  localparam integer COLUMNS = 2;
  localparam integer SUPERLANES = 1;
  localparam integer STREAMS = 2;
  localparam integer LANES = 8;
  localparam integer DATA_BITS = 8;
  localparam integer PRODUCERS = 2;
  localparam integer CONSUMERS = 2;
  localparam integer SEGMENT_BITS = LANES * DATA_BITS;
  localparam logic [63:0] PATTERN = 64'h0807_0605_0403_0201;

  logic clk;
  logic rst_n;

  logic [STREAMS-1:0] east_input_valid;
  logic [STREAMS*SEGMENT_BITS-1:0] east_input_data;
  logic [STREAMS-1:0] east_output_valid;
  logic [STREAMS*SEGMENT_BITS-1:0] east_output_data;
  logic [COLUMNS*PRODUCERS*STREAMS-1:0] east_inject_valid;
  logic [COLUMNS*PRODUCERS*STREAMS*SEGMENT_BITS-1:0] east_inject_data;
  logic [COLUMNS*CONSUMERS*STREAMS-1:0] east_consume;
  logic [COLUMNS-1:0] east_collision;
  logic [COLUMNS-1:0] east_invalid_consume;
  logic [COLUMNS*STREAMS-1:0] east_state_valid;
  logic [COLUMNS*STREAMS*SEGMENT_BITS-1:0] east_state_data;

  logic [STREAMS-1:0] west_input_valid;
  logic [STREAMS*SEGMENT_BITS-1:0] west_input_data;
  logic [STREAMS-1:0] west_output_valid;
  logic [STREAMS*SEGMENT_BITS-1:0] west_output_data;
  logic [COLUMNS*PRODUCERS*STREAMS-1:0] west_inject_valid;
  logic [COLUMNS*PRODUCERS*STREAMS*SEGMENT_BITS-1:0] west_inject_data;
  logic [COLUMNS*CONSUMERS*STREAMS-1:0] west_consume;
  logic [COLUMNS-1:0] west_collision;
  logic [COLUMNS-1:0] west_invalid_consume;
  logic [COLUMNS*STREAMS-1:0] west_state_valid;
  logic [COLUMNS*STREAMS*SEGMENT_BITS-1:0] west_state_data;

  ftlpu_sr_direction_fabric #(
    .COLUMNS(COLUMNS),
    .SUPERLANES(SUPERLANES),
    .STREAMS(STREAMS),
    .LANES(LANES),
    .DATA_BITS(DATA_BITS),
    .LOCAL_PRODUCERS(PRODUCERS),
    .LOCAL_CONSUMERS(CONSUMERS),
    .DIRECTION(0)
  ) u_east (
    .clk_i(clk), .rst_ni(rst_n),
    .stream_valid_i(east_input_valid), .stream_data_i(east_input_data),
    .stream_valid_o(east_output_valid), .stream_data_o(east_output_data),
    .inject_valid_i(east_inject_valid), .inject_data_i(east_inject_data),
    .consume_i(east_consume), .collision_o(east_collision),
    .invalid_consume_o(east_invalid_consume),
    .state_valid_o(east_state_valid), .state_data_o(east_state_data)
  );

  ftlpu_sr_direction_fabric #(
    .COLUMNS(COLUMNS),
    .SUPERLANES(SUPERLANES),
    .STREAMS(STREAMS),
    .LANES(LANES),
    .DATA_BITS(DATA_BITS),
    .LOCAL_PRODUCERS(PRODUCERS),
    .LOCAL_CONSUMERS(CONSUMERS),
    .DIRECTION(1)
  ) u_west (
    .clk_i(clk), .rst_ni(rst_n),
    .stream_valid_i(west_input_valid), .stream_data_i(west_input_data),
    .stream_valid_o(west_output_valid), .stream_data_o(west_output_data),
    .inject_valid_i(west_inject_valid), .inject_data_i(west_inject_data),
    .consume_i(west_consume), .collision_o(west_collision),
    .invalid_consume_o(west_invalid_consume),
    .state_valid_o(west_state_valid), .state_data_o(west_state_data)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  task automatic clear_inputs;
    begin
      east_input_valid = '0;
      east_input_data = '0;
      east_inject_valid = '0;
      east_inject_data = '0;
      east_consume = '0;
      west_input_valid = '0;
      west_input_data = '0;
      west_inject_valid = '0;
      west_inject_data = '0;
      west_consume = '0;
    end
  endtask

  task automatic reset_dut;
    begin
      clear_inputs();
      rst_n = 1'b0;
      repeat (2) @(posedge clk);
      @(negedge clk);
      rst_n = 1'b1;
    end
  endtask

  initial begin
    rst_n = 1'b0;
    clear_inputs();

    // East entry is column 0; each subsequent column is one additional hop.
    reset_dut();
    @(negedge clk);
    east_input_valid[0] = 1'b1;
    east_input_data[0 +: SEGMENT_BITS] = PATTERN;
    @(posedge clk);
    #1ns;
    if (!east_state_valid[0] ||
        east_state_data[0 +: SEGMENT_BITS] !== PATTERN)
      $fatal(1, "East entry hop mismatch");
    @(negedge clk);
    east_input_valid = '0;
    @(posedge clk);
    #1ns;
    if (!east_state_valid[STREAMS] ||
        east_state_data[STREAMS*SEGMENT_BITS +: SEGMENT_BITS] !== PATTERN)
      $fatal(1, "East second hop mismatch");
    $display("SRF_EAST_HOP PASS");

    // West entry is the highest-numbered column and propagates toward zero.
    reset_dut();
    @(negedge clk);
    west_input_valid[1] = 1'b1;
    west_input_data[SEGMENT_BITS +: SEGMENT_BITS] = PATTERN;
    @(posedge clk);
    #1ns;
    if (!west_state_valid[COLUMNS*STREAMS-1] ||
        west_state_data[(COLUMNS*STREAMS-1)*SEGMENT_BITS +: SEGMENT_BITS]
          !== PATTERN)
      $fatal(1, "West entry hop mismatch");
    @(negedge clk);
    west_input_valid = '0;
    @(posedge clk);
    #1ns;
    if (!west_state_valid[1] ||
        west_state_data[SEGMENT_BITS +: SEGMENT_BITS] !== PATTERN)
      $fatal(1, "West second hop mismatch");
    $display("SRF_WEST_HOP PASS");

    // One valid bit covers all eight byte lanes in the 64-bit segment.
    reset_dut();
    @(negedge clk);
    east_input_valid[0] = 1'b1;
    east_input_data[0 +: SEGMENT_BITS] = PATTERN;
    @(posedge clk);
    #1ns;
    if (east_state_valid[0] !== 1'b1 ||
        east_state_data[0*SEGMENT_BITS + 0*8 +: 8] !== 8'h01 ||
        east_state_data[0*SEGMENT_BITS + 7*8 +: 8] !== 8'h08)
      $fatal(1, "Segment-valid or byte-lane mapping mismatch");
    $display("SRF_SEGMENT_VALID PASS");

    // Consume masks the complete current segment before the next leaf samples.
    @(negedge clk);
    east_input_valid = '0;
    east_consume[0] = 1'b1;
    #1ns;
    if (east_output_valid[0] !== 1'b0)
      $fatal(1, "Consume did not mask downstream valid");
    @(posedge clk);
    #1ns;
    if (east_state_valid[STREAMS])
      $fatal(1, "Consumed segment propagated to the next column");
    $display("SRF_CONSUME PASS");

    // A local producer commits one complete 64-bit next-state segment.
    reset_dut();
    @(negedge clk);
    east_inject_valid[1] = 1'b1;
    east_inject_data[SEGMENT_BITS +: SEGMENT_BITS] = PATTERN;
    @(posedge clk);
    #1ns;
    if (!east_state_valid[1] ||
        east_state_data[SEGMENT_BITS +: SEGMENT_BITS] !== PATTERN)
      $fatal(1, "Producer segment commit mismatch");
    $display("SRF_PRODUCER PASS");

    // Two producers for one stream report collision and commit no winner.
    reset_dut();
    @(negedge clk);
    east_inject_valid[0] = 1'b1;
    east_inject_valid[STREAMS] = 1'b1;
    east_inject_data[0 +: SEGMENT_BITS] = PATTERN;
    east_inject_data[STREAMS*SEGMENT_BITS +: SEGMENT_BITS] = ~PATTERN;
    #1ns;
    if (!east_collision[0])
      $fatal(1, "Producer collision was not detected");
    @(posedge clk);
    #1ns;
    if (east_state_valid[0])
      $fatal(1, "Illegal collision selected a producer winner");
    $display("SRF_COLLISION PASS");

    $display("========================================");
    $display("VMODEL_SRF_PORT TEST_PASS");
    $display("========================================");
    $finish;
  end
endmodule
