`timescale 1ns/1ps

module lpu_c2c_tx_srf_adapter_tb;
  timeunit 1ns;
  timeprecision 1ps;

  localparam integer COLUMNS = 16;
  localparam integer SUPERLANES = 4;
  localparam integer STREAMS = 32;
  localparam integer SEGMENT_BITS = 64;
  localparam integer PRODUCERS = 1;
  localparam integer CONSUMERS = 2;
  localparam integer C2C_SLOT = 1;
  localparam integer TX_COLUMN = 13;
  localparam integer STATE_BITS = 2*COLUMNS*SUPERLANES*STREAMS;
  localparam integer INJECT_BITS =
    2*COLUMNS*SUPERLANES*PRODUCERS*STREAMS;
  localparam integer CONSUME_BITS =
    2*COLUMNS*SUPERLANES*CONSUMERS*STREAMS;

  logic clk;
  logic rst_n;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data;
  logic [INJECT_BITS-1:0] inject_valid;
  logic [INJECT_BITS*SEGMENT_BITS-1:0] inject_data;
  logic [CONSUME_BITS-1:0] srf_consume;
  logic [2*COLUMNS*SUPERLANES-1:0] collision;
  logic [2*COLUMNS*SUPERLANES-1:0] invalid_consume;
  logic [STATE_BITS-1:0] state_valid;
  logic [STATE_BITS*SEGMENT_BITS-1:0] state_data;

  logic tx_issue_valid;
  logic [2:0] tx_stream_index;
  logic [4*SEGMENT_BITS-1:0] tile_data;
  logic [3:0] tile_valid;
  logic [19:0] tile_stream_index;
  logic [3:0] gather_tile_consume;
  logic completed_valid;
  logic [4*SEGMENT_BITS-1:0] completed_payload;

  function automatic integer state_index(
    input integer direction, input integer column, input integer superlane,
    input integer stream
  );
    state_index = ((direction*COLUMNS + column)*SUPERLANES + superlane)*
                  STREAMS + stream;
  endfunction

  function automatic integer inject_index(
    input integer direction, input integer column, input integer superlane,
    input integer producer, input integer stream
  );
    inject_index = ((((direction*COLUMNS + column)*SUPERLANES + superlane)*
                    PRODUCERS + producer)*STREAMS) + stream;
  endfunction

  function automatic integer consume_index(
    input integer direction, input integer column, input integer superlane,
    input integer consumer, input integer stream
  );
    consume_index = ((((direction*COLUMNS + column)*SUPERLANES + superlane)*
                     CONSUMERS + consumer)*STREAMS) + stream;
  endfunction

  function automatic logic [63:0] pattern(
    input integer vector_id, input integer tile
  );
    pattern = 64'hc2c0_0000_0000_0000 |
              (64'(vector_id) << 8) | 64'(tile);
  endfunction

  task automatic fail(input string message);
    begin
      $display("TEST_FAIL: %s", message);
      $fatal(1, "%s", message);
    end
  endtask

  task automatic clear_drive;
    begin
      boundary_valid = '0;
      boundary_data = '0;
      inject_valid = '0;
      inject_data = '0;
      tx_issue_valid = 1'b0;
      tx_stream_index = '0;
    end
  endtask

  task automatic reset_dut;
    begin
      clear_drive();
      rst_n = 1'b0;
      repeat (2) @(posedge clk);
      @(negedge clk);
      rst_n = 1'b1;
      #1ns;
    end
  endtask

  task automatic put_inject(
    input integer superlane, input integer stream, input logic [63:0] data
  );
    integer index;
    begin
      index = inject_index(0, TX_COLUMN, superlane, 0, stream);
      inject_valid[index] = 1'b1;
      inject_data[index*SEGMENT_BITS +: SEGMENT_BITS] = data;
    end
  endtask

  task automatic check_cycle(
    input logic [3:0] expected_mask,
    input integer stream0, input integer stream1,
    input integer stream2, input integer stream3,
    input logic [63:0] data0, input logic [63:0] data1,
    input logic [63:0] data2, input logic [63:0] data3,
    input logic expected_completed,
    input logic [255:0] expected_payload
  );
    integer tile;
    integer expected_stream;
    integer consume_count;
    integer expected_consume_index;
    logic [63:0] expected_data;
    begin
      consume_count = 0;
      for (tile = 0; tile < CONSUME_BITS; tile = tile + 1)
        if (srf_consume[tile]) consume_count = consume_count + 1;

      for (tile = 0; tile < 4; tile = tile + 1) begin
        case (tile)
          0: begin expected_stream = stream0; expected_data = data0; end
          1: begin expected_stream = stream1; expected_data = data1; end
          2: begin expected_stream = stream2; expected_data = data2; end
          default: begin expected_stream = stream3; expected_data = data3; end
        endcase
        if (tile_valid[tile] !== expected_mask[tile])
          fail("Unexpected selected tile valid");
        if (expected_mask[tile]) begin
          if (tile_stream_index[tile*5 +: 5] !== expected_stream[4:0])
            fail("Selected tile stream mismatch");
          if (tile_data[tile*SEGMENT_BITS +: SEGMENT_BITS] !== expected_data)
            fail("Selected tile data mismatch");
          expected_consume_index =
            consume_index(0, TX_COLUMN, tile, C2C_SLOT, expected_stream);
          if (srf_consume[expected_consume_index] !== 1'b1)
            fail("Consume did not return to selected SRF coordinate");
        end
      end
      if (gather_tile_consume !== expected_mask)
        fail("Gather consume differs from selected tile valid");
      if (consume_count != (expected_mask[0] + expected_mask[1] +
                            expected_mask[2] + expected_mask[3]))
        fail("Consume bus contains an unintended coordinate");
      if (completed_valid !== expected_completed)
        fail("Completed-vector valid mismatch");
      if (expected_completed) begin
        if (completed_payload !== expected_payload)
          fail("Completed-vector packing mismatch");
      end
    end
  endtask

  // Prime one complete current-cycle diagonal tile set through legal SRF local
  // injection, then issue the current Send only after the SRF state commits.
  task automatic run_cycle(
    input logic issue_valid, input integer issue_stream,
    input logic [3:0] expected_mask,
    input integer stream0, input integer stream1,
    input integer stream2, input integer stream3,
    input logic [63:0] data0, input logic [63:0] data1,
    input logic [63:0] data2, input logic [63:0] data3,
    input logic expected_completed,
    input logic [255:0] expected_payload
  );
    begin
      @(negedge clk);
      inject_valid = '0;
      inject_data = '0;
      if (expected_mask[0]) put_inject(0, stream0, data0);
      if (expected_mask[1]) put_inject(1, stream1, data1);
      if (expected_mask[2]) put_inject(2, stream2, data2);
      if (expected_mask[3]) put_inject(3, stream3, data3);
      @(posedge clk);
      #1ns;
      inject_valid = '0;
      inject_data = '0;
      tx_issue_valid = issue_valid;
      tx_stream_index = issue_stream[2:0];
      #1ns;
      check_cycle(expected_mask, stream0, stream1, stream2, stream3,
                  data0, data1, data2, data3, expected_completed, expected_payload);
    end
  endtask

  ftlpu_sr_hemisphere_fabric #(
    .COLUMNS(COLUMNS), .SUPERLANES(SUPERLANES), .STREAMS(STREAMS),
    .LANES(8), .DATA_BITS(8), .LOCAL_PRODUCERS(PRODUCERS),
    .LOCAL_CONSUMERS(CONSUMERS)
  ) u_srf (
    .clk_i(clk), .rst_ni(rst_n),
    .boundary_valid_i(boundary_valid), .boundary_data_i(boundary_data),
    .boundary_valid_o(), .boundary_data_o(),
    .inject_valid_i(inject_valid), .inject_data_i(inject_data),
    .consume_i(srf_consume), .collision_o(collision),
    .invalid_consume_o(invalid_consume), .state_valid_o(state_valid),
    .state_data_o(state_data)
  );

  lpu_c2c_tx_srf_adapter #(
    .COLUMNS(COLUMNS), .SUPERLANES(SUPERLANES), .STREAMS(STREAMS),
    .SEGMENT_BITS(SEGMENT_BITS), .LOCAL_CONSUMERS(CONSUMERS),
    .C2C_CONSUMER_SLOT(C2C_SLOT), .TX_COLUMN(TX_COLUMN)
  ) u_adapter (
    .clk_i(clk), .rst_ni(rst_n), .tx_issue_valid_i(tx_issue_valid),
    .tx_stream_index_i(tx_stream_index), .srf_state_valid_i(state_valid),
    .srf_state_data_i(state_data), .tile_data_o(tile_data),
    .tile_valid_o(tile_valid), .tile_stream_idx_o(tile_stream_index),
    .gather_tile_consume_i(gather_tile_consume), .srf_consume_o(srf_consume)
  );

  c2c_tx_gather u_gather (
    .clk_i(clk), .rst_ni(rst_n), .tile_data_i(tile_data),
    .tile_valid_i(tile_valid),
    .tile_consume_o(gather_tile_consume), .completed_valid_o(completed_valid),
    .completed_payload_o(completed_payload)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  initial begin
    rst_n = 1'b0;
    clear_drive();

    // TEST 1: one stream-3 Send across the four diagonal tile cycles.
    reset_dut();
    run_cycle(1'b1, 3, 4'b0001, 3, 0, 0, 0,
              pattern(0,0), '0, '0, '0, 1'b0, '0);
    run_cycle(1'b0, 0, 4'b0010, 0, 3, 0, 0,
              '0, pattern(0,1), '0, '0, 1'b0, '0);
    run_cycle(1'b0, 0, 4'b0100, 0, 0, 3, 0,
              '0, '0, pattern(0,2), '0, 1'b0, '0);
    run_cycle(1'b0, 0, 4'b1000, 0, 0, 0, 3,
              '0, '0, '0, pattern(0,3), 1'b1,
              {pattern(0,3), pattern(0,2), pattern(0,1), pattern(0,0)});
    $display("C2C_TX_SRF_SINGLE PASS");

    // TEST 2/3/5: II=1 streams 3,6,7,1.  Cycle C+3 has four independent
    // selectors: SL0/S1, SL1/S7, SL2/S6, SL3/S3.
    reset_dut();
    run_cycle(1'b1, 3, 4'b0001, 3,0,0,0,
              pattern(0,0),'0,'0,'0,1'b0,'0);
    run_cycle(1'b1, 6, 4'b0011, 6,3,0,0,
              pattern(1,0),pattern(0,1),'0,'0,1'b0,'0);
    run_cycle(1'b1, 7, 4'b0111, 7,6,3,0,
              pattern(2,0),pattern(1,1),pattern(0,2),'0,1'b0,'0);
    run_cycle(1'b1, 1, 4'b1111, 1,7,6,3,
              pattern(3,0),pattern(2,1),pattern(1,2),pattern(0,3),1'b1,
              {pattern(0,3),pattern(0,2),pattern(0,1),pattern(0,0)});
    run_cycle(1'b0, 0, 4'b1110, 0,1,7,6,
              '0,pattern(3,1),pattern(2,2),pattern(1,3),1'b1,
              {pattern(1,3),pattern(1,2),pattern(1,1),pattern(1,0)});
    run_cycle(1'b0, 0, 4'b1100, 0,0,1,7,
              '0,'0,pattern(3,2),pattern(2,3),1'b1,
              {pattern(2,3),pattern(2,2),pattern(2,1),pattern(2,0)});
    run_cycle(1'b0, 0, 4'b1000, 0,0,0,1,
              '0,'0,'0,pattern(3,3),1'b1,
              {pattern(3,3),pattern(3,2),pattern(3,1),pattern(3,0)});
    $display("C2C_TX_SRF_STREAM_SELECT PASS");
    $display("C2C_TX_SRF_II1 PASS");
    $display("C2C_TX_SRF_CONSUME PASS");

    // TEST 4: a bubble stays in the selector-valid pipeline while V0 finishes.
    reset_dut();
    run_cycle(1'b1, 3, 4'b0001, 3,0,0,0,
              pattern(4,0),'0,'0,'0,1'b0,'0);
    run_cycle(1'b0, 0, 4'b0010, 0,3,0,0,
              '0,pattern(4,1),'0,'0,1'b0,'0);
    run_cycle(1'b1, 6, 4'b0101, 6,0,3,0,
              pattern(5,0),'0,pattern(4,2),'0,1'b0,'0);
    run_cycle(1'b0, 0, 4'b1010, 0,6,0,3,
              '0,pattern(5,1),'0,pattern(4,3),1'b1,
              {pattern(4,3),pattern(4,2),pattern(4,1),pattern(4,0)});
    run_cycle(1'b0, 0, 4'b0100, 0,0,6,0,
              '0,'0,pattern(5,2),'0,1'b0,'0);
    run_cycle(1'b0, 0, 4'b1000, 0,0,0,6,
              '0,'0,'0,pattern(5,3),1'b1,
              {pattern(5,3),pattern(5,2),pattern(5,1),pattern(5,0)});
    $display("C2C_TX_SRF_BUBBLE PASS");

    // TEST 5: target E7 and distractor E2 occupy the same superlane.
    reset_dut();
    @(negedge clk);
    inject_valid = '0;
    inject_data = '0;
    put_inject(0, 7, pattern(6,0));
    put_inject(0, 2, 64'hdead_beef_cafe_0202);
    @(posedge clk);
    #1ns;
    inject_valid = '0;
    inject_data = '0;
    tx_issue_valid = 1'b1;
    tx_stream_index = 3'd7;
    #1ns;
    check_cycle(4'b0001, 7,0,0,0, pattern(6,0),'0,'0,'0,
                1'b0,'0);
    if (srf_consume[consume_index(0, TX_COLUMN, 0, C2C_SLOT, 2)] !== 1'b0)
      fail("Wrong-stream distractor was consumed");
    $display("C2C_TX_SRF_SELECTOR_GUARD PASS");

    // TEST 6: both legal C2C attachment bounds are selectable.
    reset_dut();
    run_cycle(1'b1, 0, 4'b0001, 0,0,0,0,
              pattern(7,0),'0,'0,'0,1'b0,'0);
    reset_dut();
    run_cycle(1'b1, 7, 4'b0001, 7,0,0,0,
              pattern(8,0),'0,'0,'0,1'b0,'0);
    $display("C2C_TX_SRF_STREAM_RANGE PASS");

    // TEST 7: reset clears delayed selectors; no old selector may consume.
    reset_dut();
    run_cycle(1'b1, 3, 4'b0001, 3,0,0,0,
              pattern(9,0),'0,'0,'0,1'b0,'0);
    @(negedge clk);
    tx_issue_valid = 1'b0;
    rst_n = 1'b0;
    #1ns;
    if (tile_valid !== 4'b0000 || gather_tile_consume !== 4'b0000 ||
        srf_consume !== '0)
      fail("Reset did not suppress stale selector valid or consume");
    repeat (2) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    #1ns;
    run_cycle(1'b1, 0, 4'b0001, 0,0,0,0,
              pattern(9,1),'0,'0,'0,1'b0,'0);
    $display("C2C_TX_SRF_RESET PASS");

    if (collision !== '0 || invalid_consume !== '0)
      fail("Legal C2C TX schedule raised an SRF fault");
    $display("========================================");
    $display("C2C_TX_SRF_ADAPTER TEST_PASS");
    $display("========================================");
    $finish;
  end
endmodule
