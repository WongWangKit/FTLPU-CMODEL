`timescale 1ns/1ps

module lpu_c2c_rx_srf_integration_tb;
  localparam integer COLS=16, SL=4, STREAMS=32, PRODUCERS=2, SLOT=1;
  localparam integer STATE_BITS=2*COLS*SL*STREAMS;
  localparam integer INJECT_BITS=STATE_BITS*PRODUCERS;
  logic clk=0, rst_n=0;
  logic peer_valid, cmd_valid, cmd_pop;
  logic [255:0] peer_payload;
  logic [2:0] cmd_stream;
  wire fifo_full, fifo_empty;
  wire [1:0] fifo_count;
  wire [INJECT_BITS-1:0] inject_valid;
  wire [INJECT_BITS*64-1:0] inject_data;
  wire [STATE_BITS-1:0] state_valid;
  wire [STATE_BITS*64-1:0] state_data;
  wire [2*COLS*SL-1:0] collision, invalid_consume;
  integer cycle_number=0;

  lpu_c2c_rx_srf_integration #(
    .FIFO_DEPTH(2), .COLUMNS(COLS), .SUPERLANES(SL), .STREAMS(STREAMS),
    .LOCAL_PRODUCERS(PRODUCERS), .C2C_PRODUCER_SLOT(SLOT)
  ) u_rx (
    .clk_i(clk), .rst_ni(rst_n), .peer_vector_valid_i(peer_valid),
    .peer_vector_payload_i(peer_payload), .rx_cmd_valid_i(cmd_valid),
    .rx_cmd_stream_index_i(cmd_stream), .rx_cmd_pop_o(cmd_pop),
    .rx_ready_full_o(fifo_full), .rx_ready_empty_o(fifo_empty),
    .rx_ready_count_o(fifo_count), .srf_inject_valid_o(inject_valid),
    .srf_inject_data_o(inject_data)
  );
  ftlpu_sr_hemisphere_fabric #(
    .COLUMNS(COLS), .SUPERLANES(SL), .STREAMS(STREAMS), .LANES(8),
    .DATA_BITS(8), .LOCAL_PRODUCERS(PRODUCERS), .LOCAL_CONSUMERS(2)
  ) u_srf (
    .clk_i(clk), .rst_ni(rst_n),
    .boundary_valid_i({2*SL*STREAMS{1'b0}}),
    .boundary_data_i({2*SL*STREAMS*64{1'b0}}),
    .boundary_valid_o(), .boundary_data_o(),
    .inject_valid_i(inject_valid), .inject_data_i(inject_data),
    .consume_i({2*COLS*SL*2*STREAMS{1'b0}}),
    .collision_o(collision), .invalid_consume_o(invalid_consume),
    .state_valid_o(state_valid), .state_data_o(state_data)
  );

  always #5 clk=~clk;
  always @(posedge clk) begin
    cycle_number=cycle_number+1;
    if (rst_n && (collision !== '0 || invalid_consume !== '0))
      $fatal(1, "TEST_FAIL SRF collision/invalid-consume at edge %0d", cycle_number);
  end

  // Independent model: ordered payload queue plus a ledger of accepted
  // Receive transactions. No expected value is taken from Replay selectors.
  logic [255:0] payload_queue[0:1];
  integer queue_count, event_count, observed_segments;
  logic [255:0] event_payload[0:63];
  integer event_stream[0:63], event_cycle[0:63];
  logic [STATE_BITS-1:0] expected_state_valid, next_state_valid;
  logic [STATE_BITS*64-1:0] expected_state_data, next_state_data;
  logic [INJECT_BITS-1:0] expected_inject_valid;
  logic [INJECT_BITS*64-1:0] expected_inject_data;
  logic [3:0] expected_tile_valid;
  logic [19:0] expected_tile_stream;
  logic [255:0] expected_tile_data;
  logic log_timing, log_overlap;
  logic [4:0] distractor_tx_stream;
  integer overlap_checks, timing_checks;
  integer command_streams[0:3];
  integer test_i;

  function automatic integer cell_index(
    input integer direction, input integer column,
    input integer tile, input integer stream
  );
    cell_index=((direction*COLS+column)*SL+tile)*STREAMS+stream;
  endfunction

  function automatic integer producer_index(input integer tile, input integer stream);
    producer_index=(((COLS+13)*SL+tile)*PRODUCERS+SLOT)*STREAMS+stream;
  endfunction

  function automatic [255:0] pattern(input integer id);
    // Each tile/lane differs, including the explicit lane0-low-byte ordering.
    for (integer b=0;b<32;b=b+1)
      pattern[b*8+:8]=8'(1+id*37+b);
  endfunction

  task automatic fail(input string message);
    $fatal(1, "TEST_FAIL cycle=%0d %s", cycle_number, message);
  endtask

  task automatic reset_case;
    begin
      @(negedge clk);
      rst_n=0;
      peer_valid=0; peer_payload='0; cmd_valid=0; cmd_stream=0;
      queue_count=0; event_count=0; observed_segments=0;
      expected_state_valid='0; expected_state_data='0;
      #1;
      if (!fifo_empty || fifo_count !== 0 || u_rx.replay_inject_valid !== 0 ||
          inject_valid !== '0 || state_valid !== '0 || cmd_pop !== 0)
        fail("Reset did not clear queued/deferred/SRF validity");
      repeat (2) @(posedge clk);
      @(negedge clk);
      rst_n=1;
      #1;
      if (inject_valid !== '0 || cmd_pop !== 0)
        fail("Stale Replay transaction after reset release");
    end
  endtask

  // P labels the cycle interval BEFORE the sampling edge P+1. Pair outputs
  // and tile0 candidate must be present in P; state is checked after P+1 NBA.
  task automatic tick(
    input logic push, input integer vector_id,
    input logic receive_valid, input integer receive_stream
  );
    logic pair_expected;
    integer p, e, tile, idx, src, dst, col, s, sl, j;
    begin
      @(negedge clk);
      peer_valid=push; peer_payload=pattern(vector_id);
      cmd_valid=receive_valid; cmd_stream=receive_stream[2:0];
      #1;
      p=cycle_number;
      if (fifo_count !== queue_count || fifo_empty !== (queue_count==0) ||
          fifo_full !== (queue_count==2)) fail("RX-ready FIFO status mismatch");
      pair_expected=receive_valid && (queue_count!=0);
      if (cmd_pop !== pair_expected || u_rx.pair_valid !== pair_expected ||
          u_rx.ready_pop !== pair_expected) fail("Pair/pop atomicity mismatch");
      if (push && queue_count==2 && !pair_expected) fail("Test schedule overflow");
      if (pair_expected) begin
        if (u_rx.pair_payload !== payload_queue[0] ||
            u_rx.pair_stream !== {2'b00,receive_stream[2:0]})
          fail("Pair data or Receive destination mismatch");
        event_payload[event_count]=payload_queue[0];
        event_stream[event_count]=receive_stream;
        event_cycle[event_count]=p;
        event_count=event_count+1;
        if (log_timing) $display("C2C_RX_SRF_PAIR P=%0d Receive=W%0d", p, receive_stream);
      end

      expected_inject_valid='0; expected_inject_data='0;
      expected_tile_valid='0; expected_tile_stream='0; expected_tile_data='0;
      for (e=0;e<event_count;e=e+1) begin
        tile=p-event_cycle[e];
        if (tile>=0 && tile<4) begin
          if (expected_tile_valid[tile]) fail("Model duplicate tile issue");
          expected_tile_valid[tile]=1;
          expected_tile_stream[tile*5+:5]=5'(event_stream[e]);
          expected_tile_data[tile*64+:64]=event_payload[e][tile*64+:64];
          idx=producer_index(tile,event_stream[e]);
          expected_inject_valid[idx]=1;
          expected_inject_data[idx*64+:64]=event_payload[e][tile*64+:64];
        end
      end
      // Full bus equality proves direction, column, producer-slot, stream and
      // every other inactive coordinate; all four tiles remain independent.
      if (inject_valid !== expected_inject_valid || inject_data !== expected_inject_data)
        fail("Packed SRF injection mismatch or hidden adapter latency");
      if (u_rx.replay_inject_valid !== expected_tile_valid)
        fail("Diagonal Replay valid/bubble mismatch");
      for (tile=0;tile<4;tile=tile+1) if (expected_tile_valid[tile]) begin
        if (u_rx.replay_inject_stream[tile*5+:5] !== expected_tile_stream[tile*5+:5] ||
            u_rx.replay_inject_data[tile*64+:64] !== expected_tile_data[tile*64+:64])
          fail("Per-tile Replay selector/data mismatch");
      end
      if (log_overlap && expected_tile_valid==4'b1111) begin
        $display("C2C_RX_SRF_OVERLAP cycle=%0d SL0/W%0d SL1/W%0d SL2/W%0d SL3/W%0d",
          p,u_rx.replay_inject_stream[0+:5],u_rx.replay_inject_stream[5+:5],
          u_rx.replay_inject_stream[10+:5],u_rx.replay_inject_stream[15+:5]);
        overlap_checks=overlap_checks+1;
      end

      // Model every SRF column, not only sreg13. West advances toward col0;
      // no boundary inputs or consumers are active in these RX-only tests.
      next_state_valid='0; next_state_data='0;
      for (col=0;col<COLS-1;col=col+1)
        for (sl=0;sl<SL;sl=sl+1) for (s=0;s<STREAMS;s=s+1) begin
          src=cell_index(1,col+1,sl,s); dst=cell_index(1,col,sl,s);
          if (expected_state_valid[src]) begin
            next_state_valid[dst]=1;
            next_state_data[dst*64+:64]=expected_state_data[src*64+:64];
          end
        end
      for (tile=0;tile<4;tile=tile+1) if (expected_tile_valid[tile]) begin
        dst=cell_index(1,13,tile,expected_tile_stream[tile*5+:5]);
        if (next_state_valid[dst]) fail("Model detected illegal schedule collision");
        next_state_valid[dst]=1;
        next_state_data[dst*64+:64]=expected_tile_data[tile*64+:64];
      end
      @(posedge clk);
      if (pair_expected) begin
        payload_queue[0]=payload_queue[1]; queue_count=queue_count-1;
      end
      if (push) begin payload_queue[queue_count]=pattern(vector_id); queue_count=queue_count+1; end
      expected_state_valid=next_state_valid; expected_state_data=next_state_data;
      #1;
      if (state_valid !== expected_state_valid)
        fail("SRF state valid has wrong coordinate, timing or a ghost segment");
      for (j=0;j<STATE_BITS;j=j+1)
        if (expected_state_valid[j] && state_data[j*64+:64] !== expected_state_data[j*64+:64])
          fail("SRF state payload/byte packing/propagation mismatch");
      if (collision !== '0 || invalid_consume !== '0) fail("SRF fault in legal schedule");
      for (e=0;e<event_count;e=e+1) begin
        tile=p-event_cycle[e];
        if (tile>=0 && tile<4) begin
          observed_segments=observed_segments+1;
          if (log_timing) begin
            $display("C2C_RX_SRF_TIMING P=%0d tile=%0d candidate=%0d state_visible=%0d delta=%0d",
              event_cycle[e],tile,p,cycle_number,cycle_number-event_cycle[e]);
            timing_checks=timing_checks+1;
          end
        end
      end
    end
  endtask

  task automatic drain_replay;
    repeat (5) tick(0,0,0,0);
  endtask

  initial begin
    peer_valid=0; peer_payload=0; cmd_valid=0; cmd_stream=0;
    log_timing=0; log_overlap=0; timing_checks=0; overlap_checks=0;
    distractor_tx_stream=5'd3; // Not connected: RX has no TX metadata input.
    command_streams[0]=3;command_streams[1]=6;command_streams[2]=1;command_streams[3]=7;

    reset_case();
    log_timing=1;
    tick(1,0,0,0);
    tick(0,0,1,6);
    drain_replay();
    log_timing=0;
    if (event_count!=1 || observed_segments!=4 || timing_checks!=4 ||
        event_stream[0]!=6 || event_stream[0]==distractor_tx_stream)
      fail("Single vector/independent Receive stream was not fully observed");
    $display("C2C_RX_SRF_SINGLE PASS");
    $display("C2C_RX_SRF_STREAM_OVERRIDE PASS");

    reset_case();
    tick(1,0,0,0); tick(1,1,0,0);
    log_overlap=1;
    for (test_i=0;test_i<4;test_i=test_i+1) begin
      tick(test_i<2,test_i+2,1,command_streams[test_i]);
      if (event_count!=test_i+1) fail("Missing II=1 pair");
      if (test_i>0 && event_cycle[test_i]!=event_cycle[test_i-1]+1)
        fail("Unexpected gap between consecutive pairs");
    end
    drain_replay();
    log_overlap=0;
    if (observed_segments!=16 || overlap_checks!=1) fail("Four-vector overlap incomplete");
    $display("C2C_RX_SRF_STREAM_SELECT PASS");
    $display("C2C_RX_SRF_II1 PASS");

    reset_case();
    repeat (3) tick(0,0,1,3);
    if (event_count!=0 || observed_segments!=0) fail("Command first started Replay early");
    tick(1,10,1,3);
    if (event_count!=0) fail("Empty FIFO incorrectly fell through");
    tick(0,0,1,3);
    drain_replay();
    if (event_count!=1 || observed_segments!=4) fail("Command first lost/duplicated vector");
    $display("C2C_RX_SRF_COMMAND_FIRST PASS");

    reset_case();
    tick(1,11,0,0);
    repeat (3) tick(0,0,0,0);
    if (queue_count!=1 || event_count!=0) fail("Vector first did not wait in RX FIFO");
    tick(0,0,1,7);
    drain_replay();
    if (event_count!=1 || observed_segments!=4) fail("Vector first lost/duplicated vector");
    $display("C2C_RX_SRF_VECTOR_FIRST PASS");
    $display("C2C_RX_SRF_SELECTOR_GUARD PASS");

    reset_case();
    tick(1,20,1,3);
    tick(1,21,1,3);
    tick(0,0,1,6);
    tick(1,22,1,1);
    tick(0,0,1,1);
    drain_replay();
    if (event_count!=3 || observed_segments!=12 ||
        event_cycle[1]!=event_cycle[0]+1 || event_cycle[2]!=event_cycle[1]+2)
      fail("Vector bubble was compressed or corrupted Replay");
    $display("C2C_RX_SRF_BUBBLE PASS");

    reset_case();
    tick(1,30,0,0);
    tick(1,31,1,3);
    tick(1,32,0,0);
    if (queue_count!=2 || u_rx.replay_inject_valid==0)
      fail("Reset test did not populate both FIFO and Replay state");
    reset_case();
    repeat (2) tick(0,0,1,0);
    tick(1,33,1,0); tick(0,0,1,0);
    drain_replay();
    if (event_count!=1 || observed_segments!=4 || event_stream[0]!=0)
      fail("Reset/restart did not produce only the new W0 vector");
    $display("C2C_RX_SRF_RESET PASS");
    $display("C2C_RX_SRF_INJECT_MAPPING PASS");
    $display("C2C_RX_SRF_CYCLE PASS");
    $display("========================================");
    $display("C2C_RX_SRF_INTEGRATION TEST_PASS");
    $display("========================================");
    $finish;
  end

  initial begin
    #20000;
    $fatal(1,"TEST_FAIL C2C RX SRF integration timeout");
  end
endmodule
