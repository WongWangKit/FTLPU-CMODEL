`timescale 1ns/1ps

module c2c_peer_normal_checker #(
  parameter integer LINK_LATENCY = 1
) (
  input logic clk_i,
  output logic done_o
);
  logic rst_ni;
  logic [255:0] tx_tile_data_i;
  logic [3:0] tx_tile_valid_i, tx_tile_consume_o;
  // Independent test-only RX control, stable until the Replay capture edge.
  logic [4:0] rx_stream_idx_i;
  logic [3:0] rx_inject_valid_o;
  logic [255:0] rx_inject_data_o;
  logic [19:0] rx_inject_stream_idx_o;

  lpu_c2c_peer_normal_path #(
    .FIFO_DEPTH(2), .LINK_LATENCY(LINK_LATENCY), .P_VECTOR_CREDITS(16)
  ) dut (.*);

  function automatic logic has_start(
    input integer k, input [15:0] starts, input integer slots
  );
    has_start = 1'b0;
    if (k >= 0 && k < slots) has_start = starts[k];
  endfunction

  function automatic integer ordinal(input integer k, input [15:0] starts);
    ordinal = 0;
    for (integer i = 0; i < k; i = i + 1)
      if (starts[i]) ordinal = ordinal + 1;
  endfunction

  function automatic [63:0] segment_pattern(input integer id, input integer tile);
    for (integer lane = 0; lane < 8; lane = lane + 1)
      segment_pattern[lane*8 +: 8] = 8'(id*37 + tile*8 + lane + 1);
  endfunction

  // Generic Replay indices 11/19 are NOT legal C2C SRF attachment streams.
  function automatic [4:0] stream_for(input integer id, input logic mixed);
    stream_for = 5'd3;
    if (mixed) begin
      case (id)
        0: stream_for = 5'd3;
        1: stream_for = 5'd11;
        2: stream_for = 5'd7;
        3: stream_for = 5'd19;
        default: stream_for = 5'(id);
      endcase
    end
  endfunction

  task automatic drive_cycle(
    input integer n, input [15:0] starts, input integer slots,
    input integer base_id, input logic mixed
  );
    integer k, id;
    begin
      @(negedge clk_i);
      tx_tile_data_i = {4{64'hdeadbeef98765432}};
      tx_tile_valid_i = '0;
      for (integer tile = 0; tile < 4; tile = tile + 1) begin
        k = n - tile;
        if (has_start(k, starts, slots)) begin
          id = base_id + ordinal(k, starts);
          tx_tile_valid_i[tile] = 1'b1;
          tx_tile_data_i[tile*64 +: 64] = segment_pattern(id, tile);
        end
      end
    end
  endtask

  task automatic check_reset_state;
    if (tx_tile_consume_o !== 4'b0000 || rx_inject_valid_o !== 4'b0000 ||
        dut.peer_rx_valid !== 1'b0 ||
        {dut.u_tx.u_gather.stage2_valid_q, dut.u_tx.u_gather.stage1_valid_q,
         dut.u_tx.u_gather.stage0_valid_q} !== 3'b000 ||
        dut.u_tx.fifo_empty !== 1'b1 || dut.u_tx.fifo_full !== 1'b0 ||
        dut.u_tx.fifo_count !== 2'd0 || dut.u_tx.u_transport.valid_q !== '0 ||
        dut.u_rx.deferred_valid_q !== 3'b000)
      $fatal(1, "TEST_FAIL L=%0d reset did not clear path control", LINK_LATENCY);
  endtask

  task automatic reset_path;
    @(negedge clk_i);
    rst_ni = 1'b0;
    tx_tile_valid_i = '1;
    tx_tile_data_i = '1;
    rx_stream_idx_i = '1;
    #1;
    check_reset_state();
    repeat (2) @(posedge clk_i);
    #1;
    check_reset_state();
    @(negedge clk_i);
    tx_tile_valid_i = '0;
    rst_ni = 1'b1;
    #1;
    check_reset_state();
  endtask

  task automatic run_schedule(
    input [15:0] starts, input integer slots, input integer base_id,
    input logic mixed, input logic require_ii1, input logic trace_cycle
  );
    // Scoreboard captures actual TX segments at the capture edge. RX data
    // is compared against those entries, not just a single fixed pattern.
    logic [255:0] sent [0:15];
    logic [255:0] recovered [0:15];
    logic [4:0] streams [0:15];
    logic [3:0] captured [0:15];
    logic [3:0] received [0:15];
    integer gather_cycle [0:15];
    integer last_rx_cycle [0:3];
    integer expected_count, tx_segments, rx_segments, vectors_sent, vectors_received;
    integer n, k, id, delta;
    logic [3:0] expected_rx_valid;
    logic expected_completed, expected_peer;
    logic [3:0] previous_valid;
    logic [255:0] previous_data;
    logic [19:0] previous_stream;
    begin
      expected_count = 0;
      tx_segments = 0;
      rx_segments = 0;
      vectors_sent = 0;
      vectors_received = 0;
      previous_valid = '0;
      previous_data = '0;
      previous_stream = '0;
      for (id = 0; id < 16; id = id + 1) begin
        sent[id] = '0;
        recovered[id] = '0;
        streams[id] = stream_for(base_id+id, mixed);
        captured[id] = '0;
        received[id] = '0;
        gather_cycle[id] = -1;
        if (has_start(id, starts, slots)) expected_count = expected_count + 1;
      end
      for (integer tile = 0; tile < 4; tile = tile + 1) last_rx_cycle[tile] = -1;
      for (n = 0; n < slots + LINK_LATENCY + 8; n = n + 1) begin
        drive_cycle(n, starts, slots, base_id, mixed);
        #1;
        if (tx_tile_consume_o !== tx_tile_valid_i)
          $fatal(1, "TEST_FAIL L=%0d edge=%0d TX consume", LINK_LATENCY, n);
        @(posedge clk_i);
        // Capture-side checks use pre-NBA values. RX candidates from the
        // previous cycle must also remain stable until this sampling edge.
        if (tx_tile_consume_o !== tx_tile_valid_i || rx_inject_valid_o !== previous_valid)
          $fatal(1, "TEST_FAIL L=%0d edge=%0d capture stability", LINK_LATENCY, n);
        for (integer tile = 0; tile < 4; tile = tile + 1) begin
          if (previous_valid[tile] &&
              (rx_inject_data_o[tile*64 +: 64] !== previous_data[tile*64 +: 64] ||
               rx_inject_stream_idx_o[tile*5 +: 5] !== previous_stream[tile*5 +: 5]))
            $fatal(1, "TEST_FAIL RX candidate unstable before consumer edge");
          if (tx_tile_consume_o[tile]) begin
            k = n - tile;
            id = ordinal(k, starts);
            if (captured[id][tile]) $fatal(1, "TEST_FAIL duplicate TX capture");
            captured[id][tile] = 1'b1;
            sent[id][tile*64 +: 64] = tx_tile_data_i[tile*64 +: 64];
            if (tile == 3) gather_cycle[id] = n;
            tx_segments = tx_segments + 1;
          end
        end
        expected_completed = has_start(n-3, starts, slots);
        if (dut.u_tx.gather_completed_valid !== expected_completed ||
            dut.u_tx.u_fifo.enq_fire !== expected_completed)
          $fatal(1, "TEST_FAIL L=%0d edge=%0d Gather/FIFO completion", LINK_LATENCY, n);
        if (expected_completed) begin
          id = ordinal(n-3, starts);
          if (captured[id] !== 4'b1111 ||
              dut.u_tx.gather_completed_payload !== sent[id])
            $fatal(1, "TEST_FAIL completed vector differs from captured TX entry");
          vectors_sent = vectors_sent + 1;
        end
        #1;
        // Test-only RX control is supplied at the expected peer arrival,
        // independently of all TX inputs. Hold it until the next capture edge.
        // This utility has no Receive queue; the formal endpoint tests pairing.
        k = n - 3 - LINK_LATENCY;
        rx_stream_idx_i = has_start(k, starts, slots) ?
                          stream_for(base_id+ordinal(k, starts), mixed) : 5'd31;
        #1;
        // Observe all producer candidates after control and payload settle.
        expected_peer = has_start(k, starts, slots);
        if (dut.peer_rx_valid !== expected_peer)
          $fatal(1, "TEST_FAIL L=%0d edge=%0d peer vector timing", LINK_LATENCY, n);
        if (expected_peer) begin
          id = ordinal(k, starts);
          if (dut.peer_rx_payload !== sent[id])
            $fatal(1, "TEST_FAIL Transport payload integrity");
          vectors_received = vectors_received + 1;
        end
        expected_rx_valid = '0;
        for (integer tile = 0; tile < 4; tile = tile + 1)
          expected_rx_valid[tile] = has_start(n-3-LINK_LATENCY-tile, starts, slots);
        if (rx_inject_valid_o !== expected_rx_valid)
          $fatal(1, "TEST_FAIL L=%0d edge=%0d RX valid got=%b expected=%b",
                 LINK_LATENCY, n, rx_inject_valid_o, expected_rx_valid);
        for (integer tile = 0; tile < 4; tile = tile + 1) begin
          if (rx_inject_valid_o[tile]) begin
            k = n - 3 - LINK_LATENCY - tile;
            id = ordinal(k, starts);
            delta = n - gather_cycle[id];
            if (captured[id] !== 4'b1111 || gather_cycle[id] < 0 ||
                delta != LINK_LATENCY + tile || received[id][tile])
              $fatal(1, "TEST_FAIL L=%0d vector=%0d tile=%0d cycle/duplicate", LINK_LATENCY, id, tile);
            if (rx_inject_data_o[tile*64 +: 64] !== sent[id][tile*64 +: 64] ||
                rx_inject_stream_idx_o[tile*5 +: 5] !== streams[id])
              $fatal(1, "TEST_FAIL L=%0d vector=%0d tile=%0d RX data/stream mismatch",
                     LINK_LATENCY, id, tile);
            if (require_ii1 && last_rx_cycle[tile] >= 0 && n != last_rx_cycle[tile]+1)
              $fatal(1, "TEST_FAIL L=%0d tile=%0d RX II is not one", LINK_LATENCY, tile);
            last_rx_cycle[tile] = n;
            received[id][tile] = 1'b1;
            recovered[id][tile*64 +: 64] = rx_inject_data_o[tile*64 +: 64];
            rx_segments = rx_segments + 1;
            if (trace_cycle)
              $display("C2C_PEER_NORMAL_TIMING L=%0d G=%0d tile=%0d rx_after=%0d delta=%0d",
                       LINK_LATENCY, gather_cycle[id], tile, n, delta);
          end
        end
        previous_valid = rx_inject_valid_o;
        previous_data = rx_inject_data_o;
        previous_stream = rx_inject_stream_idx_o;
      end
      if (tx_segments != 4*expected_count || rx_segments != 4*expected_count ||
          vectors_sent != expected_count || vectors_received != expected_count)
        $fatal(1, "TEST_FAIL L=%0d TX/RX segment or vector count", LINK_LATENCY);
      for (id = 0; id < expected_count; id = id + 1)
        if (captured[id] !== 4'b1111 || received[id] !== 4'b1111 || recovered[id] !== sent[id])
          $fatal(1, "TEST_FAIL vector=%0d final packing symmetry", id);
    end
  endtask

  task automatic reset_while_active;
    // Fill every stage in the chain, including all three RX deferred stages.
    for (integer n = 0; n < LINK_LATENCY+7; n = n + 1) begin
      drive_cycle(n, 16'hffff, LINK_LATENCY+7, 0, 0);
      #1;
      if (tx_tile_consume_o !== tx_tile_valid_i)
        $fatal(1, "TEST_FAIL active-reset TX consume");
      @(posedge clk_i);
    end
    #1;
    if ({dut.u_tx.u_gather.stage2_valid_q, dut.u_tx.u_gather.stage1_valid_q,
         dut.u_tx.u_gather.stage0_valid_q} !== 3'b111 || dut.u_tx.fifo_count !== 2'd1 ||
        dut.u_tx.u_transport.valid_q !== {LINK_LATENCY{1'b1}} ||
        dut.u_rx.deferred_valid_q !== 3'b111)
      $fatal(1, "TEST_FAIL reset test did not populate all path stages");
    reset_path();
    run_schedule(16'h0000, 1, 0, 0, 0, 0);
    run_schedule(16'h0001, 1, 15, 0, 0, 0);
  endtask

  initial begin
    done_o = 1'b0;
    rst_ni = 1'b0;
    tx_tile_data_i = '0;
    tx_tile_valid_i = '0;
    rx_stream_idx_i = '0;
    reset_path();
    run_schedule(16'h0001, 1, 0, 0, 0, 1);
    if (LINK_LATENCY == 1) $display("C2C_PEER_NORMAL_SINGLE PASS");
    reset_path();
    run_schedule(16'h000f, 4, 0, 0, 1, 0);
    if (LINK_LATENCY == 1) $display("C2C_PEER_NORMAL_II1 PASS");
    reset_path();
    run_schedule(16'h000f, 4, 0, 1, 1, 0);
    if (LINK_LATENCY == 1) $display("C2C_PEER_NORMAL_STREAM PASS");
    reset_path();
    // V0, V1, one empty vector issue slot, V2, V3.
    run_schedule(16'h001b, 5, 0, 1, 0, 0);
    if (LINK_LATENCY == 1) $display("C2C_PEER_NORMAL_BUBBLE PASS");
    reset_path();
    reset_while_active();
    if (LINK_LATENCY == 1) $display("C2C_PEER_NORMAL_RESET PASS");
    done_o = 1'b1;
  end
endmodule

module lpu_c2c_peer_normal_path_tb;
  logic clk_i = 1'b0;
  logic latency1_done, latency3_done;
  always #5 clk_i = ~clk_i;
  c2c_peer_normal_checker #(.LINK_LATENCY(1)) latency1 (
    .clk_i, .done_o(latency1_done)
  );
  c2c_peer_normal_checker #(.LINK_LATENCY(3)) latency3 (
    .clk_i, .done_o(latency3_done)
  );

  initial begin
    wait (latency1_done && latency3_done);
    $display("C2C_PEER_NORMAL_LATENCY PASS");
    $display("C2C_PEER_NORMAL_DATA PASS");
    $display("C2C_PEER_NORMAL_CONSUME PASS");
    $display("C2C_PEER_NORMAL_CYCLE PASS");
    $display("C2C_PEER_NORMAL_PATH TEST_PASS");
    $finish;
  end

  initial begin
    #20000;
    $fatal(1, "TEST_FAIL simulation timeout");
  end
endmodule
