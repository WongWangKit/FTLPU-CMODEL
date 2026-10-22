`timescale 1ns/1ps

module c2c_tx_peer_path_checker #(
  parameter integer LINK_LATENCY = 1
) (
  input logic clk_i,
  output logic done_o
);
  logic rst_ni;
  logic [255:0] tile_data_i;
  logic [3:0] tile_valid_i, tile_consume_o;
  logic credit_return_i;
  logic peer_rx_valid_o;
  logic [255:0] peer_rx_payload_o;
  logic [2:0] credit_count_o;
  logic serializer_busy_o, credit_error_o;
  integer check_cycle_id;

  lpu_c2c_tx_peer_path #(.FIFO_DEPTH(2), .LINK_LATENCY(LINK_LATENCY)) dut (.*);

  function automatic logic has_start(
    input integer slot, input [15:0] starts, input integer slots
  );
    has_start = 1'b0;
    if (slot >= 0 && slot < slots) has_start = starts[slot];
  endfunction

  function automatic [255:0] entry_for(input integer slot);
    // Unique vector/tile/byte patterns, with lane0 in the least-significant byte.
    for (integer b = 0; b < 32; b = b + 1)
      entry_for[b*8 +: 8] = 8'(slot*37 + b + 1);
  endfunction

  task automatic check_peer(input logic valid, input [255:0] entry);
    if (peer_rx_valid_o !== valid)
      $fatal(1, "TEST_FAIL L=%0d edge=%0d peer valid got=%b expected=%b",
             LINK_LATENCY, check_cycle_id, peer_rx_valid_o, valid);
    if (valid && peer_rx_payload_o !== entry)
      $fatal(1, "TEST_FAIL L=%0d edge=%0d peer entry got=%h expected=%h",
             LINK_LATENCY, check_cycle_id,
             peer_rx_payload_o, entry);
  endtask

  task automatic check_fifo(input logic valid, input [255:0] entry);
    // Automatic pop limits occupancy to 0/1 for this legal chain; no test
    // backpressure or artificial FIFO fill is introduced.
    if (dut.fifo_deq_valid !== valid || dut.fifo_deq_pop !== valid ||
        dut.fifo_empty !== !valid || dut.fifo_full !== 1'b0 ||
        dut.fifo_count !== (valid ? 2'd1 : 2'd0))
      $fatal(1, "TEST_FAIL L=%0d edge=%0d FIFO status/count", LINK_LATENCY, check_cycle_id);
    if (valid && dut.fifo_deq_payload !== entry)
      $fatal(1, "TEST_FAIL L=%0d edge=%0d FIFO head/order", LINK_LATENCY, check_cycle_id);
  endtask

  task automatic check_gather(input logic valid, input [255:0] entry);
    if (tile_consume_o !== tile_valid_i)
      $fatal(1, "TEST_FAIL L=%0d edge=%0d consume got=%b expected=%b",
             LINK_LATENCY, check_cycle_id, tile_consume_o, tile_valid_i);
    if (dut.gather_completed_valid !== valid || dut.u_fifo.enq_fire !== valid)
      $fatal(1, "TEST_FAIL L=%0d edge=%0d gather/enqueue", LINK_LATENCY, check_cycle_id);
    if (valid && dut.gather_completed_payload !== entry)
      $fatal(1, "TEST_FAIL L=%0d edge=%0d gather packing", LINK_LATENCY, check_cycle_id);
  endtask

  task automatic drive_inputs(
    input integer n, input [15:0] starts, input integer slots
  );
    logic [255:0] entry;
    integer k;
    begin
      @(negedge clk_i);
      check_cycle_id = n;
      tile_data_i = {4{64'hdeadbeef98765432}};
      tile_valid_i = '0;
      for (integer tile = 0; tile < 4; tile = tile + 1) begin
        k = n - tile;
        if (has_start(k, starts, slots)) begin
          entry = entry_for(k);
          tile_valid_i[tile] = 1'b1;
          tile_data_i[tile*64 +: 64] = entry[tile*64 +: 64];
        end
      end
    end
  endtask

  task automatic reset_path;
    @(negedge clk_i);
    rst_ni = 1'b0;
      tile_valid_i = '1;
      tile_data_i = '1;
      credit_return_i = 1'b0;
    #1;
    if (tile_consume_o !== 4'b0000 || dut.gather_completed_valid !== 1'b0 ||
        {dut.u_gather.stage2_valid_q, dut.u_gather.stage1_valid_q,
         dut.u_gather.stage0_valid_q} !== 3'b000 || dut.u_transport.valid_q !== '0)
      $fatal(1, "TEST_FAIL L=%0d reset valid state", LINK_LATENCY);
    check_fifo(0, '0);
    check_peer(0, '0);
    repeat (2) @(posedge clk_i);
    @(negedge clk_i);
    tile_valid_i = '0;
    credit_return_i = 1'b0;
    rst_ni = 1'b1;
    #1;
    check_fifo(0, '0);
    check_peer(0, '0);
  endtask

  task automatic run_schedule(
    input [15:0] starts, input integer slots,
    input logic require_ii1, input logic trace_cycle
  );
    integer n, k, expected_count, enqueues, dequeues, outputs, simultaneous;
    integer last_rx, measured_delta;
    integer gather_edge [0:15];
    integer head_edge [0:15];
    integer dequeue_edge [0:15];
    logic g_valid, f_valid, rx_valid;
    begin
      expected_count = 0;
      enqueues = 0;
      dequeues = 0;
      outputs = 0;
      simultaneous = 0;
      last_rx = -1;
      for (k = 0; k < 16; k = k + 1) begin
        gather_edge[k] = -1;
        head_edge[k] = -1;
        dequeue_edge[k] = -1;
        if (has_start(k, starts, slots)) expected_count = expected_count + 1;
      end
      for (n = 0; n < slots + LINK_LATENCY + 6; n = n + 1) begin
        drive_inputs(n, starts, slots);
        g_valid = has_start(n-3, starts, slots);
        f_valid = has_start(n-4, starts, slots);
        #1;
        check_gather(g_valid, entry_for(n-3));
        check_fifo(f_valid, entry_for(n-4));
        check_peer(has_start(n-4-LINK_LATENCY, starts, slots),
                   entry_for(n-4-LINK_LATENCY));
        @(posedge clk_i);
        // Capture-side observations are pre-NBA: FIFO must NOT bypass a
        // freshly completed Gather vector to the transport at this edge.
        check_gather(g_valid, entry_for(n-3));
        check_fifo(f_valid, entry_for(n-4));
        if (dut.u_fifo.deq_fire !== f_valid)
          $fatal(1, "TEST_FAIL FIFO dequeue acceptance");
        if (dut.gather_completed_valid) begin
          gather_edge[n-3] = n;
          enqueues = enqueues + 1;
        end
        if (dut.fifo_deq_pop) begin
          dequeue_edge[n-4] = n;
          dequeues = dequeues + 1;
        end
        if (dut.gather_completed_valid && dut.fifo_deq_pop)
          simultaneous = simultaneous + 1;
        #1;
        // Post-NBA visibility: enqueue at edge G makes head valid after G.
        // It is sampled by transport at G+1. RX first becomes visible after
        // G+LINK_LATENCY (not one additional output-register cycle later).
        check_fifo(g_valid, entry_for(n-3));
        if (dut.fifo_deq_valid) head_edge[n-3] = n;
        k = n - 3 - LINK_LATENCY;
        rx_valid = has_start(k, starts, slots);
        check_peer(rx_valid, entry_for(k));
        if (peer_rx_valid_o) begin
          measured_delta = n - gather_edge[k];
          if (gather_edge[k] < 0 || head_edge[k] != gather_edge[k] ||
              dequeue_edge[k] != gather_edge[k]+1 || measured_delta != LINK_LATENCY)
            $fatal(1, "TEST_FAIL L=%0d measured G-to-RX latency=%0d",
                   LINK_LATENCY, measured_delta);
          if (require_ii1 && last_rx >= 0 && n != last_rx+1)
            $fatal(1, "TEST_FAIL L=%0d RX II is not one", LINK_LATENCY);
          if (trace_cycle)
            $display("C2C_TX_PEER_TIMING L=%0d G=%0d head_after=%0d deq_edge=%0d rx_after=%0d delta=%0d",
                     LINK_LATENCY, gather_edge[k], head_edge[k], dequeue_edge[k], n, measured_delta);
          last_rx = n;
          outputs = outputs + 1;
        end
      end
      if (enqueues != expected_count || dequeues != expected_count || outputs != expected_count)
        $fatal(1, "TEST_FAIL L=%0d counts enq=%0d deq=%0d rx=%0d expected=%0d",
               LINK_LATENCY, enqueues, dequeues, outputs, expected_count);
      if (require_ii1 && simultaneous != expected_count-1)
        $fatal(1, "TEST_FAIL L=%0d FIFO simultaneous count=%0d", LINK_LATENCY, simultaneous);
    end
  endtask

  task automatic reset_while_active;
    // Drive five overlapping vectors up through the first two completions.
    // At edge 4, partial stages, FIFO and transport stage0 are all occupied.
    for (integer n = 0; n < 5; n = n + 1) begin
      drive_inputs(n, 16'h001f, 5);
      #1;
      check_gather(has_start(n-3, 16'h001f, 5), entry_for(n-3));
      check_fifo(has_start(n-4, 16'h001f, 5), entry_for(n-4));
      @(posedge clk_i);
    end
    #1;
    if ({dut.u_gather.stage2_valid_q, dut.u_gather.stage1_valid_q,
         dut.u_gather.stage0_valid_q} !== 3'b111 || dut.fifo_count !== 2'd1 ||
        dut.u_transport.valid_q[0] !== 1'b1)
      $fatal(1, "TEST_FAIL reset scenario did not occupy the whole chain");
    reset_path();
    // No old vector may emerge during the entire drain window after reset.
    run_schedule(16'h0000, 1, 0, 0);
    run_schedule(16'h0001, 1, 0, 0);
  endtask

  initial begin
    done_o = 1'b0;
    rst_ni = 1'b0;
    tile_data_i = '0;
    tile_valid_i = '0;
    check_cycle_id = -1;
    reset_path();
    run_schedule(16'h0001, 1, 0, 1);
    if (LINK_LATENCY == 1) $display("C2C_TX_PEER_SINGLE PASS");
    reset_path();
    run_schedule(16'h000f, 4, 1, 0);
    if (LINK_LATENCY == 1) begin
      $display("C2C_TX_PEER_II1 PASS");
      $display("C2C_TX_PEER_FIFO PASS");
    end
    reset_path();
    // Two bursts separated by empty issue slots; payloads remain ordered.
    run_schedule(16'h0063, 7, 0, 0);
    if (LINK_LATENCY == 1) $display("C2C_TX_PEER_PAYLOAD_ORDER PASS");
    reset_path();
    run_schedule(16'h004d, 7, 0, 0);
    if (LINK_LATENCY == 1) $display("C2C_TX_PEER_BUBBLE PASS");
    reset_path();
    reset_while_active();
    if (LINK_LATENCY == 1) $display("C2C_TX_PEER_RESET PASS");
    done_o = 1'b1;
  end
endmodule

module lpu_c2c_tx_peer_path_tb;
  logic clk_i = 1'b0;
  logic latency1_done, latency3_done;
  always #5 clk_i = ~clk_i;

  c2c_tx_peer_path_checker #(.LINK_LATENCY(1)) latency1 (
    .clk_i, .done_o(latency1_done)
  );
  c2c_tx_peer_path_checker #(.LINK_LATENCY(3)) latency3 (
    .clk_i, .done_o(latency3_done)
  );

  initial begin
    wait (latency1_done && latency3_done);
    $display("C2C_TX_PEER_LATENCY PASS");
    $display("C2C_TX_PEER_CYCLE PASS");
    $display("C2C_TX_PEER_PATH TEST_PASS");
    $finish;
  end

  initial begin
    #20000;
    $fatal(1, "TEST_FAIL simulation timeout");
  end
endmodule
