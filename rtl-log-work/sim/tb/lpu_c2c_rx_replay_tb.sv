`timescale 1ns/1ps

module lpu_c2c_rx_replay_tb;
  logic clk_i, rst_ni;
  logic vector_valid_i;
  logic [255:0] vector_payload_i;
  logic [4:0] vector_stream_idx_i;
  logic [3:0] inject_valid_o;
  logic [255:0] inject_data_o;
  logic [19:0] inject_stream_idx_o;

  c2c_rx_replay dut (.*);
  always #5 clk_i = ~clk_i;

  function automatic [63:0] segment_pattern(input integer id, input integer tile);
    for (integer lane = 0; lane < 8; lane = lane + 1)
      segment_pattern[lane*8 +: 8] = 8'(id*64 + tile*8 + lane + 1);
  endfunction

  function automatic [255:0] vector_pattern(input integer id);
    vector_pattern = {segment_pattern(id, 3), segment_pattern(id, 2),
                      segment_pattern(id, 1), segment_pattern(id, 0)};
  endfunction

  // Generic native-interface test only; C2C SRF attachment uses streams 0..7.
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

  function automatic logic has_input(
    input integer n, input [15:0] starts, input integer slots
  );
    has_input = 1'b0;
    if (n >= 0 && n < slots) has_input = starts[n];
  endfunction

  function automatic integer vector_number(input integer n, input [15:0] starts);
    vector_number = 0;
    for (integer k = 0; k < n; k = k + 1)
      if (starts[k]) vector_number = vector_number + 1;
  endfunction

  task automatic drive_input(input logic valid, input integer id, input logic mixed);
    vector_valid_i = valid;
    vector_payload_i = valid ? vector_pattern(id) : {4{64'hdeadbeef98765432}};
    vector_stream_idx_i = valid ? stream_for(id, mixed) : 5'd31;
  endtask

  // Independent oracle: input at slot k must emit tile t at k+t.
  // Neither data nor validity expectations read DUT pipeline registers.
  task automatic check_outputs(
    input integer n, input [15:0] starts, input integer slots,
    input integer base_id, input logic mixed
  );
    logic [3:0] expected_valid;
    integer k, id;
    begin
      expected_valid = '0;
      for (integer tile = 0; tile < 4; tile = tile + 1) begin
        k = n - tile;
        if (has_input(k, starts, slots)) expected_valid[tile] = 1'b1;
      end
      if (inject_valid_o !== expected_valid)
        $fatal(1, "TEST_FAIL cycle=%0d valid got=%b expected=%b",
               n, inject_valid_o, expected_valid);
      for (integer tile = 0; tile < 4; tile = tile + 1) begin
        if (expected_valid[tile]) begin
          k = n - tile;
          id = base_id + vector_number(k, starts);
          if (inject_data_o[tile*64 +: 64] !== segment_pattern(id, tile))
            $fatal(1, "TEST_FAIL cycle=%0d vector=%0d tile=%0d data got=%h expected=%h",
                   n, id, tile, inject_data_o[tile*64 +: 64], segment_pattern(id, tile));
          if (inject_stream_idx_o[tile*5 +: 5] !== stream_for(id, mixed))
            $fatal(1, "TEST_FAIL cycle=%0d tile=%0d stream got=%0d expected=%0d",
                   n, tile, inject_stream_idx_o[tile*5 +: 5], stream_for(id, mixed));
        end
      end
    end
  endtask

  task automatic reset_replay;
    @(negedge clk_i);
    rst_ni = 1'b0;
    // Even the combinational tile0 path must be quiet during reset.
    drive_input(1, 3, 1);
    #1;
    if (inject_valid_o !== 4'b0000 || dut.deferred_valid_q !== 3'b000)
      $fatal(1, "TEST_FAIL asynchronous reset did not clear valid");
    repeat (2) @(posedge clk_i);
    #1;
    if (inject_valid_o !== 4'b0000 || dut.deferred_valid_q !== 3'b000)
      $fatal(1, "TEST_FAIL reset did not suppress active input");
    @(negedge clk_i);
    drive_input(0, 0, 0);
    rst_ni = 1'b1;
    #1;
    if (inject_valid_o !== 4'b0000)
      $fatal(1, "TEST_FAIL stale valid on reset release");
  endtask

  task automatic run_schedule(
    input [15:0] starts, input integer slots, input integer base_id,
    input logic mixed, input logic require_ii1, input logic trace_cycle
  );
    integer input_cycle [0:15];
    logic [3:0] emitted [0:15];
    logic [255:0] reconstructed [0:15];
    integer n, k, id, expected_vectors, observed_segments, last_tile3;
    begin
      expected_vectors = 0;
      observed_segments = 0;
      last_tile3 = -1;
      for (k = 0; k < 16; k = k + 1) begin
        input_cycle[k] = -1;
        emitted[k] = '0;
        reconstructed[k] = '0;
        if (has_input(k, starts, slots)) expected_vectors = expected_vectors + 1;
      end
      for (n = 0; n < slots + 5; n = n + 1) begin
        @(posedge clk_i);
        // An SRF consumer samples the previous emission cycle before NBA.
        if (n > 0) check_outputs(n-1, starts, slots, base_id, mixed);
        #1;
        // Model a registered upstream producer becoming visible after edge
        // R. A delta-settled tile0 must already be visible in this cycle;
        // no additional rising edge is allowed before its first assertion.
        id = vector_number(n < slots ? n : slots, starts);
        drive_input(has_input(n, starts, slots), base_id + id, mixed);
        if (vector_valid_i) input_cycle[id] = n;
        #1;
        check_outputs(n, starts, slots, base_id, mixed);
        for (integer tile = 0; tile < 4; tile = tile + 1) begin
          if (inject_valid_o[tile]) begin
            k = n - tile;
            id = vector_number(k, starts);
            if (input_cycle[id] < 0 || n-input_cycle[id] != tile)
              $fatal(1, "TEST_FAIL vector=%0d tile=%0d measured cycle delta=%0d",
                     id, tile, n-input_cycle[id]);
            if (emitted[id][tile])
              $fatal(1, "TEST_FAIL duplicate vector=%0d tile=%0d", id, tile);
            emitted[id][tile] = 1'b1;
            reconstructed[id][tile*64 +: 64] = inject_data_o[tile*64 +: 64];
            observed_segments = observed_segments + 1;
            if (trace_cycle)
              $display("C2C_RX_REPLAY_TIMING input_cycle=%0d tile=%0d emission_cycle=%0d delta=%0d",
                       input_cycle[id], tile, n, n-input_cycle[id]);
            if (tile == 3) begin
              if (require_ii1 && last_tile3 >= 0 && n != last_tile3+1)
                $fatal(1, "TEST_FAIL nonconsecutive vector completion cycles");
              last_tile3 = n;
            end
          end
        end
      end
      if (observed_segments != 4*expected_vectors)
        $fatal(1, "TEST_FAIL segment count got=%0d expected=%0d",
               observed_segments, 4*expected_vectors);
      for (id = 0; id < expected_vectors; id = id + 1) begin
        if (emitted[id] !== 4'b1111 || reconstructed[id] !== vector_pattern(base_id+id))
          $fatal(1, "TEST_FAIL vector=%0d packing symmetry/missing tile", id);
      end
    end
  endtask

  task automatic reset_while_active;
    // Three inputs ensure all deferred stages contain data before reset.
    for (integer n = 0; n < 3; n = n + 1) begin
      @(posedge clk_i);
      #1;
      drive_input(1, n, 1);
      #1;
      check_outputs(n, 16'h0007, 3, 0, 1);
    end
    @(posedge clk_i);
    #1;
    drive_input(0, 0, 0);
    #1;
    check_outputs(3, 16'h0007, 3, 0, 1);
    if (dut.deferred_valid_q !== 3'b111)
      $fatal(1, "TEST_FAIL reset scenario did not occupy deferred pipeline");
    reset_replay();
    run_schedule(16'h0000, 1, 0, 0, 0, 0);
    run_schedule(16'h0001, 1, 2, 0, 0, 0);
  endtask

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b0;
    vector_valid_i = 1'b0;
    vector_payload_i = '0;
    vector_stream_idx_i = '0;
    reset_replay();
    run_schedule(16'h0001, 1, 0, 0, 0, 1);
    $display("C2C_RX_REPLAY_SINGLE PASS");
    reset_replay();
    run_schedule(16'h000f, 4, 0, 0, 1, 0);
    $display("C2C_RX_REPLAY_II1 PASS");
    reset_replay();
    run_schedule(16'h000f, 4, 0, 1, 1, 0);
    $display("C2C_RX_REPLAY_STREAM PASS");
    reset_replay();
    // Whole-vector bubbles: V0, bubble, V1, bubble, V2.
    run_schedule(16'h0015, 5, 0, 1, 0, 0);
    $display("C2C_RX_REPLAY_BUBBLE PASS");
    reset_replay();
    reset_while_active();
    $display("C2C_RX_REPLAY_RESET PASS");
    $display("C2C_RX_REPLAY_PACKING PASS");
    $display("C2C_RX_REPLAY_CYCLE PASS");
    $display("C2C_RX_REPLAY TEST_PASS");
    $finish;
  end

  initial begin
    #10000;
    $fatal(1, "TEST_FAIL simulation timeout");
  end
endmodule
