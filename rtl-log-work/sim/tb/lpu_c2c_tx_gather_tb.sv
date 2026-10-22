`timescale 1ns/1ps

module lpu_c2c_tx_gather_tb;
  localparam integer SEGMENT_BITS = 64;
  logic clk_i, rst_ni;
  logic [255:0] tile_data_i;
  logic [3:0] tile_valid_i;
  logic [3:0] tile_consume_o;
  logic completed_valid_o;
  logic [255:0] completed_payload_o;
  integer checked_cycles;

  c2c_tx_gather #(
    .SEGMENT_BITS(SEGMENT_BITS)
  ) dut (.*);

  always #5 clk_i = ~clk_i;

  function automatic [63:0] segment_pattern(
    input integer vector_id, input integer tile
  );
    for (integer lane = 0; lane < 8; lane = lane + 1)
      segment_pattern[lane*8 +: 8] = 8'(vector_id*32 + tile*8 + lane + 1);
  endfunction

  function automatic [255:0] vector_pattern(input integer vector_id);
    for (integer tile = 0; tile < 4; tile = tile + 1)
      vector_pattern[tile*64 +: 64] = segment_pattern(vector_id, tile);
  endfunction

  task automatic check_cycle(
    input integer cycle_id,
    input [3:0] expected_consume,
    input logic expected_valid,
    input [255:0] expected_payload
  );
    if (tile_consume_o !== expected_consume)
      $fatal(1, "TEST_FAIL cycle=%0d consume got=%b expected=%b",
             cycle_id, tile_consume_o, expected_consume);
    if (completed_valid_o !== expected_valid)
      $fatal(1, "TEST_FAIL cycle=%0d completion got=%b expected=%b",
             cycle_id, completed_valid_o, expected_valid);
    if (expected_valid) begin
      if (completed_payload_o !== expected_payload)
        $fatal(1, "TEST_FAIL cycle=%0d payload got=%h expected=%h",
               cycle_id, completed_payload_o, expected_payload);
    end
  endtask

  task automatic reset_dut;
    @(negedge clk_i);
    rst_ni = 1'b0;
    // Even valid candidates must not be consumed during reset.
    tile_valid_i = 4'b1111;
    tile_data_i = '1;
    #1;
    check_cycle(-1, 4'b0000, 1'b0, '0);
    if ({dut.stage2_valid_q, dut.stage1_valid_q, dut.stage0_valid_q} !== 3'b000)
      $fatal(1, "TEST_FAIL reset did not clear partial valid state");
    repeat (2) @(posedge clk_i);
    @(negedge clk_i);
    tile_valid_i = '0;
    rst_ni = 1'b1;
    #1;
    check_cycle(-1, 4'b0000, 1'b0, '0);
  endtask

  // Independent schedule oracle: a start at k supplies tile t at k+t,
  // and must complete at k+3. No DUT internal state is used by the oracle.
  task automatic run_schedule(
    input [15:0] starts,
    input integer start_slots,
    input logic require_ii1
  );
    integer n, tile, k, completed_count, expected_count, last_completed;
    logic [3:0] expected_consume;
    logic expected_valid;
    logic [255:0] expected_payload;
    begin
      completed_count = 0;
      expected_count = 0;
      last_completed = -1;
      for (k = 0; k < start_slots; k = k + 1)
        if (starts[k]) expected_count = expected_count + 1;
      for (n = 0; n < start_slots + 5; n = n + 1) begin
        @(negedge clk_i);
        tile_valid_i = '0;
        // Invalid data is deliberately nonzero; it must not create work.
        tile_data_i = {4{64'hdeadbeef01234567}};
        for (tile = 0; tile < 4; tile = tile + 1) begin
          k = n - tile;
          if (k >= 0 && k < start_slots) begin
            if (starts[k]) begin
              tile_valid_i[tile] = 1'b1;
              tile_data_i[tile*64 +: 64] = segment_pattern(k, tile);
            end
          end
        end
        expected_consume = tile_valid_i;
        expected_valid = 1'b0;
        expected_payload = '0;
        k = n - 3;
        if (k >= 0 && k < start_slots) begin
          if (starts[k]) begin
            expected_valid = 1'b1;
            expected_payload = vector_pattern(k);
          end
        end
        #1;
        check_cycle(n, expected_consume, expected_valid,
                    expected_payload);
        // Check at the capture edge before NBA, not after partial stages
        // advance. A future downstream queue samples completion here too.
        @(posedge clk_i);
        check_cycle(n, expected_consume, expected_valid,
                    expected_payload);
        checked_cycles = checked_cycles + 1;
        if (expected_valid) begin
          if (require_ii1 && last_completed >= 0 && n != last_completed + 1)
            $fatal(1, "TEST_FAIL nonconsecutive completions");
          last_completed = n;
          completed_count = completed_count + 1;
        end
      end
      if (completed_count != expected_count)
        $fatal(1, "TEST_FAIL completion count got=%0d expected=%0d",
               completed_count, expected_count);
    end
  endtask

  task automatic check_reset_flush;
    integer n, tile;
    begin
      // Fill all three partial stages with distinct in-flight vectors.
      for (n = 0; n < 3; n = n + 1) begin
        @(negedge clk_i);
        tile_valid_i = '0;
        tile_data_i = '0;
        for (tile = 0; tile <= n; tile = tile + 1) begin
          tile_valid_i[tile] = 1'b1;
          tile_data_i[tile*64 +: 64] = segment_pattern(n-tile, tile);
        end
        #1;
        check_cycle(n, tile_valid_i, 1'b0, '0);
        @(posedge clk_i);
      end
      #1;
      if ({dut.stage2_valid_q, dut.stage1_valid_q, dut.stage0_valid_q} !== 3'b111)
        $fatal(1, "TEST_FAIL reset test did not fill all partial stages");
      reset_dut();
      // Empty cycles must not resurrect any discarded partial vector.
      run_schedule(16'h0000, 1, 1'b0);
    end
  endtask

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b0;
    tile_data_i = '0;
    tile_valid_i = '0;
    checked_cycles = 0;
    reset_dut();
    run_schedule(16'h0001, 1, 1'b0);
    $display("C2C_TX_GATHER_SINGLE PASS");
    reset_dut();
    run_schedule(16'h000f, 4, 1'b1);
    $display("C2C_TX_GATHER_II1 PASS");
    reset_dut();
    // A whole-vector bubble at start slot 1, not a malformed schedule.
    run_schedule(16'h000d, 4, 1'b0);
    check_reset_flush();
    $display("C2C_TX_GATHER_VALID PASS");
    if (checked_cycles != 30)
      $fatal(1, "TEST_FAIL unexpected checked cycle count=%0d", checked_cycles);
    $display("C2C_TX_GATHER_CONSUME PASS");
    $display("C2C_TX_GATHER TEST_PASS");
    $finish;
  end

  initial begin
    #10000;
    $fatal(1, "TEST_FAIL simulation timeout");
  end
endmodule
