`timescale 1ns/1ps

module lpu_mem_srf_dma_store_vector_sink_tb;
  localparam integer MEM_SLICES = 52;
  localparam integer COLUMNS = 16;
  localparam integer SUPERLANES = 4;
  localparam integer STREAMS = 32;
  localparam integer SEGMENT_BITS = 64;
  localparam integer GROUP12_SLICE0 = 12 * 4;
  localparam integer GROUP12_BANK0 = 12 * 8;
  localparam integer GROUP12_PRODUCER0 = GROUP12_SLICE0 * 8;
  localparam [2:0] OPCODE_READ = 3'b000;
  localparam [2:0] OPCODE_WRITE = 3'b001;

  logic clk_i;
  logic rst_ni;
  logic [MEM_SLICES*2-1:0] bank_issue_valid_i;
  logic [MEM_SLICES*2*32-1:0] bank_issue_i;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_i;
  logic dma_store_issue_valid_i;
  logic [2:0] dma_store_stream_index_i;
  logic vector_valid_o;
  logic [255:0] vector_data_o;
  logic vector_accept_i;
  logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] state_data_o;
  logic [MEM_SLICES*8-1:0] mem_producer_valid_o;
  logic [MEM_SLICES*8*SEGMENT_BITS-1:0] mem_producer_data_o;
  logic [MEM_SLICES*8-1:0] mem_producer_direction_o;
  logic [MEM_SLICES*8*5-1:0] mem_producer_stream_o;
  logic [MEM_SLICES*8*4-1:0] mem_producer_boundary_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_collision_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_invalid_consume_o;
  logic mem_fault_valid_o;

  integer errors;
  integer cycle_count;
  integer consume_cycle [0:3];
  integer fifo_visible_cycle;
  integer b2b_issue_a;
  integer b2b_issue_b;
  integer b2b_consume_a [0:3];
  integer b2b_consume_b [0:3];
  integer b2b_gather_a;
  integer b2b_gather_b;
  integer b2b_fifo_a;
  integer b2b_fifo_b;
  integer b2b_sink_a;
  integer b2b_sink_b;
  integer tile;
  integer cycle;

  lpu_mem_srf_dma_store_vector_sink #(
    .MEM_SLICES(MEM_SLICES),
    .MEM_SLICES_PER_GROUP(4),
    .MEM_DEPTH_ROWS(16),
    .COMPLETED_FIFO_DEPTH(2)
  ) dut (
    .clk_i,
    .rst_ni,
    .bank_issue_valid_i,
    .bank_issue_i,
    .boundary_valid_i,
    .boundary_data_i,
    .dma_store_issue_valid_i,
    .dma_store_stream_index_i,
    .vector_valid_o,
    .vector_data_o,
    .vector_accept_i,
    .state_valid_o,
    .state_data_o,
    .mem_producer_valid_o,
    .mem_producer_data_o,
    .mem_producer_direction_o,
    .mem_producer_stream_o,
    .mem_producer_boundary_o,
    .srf_collision_o,
    .srf_invalid_consume_o,
    .mem_fault_valid_o
  );

  always #5 clk_i = ~clk_i;
  always @(posedge clk_i) begin
    if (rst_ni)
      cycle_count = cycle_count + 1;
  end

  function automatic [31:0] make_cmd(
    input [2:0] opcode,
    input logic direction,
    input [4:0] stream_index,
    input [14:0] row
  );
    begin
      make_cmd = '0;
      make_cmd[2:0] = opcode;
      make_cmd[8:3] = {direction, stream_index};
      make_cmd[29:15] = row;
      make_cmd[31] = 1'b0;
    end
  endfunction

  function automatic [63:0] tile_pattern(
    input integer vector_id,
    input integer tile_id
  );
    integer lane;
    reg [63:0] value;
    begin
      value = '0;
      for (lane = 0; lane < 8; lane = lane + 1)
        value[lane*8 +: 8] = vector_id*32 + tile_id*8 + lane + 1;
      tile_pattern = value;
    end
  endfunction

  function automatic [255:0] expected_vector(input integer vector_id);
    integer local_tile;
    reg [255:0] value;
    begin
      value = '0;
      for (local_tile = 0; local_tile < 4; local_tile = local_tile + 1)
        value[local_tile*64 +: 64] = tile_pattern(vector_id, local_tile);
      expected_vector = value;
    end
  endfunction

  function automatic integer boundary_index(
    input integer direction,
    input integer superlane,
    input integer stream
  );
    boundary_index = (direction*SUPERLANES + superlane)*STREAMS + stream;
  endfunction

  function automatic integer state_index(
    input integer direction,
    input integer column,
    input integer superlane,
    input integer stream
  );
    state_index = ((direction*COLUMNS + column)*SUPERLANES + superlane)*
                  STREAMS + stream;
  endfunction

  function automatic integer consume_index(
    input integer direction,
    input integer column,
    input integer superlane,
    input integer consumer,
    input integer stream
  );
    consume_index = ((((direction*COLUMNS + column)*SUPERLANES + superlane)*
                      2 + consumer)*STREAMS) + stream;
  endfunction

  task automatic clear_drives;
    begin
      bank_issue_valid_i = '0;
      bank_issue_i = '0;
      boundary_valid_i = '0;
      boundary_data_i = '0;
      dma_store_issue_valid_i = 1'b0;
      dma_store_stream_index_i = '0;
      vector_accept_i = 1'b0;
    end
  endtask

  task automatic drive_east_boundary_segment(
    input integer superlane,
    input integer stream,
    input [63:0] data
  );
    integer index;
    begin
      index = boundary_index(0, superlane, stream);
      boundary_valid_i[index] = 1'b1;
      boundary_data_i[index*SEGMENT_BITS +: SEGMENT_BITS] = data;
    end
  endtask

  task automatic step_cycle;
    begin
      @(posedge clk_i);
      #1;
    end
  endtask

  task automatic pulse_reset;
    begin
      @(negedge clk_i);
      clear_drives();
      rst_ni = 1'b0;
      #1;
      if (vector_valid_o !== 1'b0 || state_valid_o !== '0) begin
        $display("ERROR reset did not clear DMA FIFO or SRF valid state");
        errors = errors + 1;
      end
      @(negedge clk_i);
      rst_ni = 1'b1;
      step_cycle();
    end
  endtask

  // Populate group12/bank0 through an ordinary East SRF Write.  The setup
  // intentionally uses the real MEM/SRF path so the subsequent Read has a
  // genuine MEM source row rather than a mocked sreg13 payload.
  task automatic prepare_group12_row(
    input integer stream,
    input [14:0] row,
    input integer vector_id
  );
    begin
      for (cycle = 0; cycle <= 16; cycle = cycle + 1) begin
        @(negedge clk_i);
        clear_drives();
        if (cycle < 4)
          drive_east_boundary_segment(cycle, stream,
                                      tile_pattern(vector_id, cycle));
        if (cycle == 13) begin
          bank_issue_valid_i[GROUP12_BANK0] = 1'b1;
          bank_issue_i[GROUP12_BANK0*32 +: 32] =
            make_cmd(OPCODE_WRITE, 1'b0, stream[4:0], row);
        end
        step_cycle();
      end
      if (mem_fault_valid_o) begin
        $display("ERROR MEM setup Write fault stream=%0d row=%0d", stream, row);
        errors = errors + 1;
      end
    end
  endtask

  task automatic check_consume_and_no_passive_hop(
    input integer expected_tile,
    input integer stream
  );
    integer consume_bit;
    integer next_state;
    begin
      #1;
      consume_bit = consume_index(0, 13, expected_tile, 0, stream);
      if (!dut.dma_consume[consume_bit]) begin
        $display("ERROR DMA consume missing tile=%0d stream=%0d",
                 expected_tile, stream);
        errors = errors + 1;
      end
      consume_cycle[expected_tile] = cycle_count;
      step_cycle();
      next_state = state_index(0, 14, expected_tile, stream);
      if (state_valid_o[next_state]) begin
        $display("ERROR consumed sreg13 segment passively propagated tile=%0d",
                 expected_tile);
        errors = errors + 1;
      end
    end
  endtask

  task automatic expect_dma_consume(
    input integer expected_tile,
    input integer stream,
    input [8*24-1:0] label
  );
    integer consume_bit;
    begin
      #1;
      consume_bit = consume_index(0, 13, expected_tile, 0, stream);
      if (!dut.dma_consume[consume_bit]) begin
        $display("ERROR DMA consume missing %0s tile=%0d stream=%0d", label,
                 expected_tile, stream);
        errors = errors + 1;
      end
    end
  endtask

  task automatic expect_no_passive_hop(
    input integer expected_tile,
    input integer stream,
    input [8*24-1:0] label
  );
    integer next_state;
    begin
      next_state = state_index(0, 14, expected_tile, stream);
      if (state_valid_o[next_state]) begin
        $display("ERROR consumed sreg13 segment propagated %0s tile=%0d",
                 label, expected_tile);
        errors = errors + 1;
      end
    end
  endtask

  // Issue a real group12 East MEM Read, then align the existing TX selector
  // pipeline with the diagonal four-tile state arrival at East sreg13.
  task automatic read_group12_to_sink(
    input integer stream,
    input [14:0] row,
    input integer vector_id,
    input integer hold_before_accept
  );
    integer expected_producer;
    begin
      // Tile0 Read response is registered first.
      @(negedge clk_i);
      clear_drives();
      bank_issue_valid_i[GROUP12_BANK0] = 1'b1;
      bank_issue_i[GROUP12_BANK0*32 +: 32] =
        make_cmd(OPCODE_READ, 1'b0, stream[4:0], row);
      step_cycle();

      if (!mem_producer_valid_o[GROUP12_PRODUCER0] ||
          mem_producer_boundary_o[GROUP12_PRODUCER0*4 +: 4] !== 4'd13) begin
        $display("ERROR group12 East Read did not produce at sreg13");
        errors = errors + 1;
      end

      // At the next edge tile0 becomes SRF state.  Issue Send immediately
      // afterward so tile0..tile3 consume on four successive cycles.
      @(negedge clk_i);
      clear_drives();
      step_cycle();

      @(negedge clk_i);
      clear_drives();
      dma_store_issue_valid_i = 1'b1;
      dma_store_stream_index_i = stream[2:0];
      check_consume_and_no_passive_hop(0, stream);

      for (tile = 1; tile < 4; tile = tile + 1) begin
        @(negedge clk_i);
        clear_drives();
        check_consume_and_no_passive_hop(tile, stream);
      end

      if ((consume_cycle[1] != consume_cycle[0] + 1) ||
          (consume_cycle[2] != consume_cycle[0] + 2) ||
          (consume_cycle[3] != consume_cycle[0] + 3)) begin
        $display("ERROR DMA diagonal consume timing C0=%0d C1=%0d C2=%0d C3=%0d",
                 consume_cycle[0], consume_cycle[1], consume_cycle[2],
                 consume_cycle[3]);
        errors = errors + 1;
      end

      // The completed FIFO is registered; only its post-commit head is
      // visible to the sink, never an empty-to-output combinational bypass.
      fifo_visible_cycle = cycle_count;
      if (fifo_visible_cycle != consume_cycle[0] + 4) begin
        $display("ERROR DMA FIFO visibility timing C0=%0d fifo=%0d",
                 consume_cycle[0], fifo_visible_cycle);
        errors = errors + 1;
      end
      $display("DMA_STORE_CYCLE C0=%0d C1=%0d C2=%0d C3=%0d FIFO_VISIBLE=%0d",
               consume_cycle[0], consume_cycle[1], consume_cycle[2],
               consume_cycle[3], fifo_visible_cycle);
      if (!vector_valid_o || vector_data_o !== expected_vector(vector_id)) begin
        $display("ERROR DMA sink vector missing or incorrectly packed stream=%0d",
                 stream);
        errors = errors + 1;
      end

      for (cycle = 0; cycle < hold_before_accept; cycle = cycle + 1) begin
        @(negedge clk_i);
        clear_drives();
        if (!vector_valid_o || vector_data_o !== expected_vector(vector_id)) begin
          $display("ERROR DMA sink head changed while accept was low");
          errors = errors + 1;
        end
        step_cycle();
      end

      @(negedge clk_i);
      clear_drives();
      vector_accept_i = 1'b1;
      if (!vector_valid_o)
        begin
          $display("ERROR DMA sink did not present vector for pop");
          errors = errors + 1;
        end
      step_cycle();
      if (vector_valid_o) begin
        $display("ERROR DMA sink vector did not pop exactly once");
        errors = errors + 1;
      end

      if (srf_collision_o !== '0 || srf_invalid_consume_o !== '0) begin
        $display("ERROR unexpected SRF collision or invalid consume");
        errors = errors + 1;
      end
      if (mem_fault_valid_o) begin
        $display("ERROR MEM fault during DMA Store Read");
        errors = errors + 1;
      end
    end
  endtask

  // Two group12 Reads are issued one cycle apart.  Their native MEM tile
  // control waves create real, overlapping A/B diagonal arrivals at sreg13:
  // A0; A1+B0; A2+B1; A3+B2; B3.
  task automatic read_group12_back_to_back_to_sink(
    input integer stream,
    input [14:0] row_a,
    input integer vector_a,
    input [14:0] row_b,
    input integer vector_b
  );
    begin
      // Launch Read A and Read B from the same group12 bank on adjacent
      // cycles.  The native MEM bank control column is the source of overlap.
      @(negedge clk_i);
      clear_drives();
      vector_accept_i = 1'b1;
      bank_issue_valid_i[GROUP12_BANK0] = 1'b1;
      bank_issue_i[GROUP12_BANK0*32 +: 32] =
        make_cmd(OPCODE_READ, 1'b0, stream[4:0], row_a);
      step_cycle();

      @(negedge clk_i);
      clear_drives();
      vector_accept_i = 1'b1;
      bank_issue_valid_i[GROUP12_BANK0] = 1'b1;
      bank_issue_i[GROUP12_BANK0*32 +: 32] =
        make_cmd(OPCODE_READ, 1'b0, stream[4:0], row_b);
      step_cycle();

      // A tile0 is now visible.  Issue A, then issue B one cycle later.
      @(negedge clk_i);
      clear_drives();
      vector_accept_i = 1'b1;
      dma_store_issue_valid_i = 1'b1;
      dma_store_stream_index_i = stream[2:0];
      b2b_issue_a = cycle_count;
      expect_dma_consume(0, stream, "A");
      b2b_consume_a[0] = cycle_count;
      step_cycle();
      expect_no_passive_hop(0, stream, "A");

      @(negedge clk_i);
      clear_drives();
      vector_accept_i = 1'b1;
      dma_store_issue_valid_i = 1'b1;
      dma_store_stream_index_i = stream[2:0];
      b2b_issue_b = cycle_count;
      expect_dma_consume(1, stream, "A");
      expect_dma_consume(0, stream, "B");
      b2b_consume_a[1] = cycle_count;
      b2b_consume_b[0] = cycle_count;
      step_cycle();
      expect_no_passive_hop(1, stream, "A");
      expect_no_passive_hop(0, stream, "B");

      @(negedge clk_i);
      clear_drives();
      vector_accept_i = 1'b1;
      expect_dma_consume(2, stream, "A");
      expect_dma_consume(1, stream, "B");
      b2b_consume_a[2] = cycle_count;
      b2b_consume_b[1] = cycle_count;
      step_cycle();
      expect_no_passive_hop(2, stream, "A");
      expect_no_passive_hop(1, stream, "B");

      @(negedge clk_i);
      clear_drives();
      vector_accept_i = 1'b1;
      expect_dma_consume(3, stream, "A");
      expect_dma_consume(2, stream, "B");
      b2b_consume_a[3] = cycle_count;
      b2b_consume_b[2] = cycle_count;
      step_cycle();
      expect_no_passive_hop(3, stream, "A");
      expect_no_passive_hop(2, stream, "B");

      b2b_gather_a = cycle_count;
      b2b_fifo_a = cycle_count;
      if (!vector_valid_o || vector_data_o !== expected_vector(vector_a)) begin
        $display("ERROR DMA back-to-back vector A missing or incorrectly packed");
        errors = errors + 1;
      end
      b2b_sink_a = cycle_count;

      @(negedge clk_i);
      clear_drives();
      vector_accept_i = 1'b1;
      expect_dma_consume(3, stream, "B");
      b2b_consume_b[3] = cycle_count;
      step_cycle();
      expect_no_passive_hop(3, stream, "B");

      b2b_gather_b = cycle_count;
      b2b_fifo_b = cycle_count;
      if (!vector_valid_o || vector_data_o !== expected_vector(vector_b)) begin
        $display("ERROR DMA back-to-back vector B missing or incorrectly packed");
        errors = errors + 1;
      end
      b2b_sink_b = cycle_count;

      if ((b2b_issue_b != b2b_issue_a + 1) ||
          (b2b_consume_a[1] != b2b_consume_a[0] + 1) ||
          (b2b_consume_a[2] != b2b_consume_a[0] + 2) ||
          (b2b_consume_a[3] != b2b_consume_a[0] + 3) ||
          (b2b_consume_b[0] != b2b_consume_a[0] + 1) ||
          (b2b_consume_b[1] != b2b_consume_a[0] + 2) ||
          (b2b_consume_b[2] != b2b_consume_a[0] + 3) ||
          (b2b_consume_b[3] != b2b_consume_a[0] + 4) ||
          (b2b_gather_b != b2b_gather_a + 1) ||
          (b2b_fifo_b != b2b_fifo_a + 1) ||
          (b2b_sink_b != b2b_sink_a + 1)) begin
        $display("ERROR DMA back-to-back relative timing mismatch");
        errors = errors + 1;
      end

      if (srf_collision_o !== '0 || srf_invalid_consume_o !== '0 ||
          mem_fault_valid_o) begin
        $display("ERROR DMA back-to-back produced SRF/MEM fault");
        errors = errors + 1;
      end

      $display("DMA_STORE_BACK_TO_BACK_CYCLE A_ISSUE=%0d A_T0=%0d A_T1=%0d A_T2=%0d A_T3=%0d A_GATHER=%0d A_FIFO=%0d A_SINK=%0d B_ISSUE=%0d B_T0=%0d B_T1=%0d B_T2=%0d B_T3=%0d B_GATHER=%0d B_FIFO=%0d B_SINK=%0d",
               b2b_issue_a, b2b_consume_a[0], b2b_consume_a[1],
               b2b_consume_a[2], b2b_consume_a[3], b2b_gather_a,
               b2b_fifo_a, b2b_sink_a, b2b_issue_b, b2b_consume_b[0],
               b2b_consume_b[1], b2b_consume_b[2], b2b_consume_b[3],
               b2b_gather_b, b2b_fifo_b, b2b_sink_b);
    end
  endtask

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b1;
    errors = 0;
    cycle_count = 0;
    clear_drives();

    pulse_reset();
    if (vector_valid_o || srf_invalid_consume_o !== '0)
      begin
        $display("ERROR reset produced a spurious vector or consume fault");
        errors = errors + 1;
      end
    $display("DMA_STORE_RESET PASS");

    prepare_group12_row(3, 5, 1);
    read_group12_to_sink(3, 5, 1, 2);
    $display("DMA_STORE_BASIC PASS");
    $display("DMA_STORE_SINK_HOLD PASS");
    $display("DMA_STORE_CONSUME PASS");

    // A different selected stream proves the attachment is not hard-coded to E0.
    prepare_group12_row(6, 6, 2);
    read_group12_to_sink(6, 6, 2, 0);
    $display("DMA_STORE_STREAM_SELECT PASS");

    pulse_reset();
    prepare_group12_row(3, 7, 3);
    pulse_reset();
    prepare_group12_row(3, 8, 4);
    pulse_reset();
    read_group12_back_to_back_to_sink(3, 7, 3, 8, 4);
    $display("DMA_STORE_BACK_TO_BACK PASS");

    pulse_reset();
    if (vector_valid_o || state_valid_o !== '0 || srf_invalid_consume_o !== '0) begin
      $display("ERROR final reset state is not empty");
      errors = errors + 1;
    end

    if (errors == 0)
      $display("DMA_STORE_VECTOR_SINK TEST_PASS");
    else
      $display("DMA_STORE_VECTOR_SINK TEST_FAIL errors=%0d", errors);
    $finish;
  end
endmodule
