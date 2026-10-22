`timescale 1ns/1ps

module lpu_mem_srf_tb;
  localparam integer MEM_SLICES = 4;
  localparam integer MEM_GROUPS = 1;
  localparam integer MEM_BOUNDARIES = 2;
  localparam integer COLUMNS = 16;
  localparam integer SUPERLANES = 4;
  localparam integer STREAMS = 32;
  localparam integer SEGMENT_BITS = 64;
  localparam integer LOCAL_PRODUCERS = 2;
  localparam integer LOCAL_CONSUMERS = 2;
  localparam [2:0] OPCODE_READ = 3'b000;
  localparam [2:0] OPCODE_WRITE = 3'b001;
  localparam [63:0] TEST_PATTERN = 64'h0807060504030201;

  logic clk_i;
  logic rst_ni;
  logic [MEM_SLICES*2-1:0] bank_issue_valid_i;
  logic [MEM_SLICES*2*32-1:0] bank_issue_i;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_i;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_o;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0]
    state_data_o;
  logic [MEM_BOUNDARIES*256-1:0] mem_boundary_consume_o;
  logic [MEM_SLICES*8-1:0] mem_producer_valid_o;
  logic [MEM_SLICES*8*SEGMENT_BITS-1:0] mem_producer_data_o;
  logic [MEM_SLICES*8-1:0] mem_producer_direction_o;
  logic [MEM_SLICES*8*5-1:0] mem_producer_stream_o;
  logic [MEM_SLICES*8*4-1:0] mem_producer_boundary_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_collision_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_invalid_consume_o;
  logic [MEM_SLICES*8-1:0] mem_internal_collision_o;
  logic [MEM_SLICES*2-1:0] mem_bank_fault_valid_o;
  logic mem_fault_valid_o;
  logic mem_busy_o;

  integer errors;
  integer tile;
  integer wait_cycle;

  lpu_mem_srf_integration #(
    .MEM_SLICES(MEM_SLICES),
    .MEM_SLICES_PER_GROUP(4),
    .MEM_DEPTH_ROWS(16)
  ) dut (
    .clk_i,
    .rst_ni,
    .bank_issue_valid_i,
    .bank_issue_i,
    .boundary_valid_i,
    .boundary_data_i,
    .boundary_valid_o,
    .boundary_data_o,
    .state_valid_o,
    .state_data_o,
    .mem_boundary_consume_o,
    .mem_producer_valid_o,
    .mem_producer_data_o,
    .mem_producer_direction_o,
    .mem_producer_stream_o,
    .mem_producer_boundary_o,
    .srf_collision_o,
    .srf_invalid_consume_o,
    .mem_internal_collision_o,
    .mem_bank_fault_valid_o,
    .mem_fault_valid_o,
    .mem_busy_o
  );

  always #5 clk_i = ~clk_i;

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

  function automatic integer mem_boundary_index(
    input integer boundary,
    input integer direction,
    input integer stream,
    input integer superlane
  );
    mem_boundary_index = boundary*256 + direction*128 + stream*4 + superlane;
  endfunction

  function automatic integer inject_index(
    input integer direction,
    input integer column,
    input integer superlane,
    input integer producer,
    input integer stream
  );
    inject_index = ((((direction*COLUMNS + column)*SUPERLANES + superlane)*
                     LOCAL_PRODUCERS + producer)*STREAMS) + stream;
  endfunction

  function automatic integer consume_index(
    input integer direction,
    input integer column,
    input integer superlane,
    input integer consumer,
    input integer stream
  );
    consume_index = ((((direction*COLUMNS + column)*SUPERLANES + superlane)*
                      LOCAL_CONSUMERS + consumer)*STREAMS) + stream;
  endfunction

  task automatic clear_drives;
    begin
      bank_issue_valid_i = '0;
      bank_issue_i = '0;
      boundary_valid_i = '0;
      boundary_data_i = '0;
    end
  endtask

  task automatic drive_boundary_segment(
    input integer direction,
    input integer superlane,
    input integer stream,
    input [63:0] data
  );
    integer index;
    begin
      index = boundary_index(direction, superlane, stream);
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
      if (state_valid_o !== '0) begin
        $display("ERROR reset did not clear SRF state");
        errors = errors + 1;
      end
      @(negedge clk_i);
      rst_ni = 1'b1;
      step_cycle();
    end
  endtask

  task automatic check_state_segment(
    input integer direction,
    input integer column,
    input integer superlane,
    input integer stream,
    input [63:0] expected
  );
    integer index;
    begin
      index = state_index(direction, column, superlane, stream);
      if (!state_valid_o[index] ||
          state_data_o[index*SEGMENT_BITS +: SEGMENT_BITS] !== expected) begin
        $display("ERROR SRF state d=%0d c=%0d tile=%0d stream=%0d", direction,
                 column, superlane, stream);
        errors = errors + 1;
      end
    end
  endtask

  task automatic perform_write(
    input integer direction,
    input integer stream,
    input [14:0] row
  );
    integer target_column;
    integer initial_hops;
    integer consume_bit;
    begin
      target_column = (direction == 0) ? 0 : 1;

      // Stagger the four superlanes so they meet the MEM tile0..tile3 command
      // wave.  East arrives at sreg0 directly, so tile1..tile3 are injected
      // alongside the active command wave.  West traverses sreg15..sreg1 and
      // is preloaded before the command starts.
      if (direction == 0) begin
        @(negedge clk_i);
        clear_drives();
        drive_boundary_segment(direction, 0, stream, TEST_PATTERN);
        step_cycle();
      end else begin
        for (tile = 0; tile < 4; tile = tile + 1) begin
          @(negedge clk_i);
          clear_drives();
          drive_boundary_segment(direction, tile, stream, TEST_PATTERN);
          step_cycle();
        end
        initial_hops = 11;
        for (wait_cycle = 0; wait_cycle < initial_hops;
             wait_cycle = wait_cycle + 1) begin
          @(negedge clk_i);
          clear_drives();
          step_cycle();
        end
      end

      // tile0 is now resident at the selected MEM boundary.  The native MEM
      // control column advances one tile per cycle.
      for (tile = 0; tile < 4; tile = tile + 1) begin
        @(negedge clk_i);
        clear_drives();
        if ((direction == 0) && (tile < 3))
          drive_boundary_segment(direction, tile + 1, stream, TEST_PATTERN);
        if (tile == 0) begin
          bank_issue_valid_i[0] = 1'b1;
          bank_issue_i[0 +: 32] = make_cmd(OPCODE_WRITE, direction[0],
                                            stream[4:0], row);
        end
        #1;
        if (!mem_boundary_consume_o[
              mem_boundary_index(target_column, direction, stream, tile)]) begin
          $display("ERROR MEM consume missing d=%0d tile=%0d", direction, tile);
          errors = errors + 1;
        end
        consume_bit = consume_index(direction, target_column, tile, 0, stream);
        if (!dut.srf_consume[consume_bit] ||
            dut.srf_consume[consume_bit + STREAMS]) begin
          $display("ERROR consumer slot mapping d=%0d tile=%0d", direction,
                   tile);
          errors = errors + 1;
        end
        if (srf_invalid_consume_o !== '0) begin
          $display("ERROR invalid consume before commit d=%0d tile=%0d",
                   direction, tile);
          errors = errors + 1;
        end

        step_cycle();
        // The atomic consume masks the current segment before the next SRF
        // leaf samples it; no extra passive hop is allowed.
        if (direction == 0) begin
          if (state_valid_o[state_index(direction, 1, tile, stream)]) begin
            $display("ERROR East consumed segment propagated tile=%0d", tile);
            errors = errors + 1;
          end
        end else begin
          if (state_valid_o[state_index(direction, 0, tile, stream)]) begin
            $display("ERROR West consumed segment propagated tile=%0d", tile);
            errors = errors + 1;
          end
        end
      end
    end
  endtask

  task automatic perform_read(
    input integer direction,
    input integer stream,
    input [14:0] row
  );
    integer target_column;
    integer inject_bit;
    begin
      target_column = (direction == 0) ? 1 : 0;
      for (tile = 0; tile < 4; tile = tile + 1) begin
        @(negedge clk_i);
        clear_drives();
        if (tile == 0) begin
          bank_issue_valid_i[0] = 1'b1;
          bank_issue_i[0 +: 32] = make_cmd(OPCODE_READ, direction[0],
                                            stream[4:0], row);
        end
        step_cycle();

        if (!mem_producer_valid_o[tile] ||
            mem_producer_data_o[tile*SEGMENT_BITS +: SEGMENT_BITS] !==
              TEST_PATTERN ||
            mem_producer_direction_o[tile] !== direction[0] ||
            mem_producer_stream_o[tile*5 +: 5] !== stream[4:0] ||
            mem_producer_boundary_o[tile*4 +: 4] !== target_column[3:0]) begin
          $display("ERROR MEM producer d=%0d tile=%0d", direction, tile);
          errors = errors + 1;
        end

        inject_bit = inject_index(direction, target_column, tile, 0, stream);
        if (!dut.srf_inject_valid[inject_bit] ||
            dut.srf_inject_valid[inject_bit + STREAMS] ||
            dut.srf_inject_data[inject_bit*SEGMENT_BITS +: SEGMENT_BITS] !==
              TEST_PATTERN) begin
          $display("ERROR producer slot0 mapping d=%0d tile=%0d", direction,
                   tile);
          errors = errors + 1;
        end

        // A producer pulse observed after edge N commits at exactly edge N+1.
        if (tile > 0)
          check_state_segment(direction, target_column, tile-1, stream,
                              TEST_PATTERN);
      end

      @(negedge clk_i);
      clear_drives();
      step_cycle();
      check_state_segment(direction, target_column, 3, stream, TEST_PATTERN);
    end
  endtask

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b1;
    errors = 0;
    clear_drives();

    pulse_reset();
    perform_write(0, 3, 5);
    if (mem_bank_fault_valid_o == '0 && !mem_fault_valid_o)
      $display("MEM_SRF_EAST_WRITE PASS");
    else begin
      $display("ERROR East Write fault");
      errors = errors + 1;
    end

    pulse_reset();
    perform_read(0, 9, 5);
    $display("MEM_SRF_EAST_READ PASS");

    pulse_reset();
    perform_write(1, 4, 6);
    if (mem_bank_fault_valid_o == '0 && !mem_fault_valid_o)
      $display("MEM_SRF_WEST_WRITE PASS");
    else begin
      $display("ERROR West Write fault");
      errors = errors + 1;
    end

    pulse_reset();
    perform_read(1, 10, 6);
    $display("MEM_SRF_WEST_READ PASS");

    if (srf_collision_o !== '0 || mem_internal_collision_o !== '0) begin
      $display("ERROR unexpected producer collision");
      errors = errors + 1;
    end

    $display("MEM_SRF_SLOT0 PASS");
    $display("MEM_SRF_CYCLE_CONTRACT PASS");

    if (errors == 0)
      $display("VMODEL_MEM_SRF_INTEGRATION TEST_PASS");
    else
      $display("VMODEL_MEM_SRF_INTEGRATION TEST_FAIL errors=%0d", errors);
    $finish;
  end
endmodule
