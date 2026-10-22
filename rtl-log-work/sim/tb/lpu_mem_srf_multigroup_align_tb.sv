`timescale 1ns/1ps

// Phase 4A-R: prove the native MEM/SRF static-arrival schedule before SXM is
// connected.  Four fixed MEM groups supply E0..E15 to sreg14 East without any
// retention, stall, FIFO, or additional stream-state holder.
module lpu_mem_srf_multigroup_align_tb;
  localparam integer MEM_SLICES = 16;
  localparam integer MEM_GROUPS = 4;
  localparam integer COLUMNS = 16;
  localparam integer SUPERLANES = 4;
  localparam integer STREAMS = 32;
  localparam integer SEGMENT_BITS = 64;
  localparam integer ACTIVE_STREAMS = 16;
  localparam [14:0] PRELOAD_ROW = 15'd0;

  logic clk_i, rst_ni;
  logic [MEM_SLICES*2-1:0] vmodel_issue_valid_i, native_issue_valid_o, command_fault_o;
  logic [MEM_SLICES*2*47-1:0] vmodel_issue_instruction_i;
  logic [MEM_SLICES*2*32-1:0] native_issue_o;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i, boundary_valid_o;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_i, boundary_data_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] state_data_o;
  logic [(MEM_GROUPS+1)*256-1:0] mem_boundary_consume_o;
  logic [MEM_SLICES*8-1:0] mem_producer_valid_o, mem_producer_direction_o;
  logic [MEM_SLICES*8*SEGMENT_BITS-1:0] mem_producer_data_o;
  logic [MEM_SLICES*8*5-1:0] mem_producer_stream_o;
  logic [MEM_SLICES*8*4-1:0] mem_producer_boundary_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_collision_o, srf_invalid_consume_o;
  logic [MEM_SLICES*8-1:0] mem_internal_collision_o;
  logic [MEM_SLICES*2-1:0] mem_bank_fault_valid_o;
  logic mem_fault_valid_o, mem_busy_o;
  integer errors, tile, stream, group, local_slice, wait_cycle, capture_cycle [0:3];

  lpu_mem_srf_command_integration #(
    .MEM_SLICES(MEM_SLICES), .MEM_SLICES_PER_GROUP(4), .MEM_DEPTH_ROWS(16)
  ) dut (.*);

  always #5 clk_i = ~clk_i;

  function automatic [46:0] mem_packet(input [2:0] opcode, input west,
                                        input [4:0] stream_id, input [14:0] row);
    begin
      mem_packet = '0;
      mem_packet[2:0] = opcode;
      mem_packet[8:3] = {west, stream_id};
      mem_packet[30:15] = row;
    end
  endfunction

  function automatic [63:0] stream_pattern(input integer superlane,
                                            input integer stream_id);
    integer lane;
    begin
      stream_pattern = '0;
      for (lane = 0; lane < 8; lane = lane + 1)
        stream_pattern[lane*8 +: 8] = (superlane << 6) | (stream_id << 2) | lane;
    end
  endfunction

  function automatic integer boundary_index(input integer direction,
                                             input integer superlane,
                                             input integer stream_id);
    boundary_index = (direction*SUPERLANES + superlane)*STREAMS + stream_id;
  endfunction
  function automatic integer state_index(input integer direction, input integer column,
                                          input integer superlane, input integer stream_id);
    state_index = ((direction*COLUMNS + column)*SUPERLANES + superlane)*STREAMS + stream_id;
  endfunction
  function automatic integer producer_index(input integer slice, input integer tile_id);
    producer_index = slice*8 + tile_id;
  endfunction

  task automatic clear_drives;
    begin
      vmodel_issue_valid_i = '0;
      vmodel_issue_instruction_i = '0;
      boundary_valid_i = '0;
      boundary_data_i = '0;
    end
  endtask
  task automatic step_cycle; begin @(posedge clk_i); #1; end endtask
  task automatic check(input bit condition, input [8*64-1:0] label);
    begin if (!condition) begin $display("ERROR %0s", label); errors = errors + 1; end end
  endtask
  task automatic reset_preserve_sram;
    begin
      @(negedge clk_i); clear_drives(); rst_ni = 1'b0;
      @(negedge clk_i); rst_ni = 1'b1; step_cycle();
    end
  endtask

  task automatic drive_east_tile(input integer tile_id);
    integer index;
    begin
      for (stream = 0; stream < ACTIVE_STREAMS; stream = stream + 1) begin
        index = boundary_index(0, tile_id, stream);
        boundary_valid_i[index] = 1'b1;
        boundary_data_i[index*SEGMENT_BITS +: SEGMENT_BITS] = stream_pattern(tile_id, stream);
      end
    end
  endtask

  task automatic issue_group(input integer group_id, input [2:0] opcode);
    integer slice_id, bank_id;
    begin
      for (local_slice = 0; local_slice < 4; local_slice = local_slice + 1) begin
        slice_id = group_id*4 + local_slice;
        bank_id = slice_id*2;
        vmodel_issue_valid_i[bank_id] = 1'b1;
        vmodel_issue_instruction_i[bank_id*47 +: 47] =
          mem_packet(opcode, 1'b0, slice_id[4:0], PRELOAD_ROW);
      end
    end
  endtask

  // The four external boundary tile injections establish a command-compatible
  // 4-tile wave.  Group g starts its Write exactly g cycles after group0, when
  // tile0 is resident at sreg[g]; each group owns four distinct stream IDs.
  task automatic preload_all_streams;
    begin
      @(negedge clk_i); clear_drives(); drive_east_tile(0); step_cycle();
      for (group = 0; group < MEM_GROUPS; group = group + 1) begin
        @(negedge clk_i); clear_drives();
        if (group < SUPERLANES-1) drive_east_tile(group+1);
        issue_group(group, 3'd1);
        step_cycle();
      end
      repeat (4) begin @(negedge clk_i); clear_drives(); step_cycle(); end
      check(command_fault_o === '0 && mem_bank_fault_valid_o === '0 &&
            !mem_fault_valid_o && srf_collision_o === '0 && srf_invalid_consume_o === '0,
            "legal multi-group preload status");
    end
  endtask

  task automatic check_read_producers(input integer group_id, input integer tile_id);
    integer slice_id, pindex;
    begin
      for (local_slice = 0; local_slice < 4; local_slice = local_slice + 1) begin
        slice_id = group_id*4 + local_slice;
        pindex = producer_index(slice_id, tile_id);
        check(mem_producer_valid_o[pindex] &&
              mem_producer_data_o[pindex*SEGMENT_BITS +: SEGMENT_BITS] ===
                stream_pattern(tile_id, slice_id) &&
              mem_producer_direction_o[pindex] == 1'b0 &&
              mem_producer_stream_o[pindex*5 +: 5] == slice_id &&
              mem_producer_boundary_o[pindex*4 +: 4] == group_id+1,
              "MEM producer slice/group/boundary mapping");
      end
    end
  endtask

  // Read group g is launched g cycles after group0.  With a registered native
  // MEM Read response and 13-(g) hops from sreg[g+1] to sreg14, each tile k
  // commits at sreg14 on the common cycle: read0_issue + 14 + k.
  task automatic schedule_multigroup_reads;
    begin
      for (group = 0; group < MEM_GROUPS; group = group + 1) begin
        @(negedge clk_i); clear_drives(); issue_group(group, 3'd0);
        step_cycle();
        check_read_producers(group, 0);
      end
    end
  endtask

  task automatic check_sreg14_tile(input integer tile_id);
    integer index;
    begin
      for (stream = 0; stream < ACTIVE_STREAMS; stream = stream + 1) begin
        index = state_index(0, 14, tile_id, stream);
        check(state_valid_o[index] &&
              state_data_o[index*SEGMENT_BITS +: SEGMENT_BITS] === stream_pattern(tile_id, stream),
              "sreg14 East E0..E15 aligned");
      end
    end
  endtask

  initial begin
    clk_i = 1'b0; rst_ni = 1'b1; errors = 0; clear_drives();
    reset_preserve_sram();
    preload_all_streams();
    $display("MULTIGROUP_PRELOAD PASS");

    // Reset only control/SRF state.  Native SRAM intentionally retains the
    // legal preload, allowing the read-arrival proof to be isolated.
    reset_preserve_sram();
    schedule_multigroup_reads();
    // The last group0 tile0 producer was observed at the first issue edge.
    // After group3's issue edge, eleven more edges reach the common sreg14
    // tile0 commit cycle; subsequent tile rows appear one edge apart.
    for (wait_cycle = 0; wait_cycle < 11; wait_cycle = wait_cycle + 1) begin
      @(negedge clk_i); clear_drives(); step_cycle();
    end
    for (tile = 0; tile < SUPERLANES; tile = tile + 1) begin
      check_sreg14_tile(tile);
      capture_cycle[tile] = $time / 10;
      if (tile < SUPERLANES-1) begin @(negedge clk_i); clear_drives(); step_cycle(); end
    end

    check(srf_collision_o === '0 && srf_invalid_consume_o === '0 &&
          command_fault_o === '0 && mem_bank_fault_valid_o === '0 && !mem_fault_valid_o &&
          mem_internal_collision_o === '0, "multi-group legal status");
    if (errors == 0) begin
      $display("MULTIGROUP_E0_E15_ALIGNED PASS");
      $display("MULTIGROUP_NO_COLLISION PASS");
      $display("MULTIGROUP_TILE_PIPELINE_ALIGNMENT PASS");
      $display("CAPTURE_CYCLES tile0=%0d tile1=%0d tile2=%0d tile3=%0d",
               capture_cycle[0], capture_cycle[1], capture_cycle[2], capture_cycle[3]);
      $display("PHASE4A_PREVIOUS_BLOCKER_INVALID");
    end else begin
      $display("ARCHITECTURE_BLOCKER_CONFIRMED errors=%0d", errors);
    end
    $finish;
  end
endmodule
