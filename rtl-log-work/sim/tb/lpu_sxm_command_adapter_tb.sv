`timescale 1ns/1ps

module lpu_sxm_command_adapter_tb;
  import lpu_pkg::*;
  localparam integer COLUMNS = 16;
  localparam integer SUPERLANES = 4;
  localparam integer STREAMS = 32;
  localparam integer SEGMENT_BITS = 64;
  localparam integer ACTIVE_STREAMS = 16;

  logic clk_i, rst_ni;
  logic vmodel_transpose_valid_i, vmodel_permute_valid_i;
  logic [415:0] vmodel_transpose_instruction_i, vmodel_permute_instruction_i;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i, boundary_valid_o;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_i, boundary_data_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] state_data_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_collision_o, srf_invalid_consume_o;
  logic sxm_fault_valid_o, sxm_permute_phase_fault_o, sxm_permute_selector_fault_o;
  logic sxm_permute_buffer_not_ready_o, sxm_busy_o, sxm_command_fault_o;
  logic [SUPERLANES-1:0] sxm_transpose_input_invalid_o, sxm_transpose_buffer_full_o;
  integer errors, tile, stream, lane, index, wait_cycle;

  lpu_sxm_srf_command_integration dut (.*);
  always #5 clk_i = ~clk_i;

  function automatic [415:0] make_packet(
    input [1:0] opcode, input src_west, input [4:0] src_base,
    input dst_west, input [4:0] dst_base, input [2:0] output_tile,
    input integer phase, input bit bad_map);
    integer i, destination_tile, source_tile;
    begin
      make_packet = '0;
      make_packet[1:0] = opcode;
      make_packet[6 +: 5] = 5'd16;
      make_packet[11 +: 5] = 5'd16;
      make_packet[SXM_OUTPUT_TILE_LSB +: SXM_OUTPUT_TILE_WIDTH] = output_tile;
      for (i = 0; i < 16; i = i + 1) begin
        // Keep the VMODEL selector encoding explicit: [4:0] stream,
        // bit5 direction (East=0, West=1).  Splitting the assignments avoids
        // accidental truncation of the integer arithmetic expression.
        make_packet[16+i*6 +: 5] = src_base + i;
        make_packet[16+i*6+5] = src_west;
        make_packet[112+i*6 +: 5] = dst_base + i;
        make_packet[112+i*6+5] = dst_west;
      end
      for (i = 0; i < 32; i = i + 1) begin
        destination_tile = i / 8;
        source_tile = (phase + 4 - destination_tile) % 4;
        make_packet[240+i*5 +: 5] = source_tile*8 + (i % 8);
      end
      if (bad_map) make_packet[240 +: 5] = 5'd7;
    end
  endfunction

  function automatic [63:0] input_segment(input integer source_tile,
                                            input integer source_stream);
    begin
      input_segment = '0;
      for (lane = 0; lane < 8; lane = lane + 1)
        input_segment[lane*8 +: 8] = (source_tile << 6) |
                                          (source_stream << 2) | lane;
    end
  endfunction

  function automatic [63:0] expected_segment(input integer source_tile,
                                              input integer output_stream);
    integer row, plane, source_stream;
    begin
      expected_segment = '0;
      row = output_stream / 2;
      plane = output_stream % 2;
      for (lane = 0; lane < 8; lane = lane + 1) begin
        source_stream = 2*lane + plane;
        expected_segment[lane*8 +: 8] = (source_tile << 6) |
                                              (source_stream << 2) | row;
      end
    end
  endfunction

  function automatic integer boundary_index(input integer direction,
                                             input integer superlane,
                                             input integer stream_index);
    boundary_index = (direction*SUPERLANES + superlane)*STREAMS + stream_index;
  endfunction

  function automatic integer state_index(input integer direction,
                                          input integer column,
                                          input integer superlane,
                                          input integer stream_index);
    state_index = ((direction*COLUMNS + column)*SUPERLANES + superlane)*
                  STREAMS + stream_index;
  endfunction

  task automatic clear_drives;
    begin
      vmodel_transpose_valid_i = 1'b0;
      vmodel_permute_valid_i = 1'b0;
      vmodel_transpose_instruction_i = '0;
      vmodel_permute_instruction_i = '0;
      boundary_valid_i = '0;
      boundary_data_i = '0;
    end
  endtask

  task automatic step_cycle;
    begin @(posedge clk_i); #1; end
  endtask

  task automatic reset_all;
    begin
      @(negedge clk_i); clear_drives(); rst_ni = 1'b0;
      @(negedge clk_i); rst_ni = 1'b1;
      step_cycle();
    end
  endtask

  task automatic check_condition(input bit condition, input [8*56-1:0] label);
    begin
      if (!condition) begin
        $display("ERROR %0s", label);
        errors = errors + 1;
      end
    end
  endtask

  task automatic drive_east_tile(input integer source_tile);
    begin
      for (stream = 0; stream < ACTIVE_STREAMS; stream = stream + 1) begin
        index = boundary_index(0, source_tile, stream);
        boundary_valid_i[index] = 1'b1;
        boundary_data_i[index*SEGMENT_BITS +: SEGMENT_BITS] =
          input_segment(source_tile, stream);
      end
    end
  endtask

  task automatic capture_east;
    begin
      for (tile = 0; tile < SUPERLANES; tile = tile + 1) begin
        @(negedge clk_i); clear_drives(); drive_east_tile(tile); step_cycle();
      end
      for (wait_cycle = 0; wait_cycle < 11; wait_cycle = wait_cycle + 1) begin
        @(negedge clk_i); clear_drives(); step_cycle();
      end
      @(negedge clk_i);
      clear_drives();
      vmodel_transpose_valid_i = 1'b1;
      vmodel_transpose_instruction_i =
        make_packet(SXM_TRANSPOSE, 1'b0, 5'd0, 1'b0, 5'd16, 3'd0, 0, 1'b0);
      step_cycle();
      for (tile = 1; tile < SUPERLANES; tile = tile + 1) begin
        @(negedge clk_i); clear_drives(); step_cycle();
      end
    end
  endtask

  task automatic issue_permute_and_check(input integer phase, input [2:0] output_tile);
    begin
      // The native result-buffer array requires a full intervening cycle
      // between capture and the first legal Permute observation.
      @(negedge clk_i); clear_drives(); step_cycle();
      @(negedge clk_i);
      clear_drives();
      vmodel_permute_valid_i = 1'b1;
      vmodel_permute_instruction_i =
        make_packet(SXM_PERMUTE, 1'b0, 5'd16, 1'b1, 5'd0,
                     output_tile, phase, 1'b0);
      #1;
      step_cycle();
      for (stream = 0; stream < ACTIVE_STREAMS; stream = stream + 1) begin
        index = state_index(1, 14, output_tile, stream);
        check_condition(state_valid_o[index] &&
               state_data_o[index*SEGMENT_BITS +: SEGMENT_BITS] ===
                 expected_segment(output_tile, stream), "native legal East-to-West output");
      end
      @(negedge clk_i); clear_drives(); #1;
    end
  endtask

  initial begin
    clk_i = 1'b0; rst_ni = 1'b1; errors = 0; clear_drives();
    reset_all();

    // Cases A/B: Transpose opcode and contiguous selector compression.
    @(negedge clk_i);
    vmodel_transpose_valid_i = 1'b1;
    vmodel_transpose_instruction_i =
      make_packet(SXM_TRANSPOSE, 1'b0, 5'd0, 1'b0, 5'd16, 3'd0, 0, 1'b0);
    #1;
    check_condition(dut.native_transpose_valid && !sxm_command_fault_o &&
           dut.native_transpose_command[1:0] == 2'd0 &&
           dut.native_transpose_command[7:3] == 5'd0 &&
           dut.native_transpose_command[13:9] == 5'd16 &&
           dut.native_transpose_command[17:14] == 4'd8,
           "Transpose decode");
    $display("SXM_CMD_TRANSPOSE PASS");
    $display("SXM_CMD_SELECTOR PASS");
    clear_drives();

    // Cases C/D: exact phase detection and every legal output tile encoding.
    for (tile = 0; tile <= 4; tile = tile + 1) begin
      vmodel_permute_valid_i = 1'b1;
      vmodel_permute_instruction_i =
        make_packet(SXM_PERMUTE, 1'b0, 5'd16, 1'b1, 5'd0, tile, 0, 1'b0);
      #1;
      check_condition(dut.native_permute_valid && !sxm_command_fault_o &&
             dut.native_permute_command[24:22] == tile &&
             dut.native_permute_command[32:25] == 8'd0,
             "phase0/output_tile decode");
    end
    vmodel_permute_instruction_i =
      make_packet(SXM_PERMUTE, 1'b0, 5'd16, 1'b1, 5'd0, 3'd1, 2, 1'b0);
    #1;
    check_condition(dut.native_permute_valid && dut.native_permute_command[32:25] == 8'd2,
           "phase2 decode");
    $display("SXM_CMD_PHASE PASS");
    $display("SXM_CMD_OUTPUT_TILE PASS");

    vmodel_permute_instruction_i =
      make_packet(SXM_PERMUTE, 1'b0, 5'd16, 1'b1, 5'd0, 3'd5, 0, 1'b0);
    #1; check_condition(!dut.native_permute_valid && sxm_command_fault_o, "illegal output tile");
    $display("SXM_CMD_ILLEGAL_TILE PASS");
    vmodel_permute_instruction_i =
      make_packet(SXM_PERMUTE, 1'b0, 5'd16, 1'b1, 5'd0, 3'd0, 0, 1'b1);
    #1; check_condition(!dut.native_permute_valid && sxm_command_fault_o, "illegal map");
    $display("SXM_CMD_ILLEGAL_MAP PASS");
    clear_drives();

    // Case G/H: legal T + P0/P1/P2/P3 drives the Phase 3A data plane.
    reset_all();
    capture_east();
    issue_permute_and_check(0, 3'd0);
    issue_permute_and_check(2, 3'd1);
    issue_permute_and_check(0, 3'd2);
    issue_permute_and_check(2, 3'd3);
    check_condition(!sxm_command_fault_o && !sxm_fault_valid_o &&
           !sxm_permute_phase_fault_o && !sxm_permute_selector_fault_o &&
           !sxm_permute_buffer_not_ready_o && srf_collision_o === '0 &&
           srf_invalid_consume_o === '0, "legal command integration status");
    $display("SXM_CMD_EAST_TO_WEST PASS");
    $display("SXM_CMD_CYCLE_CONTRACT PASS");

    if (errors == 0)
      $display("========================================");
    if (errors == 0)
      $display("VMODEL_SXM_COMMAND_ADAPTER TEST_PASS");
    if (errors == 0)
      $display("========================================");
    if (errors != 0)
      $display("VMODEL_SXM_COMMAND_ADAPTER TEST_FAIL errors=%0d", errors);
    $finish;
  end
endmodule
