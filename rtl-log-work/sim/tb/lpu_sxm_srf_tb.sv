`timescale 1ns/1ps

module lpu_sxm_srf_tb;
  localparam integer COLUMNS = 16;
  localparam integer SUPERLANES = 4;
  localparam integer STREAMS = 32;
  localparam integer SEGMENT_BITS = 64;
  localparam integer ACTIVE_STREAMS = 16;
  localparam logic SXM_WEST = 1'b0;
  localparam logic SXM_EAST = 1'b1;

  logic clk_i;
  logic rst_ni;
  logic transpose_cmd_valid_i;
  logic [95:0] transpose_cmd_i;
  logic permute_cmd_valid_i;
  logic [95:0] permute_cmd_i;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_i;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_o;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] state_data_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_collision_o;
  logic [2*COLUMNS*SUPERLANES-1:0] srf_invalid_consume_o;
  logic sxm_fault_valid_o;
  logic [SUPERLANES-1:0] sxm_transpose_input_invalid_o;
  logic [SUPERLANES-1:0] sxm_transpose_buffer_full_o;
  logic sxm_permute_phase_fault_o;
  logic sxm_permute_selector_fault_o;
  logic sxm_permute_buffer_not_ready_o;
  logic sxm_busy_o;
  integer errors;
  integer tile;
  integer stream;
  integer wait_cycle;

  lpu_sxm_srf_integration dut (.*);

  always #5 clk_i = ~clk_i;

  function automatic [95:0] make_transpose_cmd(
    input logic src_direction,
    input [4:0] src_base,
    input logic dst_direction,
    input [4:0] dst_base
  );
    begin
      make_transpose_cmd = '0;
      make_transpose_cmd[1:0] = 2'd0;
      make_transpose_cmd[2] = src_direction;
      make_transpose_cmd[7:3] = src_base;
      make_transpose_cmd[8] = dst_direction;
      make_transpose_cmd[13:9] = dst_base;
      make_transpose_cmd[17:14] = 4'd8;
    end
  endfunction

  function automatic [95:0] make_permute_cmd(
    input logic dst_direction,
    input [4:0] dst_base
  );
    begin
      make_permute_cmd = '0;
      make_permute_cmd[1:0] = 2'd1;
      make_permute_cmd[8] = dst_direction;
      make_permute_cmd[13:9] = dst_base;
      make_permute_cmd[21:18] = 4'd8;
      make_permute_cmd[24:22] = 3'd0;
      make_permute_cmd[32:25] = 8'd0;
    end
  endfunction

  function automatic [63:0] input_segment(input integer source_tile,
                                            input integer source_stream);
    integer lane;
    begin
      input_segment = '0;
      for (lane = 0; lane < 8; lane = lane + 1)
        input_segment[lane*8 +: 8] = ((source_tile & 3) << 6) |
                                      ((source_stream & 15) << 2) |
                                      (lane & 3);
    end
  endfunction

  // Expected output for phase0, destination tile0.  The transpose leaf maps
  // output {row,plane,lane} from input {2*lane+plane,row}.
  function automatic [63:0] expected_transpose_segment(
    input integer source_tile,
    input integer output_stream
  );
    integer lane;
    integer output_row;
    integer plane;
    integer source_stream;
    begin
      expected_transpose_segment = '0;
      output_row = output_stream / 2;
      plane = output_stream % 2;
      for (lane = 0; lane < 8; lane = lane + 1) begin
        source_stream = 2*lane + plane;
        expected_transpose_segment[lane*8 +: 8] =
          ((source_tile & 3) << 6) | ((source_stream & 15) << 2) |
          (output_row & 3);
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

  function automatic integer consume_index(input integer direction,
                                            input integer column,
                                            input integer superlane,
                                            input integer consumer,
                                            input integer stream_index);
    consume_index = ((((direction*COLUMNS + column)*SUPERLANES + superlane)*2 +
                     consumer)*STREAMS) + stream_index;
  endfunction

  function automatic integer inject_index(input integer direction,
                                           input integer column,
                                           input integer superlane,
                                           input integer producer,
                                           input integer stream_index);
    inject_index = ((((direction*COLUMNS + column)*SUPERLANES + superlane)*2 +
                    producer)*STREAMS) + stream_index;
  endfunction

  task automatic clear_drives;
    begin
      transpose_cmd_valid_i = 1'b0;
      transpose_cmd_i = '0;
      permute_cmd_valid_i = 1'b0;
      permute_cmd_i = '0;
      boundary_valid_i = '0;
      boundary_data_i = '0;
    end
  endtask

  task automatic drive_input_tile(input integer direction,
                                  input integer source_tile);
    integer local_stream;
    integer index;
    begin
      for (local_stream = 0; local_stream < ACTIVE_STREAMS;
           local_stream = local_stream + 1) begin
        index = boundary_index(direction, source_tile, local_stream);
        boundary_valid_i[index] = 1'b1;
        boundary_data_i[index*SEGMENT_BITS +: SEGMENT_BITS] =
          input_segment(source_tile, local_stream);
      end
    end
  endtask

  task automatic step_cycle;
    begin
      @(posedge clk_i);
      #1;
    end
  endtask

  task automatic reset_all;
    begin
      @(negedge clk_i);
      clear_drives();
      rst_ni = 1'b0;
      @(negedge clk_i);
      rst_ni = 1'b1;
      step_cycle();
    end
  endtask

  task automatic check_capture_slot1(input integer direction,
                                     input integer column,
                                     input integer source_tile);
    integer local_stream;
    integer index;
    begin
      for (local_stream = 0; local_stream < ACTIVE_STREAMS;
           local_stream = local_stream + 1) begin
        index = consume_index(direction, column, source_tile, 1, local_stream);
        if (!dut.srf_consume[index] || dut.srf_consume[index-STREAMS]) begin
          $display("ERROR SXM consumer slot mapping d=%0d tile=%0d stream=%0d",
                   direction, source_tile, local_stream);
          errors = errors + 1;
        end
      end
    end
  endtask

  task automatic capture_east;
    integer local_tile;
    begin
      // Tile N reaches sreg14 East at one-cycle offsets, matching the native
      // transpose command wave without any adapter storage.
      for (local_tile = 0; local_tile < SUPERLANES; local_tile = local_tile + 1) begin
        @(negedge clk_i);
        clear_drives();
        drive_input_tile(0, local_tile);
        step_cycle();
      end
      for (wait_cycle = 0; wait_cycle < 11; wait_cycle = wait_cycle + 1) begin
        @(negedge clk_i);
        clear_drives();
        step_cycle();
      end

      for (local_tile = 0; local_tile < SUPERLANES; local_tile = local_tile + 1) begin
        @(negedge clk_i);
        clear_drives();
        if (local_tile == 0) begin
          transpose_cmd_valid_i = 1'b1;
          transpose_cmd_i = make_transpose_cmd(SXM_EAST, 5'd0, SXM_EAST, 5'd0);
        end
        #1;
        check_capture_slot1(0, 14, local_tile);
        if (dut.sxm_sr_read_data[local_tile*ACTIVE_STREAMS*SEGMENT_BITS +: SEGMENT_BITS]
            !== input_segment(local_tile, 0)) begin
          $display("ERROR East selector/data tile=%0d", local_tile);
          errors = errors + 1;
        end
        step_cycle();
        if (state_valid_o[state_index(0, 15, local_tile, 0)]) begin
          $display("ERROR East consumed segment took passive hop tile=%0d",
                   local_tile);
          errors = errors + 1;
        end
      end
    end
  endtask

  task automatic capture_west;
    integer local_tile;
    begin
      // West input boundary commits directly into sreg15.  Staggering input
      // with the command wave proves that the adapter never reads sreg14 West.
      @(negedge clk_i);
      clear_drives();
      drive_input_tile(1, 0);
      step_cycle();
      for (local_tile = 0; local_tile < SUPERLANES; local_tile = local_tile + 1) begin
        @(negedge clk_i);
        clear_drives();
        if (local_tile < SUPERLANES-1)
          drive_input_tile(1, local_tile + 1);
        if (local_tile == 0) begin
          transpose_cmd_valid_i = 1'b1;
          transpose_cmd_i = make_transpose_cmd(SXM_WEST, 5'd0, SXM_WEST, 5'd0);
        end
        #1;
        check_capture_slot1(1, 15, local_tile);
        if (dut.sxm_sr_read_data[local_tile*ACTIVE_STREAMS*SEGMENT_BITS +: SEGMENT_BITS]
            !== input_segment(local_tile, 0)) begin
          $display("ERROR West selector/data tile=%0d", local_tile);
          errors = errors + 1;
        end
        step_cycle();
        if (state_valid_o[state_index(1, 14, local_tile, 0)]) begin
          $display("ERROR West consumed segment took passive hop tile=%0d",
                   local_tile);
          errors = errors + 1;
        end
      end
    end
  endtask

  task automatic permute_and_check(input logic destination_direction,
                                   input integer destination_column);
    integer local_stream;
    integer index;
    integer srf_direction;
    begin
      srf_direction = destination_direction ? 0 : 1;
      // The result buffers must remain ready for a full intervening cycle.
      @(negedge clk_i);
      clear_drives();
      step_cycle();

      @(negedge clk_i);
      clear_drives();
      permute_cmd_valid_i = 1'b1;
      permute_cmd_i = make_permute_cmd(destination_direction, 5'd0);
      #1;
      for (local_stream = 0; local_stream < ACTIVE_STREAMS;
           local_stream = local_stream + 1) begin
        index = inject_index(srf_direction, destination_column, 0, 1,
                             local_stream);
        if (!dut.srf_inject_valid[index] ||
            dut.srf_inject_valid[index-STREAMS] ||
            dut.srf_inject_data[index*SEGMENT_BITS +: SEGMENT_BITS] !==
              expected_transpose_segment(0, local_stream)) begin
          $display("ERROR SXM producer slot/data d=%0d stream=%0d",
                   destination_direction, local_stream);
          errors = errors + 1;
        end
      end
      step_cycle();
      for (local_stream = 0; local_stream < ACTIVE_STREAMS;
           local_stream = local_stream + 1) begin
        index = state_index(srf_direction, destination_column, 0,
                            local_stream);
        if (!state_valid_o[index] ||
            state_data_o[index*SEGMENT_BITS +: SEGMENT_BITS] !==
              expected_transpose_segment(0, local_stream)) begin
          $display("ERROR SXM producer SRF commit d=%0d stream=%0d",
                   destination_direction, local_stream);
          errors = errors + 1;
        end
      end
      // Native commands are issue pulses. Remove the stimulus after its
      // active edge so the released buffer is not re-evaluated as a second
      // Permute command in the following half-cycle.
      @(negedge clk_i);
      clear_drives();
      #1;
    end
  endtask

  task automatic check_legal_status;
    begin
      if (srf_collision_o !== '0 || srf_invalid_consume_o !== '0 ||
          sxm_fault_valid_o || sxm_transpose_input_invalid_o !== '0 ||
          sxm_transpose_buffer_full_o !== '0 || sxm_permute_phase_fault_o ||
          sxm_permute_selector_fault_o || sxm_permute_buffer_not_ready_o) begin
        $display("ERROR legal SXM/SRF status collision=%h invalid_consume=%h sxm_fault=%b input_invalid=%h buffer_full=%h phase=%b selector=%b not_ready=%b",
                 srf_collision_o, srf_invalid_consume_o, sxm_fault_valid_o,
                 sxm_transpose_input_invalid_o, sxm_transpose_buffer_full_o,
                 sxm_permute_phase_fault_o, sxm_permute_selector_fault_o,
                 sxm_permute_buffer_not_ready_o);
        errors = errors + 1;
      end
    end
  endtask

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b1;
    errors = 0;
    clear_drives();

    reset_all();
    capture_east();
    check_legal_status();
    $display("SXM_SRF_EAST_INPUT PASS");

    permute_and_check(SXM_EAST, 15);
    check_legal_status();
    $display("SXM_SRF_EAST_OUTPUT PASS");

    reset_all();
    capture_west();
    check_legal_status();
    $display("SXM_SRF_WEST_INPUT PASS");

    reset_all();
    capture_east();
    permute_and_check(SXM_WEST, 14);
    check_legal_status();
    $display("SXM_SRF_EAST_TO_WEST PASS");
    $display("SXM_SRF_SLOT1 PASS");
    $display("SXM_SRF_CYCLE_CONTRACT PASS");

    if (errors == 0)
      $display("VMODEL_SXM_SRF_INTEGRATION TEST_PASS");
    else
      $display("VMODEL_SXM_SRF_INTEGRATION TEST_FAIL errors=%0d", errors);
    $finish;
  end
endmodule
