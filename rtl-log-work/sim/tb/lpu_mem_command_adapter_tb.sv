`timescale 1ns/1ps

module lpu_mem_command_adapter_tb;
  localparam integer MEM_SLICES = 4;
  localparam integer COLUMNS = 16;
  localparam integer SUPERLANES = 4;
  localparam integer STREAMS = 32;
  localparam integer SEGMENT_BITS = 64;
  localparam [2:0] MEM_READ = 3'd0;
  localparam [2:0] MEM_WRITE = 3'd1;
  localparam [2:0] MEM_READ_WRITE = 3'd2;
  localparam [63:0] TEST_PATTERN = 64'h0807060504030201;

  logic clk_i;
  logic rst_ni;
  logic [MEM_SLICES*2-1:0] vmodel_issue_valid_i;
  logic [MEM_SLICES*2*47-1:0] vmodel_issue_instruction_i;
  logic [MEM_SLICES*2-1:0] native_issue_valid_o;
  logic [MEM_SLICES*2*32-1:0] native_issue_o;
  logic [MEM_SLICES*2-1:0] command_fault_o;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_i;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_i;
  logic [2*SUPERLANES*STREAMS-1:0] boundary_valid_o;
  logic [2*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] boundary_data_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS-1:0] state_valid_o;
  logic [2*COLUMNS*SUPERLANES*STREAMS*SEGMENT_BITS-1:0] state_data_o;
  logic [511:0] mem_boundary_consume_o;
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

  lpu_mem_srf_command_integration #(
    .MEM_SLICES(MEM_SLICES),
    .MEM_SLICES_PER_GROUP(4),
    .MEM_DEPTH_ROWS(16)
  ) dut (.*);

  always #5 clk_i = ~clk_i;

  function automatic [46:0] make_vmodel_cmd(
    input [2:0] opcode,
    input [5:0] stream,
    input [15:0] address,
    input [5:0] map_or_write_stream,
    input [15:0] write_address
  );
    begin
      make_vmodel_cmd = '0;
      make_vmodel_cmd[2:0] = opcode;
      make_vmodel_cmd[8:3] = stream;
      make_vmodel_cmd[14:9] = map_or_write_stream;
      make_vmodel_cmd[30:15] = address;
      make_vmodel_cmd[46:31] = write_address;
    end
  endfunction

  function automatic integer boundary_index(input integer direction,
                                             input integer superlane,
                                             input integer stream);
    boundary_index = (direction*SUPERLANES + superlane)*STREAMS + stream;
  endfunction

  function automatic integer state_index(input integer direction,
                                          input integer column,
                                          input integer superlane,
                                          input integer stream);
    state_index = ((direction*COLUMNS + column)*SUPERLANES + superlane)*
                  STREAMS + stream;
  endfunction

  function automatic integer consume_index(input integer direction,
                                            input integer column,
                                            input integer superlane,
                                            input integer stream);
    consume_index = ((((direction*COLUMNS + column)*SUPERLANES + superlane)*2)*
                     STREAMS) + stream;
  endfunction

  task automatic clear_drives;
    begin
      vmodel_issue_valid_i = '0;
      vmodel_issue_instruction_i = '0;
      boundary_valid_i = '0;
      boundary_data_i = '0;
    end
  endtask

  task automatic drive_east_segment(input integer superlane, input integer stream);
    integer index;
    begin
      index = boundary_index(0, superlane, stream);
      boundary_valid_i[index] = 1'b1;
      boundary_data_i[index*SEGMENT_BITS +: SEGMENT_BITS] = TEST_PATTERN;
    end
  endtask

  task automatic step_cycle;
    begin
      @(posedge clk_i);
      #1;
    end
  endtask

  task automatic reset_srf_control;
    begin
      @(negedge clk_i);
      clear_drives();
      rst_ni = 1'b0;
      @(negedge clk_i);
      rst_ni = 1'b1;
      step_cycle();
    end
  endtask

  task automatic check_native_command(input [2:0] opcode,
                                      input [5:0] stream,
                                      input [14:0] row);
    begin
      if (!native_issue_valid_o[0] || command_fault_o[0] ||
          native_issue_o[2:0] !== opcode ||
          native_issue_o[8:3] !== stream ||
          native_issue_o[29:15] !== row ||
          native_issue_o[14:9] !== '0 || native_issue_o[31] !== 1'b0) begin
        $display("ERROR adapter native command mismatch");
        errors = errors + 1;
      end
    end
  endtask

  task automatic perform_east_write;
    integer consume_bit;
    begin
      @(negedge clk_i);
      clear_drives();
      drive_east_segment(0, 3);
      step_cycle();

      for (tile = 0; tile < 4; tile = tile + 1) begin
        @(negedge clk_i);
        clear_drives();
        if (tile < 3)
          drive_east_segment(tile + 1, 3);
        if (tile == 0) begin
          vmodel_issue_valid_i[0] = 1'b1;
          vmodel_issue_instruction_i[0 +: 47] =
            make_vmodel_cmd(MEM_WRITE, {1'b0, 5'd3}, 16'd5, 6'd21, 16'd0);
          #1;
          check_native_command(MEM_WRITE, {1'b0, 5'd3}, 15'd5);
        end
        #1;
        consume_bit = consume_index(0, 0, tile, 3);
        if (!dut.u_phase2a_data_plane.srf_consume[consume_bit]) begin
          $display("ERROR adapter Write did not reach SRF consumer slot0 tile=%0d", tile);
          errors = errors + 1;
        end
        step_cycle();
        if (state_valid_o[state_index(0, 1, tile, 3)]) begin
          $display("ERROR consumed segment propagated tile=%0d", tile);
          errors = errors + 1;
        end
      end
    end
  endtask

  task automatic perform_east_read;
    begin
      for (tile = 0; tile < 4; tile = tile + 1) begin
        @(negedge clk_i);
        clear_drives();
        if (tile == 0) begin
          vmodel_issue_valid_i[0] = 1'b1;
          vmodel_issue_instruction_i[0 +: 47] =
            make_vmodel_cmd(MEM_READ, {1'b0, 5'd9}, 16'd5, 6'd0, 16'd0);
          #1;
          check_native_command(MEM_READ, {1'b0, 5'd9}, 15'd5);
        end
        step_cycle();
        if (!mem_producer_valid_o[tile] ||
            mem_producer_data_o[tile*SEGMENT_BITS +: SEGMENT_BITS] !== TEST_PATTERN ||
            mem_producer_direction_o[tile] !== 1'b0 ||
            mem_producer_stream_o[tile*5 +: 5] !== 5'd9 ||
            mem_producer_boundary_o[tile*4 +: 4] !== 4'd1) begin
          $display("ERROR adapter Read producer tile=%0d", tile);
          errors = errors + 1;
        end
        if (tile > 0 && (!state_valid_o[state_index(0, 1, tile-1, 9)] ||
                         state_data_o[state_index(0, 1, tile-1, 9)*SEGMENT_BITS +:
                                      SEGMENT_BITS] !== TEST_PATTERN)) begin
          $display("ERROR SRF producer commit tile=%0d", tile-1);
          errors = errors + 1;
        end
      end
      @(negedge clk_i);
      clear_drives();
      step_cycle();
      if (!state_valid_o[state_index(0, 1, 3, 9)]) begin
        $display("ERROR final SRF producer commit");
        errors = errors + 1;
      end
    end
  endtask

  initial begin
    clk_i = 1'b0;
    rst_ni = 1'b1;
    errors = 0;
    clear_drives();

    // Pure combinational direction and address mapping checks.  Deasserting
    // before the next active edge ensures these probes do not execute MEM.
    @(negedge clk_i);
    vmodel_issue_valid_i[0] = 1'b1;
    vmodel_issue_instruction_i[0 +: 47] =
      make_vmodel_cmd(MEM_READ, {1'b0, 5'd17}, 16'h1234, 6'd0, 16'd0);
    #1;
    check_native_command(MEM_READ, {1'b0, 5'd17}, 15'h1234);
    clear_drives();
    #1;

    vmodel_issue_valid_i[0] = 1'b1;
    vmodel_issue_instruction_i[0 +: 47] =
      make_vmodel_cmd(MEM_WRITE, {1'b1, 5'd6}, 16'd5, 6'd0, 16'd0);
    #1;
    check_native_command(MEM_WRITE, {1'b1, 5'd6}, 15'd5);
    $display("MEM_CMD_STREAM_DIRECTION PASS");
    $display("MEM_CMD_ADDRESS PASS");
    clear_drives();

    vmodel_issue_valid_i[0] = 1'b1;
    vmodel_issue_instruction_i[0 +: 47] =
      make_vmodel_cmd(MEM_READ_WRITE, {1'b0, 5'd1}, 16'd5, 6'd2, 16'd7);
    #1;
    if (native_issue_valid_o[0] || !command_fault_o[0]) begin
      $display("ERROR ReadWrite did not fail closed");
      errors = errors + 1;
    end
    clear_drives();

    vmodel_issue_valid_i[0] = 1'b1;
    vmodel_issue_instruction_i[0 +: 47] =
      make_vmodel_cmd(MEM_READ, {1'b0, 5'd1}, 16'h8000, 6'd0, 16'd0);
    #1;
    if (native_issue_valid_o[0] || !command_fault_o[0]) begin
      $display("ERROR high address was silently truncated");
      errors = errors + 1;
    end
    $display("MEM_CMD_ILLEGAL PASS");
    clear_drives();

    reset_srf_control();
    perform_east_write();
    if (command_fault_o !== '0 || mem_bank_fault_valid_o !== '0 ||
        mem_fault_valid_o || srf_collision_o !== '0 ||
        srf_invalid_consume_o !== '0) begin
      $display("ERROR legal Write status");
      errors = errors + 1;
    end
    $display("MEM_CMD_WRITE PASS");

    reset_srf_control();
    perform_east_read();
    if (command_fault_o !== '0 || mem_bank_fault_valid_o !== '0 ||
        mem_fault_valid_o || srf_collision_o !== '0 ||
        srf_invalid_consume_o !== '0) begin
      $display("ERROR legal Read status");
      errors = errors + 1;
    end
    $display("MEM_CMD_READ PASS");
    $display("MEM_CMD_CYCLE_CONTRACT PASS");

    if (errors == 0)
      $display("VMODEL_MEM_COMMAND_ADAPTER TEST_PASS");
    else
      $display("VMODEL_MEM_COMMAND_ADAPTER TEST_FAIL errors=%0d", errors);
    $finish;
  end
endmodule
