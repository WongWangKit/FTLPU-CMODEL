`timescale 1ns/1ps

module lpu_vxm_tile_pair_lut_tb;
  localparam integer FUNCTIONS = 3;
  localparam integer LANES = 8;
  localparam integer TILES = 2;
  localparam integer ADDRESS_WIDTH = 6;
  localparam integer STAGE_WIDTH = 4;
  localparam integer TILE_REQUESTS = FUNCTIONS*LANES;

  logic clk;
  logic rst_n;
  logic config_valid;
  logic [1:0] config_function;
  logic [15:0] config_input_min;
  logic [15:0] config_segment_width;
  logic write_valid;
  logic [1:0] write_function;
  logic [ADDRESS_WIDTH-1:0] write_address;
  logic [15:0] write_k;
  logic [15:0] write_b;
  logic [TILES*TILE_REQUESTS-1:0] request_valid;
  logic [TILES*TILE_REQUESTS*ADDRESS_WIDTH-1:0] request_address;
  logic [TILES*TILE_REQUESTS*STAGE_WIDTH-1:0] request_stage;
  wire [TILES*TILE_REQUESTS-1:0] response_valid;
  wire [TILES*TILE_REQUESTS*STAGE_WIDTH-1:0] response_stage;
  wire [TILES*TILE_REQUESTS*16-1:0] response_k;
  wire [TILES*TILE_REQUESTS*16-1:0] response_b;
  wire [FUNCTIONS-1:0] configured;
  wire [FUNCTIONS*16-1:0] input_min;
  wire [FUNCTIONS*16-1:0] segment_width;
  wire collision;
  wire fault;

  always #5 clk = ~clk;

  lpu_vxm_tile_pair_lut dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .config_valid_i(config_valid),
    .config_function_i(config_function),
    .config_input_min_i(config_input_min),
    .config_segment_width_i(config_segment_width),
    .write_valid_i(write_valid),
    .write_function_i(write_function),
    .write_address_i(write_address),
    .write_k_i(write_k),
    .write_b_i(write_b),
    .request_valid_i(request_valid),
    .request_address_i(request_address),
    .request_stage_i(request_stage),
    .response_valid_o(response_valid),
    .response_stage_o(response_stage),
    .response_k_o(response_k),
    .response_b_o(response_b),
    .configured_o(configured),
    .input_min_o(input_min),
    .segment_width_o(segment_width),
    .collision_o(collision),
    .fault_o(fault)
  );

  function automatic logic [15:0] expected_k(
    input integer function_id,
    input integer address
  );
    // EXP transports the high UQ1.25 half in 13 bits; RECIP and RSQRT use
    // UQ1.15 coefficient containers.
    expected_k = function_id == 0 ? 16'h1000 + address :
      (function_id == 1 ? 16'h8000 + address :
       16'h1000 + function_id*16'h0100 + address);
  endfunction

  function automatic logic [15:0] expected_b(
    input integer function_id,
    input integer address
  );
    expected_b = function_id == 0 ? 16'h0100 + address :
      (function_id == 1 ? 16'h4000 + address :
       16'h8000 + function_id*16'h0100 + address);
  endfunction

  function automatic integer request_index(
    input integer tile,
    input integer function_id,
    input integer lane
  );
    request_index = tile*TILE_REQUESTS + function_id*LANES + lane;
  endfunction

  task automatic configure(input integer function_id);
    begin
      @(negedge clk);
      config_valid = 1'b1;
      config_function = function_id[1:0];
      config_input_min = 16'hbc00 + function_id;
      config_segment_width = 16'h3800;
      @(posedge clk);
      @(negedge clk);
      config_valid = 1'b0;
    end
  endtask

  task automatic write_row(
    input integer function_id,
    input integer address
  );
    begin
      @(negedge clk);
      write_valid = 1'b1;
      write_function = function_id[1:0];
      write_address = address[ADDRESS_WIDTH-1:0];
      write_k = expected_k(function_id, address);
      write_b = expected_b(function_id, address);
      @(posedge clk);
      @(negedge clk);
      write_valid = 1'b0;
    end
  endtask

  task automatic drive_lane_group(
    input integer tile,
    input integer function_id,
    input integer stage_base
  );
    begin
      request_valid = '0;
      request_address = '0;
      request_stage = '0;
      for (integer lane = 0; lane < LANES; lane++) begin
        integer index;
        index = request_index(tile, function_id, lane);
        request_valid[index] = 1'b1;
        request_address[index*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
          lane[ADDRESS_WIDTH-1:0];
        request_stage[index*STAGE_WIDTH +: STAGE_WIDTH] =
          stage_base + lane;
      end
    end
  endtask

  task automatic check_lane_group(
    input integer tile,
    input integer function_id,
    input integer stage_base
  );
    begin
      for (integer lane = 0; lane < LANES; lane++) begin
        integer index;
        index = request_index(tile, function_id, lane);
        if (!response_valid[index])
          $fatal(1, "tile %0d function %0d lane %0d response missing",
            tile, function_id, lane);
        if (response_stage[index*STAGE_WIDTH +: STAGE_WIDTH] !==
              stage_base + lane ||
            response_k[index*16 +: 16] !== expected_k(function_id, lane) ||
            response_b[index*16 +: 16] !== expected_b(function_id, lane))
          $fatal(1, "tile %0d function %0d lane %0d response mismatch",
            tile, function_id, lane);
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    config_valid = 1'b0;
    config_function = '0;
    config_input_min = '0;
    config_segment_width = '0;
    write_valid = 1'b0;
    write_function = '0;
    write_address = '0;
    write_k = '0;
    write_b = '0;
    request_valid = '0;
    request_address = '0;
    request_stage = '0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    for (integer function_id = 0;
         function_id < FUNCTIONS; function_id++) begin
      configure(function_id);
      for (integer address = 0; address < LANES; address++)
        write_row(function_id, address);
    end
    if (configured !== 3'b111)
      $fatal(1, "all three function banks must be configured");

    // All eight Lane SRAMs of one function read in parallel. On the next
    // cycle, ownership changes to the adjacent Tile without a bubble.
    @(negedge clk);
    drive_lane_group(0, 0, 0);
    @(posedge clk);
    @(negedge clk);
    check_lane_group(0, 0, 0);
    drive_lane_group(1, 0, 8);
    @(posedge clk);
    @(negedge clk);
    check_lane_group(1, 0, 8);
    request_valid = '0;

    // Different function SRAMs may be used by both Tiles in the same cycle.
    @(negedge clk);
    drive_lane_group(0, 0, 0);
    for (integer lane = 0; lane < LANES; lane++) begin
      integer index;
      index = request_index(1, 1, lane);
      request_valid[index] = 1'b1;
      request_address[index*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
        lane[ADDRESS_WIDTH-1:0];
      request_stage[index*STAGE_WIDTH +: STAGE_WIDTH] =
        8 + lane;
    end
    #1;
    if (collision)
      $fatal(1, "different function/Lane SRAMs falsely collided");
    @(posedge clk);
    @(negedge clk);
    check_lane_group(0, 0, 0);
    check_lane_group(1, 1, 8);
    if (fault)
      $fatal(1, "legal fixed-phase traffic raised a fault");
    request_valid = '0;

    // EXP stores two packed 13-bit halves. RECIP uses the complete 16-bit
    // containers as UQ1.15, so only EXP has unused upper container bits.
    @(negedge clk);
    write_valid = 1'b1;
    write_function = 2'd0;
    write_address = 6'd9;
    write_k = 16'h2000;
    write_b = 16'h0400;
    @(posedge clk);
    @(negedge clk);
    if (!fault)
      $fatal(1, "EXP packed UQ1.25 half wider than 13 bits did not fault");
    write_valid = 1'b0;
    write_k = '0;
    write_b = '0;
    @(posedge clk);
    @(negedge clk);
    write_valid = 1'b1;
    write_function = 2'd1;
    write_address = 6'd9;
    write_k = 16'h8000;
    write_b = 16'h6000;
    @(posedge clk);
    @(negedge clk);
    if (fault)
      $fatal(1, "legal full-width RECIP UQ1.15 coefficients faulted");
    write_valid = 1'b0;
    write_k = '0;
    write_b = '0;

    // The same function/Lane cannot be owned by both adjacent Tiles.
    @(negedge clk);
    request_valid[request_index(0, 2, 3)] = 1'b1;
    request_valid[request_index(1, 2, 3)] = 1'b1;
    request_address[request_index(0, 2, 3)*ADDRESS_WIDTH +:
      ADDRESS_WIDTH] = 6'd3;
    request_address[request_index(1, 2, 3)*ADDRESS_WIDTH +:
      ADDRESS_WIDTH] = 6'd3;
    #1;
    if (!collision)
      $fatal(1, "same function/Lane collision was not reported");
    @(posedge clk);
    @(negedge clk);
    if (!fault)
      $fatal(1, "same function/Lane collision did not raise fault");

    $display("LPU_VXM_TILE_PAIR_LUT_TB_PASS");
    $finish;
  end
endmodule
