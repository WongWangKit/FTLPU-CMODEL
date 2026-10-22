// Two adjacent Tiles share one fully pipelined LUT set. Each function owns
// one independent single-read SRAM per Lane (3 x 8 SRAMs by default; 4 x 8 with Log). EXP
// stores one UQ1.25 base value packed as {k[12:0], b[12:0]} (26-bit
// row). RECIP and RSQRT each store two UQ1.15 coefficients (32-bit row).
// The external programming/response
// containers stay 16 bits for interface stability.
// The instruction wave guarantees that both Tiles do not request the same
// function/Lane SRAM in one cycle; a collision is reported if violated.
module lpu_vxm_tile_pair_lut #(
  parameter integer FUNCTION_COUNT = 3,
  parameter integer LANES = 8,
  parameter integer ENTRY_COUNT = 64,
  parameter integer TILE_COUNT = 2,
  parameter integer STAGE_WIDTH = 4,
  parameter integer FUNCTION_WIDTH =
    FUNCTION_COUNT <= 1 ? 1 : $clog2(FUNCTION_COUNT),
  parameter integer ADDRESS_WIDTH =
    ENTRY_COUNT <= 1 ? 1 : $clog2(ENTRY_COUNT)
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic                      config_valid_i,
  input  logic [FUNCTION_WIDTH-1:0] config_function_i,
  input  logic [15:0]               config_input_min_i,
  input  logic [15:0]               config_segment_width_i,
  input  logic                      write_valid_i,
  input  logic [FUNCTION_WIDTH-1:0] write_function_i,
  input  logic [ADDRESS_WIDTH-1:0]  write_address_i,
  input  logic [15:0]               write_k_i,
  input  logic [15:0]               write_b_i,
  input  logic [49:0]              write_log_i,

  input  logic [TILE_COUNT*FUNCTION_COUNT*LANES-1:0]
    request_valid_i,
  input  logic [TILE_COUNT*FUNCTION_COUNT*LANES*ADDRESS_WIDTH-1:0]
    request_address_i,
  input  logic [TILE_COUNT*FUNCTION_COUNT*LANES*STAGE_WIDTH-1:0]
    request_stage_i,
  output logic [TILE_COUNT*FUNCTION_COUNT*LANES-1:0]
    response_valid_o,
  output logic [TILE_COUNT*FUNCTION_COUNT*LANES*STAGE_WIDTH-1:0]
    response_stage_o,
  output logic [TILE_COUNT*FUNCTION_COUNT*LANES*16-1:0]
    response_k_o,
  output logic [TILE_COUNT*FUNCTION_COUNT*LANES*16-1:0]
    response_b_o,
  output logic [TILE_COUNT*FUNCTION_COUNT*LANES*50-1:0]
    response_log_o,

  output logic [FUNCTION_COUNT-1:0] configured_o,
  output logic [FUNCTION_COUNT*16-1:0] input_min_o,
  output logic [FUNCTION_COUNT*16-1:0] segment_width_o,
  output logic collision_o,
  output logic fault_o
);
  localparam integer SRAM_COUNT = FUNCTION_COUNT*LANES;
  localparam integer TILE_REQUESTS = FUNCTION_COUNT*LANES;

  logic [SRAM_COUNT-1:0] sram_request_valid;
  logic [SRAM_COUNT*ADDRESS_WIDTH-1:0] sram_request_address;
  logic [SRAM_COUNT-1:0] sram_response_valid;
  logic [SRAM_COUNT*16-1:0] sram_response_k;
  logic [SRAM_COUNT*16-1:0] sram_response_b;
  logic [SRAM_COUNT*50-1:0] sram_response_log;
  logic [SRAM_COUNT-1:0] sram_configured;
  logic [SRAM_COUNT*16-1:0] sram_input_min;
  logic [SRAM_COUNT*16-1:0] sram_segment_width;
  logic [SRAM_COUNT-1:0] sram_fault;
  logic [SRAM_COUNT-1:0] response_tile_q;
  logic [SRAM_COUNT*STAGE_WIDTH-1:0] response_stage_q;
  logic [SRAM_COUNT-1:0] response_tag_valid_q;
  logic request_collision;

  always_comb begin
    sram_request_valid = '0;
    sram_request_address = '0;
    request_collision = 1'b0;
    for (integer function_id = 0;
         function_id < FUNCTION_COUNT; function_id++) begin
      configured_o[function_id] =
        sram_configured[function_id*LANES];
      input_min_o[function_id*16 +: 16] =
        sram_input_min[(function_id*LANES)*16 +: 16];
      segment_width_o[function_id*16 +: 16] =
        sram_segment_width[(function_id*LANES)*16 +: 16];
      for (integer lane = 0; lane < LANES; lane++) begin
        integer sram_index;
        integer tile0_index;
        integer tile1_index;
        sram_index = function_id*LANES + lane;
        tile0_index = sram_index;
        tile1_index = TILE_REQUESTS + sram_index;
        sram_request_valid[sram_index] =
          request_valid_i[tile0_index] || request_valid_i[tile1_index];
        if (request_valid_i[tile0_index])
          sram_request_address[
            sram_index*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
            request_address_i[
              tile0_index*ADDRESS_WIDTH +: ADDRESS_WIDTH];
        else
          sram_request_address[
            sram_index*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
            request_address_i[
              tile1_index*ADDRESS_WIDTH +: ADDRESS_WIDTH];
        if (request_valid_i[tile0_index] && request_valid_i[tile1_index])
          request_collision = 1'b1;
      end
    end

    response_valid_o = '0;
    response_stage_o = '0;
    response_k_o = '0;
    response_b_o = '0;
    response_log_o = '0;
    for (integer sram_index = 0;
         sram_index < SRAM_COUNT; sram_index++) begin
      integer response_index;
      response_index = (response_tile_q[sram_index] ?
        TILE_REQUESTS : 0) + sram_index;
      if (sram_response_valid[sram_index] &&
          response_tag_valid_q[sram_index]) begin
        response_valid_o[response_index] = 1'b1;
        response_stage_o[
          response_index*STAGE_WIDTH +: STAGE_WIDTH] =
          response_stage_q[
            sram_index*STAGE_WIDTH +: STAGE_WIDTH];
        response_k_o[response_index*16 +: 16] =
          sram_response_k[sram_index*16 +: 16];
        response_b_o[response_index*16 +: 16] =
          sram_response_b[sram_index*16 +: 16];
        response_log_o[response_index*50 +: 50] =
          sram_response_log[sram_index*50 +: 50];
      end
    end
    collision_o = request_collision;
  end

  genvar function_index;
  genvar lane_index;
  generate
    for (function_index = 0; function_index < FUNCTION_COUNT;
         function_index++) begin : g_function
      for (lane_index = 0; lane_index < LANES;
           lane_index++) begin : g_lane_sram
        localparam integer SRAM_INDEX = function_index*LANES + lane_index;
        localparam integer COEFFICIENT_WIDTH =
          function_index == 0 ? 13 : (function_index == 3 ? 25 : 16);
        logic [COEFFICIENT_WIDTH-1:0] coefficient_k;
        logic [COEFFICIENT_WIDTH-1:0] coefficient_b;

        if (function_index == 3) begin : g_log_row
          assign sram_response_k[SRAM_INDEX*16 +: 16] = '0;
          assign sram_response_b[SRAM_INDEX*16 +: 16] = '0;
          assign sram_response_log[SRAM_INDEX*50 +: 50] =
            {coefficient_k, coefficient_b};
        end else begin : g_legacy_row
          assign sram_response_k[SRAM_INDEX*16 +: 16] =
            {{(16-COEFFICIENT_WIDTH){1'b0}}, coefficient_k};
          assign sram_response_b[SRAM_INDEX*16 +: 16] =
            {{(16-COEFFICIENT_WIDTH){1'b0}}, coefficient_b};
          assign sram_response_log[SRAM_INDEX*50 +: 50] = '0;
        end

        lpu_vxm_lut_sram #(
          .ENTRY_COUNT(ENTRY_COUNT),
          .COEFFICIENT_WIDTH(COEFFICIENT_WIDTH),
          .ADDRESS_WIDTH(ADDRESS_WIDTH)
        ) u_sram (
          .clk_i,
          .rst_ni,
          .config_valid_i(config_valid_i &&
            (config_function_i == function_index)),
          .config_input_min_i,
          .config_segment_width_i,
          .write_valid_i(write_valid_i &&
            (write_function_i == function_index) &&
            ((function_index == 0 &&
              !(|write_k_i[15:13]) && !(|write_b_i[15:13])) ||
             (function_index > 0))),
          .write_address_i,
          .write_k_i(function_index == 3 ?
            COEFFICIENT_WIDTH'(write_log_i[49:25]) : COEFFICIENT_WIDTH'(write_k_i)),
          .write_b_i(function_index == 3 ?
            COEFFICIENT_WIDTH'(write_log_i[24:0]) : COEFFICIENT_WIDTH'(write_b_i)),
          .read_valid_i(sram_request_valid[SRAM_INDEX]),
          .read_address_i(sram_request_address[
            SRAM_INDEX*ADDRESS_WIDTH +: ADDRESS_WIDTH]),
          .read_valid_o(sram_response_valid[SRAM_INDEX]),
          .read_k_o(coefficient_k),
          .read_b_o(coefficient_b),
          .configured_o(sram_configured[SRAM_INDEX]),
          .input_min_o(sram_input_min[SRAM_INDEX*16 +: 16]),
          .segment_width_o(
            sram_segment_width[SRAM_INDEX*16 +: 16]),
          .fault_o(sram_fault[SRAM_INDEX])
        );
      end
    end
  endgenerate

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      response_tile_q <= '0;
      response_stage_q <= '0;
      response_tag_valid_q <= '0;
      fault_o <= 1'b0;
    end else begin
      response_tag_valid_q <= sram_request_valid;
      fault_o <= request_collision || (|sram_fault) ||
        (config_valid_i && (config_function_i >= FUNCTION_COUNT)) ||
        (write_valid_i && (write_function_i >= FUNCTION_COUNT)) ||
        (write_valid_i && (write_function_i == 0) &&
         ((|write_k_i[15:13]) || (|write_b_i[15:13])));
      for (integer sram_index = 0;
           sram_index < SRAM_COUNT; sram_index++) begin
        integer tile0_index;
        integer tile1_index;
        tile0_index = sram_index;
        tile1_index = TILE_REQUESTS + sram_index;
        if (sram_request_valid[sram_index]) begin
          response_tile_q[sram_index] <=
            !request_valid_i[tile0_index] && request_valid_i[tile1_index];
          response_stage_q[
            sram_index*STAGE_WIDTH +: STAGE_WIDTH] <=
            request_valid_i[tile0_index] ?
              request_stage_i[
                tile0_index*STAGE_WIDTH +: STAGE_WIDTH] :
              request_stage_i[
                tile1_index*STAGE_WIDTH +: STAGE_WIDTH];
        end
      end
    end
  end

  initial begin
    if (((FUNCTION_COUNT != 3) && (FUNCTION_COUNT != 4)) ||
        ((LANES != 1) && (LANES != lpu_pkg::LANES_PER_TILE)) ||
        (TILE_COUNT != 2))
      $error("VXM LUT sharing requires two Tiles, three/four functions, and 1 or 8 Lanes");
  end
endmodule
