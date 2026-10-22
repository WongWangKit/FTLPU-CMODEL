module lpu_mem_hemisphere #(
  parameter integer DEPTH_ROWS    = 65536,
  parameter integer ACTIVE_COLUMNS = 52,
  parameter bit USE_SRAM_MACRO = 1'b0
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,

  input  logic [51:0] issue_valid_i,
  input  logic [52*47-1:0] issue_instruction_i,

  // Boundary 0 is adjacent to VXM; boundary 13 is adjacent to SXM.
  input  logic [4*32-1:0] east_edge_valid_i,
  input  logic [4*32*64-1:0] east_edge_data_i,
  output logic [4*32-1:0] east_edge_valid_o,
  output logic [4*32*64-1:0] east_edge_data_o,
  input  logic [4*32-1:0] west_edge_valid_i,
  input  logic [4*32*64-1:0] west_edge_data_i,
  output logic [4*32-1:0] west_edge_valid_o,
  output logic [4*32*64-1:0] west_edge_data_o,

  input  logic host_write_valid_i,
  input  logic [5:0] host_column_i,
  input  logic [1:0] host_tile_i,
  input  logic [15:0] host_address_i,
  input  logic [63:0] host_write_data_i,
  output logic [63:0] host_read_data_o,

  output logic [52*4-1:0] mem_fault_o,
  output logic conflict_o
);
  localparam integer COLUMNS    = 52;
  localparam integer GROUPS     = 13;
  localparam integer BOUNDARIES = 14;
  localparam integer TILES      = 4;
  localparam integer STREAMS    = 32;
  localparam integer WORD_WIDTH = 64;

  logic east_valid_q [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic west_valid_q [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic [WORD_WIDTH-1:0] east_data_q [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic [WORD_WIDTH-1:0] west_data_q [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic east_valid_d [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic west_valid_d [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic [WORD_WIDTH-1:0] east_data_d [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic [WORD_WIDTH-1:0] west_data_d [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic east_consumed [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];
  logic west_consumed [0:BOUNDARIES-1][0:TILES-1][0:STREAMS-1];

  logic [COLUMNS*TILES*64-1:0] column_rx_valid;
  logic [COLUMNS*TILES*64*64-1:0] column_rx_data;
  logic [COLUMNS*TILES*64-1:0] column_rx_consume;
  logic [COLUMNS*TILES-1:0] column_tx_valid;
  logic [COLUMNS*TILES*6-1:0] column_tx_stream;
  logic [COLUMNS*TILES*64-1:0] column_tx_data;
  logic [COLUMNS*TILES-1:0] column_tx_last;
  logic [COLUMNS*TILES-1:0] column_fault;
  logic [COLUMNS*64-1:0] host_read_data;
  logic conflict_d;

  generate
    for (genvar column_gen = 0; column_gen < ACTIVE_COLUMNS; column_gen++) begin : gen_column
      localparam integer COLUMN_INDEX = column_gen;
      logic [TILES*64-1:0] rx_consume;
      logic [TILES-1:0] tx_valid;
      logic [TILES*6-1:0] tx_stream;
      logic [TILES*64-1:0] tx_data;
      logic [TILES-1:0] tx_last;
      logic [TILES-1:0] fault;
      logic [63:0] host_read;

      lpu_mem_column #(
        .DEPTH_ROWS(DEPTH_ROWS),
        .USE_SRAM_MACRO(USE_SRAM_MACRO)
      ) u_column (
        .clk_i,
        .rst_ni,
        .issue_valid_i(issue_valid_i[COLUMN_INDEX]),
        .issue_instruction_i(issue_instruction_i[COLUMN_INDEX*47 +: 47]),
        .tile_rx_valid_i(column_rx_valid[COLUMN_INDEX*TILES*64 +: TILES*64]),
        .tile_rx_data_i(column_rx_data[COLUMN_INDEX*TILES*64*64 +: TILES*64*64]),
        .tile_rx_consume_o(rx_consume),
        .tile_tx_valid_o(tx_valid),
        .tile_tx_stream_o(tx_stream),
        .tile_tx_data_o(tx_data),
        .tile_tx_last_o(tx_last),
        .tile_fault_o(fault),
        .host_write_valid_i(host_write_valid_i && (host_column_i == COLUMN_INDEX)),
        .host_tile_i,
        .host_address_i,
        .host_write_data_i,
        .host_read_data_o(host_read)
      );

      assign column_rx_consume[COLUMN_INDEX*TILES*64 +: TILES*64] = rx_consume;
      assign column_tx_valid[COLUMN_INDEX*TILES +: TILES] = tx_valid;
      assign column_tx_stream[COLUMN_INDEX*TILES*6 +: TILES*6] = tx_stream;
      assign column_tx_data[COLUMN_INDEX*TILES*64 +: TILES*64] = tx_data;
      assign column_tx_last[COLUMN_INDEX*TILES +: TILES] = tx_last;
      assign column_fault[COLUMN_INDEX*TILES +: TILES] = fault;
      assign host_read_data[COLUMN_INDEX*64 +: 64] = host_read;
    end
    for (genvar unused_column_gen = ACTIVE_COLUMNS;
         unused_column_gen < COLUMNS;
         unused_column_gen++) begin : gen_unused_column
      localparam integer COLUMN_INDEX = unused_column_gen;
      assign column_rx_consume[COLUMN_INDEX*TILES*64 +: TILES*64] = '0;
      assign column_tx_valid[COLUMN_INDEX*TILES +: TILES] = '0;
      assign column_tx_stream[COLUMN_INDEX*TILES*6 +: TILES*6] = '0;
      assign column_tx_data[COLUMN_INDEX*TILES*64 +: TILES*64] = '0;
      assign column_tx_last[COLUMN_INDEX*TILES +: TILES] = '0;
      assign column_fault[COLUMN_INDEX*TILES +: TILES] = '0;
      assign host_read_data[COLUMN_INDEX*64 +: 64] = '0;
    end
  endgenerate

  assign mem_fault_o = column_fault;

  always_comb begin
    column_rx_valid = '0;
    column_rx_data = '0;
    east_edge_valid_o = '0;
    east_edge_data_o = '0;
    west_edge_valid_o = '0;
    west_edge_data_o = '0;
    host_read_data_o = '0;
    conflict_d = 1'b0;

    if (host_column_i < ACTIVE_COLUMNS)
      host_read_data_o = host_read_data[host_column_i*64 +: 64];

    for (integer boundary = 0; boundary < BOUNDARIES; boundary++) begin
      for (integer tile = 0; tile < TILES; tile++) begin
        for (integer stream = 0; stream < STREAMS; stream++) begin
          east_valid_d[boundary][tile][stream] = 1'b0;
          west_valid_d[boundary][tile][stream] = 1'b0;
          east_data_d[boundary][tile][stream] = '0;
          west_data_d[boundary][tile][stream] = '0;
          east_consumed[boundary][tile][stream] = 1'b0;
          west_consumed[boundary][tile][stream] = 1'b0;
        end
      end
    end

    // Each group of four columns sees East at its west boundary and West at
    // its east boundary.
    for (integer column = 0; column < ACTIVE_COLUMNS; column++) begin
      for (integer tile = 0; tile < TILES; tile++) begin
        for (integer stream = 0; stream < STREAMS; stream++) begin
          column_rx_valid[(column*TILES+tile)*64+stream] =
            east_valid_q[column/4][tile][stream];
          column_rx_data[((column*TILES+tile)*64+stream)*64 +: 64] =
            east_data_q[column/4][tile][stream];
          column_rx_valid[(column*TILES+tile)*64+32+stream] =
            west_valid_q[column/4+1][tile][stream];
          column_rx_data[((column*TILES+tile)*64+32+stream)*64 +: 64] =
            west_data_q[column/4+1][tile][stream];

          if (column_rx_consume[(column*TILES+tile)*64+stream])
            east_consumed[column/4][tile][stream] = 1'b1;
          if (column_rx_consume[(column*TILES+tile)*64+32+stream])
            west_consumed[column/4+1][tile][stream] = 1'b1;
        end
      end
    end

    // Passive links move an unconsumed value exactly one SR hop per cycle.
    for (integer boundary = 0; boundary < BOUNDARIES-1; boundary++) begin
      for (integer tile = 0; tile < TILES; tile++) begin
        for (integer stream = 0; stream < STREAMS; stream++) begin
          if (east_valid_q[boundary][tile][stream] &&
              !east_consumed[boundary][tile][stream]) begin
            east_valid_d[boundary+1][tile][stream] = 1'b1;
            east_data_d[boundary+1][tile][stream] =
              east_data_q[boundary][tile][stream];
          end
        end
      end
    end
    for (integer boundary = 1; boundary < BOUNDARIES; boundary++) begin
      for (integer tile = 0; tile < TILES; tile++) begin
        for (integer stream = 0; stream < STREAMS; stream++) begin
          if (west_valid_q[boundary][tile][stream] &&
              !west_consumed[boundary][tile][stream]) begin
            west_valid_d[boundary-1][tile][stream] = 1'b1;
            west_data_d[boundary-1][tile][stream] =
              west_data_q[boundary][tile][stream];
          end
        end
      end
    end

    // Edge producers participate in the same next-state collision rules.
    for (integer tile = 0; tile < TILES; tile++) begin
      for (integer stream = 0; stream < STREAMS; stream++) begin
        if (east_edge_valid_i[tile*STREAMS+stream]) begin
          if (east_valid_d[0][tile][stream]) conflict_d = 1'b1;
          else begin
            east_valid_d[0][tile][stream] = 1'b1;
            east_data_d[0][tile][stream] =
              east_edge_data_i[(tile*STREAMS+stream)*64 +: 64];
          end
        end
        if (west_edge_valid_i[tile*STREAMS+stream]) begin
          if (west_valid_d[BOUNDARIES-1][tile][stream]) conflict_d = 1'b1;
          else begin
            west_valid_d[BOUNDARIES-1][tile][stream] = 1'b1;
            west_data_d[BOUNDARIES-1][tile][stream] =
              west_edge_data_i[(tile*STREAMS+stream)*64 +: 64];
          end
        end
      end
    end

    // Match TspSliceSystem's current TileArray compatibility mapping: MEM
    // Reads inject at the same input-side boundary used by MEM Writes. The
    // standalone C-model MemArrayModel also offers downstream placement, but
    // that mode is not used by the full-system workloads yet.
    for (integer column = 0; column < ACTIVE_COLUMNS; column++) begin
      for (integer tile = 0; tile < TILES; tile++) begin
        if (column_tx_valid[column*TILES+tile]) begin
          if (column_tx_stream[(column*TILES+tile)*6 +: 6] < 32) begin
            if (east_valid_d[column/4][tile]
                            [column_tx_stream[(column*TILES+tile)*6 +: 6]])
              conflict_d = 1'b1;
            else begin
              east_valid_d[column/4][tile]
                          [column_tx_stream[(column*TILES+tile)*6 +: 6]] = 1'b1;
              east_data_d[column/4][tile]
                         [column_tx_stream[(column*TILES+tile)*6 +: 6]] =
                column_tx_data[(column*TILES+tile)*64 +: 64];
            end
          end else begin
            if (west_valid_d[column/4+1][tile]
                            [column_tx_stream[(column*TILES+tile)*6 +: 6]-32])
              conflict_d = 1'b1;
            else begin
              west_valid_d[column/4+1][tile]
                          [column_tx_stream[(column*TILES+tile)*6 +: 6]-32] = 1'b1;
              west_data_d[column/4+1][tile]
                         [column_tx_stream[(column*TILES+tile)*6 +: 6]-32] =
                column_tx_data[(column*TILES+tile)*64 +: 64];
            end
          end
        end
      end
    end

    for (integer tile = 0; tile < TILES; tile++) begin
      for (integer stream = 0; stream < STREAMS; stream++) begin
        east_edge_valid_o[tile*STREAMS+stream] =
          east_valid_q[BOUNDARIES-1][tile][stream];
        east_edge_data_o[(tile*STREAMS+stream)*64 +: 64] =
          east_data_q[BOUNDARIES-1][tile][stream];
        west_edge_valid_o[tile*STREAMS+stream] =
          west_valid_q[0][tile][stream];
        west_edge_data_o[(tile*STREAMS+stream)*64 +: 64] =
          west_data_q[0][tile][stream];
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      conflict_o <= 1'b0;
      for (integer boundary = 0; boundary < BOUNDARIES; boundary++) begin
        for (integer tile = 0; tile < TILES; tile++) begin
          for (integer stream = 0; stream < STREAMS; stream++) begin
            east_valid_q[boundary][tile][stream] <= 1'b0;
            west_valid_q[boundary][tile][stream] <= 1'b0;
            east_data_q[boundary][tile][stream] <= '0;
            west_data_q[boundary][tile][stream] <= '0;
          end
        end
      end
    end else if (!run_i) begin
      conflict_o <= 1'b0;
      for (integer boundary = 0; boundary < BOUNDARIES; boundary++) begin
        for (integer tile = 0; tile < TILES; tile++) begin
          for (integer stream = 0; stream < STREAMS; stream++) begin
            east_valid_q[boundary][tile][stream] <= 1'b0;
            west_valid_q[boundary][tile][stream] <= 1'b0;
            east_data_q[boundary][tile][stream] <= '0;
            west_data_q[boundary][tile][stream] <= '0;
          end
        end
      end
    end else begin
      if (conflict_d) conflict_o <= 1'b1;
      for (integer boundary = 0; boundary < BOUNDARIES; boundary++) begin
        for (integer tile = 0; tile < TILES; tile++) begin
          for (integer stream = 0; stream < STREAMS; stream++) begin
            east_valid_q[boundary][tile][stream] <= east_valid_d[boundary][tile][stream];
            west_valid_q[boundary][tile][stream] <= west_valid_d[boundary][tile][stream];
            east_data_q[boundary][tile][stream] <= east_data_d[boundary][tile][stream];
            west_data_q[boundary][tile][stream] <= west_data_d[boundary][tile][stream];
          end
        end
      end
    end
  end
endmodule
