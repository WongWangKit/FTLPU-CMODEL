`timescale 1ns/1ps

module lpu_vxm_slice_fp32_special_tb;
  import lpu_pkg::*;

  localparam integer TILES = 4;
  localparam integer HEMISPHERES = 2;
  localparam integer CELLS = TILES*HEMISPHERES;
  localparam integer STREAMS = STREAMS_PER_DIRECTION;
  localparam integer LANES = LANES_PER_TILE;
  localparam integer WORD_WIDTH = LANES*8;
  localparam integer LOCAL_WIDTH = VXM_LOCAL_MAX_INSTRUCTION_WIDTH;
  localparam logic [STREAMS-1:0] ALL_INPUTS = 32'hffffffff;
  localparam logic [STREAMS-1:0] ALL_CHAIN2_OUTPUTS = 32'h0000ffff;

  logic clk;
  logic rst_n;
  logic run;
  logic [7:0] local_issue_valid;
  logic [8*LOCAL_WIDTH-1:0] local_issue_instruction;
  logic global_issue_valid;
  logic [VXM_GLOBAL_CONFIG_WIDTH-1:0] global_issue_instruction;
  vxm_global_config_t global_fields;
  logic lut_config_valid;
  logic [1:0] lut_config_bank;
  logic [15:0] lut_config_min;
  logic [15:0] lut_config_width;
  logic lut_write_valid;
  logic [1:0] lut_write_bank;
  logic [5:0] lut_write_address;
  logic [15:0] lut_write_k;
  logic [15:0] lut_write_b;
  logic [CELLS*STREAMS-1:0] west_valid;
  logic [CELLS*STREAMS*WORD_WIDTH-1:0] west_data;
  logic [CELLS*STREAMS-1:0] external_east_valid;
  logic [CELLS*STREAMS*WORD_WIDTH-1:0] external_east_data;
  wire [CELLS*STREAMS-1:0] east_valid;
  wire [CELLS*STREAMS*WORD_WIDTH-1:0] east_data;
  wire fault;
  wire conflict;

  logic active_direction;
  logic [TILES-1:0] row_data_valid;
  logic [TILES-1:0] row_data_high;
  logic [2*TILES-1:0] phase_seen;
  logic [TILES-1:0] done_seen;
  logic [CELLS*STREAMS-1:0] output_low_seen;
  logic [CELLS*STREAMS-1:0] result_seen;
  integer cycle_count;
  integer row_load_cycle [0:TILES-1];
  integer low_cycle [0:TILES-1];
  integer high_cycle [0:TILES-1];

  always #5 clk = ~clk;

  lpu_vxm_slice dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .run_i(run),
    .local_issue_valid_i(local_issue_valid),
    .local_issue_instruction_i(local_issue_instruction),
    .global_issue_valid_i(global_issue_valid),
    .global_issue_instruction_i(global_issue_instruction),
    .lut_config_valid_i(lut_config_valid),
    .lut_config_bank_i(lut_config_bank),
    .lut_config_input_min_i(lut_config_min),
    .lut_config_segment_width_i(lut_config_width),
    .lut_write_valid_i(lut_write_valid),
    .lut_write_bank_i(lut_write_bank),
    .lut_write_address_i(lut_write_address),
    .lut_write_k_i(lut_write_k),
    .lut_write_b_i(lut_write_b),
    .west_from_mem_valid_i(west_valid),
    .west_from_mem_data_i(west_data),
    .external_east_valid_i(external_east_valid),
    .external_east_data_i(external_east_data),
    .east_to_mem_valid_o(east_valid),
    .east_to_mem_data_o(east_data),
    .fault_o(fault),
    .conflict_o(conflict)
  );

  function automatic integer source_cell(input integer row);
    source_cell = active_direction == VXM_FLOW_LEFT_TO_RIGHT ?
      row : TILES+row;
  endfunction

  function automatic integer destination_cell(input integer row);
    destination_cell = active_direction == VXM_FLOW_LEFT_TO_RIGHT ?
      TILES+row : row;
  endfunction

  function automatic logic [31:0] operand_for_block(input integer block);
    case (block % 4)
      0, 2: operand_for_block = 32'h00000000; // Exp(0) = 1.
      1:    operand_for_block = 32'h40000000; // Reciprocal(2) = 0.5.
      default: operand_for_block = 32'h40800000; // Rsqrt(4) = 0.5.
    endcase
  endfunction

  always_comb begin
    west_valid = '0;
    west_data = '0;
    for (integer row = 0; row < TILES; row++) begin
      if (row_data_valid[row]) begin
        west_valid[source_cell(row)*STREAMS +: STREAMS] = ALL_INPUTS;
        for (integer block = 0; block < 8; block++) begin
          logic [31:0] operand;
          logic [15:0] half;
          operand = operand_for_block(block);
          half = row_data_high[row] ? operand[31:16] : operand[15:0];
          for (integer lane = 0; lane < LANES; lane++) begin
            west_data[
              ((source_cell(row)*STREAMS+block*4)*LANES+lane)*8 +: 8] =
              half[7:0];
            west_data[
              ((source_cell(row)*STREAMS+block*4+1)*LANES+lane)*8 +: 8] =
              half[15:8];
            // The Basic chain head still requires a valid RHS even though
            // Bypass ignores its numerical value.
            west_data[
              ((source_cell(row)*STREAMS+block*4+2)*LANES+lane)*8 +: 8] =
              8'h00;
            west_data[
              ((source_cell(row)*STREAMS+block*4+3)*LANES+lane)*8 +: 8] =
              8'h00;
          end
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      phase_seen <= '0;
      done_seen <= '0;
      output_low_seen <= '0;
      result_seen <= '0;
      cycle_count <= 0;
      for (integer row = 0; row < TILES; row++) begin
        row_load_cycle[row] <= -1;
        low_cycle[row] <= -1;
        high_cycle[row] <= -1;
      end
    end else begin
      cycle_count <= cycle_count + 1;
      done_seen <= done_seen | dut.tile_config_done;
      for (integer row = 0; row < TILES; row++) begin
        integer boundary_cell;
        boundary_cell = destination_cell(row);
        if (dut.row_config_fire[row] && (row_load_cycle[row] < 0))
          row_load_cycle[row] <= cycle_count;

        if (row_data_valid[row]) begin
          if (dut.tile_stream_consumed[row*STREAMS +: STREAMS] !==
              ALL_INPUTS)
            $fatal(1,
              "direction %0d row %0d FP32 Special phase not consumed",
              active_direction, row);
          if (row_data_high[row]) begin
            phase_seen[TILES+row] <= 1'b1;
            high_cycle[row] <= cycle_count;
          end else begin
            phase_seen[row] <= 1'b1;
            low_cycle[row] <= cycle_count;
          end
        end else if (|dut.tile_stream_consumed[row*STREAMS +: STREAMS]) begin
          $fatal(1, "Slice consumed FP32 Special data outside a port beat");
        end

        for (integer block = 0; block < 8; block++) begin
          integer low_stream;
          logic pair_valid;
          logic [15:0] observed;
          logic [15:0] expected;
          low_stream = block*2;
          pair_valid = east_valid[boundary_cell*STREAMS+low_stream] &&
            east_valid[boundary_cell*STREAMS+low_stream+1];
          if (pair_valid) begin
            expected = !output_low_seen[boundary_cell*STREAMS+low_stream] ?
              16'h0000 :
              (((block % 4) == 0 || (block % 4) == 2) ?
                16'h3f80 : 16'h3f00);
            for (integer lane = 0; lane < LANES; lane++) begin
              observed[7:0] = east_data[
                ((boundary_cell*STREAMS+low_stream)*LANES+lane)*8 +: 8];
              observed[15:8] = east_data[
                ((boundary_cell*STREAMS+low_stream+1)*LANES+lane)*8 +: 8];
              if (observed !== expected)
                $fatal(1,
                  "direction %0d row %0d block %0d FP32 Special mismatch",
                  active_direction, row, block);
            end
            if (!output_low_seen[boundary_cell*STREAMS+low_stream]) begin
              output_low_seen[boundary_cell*STREAMS+low_stream] <= 1'b1;
              output_low_seen[boundary_cell*STREAMS+low_stream+1] <= 1'b1;
            end else begin
              result_seen[boundary_cell*STREAMS+low_stream] <= 1'b1;
              result_seen[boundary_cell*STREAMS+low_stream+1] <= 1'b1;
            end
          end
        end
      end
      if (fault) begin
        $display("FP32 Special fault detail: global=%b config_wave=%b tile=%b pair_lut=%b pair_collision=%b bridge_conflict=%b",
          dut.global_config_fault, dut.config_wave_fault, dut.tile_fault,
          dut.pair_lut_fault, dut.pair_lut_collision, dut.bridge_conflict);
        $display("Tile0 internals: lane_fault=%b lut_protocol=%b lut_storage=%b lut_collision=%b",
          dut.gen_tile[0].u_tile.lane_fault,
          dut.gen_tile[0].u_tile.lut_lane_protocol_fault,
          dut.gen_tile[0].u_tile.lut_storage_fault,
          dut.gen_tile[0].u_tile.internal_lut_collision);
        $display("Tile1 internals: lane_fault=%b lut_protocol=%b lut_storage=%b lut_collision=%b",
          dut.gen_tile[1].u_tile.lane_fault,
          dut.gen_tile[1].u_tile.lut_lane_protocol_fault,
          dut.gen_tile[1].u_tile.lut_storage_fault,
          dut.gen_tile[1].u_tile.internal_lut_collision);
        $fatal(1, "FP32 Special Slice reported a fault");
      end
      if (conflict)
        $fatal(1, "FP32 Special Slice reported a routing conflict");
    end
  end

  task automatic program_lut(
    input logic [1:0] bank,
    input logic [15:0] input_min,
    input logic [15:0] segment_width,
    input logic [15:0] k,
    input logic [15:0] b
  );
    begin
      @(negedge clk);
      lut_config_bank = bank;
      lut_config_min = input_min;
      lut_config_width = segment_width;
      lut_write_bank = bank;
      lut_write_address = '0;
      lut_write_k = k;
      lut_write_b = b;
      lut_config_valid = 1'b1;
      lut_write_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      lut_config_valid = 1'b0;
      lut_write_valid = 1'b0;
    end
  endtask

  task automatic run_direction(input logic direction);
    logic [CELLS*STREAMS-1:0] expected_mask;
    integer destination;
    begin
      rst_n = 1'b0;
      run = 1'b0;
      local_issue_valid = '0;
      local_issue_instruction = '0;
      global_issue_valid = 1'b0;
      global_issue_instruction = '0;
      lut_config_valid = 1'b0;
      lut_write_valid = 1'b0;
      row_data_valid = '0;
      row_data_high = '0;
      external_east_valid = '0;
      external_east_data = '0;
      active_direction = direction;

      repeat (3) @(posedge clk);
      @(negedge clk);
      rst_n = 1'b1;
      program_lut(2'd0, 16'h0000, 16'h3c00, 16'h1000, 16'h0000);
      program_lut(2'd1, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);
      program_lut(2'd2, 16'h3c00, 16'h3c00, 16'h0000, 16'h8000);

      global_fields = '0;
      global_fields.flow_direction = direction;
      global_fields.chain_length = VXM_CHAIN_LENGTH_2;
      global_fields.compute_dtype = VXM_FORMAT_FP32;
      global_fields.lhs_read_bits = VXM_READ_BITS_32;
      global_fields.lhs_dtype = VXM_FORMAT_FP32;
      global_fields.rhs_read_bits = VXM_READ_BITS_32;
      global_fields.rhs_dtype = VXM_FORMAT_FP32;
      global_issue_instruction = global_fields;
      local_issue_instruction = '0;
      local_issue_instruction[1*LOCAL_WIDTH +: 3] = VXM_LOCAL_SPECIAL0;
      local_issue_instruction[3*LOCAL_WIDTH +: 3] = VXM_LOCAL_SPECIAL0;
      local_issue_instruction[5*LOCAL_WIDTH +: 3] = VXM_LOCAL_SPECIAL0;
      local_issue_instruction[7*LOCAL_WIDTH +: 3] = VXM_LOCAL_SPECIAL1;

      expected_mask = '0;
      for (integer row = 0; row < TILES; row++) begin
        destination = direction == VXM_FLOW_LEFT_TO_RIGHT ? TILES+row : row;
        expected_mask[destination*STREAMS +: STREAMS] = ALL_CHAIN2_OUTPUTS;
      end

      run = 1'b1;
      local_issue_valid = 8'hff;
      global_issue_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      local_issue_valid = '0;
      global_issue_valid = 1'b0;

      repeat (2) @(posedge clk);
      @(negedge clk);
      for (integer wave = 0; wave <= TILES; wave++) begin
        row_data_valid = '0;
        row_data_high = '0;
        if (wave < TILES)
          row_data_valid[wave] = 1'b1;
        if (wave > 0) begin
          row_data_valid[wave-1] = 1'b1;
          row_data_high[wave-1] = 1'b1;
        end
        @(posedge clk);
        @(negedge clk);
      end
      row_data_valid = '0;
      row_data_high = '0;

      wait ((result_seen & expected_mask) == expected_mask);
      wait (done_seen == {TILES{1'b1}});
      for (integer row = 0; row < TILES; row++) begin
        if (low_cycle[row] != row_load_cycle[row] + 2)
          $fatal(1, "row %0d FP32 Special low phase was misaligned", row);
        if (high_cycle[row] != low_cycle[row] + 1)
          $fatal(1, "row %0d FP32 Special high phase was misaligned", row);
      end
      if (phase_seen != {2*TILES{1'b1}})
        $fatal(1, "not every Slice row consumed both FP32 phases");
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    run = 1'b0;
    local_issue_valid = '0;
    local_issue_instruction = '0;
    global_issue_valid = 1'b0;
    global_issue_instruction = '0;
    lut_config_valid = 1'b0;
    lut_config_bank = '0;
    lut_config_min = '0;
    lut_config_width = '0;
    lut_write_valid = 1'b0;
    lut_write_bank = '0;
    lut_write_address = '0;
    lut_write_k = '0;
    lut_write_b = '0;
    external_east_valid = '0;
    external_east_data = '0;
    active_direction = VXM_FLOW_LEFT_TO_RIGHT;
    row_data_valid = '0;
    row_data_high = '0;

    run_direction(VXM_FLOW_LEFT_TO_RIGHT);
    run_direction(VXM_FLOW_RIGHT_TO_LEFT);
    $display("LPU_VXM_SLICE_FP32_SPECIAL_TB_PASS");
    $finish;
  end

  initial begin
    #100000;
    $fatal(1, "FP32 Special Slice timeout");
  end
endmodule
