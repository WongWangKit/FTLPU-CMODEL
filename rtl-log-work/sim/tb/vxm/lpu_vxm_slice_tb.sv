`timescale 1ns/1ps

module lpu_vxm_slice_tb;
  import lpu_pkg::*;

  localparam integer HEMISPHERES = 2;
  localparam integer TILES = 4;
  localparam integer STREAMS = STREAMS_PER_DIRECTION;
  localparam integer LANES = LANES_PER_TILE;
  localparam integer WORD_WIDTH = LANES*8;
  localparam integer BOUNDARY_CELLS = HEMISPHERES*TILES;
  localparam integer LOCAL_MAX_WIDTH =
    VXM_LOCAL_MAX_INSTRUCTION_WIDTH;
  localparam integer CONFIG_WIDTH = VXM_GLOBAL_CONFIG_WIDTH;

  logic clk;
  logic rst_n;
  logic run;
  logic [7:0] local_issue_valid;
  logic [8*LOCAL_MAX_WIDTH-1:0] local_issue_instruction;
  logic global_issue_valid;
  logic [CONFIG_WIDTH-1:0] global_issue_instruction;
  vxm_global_config_t global_fields;
  logic [BOUNDARY_CELLS*STREAMS-1:0] west_valid;
  logic [BOUNDARY_CELLS*STREAMS*WORD_WIDTH-1:0] west_data;
  logic [BOUNDARY_CELLS*STREAMS-1:0] external_east_valid;
  logic [BOUNDARY_CELLS*STREAMS*WORD_WIDTH-1:0] external_east_data;
  wire [BOUNDARY_CELLS*STREAMS-1:0] east_valid;
  wire [BOUNDARY_CELLS*STREAMS*WORD_WIDTH-1:0] east_data;
  wire fault;
  wire conflict;

  logic active_direction;
  logic [TILES-1:0] data_row_valid;
  logic [TILES-1:0] data_row_high_half;
  logic [2*TILES-1:0] data_consumed_seen;
  logic [TILES-1:0] done_seen;
  logic [BOUNDARY_CELLS*STREAMS-1:0] result_seen;
  logic [BOUNDARY_CELLS*STREAMS-1:0] output_low_seen;
  integer cycle_count;
  integer row_load_cycle [0:TILES-1];
  integer data_low_cycle [0:TILES-1];
  integer data_high_cycle [0:TILES-1];

  localparam logic [STREAMS-1:0] CHAIN_HEAD_STREAM_MASK =
    32'h0f0f0f0f;

  always #5 clk = ~clk;

  // Keep the Slice waveform beside its VCS build products.  The VCS
  // +vcs+vcdpluson runtime switch only takes effect when the testbench
  // actually opens a VPD database, so do that explicitly here.
  initial begin
    $vcdplusfile("build/vcs/vxm_slice/vxm_slice.vpd");
    $vcdpluson(0, lpu_vxm_slice_tb);
  end

  lpu_vxm_slice dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .run_i(run),
    .local_issue_valid_i(local_issue_valid),
    .local_issue_instruction_i(local_issue_instruction),
    .global_issue_valid_i(global_issue_valid),
    .global_issue_instruction_i(global_issue_instruction),
    .lut_config_valid_i(1'b0),
    .lut_config_bank_i('0),
    .lut_config_input_min_i('0),
    .lut_config_segment_width_i('0),
    .lut_write_valid_i(1'b0),
    .lut_write_bank_i('0),
    .lut_write_address_i('0),
    .lut_write_k_i('0),
    .lut_write_b_i('0),
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
    source_cell = (active_direction == VXM_FLOW_LEFT_TO_RIGHT) ?
      row : TILES+row;
  endfunction

  function automatic integer destination_cell(input integer row);
    destination_cell = (active_direction == VXM_FLOW_LEFT_TO_RIGHT) ?
      TILES+row : row;
  endfunction

  function automatic logic stream_is_driven(input integer stream);
    stream_is_driven =
      ((stream >= 0)  && (stream <= 3))  ||
      ((stream >= 8)  && (stream <= 11)) ||
      ((stream >= 16) && (stream <= 19)) ||
      ((stream >= 24) && (stream <= 27));
  endfunction

  always_comb begin
    west_valid = '0;
    west_data = '0;
    for (integer row = 0; row < TILES; row++) begin
      if (data_row_valid[row]) begin
        for (integer stream = 0; stream < STREAMS; stream++) begin
          if (stream_is_driven(stream))
            west_valid[source_cell(row)*STREAMS+stream] = 1'b1;
        end
        for (integer block = 0; block < 8; block += 2) begin
          for (integer lane = 0; lane < LANES; lane++) begin
            // Two port beats encode LHS=FP32 1.0 and RHS=FP32 2.0.
            west_data[
              ((source_cell(row)*STREAMS+block*4)*LANES+lane)*8 +: 8] =
              data_row_high_half[row] ? 8'h80 : 8'h00;
            west_data[
              ((source_cell(row)*STREAMS+block*4+1)*LANES+lane)*8 +: 8] =
              data_row_high_half[row] ? 8'h3f : 8'h00;
            west_data[
              ((source_cell(row)*STREAMS+block*4+2)*LANES+lane)*8 +: 8] =
              8'h00;
            west_data[
              ((source_cell(row)*STREAMS+block*4+3)*LANES+lane)*8 +: 8] =
              data_row_high_half[row] ? 8'h40 : 8'h00;
          end
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_consumed_seen <= '0;
      done_seen <= '0;
      result_seen <= '0;
      output_low_seen <= '0;
      cycle_count <= 0;
      for (integer row = 0; row < TILES; row++) begin
        row_load_cycle[row] <= -1;
        data_low_cycle[row] <= -1;
        data_high_cycle[row] <= -1;
      end
    end else begin
      cycle_count <= cycle_count + 1;
      done_seen <= done_seen | dut.tile_config_done;

      for (integer row = 0; row < TILES; row++) begin
        if (dut.row_config_fire[row] && (row_load_cycle[row] < 0))
          row_load_cycle[row] <= cycle_count;

        // Stimulus enters only through the Slice boundary ports.  Internal
        // signals are observation points: the resident instruction and the
        // one-cycle data phase must be present when the Tile input collector
        // accepts the corresponding half of its chain-head operands.
        if (data_row_valid[row]) begin
          if (dut.tile_stream_consumed[row*STREAMS +: STREAMS] !==
              CHAIN_HEAD_STREAM_MASK)
            $fatal(1,
              "direction %0d row %0d data was not consumed on arrival",
              active_direction, row);
          if (data_row_high_half[row]) begin
            data_high_cycle[row] <= cycle_count;
            data_consumed_seen[TILES+row] <= 1'b1;
          end else begin
            data_low_cycle[row] <= cycle_count;
            data_consumed_seen[row] <= 1'b1;
          end
        end else if (|dut.tile_stream_consumed[row*STREAMS +: STREAMS]) begin
          $fatal(1,
            "direction %0d row %0d consumed data outside its input beat",
            active_direction, row);
        end

        for (integer pair = 0; pair < 4; pair++) begin
          integer low_stream;
          integer cell;
          logic pair_valid;
          logic data_matches;
          low_stream = pair*4 + 2;
          cell = destination_cell(row);
          pair_valid = east_valid[cell*STREAMS+low_stream] &&
            east_valid[cell*STREAMS+low_stream+1];
          data_matches = pair_valid;
          for (integer lane = 0; lane < LANES; lane++) begin
            if (!output_low_seen[cell*STREAMS+low_stream]) begin
              data_matches &= east_data[
                ((cell*STREAMS+low_stream)*LANES+lane)*8 +: 8] == 8'h00;
              data_matches &= east_data[
                ((cell*STREAMS+low_stream+1)*LANES+lane)*8 +: 8] == 8'h00;
            end else begin
              // Four chained Adds compute ((1+2)+1)+1 = FP32 6.0.
              data_matches &= east_data[
                ((cell*STREAMS+low_stream)*LANES+lane)*8 +: 8] == 8'hc0;
              data_matches &= east_data[
                ((cell*STREAMS+low_stream+1)*LANES+lane)*8 +: 8] == 8'h40;
            end
          end
          if (pair_valid && !data_matches)
            $fatal(1,
              "direction %0d row %0d FP32 output phase/data mismatch",
              active_direction, row);
          if (data_matches) begin
            if (!output_low_seen[cell*STREAMS+low_stream]) begin
              output_low_seen[cell*STREAMS+low_stream] <= 1'b1;
              output_low_seen[cell*STREAMS+low_stream+1] <= 1'b1;
            end else begin
              result_seen[cell*STREAMS+low_stream] <= 1'b1;
              result_seen[cell*STREAMS+low_stream+1] <= 1'b1;
            end
          end
        end
      end

      if (fault)
        $fatal(1, "VXM Slice reported a fault");
      if (conflict)
        $fatal(1, "VXM Slice boundary routing conflict");
    end
  end

  task automatic run_direction(input logic direction);
    logic [BOUNDARY_CELLS*STREAMS-1:0] expected_mask;
    integer destination;
    begin
      rst_n = 1'b0;
      run = 1'b0;
      local_issue_valid = '0;
      local_issue_instruction = '0;
      for (integer queue = 0; queue < VXM_LOCAL_QUEUE_COUNT; queue++)
        local_issue_instruction[
          queue*LOCAL_MAX_WIDTH +: 3] = VXM_LOCAL_ADD;
      global_issue_valid = 1'b0;
      global_issue_instruction = '0;
      data_row_valid = '0;
      data_row_high_half = '0;
      external_east_valid = '0;
      external_east_data = '0;
      active_direction = direction;

      global_fields = '0;
      global_fields.flow_direction = direction;
      global_fields.chain_length = VXM_CHAIN_LENGTH_4;
      global_fields.compute_dtype = VXM_FORMAT_FP32;
      global_fields.lhs_read_bits = VXM_READ_BITS_32;
      global_fields.lhs_dtype = VXM_FORMAT_FP32;
      global_fields.rhs_read_bits = VXM_READ_BITS_32;
      global_fields.rhs_dtype = VXM_FORMAT_FP32;
      global_issue_instruction = global_fields;

      expected_mask = '0;
      for (integer row = 0; row < TILES; row++) begin
        destination = (direction == VXM_FLOW_LEFT_TO_RIGHT) ?
          TILES+row : row;
        expected_mask[destination*STREAMS +: STREAMS] = 32'h0000cccc;
      end

      repeat (3) @(posedge clk);
      @(negedge clk);
      rst_n = 1'b1;
      run = 1'b1;
      local_issue_valid = 8'hff;
      global_issue_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      local_issue_valid = '0;
      global_issue_valid = 1'b0;

      // Two cycles after each row loads its resident instruction, FP32 phase
      // zero reaches that row. The high half follows one cycle later while
      // the next row receives its low half. This schedule is derived only
      // from the ICU issue cycle; no internal DUT signal drives stimulus.
      repeat (2) @(posedge clk);
      @(negedge clk);
      for (integer wave = 0; wave <= TILES; wave++) begin
        data_row_valid = '0;
        data_row_high_half = '0;
        if (wave < TILES)
          data_row_valid[wave] = 1'b1;
        if (wave > 0) begin
          data_row_valid[wave-1] = 1'b1;
          data_row_high_half[wave-1] = 1'b1;
        end
        @(posedge clk);
        @(negedge clk);
      end
      data_row_valid = '0;
      data_row_high_half = '0;

      wait ((result_seen & expected_mask) == expected_mask);
      wait (done_seen == {TILES{1'b1}});
      repeat (2) @(posedge clk);

      for (integer row = 0; row < TILES; row++) begin
        if (row_load_cycle[row] < 0)
          $fatal(1, "direction %0d row %0d was not configured",
                 direction, row);
        if ((row > 0) &&
            (row_load_cycle[row] != row_load_cycle[row-1] + 1))
          $fatal(1, "direction %0d configuration wave skipped row %0d",
                 direction, row);
        if (data_low_cycle[row] != row_load_cycle[row] + 2)
          $fatal(1,
            "direction %0d row %0d low-half latency mismatch",
            direction, row);
        if (data_high_cycle[row] != data_low_cycle[row] + 1)
          $fatal(1,
            "direction %0d row %0d FP32 phase order mismatch",
            direction, row);
        if ((row > 0) && (data_low_cycle[row] !=
                          data_low_cycle[row-1] + 1))
          $fatal(1, "direction %0d low-half wave skipped row %0d",
                 direction, row);
      end
      if (data_consumed_seen != {2*TILES{1'b1}})
        $fatal(1, "direction %0d did not consume both FP32 phases", direction);
      if (fault || conflict)
        $fatal(1, "direction %0d ended with fault/conflict", direction);
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
    external_east_valid = '0;
    external_east_data = '0;
    active_direction = VXM_FLOW_LEFT_TO_RIGHT;
    data_row_valid = '0;
    data_row_high_half = '0;

    run_direction(VXM_FLOW_LEFT_TO_RIGHT);
    run_direction(VXM_FLOW_RIGHT_TO_LEFT);
    $display("LPU_VXM_SLICE_TB_PASS");
    $finish;
  end

  initial begin
    #50000;
    $fatal(1, "VXM Slice direction regression timeout");
  end
endmodule
