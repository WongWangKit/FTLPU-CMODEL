module lpu_mxm_shared_accumulator #(
  parameter integer ACCUMULATOR_BLOCK_COUNT =
    lpu_pkg::MXM_ACCUMULATOR_BLOCK_COUNT
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,
  input  logic write_valid_i,
  input  logic write_block_mode_i,
  input  logic [8*32*32-1:0] write_values_i,
  input  logic [12:0] write_address_i,
  input  logic [5:0] write_stream_base_i,
  input  logic write_stream_destination_i,
  input  logic write_clear_i,
  output logic write_ready_o,
  input  logic [3:0] read_row_valid_i,
  input  logic [4*48-1:0] read_row_instruction_i,
  output logic [4*32-1:0] west_valid_o,
  output logic [4*32*64-1:0] west_data_o,
  output logic fault_o
);
`ifdef FTLPU_USE_SRAM128X512_MACRO
  lpu_mxm_shared_accumulator_macro #(
    .ACCUMULATOR_BLOCK_COUNT(ACCUMULATOR_BLOCK_COUNT)
  ) u_backend (.*);
`else
  logic vector_write_ready;
  logic block_write_ready;
  logic [3:0] vector_read_valid;
  logic [3:0] block_read_valid;
  logic [4*32-1:0] vector_west_valid;
  logic [4*32-1:0] block_west_valid;
  logic [4*32*64-1:0] vector_west_data;
  logic [4*32*64-1:0] block_west_data;
  logic vector_fault;
  logic block_fault;

  always_comb begin
    for (integer tile = 0; tile < 4; tile++) begin
      vector_read_valid[tile] = read_row_valid_i[tile] &&
        !read_row_instruction_i[tile*48+46];
      block_read_valid[tile] = read_row_valid_i[tile] &&
        read_row_instruction_i[tile*48+46];
    end
    write_ready_o = write_block_mode_i
      ? block_write_ready : vector_write_ready;
    west_valid_o = vector_west_valid | block_west_valid;
    west_data_o = '0;
    for (integer stream = 0; stream < 4*32; stream++) begin
      if (vector_west_valid[stream])
        west_data_o[stream*64 +: 64] = vector_west_data[stream*64 +: 64];
      else if (block_west_valid[stream])
        west_data_o[stream*64 +: 64] = block_west_data[stream*64 +: 64];
    end
    fault_o = vector_fault | block_fault |
      |(vector_west_valid & block_west_valid);
  end

  // Cycle-compatible architectural reference backend. The macro build below
  // replaces this pair with one physically shared 128 KiB banked SRAM.
  lpu_mxm_accumulator #(
    .ACCUMULATOR_BLOCK_COUNT(ACCUMULATOR_BLOCK_COUNT)
  ) u_vector_reference (
    .clk_i, .rst_ni, .run_i,
    .write_valid_i(write_valid_i && !write_block_mode_i),
    .write_values_i(write_values_i[0 +: 32*32]),
    .write_address_i, .write_stream_base_i,
    .write_stream_destination_i, .write_clear_i,
    .write_ready_o(vector_write_ready),
    .read_row_valid_i(vector_read_valid), .read_row_instruction_i,
    .west_valid_o(vector_west_valid), .west_data_o(vector_west_data),
    .fault_o(vector_fault)
  );

  lpu_mxm_block_accumulator #(
    .ACCUMULATOR_BLOCK_COUNT(ACCUMULATOR_BLOCK_COUNT)
  ) u_block_reference (
    .clk_i, .rst_ni, .run_i,
    .write_valid_i(write_valid_i && write_block_mode_i),
    .write_values_i, .write_address_i, .write_stream_base_i,
    .write_stream_destination_i, .write_clear_i,
    .write_ready_o(block_write_ready),
    .read_row_valid_i(block_read_valid), .read_row_instruction_i,
    .west_valid_o(block_west_valid), .west_data_o(block_west_data),
    .fault_o(block_fault)
  );
`endif
endmodule

module lpu_mxm_shared_accumulator_macro #(
  parameter integer ACCUMULATOR_BLOCK_COUNT =
    lpu_pkg::MXM_ACCUMULATOR_BLOCK_COUNT
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,
  input  logic write_valid_i,
  input  logic write_block_mode_i,
  input  logic [8*32*32-1:0] write_values_i,
  input  logic [12:0] write_address_i,
  input  logic [5:0] write_stream_base_i,
  input  logic write_stream_destination_i,
  input  logic write_clear_i,
  output logic write_ready_o,
  input  logic [3:0] read_row_valid_i,
  input  logic [4*48-1:0] read_row_instruction_i,
  output logic [4*32-1:0] west_valid_o,
  output logic [4*32*64-1:0] west_data_o,
  output logic fault_o
);
  import lpu_vxm_math_pkg::*;

  localparam integer ROW_BANKS = 8;
  localparam integer LANES = 8;
  localparam integer STREAMS = 32;
  localparam integer VECTOR_DEPTH = ACCUMULATOR_BLOCK_COUNT * 32;

  typedef enum logic [2:0] {
    STATE_IDLE,
    STATE_ISSUE_READ,
    STATE_READ_DATA,
    STATE_ACCUMULATE,
    STATE_WRITE_DATA
  } state_t;

  state_t state_q;
  logic operation_write_q;
  logic [1:0] operation_segment_q;
  logic [2:0] operation_row_bank_q;
  logic [1:0] operation_output_block_q;
  logic [12:0] operation_address_q;
  logic [5:0] operation_stream_base_q;
  logic operation_stream_destination_q;
  logic operation_clear_q;
  logic [32*32-1:0] operation_values_q;

  logic [3:0] read_opcode_valid;
  logic read_overlapping;
  logic read_active;
  logic [1:0] read_output_block;
  logic [47:0] read_instruction;
  logic read_supported;

  logic [8:0] physical_address;
  logic [8:0] init_address_q;
  logic init_done_q;
  logic [2:0] vector_bank;
  logic [1:0] stream_output_block;
  logic [ROW_BANKS-1:0] active_banks;
  logic [ROW_BANKS-1:0] bank_req;
  logic [255:0] bank_read_data [0:ROW_BANKS-1];
  logic [255:0] bank_stored_data [0:ROW_BANKS-1];
  logic [255:0] selected_stored_data;
  logic [255:0] selected_operand_data;
  logic [255:0] selected_result_data;
  logic [255:0] result_data_q;
  logic [2:0] accumulate_lane_q;
  logic [31:0] lane_sum;
  logic [7:0] lane_sum_status;
  logic output_active;
  logic [255:0] output_values;

  function automatic logic [8:0] vector_physical_address(
    input logic [12:0] address,
    input logic [1:0] segment
  );
    vector_physical_address =
      (address / 32) * 16 + ((address % 32) / 8) * 4 + segment;
  endfunction

  function automatic logic accumulator_read_supported(
    input logic [47:0] instruction
  );
    accumulator_read_supported =
      (instruction[1:0] == 2'd2) &&
      (instruction[8:2] == '0) &&
      (instruction[14:9] <= 6'd28) &&
      (instruction[27:15] < VECTOR_DEPTH) &&
      (instruction[45:29] == '0) &&
      !instruction[46] && !instruction[47];
  endfunction

  initial begin
    if (ACCUMULATOR_BLOCK_COUNT < 1 || ACCUMULATOR_BLOCK_COUNT > 32)
      $error("ACCUMULATOR_BLOCK_COUNT must be in the range 1..32");
  end

  always_comb begin
    read_opcode_valid = '0;
    for (integer tile = 0; tile < 4; tile++) begin
      read_opcode_valid[tile] = read_row_valid_i[tile] &&
        (read_row_instruction_i[tile*48 +: 2] == 2'd2);
    end
    read_overlapping =
      (read_opcode_valid & (read_opcode_valid - 1'b1)) != 0;
    read_active = 1'b0;
    read_output_block = '0;
    read_instruction = '0;
    for (integer tile = 0; tile < 4; tile++) begin
      if (read_opcode_valid[tile]) begin
        read_active = 1'b1;
        read_output_block = tile;
        read_instruction = read_row_instruction_i[tile*48 +: 48];
      end
    end
    read_supported = accumulator_read_supported(read_instruction);
  end

  always_comb begin
    write_ready_o = init_done_q && run_i && !write_block_mode_i &&
      state_q == STATE_IDLE && !read_active;
    stream_output_block = operation_write_q
      ? operation_segment_q : operation_output_block_q;

    physical_address = vector_physical_address(
      operation_address_q, operation_segment_q);
    vector_bank = operation_address_q[2:0];
    active_banks = 8'b1 << operation_address_q[2:0];

    for (integer bank = 0; bank < ROW_BANKS; bank++) begin
      bank_req[bank] = !init_done_q ||
        (active_banks[bank] &&
         (state_q == STATE_ISSUE_READ || state_q == STATE_WRITE_DATA));
      bank_stored_data[bank] = bank_read_data[bank];
    end

    selected_stored_data = bank_stored_data[operation_row_bank_q];
    selected_operand_data = operation_values_q[
      operation_segment_q*LANES*32 +: LANES*32];
    selected_result_data = result_data_q;
  end

  DW_fp_add #(
    .sig_width(23),
    .exp_width(8),
    .ieee_compliance(0)
  ) u_accumulate_add (
    .a(result_data_q[accumulate_lane_q*32 +: 32]),
    .b(selected_operand_data[accumulate_lane_q*32 +: 32]),
    .rnd(3'b000),
    .z(lane_sum),
    .status(lane_sum_status)
  );

  generate
    for (genvar bank = 0; bank < ROW_BANKS; bank++) begin : gen_row_bank
      lpu_mxm_accumulator_sram u_bank (
        .clk_i,
        .req_valid_i(bank_req[bank]),
        .write_i(!init_done_q || state_q == STATE_WRITE_DATA),
        .address_i(init_done_q ? physical_address : init_address_q),
        .write_data_i((!init_done_q ||
          (operation_clear_q &&
           (!operation_write_q || operation_stream_destination_q)))
          ? '0 : selected_result_data),
        .write_mask_i({256{bank_req[bank]}}),
        .read_data_o(bank_read_data[bank])
      );
    end
  endgenerate

  always_comb begin
    west_valid_o = '0;
    west_data_o = '0;
    output_active = (state_q == STATE_READ_DATA && !operation_write_q) ||
      (state_q == STATE_WRITE_DATA && operation_write_q &&
       operation_stream_destination_q);
    output_values = (state_q == STATE_READ_DATA)
      ? bank_stored_data[vector_bank] : selected_result_data;
    for (integer stream = 0; stream < 4*STREAMS; stream++) begin
      for (integer byte_index = 0; byte_index < 4; byte_index++) begin
        if (output_active &&
            stream == stream_output_block*STREAMS+
                      operation_stream_base_q+byte_index) begin
          west_valid_o[stream] = 1'b1;
          for (integer lane = 0; lane < LANES; lane++) begin
            west_data_o[stream*64+lane*8 +: 8] =
              output_values[lane*32+byte_index*8 +: 8];
          end
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= STATE_IDLE;
      operation_write_q <= 1'b0;
      operation_segment_q <= '0;
      operation_row_bank_q <= '0;
      operation_output_block_q <= '0;
      operation_address_q <= '0;
      operation_stream_base_q <= '0;
      operation_stream_destination_q <= 1'b0;
      operation_clear_q <= 1'b0;
      operation_values_q <= '0;
      result_data_q <= '0;
      accumulate_lane_q <= '0;
      init_address_q <= '0;
      init_done_q <= 1'b0;
      fault_o <= 1'b0;
    end else if (!init_done_q) begin
      state_q <= STATE_IDLE;
      fault_o <= 1'b0;
      if (init_address_q == 9'd511)
        init_done_q <= 1'b1;
      else
        init_address_q <= init_address_q + 1'b1;
    end else if (!run_i) begin
      state_q <= STATE_IDLE;
      fault_o <= 1'b0;
    end else begin
      if (read_overlapping ||
          (read_active && !read_supported) ||
          (read_active && (state_q != STATE_IDLE || write_valid_i)) ||
          (write_valid_i &&
            (!write_ready_o || write_address_i >= VECTOR_DEPTH)))
        fault_o <= 1'b1;

      case (state_q)
        STATE_IDLE: begin
          if (write_valid_i && write_ready_o) begin
            operation_write_q <= 1'b1;
            operation_segment_q <= '0;
            operation_row_bank_q <= write_address_i[2:0];
            operation_output_block_q <= '0;
            operation_address_q <= write_address_i;
            operation_stream_base_q <= write_stream_base_i;
            operation_stream_destination_q <=
              write_stream_destination_i;
            operation_clear_q <= write_clear_i;
            operation_values_q <= write_values_i[0 +: 32*32];
            state_q <= STATE_ISSUE_READ;
          end else if (read_active && !read_overlapping && read_supported) begin
            operation_write_q <= 1'b0;
            operation_segment_q <= read_output_block;
            operation_row_bank_q <= read_instruction[17:15];
            operation_output_block_q <= read_output_block;
            operation_address_q <= read_instruction[27:15];
            operation_stream_base_q <= read_instruction[14:9];
            operation_stream_destination_q <= 1'b1;
            operation_clear_q <= read_instruction[28];
            operation_values_q <= '0;
            state_q <= STATE_ISSUE_READ;
          end
        end
        STATE_ISSUE_READ: state_q <= STATE_READ_DATA;
        STATE_READ_DATA: begin
          if (operation_write_q) begin
            result_data_q <= selected_stored_data;
            accumulate_lane_q <= '0;
            state_q <= STATE_ACCUMULATE;
          end
          else begin
            if (operation_clear_q)
              state_q <= STATE_WRITE_DATA;
            else
              state_q <= STATE_IDLE;
          end
        end
        STATE_ACCUMULATE: begin
          result_data_q[accumulate_lane_q*32 +: 32] <= lane_sum;
          if (accumulate_lane_q == 3'd7)
            state_q <= STATE_WRITE_DATA;
          else
            accumulate_lane_q <= accumulate_lane_q + 1'b1;
        end
        STATE_WRITE_DATA: begin
          if (!operation_write_q)
            state_q <= STATE_IDLE;
          else if (operation_segment_q != 2'd3) begin
            operation_segment_q <= operation_segment_q + 1'b1;
            state_q <= STATE_ISSUE_READ;
          end else
            state_q <= STATE_IDLE;
        end
        default: state_q <= STATE_IDLE;
      endcase
    end
  end
endmodule
