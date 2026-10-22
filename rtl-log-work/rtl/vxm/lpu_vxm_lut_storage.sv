module lpu_vxm_lut_storage #(
  parameter integer BANK_COUNT  = 3,
  parameter integer ENTRY_COUNT = 64,
  parameter integer BANK_WIDTH =
    BANK_COUNT <= 1 ? 1 : $clog2(BANK_COUNT),
  parameter integer ADDRESS_WIDTH =
    ENTRY_COUNT <= 1 ? 1 : $clog2(ENTRY_COUNT)
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic                     config_valid_i,
  input  logic [BANK_WIDTH-1:0]    config_bank_i,
  input  logic [15:0]              config_input_min_i,
  input  logic [15:0]              config_segment_width_i,

  input  logic                     write_valid_i,
  input  logic [BANK_WIDTH-1:0]    write_bank_i,
  input  logic [ADDRESS_WIDTH-1:0] write_address_i,
  input  logic [15:0]              write_k_i,
  input  logic [15:0]              write_b_i,

  input  logic                     read_valid_i,
  input  logic [BANK_WIDTH-1:0]    read_bank_i,
  input  logic [ADDRESS_WIDTH-1:0] read_address_i,
  output logic                     read_valid_o,
  output logic [15:0]              read_k_o,
  output logic [15:0]              read_b_o,

  output logic [BANK_COUNT-1:0]    configured_o,
  output logic [BANK_COUNT*16-1:0] input_min_o,
  output logic [BANK_COUNT*16-1:0] segment_width_o,
  output logic                     fault_o
);
  import lpu_vxm_fp16_pkg::*;

  logic [15:0] k_memory [0:BANK_COUNT-1][0:ENTRY_COUNT-1];
  logic [15:0] b_memory [0:BANK_COUNT-1][0:ENTRY_COUNT-1];
  logic [15:0] input_min_q [0:BANK_COUNT-1];
  logic [15:0] segment_width_q [0:BANK_COUNT-1];
  logic [BANK_COUNT-1:0] configured_q;

  function automatic logic valid_segment_width(input logic [15:0] width);
    valid_segment_width = !width[15] &&
      !fp16_is_nan(width) && !fp16_is_inf(width) &&
      !fp16_is_zero_or_subnormal(width);
  endfunction

  always_comb begin
    configured_o = configured_q;
    input_min_o = '0;
    segment_width_o = '0;
    for (integer bank = 0; bank < BANK_COUNT; bank++) begin
      input_min_o[bank*16 +: 16] = input_min_q[bank];
      segment_width_o[bank*16 +: 16] = segment_width_q[bank];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      configured_q <= '0;
      read_valid_o <= 1'b0;
      read_k_o <= 16'b0;
      read_b_o <= 16'b0;
      fault_o <= 1'b0;
      for (integer bank = 0; bank < BANK_COUNT; bank++) begin
        input_min_q[bank] <= 16'b0;
        segment_width_q[bank] <= FP16_ONE;
      end
    end else begin
      read_valid_o <= 1'b0;
      fault_o <= 1'b0;

      if (config_valid_i) begin
        if ((config_bank_i < BANK_COUNT) &&
            valid_segment_width(config_segment_width_i)) begin
          input_min_q[config_bank_i] <=
            fp16_sanitize_ftz(config_input_min_i);
          segment_width_q[config_bank_i] <=
            fp16_sanitize_ftz(config_segment_width_i);
          configured_q[config_bank_i] <= 1'b1;
        end else begin
          fault_o <= 1'b1;
        end
      end

      if (write_valid_i) begin
        if ((write_bank_i < BANK_COUNT) &&
            (write_address_i < ENTRY_COUNT)) begin
          k_memory[write_bank_i][write_address_i] <= write_k_i;
          b_memory[write_bank_i][write_address_i] <= write_b_i;
        end else begin
          fault_o <= 1'b1;
        end
      end

      // One-cycle, in-order read response. A same-cycle write/read collision
      // returns the old entry, matching non-write-through SRAM behavior.
      if (read_valid_i) begin
        if ((read_bank_i < BANK_COUNT) &&
            (read_address_i < ENTRY_COUNT)) begin
          read_valid_o <= 1'b1;
          read_k_o <= k_memory[read_bank_i][read_address_i];
          read_b_o <= b_memory[read_bank_i][read_address_i];
        end else begin
          fault_o <= 1'b1;
        end
      end
    end
  end
endmodule
