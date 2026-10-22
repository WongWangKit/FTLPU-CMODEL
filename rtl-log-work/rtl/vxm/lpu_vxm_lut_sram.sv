// One physical LUT SRAM for one special-function type. The slope/intercept
// pair stays in one row so synthesis sees one SRAM, not
// independent k/b memories or an inferred multi-read array. COEFFICIENT_WIDTH
// is 13 for each packed half of the EXP UQ1.25 base and 16 for the RECIP or
// RSQRT UQ1.15 coefficients.
module lpu_vxm_lut_sram #(
  parameter integer ENTRY_COUNT = 64,
  parameter integer COEFFICIENT_WIDTH = 16,
  parameter integer ADDRESS_WIDTH =
    ENTRY_COUNT <= 1 ? 1 : $clog2(ENTRY_COUNT)
) (
  input  logic                     clk_i,
  input  logic                     rst_ni,
  input  logic                     config_valid_i,
  input  logic [15:0]              config_input_min_i,
  input  logic [15:0]              config_segment_width_i,
  input  logic                     write_valid_i,
  input  logic [ADDRESS_WIDTH-1:0] write_address_i,
  input  logic [COEFFICIENT_WIDTH-1:0] write_k_i,
  input  logic [COEFFICIENT_WIDTH-1:0] write_b_i,
  input  logic                     read_valid_i,
  input  logic [ADDRESS_WIDTH-1:0] read_address_i,
  output logic                     read_valid_o,
  output logic [COEFFICIENT_WIDTH-1:0] read_k_o,
  output logic [COEFFICIENT_WIDTH-1:0] read_b_o,
  output logic                     configured_o,
  output logic [15:0]              input_min_o,
  output logic [15:0]              segment_width_o,
  output logic                     fault_o
);
  import lpu_vxm_fp16_pkg::*;

  (* ram_style = "block" *) logic [2*COEFFICIENT_WIDTH-1:0] coefficient_memory
    [0:ENTRY_COUNT-1];
  logic [15:0] input_min_q;
  logic [15:0] segment_width_q;
  logic configured_q;

  function automatic logic valid_segment_width(input logic [15:0] width);
    valid_segment_width = !width[15] &&
      !fp16_is_nan(width) && !fp16_is_inf(width) &&
      !fp16_is_zero_or_subnormal(width);
  endfunction

  always_comb begin
    configured_o = configured_q;
    input_min_o = input_min_q;
    segment_width_o = segment_width_q;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      configured_q <= 1'b0;
      input_min_q <= 16'b0;
      segment_width_q <= FP16_ONE;
      read_valid_o <= 1'b0;
      read_k_o <= '0;
      read_b_o <= '0;
      fault_o <= 1'b0;
    end else begin
      read_valid_o <= 1'b0;
      fault_o <= 1'b0;

      if (config_valid_i) begin
        if (valid_segment_width(config_segment_width_i)) begin
          input_min_q <= fp16_sanitize_ftz(config_input_min_i);
          segment_width_q <=
            fp16_sanitize_ftz(config_segment_width_i);
          configured_q <= 1'b1;
        end else
          fault_o <= 1'b1;
      end

      if (write_valid_i) begin
        if (write_address_i < ENTRY_COUNT)
          coefficient_memory[write_address_i] <= {write_k_i, write_b_i};
        else
          fault_o <= 1'b1;
      end

      // Synchronous one-cycle read. A same-edge write/read returns the old
      // row, matching a non-write-through 1R1W SRAM contract.
      if (read_valid_i) begin
        if (read_address_i < ENTRY_COUNT) begin
          read_valid_o <= 1'b1;
          read_k_o <= coefficient_memory[read_address_i][
            2*COEFFICIENT_WIDTH-1 -: COEFFICIENT_WIDTH];
          read_b_o <= coefficient_memory[read_address_i][
            COEFFICIENT_WIDTH-1:0];
        end else
          fault_o <= 1'b1;
      end
    end
  end
endmodule
