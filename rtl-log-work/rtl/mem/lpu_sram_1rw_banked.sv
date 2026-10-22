module lpu_sram_1rw_banked #(
  parameter integer WIDTH = 64,
  parameter integer DEPTH = 65536,
  parameter integer ADDR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  req_valid_i,
  input  logic                  write_i,
  input  logic [ADDR_WIDTH-1:0] address_i,
  input  logic [WIDTH-1:0]      write_data_i,
  input  logic [WIDTH-1:0]      write_mask_i,
  output logic                  read_valid_o,
  output logic [WIDTH-1:0]      read_data_o
);
  localparam integer MACRO_WIDTH = 64;
  localparam integer MACRO_DEPTH = 2048;
  localparam integer WIDTH_BANKS = (WIDTH + MACRO_WIDTH - 1) / MACRO_WIDTH;
  localparam integer DEPTH_BANKS = (DEPTH + MACRO_DEPTH - 1) / MACRO_DEPTH;
  localparam integer DEPTH_BANK_WIDTH =
    (DEPTH_BANKS <= 1) ? 1 : $clog2(DEPTH_BANKS);

  logic [DEPTH_BANK_WIDTH-1:0] selected_depth_bank;
  logic [DEPTH_BANK_WIDTH-1:0] read_depth_bank_q;
  logic [WIDTH_BANKS*MACRO_WIDTH-1:0] bank_read_data [0:DEPTH_BANKS-1];

`ifndef FTLPU_USE_SRAM64X2048_MACRO
  initial begin
    if ((WIDTH % MACRO_WIDTH) != 0)
      $error("lpu_sram_1rw_banked WIDTH must be a multiple of 64");
    if ((DEPTH % MACRO_DEPTH) != 0)
      $error("lpu_sram_1rw_banked DEPTH must be a multiple of 2048");
  end
`endif

  always_comb begin
    if (DEPTH_BANKS <= 1)
      selected_depth_bank = '0;
    else
      selected_depth_bank = address_i / MACRO_DEPTH;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      read_valid_o <= 1'b0;
      read_depth_bank_q <= '0;
    end else begin
      read_valid_o <= req_valid_i && !write_i;
      if (req_valid_i && !write_i)
        read_depth_bank_q <= selected_depth_bank;
    end
  end

  always_comb begin
    read_data_o = '0;
    if (read_valid_o)
      read_data_o = bank_read_data[read_depth_bank_q][WIDTH-1:0];
  end

  generate
    for (genvar depth_bank = 0;
         depth_bank < DEPTH_BANKS;
         depth_bank++) begin : gen_depth_bank
      for (genvar width_bank = 0;
           width_bank < WIDTH_BANKS;
           width_bank++) begin : gen_width_bank
        logic [63:0] macro_q;
        logic macro_selected;

        assign macro_selected = req_valid_i &&
          (selected_depth_bank == depth_bank);
        assign bank_read_data[depth_bank]
          [width_bank*MACRO_WIDTH +: MACRO_WIDTH] = macro_q;

`ifdef FTLPU_USE_SRAM64X2048_MACRO
        sram_64_2048 u_sram (
          .Q(macro_q),
          .CLK(clk_i),
          .CEN(!macro_selected),
          .WEN(~write_mask_i[width_bank*MACRO_WIDTH +: MACRO_WIDTH]),
          .A(address_i[10:0]),
          .D(write_data_i[width_bank*MACRO_WIDTH +: MACRO_WIDTH]),
          .EMA(3'b011),
          .EMAW(2'b01),
          .EMAS(1'b0),
          .GWEN(!(macro_selected && write_i)),
          .RET1N(1'b1)
        );
`else
        logic [63:0] behavior_mem [0:MACRO_DEPTH-1];
        logic [63:0] behavior_q;

        always_ff @(posedge clk_i) begin
          if (macro_selected) begin
            if (write_i) begin
              for (integer bit_index = 0;
                   bit_index < MACRO_WIDTH;
                   bit_index++) begin
                if (write_mask_i[width_bank*MACRO_WIDTH+bit_index])
                  behavior_mem[address_i[10:0]][bit_index] <=
                    write_data_i[width_bank*MACRO_WIDTH+bit_index];
              end
            end else begin
              behavior_q <= behavior_mem[address_i[10:0]];
            end
          end
        end

        assign macro_q = behavior_q;
`endif
      end
    end
  endgenerate
endmodule
