module lpu_mxm_accumulator_sram (
  input  logic         clk_i,
  input  logic         req_valid_i,
  input  logic         write_i,
  input  logic [8:0]   address_i,
  input  logic [255:0] write_data_i,
  input  logic [255:0] write_mask_i,
  output logic [255:0] read_data_o
);
  generate
    for (genvar half = 0; half < 2; half++) begin : gen_half
      logic [127:0] macro_q;

      assign read_data_o[half*128 +: 128] = macro_q;

`ifdef FTLPU_USE_SRAM128X512_MACRO
      sram_128_512 u_sram (
        .Q(macro_q),
        .CLK(clk_i),
        .CEN(!req_valid_i),
        .WEN(~write_mask_i[half*128 +: 128]),
        .A(address_i),
        .D(write_data_i[half*128 +: 128]),
        .EMA(3'b011),
        .EMAW(2'b01),
        .EMAS(1'b0),
        .GWEN(!(req_valid_i && write_i)),
        .RET1N(1'b1)
      );
`else
      logic [127:0] behavior_mem [0:511];
      logic [127:0] behavior_q;

      always_ff @(posedge clk_i) begin
        if (req_valid_i) begin
          if (write_i) begin
            for (integer bit_index = 0; bit_index < 128; bit_index++) begin
              if (write_mask_i[half*128+bit_index])
                behavior_mem[address_i][bit_index] <=
                  write_data_i[half*128+bit_index];
            end
          end else begin
            behavior_q <= behavior_mem[address_i];
          end
        end
      end

      assign macro_q = behavior_q;
`endif
    end
  endgenerate
endmodule
