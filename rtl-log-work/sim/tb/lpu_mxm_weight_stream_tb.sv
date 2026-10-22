`timescale 1ns/1ps

module lpu_mxm_weight_stream_tb;
  timeunit 1ns;
  timeprecision 1ps;

  logic clk;
  logic rst_n;
  logic run;
  logic [3:0] load_valid;
  logic [4*48-1:0] load_instruction;
  logic [3:0] dequant_valid;
  logic [4*16-1:0] dequant_instruction;
  logic [4*32-1:0] east_valid;
  logic [4*32*64-1:0] east_data;
  logic [4*32-1:0] consumed [0:1];
  logic [2*4*4*8*8*16-1:0] weights [0:1];
  logic [2*4*4-1:0] cells [0:1];
  logic [1:0] fault;

  for (genvar mxm = 0; mxm < 2; mxm++) begin : gen_weight_buffer
    lpu_mxm_weight_buffer #(
      .LOCAL_MXM_INDEX(mxm)
    ) dut (
      .clk_i(clk),
      .rst_ni(rst_n),
      .run_i(run),
      .load_row_valid_i(load_valid),
      .load_row_instruction_i(load_instruction),
      .dequant_row_valid_i(dequant_valid),
      .dequant_row_instruction_i(dequant_instruction),
      .east_valid_i(east_valid),
      .east_data_i(east_data),
      .east_consumed_o(consumed[mxm]),
      .weight_bits_o(weights[mxm]),
      .cell_valid_o(cells[mxm]),
      .fault_o(fault[mxm])
    );
  end

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  initial begin
    rst_n = 1'b0;
    run = 1'b0;
    load_valid = '0;
    load_instruction = '0;
    dequant_valid = '0;
    dequant_instruction = '0;
    east_valid = '0;
    east_data = '0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    run = 1'b1;

    // INT8 dequant IW for supercell column block 0.  Both local MXMs see the
    // same physical bus, but MXM0 owns streams 0..7 and MXM1 owns 8..15.
    load_valid = 4'hf;
    dequant_valid = 4'hf;
    for (integer tile = 0; tile < 4; tile++) begin
      load_instruction[tile*48 +: 48] = 48'd0;
      dequant_instruction[tile*16 +: 16] = 16'h3f80;
      for (integer stream = 0; stream < 16; stream++) begin
        east_valid[tile*32+stream] = 1'b1;
        for (integer lane = 0; lane < 8; lane++)
          east_data[((tile*32+stream)*64)+lane*8 +: 8] =
            stream < 8 ? stream + 1 : stream + 12;
      end
    end
    #1ns;
    for (integer tile = 0; tile < 4; tile++) begin
      if (consumed[0][tile*32 +: 16] !== 16'h00ff)
        $fatal(1, "MXM0 INT8 stream window mismatch tile=%0d", tile);
      if (consumed[1][tile*32 +: 16] !== 16'hff00)
        $fatal(1, "MXM1 INT8 stream window mismatch tile=%0d", tile);
    end

    @(posedge clk);
    #1ns;
    load_valid = '0;
    dequant_valid = '0;
    east_valid = '0;
    if (fault != '0)
      $fatal(1, "dual MXM weight buffers raised fault=%b", fault);
    if (weights[0][0 +: 16] !== 16'h3f80)
      $fatal(1, "MXM0 loaded wrong INT8 weight: %h", weights[0][0 +: 16]);
    if (weights[1][0 +: 16] !== 16'h41a0)
      $fatal(1, "MXM1 loaded wrong INT8 weight: %h", weights[1][0 +: 16]);

    $display("FTLPU-VMODEL dual-MXM weight stream mapping passed");
    $finish;
  end
endmodule
