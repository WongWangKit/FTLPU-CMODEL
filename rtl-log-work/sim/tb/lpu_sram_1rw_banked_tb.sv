`timescale 1ns/1ps

module lpu_sram_1rw_banked_tb;
  logic clk;
  logic rst_n;
  logic req_valid;
  logic write;
  logic [11:0] address;
  logic [127:0] write_data;
  logic [127:0] write_mask;
  logic read_valid;
  logic [127:0] read_data;

  lpu_sram_1rw_banked #(
    .WIDTH(128),
    .DEPTH(4096),
    .ADDR_WIDTH(12)
  ) dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .req_valid_i(req_valid),
    .write_i(write),
    .address_i(address),
    .write_data_i(write_data),
    .write_mask_i(write_mask),
    .read_valid_o(read_valid),
    .read_data_o(read_data)
  );

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  task automatic write_word(
    input logic [11:0] word_address,
    input logic [127:0] value,
    input logic [127:0] mask
  );
    @(negedge clk);
    req_valid = 1'b1;
    write = 1'b1;
    address = word_address;
    write_data = value;
    write_mask = mask;
    @(negedge clk);
    req_valid = 1'b0;
  endtask

  task automatic check_word(
    input logic [11:0] word_address,
    input logic [127:0] expected
  );
    @(negedge clk);
    req_valid = 1'b1;
    write = 1'b0;
    address = word_address;
    @(posedge clk);
    #1;
    if (!read_valid || read_data !== expected)
      $fatal(1, "address %0d: got %h, expected %h",
             word_address, read_data, expected);
    @(negedge clk);
    req_valid = 1'b0;
  endtask

  initial begin
    rst_n = 1'b0;
    req_valid = 1'b0;
    write = 1'b0;
    address = '0;
    write_data = '0;
    write_mask = '0;
    repeat (2) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    write_word(12'd7, 128'h00112233445566778899aabbccddeeff,
               {128{1'b1}});
    write_word(12'd2055, 128'hffeeddccbbaa99887766554433221100,
               {128{1'b1}});
    check_word(12'd7, 128'h00112233445566778899aabbccddeeff);
    check_word(12'd2055, 128'hffeeddccbbaa99887766554433221100);

    write_word(12'd7, 128'hffffffffffffffff0000000000000000,
               {{64{1'b0}}, {64{1'b1}}});
    check_word(12'd7, 128'h00112233445566770000000000000000);

    $display("FTLPU banked SRAM test passed");
    $finish;
  end
endmodule
