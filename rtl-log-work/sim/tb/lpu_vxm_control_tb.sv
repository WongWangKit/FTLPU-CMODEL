`timescale 1ns/1ps

module lpu_vxm_control_tb;
  timeunit 1ns;
  timeprecision 1ps;

  localparam integer MAX_WIDTH =
    lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH;
  localparam integer CONFIG_WIDTH = lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH;

  logic clk;
  logic rst_n;
  logic [7:0] local_issue_valid;
  logic [8*MAX_WIDTH-1:0] local_issue_instruction;
  logic global_issue_valid;
  logic [CONFIG_WIDTH-1:0] global_issue_instruction;
  logic [CONFIG_WIDTH-1:0] accepted_global_config;
  logic [4*8-1:0] tile_local_valid;
  logic [4*8*MAX_WIDTH-1:0] tile_local_instruction;
  logic [4*CONFIG_WIDTH-1:0] tile_global_config;
  logic [3:0] tile_global_config_valid;
  logic global_config_valid;
  logic global_config_fault;

  lpu_vxm_control dut (
    .clk_i(clk),
    .rst_ni(rst_n),
    .datapath_idle_i(1'b1),
    .local_issue_valid_i(local_issue_valid),
    .local_issue_instruction_i(local_issue_instruction),
    .global_issue_valid_i(global_issue_valid),
    .global_issue_instruction_i(global_issue_instruction),
    .tile_local_valid_o(tile_local_valid),
    .tile_local_instruction_o(tile_local_instruction),
    .tile_global_config_o(tile_global_config),
    .tile_global_config_valid_o(tile_global_config_valid),
    .global_config_valid_o(global_config_valid),
    .global_config_fault_o(global_config_fault)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  task automatic expect_tile(input integer tile);
    begin
      if (tile_local_valid[tile*8 +: 8] != 8'hff)
        $fatal(1, "VXM tile %0d did not receive all local queues", tile);
      for (integer queue = 0; queue < 8; queue++) begin
        if (tile_local_instruction[
              (tile*8+queue)*MAX_WIDTH +: MAX_WIDTH] !=
            local_issue_instruction[queue*MAX_WIDTH +: MAX_WIDTH])
          $fatal(1, "VXM tile %0d queue %0d payload mismatch", tile, queue);
      end
      if (!tile_global_config_valid[tile] ||
          (tile_global_config[tile*CONFIG_WIDTH +: CONFIG_WIDTH] !=
           accepted_global_config))
        $fatal(1, "VXM tile %0d global wave mismatch", tile);
    end
  endtask

  initial begin
    rst_n = 1'b0;
    local_issue_valid = '0;
    local_issue_instruction = '0;
    global_issue_valid = 1'b0;
    global_issue_instruction = '0;

    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    // Values are already zero-padded to the seven-bit transport slots.  The
    // instantiated pipeline storage itself remains 6/5/7/5/7/5/7/5 bits.
    local_issue_instruction[0*MAX_WIDTH +: MAX_WIDTH] = 7'b0_10_0011;
    local_issue_instruction[1*MAX_WIDTH +: MAX_WIDTH] = 7'b00_1_0110;
    local_issue_instruction[2*MAX_WIDTH +: MAX_WIDTH] = 7'b101_0011;
    local_issue_instruction[3*MAX_WIDTH +: MAX_WIDTH] = 7'b00_1_1100;
    local_issue_instruction[4*MAX_WIDTH +: MAX_WIDTH] = 7'b011_1001;
    local_issue_instruction[5*MAX_WIDTH +: MAX_WIDTH] = 7'b00_0_1011;
    local_issue_instruction[6*MAX_WIDTH +: MAX_WIDTH] = 7'b110_0101;
    local_issue_instruction[7*MAX_WIDTH +: MAX_WIDTH] = 7'b00_1_0001;
    global_issue_instruction = 15'b0_10_01_10_01_00_10_11;
    accepted_global_config = global_issue_instruction;
    local_issue_valid = 8'hff;
    global_issue_valid = 1'b1;

    @(posedge clk);
    #1ns;
    if (!global_config_valid || global_config_fault)
      $fatal(1, "VXM initial global configuration failed");
    expect_tile(0);

    @(negedge clk);
    local_issue_valid = '0;
    global_issue_valid = 1'b0;

    @(posedge clk);
    #1ns;
    expect_tile(1);
    if (tile_local_valid[0 +: 8] != '0)
      $fatal(1, "VXM tile 0 local wave did not advance");

    // A new global word may not replace the active configuration while a
    // local wave remains in the four tile registers.
    @(negedge clk);
    global_issue_instruction = 15'b1_01_10_01_10_01_00_11;
    global_issue_valid = 1'b1;
    @(posedge clk);
    #1ns;
    expect_tile(2);
    if (!global_config_fault)
      $fatal(1, "VXM unsafe global update was not rejected");
    for (integer tile = 0; tile < 4; tile++) begin
      if (tile_global_config[tile*CONFIG_WIDTH +: CONFIG_WIDTH] ==
          global_issue_instruction)
        $fatal(1, "VXM unsafe global word reached tile %0d", tile);
    end

    @(negedge clk);
    global_issue_valid = 1'b0;
    @(posedge clk);
    #1ns;
    expect_tile(3);
    @(posedge clk);
    #1ns;
    if (tile_local_valid != '0)
      $fatal(1, "VXM four-tile local wave did not drain");
    if (tile_global_config_valid != 4'hf)
      $fatal(1, "VXM global wave did not reach all four tiles");

    $display("VXM local/global four-tile control waves passed");
    $finish;
  end
endmodule
