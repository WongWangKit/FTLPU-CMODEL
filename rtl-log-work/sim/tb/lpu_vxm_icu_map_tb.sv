`timescale 1ns/1ps

module lpu_vxm_icu_map_tb;
  timeunit 1ns;
  timeprecision 1ps;

  logic clk;
  logic rst_n;
  logic run;
  logic enqueue_valid;
  logic enqueue_ready;
  logic [7:0] enqueue_queue;
  logic enqueue_is_instruction;
  logic [31:0] enqueue_command;
  logic [415:0] enqueue_payload;
  logic [137:0] issue_valid;
  logic [138*416-1:0] issue_payload;
  logic [137:0] queue_fault;

  lpu_icu #(.QUEUE_DEPTH(4)) dut (
    .clk_i(clk), .rst_ni(rst_n), .run_i(run),
    .enqueue_valid_i(enqueue_valid), .enqueue_ready_o(enqueue_ready),
    .enqueue_queue_i(enqueue_queue),
    .enqueue_is_instruction_i(enqueue_is_instruction),
    .enqueue_command_i(enqueue_command),
    .enqueue_payload_i(enqueue_payload),
    .issue_valid_o(issue_valid), .issue_payload_o(issue_payload),
    .queue_fault_o(queue_fault)
  );

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  task automatic enqueue_vxm(input logic [7:0] queue);
    begin
      @(negedge clk);
      enqueue_queue = queue;
      enqueue_payload = '1;
      enqueue_valid = 1'b1;
      #1ps;
      if (!enqueue_ready)
        $fatal(1, "VXM queue %0d was not ready", queue);
      @(negedge clk);
      enqueue_valid = 1'b0;
    end
  endtask

  task automatic expect_payload(
    input integer queue,
    input integer width
  );
    logic [415:0] expected;
    begin
      expected = '0;
      for (integer bit_index = 0; bit_index < width; bit_index++)
        expected[bit_index] = 1'b1;
      if (issue_payload[queue*416 +: 416] != expected)
        $fatal(1, "VXM queue %0d physical width mismatch", queue);
    end
  endtask

  initial begin
    rst_n = 1'b0;
    run = 1'b0;
    enqueue_valid = 1'b0;
    enqueue_queue = '0;
    enqueue_is_instruction = 1'b1;
    enqueue_command = '0;
    enqueue_payload = '0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    for (integer queue = 112; queue <= 120; queue++)
      enqueue_vxm(queue[7:0]);

    @(negedge clk);
    enqueue_queue = 8'd121;
    enqueue_valid = 1'b1;
    #1ps;
    if (enqueue_ready)
      $fatal(1, "reserved former VXM queue 121 accepted an instruction");
    enqueue_valid = 1'b0;
    run = 1'b1;

    @(posedge clk);
    #1ns;
    if (issue_valid[120:112] != 9'h1ff)
      $fatal(1, "VXM 8 local plus global queues did not issue independently");
    expect_payload(112, 6);
    expect_payload(113, 5);
    expect_payload(114, 7);
    expect_payload(115, 6);
    expect_payload(116, 7);
    expect_payload(117, 5);
    expect_payload(118, 7);
    expect_payload(119, 6);
    expect_payload(120, lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH);
    if (queue_fault != '0)
      $fatal(1, "VXM ICU mapping fault: %h", queue_fault);

    // Q3/Q7 must retain extension bit 5 through the physical ICU FIFOs.
    @(negedge clk);run=1'b0;
    for (integer q=115;q<=119;q=q+4) begin
      @(negedge clk);enqueue_queue=8'(q);
      enqueue_payload='0;
      enqueue_payload[5:0]=lpu_pkg::VXM_LOG_INSTRUCTION;
      enqueue_valid=1'b1;#1ps;
      if(!enqueue_ready) $fatal(1,"Log ICU queue not ready");
      @(negedge clk);enqueue_valid=1'b0;
    end
    @(negedge clk);run=1'b1;
    @(posedge clk);#1ns;
    if(!issue_valid[115]||!issue_valid[119]||
       issue_payload[115*416+:6]!==lpu_pkg::VXM_LOG_INSTRUCTION||
       issue_payload[119*416+:6]!==lpu_pkg::VXM_LOG_INSTRUCTION)
      $fatal(1,"Log extension bit was truncated by ICU");

    $display("VXM physical 8 local plus global ICU queue map passed");
    $finish;
  end
endmodule
