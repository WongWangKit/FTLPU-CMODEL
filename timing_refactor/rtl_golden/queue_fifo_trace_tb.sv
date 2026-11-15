`timescale 1ns/1ps

module queue_fifo_trace_tb;
    localparam integer DATA_WIDTH = 8;
    localparam integer DEPTH = 2;

    logic clk_i;
    logic rst_ni;
    logic enqueue_valid_i;
    logic enqueue_ready_o;
    logic [DATA_WIDTH-1:0] enqueue_data_i;
    logic dequeue_valid_o;
    logic dequeue_ready_i;
    logic [DATA_WIDTH-1:0] dequeue_data_o;
    logic [$clog2(DEPTH+1)-1:0] level_o;

    integer trace_file;
    integer cycle_index;

    lpu_queue_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .clk_i,
        .rst_ni,
        .enqueue_valid_i,
        .enqueue_ready_o,
        .enqueue_data_i,
        .dequeue_valid_o,
        .dequeue_ready_i,
        .dequeue_data_o,
        .level_o
    );

    always #5 clk_i = ~clk_i;

    task automatic trace_cycle;
        input logic enqueue_valid;
        input logic [DATA_WIDTH-1:0] enqueue_data;
        input logic dequeue_ready;
        integer enqueue_ready_pre;
        integer dequeue_valid_pre;
        integer enqueue_fire;
        integer dequeue_fire;
        integer dequeue_data_pre;
        integer level_pre;
        integer write_ptr_pre;
        integer read_ptr_pre;
        begin
            @(negedge clk_i);
            enqueue_valid_i = enqueue_valid;
            enqueue_data_i = enqueue_data;
            dequeue_ready_i = dequeue_ready;
            #1;
            enqueue_ready_pre = enqueue_ready_o;
            dequeue_valid_pre = dequeue_valid_o;
            enqueue_fire = enqueue_valid_i && enqueue_ready_o;
            dequeue_fire = dequeue_valid_o && dequeue_ready_i;
            dequeue_data_pre = dequeue_valid_o ? dequeue_data_o : 0;
            level_pre = level_o;
            write_ptr_pre = dut.write_ptr_q;
            read_ptr_pre = dut.read_ptr_q;
            @(posedge clk_i);
            #1;
            $fwrite(trace_file,
                "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                cycle_index, enqueue_valid, enqueue_data, dequeue_ready,
                enqueue_ready_pre, dequeue_valid_pre, enqueue_fire,
                dequeue_fire, dequeue_data_pre, level_pre, write_ptr_pre,
                read_ptr_pre, level_o, dut.write_ptr_q, dut.read_ptr_q,
                dequeue_valid_o, dequeue_valid_o ? dequeue_data_o : 0);
            cycle_index = cycle_index + 1;
        end
    endtask

    initial begin
        clk_i = 1'b0;
        rst_ni = 1'b0;
        enqueue_valid_i = 1'b0;
        enqueue_data_i = '0;
        dequeue_ready_i = 1'b0;
        cycle_index = 0;
        trace_file = $fopen("queue_fifo_rtl_trace.csv", "w");
        if(trace_file == 0) $fatal(1, "cannot open RTL trace");
        $fwrite(trace_file,
            "cycle,enqueue_valid,enqueue_data,dequeue_ready,enqueue_ready_pre,dequeue_valid_pre,enqueue_fire,dequeue_fire,dequeue_data_pre,level_pre,write_ptr_pre,read_ptr_pre,level_post,write_ptr_post,read_ptr_post,dequeue_valid_post,dequeue_data_post\n");

        repeat(2) @(posedge clk_i);
        @(negedge clk_i);
        rst_ni = 1'b1;

        trace_cycle(1'b1, 8'd10, 1'b0);
        trace_cycle(1'b1, 8'd20, 1'b0);
        trace_cycle(1'b1, 8'd30, 1'b0);
        trace_cycle(1'b1, 8'd30, 1'b1);
        trace_cycle(1'b1, 8'd40, 1'b1);
        trace_cycle(1'b0, 8'd0, 1'b1);
        trace_cycle(1'b0, 8'd0, 1'b1);
        trace_cycle(1'b1, 8'd50, 1'b1);
        trace_cycle(1'b1, 8'd60, 1'b0);
        trace_cycle(1'b1, 8'd70, 1'b1);
        trace_cycle(1'b0, 8'd0, 1'b1);
        trace_cycle(1'b0, 8'd0, 1'b1);

        $fclose(trace_file);
        $display("queue_fifo_trace_tb wrote %0d rows", cycle_index);
        $finish;
    end
endmodule
