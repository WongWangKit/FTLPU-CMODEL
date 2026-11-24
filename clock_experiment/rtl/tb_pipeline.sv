module tb_pipeline;
    timeunit 1ns;
    timeprecision 1ps;

    localparam int STIMULUS_COUNT = 32;
    localparam time HALF_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;
    logic en;
    logic [7:0] din;
    logic [7:0] q1;
    logic [7:0] q2;
    logic [7:0] dout;
    logic [7:0] d1;
    logic [7:0] d2;
    logic [9:0] stimuli [0:STIMULUS_COUNT-1];

    string stimulus_path;
    string trace_path;
    integer trace_file;
    integer index;
    longint unsigned half_clk = 0;

    pipeline dut (.*);

    task automatic log_low(input bit valid);
        if (valid) begin
            $fdisplay(trace_file,
                "%0d,%0d,LOW,%0d,1,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                half_clk, half_clk >> 1, clk, rst, en, din,
                q1, q2, dout, d1, d2);
        end else begin
            $fdisplay(trace_file,
                "%0d,%0d,LOW,%0d,0,%0d,%0d,%0d,x,x,x,%0d,%0d",
                half_clk, half_clk >> 1, clk, rst, en, din, d1, d2);
        end
    endtask

    task automatic log_high;
        $fdisplay(trace_file,
            "%0d,%0d,HIGH,%0d,1,%0d,%0d,%0d,%0d,%0d,%0d,,",
            half_clk, half_clk >> 1, clk, rst, en, din, q1, q2, dout);
    endtask

    initial begin
        if (!$value$plusargs("STIMULUS=%s", stimulus_path)) begin
            $fatal(1, "missing +STIMULUS=<path>");
        end
        if (!$value$plusargs("TRACE=%s", trace_path)) begin
            $fatal(1, "missing +TRACE=<path>");
        end

        $readmemh(stimulus_path, stimuli);
        trace_file = $fopen(trace_path, "w");
        if (trace_file == 0) begin
            $fatal(1, "cannot open trace file %s", trace_path);
        end
        $fdisplay(trace_file,
            "half_clk,cycle,phase,clk_level,valid,rst,en,din,q1,q2,dout,d1,d2");

        for (index = 0; index < STIMULUS_COUNT; index = index + 1) begin
            {rst, en, din} = stimuli[index];
            #1ps;
            log_low(index != 0);

            #(HALF_PERIOD - 1ps);
            clk = 1'b1;
            half_clk = half_clk + 1;
            #1ps;
            log_high();

            #(HALF_PERIOD - 1ps);
            clk = 1'b0;
            half_clk = half_clk + 1;
        end

        $fclose(trace_file);
        $display("RTL trace complete: %0d cycles, %0d samples",
                 STIMULUS_COUNT, STIMULUS_COUNT * 2);
        $finish;
    end
endmodule
