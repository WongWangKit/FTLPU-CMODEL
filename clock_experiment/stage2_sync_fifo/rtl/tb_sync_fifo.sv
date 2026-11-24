module tb_sync_fifo;
    timeunit 1ns;
    timeprecision 1ps;

    localparam int STIMULUS_COUNT = 64;
    localparam time HALF_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;
    logic wr_en;
    logic [7:0] wr_data;
    logic rd_en;
    logic full;
    logic empty;
    logic wr_ready;
    logic rd_available;
    logic wr_fire;
    logic rd_fire;
    logic [1:0] wr_ptr_q;
    logic [1:0] rd_ptr_q;
    logic [2:0] count_q;
    logic [7:0] rd_data_q;
    logic rd_data_valid_q;
    logic [1:0] wr_ptr_d;
    logic [1:0] rd_ptr_d;
    logic [2:0] count_d;
    logic [7:0] rd_data_d;
    logic rd_data_valid_d;
    logic [10:0] stimuli [0:STIMULUS_COUNT-1];
    logic accepted_write;
    logic accepted_read;
    logic trace_state_valid = 1'b0;

    string stimulus_path;
    string trace_path;
    integer trace_file;
    integer index;
    longint unsigned half_clk = 0;

    sync_fifo dut (.*);

    function automatic string logic_text(input logic value);
        if (value === 1'b0) begin
            return "0";
        end
        if (value === 1'b1) begin
            return "1";
        end
        return "x";
    endfunction

    function automatic string vector_text(input logic [7:0] value);
        if (^value === 1'bx) begin
            return "x";
        end
        return $sformatf("%0d", value);
    endfunction

    function automatic string pointer_text(input logic [1:0] value);
        if (^value === 1'bx) begin
            return "x";
        end
        return $sformatf("%0d", value);
    endfunction

    function automatic string count_text(input logic [2:0] value);
        if (^value === 1'bx) begin
            return "x";
        end
        return $sformatf("%0d", value);
    endfunction

    function automatic string memory_text(input int slot);
        if (dut.mem_valid_q[slot] === 1'b1) begin
            return vector_text(dut.mem[slot]);
        end
        return "x";
    endfunction

    task automatic log_low;
        $fdisplay(trace_file,
            "%0d,%0d,LOW,%0d,%0d,%0d,%0d,%0d,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
            half_clk, half_clk >> 1, clk, trace_state_valid, rst, wr_en, wr_data, rd_en,
            logic_text(full), logic_text(empty), logic_text(wr_ready),
            logic_text(rd_available), logic_text(wr_fire), logic_text(rd_fire),
            pointer_text(wr_ptr_q), pointer_text(rd_ptr_q), count_text(count_q),
            vector_text(rd_data_q), logic_text(rd_data_valid_q),
            logic_text(dut.mem_valid_q[0]), memory_text(0),
            logic_text(dut.mem_valid_q[1]), memory_text(1),
            logic_text(dut.mem_valid_q[2]), memory_text(2),
            logic_text(dut.mem_valid_q[3]), memory_text(3),
            pointer_text(wr_ptr_d), pointer_text(rd_ptr_d), count_text(count_d),
            vector_text(rd_data_d), logic_text(rd_data_valid_d));
    endtask

    task automatic log_high;
        $fdisplay(trace_file,
            "%0d,%0d,HIGH,%0d,%0d,%0d,%0d,%0d,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,,,,,",
            half_clk, half_clk >> 1, clk, trace_state_valid, rst, wr_en, wr_data, rd_en,
            logic_text(full), logic_text(empty), logic_text(wr_ready),
            logic_text(rd_available), logic_text(accepted_write),
            logic_text(accepted_read), pointer_text(wr_ptr_q), pointer_text(rd_ptr_q),
            count_text(count_q), vector_text(rd_data_q), logic_text(rd_data_valid_q),
            logic_text(dut.mem_valid_q[0]), memory_text(0),
            logic_text(dut.mem_valid_q[1]), memory_text(1),
            logic_text(dut.mem_valid_q[2]), memory_text(2),
            logic_text(dut.mem_valid_q[3]), memory_text(3));
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
            "half_clk,cycle,phase,clk_level,state_valid,rst,wr_en,wr_data,rd_en,full,empty,wr_ready,rd_available,wr_fire,rd_fire,wr_ptr_q,rd_ptr_q,count_q,rd_data_q,rd_data_valid_q,mem0_valid,mem0,mem1_valid,mem1,mem2_valid,mem2,mem3_valid,mem3,wr_ptr_d,rd_ptr_d,count_d,rd_data_d,rd_data_valid_d");

        for (index = 0; index < STIMULUS_COUNT; index = index + 1) begin
            {rst, wr_en, wr_data, rd_en} = stimuli[index];
            #1ps;
            log_low();
            accepted_write = wr_fire;
            accepted_read = rd_fire;

            #(HALF_PERIOD - 1ps);
            clk = 1'b1;
            half_clk = half_clk + 1;
            #1ps;
            if (rst) begin
                trace_state_valid = 1'b1;
            end
            log_high();

            #(HALF_PERIOD - 1ps);
            clk = 1'b0;
            half_clk = half_clk + 1;
        end

        $fclose(trace_file);
        $display("RTL FIFO trace complete: %0d cycles, %0d samples",
                 STIMULUS_COUNT, STIMULUS_COUNT * 2);
        $finish;
    end
endmodule
