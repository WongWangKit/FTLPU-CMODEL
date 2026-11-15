module tb_two_fifo_system;
    timeunit 1ns;
    timeprecision 1ps;

    localparam int STIMULUS_COUNT = 96;
    localparam time HALF_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;
    logic producer_valid;
    logic [7:0] producer_data;
    logic consumer_ready;
    logic tc_a_rd_en;
    logic tc_b_wr_en;
    logic [7:0] tc_b_wr_data;
    logic tc_b_rd_en;
    logic producer_ready;
    logic transfer_invariant_ok;
    logic a_full, a_empty, a_wr_ready, a_rd_available, a_wr_fire, a_rd_fire;
    logic [1:0] a_wr_ptr_q, a_rd_ptr_q, a_wr_ptr_d, a_rd_ptr_d;
    logic [2:0] a_count_q, a_count_d;
    logic [7:0] a_rd_data_q, a_rd_data_d;
    logic a_rd_data_valid_q, a_rd_data_valid_d;
    logic b_full, b_empty, b_wr_ready, b_rd_available, b_wr_fire, b_rd_fire;
    logic [1:0] b_wr_ptr_q, b_rd_ptr_q, b_wr_ptr_d, b_rd_ptr_d;
    logic [2:0] b_count_q, b_count_d;
    logic [7:0] b_rd_data_q, b_rd_data_d;
    logic b_rd_data_valid_q, b_rd_data_valid_d;

    logic [10:0] stimuli [0:STIMULUS_COUNT-1];
    logic pre_tc_a_rd_en, pre_tc_b_wr_en, pre_tc_b_rd_en;
    logic [7:0] pre_tc_b_wr_data;
    logic pre_producer_ready, pre_transfer_invariant_ok;
    logic pre_a_wr_fire, pre_a_rd_fire, pre_b_wr_fire, pre_b_rd_fire;
    logic trace_state_valid = 1'b0;

    string stimulus_path;
    string trace_path;
    integer trace_file;
    integer index;
    longint unsigned half_clk = 0;

    two_fifo_system dut (.*);

    function automatic string logic_text(input logic value);
        if (value === 1'b0) return "0";
        if (value === 1'b1) return "1";
        return "x";
    endfunction

    function automatic string vector_text(input logic [7:0] value);
        if (^value === 1'bx) return "x";
        return $sformatf("%0d", value);
    endfunction

    function automatic string pointer_text(input logic [1:0] value);
        if (^value === 1'bx) return "x";
        return $sformatf("%0d", value);
    endfunction

    function automatic string count_text(input logic [2:0] value);
        if (^value === 1'bx) return "x";
        return $sformatf("%0d", value);
    endfunction

    function automatic string a_memory_text(input int slot);
        if (dut.fifo_a.mem_valid_q[slot] === 1'b1) return vector_text(dut.fifo_a.mem[slot]);
        return "x";
    endfunction

    function automatic string b_memory_text(input int slot);
        if (dut.fifo_b.mem_valid_q[slot] === 1'b1) return vector_text(dut.fifo_b.mem[slot]);
        return "x";
    endfunction

    task automatic log_common_low;
        $fwrite(trace_file,
            "%0d,%0d,LOW,%0d,%0d,%0d,%0d,%0d,%s,%s,%s,%s,%s,%s",
            half_clk, half_clk >> 1, clk, rst, producer_valid, producer_data,
            consumer_ready, logic_text(tc_a_rd_en), logic_text(tc_b_wr_en),
            vector_text(tc_b_wr_data), logic_text(tc_b_rd_en),
            logic_text(producer_ready), logic_text(transfer_invariant_ok));
    endtask

    task automatic log_common_high;
        $fwrite(trace_file,
            "%0d,%0d,HIGH,%0d,%0d,%0d,%0d,%0d,%s,%s,%s,%s,%s,%s",
            half_clk, half_clk >> 1, clk, rst, producer_valid, producer_data,
            consumer_ready, logic_text(pre_tc_a_rd_en), logic_text(pre_tc_b_wr_en),
            vector_text(pre_tc_b_wr_data), logic_text(pre_tc_b_rd_en),
            logic_text(pre_producer_ready), logic_text(pre_transfer_invariant_ok));
    endtask

    task automatic log_a_low;
        $fwrite(trace_file,
            ",%s,%s,%s,%s,%s,%s,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
            logic_text(a_full), logic_text(a_empty), logic_text(a_wr_ready),
            logic_text(a_rd_available), logic_text(a_wr_fire), logic_text(a_rd_fire),
            trace_state_valid, pointer_text(a_wr_ptr_q), pointer_text(a_rd_ptr_q),
            count_text(a_count_q), vector_text(a_rd_data_q), logic_text(a_rd_data_valid_q),
            logic_text(dut.fifo_a.mem_valid_q[0]), a_memory_text(0),
            logic_text(dut.fifo_a.mem_valid_q[1]), a_memory_text(1),
            logic_text(dut.fifo_a.mem_valid_q[2]), a_memory_text(2),
            logic_text(dut.fifo_a.mem_valid_q[3]), a_memory_text(3),
            pointer_text(a_wr_ptr_d), pointer_text(a_rd_ptr_d), count_text(a_count_d),
            vector_text(a_rd_data_d), logic_text(a_rd_data_valid_d));
    endtask

    task automatic log_b_low;
        $fwrite(trace_file,
            ",%s,%s,%s,%s,%s,%s,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
            logic_text(b_full), logic_text(b_empty), logic_text(b_wr_ready),
            logic_text(b_rd_available), logic_text(b_wr_fire), logic_text(b_rd_fire),
            trace_state_valid, pointer_text(b_wr_ptr_q), pointer_text(b_rd_ptr_q),
            count_text(b_count_q), vector_text(b_rd_data_q), logic_text(b_rd_data_valid_q),
            logic_text(dut.fifo_b.mem_valid_q[0]), b_memory_text(0),
            logic_text(dut.fifo_b.mem_valid_q[1]), b_memory_text(1),
            logic_text(dut.fifo_b.mem_valid_q[2]), b_memory_text(2),
            logic_text(dut.fifo_b.mem_valid_q[3]), b_memory_text(3),
            pointer_text(b_wr_ptr_d), pointer_text(b_rd_ptr_d), count_text(b_count_d),
            vector_text(b_rd_data_d), logic_text(b_rd_data_valid_d));
    endtask

    task automatic log_a_high;
        $fwrite(trace_file,
            ",%s,%s,%s,%s,%s,%s,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,,,,,",
            logic_text(a_full), logic_text(a_empty), logic_text(a_wr_ready),
            logic_text(a_rd_available), logic_text(pre_a_wr_fire), logic_text(pre_a_rd_fire),
            trace_state_valid, pointer_text(a_wr_ptr_q), pointer_text(a_rd_ptr_q),
            count_text(a_count_q), vector_text(a_rd_data_q), logic_text(a_rd_data_valid_q),
            logic_text(dut.fifo_a.mem_valid_q[0]), a_memory_text(0),
            logic_text(dut.fifo_a.mem_valid_q[1]), a_memory_text(1),
            logic_text(dut.fifo_a.mem_valid_q[2]), a_memory_text(2),
            logic_text(dut.fifo_a.mem_valid_q[3]), a_memory_text(3));
    endtask

    task automatic log_b_high;
        $fwrite(trace_file,
            ",%s,%s,%s,%s,%s,%s,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,,,,,",
            logic_text(b_full), logic_text(b_empty), logic_text(b_wr_ready),
            logic_text(b_rd_available), logic_text(pre_b_wr_fire), logic_text(pre_b_rd_fire),
            trace_state_valid, pointer_text(b_wr_ptr_q), pointer_text(b_rd_ptr_q),
            count_text(b_count_q), vector_text(b_rd_data_q), logic_text(b_rd_data_valid_q),
            logic_text(dut.fifo_b.mem_valid_q[0]), b_memory_text(0),
            logic_text(dut.fifo_b.mem_valid_q[1]), b_memory_text(1),
            logic_text(dut.fifo_b.mem_valid_q[2]), b_memory_text(2),
            logic_text(dut.fifo_b.mem_valid_q[3]), b_memory_text(3));
    endtask

    initial begin
        if (!$value$plusargs("STIMULUS=%s", stimulus_path)) $fatal(1, "missing +STIMULUS=<path>");
        if (!$value$plusargs("TRACE=%s", trace_path)) $fatal(1, "missing +TRACE=<path>");

        $readmemh(stimulus_path, stimuli);
        trace_file = $fopen(trace_path, "w");
        if (trace_file == 0) $fatal(1, "cannot open trace file %s", trace_path);
        $fwrite(trace_file,
            "half_clk,cycle,phase,clk_level,rst,producer_valid,producer_data,consumer_ready,tc_a_rd_en,tc_b_wr_en,tc_b_wr_data,tc_b_rd_en,producer_ready,transfer_invariant_ok");
        $fwrite(trace_file,
            ",a_full,a_empty,a_wr_ready,a_rd_available,a_wr_fire,a_rd_fire,a_state_valid,a_wr_ptr_q,a_rd_ptr_q,a_count_q,a_rd_data_q,a_rd_data_valid_q,a_mem0_valid,a_mem0,a_mem1_valid,a_mem1,a_mem2_valid,a_mem2,a_mem3_valid,a_mem3,a_wr_ptr_d,a_rd_ptr_d,a_count_d,a_rd_data_d,a_rd_data_valid_d");
        $fdisplay(trace_file,
            ",b_full,b_empty,b_wr_ready,b_rd_available,b_wr_fire,b_rd_fire,b_state_valid,b_wr_ptr_q,b_rd_ptr_q,b_count_q,b_rd_data_q,b_rd_data_valid_q,b_mem0_valid,b_mem0,b_mem1_valid,b_mem1,b_mem2_valid,b_mem2,b_mem3_valid,b_mem3,b_wr_ptr_d,b_rd_ptr_d,b_count_d,b_rd_data_d,b_rd_data_valid_d");

        for (index = 0; index < STIMULUS_COUNT; index = index + 1) begin
            {rst, producer_valid, producer_data, consumer_ready} = stimuli[index];
            #1ps;
            log_common_low();
            log_a_low();
            log_b_low();
            $fwrite(trace_file, "\n");

            pre_tc_a_rd_en = tc_a_rd_en;
            pre_tc_b_wr_en = tc_b_wr_en;
            pre_tc_b_wr_data = tc_b_wr_data;
            pre_tc_b_rd_en = tc_b_rd_en;
            pre_producer_ready = producer_ready;
            pre_transfer_invariant_ok = transfer_invariant_ok;
            pre_a_wr_fire = a_wr_fire;
            pre_a_rd_fire = a_rd_fire;
            pre_b_wr_fire = b_wr_fire;
            pre_b_rd_fire = b_rd_fire;

            if (trace_state_valid && !rst && transfer_invariant_ok !== 1'b1) begin
                $fatal(1, "transfer invariant failed at cycle %0d", index);
            end

            #(HALF_PERIOD - 1ps);
            clk = 1'b1;
            half_clk = half_clk + 1;
            #1ps;
            if (rst) trace_state_valid = 1'b1;
            log_common_high();
            log_a_high();
            log_b_high();
            $fwrite(trace_file, "\n");

            #(HALF_PERIOD - 1ps);
            clk = 1'b0;
            half_clk = half_clk + 1;
        end

        $fclose(trace_file);
        $display("RTL two-FIFO trace complete: %0d cycles, %0d samples",
                 STIMULUS_COUNT, STIMULUS_COUNT * 2);
        $finish;
    end
endmodule
