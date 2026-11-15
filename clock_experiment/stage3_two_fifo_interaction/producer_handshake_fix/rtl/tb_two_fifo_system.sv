module tb_two_fifo_system;
    timeunit 1ns;
    timeprecision 1ps;

    localparam int TRANSACTION_COUNT = 74;
    localparam int CONTROL_COUNT = 264;
    localparam time HALF_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;
    logic producer_valid;
    logic [7:0] producer_data;
    logic consumer_ready;
    logic tc_a_rd_en, tc_b_wr_en, tc_b_rd_en;
    logic [7:0] tc_b_wr_data;
    logic producer_ready, transfer_invariant_ok;
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

    logic [7:0] producer_transactions [0:TRANSACTION_COUNT-1];
    logic [1:0] consumer_controls [0:CONTROL_COUNT-1];
    logic [7:0] pending_transfer_data [0:TRANSACTION_COUNT-1];
    integer pending_transfer_cycle [0:TRANSACTION_COUNT-1];

    logic pre_tc_a_rd_en, pre_tc_b_wr_en, pre_tc_b_rd_en;
    logic [7:0] pre_tc_b_wr_data;
    logic pre_producer_ready, pre_transfer_invariant_ok;
    logic pre_producer_fire, pre_producer_stall;
    logic pre_a_wr_fire, pre_a_rd_fire, pre_b_wr_fire, pre_b_rd_fire;
    logic trace_state_valid = 1'b0;

    logic previous_cycle_valid = 1'b0;
    logic previous_stall = 1'b0;
    logic previous_fire = 1'b0;
    logic [7:0] previous_data = 8'd0;
    integer previous_transaction_index = 0;

    string transactions_path, control_path, trace_path;
    integer trace_file;
    integer index;
    integer producer_transaction_index = 0;
    integer producer_accepted_count = 0;
    integer consumer_received_count = 0;
    integer pending_transfer_head = 0;
    integer pending_transfer_tail = 0;
    integer producer_protocol_violations = 0;
    integer transfer_response_failures = 0;
    integer transfer_latency_failures = 0;
    integer consumer_order_failures = 0;
    integer producer_valid_cycles = 0;
    integer producer_ready_cycles = 0;
    integer producer_fire_cycles = 0;
    integer producer_stall_cycles = 0;
    integer consecutive_stalls = 0;
    integer max_consecutive_stalls = 0;
    longint unsigned half_clk = 0;

    wire producer_fire = producer_valid && producer_ready && !rst;
    wire producer_stall = producer_valid && !producer_ready && !rst;

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
            "%0d,%0d,LOW,%0d,%0d,%0d,%0d,%0d,%s,%s,%s,%0d,%0d,%0d,%s,%s,%s,%s,%s",
            half_clk, half_clk >> 1, clk, rst, producer_transaction_index,
            producer_valid, producer_data, logic_text(producer_ready), logic_text(producer_fire),
            logic_text(producer_stall), TRANSACTION_COUNT - producer_accepted_count,
            producer_accepted_count, consumer_ready, logic_text(tc_a_rd_en), logic_text(tc_b_wr_en),
            vector_text(tc_b_wr_data), logic_text(tc_b_rd_en), logic_text(transfer_invariant_ok));
    endtask

    task automatic log_common_high;
        integer accepted_after_edge;
        begin
            accepted_after_edge = producer_accepted_count + (pre_producer_fire ? 1 : 0);
            $fwrite(trace_file,
                "%0d,%0d,HIGH,%0d,%0d,%0d,%0d,%0d,%s,%s,%s,%0d,%0d,%0d,%s,%s,%s,%s,%s",
                half_clk, half_clk >> 1, clk, rst, producer_transaction_index,
                producer_valid, producer_data, logic_text(pre_producer_ready), logic_text(pre_producer_fire),
                logic_text(pre_producer_stall), TRANSACTION_COUNT - accepted_after_edge,
                accepted_after_edge, consumer_ready, logic_text(pre_tc_a_rd_en), logic_text(pre_tc_b_wr_en),
                vector_text(pre_tc_b_wr_data), logic_text(pre_tc_b_rd_en), logic_text(pre_transfer_invariant_ok));
        end
    endtask

    task automatic log_a_low;
        $fwrite(trace_file,
            ",%s,%s,%s,%s,%s,%s,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
            logic_text(a_full), logic_text(a_empty), logic_text(a_wr_ready), logic_text(a_rd_available),
            logic_text(a_wr_fire), logic_text(a_rd_fire), trace_state_valid,
            pointer_text(a_wr_ptr_q), pointer_text(a_rd_ptr_q), count_text(a_count_q),
            vector_text(a_rd_data_q), logic_text(a_rd_data_valid_q),
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
            logic_text(b_full), logic_text(b_empty), logic_text(b_wr_ready), logic_text(b_rd_available),
            logic_text(b_wr_fire), logic_text(b_rd_fire), trace_state_valid,
            pointer_text(b_wr_ptr_q), pointer_text(b_rd_ptr_q), count_text(b_count_q),
            vector_text(b_rd_data_q), logic_text(b_rd_data_valid_q),
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
            logic_text(a_full), logic_text(a_empty), logic_text(a_wr_ready), logic_text(a_rd_available),
            logic_text(pre_a_wr_fire), logic_text(pre_a_rd_fire), trace_state_valid,
            pointer_text(a_wr_ptr_q), pointer_text(a_rd_ptr_q), count_text(a_count_q),
            vector_text(a_rd_data_q), logic_text(a_rd_data_valid_q),
            logic_text(dut.fifo_a.mem_valid_q[0]), a_memory_text(0),
            logic_text(dut.fifo_a.mem_valid_q[1]), a_memory_text(1),
            logic_text(dut.fifo_a.mem_valid_q[2]), a_memory_text(2),
            logic_text(dut.fifo_a.mem_valid_q[3]), a_memory_text(3));
    endtask

    task automatic log_b_high;
        $fwrite(trace_file,
            ",%s,%s,%s,%s,%s,%s,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,,,,,",
            logic_text(b_full), logic_text(b_empty), logic_text(b_wr_ready), logic_text(b_rd_available),
            logic_text(pre_b_wr_fire), logic_text(pre_b_rd_fire), trace_state_valid,
            pointer_text(b_wr_ptr_q), pointer_text(b_rd_ptr_q), count_text(b_count_q),
            vector_text(b_rd_data_q), logic_text(b_rd_data_valid_q),
            logic_text(dut.fifo_b.mem_valid_q[0]), b_memory_text(0),
            logic_text(dut.fifo_b.mem_valid_q[1]), b_memory_text(1),
            logic_text(dut.fifo_b.mem_valid_q[2]), b_memory_text(2),
            logic_text(dut.fifo_b.mem_valid_q[3]), b_memory_text(3));
    endtask

    initial begin
        if (!$value$plusargs("TRANSACTIONS=%s", transactions_path)) $fatal(1, "missing +TRANSACTIONS=<path>");
        if (!$value$plusargs("CONTROL=%s", control_path)) $fatal(1, "missing +CONTROL=<path>");
        if (!$value$plusargs("TRACE=%s", trace_path)) $fatal(1, "missing +TRACE=<path>");
        $readmemh(transactions_path, producer_transactions);
        $readmemh(control_path, consumer_controls);
        trace_file = $fopen(trace_path, "w");
        if (trace_file == 0) $fatal(1, "cannot open trace file %s", trace_path);
        $fwrite(trace_file,
            "half_clk,cycle,phase,clk_level,rst,producer_transaction_index,producer_valid,producer_data,producer_ready,producer_fire,producer_stall,producer_pending_count,producer_accepted_count,consumer_ready,tc_a_rd_en,tc_b_wr_en,tc_b_wr_data,tc_b_rd_en,transfer_invariant_ok");
        $fwrite(trace_file,
            ",a_full,a_empty,a_wr_ready,a_rd_available,a_wr_fire,a_rd_fire,a_state_valid,a_wr_ptr_q,a_rd_ptr_q,a_count_q,a_rd_data_q,a_rd_data_valid_q,a_mem0_valid,a_mem0,a_mem1_valid,a_mem1,a_mem2_valid,a_mem2,a_mem3_valid,a_mem3,a_wr_ptr_d,a_rd_ptr_d,a_count_d,a_rd_data_d,a_rd_data_valid_d");
        $fdisplay(trace_file,
            ",b_full,b_empty,b_wr_ready,b_rd_available,b_wr_fire,b_rd_fire,b_state_valid,b_wr_ptr_q,b_rd_ptr_q,b_count_q,b_rd_data_q,b_rd_data_valid_q,b_mem0_valid,b_mem0,b_mem1_valid,b_mem1,b_mem2_valid,b_mem2,b_mem3_valid,b_mem3,b_wr_ptr_d,b_rd_ptr_d,b_count_d,b_rd_data_d,b_rd_data_valid_d");

        for (index = 0; index < CONTROL_COUNT; index = index + 1) begin
            {rst, consumer_ready} = consumer_controls[index];
            if (rst || producer_transaction_index >= TRANSACTION_COUNT) begin
                producer_valid = 1'b0;
                producer_data = 8'd0;
            end else begin
                producer_valid = 1'b1;
                producer_data = producer_transactions[producer_transaction_index];
            end
            #1ps;

            if (previous_cycle_valid) begin
                if (producer_transaction_index != previous_transaction_index + (previous_fire ? 1 : 0))
                    producer_protocol_violations = producer_protocol_violations + 1;
                if (previous_stall && !rst &&
                    (!producer_valid || producer_data !== previous_data ||
                     producer_transaction_index != previous_transaction_index))
                    producer_protocol_violations = producer_protocol_violations + 1;
            end
            if (producer_transaction_index != producer_accepted_count)
                producer_protocol_violations = producer_protocol_violations + 1;

            pre_tc_a_rd_en = tc_a_rd_en; pre_tc_b_wr_en = tc_b_wr_en;
            pre_tc_b_wr_data = tc_b_wr_data; pre_tc_b_rd_en = tc_b_rd_en;
            pre_producer_ready = producer_ready; pre_transfer_invariant_ok = transfer_invariant_ok;
            pre_producer_fire = producer_fire; pre_producer_stall = producer_stall;
            pre_a_wr_fire = a_wr_fire; pre_a_rd_fire = a_rd_fire;
            pre_b_wr_fire = b_wr_fire; pre_b_rd_fire = b_rd_fire;

            if (producer_valid) producer_valid_cycles = producer_valid_cycles + 1;
            if (!rst && producer_ready) producer_ready_cycles = producer_ready_cycles + 1;
            if (pre_producer_fire) producer_fire_cycles = producer_fire_cycles + 1;
            if (pre_producer_stall) begin
                producer_stall_cycles = producer_stall_cycles + 1;
                consecutive_stalls = consecutive_stalls + 1;
                if (consecutive_stalls > max_consecutive_stalls) max_consecutive_stalls = consecutive_stalls;
            end else consecutive_stalls = 0;

            if (pre_producer_fire !== pre_a_wr_fire) producer_protocol_violations = producer_protocol_violations + 1;
            if (rst && pre_producer_fire) producer_protocol_violations = producer_protocol_violations + 1;
            if (pre_producer_fire && producer_data !== producer_transactions[producer_accepted_count])
                producer_protocol_violations = producer_protocol_violations + 1;
            if (trace_state_valid && !rst && transfer_invariant_ok !== 1'b1)
                $fatal(1, "transfer invariant failed at cycle %0d", index);

            if (pre_b_wr_fire) begin
                if (pending_transfer_head >= pending_transfer_tail) transfer_response_failures = transfer_response_failures + 1;
                else begin
                    if (pre_tc_b_wr_data !== pending_transfer_data[pending_transfer_head])
                        transfer_response_failures = transfer_response_failures + 1;
                    if (index != pending_transfer_cycle[pending_transfer_head] + 1)
                        transfer_latency_failures = transfer_latency_failures + 1;
                    pending_transfer_head = pending_transfer_head + 1;
                end
            end
            if (pre_a_rd_fire) begin
                if (dut.fifo_a.mem_valid_q[a_rd_ptr_q] !== 1'b1)
                    transfer_response_failures = transfer_response_failures + 1;
                else begin
                    pending_transfer_data[pending_transfer_tail] = dut.fifo_a.mem[a_rd_ptr_q];
                    pending_transfer_cycle[pending_transfer_tail] = index;
                    pending_transfer_tail = pending_transfer_tail + 1;
                end
            end

            log_common_low(); log_a_low(); log_b_low(); $fwrite(trace_file, "\n");
            #(HALF_PERIOD - 1ps);
            clk = 1'b1;
            half_clk = half_clk + 1;
            #1ps;
            if (rst) trace_state_valid = 1'b1;
            if (pre_b_rd_fire) begin
                if (b_rd_data_valid_q !== 1'b1 || consumer_received_count >= TRANSACTION_COUNT ||
                    b_rd_data_q !== producer_transactions[consumer_received_count])
                    consumer_order_failures = consumer_order_failures + 1;
            end
            log_common_high(); log_a_high(); log_b_high(); $fwrite(trace_file, "\n");

            #(HALF_PERIOD - 1ps);
            clk = 1'b0;
            half_clk = half_clk + 1;
            previous_cycle_valid = 1'b1;
            previous_stall = pre_producer_stall;
            previous_fire = pre_producer_fire;
            previous_data = producer_data;
            previous_transaction_index = producer_transaction_index;
            if (pre_b_rd_fire) consumer_received_count = consumer_received_count + 1;
            if (pre_producer_fire) begin
                producer_transaction_index = producer_transaction_index + 1;
                producer_accepted_count = producer_accepted_count + 1;
            end
        end

        $fclose(trace_file);
        $display("producer_transaction_count=%0d", TRANSACTION_COUNT);
        $display("producer_valid_cycles=%0d", producer_valid_cycles);
        $display("producer_ready_cycles=%0d", producer_ready_cycles);
        $display("producer_fire_cycles=%0d", producer_fire_cycles);
        $display("producer_stall_cycles=%0d", producer_stall_cycles);
        $display("max_consecutive_producer_stall_cycles=%0d", max_consecutive_stalls);
        $display("producer_accepted_count=%0d", producer_accepted_count);
        $display("producer_pending_count=%0d", TRANSACTION_COUNT - producer_accepted_count);
        $display("producer_protocol_violations=%0d", producer_protocol_violations);
        $display("transfer_response_failures=%0d", transfer_response_failures);
        $display("transfer_latency_failures=%0d", transfer_latency_failures);
        $display("consumer_received_count=%0d", consumer_received_count);
        $display("consumer_order_failures=%0d", consumer_order_failures);
        $display("pending_transfer_responses=%0d", pending_transfer_tail - pending_transfer_head);
        $display("final_a_count=%0d", a_count_q);
        $display("final_b_count=%0d", b_count_q);
        if (producer_protocol_violations != 0 || transfer_response_failures != 0 ||
            transfer_latency_failures != 0 || consumer_order_failures != 0 ||
            producer_accepted_count != TRANSACTION_COUNT || consumer_received_count != TRANSACTION_COUNT ||
            pending_transfer_head != pending_transfer_tail || a_count_q != 0 || b_count_q != 0 ||
            a_rd_data_valid_q != 0)
            $fatal(1, "RTL producer-handshake scoreboard failed");
        $display("RTL two-FIFO producer-handshake trace complete: %0d cycles, %0d samples",
                 CONTROL_COUNT, CONTROL_COUNT * 2);
        $finish;
    end
endmodule
