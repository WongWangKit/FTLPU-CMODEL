module tb_two_fifo_system;
    timeunit 1ns;
    timeprecision 1ps;
    localparam int TRANSACTION_COUNT = 74;
    localparam int CONTROL_COUNT = 264;
    localparam time HALF_PERIOD = 5ns;

    logic clk = 0, rst, producer_valid, consumer_ready;
    logic [7:0] producer_data;
    logic tc_a_rd_en, tc_b_wr_en, tc_b_rd_en, producer_ready, transfer_invariant_ok;
    logic [7:0] tc_b_wr_data;
    logic mac_valid;
    logic [15:0] mac_mul0, mac_add0, mac_mul1, mac_add1, mac_mul2, mac_result16;
    logic [7:0] mac_result8;
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
    logic [7:0] pending_data [0:TRANSACTION_COUNT-1];
    integer pending_cycle [0:TRANSACTION_COUNT-1];
    logic pre_tc_a_rd_en, pre_tc_b_wr_en, pre_tc_b_rd_en, pre_ready, pre_invariant;
    logic [7:0] pre_tc_b_wr_data;
    logic pre_fire, pre_stall, pre_a_wr_fire, pre_a_rd_fire, pre_b_wr_fire, pre_b_rd_fire;
    logic pre_mac_valid;
    logic [7:0] pre_mac_input, pre_mac_result8;
    logic [15:0] pre_mul0, pre_add0, pre_mul1, pre_add1, pre_mul2, pre_result16;
    logic [1:0] prep_a_wp, prep_a_rp, prep_b_wp, prep_b_rp;
    logic [2:0] prep_a_count, prep_b_count;
    logic [7:0] prep_a_data, prep_b_data;
    logic prep_a_valid, prep_b_valid;

    logic previous_cycle_valid = 0, previous_stall = 0, previous_fire = 0;
    logic [7:0] previous_data = 0;
    integer previous_index = 0;
    logic trace_state_valid = 0;
    longint unsigned phase_count = 0;
    integer producer_index = 0, accepted_count = 0, consumer_count = 0;
    integer pending_head = 0, pending_tail = 0;
    integer protocol_failures = 0, transfer_failures = 0, latency_failures = 0, consumer_failures = 0;
    integer valid_cycles = 0, ready_cycles = 0, fire_cycles = 0, stall_cycles = 0;
    integer consecutive_stalls = 0, max_stalls = 0;
    integer index, mac_index, trace_file, mac_file;
    string transactions_path, control_path, trace_path, mac_path;

    logic [7:0] ex_x;
    logic [15:0] ex_mul0, ex_add0, ex_mul1, ex_add1, ex_mul2, ex_result16;
    logic [7:0] ex_result8;
    wire producer_fire = producer_valid && producer_ready && !rst;
    wire producer_stall = producer_valid && !producer_ready && !rst;

    two_fifo_system dut (.*);
    mac_chain exhaustive_mac(.x(ex_x), .mul0(ex_mul0), .add0(ex_add0), .mul1(ex_mul1),
        .add1(ex_add1), .mul2(ex_mul2), .result16(ex_result16), .result8(ex_result8));

    function automatic [7:0] mac_formula(input logic [7:0] x);
        integer v;
        begin v = (((x * 3 + 17) * 5 + 29) * 7 + 43); return v[7:0]; end
    endfunction
    function automatic string lt(input logic v);
        if (v === 0) return "0"; if (v === 1) return "1"; return "x";
    endfunction
    function automatic string bt(input logic [7:0] v);
        if (^v === 1'bx) return "x"; return $sformatf("%0d", v);
    endfunction
    function automatic string wt(input logic [15:0] v);
        if (^v === 1'bx) return "x"; return $sformatf("%0d", v);
    endfunction
    function automatic string pt(input logic [1:0] v);
        if (^v === 1'bx) return "x"; return $sformatf("%0d", v);
    endfunction
    function automatic string ct(input logic [2:0] v);
        if (^v === 1'bx) return "x"; return $sformatf("%0d", v);
    endfunction
    function automatic string am(input int s);
        if (dut.fifo_a.mem_valid_q[s] === 1) return bt(dut.fifo_a.mem[s]); return "x";
    endfunction
    function automatic string bm(input int s);
        if (dut.fifo_b.mem_valid_q[s] === 1) return bt(dut.fifo_b.mem[s]); return "x";
    endfunction

    task automatic log_common(input int phase_id, input string phase_text,
                              input logic use_pre, input integer accepted);
        logic mv;
        logic [7:0] mi, mr8, post8;
        logic [15:0] m0, a0, m1, a1, m2, r16;
        begin
            if (use_pre) begin
                mv=pre_mac_valid; mi=pre_mac_input; m0=pre_mul0; a0=pre_add0; m1=pre_mul1;
                a1=pre_add1; m2=pre_mul2; r16=pre_result16; mr8=pre_mac_result8; post8=pre_mac_result8;
            end else begin
                mv=mac_valid; mi=a_rd_data_q; m0=mac_mul0; a0=mac_add0; m1=mac_mul1;
                a1=mac_add1; m2=mac_mul2; r16=mac_result16; mr8=mac_result8; post8=mac_result8;
            end
            $fwrite(trace_file,
                "%0d,%0d,%0d,%0d,%s,%0d,%0d,%0d,%0d,%0d,%s,%s,%s,%0d,%0d,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
                phase_count, phase_count, phase_count>>2, phase_id, phase_text, clk, rst,
                producer_index, producer_valid, producer_data, lt(pre_ready), lt(pre_fire), lt(pre_stall),
                TRANSACTION_COUNT-accepted, accepted, consumer_ready, lt(pre_tc_a_rd_en), lt(pre_tc_b_wr_en),
                bt(pre_tc_b_wr_data), lt(pre_tc_b_rd_en), lt(pre_invariant), lt(mv), bt(mi), wt(m0), wt(a0),
                wt(m1), wt(a1), wt(m2), wt(r16), bt(mr8), bt(pre_mac_result8), bt(post8));
        end
    endtask

    task automatic log_a;
        $fwrite(trace_file,
            ",%s,%s,%s,%s,%s,%s,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
            lt(a_full),lt(a_empty),lt(a_wr_ready),lt(a_rd_available),lt(pre_a_wr_fire),lt(pre_a_rd_fire),
            trace_state_valid,pt(a_wr_ptr_q),pt(a_rd_ptr_q),ct(a_count_q),bt(a_rd_data_q),lt(a_rd_data_valid_q),
            lt(dut.fifo_a.mem_valid_q[0]),am(0),lt(dut.fifo_a.mem_valid_q[1]),am(1),
            lt(dut.fifo_a.mem_valid_q[2]),am(2),lt(dut.fifo_a.mem_valid_q[3]),am(3),
            pt(prep_a_wp),pt(prep_a_rp),ct(prep_a_count),bt(prep_a_data),lt(prep_a_valid));
    endtask
    task automatic log_b;
        $fwrite(trace_file,
            ",%s,%s,%s,%s,%s,%s,%0d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
            lt(b_full),lt(b_empty),lt(b_wr_ready),lt(b_rd_available),lt(pre_b_wr_fire),lt(pre_b_rd_fire),
            trace_state_valid,pt(b_wr_ptr_q),pt(b_rd_ptr_q),ct(b_count_q),bt(b_rd_data_q),lt(b_rd_data_valid_q),
            lt(dut.fifo_b.mem_valid_q[0]),bm(0),lt(dut.fifo_b.mem_valid_q[1]),bm(1),
            lt(dut.fifo_b.mem_valid_q[2]),bm(2),lt(dut.fifo_b.mem_valid_q[3]),bm(3),
            pt(prep_b_wp),pt(prep_b_rp),ct(prep_b_count),bt(prep_b_data),lt(prep_b_valid));
    endtask
    task automatic log_row(input int p, input string name, input logic use_pre, input integer accepted);
        begin log_common(p,name,use_pre,accepted); log_a(); log_b(); $fwrite(trace_file,"\n"); end
    endtask

    initial begin
        if (!$value$plusargs("TRANSACTIONS=%s",transactions_path)) $fatal(1,"missing TRANSACTIONS");
        if (!$value$plusargs("CONTROL=%s",control_path)) $fatal(1,"missing CONTROL");
        if (!$value$plusargs("TRACE=%s",trace_path)) $fatal(1,"missing TRACE");
        if (!$value$plusargs("MAC_TRACE=%s",mac_path)) $fatal(1,"missing MAC_TRACE");
        $readmemh(transactions_path,producer_transactions);
        $readmemh(control_path,consumer_controls);
        mac_file=$fopen(mac_path,"w");
        $fdisplay(mac_file,"input,mul0,add0,mul1,add1,mul2,result16,result8");
        for(mac_index=0;mac_index<256;mac_index++) begin
            ex_x=mac_index[7:0]; #1ps;
            $fdisplay(mac_file,"%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",mac_index,
                ex_mul0,ex_add0,ex_mul1,ex_add1,ex_mul2,ex_result16,ex_result8);
        end
        $fclose(mac_file);

        trace_file=$fopen(trace_path,"w");
        $fwrite(trace_file,"phase_count,half_clk,cycle,phase,phase_name,clk_level,rst,producer_transaction_index,producer_valid,producer_data,producer_ready,producer_fire,producer_stall,producer_pending_count,producer_accepted_count,consumer_ready,tc_a_rd_en,tc_b_wr_en,tc_b_wr_data,tc_b_rd_en,transfer_invariant_ok,mac_valid,mac_input,mac_mul0,mac_add0,mac_mul1,mac_add1,mac_mul2,mac_result16,mac_result8,mac_pre_edge_result,mac_post_edge_result");
        $fwrite(trace_file,",a_full,a_empty,a_wr_ready,a_rd_available,a_wr_fire,a_rd_fire,a_state_valid,a_wr_ptr_q,a_rd_ptr_q,a_count_q,a_rd_data_q,a_rd_data_valid_q,a_mem0_valid,a_mem0,a_mem1_valid,a_mem1,a_mem2_valid,a_mem2,a_mem3_valid,a_mem3,a_wr_ptr_d,a_rd_ptr_d,a_count_d,a_rd_data_d,a_rd_data_valid_d");
        $fdisplay(trace_file,",b_full,b_empty,b_wr_ready,b_rd_available,b_wr_fire,b_rd_fire,b_state_valid,b_wr_ptr_q,b_rd_ptr_q,b_count_q,b_rd_data_q,b_rd_data_valid_q,b_mem0_valid,b_mem0,b_mem1_valid,b_mem1,b_mem2_valid,b_mem2,b_mem3_valid,b_mem3,b_wr_ptr_d,b_rd_ptr_d,b_count_d,b_rd_data_d,b_rd_data_valid_d");

        for(index=0;index<CONTROL_COUNT;index++) begin
            {rst,consumer_ready}=consumer_controls[index];
            if(rst||producer_index>=TRANSACTION_COUNT) begin producer_valid=0; producer_data=0; end
            else begin producer_valid=1; producer_data=producer_transactions[producer_index]; end
            #1ps;
            if(previous_cycle_valid) begin
                if(producer_index!=previous_index+(previous_fire?1:0)) protocol_failures++;
                if(previous_stall&&!rst&&(!producer_valid||producer_data!==previous_data||producer_index!=previous_index)) protocol_failures++;
            end
            if(producer_index!=accepted_count) protocol_failures++;

            pre_tc_a_rd_en=tc_a_rd_en; pre_tc_b_wr_en=tc_b_wr_en; pre_tc_b_wr_data=tc_b_wr_data;
            pre_tc_b_rd_en=tc_b_rd_en; pre_ready=producer_ready; pre_invariant=transfer_invariant_ok;
            pre_fire=producer_fire; pre_stall=producer_stall; pre_a_wr_fire=a_wr_fire; pre_a_rd_fire=a_rd_fire;
            pre_b_wr_fire=b_wr_fire; pre_b_rd_fire=b_rd_fire;
            pre_mac_valid=mac_valid; pre_mac_input=a_rd_data_q; pre_mul0=mac_mul0; pre_add0=mac_add0;
            pre_mul1=mac_mul1; pre_add1=mac_add1; pre_mul2=mac_mul2; pre_result16=mac_result16; pre_mac_result8=mac_result8;
            prep_a_wp=a_wr_ptr_d; prep_a_rp=a_rd_ptr_d; prep_a_count=a_count_d; prep_a_data=a_rd_data_d; prep_a_valid=a_rd_data_valid_d;
            prep_b_wp=b_wr_ptr_d; prep_b_rp=b_rd_ptr_d; prep_b_count=b_count_d; prep_b_data=b_rd_data_d; prep_b_valid=b_rd_data_valid_d;

            if(producer_valid) valid_cycles++; if(!rst&&producer_ready) ready_cycles++;
            if(pre_fire) fire_cycles++;
            if(pre_stall) begin stall_cycles++; consecutive_stalls++; if(consecutive_stalls>max_stalls) max_stalls=consecutive_stalls; end
            else consecutive_stalls=0;
            if(pre_fire!==pre_a_wr_fire) protocol_failures++;
            if(pre_fire&&producer_data!==producer_transactions[accepted_count]) protocol_failures++;
            if(trace_state_valid&&!rst&&transfer_invariant_ok!==1) $fatal(1,"transfer invariant");

            if(pre_b_wr_fire) begin
                if(pending_head>=pending_tail) transfer_failures++;
                else begin
                    if(pre_tc_b_wr_data!==pending_data[pending_head]) transfer_failures++;
                    if(index!=pending_cycle[pending_head]+1) latency_failures++;
                    pending_head++;
                end
            end
            if(pre_a_rd_fire) begin
                if(dut.fifo_a.mem_valid_q[a_rd_ptr_q]!==1) transfer_failures++;
                else begin pending_data[pending_tail]=mac_formula(dut.fifo_a.mem[a_rd_ptr_q]);
                    pending_cycle[pending_tail]=index; pending_tail++; end
            end

            log_row(0,"LOW_EVALUATE",1,accepted_count);
            #(HALF_PERIOD-1ps); clk=1; phase_count++; #1ps;
            if(rst) trace_state_valid=1;
            if(pre_b_rd_fire) begin
                if(b_rd_data_valid_q!==1||consumer_count>=TRANSACTION_COUNT||
                    b_rd_data_q!==mac_formula(producer_transactions[consumer_count])) consumer_failures++;
            end
            log_row(1,"POSEDGE_COMMIT",1,accepted_count+(pre_fire?1:0));
            phase_count++; #1ps;
            log_row(2,"POST_EDGE_SETTLE",0,accepted_count+(pre_fire?1:0));
            #(HALF_PERIOD-2ps); clk=0; phase_count++; #1ps;
            log_row(3,"NEGEDGE",0,accepted_count+(pre_fire?1:0));

            previous_cycle_valid=1; previous_stall=pre_stall; previous_fire=pre_fire;
            previous_data=producer_data; previous_index=producer_index;
            if(pre_b_rd_fire) consumer_count++;
            if(pre_fire) begin producer_index++; accepted_count++; end
            phase_count++;
        end

        $fclose(trace_file);
        $display("producer_transaction_count=%0d",TRANSACTION_COUNT);
        $display("producer_valid_cycles=%0d",valid_cycles); $display("producer_ready_cycles=%0d",ready_cycles);
        $display("producer_fire_cycles=%0d",fire_cycles); $display("producer_stall_cycles=%0d",stall_cycles);
        $display("max_consecutive_producer_stall_cycles=%0d",max_stalls);
        $display("producer_accepted_count=%0d",accepted_count); $display("producer_pending_count=%0d",TRANSACTION_COUNT-accepted_count);
        $display("producer_protocol_failures=%0d",protocol_failures); $display("transfer_response_failures=%0d",transfer_failures);
        $display("transfer_latency_failures=%0d",latency_failures); $display("consumer_received_count=%0d",consumer_count);
        $display("consumer_order_failures=%0d",consumer_failures); $display("pending_transfer_responses=%0d",pending_tail-pending_head);
        $display("final_a_count=%0d",a_count_q); $display("final_b_count=%0d",b_count_q);
        if(protocol_failures||transfer_failures||latency_failures||consumer_failures||accepted_count!=TRANSACTION_COUNT||
           consumer_count!=TRANSACTION_COUNT||pending_head!=pending_tail||a_count_q!=0||b_count_q!=0||
           a_rd_data_valid_q!=0||b_rd_data_valid_q!=0) $fatal(1,"Stage 4 RTL scoreboard failed");
        $display("Stage 4 RTL complete: %0d cycles, %0d phase samples",CONTROL_COUNT,CONTROL_COUNT*4);
        $finish;
    end
endmodule
