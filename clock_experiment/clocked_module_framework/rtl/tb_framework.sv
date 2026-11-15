`timescale 1ns/1ps

module tb_framework;
    localparam int CYCLES = 30;
    logic clk, rst, reg_en, fifo_wr_en, fifo_rd_en;
    logic [7:0] reg_data, fifo_wr_data, sram_wdata;
    logic [1:0] sram_op, sram_addr;
    logic [31:0] stimulus [0:CYCLES-1];
    string stimulus_path, trace_path;
    integer trace_fd;
    integer cycle, i;
    bit state_valid = 0;

    logic pre_wr_fire, pre_rd_fire;
    logic [15:0] pre_mul0, pre_add0, pre_mul1, pre_add1, pre_mul2, pre_result16;
    logic [7:0] pre_result8;
    logic [7:0] n_source, n_destination, n_bank0, n_bank1;
    logic [1:0] n_fifo_wr_ptr, n_fifo_rd_ptr;
    logic [2:0] n_fifo_count;
    logic [7:0] n_fifo_rd_data;
    logic n_fifo_rd_data_valid;
    logic [7:0] n_fifo_memory [0:3];
    logic [3:0] n_fifo_valid;
    logic [7:0] n_sram_rd_data;
    logic n_sram_rd_valid, n_sram_rd_initialized;

    framework_top dut (.*);

    always @(posedge clk) if (rst) state_valid <= 1'b1;

    function automatic string fmt(input logic valid, input logic [31:0] value);
        if (valid === 1'b1) fmt = $sformatf("%0d", value);
        else fmt = "x";
    endfunction

    function automatic string raw(input logic [31:0] value);
        raw = $sformatf("%0d", value);
    endfunction

    task automatic capture_preedge;
        begin
            pre_wr_fire = dut.fifo_wr_fire;
            pre_rd_fire = dut.fifo_rd_fire;
            pre_mul0 = dut.mac_mul0; pre_add0 = dut.mac_add0;
            pre_mul1 = dut.mac_mul1; pre_add1 = dut.mac_add1;
            pre_mul2 = dut.mac_mul2; pre_result16 = dut.mac_result16;
            pre_result8 = dut.mac_result8;
            n_source = dut.source_next; n_destination = dut.destination_next;
            n_bank0 = dut.bank_next[0]; n_bank1 = dut.bank_next[1];
            n_fifo_wr_ptr = dut.fifo.wr_ptr_next;
            n_fifo_rd_ptr = dut.fifo.rd_ptr_next;
            n_fifo_count = dut.fifo.count_next;
            n_fifo_rd_data = dut.fifo.rd_data_next;
            n_fifo_rd_data_valid = dut.fifo.rd_data_valid_next;
            n_fifo_valid = dut.fifo.memory_valid_next;
            for (i = 0; i < 4; i = i + 1) n_fifo_memory[i] = dut.fifo.memory_next[i];
            n_sram_rd_data = dut.sram.read_data_next;
            n_sram_rd_valid = dut.sram.read_valid_next;
            n_sram_rd_initialized = dut.sram.read_initialized_next;
        end
    endtask

    task automatic dump_row(input int phase, input bit use_pre_mac);
        logic [15:0] m0, a0, m1, a1, m2, r16;
        logic [7:0] r8;
        begin
            if (use_pre_mac) begin
                m0=pre_mul0; a0=pre_add0; m1=pre_mul1; a1=pre_add1;
                m2=pre_mul2; r16=pre_result16; r8=pre_result8;
            end else begin
                m0=dut.mac_mul0; a0=dut.mac_add0; m1=dut.mac_mul1; a1=dut.mac_add1;
                m2=dut.mac_mul2; r16=dut.mac_result16; r8=dut.mac_result8;
            end
            $fwrite(trace_fd, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,",
                cycle, phase, (phase==1 || phase==2), rst, reg_en, reg_data,
                fifo_wr_en, fifo_wr_data, fifo_rd_en, sram_op, sram_addr, sram_wdata, state_valid);
            $fwrite(trace_fd, "%s,%s,%s,%s,%s,%s,%s,%s,",
                fmt(state_valid,dut.source_q), raw(n_source),
                fmt(state_valid,dut.destination_q), raw(n_destination),
                fmt(state_valid,dut.bank_q[0]), fmt(state_valid,dut.bank_q[1]),
                raw(n_bank0), raw(n_bank1));
            $fwrite(trace_fd, "%s,%s,%s,%s,%s,%s,%s,",
                raw(m0),raw(a0),raw(m1),raw(a1),raw(m2),raw(r16),raw(r8));
            $fwrite(trace_fd, "%s,%s,%s,%s,%0d,%0d,",
                fmt(state_valid,dut.fifo_full), fmt(state_valid,dut.fifo_empty),
                fmt(state_valid,dut.fifo_wr_ready), fmt(state_valid,dut.fifo_rd_available),
                pre_wr_fire, pre_rd_fire);
            $fwrite(trace_fd, "%s,%s,%s,%s,%s,",
                fmt(state_valid,dut.fifo.wr_ptr), fmt(state_valid,dut.fifo.rd_ptr),
                fmt(state_valid,dut.fifo.count), fmt(state_valid,dut.fifo.rd_data),
                fmt(state_valid,dut.fifo.rd_data_valid));
            for (i=0;i<4;i=i+1)
                $fwrite(trace_fd, "%s,", fmt(state_valid && dut.fifo.memory_valid[i], dut.fifo.memory[i]));
            for (i=0;i<4;i=i+1)
                $fwrite(trace_fd, "%s,", fmt(state_valid, dut.fifo.memory_valid[i]));
            $fwrite(trace_fd, "%s,%s,%s,%s,%s,",
                raw(n_fifo_wr_ptr), raw(n_fifo_rd_ptr), raw(n_fifo_count),
                raw(n_fifo_rd_data), raw(n_fifo_rd_data_valid));
            for (i=0;i<4;i=i+1)
                $fwrite(trace_fd, "%s,", fmt(n_fifo_valid[i], n_fifo_memory[i]));
            for (i=0;i<4;i=i+1) $fwrite(trace_fd, "%s,", raw(n_fifo_valid[i]));
            $fwrite(trace_fd, "%s,%s,%s,%s,%s,%s,",
                fmt(state_valid,dut.sram.read_data), fmt(state_valid,dut.sram.read_valid),
                fmt(state_valid,dut.sram.read_initialized), raw(n_sram_rd_data),
                raw(n_sram_rd_valid), raw(n_sram_rd_initialized));
            for (i=0;i<4;i=i+1)
                $fwrite(trace_fd, "%s,", fmt(dut.sram.initialized[i],dut.sram.memory[i]));
            for (i=0;i<4;i=i+1) begin
                if (i == 3) $fwrite(trace_fd, "%s\n", raw(dut.sram.initialized[i]));
                else $fwrite(trace_fd, "%s,", raw(dut.sram.initialized[i]));
            end
        end
    endtask

    task automatic drive_word(input logic [31:0] word);
        begin
            rst = word[31]; reg_en = word[30]; reg_data = word[29:22];
            fifo_wr_en = word[21]; fifo_wr_data = word[20:13]; fifo_rd_en = word[12];
            sram_op = word[11:10]; sram_addr = word[9:8]; sram_wdata = word[7:0];
        end
    endtask

    initial begin
        clk = 0; rst = 0; reg_en = 0; reg_data = 0;
        fifo_wr_en = 0; fifo_wr_data = 0; fifo_rd_en = 0;
        sram_op = 0; sram_addr = 0; sram_wdata = 0;
        if (!$value$plusargs("STIMULUS=%s", stimulus_path)) stimulus_path = "stimulus/stimulus.hex";
        if (!$value$plusargs("TRACE=%s", trace_path)) trace_path = "rtl_trace.csv";
        $readmemh(stimulus_path, stimulus);
        trace_fd = $fopen(trace_path, "w");
        if (trace_fd == 0) $fatal(1, "cannot open trace output");
        $fwrite(trace_fd, "cycle,phase,clk,rst,reg_en,reg_data,fifo_wr_en,fifo_wr_data,fifo_rd_en,");
        $fwrite(trace_fd, "sram_op,sram_addr,sram_wdata,state_valid,src_q,src_next,dst_q,dst_next,");
        $fwrite(trace_fd, "bank0_q,bank1_q,bank0_next,bank1_next,mac_mul0,mac_add0,mac_mul1,mac_add1,");
        $fwrite(trace_fd, "mac_mul2,mac_result16,mac_result8,fifo_full,fifo_empty,fifo_wr_ready,");
        $fwrite(trace_fd, "fifo_rd_available,fifo_wr_fire,fifo_rd_fire,fifo_wr_ptr,fifo_rd_ptr,fifo_count,");
        $fwrite(trace_fd, "fifo_rd_data,fifo_rd_data_valid,fifo_m0,fifo_m1,fifo_m2,fifo_m3,");
        $fwrite(trace_fd, "fifo_v0,fifo_v1,fifo_v2,fifo_v3,fifo_n_wr_ptr,fifo_n_rd_ptr,fifo_n_count,");
        $fwrite(trace_fd, "fifo_n_rd_data,fifo_n_rd_data_valid,fifo_n_m0,fifo_n_m1,fifo_n_m2,fifo_n_m3,");
        $fwrite(trace_fd, "fifo_n_v0,fifo_n_v1,fifo_n_v2,fifo_n_v3,sram_rd_data,sram_rd_valid,");
        $fwrite(trace_fd, "sram_rd_initialized,sram_n_rd_data,sram_n_rd_valid,sram_n_rd_initialized,");
        $fwrite(trace_fd, "sram_m0,sram_m1,sram_m2,sram_m3,sram_i0,sram_i1,sram_i2,sram_i3\n");

        for (cycle=0; cycle<CYCLES; cycle=cycle+1) begin
            drive_word(stimulus[cycle]);
            #1; capture_preedge(); dump_row(0, 1'b1);
            #1; clk = 1'b1;
            #1; dump_row(1, 1'b1);
            #1; dump_row(2, 1'b0);
            #1; clk = 1'b0;
            #1; dump_row(3, 1'b0);
        end
        $fclose(trace_fd);
        $display("rtl_cycles=%0d", CYCLES);
        $display("rtl_trace_rows=%0d", CYCLES*4);
        $display("result=PASS");
        $finish;
    end
endmodule

