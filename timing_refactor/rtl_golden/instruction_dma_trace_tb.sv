`timescale 1ns/1ps

module instruction_dma_trace_tb;
    logic clk_i;
    logic rst_ni;
    logic dma_cmd_valid_i;
    logic dma_cmd_ready_o;
    logic [63:0] dma_cmd_address_i;
    logic [31:0] dma_cmd_length_i;
    logic [7:0] dma_cmd_id_i;
    logic dma_cmd_valid_o;
    logic dma_cmd_ready_i;
    logic [63:0] dma_cmd_address_o;
    logic [31:0] dma_cmd_length_o;
    logic [7:0] dma_cmd_id_o;
    logic dma_rsp_valid_i;
    logic dma_rsp_ready_o;
    logic [255:0] dma_rsp_data_i;
    logic [31:0] dma_rsp_keep_i;
    logic dma_rsp_last_i;
    logic [7:0] dma_rsp_id_i;
    logic dma_rsp_error_i;
    logic response_valid_o;
    logic response_ready_i;
    logic [255:0] response_data_o;
    logic [31:0] response_keep_o;
    logic response_last_o;
    logic [7:0] response_id_o;
    logic response_error_o;
    logic [4:0] outstanding_o;
    logic idle_o;
    logic fault_o;
    logic [5:0] fault_code_o;

    integer stimulus_file;
    integer trace_file;
    integer result;
    integer cycle;
    integer scenario;
    integer reset_n_value;
    integer command_valid_value;
    longint unsigned command_address_value;
    integer command_length_value;
    integer command_id_value;
    integer request_ready_value;
    integer response_valid_value;
    logic [255:0] response_data_value;
    logic [31:0] response_keep_value;
    integer response_last_value;
    integer response_id_value;
    integer response_error_value;
    integer reassembler_ready_value;
    string stimulus_path;
    string trace_path;
    string header;

    lpu_instruction_dma dut (
        .clk_i,
        .rst_ni,
        .ctrl_dma_cmd_valid_i(dma_cmd_valid_i),
        .ctrl_dma_cmd_ready_o(dma_cmd_ready_o),
        .ctrl_dma_cmd_address_i(dma_cmd_address_i),
        .ctrl_dma_cmd_length_i(dma_cmd_length_i),
        .ctrl_dma_cmd_id_i(dma_cmd_id_i),
        .ddr_read_req_valid_o(dma_cmd_valid_o),
        .ddr_read_req_ready_i(dma_cmd_ready_i),
        .ddr_read_req_address_o(dma_cmd_address_o),
        .ddr_read_req_length_o(dma_cmd_length_o),
        .ddr_read_req_id_o(dma_cmd_id_o),
        .ddr_read_rsp_valid_i(dma_rsp_valid_i),
        .ddr_read_rsp_ready_o(dma_rsp_ready_o),
        .ddr_read_rsp_data_i(dma_rsp_data_i),
        .ddr_read_rsp_keep_i(dma_rsp_keep_i),
        .ddr_read_rsp_last_i(dma_rsp_last_i),
        .ddr_read_rsp_id_i(dma_rsp_id_i),
        .ddr_read_rsp_error_i(dma_rsp_error_i),
        .reasm_rsp_valid_o(response_valid_o),
        .reasm_rsp_ready_i(response_ready_i),
        .reasm_rsp_data_o(response_data_o),
        .reasm_rsp_keep_o(response_keep_o),
        .reasm_rsp_last_o(response_last_o),
        .reasm_rsp_id_o(response_id_o),
        .reasm_rsp_error_o(response_error_o),
        .outstanding_o,
        .idle_o,
        .fault_o,
        .fault_code_o
    );

    always #5 clk_i = ~clk_i;

    initial begin
        clk_i = 1'b0;
        rst_ni = 1'b0;
        dma_cmd_valid_i = 1'b0;
        dma_cmd_address_i = '0;
        dma_cmd_length_i = '0;
        dma_cmd_id_i = '0;
        dma_cmd_ready_i = 1'b0;
        dma_rsp_valid_i = 1'b0;
        dma_rsp_data_i = '0;
        dma_rsp_keep_i = '0;
        dma_rsp_last_i = 1'b0;
        dma_rsp_id_i = '0;
        dma_rsp_error_i = 1'b0;
        response_ready_i = 1'b0;

        if (!$value$plusargs("STIMULUS=%s", stimulus_path))
            $fatal(1, "missing +STIMULUS=<path>");
        if (!$value$plusargs("TRACE=%s", trace_path))
            $fatal(1, "missing +TRACE=<path>");
        stimulus_file = $fopen(stimulus_path, "r");
        trace_file = $fopen(trace_path, "w");
        if (stimulus_file == 0 || trace_file == 0)
            $fatal(1, "cannot open instruction DMA stimulus or trace");
        result = $fgets(header, stimulus_file);
        $fwrite(trace_file,
            "cycle,scenario,reset_n,cmd_valid,cmd_address,cmd_length,cmd_id,req_ready,rsp_valid,rsp_data,rsp_keep,rsp_last,rsp_id,rsp_error,reasm_ready,cmd_ready_pre,req_valid_pre,req_address_pre,req_length_pre,req_id_pre,cmd_fire,outstanding_enq_fire,outstanding_deq_fire,rsp_ready_pre,rsp_input_fire,resp_valid_pre,resp_data_pre,resp_keep_pre,resp_last_pre,resp_id_pre,resp_error_pre,reasm_fire,outstanding_pre,rx_pre,fault_comb_pre,fault_code_comb_pre,outstanding_post,rx_post,fault_state_post,fault_code_post,idle_post,fault_wire_post_nba\n");

        while (!$feof(stimulus_file)) begin
            result = $fscanf(stimulus_file,
                "%d,%d,%d,%d,%d,%d,%d,%d,%d,%h,%h,%d,%d,%d,%d\n",
                cycle, scenario, reset_n_value, command_valid_value,
                command_address_value, command_length_value, command_id_value,
                request_ready_value, response_valid_value, response_data_value,
                response_keep_value, response_last_value, response_id_value,
                response_error_value, reassembler_ready_value);
            if (result == 15) begin
                @(negedge clk_i);
                rst_ni = reset_n_value[0];
                dma_cmd_valid_i = command_valid_value[0];
                dma_cmd_address_i = command_address_value;
                dma_cmd_length_i = command_length_value;
                dma_cmd_id_i = command_id_value;
                dma_cmd_ready_i = request_ready_value[0];
                dma_rsp_valid_i = response_valid_value[0];
                dma_rsp_data_i = response_data_value;
                dma_rsp_keep_i = response_keep_value;
                dma_rsp_last_i = response_last_value[0];
                dma_rsp_id_i = response_id_value;
                dma_rsp_error_i = response_error_value[0];
                response_ready_i = reassembler_ready_value[0];
                #1;
                $fwrite(trace_file,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%08h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%08h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,",
                    cycle, scenario, reset_n_value, command_valid_value,
                    command_address_value, command_length_value, command_id_value,
                    request_ready_value, response_valid_value, response_data_value,
                    response_keep_value, response_last_value, response_id_value,
                    response_error_value, reassembler_ready_value,
                    dma_cmd_ready_o, dma_cmd_valid_o, dma_cmd_address_o,
                    dma_cmd_length_o, dma_cmd_id_o,
                    dma_cmd_valid_i && dma_cmd_ready_o,
                    dut.u_outstanding_id_fifo.enqueue_fire,
                    dut.u_outstanding_id_fifo.dequeue_fire,
                    dma_rsp_ready_o, dma_rsp_valid_i && dma_rsp_ready_o,
                    response_valid_o,
                    response_valid_o ? response_data_o : 256'b0,
                    response_valid_o ? response_keep_o : 32'b0,
                    response_valid_o ? response_last_o : 1'b0,
                    response_valid_o ? response_id_o : 8'b0,
                    response_valid_o ? response_error_o : 1'b0,
                    response_valid_o && response_ready_i,
                    outstanding_o, dut.response_level, fault_o, fault_code_o);
                @(posedge clk_i);
                #1;
                $fwrite(trace_file, "%0d,%0d,%0d,%0d,%0d,%0d\n",
                    outstanding_o, dut.response_level, dut.fault_q,
                    dut.fault_code_q, idle_o, fault_o);
            end
        end

        $fclose(stimulus_file);
        $fclose(trace_file);
        $display("instruction_dma_trace_tb completed");
        $finish;
    end
endmodule
