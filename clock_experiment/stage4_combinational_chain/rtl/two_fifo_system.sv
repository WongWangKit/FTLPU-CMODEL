module two_fifo_system (
    input logic clk, rst, producer_valid, input logic [7:0] producer_data,
    input logic consumer_ready,
    output logic tc_a_rd_en, tc_b_wr_en, output logic [7:0] tc_b_wr_data,
    output logic tc_b_rd_en, producer_ready, transfer_invariant_ok,
    output logic mac_valid, output logic [15:0] mac_mul0, mac_add0, mac_mul1, mac_add1, mac_mul2, mac_result16,
    output logic [7:0] mac_result8,
    output logic a_full, a_empty, a_wr_ready, a_rd_available, a_wr_fire, a_rd_fire,
    output logic [1:0] a_wr_ptr_q, a_rd_ptr_q, a_wr_ptr_d, a_rd_ptr_d,
    output logic [2:0] a_count_q, a_count_d,
    output logic [7:0] a_rd_data_q, a_rd_data_d,
    output logic a_rd_data_valid_q, a_rd_data_valid_d,
    output logic b_full, b_empty, b_wr_ready, b_rd_available, b_wr_fire, b_rd_fire,
    output logic [1:0] b_wr_ptr_q, b_rd_ptr_q, b_wr_ptr_d, b_rd_ptr_d,
    output logic [2:0] b_count_q, b_count_d,
    output logic [7:0] b_rd_data_q, b_rd_data_d,
    output logic b_rd_data_valid_q, b_rd_data_valid_d
);
    timeunit 1ns;
    timeprecision 1ps;

    mac_chain mac (
        .x(a_rd_data_q), .mul0(mac_mul0), .add0(mac_add0), .mul1(mac_mul1), .add1(mac_add1),
        .mul2(mac_mul2), .result16(mac_result16), .result8(mac_result8)
    );

    always_comb begin
        mac_valid = a_rd_data_valid_q;
        tc_a_rd_en = !rst && !a_empty && !a_rd_data_valid_q && !b_full;
        tc_b_wr_en = !rst && a_rd_data_valid_q;
        tc_b_wr_data = mac_result8;
        tc_b_rd_en = !rst && consumer_ready;
        producer_ready = !rst && !a_full;
        transfer_invariant_ok = rst ? 1'b1 : !(a_rd_data_valid_q && b_full);
    end

    sync_fifo fifo_a (
        .clk(clk), .rst(rst), .wr_en(producer_valid), .wr_data(producer_data), .rd_en(tc_a_rd_en),
        .full(a_full), .empty(a_empty), .wr_ready(a_wr_ready), .rd_available(a_rd_available),
        .wr_fire(a_wr_fire), .rd_fire(a_rd_fire), .wr_ptr_q(a_wr_ptr_q), .rd_ptr_q(a_rd_ptr_q),
        .count_q(a_count_q), .rd_data_q(a_rd_data_q), .rd_data_valid_q(a_rd_data_valid_q),
        .wr_ptr_d(a_wr_ptr_d), .rd_ptr_d(a_rd_ptr_d), .count_d(a_count_d),
        .rd_data_d(a_rd_data_d), .rd_data_valid_d(a_rd_data_valid_d)
    );

    sync_fifo fifo_b (
        .clk(clk), .rst(rst), .wr_en(tc_b_wr_en), .wr_data(tc_b_wr_data), .rd_en(tc_b_rd_en),
        .full(b_full), .empty(b_empty), .wr_ready(b_wr_ready), .rd_available(b_rd_available),
        .wr_fire(b_wr_fire), .rd_fire(b_rd_fire), .wr_ptr_q(b_wr_ptr_q), .rd_ptr_q(b_rd_ptr_q),
        .count_q(b_count_q), .rd_data_q(b_rd_data_q), .rd_data_valid_q(b_rd_data_valid_q),
        .wr_ptr_d(b_wr_ptr_d), .rd_ptr_d(b_rd_ptr_d), .count_d(b_count_d),
        .rd_data_d(b_rd_data_d), .rd_data_valid_d(b_rd_data_valid_d)
    );
endmodule
