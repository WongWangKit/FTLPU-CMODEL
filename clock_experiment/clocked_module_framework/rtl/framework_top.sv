`timescale 1ns/1ps

module sync_fifo_framework #(
    parameter int WIDTH = 8,
    parameter int DEPTH = 4,
    parameter int PTR_W = $clog2(DEPTH)
) (
    input  logic clk,
    input  logic rst,
    input  logic wr_en,
    input  logic [WIDTH-1:0] wr_data,
    input  logic rd_en,
    output logic full,
    output logic empty,
    output logic wr_ready,
    output logic rd_available,
    output logic wr_fire,
    output logic rd_fire
);
    logic [WIDTH-1:0] memory [0:DEPTH-1];
    logic [WIDTH-1:0] memory_next [0:DEPTH-1];
    logic [DEPTH-1:0] memory_valid, memory_valid_next;
    logic [PTR_W-1:0] wr_ptr, rd_ptr, wr_ptr_next, rd_ptr_next;
    logic [$clog2(DEPTH+1)-1:0] count, count_next;
    logic [WIDTH-1:0] rd_data, rd_data_next;
    logic rd_data_valid, rd_data_valid_next;

    always_comb begin
        full = (count == DEPTH);
        empty = (count == 0);
        wr_ready = !full;
        rd_available = !empty;
        wr_fire = wr_en && wr_ready && !rst;
        rd_fire = rd_en && rd_available && !rst;

        wr_ptr_next = wr_ptr;
        rd_ptr_next = rd_ptr;
        count_next = count;
        rd_data_next = rd_data;
        rd_data_valid_next = 1'b0;
        memory_valid_next = memory_valid;
        for (int i = 0; i < DEPTH; i = i + 1) memory_next[i] = memory[i];

        if (rst) begin
            wr_ptr_next = '0;
            rd_ptr_next = '0;
            count_next = '0;
            rd_data_next = '0;
            rd_data_valid_next = 1'b0;
            memory_valid_next = '0;
        end else begin
            if (wr_fire) begin
                memory_next[wr_ptr] = wr_data;
                memory_valid_next[wr_ptr] = 1'b1;
                wr_ptr_next = (wr_ptr == DEPTH-1) ? '0 : wr_ptr + 1'b1;
            end
            if (rd_fire) begin
                rd_data_next = memory[rd_ptr];
                rd_data_valid_next = 1'b1;
                rd_ptr_next = (rd_ptr == DEPTH-1) ? '0 : rd_ptr + 1'b1;
            end
            case ({wr_fire, rd_fire})
                2'b10: count_next = count + 1'b1;
                2'b01: count_next = count - 1'b1;
                default: count_next = count;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        wr_ptr <= wr_ptr_next;
        rd_ptr <= rd_ptr_next;
        count <= count_next;
        rd_data <= rd_data_next;
        rd_data_valid <= rd_data_valid_next;
        memory_valid <= memory_valid_next;
        for (int i = 0; i < DEPTH; i = i + 1) memory[i] <= memory_next[i];
    end
endmodule

module sync_sram_framework #(
    parameter int WIDTH = 8,
    parameter int DEPTH = 4,
    parameter int ADDR_W = $clog2(DEPTH)
) (
    input logic clk,
    input logic rst,
    input logic [1:0] operation,
    input logic [ADDR_W-1:0] address,
    input logic [WIDTH-1:0] write_data
);
    logic [WIDTH-1:0] memory [0:DEPTH-1];
    // A 2-state initialized bitmap starts false while Memory remains 4-state/X.
    bit [DEPTH-1:0] initialized;
    logic [WIDTH-1:0] read_data, read_data_next;
    logic read_valid, read_valid_next;
    logic read_initialized, read_initialized_next;

    always_comb begin
        read_data_next = read_data;
        read_valid_next = 1'b0;
        read_initialized_next = 1'b0;
        if (rst) begin
            read_data_next = '0;
            read_valid_next = 1'b0;
            read_initialized_next = 1'b0;
        end else if (operation == 2'd1) begin
            read_valid_next = 1'b1;
            read_initialized_next = initialized[address];
            if (initialized[address]) read_data_next = memory[address];
        end
    end

    always_ff @(posedge clk) begin
        read_data <= read_data_next;
        read_valid <= read_valid_next;
        read_initialized <= read_initialized_next;
        if (!rst && operation == 2'd2) begin
            memory[address] <= write_data;
            initialized[address] <= 1'b1;
        end
    end
endmodule

module framework_top (
    input logic clk,
    input logic rst,
    input logic reg_en,
    input logic [7:0] reg_data,
    input logic fifo_wr_en,
    input logic [7:0] fifo_wr_data,
    input logic fifo_rd_en,
    input logic [1:0] sram_op,
    input logic [1:0] sram_addr,
    input logic [7:0] sram_wdata
);
    logic [7:0] source_q, source_next;
    logic [7:0] destination_q, destination_next;
    logic [7:0] bank_q [0:1];
    logic [7:0] bank_next [0:1];
    logic [15:0] mac_mul0, mac_add0, mac_mul1, mac_add1, mac_mul2, mac_result16;
    logic [7:0] mac_result8;
    logic fifo_full, fifo_empty, fifo_wr_ready, fifo_rd_available, fifo_wr_fire, fifo_rd_fire;

    always_comb begin
        source_next = source_q;
        bank_next[0] = bank_q[0];
        bank_next[1] = bank_q[1];
        if (rst) begin
            source_next = '0;
            bank_next[0] = '0;
            bank_next[1] = '0;
        end else if (reg_en) begin
            source_next = reg_data;
            bank_next[0] = reg_data;
            bank_next[1] = reg_data ^ 8'h5a;
        end

        mac_mul0 = {8'b0, source_q} * 16'd3;
        mac_add0 = mac_mul0 + 16'd17;
        mac_mul1 = mac_add0 * 16'd5;
        mac_add1 = mac_mul1 + 16'd29;
        mac_mul2 = mac_add1 * 16'd7;
        mac_result16 = mac_mul2 + 16'd43;
        mac_result8 = mac_result16[7:0];
        destination_next = rst ? 8'd0 : mac_result8;
    end

    always_ff @(posedge clk) begin
        source_q <= source_next;
        destination_q <= destination_next;
        bank_q[0] <= bank_next[0];
        bank_q[1] <= bank_next[1];
    end

    sync_fifo_framework fifo (
        .clk, .rst, .wr_en(fifo_wr_en), .wr_data(fifo_wr_data), .rd_en(fifo_rd_en),
        .full(fifo_full), .empty(fifo_empty), .wr_ready(fifo_wr_ready),
        .rd_available(fifo_rd_available), .wr_fire(fifo_wr_fire), .rd_fire(fifo_rd_fire)
    );

    sync_sram_framework sram (
        .clk, .rst, .operation(sram_op), .address(sram_addr), .write_data(sram_wdata)
    );
endmodule

