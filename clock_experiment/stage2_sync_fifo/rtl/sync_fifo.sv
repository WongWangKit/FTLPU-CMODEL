module sync_fifo (
    input  logic       clk,
    input  logic       rst,
    input  logic       wr_en,
    input  logic [7:0] wr_data,
    input  logic       rd_en,
    output logic       full,
    output logic       empty,
    output logic       wr_ready,
    output logic       rd_available,
    output logic       wr_fire,
    output logic       rd_fire,
    output logic [1:0] wr_ptr_q,
    output logic [1:0] rd_ptr_q,
    output logic [2:0] count_q,
    output logic [7:0] rd_data_q,
    output logic       rd_data_valid_q,
    output logic [1:0] wr_ptr_d,
    output logic [1:0] rd_ptr_d,
    output logic [2:0] count_d,
    output logic [7:0] rd_data_d,
    output logic       rd_data_valid_d
);
    logic [7:0] mem [0:3];
    logic [3:0] mem_valid_q;

    always_comb begin
        full = (count_q == 3'd4);
        empty = (count_q == 3'd0);
        wr_ready = !full;
        rd_available = !empty;
        wr_fire = wr_en && wr_ready && !rst;
        rd_fire = rd_en && rd_available && !rst;

        wr_ptr_d = wr_ptr_q;
        rd_ptr_d = rd_ptr_q;
        count_d = count_q;
        rd_data_d = rd_data_q;
        rd_data_valid_d = 1'b0;

        if (rst) begin
            wr_ptr_d = 2'd0;
            rd_ptr_d = 2'd0;
            count_d = 3'd0;
            rd_data_d = 8'd0;
            rd_data_valid_d = 1'b0;
        end else begin
            if (wr_fire) begin
                wr_ptr_d = (wr_ptr_q == 2'd3) ? 2'd0 : wr_ptr_q + 2'd1;
            end
            if (rd_fire) begin
                rd_ptr_d = (rd_ptr_q == 2'd3) ? 2'd0 : rd_ptr_q + 2'd1;
                rd_data_d = mem[rd_ptr_q];
                rd_data_valid_d = 1'b1;
            end

            unique case ({wr_fire, rd_fire})
                2'b10: count_d = count_q + 3'd1;
                2'b01: count_d = count_q - 3'd1;
                default: count_d = count_q;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            wr_ptr_q <= 2'd0;
            rd_ptr_q <= 2'd0;
            count_q <= 3'd0;
            rd_data_q <= 8'd0;
            rd_data_valid_q <= 1'b0;
            mem_valid_q <= 4'b0000;
        end else begin
            wr_ptr_q <= wr_ptr_d;
            rd_ptr_q <= rd_ptr_d;
            count_q <= count_d;
            rd_data_q <= rd_data_d;
            rd_data_valid_q <= rd_data_valid_d;
            if (wr_fire) begin
                mem[wr_ptr_q] <= wr_data;
                mem_valid_q[wr_ptr_q] <= 1'b1;
            end
        end
    end
endmodule
