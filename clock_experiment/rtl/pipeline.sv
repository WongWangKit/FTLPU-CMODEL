module pipeline (
    input  logic       clk,
    input  logic       rst,
    input  logic       en,
    input  logic [7:0] din,
    output logic [7:0] q1,
    output logic [7:0] q2,
    output logic [7:0] dout,
    output logic [7:0] d1,
    output logic [7:0] d2
);
    always_comb begin
        d1 = q1;
        d2 = q2;

        if (rst) begin
            d1 = 8'h00;
            d2 = 8'h00;
        end else if (en) begin
            d1 = din;
            d2 = q1;
        end
    end

    always_ff @(posedge clk) begin
        q1 <= d1;
        q2 <= d2;
    end

    assign dout = q2;
endmodule
