module mac_chain (
    input  logic [7:0]  x,
    output logic [15:0] mul0,
    output logic [15:0] add0,
    output logic [15:0] mul1,
    output logic [15:0] add1,
    output logic [15:0] mul2,
    output logic [15:0] result16,
    output logic [7:0]  result8
);
    timeunit 1ns;
    timeprecision 1ps;
    always_comb begin
        mul0 = {8'd0, x} * 16'd3;
        add0 = mul0 + 16'd17;
        mul1 = add0 * 16'd5;
        add1 = mul1 + 16'd29;
        mul2 = add1 * 16'd7;
        result16 = mul2 + 16'd43;
        result8 = result16[7:0];
    end
endmodule
