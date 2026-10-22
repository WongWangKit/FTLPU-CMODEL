module lpu_control_pipeline #(
  parameter integer WIDTH = 47,
  parameter integer ROWS  = 4
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic issue_valid_i,
  input  logic [WIDTH-1:0] issue_payload_i,
  output logic [ROWS-1:0] row_valid_o,
  output logic [ROWS*WIDTH-1:0] row_payload_o
);
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      row_valid_o   <= '0;
      row_payload_o <= '0;
    end else begin
      row_valid_o[0] <= issue_valid_i;
      row_payload_o[0 +: WIDTH] <= issue_payload_i;
      for (integer row = 1; row < ROWS; row++) begin
        row_valid_o[row] <= row_valid_o[row-1];
        row_payload_o[row*WIDTH +: WIDTH] <=
          row_payload_o[(row-1)*WIDTH +: WIDTH];
      end
    end
  end
endmodule
