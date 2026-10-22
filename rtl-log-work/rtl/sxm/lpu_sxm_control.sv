module lpu_sxm_control (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,
  input  logic transpose_valid_i,
  input  logic [415:0] transpose_instruction_i,
  input  logic permute_valid_i,
  input  logic [415:0] permute_instruction_i,
  output logic [3:0] transpose_row_valid_o,
  output logic [4*416-1:0] transpose_row_instruction_o,
  output logic permute_valid_o,
  output logic [415:0] permute_instruction_o
);
  lpu_control_pipeline #(.WIDTH(416), .ROWS(4)) u_transpose_pipeline (
    .clk_i,
    .rst_ni,
    .issue_valid_i(transpose_valid_i),
    .issue_payload_i(transpose_instruction_i),
    .row_valid_o(transpose_row_valid_o),
    .row_payload_o(transpose_row_instruction_o)
  );

  // Permute controls the complete four-tile bank in one physical operation.
  // The register also enforces the required full-cycle separation from a
  // Transpose capture issued on the previous schedule cycle.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      permute_valid_o <= 1'b0;
      permute_instruction_o <= '0;
    end else if (!run_i) begin
      permute_valid_o <= 1'b0;
      permute_instruction_o <= '0;
    end else begin
      permute_valid_o <= permute_valid_i;
      permute_instruction_o <= permute_instruction_i;
    end
  end
endmodule
