module lpu_vxm_global_config #(
  parameter integer CONFIG_WIDTH = lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH
) (
  input  logic                    clk_i,
  input  logic                    rst_ni,

  input  logic                    write_valid_i,
  input  logic [CONFIG_WIDTH-1:0] write_data_i,
  input  logic                    commit_i,
  input  logic                    safe_i,

  output logic                    commit_ready_o,
  output logic                    apply_o,
  output logic [CONFIG_WIDTH-1:0] apply_data_o,
  output logic                    config_valid_o,
  output logic [CONFIG_WIDTH-1:0] current_config_o,
  output logic                    fault_o
);
  logic [CONFIG_WIDTH-1:0] shadow_config_q;
  logic shadow_valid_q;

  // A write and commit may occur together.  In that case the newly supplied
  // word, rather than the previous shadow word, is broadcast atomically.
  assign apply_data_o = write_valid_i ? write_data_i : shadow_config_q;
  assign commit_ready_o = safe_i && (shadow_valid_q || write_valid_i);
  assign apply_o = commit_i && commit_ready_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      shadow_config_q <= '0;
      shadow_valid_q <= 1'b0;
      current_config_o <= '0;
      config_valid_o <= 1'b0;
      fault_o <= 1'b0;
    end else begin
      if (write_valid_i) begin
        shadow_config_q <= write_data_i;
        shadow_valid_q <= 1'b1;
      end

      if (apply_o) begin
        current_config_o <= apply_data_o;
        config_valid_o <= 1'b1;
      end else if (commit_i) begin
        // Configuration changes are legal only at a boundary at which no VXM
        // data or queue sequence is active.
        fault_o <= 1'b1;
      end
    end
  end
endmodule
