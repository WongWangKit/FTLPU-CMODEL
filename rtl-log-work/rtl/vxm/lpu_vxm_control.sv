module lpu_vxm_control (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic datapath_idle_i,

  input  logic [7:0] local_issue_valid_i,
  input  logic [8*lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    local_issue_instruction_i,
  input  logic global_issue_valid_i,
  input  logic [lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH-1:0]
    global_issue_instruction_i,

  output logic [4*8-1:0] tile_local_valid_o,
  output logic [4*8*lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    tile_local_instruction_o,
  output logic [4*lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH-1:0]
    tile_global_config_o,
  output logic [3:0] tile_global_config_valid_o,
  output logic global_config_valid_o,
  output logic global_config_fault_o
);
  localparam integer LOCAL_MAX_WIDTH =
    lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH;
  localparam integer CONFIG_WIDTH = lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH;

  logic config_apply;
  logic [CONFIG_WIDTH-1:0] config_apply_data;
  logic control_idle;
  logic [3:0] global_row_valid;
  logic [4*CONFIG_WIDTH-1:0] global_row_config;
  logic [3:0] global_row_loaded_q;
  logic [4*CONFIG_WIDTH-1:0] global_row_config_q;

  assign control_idle = !(|tile_local_valid_o) && !(|global_row_valid);

  lpu_vxm_global_config #(.CONFIG_WIDTH(CONFIG_WIDTH)) u_global_config (
    .clk_i,
    .rst_ni,
    .write_valid_i(global_issue_valid_i),
    .write_data_i(global_issue_instruction_i),
    .commit_i(global_issue_valid_i),
    .safe_i(datapath_idle_i && control_idle),
    .commit_ready_o(),
    .apply_o(config_apply),
    .apply_data_o(config_apply_data),
    .config_valid_o(global_config_valid_o),
    .current_config_o(),
    .fault_o(global_config_fault_o)
  );

  // The global word follows the same four-row geometry as the local words.
  // Each row retains its most recently received word after the valid pulse.
  lpu_control_pipeline #(.WIDTH(CONFIG_WIDTH), .ROWS(4)) u_global_wave (
    .clk_i,
    .rst_ni,
    .issue_valid_i(config_apply),
    .issue_payload_i(config_apply_data),
    .row_valid_o(global_row_valid),
    .row_payload_o(global_row_config)
  );

  always_comb begin
    tile_global_config_o = global_row_config_q;
    tile_global_config_valid_o = global_row_loaded_q | global_row_valid;
    for (integer row = 0; row < 4; row++) begin
      if (global_row_valid[row])
        tile_global_config_o[row*CONFIG_WIDTH +: CONFIG_WIDTH] =
          global_row_config[row*CONFIG_WIDTH +: CONFIG_WIDTH];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      global_row_loaded_q <= '0;
      global_row_config_q <= '0;
    end else begin
      global_row_loaded_q <= global_row_loaded_q | global_row_valid;
      for (integer row = 0; row < 4; row++) begin
        if (global_row_valid[row])
          global_row_config_q[row*CONFIG_WIDTH +: CONFIG_WIDTH] <=
            global_row_config[row*CONFIG_WIDTH +: CONFIG_WIDTH];
      end
    end
  end

  generate
    for (genvar queue = 0; queue < 8; queue++) begin : gen_local_wave
      localparam integer LOCAL_WIDTH =
        lpu_pkg::vxm_local_instruction_width(queue);
      logic [3:0] row_valid;
      logic [4*LOCAL_WIDTH-1:0] row_instruction;

      lpu_control_pipeline #(.WIDTH(LOCAL_WIDTH), .ROWS(4)) u_pipeline (
        .clk_i,
        .rst_ni,
        .issue_valid_i(local_issue_valid_i[queue]),
        .issue_payload_i(
          local_issue_instruction_i[queue*LOCAL_MAX_WIDTH +: LOCAL_WIDTH]),
        .row_valid_o(row_valid),
        .row_payload_o(row_instruction)
      );

      for (genvar tile = 0; tile < 4; tile++) begin : gen_tile
        wire [LOCAL_MAX_WIDTH-1:0] padded_instruction =
          LOCAL_MAX_WIDTH'(
            row_instruction[tile*LOCAL_WIDTH +: LOCAL_WIDTH]);
        assign tile_local_valid_o[tile*8 + queue] = row_valid[tile];
        assign tile_local_instruction_o[
          (tile*8+queue)*LOCAL_MAX_WIDTH +: LOCAL_MAX_WIDTH] =
          padded_instruction;
      end
    end
  endgenerate
endmodule
