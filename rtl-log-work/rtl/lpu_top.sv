module lpu_top #(
  parameter integer ICU_QUEUE_DEPTH = 16,
  parameter integer MEM_DEPTH_ROWS  = 65536,
  parameter integer ACTIVE_MEM_COLUMNS = 52,
  parameter integer MXM_ACCUMULATOR_BLOCK_COUNT =
    lpu_pkg::MXM_ACCUMULATOR_BLOCK_COUNT,
  parameter bit USE_SRAM_MACRO = 1'b0
) (
  input  logic clk_i,
  input  logic rst_ni,

  // The C model preloads a complete static schedule before cycle zero.
  // Keep run_i low while filling queues, then hold it high while executing.
  input  logic         run_i,
  input  logic         schedule_valid_i,
  output logic         schedule_ready_o,
  input  logic [7:0]   schedule_queue_i,
  input  logic         schedule_is_instruction_i,
  input  logic [31:0]  schedule_command_i,
  input  logic [415:0] schedule_payload_i,

  // Host SRAM access is intended for cycle-zero initialization and final
  // result collection. Global MEM columns 0..51 are East and 52..103 West.
  input  logic         host_mem_write_valid_i,
  input  logic [6:0]   host_mem_column_i,
  input  logic [1:0]   host_mem_tile_i,
  input  logic [15:0]  host_mem_address_i,
  input  logic [63:0]  host_mem_write_data_i,
  output logic [63:0]  host_mem_read_data_o,

  // VXM-side East injection / West extraction and SXM-side mirror ports.
  input  logic [2*4*32-1:0]    mem_east_edge_valid_i,
  input  logic [2*4*32*64-1:0] mem_east_edge_data_i,
  output logic [2*4*32-1:0]    mem_east_edge_valid_o,
  output logic [2*4*32*64-1:0] mem_east_edge_data_o,
  input  logic [2*4*32-1:0]    mem_west_edge_valid_i,
  input  logic [2*4*32*64-1:0] mem_west_edge_data_i,
  output logic [2*4*32-1:0]    mem_west_edge_valid_o,
  output logic [2*4*32*64-1:0] mem_west_edge_data_o,

  // Architectural dispatch monitor. Queue numbering is defined in lpu_pkg;
  // MEM queue pulses also feed the integrated hemisphere datapaths below.
  output logic [137:0]       dispatch_valid_o,
  output logic [138*416-1:0] dispatch_payload_o,
  output logic [137:0]       queue_fault_o,
  output logic [2*52*4-1:0]  mem_fault_o,
  output logic [1:0]         mxm_fault_o,
  output logic [1:0]         sxm_fault_o,
  output logic               vxm_fault_o,
  output logic [1:0]         stream_conflict_o,
  output logic [63:0]        cycle_o
);
  logic [2*52*47-1:0] mem_issue_instruction;
  logic [4*48-1:0] mxm_load_instruction;
  logic [4*16-1:0] mxm_dequant_instruction;
  logic [4*48-1:0] mxm_compute_instruction;
  logic [7:0] vxm_local_issue_valid;
  logic [8*lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    vxm_local_issue_instruction;
  logic vxm_global_issue_valid;
  logic [lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH-1:0]
    vxm_global_issue_instruction;
  logic [2*64-1:0] host_mem_read_data;
  logic [2*4*32-1:0] mem_to_vxm_west_valid;
  logic [2*4*32*64-1:0] mem_to_vxm_west_data;
  logic [2*4*32-1:0] vxm_to_mem_east_valid;
  logic [2*4*32*64-1:0] vxm_to_mem_east_data;
  logic vxm_stream_conflict;

  lpu_icu #(.QUEUE_DEPTH(ICU_QUEUE_DEPTH)) u_icu (
    .clk_i,
    .rst_ni,
    .run_i,
    .enqueue_valid_i(schedule_valid_i),
    .enqueue_ready_o(schedule_ready_o),
    .enqueue_queue_i(schedule_queue_i),
    .enqueue_is_instruction_i(schedule_is_instruction_i),
    .enqueue_command_i(schedule_command_i),
    .enqueue_payload_i(schedule_payload_i),
    .issue_valid_o(dispatch_valid_o),
    .issue_payload_o(dispatch_payload_o),
    .queue_fault_o
  );

  always_comb begin
    mem_issue_instruction = '0;
    mxm_load_instruction = '0;
    mxm_dequant_instruction = '0;
    mxm_compute_instruction = '0;
    vxm_local_issue_valid = dispatch_valid_o[
      lpu_pkg::VXM_QUEUE_BASE +: lpu_pkg::VXM_LOCAL_QUEUE_COUNT];
    vxm_local_issue_instruction = '0;
    vxm_global_issue_valid = dispatch_valid_o[lpu_pkg::VXM_GLOBAL_QUEUE];
    vxm_global_issue_instruction = dispatch_payload_o[
      lpu_pkg::VXM_GLOBAL_QUEUE*416 +:
      lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH];
    host_mem_read_data_o = '0;
    for (integer column = 0; column < 104; column++)
      mem_issue_instruction[column*47 +: 47] =
        dispatch_payload_o[column*416 +: 47];
    for (integer queue = 0;
         queue < lpu_pkg::VXM_LOCAL_QUEUE_COUNT; queue++)
      vxm_local_issue_instruction[
        queue*lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH +:
        lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH] =
        dispatch_payload_o[
          (lpu_pkg::VXM_QUEUE_BASE+queue)*416 +:
          lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH];
    for (integer hemisphere = 0; hemisphere < 2; hemisphere++) begin
      mxm_load_instruction[(hemisphere*2)*48 +: 48] =
        dispatch_payload_o[(104+hemisphere)*416 +: 48];
      mxm_dequant_instruction[(hemisphere*2)*16 +: 16] =
        dispatch_payload_o[(106+hemisphere)*416 +: 16];
      mxm_compute_instruction[(hemisphere*2)*48 +: 48] =
        dispatch_payload_o[(108+hemisphere)*416 +: 48];
      mxm_load_instruction[(hemisphere*2+1)*48 +: 48] =
        dispatch_payload_o[(132+hemisphere)*416 +: 48];
      mxm_dequant_instruction[(hemisphere*2+1)*16 +: 16] =
        dispatch_payload_o[(134+hemisphere)*416 +: 16];
      mxm_compute_instruction[(hemisphere*2+1)*48 +: 48] =
        dispatch_payload_o[(136+hemisphere)*416 +: 48];
    end

    if (host_mem_column_i < 52)
      host_mem_read_data_o = host_mem_read_data[0 +: 64];
    else if (host_mem_column_i < 104)
      host_mem_read_data_o = host_mem_read_data[64 +: 64];
  end

  generate
    for (genvar hemisphere_gen = 0; hemisphere_gen < 2; hemisphere_gen++) begin : gen_mem
      localparam integer HEMISPHERE_INDEX = hemisphere_gen;
      logic [4*32-1:0] east_edge_valid;
      logic [4*32*64-1:0] east_edge_data;
      logic [4*32-1:0] west_edge_valid;
      logic [4*32*64-1:0] west_edge_data;
      logic [4*32-1:0] sxm_east_valid;
      logic [4*32*64-1:0] sxm_east_data;
      logic [4*32-1:0] sxm_west_valid;
      logic [4*32*64-1:0] sxm_west_data;
      logic [4*32-1:0] mxm_west_valid;
      logic [4*32*64-1:0] mxm_west_data;
      logic [4*32-1:0] mxm_east_valid;
      logic [4*32*64-1:0] mxm_east_data;
      logic [2*4*32-1:0] local_mxm_west_valid;
      logic [2*4*32*64-1:0] local_mxm_west_data;
      logic [2*4*32-1:0] local_mxm_east_valid;
      logic [2*4*32*64-1:0] local_mxm_east_data;
      logic [1:0] local_mxm_fault;
      logic [1:0] local_mxm_conflict;
      logic [63:0] host_read;
      logic [5:0] host_local_column;
      logic [52*4-1:0] mem_fault;
      logic mem_stream_conflict;
      logic sxm_stream_conflict;
      logic sxm_fault;
      logic mxm_stream_conflict;
      logic mxm_fault;

      if (HEMISPHERE_INDEX == 0)
        assign host_local_column = host_mem_column_i[5:0];
      else
        assign host_local_column = host_mem_column_i[5:0] - 6'd52;

      lpu_mem_hemisphere #(
        .DEPTH_ROWS(MEM_DEPTH_ROWS),
        .ACTIVE_COLUMNS(ACTIVE_MEM_COLUMNS),
        .USE_SRAM_MACRO(USE_SRAM_MACRO)
      ) u_mem (
        .clk_i,
        .rst_ni,
        .run_i,
        .issue_valid_i(dispatch_valid_o[HEMISPHERE_INDEX*52 +: 52]),
        .issue_instruction_i(mem_issue_instruction[HEMISPHERE_INDEX*52*47 +: 52*47]),
        .east_edge_valid_i(
          vxm_to_mem_east_valid[HEMISPHERE_INDEX*4*32 +: 4*32]),
        .east_edge_data_i(
          vxm_to_mem_east_data[HEMISPHERE_INDEX*4*32*64 +: 4*32*64]),
        .east_edge_valid_o(east_edge_valid),
        .east_edge_data_o(east_edge_data),
        .west_edge_valid_i(sxm_west_valid),
        .west_edge_data_i(sxm_west_data),
        .west_edge_valid_o(west_edge_valid),
        .west_edge_data_o(west_edge_data),
        .host_write_valid_i(
          host_mem_write_valid_i &&
          (host_mem_column_i >= HEMISPHERE_INDEX*52) &&
          (host_mem_column_i < (HEMISPHERE_INDEX+1)*52)),
        .host_column_i(host_local_column),
        .host_tile_i(host_mem_tile_i),
        .host_address_i(host_mem_address_i),
        .host_write_data_i(host_mem_write_data_i),
        .host_read_data_o(host_read),
        .mem_fault_o(mem_fault),
        .conflict_o(mem_stream_conflict)
      );

      lpu_sxm_slice u_sxm (
        .clk_i,
        .rst_ni,
        .run_i,
        .transpose_issue_valid_i(dispatch_valid_o[128+HEMISPHERE_INDEX]),
        .transpose_issue_instruction_i(
          dispatch_payload_o[(128+HEMISPHERE_INDEX)*416 +: 416]),
        .permute_issue_valid_i(dispatch_valid_o[130+HEMISPHERE_INDEX]),
        .permute_issue_instruction_i(
          dispatch_payload_o[(130+HEMISPHERE_INDEX)*416 +: 416]),
        .east_inner_valid_i(east_edge_valid),
        .east_inner_data_i(east_edge_data),
        .east_outer_valid_o(sxm_east_valid),
        .east_outer_data_o(sxm_east_data),
        .west_outer_valid_i(
          mxm_west_valid),
        .west_outer_data_i(
          mxm_west_data),
        .west_inner_valid_o(sxm_west_valid),
        .west_inner_data_o(sxm_west_data),
        .fault_o(sxm_fault),
        .conflict_o(sxm_stream_conflict)
      );

      for (genvar local_mxm = 0; local_mxm < 2; local_mxm++) begin : gen_mxm
        localparam integer MXM_INDEX = HEMISPHERE_INDEX*2 + local_mxm;
        localparam integer LOAD_QUEUE = local_mxm == 0
          ? 104 + HEMISPHERE_INDEX : 132 + HEMISPHERE_INDEX;
        localparam integer DEQUANT_QUEUE = local_mxm == 0
          ? 106 + HEMISPHERE_INDEX : 134 + HEMISPHERE_INDEX;
        localparam integer COMPUTE_QUEUE = local_mxm == 0
          ? 108 + HEMISPHERE_INDEX : 136 + HEMISPHERE_INDEX;

        lpu_mxm_slice #(
          .LOCAL_MXM_INDEX(local_mxm),
          .ACCUMULATOR_BLOCK_COUNT(MXM_ACCUMULATOR_BLOCK_COUNT)
        ) u_mxm (
          .clk_i,
          .rst_ni,
          .run_i,
          .load_issue_valid_i(dispatch_valid_o[LOAD_QUEUE]),
          .load_issue_instruction_i(
            mxm_load_instruction[MXM_INDEX*48 +: 48]),
          .dequant_issue_valid_i(dispatch_valid_o[DEQUANT_QUEUE]),
          .dequant_issue_instruction_i(
            mxm_dequant_instruction[MXM_INDEX*16 +: 16]),
          .compute_issue_valid_i(dispatch_valid_o[COMPUTE_QUEUE]),
          .compute_issue_instruction_i(
            mxm_compute_instruction[MXM_INDEX*48 +: 48]),
          .east_from_sxm_valid_i(sxm_east_valid),
          .east_from_sxm_data_i(sxm_east_data),
          .external_west_valid_i(local_mxm == 0
            ? mem_west_edge_valid_i[HEMISPHERE_INDEX*4*32 +: 4*32]
            : '0),
          .external_west_data_i(
            mem_west_edge_data_i[HEMISPHERE_INDEX*4*32*64 +: 4*32*64]),
          .east_external_valid_o(
            local_mxm_east_valid[local_mxm*4*32 +: 4*32]),
          .east_external_data_o(
            local_mxm_east_data[local_mxm*4*32*64 +: 4*32*64]),
          .west_to_sxm_valid_o(
            local_mxm_west_valid[local_mxm*4*32 +: 4*32]),
          .west_to_sxm_data_o(
            local_mxm_west_data[local_mxm*4*32*64 +: 4*32*64]),
          .fault_o(local_mxm_fault[local_mxm]),
          .conflict_o(local_mxm_conflict[local_mxm])
        );
      end

      always_comb begin
        mxm_east_valid = local_mxm_east_valid[0 +: 4*32] &
          local_mxm_east_valid[4*32 +: 4*32];
        mxm_east_data = local_mxm_east_data[0 +: 4*32*64];
        mxm_west_valid = local_mxm_west_valid[0 +: 4*32];
        mxm_west_data = local_mxm_west_data[0 +: 4*32*64];
        mxm_stream_conflict = |local_mxm_conflict;
        for (integer stream_cell = 0; stream_cell < 4*32; stream_cell++) begin
          if (local_mxm_west_valid[4*32+stream_cell]) begin
            if (mxm_west_valid[stream_cell])
              mxm_stream_conflict = 1'b1;
            else begin
              mxm_west_valid[stream_cell] = 1'b1;
              mxm_west_data[stream_cell*64 +: 64] =
                local_mxm_west_data[(4*32+stream_cell)*64 +: 64];
            end
          end
        end
        mxm_fault = |local_mxm_fault;
      end

      assign mem_east_edge_valid_o[HEMISPHERE_INDEX*4*32 +: 4*32] = mxm_east_valid;
      assign mem_east_edge_data_o[HEMISPHERE_INDEX*4*32*64 +: 4*32*64] = mxm_east_data;
      assign mem_west_edge_valid_o[HEMISPHERE_INDEX*4*32 +: 4*32] = west_edge_valid;
      assign mem_west_edge_data_o[HEMISPHERE_INDEX*4*32*64 +: 4*32*64] = west_edge_data;
      assign mem_to_vxm_west_valid[HEMISPHERE_INDEX*4*32 +: 4*32] =
        west_edge_valid;
      assign mem_to_vxm_west_data[HEMISPHERE_INDEX*4*32*64 +: 4*32*64] =
        west_edge_data;
      assign host_mem_read_data[HEMISPHERE_INDEX*64 +: 64] = host_read;
      assign mem_fault_o[HEMISPHERE_INDEX*52*4 +: 52*4] = mem_fault;
      assign sxm_fault_o[HEMISPHERE_INDEX] = sxm_fault;
      assign mxm_fault_o[HEMISPHERE_INDEX] = mxm_fault;
      assign stream_conflict_o[HEMISPHERE_INDEX] =
        mem_stream_conflict | sxm_stream_conflict |
        mxm_stream_conflict | vxm_stream_conflict;
    end
  endgenerate

  lpu_vxm_slice u_vxm (
    .clk_i,
    .rst_ni,
    .run_i,
    .local_issue_valid_i(vxm_local_issue_valid),
    .local_issue_instruction_i(vxm_local_issue_instruction),
      .global_issue_valid_i(vxm_global_issue_valid),
      .global_issue_instruction_i(vxm_global_issue_instruction),
      .lut_config_valid_i(1'b0),
      .lut_config_bank_i('0),
      .lut_config_input_min_i('0),
      .lut_config_segment_width_i('0),
      .lut_write_valid_i(1'b0),
      .lut_write_bank_i('0),
      .lut_write_address_i('0),
      .lut_write_k_i('0),
      .lut_write_b_i('0),
      .lut_write_log_i('0),
      .west_from_mem_valid_i(mem_to_vxm_west_valid),
    .west_from_mem_data_i(mem_to_vxm_west_data),
    .external_east_valid_i(mem_east_edge_valid_i),
    .external_east_data_i(mem_east_edge_data_i),
    .east_to_mem_valid_o(vxm_to_mem_east_valid),
    .east_to_mem_data_o(vxm_to_mem_east_data),
    .fault_o(vxm_fault_o),
    .conflict_o(vxm_stream_conflict)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      cycle_o <= '0;
    else if (run_i)
      cycle_o <= cycle_o + 1'b1;
  end
endmodule
