module lpu_icu #(
  parameter integer QUEUE_DEPTH = 16
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic run_i,

  input  logic         enqueue_valid_i,
  output logic         enqueue_ready_o,
  input  logic [7:0]   enqueue_queue_i,
  input  logic         enqueue_is_instruction_i,
  input  logic [31:0]  enqueue_command_i,
  input  logic [415:0] enqueue_payload_i,

  output logic [137:0]       issue_valid_o,
  output logic [138*416-1:0] issue_payload_o,
  output logic [137:0]       queue_fault_o
);
  logic [137:0] queue_ready;
  logic [137:0] queue_enqueue_valid;

  always_comb begin
    queue_enqueue_valid = '0;
    enqueue_ready_o = 1'b0;
    if (enqueue_queue_i < 8'd138) begin
      queue_enqueue_valid[enqueue_queue_i] = enqueue_valid_i;
      enqueue_ready_o = queue_ready[enqueue_queue_i];
    end
  end

  generate
    for (genvar queue = 0; queue < 138; queue++) begin : gen_queue
      if ((queue < lpu_pkg::VXM_QUEUE_BASE) ||
          (queue >= lpu_pkg::SXM_TRANSPOSE_QUEUE_BASE)) begin : gen_standard
        logic [$clog2(QUEUE_DEPTH+1)-1:0] unused_level;
        lpu_icu_queue #(
          .PAYLOAD_WIDTH(416),
          .DEPTH(QUEUE_DEPTH),
          .APPLY_MEM_STRIDE(queue < 104)
        ) u_queue (
          .clk_i,
          .rst_ni,
          .run_i,
          .enqueue_valid_i(queue_enqueue_valid[queue]),
          .enqueue_ready_o(queue_ready[queue]),
          .enqueue_is_instruction_i,
          .enqueue_command_i,
          .enqueue_payload_i,
          .issue_valid_o(issue_valid_o[queue]),
          .issue_payload_o(issue_payload_o[queue*416 +: 416]),
          .fault_o(queue_fault_o[queue]),
          .level_o(unused_level),
          .idle_o()
        );
      end else if (queue < lpu_pkg::VXM_GLOBAL_QUEUE) begin : gen_vxm_local
        localparam integer LOCAL_QUEUE = queue - lpu_pkg::VXM_QUEUE_BASE;
        localparam integer LOCAL_WIDTH =
          lpu_pkg::vxm_local_instruction_width(LOCAL_QUEUE);
        // Q3/Q7 include bit 5 of the backward-compatible Log extension.
        logic [LOCAL_WIDTH-1:0] local_issue_payload;
        logic [$clog2(QUEUE_DEPTH+1)-1:0] unused_level;

        lpu_icu_queue #(
          .PAYLOAD_WIDTH(LOCAL_WIDTH),
          .DEPTH(QUEUE_DEPTH),
          .APPLY_MEM_STRIDE(1'b0)
        ) u_queue (
          .clk_i,
          .rst_ni,
          .run_i,
          .enqueue_valid_i(queue_enqueue_valid[queue]),
          .enqueue_ready_o(queue_ready[queue]),
          .enqueue_is_instruction_i,
          .enqueue_command_i,
          .enqueue_payload_i(enqueue_payload_i[LOCAL_WIDTH-1:0]),
          .issue_valid_o(issue_valid_o[queue]),
          .issue_payload_o(local_issue_payload),
          .fault_o(queue_fault_o[queue]),
          .level_o(unused_level),
          .idle_o()
        );

        always_comb begin
          issue_payload_o[queue*416 +: 416] = '0;
          issue_payload_o[queue*416 +: LOCAL_WIDTH] = local_issue_payload;
        end
      end else if (queue == lpu_pkg::VXM_GLOBAL_QUEUE) begin : gen_vxm_global
        logic [lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH-1:0] global_issue_payload;
        logic [$clog2(QUEUE_DEPTH+1)-1:0] unused_level;

        lpu_icu_queue #(
          .PAYLOAD_WIDTH(lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH),
          .DEPTH(QUEUE_DEPTH),
          .APPLY_MEM_STRIDE(1'b0)
        ) u_queue (
          .clk_i,
          .rst_ni,
          .run_i,
          .enqueue_valid_i(queue_enqueue_valid[queue]),
          .enqueue_ready_o(queue_ready[queue]),
          .enqueue_is_instruction_i,
          .enqueue_command_i,
          .enqueue_payload_i(
            enqueue_payload_i[lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH-1:0]),
          .issue_valid_o(issue_valid_o[queue]),
          .issue_payload_o(global_issue_payload),
          .fault_o(queue_fault_o[queue]),
          .level_o(unused_level),
          .idle_o()
        );

        always_comb begin
          issue_payload_o[queue*416 +: 416] = '0;
          issue_payload_o[queue*416 +: lpu_pkg::VXM_GLOBAL_CONFIG_WIDTH] =
            global_issue_payload;
        end
      end else begin : gen_vxm_reserved
        assign queue_ready[queue] = 1'b0;
        assign issue_valid_o[queue] = 1'b0;
        assign issue_payload_o[queue*416 +: 416] = '0;
        assign queue_fault_o[queue] = 1'b0;
      end
    end
  endgenerate
endmodule
