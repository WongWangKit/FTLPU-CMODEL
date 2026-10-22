module lpu_icu_queue #(
  parameter integer      PAYLOAD_WIDTH    = 416,
  parameter integer      DEPTH            = 16,
  parameter bit          APPLY_MEM_STRIDE = 1'b0
) (
  input  logic                     clk_i,
  input  logic                     rst_ni,
  input  logic                     run_i,

  input  logic                     enqueue_valid_i,
  output logic                     enqueue_ready_o,
  input  logic                     enqueue_is_instruction_i,
  input  logic [31:0]              enqueue_command_i,
  input  logic [PAYLOAD_WIDTH-1:0] enqueue_payload_i,

  output logic                     issue_valid_o,
  output logic [PAYLOAD_WIDTH-1:0] issue_payload_o,
  output logic                     fault_o,
  output logic [$clog2(DEPTH+1)-1:0] level_o,
  output logic                     idle_o
);
  localparam integer PTR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
  localparam integer LEVEL_WIDTH = $clog2(DEPTH + 1);
  localparam integer STRIDE_PAYLOAD_WIDTH =
    (PAYLOAD_WIDTH < 31) ? 31 : PAYLOAD_WIDTH;

  logic [1:0] kind_mem [0:DEPTH-1];
  logic [PAYLOAD_WIDTH-1:0] payload_mem [0:DEPTH-1];
  logic [31:0] command_mem [0:DEPTH-1];
  logic [PTR_WIDTH-1:0] write_ptr_q;
  logic [PTR_WIDTH-1:0] read_ptr_q;
  logic [LEVEL_WIDTH-1:0] level_q;

  logic [PAYLOAD_WIDTH-1:0] last_instruction_q;
  logic [PAYLOAD_WIDTH-1:0] repeat_instruction_q;
  logic last_instruction_valid_q;
  logic [29:0] nop_remaining_q;
  logic [9:0] repeat_remaining_q;
  logic [7:0] repeat_interval_q;
  logic [7:0] repeat_cooldown_q;
  logic signed [11:0] repeat_stride_q;
  logic [9:0] repeat_index_q;
  logic enqueue_fire;
  logic dequeue_fire;

  function automatic [PTR_WIDTH-1:0] increment_ptr(input [PTR_WIDTH-1:0] ptr);
    if (ptr == DEPTH-1) increment_ptr = '0;
    else increment_ptr = ptr + 1'b1;
  endfunction

  function automatic [PAYLOAD_WIDTH-1:0] apply_stride(
    input [PAYLOAD_WIDTH-1:0] payload,
    input logic signed [11:0] stride,
    input logic [9:0] index
  );
    logic [STRIDE_PAYLOAD_WIDTH-1:0] wide_payload;
    logic signed [22:0] delta;
    logic signed [23:0] address;
    begin
      wide_payload = '0;
      wide_payload[PAYLOAD_WIDTH-1:0] = payload;
      if (APPLY_MEM_STRIDE) begin
        delta = stride * $signed({1'b0, index});
        address = $signed({1'b0, wide_payload[30:15]}) + delta;
        wide_payload[30:15] = address[15:0];
      end
      apply_stride = wide_payload[PAYLOAD_WIDTH-1:0];
    end
  endfunction

  assign dequeue_fire = run_i && (nop_remaining_q == 0) &&
    (repeat_remaining_q == 0) && (level_q != 0);
  // The host may refill an executing queue.  Accepting an enqueue alongside
  // a dequeue also keeps a full queue streaming without a bubble.
  assign enqueue_ready_o = (level_q < DEPTH) || dequeue_fire;
  assign enqueue_fire = enqueue_valid_i && enqueue_ready_o;
  assign level_o = level_q;
  assign idle_o = (level_q == 0) && (nop_remaining_q == 0) &&
    (repeat_remaining_q == 0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      write_ptr_q               <= '0;
      read_ptr_q                <= '0;
      level_q                   <= '0;
      last_instruction_q        <= '0;
      repeat_instruction_q      <= '0;
      last_instruction_valid_q  <= 1'b0;
      nop_remaining_q           <= '0;
      repeat_remaining_q        <= '0;
      repeat_interval_q         <= 8'd1;
      repeat_cooldown_q         <= '0;
      repeat_stride_q           <= '0;
      repeat_index_q            <= '0;
      issue_valid_o             <= 1'b0;
      issue_payload_o           <= '0;
      fault_o                   <= 1'b0;
    end else begin
      issue_valid_o <= 1'b0;
      if (enqueue_fire) begin
        kind_mem[write_ptr_q] <= enqueue_is_instruction_i
          ? 2'd0 : enqueue_command_i[1:0];
        payload_mem[write_ptr_q] <= enqueue_payload_i;
        command_mem[write_ptr_q] <= enqueue_command_i;
        write_ptr_q <= increment_ptr(write_ptr_q);
      end

      if (!run_i) begin
        if (enqueue_fire)
          level_q <= level_q + 1'b1;
      end else if (nop_remaining_q != 0) begin
        nop_remaining_q <= nop_remaining_q - 1'b1;
        if (enqueue_fire)
          level_q <= level_q + 1'b1;
      end else if (repeat_remaining_q != 0) begin
        if (enqueue_fire)
          level_q <= level_q + 1'b1;
        if (repeat_cooldown_q != 0) begin
          repeat_cooldown_q <= repeat_cooldown_q - 1'b1;
        end else begin
          issue_valid_o <= 1'b1;
          issue_payload_o <= apply_stride(
            repeat_instruction_q, repeat_stride_q, repeat_index_q);
          last_instruction_q <= apply_stride(
            repeat_instruction_q, repeat_stride_q, repeat_index_q);
          last_instruction_valid_q <= 1'b1;
          repeat_index_q <= repeat_index_q + 1'b1;
          repeat_remaining_q <= repeat_remaining_q - 1'b1;
          if (repeat_remaining_q > 1)
            repeat_cooldown_q <= repeat_interval_q - 1'b1;
        end
      end else if (level_q != 0) begin
        read_ptr_q <= increment_ptr(read_ptr_q);
        if (!enqueue_fire)
          level_q <= level_q - 1'b1;
        case (kind_mem[read_ptr_q])
          2'd0: begin
            issue_valid_o <= 1'b1;
            issue_payload_o <= payload_mem[read_ptr_q];
            last_instruction_q <= payload_mem[read_ptr_q];
            last_instruction_valid_q <= 1'b1;
          end
          2'd1: begin
            if (command_mem[read_ptr_q][31:2] != 0)
              nop_remaining_q <= command_mem[read_ptr_q][31:2] - 1'b1;
          end
          2'd2: begin
            if (!last_instruction_valid_q ||
                (command_mem[read_ptr_q][19:12] == 0)) begin
              fault_o <= 1'b1;
            end else if (command_mem[read_ptr_q][11:2] != 0) begin
              repeat_instruction_q <= last_instruction_q;
              repeat_stride_q <= command_mem[read_ptr_q][31:20];
              repeat_interval_q <= command_mem[read_ptr_q][19:12];
              repeat_index_q <= 10'd1;
              if (command_mem[read_ptr_q][19:12] == 1) begin
                issue_valid_o <= 1'b1;
                issue_payload_o <= apply_stride(
                  last_instruction_q,
                  command_mem[read_ptr_q][31:20],
                  10'd1);
                last_instruction_q <= apply_stride(
                  last_instruction_q,
                  command_mem[read_ptr_q][31:20],
                  10'd1);
                repeat_index_q <= 10'd2;
                repeat_remaining_q <= command_mem[read_ptr_q][11:2] - 1'b1;
                repeat_cooldown_q <= '0;
              end else begin
                repeat_remaining_q <= command_mem[read_ptr_q][11:2];
                repeat_cooldown_q <= command_mem[read_ptr_q][19:12] - 2'd2;
              end
            end
          end
          default: fault_o <= 1'b1;
        endcase
      end else if (enqueue_fire) begin
        level_q <= level_q + 1'b1;
      end
    end
  end
endmodule
