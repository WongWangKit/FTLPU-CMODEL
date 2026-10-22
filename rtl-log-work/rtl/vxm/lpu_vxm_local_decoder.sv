module lpu_vxm_local_decoder #(
  parameter integer LOCAL_QUEUE = 0,
  parameter integer PHYSICAL_STAGE = LOCAL_QUEUE
) (
  input  logic [lpu_pkg::VXM_LOCAL_MAX_INSTRUCTION_WIDTH-1:0]
    instruction_i,
  input  logic [1:0] chain_length_i,

  output logic [3:0] opcode_o,
  output logic [2:0] lhs_source_o,
  output logic [2:0] rhs_source_o,
  output logic       chain_head_o,
  output logic       chain_tail_o,
  output logic       illegal_instruction_o
);
  import lpu_pkg::*;

  logic [1:0] lhs_code;
  logic [1:0] rhs_code;
  logic       upper_bits_nonzero;
  logic       opcode_illegal;
  logic       source_illegal;

  always_comb begin
    opcode_o = {1'b0, instruction_i[2:0]};
    lhs_code = 2'b00;
    rhs_code = 2'b00;
    upper_bits_nonzero = 1'b0;
    opcode_illegal = 1'b0;
    source_illegal = 1'b0;
    lhs_source_o = VXM_SOURCE_INVALID;
    rhs_source_o = VXM_SOURCE_INVALID;
    chain_head_o = 1'b0;
    chain_tail_o = 1'b0;

    case (chain_length_i)
      VXM_CHAIN_LENGTH_2: begin
        chain_head_o = ((PHYSICAL_STAGE % 2) == 0);
        chain_tail_o = ((PHYSICAL_STAGE % 2) == 1);
      end
      VXM_CHAIN_LENGTH_4: begin
        chain_head_o = ((PHYSICAL_STAGE % 4) == 0);
        chain_tail_o = ((PHYSICAL_STAGE % 4) == 3);
      end
      VXM_CHAIN_LENGTH_8: begin
        chain_head_o = ((PHYSICAL_STAGE % 8) == 0);
        chain_tail_o = ((PHYSICAL_STAGE % 8) == 7);
      end
      default: begin
        chain_head_o = 1'b0;
        chain_tail_o = 1'b0;
        source_illegal = 1'b1;
      end
    endcase

    // Compact instruction layouts:
    // Q0: op[2:0], lhs[4:3], rhs[5]
    // Q1/3/5/7: op[2:0], rhs[4:3]
    // Q2/4/6: op[2:0], lhs[4:3], rhs[6:5]
    case (LOCAL_QUEUE)
      0: begin
        lhs_code = instruction_i[4:3];
        rhs_code = {1'b0, instruction_i[5]};
        upper_bits_nonzero = instruction_i[6];
      end
      1, 5: begin
        rhs_code = instruction_i[4:3];
        upper_bits_nonzero = |instruction_i[6:5];
      end
      3, 7: begin
        rhs_code = instruction_i[4:3];
        upper_bits_nonzero = instruction_i[6];
        if (instruction_i[5]) begin
          opcode_o = VXM_LOCAL_LOG;
          if (instruction_i[5:0] != VXM_LOG_INSTRUCTION)
            source_illegal = 1'b1;
        end
      end
      2, 4, 6: begin
        lhs_code = instruction_i[4:3];
        rhs_code = instruction_i[6:5];
      end
      default: begin
        upper_bits_nonzero = 1'b1;
        source_illegal = 1'b1;
      end
    endcase

    if (chain_head_o) begin
      case (lhs_code)
        2'd0: lhs_source_o = VXM_SOURCE_STREAM;
        2'd1: lhs_source_o = VXM_SOURCE_IMMEDIATE;
        2'd2: lhs_source_o = VXM_SOURCE_FEEDBACK;
        default: begin
          lhs_source_o = VXM_SOURCE_INVALID;
          source_illegal = 1'b1;
        end
      endcase
      case (rhs_code)
        2'd0: rhs_source_o = VXM_SOURCE_STREAM;
        2'd1: rhs_source_o = VXM_SOURCE_IMMEDIATE;
        default: begin
          rhs_source_o = VXM_SOURCE_INVALID;
          source_illegal = 1'b1;
        end
      endcase
    end else begin
      lhs_source_o = VXM_SOURCE_PREVIOUS;
      // LHS bits exist on Q2/Q4/Q6 only because these queues can become a
      // chain head.  They must be zero while that stage is internal.
      if ((LOCAL_QUEUE == 2 || LOCAL_QUEUE == 4 || LOCAL_QUEUE == 6) &&
          (lhs_code != 2'd0))
        source_illegal = 1'b1;
      case (rhs_code)
        2'd0: rhs_source_o = VXM_SOURCE_ORIGINAL;
        2'd1: rhs_source_o = VXM_SOURCE_AUXILIARY;
        2'd2: rhs_source_o = VXM_SOURCE_IMMEDIATE;
        2'd3: begin
          if ((LOCAL_QUEUE % 2) == 1)
            rhs_source_o = VXM_SOURCE_ACCUMULATOR;
          else begin
            rhs_source_o = VXM_SOURCE_INVALID;
            source_illegal = 1'b1;
          end
        end
      endcase
    end

    // C0/C2 positions are basic-only. C1 has Exp at opcode 6; C3 has
    // Reciprocal/Rsqrt at opcodes 6/7.
    if ((LOCAL_QUEUE % 2) == 0)
      opcode_illegal = (opcode_o > VXM_LOCAL_MAX);
    else if ((LOCAL_QUEUE == 1) || (LOCAL_QUEUE == 5))
      opcode_illegal = (opcode_o == VXM_LOCAL_SPECIAL1);

    illegal_instruction_o = upper_bits_nonzero || opcode_illegal ||
      source_illegal;
  end
endmodule
