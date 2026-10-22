`timescale 1ns/1ps

// Stateless conversion from the VMODEL 47-bit MEM issue packet to the native
// 32-bit command accepted by the Phase 2A MEM hierarchy.  This adapter only
// supports VMODEL MEM_READ and MEM_WRITE; unsupported commands fail closed.
module lpu_mem_command_adapter (
  input  logic        issue_valid_i,
  input  logic [46:0] issue_instruction_i,
  output logic        native_valid_o,
  output logic [31:0] native_command_o,
  output logic        command_fault_o
);
  localparam logic [2:0] VMODEL_MEM_READ  = 3'd0;
  localparam logic [2:0] VMODEL_MEM_WRITE = 3'd1;

  always @* begin
    native_valid_o = 1'b0;
    native_command_o = '0;
    command_fault_o = 1'b0;

    if (issue_valid_i) begin
      // VMODEL [46:31] is write_address.  Its required zero value for Read
      // and Write is checked explicitly rather than silently discarded.
      // VMODEL address[15] cannot fit the native 15-bit row field.
      if ((issue_instruction_i[2:0] == VMODEL_MEM_READ ||
           issue_instruction_i[2:0] == VMODEL_MEM_WRITE) &&
          (issue_instruction_i[46:31] == '0) &&
          !issue_instruction_i[30]) begin
        native_valid_o = 1'b1;
        native_command_o[2:0] = issue_instruction_i[2:0];
        native_command_o[8:3] = issue_instruction_i[8:3];
        native_command_o[29:15] = issue_instruction_i[29:15];
        // Native [14:9] is reserved and [31] is preserve.  VMODEL Read/Write
        // have no equivalent preserve bit, so both are fixed to zero.
      end else begin
        command_fault_o = 1'b1;
      end
    end
  end
endmodule
