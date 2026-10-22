`timescale 1ns/1ps

// One physical MEM storage leaf for a fixed
// (hemisphere, mem_slice, bank, superlane) coordinate.
//
// The SRAM array is single-port: at most one complete-row Read or Write is
// accepted in a cycle. Reset clears only control/output state; SRAM contents
// are intentionally not reset.
module mem_bank_superlane_leaf #(
    parameter P_MEM_BANK_DEPTH_ROWS = 32768,
    parameter P_LANES_PER_TILE      = 8,
    parameter P_MEM_ELEMENT_BITS    = 8,
    parameter P_MEM_READ_TO_SR_CYCLES = 1,
    parameter P_MEM_FAULT_BITS      = 2
) (
    input  wire                             clk_i,
    input  wire                             rst_ni,

    input  wire                             cmd_valid_i,
    input  wire [2:0]                       cmd_opcode_i,
    input  wire [14:0]                      cmd_row_i,
    input  wire                             cmd_stream_dir_i,
    input  wire [4:0]                       cmd_stream_idx_i,
    input  wire                             cmd_preserve_i,

    input  wire                             stream_valid_i,
    input  wire [P_LANES_PER_TILE*P_MEM_ELEMENT_BITS-1:0] stream_data_i,

    output reg                              stream_consume_o,

    output reg                              read_valid_o,
    output reg  [P_LANES_PER_TILE*P_MEM_ELEMENT_BITS-1:0] read_data_o,
    output reg                              read_stream_dir_o,
    output reg  [4:0]                       read_stream_idx_o,

    output reg                              fault_valid_o,
    output reg  [P_MEM_FAULT_BITS-1:0]      fault_code_o
);

    localparam P_ROW_BITS = P_LANES_PER_TILE * P_MEM_ELEMENT_BITS;

    localparam [2:0] MEM_OPCODE_READ  = 3'b000;
    localparam [2:0] MEM_OPCODE_WRITE = 3'b001;

    // IMPLEMENTATION CHOICE: the specification does not freeze numeric fault
    // encoding. These symbolic values are local to this first leaf model.
    localparam [P_MEM_FAULT_BITS-1:0] MEM_FAULT_NONE           = 0;
    localparam [P_MEM_FAULT_BITS-1:0] MEM_FAULT_ILLEGAL_OPCODE = 1;
    localparam [P_MEM_FAULT_BITS-1:0] MEM_FAULT_READ_PRESERVE  = 2;
    localparam [P_MEM_FAULT_BITS-1:0] MEM_FAULT_INVALID_WRITE  = 3;

    // Behavioral model of one 32768 x 64-bit SRAM in the default profile.
    // Do not add this array to the reset branch.
    reg [P_ROW_BITS-1:0] mem_array [0:P_MEM_BANK_DEPTH_ROWS-1];

    // P_MEM_READ_TO_SR_CYCLES is fixed to 1 for this implementation stage.
    // A legal Read is sampled at the active edge and produces one registered
    // response pulse carrying data and stream metadata.
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            stream_consume_o  <= 1'b0;
            read_valid_o      <= 1'b0;
            read_data_o       <= {P_ROW_BITS{1'b0}};
            read_stream_dir_o <= 1'b0;
            read_stream_idx_o <= 5'b0;
            fault_valid_o     <= 1'b0;
            fault_code_o      <= MEM_FAULT_NONE;
        end else begin
            // Event outputs are one-cycle pulses.
            stream_consume_o <= 1'b0;
            read_valid_o     <= 1'b0;
            fault_valid_o    <= 1'b0;
            fault_code_o     <= MEM_FAULT_NONE;

            if (cmd_valid_i) begin
                case (cmd_opcode_i)
                    MEM_OPCODE_READ: begin
                        if (cmd_preserve_i) begin
                            // preserve is not legal for Read.
                            fault_valid_o <= 1'b1;
                            fault_code_o  <= MEM_FAULT_READ_PRESERVE;
                        end else begin
                            read_data_o       <= mem_array[cmd_row_i];
                            read_stream_dir_o <= cmd_stream_dir_i;
                            read_stream_idx_o <= cmd_stream_idx_i;
                            read_valid_o      <= 1'b1;
                        end
                    end

                    MEM_OPCODE_WRITE: begin
                        if (cmd_preserve_i) begin
                            // WriteTap: invalid input is a tile-local no-op.
                            if (stream_valid_i) begin
                                mem_array[cmd_row_i] <= stream_data_i;
                            end
                        end else if (stream_valid_i) begin
                            // Normal Write consumes the selected stream.
                            mem_array[cmd_row_i] <= stream_data_i;
                            stream_consume_o     <= 1'b1;
                        end else begin
                            // Never replace missing stream data with zeros.
                            fault_valid_o <= 1'b1;
                            fault_code_o  <= MEM_FAULT_INVALID_WRITE;
                        end
                    end

                    default: begin
                        // 3'b010 is retired/illegal; 3'b011..111 are reserved.
                        fault_valid_o <= 1'b1;
                        fault_code_o  <= MEM_FAULT_ILLEGAL_OPCODE;
                    end
                endcase
            end
        end
    end

endmodule
