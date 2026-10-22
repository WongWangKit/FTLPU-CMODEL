`timescale 1ns/1ps

// One MEM slice contains two fully independent logical banks. The Slice/SR
// boundary is abstract: it observes only the current boundary SR state and
// emits consume/injection candidates. It does not instantiate or arbitrate SR.
module mem_slice #(
    parameter P_BANKS                    = 2,
    parameter P_TILE_ROWS                = 4,
    parameter P_STREAMS_PER_DIRECTION    = 32,
    parameter P_SEGMENT_BITS             = 64,
    parameter P_MEM_BANK_DEPTH_ROWS      = 32768,
    parameter P_MEM_FAULT_BITS           = 2,
    parameter P_SLICE_FAULT_CODE_BITS    = 3
) (
    input  wire                         clk_i,
    input  wire                         rst_ni,

    // Bank is selected structurally by the ICU issue queue location. There is
    // intentionally no runtime bank-select field in the 32-bit payload.
    input  wire [P_BANKS-1:0]           bank_issue_valid_i,
    input  wire [P_BANKS*32-1:0]        bank_issue_i,

    // SR state ordering: cell = {direction, stream_index[4:0], tile[1:0]}.
    // Cells 0..127 are East; cells 128..255 are West. Each data cell is 64b.
    input  wire [2*P_STREAMS_PER_DIRECTION*P_TILE_ROWS-1:0] sr_state_valid_i,
    input  wire [2*P_STREAMS_PER_DIRECTION*P_TILE_ROWS*P_SEGMENT_BITS-1:0] sr_state_data_i,

    // One collision-feedback bit per {bank,tile} injection candidate.
    input  wire [P_BANKS*P_TILE_ROWS-1:0] producer_collision_i,

    // Per-bank consume bitmap. Within each bank the bitmap uses the same
    // {direction,stream,tile} cell ordering as sr_state_valid_i.
    output reg  [P_BANKS*2*P_STREAMS_PER_DIRECTION*P_TILE_ROWS-1:0] sr_consume_o,

    // Injection ordering: entry = bank*P_TILE_ROWS + tile.
    output wire [P_BANKS*P_TILE_ROWS-1:0] sr_inject_valid_o,
    output wire [P_BANKS*P_TILE_ROWS*P_SEGMENT_BITS-1:0] sr_inject_data_o,
    output wire [P_BANKS*P_TILE_ROWS-1:0] sr_inject_stream_dir_o,
    output wire [P_BANKS*P_TILE_ROWS*5-1:0] sr_inject_stream_idx_o,

    output reg  [P_BANKS-1:0]           mem_fault_valid_o,
    output reg  [P_BANKS*P_SLICE_FAULT_CODE_BITS-1:0] mem_fault_code_o,
    output reg  [P_BANKS-1:0]           mem_fault_bank_id_o,
    output reg  [P_BANKS-1:0]           mem_fault_tile_valid_o,
    output reg  [P_BANKS*2-1:0]         mem_fault_tile_id_o,
    output reg  [P_BANKS*15-1:0]        mem_fault_row_o,

    output wire [P_BANKS-1:0]           bank_north_cmd_valid_o,
    output wire [P_BANKS*32-1:0]        bank_north_cmd_o,
    output wire [P_BANKS-1:0]           bank_pipeline_busy_o,
    output wire                         slice_busy_o
);

    localparam P_SR_CELLS = 2 * P_STREAMS_PER_DIRECTION * P_TILE_ROWS;
    localparam [P_SLICE_FAULT_CODE_BITS-1:0] SLICE_FAULT_NONE      = 0;
    localparam [P_SLICE_FAULT_CODE_BITS-1:0] SLICE_FAULT_ISSUE     = 1;
    localparam [P_SLICE_FAULT_CODE_BITS-1:0] SLICE_FAULT_LEAF      = 2;
    localparam [P_SLICE_FAULT_CODE_BITS-1:0] SLICE_FAULT_COLLISION = 3;

    wire [P_BANKS-1:0] issue_legal;
    wire [P_BANKS-1:0] issue_fault;

    wire [P_BANKS*P_TILE_ROWS-1:0]    current_cmd_valid;
    wire [P_BANKS*P_TILE_ROWS*3-1:0]  current_cmd_opcode;
    wire [P_BANKS*P_TILE_ROWS*15-1:0] current_cmd_row;
    wire [P_BANKS*P_TILE_ROWS-1:0]    current_cmd_stream_dir;
    wire [P_BANKS*P_TILE_ROWS*5-1:0]  current_cmd_stream_idx;
    wire [P_BANKS*P_TILE_ROWS-1:0]    current_cmd_preserve;

    wire [P_BANKS*P_TILE_ROWS-1:0]    selected_stream_valid;
    wire [P_BANKS*P_TILE_ROWS*P_SEGMENT_BITS-1:0] selected_stream_data;

    wire [P_BANKS*P_TILE_ROWS-1:0]    bank_stream_consume;
    wire [P_BANKS*P_TILE_ROWS-1:0]    bank_read_valid;
    wire [P_BANKS*P_TILE_ROWS*P_SEGMENT_BITS-1:0] bank_read_data;
    wire [P_BANKS*P_TILE_ROWS-1:0]    bank_read_stream_dir;
    wire [P_BANKS*P_TILE_ROWS*5-1:0]  bank_read_stream_idx;
    wire [P_BANKS*P_TILE_ROWS-1:0]    bank_leaf_fault_valid;
    wire [P_BANKS*P_TILE_ROWS*P_MEM_FAULT_BITS-1:0] bank_leaf_fault_code;

    // Metadata captured at the same edge as leaf response/consume pulses.
    reg [P_BANKS*P_TILE_ROWS-1:0]    response_stream_dir_q;
    reg [P_BANKS*P_TILE_ROWS*5-1:0]  response_stream_idx_q;
    reg [P_BANKS*P_TILE_ROWS*15-1:0] response_row_q;

`ifndef SYNTHESIS
    initial begin
        if (P_BANKS != 2 || P_TILE_ROWS != 4 ||
            P_STREAMS_PER_DIRECTION != 32 || P_SEGMENT_BITS != 64) begin
            $display("ERROR mem_slice supports the FTLPU_DEFAULT_4T8L profile only");
            $finish;
        end
    end
`endif

    genvar bank;
    generate
        for (bank = 0; bank < P_BANKS; bank = bank + 1) begin : gen_bank
            wire [31:0] issue_command;
            wire [2:0]  issue_opcode;
            wire        supported_opcode;
            wire        read_preserve_illegal;

            assign issue_command = bank_issue_i[bank*32 +: 32];
            assign issue_opcode = issue_command[2:0];
            assign supported_opcode = (issue_opcode == 3'b000) ||
                                      (issue_opcode == 3'b001);
            assign read_preserve_illegal =
                (issue_opcode == 3'b000) && issue_command[31];
            assign issue_legal[bank] = supported_opcode &&
                                       !read_preserve_illegal &&
                                       (issue_command[14:9] == 6'b0) &&
                                       !issue_command[30];
            assign issue_fault[bank] = bank_issue_valid_i[bank] &&
                                       !issue_legal[bank];

            mem_logical_bank_column #(
                .P_TILE_ROWS(P_TILE_ROWS),
                .P_LANES_PER_TILE(8),
                .P_MEM_ELEMENT_BITS(8),
                .P_MEM_BANK_DEPTH_ROWS(P_MEM_BANK_DEPTH_ROWS),
                .P_MEM_CONTROL_HOP_CYCLES(1),
                .P_MEM_READ_TO_SR_CYCLES(1),
                .P_MEM_FAULT_BITS(P_MEM_FAULT_BITS)
            ) u_logical_bank (
                .clk_i(clk_i),
                .rst_ni(rst_ni),
                .south_cmd_valid_i(rst_ni && bank_issue_valid_i[bank] &&
                                   issue_legal[bank]),
                .south_cmd_i(issue_command),
                .stream_valid_i(selected_stream_valid[bank*P_TILE_ROWS +: P_TILE_ROWS]),
                .stream_data_i(selected_stream_data[bank*P_TILE_ROWS*P_SEGMENT_BITS +:
                                                    P_TILE_ROWS*P_SEGMENT_BITS]),
                .stream_consume_o(bank_stream_consume[bank*P_TILE_ROWS +: P_TILE_ROWS]),
                .read_valid_o(bank_read_valid[bank*P_TILE_ROWS +: P_TILE_ROWS]),
                .read_data_o(bank_read_data[bank*P_TILE_ROWS*P_SEGMENT_BITS +:
                                            P_TILE_ROWS*P_SEGMENT_BITS]),
                .read_stream_dir_o(bank_read_stream_dir[bank*P_TILE_ROWS +: P_TILE_ROWS]),
                .read_stream_idx_o(bank_read_stream_idx[bank*P_TILE_ROWS*5 +:
                                                        P_TILE_ROWS*5]),
                .fault_valid_o(bank_leaf_fault_valid[bank*P_TILE_ROWS +: P_TILE_ROWS]),
                .fault_code_o(bank_leaf_fault_code[bank*P_TILE_ROWS*P_MEM_FAULT_BITS +:
                                                   P_TILE_ROWS*P_MEM_FAULT_BITS]),
                .current_cmd_valid_o(current_cmd_valid[bank*P_TILE_ROWS +: P_TILE_ROWS]),
                .current_cmd_opcode_o(current_cmd_opcode[bank*P_TILE_ROWS*3 +:
                                                         P_TILE_ROWS*3]),
                .current_cmd_row_o(current_cmd_row[bank*P_TILE_ROWS*15 +:
                                                   P_TILE_ROWS*15]),
                .current_cmd_stream_dir_o(current_cmd_stream_dir[bank*P_TILE_ROWS +:
                                                                 P_TILE_ROWS]),
                .current_cmd_stream_idx_o(current_cmd_stream_idx[bank*P_TILE_ROWS*5 +:
                                                                 P_TILE_ROWS*5]),
                .current_cmd_preserve_o(current_cmd_preserve[bank*P_TILE_ROWS +:
                                                             P_TILE_ROWS]),
                .north_cmd_valid_o(bank_north_cmd_valid_o[bank]),
                .north_cmd_o(bank_north_cmd_o[bank*32 +: 32]),
                .pipeline_busy_o(bank_pipeline_busy_o[bank])
            );
        end
    endgenerate

    // Every active tile independently selects its own direction/stream/tile
    // SR cell. Continuous command waves therefore do not share a selector.
    genvar select_bank;
    genvar select_tile;
    generate
        for (select_bank = 0; select_bank < P_BANKS;
             select_bank = select_bank + 1) begin : gen_select_bank
            for (select_tile = 0; select_tile < P_TILE_ROWS;
                 select_tile = select_tile + 1) begin : gen_select_tile
                localparam ENTRY = select_bank*P_TILE_ROWS + select_tile;
                localparam [1:0] TILE_ID = select_tile;
                wire [7:0] sr_cell_index;

                assign sr_cell_index = {
                    current_cmd_stream_dir[ENTRY],
                    current_cmd_stream_idx[ENTRY*5 +: 5],
                    TILE_ID
                };
                assign selected_stream_valid[ENTRY] =
                    sr_state_valid_i[sr_cell_index];
                assign selected_stream_data[ENTRY*P_SEGMENT_BITS +: P_SEGMENT_BITS] =
                    sr_state_data_i[sr_cell_index*P_SEGMENT_BITS +: P_SEGMENT_BITS];
            end
        end
    endgenerate

    assign sr_inject_valid_o      = bank_read_valid;
    assign sr_inject_data_o       = bank_read_data;
    assign sr_inject_stream_dir_o = bank_read_stream_dir;
    assign sr_inject_stream_idx_o = bank_read_stream_idx;
    assign slice_busy_o           = |bank_pipeline_busy_o;

    integer capture_entry;
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            response_stream_dir_q <= {P_BANKS*P_TILE_ROWS{1'b0}};
            response_stream_idx_q <= {P_BANKS*P_TILE_ROWS*5{1'b0}};
            response_row_q        <= {P_BANKS*P_TILE_ROWS*15{1'b0}};
        end else begin
            for (capture_entry = 0;
                 capture_entry < P_BANKS*P_TILE_ROWS;
                 capture_entry = capture_entry + 1) begin
                if (current_cmd_valid[capture_entry]) begin
                    response_stream_dir_q[capture_entry] <=
                        current_cmd_stream_dir[capture_entry];
                    response_stream_idx_q[capture_entry*5 +: 5] <=
                        current_cmd_stream_idx[capture_entry*5 +: 5];
                    response_row_q[capture_entry*15 +: 15] <=
                        current_cmd_row[capture_entry*15 +: 15];
                end
            end
        end
    end

    // A legal normal Write consumes the current SR segment in the same cycle
    // in which the MEM leaf samples it.  Deriving this candidate from the
    // existing current-command view keeps the SRF adapter combinational and
    // prevents the segment from taking another passive propagation hop.
    // The leaf's registered bank_stream_consume remains its local event pulse;
    // it is not used as the SR boundary timing source in this variant.
    integer consume_bank;
    integer consume_tile;
    integer consume_entry;
    integer consume_index;
    always @* begin
        sr_consume_o = {P_BANKS*P_SR_CELLS{1'b0}};
        if (rst_ni) begin
            for (consume_bank = 0; consume_bank < P_BANKS;
                 consume_bank = consume_bank + 1) begin
                for (consume_tile = 0; consume_tile < P_TILE_ROWS;
                     consume_tile = consume_tile + 1) begin
                    consume_entry = consume_bank*P_TILE_ROWS + consume_tile;
                    consume_index = consume_bank*P_SR_CELLS +
                        (current_cmd_stream_dir[consume_entry] ?
                         P_STREAMS_PER_DIRECTION*P_TILE_ROWS : 0) +
                        current_cmd_stream_idx[consume_entry*5 +: 5]*P_TILE_ROWS +
                        consume_tile;
                    if (current_cmd_valid[consume_entry] &&
                        (current_cmd_opcode[consume_entry*3 +: 3] == 3'b001) &&
                        !current_cmd_preserve[consume_entry] &&
                        selected_stream_valid[consume_entry]) begin
                        sr_consume_o[consume_index] = 1'b1;
                    end
                end
            end
        end
    end

    // Per-bank fault aggregation. Diagnostic metadata priority is issue
    // decode, then lowest-tile collision, then lowest-tile leaf fault. This
    // priority reports one representative fault only; it never chooses a data
    // producer, suppresses an injection, or changes bank execution.
    integer fault_bank;
    integer fault_tile;
    integer fault_entry;
    always @* begin
        mem_fault_valid_o      = {P_BANKS{1'b0}};
        mem_fault_code_o       = {P_BANKS*P_SLICE_FAULT_CODE_BITS{1'b0}};
        mem_fault_bank_id_o    = {P_BANKS{1'b0}};
        mem_fault_tile_valid_o = {P_BANKS{1'b0}};
        mem_fault_tile_id_o    = {P_BANKS*2{1'b0}};
        mem_fault_row_o        = {P_BANKS*15{1'b0}};

        for (fault_bank = 0; fault_bank < P_BANKS;
             fault_bank = fault_bank + 1) begin
            mem_fault_bank_id_o[fault_bank] = (fault_bank != 0);
            if (rst_ni && issue_fault[fault_bank]) begin
                mem_fault_valid_o[fault_bank] = 1'b1;
                mem_fault_code_o[fault_bank*P_SLICE_FAULT_CODE_BITS +:
                                 P_SLICE_FAULT_CODE_BITS] = SLICE_FAULT_ISSUE;
                mem_fault_row_o[fault_bank*15 +: 15] =
                    bank_issue_i[fault_bank*32 + 15 +: 15];
            end

            for (fault_tile = 0; fault_tile < P_TILE_ROWS;
                 fault_tile = fault_tile + 1) begin
                fault_entry = fault_bank*P_TILE_ROWS + fault_tile;
                if (rst_ni && !mem_fault_valid_o[fault_bank] &&
                    bank_read_valid[fault_entry] &&
                    producer_collision_i[fault_entry]) begin
                    mem_fault_valid_o[fault_bank] = 1'b1;
                    mem_fault_code_o[fault_bank*P_SLICE_FAULT_CODE_BITS +:
                                     P_SLICE_FAULT_CODE_BITS] =
                        SLICE_FAULT_COLLISION;
                    mem_fault_tile_valid_o[fault_bank] = 1'b1;
                    mem_fault_tile_id_o[fault_bank*2 +: 2] = fault_tile;
                    mem_fault_row_o[fault_bank*15 +: 15] =
                        response_row_q[fault_entry*15 +: 15];
                end
            end

            for (fault_tile = 0; fault_tile < P_TILE_ROWS;
                 fault_tile = fault_tile + 1) begin
                fault_entry = fault_bank*P_TILE_ROWS + fault_tile;
                if (rst_ni && !mem_fault_valid_o[fault_bank] &&
                    bank_leaf_fault_valid[fault_entry]) begin
                    mem_fault_valid_o[fault_bank] = 1'b1;
                    mem_fault_code_o[fault_bank*P_SLICE_FAULT_CODE_BITS +:
                                     P_SLICE_FAULT_CODE_BITS] = SLICE_FAULT_LEAF;
                    mem_fault_tile_valid_o[fault_bank] = 1'b1;
                    mem_fault_tile_id_o[fault_bank*2 +: 2] = fault_tile;
                    mem_fault_row_o[fault_bank*15 +: 15] =
                        response_row_q[fault_entry*15 +: 15];
                end
            end
        end
    end

endmodule
