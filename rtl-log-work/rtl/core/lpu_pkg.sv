package lpu_pkg;
  // Keep these values aligned with FTLPU-CMODEL/core/hardware_params.hpp.
  localparam integer TILE_ROWS             = 4;
  localparam integer LANES_PER_TILE        = 8;
  localparam integer PHYSICAL_VECTOR_BYTES = 32;

  localparam integer HEMISPHERES           = 2;
  localparam integer STREAMS_PER_DIRECTION = 32;
  localparam integer PACKED_STREAMS        = 64;
  localparam integer STREAM_BYTE_WIDTH     = 8;

  localparam integer MEM_SLICES_PER_HEMISPHERE = 52;
  localparam integer MEM_SLICE_COUNT            = 104;
  localparam integer MEM_SLICES_PER_GROUP       = 4;
  localparam integer MEM_GROUPS                 = 13;
  localparam integer STREAM_REGISTER_COLUMNS    = 15;
  localparam integer MEM_BOUNDARY_COLUMN        = 13;
  localparam integer MXM_BOUNDARY_COLUMN        = 14;
  localparam integer SRAM_DEPTH_ROWS            = 65536;
  localparam integer SRAM_ROW_BYTES             = 32;

  localparam integer MXMS_PER_HEMISPHERE  = 2;
  localparam integer MXM_COUNT            = 4;
  localparam integer MXM_ROWS             = 32;
  localparam integer MXM_COLUMNS          = 32;
  // One logical accumulator block holds a complete 32x32 FP32 partial sum.
  localparam integer MXM_ACCUMULATOR_BLOCK_COUNT = 32;
  localparam integer MXM_ACCUMULATOR_ROWS =
    MXM_ACCUMULATOR_BLOCK_COUNT * MXM_ROWS;
  localparam integer MXM_BLOCK_ACCUMULATOR_ROWS =
    MXM_ACCUMULATOR_BLOCK_COUNT * (MXM_ROWS / LANES_PER_TILE);
  localparam integer VXM_ALU_COUNT         = 16;
  // The replacement VXM ICU controls one mirrored pair of physical ALUs with
  // each local queue.  Global format/routing state is carried separately.
  localparam integer VXM_LOCAL_QUEUE_COUNT = 8;
  localparam integer VXM_LOCAL_MAX_INSTRUCTION_WIDTH = 7;
  localparam integer VXM_GLOBAL_CONFIG_WIDTH = 15;
  localparam integer VXM_GLOBAL_FLOW_DIRECTION_BIT = 14;
  localparam logic VXM_FLOW_LEFT_TO_RIGHT = 1'b0;
  localparam logic VXM_FLOW_RIGHT_TO_LEFT = 1'b1;
  // Zero remains the one-beat default used by existing FP16 configurations.
  // A 32-bit operand is transferred as low 16 bits followed by high 16 bits
  // over the same fixed pair of byte streams.
  localparam logic [1:0] VXM_READ_BITS_16 = 2'd0;
  localparam logic [1:0] VXM_READ_BITS_32 = 2'd1;
  localparam logic [1:0] VXM_READ_BITS_8  = 2'd2;
  localparam logic [1:0] VXM_READ_BITS_RESERVED = 2'd3;
  localparam integer VXM_Q0_INSTRUCTION_WIDTH = 6;
  localparam integer VXM_Q1_INSTRUCTION_WIDTH = 5;
  localparam integer VXM_Q2_INSTRUCTION_WIDTH = 7;
  localparam integer VXM_Q3_INSTRUCTION_WIDTH = 6;
  localparam integer VXM_Q4_INSTRUCTION_WIDTH = 7;
  localparam integer VXM_Q5_INSTRUCTION_WIDTH = 5;
  localparam integer VXM_Q6_INSTRUCTION_WIDTH = 7;
  localparam integer VXM_Q7_INSTRUCTION_WIDTH = 6;
  // One repeat-phase word accompanies each execution of a resident Tile
  // configuration. Local queues retain only execute/source fields.
  localparam integer VXM_REPEAT_CONTROL_WIDTH = 4;
  localparam integer VXM_REPEAT_LAST_ITERATION_BIT  = 0;
  localparam integer VXM_REPEAT_FIRST_ITERATION_BIT = 1;
  localparam integer VXM_REPEAT_ACCUMULATOR_BIT     = 2;
  localparam integer VXM_REPEAT_OUTPUT_ENABLE_BIT   = 3;
  typedef struct packed {
    logic output_enable;
    logic accumulator_enable;
    logic first_iteration;
    logic last_iteration;
  } vxm_repeat_control_t;
  localparam integer SXM_COUNT             = 2;

  // Encoded centrally, then propagated and latched beside every local queue.
  // External input/output format policy may evolve without changing the
  // heterogeneous local instruction widths.
  typedef struct packed {
    logic       flow_direction;
    logic [1:0] chain_length;
    logic [1:0] active_width;
    logic [1:0] compute_dtype;
    logic [1:0] lhs_read_bits;
    logic [1:0] lhs_dtype;
    logic [1:0] rhs_read_bits;
    logic [1:0] rhs_dtype;
  } vxm_global_config_t;

  // The execution interface uses one stable 32-bit container. FP16 and BF16
  // occupy data bits [15:0]; FP32 occupies all 32 bits.
  typedef enum logic [1:0] {
    VXM_FORMAT_FP16     = 2'd0,
    VXM_FORMAT_BF16     = 2'd1,
    VXM_FORMAT_FP32     = 2'd2,
    VXM_FORMAT_RESERVED = 2'd3
  } vxm_data_format_e;

  typedef enum logic [3:0] {
    VXM_LOCAL_BYPASS   = 4'd0,
    VXM_LOCAL_ADD      = 4'd1,
    VXM_LOCAL_SUBTRACT = 4'd2,
    VXM_LOCAL_MULTIPLY = 4'd3,
    VXM_LOCAL_NEGATE   = 4'd4,
    VXM_LOCAL_MAX      = 4'd5,
    // 3'd6 and 3'd7 are position-dependent special operations.
    VXM_LOCAL_SPECIAL0 = 4'd6,
    VXM_LOCAL_SPECIAL1 = 4'd7,
    VXM_LOCAL_LOG      = 4'd8
  } vxm_local_opcode_e;

  // C3/C7 compact Log: extension bit 5 plus SPECIAL0, RHS code zero.
  // Old five-bit instructions retain their encoding.
  localparam logic [5:0] VXM_LOG_INSTRUCTION = 6'b100110;
  localparam integer VXM_LOG_LUT_BANK = 3;
  localparam integer VXM_LOG_LUT_WIDTH = 50;

  // A local source field is decoded against the physical stage's current
  // chain position.  Keeping the semantic source wider than its compact
  // encoding makes the datapath MUX explicit and leaves room for new data
  // types without changing the source meaning.
  typedef enum logic [2:0] {
    VXM_SOURCE_PREVIOUS    = 3'd0,
    VXM_SOURCE_STREAM      = 3'd1,
    VXM_SOURCE_ORIGINAL    = 3'd2,
    VXM_SOURCE_AUXILIARY   = 3'd3,
    VXM_SOURCE_IMMEDIATE   = 3'd4,
    VXM_SOURCE_ACCUMULATOR = 3'd5,
    VXM_SOURCE_FEEDBACK    = 3'd6,
    VXM_SOURCE_INVALID     = 3'd7
  } vxm_operand_source_e;

  typedef enum logic [1:0] {
    VXM_CHAIN_LENGTH_2       = 2'd0,
    VXM_CHAIN_LENGTH_4       = 2'd1,
    VXM_CHAIN_LENGTH_8       = 2'd2,
    VXM_CHAIN_LENGTH_INVALID = 2'd3
  } vxm_chain_length_e;

  localparam integer VXM_SPECIAL_NONE        = 0;
  localparam integer VXM_SPECIAL_EXP         = 1;
  localparam integer VXM_SPECIAL_RECIP_RSQRT = 2;

  localparam integer MEM_INSTRUCTION_WIDTH = 47;
  localparam integer MXM_INSTRUCTION_WIDTH = 48;
  localparam integer VXM_INSTRUCTION_WIDTH = 128;
  localparam integer SXM_INSTRUCTION_WIDTH = 416;
  // VMODEL SXM packet fields.  Bits [239:208] were not consumed by the
  // existing SXM RTL; reserve [210:208] for the explicit native-compatible
  // Permute destination-tile selector.
  localparam integer SXM_OUTPUT_TILE_LSB   = 208;
  localparam integer SXM_OUTPUT_TILE_MSB   = 210;
  localparam integer SXM_OUTPUT_TILE_WIDTH =
    SXM_OUTPUT_TILE_MSB - SXM_OUTPUT_TILE_LSB + 1;
  localparam logic [SXM_OUTPUT_TILE_WIDTH-1:0] SXM_OUTPUT_TILE_ALL = 3'd4;
  localparam integer ICU_PAYLOAD_WIDTH     = SXM_INSTRUCTION_WIDTH;
  localparam integer ICU_QUEUE_COUNT       = 138;

  localparam integer MEM_QUEUE_BASE           = 0;
  localparam integer MXM_LOAD_QUEUE_BASE      = 104;
  localparam integer MXM_DEQUANT_QUEUE_BASE   = 106;
  localparam integer MXM_COMPUTE_QUEUE_BASE   = 108;
  localparam integer VXM_QUEUE_BASE           = 112;
  localparam integer VXM_GLOBAL_QUEUE         = 120;
  localparam integer VXM_RESERVED_QUEUE_BASE  = 121;
  localparam integer VXM_RESERVED_QUEUE_COUNT = 7;
  localparam integer SXM_TRANSPOSE_QUEUE_BASE = 128;
  localparam integer SXM_PERMUTE_QUEUE_BASE   = 130;
  // Preserve the original queue map for local MXM 0 in each hemisphere.
  // The second local MXM queues are appended so existing programs remain
  // binary-compatible while the topology grows to match the C model.
  localparam integer MXM_SECONDARY_LOAD_QUEUE_BASE    = 132;
  localparam integer MXM_SECONDARY_DEQUANT_QUEUE_BASE = 134;
  localparam integer MXM_SECONDARY_COMPUTE_QUEUE_BASE = 136;

  typedef enum logic {
    HEMISPHERE_EAST = 1'b0,
    HEMISPHERE_WEST = 1'b1
  } hemisphere_e;

  typedef enum logic [2:0] {
    MEM_READ       = 3'd0,
    MEM_WRITE      = 3'd1,
    MEM_READ_WRITE = 3'd2,
    MEM_GATHER     = 3'd3,
    MEM_SCATTER    = 3'd4
  } mem_opcode_e;

  typedef enum logic [1:0] {
    MXM_IW               = 2'd0,
    MXM_COMPUTE          = 2'd1,
    MXM_ACCUMULATOR_READ = 2'd2
  } mxm_opcode_e;

  typedef enum logic [4:0] {
    VXM_PASS     = 5'd0,
    VXM_ADD      = 5'd1,
    VXM_SUBTRACT = 5'd2,
    VXM_MULTIPLY = 5'd3,
    VXM_DIVIDE   = 5'd4,
    VXM_NEGATE   = 5'd5,
    VXM_ABS      = 5'd6,
    VXM_MIN      = 5'd7,
    VXM_MAX      = 5'd8,
    VXM_CLAMP    = 5'd9,
    VXM_SQUARE   = 5'd10,
    VXM_SQRT     = 5'd11,
    VXM_EXP      = 5'd12,
    VXM_LOG      = 5'd13,
    VXM_RELU     = 5'd14,
    VXM_CAST     = 5'd15
  } vxm_opcode_e;

  typedef enum logic [1:0] {
    SXM_SHIFT_SELECT = 2'd0,
    SXM_DISTRIBUTE   = 2'd1,
    SXM_TRANSPOSE    = 2'd2,
    SXM_PERMUTE      = 2'd3
  } sxm_opcode_e;

  typedef enum logic [1:0] {
    ICU_INSTRUCTION = 2'd0,
    ICU_NOP         = 2'd1,
    ICU_REPEAT      = 2'd2
  } icu_command_opcode_e;

  function automatic logic [5:0] packed_stream(
    input logic direction_west,
    input logic [4:0] index
  );
    packed_stream = {direction_west, index};
  endfunction

  function automatic integer vxm_local_instruction_width(
    input integer local_queue
  );
    case (local_queue)
      0: vxm_local_instruction_width = VXM_Q0_INSTRUCTION_WIDTH;
      1: vxm_local_instruction_width = VXM_Q1_INSTRUCTION_WIDTH;
      2: vxm_local_instruction_width = VXM_Q2_INSTRUCTION_WIDTH;
      3: vxm_local_instruction_width = VXM_Q3_INSTRUCTION_WIDTH;
      4: vxm_local_instruction_width = VXM_Q4_INSTRUCTION_WIDTH;
      5: vxm_local_instruction_width = VXM_Q5_INSTRUCTION_WIDTH;
      6: vxm_local_instruction_width = VXM_Q6_INSTRUCTION_WIDTH;
      7: vxm_local_instruction_width = VXM_Q7_INSTRUCTION_WIDTH;
      default: vxm_local_instruction_width = 1;
    endcase
  endfunction
endpackage
