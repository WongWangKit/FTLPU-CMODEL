"""Apply the Log extension to the isolated RTL work copy only."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2] / 'rtl-log-work'

def edit(name, fn):
    p = ROOT / name
    old = p.read_text(encoding='utf-8')
    new = fn(old)
    if new != old:
        p.write_text(new, encoding='utf-8', newline='\n')

def replace(name, old, new, count=1):
    def change(s):
        assert s.count(old) >= count, (name, old)
        return s.replace(old, new, count)
    edit(name, change)

pkg='rtl/core/lpu_pkg.sv'
for q in (3,7):
    replace(pkg, f'VXM_Q{q}_INSTRUCTION_WIDTH = 5;', f'VXM_Q{q}_INSTRUCTION_WIDTH = 6;')
replace(pkg, 'typedef enum logic [2:0] {\n    VXM_LOCAL_BYPASS', 'typedef enum logic [3:0] {\n    VXM_LOCAL_BYPASS')
replace(pkg, "VXM_LOCAL_SPECIAL1 = 3'd7", "VXM_LOCAL_SPECIAL1 = 3'd7,\n    VXM_LOCAL_LOG      = 4'd8")
replace(pkg, '  // A local source field', "  // C3/C7 compact Log: extension bit 5 plus SPECIAL0, RHS code zero.\n  // Old five-bit instructions retain their encoding.\n  localparam logic [5:0] VXM_LOG_INSTRUCTION = 6'b100110;\n  localparam integer VXM_LOG_LUT_BANK = 3;\n  localparam integer VXM_LOG_LUT_WIDTH = 50;\n\n  // A local source field")

for name in ('local_decoder','datapath_mux','datapath_stage','execution_stage','alu','special_alu'):
    file=f'rtl/vxm/lpu_vxm_{name}.sv'
    edit(file, lambda s: re.sub(r'logic \[2:0\](\s+opcode(?:_i|_o)?\b)', r'logic [3:0]\1', s))
replace('rtl/vxm/lpu_vxm_datapath_stage.sv', '[LANES*3-1:0] lane_opcodes', '[LANES*4-1:0] lane_opcodes')
replace('rtl/vxm/lpu_vxm_datapath_stage.sv', 'lane_opcodes[lane*3 +: 3]', 'lane_opcodes[lane*4 +: 4]')
replace('rtl/vxm/lpu_vxm_datapath_stage.sv', 'opcode_o = lane_opcodes[2:0]', 'opcode_o = lane_opcodes[3:0]')
replace('rtl/vxm/lpu_vxm_local_decoder.sv', 'opcode_o = instruction_i[2:0];', "opcode_o = {1'b0, instruction_i[2:0]};")
replace('rtl/vxm/lpu_vxm_local_decoder.sv', '      1, 3, 5, 7: begin', '      1, 5: begin')
replace('rtl/vxm/lpu_vxm_local_decoder.sv', '      2, 4, 6: begin', '''      3, 7: begin
        rhs_code = instruction_i[4:3];
        upper_bits_nonzero = instruction_i[6];
        if (instruction_i[5]) begin
          opcode_o = VXM_LOCAL_LOG;
          if (instruction_i[5:0] != VXM_LOG_INSTRUCTION)
            source_illegal = 1'b1;
        end
      end
      2, 4, 6: begin''')

# Preserve the legacy 16+16 coefficient ports and add one typed 50-bit Log row.
for name in ('alu','special_alu','execution_stage'):
    file=f'rtl/vxm/lpu_vxm_{name}.sv'
    replace(file, 'lut_read_b_i,', 'lut_read_b_i,\n  input  logic [49:0]                  lut_read_log_i,')
for name in ('alu','execution_stage'):
    file=f'rtl/vxm/lpu_vxm_{name}.sv'
    replace(file, '    .lut_read_b_i,', '    .lut_read_b_i,\n    .lut_read_log_i,')
replace('rtl/vxm/lpu_vxm_alu.sv', '    .opcode_i(opcode_i),', '    .opcode_i(opcode_i[2:0]),')
for name in ('alu','special_alu'):
    file=f'rtl/vxm/lpu_vxm_{name}.sv'
    replace(file, '(opcode_i == VXM_LOCAL_SPECIAL1)));', '(opcode_i == VXM_LOCAL_SPECIAL1) ||\n        (opcode_i == VXM_LOCAL_LOG)));')

pair='rtl/vxm/lpu_vxm_tile_pair_lut.sv'
replace(pair, 'write_b_i,', 'write_b_i,\n  input  logic [49:0]              write_log_i,')
replace(pair, '    response_b_o,', '    response_b_o,\n  output logic [TILE_COUNT*FUNCTION_COUNT*LANES*50-1:0]\n    response_log_o,')
replace(pair, '  logic [SRAM_COUNT-1:0] sram_configured;', '  logic [SRAM_COUNT*50-1:0] sram_response_log;\n  logic [SRAM_COUNT-1:0] sram_configured;')
replace(pair, "    response_b_o = '0;", "    response_b_o = '0;\n    response_log_o = '0;")
replace(pair, '          sram_response_b[sram_index*16 +: 16];', '          sram_response_b[sram_index*16 +: 16];\n        response_log_o[response_index*50 +: 50] =\n          sram_response_log[sram_index*50 +: 50];')
replace(pair, 'function_index == 0 ? 13 : 16;', 'function_index == 0 ? 13 : (function_index == 3 ? 25 : 16);')
start='''        assign sram_response_k[SRAM_INDEX*16 +: 16] ='''
end='''        lpu_vxm_lut_sram #('''
def pair_assign(s):
    a=s.index(start); b=s.index(end,a)
    return s[:a]+'''        if (function_index == 3) begin : g_log_row
          assign sram_response_k[SRAM_INDEX*16 +: 16] = '0;
          assign sram_response_b[SRAM_INDEX*16 +: 16] = '0;
          assign sram_response_log[SRAM_INDEX*50 +: 50] =
            {coefficient_k, coefficient_b};
        end else begin : g_legacy_row
          assign sram_response_k[SRAM_INDEX*16 +: 16] =
            {{(16-COEFFICIENT_WIDTH){1'b0}}, coefficient_k};
          assign sram_response_b[SRAM_INDEX*16 +: 16] =
            {{(16-COEFFICIENT_WIDTH){1'b0}}, coefficient_b};
          assign sram_response_log[SRAM_INDEX*50 +: 50] = '0;
        end

'''+s[b:]
edit(pair,pair_assign)
replace(pair, '.write_k_i(write_k_i[COEFFICIENT_WIDTH-1:0]),', ".write_k_i(function_index == 3 ?\n            COEFFICIENT_WIDTH'(write_log_i[49:25]) : COEFFICIENT_WIDTH'(write_k_i)),")
replace(pair, '.write_b_i(write_b_i[COEFFICIENT_WIDTH-1:0]),', ".write_b_i(function_index == 3 ?\n            COEFFICIENT_WIDTH'(write_log_i[24:0]) : COEFFICIENT_WIDTH'(write_b_i)),")
replace(pair, '(FUNCTION_COUNT != 3)', '((FUNCTION_COUNT != 3) && (FUNCTION_COUNT != 4))')
replace(pair, 'two Tiles, three functions, and 1 or 8 Lanes', 'two Tiles, three/four functions, and 1 or 8 Lanes')

tile='rtl/vxm/lpu_vxm_tile_execution.sv'
replace(tile, 'lut_write_b_i,', 'lut_write_b_i,\n  input  logic [49:0]                  lut_write_log_i,')
replace(tile, '    lut_lane_response_b_i,', '    lut_lane_response_b_i,\n  input  logic [LUT_BANK_COUNT*LANES*50-1:0]\n    lut_lane_response_log_i,')
replace(tile, '  logic [LUT_BANK_COUNT-1:0] lut_configured;', '  logic [STAGES*LANES*50-1:0] lut_response_log;\n  logic [LUT_BANK_COUNT*LANES*50-1:0] selected_lut_response_log;\n  logic [LUT_BANK_COUNT-1:0] lut_configured;')
replace(tile, "    lut_response_b = '0;", "    lut_response_b = '0;\n    lut_response_log = '0;")
replace(tile, '            lut_response_b[response_client*16 +: 16] =', '            lut_response_log[response_client*50 +: 50] =\n              selected_lut_response_log[lane_function_index*50 +: 50];\n            lut_response_b[response_client*16 +: 16] =')
replace(tile, '      assign selected_lut_response_b = lut_lane_response_b_i;', '      assign selected_lut_response_b = lut_lane_response_b_i;\n      assign selected_lut_response_log = lut_lane_response_log_i;')
replace(tile, '      logic [2*LANE_FUNCTIONS*16-1:0] pair_response_b;', '      logic [2*LANE_FUNCTIONS*16-1:0] pair_response_b;\n      logic [2*LANE_FUNCTIONS*50-1:0] pair_response_log;')
replace(tile, '        .write_b_i(lut_write_b_i),', '        .write_b_i(lut_write_b_i),\n        .write_log_i(lut_write_log_i),')
replace(tile, '        .response_b_o(pair_response_b),', '        .response_b_o(pair_response_b),\n        .response_log_o(pair_response_log),')
replace(tile, '        pair_response_b[0 +: LANE_FUNCTIONS*16];', '        pair_response_b[0 +: LANE_FUNCTIONS*16];\n      assign selected_lut_response_log =\n        pair_response_log[0 +: LANE_FUNCTIONS*50];')
replace(tile, '          .lut_read_b_i(lut_response_b[(stage*LANES+lane)*16 +: 16]),', '          .lut_read_b_i(lut_response_b[(stage*LANES+lane)*16 +: 16]),\n          .lut_read_log_i(lut_response_log[(stage*LANES+lane)*50 +: 50]),')

slice='rtl/vxm/lpu_vxm_slice.sv'
replace(slice, 'lut_write_b_i,', 'lut_write_b_i,\n  input  logic [49:0] lut_write_log_i,')
replace(slice, 'localparam integer LUT_FUNCTIONS = 3;', 'localparam integer LUT_FUNCTIONS = 4;')
replace(slice, '  logic [TILE_PAIRS*LUT_FUNCTIONS-1:0] pair_lut_configured;', '  logic [TILES*LUT_LANE_REQUESTS*50-1:0] tile_lut_response_log;\n  logic [TILE_PAIRS*LUT_FUNCTIONS-1:0] pair_lut_configured;')
replace(slice, 'lpu_vxm_tile_execution #(', 'lpu_vxm_tile_execution #(\n          .LUT_BANK_COUNT(LUT_FUNCTIONS),')
replace(slice, "          .lut_write_b_i('0),", "          .lut_write_b_i('0),\n          .lut_write_log_i('0),")
needle='''          .lut_lane_response_b_i('''
def slice_port(s):
    a=s.index(needle); b=s.index(']),',a)+3
    return s[:b]+'''
          .lut_lane_response_log_i(tile_lut_response_log[
            tile*LUT_LANE_REQUESTS*50 +: LUT_LANE_REQUESTS*50]),'''+s[b:]
edit(slice,slice_port)
replace(slice, '        .write_b_i(lut_write_b_i),', '        .write_b_i(lut_write_b_i),\n        .write_log_i(lut_write_log_i),')
replace(slice, '        .configured_o(pair_lut_configured[', '        .response_log_o(tile_lut_response_log[\n          FIRST_TILE*LUT_LANE_REQUESTS*50 +: 2*LUT_LANE_REQUESTS*50]),\n        .configured_o(pair_lut_configured[')
replace('rtl/lpu_top.sv', "      .lut_write_b_i('0),", "      .lut_write_b_i('0),\n      .lut_write_log_i('0),")

# The canonical queue widths drive both physical ICU memories and waves.
replace('sim/tb/lpu_vxm_icu_map_tb.sv', 'expect_payload(115, 5)', 'expect_payload(115, 6)')
replace('sim/tb/lpu_vxm_icu_map_tb.sv', 'expect_payload(119, 5)', 'expect_payload(119, 6)')
print('Extended instruction and LUT plumbing in',ROOT)
