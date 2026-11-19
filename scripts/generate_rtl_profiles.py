"""Derive current RTL geometry targets while retaining legacy target JSONs."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
for width in (16,32,64):
    p=ROOT/'config/mem-profiles'/f'ftlpu-lpu{width}.json'
    d=json.loads(p.read_text(encoding='utf-8'));rows=width//8
    d['target']['name']=f'ftlpu-rtl-lpu{width}'
    d['mxm']['accum_contexts']=1024//width
    d['mxm']['supported_modes']=['ws-vector'];d['mxm']['is_columns']=rows
    d['throughput']['vxm_fma_enabled']=0
    # These legacy schedule hints are not native block addresses, but retain
    # in-range values for software that still reads the shared JSON schema.
    depth=d['mem']['rows_per_bank']
    d['memory']['attention_mask_base_row']=depth-256
    d['memory']['matmul_result_base_row']=min(1600,depth//2)
    d['rtl_alignment']={
        'interface':'ftlpu/rtl/model.hpp',
        'mem':{'command_bits':32,'delay3_bit':30,'hops':[1,3],'read_cycles':1,'ports_per_bank':1},
        'mxm':{'sc_rows':rows,'sc_columns':rows,'cell_k':8,'cell_n':8,'input':'bf16','partial':'fp32',
               'dot_bits':64,'dot_stages':9,'partial_add_stages':3,'instruction_hop':3,'result_hop':1,
               'accumulator_result_length':width,'accumulator_depth':1024,'banks_per_result_group':2,
               'accumulator_read_to_write_cycles':5,'dequant_stages':5,
               'param_load_instruction_bits':5,'dequant_instruction_bits':3},
        'vxm':{'physical_stages':16,'lanes':8,'local_instruction_bits':[6,5,7,6,7,5,7,6],
               'global_config_bits':15,'special_cycles':8,'lut_entries':64,'lut_functions':4,
               'lut_private_per_lane':True,'fp32_input_beats':2},
        'scope':'Replacement block interfaces; legacy whole-chip TspSliceSystem remains a compatibility API.'}
    (ROOT/'config'/f'ftlpu-rtl-lpu{width}.json').write_bytes((json.dumps(d,indent=2)+'\n').encode('utf-8'))
