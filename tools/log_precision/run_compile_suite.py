from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[2]
helper=root/'tools/log_precision/check_rtl.py'
checks=[
 ('lpu_top',[]),('lpu_vxm_slice',[]),
 ('lpu_vxm_log_tb',['sim/tb/vxm/lpu_vxm_log_reference_pkg.sv','sim/tb/vxm/lpu_vxm_log_tb.sv']),
 ('lpu_vxm_log_constants',['sim/tb/vxm/lpu_vxm_log_reference_pkg.sv','sim/tb/vxm/lpu_vxm_log_constants.sv','--constants']),
 ('lpu_vxm_icu_map_tb',['sim/tb/lpu_vxm_icu_map_tb.sv']),
 ('lpu_vxm_16_alu_tb',['sim/tb/vxm/lpu_vxm_alu_stage_checker.sv','sim/tb/vxm/lpu_vxm_16_alu_tb.sv']),
 ('lpu_vxm_tile_pair_lut_tb',['sim/tb/vxm/lpu_vxm_tile_pair_lut_tb.sv']),
 ('lpu_vxm_tile_execution_tb',['sim/tb/vxm/lpu_vxm_tile_execution_tb.sv']),
 ('lpu_vxm_tile_special_paths_tb',['sim/tb/vxm/lpu_vxm_tile_special_paths_tb.sv']),
 ('lpu_vxm_slice_fp32_special_tb',['sim/tb/vxm/lpu_vxm_slice_fp32_special_tb.sv']),
 ('lpu_vxm_slice_bf16_special_tb',['sim/tb/vxm/lpu_vxm_slice_bf16_special_tb.sv']),
 ('lpu_vxm_lane_chain_tb',['sim/tb/vxm/chains/lpu_vxm_lane_chain_tb.sv']),
 ('lpu_vxm_superlane_lockstep_tb',['sim/tb/vxm/chains/lpu_vxm_superlane_lockstep_tb.sv']),
]
failed=[]
for top,extra in checks:
    result=subprocess.run([sys.executable,str(helper),top,*extra],cwd=root,capture_output=True,text=True)
    (root/'build-log'/f'{top}_compile.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(('PASS ' if result.returncode==0 else 'FAIL ')+top,flush=True)
    if result.returncode:
        failed.append(top)
        print((result.stdout+result.stderr)[-3500:],flush=True)
print('COMPILE_SUITE',len(checks)-len(failed),'/',len(checks))
raise SystemExit(bool(failed))
