from pathlib import Path
import re
root=Path(__file__).resolve().parents[2]/'rtl-log-work'
p=root/'rtl/core/lpu_pkg.sv';s=p.read_text()
s=re.sub(r"(VXM_LOCAL_\w+\s*= )3'd",lambda m:m.group(1)+"4'd",s)
p.write_text(s,newline='\n')
p=root/'rtl/vxm/lpu_vxm_tile_execution.sv'
s=re.sub(r'\bconfig\b','active_config',p.read_text())
p.write_text(s,newline='\n')
p=root/'rtl/vxm/lpu_vxm_special_alu.sv'
s=p.read_text().replace('  logic supported_opcode;\n','')
s=s.replace('  logic log_ready,','  logic supported_opcode;\n  logic log_ready,')
p.write_text(s,newline='\n')
