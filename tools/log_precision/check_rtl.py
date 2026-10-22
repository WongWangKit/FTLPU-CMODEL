import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'build-log/python-deps'))
from pyslang.driver import Driver
import os
os.chdir(ROOT/'rtl-log-work')
files=[s.strip() for s in Path('sim/filelist.f').read_text().splitlines() if s.strip().startswith('rtl/')]
d=Driver();d.addStandardArgs()
top=sys.argv[1] if len(sys.argv)>1 else 'lpu_vxm_slice'
extra=[a for a in sys.argv[2:] if a!='--constants']
args='slang --error-limit 20 --timescale 1ns/1ps --top '+top+' '+' '.join(files+extra)
if not d.parseCommandLine(args): raise SystemExit(2)
if not d.processOptions(): raise SystemExit(2)
if not d.parseAllSources(): raise SystemExit(2)
if not d.runFullCompilation(quiet=True): raise SystemExit(1)
if '--constants' in sys.argv:
    compilation=d.createCompilation()
    body=compilation.getRoot().topInstances[0].body
    count=0;failed=[]
    for member in body:
        if member.name.startswith('PASS_'):
            count+=1
            if str(member.value) not in ('1','1\'b1'):
                failed.append((member.name,str(member.value)))
    print('RTL constant-arithmetic checks:',count,'failed:',len(failed))
    if failed:
        print(failed[:30])
        raise SystemExit(1)
    if not count: raise SystemExit('No constant checks were evaluated')
print('RTL_CHECK_PASS',top)
