"""Run an actual event simulation with the workspace-local Windows Verilator."""
from pathlib import Path
import os,subprocess,sys,shutil,hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'rtl-log-work'
V=ROOT/'build-log/verilator-deps/verilator'
top=sys.argv[1] if len(sys.argv)>1 else 'lpu_vxm_log_tb'
extras=sys.argv[2:] or ['sim/tb/vxm/lpu_vxm_log_reference_pkg.sv','sim/tb/vxm/lpu_vxm_log_tb.sv']
variant='_'+hashlib.sha256(' '.join(extras).encode()).hexdigest()[:8] if any(x.startswith('-G') for x in extras) else ''
obj=ROOT/'build-log'/('verilated2_'+top+variant);obj.mkdir(exist_ok=True)
files=[s.strip() for s in (R/'sim/filelist.f').read_text().splitlines() if s.strip().startswith('rtl/')]
env=os.environ.copy();env['VERILATOR_ROOT']=V.as_posix()
cmd=[str(V/'bin/verilator_bin.exe'),'--cc','--main','--timing','--assert','-Wno-fatal',
     '--timescale','1ns/1ps','--top-module',top,'--Mdir',obj.as_posix(),*files,*extras]
def run(cmd,log):
 p=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True)
 (obj/log).write_text(p.stdout+p.stderr,encoding='utf-8')
 if p.returncode:
  print((p.stdout+p.stderr)[-6500:],flush=True);raise SystemExit(p.returncode)
run(cmd,'verilate.log');print('VERILATED',top,flush=True)
sources=list(obj.glob('*.cpp'))
sources += [V/'include'/s for s in ('verilated.cpp','verilated_timing.cpp','verilated_threads.cpp')]
exe=obj/(top+'.exe')
flags=[shutil.which('clang++'),'-std=c++20','-O0','-pthread','-DVL_TIME_CONTEXT',
       '-I'+str(V/'include'),'-I'+str(V/'include/vltstd'),'-I'+str(obj)]
objects=[obj/(s.stem+'.o') for s in sources]
def compile_one(item):
    source,output=item
    p=subprocess.run([*flags,'-c',str(source),'-o',str(output)],cwd=R,env=env,capture_output=True,text=True)
    return p.returncode,p.stdout+p.stderr
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(compile_one,zip(sources,objects)))
(obj/'build.log').write_text(''.join(s for _,s in results),encoding='utf-8')
if any(c for c,_ in results):
    print((obj/'build.log').read_text()[-6500:]);raise SystemExit(1)
run([shutil.which('clang++'),'-pthread',*map(str,objects),'-o',str(exe)],'link.log')
print('BUILT',top,flush=True)
run([str(exe)],'simulation.log')
print((obj/'simulation.log').read_text(),flush=True)
