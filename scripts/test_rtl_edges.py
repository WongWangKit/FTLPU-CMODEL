"""Public-port differential tests for MXM parameter switching and SRAM hazards."""
from pathlib import Path
import argparse,hashlib,json,random,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vmodel',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    vm=a.vmodel.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sources=[vm/s.strip() for s in (vm/'sim/filelist_mxm_supercell.f').read_text().splitlines() if s.strip()]
    def execute(argv,name):
        r=subprocess.run(list(map(str,argv)),cwd=vm,capture_output=True,text=True,timeout=180)
        (out/(name+'.log')).write_text(subprocess.list2cmdline(list(map(str,argv)))+'\n'+r.stdout+r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError(name+': '+(r.stdout+r.stderr)[-3000:])
    rng=random.Random(0x44453136);dq=[]
    # Entries are (weight format, scale mode, zero mode). The next parameter
    # load and the current D1 read target opposite buffers in the steady state.
    modes=[(0,s,z) for s in range(3) for z in range(2)]
    modes += [(1,s,0) for s in range(4)] + [(2,s,0) for s in range(3)]
    for t in range(4096):
        phase=t%512;reset=phase==0;run=phase!=256
        request=phase>=2 and phase!=257 and run;load=run and not reset
        fmt,_,_=modes[(phase-2)%len(modes)]
        _,scale_mode,zero_mode=modes[(phase-1)%len(modes)]
        pb=phase%2;lb=1-pb
        if phase==490:lb=pb # same-edge read/write
        mask=(1<<48)-1
        if phase==470:mask^=1 # missing q byte
        if phase==480:mask^=1<<9 # incomplete scale load
        if phase in (492,493):zero_mode=phase-490 # reserved zero modes
        if phase==494:scale_mode,zero_mode=3,1 # NVFP4 with explicit zero
        if phase==495:fmt=3 # reserved format
        if phase==496:scale_mode,zero_mode=2,1;mask^=1<<16 # missing explicit zero
        if phase==498:scale_mode,zero_mode=0,1
        if phase==499:fmt=2 # FP8 reading the explicit zero entry loaded above
        bytes_=[[rng.randrange(256) for _ in range(8)] for _ in range(6)]
        if phase==1:
            bytes_[1]=[0x80]*8;bytes_[2]=[0x3f]*8
        if phase==2:
            bytes_[0]=[0x7f,0x80]*4
            bytes_[1]=[0x80]*8;bytes_[2]=[0x3f]*8;bytes_[3]=[0x80,0x7f]*4
        if phase==3:bytes_[0]=[0x7f,0x80]*4 # signed differences +255 and -255
        param_word=(zero_mode<<3)|(scale_mode<<1)|lb
        dq_word=(fmt<<1)|pb
        dq.append(f'{int(reset)} {int(run)} {int(request)} {pb} {int(load)} '
                  f'{param_word} {dq_word} {mask:012x} '+
                  ' '.join(f'{byte:02x}' for stream in bytes_ for byte in stream)+'\n')
    jobs={};partial={}
    def issue(t,op,address,bf=False,missing=False):
        jobs[t]=(op,bf,address)
        if not missing:partial[t+3]=(t&65535,[rng.randrange(2**32) for _ in range(8)])
    for i in range(32):
        issue(1+i,1,i);issue(50+i,2,i);issue(82+i,3,i,i%2==0);issue(150+i,0,i,i%2==1)
    partial[154]=(151,[0x7f800001,0xff812345,0x7fa12345,0x00018000,0x80000000,0x7f7fffff,0x3f808000,0x3f818000])
    for t,op,addr in [(221,1,0),(226,3,0),(240,3,0),(261,1,0),(262,1,0),(280,3,0),(301,3,0),(321,1,0),(330,3,0)]:issue(t,op,addr)
    issue(360,0,0,missing=True);issue(379,0,0);issue(401,0,0)
    partial[340]=(340,[0x3f800000]*8)
    backend=[]
    for t in range(440):
        reset=t in (0,220,260,300,320,350);run=t!=380
        op,bf,addr=jobs.get(t,(0,False,0));tag,data=partial.get(t,(0,[0]*8))
        backend.append(f'{int(run)} {int(reset)} {int(t in jobs)} {op} {int(bf)} {addr} {int(t in partial)} {tag} '+
                       ' '.join(f'{x:08x}' for x in data)+'\n')
    controller=[]
    for t in range(512):
        controller.append(f'{int(t in (0,256,480))} {int(rng.random()<.35)} {rng.randrange(5)} {rng.randrange(8)} '+
                          f'{rng.randrange(2)} {rng.randrange(2)} {int(rng.random()<.8)} {int(rng.random()<.7)} {int(rng.random()<.3)}\n')
    results=[]
    for name,stimuli in [('dequant',dq),('backend',backend),('controller',controller)]:
        inp=out/(name+'_input.txt');inp.write_text(''.join(stimuli),encoding='ascii')
        oracle=ROOT/f'tests/unit/rtl/{name}_oracle.sv';exe=out/(name+'.vvp');rt=out/(name+'_rtl.txt');cm=out/(name+'_cmodel.txt')
        iv=shutil.which('iverilog') or vm/'build/iverilog/bin/iverilog.exe';vvp=shutil.which('vvp') or vm/'build/iverilog/bin/vvp.exe'
        case_sources=sources if name!='controller' else [vm/'rtl/core/lpu_pkg.sv',vm/'rtl/vxm/lpu_vxm_instruction_controller.sv']
        execute([iv,'-g2012','-s',name+'_oracle','-o',exe,*case_sources,oracle],name+'_compile')
        execute([vvp,exe,f'+INPUT={inp.as_posix()}',f'+OUTPUT={rt.as_posix()}'],name+'_run')
        execute([a.probe.resolve(),name,inp,cm],name+'_model')
        expected=rt.read_text().splitlines();actual=cm.read_text().splitlines()
        differences=[(i,e,c) for i,(e,c) in enumerate(zip(expected,actual)) if e.split()!=c.split()]
        result=dict(name=name,passed=not differences and len(expected)==len(actual),observations=len(expected),differences=differences[:20]);results.append(result)
        (out/'results.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in results),results=results),indent=2))
        print(name,'PASS' if result['passed'] else 'FAIL',len(expected),'cycles',len(differences),'mismatches',flush=True)
        if not result['passed']:raise RuntimeError(str(differences[:5]))
    identities={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources+[vm/'rtl/core/lpu_pkg.sv',vm/'rtl/vxm/lpu_vxm_instruction_controller.sv',Path(__file__),ROOT/'tests/unit/rtl/probe.cpp']+list((ROOT/'include/ftlpu/rtl').glob('*.hpp'))+list((ROOT/'tests/unit/rtl').glob('*oracle.sv'))}
    (out/'results.json').write_text(json.dumps(dict(passed=True,results=results,source_sha256=identities),indent=2)+'\n')
if __name__=='__main__':main()
