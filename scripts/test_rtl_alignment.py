"""Compare native CModel public-port traces against the current RTL.

Build rtl_probe first. This runner reads (never edits) --vmodel, and keeps all
stimuli, simulator binaries, traces, logs and source hashes under --output.
"""
from pathlib import Path
import argparse, hashlib, json, math, random, re, shutil, struct, subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vmodel',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'build/rtl_alignment')
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--jobs',type=int,default=4,help='Independent VXM Tile simulators (default: 4)')
    args=p.parse_args();vm=args.vmodel.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def tool(name):
        return shutil.which(name) or str(vm/'build/iverilog/bin'/(name+'.exe'))
    def run(argv,name,cwd=ROOT):
        try:r=subprocess.run([str(a) for a in argv],cwd=cwd,capture_output=True,text=True,timeout=180)
        except subprocess.TimeoutExpired as e:
            (out/(name+'.log')).write_text('TIMEOUT\n'+str(e.stdout or '')+'\n'+str(e.stderr or ''))
            raise
        (out/(name+'.log')).write_text(subprocess.list2cmdline([str(a) for a in argv])+'\n'+r.stdout+r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError(name+' failed: '+(r.stdout+r.stderr)[-2000:])
        return r
    results=[];all_sources=[]
    def compare(name,rtl,model):
        expected=rtl.read_text().splitlines();actual=model.read_text().splitlines()
        differences=[(i,a,b) for i,(a,b) in enumerate(zip(expected,actual)) if a.split()!=b.split()]
        passed=not differences and len(expected)==len(actual)
        result=dict(name=name,passed=passed,observations=len(expected),differences=differences[:30]);results.append(result)
        (out/'results.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in results),results=results),indent=2)+'\n')
        print(name,'PASS' if passed else 'FAIL','observations='+str(len(expected)),'mismatches='+str(len(differences)),flush=True)
        if not passed:raise RuntimeError('Trace mismatch: '+str(differences[:8]))
        return result
    mem_sources=[vm/'rtl/mem_native'/f'{n}.sv' for n in ['mem_bank_superlane_leaf','mem_bank_control_column','mem_logical_bank_column']]
    for rows_count in [2,4,8]:
        name=f'mem_rows{rows_count}';exe=out/(name+'.vvp');rtl=out/(name+'_rtl.txt');model=out/(name+'_cmodel.txt')
        oracle=ROOT/'tests/unit/rtl/mem_oracle.sv'
        run([tool('iverilog'),'-g2012','-s','mem_oracle',f'-Pmem_oracle.ROWS={rows_count}','-o',exe,*mem_sources,oracle],name+'_compile',vm)
        run([tool('vvp'),exe,f'+OUTPUT={rtl.as_posix()}'],name+'_run',vm)
        run([args.probe.resolve(),'mem',rows_count,model],name+'_cmodel');compare(name,rtl,model)
        all_sources+=mem_sources+[oracle]
    mxm_sources=[vm/s.strip() for s in (vm/'sim/filelist_mxm_supercell.f').read_text().splitlines() if s.strip()]
    oracle=ROOT/'tests/unit/rtl/mxm_core_oracle.sv';exe=out/'mxm.vvp';rtl=out/'mxm_rtl.txt';model=out/'mxm_cmodel.txt'
    run([tool('iverilog'),'-g2012','-s','mxm_core_oracle','-o',exe,*mxm_sources,oracle],'mxm_compile',vm)
    run([tool('vvp'),exe,f'+OUTPUT={rtl.as_posix()}'],'mxm_run',vm)
    run([args.probe.resolve(),'core',model],'mxm_cmodel');compare('mxm_core',rtl,model)
    all_sources+=mxm_sources+[oracle]
    cell_rng=random.Random(0x43534136);stim=[]
    for t in range(16):stim.append((0,0,1,t//8,0,[0]*8,[cell_rng.randrange(65536) for _ in range(8)]))
    for t in range(1024):
        a=[cell_rng.randrange(65536) for _ in range(8)];w=[cell_rng.randrange(65536) for _ in range(8)]
        if t%17==0:a=[0x4b80,0x3f80,0xcb80,0,0,0,0,0]
        buffer=(t//64)%2;stim.append((t%11!=0,buffer,t%7==0,not buffer,t,a,w))
    stim.extend([(0,0,0,0,0,[0]*8,[0]*8)]*16)
    inp=out/'cell_input.txt';inp.write_text(''.join(f'{int(av)} {int(ab)} {int(wv)} {int(wb)} {tag} '+ ' '.join(f'{x:04x}' for x in a+w)+'\n' for av,ab,wv,wb,tag,a,w in stim))
    oracle=ROOT/'tests/unit/rtl/cell_oracle.sv';exe=out/'cell.vvp';rtl=out/'cell_rtl.txt';model=out/'cell_cmodel.txt'
    run([tool('iverilog'),'-g2012','-s','cell_oracle','-o',exe,*mxm_sources,oracle],'cell_compile',vm)
    run([tool('vvp'),exe,f'+INPUT={inp.as_posix()}',f'+OUTPUT={rtl.as_posix()}'],'cell_run',vm)
    run([args.probe.resolve(),'cell',inp,model],'cell_cmodel');compare('mxm_supercell_random',rtl,model)
    all_sources+=[oracle]
    rows=[]
    for i in range(64):rows.append(round(math.exp(-.5+i/64)*2**25))
    for i in range(64):
        x=1+i/64;rows.append((round(64*(1/x-1/(x+1/64))*2**15)<<16)|round(1/x*2**15))
    for i in range(64):
        parity=2 if i>=32 else 1;x=parity*(1+(i%32)/32)
        rows.append((round(32*(1/math.sqrt(x)-1/math.sqrt(x+parity/32))*2**15)<<16)|round(1/math.sqrt(x)*2**15))
    for i in range(64):
        x=1+i/64;rows.append((round(math.log(x)*2**28)<<22)|round(1/x*2**21))
    lut=out/'lut.hex';lut.write_text(''.join(f'{r:016x}\n' for r in rows),encoding='ascii')
    sources=[s.strip() for s in (vm/'sim/filelist_vxm_log.f').read_text().splitlines() if s.strip()]
    sources=[vm/s for s in sources if s=='rtl/core/lpu_pkg.sv' or s.startswith('rtl/vxm/')]
    sources=[s for s in sources if s.name!='lpu_vxm_slice.sv']
    originals=list(sources)
    # Icarus needs package qualification in generate/port expressions and
    # does not support the packed-struct cast. The adder's acyclic front end,
    # shift control and packer are partitioned into separate comb processes
    # to prevent Icarus delta oscillation through its structural children.
    # Arithmetic statements and all registers remain unchanged.
    names=set(re.findall(r'\bVXM_[A-Z0-9_]+\b',(vm/'rtl/core/lpu_pkg.sv').read_text()))
    for i,source in enumerate(sources):
        if source.parent.name=='vxm':
            copy=out/'icarus_rtl'/source.name;copy.parent.mkdir(parents=True,exist_ok=True)
            text=re.sub(r'(?<![\w:])VXM_[A-Z0-9_]+',lambda m:'lpu_pkg::'+m[0] if m[0] in names else m[0],source.read_text())
            text=text.replace("vxm_global_config_t'(global_config_i)",'global_config_i')
            if source.name=='lpu_vxm_shared_float_adder.sv':
                text=text.replace("    subtract_shift = 5'd0;","  end\n  always_comb begin\n    subtract_shift = 5'd0;")
                text=text.replace("    normal_result = 32'b0;","  end\n  always_comb begin\n    normal_result = 32'b0;")
            copy.write_text(text,encoding='utf-8');sources[i]=copy
    oracle=ROOT/'tests/unit/rtl/alu_oracle.sv'
    rng=random.Random(0x4c505532)
    for kind,basic in ((2,0),(1,0),(2,1)):
        stimuli=[];operations=range(6) if basic else range(6,9) if kind==2 else [6]
        for fmt in range(3):
            fb=(10,7,23)[fmt];eb=(5,8,8)[fmt];bias=(15,127,127)[fmt]
            one=bias<<fb;sign=1<<(fb+eb);infinity=((1<<eb)-1)<<fb
            edges=[0,sign,1,one,one-1,one+1,one+2,infinity,sign|infinity,infinity|1,sign|one,(1<<fb)-1,1<<fb]
            values=edges+[rng.randrange(1<<(fb+eb+1)) for _ in range(args.samples)]
            for op in operations:
                for a in values:
                    b=rng.choice(values);stimuli.append((1,fmt,op,a,b));stimuli.extend([(0,fmt,op,0,0)]*8)
        label=f'alu_{kind}_basic{basic}';path=out/f'{label}_input.txt'
        path.write_text(''.join(f'{v} {f} {op} {a:08x} {b:08x}\n' for v,f,op,a,b in stimuli),encoding='ascii')
        exe=out/f'{label}.vvp';rtl=out/f'{label}_rtl.txt';model=out/f'{label}_cmodel.txt'
        run([tool('iverilog'),'-g2012','-s','alu_oracle',f'-Palu_oracle.KIND={kind}',f'-Palu_oracle.BASIC={basic}','-o',exe,*sources,oracle],label+'_compile',vm)
        run([tool('vvp'),exe,f'+INPUT={path.as_posix()}',f'+OUTPUT={rtl.as_posix()}',f'+LUT={lut.as_posix()}'],label+'_run',vm)
        run([args.probe.resolve(),'alu',kind,path,model,lut],label+'_cmodel')
        compare(f'vxm_basic' if basic else f'vxm_special_kind{kind}',rtl,model)['requests']=sum(s[0] for s in stimuli)
    all_sources+=[oracle]
    conversion=[]
    for src in range(4):
        for dst in range(4):
            values=[0,1,0x80000000,0x007fffff,0x00800000,0x7f800000,0x7fa12345,0xffff7e01,0x7f81,0x3c00]+[rng.randrange(2**32) for _ in range(args.samples)]
            conversion.extend((1,src,dst,raw) for raw in values);conversion.append((0,src,dst,0xffffffff))
    inp=out/'conversion_input.txt';inp.write_text(''.join(f'{v} {s} {d} {raw:08x}\n' for v,s,d,raw in conversion))
    oracle=ROOT/'tests/unit/rtl/conversion_oracle.sv';exe=out/'conversion.vvp';rtl=out/'conversion_rtl.txt';model=out/'conversion_cmodel.txt'
    run([tool('iverilog'),'-g2012','-s','conversion_oracle','-o',exe,*sources,oracle],'conversion_compile',vm)
    run([tool('vvp'),exe,f'+INPUT={inp.as_posix()}',f'+OUTPUT={rtl.as_posix()}'],'conversion_run',vm)
    run([args.probe.resolve(),'conversion',inp,model],'conversion_model');compare('vxm_conversion',rtl,model)
    all_sources+=[oracle];oracle=ROOT/'tests/unit/rtl/vxm_tile_oracle.sv';exe=out/'tile.vvp'
    run([tool('iverilog'),'-g2012','-s','vxm_tile_oracle','-o',exe,*sources,oracle],'tile_compile',vm)
    cases=[(f'tile_c{c}_f{f}_op{o}',c,f,o,0) for c in range(3) for f in range(3) for o in range(6)]
    cases += [(f'tile_mode{m}_f{f}',2,f,1 if m in (3,4) else 0,m) for m in range(1,6) for f in range(3)]
    def tile_case(case):
        label,chain,fmt,op,mode=case;rtl=out/(label+'_rtl.txt');model=out/(label+'_cmodel.txt')
        run([tool('vvp'),exe,f'+OUTPUT={rtl.as_posix()}',f'+CHAIN={chain}',f'+FORMAT={fmt}',f'+OP={op}',f'+MODE={mode}'],label+'_run',vm)
        run([args.probe.resolve(),'tile',chain,fmt,op,mode,model],label+'_cmodel')
        return label,rtl,model
    with ThreadPoolExecutor(max_workers=max(1,args.jobs)) as pool:
        futures=[pool.submit(tile_case,c) for c in cases]
        for future in futures:compare(*future.result())
    identities={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in all_sources+originals+sources+[oracle,Path(__file__),ROOT/'tests/unit/rtl/probe.cpp']+list((ROOT/'include/ftlpu/rtl').glob('*.hpp'))}
    (out/'results.json').write_text(json.dumps(dict(passed=True,results=results,
        simulator_adaptation='Read-only copies: qualify imported VXM package constants, replace identity packed-struct cast, partition acyclic adder comb cone into front/shift/pack processes; no arithmetic statement/register changes',source_sha256=identities,probe_sha256=hashlib.sha256(args.probe.resolve().read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
