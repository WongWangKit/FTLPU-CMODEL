from pathlib import Path
import math, struct, random
ROOT=Path(__file__).resolve().parents[2]
def f32(x):
    return struct.unpack('<I',struct.pack('<f',x))[0]
def fp16(x):
    return int.from_bytes(struct.pack('<e',x),'little')
def bf16(x):
    b=f32(x);return (b+0x7fff+((b>>16)&1))>>16
rows=[]
for i in range(64):
    c=1+i/64
    base=round(math.log(c)*2**28)
    inv=round((1/c)*2**21)
    rows.append((base<<22)|inv)
(ROOT/'sim/vectors/vxm_log_lut.hex').write_text(''.join(f'{r:013x}\n' for r in rows))
rng=random.Random(20261020)
vectors=[]
for fmt in range(3):
    fb=(10,7,23)[fmt];eb=(5,8,8)[fmt];bias=(15,127,127)[fmt]
    values=[0,1,1<<fb,(bias<<fb),((bias<<fb)-1),((bias<<fb)+1),
            ((2**eb-1)<<fb),((2**eb-1)<<fb)|1,1<<(fb+eb),
            (1<<(fb+eb))|1,(1<<(fb+eb))|(bias<<fb),
            (1<<(fb+eb))|((2**eb-1)<<fb)]
    # Both sides of every segment boundary and near-one threshold.
    for e in (bias-1,bias,bias+1):
        for a in range(65):
            for offset in (-1,0,1):
                b=(e<<fb)+(a<<(fb-6))+offset
                if 0<b<((2**eb-1)<<fb):values.append(b)
    values += [rng.randrange(1,(2**eb-1)<<fb) for _ in range(256)]
    if fmt==2:values += [0x3f7b43f0,0x3f7c0000,0x3f820000,0x3f81004a]
    for b in dict.fromkeys(values):
        ex=(b>>fb)&(2**eb-1);fraction=b&((1<<fb)-1);sign=b>>(fb+eb)
        if ex==2**eb-1 and fraction:
            expected=(0x7e00,0x7fc0,0x7fc00000)[fmt]
        elif ex==0:
            expected=(0xfc00,0xff80,0xff800000)[fmt]
        elif sign:
            expected=(0x7e00,0x7fc0,0x7fc00000)[fmt]
        elif ex==2**eb-1:
            expected=(0x7c00,0x7f80,0x7f800000)[fmt]
        else:
            x=math.ldexp(1+fraction/2**fb,ex-bias)
            y=math.log(x)
            expected=(fp16(y),bf16(y),f32(y))[fmt]
        address=fraction>>(fb-6)
        vectors.append((fmt,b,expected,address))
(ROOT/'sim/vectors/vxm_log_cases.hex').write_text(''.join(f'{(f<<64)|(b<<32)|e:017x}\n' for f,b,e,a in vectors))
source=['module lpu_vxm_log_constants;','import lpu_vxm_log_reference_pkg::*;']
for i,(fmt,b,expected,a) in enumerate(vectors):
    source.append(f"localparam bit PASS_{i} = log_within_ulp(log_evaluate_bits(32'h{b:08x},2'd{fmt},50'h{rows[a]:013x}),32'h{expected:08x},2'd{fmt});")
source.append(f'localparam integer CASE_COUNT = {len(vectors)};')
source.append('endmodule')
(ROOT/'sim/tb/vxm/lpu_vxm_log_constants.sv').write_text('\n'.join(source)+'\n')
tb=ROOT/'sim/tb/vxm/lpu_vxm_log_tb.sv'
import re
tb.write_text(re.sub(r'CASE_COUNT = [0-9]+;', f'CASE_COUNT = {len(vectors)};', tb.read_text()))
print('Generated',len(vectors),'independent Log cases and 64 packed LUT rows.')
