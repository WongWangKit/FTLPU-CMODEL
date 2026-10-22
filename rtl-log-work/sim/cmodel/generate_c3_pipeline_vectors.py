"""Independent mathematical golden vectors: instructions + operand bits.

No RTL arithmetic helper or internal pipeline state is used as the oracle.
"""
from pathlib import Path
import math,random,struct,json
R=Path(__file__).resolve().parents[2]
rng=random.Random(20261021)
def f32bits(x):
    try:return int.from_bytes(struct.pack('<f',x),'little')
    except OverflowError:return 0xff800000 if x<0 else 0x7f800000
def pack(x,fmt):
    if fmt==0:
        try:b=int.from_bytes(struct.pack('<e',x),'little')
        except OverflowError:b=0xfc00 if x<0 else 0x7c00
        return b&0x8000 if b&0x7c00==0 else b
    b=f32bits(x)
    if fmt==1:
        if b&0x7f800000!=0x7f800000:b=(b+0x7fff+((b>>16)&1))&0xffffffff
        b>>=16
        return b&0x8000 if b&0x7f80==0 else b
    return b&0x80000000 if b&0x7f800000==0 else b
def unpack(b,fmt):
    fb,eb,bias=((10,5,15),(7,8,127),(23,8,127))[fmt]
    sign=-1 if b>>(fb+eb) else 1
    ex=(b>>fb)&((1<<eb)-1);fr=b&((1<<fb)-1)
    if ex==(1<<eb)-1:return math.nan if fr else sign*math.inf
    if ex==0:return math.copysign(0.,sign)
    return sign*math.ldexp(1+fr/2**fb,ex-bias)
def expected(b,fmt,ins):
    x=unpack(b,fmt);nan=(0x7e00,0x7fc0,0x7fc00000)[fmt]
    if math.isnan(x):return nan
    if ins==0x26:
        if x==0:return pack(-math.inf,fmt)
        if x<0:return nan
        return pack(math.log(x),fmt)
    if ins==6:
        if x==0:return pack(math.copysign(math.inf,x),fmt)
        return pack(1/x,fmt)
    if x==0:return pack(math.inf,fmt)
    if x<0:return nan
    return pack(1/math.sqrt(x),fmt)
rows=[]
for fn in (1,2):
    for i in range(64):
        parity=i>>5 if fn==2 else 0
        index=i&31 if fn==2 else i
        c=1+index/(32 if fn==2 else 64)
        step=1/(32 if fn==2 else 64)
        if fn==1:b=1/c;k=(b-1/(c+step))/step
        else:
            # The residual always represents fraction/2^fb. Fold exponent
            # parity into the function before deriving its secant slope.
            b=1/math.sqrt(c*2**parity)
            k=(b-1/math.sqrt((c+step)*2**parity))/step
        rows.append((round(k*2**15)<<16)|round(b*2**15))
(R/'sim/vectors/vxm_c3_recip_rsqrt_lut.hex').write_text(''.join(f'{v:08x}\n' for v in rows))
vectors=[];counts={}
def append(fmt,ins,lhs,rhs,ref,tolerance):
    vectors.append((fmt,ins,lhs,rhs,ref,tolerance))
    key=f'{fmt}:{ins:02x}';counts[key]=counts.get(key,0)+1
for fmt in range(3):
    fb,eb,bias=((10,5,15),(7,8,127),(23,8,127))[fmt]
    limit=((1<<eb)-1)<<fb
    special=[0,1,(1<<fb)-1,1<<fb,limit-1,limit,limit|1,limit|(1<<(fb-1)),
             1<<(fb+eb),(1<<(fb+eb))|1,(1<<(fb+eb))|limit,
             (1<<(fb+eb))|(bias<<fb)]
    values=list(range(1<<fb,limit)) if fmt<2 else [rng.randrange(1<<fb,limit) for _ in range(12000)]
    if fmt==2:
        for ex in (1,63,125,126,127,128,190,253,254):
            for addr in range(65):
                for offset in (-1,0,1):
                    b=(ex<<fb)+(addr<<(fb-6))+offset
                    if 0<b<limit:values.append(b)
        values += [0x3f7b43f0,0x3f7c0000,0x3f820000,0x3f81004a]
    values+=special
    values += [v|(1<<(fb+eb)) for v in values[::113]]
    for ins in (6,7,0x26):
        for b in values:
            append(fmt,ins,b,0,expected(b,fmt,ins),1 if fmt<2 else 2)
    # Mix Basic instructions with special functions; all share C3 output timing.
    for _ in range(384):
        a=rng.randint(-16,16)/4;b=rng.randint(-16,16)/4
        lhs=pack(a,fmt);rhs=pack(b,fmt)
        for ins,y in ((0,a),(1,a+b),(2,a-b),(3,a*b),(4,-a),(5,max(a,b))):
            append(fmt,ins,lhs,rhs,pack(y,fmt),0)
rng.shuffle(vectors)
# Explicit long bursts exercise each precision/function at full rate as well.
bursts=[]
for fmt in range(3):
    for ins in (6,7,0x26):
        for i in range(80):
            b=pack(0.75+i/32,fmt)
            bursts.append((fmt,ins,b,0,expected(b,fmt,ins),1 if fmt<2 else 2))
vectors=bursts+vectors
def encoded(v):
    fmt,ins,lhs,rhs,ref,tol=v
    return (fmt<<106)|(ins<<99)|(lhs<<67)|(rhs<<35)|(ref<<3)|tol
(R/'sim/vectors/vxm_c3_pipeline_cases.hex').write_text(''.join(f'{encoded(v):027x}\n' for v in vectors))
info=dict(count=len(vectors),seed=20261021,counts=counts,
          oracle='Python math double precision; RNE output conversion; DAZ/FTZ contract',
          tolerance='special: FP16/BF16 <=1 ULP, FP32 <=2 ULP; Basic exact')
(R/'sim/vectors/vxm_c3_pipeline_cases.json').write_text(json.dumps(info,indent=2)+'\n')
(R/'sim/vectors/vxm_c3_pipeline_count.svh').write_text(f'localparam integer CASE_COUNT={len(vectors)};\n')
print(json.dumps(info,indent=2))
