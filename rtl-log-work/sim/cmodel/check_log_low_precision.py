"""Exhaust low-precision normal inputs using the selected RTL flow."""
import math,struct,json
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def low(x,fmt):
    if fmt==0:
        result=struct.unpack('<e',struct.pack('<e',x))[0]
        return math.copysign(0.0,result) if abs(result)<2**-14 else result
    v=bits(x);v=(v+0x7fff+((v>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',v))[0]
def code(x,fmt):
    return int.from_bytes(struct.pack('<e',x),'little') if fmt==0 else bits(x)>>16
LN2=round(math.log(2)*2**40)
rows=[(round(math.log(1+i/64)*2**28),round(1/(1+i/64)*2**21)) for i in range(64)]
results=[]
for fmt,fb,bias,maxe,bfrac,ifrac in [(0,10,15,30,15,12),(1,7,127,254,14,9)]:
    maximum=0;worst=0;over=0;count=0
    def r(x):return low(f32(x),fmt)
    for ex in range(1,maxe+1):
        for fraction in range(1<<fb):
            x=math.ldexp(1+fraction/2**fb,ex-bias)
            e=ex-bias;i=fraction>>(fb-6)
            delta=(fraction&((1<<(fb-6))-1))/2**fb
            near=abs(x-1)<= (1/128 if fmt==1 else 1/64)
            b,inv=rows[i]
            b=round(b/2**(28-bfrac))*2**(28-bfrac)
            inv=round(inv/2**(21-ifrac))*2**(21-ifrac)
            if near:delta=x-1;b=0;inv=2**21;e=0
            u=r(delta*(inv/2**21))
            h=1.0 if fmt==1 else r(r(-0.5*u)+1.0)
            p=r(u*h)
            y=p if near else f32((e*LN2+(b<<12)+round(p*2**40))/2**40)
            got=code(low(y,fmt),fmt)
            expected=code(low(f32(math.log(x)),fmt),fmt)
            error=abs(got-expected);count+=1;over+=error>1
            if error>maximum:maximum=error;worst=(ex<<fb)|fraction
    results.append(dict(format=['FP16','BF16'][fmt],count=count,max_ulp=maximum,over_1ulp=over,worst_input=hex(worst)))
print(json.dumps(results,indent=2))
if any(r['max_ulp']>1 for r in results):raise SystemExit(1)
