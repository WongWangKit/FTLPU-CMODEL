#define main baseline_main
#include "log_precision.cpp"
#undef main

double qfixed(double x, int n) { return std::ldexp(std::nearbyint(std::ldexp(x,n)),-n); }
struct RawPlan { int b,i; std::array<double,64> base; std::array<float,64> inv; };
RawPlan raw_plan(int b,int inv) {
    RawPlan p{b,inv,{},{}};
    for(int j=0;j<64;++j) {
        double c=1.0+j/64.0;
        p.base[j]=qfixed(std::log(c),b);
        p.inv[j]=static_cast<float>(qfixed(1/c,inv));
    }
    return p;
}
float raw_eval(float x,const RawPlan& p) {
    auto bits=std::bit_cast<std::uint32_t>(x);
    auto fraction=bits&0x7fffff;
    int e=static_cast<int>((bits>>23)&255)-127;
    int i=fraction>>17;
    float m=std::bit_cast<float>(0x3f800000u|fraction);
    float c=1.0f+i/64.0f;
    bool near=x>=0.984375f&&x<=1.015625f;
    float u;
    if(near) {u=x-1.0f;i=0;e=0;}
    else u=(m-c)*p.inv[i];
    float h=-0.25f;
    h=h*u+1.0f/3.0f;
    h=h*u-0.5f;
    h=h*u+1.0f;
    float polynomial=u*h;
    if(near) return polynomial;
    // Signed 48-bit, 40-fraction-bit restoration. No early FP32 conversion
    // of the fixed LUT base, hence its guard bits survive cancellation.
    double restored=e*qfixed(LN2,40)+p.base[i]+qfixed(polynomial,40);
    return static_cast<float>(restored);
}
int main(int argc,char**) {
    std::cout<<"scope,plan,count,max_ulp,over_1ulp,over_2ulp,correctly_rounded,max_abs,max_rel,worst_x_fp32_bits\n";
    std::vector<RawPlan> plans;
    if(argc>1) plans.push_back(raw_plan(28,21));
    else for(int b:{24,26,28,29,30,31}) for(int i:{18,20,21,22,23}) plans.push_back(raw_plan(b,i));
    std::vector<Stats> stats(plans.size());
    int start=argc>1?1:126,end=argc>1?254:128;
    for(int be=start;be<=end;++be) {
        for(std::uint32_t f=0;f<(1u<<23);++f) {
            float x=std::bit_cast<float>((static_cast<std::uint32_t>(be)<<23)|f);
            double ref=std::log(static_cast<double>(x));
            for(std::size_t j=0;j<plans.size();++j) stats[j].observe(x,raw_eval(x,plans[j]),ref,32);
        }
        if(argc>1&&be%32==0) std::cerr<<"Completed exponent "<<be<<"/254\n";
    }
    for(std::size_t j=0;j<plans.size();++j) {
        auto dummy=make_plan("B"+std::to_string(plans[j].b)+"_I"+std::to_string(plans[j].i)+"_fixed_restore40",false,4,-1,false);
        print(argc>1?"FP32_ALL_POSITIVE_NORMAL":"FP32_MANTISSAS_E_MINUS1_0_1",dummy,stats[j]);
    }
}
