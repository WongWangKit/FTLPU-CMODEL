#include "sxm_matrix_test_support.hpp"
#include <iostream>

using namespace sxm_test;

static void run(SxmMatrixDtype dtype, bool mapped, std::uint64_t origin)
{
    auto j=job(dtype,mapped), copy=j;
    // Lane maps can repeat within a tile, unlike the tile bijection.
    if (mapped) j.lane_map[0][7]=j.lane_map[0][0];
    copy=j;
    SxmMatrixCore core; core.reset(origin); core.start(j);
    j.tile_map.fill(0); j.lane_map[0].fill(7); j.matrices=1; // private snapshot
    const auto end=9+8*(copy.matrices*T+T-2);
    std::size_t blocks=0;
    for (std::size_t c=0;c<=end;++c) {
        auto in=inputs(copy,c);
        if (dtype==SxmMatrixDtype::BFloat16)
            for (auto& row:in) if(row) for(auto& v:row->words) v|=0xabcd0000u;
        auto out=core.tick(in);
        for(std::size_t d=0;d<T;++d) {
            bool due=false;
            // Oracle enumerates original matrix/block coordinates independently.
            for(std::size_t m=0;m<copy.matrices;++m) for(std::size_t a=0;a<T;++a)
                for(std::size_t s=0;s<T;++s) if(copy.tile_map[a]==d && c==9+8*T*m+8*(a+s)) {
                    due=true; assert(out[d]); ++blocks;
                    assert(out[d]->matrix==m && out[d]->source==s);
                    for(std::size_t q=0;q<8;++q) for(std::size_t l=0;l<8;++l)
                        assert(out[d]->words[q][l]==expected(copy,m,s*8+q,d,l));
                }
            assert(bool(out[d])==due);
        }
    }
    assert(!core.busy() && !core.fault() && blocks==2*T*T);
    for(std::size_t s=0;s<T;++s) {
        assert(core.peaks()[s]==s+1 && core.occupancy(s)==0);
        assert(core.received()[s]==2*8*T);
    }
    core.start(copy); // new private context without resetting the timer
    core.tick(); assert(core.busy());
}

int main() try {
    for(auto dt:{SxmMatrixDtype::BFloat16,SxmMatrixDtype::Float32})
        for(bool map:{false,true}) for(auto start:{0ull,65530ull}) run(dt,map,start);
    for(int bad=0;bad<5;++bad) {
        auto j=job(SxmMatrixDtype::Float32); SxmMatrixCore c;
        if(bad==0) j.matrices=0;
        if(bad==1) j.matrices=65536;
        if(bad==2) j.dtype=static_cast<SxmMatrixDtype>(9);
        if(bad==3) j.tile_map[1]=j.tile_map[0];
        if(bad==4) j.lane_map[0][0]=8;
        rejects([&]{c.start(j);}); assert(c.fault()); c.reset(); assert(!c.fault());
    }
    {
        auto j=job(SxmMatrixDtype::Float32); SxmMatrixCore c; c.start(j);
        rejects([&]{c.start(j);}); assert(c.fault());
    }
    for(int bad=0;bad<3;++bad) {
        auto j=job(SxmMatrixDtype::Float32); SxmMatrixCore c; c.start(j);
        if(bad==0) { auto in=inputs(j,1); rejects([&]{c.tick(in);}); }
        if(bad==1) { c.tick(); auto in=inputs(j,1); in[0]->lane_valid=0x7f; rejects([&]{c.tick(in);}); }
        if(bad==2) { for(int t=0;t<9;++t)c.tick(inputs(j,t)); auto ready=c.all_ready(); ready[0]=false;
                    rejects([&]{c.tick(inputs(j,9),ready);}); }
        assert(c.fault()); c.reset(); assert(!c.busy() && !c.fault());
    }
    // Incomplete blocks keep the core busy; reset discards validity/pointers.
    {
        SxmMatrixCore c; auto j=job(SxmMatrixDtype::Float32);
        c.start(j); c.tick(); c.tick(inputs(j,1));
        for(int i=0;i<20;++i) c.tick();
        assert(c.busy()); c.reset(); assert(!c.busy());
    }
    std::cout<<"PASS matrix core T="<<T<<" slots="<<SxmMatrixCore::kSlots
             <<" payload_bytes="<<SxmMatrixCore::kPayloadBytes<<" timer_wrap/private_config/same_edge/faults\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
