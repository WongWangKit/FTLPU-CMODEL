#include "ftlpu/c2c/axi_dma.hpp"
#include <iostream>
#include <random>
#include <map>
#include <string>
using namespace ftlpu;
static void check(bool v,const char* s) {if(!v)throw std::runtime_error(s);}
template<class F> static void rejects(F f) {bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"negative test not detected");}
static std::uint8_t pattern(std::uint64_t a) {return static_cast<std::uint8_t>((a*37)^(a>>8));}
struct PendingRead {AxiAddress a;std::size_t pos{};};
struct PendingB {std::size_t id{},due{};};

// Independent subordinate scoreboard: AW and W queues are distinct; read
// responses interleave across ID but never reorder bursts within one ID.
// Software containers here are stimulus/scoreboards, not modeled DUT FIFOs.
static void randomized(unsigned seed,std::size_t depth,std::size_t beat) {
    AxiDmaConfig c;c.beat_bytes=beat;c.max_burst_beats=4;c.transactions=depth;
    c.ids=3;c.per_id=3;c.rx_beats=depth*4;c.tx_beats=depth*4;
    AxiDmaMaster m(c);std::mt19937 rng(seed);
    std::deque<PendingRead> reads;
    std::deque<AxiAddress> aws;
    std::deque<std::vector<AxiWriteBeat>> ws;std::vector<AxiWriteBeat> partial;
    std::deque<PendingB> bs;
    std::map<std::uint64_t,std::uint8_t> written;
    std::vector<std::deque<AxiDmaMaster::Ticket>> tickets(3);
    std::optional<AxiAddress> heldAr,heldAw;std::optional<AxiWriteBeat> heldW;
    std::size_t submitted=0,retired=0,w_before_aw=0,reordered=0;
    for(std::size_t cycle=0;cycle<100000;++cycle) {
        if(submitted<240 && rng()%3) {
            const auto client=submitted%3,addr=(submitted+1)*256;
            const auto length=1+rng()%(4*beat);
            std::optional<std::uint64_t> t;
            if(submitted%2) {
                std::vector<std::uint8_t> bytes(length);
                for(std::size_t b=0;b<length;++b)bytes[b]=pattern(addr+b);
                t=m.submit_write(client,addr,bytes,7);
            } else t=m.submit_read(client,addr,length,7);
            if(t) {tickets[client].push_back(*t);++submitted;}
        }
        const auto o=m.eval();
        if(heldAr)check(o.ar==heldAr,"AR changed while stalled");
        if(heldAw)check(o.aw==heldAw,"AW changed while stalled");
        if(heldW)check(o.w==heldW,"W changed while stalled");
        AxiSlaveSignals i;
        i.arready=cycle%37>7 && rng()%3;
        i.awready=cycle%41>12 && rng()%3;
        i.wready=cycle%43>8 && rng()%3;
        // Choose newest eligible burst; earlier same-ID transactions win.
        if(!reads.empty() && rng()%4) {
            std::size_t chosen=0;
            for(std::size_t j=0;j<reads.size();++j) {
                bool first=true;
                for(std::size_t k=0;k<j;++k) if(reads[k].a.id==reads[j].a.id) first=false;
                if(first)chosen=j;
            }
            auto& r=reads[chosen];if(chosen)++reordered;
            AxiReadBeat rb;rb.id=r.a.id;rb.last=r.pos+1==r.a.beats;rb.bytes.resize(beat);
            for(std::size_t b=0;b<beat;++b)rb.bytes[b]=pattern(r.a.address+r.pos*beat+b);
            i.r=rb;++r.pos;
            if(rb.last)reads.erase(reads.begin()+chosen);
        }
        for(std::size_t j=bs.size();j>0;--j) {
            const auto k=j-1;bool first=true;
            for(std::size_t z=0;z<k;++z)if(bs[z].id==bs[k].id)first=false;
            if(first && cycle>=bs[k].due) {i.b=AxiWriteResponse{bs[k].id,0};bs.erase(bs.begin()+k);break;}
        }
        heldAr=o.ar && !i.arready?o.ar:std::nullopt;
        heldAw=o.aw && !i.awready?o.aw:std::nullopt;
        heldW=o.w && !i.wready?o.w:std::nullopt;
        m.tick(i);
        if(o.ar && i.arready) reads.push_back({*o.ar,0});
        if(o.aw && i.awready)aws.push_back(*o.aw);
        if(o.w && i.wready) {
            if(aws.empty())++w_before_aw;
            partial.push_back(*o.w);
            if(o.w->last){ws.push_back(partial);partial.clear();}
        }
        if(!aws.empty() && !ws.empty()) {
            const auto a=aws.front();aws.pop_front();const auto w=ws.front();ws.pop_front();
            check(w.size()==a.beats,"W interleaved/burst length mismatch");
            for(std::size_t n=0;n<w.size();++n)for(std::size_t b=0;b<beat;++b)
                if(w[n].strobe[b]) {
                    const auto addr=a.address+n*beat+b;
                    check(w[n].bytes[b]==pattern(addr),"W payload/STRB corruption");
                    check(!written.contains(addr),"duplicate write");written[addr]=w[n].bytes[b];
                }
            bs.push_back({a.id,cycle+1+rng()%19});
        }
        // Long client hold simulates stopped consumers without stopping AXI.
        if(cycle>500 && rng()%3) for(std::size_t client=0;client<3;++client)
            if(m.completion_ready(client)) {
                const auto result=m.pop(client);
                check(result.ticket==tickets[client].front(),"client reorder");
                tickets[client].pop_front();check(!result.error && result.generation==7,"completion status");
                if(!result.write)for(std::size_t b=0;b<result.bytes.size();++b)
                    check(result.bytes[b]==pattern(result.address+b),"R data corruption");
                ++retired;
            }
        if(retired==240) {
            check(m.drained(),"leaked transaction/credit");
            check(m.counters().rx_reserved==0 && m.counters().tx_reserved==0,"credit leak");
            std::cout<<"random seed="<<seed<<" depth="<<depth<<" beat="<<beat
                <<" cycles="<<cycle<<" reordered="<<reordered<<" w_before_aw="<<w_before_aw<<" PASS\n";
            return;
        }
    }
    throw std::runtime_error("random test timeout");
}
static void directed() {
    AxiDmaConfig c;c.beat_bytes=8;c.max_burst_beats=256;
    auto p=plan_axi_bursts(c,4088,2057);
    check(p.size()==3 && p[0].bytes==8 && p[1].bytes==2048 && p[2].bytes==1,"4K/maximum/tail split");
    check(plan_axi_bursts(c,0,0).empty(),"zero length plan");
    rejects([&]{plan_axi_bursts(c,1,2);});
    rejects([&]{plan_axi_bursts(c,c.address_limit-8,9);});
    c.rx_beats=1;c.tx_beats=1;c.transactions=3;c.ids=2;c.per_id=2;
    AxiDmaMaster m(c);
    rejects([&]{m.submit_read(0,0,c.address_limit);});
    rejects([&]{m.submit_read(0,4088,9);});
    auto a=m.submit_read(0,0,8);check(a.has_value(),"one beat admission");
    check(!m.submit_read(1,8,1),"held offer did not reserve capacity");
    const auto old=m.eval().ar;
    for(int n=0;n<10000;++n){m.tick({});check(m.eval().ar==old,"long hold AR");}
    AxiSlaveSignals i;i.arready=true;m.tick(i);
    i={};i.r=AxiReadBeat{old->id,std::vector<std::uint8_t>(8,5),true,2};m.tick(i);
    check(m.client_stopped(0),"R error must stop new client work");
    check(!m.submit_read(0,16,8),"error client admission");
    check(m.pop(0).error,"R error missing");m.resume_client(0);
    // WVALID must not wait for AWREADY. W may handshake first.
    auto w=m.submit_write(1,24,{1,2,3});check(w.has_value(),"write admission");
    const auto out=m.eval();check(out.aw && out.w,"independent AW/W");
    check(out.w->strobe==std::vector<std::uint8_t>({1,1,1,0,0,0,0,0}),"tail STRB");
    i={};i.wready=true;m.tick(i);
    check(m.eval().aw && !m.eval().w,"W before AW state");
    rejects([&]{AxiSlaveSignals bad;bad.b=AxiWriteResponse{out.aw->id,0};m.tick(bad);});
    i={};i.awready=true;m.tick(i);
    m.abort_client(1);i={};i.b=AxiWriteResponse{out.aw->id,2};m.tick(i);
    const auto done=m.pop(1);check(done.error && done.aborted && done.transferred_bytes==3,"partial/error completion");
    m.resume_client(1);
    rejects([&]{AxiSlaveSignals bad;bad.r=AxiReadBeat{99,std::vector<std::uint8_t>(8),true,0};m.tick(bad);});
    check(m.drained(),"directed drain");
    std::cout<<"directed capacity/4K/error/abort/negative PASS\n";
}
static void malformed_late_returns() {
    AxiDmaConfig c;c.beat_bytes=8;c.ids=2;c.rx_beats=2;
    AxiDmaMaster m(c);check(m.submit_read(0,0,11).has_value(),"two-beat tail admission");
    const auto id=m.eval().ar->id;
    AxiSlaveSignals s;s.arready=true;m.tick(s);
    m.abort_client(0);
    auto send=[&](std::size_t rid,bool last){AxiSlaveSignals x;x.r=AxiReadBeat{rid,std::vector<std::uint8_t>(8,7),last,0};m.tick(x);};
    auto illegal=[&](std::size_t rid,bool last){
        const auto before=m.counters();rejects([&]{send(rid,last);});
        const auto after=m.counters();
        check(before.r==after.r&&before.live==after.live&&before.rx_reserved==after.rx_reserved&&before.cycles==after.cycles,
            "illegal response changed ownership or advanced the edge");
    };
    illegal(id,true);illegal(id^1,false);send(id,false);illegal(id,false);
    check(!m.completion_ready(0)&&!m.drained(),"abort dropped accepted tail");send(id,true);
    const auto x=m.pop(0);check(x.aborted&&x.transferred_bytes==11&&m.drained(),"late legal tail lost or duplicated");
    illegal(id,true);
}
int main() {
    try {
        directed();malformed_late_returns();
        for(auto seed:{1U,7U,29U}) {
            for(auto depth:{1U,3U,8U}) {
                for(auto beat:{8U,16U,32U}) randomized(seed,depth,beat);
            }
        }
        std::cout<<"AXI DMA component tests PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

