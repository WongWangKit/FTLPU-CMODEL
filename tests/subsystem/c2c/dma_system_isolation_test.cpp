#include "ftlpu/system/c2c_dma_system.hpp"
#include <iostream>
#include <set>
using namespace ftlpu;
static void check(bool ok,const char* text){if(!ok) throw std::runtime_error(text);}
static void wire_fault(unsigned which) {
    Ddr4Config cfg; cfg.beat_bytes=hw::kPhysicalVectorBytes/2;cfg.request_queue_depth=16;
    auto s=std::make_unique<C2cDmaSystem>(cfg,4);auto& d=s->ddr4();
    auto& east=s->dma(Hemisphere::East);auto& west=s->dma(Hemisphere::West);
    east.issue(C2cDmaInstruction::Load(0x1000,12));west.issue(C2cDmaInstruction::Store(0x4000,12));
    bool fired=false,caught=false;AxiDmaMaster::Counters before;
    d.set_response_wire([&](auto& r,auto& b){
        if(fired)return;
        if(which<3 && r && (which==0 || r->last==(which==2))){
            before=d.axi().counters();if(!which)r->id=d.axi().config().ids;else r->last=!r->last;fired=true;
        } else if(which==3 && b){before=d.axi().counters();b->id=d.axi().config().ids;fired=true;}
    });
    for(int n=0;n<1000 && !caught;n++){
        if(n<12 && west.can_send())west.send({});
        try{s->tick();}catch(const AxiResponseProtocolError&){caught=true;}
    }
    check(caught && fired && s->protocol_isolated() && !s->kernel_done() && !s->can_load_program(),"uncontained wire error");
    const auto& a=d.axi().counters();
    check(a.live==before.live && a.rx_reserved==before.rx_reserved && a.tx_reserved==before.tx_reserved &&
        a.r==before.r && a.b==before.b && a.w==before.w && a.live,"fault edge reassigned a response/credit");
    check(!east.can_issue() && !d.can_accept_request() && !d.drained() &&
        !d.axi().submit_read(Ddr4Model::kProgramClient,0,16),"quarantine allowed fresh traffic");
    const auto t=s->cycle();
    for(unsigned i=0;i<3;i++){
        bool no=false;try{if(i==0)s->tick();if(i==1)s->begin_task_error_recovery();if(i==2)s->reset_execution_state();}
        catch(const std::logic_error&){no=true;}check(no && t==s->cycle(),"quarantine escaped");
    }
}
static void capacity(){
    // 16 transactions, eight IDs x two tickets, 16 beats per direction;
    // 8 two-beat reads and 8 two-beat writes fill *both* payload pools.
    Ddr4Config cfg;cfg.request_queue_depth=16;
    auto s=std::make_unique<C2cDmaSystem>(cfg,4);auto& d=s->ddr4();auto& m=d.axi();
    const auto n=m.config().rx_beats/8*m.config().beat_bytes;
    std::vector<std::uint8_t> data(n);
    for(unsigned epoch=0;epoch<32;epoch++){
        std::vector<AxiDmaMaster::Ticket> ts;std::set<std::size_t> ids;
        for(unsigned i=0;i<16;i++){
            const auto addr=4096ull*(i+1);const auto owner=i==0?Ddr4Model::kProgramClient:i;
            std::fill(data.begin(),data.end(),std::uint8_t(i+epoch));
            if(!(i%2)){C2cVector v;for(auto& t:v.payload)t.fill(i+epoch);
                for(unsigned j=0;j<n;j+=hw::kPhysicalVectorBytes)d.initialize_vector(addr+j,v);}
            auto id=i%2?m.submit_write(owner,addr,data):m.submit_read(owner,addr,n);
            check(id.has_value(),"premature full");ts.push_back(*id);
        }
        check(m.counters().live==16 && m.counters().rx_reserved==m.config().rx_beats &&
            m.counters().tx_reserved==m.config().tx_beats&&!d.can_accept_request(),"incomplete saturation");
        d.set_channel_service([&](auto cycle){return AxiMemory::ChannelService{true,true,true,cycle%23>=7,cycle%17>=5};});
        for(unsigned k=0;k<2000 && !m.transport_idle();k++){
            auto v=m.eval();if(v.ar)ids.insert(v.ar->id);if(v.aw)ids.insert(v.aw->id);s->tick();
        }
        check(ids.size()==8 && m.transport_idle() && !d.drained() && !m.can_submit(1,false),"ID/completion ownership");
        for(unsigned j=0;j<128;j++)s->tick(); // delayed owners keep all reservations
        for(int i=15;i>=0;i--){
            auto v=m.pop(i?i:Ddr4Model::kProgramClient);check(v.ticket==ts[i] && !v.error && !v.aborted,"wrong owner");
            if(i%2){for(unsigned j=0;j<n;j+=hw::kPhysicalVectorBytes){const auto vec=d.read_vector(4096ull*(i+1)+j);
                for(auto& t:vec.payload)for(auto b:t)check(b==i+epoch,"wrong write");}}
            else check(v.bytes==std::vector<std::uint8_t>(n,i+epoch),"wrong read");
        }
        check(d.drained() && !m.counters().rx_reserved && !m.counters().tx_reserved,"leak");
    }
}
static void orphan_after_drain(bool write) {
    auto s=std::make_unique<C2cDmaSystem>();auto& d=s->ddr4();
    if(write)d.request_write(0,{});else d.request_read(0);
    std::optional<AxiReadBeat> old_r;std::optional<AxiWriteResponse> old_b;
    d.set_response_wire([&](auto& r,auto& b){if(r)old_r=r;if(b)old_b=b;});
    for(unsigned i=0;i<2000 && !d.idle();i++)s->tick();
    const auto owner=Ddr4Model::kDirectClient;
    const auto ticket=d.axi().peek(owner).ticket;
    if(write)d.pop_write_completion(ticket);else d.pop_read_completion(ticket);
    check(d.drained() && (write?bool(old_b):bool(old_r)),"missing original response");
    d.set_response_wire([&](auto& r,auto& b){r=old_r;b=old_b;});
    bool caught=false;try{s->tick();}catch(const AxiResponseProtocolError&){caught=true;}
    check(caught && s->protocol_isolated() && !d.drained() && !s->can_load_program(),"late duplicate reopened empty system");
    bool blocked=false;try{d.configure(d.config());}catch(const std::logic_error&){blocked=true;}
    check(blocked,"configure cleared a wire fault");
}
static void stopped_consumers(){
    constexpr auto v=hw::kPhysicalVectorBytes;constexpr unsigned hold=100000, per=4, words=2048;
    Ddr4Config cfg;cfg.request_queue_depth=16;Ddr4Model d(cfg);
    C2cDmaEngine rx(d,4,8),tx(d,4,8,8);InstructionFetchDma pg(d);
    rx.enable_clocked_mode();tx.enable_clocked_mode();pg.enable_clocked_mode();d.enable_clocked_mode();
    auto vec=[](std::size_t a){C2cVector x;for(auto& t:x.payload)for(auto& b:t)b=(a*37+(a>>6))%251,++a;return x;};
    constexpr unsigned program=0x100000,src=0x200000,dst=0x300000;
    for(unsigned i=0;i<words*12;i+=v)d.initialize_vector(program+i,vec(program+i));
    pg.start({program,words,PhysicalInstructionWordWidth::Bits96,9});
    for(unsigned i=0;i<C2cDmaEngine::kDescriptorSlots;++i){
        const auto a=i*per*v;
        rx.issue(C2cDmaInstruction::Load(src+a,per,v,0,i%8));
        tx.issue(C2cDmaInstruction::Store(dst+a,per,v,i%8));
        for(unsigned j=0;j<per;j++)d.initialize_vector(src+a+j*v,vec(src+a+j*v));
    }
    check(!rx.can_issue()&&!tx.can_issue(),"descriptor table not saturated");
    d.set_channel_service([](auto t){AxiMemory::ChannelService s;
        s.ar=t%23>3;s.aw=t%19>2;s.w=t%17>3;s.r=t%31>5;s.b=t%29>7;
        if(t>=400 && t<100400)s.r=false;return s;});
    std::array<unsigned,8> nr{},nw{};unsigned wp=0,cr=0,cw=0;bool full=false;std::size_t total=0;
    for(;total<130000;++total){
        if(total>=hold && pg.word_ready()){
            auto word=pg.front_word();check(word.index==wp,"program word order");
            for(unsigned j=0;j<12;j++){auto a=program+wp*12+j;check(word.bytes[j]==(a*37+(a>>6))%251,"program word data");}
            pg.pop_word();wp++;
        }
        for(unsigned l=0;l<8;l++){
            auto offset=[&](unsigned n){return ((n/per*8+l)*per+n%per)*v;};
            if((total>=hold || (l&1))&&rx.receive_ready(l)){
                check(rx.pop_received(l).payload==vec(src+offset(nr[l])).payload,"RX data overwritten");++nr[l];}
            if(nw[l]<32 && tx.can_send(l)){tx.send(vec(src+offset(nw[l])),l);++nw[l];}
            full|= !(l&1) && rx.receive_queue_size(l)==4 && total<hold;
        }
        pg.low();rx.low();tx.low();d.low();pg.posedge();rx.posedge();tx.posedge();d.posedge();
        pg.high();rx.high();tx.high();d.high();
        while(rx.take_completion_notification())++cr;while(tx.take_completion_notification())++cw;
        while(rx.take_completion_sync_tag()){}while(tx.take_completion_sync_tag()){}
        if(total==hold-1)check(full && !wp && nr[0]==0 && nr[1]>0 && nw[0]>0,"stop did not exercise independent owners");
        if(cr==64&&cw==64&&wp==words&&rx.drained()&&tx.drained()&&pg.drained()&&d.drained())break;
    }
    check(total<130000 && total>=100400 && d.axi().counters().live==0,"long stall did not retire");
    for(unsigned l=0;l<8;l++)check(nr[l]==32 && nw[l]==32,"missing vector");
    for(unsigned n=0;n<256;n++)check(d.read_vector(dst+n*v).payload==vec(src+n*v).payload,"wrong stored vector");
}
static void atomic_descriptor_admission(){
    constexpr auto v=hw::kPhysicalVectorBytes;Ddr4Config cfg;cfg.beat_bytes=v/2;cfg.request_queue_depth=1;
    Ddr4Model d(cfg);C2cDmaEngine dma(d);
    for(bool write:{false,true}){bool rejected=false;
        // First vector fits; only the second would cross 4 KiB.
        try{dma.issue(write?C2cDmaInstruction::Store(4096-v,2,v/2):C2cDmaInstruction::Load(4096-v,2,v/2));}
        catch(const std::invalid_argument&){rejected=true;}
        check(rejected&&dma.idle()&&d.drained()&&d.cycle()==0,"partial admission of impossible vector");
    }
    dma.issue(C2cDmaInstruction::Load(4096-v,2,v));
    unsigned got=0;
    for(unsigned i=0;i<1024 && got<2;i++){dma.tick();d.tick();if(dma.receive_ready()){dma.pop_received();++got;}}
    check(got==2 && dma.drained() && d.drained(),"Q1 rejected aligned non-split transfers");
}
int main()try{for(unsigned i=0;i<4;i++)wire_fault(i);orphan_after_drain(false);orphan_after_drain(true);capacity();stopped_consumers();atomic_descriptor_admission();std::cout<<"PASS wire=6 full_pool_epochs=32 descriptors=128 stalled_R=100000 Q1=atomic\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
