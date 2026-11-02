#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/icu/queue_completion.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include <iostream>
#include <stdexcept>
#include <tuple>

using namespace ftlpu;
using Queue=DistributedIcuQueue<MemInstruction,96,64,8,2,4,IcuQueueRole::Mem>;
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f) { bool failed=false;try {f();}catch(const StaticScheduleError&){failed=true;}require(failed,"invalid control accepted"); }
std::vector<IcuRawImemWord96> image()
{
    std::vector<IcuRawImemWord96> words(2);
    words[0].lanes[0]=0x17;
    unsigned offset=0;
    const auto put=[&](std::uint32_t value,unsigned width) {
        for(unsigned b=0;b<width;++b,++offset)
            if ((value>>b)&1u) words[1].lanes[offset/32]|=1u<<(offset%32);
    };
    put(0,3);put(20,32);put(100,13);put(0,1);
    put(0,1);put(1,2);put(0,1);put(2,6);put(0,1);
    put(6,11);put(2,8);put(1,10); // 7 reads at local 20,23,...38
    return words;
}
void compare(bool late, bool shared=false)
{
    Queue raw,native,independent;
    raw.load_raw_macro(IcuMacroQueueKind::Mem,image());
    IcuQueueControlPool pool;
    std::vector<IcuQueueControl> controls{{26,IcuControlOpcode::WaitEvent,7},
        {26,IcuControlOpcode::WaitEvent,8},{29,IcuControlOpcode::Sync,0},
        {29,IcuControlOpcode::Notify,0}};
    if(shared) {
        pool.configure(512,1,4,2);
        raw.configure_shared_queue_controls(pool,controls,16,39);
    } else raw.configure_queue_controls(controls,16,8,2);
    while(!raw.raw_macro_ready()) {pool.begin_cycle();raw.prefetch_raw_macro();pool.end_cycle(true);}
    native.push_nop(20);
    for(unsigned i=0;i<7;++i) {
        if(i==2) {native.append_control(IcuControlInstruction::WaitEvent(7));native.append_control(IcuControlInstruction::WaitEvent(8));}
        if(i==3) {native.append_control(IcuControlInstruction::Sync());native.append_control(IcuControlInstruction::Notify());}
        native.push_instruction(MemInstruction::Read(100+i,2));
        if(i!=6) native.push_nop(2);
    }
    independent.push_nop(27);independent.push_instruction(MemInstruction::Read(999,0));
    native.launch();independent.launch();
    std::vector<std::pair<unsigned,std::size_t>> a,b;
    unsigned independentCycle=0,notifyA=0,notifyB=0;
    for(unsigned p=0;p<90;++p) {
        pool.begin_cycle();
        if(p==(late?32u:1u)) {raw.notify(7);native.notify(7);}
        if(p==(late?35u:2u)) {raw.notify(8);native.notify(8);}
        if(p==(late?43u:3u)) {raw.notify();native.notify();}
        if(auto x=raw.tick()) a.emplace_back(p,isa::encode_mem_instruction(*x));
        if(auto x=native.tick()) b.emplace_back(p,isa::encode_mem_instruction(*x));
        if(independent.tick()) independentCycle=p;
        const bool ar=raw.take_notify(),br=native.take_notify();
        require(ar==br,"NOTIFY differs by physical cycle");notifyA+=ar;notifyB+=br;
        if(p==27 && late) require(raw.blocked_on_event(7),"active Macro hid its pseudo WAIT");
        pool.end_cycle();
    }
    require(a==b && a.size()==7,"pseudo controls changed physical FU issue trace");
    require(independentCycle==27,"waiting queue stopped independent computation");
    require(raw.queue_controls_retired()==4 && notifyA==1 && notifyB==1,"control consumed twice or omitted");
    require(raw.done() && native.done(),"controls prevented completion");
}
void finite_frontend()
{
    IcuQueueControlFrontend f;
    f.configure(std::vector<IcuQueueControl>(16),16,8,2);
    require(f.size()==16,"16-entry control RAM lost a resident entry");
    rejects([&]{f.configure(std::vector<IcuQueueControl>(17),16,8,2);});
    rejects([&]{f.configure({{4,IcuControlOpcode::WaitEvent,1},{3,IcuControlOpcode::Sync,0}},4,2,2);});
    rejects([&]{f.configure({{0,IcuControlOpcode::Nop,0}},4,2,2);});
    rejects([&]{f.configure({{0,IcuControlOpcode::Sync,1}},4,2,2);});
    rejects([&]{f.configure(std::vector<IcuQueueControl>(5),4,2,2);});
    f.configure({{0,IcuControlOpcode::WaitEvent,1},{0,IcuControlOpcode::WaitEvent,2}},4,1,3);
    rejects([&]{f.due(0);});
    for(unsigned n=0;n<4;++n) f.prime_cycle();
    require(f.primed() && f.due(0)->operand==1,"RAM response latency lost control");
    f.retire();rejects([&]{f.due(0);}); // no ideal second port / cache bypass
}
void transport_order()
{
    Queue raw,native,transport;
    raw.load_raw_macro(IcuMacroQueueKind::Mem,image());
    raw.configure_queue_controls({{23,kQueueControlMemPacket,1}},16,8,2);
    transport.push_synchronized(2,19,2,1,MemInstruction::Write(900,1));
    transport.launch();
    bool ticked=false;
    raw.set_queue_transport_executor([&](std::uint32_t ordinal) {
        require(ordinal==1,"wrong transport ordinal");ticked=true;
        auto instruction=transport.tick();
        return Queue::QueueTransportResult{instruction,transport.synchronized_completed_count()==1,
            transport.last_trace().action,transport.last_trace().issue_pc};
    });
    while(!raw.raw_macro_ready()) raw.prefetch_raw_macro();
    native.push_nop(20);
    for(unsigned i=0;i<7;++i) {
        if(i==1) native.push_synchronized(2,19,2,1,MemInstruction::Write(900,1));
        native.push_instruction(MemInstruction::Read(100+i,2));
        if(i!=6) native.push_nop(2);
    }
    native.launch();
    std::vector<std::tuple<unsigned,MemOpcode,std::size_t>> a,b;
    for(unsigned p=0;p<90;++p) {
        if(p==31 || p==40) {transport.notify(19);native.notify(19);}
        ticked=false;
        if(auto x=raw.tick()) a.emplace_back(p,x->opcode,isa::encode_mem_instruction(*x));
        if(!ticked) transport.prefetch_transport_cycle();
        if(auto x=native.tick()) b.emplace_back(p,x->opcode,isa::encode_mem_instruction(*x));
    }
    require(a==b && a.size()==9,"shared MEM ownership differs from Native packet order");
    require(transport.synchronized_completed_count()==1 && raw.queue_controls_retired()==1,
        "transport packet retired twice or blocked subsequent compute");
}
void completion_generation()
{
    IcuQueueCompletionEvents e;
    e.configure(4,{{9,{0,1}},{10,{2}}});
    e.notify_mem_commit(3,0); require(!e.ready(9),"old generation completed page");
    e.notify_mem_commit(4,0); e.notify_mem_commit(4,0);
    require(!e.ready(9),"duplicate completion satisfied all-of");
    e.notify_mem_commit(4,1);require(e.ready(9) && !e.ready(10),"same-page reload reused completion");
    rejects([&]{e.notify_mem_commit(5,0);});
    rejects([&]{e.notify_mem_commit(4,3);});
}
void trailing_nops()
{
    Queue raw,native;
    raw.load_raw_macro(IcuMacroQueueKind::Mem,image());
    raw.configure_queue_controls({},16,8,2,47);
    while(!raw.raw_macro_ready()) raw.prefetch_raw_macro();
    native.push_nop(20);
    for(unsigned i=0;i<7;++i) {
        native.push_instruction(MemInstruction::Read(100+i,0));
        native.push_nop(i==6 ? 8 : 2);
    }
    native.launch();
    for(unsigned p=0;p<47;++p) {
        raw.tick();native.tick();
        require(raw.done()==native.done(),"trailing Native NOP extent changed completion edge");
    }
    require(raw.done(),"queue completion extent did not retire");
}
void overlapping_contexts()
{
    std::vector<IcuRawImemWord96> words(3);
    words[0].lanes[0]=0x27;
    unsigned offset=96;
    const auto put=[&](std::uint32_t value,unsigned width) {
        for(unsigned b=0;b<width;++b,++offset)
            if((value>>b)&1u) words[offset/96].lanes[(offset%96)/32]|=1u<<(offset%32);
    };
    put(1,3);put(3,22);put(100,14); // dictionary: next Macro starts +3, address +100
    put(20,32);put(100,13);
    for(unsigned m=0;m<2;++m) {
        put(0,1);put(0,1);put(1,2);put(0,1);put(2,6);put(0,1);
        put(6,11);put(5,8);put(1,10);
        if(m==1)put(0,1); // a later descriptor's delta follows its template
    }
    Queue raw,native;
    raw.load_raw_macro(IcuMacroQueueKind::Mem,words);
    raw.configure_queue_controls({{26,IcuControlOpcode::WaitEvent,7}},16,8,2);
    while(!raw.raw_macro_ready())raw.prefetch_raw_macro();
    native.push_nop(20);
    for(unsigned i=0;i<14;++i) {
        if(i==2)native.append_control(IcuControlInstruction::WaitEvent(7));
        native.push_instruction(MemInstruction::Read((i%2?200:100)+i/2,2));
        if(i!=13)native.push_nop(2);
    }
    native.launch();
    std::vector<std::pair<unsigned,std::size_t>> a,b;
    for(unsigned p=0;p<90;++p) {
        if(p==24){raw.notify(8);native.notify(8);} // wrong tag cannot release
        if(p==40){raw.notify(7);native.notify(7);}
        if(auto x=raw.tick())a.emplace_back(p,isa::encode_mem_instruction(*x));
        if(auto x=native.tick())b.emplace_back(p,isa::encode_mem_instruction(*x));
        if(p==30)require(raw.active_macro_count()==2 && raw.cycle()==26,
            "queue wait discarded or advanced an active Context");
    }
    if(a!=b) {
        for(const auto& [p,x]:a)std::cerr<<"raw "<<p<<' '<<x<<'\n';
        for(const auto& [p,x]:b)std::cerr<<"native "<<p<<' '<<x<<'\n';
    }
    require(a==b && a.size()==14 && raw.peak_active_macros()==2,
        "interleaved Contexts changed Native issue order across WAIT");
}
void response_during_wait()
{
    IcuQueueControlFrontend f;
    f.configure({{0,IcuControlOpcode::WaitEvent,1},{0,IcuControlOpcode::WaitEvent,2},
        {0,IcuControlOpcode::WaitEvent,3}},4,2,3);
    while(!f.primed())f.prime_cycle();
    require(f.due(0)->operand==1,"first control absent");f.retire();
    f.end_cycle(); // reserve cache credit and issue the third RAM request
    for(unsigned p=0;p<4;++p) {
        f.begin_cycle();require(f.due(0)->operand==2,"waiting head overwritten by response");f.end_cycle();
    }
    f.retire();require(f.due(0)->operand==3,"pending RAM response lost during WAIT");
    f.retire();require(f.done(),"RAM response consumed twice");
}
void issued_data_keeps_flowing(bool transport)
{
    auto system=std::make_unique<TspSliceSystem>();
    auto& icu=system->icu();
    auto& raw=icu.mem_iq(0);
    raw.load_raw_macro(IcuMacroQueueKind::Mem,image());
    std::vector<IcuQueueControl> controls;
    if(transport) controls.push_back({0,kQueueControlMemPacket,1});
    controls.push_back({0,IcuControlOpcode::WaitEvent,7});
    raw.configure_queue_controls(std::move(controls),16,8,2);
    const std::size_t slice=transport ? 0 : 1;
    const auto writer=InstructionControlUnit::mem_queue(Hemisphere::East,slice,0);
    if(transport) {
        icu.push_mem_transport_synchronized_raw(writer,
            InstructionControlUnit::encode_mem_synchronized_packet(1,90,0,1,
                MemInstruction::Write(30,StreamId::East(0)),1));
        icu.configure_mem_transport_owner(writer,0,40,90,true);
        system->mem_array(Hemisphere::East).configure_transport_commit(40,90,slice,0,1);
        icu.configure_queue_completion_events(40,{{7,{400}}});
        icu.enable_queue_control_transport(writer);
        icu.mem_transport_iq(writer).launch();
    } else {
        auto& native=icu.mem_iq(writer);
        native.push_instruction(MemInstruction::Write(30,StreamId::East(0)));
        native.append_control(IcuControlInstruction::WaitEvent(8));
        native.launch();
    }
    icu.prime_raw_macro_frontends();
    if(transport) {
        system->tick(TspSliceSystem::LogSinks{});
        icu.notify_mem_synchronized(IcuLocation::Mem(writer),90);
        system->tick(TspSliceSystem::LogSinks{}); // synchronized notification D+1
    }
    const auto column=system->mem_array(Hemisphere::East).ports().input_column(slice,StreamDirection::East);
    bool retiredBeforeCommit=false;
    for(std::size_t tile=0;tile<hw::kTileRows;++tile) {
        // Physical data already in flight arrives while later commands wait.
        for(std::size_t lane=0;lane<hw::kLanesPerTile;++lane) {
            auto cell=StreamCell::Valid(static_cast<std::uint8_t>(42+tile));
            cell.transport=transport;
            system->stream_fabric(Hemisphere::East).initialize_cell(column,tile,lane,StreamId::East(0),cell);
        }
        system->tick(TspSliceSystem::LogSinks{});
        require(raw.cycle()==0 && icu.compute_enabled(),"queue WAIT froze global computation");
        for(std::size_t lane=0;lane<hw::kLanesPerTile;++lane)
            require(system->mem_array(Hemisphere::East).sram_lane_byte(slice,0,tile,30,lane)==42+tile,
                "queue WAIT stopped an already issued consumer from accepting in-flight data");
        if(!transport && tile>0) require(icu.mem_iq(writer).blocked_on_event(8),"consumer was not waiting");
        if(transport) {
            const auto committed=system->mem_array(Hemisphere::East).transport_committed_segments(40,90,slice,0);
            require(committed==tile+1,"commit tracking differs from physical SRAM writes");
            retiredBeforeCommit|=icu.mem_transport_iq(writer).synchronized_completed_count()==1 && committed<hw::kTileRows;
            require(!icu.queue_completion_ready(7),"packet retirement became page readiness");
        }
    }
    if(transport) {
        require(retiredBeforeCommit,"test did not separate packet retirement from final tile commit");
        icu.notify_queue_mem_commit(40,400);
        require(icu.queue_completion_ready(7),"actual completed instance did not become ready");
        require(raw.blocked_on_event(7),"completion skipped queue release phase");
        const auto retired=icu.mem_transport_iq(writer).synchronized_completed_count();
        icu.reset_compute_program();
        require(!icu.queue_control_transport_enabled(writer),
            "kernel switch retained the previous QCTL transport activation gate");
        require(icu.mem_transport_iq(writer).synchronized_completed_count()==retired,
            "compute program reset discarded independent transport state");
        icu.configure_queue_completion_events(41,{{7,{401}}});
        icu.notify_queue_mem_commit(40,400);
        require(!icu.queue_completion_ready(7),"kernel switch reused old page completion");
    }
}
void shared_pool_refill() {
    IcuQueueControlPool pool;pool.configure(512,4,4,2);
    std::vector<IcuQueueControlPool::Handle> handles;
    for(unsigned q=0;q<4;++q) {
        std::vector<IcuQueueControl> entries;
        for(unsigned i=0;i<16;++i)entries.push_back({5+20*i+q,IcuControlOpcode::WaitEvent,i+1});
        handles.push_back(pool.add_partition(entries,16));
    }
    const auto ready=[&]{return std::all_of(handles.begin(),handles.end(),[&](auto h){return pool.primed(h);});};
    while(!ready()) {pool.begin_cycle();pool.end_cycle(true);}
    for(unsigned p=0;p<330;++p) {
        const auto reads=pool.statistics().reads;
        pool.begin_cycle();
        for(unsigned q=0;q<4;++q) {
            const auto* control=pool.due(handles[q],p);
            if(p>=5+q && (p-5-q)%20==0 && p<=305+q) {
                require(control && control->operand==(p-5-q)/20+1,"shared refill lost or reordered control");
                pool.retire(handles[q]);
            } else require(!control,"shared control issued on wrong physical cycle");
        }
        pool.end_cycle();require(pool.statistics().reads-reads<=1,"shared RAM silently gained read ports");
    }
    require(pool.statistics().prime_reads==16 && pool.statistics().run_reads==48,"runtime refill was not exercised");
    require(pool.statistics().peak_cached<=4 && pool.statistics().peak_pending<=2,"shared finite storage exceeded");
    require(std::all_of(handles.begin(),handles.end(),[&](auto h){return pool.done(h);}),"shared control omitted");
    pool.configure(2,1,1,2);
    std::vector<IcuQueueControl> burst{{0,IcuControlOpcode::WaitEvent,1},{0,IcuControlOpcode::WaitEvent,2}};
    auto h=pool.add_partition(burst,16);
    require(!pool.primed(h),"shared RAM has zero latency");
    pool.begin_cycle();pool.end_cycle(true);
    require(!pool.primed(h),"shared RAM returned before read latency");
    while(!pool.primed(h)){pool.begin_cycle();pool.end_cycle(true);}
    pool.retire(h);rejects([&]{pool.due(h,0);});
    pool.begin_cycle();pool.end_cycle(); // second response is in flight at reset
    pool.reset();rejects([&]{pool.due(h,0);});
    pool.configure(1,1,1,2);rejects([&]{pool.add_partition(burst,16);});
    std::vector<IcuQueueControl> overflow(17);
    pool.configure(512,1,4,2);rejects([&]{pool.add_partition(overflow,16);});
}
int main() try {
    shared_pool_refill();compare(false,true);compare(true,true);
    finite_frontend();compare(false);compare(true);transport_order();completion_generation();trailing_nops();
    overlapping_contexts();response_during_wait();
    issued_data_keeps_flowing(false);issued_data_keeps_flowing(true);
    std::cout<<"queue controls passed: native physical issue equivalence, same-cycle WAIT chain, SYNC/NOTIFY slots, active Macro, independent queue, finite RAM, shared 4-entry caches and runtime refill\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
