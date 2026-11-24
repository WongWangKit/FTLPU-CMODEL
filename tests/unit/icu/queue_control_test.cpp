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
    put(0,3);put(20,32);put(100,std::bit_width(hw::kSramDepthRows-1));put(0,1);
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
    put(20,32);put(100,std::bit_width(hw::kSramDepthRows-1));
    for(unsigned m=0;m<2;++m) {
        put(0,1);put(0,1);put(1,2);put(0,1);put(2,6);put(0,1);
        put(6,11);put(5,8);put(1,10);
        if(m==1)put(0,1); // a later descriptor's delta follows its template
    }
    rejects([&] {
        Queue raw;
        raw.load_raw_macro(IcuMacroQueueKind::Mem,words);
        raw.configure_queue_controls({{26,IcuControlOpcode::WaitEvent,7}},16,8,2);
        while(!raw.raw_macro_ready())raw.prefetch_raw_macro();
        for(unsigned p=0;p<90;++p) {
            if(p==40)raw.notify(7);
            raw.tick();
        }
    }); // A WAIT does not authorize interleaved execution of two Macros.
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
// Two real MEM_PACKET writers, distinct banks' queues and independently late waves.
void two_page_last_tile(bool reverse)
{
    auto sys=std::make_unique<TspSliceSystem>();auto& icu=sys->icu();
    auto& mem=sys->mem_array(Hemisphere::East);
    icu.configure_queue_completion_events(77,{{17,{400,401}}});
    const auto queue=[](unsigned p){return InstructionControlUnit::mem_queue(Hemisphere::East,p,0);};
    for(unsigned p=0;p<2;p++){
        auto q=queue(p);auto& raw=icu.mem_iq(q);raw.load_raw_macro(IcuMacroQueueKind::Mem,image());
        raw.configure_queue_controls({{0,kQueueControlMemPacket,1},{0,IcuControlOpcode::WaitEvent,17}},16,8,2);
        icu.push_mem_transport_synchronized_raw(q,InstructionControlUnit::encode_mem_synchronized_packet(
            1,90+p,0,1,MemInstruction::Write(30+p,StreamId::East(p)),1));
        icu.configure_mem_transport_owner(q,0,77,90+p,true);mem.configure_transport_commit(77,90+p,p,0,1);
        icu.enable_queue_control_transport(q);icu.mem_transport_iq(q).launch();
    }
    icu.prime_raw_macro_frontends();auto tick=[&]{sys->tick(TspSliceSystem::LogSinks{});};
    tick();
    for(unsigned visit=0;visit<2;visit++){
        auto p=visit^unsigned(reverse);auto q=queue(p);icu.notify_mem_synchronized(IcuLocation::Mem(Hemisphere::East,p,0),90+p);tick();
        for(unsigned tile=0;tile<hw::kTileRows;tile++){
            for(unsigned b=0;b<hw::kLanesPerTile;b++){
                auto cell=StreamCell::Valid(31+64*p+tile*8+b);cell.transport=true;
                sys->stream_fabric(Hemisphere::East).initialize_cell(mem.ports().input_column(p,StreamDirection::East),tile,b,StreamId::East(p),cell);
            }
            tick();require(mem.transport_committed_segments(77,90+p,p,0)==tile+1,"tile retirement mismatch");
            require(!icu.queue_completion_ready(17),"all-of accepted an uncommitted page");
            for(unsigned b=0;b<hw::kLanesPerTile;b++)
                require(mem.sram_lane_byte(p,0,tile,30+p,b)==31+64*p+tile*8+b,"physical page contents");
        }
        icu.notify_queue_mem_commit(77,400+p);icu.notify_queue_mem_commit(77,400+p);
        require(icu.queue_completion_ready(17)==bool(visit),"duplicate completed the other page");
        if(!visit){for(unsigned t=0;t<7;t++){tick();require(icu.mem_iq(q).blocked_on_event(17),"early first-bank release");}}
    }
    auto a=IcuLocation::Mem(Hemisphere::East,0,0),b=IcuLocation::Mem(Hemisphere::East,1,0);
    icu.push_event(76,17,{a,b},true);tick();
    require(icu.blocked_on_tagged_event(a,17)&&icu.blocked_on_tagged_event(b,17),"old generation released new pages");
    icu.push_event(77,17,{a,b},true);tick();
    require(icu.take_phase_fire(77,17).has_value() && !icu.blocked_on_tagged_event(a,17) &&
        !icu.blocked_on_tagged_event(b,17),"all-of did not release both destinations together");
}
void staged_reset_preserves_transport_owner()
{
    auto system=std::make_unique<TspSliceSystem>();
    auto& icu=system->icu();
    constexpr std::size_t slice=0;
    const auto writer=InstructionControlUnit::mem_queue(
        Hemisphere::East,slice,0);
    icu.stage_program_reset();
    icu.configure_mem_transport_owner(writer,0,40,90,true);
    system->tick(TspSliceSystem::LogSinks{});

    icu.push_mem_transport_synchronized_raw(writer,
        InstructionControlUnit::encode_mem_synchronized_packet(1,90,0,1,
            MemInstruction::Write(30,StreamId::East(0)),1));
    system->mem_array(Hemisphere::East).configure_transport_commit(
        40,90,slice,0,1);
    icu.mem_transport_iq(writer).launch();
    icu.stage_program_loading(false);
    icu.stage_program_issue_enabled(true);
    system->tick(TspSliceSystem::LogSinks{});
    icu.notify_mem_synchronized(IcuLocation::Mem(writer),90);
    system->tick(TspSliceSystem::LogSinks{});

    const auto column=system->mem_array(Hemisphere::East)
        .ports().input_column(slice,StreamDirection::East);
    for(std::size_t tile=0;tile<hw::kTileRows;++tile) {
        for(std::size_t lane=0;lane<hw::kLanesPerTile;++lane) {
            auto cell=StreamCell::Valid(static_cast<std::uint8_t>(42+tile));
            cell.transport=true;
            system->stream_fabric(Hemisphere::East).initialize_cell(
                column,tile,lane,StreamId::East(0),cell);
        }
        system->tick(TspSliceSystem::LogSinks{});
        require(system->mem_array(Hemisphere::East)
                .transport_committed_segments(40,90,slice,0)==tile+1,
            "staged reset discarded the new MEM transport owner");
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
// Product dispatch/clock path; use two physical banks, not a standalone arbiter.
void qctl_event_edges()
{
    auto system=std::make_unique<TspSliceSystem>();
    auto& icu=system->icu();
    const auto a=IcuLocation::Mem(Hemisphere::East,0,0), b=IcuLocation::Mem(Hemisphere::East,0,1);
    auto setup=[&](unsigned bank,std::vector<IcuQueueControl> controls){
        auto& q=icu.mem_iq(bank); q.load_raw_macro(IcuMacroQueueKind::Mem,image());
        q.configure_queue_controls(std::move(controls),16,8,2);
    };
    setup(0,{{0,IcuControlOpcode::WaitEvent,53},{1,IcuControlOpcode::WaitEvent,55},
             {2,IcuControlOpcode::WaitEvent,55}});
    setup(1,{{3,IcuControlOpcode::WaitEvent,54},{5,IcuControlOpcode::WaitEvent,55}});
    icu.set_program_generation(41);icu.prime_raw_macro_frontends();
    auto tick=[&]{system->tick(TspSliceSystem::LogSinks{});};
    auto retired=[&](unsigned i){return icu.mem_iq(i).queue_controls_retired();};
    rejects([&]{icu.broadcast_tagged_notification(53);});
    icu.push_event(40,53,{a}); tick();
    require(!retired(0),"stale generation released same tag");
    icu.push_event(41,54,{b}); icu.push_event(41,53,{a}); tick();
    require(retired(0)==1 && !retired(1),"disjoint wait did not bypass oldest unready target");
    icu.push_event(41,55,{a,b}); icu.push_event(41,55,{a});
    for(unsigned i=0;i<9;++i) tick();
    require(retired(0)==3 && retired(1)==2,"repeated tag or staggered multicast lost a target");
    std::cout<<"QCTL: stale=0; tag53 before 54; multicast + repeated 55: retired=3/2\n";
}
void qctl_wrong_tag_and_phase()
{
    auto system=std::make_unique<TspSliceSystem>();auto& icu=system->icu();
    const auto a=IcuLocation::Mem(Hemisphere::East,0,0),b=IcuLocation::Mem(Hemisphere::East,0,1);
    auto& qa=icu.mem_iq(0);qa.load_raw_macro(IcuMacroQueueKind::Mem,image());
    qa.configure_queue_controls({{0,IcuControlOpcode::WaitEvent,61},{1,IcuControlOpcode::WaitEvent,62}},16,8,2);
    auto& qb=icu.mem_iq(1);qb.push_nop(3);qb.append_control(IcuControlInstruction::WaitEvent(61));qb.launch();
    icu.set_program_generation(5);icu.prime_raw_macro_frontends();
    auto tick=[&]{system->tick(TspSliceSystem::LogSinks{});};
    icu.push_event(5,61,{a,b},true);
    unsigned held=0;
    while(qa.queue_controls_retired()==0) {
        require(held++<12,"mixed phase stalled");tick();
        if (!qa.queue_controls_retired()) require(qa.blocked_on_event(61),"partial phase release");
    }
    require(icu.take_phase_fire(5,61).has_value(),"missing phase release receipt");
    require(held>=3 && !qb.blocked_on_event(61),"Native and QCTL did not clear phase together");
    // Both records name the same waiting target. The oldest is selected
    // even with a mismatched tag (no tag search or arbitrary reordering).
    icu.push_event(5,60,{a});icu.push_event(5,62,{a});tick();tick();
    require(qa.blocked_on_event(62) && qa.queue_controls_retired()==1,"tag match skipped selected record");
    for (unsigned n=0;n<6;++n) {
        require(icu.event_in_ready({a}),"lost event credit");icu.push_event(5,60,{a});
    }
    require(!icu.event_in_ready({a}),"uncommitted events overbooked full buffer");
    rejects([&]{icu.push_event(5,60,{a});});
    tick();require(!icu.event_in_ready({a}),"full buffer lost an unconsumed record");
    icu.set_program_generation(6);tick();
    require(icu.event_in_ready({a}) && qa.blocked_on_event(62),"old buffered records advanced new generation");
    icu.push_event(6,62,{a});tick();require(qa.queue_controls_retired()==2,"same-tag reload failed");
}
int main() try {
    if constexpr (!hw::kCommonBarrierCapacity) rejects([] { IcuCommonBarrierController c; c.configure(1,{{0,1}},{{1,{9}}}); });
    qctl_wrong_tag_and_phase();qctl_event_edges();shared_pool_refill();compare(false,true);compare(true,true);
    two_page_last_tile(false);two_page_last_tile(true);staged_reset_preserves_transport_owner();
    finite_frontend();compare(false);compare(true);transport_order();completion_generation();trailing_nops();
    overlapping_contexts();response_during_wait();
    issued_data_keeps_flowing(false);issued_data_keeps_flowing(true);
    std::cout<<"queue controls passed: native physical issue equivalence, same-cycle WAIT chain, SYNC/NOTIFY slots, active Macro, independent queue, finite RAM, shared 4-entry caches and runtime refill\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
