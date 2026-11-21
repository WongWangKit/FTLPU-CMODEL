#include "ftlpu/system/c2c_dma_system.hpp"
#include <iostream>
#include <memory>
using namespace ftlpu;
static void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
template<class F>static void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"negative not caught");}
static void stage(C2cDmaSystem& s,std::uint64_t address,const std::vector<std::uint8_t>& bytes) {
    for(std::size_t n=0;n<bytes.size();n+=hw::kPhysicalVectorBytes) {
        C2cVector v{};for(std::size_t b=0;b<hw::kPhysicalVectorBytes && n+b<bytes.size();++b)
            v.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]=bytes[n+b];
        s.ddr4().initialize_vector(address+n,v);
    }
}
// Exercise descriptor admission without allocating millions of instruction words.
static void target_limits(InstructionControlUnit& icu) {
    using K=IcuMacroQueueKind; using S=ProgramStorage; using M=IcuQueueMode;
    struct Target {K kind; std::size_t count, depth;};
    const Target targets[]={
        {K::Mem,icu.kMemQueues,hw::kIcuMemImemDepth},
        {K::MxmLoad,icu.kMxmQueues,hw::kIcuMxmImemDepth},
        {K::MxmCompute,icu.kMxmQueues,hw::kIcuMxmImemDepth},
        {K::MxmDequant,icu.kMxmQueues,hw::kIcuMxmImemDepth},
        {K::Vxm,icu.kVxmQueues,hw::kIcuVxmImemDepth},
        {K::SxmTranspose,hw::kHemispheres,hw::kIcuSxmImemDepth},
        {K::SxmPermute,hw::kHemispheres,hw::kIcuSxmImemDepth},
        {K::C2cRx,hw::kHemispheres,hw::kIcuC2cImemDepth},
        {K::C2cTx,hw::kHemispheres,hw::kIcuC2cImemDepth},
        {K::C2cDma,hw::kHemispheres,hw::kIcuC2cImemDepth}};
    std::size_t checked=0;
    auto check=[&](const ProgramImage& im,bool legal){
        bool accepted=true;
        try { icu.validate_program_image(im); }
        catch(const std::invalid_argument&) { accepted=false; }
        require(accepted==legal,"target/format/capacity admission mismatch");
        ++checked;
    };
    for(const auto& t:targets) for(auto storage:{S::Imem,S::TransportImem})
        for(auto mode:{M::Native,M::Macro}) for(std::size_t q=0;q<=t.count;++q)
            for(auto n:{std::size_t{0},std::size_t{1},t.depth-1,t.depth,t.depth+1}) {
                ProgramImage im;im.kind=t.kind;im.storage=storage;im.mode=mode;im.queue_index=q;
                im.byte_count=n*im.word_bytes();
                const bool macro=t.kind==K::Mem || t.kind==K::MxmLoad || t.kind==K::MxmCompute ||
                    t.kind==K::MxmDequant || t.kind==K::SxmTranspose || t.kind==K::SxmPermute;
                check(im,q<t.count && n<=t.depth &&
                    (mode==M::Native || (macro && n!=0)) &&
                    (storage==S::Imem || (mode==M::Native && t.kind==K::Mem)));
            }
    for(const auto& t:targets)for(auto st:{S::Qctl,S::Barr,S::BarrierEvents,S::QueueEvents}) {
        const std::size_t cap=st==S::Qctl?hw::kQueueControlCapacity:
            st==S::Barr?hw::kCommonBarrierCapacity:4*65535;
        std::vector<std::size_t> sizes{0,1,cap?cap-1:0,cap,cap+1};
        std::ranges::sort(sizes);
        sizes.erase(std::unique(sizes.begin(),sizes.end()),sizes.end());
        for(std::size_t q=0;q<=t.count;++q)for(auto n:sizes) {
            ProgramImage im;im.kind=t.kind;im.queue_index=q;im.storage=st;im.generation=1;
            im.byte_count=n*im.word_bytes();
            const bool target=st==S::Qctl ? (q<t.count && t.kind!=K::Vxm &&
                t.kind!=K::C2cTx && t.kind!=K::C2cRx && t.kind!=K::C2cDma)
                : (t.kind==K::Mem && q==0);
            check(im,target && n<=cap);
            im.generation=0;check(im,false);
        }
    }
    std::cout<<"descriptor admission cases="<<checked<<'\n';
}
int main() try {
    auto system=std::make_unique<C2cDmaSystem>();auto& loader=system->program_loader();
    const auto sizing=hw::config::kC2cDmaSizing;
    const auto& axi=system->ddr4().axi().config();
    const auto beats=(hw::kPhysicalVectorBytes+axi.beat_bytes-1)/axi.beat_bytes;
    require(loader.dma().config().response_fifo_depth==sizing.p &&
        system->dma(Hemisphere::East).fifo_depth_vectors()==sizing.rt &&
        system->dma(Hemisphere::West).fifo_depth_vectors()==sizing.rt &&
        system->ddr4().config().request_queue_depth==sizing.q &&
        axi.transactions==sizing.q && axi.per_id==(sizing.q+7)/8 &&
        axi.rx_beats==sizing.q*beats && axi.tx_beats==sizing.q*beats,
        "profile DMA sizing not connected to default device");
    std::cout<<"DMA defaults P="<<sizing.p<<" RT="<<sizing.rt<<" Q="<<sizing.q<<'\n';
    target_limits(system->chip().icu());
    // Every data bank is writable, including the former reserved final bank.
    for(std::size_t h=0;h<2;++h)for(std::size_t s=0;s<hw::kMemSliceColumns;++s)
        for(std::size_t b=0;b<2;++b)system->chip().mem_array(static_cast<Hemisphere>(h)).set_sram_byte(s,b,0,0,0x5a);
    require(system->chip().mem_array(Hemisphere::East).read_timing()==MemArrayModel::ReadTiming::RegisteredR1,"R1 lost");
    std::array<ProgramImage,3> images;std::array<std::vector<std::uint8_t>,3> gold;
    for(std::size_t seg=0;seg<3;++seg) {
        auto encoder=std::make_unique<InstructionControlUnit>();
        for(int n=0;n<5;++n)encoder->mem_iq(0).append_control(IcuControlInstruction::Nop(1));
        encoder->enqueue_mem(0,MemInstruction::Read(100+seg,StreamId::East(0)));
        images[seg]={0x1000+seg*256,IcuMacroQueueKind::Mem,0,72,IcuQueueMode::Native};
        gold[seg]=encoder->read_program_image(images[seg]);stage(*system,images[seg].source_address,gold[seg]);
    }
    auto tooLarge=images[0];tooLarge.byte_count=12*(hw::kIcuMemImemDepth+1);
    rejects([&]{loader.submit({tooLarge},0);});
    TspSliceSystem::LogSinks sinks;sinks.capture_mem_trace=true;
    std::size_t bytes=0;
    const std::array<int,8> itinerary{0,1,2,0,2,1,0,1};
    std::array<unsigned,3> visits{};
    for(auto seg:itinerary) {
        const auto row=100+seg+8*visits[seg]++;
        if(visits[seg]>1) {
            auto encoder=std::make_unique<InstructionControlUnit>();
            for(int n=0;n<5;++n)encoder->mem_iq(0).append_control(IcuControlInstruction::Nop(1));
            encoder->enqueue_mem(0,MemInstruction::Read(row,StreamId::East(0)));
            gold[seg]=encoder->read_program_image(images[seg]);stage(*system,images[seg].source_address,gold[seg]);
        }
        const auto ar=system->ddr4().axi().counters().ar;
        auto im=images[seg];im.generation=loader.last_generation()+1;
        loader.submit({im},system->cycle());loader.start();loader.set_dependencies_ready(false);
        std::size_t n=0;
        while(!loader.ready_to_start()){require(n++<5000,"load deadlock");system->tick(sinks);}
        require(!loader.started(),"early START ignored dependency");
        require(system->chip().icu().read_program_image(im)==gold[seg],"direct iMEM mismatch");
        require(system->ddr4().axi().counters().ar>ar,"nonresident segment reused a hidden cache");
        require(loader.statistics().relay_reads==loader.statistics().relay_writes,"Relay record leaked");
        loader.set_dependencies_ready(true);std::size_t observed=0;
        do {
            require(n++<6000,"execution drain deadlock");system->tick(sinks);
            for(const auto& x:system->chip().mem_array(Hemisphere::East).executed_instructions())
                if(x.mem_slice==0 && x.instruction.opcode==MemOpcode::Read) {require(x.instruction.address==row,"stale code after page back edge");++observed;}
        } while(!loader.kernel_done());
        require(observed==hw::kTileRows,"instruction did not execute on each tile");
        require(loader.statistics().all_images_committed<loader.statistics().program_start,"START before commit");
        rejects([&]{loader.submit({im},system->cycle());});
        bytes+=gold[seg].size();
        std::cout<<"segment="<<seg<<" cycle="<<system->cycle()<<" bytes="<<loader.statistics().dma_bytes<<'\n';
    }
    require(loader.total_committed_bytes()==bytes && loader.dma().drained(),"private commit/return accounting");
    for(std::size_t h=0;h<2;++h)for(std::size_t s=0;s<hw::kMemSliceColumns;++s)
        for(std::size_t b=0;b<2;++b)require(system->chip().mem_array(static_cast<Hemisphere>(h)).sram_byte(s,b,0,0)==0x5a,"program loader modified data MEM");
    // A task error retires the right to submit same-generation continuations.
    // Use a legal empty native target so the negative does not depend on a
    // failed underlying DMA rejecting a subsequent read.
    auto fault=images[0];fault.generation=loader.last_generation()+1;
    system->ddr4().set_response_fault([](auto,bool){return std::uint8_t{2};});
    loader.submit({fault},system->cycle());
    for(int t=0;!loader.failed();++t){require(t<1000,"missing loader error");system->tick();}
    system->begin_task_error_recovery();
    for(int t=0;!system->ddr4().drained()||!loader.dma().drained();++t){
        require(t<1000,"failed task did not drain");system->tick();
    }
    auto empty=fault;empty.byte_count=0;
    rejects([&]{loader.submit({empty},system->cycle(),true);});
    require(loader.failed(),"continuation cleared the task error");
    for(int t=0;!system->finish_task_error_recovery();++t){require(t<1000,"recovery hung");system->tick();}
    rejects([&]{loader.submit({empty},system->cycle(),true);});
    require(system->can_load_program()&&loader.last_generation()==fault.generation,"stale segment changed recovered task");
    // Hold an accepted program R while a different client gets a legal
    // BRESP error. Timeout must neither release that R owner nor allow a
    // retry. Only restoring service can finish this recovery.
    bool r_service=false;
    system->ddr4().set_channel_service([&](auto){AxiMemory::ChannelService x;x.r=r_service;return x;});
    system->ddr4().set_response_fault([](auto,bool write){return std::uint8_t(write?2:0);});
    ++fault.generation;const auto ar=system->ddr4().axi().counters().ar;
    loader.submit({fault},system->cycle());
    for(int t=0;system->ddr4().axi().counters().ar==ar;++t){require(t<1000,"AR not accepted");system->tick();}
    auto& sibling=system->dma(Hemisphere::West);
    sibling.issue(C2cDmaInstruction::Store(0x50000,1));sibling.send(C2cVector{});
    for(int t=0;!system->data_error();++t){require(t<1000,"no sibling B error");system->tick();}
    system->begin_task_error_recovery();
    for(int t=0;t<1024;++t){
        require(!system->finish_task_error_recovery()&&!system->can_load_program()&&!system->kernel_done()
            &&system->ddr4().axi().counters().rx_reserved,"blocked accepted R was forgotten");system->tick();
    }
    r_service=true;
    for(int t=0;!system->finish_task_error_recovery();++t){require(t<1000,"released R did not drain");system->tick();}
    const auto sibling_error=system->last_data_errors()[hemisphere_index(Hemisphere::West)];
    require(system->can_load_program()&&system->ddr4().drained()&&sibling_error&&sibling_error->write,
        "recovery did not retire delayed R and preserve sibling B error");
    system->ddr4().set_channel_service({});system->ddr4().set_response_fault({});
    std::cout<<"DDR -> shared AXI -> bounded Loader -> Relay -> private iMEM -> MEM FU PASS\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
