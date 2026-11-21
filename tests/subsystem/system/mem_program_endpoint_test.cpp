#include "ftlpu/system/c2c_dma_system.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace ftlpu;
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main() try {
    using E=MemProgramEndpoint;
    E ep;
    check(!ep.write(0,15,true,1)&&ep.error==E::Error::Write,"bad strb");
    check(!ep.write(0,7,true,1),"sticky fault");ep={};
    check(!ep.write(1,7,true,2),"nonzero base");ep={};
    check(!ep.write(0,7,true,2),"early LAST");ep={};
    check(!ep.write(0,7,false,1),"missing LAST");ep={};
    check(!ep.commit(0),"unowned commit");ep={};
    check(ep.write(0,7,true,1)&&!ep.prime(1),"PRIME before commit");ep={};
    check(ep.write(0,7,true,1)&&!ep.write(1,7,true,2),"write after LAST");ep={};
    check(ep.write(0,7,true,1)&&ep.commit(0)&&ep.prime(1)&&!ep.prime(1),"duplicate PRIME");ep={};
    check(ep.write(0,7,false,2)&&ep.write(1,7,true,2)&&!ep.commit(1),"out of order ACK");
    ProgramLoadConfig config;config.imem_write_commit_latency_cycles=5;
    Ddr4Config ddr;ddr.beat_bytes=std::min<std::size_t>(32,hw::kPhysicalVectorBytes);
    auto sys=std::make_unique<C2cDmaSystem>(ddr,8,SystemHardwareConfiguration{},config);
    // Same DEV queue used on a real ICU commit, independent of the loader's
    // zero-based descriptor policy. A sticky endpoint fault must survive until reset.
    auto& q=sys->chip().icu().mem_iq(1);
    auto rejects=[&](auto op){try{op();return false;}catch(const StaticScheduleError&){return true;}};
    std::array<std::uint32_t,4> full{39,0,0,0};
    q.load_mem_program_word(0,full,7,false);
    check(rejects([&]{q.load_mem_program_word(2,full,7,true);})
        &&q.mem_local_imem_load_port_signals().fault_code_o==1,"queue accepted a hole");
    check(rejects([&]{q.load_mem_program_word(1,full,7,true);}),"queue lost sticky fault");
    q.reset();
    check(rejects([&]{q.prime_mem_program(IcuMacroQueueKind::Mem,1);})
        &&q.mem_local_imem_load_port_signals().fault_code_o==2,"empty queue PRIME");
    q.reset();
    auto& loader=sys->program_loader();
    std::array<std::uint32_t,9> lanes{0x27,0,0,0x02000021,0x180,0x00303200,0x400,0,0};
    auto byte=[&](std::size_t b){return std::uint8_t(lanes[b/4]>>(8*(b%4)));};
    ProgramImage im{0x1000,IcuMacroQueueKind::Mem,0,36};im.generation=42;
    for(std::size_t i=0;i<36;i+=hw::kPhysicalVectorBytes){C2cVector v{};for(std::size_t j=0;j<hw::kPhysicalVectorBytes&&i+j<36;++j)v.payload[j/8][j%8]=byte(i+j);sys->ddr4().initialize_vector(0x1000+i,v);}
    loader.submit({im},0);unsigned w=0,c=0,p=0;bool held=false;std::size_t issued[3]{};
    for(unsigned t=0;!loader.ready_to_start();++t){
        check(t<2048&&!loader.failed(),"ready timeout");sys->tick();
        const auto& x=loader.mem_program_edge();
        if(x.write_fire){check(w<3&&x.queue==0&&x.generation==42&&x.address==w&&x.strb==7&&x.last==(w==2),"WRITE pins");
            for(int k=0;k<12;k++)check(x.data[k]==byte(w*12+k),"accepted bytes");issued[w++]=t;}
        if(x.commit){check(c<w&&x.commit_address==c&&t-issued[c]==5,"commit latency/order");
            const auto z=sys->chip().icu().mem_iq(0).read_raw_imem(c);
            for(int k=0;k<3;k++)check(z.lanes[k]==lanes[c*3+k],"actual SRAM at commit");++c;}
        held|=x.prime_valid&&!x.ready;
        if(x.prime_fire){
            check(c==3&&!p,"premature PRIME");
            check(sys->chip().icu().mem_iq(0).mem_local_imem_load_port_signals().prime_fire
                && loader.statistics().images[0].image_committed==loader.statistics().images[0].prime_accepted,
                "queue PRIME did not act on the accepted edge");++p;
        }
    }
    check(w==3&&c==3&&p==1&&held,"edge coverage");
    std::cout<<"PASS 3 writes/commits, stalled PRIME once, illegal records rejected\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
