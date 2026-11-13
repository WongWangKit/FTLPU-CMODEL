#include "ftlpu/system/tsp_slice_system.hpp"
#include "../../support/sxm_macro_fixture.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace ftlpu;
constexpr std::size_t N=hw::kPhysicalVectorBytes, first=64, inputSlice=18, outputSlice=40;
std::uint16_t value(std::size_t matrix,std::size_t row,std::size_t col){
    return static_cast<std::uint16_t>(0x1000+matrix*139+row*43+col*7);
}
void run(Hemisphere side,std::size_t matrices){
    auto system=std::make_unique<TspSliceSystem>(SystemHardwareConfiguration{},SxmTransposeArchitecture::WideStreaming);
    for(std::size_t m=0;m<matrices;++m)for(std::size_t r=0;r<N;++r)for(std::size_t c=0;c<N;++c)
        for(std::size_t b=0;b<2;++b)
            system->initialize_mem_sram_lane_byte(side,inputSlice+b,c/hw::kLanesPerTile,m*N+r,c%hw::kLanesPerTile,
                static_cast<std::uint8_t>(value(m,r,c)>>(8*b)));
    const auto readCycle=first-(hw::kMemGroups+hw::kC2cToSxmStreamRegisterColumns-inputSlice/hw::kMemSlicesPerGroup);
    const auto writeCycle=first+N+SxmWideTranspose::skew
        +hw::kMemGroups+hw::kC2cToSxmStreamRegisterColumns-outputSlice/hw::kMemSlicesPerGroup;
    for(std::size_t b=0;b<2;++b){
        auto reader=InstructionControlUnit::mem_queue(side,inputSlice+b,0);
        auto writer=InstructionControlUnit::mem_queue(side,outputSlice+b,0);
        system->icu().enqueue_mem_nop(reader,readCycle);
        system->icu().enqueue_mem(reader,MemInstruction::Read(0,StreamId::East(16+b)));
        for (std::size_t row=1;row<matrices*N;++row)
            system->icu().enqueue_mem(reader,MemInstruction::Read(row,StreamId::East(16+b)));
        system->icu().enqueue_mem_nop(writer,writeCycle);
        system->icu().enqueue_mem(writer,MemInstruction::Write(32,StreamId::West(b)));
        for (std::size_t row=1;row<matrices*N;++row)
            system->icu().enqueue_mem(writer,MemInstruction::Write(32+row,StreamId::West(b)));
    }
    test::SxmMacroFixture program;
    program.add(first,matrices,N,SxmInstruction::Transpose({{16},{17}},{{32},{33}}));
    program.load(system->icu().sxm_transpose_iq(side),true);
    system->icu().prime_raw_macro_frontends();
    for(std::size_t cycle=0;cycle<writeCycle+matrices*N+32;++cycle)system->tick({});
    for(std::size_t m=0;m<matrices;++m)for(std::size_t r=0;r<N;++r)for(std::size_t c=0;c<N;++c){
        std::uint16_t actual=0;
        for(std::size_t b=0;b<2;++b)actual|=std::uint16_t(system->read_mem_sram_lane_byte(side,outputSlice+b,
            c/hw::kLanesPerTile,32+m*N+r,c%hw::kLanesPerTile))<<(8*b);
        if(actual!=value(m,c,r))throw std::runtime_error("MEM/SXM transpose value mismatch");
    }
    if(!system->icu().program_queues_done() || !system->program_resources_drained())
        throw std::runtime_error("MEM/SXM did not drain");
}
int main(int argc,char** argv)try{
    auto matrices=argc==2?std::stoul(argv[1]):96u;
    if(!matrices || matrices>96)throw std::invalid_argument("matrix-count must be 1..96");
    run(Hemisphere::East,matrices);run(Hemisphere::West,matrices);
    std::cout<<"MEM -> packed Macro wide SXM -> MEM passed: matrices="<<matrices<<" both hemispheres\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
