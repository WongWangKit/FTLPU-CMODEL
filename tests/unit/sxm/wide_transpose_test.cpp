#include "../../support/sxm_macro_fixture.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include <iostream>
#include <stdexcept>
using namespace ftlpu;
void check(bool v){if(!v)throw std::runtime_error("wide transpose regression failed");}
void run(bool reverse,std::size_t gap){
    constexpr auto N=SxmWideTranspose::N;
    auto slice=SxmSlice(SxmStreamPortMap::BetweenColumns(0,1,1,0),SxmTransposeArchitecture::WideStreaming);
    StreamRegisterFabric fabric(2);
    const auto direction=reverse?StreamDirection::West:StreamDirection::East;
    const std::size_t offset=reverse?hw::kStreamsPerDirection:0,in=reverse?1:0,out=reverse?0:1;
    const auto original=SxmInstruction::Transpose({{offset},{offset+1}},{{offset+2},{offset+3}});
    const auto instruction=IcuSxmActiveMacroContextCodec<true>::unpack(
        IcuSxmActiveMacroContextCodec<true>::pack_initial(
            {0,1,1,0,1,1,0,IcuInductionTarget::None},original)).instruction;
    check(SxmWideTranspose::accepts(instruction));
    const std::size_t starts[]={0,N+gap,2*(N+gap)};
    test::SxmMacroFixture program;
    program.add(0,3,N+gap,instruction);
    InstructionControlUnit::SxmTransposeIcu queue;
    program.load(queue,true);
    while(!queue.raw_macro_ready()) queue.prefetch_raw_macro();
    auto value=[](std::size_t tile,std::size_t r,std::size_t c){return static_cast<unsigned>((tile*1723+r*43+c*7)&65535);};
    auto inject=[&](std::size_t t){for(std::size_t k=0;k<3;k++)for(std::size_t sl=0;sl<hw::kTileRows;sl++){
        if(t<starts[k]+sl||t>=starts[k]+sl+N)continue;
        auto row=t-starts[k]-sl;
        for(std::size_t p=0;p<2;p++){
            StreamTileSegment seg{};for(std::size_t l=0;l<hw::kLanesPerTile;l++){
                auto col=sl*hw::kLanesPerTile+l;
                seg[l]=StreamCell::Valid(static_cast<std::uint8_t>(value(k,row,col)>>(8*p)),col==N-1,1000*k+row);
            }
            StreamOutputPort(fabric,in,direction,"test source").write_segment(sl,p,seg);
        }
    }};
    fabric.begin_cycle();inject(0);fabric.commit_cycle();
    std::size_t checked=0;
    for(std::size_t t=0;t<=starts[2]+2*N+2*SxmWideTranspose::skew;t++){
        fabric.begin_cycle();
        if(auto issued=queue.tick()){check(slice.can_issue(*issued));slice.issue(*issued);check(!slice.can_issue(*issued));}
        slice.evaluate(fabric);inject(t+1);fabric.commit_cycle();
        for(std::size_t sl=0;sl<hw::kTileRows;sl++){
            std::optional<std::pair<std::size_t,std::size_t>> expected;
            for(std::size_t k=0;k<3;k++){
                auto first=starts[k]+N+SxmWideTranspose::skew+sl;
                if(t>=first&&t<first+N)expected={{k,t-first}};
            }
            for(std::size_t p=0;p<2;p++)for(std::size_t l=0;l<hw::kLanesPerTile;l++){
                auto cell=fabric.cell(out,sl,l,StreamId::from_packed(offset+2+p));
                check(cell.valid==expected.has_value());
                if(expected){auto [k,col]=*expected;auto row=sl*hw::kLanesPerTile+l;
                    check(cell.data==static_cast<std::uint8_t>(value(k,row,col)>>(8*p)));
                    check(cell.last==(col==N-1));check(cell.vector_tag==1000*k+row);checked++;
                }
            }
        }
    }
    check(checked==3*N*N*2);check(slice.idle());slice.reset();check(slice.cycle()==0&&slice.idle());
}
int main(){
    auto system=std::make_unique<TspSliceSystem>(SystemHardwareConfiguration{},SxmTransposeArchitecture::WideStreaming);

    for(bool reverse:{false,true})for(std::size_t gap:{0u,7u})run(reverse,gap);
    SxmSlice legacy(SxmStreamPortMap::SameDirection(0,1));
    auto command=SxmInstruction::Transpose({{0},{1}},{{2},{3}});
    check(!legacy.can_issue(command));
    SxmSlice wide(SxmStreamPortMap::SameDirection(0,1),SxmTransposeArchitecture::WideStreaming);
    check(wide.can_issue(command));wide.issue(command);
    StreamRegisterFabric missing(2);missing.begin_cycle();bool threw=false;
    try{wide.evaluate(missing);}catch(const std::logic_error&){threw=true;}check(threw);
    wide.reset();check(wide.idle());
    std::cout<<"wide transpose: 3 tiles, both directions, gap/reset/missing-input checks passed; payload bytes="<<SxmWideTranspose::payload_capacity_bytes<<'\n';
}
