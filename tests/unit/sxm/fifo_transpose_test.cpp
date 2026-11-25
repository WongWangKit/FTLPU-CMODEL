#include "../../support/sxm_macro_fixture.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include <iostream>
#include <stdexcept>
using namespace ftlpu;
void check(bool v){if(!v)throw std::runtime_error("FIFO transpose regression failed");}
void run(bool reverse,std::size_t gap){
    constexpr auto N=sxm_fifo::width;
    auto slice=SxmSlice(SxmStreamPortMap::BetweenColumns(0,1,1,0),SxmTransposeArchitecture::Fifo);
    StreamRegisterFabric fabric(2);
    const auto direction=reverse?StreamDirection::West:StreamDirection::East;
    const std::size_t offset=reverse?hw::kStreamsPerDirection:0,in=reverse?1:0,out=reverse?0:1;
    const auto original=SxmInstruction::Transpose({{offset},{offset+1}},{{offset+2},{offset+3}});
    const auto instruction=IcuSxmActiveMacroContextCodec<true>::unpack(
        IcuSxmActiveMacroContextCodec<true>::pack_initial(
            {0,1,1,0,1,1,0,IcuInductionTarget::None},original)).instruction;
    check(SxmFifoTranspose::accepts(instruction));
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
    for(std::size_t t=0;t<=starts[2]+sxm_fifo::execution_cycles;t++){
        fabric.begin_cycle();
        if(auto issued=queue.tick()){check(slice.can_issue(*issued));slice.issue(*issued);check(!slice.can_issue(*issued));}
        slice.evaluate(fabric);inject(t+1);fabric.commit_cycle();
        for(std::size_t sl=0;sl<hw::kTileRows;sl++){
            std::optional<std::pair<std::size_t,std::size_t>> expected;
            for(std::size_t k=0;k<3;k++){
                auto first=starts[k]+sxm_fifo::output_offset+sl;
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
// Verify the routing rule independently of the target selected by CMake.
// Each occupied wire must write a different register at most one SL away.
void check_interval_routes() {
    for (int count = 1; count <= 32; ++count) {
        for (int sum = 0; sum <= 2 * (count - 1); ++sum) {
            std::vector<int> wires(count, -1);
            for (int sl = 0; sl < count; ++sl)
                if (sum - sl >= 0 && sum - sl < count) wires[sl] = sl;
            for (int stage = 0; stage < count; ++stage) {
                std::vector<int> next(count, -1);
                for (int sl = 0; sl < count; ++sl) {
                    if (wires[sl] < 0) continue;
                    const int dst = sxm_fifo::interval_exchange_port(stage, sl, sum, count);
                    check(dst >= 0 && dst < count && dst >= sl - 1 && dst <= sl + 1);
                    check(next[dst] == -1);
                    next[dst] = wires[sl];
                }
                wires.swap(next);
            }
            for (int sl = 0; sl < count; ++sl) {
                const int source = sum - sl;
                check(wires[sl] == (source >= 0 && source < count ? source : -1));
            }
        }
    }
}
int main(){
    check_interval_routes();
    auto system=std::make_unique<TspSliceSystem>(SystemHardwareConfiguration{},SxmTransposeArchitecture::Fifo);

    for(bool reverse:{false,true})for(std::size_t gap:{0u,7u})run(reverse,gap);
    SxmSlice slice(SxmStreamPortMap::SameDirection(0,1));
    auto command=SxmInstruction::Transpose({{0},{1}},{{2},{3}});
    auto local = command;
    local.input_row = 0;
    const auto last_stream = hw::kStreamsPerDirection - 1;
    for (const auto& retired : {
            local,
            SxmInstruction::Distribute({0}, {2}, {}),
            SxmInstruction::ShiftSelect({{0}}, {{2}}, SxmShiftSource::NorthShifted),
            SxmInstruction::Permute({{0}}, {{2}}, {}),
            SxmInstruction::PermuteFromBank(0, 0, {2}, {}),
            SxmInstruction::Transpose({{last_stream}, {last_stream+1}}, {{2}, {3}})}) {
        check(!slice.can_issue(retired));
        bool refused = false;
        try { slice.issue(retired); } catch (const std::invalid_argument&) { refused = true; }
        check(refused && slice.idle());
    }
    for (auto row : {0u, 15u}) {
        const unsigned fields[]{0u, 2u, row};
        unsigned index = 0;
        bool rejected = false;
        try { isa::read_sxm_macro_native(true, [&](unsigned) { return fields[index++]; }); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected);
    }
    check(slice.can_issue(command)); slice.issue(command);
    StreamRegisterFabric missing(2); missing.begin_cycle(); bool threw=false;
    try { slice.evaluate(missing); } catch (const std::logic_error&) { threw=true; }
    check(threw); check(slice.cycle()==0);
    slice.reset(); check(slice.idle());
    std::cout<<"FIFO transpose checked data, tags, SL skew, consecutive tiles, gaps, both directions, reset and missing input\n";
}
