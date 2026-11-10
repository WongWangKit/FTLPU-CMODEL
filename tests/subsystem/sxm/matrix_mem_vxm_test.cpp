#include "ftlpu/sxm/matrix_vxm.hpp"
#include "sxm_matrix_test_support.hpp"
#include <iostream>
#include <memory>
#include <string>

using namespace sxm_test;
using Layout=SxmMatrixMemLayout;
using Transfer=MemArrayModel::MemTransfer;

// Ordinary MEM Read -> actual SR links -> the four-group read boundary. No direct SRAM
// injection into VXM; verify values and every arrival phase before collection.
template<class Consumer>
static std::size_t read_row(MemArrayModel& mem, StreamRegisterFabric& fabric,
    const SxmMatrixStart& j, std::size_t m, std::size_t row, std::size_t stream,
    Consumer consumer)
{
    const auto arrival=mem.cycle()+Layout::read_hops(0)+2;
    const auto commands=SxmMatrixRowRead::plan(j,m,row,stream,arrival);
    const auto phases=j.bytes()/2;
    std::size_t reads=0,segments=0,issued=0;
    for(auto c=mem.cycle();c<=arrival+phases+T;++c) {
        assert(mem.cycle()==c && fabric.cycle()==c);
        fabric.begin_cycle();
        for(const auto& cmd:commands) if(cmd.cycle==c) {
            mem.enqueue_instruction(cmd.slice,cmd.bank,cmd.instruction); ++issued;
        }
        mem.evaluate(fabric);
        for(const auto& x:mem.executed_transfers()) {
            assert(x.kind==Transfer::Kind::LoadSramToStream); ++reads;
            assert(x.address==j.mem_base+m*T+row/8);
        }
        for(std::size_t t=0;t<T;++t) {
            const bool due=c>=arrival+t && c<arrival+t+phases;
            for(std::size_t s=0;s<hw::kStreamsPerDirection;++s) {
                const bool expected_valid=due && s>=stream && s<stream+2;
                assert(fabric.segment_valid(Layout::read_column,t,StreamId::West(s))==expected_valid);
            }
            if(due) {
                const auto phase=c-arrival-t;
                for(std::size_t b=0;b<2;++b) {
                    const auto sr=StreamId::West(stream+b);
                    const auto bytes=fabric.segment(Layout::read_column,t,sr);
                    for(std::size_t l=0;l<8;++l) {
                        const auto want=expected(j,m,row,t,l);
                        assert(bytes[l].data==std::uint8_t(want>>(8*(2*phase+b))));
                    }
                    ++segments;
                }
                consumer(t,phase);
                for(std::size_t b=0;b<2;++b)
                    fabric.consume_segment(Layout::read_column,t,StreamId::West(stream+b),"test row sink");
            }
        }
        fabric.stage_linear_links();fabric.commit_cycle();
    }
    assert(issued==j.bytes() && reads==T*j.bytes() && segments==reads);
    assert(mem.idle() && fabric.empty());
    return reads;
}

static std::size_t verify_vxm(MemArrayModel& mem, StreamRegisterFabric& fabric,
                             const SxmMatrixStart& j)
{
    std::size_t checked=0;
    for(std::size_t depth:{2,4,8}) for(std::size_t head=0;head<8;head+=depth) {
        SxmMatrixVxmPairAdapter adapter(j.dtype,depth,head);
        std::array<std::size_t,4> rows{}, matrices{};
        for(std::size_t op=0;op<4;++op) {
            // Four genuinely different vectors, including different matrices;
            // no LHS/RHS alias and no duplicated mirror payload.
            rows[op]=(head+depth+op*5)%(8*T); matrices[op]=(op/2)%j.matrices;
            const auto stream=adapter.stream_base(op/2,op%2);
            read_row(mem,fabric,j,matrices[op],rows[op],stream,[&](auto t,auto phase) {
                adapter.capture(fabric,t,op/2,op%2,phase);
            });
        }
        for(std::size_t t=0;t<T;++t) for(std::size_t l=0;l<8;++l) {
            assert(adapter.ready(t));
            VxmLane lane; lane.set_chain_depth(static_cast<VxmChainDepth>(depth));
            VxmLane::Configs configs{};
            for(std::size_t mirror=0;mirror<2;++mirror) {
                const auto h=head+mirror*8;
                auto operand=j.dtype==SxmMatrixDtype::Float32
                    ? VxmLaneOperand::StreamFloat32() : VxmLaneOperand::StreamBFloat16();
                VxmLaneAluInstruction add{VxmAluOpcode::Add,operand,operand};
                add.precision=VxmAluPrecision::Float32; configs[h]=add;
                for(std::size_t stage=h+1;stage<h+depth;++stage) {
                    VxmLaneAluInstruction bypass{VxmAluOpcode::Bypass,VxmLaneOperand::Previous()};
                    bypass.precision=VxmAluPrecision::Float32;
                    if(stage==h+depth-1) {
                        bypass.output_type=VxmCastTarget::Float32;
                        bypass.output_stream=VxmLane::fixed_output_stream_for_block(stage/2);
                    }
                    configs[stage]=bypass;
                }
            }
            lane.set_stream_inputs(adapter.streams(t,l));
            std::array<bool,2> seen{};
            for(std::size_t c=0;c<depth+4;++c) {
                lane.tick(configs);
                for(const auto& out:lane.outputs()) {
                    bool recognized=false;
                    for(std::size_t mirror=0;mirror<2;++mirror) {
                        const auto tail=head+mirror*8+depth-1;
                        if(out.stream!=VxmLane::fixed_output_stream_for_block(tail/2))continue;
                        assert(!seen[mirror]); seen[mirror]=true; recognized=true; ++checked;
                        const auto o=mirror*2;
                        const auto lhs=number(j.dtype,expected(j,matrices[o],rows[o],t,l));
                        const auto rhs=number(j.dtype,expected(j,matrices[o+1],rows[o+1],t,l));
                        assert(out.byte_count==4 && VxmLane::unpack_float32(out.bytes)==lhs+rhs);
                    }
                    assert(recognized);
                }
            }
            assert(seen[0] && seen[1] && lane.datapath_idle());
        }
    }
    return checked;
}

static void run(SxmMatrixDtype dtype, bool mapped)
{
    auto mem=std::make_unique<MemArrayModel>();mem->set_sram_depth_rows(128);
    auto fabric=std::make_unique<StreamRegisterFabric>(hw::kSystemStreamRegisterColumns);
    SxmMatrixMemReference ref(*mem,*fabric,128);
    auto j=job(dtype,mapped), original=j;
    // Guard rows/bank outside the reserved payload range.
    mem->set_sram_lane_byte(Layout::slice(0),Layout::bank(0),0,j.mem_base-1,0,0xa5);
    mem->set_sram_lane_byte(0,0,0,j.mem_base,0,0x5a);
    rejects([&]{ref.start(j,false);}); assert(mem->cycle()==0 && !ref.fault());
    ref.start(j,true);
    j.mem_base=100;j.matrices=1;j.tile_map.fill(0); // writer-private config too
    const auto last_p=9+8*(2*T+T-2);
    const auto done=last_p+original.bytes()+1+T+1;
    std::size_t transfer_sum=0, executed_taps=0;
    for(std::size_t c=0;c<=done;++c) {
        assert(ref.busy() && !ref.done());
        ref.tick(inputs(original,c));
        // Validate the real root -> Tile instruction propagation as well as
        // successful payload writes. Harmless empty Tap visits also count here.
        for(const auto& x:mem->executed_instructions()) {
            ++executed_taps;
            const auto& cmd=x.instruction;const auto plane=cmd.stream_id().index();
            assert(cmd.opcode==MemOpcode::Write && cmd.preserve_stream);
            assert(cmd.stream_id().direction()==StreamDirection::West && plane<original.bytes()*8);
            assert(x.mem_slice==Layout::slice(plane) && x.bank==Layout::bank(plane));
            const auto n=cmd.address-original.mem_base;assert(n<2*T);
            const auto m=n/T, s=n%T;
            bool match=false;
            for(std::size_t a=0;a<T;++a) {
                const auto d=original.tile_map[a];
                const auto root=9+8*T*m+8*(a+s)+2+plane/8-d;
                if(c==root+x.tile){assert(!match);match=true;}
            }
            assert(match);
        }
        for(const auto& x:mem->executed_transfers()) {
            ++transfer_sum;
            const auto plane=x.stream.index(); const auto m=(x.address-original.mem_base)/T;
            const auto s=(x.address-original.mem_base)%T;
            const auto a=inverse(original,x.tile);
            assert(c==9+8*T*m+8*(a+s)+2+plane/8);
            for(std::size_t l=0;l<8;++l)
                assert(x.bytes[l]==std::uint8_t(expected(original,m,s*8+plane/original.bytes(),x.tile,l)
                       >> (8*(plane%original.bytes()))));
        }
    }
    const auto stats=ref.statistics();
    assert(ref.done() && !ref.busy() && !ref.fault());
    assert(stats.blocks==2*T*T && stats.first_p==9 && stats.last_p==last_p);
    assert(stats.commits==2*8*T*T*original.bytes() && stats.commits==transfer_sum);
    assert(stats.tap_issues==stats.commits && stats.last_commit==last_p+original.bytes()+1);
    assert(executed_taps==stats.tap_issues*T);
    assert(stats.done_cycle==done && mem->idle() && fabric->empty());
    ref.tick();assert(!ref.done() && !ref.busy()); // done is a pulse, not sticky
    assert(mem->read_timing()==MemArrayModel::ReadTiming::Main);
    assert(mem->sram_lane_byte(Layout::slice(0),Layout::bank(0),0,original.mem_base-1,0)==0xa5);
    assert(mem->sram_lane_byte(0,0,0,original.mem_base,0)==0x5a);
    // Exhaustive ordinary-Read roundtrip of both matrices, all rows and lanes.
    std::size_t reads=0;
    for(std::size_t m=0;m<2;++m) for(std::size_t row=0;row<8*T;++row)
        reads+=read_row(*mem,*fabric,original,m,row,0,[](auto,auto){});
    const auto numeric=verify_vxm(*mem,*fabric,original);
    // Re-entry after a read phase: new configuration/base, no timer/SRAM reset.
    auto next=job(dtype,!mapped);next.matrices=1;next.mem_base=64;
    const auto launch=mem->cycle();ref.start(next,true);
    for(std::size_t age=0;ref.busy();++age) { assert(age<200); ref.tick(inputs(next,age)); }
    assert(ref.done() && ref.statistics().first_p==launch+9);
    read_row(*mem,*fabric,next,0,0,0,[](auto,auto){});
    std::cout<<"CASE T="<<T<<" dtype="<<(dtype==SxmMatrixDtype::Float32?"FP32":"BF16")
        <<" map="<<mapped<<" matrices=2 first_P="<<stats.first_p<<" last_P="<<stats.last_p
        <<" last_commit="<<stats.last_commit<<" done_edge="<<stats.done_cycle
        <<" commits="<<stats.commits<<" tap_visits="<<executed_taps
        <<" roundtrip_reads="<<reads<<" vxm_outputs="<<numeric
        <<" peak_slots=";
    for(std::size_t s=0;s<T;++s)std::cout<<(s?",":"")<<ref.core().peaks()[s];
    std::cout<<" PASS\n";
}

static void negative()
{
    auto mem=std::make_unique<MemArrayModel>();mem->set_sram_depth_rows(128);
    auto f=std::make_unique<StreamRegisterFabric>(hw::kSystemStreamRegisterColumns);
    SxmMatrixMemReference ref(*mem,*f,128);auto j=job(SxmMatrixDtype::Float32);
    auto bad=j;bad.mem_base=127;rejects([&]{ref.start(bad,true);});
    ref.start(j,true);rejects([&]{ref.start(j,true);});
    for(std::size_t c=0;c<9;++c) {
        auto in=inputs(j,c);if(c==4)in[0].reset();ref.tick(in);
    }
    rejects([&]{ref.tick(inputs(j,9));});assert(ref.fault());
    ref.reset_execution_state();assert(!ref.busy() && !ref.fault() && mem->idle() && f->empty());
    // Abort with actual SR/write activity, then cleanly start from a new context.
    ref.start(j,true);for(int c=0;c<14;++c)ref.tick(inputs(j,c));ref.reset_execution_state();
    ref.start(j,true);for(std::size_t c=0;ref.busy();++c){assert(c<300);ref.tick(inputs(j,c));}
    assert(ref.done());
    // The reserved write batch cannot admit an interleaved ordinary Read.
    ref.reset_execution_state();ref.start(j,true);
    mem->enqueue_instruction(Layout::slice(0),Layout::bank(0),MemInstruction::Read(j.mem_base,StreamId::West(0)));
    rejects([&]{ref.tick();});assert(ref.fault());ref.reset_execution_state();
    auto r1=std::make_unique<MemArrayModel>(MemStreamPortMap::BetweenBoundaries(),
        MemArrayModel::MissingStreamPolicy::Error,MemArrayModel::ReadTiming::RegisteredR1);
    rejects([&]{SxmMatrixMemReference wrong(*r1,*f,128);});
    rejects([&]{SxmMatrixRowRead::plan(j,2,0,0,20);});
    rejects([&]{SxmMatrixRowRead::plan(j,0,8*T,0,20);});
    rejects([&]{SxmMatrixRowRead::plan(j,0,0,31,20);});
    rejects([&]{SxmMatrixRowRead::plan(j,0,0,0,20,Layout::read_column+1);});
    rejects([&]{SxmMatrixVxmPairAdapter wrong(j.dtype,4,2);});
    SxmMatrixVxmPairAdapter adapter(j.dtype,2,0);
    rejects([&]{adapter.streams(0,0);});
    f->begin_cycle();rejects([&]{adapter.capture(*f,0,false,false,1);});f->commit_cycle();
    // Low half accepted, but a high half two clocks later must be rejected.
    f->begin_cycle();
    StreamPayloadTileSegment byte{};byte.fill(0x5a);
    f->stage_payload_segment(Layout::read_column,0,StreamId::West(0),byte);
    f->stage_payload_segment(Layout::read_column,0,StreamId::West(1),byte);f->commit_cycle();
    f->begin_cycle();adapter.capture(*f,0,false,false,0);
    rejects([&]{adapter.capture(*f,0,false,false,0);});f->commit_cycle();
    f->begin_cycle();f->commit_cycle();
    f->begin_cycle();rejects([&]{adapter.capture(*f,0,false,false,1);});f->commit_cycle();
    std::cout<<"PASS admission/missing_P/reset/Main_only/read_conflict/ranges/phase_order\n";
}

int main() try {
    for(auto dt:{SxmMatrixDtype::BFloat16,SxmMatrixDtype::Float32})
        for(bool mapped:{false,true})run(dt,mapped);
    negative();
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
