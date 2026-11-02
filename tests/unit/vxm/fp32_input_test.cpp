#include "ftlpu/vxm/slice.hpp"
#include "lane_test_driver.hpp"
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <iostream>

using namespace ftlpu;

static void put(VxmLane::StreamBytes& streams, std::size_t base, float value)
{ std::memcpy(streams.data()+base,&value,4); }

int main() try
{
#ifdef _MSC_VER
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    // Check every physical head, both operands, real lower mantissa bits,
    // and mirrored-side fixed bindings through compact decode and execution.
    for (std::size_t head=0; head<8; head+=2) {
        VxmLaneTestDriver lane;
        lane.set_chain_depth(VxmChainDepth::Two);
        VxmLaneAluInstruction sub{VxmAluOpcode::Subtract,
            VxmLaneOperand::StreamFloat32(),VxmLaneOperand::StreamFloat32()};
        sub.precision=VxmAluPrecision::Float32;
        const auto packet=VxmCompactInstructionCodec::encode(head,VxmChainDepth::Two,sub);
        assert(packet.control & (1ull<<52));
        assert(packet.control & (1ull<<53));
        auto decoded=VxmCompactInstructionCodec::decode(head,packet);
        assert(decoded.instruction.lhs.kind==VxmLaneOperandKind::StreamFloat32);
        assert(decoded.instruction.rhs.kind==VxmLaneOperandKind::StreamFloat32);
        lane.enqueue_instruction(head,decoded.instruction);
        VxmLaneAluInstruction tail{VxmAluOpcode::Bypass,VxmLaneOperand::Previous()};
        tail.precision=VxmAluPrecision::Float32;
        tail.output_type=VxmCastTarget::Float32;
        tail.output_stream=VxmLane::fixed_output_stream_for_block(head/2);
        lane.enqueue_instruction(head+1,tail);
        VxmLane::StreamBytes bytes{};
        const float lhs=-128.001953125f, rhs=-128.0f;
        const auto lb=VxmLane::fixed_input_base_for_stage(head,false,sub.lhs.kind);
        const auto rb=VxmLane::fixed_input_base_for_stage(head,true,sub.rhs.kind);
        assert(lb+4<=bytes.size() && rb+4<=bytes.size() && rb==lb+4);
        put(bytes,lb,lhs); put(bytes,rb,rhs);
        put(bytes,lb+16,lhs); put(bytes,rb+16,rhs);
        lane.tick(); // Decode configuration before the first data beat.
        lane.set_stream_inputs(bytes);
        bool seen=false;
        for(int t=0;t<12;++t) {
            lane.tick();
            for(const auto& output:lane.outputs()) {
                assert(VxmLane::unpack_float32(output.bytes)==lhs-rhs);
                seen=true;
            }
        }
        assert(seen && lane.idle());
        auto malformed=packet; malformed.control|=1ull<<4;
        bool rejected=false;
        try { (void)VxmCompactInstructionCodec::decode(head,malformed); }
        catch(const std::invalid_argument&) { rejected=true; }
        assert(rejected);
    }
    // Legacy encoding remains bit-for-bit unchanged.
    VxmLaneAluInstruction old{VxmAluOpcode::Add,
        VxmLaneOperand::StreamBFloat16(),VxmLaneOperand::StreamFloat16()};
    auto legacy=VxmCompactInstructionCodec::encode(0,VxmChainDepth::Two,old);
    assert(legacy.control==(1ull|(5ull<<4)|(4ull<<7)|(1ull<<11)|(1ull<<16)|(1ull<<19)));
    // Slice readiness must wait for all eight bytes, not just high planes.
    VxmSlice slice;
    VxmLaneAluInstruction op{VxmAluOpcode::Add,
        VxmLaneOperand::StreamFloat32(),VxmLaneOperand::StreamFloat32()};
    op.precision=VxmAluPrecision::Float32;
    slice.issue_south(0,VxmCompactInstructionCodec::encode(0,VxmChainDepth::Eight,op));
    slice.tick(); slice.prepare_cycle();
    auto required=slice.required_streams_at(0);
    assert(required && std::count(required->begin(),required->end(),true)==16); // 8 bytes on each mirrored side.
    for (std::size_t s=0;s<required->size();++s)
        assert((*required)[s]==(s<8 || (s>=16 && s<24)));
}
catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
