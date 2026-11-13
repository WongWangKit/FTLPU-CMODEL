#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/vxm/superlane.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() try {
    using namespace ftlpu;
    using Queue = DistributedIcuQueue<VxmCompactInstruction, 96, 256, 8, 2, 1, IcuQueueRole::Vxm>;
    auto alu = VxmLaneAluInstruction{};
    alu.operation = VxmAluOpcode::Add;
    alu.lhs = VxmLaneOperand::Previous();
    alu.rhs = VxmLaneOperand::Acc();
    alu.accumulator_write = true;
    alu.output_type = VxmCastTarget::BFloat16;
    std::vector<VxmCompactInstruction> configs;
    for (std::size_t i=0; i<128; ++i) {
        alu.accumulator_reset = i%2==0;
        alu.accumulator_emit = i%2!=0;
        configs.push_back(isa::encode_vxm_instruction(1,VxmChainDepth::Two,alu));
        const auto word=isa::encode_vxm_icu_instruction(configs.back());
        require(isa::decode_vxm_icu_instruction(word)==configs.back(),"96-bit codec mismatch");
    }
    // A full, sustained alternating stream must fetch/decode one config per
    // cycle, beyond the initial IQ contents and across physical fetch latency.
    Queue queue;
    queue.push_nop(3);
    for(auto config:configs)queue.push_instruction(config);
    require(queue.imem_occupancy()==129,"each VXM config must occupy exactly one word");
    VxmSuperlaneInstructionControl control;
    std::size_t issued=0,executed=0;
    for(std::size_t cycle=0;cycle<135;++cycle) {
        if(auto config=queue.tick()) {
            require(cycle==3+issued,"sequential ICU stream has a fetch bubble");
            require(*config==configs[issued++],"ICU reordered configs");
            control.enqueue_compact(1,*config);
        }
        auto active=control.issue(VxmChainDepth::Two);
        if(active[1]) {
            require(cycle==4+executed,"Superlane decoder has a bubble");
            require(active[1]->accumulator_reset==(executed%2==0),"reset changed");
            require(active[1]->accumulator_emit==(executed%2!=0),"emit changed");
            VxmLaneExecutionMask mask{};mask[1]=mask[9]=true;control.consume(mask);++executed;
        }
    }
    require(issued==128 && executed==128 && queue.done(),"stream did not finish");
    // Issuing the next configuration early is legal prefetch, not execution
    // interleaving: Current must consume all three tokens before Next runs.
    VxmSuperlaneInstructionControl prefetched;
    auto repeated=isa::decode_vxm_instruction(1,configs[0]).instruction;
    repeated.repeat_count=3;
    std::size_t consumed=0;
    for(std::size_t cycle=0;cycle<6;++cycle) {
        if(cycle==0)prefetched.enqueue_compact(1,isa::encode_vxm_instruction(1,VxmChainDepth::Two,repeated));
        if(cycle==1)prefetched.enqueue_compact(1,configs[1]);
        auto active=prefetched.issue(VxmChainDepth::Two);
        if(active[1]) {
            require(cycle==consumed+1,"prefetched configuration introduced a bubble");
            require(active[1]->accumulator_reset==(consumed<3),"prefetched configuration interleaved Current");
            VxmLaneExecutionMask mask{};mask[1]=mask[9]=true;prefetched.consume(mask);++consumed;
        }
    }
    require(consumed==4 && prefetched.idle(),"prefetched FU repeat did not retire");
    // Cover all immediate bits and the FP32 operand-extension bits.
    alu={};alu.operation=VxmAluOpcode::Multiply;
    alu.lhs=VxmLaneOperand::StreamFloat32();alu.rhs=VxmLaneOperand::Imm(-123.25f);
    alu.repeat_count=0xffffffffU;
    const auto config=isa::encode_vxm_instruction(0,VxmChainDepth::Eight,alu);
    require(isa::decode_vxm_icu_instruction(isa::encode_vxm_icu_instruction(config))==config,
        "immediate/FP32/repeat bits lost");
    bool rejected=false;
    try {
        Queue bad;
        // Old local RUN packet discriminator: Extended + FU-loop subtype 2.
        bad.write_imem(0,isa::EncodedVxmIcuWord{{0x43U,0,0x02000000U}});
        bad.dispatch_next();
    } catch(const std::exception&) {rejected=true;}
    require(rejected,"VXM raw decoder accepted an obsolete RUN packet");
    std::cout<<"vxm_sequential_instruction_test passed: 128 consecutive alternating configs, no loop context\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
