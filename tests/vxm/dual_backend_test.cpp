#include "ftlpu/system/icu.hpp"
#include "ftlpu/vxm/backend.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

void put_fp16(
    ftlpu::distributed_vxm::VxmLane::StreamBytes& streams,
    std::size_t base,
    float value)
{
    const auto bytes =
        ftlpu::distributed_vxm::VxmLane::pack_float16(value);
    streams[base] = bytes[0];
    streams[base + 1] = bytes[1];
}

void tick_with_inputs(
    ftlpu::DistributedVxmSlice& slice,
    const ftlpu::DistributedVxmSlice::StreamMatrix& inputs)
{
    slice.prepare_cycle();
    for (std::size_t tile = 0;
         tile < ftlpu::DistributedVxmSlice::kTileCount;
         ++tile) {
        if (slice.required_streams_at(tile)
            && slice.input_buffer(tile).empty()) {
            slice.set_stream_inputs(tile, inputs);
        }
    }
    slice.tick();
}

} // namespace

int main()
{
    static_assert(ftlpu::EstablishedVxmSlice::kAluQueues == 16);
    static_assert(ftlpu::DistributedVxmSlice::kAluQueues == 8);

    using namespace ftlpu::distributed_vxm;
    auto slice = ftlpu::DistributedVxmSlice{};
    slice.set_chain_depth(VxmChainDepth::Two);

    const auto head = VxmLaneAluInstruction {
        VxmAluOpcode::Add,
        VxmLaneOperand::StreamFloat16(),
        VxmLaneOperand::StreamFloat16()};
    slice.issue_south(0, VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Two, head));

    auto tail = VxmLaneAluInstruction {
        VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    tail.output_type = VxmCastTarget::Int8;
    tail.output_scale = 1.0f;
    tail.output_stream = 0;
    slice.issue_south(1, VxmCompactInstructionCodec::encode(
        1, VxmChainDepth::Two, tail));

    auto inputs = ftlpu::DistributedVxmSlice::StreamMatrix{};
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        put_fp16(inputs[lane], 0, static_cast<float>(lane));
        put_fp16(inputs[lane], 2, 10.0f);
    }

    slice.tick();
    tick_with_inputs(slice, inputs);
    tick_with_inputs(slice, inputs);
    tick_with_inputs(slice, inputs);

    assert(slice.output_at(0));
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        assert(slice.output_at(0)->values[lane]
               == static_cast<std::int8_t>(lane + 10));
    }

    auto icu = ftlpu::InstructionControlUnit{};
    auto dispatched = ftlpu::DistributedVxmSlice{};
    icu.enqueue_control(
        ftlpu::IcuLocation::DistributedVxm(0),
        ftlpu::IcuControlInstruction::Nop(1));
    icu.enqueue_distributed_vxm(
        0,
        VxmCompactInstructionCodec::encode(
            0, VxmChainDepth::Two, head));
    assert(icu.distributed_vxm_iq(0).imem_occupancy() == 2);
    icu.dispatch_vxm(dispatched);
    dispatched.tick();
    assert(!dispatched.instruction_at(0, 0));
    icu.dispatch_vxm(dispatched);
    dispatched.tick();
    assert(dispatched.instruction_at(0, 0));
}
