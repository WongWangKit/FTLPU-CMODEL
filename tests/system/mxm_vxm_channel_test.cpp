#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <cstdint>
#include <memory>

namespace {

ftlpu::Mxm::ColumnOutput raw_result(
    std::int32_t value, std::size_t stream_base)
{
    auto output = ftlpu::Mxm::ColumnOutput {
        0,
        0,
        {},
        stream_base,
        0,
        ftlpu::MxmAccumulatorMode::DirectFinal,
        ftlpu::MxmPairMode::Independent};
    output.values.fill(value);
    return output;
}

} // namespace

int main()
{
    using namespace ftlpu;

    // A logical VXM stream-group position is fixed, while the compiler may
    // select the east or west copy of that group.  The mirrored depth-two
    // chain therefore consumes the west groups without changing its local
    // ALU wiring.
    {
        auto routing = VxmSlice{};
        routing.set_chain_depth(VxmChainDepth::Two);
        routing.configure_input_group_source(8, Hemisphere::West);
        routing.configure_input_group_source(9, Hemisphere::West);
        routing.configure_output_block_destination(0, Hemisphere::East);
        routing.configure_output_block_destination(4, Hemisphere::West);

        auto add = VxmLaneAluInstruction {
            VxmAluOpcode::Add,
            VxmLaneOperand::StreamFloat16(),
            VxmLaneOperand::StreamFloat16()};
        routing.issue_south(
            0, VxmCompactInstructionCodec::encode(
                   0, VxmChainDepth::Two, add));
        routing.prepare_cycle();
        const auto& required = routing.required_streams_at(0);
        assert(required.has_value());
        assert((*required)[0] && (*required)[1]);
        assert((*required)[2] && (*required)[3]);
        assert((*required)[hw::kStreamsPerDirection + 16]);
        assert((*required)[hw::kStreamsPerDirection + 17]);
        assert((*required)[hw::kStreamsPerDirection + 18]);
        assert((*required)[hw::kStreamsPerDirection + 19]);
        assert(routing.output_stream_destination(0) == Hemisphere::East);
        assert(routing.output_stream_destination(8) == Hemisphere::West);
    }

    auto system = std::make_unique<TspSliceSystem>();
    auto& fabric =
        system->mem(Hemisphere::East).stream_fabric();
    system->mxm_accumulator(Hemisphere::East)
        .configure_output_dequant_scale(0, 0.5f, 1.0f);
    system->mxm_accumulator(Hemisphere::East)
        .configure_output_dequant_scale(1, 0.25f, 1.0f);

    // The two fixed MXM paths independently feed the two mirrored VXM
    // chains. Results enter the shared ACC as raw int32 values.
    fabric.begin_cycle();
    system->mxm_accumulator(Hemisphere::East).evaluate(
        fabric,
        {raw_result(10, 0)},
        {raw_result(20, 16)});
    fabric.commit_cycle();

    // Two ACC/Cast cycles plus the fixed westbound SR distance from the
    // MXM/MEM boundary to VXM column zero.
    for (std::size_t cycle = 0;
         cycle < hw::kMemBoundaryStreamRegisterColumns + 1;
         ++cycle) {
        system->tick({});
    }

    system->vxm().set_chain_depth(VxmChainDepth::Two);
    auto head = VxmLaneAluInstruction {
        VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamFloat16()};
    auto tail = VxmLaneAluInstruction {
        VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    tail.output_stream = 0;
    tail.output_type = VxmCastTarget::Int8;
    tail.output_scale = 1.0f;
    system->vxm().issue_south(
        0, VxmCompactInstructionCodec::encode(
               0, VxmChainDepth::Two, head));
    system->vxm().issue_south(
        1, VxmCompactInstructionCodec::encode(
               1, VxmChainDepth::Two, tail));

    // Tile zero consumes the two fixed FP16 stream groups. Other tiles are
    // prefilled only to keep the broadcast instruction schedule legal.
    for (std::size_t tile = 1; tile < VxmSlice::kTileCount; ++tile) {
        system->vxm().set_stream_inputs(
            tile, VxmSlice::StreamMatrix{});
    }

    system->tick({});
    assert(system->vxm().input_buffer(0).ready());
    assert(system->vxm().input_buffer(0).fill_count() == 2);
    system->tick({});
    system->tick({});
    assert(system->vxm().outputs_at(0).empty());
    system->tick({});

    assert(system->vxm().outputs_at(0).size() == 2);
    assert(system->vxm().outputs_at(0)[0].values[0] == 5);
    assert(system->vxm().outputs_at(0)[1].values[0] == 5);
    return 0;
}
