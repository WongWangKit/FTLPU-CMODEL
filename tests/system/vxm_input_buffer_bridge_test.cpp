#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace {

void initialize_group(
    ftlpu::StreamRegisterFabric& fabric,
    std::size_t group,
    std::int32_t base)
{
    const auto stream_base =
        group * ftlpu::VxmLane::kStreamGroupBytes;
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        const auto bytes = ftlpu::VxmLane::pack_float16(
            static_cast<float>(base + lane));
        for (std::size_t byte = 0;
             byte < bytes.size();
             ++byte) {
            fabric.initialize_cell(
                0,
                0,
                lane,
                ftlpu::StreamId::West(
                    stream_base + byte),
                ftlpu::StreamCell::Valid(
                    bytes[byte],
                    lane + 1
                        == ftlpu::hw::kLanesPerTile));
        }
    }
}

} // namespace

int main()
{
    using namespace ftlpu;

    auto system = std::make_unique<TspSliceSystem>();
    system->vxm().set_chain_depth(VxmChainDepth::Two);

    auto head = VxmLaneAluInstruction{
        VxmAluOpcode::Add,
        VxmLaneOperand::StreamFloat16(),
        VxmLaneOperand::StreamFloat16()};
    auto tail = VxmLaneAluInstruction{
        VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    tail.output_stream = 0;
    tail.output_type = VxmCastTarget::Int8;
    tail.output_scale = 1.0f;
    system->vxm().issue_south(
        0,
        VxmCompactInstructionCodec::encode(
            0, VxmChainDepth::Two, head));
    system->vxm().issue_south(
        1,
        VxmCompactInstructionCodec::encode(
            1, VxmChainDepth::Two, tail));

    auto& fabric =
        system->mem(Hemisphere::East).stream_fabric();

    // This test has no compiler-side collect-window instruction, so present
    // all four required groups before the statically scheduled VXM issue.
    // Uneven multi-cycle collection is covered by input_buffer_test; a real
    // compiler must open that collection window sufficiently early.
    initialize_group(fabric, 0, 0);
    initialize_group(fabric, 1, 10);
    initialize_group(fabric, 8, 20);
    initialize_group(fabric, 9, 30);

    // Keep later broadcast tiles on a legal static schedule while this test
    // observes tile 0's SR-to-Buffer bridge.
    for (std::size_t tile = 1;
         tile < VxmSlice::kTileCount;
         ++tile) {
        system->vxm().set_stream_inputs(
            tile, VxmSlice::StreamMatrix{});
    }

    fabric.begin_cycle();
    system->tick_vxm_stream_bridge({});
    assert(system->vxm().input_buffer(0).ready());
    assert(system->vxm().input_buffer(0).fill_count() == 4);
    assert(!system->vxm().output_at(0));
    for (const auto group : {0U, 1U, 8U, 9U}) {
        for (std::size_t lane = 0;
             lane < hw::kLanesPerTile;
             ++lane) {
            const auto stream =
                group * VxmLane::kStreamGroupBytes;
            assert(fabric.cell_consumed(
                0, 0, lane, StreamId::West(stream)));
            assert(fabric.cell_consumed(
                0, 0, lane, StreamId::West(stream + 1)));
        }
    }
    fabric.commit_cycle();

    // The compiler-scheduled issue is atomic across both mirrored consumers,
    // and releases the Buffer at the edge without ALU acknowledgement.
    fabric.begin_cycle();
    system->tick_vxm_stream_bridge({});
    // The same-cycle refill phase opens the next compiler collection window.
    // No new SR groups are present in this one-shot test, so it remains an
    // empty Collecting bank rather than reverting to the preconfigured state.
    assert(system->vxm().input_buffer(0).collecting());
    assert(system->vxm().input_buffer(0).fill_count() == 0);
    assert(!system->vxm().output_at(0));
    fabric.commit_cycle();

    fabric.begin_cycle();
    system->tick_vxm_stream_bridge({});
    assert(!system->vxm().output_at(0));
    fabric.commit_cycle();

    // INT8 output leaves the per-lane static quantizer one cycle later.
    fabric.begin_cycle();
    system->tick_vxm_stream_bridge({});
    assert(system->vxm().output_at(0));
    for (std::size_t lane = 0;
         lane < hw::kLanesPerTile;
         ++lane) {
        assert(
            system->vxm().output_at(0)->values[lane]
            == static_cast<std::int8_t>(
                10 + 2 * lane));
    }
    fabric.commit_cycle();
    return 0;
}
