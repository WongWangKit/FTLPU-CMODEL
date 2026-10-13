#include "ftlpu/vxm/stream_fabric.hpp"

#include <cassert>

namespace {

ftlpu::StreamTileSegment segment(std::uint8_t value)
{
    auto result = ftlpu::StreamTileSegment {};
    for (std::size_t lane = 0; lane < result.size(); ++lane)
        result[lane] = ftlpu::StreamCell::Valid(
            static_cast<std::uint8_t>(value + lane),
            lane + 1 == result.size());
    return result;
}

} // namespace

int main()
{
    using namespace ftlpu;
    VxmStreamFabric vxm {};
    auto& sr = vxm.registers();
    const auto east_data = StreamId::East(2);
    const auto east_bypass = StreamId::East(6);
    const auto east_result = StreamId::East(10);
    const auto west_bypass = StreamId::West(4);

    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        sr.initialize_cell(0, 0, lane, east_data,
            segment(10)[lane]);
        sr.initialize_cell(0, 0, lane, east_bypass,
            segment(30)[lane]);
        sr.initialize_cell(VxmStreamFabric::kAluColumns, 0, lane,
            west_bypass, segment(50)[lane]);
    }

    // Reach the SR immediately before C5 from both directions.
    for (std::size_t cycle = 0; cycle < 5; ++cycle) {
        sr.begin_cycle();
        sr.stage_linear_links();
        sr.commit_cycle();
    }
    assert(sr.segment_valid(5, 0, east_data));
    assert(sr.segment_valid(5, 0, east_bypass));
    assert(sr.segment_valid(11, 0, west_bypass));

    // C5 consumes only E2 and writes E10. E6 and the unrelated westbound SR
    // continue through the same ordinary passive-link mechanism.
    sr.begin_cycle();
    vxm.consume_operand(5, 0, east_data);
    vxm.stage_result(5, 0, east_result, segment(90));
    sr.stage_linear_links();
    sr.commit_cycle();
    assert(!sr.segment_valid(6, 0, east_data));
    assert(sr.segment_valid(6, 0, east_bypass));
    assert(sr.segment_valid(6, 0, east_result));
    assert(sr.segment_valid(10, 0, west_bypass));
    return 0;
}
