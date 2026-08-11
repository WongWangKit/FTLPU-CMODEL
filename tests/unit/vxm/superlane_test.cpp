#include "ftlpu/vxm/superlane.hpp"

#include <cassert>

int main()
{
    auto superlane = ftlpu::VxmSuperlane {};
    superlane.enqueue_instruction(0, {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::StreamInt8(3),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f, 0, ftlpu::VxmCastTarget::Int8, 9,
        ftlpu::Hemisphere::East, ftlpu::Hemisphere::West});
    auto streams = ftlpu::VxmSuperlane::StreamMatrix {};
    for (std::size_t lane = 0; lane < streams.size(); ++lane) streams[lane][3] = static_cast<std::uint8_t>(lane + 1);
    superlane.set_stream_inputs(ftlpu::Hemisphere::East, streams);
    superlane.tick();
    assert(superlane.output().has_value());
    assert(superlane.output()->stream == 9);
    assert(superlane.output()->hemisphere == ftlpu::Hemisphere::West);
    for (std::size_t lane = 0; lane < streams.size(); ++lane)
        assert(superlane.output()->values[lane] == static_cast<std::int8_t>(lane + 1));
}
