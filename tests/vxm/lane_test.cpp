#include "ftlpu/vxm/lane.hpp"

#include <cassert>
#include <cmath>

int main()
{
    auto lane = ftlpu::VxmLane {};
    lane.enqueue_instruction(0, {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::StreamInt8(32),
        ftlpu::VxmLaneOperand::Imm(0.25f),
        1.0f, 0, ftlpu::VxmCastTarget::Float32, std::nullopt,
        ftlpu::Hemisphere::West, ftlpu::Hemisphere::East});
    lane.enqueue_instruction(1, {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::Alu(0),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f, 0, ftlpu::VxmCastTarget::Float16, 4,
        ftlpu::Hemisphere::West, ftlpu::Hemisphere::East});

    auto streams = ftlpu::VxmLane::StreamBytes {};
    streams[32] = static_cast<std::uint8_t>(static_cast<std::int8_t>(-8));
    lane.set_stream_inputs(ftlpu::Hemisphere::West, streams);
    lane.tick();
    assert(!lane.output().has_value());
    assert(lane.alu_output(0).has_value() && std::fabs(*lane.alu_output(0) + 2.0f) < 1.0e-6f);

    lane.tick();
    assert(lane.output().has_value());
    assert(lane.output()->stream == 4);
    assert(lane.output()->byte_count == 2);
    assert(lane.output()->hemisphere == ftlpu::Hemisphere::East);
    const auto bits = static_cast<std::uint16_t>(lane.output()->bytes[0])
        | (static_cast<std::uint16_t>(lane.output()->bytes[1]) << 8);
    assert(std::fabs(ftlpu::Fp16::from_bits(bits).to_float() + 2.0f) < 1.0e-3f);
}
