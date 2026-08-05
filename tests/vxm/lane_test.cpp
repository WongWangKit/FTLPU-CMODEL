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
    auto streams = ftlpu::VxmLane::StreamBytes {};
    streams[32] = static_cast<std::uint8_t>(static_cast<std::int8_t>(-8));
    lane.set_stream_inputs(ftlpu::Hemisphere::West, streams);
    lane.tick();
    assert(!lane.output().has_value());
    assert(lane.alu_output(0).has_value() && std::fabs(*lane.alu_output(0) + 2.0f) < 1.0e-6f);

    lane.enqueue_instruction(1, {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::Alu(0),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f, 0, ftlpu::VxmCastTarget::Float16, 4,
        ftlpu::Hemisphere::West, ftlpu::Hemisphere::East});
    lane.tick();
    assert(lane.output().has_value());
    assert(lane.output()->stream == 4);
    assert(lane.output()->byte_count == 2);
    assert(lane.output()->hemisphere == ftlpu::Hemisphere::East);
    const auto bits = static_cast<std::uint16_t>(lane.output()->bytes[0])
        | (static_cast<std::uint16_t>(lane.output()->bytes[1]) << 8);
    assert(std::fabs(ftlpu::Fp16::from_bits(bits).to_float() + 2.0f) < 1.0e-3f);

    constexpr float kBf16Input = 1.00390625f;
    lane.enqueue_instruction(2, {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::Imm(kBf16Input),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f, 0, ftlpu::VxmCastTarget::BFloat16, 8});
    lane.tick();
    assert(lane.output().has_value());
    assert(lane.output()->stream == 8);
    assert(lane.output()->byte_count == 2);
    const auto bf16_bits = static_cast<std::uint16_t>(lane.output()->bytes[0])
        | (static_cast<std::uint16_t>(lane.output()->bytes[1]) << 8);
    assert(bf16_bits == ftlpu::Bf16::from_float(kBf16Input).bits());
    assert(*lane.alu_output(2)
        == ftlpu::Bf16::from_float(kBf16Input).to_float());

    auto bf16_input_lane = ftlpu::VxmLane {};
    bf16_input_lane.enqueue_instruction(0, {
        ftlpu::VxmAluOpcode::Pass,
        ftlpu::VxmLaneOperand::StreamBFloat16(10),
        ftlpu::VxmLaneOperand::Imm(0.0f)});
    auto bf16_streams = ftlpu::VxmLane::StreamBytes {};
    bf16_streams[10] = static_cast<std::uint8_t>(bf16_bits & 0xffu);
    bf16_streams[11] = static_cast<std::uint8_t>(bf16_bits >> 8);
    bf16_input_lane.set_stream_inputs(bf16_streams);
    bf16_input_lane.tick();
    assert(*bf16_input_lane.alu_output(0)
        == ftlpu::Bf16::from_bits(bf16_bits).to_float());

    auto invalid_lane = ftlpu::VxmLane {};
    invalid_lane.enqueue_instruction(0, {
        ftlpu::VxmAluOpcode::Pass,
        ftlpu::VxmLaneOperand::StreamInt8(0),
        ftlpu::VxmLaneOperand::Imm(0.0f)});
    auto rejected = false;
    try {
        invalid_lane.tick();
    }
    catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);
}
