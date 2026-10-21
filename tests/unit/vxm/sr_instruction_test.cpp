#include "ftlpu/vxm/sr_instruction.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

int main()
{
    using namespace ftlpu;
    VxmSrInstruction instruction {};
    instruction.operation = VxmAluOpcode::Multiply;
    instruction.lhs = VxmSrOperand::StreamFloat32(StreamId::West(12).packed());
    instruction.rhs = VxmSrOperand::Immediate(0.5f);
    instruction.result_sr = static_cast<std::uint8_t>(StreamId::East(20).packed());
    instruction.precision = VxmAluPrecision::Float32;
    instruction.result_type = VxmCastTarget::Float32;
    instruction.repeat_count = 123456;
    instruction.lhs_stream_tap = true;
    instruction.lhs_stream_tap = true;

    const auto packet = VxmSrCompactInstructionCodec::encode(instruction);
    const auto decoded = VxmSrCompactInstructionCodec::decode(packet);
    assert(std::get<VxmAluOpcode>(decoded.operation)
        == VxmAluOpcode::Multiply);
    assert(decoded.lhs.kind == VxmSrOperandKind::StreamFloat32);
    assert(decoded.lhs.sr == StreamId::West(12).packed());
    assert(decoded.rhs.kind == VxmSrOperandKind::Immediate);
    assert(std::fabs(decoded.rhs.immediate - 0.5f) < 1.0e-6f);
    assert(decoded.result_sr == StreamId::East(20).packed());
    assert(decoded.precision == VxmAluPrecision::Float32);
    assert(decoded.result_type == VxmCastTarget::Float32);
    assert(decoded.repeat_count == 123456);
    if (!decoded.lhs_stream_tap || decoded.rhs_stream_tap)
        throw std::logic_error("VXM stream tap bits did not round-trip");
    assert(decoded.lhs_stream_tap);
    assert(!decoded.rhs_stream_tap);

    bool rejected = false;
    try {
        auto invalid = instruction;
        invalid.result_sr = static_cast<std::uint8_t>(StreamId::West(31).packed());
        invalid.result_type = VxmCastTarget::Float16;
        (void)VxmSrCompactInstructionCodec::encode(invalid);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);

    auto reciprocal = VxmLaneAluInstruction {
        VxmSpecialAluOpcode::Reciprocal,
        VxmLaneOperand::Acc(),
        VxmLaneOperand::Imm(0.0f)};
    reciprocal.local_scalar_write = true;
    const auto lowered = lower_legacy_vxm_to_explicit_sr(
        3, VxmChainDepth::Four, reciprocal);
    assert(lowered.lhs.kind == VxmSrOperandKind::Accumulator);
    assert(lowered.accumulator_write);
    assert(!lowered.accumulator_emit);
    return 0;
}
