#pragma once

#include "ftlpu/icu/fu_3d_instruction.hpp"
#include "ftlpu/icu/instruction.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ftlpu::detail {

// Transitional adapter for compiler binaries that still carry the old
// absolute-cycle Macro/STREAM_ND descriptors. The caller must translate the
// legacy start cycle into a preceding queue-local NOP before lowering. The
// resulting hardware FU command always starts at relative cycle zero and this
// adapter never enumerates the descriptor's issue points.
inline IcuLoop3D legacy_stream_nd_loop_3d(
    const IcuStreamNdSchedule& schedule)
{
    if (schedule.rank == 0
        || schedule.rank > IcuStreamNdSchedule::kMaxRank)
        throw std::invalid_argument(
            "legacy STREAM_ND rank must be between one and three");

    IcuLoop3D loop;
    loop.start_cycle = 0;
    for (std::size_t dimension = 0; dimension < schedule.rank; ++dimension) {
        loop.counts[dimension] = schedule.counts[dimension];
        loop.cycle_strides[dimension] =
            schedule.cycle_strides[dimension];
    }
    return loop;
}

inline IcuLoop3D legacy_macro_loop_3d(const IcuMacroSchedule& schedule)
{
    return IcuLoop3D {
        0,
        {schedule.inner_count, schedule.outer_count, 1},
        {schedule.inner_interval, schedule.outer_interval, 1},
    };
}

inline std::array<std::int64_t, 3> legacy_stream_nd_strides(
    const IcuStreamNdSchedule& schedule)
{
    std::array<std::int64_t, 3> result {0, 0, 0};
    for (std::size_t dimension = 0; dimension < schedule.rank; ++dimension)
        result[dimension] = schedule.operand_strides[dimension];
    return result;
}

inline std::array<std::int64_t, 3> legacy_macro_strides(
    const IcuMacroSchedule& schedule) noexcept
{
    return {schedule.inner_stride, schedule.outer_stride, 0};
}

inline bool has_legacy_operand_stride(
    const std::array<std::int64_t, 3>& strides) noexcept
{
    return std::any_of(strides.begin(), strides.end(),
        [](std::int64_t stride) { return stride != 0; });
}

inline std::array<std::int64_t, 3> checked_legacy_operand_strides(
    std::array<std::int64_t, 3> strides,
    IcuInductionTarget target,
    IcuInductionTarget supportedTarget,
    const char* command)
{
    const auto hasStride = has_legacy_operand_stride(strides);
    if (target == IcuInductionTarget::None) {
        if (hasStride)
            throw std::invalid_argument(std::string(command)
                + " has operand strides without an induction target");
        return {0, 0, 0};
    }
    if (target != supportedTarget)
        throw std::invalid_argument(std::string(command)
            + " uses an induction target unsupported by this FU");
    return strides;
}

inline MemIcuInstruction lower_legacy_mem_to_3d(
    IcuLoop3D loop,
    std::array<std::int64_t, 3> strides,
    IcuInductionTarget target,
    const MemInstruction& instruction)
{
    strides = checked_legacy_operand_strides(strides, target,
        IcuInductionTarget::MemAddress, "legacy MEM descriptor");
    const auto address = MemIcuAddress3D::Affine(
        instruction.address, strides);
    const auto stream = instruction.stream_id();
    switch (instruction.opcode) {
    case MemOpcode::Read:
        if (instruction.preserve_stream)
            throw std::invalid_argument(
                "legacy MEM read cannot carry write-tap semantics");
        return MemIcuInstruction::Read3D(loop, address, stream);
    case MemOpcode::Write:
        return instruction.preserve_stream
            ? MemIcuInstruction::WriteTap3D(loop, address, stream)
            : MemIcuInstruction::Write3D(loop, address, stream);
    case MemOpcode::Gather:
    case MemOpcode::Scatter:
        throw std::invalid_argument(
            "legacy MEM gather/scatter has no MEM FU 3-D opcode");
    }
    throw std::invalid_argument("legacy MEM opcode is invalid");
}

inline MxmLoadIcuInstruction lower_legacy_mxm_load_to_3d(
    IcuLoop3D loop,
    std::array<std::int64_t, 3> strides,
    IcuInductionTarget target,
    const MxmControlInstruction& instruction)
{
    if (instruction.opcode != MxmControlOpcode::IW)
        throw std::invalid_argument(
            "legacy MXM load descriptor is not an IW instruction");
    if (instruction.weight_load_mode != MxmWeightLoadMode::Supercell
        || instruction.weight_inner_column != 0)
        throw std::invalid_argument(
            "legacy MXM IWColumn has no MXM LOAD_3D encoding");
    strides = checked_legacy_operand_strides(strides, target,
        IcuInductionTarget::MxmWeightColumn,
        "legacy MXM load descriptor");
    return MxmLoadIcuInstruction::Load3D(loop,
        instruction.weight_buffer,
        MxmIcuBufferMode::Fixed,
        instruction.weight_column,
        strides,
        instruction.weight_stream_base,
        instruction.weight_input_mode);
}

inline MxmDequantIcuInstruction lower_legacy_mxm_dequant_to_3d(
    IcuLoop3D loop,
    std::array<std::int64_t, 3> strides,
    IcuInductionTarget target,
    const MxmDequantInstruction& instruction)
{
    static_cast<void>(checked_legacy_operand_strides(strides, target,
        IcuInductionTarget::None,
        "legacy MXM dequant descriptor"));
    return MxmDequantIcuInstruction::Dequant3D(loop, instruction);
}

inline MxmComputeIcuInstruction lower_legacy_mxm_compute_to_3d(
    IcuLoop3D loop,
    std::array<std::int64_t, 3> strides,
    IcuInductionTarget target,
    const MxmControlInstruction& instruction)
{
    if (instruction.opcode == MxmControlOpcode::Decode)
        throw std::invalid_argument(
            "legacy MXM decode has no MXM compute ICU 3-D encoding");
    strides = checked_legacy_operand_strides(strides, target,
        IcuInductionTarget::MxmAccumulatorAddress,
        "legacy MXM compute descriptor");
    if (instruction.opcode == MxmControlOpcode::AccumulatorRead)
        return MxmComputeIcuInstruction::AccumulatorRead3D(loop,
            instruction.stream_base,
            instruction.accumulator_address,
            strides,
            instruction.accumulator_clear,
            instruction.accumulator_output_format,
            instruction.accumulator_destination);
    if (instruction.opcode != MxmControlOpcode::Compute)
        throw std::invalid_argument(
            "legacy MXM compute descriptor has an unsupported opcode");
    const MxmComputeIcuMode mode {
        instruction.accumulator_destination,
        instruction.accumulator_clear,
        instruction.accumulator_output_format,
    };
    return MxmComputeIcuInstruction::Compute3D(loop,
        instruction.weight_buffer,
        MxmIcuBufferMode::Fixed,
        instruction.activation_stream_base,
        instruction.stream_base,
        instruction.accumulator_address,
        strides,
        instruction.accumulator_row_stride,
        instruction.data_format,
        mode,
        MxmComputeIcuInstruction::kNoTerminalDimension,
        mode);
}

} // namespace ftlpu::detail
