#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/mxm/weight_dequantizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ftlpu {

// Hardware-visible three-counter launch domain. Counter 0 is innermost.
struct IcuLoop3D {
    static constexpr std::size_t kDimensions = 3;

    // Compiler-side absolute cycle. Materialization clears this before
    // encoding; the physical queue uses wait_cycle instead.
    std::size_t start_cycle{0};
    std::array<std::size_t, kDimensions> counts{1, 1, 1};
    std::array<std::size_t, kDimensions> cycle_strides{1, 1, 1};
    // Queue-local delay after this packet reaches the ICU head.
    std::size_t wait_cycle{0};
};

struct IcuCoordinate3D {
    std::array<std::size_t, IcuLoop3D::kDimensions> index{0, 0, 0};

    std::size_t operator[](std::size_t dimension) const
    {
        return index.at(dimension);
    }
};

// The outer counter may use a power-of-two carry/wrap address generator.
// Hardware takes the group index and within-group index from a shift and a
// mask; it does not implement a general divider or a hidden launch counter.
// For group_size=1 this reduces to an ordinary affine 3-D address with
// outer_group_stride as the counter-2 stride.
struct MemIcuAddress3D {
    std::size_t base_address{0};
    std::int64_t inner_stride{0};
    std::int64_t middle_stride{0};
    std::size_t outer_group_size{1};
    std::int64_t outer_inner_stride{0};
    std::int64_t outer_group_stride{0};

    static constexpr MemIcuAddress3D Affine(
        std::size_t base_address,
        std::array<std::int64_t, 3> strides) noexcept
    {
        return {base_address, strides[0], strides[1], 1, 0, strides[2]};
    }

    static constexpr MemIcuAddress3D BlockedOuter(
        std::size_t base_address,
        std::int64_t inner_stride,
        std::int64_t middle_stride,
        std::size_t outer_group_size,
        std::int64_t outer_inner_stride,
        std::int64_t outer_group_stride) noexcept
    {
        return {base_address, inner_stride, middle_stride,
            outer_group_size, outer_inner_stride, outer_group_stride};
    }
};

enum class MemIcuOpcode : std::uint8_t {
    Read3D = 0,
    Write3D = 1,
    WriteTap3D = 2,
};

struct MemIcuInstruction {
    MemIcuOpcode opcode{MemIcuOpcode::Read3D};
    IcuLoop3D loop{};
    MemIcuAddress3D address{};
    std::size_t stream{0};

    static MemIcuInstruction Read3D(
        IcuLoop3D loop,
        MemIcuAddress3D address,
        StreamId stream)
    {
        return {MemIcuOpcode::Read3D, loop, address, stream.packed()};
    }

    static MemIcuInstruction Write3D(
        IcuLoop3D loop,
        MemIcuAddress3D address,
        StreamId stream)
    {
        return {MemIcuOpcode::Write3D, loop, address, stream.packed()};
    }

    static MemIcuInstruction WriteTap3D(
        IcuLoop3D loop,
        MemIcuAddress3D address,
        StreamId stream)
    {
        return {MemIcuOpcode::WriteTap3D, loop, address, stream.packed()};
    }
};

// One physical MEM bank alternates two statically scheduled 2-D event
// streams. Both streams address the same affine rows; there is no data FIFO,
// ready/valid handshake, or run-time empty/full test in the ICU.
struct MemIcuWriteRead2DInstruction {
    std::size_t start_wait{0};
    std::array<std::size_t, 2> counts{1, 1};
    std::array<std::size_t, 2> write_cycle_strides{1, 1};
    std::array<std::size_t, 2> read_cycle_strides{1, 1};
    std::size_t read_start_offset{1};
    std::size_t base_address{0};
    std::array<std::int64_t, 2> address_strides{0, 0};
    std::size_t write_stream{0};
    std::size_t read_stream_base{0};
    std::int64_t read_stream_outer_stride{0};
};

// Two physical MXM buffers make dimension-parity selection a small XOR in
// the ICU instead of a model-specific block-number calculation.
enum class MxmIcuBufferMode : std::uint8_t {
    Fixed = 0,
    ToggleDimension0 = 1,
    ToggleDimension1 = 2,
    ToggleDimension2 = 3,
};

enum class MxmLoadIcuOpcode : std::uint8_t {
    Load3D = 0,
    DecodeLoadActivation3D = 1,
};

struct MxmLoadIcuInstruction {
    MxmLoadIcuOpcode opcode{MxmLoadIcuOpcode::Load3D};
    IcuLoop3D loop{};
    std::size_t weight_buffer_base{0};
    MxmIcuBufferMode weight_buffer_mode{MxmIcuBufferMode::Fixed};
    std::size_t weight_column_base{0};
    std::array<std::int64_t, 3> weight_column_strides{1, 0, 0};
    std::size_t weight_stream_base{0};
    MxmWeightInputMode weight_input_mode{
        MxmWeightInputMode::Int8DequantBf16};
    MxmDataFormat data_format{MxmDataFormat::BFloat16};
    MxmDecodeLayout decode_layout{MxmDecodeLayout::Linear1x16};

    static MxmLoadIcuInstruction Load3D(
        IcuLoop3D loop,
        std::size_t weight_buffer_base,
        MxmIcuBufferMode weight_buffer_mode,
        std::size_t weight_column_base,
        std::array<std::int64_t, 3> weight_column_strides,
        std::size_t weight_stream_base,
        MxmWeightInputMode weight_input_mode =
            MxmWeightInputMode::Int8DequantBf16)
    {
        auto instruction = MxmLoadIcuInstruction {};
        instruction.opcode = MxmLoadIcuOpcode::Load3D;
        instruction.loop = loop;
        instruction.weight_buffer_base = weight_buffer_base;
        instruction.weight_buffer_mode = weight_buffer_mode;
        instruction.weight_column_base = weight_column_base;
        instruction.weight_column_strides = weight_column_strides;
        instruction.weight_stream_base = weight_stream_base;
        instruction.weight_input_mode = weight_input_mode;
        return instruction;
    }

    static MxmLoadIcuInstruction DecodeLoadActivation3D(
        IcuLoop3D loop,
        std::size_t activation_buffer_base,
        MxmIcuBufferMode activation_buffer_mode,
        std::size_t activation_stream_base,
        MxmDataFormat data_format,
        MxmDecodeLayout decode_layout)
    {
        auto instruction = MxmLoadIcuInstruction {};
        instruction.opcode = MxmLoadIcuOpcode::DecodeLoadActivation3D;
        instruction.loop = loop;
        instruction.weight_buffer_base = activation_buffer_base;
        instruction.weight_buffer_mode = activation_buffer_mode;
        instruction.weight_stream_base = activation_stream_base;
        instruction.data_format = data_format;
        instruction.decode_layout = decode_layout;
        return instruction;
    }
};

struct MxmDequantIcuInstruction {
    IcuLoop3D loop{};
    MxmDequantInstruction instruction{};

    static MxmDequantIcuInstruction Dequant3D(
        IcuLoop3D loop,
        MxmDequantInstruction instruction)
    {
        return {loop, instruction};
    }
};

struct MxmComputeIcuMode {
    MxmAccumulatorDestination accumulator_destination{
        MxmAccumulatorDestination::Sram};
    bool accumulator_clear{false};
    MxmAccumulatorOutputFormat accumulator_output_format{
        MxmAccumulatorOutputFormat::Float32};
};

// FU-local operations decoded by one physical MXM compute ICU.  These values
// are carried by the local-operation field in every word of its 3-D packet;
// they are not global ICU opcodes or functional-unit selectors.
enum class MxmComputeIcuOpcode : std::uint8_t {
    Compute3D = 0,
    AccumulatorRead3D = 1,
    DecodeStreamCompute3D = 2,
};

struct MxmComputeIcuInstruction {
    static constexpr std::size_t kNoTerminalDimension = 3;

    MxmComputeIcuOpcode opcode{MxmComputeIcuOpcode::Compute3D};
    IcuLoop3D loop{};
    std::size_t weight_buffer_base{0};
    MxmIcuBufferMode weight_buffer_mode{MxmIcuBufferMode::Fixed};
    std::size_t activation_stream_base{0};
    std::size_t result_stream_base{0};
    std::size_t accumulator_address_base{0};
    std::array<std::int64_t, 3> accumulator_address_strides{0, 0, 0};
    std::size_t accumulator_row_stride{1};
    MxmDataFormat data_format{MxmDataFormat::BFloat16};
    std::size_t accumulator_column{0};
    MxmDecodeLayout decode_layout{MxmDecodeLayout::Linear1x16};
    MxmComputeIcuMode regular_mode{};
    std::size_t terminal_dimension{kNoTerminalDimension};
    MxmComputeIcuMode terminal_mode{};

    static MxmComputeIcuInstruction Compute3D(
        IcuLoop3D loop,
        std::size_t weight_buffer_base,
        MxmIcuBufferMode weight_buffer_mode,
        std::size_t activation_stream_base,
        std::size_t result_stream_base,
        std::size_t accumulator_address_base,
        std::array<std::int64_t, 3> accumulator_address_strides,
        std::size_t accumulator_row_stride,
        MxmDataFormat data_format,
        MxmComputeIcuMode regular_mode,
        std::size_t terminal_dimension = kNoTerminalDimension,
        MxmComputeIcuMode terminal_mode = {})
    {
        auto instruction = MxmComputeIcuInstruction {};
        instruction.opcode = MxmComputeIcuOpcode::Compute3D;
        instruction.loop = loop;
        instruction.weight_buffer_base = weight_buffer_base;
        instruction.weight_buffer_mode = weight_buffer_mode;
        instruction.activation_stream_base = activation_stream_base;
        instruction.result_stream_base = result_stream_base;
        instruction.accumulator_address_base = accumulator_address_base;
        instruction.accumulator_address_strides = accumulator_address_strides;
        instruction.accumulator_row_stride = accumulator_row_stride;
        instruction.data_format = data_format;
        instruction.regular_mode = regular_mode;
        instruction.terminal_dimension = terminal_dimension;
        instruction.terminal_mode = terminal_mode;
        return instruction;
    }

    static MxmComputeIcuInstruction AccumulatorRead3D(
        IcuLoop3D loop,
        std::size_t result_stream_base,
        std::size_t accumulator_address_base,
        std::array<std::int64_t, 3> accumulator_address_strides,
        bool accumulator_clear = true,
        MxmAccumulatorOutputFormat accumulator_output_format =
            MxmAccumulatorOutputFormat::Float32,
        MxmAccumulatorDestination accumulator_destination =
            MxmAccumulatorDestination::Stream)
    {
        auto instruction = MxmComputeIcuInstruction {};
        instruction.opcode = MxmComputeIcuOpcode::AccumulatorRead3D;
        instruction.loop = loop;
        instruction.result_stream_base = result_stream_base;
        instruction.accumulator_address_base = accumulator_address_base;
        instruction.accumulator_address_strides = accumulator_address_strides;
        instruction.regular_mode = {
            accumulator_destination,
            accumulator_clear,
            accumulator_output_format,
        };
        return instruction;
    }

    static MxmComputeIcuInstruction DecodeStreamCompute3D(
        IcuLoop3D loop,
        std::size_t activation_buffer_base,
        MxmIcuBufferMode activation_buffer_mode,
        std::size_t result_stream_base,
        MxmDataFormat data_format,
        std::size_t accumulator_address_base,
        std::array<std::int64_t, 3> accumulator_address_strides,
        std::size_t accumulator_column,
        MxmAccumulatorDestination accumulator_destination,
        bool accumulator_clear,
        MxmDecodeLayout decode_layout)
    {
        auto instruction = MxmComputeIcuInstruction {};
        instruction.opcode = MxmComputeIcuOpcode::DecodeStreamCompute3D;
        instruction.loop = loop;
        instruction.weight_buffer_base = activation_buffer_base;
        instruction.weight_buffer_mode = activation_buffer_mode;
        instruction.result_stream_base = result_stream_base;
        instruction.accumulator_address_base = accumulator_address_base;
        instruction.accumulator_address_strides = accumulator_address_strides;
        instruction.data_format = data_format;
        instruction.accumulator_column = accumulator_column;
        instruction.regular_mode.accumulator_destination =
            accumulator_destination;
        instruction.regular_mode.accumulator_clear = accumulator_clear;
        instruction.decode_layout = decode_layout;
        return instruction;
    }
};

namespace detail {

inline void validate_icu_loop_3d(const IcuLoop3D& loop)
{
    if (loop.start_cycle != 0)
        throw std::invalid_argument(
            "hardware ICU loops must use a zero-relative start");
    if (loop.wait_cycle >= (std::size_t {1} << 24))
        throw std::invalid_argument(
            "ICU 3-D wait_cycle exceeds its 24-bit field");
    std::size_t lower_span = 0;
    for (std::size_t dimension = 0;
         dimension < IcuLoop3D::kDimensions; ++dimension) {
        const auto count = loop.counts[dimension];
        const auto stride = loop.cycle_strides[dimension];
        if (count == 0 || count > 65536 || stride == 0
            || stride >= (std::size_t {1} << 24))
            throw std::invalid_argument(
                "ICU 3-D loop dimension " + std::to_string(dimension)
                + " exceeds its fixed field width: count="
                + std::to_string(count) + " stride="
                + std::to_string(stride));
        if (dimension != 0 && count > 1 && stride <= lower_span)
            throw std::invalid_argument(
                "ICU 3-D loop dimensions overlap in issue time");
        const auto steps = count - 1;
        if (steps != 0
            && stride > (std::numeric_limits<std::size_t>::max()
                    - lower_span) / steps)
            throw std::overflow_error("ICU 3-D cycle span overflows");
        lower_span += steps * stride;
    }
    constexpr auto cycleLimit = std::size_t {1} << 24;
    if (lower_span >= cycleLimit)
        throw std::invalid_argument(
            "ICU 3-D relative cycle span exceeds the 24-bit schedule domain");
}

inline std::size_t icu_loop_3d_point_count(const IcuLoop3D& loop)
{
    std::size_t points = 1;
    for (const auto count : loop.counts) {
        if (points > std::numeric_limits<std::size_t>::max() / count)
            throw std::overflow_error("ICU 3-D point count overflows");
        points *= count;
    }
    return points;
}

inline std::size_t icu_loop_3d_issue_cycle(
    const IcuLoop3D& loop,
    const IcuCoordinate3D& coordinate)
{
    // This is an instruction-local cycle offset. The physical ICU never
    // compares it with a program-global or system-global cycle counter.
    auto cycle = loop.wait_cycle;
    for (std::size_t dimension = 0;
         dimension < IcuLoop3D::kDimensions; ++dimension)
        cycle += coordinate.index[dimension]
            * loop.cycle_strides[dimension];
    return cycle;
}

inline bool advance_icu_coordinate_3d(
    const IcuLoop3D& loop,
    IcuCoordinate3D& coordinate) noexcept
{
    for (std::size_t dimension = 0;
         dimension < IcuLoop3D::kDimensions; ++dimension) {
        if (coordinate.index[dimension] < loop.counts[dimension] - 1) {
            ++coordinate.index[dimension];
            return true;
        }
        coordinate.index[dimension] = 0;
    }
    return false;
}

inline std::int64_t checked_icu_3d_operand(
    std::size_t base,
    const std::array<std::int64_t, 3>& strides,
    const IcuCoordinate3D& coordinate,
    const char* field)
{
    if (base > static_cast<std::size_t>(
                   std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error(std::string(field) + " base overflows");
    auto value = static_cast<std::int64_t>(base);
    for (std::size_t dimension = 0;
         dimension < IcuLoop3D::kDimensions; ++dimension) {
        const auto index = coordinate.index[dimension];
        const auto stride = strides[dimension];
        if (index > static_cast<std::size_t>(
                        std::numeric_limits<std::int64_t>::max())
            || (stride > 0
                && index > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max() / stride))
            || (stride < 0 && stride != -1
                && index > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::min() / stride)))
            throw std::overflow_error(
                std::string(field) + " induction multiply overflows");
        const auto delta = static_cast<std::int64_t>(index) * stride;
        if ((delta > 0
                && value > std::numeric_limits<std::int64_t>::max() - delta)
            || (delta < 0
                && value < std::numeric_limits<std::int64_t>::min() - delta))
            throw std::overflow_error(
                std::string(field) + " induction addition overflows");
        value += delta;
    }
    return value;
}

inline std::size_t mem_icu_address_3d(
    const MemIcuInstruction& instruction,
    const IcuCoordinate3D& coordinate)
{
    if (instruction.address.outer_group_size == 0
        || !std::has_single_bit(
            instruction.address.outer_group_size))
        throw std::invalid_argument(
            "MEM ICU 3-D outer group size must be a power of two");
    const auto outer = coordinate.index[2];
    const auto groupShift = std::countr_zero(
        instruction.address.outer_group_size);
    const auto groupMask = instruction.address.outer_group_size - 1;
    const IcuCoordinate3D transformed {{
        coordinate.index[0],
        coordinate.index[1],
        outer >> groupShift,
    }};
    const std::array<std::int64_t, 3> strides {
        instruction.address.inner_stride,
        instruction.address.middle_stride,
        instruction.address.outer_group_stride,
    };
    auto address = checked_icu_3d_operand(
        instruction.address.base_address,
        strides,
        transformed,
        "MEM ICU 3-D address");
    const auto remainder = outer & groupMask;
    const auto extra = checked_icu_3d_operand(
        0,
        {instruction.address.outer_inner_stride, 0, 0},
        IcuCoordinate3D {{remainder, 0, 0}},
        "MEM ICU 3-D blocked address");
    if ((extra > 0
            && address > std::numeric_limits<std::int64_t>::max() - extra)
        || (extra < 0
            && address < std::numeric_limits<std::int64_t>::min() - extra))
        throw std::overflow_error("MEM ICU 3-D blocked address overflows");
    address += extra;
    if (address < 0
        || static_cast<std::size_t>(address) >= hw::kSramDepthRows)
        throw std::out_of_range(
            "MEM ICU 3-D address is outside the queue-local SRAM bank");
    return static_cast<std::size_t>(address);
}

inline std::size_t mxm_icu_buffer_3d(
    std::size_t base,
    MxmIcuBufferMode mode,
    const IcuCoordinate3D& coordinate)
{
    MxmControlInstruction::check_weight_buffer(base);
    if (mode == MxmIcuBufferMode::Fixed) return base;
    const auto encoded = static_cast<std::size_t>(mode);
    if (encoded < 1 || encoded > 3)
        throw std::invalid_argument("MXM ICU 3-D buffer mode is invalid");
    const auto dimension = encoded - 1;
    const auto buffer = base ^ (coordinate.index[dimension] & 1U);
    MxmControlInstruction::check_weight_buffer(buffer);
    return buffer;
}

inline void validate_mem_icu_instruction(
    const MemIcuInstruction& instruction)
{
    validate_icu_loop_3d(instruction.loop);
    static_cast<void>(StreamId::from_packed(instruction.stream));
    switch (instruction.opcode) {
    case MemIcuOpcode::Read3D:
    case MemIcuOpcode::Write3D:
    case MemIcuOpcode::WriteTap3D:
        break;
    default:
        throw std::invalid_argument("MEM ICU 3-D opcode is invalid");
    }
    for (std::size_t outer = 0; outer < instruction.loop.counts[2]; ++outer) {
        for (const auto inner : {std::size_t {0},
                 instruction.loop.counts[0] - 1}) {
            for (const auto middle : {std::size_t {0},
                     instruction.loop.counts[1] - 1})
                static_cast<void>(mem_icu_address_3d(
                    instruction, IcuCoordinate3D {{inner, middle, outer}}));
        }
    }
}

inline std::size_t mem_icu_write_read_2d_last_issue_cycle(
    const MemIcuWriteRead2DInstruction& instruction) noexcept
{
    const auto lastWrite = instruction.start_wait
        + (instruction.counts[0] - 1) * instruction.write_cycle_strides[0]
        + (instruction.counts[1] - 1) * instruction.write_cycle_strides[1];
    const auto lastRead = instruction.start_wait + instruction.read_start_offset
        + (instruction.counts[0] - 1) * instruction.read_cycle_strides[0]
        + (instruction.counts[1] - 1) * instruction.read_cycle_strides[1];
    return std::max(lastWrite, lastRead);
}

inline void validate_mem_icu_write_read_2d_instruction(
    const MemIcuWriteRead2DInstruction& instruction)
{
    constexpr auto cycleLimit = std::size_t {1} << 24;
    constexpr auto countLimit = std::size_t {1} << 16;
    if (instruction.start_wait >= cycleLimit
        || instruction.read_start_offset >= cycleLimit
        || instruction.base_address >= hw::kSramDepthRows)
        throw std::invalid_argument(
            "MEM WRITE_READ_2D wait, offset, or base is out of range");
    static_cast<void>(StreamId::from_packed(instruction.write_stream));
    static_cast<void>(StreamId::from_packed(instruction.read_stream_base));
    if (instruction.read_stream_outer_stride < -32
        || instruction.read_stream_outer_stride > 31)
        throw std::invalid_argument(
            "MEM WRITE_READ_2D read stream stride exceeds 6 bits");

    const auto validateSweep = [&](const std::array<std::size_t, 2>& strides) {
        for (std::size_t dim = 0; dim < 2; ++dim)
            if (instruction.counts[dim] == 0
                || instruction.counts[dim] > countLimit
                || strides[dim] == 0 || strides[dim] >= cycleLimit)
                throw std::invalid_argument(
                    "MEM WRITE_READ_2D count or cycle stride is out of range");
        const auto innerSpan = (instruction.counts[0] - 1) * strides[0];
        if (instruction.counts[1] > 1 && strides[1] <= innerSpan)
            throw std::invalid_argument(
                "MEM WRITE_READ_2D sweep overlaps its own issue cycles");
        return innerSpan + (instruction.counts[1] - 1) * strides[1];
    };
    const auto writeSpan = validateSweep(instruction.write_cycle_strides);
    const auto readSpan = validateSweep(instruction.read_cycle_strides);
    if (instruction.start_wait + writeSpan >= cycleLimit
        || instruction.start_wait + instruction.read_start_offset
                + readSpan >= cycleLimit)
        throw std::invalid_argument(
            "MEM WRITE_READ_2D relative cycle span exceeds 24 bits");

    // The same coordinate must have been written before it is read. Because
    // this difference is affine, its minimum occurs at a rectangle corner.
    auto minReadAfterWrite = static_cast<std::int64_t>(
        instruction.read_start_offset);
    for (std::size_t dim = 0; dim < 2; ++dim) {
        const auto difference = static_cast<std::int64_t>(
            instruction.read_cycle_strides[dim])
            - static_cast<std::int64_t>(
                instruction.write_cycle_strides[dim]);
        if (difference < 0)
            minReadAfterWrite +=
                static_cast<std::int64_t>(instruction.counts[dim] - 1)
                * difference;
    }
    if (minReadAfterWrite <= 0)
        throw std::invalid_argument(
            "MEM WRITE_READ_2D may read a coordinate before its write");

    const auto lastReadStream = static_cast<std::int64_t>(
        instruction.read_stream_base)
        + static_cast<std::int64_t>(instruction.counts[1] - 1)
            * instruction.read_stream_outer_stride;
    if (lastReadStream < 0
        || lastReadStream >= static_cast<std::int64_t>(hw::kStreams))
        throw std::invalid_argument(
            "MEM WRITE_READ_2D read stream induction is out of range");

    for (std::size_t dim = 0; dim < 2; ++dim)
        if (instruction.address_strides[dim] < -(std::int64_t {1} << 19)
            || instruction.address_strides[dim]
                >= (std::int64_t {1} << 19))
            throw std::invalid_argument(
                "MEM WRITE_READ_2D address stride exceeds 20 bits");
    for (std::size_t i : {std::size_t {0}, instruction.counts[0] - 1})
        for (std::size_t j : {std::size_t {0}, instruction.counts[1] - 1}) {
            const auto address = static_cast<std::int64_t>(
                instruction.base_address)
                + static_cast<std::int64_t>(i)
                    * instruction.address_strides[0]
                + static_cast<std::int64_t>(j)
                    * instruction.address_strides[1];
            if (address < 0
                || address >= static_cast<std::int64_t>(hw::kSramDepthRows))
                throw std::invalid_argument(
                    "MEM WRITE_READ_2D affine address exceeds bank rows");
        }

    // A compact conservative proof of no row overwrite: addresses are
    // injective over the rectangle. A future ISA may permit safe row reuse.
    const auto a = instruction.address_strides[0] < 0
        ? -instruction.address_strides[0]
        : instruction.address_strides[0];
    const auto b = instruction.address_strides[1] < 0
        ? -instruction.address_strides[1]
        : instruction.address_strides[1];
    if ((instruction.counts[0] > 1 && a == 0)
        || (instruction.counts[1] > 1 && b == 0))
        throw std::invalid_argument(
            "MEM WRITE_READ_2D row reuse is not statically proven safe");
    if (a != 0 && b != 0) {
        const auto divisor = std::gcd(a, b);
        if (b / divisor <= static_cast<std::int64_t>(
                instruction.counts[0] - 1)
            && a / divisor <= static_cast<std::int64_t>(
                instruction.counts[1] - 1))
            throw std::invalid_argument(
                "MEM WRITE_READ_2D row reuse is not statically proven safe");
    }

    // Either all writes finish before reads begin, or the two periodic
    // sweeps occupy disjoint residue classes. Both proofs avoid enumerating
    // a potentially 65536 x 65536 domain during packet decode.
    if (instruction.read_start_offset <= writeSpan) {
        std::size_t period = 0;
        for (std::size_t dim = 0; dim < 2; ++dim) {
            if (instruction.counts[dim] <= 1) continue;
            period = std::gcd(period,
                instruction.write_cycle_strides[dim]);
            period = std::gcd(period,
                instruction.read_cycle_strides[dim]);
        }
        if (period == 0
            || instruction.read_start_offset % period == 0)
            throw std::invalid_argument(
                "MEM WRITE_READ_2D single-port conflict is not statically excluded");
    }
}

inline MemInstruction expand_mem_icu_write_read_2d_instruction(
    const MemIcuWriteRead2DInstruction& instruction,
    std::size_t inner, std::size_t outer, bool read)
{
    const auto address = static_cast<std::size_t>(
        static_cast<std::int64_t>(instruction.base_address)
        + static_cast<std::int64_t>(inner)
            * instruction.address_strides[0]
        + static_cast<std::int64_t>(outer)
            * instruction.address_strides[1]);
    if (read)
        return MemInstruction::Read(address, static_cast<std::size_t>(
            static_cast<std::int64_t>(instruction.read_stream_base)
            + static_cast<std::int64_t>(outer)
                * instruction.read_stream_outer_stride));
    return MemInstruction::Write(address, instruction.write_stream);
}

inline MemInstruction expand_mem_icu_instruction(
    const MemIcuInstruction& instruction,
    const IcuCoordinate3D& coordinate)
{
    const auto address = mem_icu_address_3d(instruction, coordinate);
    switch (instruction.opcode) {
    case MemIcuOpcode::Read3D:
        return MemInstruction::Read(address, instruction.stream);
    case MemIcuOpcode::Write3D:
        return MemInstruction::Write(address, instruction.stream);
    case MemIcuOpcode::WriteTap3D:
        return MemInstruction::WriteTap(address, instruction.stream);
    }
    throw std::logic_error("MEM ICU 3-D opcode is invalid");
}

inline void validate_mxm_load_icu_instruction(
    const MxmLoadIcuInstruction& instruction)
{
    validate_icu_loop_3d(instruction.loop);
    if (instruction.opcode
        == MxmLoadIcuOpcode::DecodeLoadActivation3D) {
        MxmControlInstruction::check_data_format(instruction.data_format);
        MxmControlInstruction::check_decode_layout(
            instruction.decode_layout);
        for (std::size_t mask = 0; mask < 8; ++mask) {
            const IcuCoordinate3D coordinate {{
                (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
                (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
                (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
            }};
            static_cast<void>(mxm_icu_buffer_3d(
                instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate));
        }
        static_cast<void>(MxmControlInstruction::DecodeLoadActivation(
            instruction.weight_buffer_base,
            instruction.weight_stream_base,
            instruction.data_format,
            instruction.decode_layout));
        return;
    }
    if (instruction.opcode != MxmLoadIcuOpcode::Load3D)
        throw std::invalid_argument("MXM load ICU 3-D opcode is invalid");
    for (std::size_t mask = 0; mask < 8; ++mask) {
        IcuCoordinate3D coordinate {{
            (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
            (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
            (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
        }};
        const auto column = checked_icu_3d_operand(
            instruction.weight_column_base,
            instruction.weight_column_strides,
            coordinate,
            "MXM LOAD_3D weight column");
        if (column < 0)
            throw std::out_of_range(
                "MXM LOAD_3D weight column underflows");
        MxmControlInstruction::check_column(
            static_cast<std::size_t>(column));
        static_cast<void>(mxm_icu_buffer_3d(
            instruction.weight_buffer_base,
            instruction.weight_buffer_mode,
            coordinate));
    }
    static_cast<void>(MxmControlInstruction::IW(
        instruction.weight_buffer_base,
        instruction.weight_column_base,
        instruction.weight_input_mode,
        instruction.weight_stream_base));
}

inline MxmControlInstruction expand_mxm_load_icu_instruction(
    const MxmLoadIcuInstruction& instruction,
    const IcuCoordinate3D& coordinate)
{
    if (instruction.opcode
        == MxmLoadIcuOpcode::DecodeLoadActivation3D)
        return MxmControlInstruction::DecodeLoadActivation(
            mxm_icu_buffer_3d(instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate),
            instruction.weight_stream_base,
            instruction.data_format,
            instruction.decode_layout);
    if (instruction.opcode != MxmLoadIcuOpcode::Load3D)
        throw std::logic_error("MXM load ICU 3-D opcode is invalid");
    const auto column = checked_icu_3d_operand(
        instruction.weight_column_base,
        instruction.weight_column_strides,
        coordinate,
        "MXM LOAD_3D weight column");
    if (column < 0)
        throw std::out_of_range("MXM LOAD_3D weight column underflows");
    return MxmControlInstruction::IW(
        mxm_icu_buffer_3d(instruction.weight_buffer_base,
            instruction.weight_buffer_mode, coordinate),
        static_cast<std::size_t>(column),
        instruction.weight_input_mode,
        instruction.weight_stream_base);
}

inline void validate_mxm_dequant_icu_instruction(
    const MxmDequantIcuInstruction& instruction)
{
    validate_icu_loop_3d(instruction.loop);
}

inline void validate_mxm_compute_mode(const MxmComputeIcuMode& mode)
{
    switch (mode.accumulator_destination) {
    case MxmAccumulatorDestination::Sram:
    case MxmAccumulatorDestination::Stream:
        break;
    default:
        throw std::invalid_argument("MXM COMPUTE_3D destination is invalid");
    }
    switch (mode.accumulator_output_format) {
    case MxmAccumulatorOutputFormat::Float32:
    case MxmAccumulatorOutputFormat::BFloat16:
        break;
    default:
        throw std::invalid_argument("MXM COMPUTE_3D output format is invalid");
    }
}

inline MxmComputeIcuMode mxm_compute_mode_3d(
    const MxmComputeIcuInstruction& instruction,
    const IcuCoordinate3D& coordinate)
{
    if (instruction.terminal_dimension
            < IcuLoop3D::kDimensions
        && coordinate.index[instruction.terminal_dimension] + 1
            == instruction.loop.counts[instruction.terminal_dimension])
        return instruction.terminal_mode;
    return instruction.regular_mode;
}

inline void validate_mxm_compute_icu_instruction(
    const MxmComputeIcuInstruction& instruction)
{
    validate_icu_loop_3d(instruction.loop);
    if (instruction.opcode == MxmComputeIcuOpcode::AccumulatorRead3D) {
        validate_mxm_compute_mode(instruction.regular_mode);
        for (std::size_t mask = 0; mask < 8; ++mask) {
            const IcuCoordinate3D coordinate {{
                (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
                (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
                (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
            }};
            const auto address = checked_icu_3d_operand(
                instruction.accumulator_address_base,
                instruction.accumulator_address_strides,
                coordinate,
                "MXM ACCUMULATOR_READ_3D accumulator address");
            if (address < 0)
                throw std::out_of_range(
                    "MXM ACCUMULATOR_READ_3D accumulator address underflows");
            MxmControlInstruction::check_accumulator_address(
                static_cast<std::size_t>(address));
        }
        static_cast<void>(MxmControlInstruction::AccumulatorRead(
            instruction.accumulator_address_base,
            instruction.result_stream_base,
            instruction.regular_mode.accumulator_clear,
            instruction.regular_mode.accumulator_output_format,
            instruction.regular_mode.accumulator_destination));
        return;
    }
    if (instruction.opcode
        == MxmComputeIcuOpcode::DecodeStreamCompute3D) {
        validate_mxm_compute_mode(instruction.regular_mode);
        MxmControlInstruction::check_data_format(instruction.data_format);
        MxmControlInstruction::check_decode_layout(
            instruction.decode_layout);
        MxmControlInstruction::check_column(
            instruction.accumulator_column);
        for (std::size_t mask = 0; mask < 8; ++mask) {
            const IcuCoordinate3D coordinate {{
                (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
                (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
                (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
            }};
            static_cast<void>(mxm_icu_buffer_3d(
                instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate));
            const auto address = checked_icu_3d_operand(
                instruction.accumulator_address_base,
                instruction.accumulator_address_strides,
                coordinate,
                "MXM DECODE_STREAM_COMPUTE_3D accumulator address");
            if (address < 0)
                throw std::out_of_range(
                    "MXM DECODE_STREAM_COMPUTE_3D accumulator address underflows");
            MxmControlInstruction::check_accumulator_address(
                static_cast<std::size_t>(address));
        }
        static_cast<void>(MxmControlInstruction::DecodeStreamCompute(
            instruction.weight_buffer_base,
            instruction.result_stream_base,
            instruction.data_format,
            instruction.accumulator_address_base,
            instruction.accumulator_column,
            instruction.regular_mode.accumulator_destination,
            instruction.regular_mode.accumulator_clear,
            instruction.decode_layout));
        return;
    }
    if (instruction.opcode != MxmComputeIcuOpcode::Compute3D)
        throw std::invalid_argument("MXM compute ICU 3-D opcode is invalid");
    if (instruction.loop.counts[0] > hw::kMxmRows)
        throw std::out_of_range(
            "MXM COMPUTE_3D inner count exceeds one row wave");
    if (instruction.accumulator_row_stride == 0)
        throw std::invalid_argument(
            "MXM COMPUTE_3D accumulator row stride must be non-zero");
    if (instruction.accumulator_address_strides[0] != 0)
        throw std::invalid_argument(
            "MXM COMPUTE_3D counter 0 cannot induce the accumulator base because the MXM row wave applies row_stride");
    if (instruction.terminal_dimension
            > MxmComputeIcuInstruction::kNoTerminalDimension)
        throw std::invalid_argument(
            "MXM COMPUTE_3D terminal dimension is invalid");
    validate_mxm_compute_mode(instruction.regular_mode);
    validate_mxm_compute_mode(instruction.terminal_mode);
    MxmControlInstruction::check_data_format(instruction.data_format);
    const auto rowCount = instruction.loop.counts[0];
    if (rowCount > 1 && instruction.accumulator_row_stride
        > std::numeric_limits<std::size_t>::max() / (rowCount - 1))
        throw std::overflow_error(
            "MXM COMPUTE_3D accumulator row span overflows");
    const auto rowSpan = (rowCount - 1)
        * instruction.accumulator_row_stride;
    for (std::size_t mask = 0; mask < 8; ++mask) {
        const IcuCoordinate3D coordinate {{
            (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
            (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
            (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
        }};
        static_cast<void>(mxm_icu_buffer_3d(
            instruction.weight_buffer_base,
            instruction.weight_buffer_mode,
            coordinate));
        const auto address = checked_icu_3d_operand(
            instruction.accumulator_address_base,
            instruction.accumulator_address_strides,
            coordinate,
            "MXM COMPUTE_3D accumulator address");
        if (address < 0)
            throw std::out_of_range(
                "MXM COMPUTE_3D accumulator address underflows");
        const auto base = static_cast<std::size_t>(address);
        if (base > std::numeric_limits<std::size_t>::max() - rowSpan)
            throw std::overflow_error(
                "MXM COMPUTE_3D accumulator row address overflows");
        MxmControlInstruction::check_accumulator_address(base + rowSpan);
    }
    for (const auto mode : {instruction.regular_mode,
             instruction.terminal_mode})
        static_cast<void>(MxmControlInstruction::Compute(
            instruction.weight_buffer_base,
            instruction.activation_stream_base,
            instruction.result_stream_base,
            instruction.accumulator_address_base,
            instruction.accumulator_row_stride,
            mode.accumulator_destination,
            instruction.data_format,
            mode.accumulator_clear,
            mode.accumulator_output_format));
}

inline MxmControlInstruction expand_mxm_compute_icu_instruction(
    const MxmComputeIcuInstruction& instruction,
    const IcuCoordinate3D& coordinate)
{
    const auto address = checked_icu_3d_operand(
        instruction.accumulator_address_base,
        instruction.accumulator_address_strides,
        coordinate,
        "MXM COMPUTE_3D accumulator address");
    if (address < 0)
        throw std::out_of_range(
            "MXM COMPUTE_3D accumulator address underflows");
    if (instruction.opcode == MxmComputeIcuOpcode::AccumulatorRead3D)
        return MxmControlInstruction::AccumulatorRead(
            static_cast<std::size_t>(address),
            instruction.result_stream_base,
            instruction.regular_mode.accumulator_clear,
            instruction.regular_mode.accumulator_output_format,
            instruction.regular_mode.accumulator_destination);
    if (instruction.opcode
        == MxmComputeIcuOpcode::DecodeStreamCompute3D)
        return MxmControlInstruction::DecodeStreamCompute(
            mxm_icu_buffer_3d(instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate),
            instruction.result_stream_base,
            instruction.data_format,
            static_cast<std::size_t>(address),
            instruction.accumulator_column,
            instruction.regular_mode.accumulator_destination,
            instruction.regular_mode.accumulator_clear,
            instruction.decode_layout);
    if (instruction.opcode != MxmComputeIcuOpcode::Compute3D)
        throw std::logic_error("MXM compute ICU 3-D opcode is invalid");
    const auto mode = mxm_compute_mode_3d(instruction, coordinate);
    return MxmControlInstruction::Compute(
        mxm_icu_buffer_3d(instruction.weight_buffer_base,
            instruction.weight_buffer_mode, coordinate),
        instruction.activation_stream_base,
        instruction.result_stream_base,
        static_cast<std::size_t>(address),
        instruction.accumulator_row_stride,
        mode.accumulator_destination,
        instruction.data_format,
        mode.accumulator_clear,
        mode.accumulator_output_format);
}

} // namespace detail

} // namespace ftlpu
