#pragma once

#include "ftlpu/c2c/icu_instruction.hpp"
#include "ftlpu/icu/fu_3d_codec.hpp"
#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/legacy_fu_3d_adapter.hpp"
#include "ftlpu/icu/sxm_run_2d.hpp"
#include "ftlpu/icu/vxm_run_2d.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/sxm/instruction.hpp"
#include "ftlpu/vxm/compact_instruction.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ftlpu {

template <typename FuncInstruction>
struct IcuMacroInstruction {
    IcuMacroSchedule schedule{};
    FuncInstruction instruction{};
};

template <typename FuncInstruction>
struct IcuStreamNdInstruction {
    IcuStreamNdSchedule schedule{};
    FuncInstruction instruction{};
};

template <typename FuncInstruction>
struct IcuStreamProgramBodyEntry {
    std::size_t cycle_offset{0};
    std::array<std::int64_t, IcuStreamNdSchedule::kMaxRank>
        operand_strides{0, 0, 0};
    FuncInstruction instruction{};
};

// One queue-local launch domain shared by a short functional-unit program.
// Each body entry has its own issue offset and affine operand induction.
template <typename FuncInstruction>
struct IcuStreamProgramInstruction {
    static constexpr std::size_t kMaxBodyEntries = 16;

    IcuStreamNdSchedule schedule{};
    std::vector<IcuStreamProgramBodyEntry<FuncInstruction>> body{};
};

using IcuMemSliceProgramEntry = IcuStreamProgramBodyEntry<MemInstruction>;
using IcuMemSliceProgram = IcuStreamProgramInstruction<MemInstruction>;

template <typename FuncInstruction>
struct IcuSynchronizedInstruction {
    std::size_t count{0};
    std::size_t synchronization_tag{0};
    std::size_t transport_delay{0};
    // Minimum number of ICU clock cycles for which this command owns the
    // queue after activation. Early C2C completion waits out the reservation;
    // late completion extends it. This keeps the following static schedule
    // aligned without a second MEM command stream.
    std::size_t reservation_cycles{1};
    std::int64_t address_stride{0};
    FuncInstruction instruction{};
};

// The native instruction type alone cannot distinguish the two MXM control
// endpoints: LOAD_3D and COMPUTE_3D both expand to MxmControlInstruction.
// Keep that physical queue identity in the queue type so an i-MEM word cannot
// contain a coarse instruction owned by another functional unit.
enum class IcuQueueRole : std::uint8_t {
    // Software-only compatibility executor for historical absolute schedules.
    Legacy,
    Mem,
    MxmLoad,
    MxmDequant,
    MxmCompute,
    Vxm,
    Sxm,
    C2cTx,
    C2cRx,
    C2cDma,
};

constexpr bool is_fu_loop_queue_role(IcuQueueRole role) noexcept
{
    return role == IcuQueueRole::Mem
        || role == IcuQueueRole::MxmLoad
        || role == IcuQueueRole::MxmDequant
        || role == IcuQueueRole::MxmCompute
        || role == IcuQueueRole::Vxm
        || role == IcuQueueRole::Sxm;
}

template <typename FuncInstruction, typename... ThreeDInstructions>
using IqEntryWith3D = std::variant<IcuControlInstruction, FuncInstruction,
    IcuMacroInstruction<FuncInstruction>,
    IcuStreamNdInstruction<FuncInstruction>,
    IcuStreamProgramInstruction<FuncInstruction>,
    IcuSynchronizedInstruction<FuncInstruction>,
    ThreeDInstructions...>;

template <typename FuncInstruction, IcuQueueRole Role>
struct IqEntryForRole;

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::Legacy> {
    using Type = IqEntryWith3D<FuncInstruction>;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::Mem> {
    using Type = isa::EncodedMemIcu3DWord;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::MxmLoad> {
    using Type = isa::EncodedMxmLoadIcu3DWord;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::MxmDequant> {
    using Type = isa::EncodedMxmDequantIcu3DWord;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::MxmCompute> {
    using Type = isa::EncodedMxmComputeIcu3DWord;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::Vxm> {
    using Type = isa::EncodedVxmIcuRun2DWord;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::Sxm> {
    using Type = isa::EncodedSxmIcuRun2DWord;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::C2cTx> {
    using Type = C2cEndpointIcuPacket;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::C2cRx> {
    using Type = C2cEndpointIcuPacket;
};

template <typename FuncInstruction>
struct IqEntryForRole<FuncInstruction, IcuQueueRole::C2cDma> {
    using Type = C2cEndpointIcuPacket;
};

template <typename FuncInstruction,
    IcuQueueRole Role = IcuQueueRole::Legacy>
using IqEntry = typename IqEntryForRole<FuncInstruction, Role>::Type;

template <IcuQueueRole Role>
struct Icu3DInstructionForRole;

struct NoIcu3DInstruction {};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::Legacy> {
    using Type = NoIcu3DInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::Mem> {
    using Type = MemIcuInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::MxmLoad> {
    using Type = MxmLoadIcuInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::MxmDequant> {
    using Type = MxmDequantIcuInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::MxmCompute> {
    using Type = MxmComputeIcuInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::Vxm> {
    using Type = VxmIcuRun2DInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::Sxm> {
    using Type = SxmIcuRun2DInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::C2cTx> {
    using Type = NoIcu3DInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::C2cRx> {
    using Type = NoIcu3DInstruction;
};

template <>
struct Icu3DInstructionForRole<IcuQueueRole::C2cDma> {
    using Type = NoIcu3DInstruction;
};

struct NoEncodedIcu3DWord {};

struct NoEncodedIcu3DPacket {
    static constexpr std::size_t kWordBits = 0;
    static constexpr std::size_t kWordCount = 0;
};

struct EncodedC2cEndpointIcuPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 1;
    std::array<C2cEndpointIcuPacket, kWordCount> words{};
};

// A synchronized MEM write is a physical pair of 96-bit i-MEM words: the
// synchronization header followed by one native MEM instruction word.
struct EncodedMemIcuSynchronizedPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 2;
    std::array<isa::EncodedMemIcu3DWord, kWordCount> words{};
};

static_assert(sizeof(EncodedMemIcuSynchronizedPacket) == 24,
    "a synchronized MEM packet must occupy exactly two 96-bit words");

template <IcuQueueRole Role>
struct Icu3DEncodingForRole;

template <>
struct Icu3DEncodingForRole<IcuQueueRole::Legacy> {
    using Word = NoEncodedIcu3DWord;
    using Packet = NoEncodedIcu3DPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::Mem> {
    using Word = isa::EncodedMemIcu3DWord;
    using Packet = isa::EncodedMemIcu3DPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::MxmLoad> {
    using Word = isa::EncodedMxmLoadIcu3DWord;
    using Packet = isa::EncodedMxmLoadIcu3DPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::MxmDequant> {
    using Word = isa::EncodedMxmDequantIcu3DWord;
    using Packet = isa::EncodedMxmDequantIcu3DPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::MxmCompute> {
    using Word = isa::EncodedMxmComputeIcu3DWord;
    using Packet = isa::EncodedMxmComputeIcu3DPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::Vxm> {
    using Word = isa::EncodedVxmIcuRun2DWord;
    using Packet = isa::EncodedVxmIcuRun2DPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::Sxm> {
    using Word = isa::EncodedSxmIcuRun2DWord;
    using Packet = isa::EncodedSxmIcuRun2DPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::C2cTx> {
    using Word = C2cEndpointIcuPacket;
    using Packet = EncodedC2cEndpointIcuPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::C2cRx> {
    using Word = C2cEndpointIcuPacket;
    using Packet = EncodedC2cEndpointIcuPacket;
};

template <>
struct Icu3DEncodingForRole<IcuQueueRole::C2cDma> {
    using Word = C2cEndpointIcuPacket;
    using Packet = C2cDmaIcuPacket;
};

template <IcuQueueRole Role>
struct Icu3DContextForRole;

template <>
struct Icu3DContextForRole<IcuQueueRole::Legacy> {
    static constexpr std::size_t depth = 0;
    static constexpr std::size_t bits = 0;
    static constexpr std::size_t minimum_bits = 0;
};

template <>
struct Icu3DContextForRole<IcuQueueRole::Mem> {
    static constexpr std::size_t depth = hw::kIcuMem3DContextDepth;
    static constexpr std::size_t bits = hw::kIcuMem3DContextBits;
    // WRITE_READ_2D worst case: 247 static bits, four 16-bit counters,
    // two completion flags, a 24-bit relative cycle, and a 16-bit i-MEM PC.
    static constexpr std::size_t minimum_bits = 353;
};

template <>
struct Icu3DContextForRole<IcuQueueRole::MxmLoad> {
    static constexpr std::size_t depth = hw::kIcuMxmLoad3DContextDepth;
    static constexpr std::size_t bits = hw::kIcuMxmLoad3DContextBits;
    static constexpr std::size_t minimum_bits = 268;
};

template <>
struct Icu3DContextForRole<IcuQueueRole::MxmDequant> {
    static constexpr std::size_t depth = hw::kIcuMxmDequant3DContextDepth;
    static constexpr std::size_t bits = hw::kIcuMxmDequant3DContextBits;
    static constexpr std::size_t minimum_bits = 224;
};

template <>
struct Icu3DContextForRole<IcuQueueRole::MxmCompute> {
    static constexpr std::size_t depth = hw::kIcuMxmCompute3DContextDepth;
    static constexpr std::size_t bits = hw::kIcuMxmCompute3DContextBits;
    static constexpr std::size_t minimum_bits = 303;
};

template <>
struct Icu3DContextForRole<IcuQueueRole::Vxm> {
    static constexpr std::size_t depth = hw::kIcuVxmRun2DContextDepth;
    static constexpr std::size_t bits = hw::kIcuVxmRun2DContextBits;
    static constexpr std::size_t minimum_bits = 243;
};

template <>
struct Icu3DContextForRole<IcuQueueRole::Sxm> {
    static constexpr std::size_t depth = hw::kIcuSxmRun2DContextDepth;
    static constexpr std::size_t bits = hw::kIcuSxmRun2DContextBits;
    static constexpr std::size_t minimum_bits = 491;
};

template <>
struct Icu3DContextForRole<IcuQueueRole::C2cTx> {
    static constexpr std::size_t depth = 1;
    static constexpr std::size_t bits = 0;
    static constexpr std::size_t minimum_bits = 0;
};

template <>
struct Icu3DContextForRole<IcuQueueRole::C2cRx>
    : Icu3DContextForRole<IcuQueueRole::C2cTx> {};

template <>
struct Icu3DContextForRole<IcuQueueRole::C2cDma>
    : Icu3DContextForRole<IcuQueueRole::C2cTx> {};

struct IcuProgramDescriptor {
    std::size_t base_pc{0};
    // Counts physical local-iMEM words. A FU 3-D launch occupies a fixed
    // multiword packet even though it is one logical coarse instruction.
    std::size_t instruction_count{0};
};

enum class IcuQueueAction : std::uint8_t {
    Idle,
    WaitingForStart,
    PrefetchOnly,
    FunctionalIssue,
    Nop,
    NopWait,
    RepeatIssue,
    RepeatWait,
    Repeat2DIssue,
    Repeat2DWait,
    MacroIssue,
    MacroWait,
    MemStreamNdIssue,
    MemStreamNdWait,
    MemSliceProgramIssue,
    MemSliceProgramWait,
    MxmStreamNdIssue,
    MxmStreamNdWait,
    VxmStreamNdIssue,
    VxmStreamNdWait,
    VxmRun2DIssue,
    VxmRun2DWait,
    SxmRun2DIssue,
    SxmRun2DWait,
    SxmTileProgramIssue,
    SxmTileProgramWait,
    Mem3DIssue,
    Mem3DWait,
    MemWriteRead2DIssue,
    MemWriteRead2DWait,
    MxmLoad3DIssue,
    MxmLoad3DWait,
    MxmDequant3DIssue,
    MxmDequant3DWait,
    MxmCompute3DIssue,
    MxmCompute3DWait,
    ThreeDDecodeWait,
    ThreeDContextFullWait,
    ProgramPaused,
    SynchronizedWait,
    SynchronizedDelay,
    SynchronizedIssue,
    SyncWait,
    SyncRelease,
    EventWait,
    EventRelease,
    Notify,
    Underflow,
};

struct IcuQueueCycleTrace {
    std::size_t cycle{0};
    std::optional<std::size_t> fetch_started_pc{};
    std::optional<std::size_t> fetch_completed_pc{};
    std::optional<std::size_t> issue_pc{};
    std::optional<std::size_t> three_d_decode_header_pc{};
    std::size_t three_d_decode_word_count{0};
    std::size_t iq_before{0};
    std::size_t iq_after{0};
    IcuQueueAction action{IcuQueueAction::Idle};
};

namespace detail {

template <typename RawWord>
constexpr std::size_t raw_icu_word_bits = sizeof(RawWord) * 8;

template <typename RawWord>
void write_raw_icu_bits(RawWord& word, std::size_t offset,
    unsigned width, std::uint64_t value)
{
    if (width > 64 || offset > raw_icu_word_bits<RawWord>
        || width > raw_icu_word_bits<RawWord> - offset)
        throw std::logic_error("ICU raw-word bit write is out of range");
    for (unsigned bit = 0; bit < width; ++bit) {
        if (((value >> bit) & 1U) != 0) {
            const auto position = offset + bit;
            word.lanes[position / 32]
                |= std::uint32_t {1} << (position % 32);
        }
    }
}

template <typename RawWord>
std::uint64_t read_raw_icu_bits(const RawWord& word,
    std::size_t offset, unsigned width)
{
    if (width > 64 || offset > raw_icu_word_bits<RawWord>
        || width > raw_icu_word_bits<RawWord> - offset)
        throw std::logic_error("ICU raw-word bit read is out of range");
    std::uint64_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const auto position = offset + bit;
        if ((word.lanes[position / 32]
                & (std::uint32_t {1} << (position % 32))) != 0)
            value |= std::uint64_t {1} << bit;
    }
    return value;
}

template <typename RawWord>
bool raw_icu_bits_are_zero(const RawWord& word,
    std::size_t begin, std::size_t end)
{
    if (begin > end || end > raw_icu_word_bits<RawWord>)
        throw std::logic_error("ICU raw-word zero check is out of range");
    for (auto bit = begin; bit < end; ++bit) {
        if ((word.lanes[bit / 32]
                & (std::uint32_t {1} << (bit % 32))) != 0)
            return false;
    }
    return true;
}

template <typename RawWord>
isa::IcuCommandOpcode raw_icu_opcode(const RawWord& word) noexcept
{
    return static_cast<isa::IcuCommandOpcode>(word.lanes[0] & 3U);
}

template <typename RawWord>
std::uint8_t raw_icu_extended_subtype(const RawWord& word)
{
    return static_cast<std::uint8_t>(
        read_raw_icu_bits(word, 88, 4));
}

template <typename FuncInstruction>
FuncInstruction apply_icu_repeat_stride(
    FuncInstruction instruction,
    std::int64_t,
    std::size_t)
{
    return instruction;
}

inline MemInstruction apply_icu_repeat_stride(
    MemInstruction instruction,
    std::int64_t address_stride,
    std::size_t repeat_index)
{
    const auto delta = address_stride * static_cast<std::int64_t>(repeat_index);
    const auto address = static_cast<std::int64_t>(instruction.address);
    if (delta < 0 && address < -delta) {
        throw std::out_of_range("ICU MEM Repeat address stride underflow");
    }
    instruction.address = static_cast<std::size_t>(address + delta);
    return instruction;
}

inline MxmControlInstruction apply_icu_repeat_stride(
    MxmControlInstruction instruction,
    std::int64_t address_stride,
    std::size_t repeat_index)
{
    if (address_stride == 0) return instruction;
    const auto delta = address_stride
        * static_cast<std::int64_t>(repeat_index);
    if (instruction.opcode == MxmControlOpcode::Compute
        || instruction.opcode == MxmControlOpcode::AccumulatorRead) {
        const auto address = static_cast<std::int64_t>(
            instruction.accumulator_address) + delta;
        if (address < 0)
            throw std::out_of_range(
                "ICU MXM Repeat accumulator-address stride underflow");
        MxmControlInstruction::check_accumulator_address(
            static_cast<std::size_t>(address));
        instruction.accumulator_address = static_cast<std::size_t>(address);
        return instruction;
    }
    if (instruction.opcode == MxmControlOpcode::IW) {
        const auto column = static_cast<std::int64_t>(
            instruction.weight_column) + delta;
        if (column < 0
            || column >= static_cast<std::int64_t>(hw::kMxmColumns))
            throw std::out_of_range(
                "ICU MXM Repeat weight-column stride is outside the MXM");
        instruction.weight_column = static_cast<std::size_t>(column);
        return instruction;
    }
    throw std::invalid_argument(
        "ICU MXM Repeat stride requires compute, accumulator-read, or IW");
}

template <typename FuncInstruction>
FuncInstruction apply_icu_repeat_2d_stride(
    FuncInstruction instruction,
    IcuInductionTarget target,
    std::int64_t delta)
{
    if (target != IcuInductionTarget::None || delta != 0)
        throw std::invalid_argument(
            "ICU Repeat2D induction target is invalid for this queue");
    return instruction;
}

inline MemInstruction apply_icu_repeat_2d_stride(
    MemInstruction instruction,
    IcuInductionTarget target,
    std::int64_t delta)
{
    if (target == IcuInductionTarget::None) {
        if (delta != 0)
            throw std::invalid_argument(
                "ICU Repeat2D has a stride without a MEM induction target");
        return instruction;
    }
    if (target != IcuInductionTarget::MemAddress)
        throw std::invalid_argument(
            "ICU Repeat2D induction target is invalid for a MEM queue");
    return apply_icu_repeat_stride(instruction, delta, 1);
}

inline MxmControlInstruction apply_icu_repeat_2d_stride(
    MxmControlInstruction instruction,
    IcuInductionTarget target,
    std::int64_t delta)
{
    if (target == IcuInductionTarget::None) {
        if (delta != 0)
            throw std::invalid_argument(
                "ICU Repeat2D has a stride without an MXM induction target");
        return instruction;
    }
    if (target == IcuInductionTarget::MxmWeightColumn) {
        if (instruction.opcode != MxmControlOpcode::IW)
            throw std::invalid_argument(
                "ICU Repeat2D weight-column induction requires an MXM IW instruction");
        const auto column =
            static_cast<std::int64_t>(instruction.weight_column) + delta;
        if (column < 0
            || column >= static_cast<std::int64_t>(hw::kMxmColumns))
            throw std::out_of_range(
                "ICU Repeat2D MXM weight-column induction is outside the MXM");
        instruction.weight_column = static_cast<std::size_t>(column);
        return instruction;
    }
    if (target == IcuInductionTarget::MxmAccumulatorAddress) {
        if (instruction.opcode != MxmControlOpcode::Compute
            && instruction.opcode != MxmControlOpcode::AccumulatorRead)
            throw std::invalid_argument(
                "ICU Repeat2D accumulator induction requires an MXM compute or accumulator-read instruction");
        const auto address =
            static_cast<std::int64_t>(instruction.accumulator_address)
            + delta;
        if (address < 0)
            throw std::out_of_range(
                "ICU Repeat2D MXM accumulator-address induction underflow");
        MxmControlInstruction::check_accumulator_address(
            static_cast<std::size_t>(address));
        instruction.accumulator_address = static_cast<std::size_t>(address);
        return instruction;
    }
    throw std::invalid_argument(
        "ICU Repeat2D induction target is invalid for an MXM queue");
}

} // namespace detail

// One independently scheduled ICU endpoint. The instruction memory is local
// to this queue, so runtime fetches never consume MEM capacity or data Stream
// Register bandwidth. InstructionBits, ImemDepth and IqDepth are architectural
// parameters and may differ between function types.
template <
    typename FuncInstruction,
    std::size_t InstructionBits,
    std::size_t ImemDepth,
    std::size_t IqDepth,
    std::size_t FetchLatency = 1,
    std::size_t MacroContextDepth = IqDepth,
    IcuQueueRole QueueRole = IcuQueueRole::Legacy>
class DistributedIcuQueue {
public:
    using FunctionalInstruction = FuncInstruction;
    using Entry = IqEntry<FuncInstruction, QueueRole>;
    using Encoded3DWord =
        typename Icu3DEncodingForRole<QueueRole>::Word;
    using Encoded3DPacket =
        typename Icu3DEncodingForRole<QueueRole>::Packet;

    static_assert(
        InstructionBits >= 32,
        "every distributed ICU word must hold a 32-bit ICU control command");
    static_assert(ImemDepth > 0);
    static_assert(IqDepth > 0);
    static_assert(FetchLatency > 0);
    static_assert(MacroContextDepth > 0);
    static_assert(QueueRole == IcuQueueRole::Legacy
            || InstructionBits == Encoded3DPacket::kWordBits,
        "a raw packet word must exactly match its local i-MEM width");
    static_assert(QueueRole == IcuQueueRole::Legacy
            || IqDepth >= Encoded3DPacket::kWordCount,
        "the IQ must hold one complete raw packet");
    static_assert(!is_fu_loop_queue_role(QueueRole)
            || Icu3DContextForRole<QueueRole>::depth > 0,
        "a FU 3-D queue must provide at least one decoded context");
    static_assert(!is_fu_loop_queue_role(QueueRole)
            || Icu3DContextForRole<QueueRole>::bits
                >= Icu3DContextForRole<QueueRole>::minimum_bits,
        "the FU 3-D context bit budget cannot hold its decoded state");
    static_assert(QueueRole == IcuQueueRole::Legacy
            || (QueueRole == IcuQueueRole::Mem
                && std::is_same_v<FuncInstruction, MemInstruction>)
            || (QueueRole == IcuQueueRole::MxmLoad
                && std::is_same_v<FuncInstruction,
                    MxmControlInstruction>)
            || (QueueRole == IcuQueueRole::MxmDequant
                && std::is_same_v<FuncInstruction,
                    MxmDequantInstruction>)
            || (QueueRole == IcuQueueRole::MxmCompute
                && std::is_same_v<FuncInstruction,
                    MxmControlInstruction>)
            || (QueueRole == IcuQueueRole::Vxm
                && std::is_same_v<FuncInstruction,
                    VxmCompactInstruction>)
            || (QueueRole == IcuQueueRole::Sxm
                && std::is_same_v<FuncInstruction, SxmInstruction>)
            || ((QueueRole == IcuQueueRole::C2cTx
                    || QueueRole == IcuQueueRole::C2cRx)
                && std::is_same_v<FuncInstruction, C2cInstruction>)
            || (QueueRole == IcuQueueRole::C2cDma
                && std::is_same_v<FuncInstruction,
                    C2cDmaInstruction>),
        "the ICU queue role does not match its native instruction type");

    static constexpr std::size_t instruction_bits = InstructionBits;
    static constexpr std::size_t imem_depth = ImemDepth;
    static constexpr std::size_t iq_depth = IqDepth;
    static constexpr std::size_t fetch_latency = FetchLatency;
    static constexpr std::size_t macro_context_depth = MacroContextDepth;
    static constexpr std::size_t three_d_context_depth =
        Icu3DContextForRole<QueueRole>::depth;
    static constexpr std::size_t three_d_context_bits =
        Icu3DContextForRole<QueueRole>::bits;
    static constexpr IcuQueueRole queue_role = QueueRole;
    static constexpr std::size_t three_d_packet_word_count =
        Encoded3DPacket::kWordCount;

    void reset()
    {
        imem_.clear();
        reset_execution();
        raw_compatibility_cycle_ = 0;
        raw_compatibility_cycle_known_ = true;
        configured_ = false;
        cycle_ = 0;
    }

    void reset_execution()
    {
        iq_.clear();
        iq_pcs_.clear();
        pending_fetches_.clear();
        last_dispatched_.reset();
        repeat_instruction_.reset();
        repeat_2d_instruction_.reset();
        nop_remaining_ = 0;
        repeat_remaining_ = 0;
        repeat_interval_ = 1;
        repeat_cooldown_ = 0;
        repeat_address_stride_ = 0;
        repeat_index_ = 0;
        repeat_2d_active_ = false;
        repeat_2d_inner_ = 0;
        repeat_2d_outer_ = 0;
        repeat_2d_cooldown_ = 0;
        active_macros_ = {};
        macro_remaining_points_ = 0;
        peak_active_macros_ = 0;
        active_stream_nd_ = {};
        stream_nd_remaining_points_ = 0;
        active_3d_.reset();
        active_write_read_2d_.reset();
        three_d_remaining_points_ = 0;
        write_read_2d_remaining_points_ = 0;
        peak_active_3d_contexts_ = 0;
        synchronized_instruction_.reset();
        synchronized_remaining_ = 0;
        synchronized_index_ = 0;
        synchronized_elapsed_cycles_ = 0;
        synchronized_waiting_ = false;
        synchronized_descriptor_pc_.reset();
        synchronized_issued_count_ = 0;
        synchronized_completed_count_ = 0;
        last_dispatched_pc_.reset();
        notification_tokens_ = 0;
        synchronized_notifications_.clear();
        notify_emitted_ = false;
        fetch_pc_ = 0;
        program_end_pc_ = 0;
        fetched_count_ = 0;
        issued_count_ = 0;
        launched_ = false;
        underflowed_ = false;
    }

    void write_imem(std::size_t address, Entry entry)
    {
        if (launched_) {
            throw StaticScheduleError(
                "ICU local i-MEM cannot be modified after program launch");
        }
        if (address >= ImemDepth) {
            std::ostringstream os;
            os << "ICU i-MEM address " << address
               << " exceeds configured depth " << ImemDepth
               << " for " << InstructionBits << "-bit instructions";
            throw StaticScheduleError(os.str());
        }
        if (imem_.size() <= address) {
            imem_.resize(address + 1);
        }
        imem_[address] = std::move(entry);
    }

    void write_imem(std::size_t address, FuncInstruction instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            write_imem(address, Entry {std::in_place_type<FuncInstruction>,
                                    std::move(instruction)});
        } else {
            write_imem(address, encode_native_word(instruction));
        }
    }

    void write_imem_control(
        std::size_t address, IcuControlInstruction instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            write_imem(address,
                Entry {std::in_place_type<IcuControlInstruction>,
                    instruction});
        } else {
            write_imem(address, encode_control_word(instruction));
        }
    }

    void load_imem(std::size_t base, std::vector<Entry> program)
    {
        if (base > ImemDepth || program.size() > ImemDepth - base) {
            throw StaticScheduleError("ICU program exceeds configured i-MEM depth");
        }
        for (std::size_t index = 0; index < program.size(); ++index) {
            write_imem(base + index, std::move(program[index]));
        }
    }

    // Compatibility/configuration helper: append compiler output to local
    // i-MEM. It does not inject an instruction directly into the runtime IQ.
    void append_program(Entry entry)
    {
        const auto address = imem_.size();
        write_imem(address, std::move(entry));
        configured_ = false;
    }

    void append_program(FuncInstruction instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            append_program(Entry {std::in_place_type<FuncInstruction>,
                std::move(instruction)});
        } else if constexpr (QueueRole == IcuQueueRole::C2cTx
            || QueueRole == IcuQueueRole::C2cRx
            || QueueRole == IcuQueueRole::C2cDma) {
            throw std::logic_error(
                "C2C raw ICU queues require a packed hardware packet");
        } else {
            const auto word = encode_native_word(instruction);
            const auto nextCompatibilityCycle =
                raw_compatibility_cycle_after(1, "native ICU instruction");
            append_program(word);
            commit_raw_compatibility_cycle(nextCompatibilityCycle);
        }
    }

    void append_control(IcuControlInstruction instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            append_program(Entry {std::in_place_type<IcuControlInstruction>,
                instruction});
        } else {
            const auto word = encode_control_word(instruction);
            auto nextCompatibilityCycle = raw_compatibility_cycle_;
            auto nextCompatibilityCycleKnown =
                raw_compatibility_cycle_known_;
            if (nextCompatibilityCycleKnown) {
                switch (instruction.opcode) {
                case IcuControlOpcode::Nop:
                    nextCompatibilityCycle = checked_compatibility_add(
                        nextCompatibilityCycle, instruction.count,
                        "ICU NOP compatibility cursor");
                    break;
                case IcuControlOpcode::Repeat:
                    nextCompatibilityCycle = checked_compatibility_add(
                        nextCompatibilityCycle,
                        checked_compatibility_multiply(instruction.count,
                            instruction.interval,
                            "ICU Repeat compatibility cursor"),
                        "ICU Repeat compatibility cursor");
                    break;
                case IcuControlOpcode::Repeat2D:
                    nextCompatibilityCycle = checked_compatibility_add(
                        nextCompatibilityCycle,
                        repeat_2d_final_offset(instruction.repeat_2d),
                        "ICU Repeat2D compatibility cursor");
                    break;
                case IcuControlOpcode::Notify:
                    nextCompatibilityCycle = checked_compatibility_add(
                        nextCompatibilityCycle, 1,
                        "ICU Notify compatibility cursor");
                    break;
                case IcuControlOpcode::Sync:
                case IcuControlOpcode::WaitEvent:
                    // Their release time depends on an external event. An
                    // absolute legacy launch cannot be translated after one.
                    nextCompatibilityCycleKnown = false;
                    break;
                }
            }
            append_program(word);
            raw_compatibility_cycle_ = nextCompatibilityCycle;
            raw_compatibility_cycle_known_ =
                nextCompatibilityCycleKnown;
        }
    }

    // Compatibility surface used by the existing schedule builders. These
    // calls append to local i-MEM; runtime IQ entries are still produced only
    // by the modeled fetch frontend.
    void push_instruction(FuncInstruction instruction)
    {
        append_program(std::move(instruction));
    }

    void push_nop(std::size_t cycles)
    {
        if (cycles != 0) append_control(IcuControlInstruction::Nop(cycles));
    }

    void push_repeat(IcuRepeat repeat)
    {
        if (repeat.count != 0) {
            append_control(IcuControlInstruction::Repeat(
                repeat.count, repeat.interval, repeat.address_stride));
        }
    }

    void push_repeat_2d(IcuRepeat2D repeat)
    {
        append_control(IcuControlInstruction::Repeat2D(repeat));
    }

    void push_macro(
        IcuMacroSchedule schedule, FuncInstruction instruction)
    {
        [[maybe_unused]] const auto absoluteStartCycle = schedule.start_cycle;
        if constexpr (QueueRole != IcuQueueRole::Legacy)
            schedule.start_cycle = 0;
        if constexpr (QueueRole == IcuQueueRole::Mem) {
            validate_macro_schedule(schedule);
            auto lowered = detail::lower_legacy_mem_to_3d(
                detail::legacy_macro_loop_3d(schedule),
                detail::legacy_macro_strides(schedule),
                schedule.induction_target, instruction);
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy MEM macro");
            push_mem_3d(std::move(lowered));
        } else if constexpr (QueueRole == IcuQueueRole::MxmLoad) {
            validate_macro_schedule(schedule);
            auto lowered = detail::lower_legacy_mxm_load_to_3d(
                detail::legacy_macro_loop_3d(schedule),
                detail::legacy_macro_strides(schedule),
                schedule.induction_target, instruction);
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy MXM load macro");
            push_mxm_load_3d(std::move(lowered));
        } else if constexpr (QueueRole == IcuQueueRole::MxmDequant) {
            validate_macro_schedule(schedule);
            auto lowered = detail::lower_legacy_mxm_dequant_to_3d(
                detail::legacy_macro_loop_3d(schedule),
                detail::legacy_macro_strides(schedule),
                schedule.induction_target, instruction);
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy MXM dequant macro");
            push_mxm_dequant_3d(std::move(lowered));
        } else if constexpr (QueueRole == IcuQueueRole::MxmCompute) {
            validate_macro_schedule(schedule);
            auto lowered = detail::lower_legacy_mxm_compute_to_3d(
                detail::legacy_macro_loop_3d(schedule),
                detail::legacy_macro_strides(schedule),
                schedule.induction_target, instruction);
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy MXM compute macro");
            push_mxm_compute_3d(std::move(lowered));
        } else if constexpr (QueueRole != IcuQueueRole::Legacy) {
            throw std::invalid_argument(
                "legacy ICU macro entries are not encoded on FU raw-word queues");
        } else {
            validate_macro_schedule(schedule);
            append_program(Entry {std::in_place_type<
                IcuMacroInstruction<FuncInstruction>>,
                IcuMacroInstruction<FuncInstruction> {
                    schedule, std::move(instruction)}});
        }
    }

    void push_mem_stream_nd(IcuMemStreamNdSchedule schedule,
        FuncInstruction instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::Mem) {
            if (schedule.induction_target != IcuInductionTarget::None
                && schedule.induction_target
                    != IcuInductionTarget::MemAddress)
                throw std::invalid_argument(
                    "MEM_STREAM_ND requires MEM-address induction");
            schedule.induction_target = IcuInductionTarget::MemAddress;
            validate_stream_nd_schedule(schedule);
            const auto absoluteStartCycle = schedule.start_cycle;
            schedule.start_cycle = 0;
            auto lowered = detail::lower_legacy_mem_to_3d(
                detail::legacy_stream_nd_loop_3d(schedule),
                detail::legacy_stream_nd_strides(schedule),
                schedule.induction_target, instruction);
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy MEM_STREAM_ND");
            push_mem_3d(std::move(lowered));
        } else if constexpr (QueueRole != IcuQueueRole::Legacy) {
            throw std::invalid_argument(
                "legacy MEM_STREAM_ND is not encoded on FU raw-word queues");
        } else if constexpr (!std::is_same_v<FuncInstruction, MemInstruction>) {
            throw std::invalid_argument(
                "MEM_STREAM_ND is valid only on a MEM ICU queue");
        } else {
            if (schedule.induction_target != IcuInductionTarget::None
                && schedule.induction_target
                    != IcuInductionTarget::MemAddress)
                throw std::invalid_argument(
                    "MEM_STREAM_ND requires MEM-address induction");
            schedule.induction_target = IcuInductionTarget::MemAddress;
            validate_stream_nd_schedule(schedule);
            append_program(Entry {std::in_place_type<
                IcuStreamNdInstruction<FuncInstruction>>,
                IcuStreamNdInstruction<FuncInstruction> {
                    schedule, std::move(instruction)}});
        }
    }

    void push_mem_slice_program(IcuMemSliceProgram program)
    {
        if constexpr (QueueRole != IcuQueueRole::Legacy) {
            throw std::invalid_argument(
                "legacy MEM_SLICE_PROGRAM is not encoded on FU raw-word queues");
        } else if constexpr (!std::is_same_v<FuncInstruction, MemInstruction>) {
            throw std::invalid_argument(
                "MEM_SLICE_PROGRAM is valid only on a MEM ICU queue");
        } else {
            validate_stream_program(program);
            append_program(Entry {std::in_place_type<
                IcuStreamProgramInstruction<FuncInstruction>>,
                std::move(program)});
        }
    }

    void push_mxm_stream_nd(IcuMxmStreamNdSchedule schedule,
        FuncInstruction instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::MxmLoad) {
            validate_stream_nd_schedule(schedule);
            const auto absoluteStartCycle = schedule.start_cycle;
            schedule.start_cycle = 0;
            auto lowered = detail::lower_legacy_mxm_load_to_3d(
                detail::legacy_stream_nd_loop_3d(schedule),
                detail::legacy_stream_nd_strides(schedule),
                schedule.induction_target, instruction);
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy MXM load STREAM_ND");
            push_mxm_load_3d(std::move(lowered));
        } else if constexpr (QueueRole == IcuQueueRole::MxmDequant) {
            validate_stream_nd_schedule(schedule);
            const auto absoluteStartCycle = schedule.start_cycle;
            schedule.start_cycle = 0;
            auto lowered = detail::lower_legacy_mxm_dequant_to_3d(
                detail::legacy_stream_nd_loop_3d(schedule),
                detail::legacy_stream_nd_strides(schedule),
                schedule.induction_target, instruction);
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy MXM dequant STREAM_ND");
            push_mxm_dequant_3d(std::move(lowered));
        } else if constexpr (QueueRole == IcuQueueRole::MxmCompute) {
            validate_stream_nd_schedule(schedule);
            const auto absoluteStartCycle = schedule.start_cycle;
            schedule.start_cycle = 0;
            auto lowered = detail::lower_legacy_mxm_compute_to_3d(
                detail::legacy_stream_nd_loop_3d(schedule),
                detail::legacy_stream_nd_strides(schedule),
                schedule.induction_target, instruction);
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy MXM compute STREAM_ND");
            push_mxm_compute_3d(std::move(lowered));
        } else if constexpr (QueueRole != IcuQueueRole::Legacy) {
            throw std::invalid_argument(
                "legacy MXM_STREAM_ND is not encoded on FU raw-word queues");
        } else if constexpr (!std::is_same_v<FuncInstruction,
                          MxmControlInstruction>
            && !std::is_same_v<FuncInstruction,
                MxmDequantInstruction>) {
            throw std::invalid_argument(
                "MXM_STREAM_ND is valid only on an MXM ICU queue");
        } else {
            validate_stream_nd_schedule(schedule);
            const bool hasOperandInduction = std::any_of(
                schedule.operand_strides.begin(),
                schedule.operand_strides.begin() + schedule.rank,
                [](std::int64_t stride) { return stride != 0; });
            if (!hasOperandInduction)
                schedule.induction_target = IcuInductionTarget::None;
            if constexpr (std::is_same_v<FuncInstruction,
                              MxmDequantInstruction>) {
                if (hasOperandInduction)
                    throw std::invalid_argument(
                        "MXM dequant STREAM_ND cannot induce an operand");
            } else if (hasOperandInduction) {
                if (schedule.induction_target
                        == IcuInductionTarget::MxmWeightColumn
                    && instruction.opcode != MxmControlOpcode::IW)
                    throw std::invalid_argument(
                        "MXM load STREAM_ND induction requires IW");
                if (schedule.induction_target
                        == IcuInductionTarget::MxmAccumulatorAddress
                    && instruction.opcode != MxmControlOpcode::Compute
                    && instruction.opcode
                        != MxmControlOpcode::AccumulatorRead)
                    throw std::invalid_argument(
                        "MXM compute STREAM_ND induction requires compute or accumulator-read");
            }
            append_program(Entry {std::in_place_type<
                IcuStreamNdInstruction<FuncInstruction>>,
                IcuStreamNdInstruction<FuncInstruction> {
                    schedule, std::move(instruction)}});
        }
    }

    void push_mem_3d(MemIcuInstruction instruction)
        requires (std::is_same_v<FuncInstruction, MemInstruction>
            && QueueRole == IcuQueueRole::Mem)
    {
        append_3d_packet(
            isa::encode_mem_icu_3d_instruction(instruction));
    }

    void push_mem_write_read_2d(
        MemIcuWriteRead2DInstruction instruction)
        requires (std::is_same_v<FuncInstruction, MemInstruction>
            && QueueRole == IcuQueueRole::Mem)
    {
        push_encoded_mem_write_read_2d_packet(
            isa::encode_mem_icu_write_read_2d_instruction(instruction));
    }

    void push_encoded_mem_write_read_2d_packet(
        isa::EncodedMemIcuWriteRead2DPacket packet)
        requires (std::is_same_v<FuncInstruction, MemInstruction>
            && QueueRole == IcuQueueRole::Mem)
    {
        const auto instruction =
            isa::decode_mem_icu_write_read_2d_instruction(packet);
        const auto nextCompatibilityCycle = raw_compatibility_cycle_after(
            detail::mem_icu_write_read_2d_last_issue_cycle(instruction) + 1,
            "MEM WRITE_READ_2D compatibility cursor");
        if (launched_)
            throw StaticScheduleError(
                "ICU local i-MEM cannot be modified after program launch");
        constexpr auto wordCount =
            isa::EncodedMemIcuWriteRead2DPacket::kWordCount;
        if (imem_.size() > ImemDepth
            || wordCount > ImemDepth - imem_.size())
            throw StaticScheduleError(
                "MEM WRITE_READ_2D packet exceeds local i-MEM depth");
        const auto base = imem_.size();
        imem_.resize(base + wordCount);
        for (std::size_t word = 0; word < wordCount; ++word)
            imem_[base + word] = packet.words[word];
        configured_ = false;
        commit_raw_compatibility_cycle(nextCompatibilityCycle);
    }

    void push_mxm_load_3d(MxmLoadIcuInstruction instruction)
        requires (std::is_same_v<FuncInstruction, MxmControlInstruction>
            && QueueRole == IcuQueueRole::MxmLoad)
    {
        append_3d_packet(
            isa::encode_mxm_load_icu_3d_instruction(instruction));
    }

    void push_mxm_dequant_3d(MxmDequantIcuInstruction instruction)
        requires (std::is_same_v<FuncInstruction, MxmDequantInstruction>
            && QueueRole == IcuQueueRole::MxmDequant)
    {
        append_3d_packet(
            isa::encode_mxm_dequant_icu_3d_instruction(instruction));
    }

    void push_mxm_compute_3d(MxmComputeIcuInstruction instruction)
        requires (std::is_same_v<FuncInstruction, MxmControlInstruction>
            && QueueRole == IcuQueueRole::MxmCompute)
    {
        append_3d_packet(
            isa::encode_mxm_compute_icu_3d_instruction(instruction));
    }

    void push_vxm_run_2d(VxmIcuRun2DInstruction instruction)
        requires (std::is_same_v<FuncInstruction, VxmCompactInstruction>
            && QueueRole == IcuQueueRole::Vxm)
    {
        append_3d_packet(
            isa::encode_vxm_icu_run_2d_instruction(instruction));
    }

    void push_sxm_run_2d(SxmIcuRun2DInstruction instruction)
        requires (std::is_same_v<FuncInstruction, SxmInstruction>
            && QueueRole == IcuQueueRole::Sxm)
    {
        append_3d_packet(
            isa::encode_sxm_icu_run_2d_instruction(instruction));
    }

    // Binary loaders already own hardware-formatted FU packets. Preserve
    // those bits and append the whole packet atomically so an i-MEM capacity
    // failure can never leave a truncated descriptor behind.
    void push_encoded_3d_packet(Encoded3DPacket packet)
        requires (is_fu_loop_queue_role(QueueRole))
    {
        append_3d_packet(packet);
    }

    void push_encoded_c2c_endpoint_packet(
        const C2cEndpointIcuPacket& packet)
        requires (QueueRole == IcuQueueRole::C2cTx
            || QueueRole == IcuQueueRole::C2cRx)
    {
        if constexpr (QueueRole == IcuQueueRole::C2cTx)
            (void)C2cIcuPacketCodec::decode_tx(packet);
        else
            (void)C2cIcuPacketCodec::decode_rx(packet);
        const auto nextCompatibilityCycle =
            raw_compatibility_cycle_after(1, "C2C endpoint instruction");
        append_program(packet);
        commit_raw_compatibility_cycle(nextCompatibilityCycle);
    }

    void push_encoded_c2c_dma_packet(
        const C2cDmaIcuPacket& packet)
        requires (QueueRole == IcuQueueRole::C2cDma)
    {
        (void)C2cIcuPacketCodec::decode_dma(packet);
        if (imem_.size() > ImemDepth
            || packet.words.size() > ImemDepth - imem_.size())
            throw StaticScheduleError(
                "C2C DMA packet exceeds configured local i-MEM depth");
        const auto nextCompatibilityCycle =
            raw_compatibility_cycle_after(1, "C2C DMA instruction");
        for (const auto& word : packet.words)
            append_program(word);
        commit_raw_compatibility_cycle(nextCompatibilityCycle);
    }

    // Loader-facing physical-word ingress. The caller presents a header
    // followed immediately by its continuation word. Every call writes one
    // real 96-bit local-iMEM slot.
    void push_encoded_c2c_dma_word(
        const C2cEndpointIcuPacket& word)
        requires (QueueRole == IcuQueueRole::C2cDma)
    {
        append_program(word);
    }

    void push_vxm_stream_nd(IcuVxmStreamNdSchedule schedule,
        FuncInstruction instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::Vxm) {
            validate_stream_nd_schedule(schedule);
            const bool hasOperandInduction = std::any_of(
                schedule.operand_strides.begin(),
                schedule.operand_strides.begin() + schedule.rank,
                [](std::int64_t stride) { return stride != 0; });
            if (hasOperandInduction
                || schedule.induction_target != IcuInductionTarget::None)
                throw std::invalid_argument(
                    "VXM_STREAM_ND cannot induce a compact-config operand");
            if (schedule.rank > 2)
                throw std::invalid_argument(
                    "VXM RUN_2D cannot encode a rank-3 launch domain");
            const auto absoluteStartCycle = schedule.start_cycle;
            schedule.start_cycle = 0;
            const auto secondCount = schedule.rank == 2
                ? schedule.counts[1] : std::size_t {1};
            const auto secondStride = schedule.rank == 2
                ? schedule.cycle_strides[1] : std::size_t {1};
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy VXM_STREAM_ND");
            push_vxm_run_2d(VxmIcuRun2DInstruction::Run2D(
                0,
                {schedule.counts[0], secondCount},
                {schedule.cycle_strides[0], secondStride},
                std::move(instruction)));
        } else if constexpr (QueueRole != IcuQueueRole::Legacy) {
            throw std::invalid_argument(
                "legacy VXM_STREAM_ND is not encoded on FU raw-word queues");
        } else if constexpr (!std::is_same_v<FuncInstruction,
                          VxmCompactInstruction>) {
            throw std::invalid_argument(
                "VXM_STREAM_ND is valid only on a VXM ICU queue");
        } else {
            validate_stream_nd_schedule(schedule);
            const bool hasOperandInduction = std::any_of(
                schedule.operand_strides.begin(),
                schedule.operand_strides.begin() + schedule.rank,
                [](std::int64_t stride) { return stride != 0; });
            if (hasOperandInduction
                || schedule.induction_target != IcuInductionTarget::None)
                throw std::invalid_argument(
                    "VXM_STREAM_ND cannot induce a compact-config operand");
            append_program(Entry {std::in_place_type<
                IcuStreamNdInstruction<FuncInstruction>>,
                IcuStreamNdInstruction<FuncInstruction> {
                    schedule, std::move(instruction)}});
        }
    }

    void push_sxm_tile_program(IcuSxmTileProgramSchedule schedule,
        FuncInstruction instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::Sxm) {
            validate_stream_nd_schedule(schedule);
            const bool hasOperandInduction = std::any_of(
                schedule.operand_strides.begin(),
                schedule.operand_strides.begin() + schedule.rank,
                [](std::int64_t stride) { return stride != 0; });
            if (hasOperandInduction
                || schedule.induction_target != IcuInductionTarget::None)
                throw std::invalid_argument(
                    "SXM_TILE_PROGRAM cannot induce an instruction field");
            if (schedule.rank > 2)
                throw std::invalid_argument(
                    "SXM RUN_2D cannot encode a rank-3 launch domain");
            const auto absoluteStartCycle = schedule.start_cycle;
            schedule.start_cycle = 0;
            const auto secondCount = schedule.rank == 2
                ? schedule.counts[1] : std::size_t {1};
            const auto secondStride = schedule.rank == 2
                ? schedule.cycle_strides[1] : std::size_t {1};
            append_raw_compatibility_start_nop(
                absoluteStartCycle, "legacy SXM_TILE_PROGRAM");
            push_sxm_run_2d(SxmIcuRun2DInstruction::Run2D(
                0,
                {schedule.counts[0], secondCount},
                {schedule.cycle_strides[0], secondStride},
                std::move(instruction)));
        } else if constexpr (QueueRole != IcuQueueRole::Legacy) {
            throw std::invalid_argument(
                "legacy SXM_TILE_PROGRAM is not encoded on FU raw-word queues");
        } else if constexpr (!std::is_same_v<FuncInstruction, SxmInstruction>) {
            throw std::invalid_argument(
                "SXM_TILE_PROGRAM is valid only on an SXM ICU queue");
        } else {
            validate_stream_nd_schedule(schedule);
            const bool hasOperandInduction = std::any_of(
                schedule.operand_strides.begin(),
                schedule.operand_strides.begin() + schedule.rank,
                [](std::int64_t stride) { return stride != 0; });
            if (hasOperandInduction
                || schedule.induction_target != IcuInductionTarget::None)
                throw std::invalid_argument(
                    "SXM_TILE_PROGRAM cannot induce an instruction field");
            append_program(Entry {std::in_place_type<
                IcuStreamNdInstruction<FuncInstruction>>,
                IcuStreamNdInstruction<FuncInstruction> {
                    schedule, std::move(instruction)}});
        }
    }

    void push_synchronized(std::size_t count,
        std::size_t synchronization_tag, std::size_t transport_delay,
        std::int64_t address_stride,
        FuncInstruction instruction,
        std::size_t reservation_cycles = 1)
    {
        if (count == 0)
            throw std::invalid_argument(
                "ICU synchronized instruction count must be non-zero");
        if constexpr (QueueRole != IcuQueueRole::Legacy) {
            append_synchronized_packet(count, synchronization_tag,
                transport_delay, reservation_cycles, address_stride,
                instruction);
            raw_compatibility_cycle_known_ = false;
        } else {
            append_program(Entry {std::in_place_type<
                IcuSynchronizedInstruction<FuncInstruction>>,
                IcuSynchronizedInstruction<FuncInstruction> {count,
                    synchronization_tag, transport_delay, reservation_cycles,
                    address_stride, std::move(instruction)}});
        }
    }

    std::optional<FuncInstruction> dispatch_next()
    {
        return tick();
    }
    void configure(IcuProgramDescriptor descriptor)
    {
        if (descriptor.base_pc > ImemDepth
            || descriptor.instruction_count
                > ImemDepth - descriptor.base_pc
            || descriptor.base_pc + descriptor.instruction_count
                > imem_.size()) {
            throw StaticScheduleError(
                "ICU program descriptor exceeds initialized local i-MEM");
        }
        reset_execution();
        fetch_pc_ = descriptor.base_pc;
        program_end_pc_ = descriptor.base_pc + descriptor.instruction_count;
        configured_ = true;
        launched_ = true;
    }

    void configure_all()
    {
        configure(IcuProgramDescriptor {0, imem_.size()});
    }

    // Represents compiler/loader controlled pre-launch time. Each call is one
    // frontend cycle: a local i-MEM read completes after FetchLatency cycles.
    void prefetch_only()
    {
        ensure_configured();
        begin_trace(IcuQueueAction::PrefetchOnly);
        commit_ready_fetch();
        begin_fetch_if_possible();
        age_pending_fetches();
        finish_trace();
        age_synchronized_notifications();
        ++cycle_;
    }

    std::optional<FuncInstruction> tick()
    {
        ensure_configured();
        notify_emitted_ = false;
        begin_trace(IcuQueueAction::Idle);
        commit_ready_fetch();

        auto result = dispatch_ready_entry();

        begin_fetch_if_possible();
        age_pending_fetches();
        finish_trace();
        age_synchronized_notifications();
        ++cycle_;
        return result;
    }

    // While the executable program is held for a runtime dependency, this mode
    // advances the fetch frontend and relative notification ages, and permits
    // only an in-order MEM_WRITE_SYNC already at the queue head to make
    // progress. Ordinary program state and logical time do not advance, and
    // the synchronized packet never bypasses an earlier command in the same
    // i-MEM stream.
    std::optional<FuncInstruction> tick_transport_only()
    {
        if constexpr (QueueRole != IcuQueueRole::Mem) {
            throw std::logic_error(
                "transport-only ticking is defined only for MEM ICU queues");
        } else {
            ensure_configured();
            notify_emitted_ = false;
            begin_trace(IcuQueueAction::ProgramPaused);
            commit_ready_fetch();

            auto result = std::optional<FuncInstruction> {};
            if (synchronized_instruction_.has_value()
                || (!iq_.empty()
                    && is_synchronized_entry(iq_.front()))) {
                result = tick_synchronized();
            }

            begin_fetch_if_possible();
            age_pending_fetches();
            finish_trace();
            age_synchronized_notifications();
            return result;
        }
    }

    std::optional<FuncInstruction> dispatch()
    {
        return tick();
    }

    void notify()
    {
        ++notification_tokens_;
    }

    void notify(std::size_t synchronization_tag)
    {
        synchronized_notifications_.push_back(
            TaggedNotification {synchronization_tag, 0});
    }

    bool take_notify()
    {
        const auto emitted = notify_emitted_;
        notify_emitted_ = false;
        return emitted;
    }

    bool blocked_on_sync() const
    {
        if (nop_remaining_ != 0 || repeat_remaining_ != 0
            || repeat_2d_active_
            || !active_macros_.empty()
            || !active_stream_nd_.empty()
            || active_3d_.has_value()
            || active_write_read_2d_.has_value()) {
            return false;
        }
        auto lacksTaggedNotification = [&](std::size_t tag) {
            return std::none_of(synchronized_notifications_.begin(),
                synchronized_notifications_.end(),
                [&](const TaggedNotification& notification) {
                    return notification.tag == tag;
                });
        };
        if (synchronized_instruction_.has_value()) {
            return synchronized_waiting_
                && lacksTaggedNotification(
                    synchronized_instruction_->synchronization_tag);
        }
        if (iq_.empty()) return false;
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            if (const auto* descriptor = std::get_if<
                    IcuSynchronizedInstruction<FuncInstruction>>(
                        &iq_.front()))
                return lacksTaggedNotification(
                    descriptor->synchronization_tag);
            const auto* control =
                std::get_if<IcuControlInstruction>(&iq_.front());
            if (control == nullptr) return false;
            if (control->opcode == IcuControlOpcode::Sync)
                return notification_tokens_ == 0;
            if (control->opcode != IcuControlOpcode::WaitEvent)
                return false;
            return lacksTaggedNotification(control->event_tag);
        } else {
            if (detail::raw_icu_opcode(iq_.front())
                != isa::IcuCommandOpcode::Extended)
                return false;
            const auto subtype =
                detail::raw_icu_extended_subtype(iq_.front());
            if (subtype == kSyncExtendedSubtype)
                return notification_tokens_ == 0;
            if (subtype == kSynchronizedExtendedSubtype)
                return lacksTaggedNotification(
                    decode_synchronized_header(iq_.front())
                        .synchronization_tag);
            if (subtype != kWaitEventExtendedSubtype)
                return false;
            const auto control = decode_control_word(iq_.front());
            return lacksTaggedNotification(control.event_tag);
        }
    }

    // Raw i-MEM surface for pager clients. A packet is appended as its two
    // physical words after validating both the synchronized header and the
    // native word. It is intentionally available only on the MEM queue.
    using EncodedSynchronizedPacket = std::array<Entry, 2>;
    static constexpr std::size_t synchronized_packet_word_count = 2;

    static Entry encode_control_raw_word(
        const IcuControlInstruction& instruction)
        requires (QueueRole != IcuQueueRole::Legacy)
    {
        return encode_control_word(instruction);
    }

    static IcuControlInstruction decode_control_raw_word(
        const Entry& word)
        requires (QueueRole != IcuQueueRole::Legacy)
    {
        return decode_control_word(word);
    }

    static EncodedSynchronizedPacket encode_synchronized_raw_packet(
        std::size_t count, std::size_t synchronization_tag,
        std::size_t transport_delay, std::int64_t address_stride,
        const FuncInstruction& instruction,
        std::size_t reservation_cycles = 1)
    {
        if constexpr (QueueRole != IcuQueueRole::Mem) {
            throw std::logic_error(
                "raw synchronized packets are only defined for MEM ICU queues");
        } else {
            if (instruction.opcode != MemOpcode::Write)
                throw std::invalid_argument(
                    "MEM synchronized packet must carry a write template");
            return {encode_synchronized_header(count, synchronization_tag,
                        transport_delay, reservation_cycles, address_stride),
                encode_native_word(instruction)};
        }
    }

    void push_encoded_synchronized_packet(
        const EncodedSynchronizedPacket& packet)
    {
        if constexpr (QueueRole != IcuQueueRole::Mem) {
            throw std::logic_error(
                "raw synchronized packets are only accepted by MEM ICU queues");
        } else {
            insert_encoded_synchronized_packet(imem_.size(), packet);
        }
    }

    // Linker surface for placing a two-word MEM_WRITE_SYNC packet into an
    // already assembled, but not launched, local i-MEM image. Insertion is
    // atomic and may occur only at a coarse-instruction packet boundary.
    void insert_encoded_synchronized_packet(std::size_t address,
        const EncodedSynchronizedPacket& packet)
    {
        if constexpr (QueueRole != IcuQueueRole::Mem) {
            throw std::logic_error(
                "raw synchronized packets are only accepted by MEM ICU queues");
        } else {
            (void)decode_synchronized_header(packet[0]);
            if (detail::raw_icu_opcode(packet[1])
                != isa::IcuCommandOpcode::Instruction)
                throw std::logic_error(
                    "synchronized MEM template is not a native FU word");
            const auto instruction = decode_native_word(packet[1]);
            if (instruction.opcode != MemOpcode::Write)
                throw std::logic_error(
                    "synchronized MEM packet must carry a write template");
            if (launched_)
                throw StaticScheduleError(
                    "ICU local i-MEM cannot be modified after program launch");
            if (address > imem_.size())
                throw StaticScheduleError(
                    "ICU synchronized insertion address is outside local i-MEM");
            if (imem_.size() > ImemDepth
                || kSynchronizedPacketWordCount
                    > ImemDepth - imem_.size())
                throw StaticScheduleError(
                    "ICU synchronized packet exceeds configured local i-MEM depth");
            validate_imem_packet_boundary(address);

            auto updated = imem_;
            updated.insert(updated.begin()
                    + static_cast<std::ptrdiff_t>(address),
                kSynchronizedPacketWordCount, std::nullopt);
            updated[address] = packet[0];
            updated[address + 1] = packet[1];
            imem_.swap(updated);
            configured_ = false;
            raw_compatibility_cycle_known_ = false;
        }
    }

    bool done() const
    {
        if (!configured_) {
            return imem_.empty();
        }
        return fetch_pc_ == program_end_pc_
            && pending_fetches_.empty()
            && iq_.empty()
            && nop_remaining_ == 0
            && repeat_remaining_ == 0
            && !repeat_2d_active_ && active_macros_.empty()
            && active_stream_nd_.empty()
            && !active_3d_.has_value()
            && !active_write_read_2d_.has_value()
            && !synchronized_instruction_.has_value();
    }

    bool running() const { return !done(); }
    bool underflowed() const noexcept { return underflowed_; }
    std::size_t cycle() const noexcept { return cycle_; }
    // i-MEM, IQ, and fetch counters are all measured in physical words.
    std::size_t imem_occupancy() const noexcept { return imem_.size(); }
    std::size_t iq_occupancy() const noexcept { return iq_.size(); }
    std::size_t pending_fetch_count() const noexcept
    {
        return pending_fetches_.size();
    }
    std::size_t fetch_pc() const noexcept { return fetch_pc_; }
    std::size_t fetched_count() const noexcept { return fetched_count_; }
    std::size_t issued_count() const noexcept { return issued_count_; }
    std::size_t synchronized_issued_count() const noexcept
    {
        return synchronized_issued_count_;
    }
    std::size_t synchronized_completed_count() const noexcept
    {
        return synchronized_completed_count_;
    }
    std::size_t peak_active_macros() const noexcept
    {
        return peak_active_macros_;
    }
    std::size_t active_3d_context_count() const noexcept
    {
        return (active_3d_.has_value() || active_write_read_2d_.has_value())
            ? 1 : 0;
    }
    std::size_t peak_active_3d_contexts() const noexcept
    {
        return peak_active_3d_contexts_;
    }
    const IcuQueueCycleTrace& last_trace() const noexcept
    {
        return last_trace_;
    }
    const std::optional<FuncInstruction>& last_dispatched() const noexcept
    {
        return last_dispatched_;
    }
    std::size_t free_iq_entries() const noexcept
    {
        return IqDepth - iq_.size() - pending_fetch_count();
    }
    std::size_t pending_issue_cycles() const noexcept
    {
        if (repeat_remaining_ == 0) {
            if (repeat_2d_active_)
                return nop_remaining_ + repeat_2d_cooldown_
                    + repeat_2d_remaining_points();
            return nop_remaining_;
        }
        return nop_remaining_ + repeat_cooldown_ + 1
            + (repeat_remaining_ - 1) * repeat_interval_;
    }
    std::size_t queued_count() const noexcept
    {
        if (!configured_) {
            return imem_.size();
        }
        return iq_.size() + pending_fetches_.size()
            + (program_end_pc_ - fetch_pc_)
            + repeat_remaining_ + nop_remaining_
            + repeat_2d_remaining_points()
            + macro_remaining_points()
            + stream_nd_remaining_points()
            + three_d_remaining_points()
            + write_read_2d_remaining_points()
            + synchronized_remaining_;
    }

private:
    static constexpr std::uint8_t kRepeat2DExtendedSubtype = 1;
    static constexpr std::uint8_t kFu3DExtendedSubtype = 2;
    static constexpr std::uint8_t kMemWriteRead2DExtendedSubtype = 7;
    static constexpr std::uint8_t kSyncExtendedSubtype = 3;
    static constexpr std::uint8_t kNotifyExtendedSubtype = 4;
    static constexpr std::uint8_t kWaitEventExtendedSubtype = 5;
    static constexpr std::uint8_t kSynchronizedExtendedSubtype = 6;
    static constexpr std::size_t kExtendedSubtypeOffset = 88;
    static constexpr unsigned kExtendedSubtypeBits = 4;
    static constexpr std::size_t kExtendedTagOffset = 2;
    static constexpr unsigned kExtendedTagBits = 64;
    static constexpr std::size_t kSynchronizedPacketWordCount = 2;

    // SYNCHRONIZED word 0 (one fixed-width FU raw word):
    //   [1:0]   Extended envelope
    //   [17:2]  issue count minus one
    //   [33:18] notification tag
    //   [49:34] transport delay
    //   [63:50] signed native-address stride
    //   [87:64] reserved queue window minus one
    //   [91:88] subtype 6
    // Word 1 is a normal Instruction-envelope native FU template.
    struct DecodedSynchronizedHeader {
        std::size_t count{0};
        std::size_t synchronization_tag{0};
        std::size_t transport_delay{0};
        std::size_t reservation_cycles{1};
        std::int64_t address_stride{0};
    };

    static Entry encode_synchronized_header(std::size_t count,
        std::size_t synchronizationTag, std::size_t transportDelay,
        std::size_t reservationCycles, std::int64_t addressStride)
    {
        static_assert(QueueRole != IcuQueueRole::Legacy);
        constexpr auto kMaxCount = std::size_t {1} << 16;
        constexpr auto kMax16 = std::size_t {0xffff};
        constexpr auto kMaxReservation = std::size_t {1} << 24;
        constexpr auto kStrideMin = -(std::int64_t {1} << 13);
        constexpr auto kStrideMax = (std::int64_t {1} << 13) - 1;
        if (count == 0 || count > kMaxCount)
            throw std::out_of_range(
                "ICU synchronized count does not fit raw command");
        if (synchronizationTag > kMax16)
            throw std::out_of_range(
                "ICU synchronized tag does not fit raw command");
        if (transportDelay > kMax16)
            throw std::out_of_range(
                "ICU synchronized transport delay does not fit raw command");
        if (reservationCycles == 0 || reservationCycles > kMaxReservation)
            throw std::out_of_range(
                "ICU synchronized reservation does not fit raw command");
        if (addressStride < kStrideMin || addressStride > kStrideMax)
            throw std::out_of_range(
                "ICU synchronized address stride does not fit raw command");

        Entry word{};
        detail::write_raw_icu_bits(word, 0, 2,
            static_cast<std::uint8_t>(isa::IcuCommandOpcode::Extended));
        detail::write_raw_icu_bits(word, 2, 16, count - 1);
        detail::write_raw_icu_bits(word, 18, 16, synchronizationTag);
        detail::write_raw_icu_bits(word, 34, 16, transportDelay);
        constexpr auto kStrideMask = (std::uint64_t {1} << 14) - 1;
        detail::write_raw_icu_bits(word, 50, 14,
            static_cast<std::uint64_t>(addressStride) & kStrideMask);
        detail::write_raw_icu_bits(word, 64, 24, reservationCycles - 1);
        detail::write_raw_icu_bits(word, kExtendedSubtypeOffset,
            kExtendedSubtypeBits, kSynchronizedExtendedSubtype);
        return word;
    }

    static DecodedSynchronizedHeader decode_synchronized_header(
        const Entry& word)
    {
        static_assert(QueueRole != IcuQueueRole::Legacy);
        if (detail::raw_icu_opcode(word)
                != isa::IcuCommandOpcode::Extended
            || detail::raw_icu_extended_subtype(word)
                != kSynchronizedExtendedSubtype)
            throw std::logic_error(
                "encoded ICU raw word is not a synchronized header");
        if (!detail::raw_icu_bits_are_zero(word, 92, InstructionBits))
            throw std::logic_error(
                "encoded ICU synchronized header has non-zero reserved bits");

        constexpr auto kStrideWidth = 14U;
        constexpr auto kStrideSign =
            std::uint64_t {1} << (kStrideWidth - 1);
        constexpr auto kStrideMask =
            (std::uint64_t {1} << kStrideWidth) - 1;
        const auto strideBits =
            detail::read_raw_icu_bits(word, 50, kStrideWidth);
        const auto stride = (strideBits & kStrideSign) == 0
            ? static_cast<std::int64_t>(strideBits)
            : -static_cast<std::int64_t>(
                ((~strideBits) & kStrideMask) + 1);
        return DecodedSynchronizedHeader {
            static_cast<std::size_t>(
                detail::read_raw_icu_bits(word, 2, 16)) + 1,
            static_cast<std::size_t>(
                detail::read_raw_icu_bits(word, 18, 16)),
            static_cast<std::size_t>(
                detail::read_raw_icu_bits(word, 34, 16)),
            static_cast<std::size_t>(
                detail::read_raw_icu_bits(word, 64, 24)) + 1,
            stride};
    }

    static void validate_native_instruction_role(
        const FuncInstruction& instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::MxmLoad) {
            const auto isLoadInstruction =
                instruction.opcode == MxmControlOpcode::IW
                || (instruction.opcode == MxmControlOpcode::Decode
                    && instruction.decode_operation
                        == MxmDecodeOperation::LoadActivation);
            if (!isLoadInstruction) {
                throw std::invalid_argument(
                    "MXM LOAD ICU native queue accepts only IW and "
                    "DecodeLoadActivation instructions");
            }
        } else if constexpr (QueueRole == IcuQueueRole::MxmCompute) {
            const auto isComputeInstruction =
                instruction.opcode == MxmControlOpcode::Compute
                || instruction.opcode == MxmControlOpcode::AccumulatorRead
                || (instruction.opcode == MxmControlOpcode::Decode
                    && instruction.decode_operation
                        == MxmDecodeOperation::StreamCompute);
            if (!isComputeInstruction) {
                throw std::invalid_argument(
                    "MXM COMPUTE ICU native queue accepts only Compute, "
                    "AccumulatorRead, and DecodeStreamCompute instructions");
            }
        }
    }

    static Entry encode_native_word(const FuncInstruction& instruction)
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            throw std::logic_error(
                "legacy ICU entries do not use the raw-word encoder");
        } else {
            validate_native_instruction_role(instruction);
            Entry word{};
            std::uint64_t encoded = 0;
            if constexpr (QueueRole == IcuQueueRole::Mem) {
                encoded = isa::encode_mem_instruction(instruction);
            } else if constexpr (QueueRole == IcuQueueRole::MxmLoad
                || QueueRole == IcuQueueRole::MxmCompute) {
                encoded = isa::encode_mxm_instruction(instruction);
            } else if constexpr (QueueRole == IcuQueueRole::MxmDequant) {
                encoded = isa::encode_mxm_dequant_instruction(instruction);
            } else if constexpr (QueueRole == IcuQueueRole::Vxm
                || QueueRole == IcuQueueRole::Sxm) {
                throw std::invalid_argument(
                    "VXM/SXM raw queues accept RUN_2D packets, not standalone words");
            } else if constexpr (QueueRole == IcuQueueRole::C2cTx
                || QueueRole == IcuQueueRole::C2cRx
                || QueueRole == IcuQueueRole::C2cDma) {
                throw std::invalid_argument(
                    "C2C raw queues require endpoint/DMA packets");
            }
            // Global envelope opcode Instruction is zero. The FU-native word
            // begins immediately above it at physical bit two.
            detail::write_raw_icu_bits(word, 2, 64, encoded);
            return word;
        }
    }

    static FuncInstruction decode_native_word(const Entry& word)
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            throw std::logic_error(
                "legacy ICU entries do not use the raw-word decoder");
        } else if constexpr (QueueRole == IcuQueueRole::Mem) {
            if (!detail::raw_icu_bits_are_zero(
                    word, 66, InstructionBits))
                throw std::logic_error(
                    "encoded MEM ICU native word has non-zero reserved bits");
            return isa::decode_mem_instruction(
                detail::read_raw_icu_bits(word, 2, 64));
        } else if constexpr (QueueRole == IcuQueueRole::MxmLoad
            || QueueRole == IcuQueueRole::MxmCompute) {
            if (!detail::raw_icu_bits_are_zero(
                    word, 66, InstructionBits))
                throw std::logic_error(
                    "encoded MXM ICU native word has non-zero reserved bits");
            auto instruction = isa::decode_mxm_instruction(
                detail::read_raw_icu_bits(word, 2, 64));
            validate_native_instruction_role(instruction);
            return instruction;
        } else if constexpr (QueueRole == IcuQueueRole::MxmDequant) {
            if (!detail::raw_icu_bits_are_zero(
                    word, 18, InstructionBits))
                throw std::logic_error(
                    "encoded MXM dequant ICU native word has non-zero reserved bits");
            return isa::decode_mxm_dequant_instruction(
                static_cast<isa::EncodedMxmDequantInstruction>(
                    detail::read_raw_icu_bits(word, 2, 16)));
        } else if constexpr (QueueRole == IcuQueueRole::C2cTx) {
            return C2cIcuPacketCodec::decode_tx(word).to_legacy();
        } else if constexpr (QueueRole == IcuQueueRole::C2cRx) {
            // SRAM placement is owned by the synchronized MEM write packet.
            return C2cIcuPacketCodec::decode_rx(word).to_legacy(0, 1);
        } else if constexpr (QueueRole == IcuQueueRole::C2cDma) {
            throw std::logic_error(
                "C2C DMA native decode requires two consecutive words");
        } else {
            static_cast<void>(word);
            throw std::invalid_argument(
                "VXM/SXM raw queues accept RUN_2D packets, not standalone words");
        }
    }

    static Entry encode_control_word(const IcuControlInstruction& control)
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            throw std::logic_error(
                "legacy ICU entries do not use the raw-word encoder");
        } else {
            Entry word{};
            switch (control.opcode) {
            case IcuControlOpcode::Nop:
                word.lanes[0] = isa::encode_icu_nop(control.count);
                return word;
            case IcuControlOpcode::Repeat:
                word.lanes[0] = isa::encode_icu_repeat(IcuRepeat {
                    control.count, control.interval,
                    control.address_stride});
                return word;
            case IcuControlOpcode::Repeat2D: {
                const auto encoded =
                    isa::encode_icu_repeat_2d(control.repeat_2d);
                std::copy(encoded.words.begin(), encoded.words.end(),
                    word.lanes.begin());
                return word;
            }
            case IcuControlOpcode::Sync:
            case IcuControlOpcode::Notify:
            case IcuControlOpcode::WaitEvent: {
                const auto subtype = control.opcode
                        == IcuControlOpcode::Sync
                    ? kSyncExtendedSubtype
                    : control.opcode == IcuControlOpcode::Notify
                    ? kNotifyExtendedSubtype
                    : kWaitEventExtendedSubtype;
                detail::write_raw_icu_bits(word, 0, 2,
                    static_cast<std::uint8_t>(
                        isa::IcuCommandOpcode::Extended));
                detail::write_raw_icu_bits(word,
                    kExtendedTagOffset, kExtendedTagBits,
                    static_cast<std::uint64_t>(control.event_tag));
                detail::write_raw_icu_bits(word,
                    kExtendedSubtypeOffset, kExtendedSubtypeBits,
                    subtype);
                return word;
            }
            }
            throw std::logic_error("unknown ICU control opcode");
        }
    }

    static IcuControlInstruction decode_control_word(const Entry& word)
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            throw std::logic_error(
                "legacy ICU entries do not use the raw-word decoder");
        } else {
            const auto opcode = detail::raw_icu_opcode(word);
            if (opcode == isa::IcuCommandOpcode::Nop) {
                if (!detail::raw_icu_bits_are_zero(
                        word, 32, InstructionBits))
                    throw std::logic_error(
                        "encoded ICU NOP has non-zero reserved bits");
                return IcuControlInstruction::Nop(
                    isa::decode_icu_nop_cycles(word.lanes[0]));
            }
            if (opcode == isa::IcuCommandOpcode::Repeat) {
                if (!detail::raw_icu_bits_are_zero(
                        word, 32, InstructionBits))
                    throw std::logic_error(
                        "encoded ICU Repeat has non-zero reserved bits");
                const auto repeat = isa::decode_icu_repeat(word.lanes[0]);
                return IcuControlInstruction::Repeat(repeat.count,
                    repeat.interval, repeat.address_stride);
            }
            if (opcode != isa::IcuCommandOpcode::Extended)
                throw std::logic_error(
                    "encoded ICU raw word is not a control command");

            const auto subtype =
                detail::raw_icu_extended_subtype(word);
            if (subtype == kRepeat2DExtendedSubtype) {
                if (!detail::raw_icu_bits_are_zero(
                        word, 92, InstructionBits))
                    throw std::logic_error(
                        "encoded ICU Repeat2D has non-zero reserved bits");
                isa::EncodedIcuRepeat2D encoded{};
                std::copy_n(word.lanes.begin(), encoded.words.size(),
                    encoded.words.begin());
                return IcuControlInstruction::Repeat2D(
                    isa::decode_icu_repeat_2d(encoded));
            }
            if (subtype != kSyncExtendedSubtype
                && subtype != kNotifyExtendedSubtype
                && subtype != kWaitEventExtendedSubtype)
                throw std::logic_error(
                    "encoded ICU Extended subtype is unsupported");
            if (!detail::raw_icu_bits_are_zero(word, 66, 88)
                || !detail::raw_icu_bits_are_zero(
                    word, 92, InstructionBits))
                throw std::logic_error(
                    "encoded ICU Extended control has non-zero reserved bits");
            const auto tag = static_cast<std::size_t>(
                detail::read_raw_icu_bits(
                    word, kExtendedTagOffset, kExtendedTagBits));
            if (subtype == kSyncExtendedSubtype)
                return IcuControlInstruction::Sync();
            if (subtype == kNotifyExtendedSubtype)
                return IcuControlInstruction::Notify();
            return IcuControlInstruction::WaitEvent(tag);
        }
    }

    void append_synchronized_packet(std::size_t count,
        std::size_t synchronizationTag, std::size_t transportDelay,
        std::size_t reservationCycles, std::int64_t addressStride,
        const FuncInstruction& instruction)
    {
        static_assert(QueueRole != IcuQueueRole::Legacy);
        // Encode both words before touching i-MEM so validation failures and
        // capacity failures can never leave a partial command behind.
        const auto header = encode_synchronized_header(count,
            synchronizationTag, transportDelay, reservationCycles,
            addressStride);
        const auto native = encode_native_word(instruction);
        if (launched_)
            throw StaticScheduleError(
                "ICU local i-MEM cannot be modified after program launch");
        if (imem_.size() > ImemDepth
            || kSynchronizedPacketWordCount
                > ImemDepth - imem_.size())
            throw StaticScheduleError(
                "ICU synchronized packet exceeds configured local i-MEM depth");

        const auto base = imem_.size();
        imem_.resize(base + kSynchronizedPacketWordCount);
        imem_[base] = header;
        imem_[base + 1] = native;
        configured_ = false;
    }

    static std::size_t checked_compatibility_add(std::size_t lhs,
        std::size_t rhs, const char* command)
    {
        if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
            throw std::overflow_error(
                std::string(command) + " overflows");
        return lhs + rhs;
    }

    static std::size_t checked_compatibility_multiply(std::size_t lhs,
        std::size_t rhs, const char* command)
    {
        if (lhs != 0
            && rhs > std::numeric_limits<std::size_t>::max() / lhs)
            throw std::overflow_error(
                std::string(command) + " overflows");
        return lhs * rhs;
    }

    static std::size_t repeat_2d_final_offset(const IcuRepeat2D& repeat)
    {
        const auto innerOffset = checked_compatibility_multiply(
            repeat.inner_count - 1, repeat.inner_interval,
            "ICU Repeat2D inner compatibility cursor");
        const auto outerOffset = checked_compatibility_multiply(
            repeat.outer_count - 1, repeat.outer_interval,
            "ICU Repeat2D outer compatibility cursor");
        return checked_compatibility_add(innerOffset, outerOffset,
            "ICU Repeat2D compatibility cursor");
    }

    static std::size_t loop_duration(const IcuLoop3D& loop)
    {
        detail::validate_icu_loop_3d(loop);
        auto finalOffset = loop.wait_cycle;
        for (std::size_t dimension = 0;
             dimension < IcuLoop3D::kDimensions; ++dimension) {
            finalOffset = checked_compatibility_add(finalOffset,
                checked_compatibility_multiply(loop.counts[dimension] - 1,
                    loop.cycle_strides[dimension],
                    "FU loop compatibility cursor"),
                "FU loop compatibility cursor");
        }
        return checked_compatibility_add(finalOffset, 1,
            "FU loop compatibility cursor");
    }

    std::optional<std::size_t> raw_compatibility_cycle_after(
        std::size_t delta, const char* command) const
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            static_cast<void>(delta);
            static_cast<void>(command);
            return std::nullopt;
        } else {
            if (!raw_compatibility_cycle_known_)
                return std::nullopt;
            return checked_compatibility_add(
                raw_compatibility_cycle_, delta, command);
        }
    }

    void commit_raw_compatibility_cycle(
        std::optional<std::size_t> nextCycle) noexcept
    {
        if constexpr (QueueRole != IcuQueueRole::Legacy) {
            if (nextCycle.has_value())
                raw_compatibility_cycle_ = *nextCycle;
        }
    }

    void append_raw_compatibility_start_nop(
        std::size_t absoluteStartCycle, const char* command)
    {
        static_assert(QueueRole != IcuQueueRole::Legacy);
        if (!raw_compatibility_cycle_known_)
            throw StaticScheduleError(std::string(command)
                + " follows a dynamic synchronization point, so its legacy "
                  "absolute start cannot be translated to queue-local NOPs");
        if (absoluteStartCycle < raw_compatibility_cycle_) {
            auto message = std::ostringstream {};
            message << command << " starts at legacy cycle "
                    << absoluteStartCycle << " before the queue-local "
                    << "compatibility cursor " << raw_compatibility_cycle_;
            throw StaticScheduleError(message.str());
        }
        push_nop(absoluteStartCycle - raw_compatibility_cycle_);
    }

    template <typename Packet>
    void append_3d_packet(const Packet& packet)
    {
        static_assert(QueueRole != IcuQueueRole::Legacy);
        static_assert(std::is_same_v<Packet, Encoded3DPacket>);
        const auto instruction = decode_3d_packet(packet);
        const auto nextCompatibilityCycle =
            raw_compatibility_cycle_after(loop_duration(instruction.loop),
                "FU loop compatibility cursor");
        if (launched_) {
            throw StaticScheduleError(
                "ICU local i-MEM cannot be modified after program launch");
        }
        constexpr auto wordCount = Encoded3DPacket::kWordCount;
        if (imem_.size() > ImemDepth
            || wordCount > ImemDepth - imem_.size()) {
            throw StaticScheduleError(
                "FU 3-D packet exceeds configured local i-MEM depth");
        }

        const auto base = imem_.size();
        imem_.resize(base + wordCount);
        for (std::size_t word = 0; word < wordCount; ++word) {
            imem_[base + word] = packet.words[word];
        }
        configured_ = false;
        commit_raw_compatibility_cycle(nextCompatibilityCycle);
    }

    struct TaggedNotification {
        std::size_t tag{0};
        // Age is relative to this token's arrival. A 16-bit transport delay
        // needs one extra state for the hardware-visible D+1 boundary.
        std::size_t age_cycles{0};
    };

    static constexpr std::size_t kMaxTaggedNotificationAge =
        std::size_t {1} << 16;

    void age_synchronized_notifications() noexcept
    {
        for (auto& notification : synchronized_notifications_)
            if (notification.age_cycles < kMaxTaggedNotificationAge)
                ++notification.age_cycles;
    }

    struct PendingFetch {
        std::size_t pc{0};
        Entry entry;
        std::size_t remaining_cycles{FetchLatency};
    };

    void begin_trace(IcuQueueAction action)
    {
        last_trace_ = IcuQueueCycleTrace{};
        last_trace_.cycle = cycle_;
        last_trace_.iq_before = iq_.size();
        last_trace_.action = action;
    }

    void finish_trace()
    {
        last_trace_.iq_after = iq_.size();
    }

    void ensure_configured()
    {
        if (!configured_) {
            configure_all();
            // Direct CModel enqueue calls represent a program already loaded
            // before cycle zero. Prime the finite IQ; all subsequent refills
            // still traverse the modeled local one-instruction/cycle frontend.
            while (iq_.size() < IqDepth && fetch_pc_ < program_end_pc_) {
                iq_.push_back(read_imem(fetch_pc_++));
                iq_pcs_.push_back(fetch_pc_ - 1);
                ++fetched_count_;
            }
        }
    }

    Entry read_imem(std::size_t address) const
    {
        if (address >= imem_.size() || !imem_[address].has_value()) {
            throw StaticScheduleError("ICU fetched an uninitialized local i-MEM address");
        }
        return *imem_[address];
    }

    void commit_ready_fetch()
    {
        while (!pending_fetches_.empty()
               && pending_fetches_.front().remaining_cycles == 0) {
            if (iq_.size() == IqDepth) {
                throw std::logic_error("ICU frontend committed into a full IQ");
            }
            iq_.push_back(std::move(pending_fetches_.front().entry));
            iq_pcs_.push_back(pending_fetches_.front().pc);
            last_trace_.fetch_completed_pc = pending_fetches_.front().pc;
            pending_fetches_.pop_front();
            ++fetched_count_;
        }
    }

    void begin_fetch_if_possible()
    {
        if (!launched_ || fetch_pc_ == program_end_pc_
            || free_iq_entries() == 0) {
            return;
        }
        last_trace_.fetch_started_pc = fetch_pc_;
        pending_fetches_.push_back(PendingFetch {
            fetch_pc_, read_imem(fetch_pc_), FetchLatency});
        ++fetch_pc_;
    }

    void age_pending_fetches()
    {
        for (auto& fetch : pending_fetches_) {
            if (fetch.remaining_cycles != 0) {
                --fetch.remaining_cycles;
            }
        }
    }

    std::optional<FuncInstruction> dispatch_ready_entry()
    {
        if (nop_remaining_ > 0) {
            --nop_remaining_;
            last_trace_.action = IcuQueueAction::NopWait;
            return std::nullopt;
        }
        if (repeat_remaining_ > 0) {
            return tick_repeat();
        }
        if (repeat_2d_active_) {
            return tick_repeat_2d();
        }
        if constexpr (QueueRole != IcuQueueRole::Legacy) {
            return dispatch_ready_raw_entry();
        } else {
        if (active_3d_.has_value()
            || (!iq_.empty() && is_3d_entry(iq_.front())))
            return tick_3d();
        if (!active_stream_nd_.empty()
            || (!iq_.empty() && std::holds_alternative<
                IcuStreamNdInstruction<FuncInstruction>>(iq_.front()))
            || (!iq_.empty() && std::holds_alternative<
                IcuStreamProgramInstruction<FuncInstruction>>(
                    iq_.front())))
            return tick_stream_nd();
        if (synchronized_instruction_.has_value()
            || (!iq_.empty() && std::holds_alternative<
                IcuSynchronizedInstruction<FuncInstruction>>(iq_.front())))
            return tick_synchronized();
        if (!active_macros_.empty()
            || (!iq_.empty() && std::holds_alternative<
                IcuMacroInstruction<FuncInstruction>>(iq_.front())))
            return tick_macros();
        if (iq_.empty()) {
            if (fetch_pc_ != program_end_pc_ || pending_fetch_count() != 0) {
                underflowed_ = true;
                last_trace_.action = IcuQueueAction::Underflow;
                std::ostringstream os;
                os << "ICU IQ underflow at cycle " << cycle_
                   << "; compiler did not prefetch before the scheduled issue";
                throw StaticScheduleError(os.str());
            }
            return std::nullopt;
        }

        if (auto* instruction = std::get_if<FuncInstruction>(&iq_.front())) {
            auto result = std::move(*instruction);
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::FunctionalIssue;
            last_dispatched_pc_ = iq_pcs_.front();
            iq_.pop_front();
            iq_pcs_.pop_front();
            last_dispatched_ = result;
            ++issued_count_;
            return result;
        }

        const auto control = std::get<IcuControlInstruction>(iq_.front());
        return execute_control(control);
        }
    }

    std::optional<FuncInstruction> dispatch_ready_raw_entry()
    {
        static_assert(QueueRole != IcuQueueRole::Legacy);
        // An active decoded command owns the single issue port. A following
        // raw packet stays at the IQ head until that command retires.
        if constexpr (is_fu_loop_queue_role(QueueRole)) {
            if (active_3d_.has_value())
                return tick_3d();
        }
        if constexpr (QueueRole == IcuQueueRole::Mem) {
            if (active_write_read_2d_.has_value())
                return tick_mem_write_read_2d();
            if (synchronized_instruction_.has_value())
                return tick_synchronized();
        }
        if constexpr (is_fu_loop_queue_role(QueueRole)) {
            if (!iq_.empty() && is_3d_entry(iq_.front()))
                return tick_3d();
        }
        if constexpr (QueueRole == IcuQueueRole::Mem) {
            if (!iq_.empty() && is_write_read_2d_entry(iq_.front()))
                return tick_mem_write_read_2d();
            if (!iq_.empty() && is_synchronized_entry(iq_.front()))
                return tick_synchronized();
        }
        if (iq_.empty()) {
            if (fetch_pc_ != program_end_pc_ || pending_fetch_count() != 0) {
                underflowed_ = true;
                last_trace_.action = IcuQueueAction::Underflow;
                std::ostringstream os;
                os << "ICU IQ underflow at cycle " << cycle_
                   << "; compiler did not prefetch before the scheduled issue";
                throw StaticScheduleError(os.str());
            }
            return std::nullopt;
        }

        const auto opcode = detail::raw_icu_opcode(iq_.front());
        if (opcode == isa::IcuCommandOpcode::Instruction) {
            if constexpr (QueueRole == IcuQueueRole::C2cDma)
                return dispatch_c2c_dma_packet();
            FuncInstruction result{};
            try {
                result = decode_native_word(iq_.front());
            } catch (const std::exception& error) {
                std::ostringstream os;
                os << "failed to decode FU native word at PC "
                   << iq_pcs_.front() << ": " << error.what();
                throw StaticScheduleError(os.str());
            }
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::FunctionalIssue;
            last_dispatched_pc_ = iq_pcs_.front();
            iq_.pop_front();
            iq_pcs_.pop_front();
            last_dispatched_ = result;
            ++issued_count_;
            return result;
        }

        IcuControlInstruction control{};
        try {
            control = decode_control_word(iq_.front());
        } catch (const std::exception& error) {
            std::ostringstream os;
            os << "failed to decode FU control word at PC "
               << iq_pcs_.front() << ": " << error.what();
            throw StaticScheduleError(os.str());
        }
        return execute_control(control);
    }

    std::optional<FuncInstruction> dispatch_c2c_dma_packet()
        requires (QueueRole == IcuQueueRole::C2cDma)
    {
        constexpr auto wordCount = C2cDmaIcuPacket::kWordCount;
        const auto headerPc = iq_pcs_.front();
        if (headerPc >= program_end_pc_
            || wordCount > program_end_pc_ - headerPc) {
            std::ostringstream os;
            os << "truncated C2C DMA packet at header PC " << headerPc
               << "; expected two consecutive 96-bit words";
            throw StaticScheduleError(os.str());
        }
        if (iq_.size() < wordCount) {
            last_trace_.issue_pc = headerPc;
            last_trace_.action = IcuQueueAction::ThreeDDecodeWait;
            return std::nullopt;
        }
        if (iq_pcs_.size() < wordCount || iq_pcs_[1] != headerPc + 1)
            throw StaticScheduleError(
                "C2C DMA packet words are not consecutive in local i-MEM");

        C2cDmaIcuPacket packet {};
        packet.words[0] = iq_[0];
        packet.words[1] = iq_[1];
        FuncInstruction result {};
        try {
            result = C2cIcuPacketCodec::decode_dma(packet).to_legacy();
        } catch (const std::exception& error) {
            std::ostringstream os;
            os << "failed to decode C2C DMA packet at PC " << headerPc
               << ": " << error.what();
            throw StaticScheduleError(os.str());
        }
        for (std::size_t word = 0; word < wordCount; ++word) {
            iq_.pop_front();
            iq_pcs_.pop_front();
        }
        last_trace_.issue_pc = headerPc;
        last_trace_.three_d_decode_header_pc = headerPc;
        last_trace_.three_d_decode_word_count = wordCount;
        last_trace_.action = IcuQueueAction::FunctionalIssue;
        last_dispatched_pc_ = headerPc;
        last_dispatched_ = result;
        ++issued_count_;
        return result;
    }

    std::optional<FuncInstruction> execute_control(
        const IcuControlInstruction& control)
    {
        switch (control.opcode) {
        case IcuControlOpcode::Nop:
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::Nop;
            iq_.pop_front();
            iq_pcs_.pop_front();
            nop_remaining_ = control.count;
            if (nop_remaining_ > 0) {
                --nop_remaining_;
            }
            return std::nullopt;
        case IcuControlOpcode::Repeat:
            return begin_repeat(control);
        case IcuControlOpcode::Repeat2D:
            return begin_repeat_2d(control);
        case IcuControlOpcode::Sync:
            if (notification_tokens_ == 0) {
                last_trace_.issue_pc = iq_pcs_.front();
                last_trace_.action = IcuQueueAction::SyncWait;
                return std::nullopt;
            }
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::SyncRelease;
            --notification_tokens_;
            iq_.pop_front();
            iq_pcs_.pop_front();
            return std::nullopt;
        case IcuControlOpcode::WaitEvent: {
            const auto notification = std::find_if(
                synchronized_notifications_.begin(),
                synchronized_notifications_.end(),
                [&](const TaggedNotification& candidate) {
                    return candidate.tag == control.event_tag;
                });
            if (notification == synchronized_notifications_.end()) {
                last_trace_.issue_pc = iq_pcs_.front();
                last_trace_.action = IcuQueueAction::EventWait;
                return std::nullopt;
            }
            synchronized_notifications_.erase(notification);
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::EventRelease;
            iq_.pop_front();
            iq_pcs_.pop_front();
            return std::nullopt;
        }
        case IcuControlOpcode::Notify:
            last_trace_.issue_pc = iq_pcs_.front();
            last_trace_.action = IcuQueueAction::Notify;
            iq_.pop_front();
            iq_pcs_.pop_front();
            notify_emitted_ = true;
            return std::nullopt;
        }
        throw std::logic_error("unknown ICU control opcode");
    }

    std::optional<FuncInstruction> begin_repeat(
        const IcuControlInstruction& control)
    {
        if (control.interval == 0) {
            throw std::invalid_argument("ICU Repeat interval must be at least one cycle");
        }
        if (!last_dispatched_.has_value()) {
            throw std::logic_error("ICU Repeat needs a prior functional instruction in the same IQ");
        }
        last_trace_.issue_pc = iq_pcs_.front();
        iq_.pop_front();
        iq_pcs_.pop_front();
        if (control.count == 0) {
            return std::nullopt;
        }
        repeat_remaining_ = control.count;
        repeat_instruction_ = *last_dispatched_;
        repeat_interval_ = control.interval;
        repeat_cooldown_ = control.interval - 1;
        repeat_address_stride_ = control.address_stride;
        repeat_index_ = 1;
        return tick_repeat();
    }

    std::optional<FuncInstruction> tick_repeat()
    {
        if (repeat_cooldown_ > 0) {
            --repeat_cooldown_;
            last_trace_.action = IcuQueueAction::RepeatWait;
            return std::nullopt;
        }
        auto result = detail::apply_icu_repeat_stride(
            *repeat_instruction_, repeat_address_stride_, repeat_index_);
        --repeat_remaining_;
        ++repeat_index_;
        if (repeat_remaining_ > 0) {
            repeat_cooldown_ = repeat_interval_ - 1;
        }
        last_dispatched_ = result;
        last_trace_.issue_pc = last_dispatched_pc_;
        last_trace_.action = IcuQueueAction::RepeatIssue;
        ++issued_count_;
        return result;
    }

    bool try_activate_front_synchronized()
    {
        if constexpr (QueueRole == IcuQueueRole::Legacy) {
            throw std::logic_error(
                "legacy ICU queue has no raw synchronized packet");
        } else {
            if (iq_.empty() || !is_synchronized_entry(iq_.front()))
                throw std::logic_error(
                    "ICU synchronized decoder lost its packet header");
            const auto headerPc = iq_pcs_.front();
            if (headerPc >= program_end_pc_
                || kSynchronizedPacketWordCount
                    > program_end_pc_ - headerPc) {
                std::ostringstream os;
                os << "truncated ICU synchronized packet at header PC "
                   << headerPc << "; expected "
                   << kSynchronizedPacketWordCount
                   << " consecutive local i-MEM words";
                throw StaticScheduleError(os.str());
            }
            if (iq_.size() < kSynchronizedPacketWordCount)
                return false;
            if (iq_pcs_.size() < kSynchronizedPacketWordCount)
                throw std::logic_error(
                    "ICU IQ entry/PC bookkeeping diverged");
            if (iq_pcs_[1] != headerPc + 1) {
                std::ostringstream os;
                os << "ICU synchronized packet at header PC "
                   << headerPc << " has a non-consecutive native word PC";
                throw StaticScheduleError(os.str());
            }

            DecodedSynchronizedHeader header{};
            FuncInstruction instruction{};
            try {
                header = decode_synchronized_header(iq_[0]);
                if (detail::raw_icu_opcode(iq_[1])
                    != isa::IcuCommandOpcode::Instruction)
                    throw std::logic_error(
                        "synchronized template is not a native FU word");
                instruction = decode_native_word(iq_[1]);
            } catch (const std::exception& error) {
                std::ostringstream os;
                os << "failed to decode ICU synchronized packet at header PC "
                   << headerPc << ": " << error.what();
                throw StaticScheduleError(os.str());
            }

            synchronized_instruction_ =
                IcuSynchronizedInstruction<FuncInstruction> {
                    header.count, header.synchronization_tag,
                    header.transport_delay, header.reservation_cycles,
                    header.address_stride, std::move(instruction)};
            synchronized_remaining_ = header.count;
            synchronized_index_ = 0;
            synchronized_elapsed_cycles_ = 0;
            synchronized_waiting_ = true;
            synchronized_descriptor_pc_ = headerPc;
            for (std::size_t word = 0;
                 word < kSynchronizedPacketWordCount; ++word) {
                iq_.pop_front();
                iq_pcs_.pop_front();
            }
            return true;
        }
    }

    std::optional<FuncInstruction> tick_synchronized()
    {
        if (!synchronized_instruction_.has_value()) {
            if constexpr (QueueRole == IcuQueueRole::Legacy) {
                auto* descriptor = std::get_if<
                    IcuSynchronizedInstruction<FuncInstruction>>(
                        &iq_.front());
                if (descriptor == nullptr)
                    throw std::logic_error(
                        "ICU synchronized dispatcher lost its descriptor");
                synchronized_instruction_ = std::move(*descriptor);
                synchronized_remaining_ =
                    synchronized_instruction_->count;
                synchronized_index_ = 0;
                synchronized_elapsed_cycles_ = 0;
                synchronized_waiting_ = true;
                synchronized_descriptor_pc_ = iq_pcs_.front();
                iq_.pop_front();
                iq_pcs_.pop_front();
            } else {
                if (!try_activate_front_synchronized()) {
                    last_trace_.action =
                        IcuQueueAction::SynchronizedWait;
                    return std::nullopt;
                }
            }
        }

        const auto finishReservationCycle = [&]() {
            ++synchronized_elapsed_cycles_;
            if (synchronized_remaining_ == 0
                && synchronized_elapsed_cycles_
                    >= synchronized_instruction_->reservation_cycles) {
                ++synchronized_completed_count_;
                synchronized_instruction_.reset();
                synchronized_descriptor_pc_.reset();
            }
        };

        if (synchronized_remaining_ == 0) {
            last_trace_.issue_pc = synchronized_descriptor_pc_;
            last_trace_.action = IcuQueueAction::SynchronizedDelay;
            finishReservationCycle();
            return std::nullopt;
        }

        if (synchronized_waiting_) {
            const auto notification = std::find_if(
                synchronized_notifications_.begin(),
                synchronized_notifications_.end(),
                [&](const TaggedNotification& candidate) {
                    return candidate.tag
                        == synchronized_instruction_->synchronization_tag;
                });
            if (notification == synchronized_notifications_.end()) {
                last_trace_.issue_pc = synchronized_descriptor_pc_;
                last_trace_.action = IcuQueueAction::SynchronizedWait;
                finishReservationCycle();
                return std::nullopt;
            }

            // Notifications age independently while earlier vectors cross the
            // SR fabric. The ICU compares only a token-local age with the
            // descriptor's relative delay; it never needs a global cycle.
            if (notification->age_cycles
                < synchronized_instruction_->transport_delay + 1) {
                last_trace_.issue_pc = synchronized_descriptor_pc_;
                last_trace_.action = IcuQueueAction::SynchronizedDelay;
                finishReservationCycle();
                return std::nullopt;
            }
            synchronized_notifications_.erase(notification);
            synchronized_waiting_ = false;
        }

        auto result = detail::apply_icu_repeat_stride(
            synchronized_instruction_->instruction,
            synchronized_instruction_->address_stride,
            synchronized_index_);
        ++synchronized_index_;
        --synchronized_remaining_;
        last_dispatched_ = result;
        last_dispatched_pc_ = synchronized_descriptor_pc_;
        last_trace_.issue_pc = synchronized_descriptor_pc_;
        last_trace_.action = IcuQueueAction::SynchronizedIssue;
        ++issued_count_;
        ++synchronized_issued_count_;
        if (synchronized_remaining_ != 0) {
            synchronized_waiting_ = true;
        }
        finishReservationCycle();
        return result;
    }

    std::optional<FuncInstruction> begin_repeat_2d(
        const IcuControlInstruction& control)
    {
        const auto& repeat = control.repeat_2d;
        if (!last_dispatched_.has_value())
            throw std::logic_error(
                "ICU Repeat2D needs a prior functional instruction");
        if (repeat.inner_count == 0 || repeat.outer_count == 0
            || repeat.inner_interval == 0 || repeat.outer_interval == 0
            || repeat.inner_count * repeat.outer_count <= 1
            || (repeat.outer_count > 1
                && repeat.outer_interval
                    <= (repeat.inner_count - 1) * repeat.inner_interval))
            throw std::invalid_argument(
                "ICU Repeat2D has an invalid iteration space");
        last_trace_.issue_pc = iq_pcs_.front();
        iq_.pop_front();
        iq_pcs_.pop_front();
        repeat_2d_ = repeat;
        repeat_2d_instruction_ = *last_dispatched_;
        repeat_2d_outer_ = 0;
        repeat_2d_inner_ = repeat.inner_count > 1 ? 1 : 0;
        if (repeat.inner_count == 1) repeat_2d_outer_ = 1;
        repeat_2d_active_ = true;
        const auto firstOffset = repeat_2d_issue_offset();
        repeat_2d_cooldown_ = firstOffset - 1;
        return tick_repeat_2d();
    }

    std::optional<FuncInstruction> tick_repeat_2d()
    {
        if (repeat_2d_cooldown_ > 0) {
            --repeat_2d_cooldown_;
            last_trace_.action = IcuQueueAction::Repeat2DWait;
            return std::nullopt;
        }
        const auto delta = static_cast<std::int64_t>(repeat_2d_inner_)
                * repeat_2d_.inner_stride
            + static_cast<std::int64_t>(repeat_2d_outer_)
                * repeat_2d_.outer_stride;
        auto result = detail::apply_icu_repeat_2d_stride(
            *repeat_2d_instruction_, repeat_2d_.induction_target, delta);
        last_trace_.issue_pc = last_dispatched_pc_;
        last_trace_.action = IcuQueueAction::Repeat2DIssue;
        last_dispatched_ = result;
        ++issued_count_;

        const auto previousOffset = repeat_2d_issue_offset();
        ++repeat_2d_inner_;
        if (repeat_2d_inner_ == repeat_2d_.inner_count) {
            repeat_2d_inner_ = 0;
            ++repeat_2d_outer_;
        }
        if (repeat_2d_outer_ == repeat_2d_.outer_count) {
            repeat_2d_active_ = false;
        } else {
            const auto nextOffset = repeat_2d_issue_offset();
            repeat_2d_cooldown_ = nextOffset - previousOffset - 1;
        }
        return result;
    }

    std::size_t repeat_2d_issue_offset() const noexcept
    {
        return repeat_2d_outer_ * repeat_2d_.outer_interval
            + repeat_2d_inner_ * repeat_2d_.inner_interval;
    }

    std::size_t repeat_2d_remaining_points() const noexcept
    {
        if (!repeat_2d_active_) return 0;
        return (repeat_2d_.outer_count - repeat_2d_outer_ - 1)
                * repeat_2d_.inner_count
            + repeat_2d_.inner_count - repeat_2d_inner_;
    }

    using Icu3DInstruction =
        typename Icu3DInstructionForRole<QueueRole>::Type;

    struct Active3D {
        Icu3DInstruction instruction{};
        std::size_t pc{0};
        IcuCoordinate3D coordinate{};
        std::size_t elapsed_cycle{0};

        std::size_t issue_offset() const
        {
            if constexpr (QueueRole != IcuQueueRole::Legacy) {
                return detail::icu_loop_3d_issue_cycle(
                    instruction.loop, coordinate);
            } else {
                return std::numeric_limits<std::size_t>::max();
            }
        }

        bool advance() noexcept
        {
            if constexpr (QueueRole != IcuQueueRole::Legacy) {
                return detail::advance_icu_coordinate_3d(
                    instruction.loop, coordinate);
            } else {
                return false;
            }
        }
    };

    struct ActiveMemWriteRead2D {
        MemIcuWriteRead2DInstruction instruction{};
        std::size_t pc{0};
        std::array<std::size_t, 2> inner{0, 0};
        std::array<std::size_t, 2> outer{0, 0};
        std::array<bool, 2> complete{false, false};
        std::size_t elapsed_cycle{0};

        std::size_t issue_offset(bool read) const noexcept
        {
            const auto index = read ? 1U : 0U;
            const auto& strides = read
                ? instruction.read_cycle_strides
                : instruction.write_cycle_strides;
            return instruction.start_wait
                + (read ? instruction.read_start_offset : 0)
                + inner[index] * strides[0]
                + outer[index] * strides[1];
        }

        void advance(bool read) noexcept
        {
            const auto index = read ? 1U : 0U;
            if (++inner[index] < instruction.counts[0]) return;
            inner[index] = 0;
            if (++outer[index] < instruction.counts[1]) return;
            complete[index] = true;
        }
    };

    bool try_activate_front_mem_write_read_2d()
    {
        if constexpr (QueueRole != IcuQueueRole::Mem) {
            throw std::logic_error(
                "WRITE_READ_2D is valid only on a MEM ICU");
        } else {
            constexpr auto wordCount =
                isa::EncodedMemIcuWriteRead2DPacket::kWordCount;
            const auto headerPc = iq_pcs_.front();
            if (headerPc >= program_end_pc_
                || wordCount > program_end_pc_ - headerPc)
                throw StaticScheduleError(
                    "truncated MEM WRITE_READ_2D packet");
            if (iq_.size() < wordCount) {
                last_trace_.action = IcuQueueAction::ThreeDDecodeWait;
                return false;
            }
            isa::EncodedMemIcuWriteRead2DPacket packet{};
            for (std::size_t word = 0; word < wordCount; ++word) {
                if (iq_pcs_[word] != headerPc + word)
                    throw StaticScheduleError(
                        "MEM WRITE_READ_2D packet has a non-consecutive word PC");
                packet.words[word] = iq_[word];
            }
            MemIcuWriteRead2DInstruction instruction{};
            try {
                instruction =
                    isa::decode_mem_icu_write_read_2d_instruction(packet);
            } catch (const std::exception& error) {
                throw StaticScheduleError(std::string(
                    "failed to decode MEM WRITE_READ_2D packet: ")
                    + error.what());
            }
            if (active_3d_.has_value()
                || active_write_read_2d_.has_value())
                throw std::logic_error(
                    "MEM ICU activated overlapping coarse instructions");
            const auto points = instruction.counts[0]
                * instruction.counts[1];
            if (points > std::numeric_limits<std::size_t>::max() / 2)
                throw std::overflow_error(
                    "MEM WRITE_READ_2D point count overflows");
            write_read_2d_remaining_points_ = points * 2;
            active_write_read_2d_.emplace(ActiveMemWriteRead2D {
                std::move(instruction), headerPc});
            peak_active_3d_contexts_ = 1;
            for (std::size_t word = 0; word < wordCount; ++word) {
                iq_.pop_front();
                iq_pcs_.pop_front();
            }
            last_trace_.three_d_decode_header_pc = headerPc;
            last_trace_.three_d_decode_word_count = wordCount;
            return true;
        }
    }

    std::optional<FuncInstruction> tick_mem_write_read_2d()
    {
        if constexpr (QueueRole != IcuQueueRole::Mem) {
            throw std::logic_error(
                "WRITE_READ_2D is valid only on a MEM ICU");
        } else {
            if (!active_write_read_2d_.has_value()) {
                if (!try_activate_front_mem_write_read_2d())
                    return std::nullopt;
            }
            auto& active = *active_write_read_2d_;
            const auto writeCycle = active.complete[0]
                ? std::numeric_limits<std::size_t>::max()
                : active.issue_offset(false);
            const auto readCycle = active.complete[1]
                ? std::numeric_limits<std::size_t>::max()
                : active.issue_offset(true);
            if (writeCycle == readCycle)
                throw StaticScheduleError(
                    "MEM WRITE_READ_2D violates the single-port bank contract");
            const auto due = std::min(writeCycle, readCycle);
            if (due < active.elapsed_cycle)
                throw StaticScheduleError(
                    "MEM WRITE_READ_2D missed a relative issue cycle");
            if (due > active.elapsed_cycle) {
                last_trace_.action = IcuQueueAction::MemWriteRead2DWait;
                ++active.elapsed_cycle;
                return std::nullopt;
            }
            const auto read = readCycle < writeCycle;
            auto result = detail::expand_mem_icu_write_read_2d_instruction(
                active.instruction, active.inner[read ? 1 : 0],
                active.outer[read ? 1 : 0], read);
            last_dispatched_ = result;
            last_dispatched_pc_ = active.pc;
            last_trace_.issue_pc = active.pc;
            last_trace_.action = IcuQueueAction::MemWriteRead2DIssue;
            ++issued_count_;
            --write_read_2d_remaining_points_;
            active.advance(read);
            ++active.elapsed_cycle;
            if (active.complete[0] && active.complete[1])
                active_write_read_2d_.reset();
            return result;
        }
    }


    void validate_imem_packet_boundary(std::size_t address) const
    {
        std::size_t pc = 0;
        while (pc < address) {
            if (pc >= imem_.size() || !imem_[pc].has_value())
                throw StaticScheduleError(
                    "ICU synchronized insertion crosses an uninitialized i-MEM word");
            const auto& word = *imem_[pc];
            std::size_t packetWords = 1;
            if (is_3d_entry(word)) {
                packetWords = Encoded3DPacket::kWordCount;
            } else if constexpr (QueueRole == IcuQueueRole::Mem) {
                if (is_write_read_2d_entry(word))
                    packetWords =
                        isa::EncodedMemIcuWriteRead2DPacket::kWordCount;
                else if (is_synchronized_entry(word))
                    packetWords = kSynchronizedPacketWordCount;
            }
            if (packetWords > imem_.size() - pc)
                throw StaticScheduleError(
                    "ICU synchronized insertion follows a truncated i-MEM packet");
            for (std::size_t offset = 0; offset < packetWords; ++offset)
                if (!imem_[pc + offset].has_value())
                    throw StaticScheduleError(
                        "ICU synchronized insertion crosses an uninitialized i-MEM packet");
            if (address < pc + packetWords)
                throw StaticScheduleError(
                    "ICU synchronized insertion address splits an i-MEM packet");
            pc += packetWords;
        }
        if (pc != address)
            throw StaticScheduleError(
                "ICU synchronized insertion address is not an i-MEM packet boundary");
    }

    static bool is_3d_entry(const Entry& entry) noexcept
    {
        if constexpr (QueueRole != IcuQueueRole::Legacy) {
            return detail::raw_icu_opcode(entry)
                    == isa::IcuCommandOpcode::Extended
                && detail::raw_icu_extended_subtype(entry)
                    == kFu3DExtendedSubtype;
        } else {
            static_cast<void>(entry);
            return false;
        }
    }

    static bool is_write_read_2d_entry(const Entry& entry) noexcept
    {
        if constexpr (QueueRole == IcuQueueRole::Mem) {
            return detail::raw_icu_opcode(entry)
                    == isa::IcuCommandOpcode::Extended
                && detail::raw_icu_extended_subtype(entry)
                    == kMemWriteRead2DExtendedSubtype;
        } else {
            static_cast<void>(entry);
            return false;
        }
    }

    static bool is_synchronized_entry(const Entry& entry) noexcept
    {
        if constexpr (QueueRole != IcuQueueRole::Legacy) {
            return detail::raw_icu_opcode(entry)
                    == isa::IcuCommandOpcode::Extended
                && detail::raw_icu_extended_subtype(entry)
                    == kSynchronizedExtendedSubtype;
        } else {
            static_cast<void>(entry);
            return false;
        }
    }

    template <typename IcuInstruction>
    static IcuQueueAction three_d_action_for(bool issue)
    {
        if constexpr (std::is_same_v<IcuInstruction, MemIcuInstruction>)
            return issue ? IcuQueueAction::Mem3DIssue
                         : IcuQueueAction::Mem3DWait;
        else if constexpr (std::is_same_v<IcuInstruction,
                               MxmLoadIcuInstruction>)
            return issue ? IcuQueueAction::MxmLoad3DIssue
                         : IcuQueueAction::MxmLoad3DWait;
        else if constexpr (std::is_same_v<IcuInstruction,
                               MxmDequantIcuInstruction>)
            return issue ? IcuQueueAction::MxmDequant3DIssue
                         : IcuQueueAction::MxmDequant3DWait;
        else if constexpr (std::is_same_v<IcuInstruction,
                               MxmComputeIcuInstruction>)
            return issue ? IcuQueueAction::MxmCompute3DIssue
                         : IcuQueueAction::MxmCompute3DWait;
        else if constexpr (std::is_same_v<IcuInstruction,
                               VxmIcuRun2DInstruction>)
            return issue ? IcuQueueAction::VxmRun2DIssue
                         : IcuQueueAction::VxmRun2DWait;
        else if constexpr (std::is_same_v<IcuInstruction,
                               SxmIcuRun2DInstruction>)
            return issue ? IcuQueueAction::SxmRun2DIssue
                         : IcuQueueAction::SxmRun2DWait;
        else
            throw std::logic_error(
                "ICU active 3-D instruction is invalid");
    }

    static IcuQueueAction three_d_action(
        const Icu3DInstruction& instruction, bool issue)
    {
        if constexpr (QueueRole != IcuQueueRole::Legacy) {
            return three_d_action_for<Icu3DInstruction>(issue);
        } else {
            static_cast<void>(instruction);
            throw std::logic_error(
                "a legacy ICU queue has no 3-D descriptor context");
        }
    }

    static Icu3DInstruction decode_3d_packet(
        const Encoded3DPacket& packet)
    {
        if constexpr (QueueRole == IcuQueueRole::Mem) {
            return isa::decode_mem_icu_3d_instruction(packet);
        } else if constexpr (QueueRole == IcuQueueRole::MxmLoad) {
            return isa::decode_mxm_load_icu_3d_instruction(packet);
        } else if constexpr (QueueRole == IcuQueueRole::MxmDequant) {
            return isa::decode_mxm_dequant_icu_3d_instruction(packet);
        } else if constexpr (QueueRole == IcuQueueRole::MxmCompute) {
            return isa::decode_mxm_compute_icu_3d_instruction(packet);
        } else if constexpr (QueueRole == IcuQueueRole::Vxm) {
            return isa::decode_vxm_icu_run_2d_instruction(packet);
        } else if constexpr (QueueRole == IcuQueueRole::Sxm) {
            return isa::decode_sxm_icu_run_2d_instruction(packet);
        } else {
            static_cast<void>(packet);
            throw std::logic_error(
                "a legacy ICU queue cannot decode a 3-D packet");
        }
    }

    void activate_3d(Icu3DInstruction instruction, std::size_t headerPc)
    {
        if constexpr (QueueRole == IcuQueueRole::Mem) {
            detail::validate_mem_icu_instruction(instruction);
        } else if constexpr (QueueRole == IcuQueueRole::MxmLoad) {
            detail::validate_mxm_load_icu_instruction(instruction);
        } else if constexpr (QueueRole == IcuQueueRole::MxmDequant) {
            detail::validate_mxm_dequant_icu_instruction(instruction);
        } else if constexpr (QueueRole == IcuQueueRole::MxmCompute) {
            detail::validate_mxm_compute_icu_instruction(instruction);
        } else if constexpr (QueueRole == IcuQueueRole::Vxm) {
            detail::validate_vxm_icu_run_2d_instruction(instruction);
        } else if constexpr (QueueRole == IcuQueueRole::Sxm) {
            detail::validate_sxm_icu_run_2d_instruction(instruction);
        } else {
            static_cast<void>(instruction);
            static_cast<void>(headerPc);
            throw std::logic_error(
                "a legacy ICU queue cannot activate a 3-D instruction");
        }

        const auto loop = instruction.loop;
        static_assert(three_d_context_depth == 1,
            "physical ICU queues have one active coarse-instruction context");
        if (active_3d_.has_value())
            throw std::logic_error(
                "ICU activated a coarse instruction before the current instruction retired");
        const auto points = detail::icu_loop_3d_point_count(loop);
        if (points > std::numeric_limits<std::size_t>::max()
                - three_d_remaining_points_)
            throw std::overflow_error(
                "ICU active 3-D point count overflows");
        three_d_remaining_points_ += points;
        active_3d_.emplace(Active3D {
            std::move(instruction), headerPc, {}, 0});
        peak_active_3d_contexts_ = 1;
    }

    bool try_activate_front_3d()
    {
        if constexpr (QueueRole != IcuQueueRole::Legacy) {
            constexpr auto wordCount = Encoded3DPacket::kWordCount;
            const auto headerPc = iq_pcs_.front();
            if (headerPc >= program_end_pc_
                || wordCount > program_end_pc_ - headerPc) {
                std::ostringstream os;
                os << "truncated FU 3-D packet at header PC " << headerPc
                   << "; expected " << wordCount
                   << " consecutive local i-MEM words";
                throw StaticScheduleError(os.str());
            }
            if (iq_.size() < wordCount) {
                last_trace_.action = IcuQueueAction::ThreeDDecodeWait;
                return false;
            }
            if (iq_pcs_.size() < wordCount)
                throw std::logic_error(
                    "ICU IQ entry/PC bookkeeping diverged");

            Encoded3DPacket packet{};
            for (std::size_t word = 0; word < wordCount; ++word) {
                if (iq_pcs_[word] != headerPc + word) {
                    std::ostringstream os;
                    os << "FU 3-D packet at header PC " << headerPc
                       << " has a non-consecutive word PC";
                    throw StaticScheduleError(os.str());
                }
                packet.words[word] = iq_[word];
            }

            Icu3DInstruction instruction{};
            try {
                instruction = decode_3d_packet(packet);
            } catch (const std::exception& error) {
                std::ostringstream os;
                os << "failed to decode FU 3-D packet at header PC "
                   << headerPc << ": " << error.what();
                throw StaticScheduleError(os.str());
            }

            activate_3d(std::move(instruction), headerPc);
            for (std::size_t word = 0; word < wordCount; ++word) {
                iq_.pop_front();
                iq_pcs_.pop_front();
            }
            last_trace_.three_d_decode_header_pc = headerPc;
            last_trace_.three_d_decode_word_count = wordCount;
            return true;
        } else {
            throw std::logic_error(
                "a legacy ICU queue has no 3-D packet entry");
        }
    }

    FuncInstruction expand_3d(const Active3D& active) const
    {
        if constexpr (QueueRole == IcuQueueRole::Mem) {
            return detail::expand_mem_icu_instruction(
                active.instruction, active.coordinate);
        } else if constexpr (QueueRole == IcuQueueRole::MxmLoad) {
            return detail::expand_mxm_load_icu_instruction(
                active.instruction, active.coordinate);
        } else if constexpr (QueueRole == IcuQueueRole::MxmDequant) {
            return active.instruction.instruction;
        } else if constexpr (QueueRole == IcuQueueRole::MxmCompute) {
            return detail::expand_mxm_compute_icu_instruction(
                active.instruction, active.coordinate);
        } else if constexpr (QueueRole == IcuQueueRole::Vxm) {
            return active.instruction.instruction;
        } else if constexpr (QueueRole == IcuQueueRole::Sxm) {
            auto result = active.instruction.instruction;
            if (result.opcode == SxmOpcode::Permute
                && active.instruction.permute_map_stride != 0) {
                const auto linearIndex = active.coordinate.index[0]
                    + active.instruction.loop.counts[0]
                        * active.coordinate.index[1];
                const auto delta = linearIndex
                    * active.instruction.permute_map_stride;
                for (auto& lane : result.permute_map)
                    if (lane != SxmInstruction::kZeroFill)
                        lane = (lane + delta)
                            % SxmInstruction::kTotalLanes;
            }
            return result;
        } else {
            static_cast<void>(active);
            throw std::logic_error(
                "a legacy ICU queue cannot expand a 3-D descriptor");
        }
    }

    std::optional<FuncInstruction> tick_3d()
    {
        if (!active_stream_nd_.empty() || !active_macros_.empty()
            || synchronized_instruction_.has_value())
            throw StaticScheduleError(
                "ICU queue mixes an in-flight 3-D instruction with legacy commands");

        bool decodeWait = false;
        if (!active_3d_.has_value() && !iq_.empty()) {
            if (is_3d_entry(iq_.front())) {
                decodeWait = !try_activate_front_3d();
            }
        }

        if (!active_3d_.has_value()) {
            if (decodeWait)
                last_trace_.action = IcuQueueAction::ThreeDDecodeWait;
            return std::nullopt;
        }
        if (active_3d_->issue_offset() > active_3d_->elapsed_cycle) {
            last_trace_.action =
                three_d_action(active_3d_->instruction, false);
            ++active_3d_->elapsed_cycle;
            return std::nullopt;
        }
        if (active_3d_->issue_offset() < active_3d_->elapsed_cycle)
            throw StaticScheduleError(
                "ICU 3-D expansion missed a relative issue offset");

        auto due = std::move(*active_3d_);
        active_3d_.reset();

        auto result = expand_3d(due);
        last_dispatched_ = result;
        last_dispatched_pc_ = due.pc;
        last_trace_.issue_pc = due.pc;
        last_trace_.action = three_d_action(due.instruction, true);
        ++issued_count_;
        --three_d_remaining_points_;
        if (due.advance()) {
            ++due.elapsed_cycle;
            active_3d_.emplace(std::move(due));
        }
        return result;
    }


    std::size_t three_d_remaining_points() const noexcept
    {
        return three_d_remaining_points_;
    }

    std::size_t write_read_2d_remaining_points() const noexcept
    {
        return write_read_2d_remaining_points_;
    }

    static void validate_stream_nd_schedule(
        const IcuStreamNdSchedule& schedule)
    {
        if (schedule.rank == 0
            || schedule.rank > IcuMemStreamNdSchedule::kMaxRank)
            throw std::invalid_argument(
                "STREAM_ND rank must be between one and three");

        std::size_t lowerSpan = 0;
        std::size_t points = 1;
        for (std::size_t dimension = 0;
             dimension < schedule.rank; ++dimension) {
            const auto count = schedule.counts[dimension];
            const auto stride = schedule.cycle_strides[dimension];
            if (count == 0 || stride == 0)
                throw std::invalid_argument(
                    "STREAM_ND counts and cycle strides must be non-zero");
            if (dimension != 0 && count > 1 && stride <= lowerSpan)
                throw std::invalid_argument(
                    "STREAM_ND dimensions overlap in issue time");
            if (count > std::numeric_limits<std::size_t>::max() / points)
                throw std::overflow_error(
                    "STREAM_ND iteration count overflows");
            points *= count;
            const auto steps = count - 1;
            if (steps != 0
                && stride > (std::numeric_limits<std::size_t>::max()
                    - lowerSpan) / steps)
                throw std::overflow_error(
                    "STREAM_ND cycle span overflows");
            lowerSpan += steps * stride;
        }
        if (schedule.start_cycle
            > std::numeric_limits<std::size_t>::max() - lowerSpan)
            throw std::overflow_error(
                "STREAM_ND final issue cycle overflows");
        const bool hasOperandInduction = std::any_of(
            schedule.operand_strides.begin(),
            schedule.operand_strides.begin() + schedule.rank,
            [](std::int64_t stride) { return stride != 0; });
        if (hasOperandInduction
            && schedule.induction_target == IcuInductionTarget::None)
            throw std::invalid_argument(
                "STREAM_ND operand stride requires an induction target");
    }

    struct ActiveStreamNd {
        IcuStreamNdInstruction<FuncInstruction> stream;
        std::size_t pc{0};
        std::array<std::size_t, IcuStreamNdSchedule::kMaxRank>
            indices{};
        bool mem_slice_program{false};

        std::size_t issue_cycle() const noexcept
        {
            auto result = stream.schedule.start_cycle;
            for (std::size_t dimension = 0;
                 dimension < stream.schedule.rank; ++dimension)
                result += indices[dimension]
                    * stream.schedule.cycle_strides[dimension];
            return result;
        }

        std::int64_t operand_delta() const noexcept
        {
            auto result = std::int64_t {0};
            for (std::size_t dimension = 0;
                 dimension < stream.schedule.rank; ++dimension)
                result += static_cast<std::int64_t>(indices[dimension])
                    * stream.schedule.operand_strides[dimension];
            return result;
        }

        bool advance() noexcept
        {
            for (std::size_t dimension = 0;
                 dimension < stream.schedule.rank; ++dimension) {
                ++indices[dimension];
                if (indices[dimension]
                    != stream.schedule.counts[dimension])
                    return true;
                indices[dimension] = 0;
            }
            return false;
        }
    };

    static void validate_stream_program(
        const IcuStreamProgramInstruction<FuncInstruction>& program)
    {
        if constexpr (!std::is_same_v<FuncInstruction, MemInstruction>) {
            throw std::invalid_argument(
                "MEM_SLICE_PROGRAM is valid only on a MEM ICU queue");
        } else {
            if (program.body.empty()
                || program.body.size()
                    > IcuMemSliceProgram::kMaxBodyEntries)
                throw std::invalid_argument(
                    "MEM_SLICE_PROGRAM body must contain between one and sixteen entries");
            if (program.schedule.induction_target
                    != IcuInductionTarget::None
                || std::any_of(program.schedule.operand_strides.begin(),
                    program.schedule.operand_strides.end(),
                    [](std::int64_t stride) { return stride != 0; }))
                throw std::invalid_argument(
                    "MEM_SLICE_PROGRAM launch domain cannot carry operand induction");
            validate_stream_nd_schedule(program.schedule);
            for (const auto& entry : program.body) {
                if (entry.cycle_offset
                    > std::numeric_limits<std::size_t>::max()
                        - program.schedule.start_cycle)
                    throw std::overflow_error(
                        "MEM_SLICE_PROGRAM body start cycle overflows");
                auto schedule = program.schedule;
                schedule.start_cycle += entry.cycle_offset;
                schedule.operand_strides = entry.operand_strides;
                schedule.induction_target = IcuInductionTarget::MemAddress;
                validate_stream_nd_schedule(schedule);
            }
        }
    }

    struct LaterStreamNdIssue {
        bool operator()(const ActiveStreamNd& lhs,
            const ActiveStreamNd& rhs) const noexcept
        {
            return lhs.issue_cycle() > rhs.issue_cycle();
        }
    };

    std::optional<FuncInstruction> tick_stream_nd()
    {
        if (!iq_.empty()) {
            if (auto* stream = std::get_if<
                    IcuStreamNdInstruction<FuncInstruction>>(
                        &iq_.front())) {
                validate_stream_nd_schedule(stream->schedule);
                if (cycle_ > stream->schedule.start_cycle) {
                    std::ostringstream os;
                    os << "STREAM_ND missed start cycle "
                       << stream->schedule.start_cycle << " at cycle "
                       << cycle_;
                    throw StaticScheduleError(os.str());
                }
                std::size_t points = 1;
                for (std::size_t dimension = 0;
                     dimension < stream->schedule.rank; ++dimension)
                    points *= stream->schedule.counts[dimension];
                stream_nd_remaining_points_ += points;
                active_stream_nd_.push(ActiveStreamNd {
                    std::move(*stream), iq_pcs_.front(), {}, false});
                iq_.pop_front();
                iq_pcs_.pop_front();
            } else if (auto* program = std::get_if<
                           IcuStreamProgramInstruction<FuncInstruction>>(
                           &iq_.front())) {
                validate_stream_program(*program);
                std::size_t firstIssue =
                    std::numeric_limits<std::size_t>::max();
                for (const auto& entry : program->body)
                    firstIssue = std::min(firstIssue,
                        program->schedule.start_cycle
                            + entry.cycle_offset);
                if (cycle_ > firstIssue) {
                    std::ostringstream os;
                    os << "MEM_SLICE_PROGRAM missed first issue cycle "
                       << firstIssue << " at cycle " << cycle_;
                    throw StaticScheduleError(os.str());
                }
                std::size_t points = 1;
                for (std::size_t dimension = 0;
                     dimension < program->schedule.rank; ++dimension)
                    points *= program->schedule.counts[dimension];
                const auto pc = iq_pcs_.front();
                for (auto& entry : program->body) {
                    auto schedule = program->schedule;
                    schedule.start_cycle += entry.cycle_offset;
                    schedule.operand_strides = entry.operand_strides;
                    schedule.induction_target =
                        IcuInductionTarget::MemAddress;
                    stream_nd_remaining_points_ += points;
                    active_stream_nd_.push(ActiveStreamNd {
                        IcuStreamNdInstruction<FuncInstruction> {
                            schedule, std::move(entry.instruction)},
                        pc, {}, true});
                }
                iq_.pop_front();
                iq_pcs_.pop_front();
            } else if (!active_stream_nd_.empty()) {
                throw StaticScheduleError(
                    "ICU mixes an in-flight STREAM_ND with legacy commands");
            }
        }

        constexpr bool isMem =
            std::is_same_v<FuncInstruction, MemInstruction>;
        constexpr bool isVxm =
            std::is_same_v<FuncInstruction, VxmCompactInstruction>;
        constexpr bool isSxm =
            std::is_same_v<FuncInstruction, SxmInstruction>;
        if (active_stream_nd_.empty()
            || active_stream_nd_.top().issue_cycle() > cycle_) {
            last_trace_.action = isMem
                ? (!active_stream_nd_.empty()
                            && active_stream_nd_.top().mem_slice_program
                        ? IcuQueueAction::MemSliceProgramWait
                        : IcuQueueAction::MemStreamNdWait)
                : isVxm ? IcuQueueAction::VxmStreamNdWait
                : isSxm ? IcuQueueAction::SxmTileProgramWait
                        : IcuQueueAction::MxmStreamNdWait;
            return std::nullopt;
        }
        if (active_stream_nd_.top().issue_cycle() < cycle_)
            throw StaticScheduleError(
                "STREAM_ND expansion missed an issue cycle");
        auto due = active_stream_nd_.top();
        active_stream_nd_.pop();
        if (!active_stream_nd_.empty()
            && active_stream_nd_.top().issue_cycle() == cycle_)
            throw StaticScheduleError(
                "overlapping STREAM_ND descriptors issue on one ICU queue cycle");

        auto result = detail::apply_icu_repeat_2d_stride(
            due.stream.instruction,
            due.stream.schedule.induction_target, due.operand_delta());
        last_dispatched_ = result;
        last_dispatched_pc_ = due.pc;
        last_trace_.issue_pc = due.pc;
        last_trace_.action = isMem
            ? (due.mem_slice_program
                    ? IcuQueueAction::MemSliceProgramIssue
                    : IcuQueueAction::MemStreamNdIssue)
            : isVxm ? IcuQueueAction::VxmStreamNdIssue
            : isSxm ? IcuQueueAction::SxmTileProgramIssue
                    : IcuQueueAction::MxmStreamNdIssue;
        ++issued_count_;
        --stream_nd_remaining_points_;
        if (due.advance()) active_stream_nd_.push(std::move(due));
        return result;
    }

    std::size_t stream_nd_remaining_points() const noexcept
    {
        return stream_nd_remaining_points_;
    }

    static void validate_macro_schedule(const IcuMacroSchedule& schedule)
    {
        if (schedule.inner_count == 0 || schedule.outer_count == 0
            || schedule.inner_interval == 0
            || schedule.outer_interval == 0
            || (schedule.outer_count > 1
                && schedule.outer_interval
                    <= (schedule.inner_count - 1)
                        * schedule.inner_interval)) {
            throw std::invalid_argument(
                "ICU macro has an invalid iteration space");
        }
    }

    struct ActiveMacro {
        IcuMacroInstruction<FuncInstruction> macro;
        std::size_t pc{0};
        std::size_t inner{0};
        std::size_t outer{0};

        std::size_t issue_cycle() const noexcept
        {
            return macro.schedule.start_cycle
                + outer * macro.schedule.outer_interval
                + inner * macro.schedule.inner_interval;
        }
    };

    struct LaterMacroIssue {
        bool operator()(const ActiveMacro& lhs,
            const ActiveMacro& rhs) const noexcept
        {
            return lhs.issue_cycle() > rhs.issue_cycle();
        }
    };

    std::optional<FuncInstruction> tick_macros()
    {
        if (!iq_.empty()) {
            if (auto* macro = std::get_if<
                    IcuMacroInstruction<FuncInstruction>>(&iq_.front())) {
                validate_macro_schedule(macro->schedule);
                if (cycle_ > macro->schedule.start_cycle) {
                    std::ostringstream os;
                    os << "ICU macro missed start cycle "
                       << macro->schedule.start_cycle << " at cycle "
                       << cycle_;
                    throw StaticScheduleError(os.str());
                }
                if (active_macros_.size() >= MacroContextDepth) {
                    if (macro->schedule.start_cycle <= cycle_) {
                        std::ostringstream os;
                        os << "ICU Macro context capacity "
                           << MacroContextDepth
                           << " is exhausted at cycle " << cycle_;
                        throw StaticScheduleError(os.str());
                    }
                } else {
                    macro_remaining_points_ += macro->schedule.inner_count
                        * macro->schedule.outer_count;
                    active_macros_.push(ActiveMacro {
                        std::move(*macro), iq_pcs_.front(), 0, 0});
                    peak_active_macros_ = std::max(
                        peak_active_macros_, active_macros_.size());
                    iq_.pop_front();
                    iq_pcs_.pop_front();
                }
            } else if (!active_macros_.empty()) {
                throw StaticScheduleError(
                    "ICU queue mixes an in-flight macro with legacy commands");
            }
        }

        if (active_macros_.empty()
            || active_macros_.top().issue_cycle() > cycle_) {
            last_trace_.action = IcuQueueAction::MacroWait;
            return std::nullopt;
        }
        if (active_macros_.top().issue_cycle() < cycle_)
            throw StaticScheduleError(
                "ICU macro expansion missed an issue cycle");
        auto due = active_macros_.top();
        active_macros_.pop();
        if (!active_macros_.empty()
            && active_macros_.top().issue_cycle() == cycle_)
            throw StaticScheduleError(
                "overlapping ICU macros issue on the same queue cycle");

        const auto delta = static_cast<std::int64_t>(due.inner)
                * due.macro.schedule.inner_stride
            + static_cast<std::int64_t>(due.outer)
                * due.macro.schedule.outer_stride;
        auto result = detail::apply_icu_repeat_2d_stride(
            due.macro.instruction,
            due.macro.schedule.induction_target, delta);
        last_dispatched_ = result;
        last_dispatched_pc_ = due.pc;
        last_trace_.issue_pc = due.pc;
        last_trace_.action = IcuQueueAction::MacroIssue;
        ++issued_count_;
        --macro_remaining_points_;

        ++due.inner;
        if (due.inner == due.macro.schedule.inner_count) {
            due.inner = 0;
            ++due.outer;
        }
        if (due.outer != due.macro.schedule.outer_count)
            active_macros_.push(std::move(due));
        return result;
    }

    std::size_t macro_remaining_points() const noexcept
    {
        return macro_remaining_points_;
    }

    std::vector<std::optional<Entry>> imem_{};
    std::deque<Entry> iq_{};
    std::deque<std::size_t> iq_pcs_{};
    std::deque<PendingFetch> pending_fetches_{};
    std::optional<FuncInstruction> last_dispatched_{};
    std::optional<FuncInstruction> repeat_instruction_{};
    std::optional<FuncInstruction> repeat_2d_instruction_{};
    std::priority_queue<ActiveMacro, std::vector<ActiveMacro>,
        LaterMacroIssue> active_macros_{};
    std::size_t macro_remaining_points_{0};
    std::size_t peak_active_macros_{0};
    std::priority_queue<ActiveStreamNd,
        std::vector<ActiveStreamNd>, LaterStreamNdIssue>
        active_stream_nd_{};
    std::size_t stream_nd_remaining_points_{0};
    std::optional<Active3D> active_3d_{};
    std::optional<ActiveMemWriteRead2D> active_write_read_2d_{};
    std::size_t three_d_remaining_points_{0};
    std::size_t write_read_2d_remaining_points_{0};
    std::size_t peak_active_3d_contexts_{0};
    std::optional<IcuSynchronizedInstruction<FuncInstruction>>
        synchronized_instruction_{};
    std::optional<std::size_t> synchronized_descriptor_pc_{};
    std::size_t synchronized_remaining_{0};
    std::size_t synchronized_index_{0};
    std::size_t synchronized_elapsed_cycles_{0};
    bool synchronized_waiting_{false};
    std::optional<std::size_t> last_dispatched_pc_{};
    std::size_t nop_remaining_{0};
    std::size_t repeat_remaining_{0};
    std::size_t repeat_interval_{1};
    std::size_t repeat_cooldown_{0};
    std::int64_t repeat_address_stride_{0};
    std::size_t repeat_index_{0};
    IcuRepeat2D repeat_2d_{};
    bool repeat_2d_active_{false};
    std::size_t repeat_2d_inner_{0};
    std::size_t repeat_2d_outer_{0};
    std::size_t repeat_2d_cooldown_{0};
    std::size_t notification_tokens_{0};
    std::deque<TaggedNotification> synchronized_notifications_{};
    bool notify_emitted_{false};
    std::size_t fetch_pc_{0};
    std::size_t program_end_pc_{0};
    std::size_t fetched_count_{0};
    std::size_t issued_count_{0};
    std::size_t synchronized_issued_count_{0};
    std::size_t synchronized_completed_count_{0};
    // Simulator trace/error timestamp. Raw physical queue roles never compare
    // this value to decide when to issue; only the software Legacy role does.
    std::size_t cycle_{0};
    // Compiler compatibility only. Physical ICU packets never observe this
    // value: old absolute starts are converted to queue-local NOPs before the
    // zero-relative packet is appended. QueueRole::Legacy remains the
    // explicitly non-hardware simulation path with absolute scheduling.
    std::size_t raw_compatibility_cycle_{0};
    bool raw_compatibility_cycle_known_{true};
    bool configured_{false};
    bool launched_{false};
    bool underflowed_{false};
    IcuQueueCycleTrace last_trace_{};
};

} // namespace ftlpu
