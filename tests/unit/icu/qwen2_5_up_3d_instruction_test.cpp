#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/icu/distributed_queue.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

ftlpu::IcuCoordinate3D coordinate_for_ordinal(
    std::size_t ordinal,
    const ftlpu::IcuLoop3D& loop)
{
    ftlpu::IcuCoordinate3D coordinate{};
    for (std::size_t dimension = 0;
         dimension < ftlpu::IcuLoop3D::kDimensions; ++dimension) {
        coordinate.index[dimension] = ordinal % loop.counts[dimension];
        ordinal /= loop.counts[dimension];
    }
    require(ordinal == 0, "3-D issue ordinal exceeded its launch domain");
    return coordinate;
}

std::size_t point_count(const ftlpu::IcuLoop3D& loop)
{
    return loop.counts[0] * loop.counts[1] * loop.counts[2];
}

std::size_t issue_cycle(
    const ftlpu::IcuLoop3D& loop,
    const ftlpu::IcuCoordinate3D& coordinate)
{
    require(loop.start_cycle == 0,
        "hardware FU loop carried an absolute start cycle");
    return coordinate[0] * loop.cycle_strides[0]
        + coordinate[1] * loop.cycle_strides[1]
        + coordinate[2] * loop.cycle_strides[2];
}

template <typename Queue>
constexpr bool accepts_mem_3d = requires(
    Queue& queue, ftlpu::MemIcuInstruction instruction) {
    queue.push_mem_3d(instruction);
};

template <typename Queue>
constexpr bool accepts_mxm_load_3d = requires(
    Queue& queue, ftlpu::MxmLoadIcuInstruction instruction) {
    queue.push_mxm_load_3d(instruction);
};

template <typename Queue>
constexpr bool accepts_mxm_dequant_3d = requires(
    Queue& queue, ftlpu::MxmDequantIcuInstruction instruction) {
    queue.push_mxm_dequant_3d(instruction);
};

template <typename Queue>
constexpr bool accepts_mxm_compute_3d = requires(
    Queue& queue, ftlpu::MxmComputeIcuInstruction instruction) {
    queue.push_mxm_compute_3d(instruction);
};

template <typename Queue, typename Instruction>
constexpr bool accepts_native_instruction = requires(
    Queue& queue, Instruction instruction) {
    queue.push_instruction(instruction);
};

template <typename Queue>
void prefetch_program(Queue& queue, std::size_t physical_word_count)
{
    static_assert(Queue::fetch_latency == 1,
        "this per-cycle frontend check models the architectural one-cycle fetch");
    require(queue.imem_occupancy() == physical_word_count,
        "FU 3-D program occupied the wrong number of physical iMEM words");
    queue.configure(ftlpu::IcuProgramDescriptor {
        0, queue.imem_occupancy()});

    // One local iMEM word may start and one may complete per cycle.  Keep the
    // packet as physical words through fetch and IQ; decode happens only when
    // the complete packet reaches the IQ head.
    for (std::size_t cycle = 0; cycle <= physical_word_count; ++cycle) {
        require(queue.cycle() == cycle,
            "FU 3-D frontend cycle changed during prefetch");
        queue.prefetch_only();
        const auto& trace = queue.last_trace();
        require(trace.cycle == cycle
                && trace.action == ftlpu::IcuQueueAction::PrefetchOnly,
            "FU 3-D frontend did not use the prefetch-only path");

        if (cycle < physical_word_count) {
            require(trace.fetch_started_pc.has_value()
                    && *trace.fetch_started_pc == cycle,
                "FU 3-D frontend did not start exactly one physical word");
        } else {
            require(!trace.fetch_started_pc.has_value(),
                "FU 3-D frontend fetched past its descriptor");
        }

        if (cycle == 0) {
            require(!trace.fetch_completed_pc.has_value(),
                "FU 3-D physical word completed without fetch latency");
        } else {
            require(trace.fetch_completed_pc.has_value()
                    && *trace.fetch_completed_pc == cycle - 1,
                "FU 3-D frontend did not complete one physical word per cycle");
        }

        require(queue.fetched_count()
                    == (cycle < physical_word_count
                            ? cycle
                            : physical_word_count)
                && queue.iq_occupancy() == queue.fetched_count()
                && queue.pending_fetch_count()
                    == (cycle < physical_word_count ? 1U : 0U),
            "FU 3-D physical word did not traverse fetch latency into the IQ");
    }
}

template <typename Queue>
void require_mxm_native_enqueue_rejected(
    ftlpu::MxmControlInstruction instruction,
    const char* expected_diagnostic)
{
    Queue queue;
    bool rejected = false;
    try {
        queue.push_instruction(instruction);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find(expected_diagnostic)
            != std::string::npos;
    }
    require(rejected,
        "wrong-role MXM native instruction was accepted by the encoder");
    require(queue.imem_occupancy() == 0,
        "rejected MXM native instruction changed local iMEM");
}

template <typename Queue>
void require_mxm_native_raw_rejected(
    const ftlpu::MxmControlInstruction& instruction,
    const char* expected_diagnostic)
{
    typename Queue::Entry word{};
    // Model a loader that writes a syntactically valid MXM native payload
    // directly into the wrong local iMEM, bypassing push_instruction().
    ftlpu::detail::write_raw_icu_bits(word, 2, 64,
        ftlpu::isa::encode_mxm_instruction(instruction));

    Queue queue;
    queue.write_imem(0, word);
    prefetch_program(queue, 1);
    bool rejected = false;
    try {
        static_cast<void>(queue.tick());
    } catch (const ftlpu::StaticScheduleError& error) {
        rejected = std::string(error.what()).find(expected_diagnostic)
            != std::string::npos;
    }
    require(rejected,
        "wrong-role MXM raw word was accepted by the IQ decoder");
}

template <typename Queue>
void fetch_one_descriptor(Queue& queue,
    std::size_t& total_physical_words,
    std::size_t leading_nop_cycles = 0)
{
    constexpr auto kPacketWords = Queue::three_d_packet_word_count;
    const auto expectedWords = kPacketWords
        + (leading_nop_cycles == 0 ? 0U : 1U);
    require(queue.imem_occupancy() == expectedWords,
        "FU 3-D program did not contain the expected NOP and packet words");
    total_physical_words += expectedWords;
    prefetch_program(queue, expectedWords);
}

template <typename Queue, typename Validate>
std::size_t run_one_descriptor(
    Queue& queue,
    const ftlpu::IcuLoop3D& loop,
    std::size_t leading_nop_cycles,
    ftlpu::IcuQueueAction wait_action,
    ftlpu::IcuQueueAction issue_action,
    std::size_t& total_physical_words,
    Validate validate)
{
    fetch_one_descriptor(
        queue, total_physical_words, leading_nop_cycles);
    const auto executionStart = queue.cycle();
    const auto activationCycle = executionStart + leading_nop_cycles;
    const auto descriptorPc = leading_nop_cycles == 0 ? 0U : 1U;

    bool saw_activation = false;
    bool saw_nop = leading_nop_cycles == 0;
    bool saw_wait = false;
    std::size_t issues = 0;
    const auto expected_points = point_count(loop);
    const auto final_coordinate = ftlpu::IcuCoordinate3D {{
        loop.counts[0] - 1,
        loop.counts[1] - 1,
        loop.counts[2] - 1,
    }};
    const auto final_cycle = activationCycle
        + issue_cycle(loop, final_coordinate);
    const auto expects_wait =
        issue_cycle(loop, final_coordinate) + 1 > expected_points;

    while (!queue.done()) {
        const auto cycle = queue.cycle();
        const auto instruction = queue.tick();
        const auto& trace = queue.last_trace();

        if (trace.three_d_decode_header_pc.has_value()) {
            require(trace.iq_before == Queue::three_d_packet_word_count
                    && trace.iq_after == 0
                    && *trace.three_d_decode_header_pc == descriptorPc
                    && trace.three_d_decode_word_count
                        == Queue::three_d_packet_word_count,
                "FU 3-D packet did not decode atomically from the IQ");
            require(!saw_activation && cycle == activationCycle
                    && trace.action == issue_action,
                "FU 3-D descriptor did not activate at relative cycle zero");
            saw_activation = true;
        }
        if (!saw_activation) {
            require(!instruction.has_value(),
                "FU instruction issued before its leading NOP retired");
            if (trace.action == ftlpu::IcuQueueAction::Nop
                || trace.action == ftlpu::IcuQueueAction::NopWait)
                saw_nop = true;
        }
        if (trace.action == wait_action) saw_wait = true;

        if (instruction.has_value()) {
            require(issues < expected_points,
                "FU 3-D descriptor emitted too many native instructions");
            const auto coordinate = coordinate_for_ordinal(issues, loop);
            require(cycle == activationCycle + issue_cycle(loop, coordinate),
                "FU 3-D descriptor emitted on the wrong cycle");
            require(trace.action == issue_action
                    && trace.issue_pc.has_value()
                    && *trace.issue_pc == descriptorPc,
                "FU 3-D native issue lost its descriptor PC/action");
            validate(*instruction, coordinate);
            ++issues;
        }

        require(cycle <= final_cycle,
            "FU 3-D descriptor did not retire after its final issue");
    }

    require(saw_activation && saw_nop && saw_wait == expects_wait,
        "FU 3-D descriptor exposed the wrong NOP/activation/wait sequence");
    require(issues == expected_points
            && queue.issued_count() == expected_points,
        "FU 3-D descriptor expanded to the wrong native issue count");
    return issues;
}

} // namespace

int main()
try {
    using namespace ftlpu;

    // This is the direct-lowering contract for Qwen2.5-1.5B, seq=32, FFN
    // Up only. The loops below construct FU coarse instructions directly
    // from shape, placement, and schedule parameters. No native instruction
    // list is materialized and then pattern-compressed; native instructions
    // exist only as the ICU expands a decoded context on each issue cycle.
    //   M=32 rows, K=1536 (48 reduction tiles), N=8960
    //   split as 140 output-pair tiles per hemisphere.
    // Weight pages are already resident in the selected MEM banks.  C2C/DMA
    // page movement belongs to its own ICU queues and is outside this
    // compute-kernel instruction count.
    // The delays below are emitted as queue-local NOPs. They preserve the real
    // three-counter shapes, K double buffering, and output delay without
    // placing a global cycle number in any FU packet.
    constexpr std::size_t kRows = 32;
    constexpr std::size_t kReductionTiles = 1536 / 32;
    constexpr std::size_t kPairsPerHemisphere = 8960 / 32 / 2;
    constexpr std::size_t kTileCycles = 32;
    constexpr std::size_t kPairCycles =
        kReductionTiles * kTileCycles;
    constexpr std::size_t kWeightStart = 8;
    constexpr std::size_t kLoadStart = 32;
    constexpr std::size_t kComputeStart = 64;
    constexpr std::size_t kMxmOutputDelay = 16;
    constexpr std::size_t kResultStart = kComputeStart
        + (kReductionTiles - 1) * kTileCycles + kMxmOutputDelay;

    static_assert(kRows == hw::kMxmRows);
    static_assert(kReductionTiles == 48);
    static_assert(kPairsPerHemisphere == 140);

    using MemQueue = DistributedIcuQueue<MemInstruction,
        hw::kIcuMemInstructionBits,
        hw::kIcuMemImemDepth,
        hw::kIcuMemIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuMemMacroContextDepth,
        IcuQueueRole::Mem>;
    using MxmLoadQueue = DistributedIcuQueue<MxmControlInstruction,
        hw::kIcuMxmInstructionBits,
        hw::kIcuMxmImemDepth,
        hw::kIcuMxmIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuMxmMacroContextDepth,
        IcuQueueRole::MxmLoad>;
    using MxmComputeQueue = DistributedIcuQueue<MxmControlInstruction,
        hw::kIcuMxmInstructionBits,
        hw::kIcuMxmImemDepth,
        hw::kIcuMxmIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuMxmMacroContextDepth,
        IcuQueueRole::MxmCompute>;
    using DequantQueue = DistributedIcuQueue<MxmDequantInstruction,
        hw::kIcuMxmInstructionBits,
        hw::kIcuMxmImemDepth,
        hw::kIcuMxmIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuMxmMacroContextDepth,
        IcuQueueRole::MxmDequant>;

    static_assert(accepts_mem_3d<MemQueue>);
    static_assert(!accepts_mxm_load_3d<MemQueue>);
    static_assert(!accepts_mxm_dequant_3d<MemQueue>);
    static_assert(!accepts_mxm_compute_3d<MemQueue>);
    static_assert(!accepts_mem_3d<MxmLoadQueue>);
    static_assert(accepts_mxm_load_3d<MxmLoadQueue>);
    static_assert(!accepts_mxm_dequant_3d<MxmLoadQueue>);
    static_assert(!accepts_mxm_compute_3d<MxmLoadQueue>);
    static_assert(!accepts_mem_3d<MxmComputeQueue>);
    static_assert(!accepts_mxm_load_3d<MxmComputeQueue>);
    static_assert(!accepts_mxm_dequant_3d<MxmComputeQueue>);
    static_assert(accepts_mxm_compute_3d<MxmComputeQueue>);
    static_assert(!accepts_mem_3d<DequantQueue>);
    static_assert(!accepts_mxm_load_3d<DequantQueue>);
    static_assert(accepts_mxm_dequant_3d<DequantQueue>);
    static_assert(!accepts_mxm_compute_3d<DequantQueue>);

    static_assert(accepts_native_instruction<MxmLoadQueue,
        MxmControlInstruction>);
    static_assert(!accepts_native_instruction<MxmLoadQueue,
        MxmDequantInstruction>);
    static_assert(accepts_native_instruction<DequantQueue,
        MxmDequantInstruction>);
    static_assert(!accepts_native_instruction<DequantQueue,
        MxmControlInstruction>);
    static_assert(accepts_native_instruction<MxmComputeQueue,
        MxmControlInstruction>);
    static_assert(!accepts_native_instruction<MxmComputeQueue,
        MxmDequantInstruction>);

    static_assert(std::is_same_v<MemQueue::Entry,
        isa::EncodedMemIcu3DWord>);
    static_assert(sizeof(MemQueue::Entry) * 8
        == hw::kIcuMemInstructionBits);
    static_assert(!std::is_constructible_v<MemQueue::Entry,
        isa::EncodedMxmLoadIcu3DWord>);
    static_assert(!std::is_constructible_v<MemQueue::Entry,
        isa::EncodedMxmDequantIcu3DWord>);
    static_assert(!std::is_constructible_v<MemQueue::Entry,
        isa::EncodedMxmComputeIcu3DWord>);
    static_assert(!std::is_constructible_v<MxmLoadQueue::Entry,
        isa::EncodedMemIcu3DWord>);
    static_assert(std::is_same_v<MxmLoadQueue::Entry,
        isa::EncodedMxmLoadIcu3DWord>);
    static_assert(sizeof(MxmLoadQueue::Entry) * 8
        == hw::kIcuMxmInstructionBits);
    static_assert(!std::is_constructible_v<MxmLoadQueue::Entry,
        isa::EncodedMxmDequantIcu3DWord>);
    static_assert(!std::is_constructible_v<MxmLoadQueue::Entry,
        isa::EncodedMxmComputeIcu3DWord>);
    static_assert(!std::is_constructible_v<DequantQueue::Entry,
        isa::EncodedMemIcu3DWord>);
    static_assert(!std::is_constructible_v<DequantQueue::Entry,
        isa::EncodedMxmLoadIcu3DWord>);
    static_assert(std::is_same_v<DequantQueue::Entry,
        isa::EncodedMxmDequantIcu3DWord>);
    static_assert(sizeof(DequantQueue::Entry) * 8
        == hw::kIcuMxmInstructionBits);
    static_assert(!std::is_constructible_v<DequantQueue::Entry,
        isa::EncodedMxmComputeIcu3DWord>);
    static_assert(!std::is_constructible_v<MxmComputeQueue::Entry,
        isa::EncodedMemIcu3DWord>);
    static_assert(!std::is_constructible_v<MxmComputeQueue::Entry,
        isa::EncodedMxmLoadIcu3DWord>);
    static_assert(!std::is_constructible_v<MxmComputeQueue::Entry,
        isa::EncodedMxmDequantIcu3DWord>);
    static_assert(std::is_same_v<MxmComputeQueue::Entry,
        isa::EncodedMxmComputeIcu3DWord>);
    static_assert(sizeof(MxmComputeQueue::Entry) * 8
        == hw::kIcuMxmInstructionBits);

    // The LOAD and COMPUTE ICUs share the 64-bit MXM native codec, so queue
    // identity must constrain both host-side encoding and raw IQ decode.
    // DEQUANT has its own native instruction type and raw Entry type; exercise
    // that independent codec here as well as the compile-time isolation above.
    for (const auto& instruction : std::array {
             MxmControlInstruction::Compute(),
             MxmControlInstruction::AccumulatorRead(0),
             MxmControlInstruction::DecodeStreamCompute(0)}) {
        require_mxm_native_enqueue_rejected<MxmLoadQueue>(
            instruction, "MXM LOAD ICU");
    }
    for (const auto& instruction : std::array {
             MxmControlInstruction::IW(),
             MxmControlInstruction::DecodeLoadActivation()}) {
        require_mxm_native_enqueue_rejected<MxmComputeQueue>(
            instruction, "MXM COMPUTE ICU");
    }

    require_mxm_native_raw_rejected<MxmLoadQueue>(
        MxmControlInstruction::Compute(), "MXM LOAD ICU");
    require_mxm_native_raw_rejected<MxmLoadQueue>(
        MxmControlInstruction::DecodeStreamCompute(0), "MXM LOAD ICU");
    require_mxm_native_raw_rejected<MxmComputeQueue>(
        MxmControlInstruction::IW(), "MXM COMPUTE ICU");
    require_mxm_native_raw_rejected<MxmComputeQueue>(
        MxmControlInstruction::DecodeLoadActivation(), "MXM COMPUTE ICU");

    {
        MxmLoadQueue queue;
        queue.push_instruction(MxmControlInstruction::IW(1, 2));
        queue.push_instruction(
            MxmControlInstruction::DecodeLoadActivation());
        prefetch_program(queue, 2);
        const auto iw = queue.tick();
        const auto decode = queue.tick();
        require(iw.has_value() && decode.has_value()
                && iw->opcode == MxmControlOpcode::IW
                && iw->weight_buffer == 1 && iw->weight_column == 2
                && decode->opcode == MxmControlOpcode::Decode
                && decode->decode_operation
                    == MxmDecodeOperation::LoadActivation
                && queue.done(),
            "valid MXM LOAD native words did not round-trip through its IQ");
    }
    {
        MxmComputeQueue queue;
        queue.push_instruction(MxmControlInstruction::Compute());
        queue.push_instruction(MxmControlInstruction::AccumulatorRead(0));
        queue.push_instruction(MxmControlInstruction::DecodeStreamCompute(0));
        prefetch_program(queue, 3);
        const auto compute = queue.tick();
        const auto accumulator_read = queue.tick();
        const auto decode = queue.tick();
        require(compute.has_value() && accumulator_read.has_value()
                && decode.has_value()
                && compute->opcode == MxmControlOpcode::Compute
                && accumulator_read->opcode
                    == MxmControlOpcode::AccumulatorRead
                && decode->opcode == MxmControlOpcode::Decode
                && decode->decode_operation
                    == MxmDecodeOperation::StreamCompute
                && queue.done(),
            "valid MXM COMPUTE native words did not round-trip through its IQ");
    }
    {
        DequantQueue queue;
        constexpr std::uint16_t kScaleBits = 0x3f00;
        queue.push_instruction(MxmDequantInstruction::ScaleBits(kScaleBits));
        prefetch_program(queue, 1);
        const auto scale = queue.tick();
        require(scale.has_value() && scale->scale_bf16 == kScaleBits
                && queue.done(),
            "valid MXM DEQUANT native word did not round-trip through its IQ");
    }

    // A finite descriptor-context pool must backpressure at packet granularity:
    // the second packet remains intact in the IQ until the first context retires.
    using OneContextMemQueue = DistributedIcuQueue<MemInstruction,
        hw::kIcuMemInstructionBits, 8, 8, 1, 1, IcuQueueRole::Mem>;
    {
        const IcuLoop3D first_loop {0, {2, 1, 1}, {4, 1, 1}};
        const IcuLoop3D second_loop {0, {1, 1, 1}, {1, 1, 1}};
        OneContextMemQueue queue;
        queue.push_mem_3d(MemIcuInstruction::Read3D(
            first_loop, MemIcuAddress3D::Affine(0, {1, 0, 0}),
            StreamId::East(0)));
        queue.push_mem_3d(MemIcuInstruction::Read3D(
            second_loop, MemIcuAddress3D::Affine(64, {0, 0, 0}),
            StreamId::East(1)));
        require(queue.imem_occupancy()
                    == 2 * OneContextMemQueue::three_d_packet_word_count,
            "two MEM commands did not occupy two complete encoded packets");
        prefetch_program(queue, queue.imem_occupancy());

        bool saw_first_decode = false;
        bool saw_blocked_packet = false;
        bool saw_second_decode = false;
        std::size_t native_issues = 0;
        while (!queue.done()) {
            const auto instruction = queue.tick();
            const auto& trace = queue.last_trace();
            if (trace.three_d_decode_header_pc.has_value()) {
                require(trace.three_d_decode_word_count
                            == OneContextMemQueue::three_d_packet_word_count,
                    "finite-context queue decoded a partial MEM packet");
                if (*trace.three_d_decode_header_pc == 0) {
                    require(!saw_first_decode && trace.iq_before == 6
                            && trace.iq_after == 3,
                        "first MEM packet was not decoded atomically");
                    saw_first_decode = true;
                } else {
                    require(*trace.three_d_decode_header_pc == 3
                            && !saw_second_decode
                            && trace.iq_before == 3
                            && trace.iq_after == 0,
                        "second MEM packet lost its header PC while blocked");
                    saw_second_decode = true;
                }
            }
            if (saw_first_decode && !saw_second_decode
                && queue.active_3d_context_count() == 1
                && trace.iq_before == 3 && trace.iq_after == 3)
                saw_blocked_packet = true;
            if (instruction.has_value()) {
                const auto expected_pc = native_issues < 2 ? 0U : 3U;
                const auto expected_address = native_issues < 2
                    ? native_issues
                    : 64U;
                require(trace.issue_pc == expected_pc
                        && instruction->opcode == MemOpcode::Read
                        && instruction->address == expected_address,
                    "finite-context MEM expansion lost its packet PC/data");
                ++native_issues;
            }
        }
        require(saw_first_decode && saw_blocked_packet
                && saw_second_decode && native_issues == 3
                && queue.peak_active_3d_contexts() == 1,
            "finite 3-D context backpressure coverage was incomplete");
    }

    // A following control/native word may already be resident in the IQ, but
    // program order keeps it behind the active 3-D context until that context
    // emits its final native instruction.
    using OrderedMemQueue = DistributedIcuQueue<MemInstruction,
        hw::kIcuMemInstructionBits, 8, 8, 1, 1, IcuQueueRole::Mem>;
    {
        OrderedMemQueue queue;
        queue.push_mem_3d(MemIcuInstruction::Read3D(
            IcuLoop3D {0, {2, 1, 1}, {4, 1, 1}},
            MemIcuAddress3D::Affine(0, {1, 0, 0}),
            StreamId::East(0)));
        queue.push_nop(2);
        queue.push_instruction(MemInstruction::Read(
            96, StreamId::East(2).packed()));
        require(queue.imem_occupancy()
                    == OrderedMemQueue::three_d_packet_word_count + 2,
            "3-D/NOP/native sequence did not encode to five raw words");
        prefetch_program(queue, queue.imem_occupancy());

        std::size_t three_d_issues = 0;
        bool saw_nop = false;
        bool saw_nop_wait = false;
        bool saw_native = false;
        while (!queue.done()) {
            const auto cycle = queue.cycle();
            const auto instruction = queue.tick();
            const auto& trace = queue.last_trace();

            if (trace.three_d_decode_header_pc.has_value()) {
                require(cycle == 6
                        && *trace.three_d_decode_header_pc == 0
                        && trace.three_d_decode_word_count == 3
                        && trace.iq_before == 5 && trace.iq_after == 2,
                    "ordered sequence did not atomically decode its 3-D head");
            }
            if (cycle >= 7 && cycle <= 9) {
                require(trace.iq_before == 2 && trace.iq_after == 2,
                    "NOP/native raw words advanced before the 3-D context retired");
            }

            if (trace.action == IcuQueueAction::Nop) {
                require(cycle == 11 && three_d_issues == 2
                        && trace.issue_pc == 3
                        && trace.iq_before == 2 && trace.iq_after == 1,
                    "NOP did not execute immediately after the 3-D context");
                saw_nop = true;
            } else if (trace.action == IcuQueueAction::NopWait) {
                require(cycle == 12 && saw_nop
                        && trace.iq_before == 1 && trace.iq_after == 1,
                    "NOP delay did not retain the following native raw word");
                saw_nop_wait = true;
            }

            if (!instruction.has_value()) continue;
            if (trace.action == IcuQueueAction::Mem3DIssue) {
                require(!saw_nop && !saw_native
                        && trace.issue_pc == 0
                        && instruction->address == three_d_issues,
                    "3-D issue was reordered with a later raw word");
                ++three_d_issues;
            } else {
                require(trace.action == IcuQueueAction::FunctionalIssue
                        && cycle == 13 && three_d_issues == 2
                        && saw_nop && saw_nop_wait
                        && trace.issue_pc == 4
                        && instruction->opcode == MemOpcode::Read
                        && instruction->address == 96
                        && instruction->stream
                            == StreamId::East(2).packed(),
                    "native raw word did not execute after 3-D and NOP");
                saw_native = true;
            }
        }
        require(three_d_issues == 2 && saw_nop && saw_nop_wait
                && saw_native && queue.issued_count() == 3,
            "3-D/NOP/native ordered coexistence coverage was incomplete");
    }

    // Loader corruption that leaves only the first physical word of a packet
    // must be diagnosed at the IQ head instead of waiting forever.
    using TruncatedMemQueue = DistributedIcuQueue<MemInstruction,
        hw::kIcuMemInstructionBits, 4, 3, 1, 1, IcuQueueRole::Mem>;
    {
        const auto packet = isa::encode_mem_icu_3d_instruction(
            MemIcuInstruction::Read3D(
                IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
                MemIcuAddress3D::Affine(0, {0, 0, 0}),
                StreamId::East(0)));
        TruncatedMemQueue queue;
        queue.write_imem(0, packet.words[0]);
        prefetch_program(queue, 1);
        bool diagnosed_truncation = false;
        try {
            static_cast<void>(queue.tick());
        } catch (const std::exception& error) {
            diagnosed_truncation =
                std::string(error.what()).find("truncated")
                != std::string::npos;
        }
        require(diagnosed_truncation,
            "truncated FU 3-D packet did not produce an explicit diagnostic");
    }

    constexpr std::size_t kMemQueuesPerHemisphere =
        hw::kMemSliceColumns * hw::kMemBanksPerSlice;
    const auto mem_queue_id = [](std::size_t hemisphere,
                                  std::size_t slice,
                                  std::size_t bank) {
        return hemisphere * kMemQueuesPerHemisphere
            + slice * hw::kMemBanksPerSlice + bank;
    };

    std::set<std::size_t> used_mem_queues;
    std::size_t physical_imem_words = 0;
    std::size_t coarse_weight_reads = 0;
    std::size_t coarse_activation_reads = 0;
    std::size_t coarse_result_writes = 0;
    std::size_t fine_weight_reads = 0;
    std::size_t fine_activation_reads = 0;
    std::size_t fine_result_writes = 0;

    struct WeightGroup {
        std::size_t first_pair;
        std::size_t pair_count;
        std::size_t page;
        std::size_t bank;
        std::size_t first_slice;
    };
    constexpr std::array<WeightGroup, 4> kWeightGroups {{
        {0, 42, 0, 1, 36},
        {42, 42, 0, 1, 44},
        {84, 42, 1, 0, 36},
        {126, 14, 1, 0, 44},
    }};

    // Each weight descriptor is one READ_3D in one physical MEM bank.  Its
    // outer counter uses a two-pair blocked layout:
    //   address = r + 8*k + 384*floor(pair/2) + 4*(pair mod 2).
    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres;
         ++hemisphere) {
        for (const auto& group : kWeightGroups) {
            require(group.page == (group.bank == 0 ? 1U : 0U),
                "Qwen Up weight page/bank placement changed");
            for (std::size_t local_slice = 0; local_slice < 8;
                 ++local_slice) {
                const auto slice = group.first_slice + local_slice;
                const auto queue_id = mem_queue_id(
                    hemisphere, slice, group.bank);
                require(used_mem_queues.insert(queue_id).second,
                    "Qwen Up assigned two weight descriptors to one MEM ICU");

                const auto startCycle =
                    kWeightStart + group.first_pair * kPairCycles;
                const IcuLoop3D loop {
                    0,
                    {4, kReductionTiles, group.pair_count},
                    {1, kTileCycles, kPairCycles},
                };
                const auto stream = StreamId::East(8 + local_slice);
                MemQueue queue;
                queue.push_nop(startCycle);
                queue.push_mem_3d(MemIcuInstruction::Read3D(
                    loop,
                    MemIcuAddress3D::BlockedOuter(
                        0, 1, 8, 2, 4, 384),
                    stream));
                ++coarse_weight_reads;
                fine_weight_reads += run_one_descriptor(
                    queue, loop, startCycle,
                    IcuQueueAction::Mem3DWait,
                    IcuQueueAction::Mem3DIssue,
                    physical_imem_words,
                    [&](const MemInstruction& instruction,
                        const IcuCoordinate3D& coordinate) {
                        const auto expected_address = coordinate[0]
                            + 8 * coordinate[1]
                            + 384 * (coordinate[2] / 2)
                            + 4 * (coordinate[2] % 2);
                        require(instruction.opcode == MemOpcode::Read
                                && instruction.address == expected_address
                                && instruction.stream == stream.packed()
                                && !instruction.preserve_stream,
                            "Qwen Up weight READ_3D expanded incorrectly");
                    });
            }
        }
    }

    constexpr std::array<std::size_t, 8> kActivationStartOffsets {
        0, 1, 3, 4, 6, 7, 9, 10,
    };
    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres;
         ++hemisphere) {
        for (std::size_t slice = 0; slice < 16; ++slice) {
            constexpr std::size_t kActivationBank = 1;
            const auto queue_id = mem_queue_id(
                hemisphere, slice, kActivationBank);
            require(used_mem_queues.insert(queue_id).second,
                "Qwen Up activation descriptor reused a MEM ICU");
            const auto startCycle =
                kComputeStart + kActivationStartOffsets[slice / 2];
            const IcuLoop3D loop {
                0,
                {4, kReductionTiles, kPairsPerHemisphere},
                {8, kTileCycles, kPairCycles},
            };
            const auto stream = StreamId::East(16 + slice % 2);
            MemQueue queue;
            queue.push_nop(startCycle);
            queue.push_mem_3d(MemIcuInstruction::Read3D(
                loop,
                MemIcuAddress3D::Affine(0, {1, 4, 0}),
                stream));
            ++coarse_activation_reads;
            fine_activation_reads += run_one_descriptor(
                queue, loop, startCycle,
                IcuQueueAction::Mem3DWait,
                IcuQueueAction::Mem3DIssue,
                physical_imem_words,
                [&](const MemInstruction& instruction,
                    const IcuCoordinate3D& coordinate) {
                    require(instruction.opcode == MemOpcode::Read
                            && instruction.address
                                == coordinate[0] + 4 * coordinate[1]
                            && instruction.stream == stream.packed(),
                        "Qwen Up activation READ_3D expanded incorrectly");
                });
        }
    }

    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres;
         ++hemisphere) {
        for (std::size_t local_slice = 0; local_slice < 2;
             ++local_slice) {
            constexpr std::size_t kResultBank = 0;
            const auto slice = 8 + local_slice;
            const auto queue_id = mem_queue_id(
                hemisphere, slice, kResultBank);
            require(used_mem_queues.insert(queue_id).second,
                "Qwen Up result descriptor reused a MEM ICU");
            const IcuLoop3D loop {
                0,
                {kRows, kPairsPerHemisphere, 1},
                {1, kPairCycles, 1},
            };
            const auto stream = StreamId::West(
                (hemisphere == 0 ? 12 : 20) + local_slice);
            MemQueue queue;
            queue.push_nop(kResultStart);
            queue.push_mem_3d(MemIcuInstruction::Write3D(
                loop,
                MemIcuAddress3D::Affine(0, {1, 32, 0}),
                stream));
            ++coarse_result_writes;
            fine_result_writes += run_one_descriptor(
                queue, loop, kResultStart,
                IcuQueueAction::Mem3DWait,
                IcuQueueAction::Mem3DIssue,
                physical_imem_words,
                [&](const MemInstruction& instruction,
                    const IcuCoordinate3D& coordinate) {
                    require(instruction.opcode == MemOpcode::Write
                            && instruction.address
                                == coordinate[0] + 32 * coordinate[1]
                            && instruction.stream == stream.packed()
                            && !instruction.preserve_stream,
                        "Qwen Up result WRITE_3D expanded incorrectly");
                });
        }
    }

    require(used_mem_queues.size() == 100,
        "Qwen Up did not use the expected 100 physical MEM ICU queues");

    std::size_t coarse_loads = 0;
    std::size_t coarse_dequants = 0;
    std::size_t fine_loads = 0;
    std::size_t fine_dequants = 0;
    const IcuLoop3D load_loop {
        0,
        {4, kReductionTiles, kPairsPerHemisphere},
        {1, kTileCycles, kPairCycles},
    };
    const auto scale = MxmDequantInstruction::Scale(0.0078125f);

    // One LOAD_3D and one DEQUANT_3D per hemisphere share an identical
    // launch domain.  Tick them together to prove cycle-for-cycle lockstep.
    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres;
         ++hemisphere) {
        MxmLoadQueue load;
        DequantQueue dequant;
        load.push_nop(kLoadStart);
        dequant.push_nop(kLoadStart);
        load.push_mxm_load_3d(MxmLoadIcuInstruction::Load3D(
            load_loop,
            0,
            MxmIcuBufferMode::ToggleDimension1,
            0,
            {1, 0, 0},
            8,
            MxmWeightInputMode::Int8DequantBf16));
        dequant.push_mxm_dequant_3d(
            MxmDequantIcuInstruction::Dequant3D(load_loop, scale));
        ++coarse_loads;
        ++coarse_dequants;

        fetch_one_descriptor(load, physical_imem_words, kLoadStart);
        fetch_one_descriptor(dequant, physical_imem_words, kLoadStart);
        require(load.cycle() == dequant.cycle(),
            "load/dequant frontends lost cycle alignment");
        const auto loadActivationCycle = load.cycle() + kLoadStart;

        bool saw_load_activation = false;
        bool saw_dequant_activation = false;
        std::size_t pair_issues = 0;
        while (!load.done() || !dequant.done()) {
            require(load.cycle() == dequant.cycle(),
                "load/dequant ICU clocks diverged");
            const auto cycle = load.cycle();
            const auto load_instruction = load.tick();
            const auto dequant_instruction = dequant.tick();
            require(load_instruction.has_value()
                    == dequant_instruction.has_value(),
                "LOAD_3D and DEQUANT_3D did not issue in lockstep");

            if (load.last_trace().three_d_decode_header_pc.has_value()) {
                require(load.last_trace().iq_before
                            == MxmLoadQueue::three_d_packet_word_count
                        && load.last_trace().iq_after == 0
                        && load.last_trace()
                               .three_d_decode_header_pc == 1
                        && load.last_trace()
                               .three_d_decode_word_count
                            == MxmLoadQueue::three_d_packet_word_count
                        && load.last_trace().action
                            == IcuQueueAction::MxmLoad3DIssue
                        && cycle == loadActivationCycle,
                    "LOAD_3D did not decode atomically from its IQ");
                require(!saw_load_activation,
                    "LOAD_3D decoded its packet more than once");
                saw_load_activation = true;
            }
            if (dequant.last_trace().three_d_decode_header_pc.has_value()) {
                require(dequant.last_trace().iq_before
                            == DequantQueue::three_d_packet_word_count
                        && dequant.last_trace().iq_after == 0
                        && dequant.last_trace()
                               .three_d_decode_header_pc == 1
                        && dequant.last_trace()
                               .three_d_decode_word_count
                            == DequantQueue::three_d_packet_word_count
                        && dequant.last_trace().action
                            == IcuQueueAction::MxmDequant3DIssue
                        && cycle == loadActivationCycle,
                    "DEQUANT_3D did not decode atomically from its IQ");
                require(!saw_dequant_activation,
                    "DEQUANT_3D decoded its packet more than once");
                saw_dequant_activation = true;
            }

            if (load_instruction.has_value()) {
                const auto coordinate = coordinate_for_ordinal(
                    pair_issues, load_loop);
                require(cycle
                        == loadActivationCycle
                            + issue_cycle(load_loop, coordinate),
                    "load/dequant lockstep pair issued on the wrong cycle");
                require(load.last_trace().action
                            == IcuQueueAction::MxmLoad3DIssue
                        && dequant.last_trace().action
                            == IcuQueueAction::MxmDequant3DIssue
                        && load.last_trace().issue_pc == 1
                        && dequant.last_trace().issue_pc == 1,
                    "load/dequant issue trace lost its hardware opcode/PC");
                require(load_instruction->opcode == MxmControlOpcode::IW
                        && load_instruction->weight_buffer
                            == (coordinate[1] & 1U)
                        && load_instruction->weight_column == coordinate[0]
                        && load_instruction->weight_stream_base == 8
                        && load_instruction->weight_input_mode
                            == MxmWeightInputMode::Int8DequantBf16,
                    "Qwen Up LOAD_3D expanded buffer parity/column incorrectly");
                require(dequant_instruction->scale_bf16 == scale.scale_bf16,
                    "Qwen Up DEQUANT_3D changed its BF16 scale");
                ++pair_issues;
            }
        }
        require(pair_issues == point_count(load_loop)
                && load.issued_count() == point_count(load_loop)
                && dequant.issued_count() == point_count(load_loop),
            "Qwen Up load/dequant descriptor count is incorrect");
        fine_loads += pair_issues;
        fine_dequants += pair_issues;
    }

    std::size_t coarse_computes = 0;
    std::size_t fine_computes = 0;
    std::size_t terminal_computes = 0;
    const IcuLoop3D compute_loop {
        0,
        {kRows, kReductionTiles, kPairsPerHemisphere},
        {1, kTileCycles, kPairCycles},
    };
    const MxmComputeIcuMode partial_mode {
        MxmAccumulatorDestination::Sram,
        false,
        MxmAccumulatorOutputFormat::Float32,
    };
    const MxmComputeIcuMode terminal_mode {
        MxmAccumulatorDestination::Stream,
        true,
        MxmAccumulatorOutputFormat::BFloat16,
    };

    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres;
         ++hemisphere) {
        const auto result_stream_base = hemisphere == 0 ? 12U : 20U;
        MxmComputeQueue compute;
        compute.push_nop(kComputeStart);
        compute.push_mxm_compute_3d(MxmComputeIcuInstruction::Compute3D(
            compute_loop,
            0,
            MxmIcuBufferMode::ToggleDimension1,
            16,
            result_stream_base,
            32,
            {0, 0, 0},
            1,
            MxmDataFormat::BFloat16,
            partial_mode,
            1,
            terminal_mode));
        ++coarse_computes;
        fine_computes += run_one_descriptor(
            compute, compute_loop, kComputeStart,
            IcuQueueAction::MxmCompute3DWait,
            IcuQueueAction::MxmCompute3DIssue,
            physical_imem_words,
            [&](const MxmControlInstruction& instruction,
                const IcuCoordinate3D& coordinate) {
                const bool terminal =
                    coordinate[1] + 1 == kReductionTiles;
                require(instruction.opcode == MxmControlOpcode::Compute
                        && instruction.weight_buffer
                            == (coordinate[1] & 1U)
                        && instruction.activation_stream_base == 16
                        && instruction.stream_base == result_stream_base
                        && instruction.accumulator_address == 32
                        && instruction.accumulator_row_stride == 1
                        && instruction.data_format == MxmDataFormat::BFloat16,
                    "Qwen Up COMPUTE_3D expanded common operands incorrectly");
                require(instruction.accumulator_destination
                            == (terminal
                                    ? MxmAccumulatorDestination::Stream
                                    : MxmAccumulatorDestination::Sram)
                        && instruction.accumulator_clear == terminal
                        && instruction.accumulator_output_format
                            == (terminal
                                    ? MxmAccumulatorOutputFormat::BFloat16
                                    : MxmAccumulatorOutputFormat::Float32),
                    "Qwen Up COMPUTE_3D terminal-k mode is incorrect");
                if (terminal) ++terminal_computes;
            });
    }

    constexpr std::size_t kExpectedWeightFine = 430080;
    constexpr std::size_t kExpectedActivationFine = 860160;
    constexpr std::size_t kExpectedResultFine = 17920;
    constexpr std::size_t kExpectedLoadFine = 53760;
    constexpr std::size_t kExpectedDequantFine = 53760;
    constexpr std::size_t kExpectedComputeFine = 430080;
    constexpr std::size_t kExpectedTerminalFine = 8960;

    require(coarse_weight_reads == 64
            && coarse_activation_reads == 32
            && coarse_result_writes == 4
            && coarse_loads == 2
            && coarse_dequants == 2
            && coarse_computes == 2,
        "Qwen Up hardware FU coarse instruction mix is not 64/32/4/2/2/2");
    require(coarse_weight_reads + coarse_activation_reads
                + coarse_result_writes + coarse_loads + coarse_dequants
                + coarse_computes
            == 106,
        "Qwen Up hardware FU schedule is not exactly 106 coarse instructions");
    require(physical_imem_words == 418,
        "Qwen Up schedule did not encode to 312 packet words plus 106 NOPs");
    require(fine_weight_reads == kExpectedWeightFine
            && fine_activation_reads == kExpectedActivationFine
            && fine_result_writes == kExpectedResultFine
            && fine_loads == kExpectedLoadFine
            && fine_dequants == kExpectedDequantFine
            && fine_computes == kExpectedComputeFine
            && terminal_computes == kExpectedTerminalFine,
        "Qwen Up hardware FU fine issue totals changed");

    std::cout << "Qwen2.5-1.5B seq32 Up FU 3-D instructions passed: "
              << "coarse=106"
              << " physical_words=" << physical_imem_words
              << " weight=" << fine_weight_reads
              << " activation=" << fine_activation_reads
              << " result=" << fine_result_writes
              << " load=" << fine_loads
              << " dequant=" << fine_dequants
              << " compute=" << fine_computes
              << " terminal=" << terminal_computes << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Qwen2.5 Up FU 3-D instruction test failed: "
              << error.what() << '\n';
    return 1;
}
