#include "softmax_dataflow_harness.hpp"
#include "smollm2_layer_phases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kSequence = 8;
constexpr std::size_t kHeads = 4;
constexpr std::size_t kHeadDim = 8;
constexpr std::size_t kHidden = kHeads * kHeadDim;
constexpr float kAttentionScale = 0.3535533905932738f; // 1 / sqrt(8)

constexpr std::array<std::size_t, 16> kWeightSlices {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
};
constexpr std::array<std::size_t, 2> kQuerySlices {32, 33};
constexpr std::array<std::size_t, 2> kScoreSlices {34, 35};
constexpr std::array<std::size_t, 2> kScoreMirrorSlices {36, 37};
constexpr std::array<std::size_t, 2> kXSlices {38, 39};
constexpr std::array<std::size_t, 2> kXMirrorSlices {40, 41};
constexpr std::array<std::size_t, 2> kMaxSlices {42, 43};
constexpr std::array<std::size_t, 2> kMaxMirrorSlices {44, 45};
constexpr std::array<std::size_t, 2> kProbabilitySlices {46, 47};
constexpr std::array<std::size_t, 2> kProbabilityMirrorSlices {48, 49};
constexpr std::array<std::size_t, 2> kOutputSlices {50, 51};

constexpr std::size_t kKeyWeightAddress = 64;
constexpr std::size_t kValueWeightAddress = 80;
constexpr std::size_t kQueryAddress = 96;
constexpr std::size_t kScoreAddress = 160;
constexpr std::size_t kXAddress = 192;
constexpr std::size_t kMaxAddress = 224;
constexpr std::size_t kProbabilityAddress = 256;
constexpr std::size_t kOutputAddress = 288;
constexpr std::size_t kQkAccumulatorAddress = 512;
constexpr std::size_t kPvAccumulatorAddress = 1024;

static_assert(kHidden == ftlpu::hw::kMxmRows);
static_assert(kSequence == ftlpu::hw::kLanesPerTile);
static_assert(kHeads == ftlpu::hw::kTileRows);

thread_local const std::vector<float>* active_queries = nullptr;
thread_local const std::vector<float>* active_keys = nullptr;
thread_local const std::vector<float>* active_values = nullptr;
thread_local bool quiet_verification = false;

std::size_t mem_queue(std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(
        ftlpu::Hemisphere::East, slice);
}

std::size_t mem_to_mxm_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mem_to_sxm_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 1
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t sxm_to_mem_latency(std::size_t slice)
{
    return ftlpu::hw::kMemEastBoundaryStreamRegisterColumn
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mem_to_vxm_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t vxm_to_mem_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 1;
}

ftlpu::SxmInstruction::StreamList streams(
    ftlpu::StreamDirection direction,
    std::size_t first, std::size_t count)
{
    auto result = ftlpu::SxmInstruction::StreamList{};
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto stream = direction == ftlpu::StreamDirection::East
            ? ftlpu::StreamId::East(first + index)
            : ftlpu::StreamId::West(first + index);
        result.push_back(ftlpu::SxmStreamId{stream.packed()});
    }
    return result;
}

class Schedule {
public:
    explicit Schedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(
        std::size_t slice, std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        auto& cursor = mem_[mem_queue(slice)];
        if (cycle < cursor) {
            throw std::logic_error(
                "full attention schedule overlaps MEM slice "
                + std::to_string(slice)
                + " at cycle " + std::to_string(cycle)
                + " (cursor " + std::to_string(cursor) + ")");
        }
        icu_.enqueue_mem_nop(mem_queue(slice), cycle - cursor);
        icu_.enqueue_mem(mem_queue(slice), std::move(instruction));
        cursor = cycle + 1;
        end_ = std::max(end_, cursor);
    }

    void mem_repeat_at(
        std::size_t slice, std::size_t cycle,
        ftlpu::MemInstruction instruction,
        std::size_t count, std::int64_t address_stride)
    {
        mem_at(slice, cycle, std::move(instruction));
        if (count > 1) {
            icu_.enqueue_mem_repeat(
                mem_queue(slice), count - 1, 1, address_stride);
        }
        mem_[mem_queue(slice)] = cycle + count;
        end_ = std::max(end_, cycle + count);
    }

    void mxm_load_at(
        std::size_t cycle, ftlpu::MxmControlInstruction instruction)
    {
        require_available(mxm_load_, cycle, "MXM load");
        icu_.enqueue_mxm_load_nop(0, cycle - mxm_load_);
        icu_.enqueue_mxm(0, std::move(instruction));
        mxm_load_ = cycle + 1;
        end_ = std::max(end_, mxm_load_);
    }

    void mxm_compute_at(
        std::size_t cycle, ftlpu::MxmControlInstruction instruction)
    {
        require_available(mxm_compute_, cycle, "MXM compute");
        icu_.enqueue_mxm_compute_nop(0, cycle - mxm_compute_);
        icu_.enqueue_mxm(0, std::move(instruction));
        mxm_compute_ = cycle + 1;
        end_ = std::max(end_, mxm_compute_);
    }

    void sxm_at(
        std::size_t cycle, ftlpu::SxmInstruction transpose,
        ftlpu::SxmInstruction permute)
    {
        require_available(sxm_transpose_, cycle, "SXM transpose");
        require_available(sxm_permute_, cycle, "SXM permute");
        icu_.enqueue_sxm_transpose_nop(cycle - sxm_transpose_);
        icu_.enqueue_sxm_permute_nop(cycle - sxm_permute_);
        icu_.enqueue_sxm_transpose(std::move(transpose));
        icu_.enqueue_sxm_permute(std::move(permute));
        sxm_transpose_ = cycle + 1;
        sxm_permute_ = cycle + 1;
        end_ = std::max(end_, cycle + 1);
    }

    void vxm_at(
        std::size_t stage, std::size_t cycle,
        ftlpu::VxmChainDepth depth,
        const ftlpu::VxmLaneAluInstruction& instruction)
    {
        auto& cursor = vxm_[stage];
        require_available(cursor, cycle, "VXM");
        icu_.enqueue_vxm_nop(stage, cycle - cursor);
        icu_.enqueue_vxm(stage, depth, instruction);
        cursor = cycle + 1;
        end_ = std::max(end_, cursor);
    }

    std::size_t end_cycle() const noexcept { return end_; }

private:
    static void require_available(
        std::size_t cursor, std::size_t cycle, const char* queue)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                std::string("full attention schedule overlaps ") + queue);
        }
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kVxmQueues> vxm_{};
    std::size_t mxm_load_{0};
    std::size_t mxm_compute_{0};
    std::size_t sxm_transpose_{0};
    std::size_t sxm_permute_{0};
    std::size_t end_{0};
};

float query_value(std::size_t query, std::size_t head, std::size_t dim)
{
    if (active_queries != nullptr) {
        return (*active_queries)[query * kHidden + head * kHeadDim + dim];
    }
    const auto pattern = static_cast<int>(
        (query * 11 + head * 7 + dim * 5) % 19) - 9;
    return static_cast<float>(pattern) / 8.0f;
}

float key_value(std::size_t head, std::size_t key, std::size_t dim)
{
    if (active_keys != nullptr) {
        return (*active_keys)[key * kHidden + head * kHeadDim + dim];
    }
    const auto pattern = static_cast<int>(
        (head * 13 + key * 5 + dim * 3) % 17) - 8;
    return static_cast<float>(pattern) / 8.0f;
}

float value_value(std::size_t head, std::size_t key, std::size_t dim)
{
    if (active_values != nullptr) {
        return (*active_values)[key * kHidden + head * kHeadDim + dim];
    }
    const auto pattern = static_cast<int>(
        (head * 17 + key * 7 + dim * 3) % 23) - 11;
    return static_cast<float>(pattern) / 16.0f;
}

float key_weight(std::size_t row, std::size_t column)
{
    const auto row_head = row / kHeadDim;
    const auto column_head = column / kSequence;
    if (row_head != column_head) return 0.0f;
    return key_value(column_head, column % kSequence, row % kHeadDim);
}

float value_weight(std::size_t row, std::size_t column)
{
    const auto row_head = row / kSequence;
    const auto column_head = column / kHeadDim;
    if (row_head != column_head) return 0.0f;
    return value_value(column_head, row % kSequence, column % kHeadDim);
}

void initialize_weight_matrix(
    ftlpu::TspSliceSystem& system, std::size_t base,
    float (*value)(std::size_t, std::size_t))
{
    for (std::size_t column_block = 0;
         column_block < ftlpu::hw::kMxmSupercellsPerPlane;
         ++column_block) {
        const auto address = base + column_block;
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto row = tile * ftlpu::hw::kLanesPerTile + lane;
                for (std::size_t local_column = 0;
                     local_column < ftlpu::hw::kMxmSupercellColumns;
                     ++local_column) {
                    const auto column = column_block
                            * ftlpu::hw::kMxmSupercellColumns
                        + local_column;
                    const auto bits = ftlpu::Bf16::from_float(
                        value(row, column)).bits();
                    system.initialize_mem_sram_lane_byte(
                        kWeightSlices[local_column * 2], tile,
                        address, lane,
                        static_cast<std::uint8_t>(bits & 0xffu));
                    system.initialize_mem_sram_lane_byte(
                        kWeightSlices[local_column * 2 + 1], tile,
                        address, lane,
                        static_cast<std::uint8_t>(bits >> 8));
                }
            }
        }
    }
}

void initialize_inputs(ftlpu::TspSliceSystem& system)
{
    initialize_weight_matrix(system, kKeyWeightAddress, key_weight);
    initialize_weight_matrix(system, kValueWeightAddress, value_weight);
    for (std::size_t query = 0; query < kSequence; ++query) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto hidden = tile * ftlpu::hw::kLanesPerTile + lane;
                const auto bits = ftlpu::Bf16::from_float(query_value(
                    query, hidden / kHeadDim, hidden % kHeadDim)).bits();
                system.initialize_mem_sram_lane_byte(
                    kQuerySlices[0], tile, kQueryAddress + query,
                    lane, static_cast<std::uint8_t>(bits & 0xffu));
                system.initialize_mem_sram_lane_byte(
                    kQuerySlices[1], tile, kQueryAddress + query,
                    lane, static_cast<std::uint8_t>(bits >> 8));
            }
        }
    }
}

ftlpu::VxmLaneAluInstruction basic(
    ftlpu::VxmAluOpcode opcode, ftlpu::VxmLaneOperand lhs,
    ftlpu::VxmLaneOperand rhs = ftlpu::VxmLaneOperand::Imm(0.0f),
    std::size_t repeat = 1)
{
    auto instruction = ftlpu::VxmLaneAluInstruction{opcode, lhs, rhs};
    instruction.precision = ftlpu::VxmAluPrecision::Float32;
    instruction.repeat_count = repeat;
    return instruction;
}

ftlpu::VxmLaneAluInstruction special(
    ftlpu::VxmSpecialAluOpcode opcode,
    ftlpu::VxmLaneOperand lhs, std::size_t repeat = 1)
{
    auto instruction = ftlpu::VxmLaneAluInstruction{opcode, lhs};
    instruction.repeat_count = repeat;
    return instruction;
}

ftlpu::VxmLaneAluInstruction accumulator(
    ftlpu::VxmAluOpcode opcode, bool reset,
    bool emit, std::size_t repeat)
{
    auto instruction = basic(
        opcode, ftlpu::VxmLaneOperand::Previous(),
        ftlpu::VxmLaneOperand::Acc(), repeat);
    instruction.accumulator_reset = reset;
    instruction.accumulator_write = true;
    instruction.accumulator_emit = emit;
    return instruction;
}

void schedule_weight_load(
    Schedule& schedule, std::size_t base, std::size_t first_cycle)
{
    for (std::size_t block = 0;
         block < ftlpu::hw::kMxmSupercellsPerPlane; ++block) {
        const auto load_cycle = first_cycle + block;
        for (std::size_t stream = 0;
             stream < kWeightSlices.size(); ++stream) {
            const auto slice = kWeightSlices[stream];
            schedule.mem_at(
                slice, load_cycle - mem_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    base + block, ftlpu::StreamId::East(stream)));
        }
        schedule.mxm_load_at(
            load_cycle,
            ftlpu::MxmControlInstruction::IWDirect16(0, block));
    }
}

void schedule_bf16_read(
    Schedule& schedule, const std::array<std::size_t, 2>& slices,
    std::size_t address, std::size_t stream,
    std::size_t input_cycle, std::size_t count,
    std::int64_t stride)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto slice = slices[byte];
        schedule.mem_repeat_at(
            slice, input_cycle - mem_to_vxm_latency(slice),
            ftlpu::MemInstruction::Read(
                address, ftlpu::StreamId::West(stream + byte)),
            count, stride);
    }
}

void schedule_bf16_write(
    Schedule& schedule, const std::array<std::size_t, 2>& slices,
    std::size_t address, std::size_t stream,
    std::size_t output_cycle, std::size_t count,
    std::int64_t stride)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto slice = slices[byte];
        schedule.mem_repeat_at(
            slice, output_cycle + vxm_to_mem_latency(slice),
            ftlpu::MemInstruction::Write(
                address, ftlpu::StreamId::East(stream + byte)),
            count, stride);
    }
}

void schedule_qk(Schedule& schedule)
{
    constexpr std::size_t kLoadCycle = 20;
    constexpr std::size_t kComputeCycle = 30;
    constexpr std::size_t kCaptureCycle =
        kComputeCycle + ftlpu::hw::kMxmSupercellsPerPlane
        + ftlpu::Mxm::kLocalMacStages - 1;
    constexpr std::size_t kFirstOutputCycle =
        kCaptureCycle + kSequence + 2;
    schedule_weight_load(schedule, kKeyWeightAddress, kLoadCycle);

    for (std::size_t query = 0; query < kSequence; ++query) {
        const auto cycle = kComputeCycle + query;
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kQuerySlices[byte];
            schedule.mem_at(
                slice, cycle - mem_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    kQueryAddress + query,
                    ftlpu::StreamId::East(byte)));
        }
        schedule.mxm_compute_at(
            cycle,
            ftlpu::MxmControlInstruction::Compute(
                0, 0, query * 2, kQkAccumulatorAddress, 1,
                ftlpu::MxmAccumulatorDestination::Stream,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmComputeMode::Vector, true,
                ftlpu::MxmAccumulatorOutputFormat::BFloat16));
    }

    const auto source = streams(ftlpu::StreamDirection::West, 0, 16);
    const auto internal = streams(ftlpu::StreamDirection::West, 16, 16);
    schedule.sxm_at(
        kCaptureCycle,
        ftlpu::SxmInstruction::Transpose(source, internal),
        ftlpu::SxmInstruction::Permute(
            internal, source, ftlpu::Permute320::identity_map()));

    for (std::size_t key = 0; key < kSequence; ++key) {
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto stream = key * 2 + byte;
            for (const auto& destination :
                 {kScoreSlices, kScoreMirrorSlices}) {
                const auto slice = destination[byte];
                const auto write = destination == kScoreMirrorSlices
                    ? ftlpu::MemInstruction::WriteTap(
                        kScoreAddress + key,
                        ftlpu::StreamId::West(stream))
                    : ftlpu::MemInstruction::Write(
                        kScoreAddress + key,
                        ftlpu::StreamId::West(stream));
                schedule.mem_at(
                    slice,
                    kFirstOutputCycle + key + sxm_to_mem_latency(slice),
                    write);
            }
        }
    }
}

void schedule_softmax(Schedule& schedule)
{
    constexpr auto kMxmPipelineDelay = ftlpu::Mxm::kLocalMacStages - 1;
    constexpr std::size_t kGenerateCycle = 70 + kMxmPipelineDelay;
    constexpr std::size_t kMaxCycle = 104 + kMxmPipelineDelay;
    constexpr std::size_t kSumCycle = 139 + kMxmPipelineDelay;
    constexpr std::size_t kNormalizeCycle = 170 + kMxmPipelineDelay;

    auto multiply = basic(
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::StreamBFloat16(),
        ftlpu::VxmLaneOperand::Imm(kAttentionScale), kSequence);
    auto add = basic(
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::Previous(),
        ftlpu::VxmLaneOperand::Imm(0.0f), kSequence);
    add.output_type = ftlpu::VxmCastTarget::BFloat16;
    add.output_stream = 0;
    schedule.vxm_at(0, kGenerateCycle, ftlpu::VxmChainDepth::Two, multiply);
    schedule.vxm_at(1, kGenerateCycle, ftlpu::VxmChainDepth::Two, add);
    const auto generate_input = kGenerateCycle + 1;
    schedule_bf16_read(
        schedule, kScoreSlices, kScoreAddress, 0,
        generate_input, kSequence, 1);
    schedule_bf16_read(
        schedule, kScoreMirrorSlices, kScoreAddress, 16,
        generate_input, kSequence, 1);
    schedule_bf16_write(
        schedule, kXSlices, kXAddress, 0,
        generate_input + 2, kSequence, 1);
    schedule_bf16_write(
        schedule, kXMirrorSlices, kXAddress, 8,
        generate_input + 2, kSequence, 1);

    schedule.vxm_at(
        0, kMaxCycle, ftlpu::VxmChainDepth::Two,
        basic(
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::StreamBFloat16(),
            ftlpu::VxmLaneOperand::Imm(0.0f), kSequence));
    schedule.vxm_at(
        1, kMaxCycle, ftlpu::VxmChainDepth::Two,
        accumulator(ftlpu::VxmAluOpcode::Max, true, false, 1));
    schedule.vxm_at(
        1, kMaxCycle + 1, ftlpu::VxmChainDepth::Two,
        accumulator(ftlpu::VxmAluOpcode::Max, false, false, kSequence - 2));
    auto max_final = accumulator(
        ftlpu::VxmAluOpcode::Max, false, true, 1);
    max_final.output_type = ftlpu::VxmCastTarget::BFloat16;
    max_final.output_stream = 0;
    schedule.vxm_at(
        1, kMaxCycle + 2, ftlpu::VxmChainDepth::Two, max_final);
    const auto max_input = kMaxCycle + 1;
    schedule_bf16_read(
        schedule, kXSlices, kXAddress, 0,
        max_input, kSequence, 1);
    schedule_bf16_read(
        schedule, kXMirrorSlices, kXAddress, 16,
        max_input, kSequence, 1);
    schedule_bf16_write(
        schedule, kMaxSlices, kMaxAddress, 0,
        max_input + kSequence, 1, 0);
    schedule_bf16_write(
        schedule, kMaxMirrorSlices, kMaxAddress, 8,
        max_input + kSequence, 1, 0);

    schedule.vxm_at(
        0, kSumCycle, ftlpu::VxmChainDepth::Four,
        basic(
            ftlpu::VxmAluOpcode::Subtract,
            ftlpu::VxmLaneOperand::StreamBFloat16(),
            ftlpu::VxmLaneOperand::StreamBFloat16(), kSequence));
    schedule.vxm_at(
        1, kSumCycle, ftlpu::VxmChainDepth::Four,
        special(
            ftlpu::VxmSpecialAluOpcode::Exp,
            ftlpu::VxmLaneOperand::Previous(), kSequence));
    schedule.vxm_at(
        2, kSumCycle, ftlpu::VxmChainDepth::Four,
        basic(
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Imm(0.0f), kSequence));
    schedule.vxm_at(
        3, kSumCycle, ftlpu::VxmChainDepth::Four,
        accumulator(ftlpu::VxmAluOpcode::Add, true, false, 1));
    schedule.vxm_at(
        3, kSumCycle + 1, ftlpu::VxmChainDepth::Four,
        accumulator(ftlpu::VxmAluOpcode::Add, false, false, kSequence - 2));
    schedule.vxm_at(
        3, kSumCycle + 2, ftlpu::VxmChainDepth::Four,
        accumulator(ftlpu::VxmAluOpcode::Add, false, true, 1));
    schedule.vxm_at(
        0, kSumCycle + 1, ftlpu::VxmChainDepth::Four,
        basic(
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::Feedback()));
    schedule.vxm_at(
        1, kSumCycle + 1, ftlpu::VxmChainDepth::Four,
        basic(ftlpu::VxmAluOpcode::Bypass,
              ftlpu::VxmLaneOperand::Previous()));
    schedule.vxm_at(
        2, kSumCycle + 1, ftlpu::VxmChainDepth::Four,
        basic(ftlpu::VxmAluOpcode::Bypass,
              ftlpu::VxmLaneOperand::Previous()));
    auto reciprocal = special(
        ftlpu::VxmSpecialAluOpcode::Reciprocal,
        ftlpu::VxmLaneOperand::Previous());
    reciprocal.local_scalar_write = true;
    schedule.vxm_at(
        3, kSumCycle + 3,
        ftlpu::VxmChainDepth::Four, reciprocal);
    const auto sum_input = kSumCycle + 1;
    schedule_bf16_read(
        schedule, kXSlices, kXAddress, 0,
        sum_input, kSequence, 1);
    schedule_bf16_read(
        schedule, kMaxSlices, kMaxAddress, 2,
        sum_input, kSequence, 0);
    schedule_bf16_read(
        schedule, kXMirrorSlices, kXAddress, 16,
        sum_input, kSequence, 1);
    schedule_bf16_read(
        schedule, kMaxMirrorSlices, kMaxAddress, 18,
        sum_input, kSequence, 0);

    schedule.vxm_at(
        0, kNormalizeCycle, ftlpu::VxmChainDepth::Four,
        basic(
            ftlpu::VxmAluOpcode::Subtract,
            ftlpu::VxmLaneOperand::StreamBFloat16(),
            ftlpu::VxmLaneOperand::StreamBFloat16(), kSequence));
    schedule.vxm_at(
        1, kNormalizeCycle, ftlpu::VxmChainDepth::Four,
        special(
            ftlpu::VxmSpecialAluOpcode::Exp,
            ftlpu::VxmLaneOperand::Previous(), kSequence));
    schedule.vxm_at(
        2, kNormalizeCycle, ftlpu::VxmChainDepth::Four,
        basic(
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Imm(0.0f), kSequence));
    auto normalize = basic(
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Previous(),
        ftlpu::VxmLaneOperand::Acc(), kSequence);
    normalize.output_type = ftlpu::VxmCastTarget::BFloat16;
    normalize.output_stream = 2;
    schedule.vxm_at(
        3, kNormalizeCycle,
        ftlpu::VxmChainDepth::Four, normalize);
    const auto normalize_input = kNormalizeCycle + 1;
    schedule_bf16_read(
        schedule, kXSlices, kXAddress, 0,
        normalize_input, kSequence, 1);
    schedule_bf16_read(
        schedule, kMaxSlices, kMaxAddress, 2,
        normalize_input, kSequence, 0);
    schedule_bf16_read(
        schedule, kXMirrorSlices, kXAddress, 16,
        normalize_input, kSequence, 1);
    schedule_bf16_read(
        schedule, kMaxMirrorSlices, kMaxAddress, 18,
        normalize_input, kSequence, 0);
    schedule_bf16_write(
        schedule, kProbabilitySlices, kProbabilityAddress, 2,
        normalize_input + 8, kSequence, 1);
    schedule_bf16_write(
        schedule, kProbabilityMirrorSlices, kProbabilityAddress, 10,
        normalize_input + 8, kSequence, 1);
}

void schedule_pv(Schedule& schedule)
{
    constexpr auto kMxmPipelineDelay = ftlpu::Mxm::kLocalMacStages - 1;
    constexpr std::size_t kValueLoadCycle = 205 + kMxmPipelineDelay;
    constexpr std::size_t kProbabilityCaptureCycle =
        225 + kMxmPipelineDelay;
    constexpr std::size_t kFirstProbabilityOutput =
        kProbabilityCaptureCycle + kSequence + 2;
    constexpr std::size_t kContextComputeCycle =
        kFirstProbabilityOutput + 1;
    constexpr std::size_t kContextCaptureCycle =
        kContextComputeCycle + ftlpu::hw::kMxmSupercellsPerPlane
        + ftlpu::Mxm::kLocalMacStages - 1;
    constexpr std::size_t kFirstContextOutput =
        kContextCaptureCycle + kSequence + 2;

    schedule_weight_load(
        schedule, kValueWeightAddress, kValueLoadCycle);
    for (std::size_t key = 0; key < kSequence; ++key) {
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kProbabilitySlices[byte];
            schedule.mem_at(
                slice,
                kProbabilityCaptureCycle + key
                    - mem_to_sxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    kProbabilityAddress + key,
                    ftlpu::StreamId::East(key * 2 + byte)));
        }
    }
    const auto east_source = streams(
        ftlpu::StreamDirection::East, 0, 16);
    const auto east_internal = streams(
        ftlpu::StreamDirection::East, 16, 16);
    schedule.sxm_at(
        kProbabilityCaptureCycle,
        ftlpu::SxmInstruction::Transpose(
            east_source, east_internal),
        ftlpu::SxmInstruction::Permute(
            east_internal, east_source,
            ftlpu::Permute320::identity_map()));

    for (std::size_t query = 0; query < kSequence; ++query) {
        schedule.mxm_compute_at(
            kContextComputeCycle + query,
            ftlpu::MxmControlInstruction::Compute(
                0, query * 2, query * 2,
                kPvAccumulatorAddress, 1,
                ftlpu::MxmAccumulatorDestination::Stream,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmComputeMode::Vector, true,
                ftlpu::MxmAccumulatorOutputFormat::BFloat16));
    }

    const auto west_source = streams(
        ftlpu::StreamDirection::West, 0, 16);
    const auto west_internal = streams(
        ftlpu::StreamDirection::West, 16, 16);
    schedule.sxm_at(
        kContextCaptureCycle,
        ftlpu::SxmInstruction::Transpose(
            west_source, west_internal),
        ftlpu::SxmInstruction::Permute(
            west_internal, west_source,
            ftlpu::Permute320::identity_map()));

    for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kOutputSlices[byte];
            schedule.mem_at(
                slice,
                kFirstContextOutput + dim + sxm_to_mem_latency(slice),
                ftlpu::MemInstruction::Write(
                    kOutputAddress + dim,
                    ftlpu::StreamId::West(dim * 2 + byte)));
        }
    }
}

ftlpu::Bf16 read_bf16(
    const ftlpu::TspSliceSystem& system,
    const std::array<std::size_t, 2>& slices,
    std::size_t tile, std::size_t address, std::size_t lane)
{
    const auto low = system.read_mem_sram_lane_byte(
        slices[0], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        slices[1], tile, address, lane);
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8));
}

bool verify(
    const ftlpu::TspSliceSystem& system,
    const ftlpu::VxmSpecialAlu& lut)
{
    float max_probability_error = 0.0f;
    float max_context_error = 0.0f;
    for (std::size_t head = 0; head < kHeads; ++head) {
        for (std::size_t query = 0; query < kSequence; ++query) {
            auto score = std::array<float, kSequence>{};
            auto x = std::array<float, kSequence>{};
            for (std::size_t key = 0; key < kSequence; ++key) {
                float sum = 0.0f;
                for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
                    const auto q = ftlpu::Bf16::from_float(
                        query_value(query, head, dim)).to_float();
                    const auto k = ftlpu::Bf16::from_float(
                        key_value(head, key, dim)).to_float();
                    sum += q * k;
                }
                score[key] = ftlpu::Bf16::from_float(sum).to_float();
                const auto actual_score = read_bf16(
                    system, kScoreSlices, head,
                    kScoreAddress + key, query);
                if (actual_score.bits()
                    != ftlpu::Bf16::from_float(score[key]).bits()) {
                    std::cerr << "QK score mismatch head=" << head
                              << " query=" << query
                              << " key=" << key
                              << " actual=" << actual_score.to_float()
                              << " expected=" << score[key] << '\n';
                    return false;
                }
                x[key] = ftlpu::Bf16::from_float(
                    score[key] * kAttentionScale).to_float();
            }
            const auto maximum =
                *std::max_element(x.begin(), x.end());
            auto exp_values = std::array<float, kSequence>{};
            float exp_sum = 0.0f;
            for (std::size_t key = 0; key < kSequence; ++key) {
                exp_values[key] = lut.execute(
                    ftlpu::VxmSpecialAluOpcode::Exp,
                    x[key] - maximum);
                exp_sum += exp_values[key];
            }
            const auto inverse_sum = lut.execute(
                ftlpu::VxmSpecialAluOpcode::Reciprocal, exp_sum);
            auto probability = std::array<float, kSequence>{};
            float probability_sum = 0.0f;
            for (std::size_t key = 0; key < kSequence; ++key) {
                const auto expected = ftlpu::Bf16::from_float(
                    exp_values[key] * inverse_sum);
                const auto actual = read_bf16(
                    system, kProbabilitySlices, head,
                    kProbabilityAddress + key, query);
                if (actual.bits() != expected.bits()) {
                    std::cerr << "Softmax probability mismatch head=" << head
                              << " query=" << query
                              << " key=" << key
                              << " actual=" << actual.to_float()
                              << " expected=" << expected.to_float() << '\n';
                    return false;
                }
                probability[key] = expected.to_float();
                probability_sum += actual.to_float();
                const auto exact = std::exp(x[key] - maximum);
                float exact_sum = 0.0f;
                for (const auto value : x) {
                    exact_sum += std::exp(value - maximum);
                }
                max_probability_error = std::max(
                    max_probability_error,
                    std::fabs(actual.to_float() - exact / exact_sum));
            }
            if (std::fabs(probability_sum - 1.0f) > 0.03f) {
                std::cerr << "Softmax sum mismatch head=" << head
                          << " query=" << query
                          << " sum=" << probability_sum << '\n';
                return false;
            }
            for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
                float context = 0.0f;
                for (std::size_t key = 0; key < kSequence; ++key) {
                    context += probability[key]
                        * ftlpu::Bf16::from_float(
                            value_value(head, key, dim)).to_float();
                }
                const auto expected = ftlpu::Bf16::from_float(context);
                const auto actual = read_bf16(
                    system, kOutputSlices, head,
                    kOutputAddress + dim, query);
                if (actual.bits() != expected.bits()) {
                    std::cerr << "P*V context mismatch head=" << head
                              << " query=" << query
                              << " dim=" << dim
                              << " actual=" << actual.to_float()
                              << " expected=" << expected.to_float() << '\n';
                    return false;
                }
                max_context_error = std::max(
                    max_context_error,
                    std::fabs(actual.to_float() - context));
            }
        }
    }
    if (!quiet_verification) {
        std::cout
            << "SmolLM2 full attention passed: MEM Q/K/V -> MXM QK^T -> "
               "SXM score layout -> MEM -> VXM scaled Softmax -> MEM -> "
               "SXM probability layout -> MXM P*V -> SXM -> MEM, "
            << "heads=4 sequence=8 head_dim=8"
            << " max_probability_error=" << max_probability_error
            << " max_context_error=" << max_context_error << '\n';
    }
    return true;
}

struct TileResult {
    std::array<float, kSequence * kHidden> output{};
    std::size_t cycles{0};
};

TileResult run_attention_tile(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& queries,
    const std::vector<float>& keys,
    const std::vector<float>& values)
{
    if (queries.size() != kSequence * kHidden
        || keys.size() != kSequence * kHidden
        || values.size() != kSequence * kHidden) {
        throw std::invalid_argument("attention tile must be [8,32]");
    }

    system.reset_execution_state();
    active_queries = &queries;
    active_keys = &keys;
    active_values = &values;
    quiet_verification = true;
    const auto reference_lut =
        ftlpu::test::softmax_dataflow::configure_luts(system);
    initialize_inputs(system);
    auto schedule = Schedule(system.icu());
    schedule_qk(schedule);
    schedule_softmax(schedule);
    schedule_pv(schedule);
    const auto run_cycles = schedule.end_cycle()
        + ftlpu::hw::kTileRows + 24;
    for (std::size_t cycle = 0; cycle < run_cycles; ++cycle) {
        system.tick({});
    }
    if (!verify(system, reference_lut)) {
        throw std::runtime_error("attention tile black-box verification failed");
    }

    auto result = TileResult{};
    result.cycles = run_cycles;
    for (std::size_t token = 0; token < kSequence; ++token) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto head = hidden / kHeadDim;
            const auto dim = hidden % kHeadDim;
            result.output[token * kHidden + hidden] = read_bf16(
                system, kOutputSlices, head,
                kOutputAddress + dim, token).to_float();
        }
    }
    active_queries = nullptr;
    active_keys = nullptr;
    active_values = nullptr;
    quiet_verification = false;
    return result;
}

void write_phase_trace(
    const std::filesystem::path& path,
    std::size_t cycles, const char* detail)
{
    if (path.empty()) return;
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write attention phase trace");
    }
    output << "start,end,resource,detail\n"
           << "0," << cycles << ",Attention," << detail << '\n';
}

bool full_layer_attention_tiles()
{
    const auto* value = std::getenv("FTLPU_FULL_INTEGRATION");
    return value != nullptr && std::string {value} == "1";
}

} // namespace

#ifndef FTLPU_SMOLLM2_LAYER_PHASE_ONLY
int main()
try {
    auto system = ftlpu::TspSliceSystem{};
    const auto reference_lut =
        ftlpu::test::softmax_dataflow::configure_luts(system);
    initialize_inputs(system);
    auto schedule = Schedule(system.icu());
    schedule_qk(schedule);
    schedule_softmax(schedule);
    schedule_pv(schedule);
    auto timing = integration_timing::SystemGanttTrace {};
    const auto collect_timing =
        integration_timing::SystemGanttTrace::enabled();

    const auto run_cycles = schedule.end_cycle()
        + ftlpu::hw::kTileRows + 24;
    for (std::size_t cycle = 0; cycle < run_cycles; ++cycle) {
        try {
            system.tick({});
            if (collect_timing) timing.capture(system);
        } catch (const std::exception& error) {
            std::cerr << "full attention hardware schedule failed at cycle "
                      << cycle << ": " << error.what() << '\n';
            return 1;
        }
    }
    const auto passed = verify(system, reference_lut);
    if (collect_timing) {
        timing.write(
            "smollm2_full_attention_system",
            "SmolLM2 full Attention system timing");
    }
    return passed ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "full attention setup failed: " << error.what() << '\n';
    return 1;
}
#endif

namespace ftlpu::test::smollm2_layer {

PhaseResult run_prefill_attention(
    TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path)
{
    if (input.size() != kPrefillLength * kHidden) {
        throw std::invalid_argument("prefill attention input must be [128,576]");
    }
    auto result = PhaseResult{};
    result.output = input;
    result.key_cache = input;
    result.value_cache = input;
    auto query_tile = std::vector<float>(::kSequence * ::kHidden);
    auto key_tile = query_tile;
    auto value_tile = query_tile;

    for (std::size_t token_base = 0;
         token_base < kPrefillLength; token_base += ::kSequence) {
        const auto hidden_limit = ::full_layer_attention_tiles()
            ? kHidden : ::kHidden;
        for (std::size_t hidden_base = 0;
             hidden_base < hidden_limit; hidden_base += ::kHidden) {
            for (std::size_t token = 0; token < ::kSequence; ++token) {
                for (std::size_t hidden = 0; hidden < ::kHidden; ++hidden) {
                    const auto value = input[
                        (token_base + token) * kHidden
                        + hidden_base + hidden];
                    const auto index = token * ::kHidden + hidden;
                    query_tile[index] = value;
                    key_tile[index] = value;
                    value_tile[index] = value;
                }
            }
            const auto tile = ::run_attention_tile(
                system, query_tile, key_tile, value_tile);
            result.cycles += tile.cycles;
            for (std::size_t token = 0; token < ::kSequence; ++token) {
                for (std::size_t hidden = 0; hidden < ::kHidden; ++hidden) {
                    result.output[
                        (token_base + token) * kHidden
                        + hidden_base + hidden]
                        = tile.output[token * ::kHidden + hidden];
                }
            }
        }
    }
    ::write_phase_trace(
        trace_path, result.cycles,
        ::full_layer_attention_tiles()
            ? "prefill [128,576] full 8-token x 32-hidden black-box tiles"
            : "prefill [128,576] representative hardware tile per token block");
    return result;
}

PhaseResult run_decode_attention(
    TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path,
    const std::filesystem::path&,
    const std::vector<float>& prefill_keys,
    const std::vector<float>& prefill_values)
{
    if (input.size() != kHidden
        || prefill_keys.size() != kPrefillLength * kHidden
        || prefill_values.size() != kPrefillLength * kHidden) {
        throw std::invalid_argument("decode attention/cache shape mismatch");
    }
    auto result = PhaseResult{};
    result.output = input;
    result.key_cache = prefill_keys;
    result.value_cache = prefill_values;
    result.key_cache.insert(result.key_cache.end(), input.begin(), input.end());
    result.value_cache.insert(result.value_cache.end(), input.begin(), input.end());
    auto query_tile = std::vector<float>(::kSequence * ::kHidden);
    auto key_tile = query_tile;
    auto value_tile = query_tile;
    constexpr auto history_base = kPrefillLength - (::kSequence - 1);

    const auto hidden_limit = ::full_layer_attention_tiles()
        ? kHidden : ::kHidden;
    for (std::size_t hidden_base = 0;
         hidden_base < hidden_limit; hidden_base += ::kHidden) {
        for (std::size_t token = 0; token < ::kSequence; ++token) {
            for (std::size_t hidden = 0; hidden < ::kHidden; ++hidden) {
                const auto index = token * ::kHidden + hidden;
                const auto value = token + 1 == ::kSequence
                    ? input[hidden_base + hidden]
                    : prefill_keys[
                        (history_base + token) * kHidden
                        + hidden_base + hidden];
                query_tile[index] = value;
                key_tile[index] = value;
                value_tile[index] = token + 1 == ::kSequence
                    ? input[hidden_base + hidden]
                    : prefill_values[
                        (history_base + token) * kHidden
                        + hidden_base + hidden];
            }
        }
        const auto tile = ::run_attention_tile(
            system, query_tile, key_tile, value_tile);
        result.cycles += tile.cycles;
        for (std::size_t hidden = 0; hidden < ::kHidden; ++hidden) {
            result.output[hidden_base + hidden] = tile.output[
                (::kSequence - 1) * ::kHidden + hidden];
        }
    }
    ::write_phase_trace(
        trace_path, result.cycles,
        ::full_layer_attention_tiles()
            ? "decode full hidden last-7 cache + new token black-box tiles"
            : "decode representative last-7 cache + new token hardware tile");
    return result;
}

} // namespace ftlpu::test::smollm2_layer
