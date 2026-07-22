#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "vxm_alu_program.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSeqLen = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kQueryHeads = 9;
constexpr std::size_t kKvHeads = 3;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;
constexpr std::size_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr std::size_t kTile = ftlpu::hw::kMxmRows;
constexpr float kRopeTheta = 100000.0f;
constexpr std::size_t kWeightToIwLatency = 14;
constexpr std::size_t kActivationLatency = 5;
constexpr std::size_t kMxm0AccumulatorLatency = 6;
constexpr std::size_t kMxm1AccumulatorLatency = 5;
constexpr std::size_t kMxmInputBlockIssueCycles =
    kTile + 2 * (ftlpu::hw::kTileRows - 1);
constexpr std::size_t kRopeWriteLatency = 2;
constexpr std::size_t kCastWriteLatency = 1;
constexpr std::array<std::size_t, 8> kWeightSlices {0, 4, 8, 12, 16, 20, 24, 28};
constexpr std::array<std::size_t, 4> kActivationSlices {32, 33, 34, 35};
constexpr std::array<std::size_t, 4> kOutputSlices {0, 1, 2, 3};
constexpr std::array<std::size_t, 4> kRopeTableSlices {4, 5, 6, 7};
// RoPE tables are no longer needed after Q/K projection.  QK reuses these
// slices as a second K copy so both local MXMs can consume K every cycle.
constexpr std::array<std::size_t, 4> kKeyReplicaSlices {4, 5, 6, 7};
constexpr std::array<std::size_t, 4> kScaledScoreSlices {8, 9, 10, 11};
constexpr std::array<std::size_t, 4> kExpScoreSlices {12, 13, 14, 15};
constexpr std::array<std::size_t, 8> kContextSlices {20, 21, 22, 23, 24, 25, 26, 27};
constexpr std::array<std::size_t, 4> kAttentionOutputSlices {28, 29, 30, 31};
constexpr std::size_t kRopeTableAddressBase = 7000;
constexpr std::size_t kScoreAddressBase = 3000;
constexpr std::size_t kContextAccumulatorAddressBase = 2000;
constexpr std::size_t kOutputAccumulatorAddressBase = 2200;
constexpr std::size_t kProbabilityPackAddressBase = 6000;
constexpr std::size_t kQueryIwAddressBase = 7600;
constexpr std::size_t kValuePackAddressBase = 7800;
constexpr std::size_t kProbabilityDiagonalAddressBase = kRopeTableAddressBase;
constexpr std::size_t kMemToSxmLatency = 12;
constexpr std::size_t kMemToMxmLatency = 13;
constexpr float kAttentionScale = 1.0f / 8.0f;
constexpr std::size_t kSoftmaxOutputStream = 8;
constexpr std::array<std::array<std::size_t, 16>, 2> kQueryIwSlices {{
    {{0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33}},
    {{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35}},
}};
constexpr std::array<std::array<std::size_t, 16>, 2> kValuePackSlices {{
    {{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33}},
    {{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35}},
}};

static_assert(kSeqLen % kTile == 0);
static_assert(kHidden % kTile == 0);
static_assert(kHeadDim == 2 * kTile);
static_assert(kKvWidth == 6 * kTile);
static_assert(kQueryWidth == 18 * kTile);
static_assert(kRopeTableAddressBase + kSeqLen <= ftlpu::hw::kSramDepthRows);
static_assert(kScoreAddressBase + kQueryHeads * (kSeqLen / kTile) * kSeqLen
    <= ftlpu::hw::kSramDepthRows);
static_assert(kOutputAccumulatorAddressBase + kSeqLen <= ftlpu::hw::kSramDepthRows);
static_assert(kProbabilityPackAddressBase
        + kQueryHeads * (kSeqLen / kTile)
            * (kSeqLen / ftlpu::hw::kLanesPerTile)
    <= kRopeTableAddressBase);
static_assert(kQueryIwAddressBase
        + kQueryHeads * (kSeqLen / kTile) * ftlpu::hw::kTileRows
    <= ftlpu::hw::kSramDepthRows);
static_assert(kProbabilityDiagonalAddressBase
        + kQueryHeads * (kSeqLen / kTile) * (kSeqLen / kTile)
            * ftlpu::hw::kTileRows
    <= kQueryIwAddressBase);

enum class Projection : std::size_t { Query, Key, Value };

constexpr std::size_t projection_heads(Projection projection)
{
    return projection == Projection::Query ? kQueryHeads : kKvHeads;
}

constexpr std::size_t projection_width(Projection projection)
{
    return projection_heads(projection) * kHeadDim;
}

constexpr std::size_t projection_head_offset(Projection projection)
{
    switch (projection) {
    case Projection::Query: return 0;
    case Projection::Key: return kQueryHeads;
    case Projection::Value: return kQueryHeads + kKvHeads;
    }
    return 0;
}

const char* projection_name(Projection projection)
{
    switch (projection) {
    case Projection::Query: return "Q";
    case Projection::Key: return "K";
    case Projection::Value: return "V";
    }
    return "?";
}

std::size_t x_index(std::size_t token, std::size_t hidden)
{
    return token * kHidden + hidden;
}

std::size_t weight_index(Projection projection, std::size_t hidden, std::size_t column)
{
    return hidden * projection_width(projection) + column;
}

float input_value(std::size_t token, std::size_t hidden)
{
    return static_cast<float>(static_cast<int>((token * 7 + hidden * 5) % 29) - 14) * 0.046875f;
}

float weight_value(Projection projection, std::size_t hidden, std::size_t column)
{
    const auto p = static_cast<std::size_t>(projection);
    const auto raw = static_cast<int>(
        (hidden * (13 + p * 4) + column * (7 + p * 2) + p * 11) % 43) - 21;
    return static_cast<float>(raw)
        * (0.007f + static_cast<float>((column + p * 3) % 9) * 0.001f);
}

float output_weight_value(std::size_t hidden, std::size_t column)
{
    const auto raw = static_cast<int>((hidden * 19 + column * 11 + 5) % 47) - 23;
    return static_cast<float>(raw)
        * (0.006f + static_cast<float>(column % 7) * 0.001f);
}

class OfflineSchedule {
public:
    struct TraceEvent {
        std::size_t start{0};
        std::size_t end{0};
        std::string resource;
        std::string detail;
    };

    explicit OfflineSchedule(ftlpu::InstructionControlUnit& icu)
        : icu_(icu)
        , trace_enabled_(std::getenv("FTLPU_SCHEDULE_TRACE") != nullptr)
    {
    }

    void mem_at(std::size_t queue, std::size_t cycle, ftlpu::MemInstruction instruction)
    {
        require_available(
            mem_[queue],
            cycle,
            "MEM q" + std::to_string(queue)
                + " opcode=" + std::to_string(static_cast<std::size_t>(instruction.opcode))
                + " address=" + std::to_string(instruction.address)
                + " stream=" + std::to_string(instruction.stream));
        pad(mem_[queue], cycle, [&](std::size_t count) { icu_.enqueue_mem_nop(queue, count); });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
        trace_mem(queue, cycle, cycle + 1, instruction);
    }

    void mem_repeat_at(
        std::size_t queue,
        std::size_t cycle,
        ftlpu::MemInstruction instruction,
        std::size_t count,
        std::int64_t stride)
    {
        mem_at(queue, cycle, instruction);
        if (count > 1) icu_.enqueue_mem_repeat(queue, count - 1, 1, stride);
        advance(mem_[queue], cycle + count);
        if (count > 1) {
            trace_mem(queue, cycle + 1, cycle + count, instruction, stride);
        }
    }

    void dequant_at(std::size_t cycle, const ftlpu::test::DequantSpec& instruction)
    {
        ftlpu::test::enqueue_dequant(icu_, vxm_, cycle, instruction);
        end_cycle_ = std::max(end_cycle_, cycle + 2);
        trace(cycle, cycle + 1, "VXM.ALU0-7", "dequant mul");
        trace(cycle + 1, cycle + 2, "VXM.ALU8-15", "dequant cast FP16");
    }

    void rope_at(std::size_t cycle, ftlpu::Hemisphere hemisphere)
    {
        const auto alu_base = hemisphere == ftlpu::Hemisphere::East ? 0u : 8u;
        if (alu_base + 5 >= ftlpu::VxmLane::kAluCount) {
            throw std::out_of_range("RoPE exceeds VXM ALU range");
        }
        const auto instruction = [&](ftlpu::VxmAluOpcode opcode,
                                     ftlpu::VxmLaneOperand lhs,
                                     ftlpu::VxmLaneOperand rhs,
                                     ftlpu::VxmCastTarget cast = ftlpu::VxmCastTarget::Float32,
                                     std::optional<std::size_t> output = std::nullopt) {
            return ftlpu::VxmLaneAluInstruction {
                opcode, lhs, rhs, 1.0f, 0, cast, output, hemisphere, hemisphere};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamFloat32(32),
            ftlpu::VxmLaneOperand::StreamFloat16(40)));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base + 1, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamFloat32(36),
            ftlpu::VxmLaneOperand::StreamFloat16(42)));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base + 3, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamFloat32(36),
            ftlpu::VxmLaneOperand::StreamFloat16(40)));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base + 4, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamFloat32(32),
            ftlpu::VxmLaneOperand::StreamFloat16(42)));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base + 2, cycle + 1, instruction(
            ftlpu::VxmAluOpcode::Subtract,
            ftlpu::VxmLaneOperand::Alu(alu_base),
            ftlpu::VxmLaneOperand::Alu(alu_base + 1),
            ftlpu::VxmCastTarget::Float16, 0));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base + 5, cycle + 1, instruction(
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(alu_base + 3),
            ftlpu::VxmLaneOperand::Alu(alu_base + 4),
            ftlpu::VxmCastTarget::Float16, 2));
        end_cycle_ = std::max(end_cycle_, cycle + 2);
        for (const auto alu : {0u, 1u, 3u, 4u}) {
            trace(cycle, cycle + 1,
                "VXM.ALU" + std::to_string(alu_base + alu), "RoPE mul");
        }
        trace(cycle + 1, cycle + 2,
            "VXM.ALU" + std::to_string(alu_base + 2), "RoPE sub -> FP16");
        trace(cycle + 1, cycle + 2,
            "VXM.ALU" + std::to_string(alu_base + 5), "RoPE add -> FP16");
    }

    void cast_pair_at(std::size_t cycle, ftlpu::Hemisphere hemisphere)
    {
        cast_pair_to_at(
            cycle, hemisphere, 0, hemisphere == ftlpu::Hemisphere::East ? 0 : 8);
    }

    void cast_pair_to_at(
        std::size_t cycle,
        ftlpu::Hemisphere hemisphere,
        std::size_t output_stream_base)
    {
        cast_pair_to_at(cycle, hemisphere, output_stream_base, 0);
    }

    void cast_pair_to_at(
        std::size_t cycle,
        ftlpu::Hemisphere hemisphere,
        std::size_t output_stream_base,
        std::size_t alu_base)
    {
        if (alu_base + 1 >= ftlpu::VxmLane::kAluCount) {
            throw std::out_of_range("VXM output cast exceeds ALU range");
        }
        const auto instruction = [&](std::size_t input_stream, std::size_t output_stream) {
            return ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamFloat32(input_stream),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Float16, output_stream,
                hemisphere, hemisphere};
        };
        ftlpu::test::enqueue_alu_at(
            icu_, vxm_, alu_base, cycle, instruction(32, output_stream_base));
        ftlpu::test::enqueue_alu_at(
            icu_, vxm_, alu_base + 1, cycle, instruction(36, output_stream_base + 2));
        end_cycle_ = std::max(end_cycle_, cycle + 1);
        trace(cycle, cycle + 1, "VXM.ALU" + std::to_string(alu_base), "cast output low pair");
        trace(cycle, cycle + 1,
            "VXM.ALU" + std::to_string(alu_base + 1), "cast output high pair");
    }

    void cast_pair_with_duplicate_at(std::size_t cycle, ftlpu::Hemisphere hemisphere)
    {
        cast_pair_with_duplicate_at(cycle, hemisphere, 0);
    }

    void cast_pair_with_duplicate_at(
        std::size_t cycle,
        ftlpu::Hemisphere hemisphere,
        std::size_t alu_base)
    {
        if (alu_base + 3 >= ftlpu::VxmLane::kAluCount) {
            throw std::out_of_range("VXM context cast exceeds ALU range");
        }
        const auto cast = [&](std::size_t input_stream, std::size_t output_stream) {
            return ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamFloat32(input_stream),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Float16, output_stream,
                hemisphere, hemisphere};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base, cycle, cast(32, 0));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base + 1, cycle, cast(36, 2));
        const auto instruction = [&](std::size_t alu, std::size_t output_stream) {
            return ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::Alu(alu),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Float16, output_stream,
                hemisphere, hemisphere};
        };
        ftlpu::test::enqueue_alu_at(
            icu_, vxm_, alu_base + 2, cycle + 1, instruction(alu_base, 4));
        ftlpu::test::enqueue_alu_at(
            icu_, vxm_, alu_base + 3, cycle + 1, instruction(alu_base + 1, 6));
        end_cycle_ = std::max(end_cycle_, cycle + 2);
        trace(cycle, cycle + 1, "VXM.ALU" + std::to_string(alu_base), "cast context pair 0");
        trace(cycle, cycle + 1, "VXM.ALU" + std::to_string(alu_base + 1), "cast context pair 1");
        trace(cycle + 1, cycle + 2,
            "VXM.ALU" + std::to_string(alu_base + 2), "duplicate FP16 pair 0");
        trace(cycle + 1, cycle + 2,
            "VXM.ALU" + std::to_string(alu_base + 3), "duplicate FP16 pair 1");
    }

    void softmax_scale_max_at(
        std::size_t cycle,
        bool first_key,
        ftlpu::Hemisphere hemisphere)
    {
        const auto instruction = [&](ftlpu::VxmAluOpcode opcode,
                                     ftlpu::VxmLaneOperand lhs,
                                     ftlpu::VxmLaneOperand rhs,
                                     std::optional<std::size_t> output = std::nullopt) {
            return ftlpu::VxmLaneAluInstruction {
                opcode, lhs, rhs, 1.0f, 0, ftlpu::VxmCastTarget::Float32,
                output, hemisphere, hemisphere};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 0, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamFloat32(32),
            ftlpu::VxmLaneOperand::Imm(kAttentionScale),
            kSoftmaxOutputStream));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 1, cycle + 1, instruction(
            first_key ? ftlpu::VxmAluOpcode::Pass : ftlpu::VxmAluOpcode::Max,
            first_key ? ftlpu::VxmLaneOperand::Alu(0) : ftlpu::VxmLaneOperand::Alu(1),
            first_key ? ftlpu::VxmLaneOperand::Imm(0.0f) : ftlpu::VxmLaneOperand::Alu(0)));
        end_cycle_ = std::max(end_cycle_, cycle + 2);
        trace(cycle, cycle + 1, "VXM.ALU0", "softmax P1 scale -> E8");
        trace(cycle + 1, cycle + 2, "VXM.ALU1", first_key ? "softmax P1 max init" : "softmax P1 recurrent max");
    }

    void softmax_exp_sum_at(
        std::size_t cycle,
        bool first_key,
        ftlpu::Hemisphere hemisphere)
    {
        const auto instruction = [&](ftlpu::VxmAluOpcode opcode,
                                     ftlpu::VxmLaneOperand lhs,
                                     ftlpu::VxmLaneOperand rhs,
                                     std::optional<std::size_t> output = std::nullopt) {
            return ftlpu::VxmLaneAluInstruction {
                opcode, lhs, rhs, 1.0f, 0, ftlpu::VxmCastTarget::Float32,
                output, hemisphere, hemisphere};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 2, cycle, instruction(
            ftlpu::VxmAluOpcode::Subtract,
            ftlpu::VxmLaneOperand::StreamFloat32(32),
            ftlpu::VxmLaneOperand::Alu(1)));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 3, cycle + 1, instruction(
            ftlpu::VxmAluOpcode::Exp,
            ftlpu::VxmLaneOperand::Alu(2),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            kSoftmaxOutputStream));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 4, cycle + 2, instruction(
            first_key ? ftlpu::VxmAluOpcode::Pass : ftlpu::VxmAluOpcode::Add,
            first_key ? ftlpu::VxmLaneOperand::Alu(3) : ftlpu::VxmLaneOperand::Alu(4),
            first_key ? ftlpu::VxmLaneOperand::Imm(0.0f) : ftlpu::VxmLaneOperand::Alu(3)));
        end_cycle_ = std::max(end_cycle_, cycle + 3);
        trace(cycle, cycle + 1, "VXM.ALU2", "softmax P2 subtract max");
        trace(cycle + 1, cycle + 2, "VXM.ALU3", "softmax P2 exp -> E8");
        trace(cycle + 2, cycle + 3, "VXM.ALU4", first_key ? "softmax P2 sum init" : "softmax P2 recurrent sum");
    }

    void softmax_normalize_at(std::size_t cycle, ftlpu::Hemisphere hemisphere)
    {
        const auto instruction = [&](ftlpu::VxmAluOpcode opcode,
                                     ftlpu::VxmLaneOperand lhs,
                                     ftlpu::VxmLaneOperand rhs,
                                     ftlpu::VxmCastTarget cast,
                                     std::optional<std::size_t> output = std::nullopt) {
            return ftlpu::VxmLaneAluInstruction {
                opcode, lhs, rhs, 1.0f, 0, cast, output, hemisphere, hemisphere};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 5, cycle, instruction(
            ftlpu::VxmAluOpcode::Divide,
            ftlpu::VxmLaneOperand::StreamFloat32(32),
            ftlpu::VxmLaneOperand::Alu(4),
            ftlpu::VxmCastTarget::Float32));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 6, cycle + 1, instruction(
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(5),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            ftlpu::VxmCastTarget::Float16,
            kSoftmaxOutputStream));
        end_cycle_ = std::max(end_cycle_, cycle + 2);
        trace(cycle, cycle + 1, "VXM.ALU5", "softmax P3 divide by sum");
        trace(cycle + 1, cycle + 2, "VXM.ALU6", "softmax P3 cast FP16 -> E8");
    }

    void copy_fp16_pair_at(
        std::size_t cycle,
        ftlpu::Hemisphere input_hemisphere,
        ftlpu::Hemisphere output_hemisphere)
    {
        const auto instruction = [&](std::size_t input_stream, std::size_t output_stream) {
            return ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamFloat16(input_stream),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Float16, output_stream,
                input_hemisphere, output_hemisphere};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 0, cycle, instruction(32, 0));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 1, cycle, instruction(34, 2));
        end_cycle_ = std::max(end_cycle_, cycle + 1);
        trace(cycle, cycle + 1, "VXM.ALU0", "hemisphere copy FP16 pair 0");
        trace(cycle, cycle + 1, "VXM.ALU1", "hemisphere copy FP16 pair 1");
    }

    void copy_byte_vector_at(
        std::size_t cycle,
        ftlpu::Hemisphere input_hemisphere,
        ftlpu::Hemisphere output_hemisphere)
    {
        for (std::size_t byte = 0; byte < ftlpu::VxmLane::kAluCount; ++byte) {
            ftlpu::test::enqueue_alu_at(icu_, vxm_, byte, cycle, {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamInt8(ftlpu::hw::kEastStreams + byte),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Int8, byte,
                input_hemisphere, output_hemisphere});
            trace(cycle, cycle + 1, "VXM.ALU" + std::to_string(byte),
                "copy Q IW byte stream " + std::to_string(byte));
        }
        end_cycle_ = std::max(end_cycle_, cycle + 1);
    }

    void route_byte_streams_at(
        std::size_t cycle,
        std::size_t input_stream_base,
        std::size_t output_stream_base,
        std::size_t byte_count,
        ftlpu::Hemisphere input_hemisphere,
        ftlpu::Hemisphere output_hemisphere,
        const std::string& label)
    {
        if (byte_count > ftlpu::VxmLane::kAluCount) {
            throw std::out_of_range("VXM route exceeds the 16 ALUs in one lane");
        }
        for (std::size_t byte = 0; byte < byte_count; ++byte) {
            ftlpu::test::enqueue_alu_at(icu_, vxm_, byte, cycle, {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamInt8(
                    ftlpu::hw::kEastStreams + input_stream_base + byte),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Int8, output_stream_base + byte,
                input_hemisphere, output_hemisphere});
            trace(cycle, cycle + 1, "VXM.ALU" + std::to_string(byte), label);
        }
        end_cycle_ = std::max(end_cycle_, cycle + 1);
    }

    void sxm_transpose_at(
        ftlpu::Hemisphere hemisphere,
        std::size_t cycle,
        ftlpu::SxmInstruction instruction)
    {
        const auto index = ftlpu::hemisphere_index(hemisphere);
        require_available(sxm_transpose_[index], cycle, "SXM transpose");
        pad(sxm_transpose_[index], cycle, [&](std::size_t count) {
            icu_.enqueue_sxm_transpose_nop(hemisphere, count);
        });
        icu_.enqueue_sxm_transpose(hemisphere, std::move(instruction));
        advance(sxm_transpose_[index], cycle + 1);
        trace(cycle, cycle + 1,
            std::string("SXM.") + ftlpu::hemisphere_short_name(hemisphere) + ".Transpose",
            "Transpose sg16 capture/advance");
    }

    void sxm_permute_at(
        ftlpu::Hemisphere hemisphere,
        std::size_t cycle,
        ftlpu::SxmInstruction instruction,
        bool tail = false)
    {
        const auto index = ftlpu::hemisphere_index(hemisphere);
        require_available(sxm_permute_[index], cycle, "SXM permute");
        pad(sxm_permute_[index], cycle, [&](std::size_t count) {
            icu_.enqueue_sxm_permute_nop(hemisphere, count);
        });
        icu_.enqueue_sxm_permute(hemisphere, std::move(instruction));
        advance(sxm_permute_[index], cycle + 1);
        trace(cycle, cycle + 1,
            std::string("SXM.") + ftlpu::hemisphere_short_name(hemisphere)
                + (tail ? ".Tail" : ".Permute"),
            tail ? "Permute northbound drain" : "Permute/emit");
    }

    void mxm_load_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t weight_buffer = 0,
        std::size_t weight_column = 0)
    {
        require_available(mxm_load_[mxm], cycle, "MXM load " + std::to_string(mxm));
        pad(mxm_load_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_load_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::IW(weight_buffer, weight_column));
        advance(mxm_load_[mxm], cycle + 1);
        trace(cycle, cycle + 1, mxm_name(mxm) + ".Load",
            "IW buffer=" + std::to_string(weight_buffer)
                + " column=" + std::to_string(weight_column));
    }

    void mxm_compute_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t activation_stream,
        std::size_t output_stream,
        std::size_t weight_buffer = 0)
    {
        require_available(mxm_compute_[mxm], cycle, "MXM compute " + std::to_string(mxm));
        pad(mxm_compute_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_compute_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::Compute(
                weight_buffer, activation_stream, output_stream));
        icu_.enqueue_mxm_compute_repeat(mxm, kTile - 1, 1);
        advance(mxm_compute_[mxm], cycle + kTile);
        trace(cycle, cycle + kTile, mxm_name(mxm) + ".Compute",
            "Compute buffer=" + std::to_string(weight_buffer)
                + " act=E" + std::to_string(activation_stream)
                + " out=W" + std::to_string(output_stream));
        trace(cycle + kTile, cycle + kMxmInputBlockIssueCycles,
            mxm_name(mxm) + ".Tail", "control + datapath drain");
    }

    std::size_t end_cycle() const { return end_cycle_; }

    void write_trace_csv(const std::string& path) const
    {
        auto output = std::ofstream(path, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open schedule trace output: " + path);
        }
        output << "start,end,resource,detail\n";
        for (const auto& event : trace_events_) {
            output << event.start << ',' << event.end << ','
                   << csv_field(event.resource) << ',' << csv_field(event.detail) << '\n';
        }
    }

private:
    static const char* mem_opcode_name(ftlpu::MemOpcode opcode)
    {
        switch (opcode) {
        case ftlpu::MemOpcode::Read: return "Read";
        case ftlpu::MemOpcode::Write: return "Write";
        case ftlpu::MemOpcode::Gather: return "Gather";
        case ftlpu::MemOpcode::Scatter: return "Scatter";
        case ftlpu::MemOpcode::Accumulate: return "Accumulate";
        }
        return "Unknown";
    }

    static std::string stream_name(std::size_t packed)
    {
        const auto stream = ftlpu::StreamId::from_packed(packed);
        return std::string(stream.direction() == ftlpu::StreamDirection::East ? "E" : "W")
            + std::to_string(stream.index());
    }

    static std::string mxm_name(std::size_t mxm)
    {
        const auto hemisphere = mxm / ftlpu::TspSliceSystem::kMxmCountPerHemisphere;
        const auto local = mxm % ftlpu::TspSliceSystem::kMxmCountPerHemisphere;
        return std::string("MXM.") + (hemisphere == 0 ? "E" : "W")
            + std::to_string(local);
    }

    static std::string csv_field(const std::string& value)
    {
        auto result = std::string {"\""};
        for (const auto ch : value) {
            if (ch == '"') result += '"';
            result += ch;
        }
        result += '"';
        return result;
    }

    void trace(std::size_t start, std::size_t end, std::string resource, std::string detail)
    {
        if (!trace_enabled_) return;
        trace_events_.push_back(TraceEvent {start, end, std::move(resource), std::move(detail)});
    }

    void trace_mem(
        std::size_t queue,
        std::size_t start,
        std::size_t end,
        const ftlpu::MemInstruction& instruction,
        std::int64_t stride = 0)
    {
        const auto hemisphere = queue / ftlpu::InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto slice = queue % ftlpu::InstructionControlUnit::kMemQueuesPerHemisphere;
        auto detail = std::string("slice=") + std::to_string(slice)
            + " addr=" + std::to_string(instruction.address)
            + " stream=" + stream_name(instruction.stream);
        if (end > start + 1) {
            detail += " count=" + std::to_string(end - start);
            detail += " stride=" + std::to_string(stride);
        }
        if (instruction.opcode == ftlpu::MemOpcode::Accumulate) {
            detail += instruction.accumulator_destination
                    == ftlpu::MemAccumulatorDestination::Stream
                ? " dst=stream+clear" : " dst=sram";
        }
        trace(start, end,
            std::string("MEM.") + (hemisphere == 0 ? "E." : "W.")
                + mem_opcode_name(instruction.opcode),
            std::move(detail));
    }

    static void require_available(std::size_t cursor, std::size_t cycle, const std::string& queue)
    {
        if (cycle < cursor) {
            throw std::logic_error(queue + " requests cycle " + std::to_string(cycle)
                + " but its cursor is " + std::to_string(cursor));
        }
    }

    template <typename Emit>
    static void pad(std::size_t cursor, std::size_t cycle, Emit emit)
    {
        if (cycle < cursor) throw std::logic_error("offline ICU queue schedule overlaps itself");
        emit(cycle - cursor);
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_cycle_ = std::max(end_cycle_, next);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMxmQueues> mxm_load_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMxmQueues> mxm_compute_{};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_{};
    std::array<std::size_t, ftlpu::hw::kHemispheres> sxm_transpose_{};
    std::array<std::size_t, ftlpu::hw::kHemispheres> sxm_permute_{};
    std::vector<TraceEvent> trace_events_{};
    bool trace_enabled_{false};
    std::size_t end_cycle_{0};
};

std::size_t mem_queue(ftlpu::Hemisphere hemisphere, std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(hemisphere, slice);
}

std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t east_read_to_sxm_latency(std::size_t slice)
{
    return kMemToSxmLatency - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t east_read_to_mxm_latency(std::size_t slice)
{
    return kMemToMxmLatency - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t activation_address(std::size_t k_block, std::size_t token)
{
    return k_block * kSeqLen + token;
}

std::size_t projection_address(Projection projection, std::size_t head, std::size_t token)
{
    return (projection_head_offset(projection) + head) * kSeqLen + token;
}

std::size_t query_iw_address(
    std::size_t head,
    std::size_t query_block,
    std::size_t phase)
{
    return kQueryIwAddressBase
        + (head * (kSeqLen / kTile) + query_block) * ftlpu::hw::kTileRows
        + phase;
}

std::size_t value_pack_address(
    std::size_t head,
    std::size_t reduction_block,
    std::size_t token_block,
    std::size_t block_row)
{
    return kValuePackAddressBase
        + ((head * (kHeadDim / kTile) + reduction_block) * (kSeqLen / kTile)
              + token_block)
            * ftlpu::hw::kTileRows
        + block_row;
}

std::size_t rope_table_address(std::size_t token)
{
    return kRopeTableAddressBase + token;
}

std::size_t score_address(
    std::size_t query_head,
    std::size_t query_block,
    std::size_t key_token)
{
    return kScoreAddressBase
        + (query_head * (kSeqLen / kTile) + query_block) * kSeqLen
        + key_token;
}

std::size_t probability_pack_address(
    std::size_t query_head,
    std::size_t query_block,
    std::size_t key_block)
{
    return kProbabilityPackAddressBase
        + (query_head * (kSeqLen / kTile) + query_block)
        * (kSeqLen / ftlpu::hw::kLanesPerTile)
        + key_block;
}

std::size_t probability_diagonal_address(
    std::size_t query_head,
    std::size_t query_block,
    std::size_t key_block,
    std::size_t diagonal)
{
    return kProbabilityDiagonalAddressBase
        + ((query_head * (kSeqLen / kTile) + query_block) * (kSeqLen / kTile)
              + key_block)
            * ftlpu::hw::kTileRows
        + diagonal;
}

std::size_t context_address(std::size_t query_head, std::size_t token)
{
    return query_head * kSeqLen + token;
}

std::size_t context_accumulator_address(std::size_t query_head, std::size_t token)
{
    return kContextAccumulatorAddressBase + query_head * kSeqLen + token;
}

ftlpu::Hemisphere context_hemisphere(std::size_t query_head)
{
    // The final GQA group reuses East KV weights but executes head 7 on the
    // West MXM pair through a stream route, so context placement follows the
    // P x V consumer rather than the KV source.
    if (query_head == 7) return ftlpu::Hemisphere::West;
    const auto kv_head = query_head / (kQueryHeads / kKvHeads);
    return static_cast<ftlpu::Hemisphere>(kv_head % ftlpu::hw::kHemispheres);
}

std::size_t attention_output_address(std::size_t output_block, std::size_t token)
{
    return output_block * kSeqLen + token;
}

std::size_t output_accumulator_address(std::size_t output_pair, std::size_t token)
{
    static_cast<void>(output_pair);
    return kOutputAccumulatorAddressBase + token;
}

ftlpu::SxmInstruction::StreamList sxm_streams(std::initializer_list<std::size_t> streams)
{
    auto result = ftlpu::SxmInstruction::StreamList {};
    for (const auto stream : streams) result.push_back(ftlpu::SxmStreamId {stream});
    return result;
}

ftlpu::SxmInstruction::StreamList sxm_stream_range(std::size_t first, std::size_t count)
{
    auto result = ftlpu::SxmInstruction::StreamList {};
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(ftlpu::SxmStreamId {first + index});
    }
    return result;
}

ftlpu::SxmInstruction::StreamList sxm_west_stream_range(std::size_t first, std::size_t count)
{
    auto result = ftlpu::SxmInstruction::StreamList {};
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(ftlpu::SxmStreamId {ftlpu::StreamId::West(first + index).packed()});
    }
    return result;
}

ftlpu::SxmInstruction::PermuteMap block_diagonal_map(std::size_t diagonal)
{
    auto map = ftlpu::Permute320::identity_map();
    for (std::size_t destination = 0; destination < ftlpu::hw::kTileRows; ++destination) {
        const auto source =
            (diagonal + ftlpu::hw::kTileRows - destination) % ftlpu::hw::kTileRows;
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            map[destination * ftlpu::hw::kLanesPerTile + lane] =
                source * ftlpu::hw::kLanesPerTile + lane;
        }
    }
    return map;
}


void initialize_inputs(ftlpu::TspSliceSystem& system, const std::vector<float>& input)
{
    for (std::size_t hemisphere_index = 0; hemisphere_index < ftlpu::hw::kHemispheres;
         ++hemisphere_index) {
        const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
        for (std::size_t k_block = 0; k_block < kHidden / kTile; ++k_block) {
            for (std::size_t token = 0; token < kSeqLen; ++token) {
                const auto address = activation_address(k_block, token);
                for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                        const auto hidden = k_block * kTile
                            + tile * ftlpu::hw::kLanesPerTile + lane;
                        const auto bits = ftlpu::Fp16::from_float(
                            input[x_index(token, hidden)]).bits();
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kActivationSlices[0], tile, address, lane, bits & 0xffu);
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kActivationSlices[1], tile, address, lane, bits >> 8);
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kActivationSlices[2], tile, address, lane, bits & 0xffu);
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kActivationSlices[3], tile, address, lane, bits >> 8);
                    }
                }
            }
        }
    }
}

void initialize_rope_tables(ftlpu::TspSliceSystem& system)
{
    for (std::size_t hemisphere_index = 0; hemisphere_index < ftlpu::hw::kHemispheres;
         ++hemisphere_index) {
        const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
        for (std::size_t token = 0; token < kSeqLen; ++token) {
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    const auto dimension = tile * ftlpu::hw::kLanesPerTile + lane;
                    const auto inverse_frequency = 1.0f / std::pow(
                        kRopeTheta,
                        static_cast<float>(2 * dimension) / static_cast<float>(kHeadDim));
                    const auto angle = static_cast<float>(token) * inverse_frequency;
                    const auto cos_bits = ftlpu::Fp16::from_float(std::cos(angle)).bits();
                    const auto sin_bits = ftlpu::Fp16::from_float(std::sin(angle)).bits();
                    system.initialize_mem_sram_lane_byte(
                        hemisphere, kRopeTableSlices[0], tile, rope_table_address(token), lane,
                        cos_bits & 0xffu);
                    system.initialize_mem_sram_lane_byte(
                        hemisphere, kRopeTableSlices[1], tile, rope_table_address(token), lane,
                        cos_bits >> 8);
                    system.initialize_mem_sram_lane_byte(
                        hemisphere, kRopeTableSlices[2], tile, rope_table_address(token), lane,
                        sin_bits & 0xffu);
                    system.initialize_mem_sram_lane_byte(
                        hemisphere, kRopeTableSlices[3], tile, rope_table_address(token), lane,
                        sin_bits >> 8);
                }
            }
        }
    }
}

std::uint16_t read_output(
    const ftlpu::TspSliceSystem& system,
    Projection projection,
    std::size_t token,
    std::size_t column)
{
    const auto head = column / kHeadDim;
    const auto dimension = column % kHeadDim;
    const auto hemisphere = static_cast<ftlpu::Hemisphere>(head % ftlpu::hw::kHemispheres);
    const auto physical = dimension % kTile;
    const auto tile = physical / ftlpu::hw::kLanesPerTile;
    const auto lane = physical % ftlpu::hw::kLanesPerTile;
    if (projection == Projection::Query) {
        const auto reduction_block = dimension / kTile;
        const auto local_query = token % kTile;
        const auto stream = (local_query % ftlpu::hw::kLanesPerTile) * 2;
        const auto address = query_iw_address(
            head,
            token / kTile,
            local_query / ftlpu::hw::kLanesPerTile);
        const auto low = system.read_mem_sram_lane_byte(
            hemisphere, kQueryIwSlices[reduction_block][stream], tile, address, lane);
        const auto high = system.read_mem_sram_lane_byte(
            hemisphere, kQueryIwSlices[reduction_block][stream + 1], tile, address, lane);
        return static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8);
    }
    if (projection == Projection::Value) {
        const auto reduction_block = dimension / kTile;
        const auto packed_stream = (token % ftlpu::hw::kLanesPerTile) * 2;
        const auto address = value_pack_address(
            head,
            reduction_block,
            token / kTile,
            (token % kTile) / ftlpu::hw::kLanesPerTile);
        const auto low = system.read_mem_sram_lane_byte(
            hemisphere,
            kValuePackSlices[reduction_block][packed_stream],
            tile,
            address,
            lane);
        const auto high = system.read_mem_sram_lane_byte(
            hemisphere,
            kValuePackSlices[reduction_block][packed_stream + 1],
            tile,
            address,
            lane);
        return static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8);
    }
    const auto slice = dimension < kTile ? kOutputSlices[0] : kOutputSlices[2];
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, slice, tile, projection_address(projection, head, token), lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, slice + 1, tile, projection_address(projection, head, token), lane);
    return static_cast<std::uint16_t>(low) | (static_cast<std::uint16_t>(high) << 8);
}

float read_probability(
    const ftlpu::TspSliceSystem& system,
    std::size_t query_head,
    std::size_t query_token,
    std::size_t key_token)
{
    const auto kv_head = query_head / (kQueryHeads / kKvHeads);
    const auto hemisphere = static_cast<ftlpu::Hemisphere>(
        kv_head % ftlpu::hw::kHemispheres);
    const auto query_block = query_token / kTile;
    const auto local_query = query_token % kTile;
    const auto tile = local_query / ftlpu::hw::kLanesPerTile;
    const auto lane = local_query % ftlpu::hw::kLanesPerTile;
    const auto packed_stream = (key_token % ftlpu::hw::kLanesPerTile) * 2;
    const auto address = probability_pack_address(
        query_head, query_block, key_token / ftlpu::hw::kLanesPerTile);
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, kQueryIwSlices[1][packed_stream], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, kQueryIwSlices[1][packed_stream + 1], tile, address, lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

float read_attention_output(
    const ftlpu::TspSliceSystem& system,
    std::size_t token,
    std::size_t column)
{
    const auto output_block = column / kTile;
    const auto output_pair = output_block / 2;
    const auto hemisphere = output_pair % ftlpu::hw::kHemispheres == 0
        ? ftlpu::Hemisphere::East
        : ftlpu::Hemisphere::West;
    const auto local = column % kTile;
    const auto tile = local / ftlpu::hw::kLanesPerTile;
    const auto lane = local % ftlpu::hw::kLanesPerTile;
    const auto slice = output_block % 2 == 0
        ? kAttentionOutputSlices[0]
        : kAttentionOutputSlices[2];
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, slice, tile,
        attention_output_address(output_block, token), lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, slice + 1, tile,
        attention_output_address(output_block, token), lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

float read_context(
    const ftlpu::TspSliceSystem& system,
    std::size_t query_head,
    std::size_t token,
    std::size_t dimension)
{
    const auto hemisphere = context_hemisphere(query_head);
    const auto local = dimension % kTile;
    const auto tile = local / ftlpu::hw::kLanesPerTile;
    const auto lane = local % ftlpu::hw::kLanesPerTile;
    const auto slice = dimension < kTile ? kContextSlices[0] : kContextSlices[2];
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, slice, tile, context_address(query_head, token), lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, slice + 1, tile, context_address(query_head, token), lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

} // namespace

int main() try
{
    std::vector<float> input(kSeqLen * kHidden);
    auto max_o_proj_error = 0.0f;
    auto max_o_proj_token = std::size_t {0};
    auto max_o_proj_column = std::size_t {0};
    auto max_o_proj_actual = 0.0f;
    auto max_o_proj_expected = 0.0f;
    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            input[x_index(token, hidden)] = ftlpu::Fp16::from_float(
                input_value(token, hidden)).to_float();
        }
    }

    constexpr std::array<Projection, 3> kProjections {
        Projection::Query, Projection::Key, Projection::Value};
    auto scales = std::array<std::vector<float>, kProjections.size()> {};
    auto weights = std::array<std::vector<std::int8_t>, kProjections.size()> {};
    auto dequantized = std::array<std::vector<float>, kProjections.size()> {};

    auto golden_outputs = std::array<std::vector<float>, kProjections.size()> {};
    for (const auto projection : kProjections) {
        const auto p = static_cast<std::size_t>(projection);
        const auto width = projection_width(projection);
        golden_outputs[p].resize(kSeqLen * width);
        scales[p].resize(width);
        weights[p].resize(kHidden * width);
        dequantized[p].resize(kHidden * width);
        for (std::size_t column = 0; column < width; ++column) {
            float max_abs = 0.0f;
            for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                max_abs = std::max(max_abs, std::fabs(weight_value(projection, hidden, column)));
            }
            scales[p][column] = max_abs / 127.0f;
            for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                const auto quantized = std::clamp(static_cast<int>(std::lround(
                    weight_value(projection, hidden, column) / scales[p][column])), -127, 127);
                weights[p][weight_index(projection, hidden, column)] =
                    static_cast<std::int8_t>(quantized);
                dequantized[p][weight_index(projection, hidden, column)] =
                    ftlpu::Fp16::from_float(
                        static_cast<float>(quantized) * scales[p][column]).to_float();
            }
        }
    }

    auto output_scales = std::vector<float>(kHidden);
    auto output_weights = std::vector<std::int8_t>(kHidden * kHidden);
    auto output_dequantized = std::vector<float>(kHidden * kHidden);
    for (std::size_t column = 0; column < kHidden; ++column) {
        auto max_abs = 0.0f;
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            max_abs = std::max(max_abs, std::fabs(output_weight_value(hidden, column)));
        }
        output_scales[column] = max_abs / 127.0f;
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto index = hidden * kHidden + column;
            const auto quantized = std::clamp(static_cast<int>(std::lround(
                output_weight_value(hidden, column) / output_scales[column])), -127, 127);
            output_weights[index] = static_cast<std::int8_t>(quantized);
            output_dequantized[index] = ftlpu::Fp16::from_float(
                static_cast<float>(quantized) * output_scales[column]).to_float();
        }
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system, input);
    initialize_rope_tables(system);
    auto schedule = OfflineSchedule(system.icu());

    std::size_t phase_start = 0;
    auto phase_markers = std::vector<std::pair<std::string, std::size_t>> {
        {"Q projection start", phase_start},
    };
    std::size_t weight_address = 0;
    for (const auto projection : kProjections) {
        const auto p = static_cast<std::size_t>(projection);
        for (std::size_t head_base = 0;
             head_base < projection_heads(projection);
             head_base += ftlpu::hw::kHemispheres) {
            for (std::size_t k_block = 0; k_block < kHidden / kTile; ++k_block) {
                const auto dequant_start = phase_start + 10;
                for (std::size_t hemisphere_index = 0;
                     hemisphere_index < ftlpu::hw::kHemispheres;
                     ++hemisphere_index) {
                    const auto head = head_base + hemisphere_index;
                    if (head >= projection_heads(projection)) continue;
                    const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                    for (std::size_t pulse = 0; pulse < 8; ++pulse) {
                        const auto mxm = pulse / 4;
                        const auto block = 3 - pulse % 4;
                        const auto cycle = dequant_start + hemisphere_index * 8 + pulse;
                        auto instruction = ftlpu::test::DequantSpec {};
                        instruction.input_stream_base = ftlpu::hw::kEastStreams;
                        instruction.output_stream_base = mxm * 16;
                        instruction.input_hemisphere = hemisphere;
                        instruction.output_hemisphere = hemisphere;
                        for (std::size_t stream = 0; stream < ftlpu::hw::kLanesPerTile; ++stream) {
                            const auto column = head * kHeadDim + mxm * kTile
                                + block * ftlpu::hw::kLanesPerTile + stream;
                            instruction.scales[stream] = scales[p][column];
                            const auto slice = kWeightSlices[stream];
                            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                                for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                                    const auto hidden = k_block * kTile
                                        + tile * ftlpu::hw::kLanesPerTile + lane;
                                    system.initialize_mem_sram_lane_byte(
                                        hemisphere,
                                        slice,
                                        tile,
                                        weight_address,
                                        lane,
                                        static_cast<std::uint8_t>(weights[p][
                                            weight_index(projection, hidden, column)]));
                                }
                            }
                            schedule.mem_at(
                                mem_queue(hemisphere, slice),
                                cycle - west_read_latency(slice),
                                ftlpu::MemInstruction::Read(
                                    weight_address, ftlpu::StreamId::West(stream)));
                        }
                        schedule.dequant_at(cycle, instruction);
                        schedule.mxm_load_at(
                            ftlpu::InstructionControlUnit::mxm_queue(hemisphere, mxm),
                            cycle + kWeightToIwLatency,
                            0,
                            block);
                        ++weight_address;
                    }
                }

                const auto first_compute = dequant_start + 32;
                const auto final_reduction = k_block + 1 == kHidden / kTile;
                const auto compute_block_cycles = final_reduction
                    ? 2 * kTile : kMxmInputBlockIssueCycles;
                for (std::size_t token_block = 0; token_block < kSeqLen / kTile; ++token_block) {
                    for (std::size_t hemisphere_index = 0;
                         hemisphere_index < ftlpu::hw::kHemispheres;
                         ++hemisphere_index) {
                        const auto head = head_base + hemisphere_index;
                        if (head >= projection_heads(projection)) continue;
                        const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                        const auto compute_cycle = first_compute
                            + token_block * compute_block_cycles;
                        const auto input_address = activation_address(k_block, token_block * kTile);
                        const auto output_address = projection_address(
                            projection, head, token_block * kTile);
                        for (std::size_t byte = 0; byte < kActivationSlices.size(); ++byte) {
                            schedule.mem_repeat_at(
                                mem_queue(hemisphere, kActivationSlices[byte]),
                                compute_cycle - kActivationLatency,
                                ftlpu::MemInstruction::Read(
                                    input_address, ftlpu::StreamId::East(byte)),
                                kTile,
                                1);
                        }
                        const auto destination = k_block + 1 == kHidden / kTile
                            ? ftlpu::MemAccumulatorDestination::Stream
                            : ftlpu::MemAccumulatorDestination::Sram;
                        schedule.mem_repeat_at(
                            mem_queue(hemisphere, ftlpu::hw::kWestAccumulatorMemSliceBase),
                            compute_cycle + kMxm0AccumulatorLatency,
                            ftlpu::MemInstruction::Accumulate(
                                output_address, ftlpu::StreamId::West(0), destination),
                            kTile,
                            1);
                        schedule.mem_repeat_at(
                            mem_queue(hemisphere, ftlpu::hw::kEastAccumulatorMemSliceBase),
                            compute_cycle + kMxm1AccumulatorLatency,
                            ftlpu::MemInstruction::Accumulate(
                                output_address, ftlpu::StreamId::West(4), destination),
                            kTile,
                            1);
                        schedule.mxm_compute_at(
                            ftlpu::InstructionControlUnit::mxm_queue(hemisphere, 0),
                            compute_cycle, 0, 0);
                        schedule.mxm_compute_at(
                            ftlpu::InstructionControlUnit::mxm_queue(hemisphere, 1),
                            compute_cycle, 2, 4);

                        if (destination == ftlpu::MemAccumulatorDestination::Stream) {
                            constexpr auto kAccumulatorToVxmLatency =
                                kMxm0AccumulatorLatency
                                + (ftlpu::hw::kWestAccumulatorMemSliceBase
                                    / ftlpu::hw::kMemSlicesPerGroup + 1);
                            for (std::size_t token_offset = 0; token_offset < kTile; ++token_offset) {
                                const auto token = token_block * kTile + token_offset;
                                const auto vxm_cycle = compute_cycle
                                    + kAccumulatorToVxmLatency + token_offset;
                                if (projection != Projection::Value) {
                                    for (std::size_t byte = 0; byte < kRopeTableSlices.size(); ++byte) {
                                        const auto slice = kRopeTableSlices[byte];
                                        schedule.mem_at(
                                            mem_queue(hemisphere, slice),
                                            vxm_cycle - west_read_latency(slice),
                                            ftlpu::MemInstruction::Read(
                                                rope_table_address(token),
                                                ftlpu::StreamId::West(8 + byte)));
                                    }
                                    schedule.rope_at(vxm_cycle, hemisphere);
                                } else {
                                    schedule.cast_pair_at(vxm_cycle, hemisphere);
                                }
                                const auto write_cycle = vxm_cycle
                                    + (projection == Projection::Value
                                           ? kCastWriteLatency : kRopeWriteLatency);
                                if (projection == Projection::Query) {
                                    const auto local_query = token % kTile;
                                    const auto phase = local_query / ftlpu::hw::kLanesPerTile;
                                    const auto local_column = local_query % ftlpu::hw::kLanesPerTile;
                                    for (std::size_t reduction_block = 0;
                                         reduction_block < kHeadDim / kTile;
                                         ++reduction_block) {
                                        for (std::size_t byte = 0; byte < 2; ++byte) {
                                            const auto stream = reduction_block * 2 + byte;
                                            const auto iw_stream = local_column * 2 + byte;
                                            const auto slice = kQueryIwSlices[reduction_block][iw_stream];
                                            schedule.mem_at(
                                                mem_queue(hemisphere, slice),
                                                write_cycle
                                                    + slice / ftlpu::hw::kMemSlicesPerGroup,
                                                ftlpu::MemInstruction::Write(
                                                    query_iw_address(
                                                        head, token / kTile, phase),
                                                    ftlpu::StreamId::East(stream)));
                                        }
                                    }
                                } else if (projection == Projection::Key) {
                                    for (std::size_t byte = 0; byte < kOutputSlices.size(); ++byte) {
                                        schedule.mem_at(
                                            mem_queue(hemisphere, kOutputSlices[byte]),
                                            write_cycle,
                                            ftlpu::MemInstruction::Write(
                                                projection_address(projection, head, token),
                                                ftlpu::StreamId::East(byte)));
                                    }
                                } else {
                                    const auto packed_stream =
                                        (token % ftlpu::hw::kLanesPerTile) * 2;
                                    for (std::size_t reduction_block = 0;
                                         reduction_block < kHeadDim / kTile;
                                         ++reduction_block) {
                                        for (std::size_t byte = 0; byte < 2; ++byte) {
                                            const auto slice = kValuePackSlices[
                                                reduction_block][packed_stream + byte];
                                            schedule.mem_at(
                                                mem_queue(hemisphere, slice),
                                                write_cycle
                                                    + slice / ftlpu::hw::kMemSlicesPerGroup,
                                                ftlpu::MemInstruction::Write(
                                                    value_pack_address(
                                                        head,
                                                        reduction_block,
                                                        token / kTile,
                                                        (token % kTile)
                                                            / ftlpu::hw::kLanesPerTile),
                                                    ftlpu::StreamId::East(
                                                        reduction_block * 2 + byte)));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                phase_start = first_compute
                    + (kSeqLen / kTile) * compute_block_cycles
                    + (final_reduction ? 16 : 0);
            }
        }
        phase_markers.emplace_back(
            std::string(projection_name(projection)) + " projection end", phase_start);
        if (projection == Projection::Value) {
            phase_markers.emplace_back("V x16 input layout ready", phase_start);
        }
    }

    // GQA requires each Q head to execute beside its shared KV head. Copy only
    // the three Q heads whose projection hemisphere differs from that KV home.
    auto copy_cycle = phase_start + 16;
    for (std::size_t query_head = 0; query_head < kQueryHeads; ++query_head) {
        const auto kv_head = query_head / (kQueryHeads / kKvHeads);
        const auto source = static_cast<ftlpu::Hemisphere>(
            query_head % ftlpu::hw::kHemispheres);
        const auto destination = static_cast<ftlpu::Hemisphere>(
            kv_head % ftlpu::hw::kHemispheres);
        if (source == destination) continue;
        for (std::size_t query_block = 0; query_block < kSeqLen / kTile; ++query_block) {
            for (std::size_t reduction_block = 0;
                 reduction_block < kHeadDim / kTile;
                 ++reduction_block) {
                for (std::size_t phase = 0;
                     phase < ftlpu::hw::kTileRows;
                     ++phase, ++copy_cycle) {
                    for (std::size_t stream = 0;
                         stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
                         ++stream) {
                        const auto slice = kQueryIwSlices[reduction_block][stream];
                        schedule.mem_at(
                            mem_queue(source, slice),
                            copy_cycle - west_read_latency(slice),
                            ftlpu::MemInstruction::Read(
                                query_iw_address(query_head, query_block, phase),
                                ftlpu::StreamId::West(stream)));
                    }
                    schedule.copy_byte_vector_at(copy_cycle, source, destination);
                    for (std::size_t stream = 0;
                         stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
                         ++stream) {
                        const auto slice = kQueryIwSlices[reduction_block][stream];
                        schedule.mem_at(
                            mem_queue(destination, slice),
                            copy_cycle + 1
                                + slice / ftlpu::hw::kMemSlicesPerGroup,
                            ftlpu::MemInstruction::Write(
                                query_iw_address(query_head, query_block, phase),
                                ftlpu::StreamId::East(stream)));
                    }
                }
            }
        }
        // The farthest mapped slice writes ten cycles after the VXM copy and
        // needs another westbound flight before the next copied head arrives.
        copy_cycle += 20;
    }
    phase_start = copy_cycle + 16;
    phase_markers.emplace_back("GQA Q copy end", phase_start);

    // K is duplicated after RoPE so MXM0 and MXM1 can consume distinct east
    // streams in the same cycle.  The RoPE table SRAM is dead at this point.
    auto key_copy_cycle = phase_start + 16;
    for (std::size_t key_head = 0; key_head < kKvHeads; ++key_head) {
        const auto hemisphere = static_cast<ftlpu::Hemisphere>(
            key_head % ftlpu::hw::kHemispheres);
        for (std::size_t token = 0; token < kSeqLen; ++token, ++key_copy_cycle) {
            for (std::size_t byte = 0; byte < kOutputSlices.size(); ++byte) {
                const auto slice = kOutputSlices[byte];
                schedule.mem_at(
                    mem_queue(hemisphere, slice),
                    key_copy_cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(
                        projection_address(Projection::Key, key_head, token),
                        ftlpu::StreamId::West(byte)));
            }
            schedule.copy_fp16_pair_at(key_copy_cycle, hemisphere, hemisphere);
            for (std::size_t byte = 0; byte < kKeyReplicaSlices.size(); ++byte) {
                const auto slice = kKeyReplicaSlices[byte];
                schedule.mem_at(
                    mem_queue(hemisphere, slice),
                    key_copy_cycle + 1 + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        projection_address(Projection::Key, key_head, token),
                        ftlpu::StreamId::East(byte)));
            }
        }
        key_copy_cycle += 12;
    }
    phase_start = key_copy_cycle + 16;
    phase_markers.emplace_back("K replica ready", phase_start);

    const auto schedule_query_iw = [&] (
        std::size_t head,
        std::size_t query_block,
        std::size_t reduction_block,
        ftlpu::Hemisphere hemisphere,
        std::size_t local_mxm,
        std::size_t first_iw_cycle,
        std::size_t weight_buffer) {
        const auto global_mxm = ftlpu::InstructionControlUnit::mxm_queue(hemisphere, local_mxm);
        for (std::size_t phase = 0;
             phase < ftlpu::hw::kTileRows;
             ++phase) {
            const auto iw_cycle = first_iw_cycle + phase;
            for (std::size_t stream = 0;
                 stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
                 ++stream) {
                const auto slice = kQueryIwSlices[reduction_block][stream];
                const auto source_phase = ftlpu::hw::kTileRows - 1 - phase;
                schedule.mem_at(
                    mem_queue(hemisphere, slice),
                    iw_cycle - east_read_to_mxm_latency(slice),
                    ftlpu::MemInstruction::Read(
                        query_iw_address(head, query_block, source_phase),
                        ftlpu::StreamId::East(
                            local_mxm * ftlpu::hw::kMxmLoadStreamsPerCycle + stream)));
            }
            schedule.mxm_load_at(
                global_mxm,
                iw_cycle,
                weight_buffer,
                ftlpu::hw::kTileRows - 1 - phase);
        }
    };

    struct QkWork {
        std::size_t query_head;
        std::size_t query_block;
        ftlpu::Hemisphere hemisphere;
        std::size_t local_mxm;
    };
    using QkWave = std::array<
        std::array<std::optional<QkWork>, ftlpu::TspSliceSystem::kMxmCountPerHemisphere>,
        ftlpu::hw::kHemispheres>;

    // A wave contains independent query blocks: MXM0 and MXM1 receive their
    // own Q weights, K replicas, accumulator group, and west output streams.
    auto qk_waves = std::vector<QkWave> {};
    const auto add_cross_hemisphere_heads = [&] (
        std::size_t east_head,
        std::size_t west_head) {
        for (std::size_t query_block = 0;
             query_block < kSeqLen / kTile;
             query_block += 2) {
            auto wave = QkWave {};
            wave[hemisphere_index(ftlpu::Hemisphere::East)][0] = QkWork {
                east_head, query_block, ftlpu::Hemisphere::East, 0};
            wave[hemisphere_index(ftlpu::Hemisphere::East)][1] = QkWork {
                east_head, query_block + 1, ftlpu::Hemisphere::East, 1};
            wave[hemisphere_index(ftlpu::Hemisphere::West)][0] = QkWork {
                west_head, query_block, ftlpu::Hemisphere::West, 0};
            wave[hemisphere_index(ftlpu::Hemisphere::West)][1] = QkWork {
                west_head, query_block + 1, ftlpu::Hemisphere::West, 1};
            qk_waves.push_back(std::move(wave));
        }
    };
    add_cross_hemisphere_heads(0, 3);
    add_cross_hemisphere_heads(1, 4);
    add_cross_hemisphere_heads(2, 5);
    for (std::size_t query_block = 0; query_block < kSeqLen / kTile; ++query_block) {
        auto wave = QkWave {};
        wave[hemisphere_index(ftlpu::Hemisphere::East)][0] = QkWork {
            6, query_block, ftlpu::Hemisphere::East, 0};
        wave[hemisphere_index(ftlpu::Hemisphere::East)][1] = QkWork {
            7, query_block, ftlpu::Hemisphere::East, 1};
        qk_waves.push_back(std::move(wave));
    }
    for (std::size_t query_block = 0;
         query_block < kSeqLen / kTile;
         query_block += 2) {
        auto wave = QkWave {};
        wave[hemisphere_index(ftlpu::Hemisphere::East)][0] = QkWork {
            8, query_block, ftlpu::Hemisphere::East, 0};
        wave[hemisphere_index(ftlpu::Hemisphere::East)][1] = QkWork {
            8, query_block + 1, ftlpu::Hemisphere::East, 1};
        qk_waves.push_back(std::move(wave));
    }

    auto completed_qk_works = std::vector<QkWork> {};
    for (const auto& wave : qk_waves) {
        const auto first_iw_cycle = phase_start + kMemToMxmLatency;
        const auto first_compute = first_iw_cycle + 24;
        for (std::size_t hemisphere_value = 0;
             hemisphere_value < ftlpu::hw::kHemispheres;
             ++hemisphere_value) {
            for (std::size_t local_mxm = 0;
                 local_mxm < ftlpu::TspSliceSystem::kMxmCountPerHemisphere;
                 ++local_mxm) {
                const auto& work = wave[hemisphere_value][local_mxm];
                if (!work.has_value()) continue;
                const auto iw_cycle = first_iw_cycle + local_mxm * 8;
                schedule_query_iw(
                    work->query_head, work->query_block, 0, work->hemisphere,
                    local_mxm, iw_cycle, 0);
                schedule_query_iw(
                    work->query_head, work->query_block, 1, work->hemisphere,
                    local_mxm, iw_cycle + 4, 1);
                completed_qk_works.push_back(*work);
            }
        }

        for (const auto& hemisphere_works : wave) {
            for (const auto& work : hemisphere_works) {
                if (!work.has_value()) continue;
                const auto kv_head = work->query_head / (kQueryHeads / kKvHeads);
                const auto key_slices = work->local_mxm == 0
                    ? kOutputSlices : kKeyReplicaSlices;
                const auto accumulator_base = work->local_mxm == 0
                    ? ftlpu::hw::kWestAccumulatorMemSliceBase
                    : ftlpu::hw::kEastAccumulatorMemSliceBase;
                const auto accumulator_latency = work->local_mxm == 0
                    ? kMxm0AccumulatorLatency : kMxm1AccumulatorLatency;
                const auto activation_stream = work->local_mxm * 2;
                const auto output_stream = work->local_mxm * 4;
                const auto global_mxm = ftlpu::InstructionControlUnit::mxm_queue(
                    work->hemisphere, work->local_mxm);
                for (std::size_t reduction_block = 0;
                     reduction_block < kHeadDim / kTile;
                     ++reduction_block) {
                    const auto reduction_compute = first_compute
                        + reduction_block * (kSeqLen / kTile) * kMxmInputBlockIssueCycles;
                    for (std::size_t key_block = 0;
                         key_block < kSeqLen / kTile;
                         ++key_block) {
                        const auto compute_cycle = reduction_compute
                            + key_block * kMxmInputBlockIssueCycles;
                        const auto key_address = projection_address(
                            Projection::Key, kv_head, key_block * kTile);
                        for (std::size_t byte = 0; byte < 2; ++byte) {
                            const auto slice = key_slices[reduction_block * 2 + byte];
                            schedule.mem_repeat_at(
                                mem_queue(work->hemisphere, slice),
                                compute_cycle - east_read_to_mxm_latency(slice),
                                ftlpu::MemInstruction::Read(
                                    key_address,
                                    ftlpu::StreamId::East(activation_stream + byte)),
                                kTile, 1);
                        }
                        schedule.mem_repeat_at(
                            mem_queue(work->hemisphere, accumulator_base),
                            compute_cycle + accumulator_latency,
                            ftlpu::MemInstruction::Accumulate(
                                score_address(
                                    work->query_head,
                                    work->query_block,
                                    key_block * kTile),
                                ftlpu::StreamId::West(output_stream),
                                ftlpu::MemAccumulatorDestination::Sram),
                            kTile, 1);
                        schedule.mxm_compute_at(
                            global_mxm, compute_cycle, activation_stream, output_stream,
                            reduction_block);
                    }
                }
            }
        }
        phase_start = first_compute
            + (kHeadDim / kTile) * (kSeqLen / kTile) * kMxmInputBlockIssueCycles
            + 16;
    }

    // VXM has one 16-ALU lane.  The four MXMs produce score matrices in
    // parallel, then the three recurrent softmax passes drain those matrices
    // one at a time from their independent accumulator groups.
    auto softmax_cycle = phase_start + 16;
    constexpr std::size_t kSoftmaxKeyStride = 1;
    for (const auto& work : completed_qk_works) {
        const auto accumulator_base = work.local_mxm == 0
            ? ftlpu::hw::kWestAccumulatorMemSliceBase
            : ftlpu::hw::kEastAccumulatorMemSliceBase;
        for (std::size_t key = 0; key < kSeqLen; ++key) {
            const auto vxm_cycle = softmax_cycle + key * kSoftmaxKeyStride;
            for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                const auto slice = accumulator_base + byte;
                schedule.mem_at(
                    mem_queue(work.hemisphere, slice),
                    vxm_cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(
                        score_address(work.query_head, work.query_block, key),
                        ftlpu::StreamId::West(byte)));
            }
            schedule.softmax_scale_max_at(vxm_cycle, key == 0, work.hemisphere);
            for (const auto slice : kScaledScoreSlices) {
                schedule.mem_at(
                    mem_queue(work.hemisphere, slice),
                    vxm_cycle + 1 + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        key,
                        ftlpu::StreamId::East(kSoftmaxOutputStream + slice % 4)));
            }
        }

        const auto pass2_start =
            softmax_cycle + kSeqLen * kSoftmaxKeyStride + 8;
        for (std::size_t key = 0; key < kSeqLen; ++key) {
            const auto vxm_cycle = pass2_start + key * kSoftmaxKeyStride;
            for (std::size_t byte = 0; byte < kScaledScoreSlices.size(); ++byte) {
                const auto slice = kScaledScoreSlices[byte];
                schedule.mem_at(
                    mem_queue(work.hemisphere, slice),
                    vxm_cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(key, ftlpu::StreamId::West(byte)));
            }
            schedule.softmax_exp_sum_at(vxm_cycle, key == 0, work.hemisphere);
            for (std::size_t byte = 0; byte < kExpScoreSlices.size(); ++byte) {
                const auto slice = kExpScoreSlices[byte];
                schedule.mem_at(
                    mem_queue(work.hemisphere, slice),
                    vxm_cycle + 2 + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        key, ftlpu::StreamId::East(kSoftmaxOutputStream + byte)));
            }
        }

        const auto pass3_start =
            pass2_start + kSeqLen * kSoftmaxKeyStride + 12;
        for (std::size_t key = 0; key < kSeqLen; ++key) {
            const auto vxm_cycle = pass3_start + key * kSoftmaxKeyStride;
            for (std::size_t byte = 0; byte < kExpScoreSlices.size(); ++byte) {
                const auto slice = kExpScoreSlices[byte];
                schedule.mem_at(
                    mem_queue(work.hemisphere, slice),
                    vxm_cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(key, ftlpu::StreamId::West(byte)));
            }
            schedule.softmax_normalize_at(vxm_cycle, work.hemisphere);
            const auto packed_stream = (key % ftlpu::hw::kLanesPerTile) * 2;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kQueryIwSlices[1][packed_stream + byte];
                schedule.mem_at(
                    mem_queue(work.hemisphere, slice),
                    vxm_cycle + 2 + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        probability_pack_address(
                            work.query_head,
                            work.query_block,
                            key / ftlpu::hw::kLanesPerTile),
                        ftlpu::StreamId::East(kSoftmaxOutputStream + byte)));
            }
        }
        softmax_cycle = pass3_start + kSeqLen * kSoftmaxKeyStride + 8;
    }
    phase_start = softmax_cycle;
    phase_markers.emplace_back("QK and softmax end", phase_start);

    // P3 writes each FP16 probability directly into a 16-stream packed row.
    // Once softmax releases its scratch slices, SXM transposes every P block
    // into the persistent diagonal layout consumed by the 2-stream replay.
    auto probability_transpose_ready =
        std::array<std::size_t, ftlpu::hw::kHemispheres> {
            phase_start, phase_start};
    for (const auto& work : completed_qk_works) {
        const auto hemisphere = hemisphere_index(work.hemisphere);
        for (std::size_t key_block = 0;
             key_block < kSeqLen / kTile;
             ++key_block) {
            const auto start_cycle = probability_transpose_ready[hemisphere];
            const auto capture_start = start_cycle + kMemToSxmLatency;
            for (std::size_t beat = 0; beat < ftlpu::hw::kTileRows; ++beat) {
                const auto block_row = beat;
                for (std::size_t stream = 0;
                     stream < 2 * ftlpu::hw::kLanesPerTile;
                     ++stream) {
                    const auto slice = kQueryIwSlices[1][stream];
                    schedule.mem_at(
                        mem_queue(work.hemisphere, slice),
                        capture_start + beat - east_read_to_sxm_latency(slice),
                        ftlpu::MemInstruction::Read(
                            probability_pack_address(
                                work.query_head,
                                work.query_block,
                                key_block * ftlpu::hw::kTileRows + block_row),
                            ftlpu::StreamId::East(stream)));
                }
            }
            for (std::size_t wave = 0; wave < ftlpu::hw::kTileRows; ++wave) {
                const auto transpose_cycle = capture_start + wave;
                const auto permute_cycle = transpose_cycle + 1;
                schedule.sxm_transpose_at(
                    work.hemisphere,
                    transpose_cycle,
                    ftlpu::SxmInstruction::Transpose(
                        sxm_stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
                        sxm_stream_range(16, 2 * ftlpu::hw::kLanesPerTile)));
                schedule.sxm_permute_at(
                    work.hemisphere,
                    permute_cycle,
                    ftlpu::SxmInstruction::Permute(
                        sxm_stream_range(16, 2 * ftlpu::hw::kLanesPerTile),
                        sxm_west_stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
                        block_diagonal_map(wave)));
                for (std::size_t stream = 0;
                     stream < 2 * ftlpu::hw::kLanesPerTile;
                     ++stream) {
                    const auto slice = kQueryIwSlices[0][stream];
                    schedule.mem_at(
                        mem_queue(work.hemisphere, slice),
                        permute_cycle + ftlpu::hw::kMemGroups
                            - slice / ftlpu::hw::kMemSlicesPerGroup,
                        ftlpu::MemInstruction::Write(
                            probability_diagonal_address(
                                work.query_head,
                                work.query_block,
                                key_block,
                                wave),
                            ftlpu::StreamId::West(stream)));
                }
            }
            probability_transpose_ready[hemisphere] =
                start_cycle + ftlpu::hw::kTileRows;
        }
    }
    // Consecutive probability blocks share W0..W15 and start every four
    // cycles.  Only the final block on each hemisphere needs the three-cycle
    // northbound control/Permute tail.
    for (std::size_t hemisphere = 0; hemisphere < ftlpu::hw::kHemispheres; ++hemisphere) {
        const auto side = static_cast<ftlpu::Hemisphere>(hemisphere);
        for (std::size_t tail = 0; tail + 1 < ftlpu::hw::kTileRows; ++tail) {
            const auto transpose_cycle = probability_transpose_ready[hemisphere]
                + kMemToSxmLatency + tail;
            schedule.sxm_permute_at(
                side,
                transpose_cycle + 1,
                ftlpu::SxmInstruction::Permute(
                    sxm_stream_range(16, 2 * ftlpu::hw::kLanesPerTile),
                    sxm_west_stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
                    block_diagonal_map(tail)),
                true);
        }
        probability_transpose_ready[hemisphere] += ftlpu::hw::kTileRows - 1;
    }
    phase_start = *std::max_element(
        probability_transpose_ready.begin(), probability_transpose_ready.end())
        + kMemToSxmLatency + ftlpu::hw::kMemGroups + 1;
    phase_markers.emplace_back("P replay layout ready", phase_start);

    struct PvWork {
        std::size_t query_head;
        ftlpu::Hemisphere source;
        ftlpu::Hemisphere destination;
    };
    const auto schedule_pv_v_load = [&](const PvWork& work, std::size_t key_block,
                                         std::size_t start_cycle) {
        const auto kv_head = work.query_head / (kQueryHeads / kKvHeads);
        auto ready_cycle = start_cycle;
        auto route_start = start_cycle;
        for (std::size_t mxm = 0; mxm < 2; ++mxm) {
            const auto global_mxm = ftlpu::InstructionControlUnit::mxm_queue(
                work.destination, mxm);
            const auto capture_start = route_start + kMemToSxmLatency;
            for (std::size_t beat = 0; beat < ftlpu::hw::kTileRows; ++beat) {
                const auto block_row = beat;
                for (std::size_t stream = 0;
                     stream < 2 * ftlpu::hw::kLanesPerTile;
                     ++stream) {
                    const auto slice = kValuePackSlices[mxm][stream];
                    if (work.source == work.destination) {
                        schedule.mem_at(
                            mem_queue(work.source, slice),
                            capture_start + beat - east_read_to_sxm_latency(slice),
                            ftlpu::MemInstruction::Read(
                                value_pack_address(kv_head, mxm, key_block, block_row),
                                ftlpu::StreamId::East(stream)));
                    } else {
                        schedule.mem_at(
                            mem_queue(work.source, slice),
                            route_start + beat - west_read_latency(slice),
                            ftlpu::MemInstruction::Read(
                                value_pack_address(kv_head, mxm, key_block, block_row),
                                ftlpu::StreamId::West(stream)));
                    }
                }
            }
            for (std::size_t wave = 0; wave < ftlpu::hw::kTileRows; ++wave) {
                const auto transpose_cycle = capture_start + wave;
                const auto permute_cycle = transpose_cycle + 1;
                schedule.sxm_transpose_at(
                    work.destination,
                    transpose_cycle,
                    ftlpu::SxmInstruction::Transpose(
                        sxm_stream_range(0, 2 * ftlpu::hw::kLanesPerTile),
                        sxm_stream_range(16, 2 * ftlpu::hw::kLanesPerTile)));
                schedule.sxm_permute_at(
                    work.destination,
                    permute_cycle,
                    ftlpu::SxmInstruction::Permute(
                        sxm_stream_range(16, 2 * ftlpu::hw::kLanesPerTile),
                        sxm_stream_range(
                            mxm * ftlpu::hw::kMxmLoadStreamsPerCycle,
                            ftlpu::hw::kMxmLoadStreamsPerCycle),
                        block_diagonal_map(wave),
                        ftlpu::SxmWeightLayout::MatrixColumns));
                schedule.mxm_load_at(global_mxm, permute_cycle + 1, 0, wave);
            }
            // The next MXM uses a different 16-stream destination. Drain the
            // three northbound tail waves before changing the Permute route.
            for (std::size_t tail = 0; tail + 1 < ftlpu::hw::kTileRows; ++tail) {
                const auto wave = ftlpu::hw::kTileRows + tail;
                const auto transpose_cycle = capture_start + wave;
                schedule.sxm_permute_at(
                    work.destination,
                    transpose_cycle + 1,
                    ftlpu::SxmInstruction::Permute(
                        sxm_stream_range(16, 2 * ftlpu::hw::kLanesPerTile),
                        sxm_stream_range(
                            mxm * ftlpu::hw::kMxmLoadStreamsPerCycle,
                            ftlpu::hw::kMxmLoadStreamsPerCycle),
                        block_diagonal_map(wave),
                        ftlpu::SxmWeightLayout::MatrixColumns),
                    true);
            }
            ready_cycle = std::max(
                ready_cycle, capture_start + 2 * ftlpu::hw::kTileRows + 1);
            route_start += 2 * ftlpu::hw::kTileRows - 1;
        }
        return ready_cycle;
    };
    const auto schedule_pv_query_block = [&](const PvWork& work, std::size_t key_block,
                                              std::size_t query_block, std::size_t start_cycle) {
        const auto replay_start = start_cycle;
        const auto first_compute = replay_start + kMemToMxmLatency;
        for (std::size_t query = 0; query < kTile; ++query) {
            const auto row = query % ftlpu::hw::kLanesPerTile;
            const auto query_block_row = query / ftlpu::hw::kLanesPerTile;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kQueryIwSlices[0][row * 2 + byte];
                const auto address = probability_diagonal_address(
                    work.query_head,
                    query_block,
                    key_block,
                    query_block_row);
                if (work.source == work.destination) {
                    schedule.mem_at(
                        mem_queue(work.source, slice),
                        first_compute + query - east_read_to_mxm_latency(slice),
                        ftlpu::MemInstruction::Read(
                            address, ftlpu::StreamId::East(byte)));
                } else {
                    schedule.mem_at(
                        mem_queue(work.source, slice),
                        replay_start + query - west_read_latency(slice),
                        ftlpu::MemInstruction::Read(
                            address, ftlpu::StreamId::West(byte)));
                }
            }
        }
        if (key_block == 0) {
            for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                schedule.mem_repeat_at(
                    mem_queue(
                        work.destination,
                        ftlpu::hw::kWestAccumulatorMemSliceBase + byte),
                    first_compute + kMxm0AccumulatorLatency,
                    ftlpu::MemInstruction::Write(
                        context_accumulator_address(
                            work.query_head, query_block * kTile),
                        ftlpu::StreamId::West(byte)),
                    kTile, 1);
                schedule.mem_repeat_at(
                    mem_queue(
                        work.destination,
                        ftlpu::hw::kEastAccumulatorMemSliceBase + byte),
                    first_compute + kMxm1AccumulatorLatency,
                    ftlpu::MemInstruction::Write(
                        context_accumulator_address(
                            work.query_head, query_block * kTile),
                        ftlpu::StreamId::West(4 + byte)),
                    kTile, 1);
            }
        } else {
            const auto destination = ftlpu::MemAccumulatorDestination::Sram;
            schedule.mem_repeat_at(
                mem_queue(work.destination, ftlpu::hw::kWestAccumulatorMemSliceBase),
                first_compute + kMxm0AccumulatorLatency,
                ftlpu::MemInstruction::Accumulate(
                    context_accumulator_address(work.query_head, query_block * kTile),
                    ftlpu::StreamId::West(0), destination),
                kTile, 1);
            schedule.mem_repeat_at(
                mem_queue(work.destination, ftlpu::hw::kEastAccumulatorMemSliceBase),
                first_compute + kMxm1AccumulatorLatency,
                ftlpu::MemInstruction::Accumulate(
                    context_accumulator_address(work.query_head, query_block * kTile),
                    ftlpu::StreamId::West(4), destination),
                kTile, 1);
        }
        schedule.mxm_compute_at(
            ftlpu::InstructionControlUnit::mxm_queue(work.destination, 0),
            first_compute, 0, 0);
        schedule.mxm_compute_at(
            ftlpu::InstructionControlUnit::mxm_queue(work.destination, 1),
            first_compute, 0, 4);
        if (key_block + 1 == kSeqLen / kTile) {
            const auto context_read_start = first_compute
                + kMxm0AccumulatorLatency + kTile + 12;
            const auto alu_base = work.destination == ftlpu::Hemisphere::East ? 8 : 12;
            for (std::size_t offset = 0; offset < kTile; ++offset) {
                const auto token = query_block * kTile + offset;
                const auto vxm_cycle = context_read_start + offset;
                for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                    const auto west_slice = ftlpu::hw::kWestAccumulatorMemSliceBase + byte;
                    const auto east_slice = ftlpu::hw::kEastAccumulatorMemSliceBase + byte;
                    schedule.mem_at(
                        mem_queue(work.destination, west_slice),
                        vxm_cycle - west_read_latency(west_slice),
                        ftlpu::MemInstruction::Read(
                            context_accumulator_address(
                                work.query_head, query_block * kTile + offset),
                            ftlpu::StreamId::West(byte)));
                    schedule.mem_at(
                        mem_queue(work.destination, east_slice),
                        vxm_cycle - west_read_latency(east_slice),
                        ftlpu::MemInstruction::Read(
                            context_accumulator_address(
                                work.query_head, query_block * kTile + offset),
                            ftlpu::StreamId::West(4 + byte)));
                }
                schedule.cast_pair_with_duplicate_at(vxm_cycle, work.destination, alu_base);
                for (std::size_t byte = 0; byte < 4; ++byte) {
                    const auto slice = kContextSlices[byte];
                    schedule.mem_at(
                        mem_queue(work.destination, slice),
                        vxm_cycle + 1 + slice / ftlpu::hw::kMemSlicesPerGroup,
                        ftlpu::MemInstruction::Write(
                            context_address(work.query_head, token),
                            ftlpu::StreamId::East(byte)));
                }
                for (std::size_t byte = 4; byte < kContextSlices.size(); ++byte) {
                    const auto slice = kContextSlices[byte];
                    schedule.mem_at(
                        mem_queue(work.destination, slice),
                        vxm_cycle + 2 + slice / ftlpu::hw::kMemSlicesPerGroup,
                        ftlpu::MemInstruction::Write(
                            context_address(work.query_head, token),
                            ftlpu::StreamId::East(byte)));
                }
            }
        }
        return first_compute + 2 * kTile
            + (key_block + 1 == kSeqLen / kTile ? 40 : 16);
    };
    const std::array<std::array<std::optional<PvWork>, 2>, 7> pv_waves {{
        {{PvWork {0, ftlpu::Hemisphere::East, ftlpu::Hemisphere::East},
          PvWork {3, ftlpu::Hemisphere::West, ftlpu::Hemisphere::West}}},
        {{PvWork {1, ftlpu::Hemisphere::East, ftlpu::Hemisphere::East},
          PvWork {4, ftlpu::Hemisphere::West, ftlpu::Hemisphere::West}}},
        {{PvWork {2, ftlpu::Hemisphere::East, ftlpu::Hemisphere::East},
          PvWork {5, ftlpu::Hemisphere::West, ftlpu::Hemisphere::West}}},
        {{PvWork {6, ftlpu::Hemisphere::East, ftlpu::Hemisphere::East}, std::nullopt}},
        {{PvWork {7, ftlpu::Hemisphere::East, ftlpu::Hemisphere::West}, std::nullopt}},
        {{PvWork {8, ftlpu::Hemisphere::East, ftlpu::Hemisphere::East}, std::nullopt}},
    }};
    for (const auto& wave : pv_waves) {
        for (std::size_t key_block = 0; key_block < kSeqLen / kTile; ++key_block) {
            auto load_ready = phase_start;
            auto source_load_ready = std::array<std::size_t, ftlpu::hw::kHemispheres> {
                phase_start, phase_start};
            for (const auto& work : wave) {
                if (!work.has_value()) continue;
                const auto work_start = std::max(
                    phase_start, source_load_ready[ftlpu::hemisphere_index(work->source)]);
                load_ready = std::max(
                    load_ready, schedule_pv_v_load(*work, key_block, work_start));
                // Eastbound local reads and westbound VXM-route reads have
                // different group latencies. Leave a full 16-stream burst
                // between consumers of the same single-port source slices.
                source_load_ready[ftlpu::hemisphere_index(work->source)] = work_start + 16;
            }
            phase_start = load_ready + 8;
            for (std::size_t query_block = 0; query_block < kSeqLen / kTile; ++query_block) {
                auto block_end = phase_start;
                auto source_probability_ready = std::array<std::size_t, ftlpu::hw::kHemispheres> {
                    phase_start, phase_start};
                for (const auto& work : wave) {
                    if (!work.has_value()) continue;
                    const auto work_start = std::max(
                        phase_start,
                        source_probability_ready[ftlpu::hemisphere_index(work->source)]);
                    block_end = std::max(
                        block_end,
                        schedule_pv_query_block(*work, key_block, query_block, work_start));
                    source_probability_ready[ftlpu::hemisphere_index(work->source)] = work_start + 16;
                }
                phase_start = block_end;
            }
        }
    }
    phase_markers.emplace_back("P x V end", phase_start);

    // Context stays in the hemisphere where P x V produced it.  o_proj reads
    // local context directly, while a non-local half is routed through VXM on
    // the exact cycles its East MXM consumer needs it; no remote MEM copy is
    // materialized.
    phase_start += 16;
    phase_markers.emplace_back("context route ready", phase_start);
    weight_address = std::max(weight_address, std::size_t {5000});

    // Concat(head contexts) is represented by the MEM address layout.  Each
    // reduction block reads the corresponding head half, duplicates it onto
    // both MXM activation pairs, and accumulates the 576-wide O projection.
    for (std::size_t output_pair = 0;
         output_pair < kHidden / (2 * kTile);
         ++output_pair) {
        const auto output_hemisphere = output_pair % ftlpu::hw::kHemispheres == 0
            ? ftlpu::Hemisphere::East
            : ftlpu::Hemisphere::West;
        for (std::size_t reduction_block = 0;
             reduction_block < kHidden / kTile;
             ++reduction_block) {
            const auto dequant_start = phase_start + 10;
            for (std::size_t pulse = 0; pulse < 8; ++pulse) {
                const auto mxm = pulse / 4;
                const auto block = 3 - pulse % 4;
                const auto cycle = dequant_start + pulse;
                auto instruction = ftlpu::test::DequantSpec {};
                instruction.input_stream_base = ftlpu::hw::kEastStreams;
                instruction.output_stream_base = mxm * 16;
                instruction.input_hemisphere = output_hemisphere;
                instruction.output_hemisphere = output_hemisphere;
                for (std::size_t stream = 0;
                     stream < ftlpu::hw::kLanesPerTile;
                     ++stream) {
                    const auto column = output_pair * 2 * kTile + mxm * kTile
                        + block * ftlpu::hw::kLanesPerTile + stream;
                    instruction.scales[stream] = output_scales[column];
                    const auto slice = kWeightSlices[stream];
                    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                        for (std::size_t lane = 0;
                             lane < ftlpu::hw::kLanesPerTile;
                             ++lane) {
                            const auto hidden = reduction_block * kTile
                                + tile * ftlpu::hw::kLanesPerTile + lane;
                            system.initialize_mem_sram_lane_byte(
                                output_hemisphere,
                                slice,
                                tile,
                                weight_address,
                                lane,
                                static_cast<std::uint8_t>(
                                    output_weights[hidden * kHidden + column]));
                        }
                    }
                    schedule.mem_at(
                        mem_queue(output_hemisphere, slice),
                        cycle - west_read_latency(slice),
                        ftlpu::MemInstruction::Read(
                            weight_address, ftlpu::StreamId::West(stream)));
                }
                schedule.dequant_at(cycle, instruction);
                schedule.mxm_load_at(
                        ftlpu::InstructionControlUnit::mxm_queue(
                        output_hemisphere, mxm),
                    cycle + kWeightToIwLatency,
                    0,
                    block);
                ++weight_address;
            }

            const auto first_compute = dequant_start + 32;
            const auto context_head = reduction_block / (kHeadDim / kTile);
            const auto context_half = reduction_block % (kHeadDim / kTile);
            const auto context_source = context_hemisphere(context_head);
            for (std::size_t token_block = 0;
                 token_block < kSeqLen / kTile;
                 ++token_block) {
                const auto compute_cycle = first_compute
                    + token_block * kMxmInputBlockIssueCycles;
                const auto input_address = context_address(
                    context_head, token_block * kTile);
                const std::array<std::size_t, 4> input_slices {
                    kContextSlices[context_half * 2],
                    kContextSlices[context_half * 2 + 1],
                    kContextSlices[4 + context_half * 2],
                    kContextSlices[4 + context_half * 2 + 1],
                };
                if (context_source == output_hemisphere) {
                    for (std::size_t byte = 0; byte < input_slices.size(); ++byte) {
                        schedule.mem_repeat_at(
                            mem_queue(context_source, input_slices[byte]),
                            compute_cycle - east_read_to_mxm_latency(input_slices[byte]),
                            ftlpu::MemInstruction::Read(
                                input_address, ftlpu::StreamId::East(byte)),
                            kTile, 1);
                    }
                } else {
                    const auto route_cycle = compute_cycle - kMemToMxmLatency;
                    for (std::size_t byte = 0; byte < input_slices.size(); ++byte) {
                        const auto slice = input_slices[byte];
                        schedule.mem_repeat_at(
                            mem_queue(context_source, slice),
                            route_cycle - west_read_latency(slice),
                            ftlpu::MemInstruction::Read(
                                input_address, ftlpu::StreamId::West(byte)),
                            kTile, 1);
                    }
                    for (std::size_t offset = 0; offset < kTile; ++offset) {
                        schedule.route_byte_streams_at(
                            route_cycle + offset, 0, 0, input_slices.size(),
                            context_source, output_hemisphere,
                            "route context to o_proj");
                    }
                }
                const auto destination = reduction_block + 1 == kHidden / kTile
                    ? ftlpu::MemAccumulatorDestination::Stream
                    : ftlpu::MemAccumulatorDestination::Sram;
                schedule.mem_repeat_at(
                    mem_queue(output_hemisphere,
                        ftlpu::hw::kWestAccumulatorMemSliceBase),
                    compute_cycle + kMxm0AccumulatorLatency,
                    ftlpu::MemInstruction::Accumulate(
                        output_accumulator_address(
                            output_pair, token_block * kTile),
                        ftlpu::StreamId::West(0), destination),
                    kTile, 1);
                schedule.mem_repeat_at(
                    mem_queue(output_hemisphere,
                        ftlpu::hw::kEastAccumulatorMemSliceBase),
                    compute_cycle + kMxm1AccumulatorLatency,
                    ftlpu::MemInstruction::Accumulate(
                        output_accumulator_address(
                            output_pair, token_block * kTile),
                        ftlpu::StreamId::West(4), destination),
                    kTile, 1);
                schedule.mxm_compute_at(
                        ftlpu::InstructionControlUnit::mxm_queue(
                        output_hemisphere, 0),
                    compute_cycle, 0, 0);
                schedule.mxm_compute_at(
                        ftlpu::InstructionControlUnit::mxm_queue(
                        output_hemisphere, 1),
                    compute_cycle, 2, 4);
                if (destination == ftlpu::MemAccumulatorDestination::Stream) {
                    constexpr auto kOutputToVxmLatency = kMxm0AccumulatorLatency
                        + (ftlpu::hw::kWestAccumulatorMemSliceBase
                            / ftlpu::hw::kMemSlicesPerGroup + 1);
                    for (std::size_t offset = 0; offset < kTile; ++offset) {
                        const auto token = token_block * kTile + offset;
                        const auto vxm_cycle = compute_cycle
                            + kOutputToVxmLatency + offset;
                        schedule.cast_pair_to_at(
                            vxm_cycle, output_hemisphere, kSoftmaxOutputStream,
                            output_hemisphere == ftlpu::Hemisphere::East ? 0 : 8);
                        for (std::size_t byte = 0;
                             byte < kAttentionOutputSlices.size();
                             ++byte) {
                            const auto slice = kAttentionOutputSlices[byte];
                            schedule.mem_at(
                                mem_queue(output_hemisphere, slice),
                                vxm_cycle + 1
                                    + slice / ftlpu::hw::kMemSlicesPerGroup,
                                ftlpu::MemInstruction::Write(
                                    attention_output_address(
                                        output_pair * 2 + byte / 2, token),
                                    ftlpu::StreamId::East(
                                        kSoftmaxOutputStream + byte)));
                        }
                    }
                }
            }
            phase_start = first_compute
                + (kSeqLen / kTile) * kMxmInputBlockIssueCycles
                + (reduction_block + 1 == kHidden / kTile ? 24 : 0);
        }
    }

    phase_markers.emplace_back("o_proj end", schedule.end_cycle() + 16);
    if (const auto* trace_path = std::getenv("FTLPU_SCHEDULE_TRACE")) {
        schedule.write_trace_csv(trace_path);
    }
    const auto report_schedule = std::getenv("FTLPU_SCHEDULE_REPORT") != nullptr;
    if (report_schedule) {
        for (const auto& [name, cycle] : phase_markers) {
            std::cout << "schedule_phase cycle=" << cycle << " name=" << name << '\n';
        }
        std::cout << std::flush;
    }
    if (std::getenv("FTLPU_SCHEDULE_TRACE_ONLY") != nullptr) {
        std::cout << "SmolLM2 attention schedule trace generated; scheduled_cycles="
                  << schedule.end_cycle() + 16 << '\n';
        return 0;
    }

    for (std::size_t cycle = 0; cycle < schedule.end_cycle() + 16; ++cycle) {
        try {
            system.tick({});
        } catch (const std::exception& ex) {
            throw std::runtime_error(
                "system cycle " + std::to_string(cycle) + ": " + ex.what());
        }
    }

    if (report_schedule) {
        for (std::size_t mxm = 0; mxm < ftlpu::TspSliceSystem::kMxmCount; ++mxm) {
            system.mxm_performance(mxm).print(
                std::cout, "mxm" + std::to_string(mxm));
        }
    }

    for (const auto projection : kProjections) {
        const auto p = static_cast<std::size_t>(projection);
        const auto width = projection_width(projection);
        auto projected = std::vector<float>(kSeqLen * width, 0.0f);
        for (std::size_t token = 0; token < kSeqLen; ++token) {
            for (std::size_t column = 0; column < width; ++column) {
                for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                    projected[token * width + column] += input[x_index(token, hidden)]
                        * dequantized[p][weight_index(projection, hidden, column)];
                }
            }
        }
        for (std::size_t token = 0; token < kSeqLen; ++token) {
            for (std::size_t column = 0; column < width; ++column) {
                auto expected = projected[token * width + column];
                if (projection != Projection::Value) {
                    const auto dimension = column % kHeadDim;
                    const auto pair_dimension = dimension % kTile;
                    const auto head_base = column - dimension;
                    const auto lo = projected[token * width + head_base + pair_dimension];
                    const auto hi = projected[token * width + head_base + pair_dimension + kTile];
                    const auto inverse_frequency = 1.0f / std::pow(
                        kRopeTheta,
                        static_cast<float>(2 * pair_dimension) / static_cast<float>(kHeadDim));
                    const auto angle = static_cast<float>(token) * inverse_frequency;
                    const auto cos_value = ftlpu::Fp16::from_float(std::cos(angle)).to_float();
                    const auto sin_value = ftlpu::Fp16::from_float(std::sin(angle)).to_float();
                    expected = dimension < kTile
                        ? lo * cos_value - hi * sin_value
                        : hi * cos_value + lo * sin_value;
                }
                const auto expected_bits = ftlpu::Fp16::from_float(expected).bits();
                golden_outputs[p][token * width + column] =
                    ftlpu::Fp16::from_bits(expected_bits).to_float();
                const auto actual_bits = read_output(system, projection, token, column);
                if (actual_bits != expected_bits) {
                    std::cerr << projection_name(projection)
                              << " output mismatch at token=" << token
                              << " column=" << column
                              << " actual=" << ftlpu::Fp16::from_bits(actual_bits).to_float()
                              << " expected=" << ftlpu::Fp16::from_bits(expected_bits).to_float()
                              << '\n';
                    return 1;
                }
            }
        }
    }

    auto probabilities = std::vector<float>(
        kQueryHeads * kSeqLen * kSeqLen, 0.0f);
    const auto probability_index = [] (
        std::size_t head, std::size_t query, std::size_t key) {
        return (head * kSeqLen + query) * kSeqLen + key;
    };
    for (std::size_t query_head = 0; query_head < kQueryHeads; ++query_head) {
        const auto kv_head = query_head / (kQueryHeads / kKvHeads);
        for (std::size_t query_token = 0; query_token < kSeqLen; ++query_token) {
            auto logits = std::array<float, kSeqLen> {};
            auto maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t key_token = 0; key_token < kSeqLen; ++key_token) {
                auto score = 0.0f;
                for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
                    score += golden_outputs[static_cast<std::size_t>(Projection::Query)][
                        query_token * kQueryWidth + query_head * kHeadDim + dimension]
                        * golden_outputs[static_cast<std::size_t>(Projection::Key)][
                            key_token * kKvWidth + kv_head * kHeadDim + dimension];
                }
                logits[key_token] = score * kAttentionScale;
                maximum = std::max(maximum, logits[key_token]);
            }
            auto sum = 0.0f;
            for (auto& value : logits) {
                value = std::exp(value - maximum);
                sum += value;
            }
            for (std::size_t key_token = 0; key_token < kSeqLen; ++key_token) {
                const auto expected = ftlpu::Fp16::from_float(
                    logits[key_token] / sum).to_float();
                probabilities[probability_index(
                    query_head, query_token, key_token)] = expected;
                const auto actual = read_probability(
                    system, query_head, query_token, key_token);
                if (std::fabs(actual - expected) > 2.0e-3f) {
                    std::cerr << "softmax mismatch at head=" << query_head
                              << " query=" << query_token
                              << " key=" << key_token
                              << " actual=" << actual
                              << " expected=" << expected << '\n';
                    return 1;
                }
            }
        }
    }

    auto contexts = std::vector<float>(kSeqLen * kHidden, 0.0f);
    for (std::size_t query_head = 0; query_head < kQueryHeads; ++query_head) {
        const auto kv_head = query_head / (kQueryHeads / kKvHeads);
        for (std::size_t query_token = 0; query_token < kSeqLen; ++query_token) {
            for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
                auto expected = 0.0f;
                for (std::size_t key_token = 0; key_token < kSeqLen; ++key_token) {
                    expected += probabilities[probability_index(
                        query_head, query_token, key_token)]
                        * golden_outputs[static_cast<std::size_t>(Projection::Value)][
                            key_token * kKvWidth + kv_head * kHeadDim + dimension];
                }
                expected = ftlpu::Fp16::from_float(expected).to_float();
                contexts[query_token * kHidden + query_head * kHeadDim + dimension] = expected;
                const auto actual = read_context(
                    system, query_head, query_token, dimension);
                if (std::fabs(actual - expected) > 5.0e-3f) {
                    std::cerr << "context mismatch at head=" << query_head
                              << " query=" << query_token
                              << " dimension=" << dimension
                              << " actual=" << actual
                              << " expected=" << expected << '\n';
                    return 1;
                }
            }
        }
    }

    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            auto expected = 0.0f;
            for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                expected += contexts[token * kHidden + hidden]
                    * output_dequantized[hidden * kHidden + column];
            }
            expected = ftlpu::Fp16::from_float(expected).to_float();
            const auto actual = read_attention_output(system, token, column);
            const auto error = std::fabs(actual - expected);
            if (error > max_o_proj_error) {
                max_o_proj_error = error;
                max_o_proj_token = token;
                max_o_proj_column = column;
                max_o_proj_actual = actual;
                max_o_proj_expected = expected;
            }
        }
    }
    if (max_o_proj_error > 7.0e-2f) {
        std::cerr << "o_proj max mismatch at token=" << max_o_proj_token
                  << " column=" << max_o_proj_column
                  << " actual=" << max_o_proj_actual
                  << " expected=" << max_o_proj_expected
                  << " error=" << max_o_proj_error << '\n';
        return 1;
    }

    std::cout << "SmolLM2 attention passed: Q/K/V projection, RoPE, scaled softmax, "
              << "GQA context, and o_proj[128,576] verified; scheduled_cycles="
              << schedule.end_cycle() + 16 << '\n';
    return 0;
}
catch (const std::exception& ex)
{
    std::cerr << "SmolLM2 attention test failed: " << ex.what() << '\n';
    return 1;
}
