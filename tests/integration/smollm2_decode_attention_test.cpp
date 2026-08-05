#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "vxm_alu_program.hpp"
#include "smollm2_layer_phases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kPrefillLength = 128;
constexpr std::size_t kDecodePosition = kPrefillLength;
constexpr std::size_t kSequenceLength = kPrefillLength + 1;
constexpr std::size_t kPaddedSequenceLength = 144;
constexpr std::size_t kProbabilityTokensPerReduction =
    ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kProbabilityReductions =
    kPaddedSequenceLength / kProbabilityTokensPerReduction;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kQueryHeads = 9;
constexpr std::size_t kKvHeads = 3;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;
constexpr std::size_t kQueriesPerKv = kQueryHeads / kKvHeads;
constexpr std::size_t kDecodeStages =
    ftlpu::hw::kTileRows * ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kDecodeBlock =
    kDecodeStages * ftlpu::hw::kLanesPerTile;
constexpr std::size_t kProbabilityRows =
    kProbabilityReductions * kDecodeBlock;
constexpr std::size_t kOutputGroup = ftlpu::hw::kMxmSupercellColumns;
constexpr std::size_t kContextChunks = kHidden / kOutputGroup;
constexpr std::size_t kContextReductions =
    (kContextChunks + ftlpu::hw::kMxmSupercellsPerPlane - 1)
    / ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kContextRows = kContextReductions * kDecodeBlock;
constexpr std::size_t kGroupsPerAccumulatorRow =
    ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kActivationAddressBase = 512;
constexpr std::size_t kWeightAddressBase = 1024;
constexpr std::size_t kOutputAddressBase = 20000;
constexpr std::size_t kScaledScoreAddressBase = 22000;
constexpr std::size_t kExpScoreAddressBase = 23000;
constexpr std::size_t kKeyCacheAddressBase = 24000;
constexpr std::size_t kValueCacheAddressBase = 27000;
constexpr std::size_t kKvCacheHeadStride = 512;
constexpr std::array<std::size_t, 16> kProbabilityPackSlices {
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31};
constexpr std::array<std::size_t, 4> kScaledScoreSlices {32, 33, 34, 35};
constexpr std::array<std::size_t, 4> kExpScoreSlices {36, 37, 38, 39};
constexpr std::size_t kSoftmaxOutputStream = 8;
constexpr std::size_t kProbabilityPackAddressBase = 30000;
constexpr std::size_t kRopeTableAddressBase = 31000;
constexpr std::size_t kQueryRopeAddressBase = 32000;
constexpr std::size_t kContextActivationAddressBase = 33000;
constexpr std::array<std::size_t, 4> kRopeTableSlices {8, 9, 10, 11};
constexpr std::array<float, kKvHeads> kKeyCacheScales {
    0.0078125f, 0.009765625f, 0.01171875f};
constexpr std::array<float, kKvHeads> kValueCacheScales {
    0.0078125f, 0.009765625f, 0.01171875f};
constexpr std::array<std::size_t, 8> kActivationSlices {
    44, 45, 46, 47, 48, 49, 50, 51};
constexpr std::array<std::size_t, 2> kOutputSlices {40, 41};
constexpr float kRopeTheta = 100000.0f;
const float kAttentionScale =
    1.0f / std::sqrt(static_cast<float>(kHeadDim));

static_assert(kDecodeBlock == 128);
static_assert(kHidden == kQueryHeads * kHeadDim);
static_assert(kKvWidth == 192);
static_assert(kQueryHeads % kKvHeads == 0);

using HeadVector = std::array<float, kHeadDim>;
using KvCache =
    std::array<std::array<HeadVector, kSequenceLength>, kKvHeads>;

using ResidentHeads = std::vector<std::vector<float>>;

struct TraceEvent {
    std::size_t start{0};
    std::size_t end{0};
    std::string resource;
    std::string detail;
};

struct QuantizedMatrix {
    std::size_t rows{0};
    std::size_t columns{0};
    std::size_t padded_columns{0};
    std::vector<float> scales;
    std::vector<std::int8_t> values;
    std::vector<float> dequantized;

    std::size_t groups() const noexcept
    {
        return padded_columns / kOutputGroup;
    }
};

std::vector<float> softmax(const std::vector<float>& scores);
void apply_rope(HeadVector& vector, std::size_t position);
HeadVector slice_head(const std::vector<float>& values, std::size_t head);

float bf16(float value)
{
    return ftlpu::Bf16::from_float(value).to_float();
}

std::size_t east_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t west_latency(std::size_t slice)
{
    return ftlpu::hw::kSystemStreamRegisterColumns - 1
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t mem_queue(ftlpu::Hemisphere hemisphere, std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(hemisphere, slice);
}

template <typename Generator>
QuantizedMatrix quantize_matrix(
    std::size_t rows,
    std::size_t columns,
    Generator generator)
{
    const auto groups = (columns + kOutputGroup - 1) / kOutputGroup;
    const auto padded_columns = groups * kOutputGroup;
    auto result = QuantizedMatrix {
        rows,
        columns,
        padded_columns,
        std::vector<float>(groups),
        std::vector<std::int8_t>(rows * padded_columns),
        std::vector<float>(rows * padded_columns)};

    for (std::size_t group = 0; group < groups; ++group) {
        auto max_abs = 0.0f;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t lane = 0; lane < kOutputGroup; ++lane) {
                const auto column = group * kOutputGroup + lane;
                if (column < columns) {
                    max_abs = std::max(
                        max_abs, std::fabs(generator(row, column)));
                }
            }
        }
        const auto scale = max_abs == 0.0f
            ? bf16(1.0f)
            : bf16(max_abs / 127.0f);
        result.scales[group] = scale;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t lane = 0; lane < kOutputGroup; ++lane) {
                const auto column = group * kOutputGroup + lane;
                const auto source = column < columns
                    ? generator(row, column)
                    : 0.0f;
                const auto quantized = std::clamp(
                    static_cast<int>(std::lround(source / scale)),
                    -127,
                    127);
                const auto index = row * padded_columns + column;
                result.values[index] = static_cast<std::int8_t>(quantized);
                result.dequantized[index] = bf16(
                    static_cast<float>(quantized) * scale);
            }
        }
    }
    return result;
}

template <typename Generator>
QuantizedMatrix quantize_matrix_fixed(
    std::size_t rows,
    std::size_t columns,
    float scale,
    Generator generator)
{
    const auto groups = (columns + kOutputGroup - 1) / kOutputGroup;
    const auto padded_columns = groups * kOutputGroup;
    auto result = QuantizedMatrix {
        rows,
        columns,
        padded_columns,
        std::vector<float>(groups, bf16(scale)),
        std::vector<std::int8_t>(rows * padded_columns),
        std::vector<float>(rows * padded_columns)};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < padded_columns; ++column) {
            const auto source = column < columns
                ? generator(row, column)
                : 0.0f;
            const auto quantized = std::clamp(
                static_cast<int>(std::nearbyint(source / scale)),
                -128,
                127);
            const auto index = row * padded_columns + column;
            result.values[index] = static_cast<std::int8_t>(quantized);
            result.dequantized[index] = bf16(
                static_cast<float>(quantized) * bf16(scale));
        }
    }
    return result;
}
std::vector<float> reference_gemv(
    const std::vector<float>& input,
    const QuantizedMatrix& matrix)
{
    auto output = std::vector<float>(matrix.columns);
    const auto reductions =
        (matrix.rows + kDecodeBlock - 1) / kDecodeBlock;
    for (std::size_t column = 0; column < matrix.columns; ++column) {
        auto accumulated = 0.0f;
        for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
            auto wave = 0.0f;
            for (std::size_t stage = 0; stage < kDecodeStages; ++stage) {
                auto partial = 0.0f;
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile;
                     ++lane) {
                    const auto row = reduction * kDecodeBlock
                        + stage * ftlpu::hw::kLanesPerTile + lane;
                    if (row < matrix.rows) {
                        partial += input[row]
                            * matrix.dequantized[
                                row * matrix.padded_columns + column];
                    }
                }
                wave += partial;
            }
            accumulated += wave;
        }
        output[column] = bf16(accumulated);
    }
    return output;
}
class Schedule {
public:
    Schedule(
        ftlpu::InstructionControlUnit& icu,
        std::vector<TraceEvent>& trace,
        std::size_t trace_offset)
        : icu_(icu), trace_(trace), trace_offset_(trace_offset)
    {
    }

    void mem_at(
        ftlpu::Hemisphere hemisphere,
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        const auto queue = mem_queue(hemisphere, slice);
        pad(mem_[queue], cycle, [&](std::size_t count) {
            icu_.enqueue_mem_nop(queue, count);
        });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
    }

    void load_at(std::size_t mxm, std::size_t cycle, std::size_t buffer)
    {
        pad(mxm_load_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_load_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::DecodeLoadActivation(buffer));
        advance(mxm_load_[mxm], cycle + 1);
    }

    void decode_compute_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t buffer,
        float scale,
        std::size_t accumulator_address,
        std::size_t accumulator_column,
        ftlpu::MxmAccumulatorDestination destination,
        bool clear)
    {
        pad(mxm_dequant_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_dequant_nop(mxm, count);
        });
        icu_.enqueue_mxm_dequant(
            mxm, ftlpu::MxmDequantInstruction::Scale(scale));
        advance(mxm_dequant_[mxm], cycle + 1);

        pad(mxm_compute_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_compute_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::DecodeStreamCompute(
                buffer,
                0,
                ftlpu::MxmDataFormat::BFloat16,
                accumulator_address,
                accumulator_column,
                destination,
                clear));
        advance(mxm_compute_[mxm], cycle + 1);
    }

    void rope_hold_low_at(std::size_t cycle)
    {
        ftlpu::test::enqueue_alu_at(
            icu_, vxm_, 0, cycle,
            ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamBFloat16(32),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Float32,
                std::nullopt,
                ftlpu::Hemisphere::East,
                ftlpu::Hemisphere::East});
        trace(cycle, cycle + 1, "VXM.ALU0", "RoPE hold low half");
        end_ = std::max(end_, cycle + 1);
    }

    void rope_rotate_at(std::size_t cycle)
    {
        const auto instruction = [](ftlpu::VxmAluOpcode opcode,
                                    ftlpu::VxmLaneOperand lhs,
                                    ftlpu::VxmLaneOperand rhs,
                                    ftlpu::VxmCastTarget cast,
                                    std::optional<std::size_t> output = std::nullopt) {
            return ftlpu::VxmLaneAluInstruction {
                opcode, lhs, rhs, 1.0f, 0, cast, output,
                ftlpu::Hemisphere::East, ftlpu::Hemisphere::East};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 1, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::StreamBFloat16(36),
            ftlpu::VxmCastTarget::Float32));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 2, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamBFloat16(32),
            ftlpu::VxmLaneOperand::StreamBFloat16(38),
            ftlpu::VxmCastTarget::Float32));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 3, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamBFloat16(32),
            ftlpu::VxmLaneOperand::StreamBFloat16(36),
            ftlpu::VxmCastTarget::Float32));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 4, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::StreamBFloat16(38),
            ftlpu::VxmCastTarget::Float32));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 5, cycle + 1, instruction(
            ftlpu::VxmAluOpcode::Subtract,
            ftlpu::VxmLaneOperand::Alu(1),
            ftlpu::VxmLaneOperand::Alu(2),
            ftlpu::VxmCastTarget::BFloat16, 0));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 6, cycle + 1, instruction(
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(3),
            ftlpu::VxmLaneOperand::Alu(4),
            ftlpu::VxmCastTarget::BFloat16));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 7, cycle + 2, instruction(
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(6),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            ftlpu::VxmCastTarget::BFloat16, 2));
        trace(cycle, cycle + 1, "VXM.ALU1", "RoPE low * cos");
        trace(cycle, cycle + 1, "VXM.ALU2", "RoPE high * sin");
        trace(cycle, cycle + 1, "VXM.ALU3", "RoPE high * cos");
        trace(cycle, cycle + 1, "VXM.ALU4", "RoPE low * sin");
        trace(cycle + 1, cycle + 2, "VXM.ALU5", "RoPE low*cos - high*sin");
        trace(cycle + 1, cycle + 2, "VXM.ALU6", "RoPE high*cos + low*sin");
        trace(cycle + 2, cycle + 3, "VXM.ALU7", "RoPE BF16 high-half write");
        end_ = std::max(end_, cycle + 3);
    }
    void quantize_bf16_at(
        std::size_t cycle,
        std::size_t input_stream_base,
        float scale,
        std::size_t output_stream,
        const std::string& label)
    {
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 0, cycle, {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamBFloat16(input_stream_base),
            ftlpu::VxmLaneOperand::Imm(1.0f / scale),
            1.0f, 0, ftlpu::VxmCastTarget::Float32, std::nullopt,
            ftlpu::Hemisphere::East, ftlpu::Hemisphere::East});
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 1, cycle + 1, {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f, 0, ftlpu::VxmCastTarget::Int8, output_stream,
            ftlpu::Hemisphere::East, ftlpu::Hemisphere::East});
        trace(cycle, cycle + 1, "VXM.ALU0", label + " divide by scale");
        trace(cycle + 1, cycle + 2, "VXM.ALU1", label + " round/saturate INT8");
        end_ = std::max(end_, cycle + 2);
    }
    void quantize_bf16_broadcast8_at(
        std::size_t cycle,
        std::size_t input_stream_base,
        float scale,
        std::size_t output_stream_base,
        const std::string& label)
    {
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 0, cycle, {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamBFloat16(input_stream_base),
            ftlpu::VxmLaneOperand::Imm(1.0f / scale),
            1.0f, 0, ftlpu::VxmCastTarget::Float32, std::nullopt,
            ftlpu::Hemisphere::East, ftlpu::Hemisphere::East});
        trace(cycle, cycle + 1, "VXM.ALU0", label + " divide by scale");
        for (std::size_t duplicate = 0; duplicate < kOutputGroup; ++duplicate) {
            const auto alu = 1 + duplicate;
            ftlpu::test::enqueue_alu_at(icu_, vxm_, alu, cycle + 1, {
                ftlpu::VxmAluOpcode::Cast,
                ftlpu::VxmLaneOperand::Alu(0),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Int8,
                output_stream_base + duplicate,
                ftlpu::Hemisphere::East, ftlpu::Hemisphere::East});
            trace(
                cycle + 1, cycle + 2,
                "VXM.ALU" + std::to_string(alu),
                label + " INT8 duplicate=" + std::to_string(duplicate));
        }
        end_ = std::max(end_, cycle + 2);
    }
    void int8_pass_at(
        std::size_t cycle,
        std::size_t alu,
        std::size_t input_stream,
        std::size_t output_stream,
        const std::string& label)
    {
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu, cycle, {
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::StreamInt8(input_stream),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f, 0, ftlpu::VxmCastTarget::Int8, output_stream,
            ftlpu::Hemisphere::East, ftlpu::Hemisphere::East});
        trace(
            cycle,
            cycle + 1,
            "VXM.ALU" + std::to_string(alu),
            label);
        end_ = std::max(end_, cycle + 1);
    }

    void softmax_pass1_at(
        std::size_t cycle,
        bool first_token,
        bool valid_token)
    {
        const auto instruction = [](ftlpu::VxmAluOpcode opcode,
                                    ftlpu::VxmLaneOperand lhs,
                                    ftlpu::VxmLaneOperand rhs,
                                    ftlpu::VxmCastTarget cast,
                                    std::optional<std::size_t> output = std::nullopt) {
            return ftlpu::VxmLaneAluInstruction {
                opcode, lhs, rhs, 1.0f, 0, cast, output,
                ftlpu::Hemisphere::East, ftlpu::Hemisphere::East};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 0, cycle, instruction(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamBFloat16(32),
            ftlpu::VxmLaneOperand::Imm(kAttentionScale),
            ftlpu::VxmCastTarget::Float32));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 7, cycle + 1, instruction(
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::Imm(valid_token ? 0.0f : -1.0e9f),
            ftlpu::VxmCastTarget::Float32, kSoftmaxOutputStream));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 1, cycle + 2, instruction(
            first_token ? ftlpu::VxmAluOpcode::Pass
                        : ftlpu::VxmAluOpcode::Max,
            first_token ? ftlpu::VxmLaneOperand::Alu(7)
                        : ftlpu::VxmLaneOperand::Alu(1),
            first_token ? ftlpu::VxmLaneOperand::Imm(0.0f)
                        : ftlpu::VxmLaneOperand::Alu(7),
            ftlpu::VxmCastTarget::Float32));
        end_ = std::max(end_, cycle + 3);
    }

    void softmax_pass2_at(std::size_t cycle, bool first_token)
    {
        const auto instruction = [](ftlpu::VxmAluOpcode opcode,
                                    ftlpu::VxmLaneOperand lhs,
                                    ftlpu::VxmLaneOperand rhs,
                                    std::optional<std::size_t> output = std::nullopt) {
            return ftlpu::VxmLaneAluInstruction {
                opcode, lhs, rhs, 1.0f, 0,
                ftlpu::VxmCastTarget::Float32, output,
                ftlpu::Hemisphere::East, ftlpu::Hemisphere::East};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 2, cycle, instruction(
            ftlpu::VxmAluOpcode::Subtract,
            ftlpu::VxmLaneOperand::StreamFloat32(32),
            ftlpu::VxmLaneOperand::Alu(1)));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 3, cycle + 1, instruction(
            ftlpu::VxmAluOpcode::Exp,
            ftlpu::VxmLaneOperand::Alu(2),
            ftlpu::VxmLaneOperand::Imm(0.0f), kSoftmaxOutputStream));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 4, cycle + 2, instruction(
            first_token ? ftlpu::VxmAluOpcode::Pass
                        : ftlpu::VxmAluOpcode::Add,
            first_token ? ftlpu::VxmLaneOperand::Alu(3)
                        : ftlpu::VxmLaneOperand::Alu(4),
            first_token ? ftlpu::VxmLaneOperand::Imm(0.0f)
                        : ftlpu::VxmLaneOperand::Alu(3)));
        end_ = std::max(end_, cycle + 3);
    }

    void softmax_pass3_at(std::size_t cycle)
    {
        const auto instruction = [](ftlpu::VxmAluOpcode opcode,
                                    ftlpu::VxmLaneOperand lhs,
                                    ftlpu::VxmLaneOperand rhs,
                                    ftlpu::VxmCastTarget cast,
                                    std::optional<std::size_t> output = std::nullopt) {
            return ftlpu::VxmLaneAluInstruction {
                opcode, lhs, rhs, 1.0f, 0, cast, output,
                ftlpu::Hemisphere::East, ftlpu::Hemisphere::East};
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
            ftlpu::VxmCastTarget::BFloat16, kSoftmaxOutputStream));
        end_ = std::max(end_, cycle + 2);
    }
    void trace(
        std::size_t start,
        std::size_t end,
        std::string resource,
        std::string detail)
    {
        trace_.push_back(TraceEvent {
            trace_offset_ + start,
            trace_offset_ + end,
            std::move(resource),
            std::move(detail)});
    }

    std::size_t end_cycle() const noexcept { return end_; }

private:
    template <typename Emit>
    static void pad(std::size_t cursor, std::size_t cycle, Emit emit)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                "offline decode attention schedule overlaps an ICU queue");
        }
        emit(cycle - cursor);
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_ = std::max(end_, next);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::vector<TraceEvent>& trace_;
    std::size_t trace_offset_{0};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, ftlpu::TspSliceSystem::kMxmCount> mxm_load_{};
    std::array<std::size_t, ftlpu::TspSliceSystem::kMxmCount> mxm_compute_{};
    std::array<std::size_t, ftlpu::TspSliceSystem::kMxmCount> mxm_dequant_{};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_{};
    std::size_t end_{0};
};

class DecodeAttentionHarness {
public:
    explicit DecodeAttentionHarness(ftlpu::TspSliceSystem& system)
        : system_(system)
    {
    }

    void enable_logs(const std::filesystem::path& directory)
    {
        std::filesystem::create_directories(directory);
        icu_log_.open(directory / "icu.log", std::ios::trunc);
        mem_log_.open(directory / "mem.log", std::ios::trunc);
        if (!icu_log_ || !mem_log_) {
            throw std::runtime_error(
                "cannot open SmolLM2 decode attention logs");
        }
        logs_enabled_ = true;
    }

    std::vector<float> gemv(
        const std::vector<float>& source_input,
        const QuantizedMatrix& matrix,
        const std::string& label,
        std::optional<std::size_t> resident_weight_base = std::nullopt,
        bool activation_is_resident = false,
        std::size_t activation_address_base = kActivationAddressBase,
        std::optional<std::size_t> resident_output_chunk_base = std::nullopt,
        std::size_t resident_output_address_base =
            kContextActivationAddressBase)
    {
        if (source_input.size() != matrix.rows) {
            throw std::invalid_argument(
                label + ": activation size does not match matrix rows");
        }
        auto input = source_input;
        for (auto& value : input) value = bf16(value);

        system_.reset_execution_state();
        if (!activation_is_resident) {
            initialize_activation(input, matrix.rows, activation_address_base);
        }
        if (!resident_weight_base.has_value()) {
            initialize_weights(matrix, kWeightAddressBase);
        }

        auto schedule = Schedule(system_.icu(), trace_, total_cycles_);
        auto phase_cycle = std::size_t {24};
        const auto reductions =
            (matrix.rows + kDecodeBlock - 1) / kDecodeBlock;
        for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
            const auto buffer = reduction % 2;
            schedule_activation_load(
                schedule, phase_cycle, reduction, buffer,
                activation_address_base);
            const auto compute_start = phase_cycle + 2;
            const auto final_reduction = reduction + 1 == reductions;
            schedule_weight_waves(
                schedule,
                compute_start,
                reduction,
                matrix,
                buffer,
                final_reduction,
                resident_weight_base.value_or(kWeightAddressBase),
                resident_output_chunk_base,
                resident_output_address_base);
            schedule.trace(
                phase_cycle,
                phase_cycle + 1,
                "MXM.E0.ActivationLoad",
                label + " activation reduction="
                    + std::to_string(reduction));
            schedule.trace(
                compute_start - east_latency(0),
                compute_start + matrix.groups()
                    + (ftlpu::hw::kMxmSupercellsPerPlane - 1)
                        * ftlpu::hw::kTileRows,
                "MEM.E.Read",
                label + " streamed INT8 weights reduction="
                    + std::to_string(reduction));
            schedule.trace(
                compute_start,
                compute_start + matrix.groups(),
                "MXM.E0.Dequant",
                label + " BF16 group scales reduction="
                    + std::to_string(reduction));
            schedule.trace(
                compute_start,
                compute_start + matrix.groups(),
                "MXM.E0.Compute",
                label + " decode waves reduction="
                    + std::to_string(reduction)
                    + (final_reduction
                        ? " dst=stream+clear BF16"
                        : " dst=sram retain"));
            phase_cycle = compute_start + matrix.groups()
                + kDecodeStages + 4;
        }
        const auto cycles = schedule.end_cycle() + 8;
        run(cycles, label);
        total_cycles_ += cycles;

        auto actual = resident_output_chunk_base.has_value()
            ? read_resident_output(
                matrix.columns,
                *resident_output_chunk_base,
                resident_output_address_base)
            : read_output(matrix.columns);
        const auto expected = reference_gemv(input, matrix);
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const auto actual_bits =
                ftlpu::Bf16::from_float(actual[index]).bits();
            const auto expected_bits =
                ftlpu::Bf16::from_float(expected[index]).bits();
            if (actual_bits != expected_bits) {
                throw std::runtime_error(
                    label + " mismatch at " + std::to_string(index)
                    + " actual=" + std::to_string(actual[index])
                    + " expected=" + std::to_string(expected[index]));
            }
        }
        return actual;
    }

    void install_resident_weights(
        const QuantizedMatrix& matrix,
        std::size_t address_base)
    {
        initialize_weights(matrix, address_base);
    }

    void load_resident_weights_via_instructions(
        const QuantizedMatrix& matrix,
        std::size_t target_address_base,
        const std::string& label)
    {
        constexpr auto kStagingAddressBase = std::size_t {40000};
        initialize_weights(matrix, kStagingAddressBase);
        system_.reset_execution_state();
        auto schedule = Schedule(system_.icu(), trace_, total_cycles_);
        auto cycle = std::size_t {20};
        const auto reductions =
            (matrix.rows + kDecodeBlock - 1) / kDecodeBlock;
        const auto addresses = reductions * matrix.groups();
        for (std::size_t address = 0; address < addresses; ++address) {
            for (std::size_t wave = 0; wave < 2; ++wave) {
                const auto vxm_cycle = cycle;
                for (std::size_t stream = 0;
                     stream < ftlpu::VxmLane::kAluCount;
                     ++stream) {
                    const auto slice =
                        wave * ftlpu::VxmLane::kAluCount + stream;
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        vxm_cycle - west_read_latency(slice),
                        ftlpu::MemInstruction::Read(
                            kStagingAddressBase + address,
                            ftlpu::StreamId::West(slice)));
                    schedule.int8_pass_at(
                        vxm_cycle,
                        stream,
                        ftlpu::hw::kEastStreams + slice,
                        stream,
                        label + " address=" + std::to_string(address)
                            + " slice=" + std::to_string(slice));
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        vxm_cycle + 1
                            + slice / ftlpu::hw::kMemSlicesPerGroup,
                        ftlpu::MemInstruction::Write(
                            target_address_base + address,
                            ftlpu::StreamId::East(stream)));
                }
                schedule.trace(
                    vxm_cycle - west_read_latency(
                        wave * ftlpu::VxmLane::kAluCount
                        + ftlpu::VxmLane::kAluCount - 1),
                    vxm_cycle + 2
                        + (wave * ftlpu::VxmLane::kAluCount
                            + ftlpu::VxmLane::kAluCount - 1)
                            / ftlpu::hw::kMemSlicesPerGroup,
                    "KVCache.Load",
                    label + " address=" + std::to_string(address)
                        + " streams=" + std::to_string(
                            wave * ftlpu::VxmLane::kAluCount)
                        + ".." + std::to_string(
                            (wave + 1) * ftlpu::VxmLane::kAluCount - 1));
                cycle += 12;
            }
        }
        const auto cycles = schedule.end_cycle() + 16;
        run(cycles, label + " instruction load");
        total_cycles_ += cycles;

        for (std::size_t address = 0; address < addresses; ++address) {
            for (std::size_t slice = 0;
                 slice < ftlpu::hw::kEastStreams;
                 ++slice) {
                for (std::size_t tile = 0;
                     tile < ftlpu::hw::kTileRows;
                     ++tile) {
                    for (std::size_t lane = 0;
                         lane < ftlpu::hw::kLanesPerTile;
                         ++lane) {
                        const auto expected = system_.read_mem_sram_lane_byte(
                            ftlpu::Hemisphere::East,
                            slice,
                            tile,
                            kStagingAddressBase + address,
                            lane);
                        const auto actual = system_.read_mem_sram_lane_byte(
                            ftlpu::Hemisphere::East,
                            slice,
                            tile,
                            target_address_base + address,
                            lane);
                        if (actual != expected) {
                            throw std::runtime_error(
                                label + " instruction cache load mismatch");
                        }
                    }
                }
            }
        }
    }

    void clear_resident_activation(
        std::size_t address_base,
        std::size_t reductions)
    {
        for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
            for (const auto slice : kActivationSlices) {
                for (std::size_t tile = 0;
                     tile < ftlpu::hw::kTileRows;
                     ++tile) {
                    for (std::size_t lane = 0;
                         lane < ftlpu::hw::kLanesPerTile;
                         ++lane) {
                        system_.initialize_mem_sram_lane_byte(
                            ftlpu::Hemisphere::East,
                            slice,
                            tile,
                            address_base + reduction,
                            lane,
                            0);
                    }
                }
            }
        }
    }
    ResidentHeads rope_heads_from_projection(
        const std::vector<float>& projection,
        std::size_t head_count,
        const std::string& label)
    {
        if (projection.size() != head_count * kHeadDim) {
            throw std::invalid_argument(
                label + " projection width does not match head count");
        }
        for (std::size_t beat = 0; beat < kHeadDim / 2 / kOutputGroup; ++beat) {
            for (std::size_t tile = 0;
                 tile < ftlpu::hw::kTileRows;
                 ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile;
                     ++lane) {
                    const auto pair = beat * kOutputGroup + lane;
                    const auto exponent = 2.0f * static_cast<float>(pair)
                        / static_cast<float>(kHeadDim);
                    const auto angle = static_cast<float>(kDecodePosition)
                        / std::pow(kRopeTheta, exponent);
                    const auto cosine = ftlpu::Bf16::from_float(
                        std::cos(angle)).bits();
                    const auto sine = ftlpu::Bf16::from_float(
                        std::sin(angle)).bits();
                    for (std::size_t byte = 0; byte < 2; ++byte) {
                        system_.initialize_mem_sram_lane_byte(
                            ftlpu::Hemisphere::East,
                            kRopeTableSlices[byte],
                            tile,
                            kRopeTableAddressBase + beat,
                            lane,
                            static_cast<std::uint8_t>(
                                (cosine >> (byte * 8)) & 0xffu));
                        system_.initialize_mem_sram_lane_byte(
                            ftlpu::Hemisphere::East,
                            kRopeTableSlices[2 + byte],
                            tile,
                            kRopeTableAddressBase + beat,
                            lane,
                            static_cast<std::uint8_t>(
                                (sine >> (byte * 8)) & 0xffu));
                    }
                }
            }
        }
        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t reduction = 0; reduction < 2; ++reduction) {
                const auto address = kQueryRopeAddressBase + head * 2 + reduction;
                for (const auto slice : kActivationSlices) {
                    for (std::size_t tile = 0;
                         tile < ftlpu::hw::kTileRows;
                         ++tile) {
                        for (std::size_t lane = 0;
                             lane < ftlpu::hw::kLanesPerTile;
                             ++lane) {
                            system_.initialize_mem_sram_lane_byte(
                                ftlpu::Hemisphere::East,
                                slice,
                                tile,
                                address,
                                lane,
                                0);
                        }
                    }
                }
            }
        }

        system_.reset_execution_state();
        auto schedule = Schedule(system_.icu(), trace_, total_cycles_);
        auto cycle = std::size_t {24};
        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t beat = 0;
                 beat < kHeadDim / 2 / kOutputGroup;
                 ++beat) {
                const auto low_cycle = cycle;
                const auto high_cycle = cycle + 1;
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    const auto slice = kOutputSlices[byte];
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        low_cycle - west_read_latency(slice),
                        ftlpu::MemInstruction::Read(
                            kOutputAddressBase + head * 8 + beat,
                            ftlpu::StreamId::West(byte)));
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        high_cycle - west_read_latency(slice),
                        ftlpu::MemInstruction::Read(
                            kOutputAddressBase + head * 8 + 4 + beat,
                            ftlpu::StreamId::West(byte)));
                }
                for (std::size_t byte = 0;
                     byte < kRopeTableSlices.size();
                     ++byte) {
                    const auto slice = kRopeTableSlices[byte];
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        high_cycle - west_read_latency(slice),
                        ftlpu::MemInstruction::Read(
                            kRopeTableAddressBase + beat,
                            ftlpu::StreamId::West(4 + byte)));
                }
                schedule.rope_hold_low_at(low_cycle);
                schedule.rope_rotate_at(high_cycle);
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    const auto slice = kActivationSlices[beat * 2 + byte];
                    const auto write_cycle = high_cycle + 2
                        + slice / ftlpu::hw::kMemSlicesPerGroup;
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        write_cycle,
                        ftlpu::MemInstruction::Write(
                            kQueryRopeAddressBase + head * 2,
                            ftlpu::StreamId::East(byte)));
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        write_cycle + 1,
                        ftlpu::MemInstruction::Write(
                            kQueryRopeAddressBase + head * 2 + 1,
                            ftlpu::StreamId::East(2 + byte)));
                }
                cycle += 4;
            }
        }

        const auto cycles = schedule.end_cycle() + 16;
        run(cycles, label + " RoPE");
        total_cycles_ += cycles;

        auto result = ResidentHeads(head_count);
        for (std::size_t head = 0; head < head_count; ++head) {
            auto expected = slice_head(projection, head);
            apply_rope(expected, kDecodePosition);
            result[head] = std::vector<float>(2 * kDecodeBlock);
            for (std::size_t reduction = 0; reduction < 2; ++reduction) {
                for (std::size_t column = 0;
                     column < ftlpu::hw::kMxmSupercellsPerPlane;
                     ++column) {
                    for (std::size_t lane = 0;
                         lane < ftlpu::hw::kLanesPerTile;
                         ++lane) {
                        const auto dimension = reduction * 32
                            + column * kOutputGroup + lane;
                        const auto row = reduction * kDecodeBlock
                            + (column * ftlpu::hw::kTileRows
                                + ftlpu::hw::kTileRows - 1)
                                * ftlpu::hw::kLanesPerTile
                            + lane;
                        std::uint16_t bits = 0;
                        for (std::size_t byte = 0; byte < 2; ++byte) {
                            bits |= static_cast<std::uint16_t>(
                                system_.read_mem_sram_lane_byte(
                                    ftlpu::Hemisphere::East,
                                    kActivationSlices[column * 2 + byte],
                                    ftlpu::hw::kTileRows - 1,
                                    kQueryRopeAddressBase + head * 2 + reduction,
                                    lane)) << (byte * 8);
                        }
                        const auto actual =
                            ftlpu::Bf16::from_bits(bits).to_float();
                        result[head][row] = actual;
                        if (actual != expected[dimension]) {
                            throw std::runtime_error(
                                label + " hardware RoPE mismatch head="
                                + std::to_string(head)
                                + " dim=" + std::to_string(dimension));
                        }
                    }
                }
            }
        }
        return result;
    }
    void quantize_resident_keys_to_cache()
    {
        system_.reset_execution_state();
        auto schedule = Schedule(system_.icu(), trace_, total_cycles_);
        constexpr auto kStart = std::size_t {24};
        for (std::size_t head = 0; head < kKvHeads; ++head) {
            for (std::size_t group = 0; group < kHeadDim / kOutputGroup; ++group) {
                const auto cycle = kStart + head * (kHeadDim / kOutputGroup) + group;
                const auto reduction = group / ftlpu::hw::kMxmSupercellsPerPlane;
                const auto column = group % ftlpu::hw::kMxmSupercellsPerPlane;
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    const auto slice = kActivationSlices[column * 2 + byte];
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        cycle - west_read_latency(slice),
                        ftlpu::MemInstruction::Read(
                            kQueryRopeAddressBase + head * 2 + reduction,
                            ftlpu::StreamId::West(byte)));
                }
                schedule.quantize_bf16_broadcast8_at(
                    cycle, ftlpu::hw::kEastStreams,
                    kKeyCacheScales[head], kSoftmaxOutputStream,
                    "K cache head=" + std::to_string(head));
                for (std::size_t duplicate = 0;
                     duplicate < kOutputGroup;
                     ++duplicate) {
                    const auto slice = column * kOutputGroup + duplicate;
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        slice,
                        cycle + 2
                            + slice / ftlpu::hw::kMemSlicesPerGroup,
                        ftlpu::MemInstruction::Write(
                            kKeyCacheAddressBase
                                + head * kKvCacheHeadStride
                                + reduction * kPaddedSequenceLength
                                + kDecodePosition,
                            ftlpu::StreamId::East(
                                kSoftmaxOutputStream + duplicate)));
                }
            }
        }
        schedule.trace(
            kStart,
            kStart + kKvWidth / kOutputGroup
                + ftlpu::hw::kMemGroups,
            "MEM.E.Write",
            "instruction-written INT8 decode K cache token=128");
        const auto cycles = schedule.end_cycle() + 16;
        run(cycles, "K cache quantize/write");
        total_cycles_ += cycles;
    }

    void quantize_packed_values_to_cache()
    {
        system_.reset_execution_state();
        auto schedule = Schedule(system_.icu(), trace_, total_cycles_);
        constexpr auto kStart = std::size_t {24};
        constexpr auto kValueGroups = kHeadDim / kOutputGroup;
        constexpr auto kValueReduction =
            kDecodePosition / ftlpu::hw::kMxmSupercellsPerPlane;
        constexpr auto kValueColumn =
            kDecodePosition % ftlpu::hw::kMxmSupercellsPerPlane;
        for (std::size_t dimension = 0; dimension < kKvWidth; ++dimension) {
            const auto cycle = kStart + dimension;
            const auto head = dimension / kHeadDim;
            const auto local_dimension = dimension % kHeadDim;
            const auto output_group = local_dimension / kOutputGroup;
            const auto output_lane = local_dimension % kOutputGroup;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kOutputSlices[byte];
                schedule.mem_at(
                    ftlpu::Hemisphere::East,
                    slice,
                    cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(
                        kOutputAddressBase + dimension,
                        ftlpu::StreamId::West(byte)));
            }
            schedule.quantize_bf16_at(
                cycle, ftlpu::hw::kEastStreams,
                kValueCacheScales[head], kSoftmaxOutputStream,
                "V cache head=" + std::to_string(head));
            const auto slice = kValueColumn * kOutputGroup + output_lane;
            schedule.mem_at(
                ftlpu::Hemisphere::East,
                slice,
                cycle + 2 + slice / ftlpu::hw::kMemSlicesPerGroup,
                ftlpu::MemInstruction::Write(
                    kValueCacheAddressBase
                        + head * kKvCacheHeadStride
                        + kValueReduction * kValueGroups
                        + output_group,
                    ftlpu::StreamId::East(kSoftmaxOutputStream)));
        }
        schedule.trace(
            kStart,
            kStart + kKvWidth + ftlpu::hw::kMemGroups,
            "MEM.E.Write",
            "instruction-written packed INT8 decode V cache token=128");
        const auto cycles = schedule.end_cycle() + 16;
        run(cycles, "V cache quantize/write");
        total_cycles_ += cycles;
    }
    std::vector<float> softmax_from_resident_scores(
        const std::vector<float>& expanded_scores,
        const std::string& label)
    {
        if (expanded_scores.size()
            != kPaddedSequenceLength * kOutputGroup) {
            throw std::invalid_argument(
                "softmax score layout must contain 144 eight-lane waves");
        }
system_.reset_execution_state();
        for (std::size_t reduction = 0;
             reduction < kProbabilityReductions;
             ++reduction) {
            for (const auto slice : kActivationSlices) {
                for (std::size_t tile = 0;
                     tile < ftlpu::hw::kTileRows;
                     ++tile) {
                    for (std::size_t lane = 0;
                         lane < ftlpu::hw::kLanesPerTile;
                         ++lane) {
                        system_.initialize_mem_sram_lane_byte(
                            ftlpu::Hemisphere::East,
                            slice,
                            tile,
                            kActivationAddressBase + reduction,
                            lane,
                            0);
                    }
                }
            }
        }
        auto schedule = Schedule(system_.icu(), trace_, total_cycles_);
        auto pass1_start = std::size_t {24};
        for (std::size_t token = 0;
             token < kPaddedSequenceLength;
             ++token) {
            const auto cycle = pass1_start + token;
            for (std::size_t byte = 0; byte < kOutputSlices.size(); ++byte) {
                const auto slice = kOutputSlices[byte];
                schedule.mem_at(
                    ftlpu::Hemisphere::East,
                    slice,
                    cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(
                        kOutputAddressBase + token,
                        ftlpu::StreamId::West(byte)));
            }
            schedule.softmax_pass1_at(
                cycle, token == 0, token < kSequenceLength);
            for (std::size_t byte = 0;
                 byte < kScaledScoreSlices.size();
                 ++byte) {
                const auto slice = kScaledScoreSlices[byte];
                schedule.mem_at(
                    ftlpu::Hemisphere::East,
                    slice,
                    cycle + 2
                        + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        kScaledScoreAddressBase + token,
                        ftlpu::StreamId::East(
                            kSoftmaxOutputStream + byte)));
            }
        }
        schedule.trace(
            pass1_start,
            pass1_start + kPaddedSequenceLength,
            "VXM.ALU0",
            label + " P1 scale by 1/sqrt(head_dim)");
        schedule.trace(
            pass1_start + 1,
            pass1_start + kPaddedSequenceLength + 1,
            "VXM.ALU7",
            label + " P1 padded mask and FP32 scratch write");
        schedule.trace(
            pass1_start + 2,
            pass1_start + kPaddedSequenceLength + 2,
            "VXM.ALU1",
            label + " P1 recurrent max");

        const auto pass2_start = pass1_start + kPaddedSequenceLength + 32;
        for (std::size_t token = 0;
             token < kPaddedSequenceLength;
             ++token) {
            const auto cycle = pass2_start + token;
            for (std::size_t byte = 0;
                 byte < kScaledScoreSlices.size();
                 ++byte) {
                const auto slice = kScaledScoreSlices[byte];
                schedule.mem_at(
                    ftlpu::Hemisphere::East,
                    slice,
                    cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(
                        kScaledScoreAddressBase + token,
                        ftlpu::StreamId::West(byte)));
            }
            schedule.softmax_pass2_at(cycle, token == 0);
            for (std::size_t byte = 0;
                 byte < kExpScoreSlices.size();
                 ++byte) {
                const auto slice = kExpScoreSlices[byte];
                schedule.mem_at(
                    ftlpu::Hemisphere::East,
                    slice,
                    cycle + 2
                        + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        kExpScoreAddressBase + token,
                        ftlpu::StreamId::East(
                            kSoftmaxOutputStream + byte)));
            }
        }
        schedule.trace(
            pass2_start,
            pass2_start + kPaddedSequenceLength,
            "VXM.ALU2",
            label + " P2 subtract max");
        schedule.trace(
            pass2_start + 1,
            pass2_start + kPaddedSequenceLength + 1,
            "VXM.ALU3",
            label + " P2 exp and FP32 scratch write");
        schedule.trace(
            pass2_start + 2,
            pass2_start + kPaddedSequenceLength + 2,
            "VXM.ALU4",
            label + " P2 recurrent sum");

        const auto pass3_start = pass2_start + kPaddedSequenceLength + 32;
        for (std::size_t token = 0;
             token < kPaddedSequenceLength;
             ++token) {
            const auto cycle = pass3_start + token;
            for (std::size_t byte = 0;
                 byte < kExpScoreSlices.size();
                 ++byte) {
                const auto slice = kExpScoreSlices[byte];
                schedule.mem_at(
                    ftlpu::Hemisphere::East,
                    slice,
                    cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(
                        kExpScoreAddressBase + token,
                        ftlpu::StreamId::West(byte)));
            }
            schedule.softmax_pass3_at(cycle);
            const auto column = token % kProbabilityTokensPerReduction;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kActivationSlices[column * 2 + byte];
                schedule.mem_at(
                    ftlpu::Hemisphere::East,
                    slice,
                    cycle + 2
                        + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        kActivationAddressBase
                            + token / kProbabilityTokensPerReduction,
                        ftlpu::StreamId::East(
                            kSoftmaxOutputStream + byte)));
            }
        }
        schedule.trace(
            pass3_start,
            pass3_start + kPaddedSequenceLength,
            "VXM.ALU5",
            label + " P3 divide by sum");
        schedule.trace(
            pass3_start + 1,
            pass3_start + kPaddedSequenceLength + 1,
            "VXM.ALU6",
            label + " P3 BF16 cast and packed P write");

        const auto cycles = schedule.end_cycle() + 16;
        run(cycles, label + " softmax");
        total_cycles_ += cycles;

        auto actual = std::vector<float>(kPaddedSequenceLength);
        constexpr auto kQueryTile = ftlpu::hw::kTileRows - 1;
        constexpr auto kQueryLane = std::size_t {0};
        for (std::size_t token = 0;
             token < kPaddedSequenceLength;
             ++token) {
            const auto column = token % kProbabilityTokensPerReduction;
            const auto address = kActivationAddressBase
                + token / kProbabilityTokensPerReduction;
            const auto low = system_.read_mem_sram_lane_byte(
                ftlpu::Hemisphere::East,
                kActivationSlices[column * 2],
                kQueryTile,
                address,
                kQueryLane);
            const auto high = system_.read_mem_sram_lane_byte(
                ftlpu::Hemisphere::East,
                kActivationSlices[column * 2 + 1],
                kQueryTile,
                address,
                kQueryLane);
            actual[token] = ftlpu::Bf16::from_bits(
                static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8)).to_float();
        }

        auto scores = std::vector<float>(kSequenceLength);
        for (std::size_t token = 0; token < kSequenceLength; ++token) {
            scores[token] = expanded_scores[token * kOutputGroup];
        }
        const auto expected = softmax(scores);
        for (std::size_t token = 0; token < kSequenceLength; ++token) {
            if (std::fabs(actual[token] - expected[token]) > 0.003f) {
                throw std::runtime_error(
                    label + " hardware softmax mismatch at token="
                    + std::to_string(token)
                    + " actual=" + std::to_string(actual[token])
                    + " expected=" + std::to_string(expected[token]));
            }
        }
        for (std::size_t token = kSequenceLength;
             token < kPaddedSequenceLength;
             ++token) {
            if (std::fabs(actual[token]) > 1.0e-6f) {
                throw std::runtime_error(
                    label + " padded probability is not zero at token="
                    + std::to_string(token));
            }
        }
        return actual;
    }
    void mark(std::string operation, std::string detail)
    {
        trace_.push_back(TraceEvent {
            total_cycles_, total_cycles_ + 1,
            std::move(operation), std::move(detail)});
    }

    std::size_t total_cycles() const noexcept { return total_cycles_; }

    void write_trace_csv(const std::filesystem::path& path) const
    {
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        auto output = std::ofstream(path, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot open decode attention trace: " + path.string());
        }
        auto events = trace_;
        std::stable_sort(
            events.begin(), events.end(),
            [](const TraceEvent& lhs, const TraceEvent& rhs) {
                if (lhs.start != rhs.start) return lhs.start < rhs.start;
                if (lhs.end != rhs.end) return lhs.end < rhs.end;
                return lhs.resource < rhs.resource;
            });
        const auto quote = [](const std::string& value) {
            auto result = std::string {"\""};
            for (const auto ch : value) {
                result.push_back(ch);
                if (ch == '"') result.push_back('"');
            }
            result.push_back('"');
            return result;
        };
        output << "start,end,resource,detail\n";
        for (const auto& event : events) {
            output << event.start << ',' << event.end << ','
                   << quote(event.resource) << ','
                   << quote(event.detail) << '\n';
        }
    }

private:
    void initialize_activation(
        const std::vector<float>& input,
        std::size_t rows,
        std::size_t activation_address_base)
    {
        const auto reductions = (rows + kDecodeBlock - 1) / kDecodeBlock;
        for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
            for (std::size_t column = 0;
                 column < ftlpu::hw::kMxmSupercellsPerPlane;
                 ++column) {
                for (std::size_t tile = 0;
                     tile < ftlpu::hw::kTileRows;
                     ++tile) {
                    for (std::size_t lane = 0;
                         lane < ftlpu::hw::kLanesPerTile;
                         ++lane) {
                        const auto row = reduction * kDecodeBlock
                            + (column * ftlpu::hw::kTileRows + tile)
                                * ftlpu::hw::kLanesPerTile
                            + lane;
                        const auto bits = ftlpu::Bf16::from_float(
                            row < rows ? input[row] : 0.0f).bits();
                        for (std::size_t byte = 0; byte < 2; ++byte) {
                            system_.initialize_mem_sram_lane_byte(
                                ftlpu::Hemisphere::East,
                                kActivationSlices[column * 2 + byte],
                                tile,
                                activation_address_base + reduction,
                                lane,
                                static_cast<std::uint8_t>(
                                    (bits >> (byte * 8)) & 0xffu));
                        }
                    }
                }
            }
        }
    }

    void initialize_weights(
        const QuantizedMatrix& matrix,
        std::size_t weight_address_base)
    {
        const auto reductions =
            (matrix.rows + kDecodeBlock - 1) / kDecodeBlock;
        for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
            for (std::size_t group = 0; group < matrix.groups(); ++group) {
                const auto address = weight_address_base
                    + reduction * matrix.groups() + group;
                for (std::size_t stream = 0;
                     stream < ftlpu::hw::kEastStreams;
                     ++stream) {
                    const auto column = stream / kOutputGroup;
                    const auto output = group * kOutputGroup
                        + stream % kOutputGroup;
                    for (std::size_t tile = 0;
                         tile < ftlpu::hw::kTileRows;
                         ++tile) {
                        for (std::size_t lane = 0;
                             lane < ftlpu::hw::kLanesPerTile;
                             ++lane) {
                            const auto row = reduction * kDecodeBlock
                                + (column * ftlpu::hw::kTileRows + tile)
                                    * ftlpu::hw::kLanesPerTile
                                + lane;
                            const auto value = row < matrix.rows
                                ? matrix.values[
                                    row * matrix.padded_columns + output]
                                : std::int8_t {0};
                            system_.initialize_mem_sram_lane_byte(
                                ftlpu::Hemisphere::East,
                                stream,
                                tile,
                                address,
                                lane,
                                static_cast<std::uint8_t>(value));
                        }
                    }
                }
            }
        }
    }

    static void schedule_activation_load(
        Schedule& schedule,
        std::size_t cycle,
        std::size_t reduction,
        std::size_t buffer,
        std::size_t activation_address_base)
    {
        for (std::size_t stream = 0;
             stream < kActivationSlices.size();
             ++stream) {
            const auto slice = kActivationSlices[stream];
            schedule.mem_at(
                ftlpu::Hemisphere::East,
                slice,
                cycle - east_latency(slice),
                ftlpu::MemInstruction::Read(
                    activation_address_base + reduction,
                    ftlpu::StreamId::East(stream)));
        }
        schedule.load_at(0, cycle, buffer);
    }

    static void schedule_weight_waves(
        Schedule& schedule,
        std::size_t compute_start,
        std::size_t reduction,
        const QuantizedMatrix& matrix,
        std::size_t buffer,
        bool final_reduction,
        std::size_t weight_address_base,
        std::optional<std::size_t> resident_output_chunk_base,
        std::size_t resident_output_address_base)
    {
        for (std::size_t group = 0; group < matrix.groups(); ++group) {
            for (std::size_t column = 0;
                 column < ftlpu::hw::kMxmSupercellsPerPlane;
                 ++column) {
                const auto boundary_cycle = compute_start + group
                    + column * ftlpu::hw::kTileRows;
                for (std::size_t lane = 0; lane < kOutputGroup; ++lane) {
                    const auto stream = column * kOutputGroup + lane;
                    schedule.mem_at(
                        ftlpu::Hemisphere::East,
                        stream,
                        boundary_cycle - east_latency(stream),
                        ftlpu::MemInstruction::Read(
                            weight_address_base
                                + reduction * matrix.groups() + group,
                            ftlpu::StreamId::East(stream)));
                }
            }
            schedule.decode_compute_at(
                0,
                compute_start + group,
                buffer,
                matrix.scales[group],
                group / kGroupsPerAccumulatorRow,
                group % kGroupsPerAccumulatorRow,
                final_reduction
                    ? ftlpu::MxmAccumulatorDestination::Stream
                    : ftlpu::MxmAccumulatorDestination::Sram,
                final_reduction);

            if (final_reduction) {
                if (resident_output_chunk_base.has_value()) {
                    const auto chunk = *resident_output_chunk_base + group;
                    const auto column = chunk
                        % ftlpu::hw::kMxmSupercellsPerPlane;
                    const auto reduction = chunk
                        / ftlpu::hw::kMxmSupercellsPerPlane;
                    for (std::size_t byte = 0; byte < 2; ++byte) {
                        const auto slice =
                            kActivationSlices[column * 2 + byte];
                        schedule.mem_at(
                            ftlpu::Hemisphere::East,
                            slice,
                            compute_start + group + kDecodeStages - 1
                                + west_latency(slice),
                            ftlpu::MemInstruction::Write(
                                resident_output_address_base + reduction,
                                ftlpu::StreamId::West(byte)));
                    }
                } else {
                    for (std::size_t byte = 0;
                         byte < kOutputSlices.size();
                         ++byte) {
                        const auto slice = kOutputSlices[byte];
                        schedule.mem_at(
                            ftlpu::Hemisphere::East,
                            slice,
                            compute_start + group + kDecodeStages - 1
                                + west_latency(slice),
                            ftlpu::MemInstruction::Write(
                                kOutputAddressBase + group,
                                ftlpu::StreamId::West(byte)));
                    }
                }
            }
        }        if (final_reduction) {
            schedule.trace(
                compute_start + kDecodeStages - 1
                    + west_latency(kOutputSlices.front()),
                compute_start + matrix.groups() + kDecodeStages - 1
                    + west_latency(kOutputSlices.back()),
                "MEM.E.Write",
                "BF16 final reduction output");
        }
    }

    std::vector<float> read_resident_output(
        std::size_t columns,
        std::size_t chunk_base,
        std::size_t address_base) const
    {
        auto result = std::vector<float>(columns);
        constexpr auto kOutputTile = ftlpu::hw::kTileRows - 1;
        for (std::size_t output = 0; output < columns; ++output) {
            const auto chunk = chunk_base + output / kOutputGroup;
            const auto column = chunk
                % ftlpu::hw::kMxmSupercellsPerPlane;
            const auto reduction = chunk
                / ftlpu::hw::kMxmSupercellsPerPlane;
            const auto lane = output % kOutputGroup;
            const auto low = system_.read_mem_sram_lane_byte(
                ftlpu::Hemisphere::East,
                kActivationSlices[column * 2],
                kOutputTile,
                address_base + reduction,
                lane);
            const auto high = system_.read_mem_sram_lane_byte(
                ftlpu::Hemisphere::East,
                kActivationSlices[column * 2 + 1],
                kOutputTile,
                address_base + reduction,
                lane);
            result[output] = ftlpu::Bf16::from_bits(
                static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8)).to_float();
        }
        return result;
    }
    std::vector<float> read_output(std::size_t columns) const
    {
        auto result = std::vector<float>(columns);
        constexpr auto kOutputTile = ftlpu::hw::kTileRows - 1;
        for (std::size_t output = 0; output < columns; ++output) {
            const auto group = output / kOutputGroup;
            const auto lane = output % kOutputGroup;
            const auto low = system_.read_mem_sram_lane_byte(
                ftlpu::Hemisphere::East,
                kOutputSlices[0],
                kOutputTile,
                kOutputAddressBase + group,
                lane);
            const auto high = system_.read_mem_sram_lane_byte(
                ftlpu::Hemisphere::East,
                kOutputSlices[1],
                kOutputTile,
                kOutputAddressBase + group,
                lane);
            result[output] = ftlpu::Bf16::from_bits(
                static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8)).to_float();
        }
        return result;
    }

    void run(std::size_t cycles, const std::string& label)
    {
        if (logs_enabled_) {
            icu_log_ << "\n=== " << label
                     << " global_cycle " << total_cycles_ << " ===\n";
            mem_log_ << "\n=== " << label
                     << " global_cycle " << total_cycles_ << " ===\n";
        }
        for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
            try {
                system_.tick({
                    .icu = logs_enabled_ ? &icu_log_ : nullptr,
                    .mem = logs_enabled_ ? &mem_log_ : nullptr,
                    .mem_log_tile = logs_enabled_
                        ? std::optional<std::size_t>{0}
                        : std::nullopt,
                });
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    label + " cycle " + std::to_string(cycle)
                    + ": " + error.what());
            }
        }
    }

    ftlpu::TspSliceSystem& system_;
    std::vector<TraceEvent> trace_{};
    std::size_t total_cycles_{0};
    std::ofstream icu_log_{};
    std::ofstream mem_log_{};
    bool logs_enabled_{false};
};
float projection_weight(
    std::size_t tag,
    std::size_t row,
    std::size_t column)
{
    if (tag < 3) {
        const auto raw = static_cast<int>(
            (row * (13 + tag * 4)
                + column * (7 + tag * 2)
                + tag * 11)
            % 43)
            - 21;
        return static_cast<float>(raw)
            * (0.007f
                + static_cast<float>((column + tag * 3) % 9) * 0.001f);
    }
    const auto raw =
        static_cast<int>((row * 19 + column * 11 + 5) % 47) - 23;
    return static_cast<float>(raw)
        * (0.006f + static_cast<float>(column % 7) * 0.001f);
}

void apply_rope(HeadVector& vector, std::size_t position)
{
    for (std::size_t pair = 0; pair < kHeadDim / 2; ++pair) {
        const auto exponent = 2.0f * static_cast<float>(pair)
            / static_cast<float>(kHeadDim);
        const auto angle = static_cast<float>(position)
            / std::pow(kRopeTheta, exponent);
        const auto cosine = bf16(std::cos(angle));
        const auto sine = bf16(std::sin(angle));
        const auto low = vector[pair];
        const auto high = vector[pair + kHeadDim / 2];
        vector[pair] = bf16(low * cosine - high * sine);
        vector[pair + kHeadDim / 2] = bf16(
            high * cosine + low * sine);
    }
}

std::vector<float> softmax(const std::vector<float>& scores)
{
    const auto maximum = *std::max_element(scores.begin(), scores.end());
    auto probabilities = std::vector<float>(scores.size());
    auto denominator = 0.0f;
    for (std::size_t index = 0; index < scores.size(); ++index) {
        probabilities[index] = std::exp(
            (scores[index] - maximum) * kAttentionScale);
        denominator += probabilities[index];
    }
    for (auto& probability : probabilities) {
        probability = bf16(probability / denominator);
    }
    return probabilities;
}

void initialize_prefill_cache(KvCache& keys, KvCache& values)
{
    for (std::size_t head = 0; head < kKvHeads; ++head) {
        for (std::size_t token = 0; token < kPrefillLength; ++token) {
            for (std::size_t dimension = 0;
                 dimension < kHeadDim;
                 ++dimension) {
                const auto key_raw = static_cast<int>(
                    (token * 13 + head * 17 + dimension * 7) % 43) - 21;
                const auto value_raw = static_cast<int>(
                    (token * 5 + head * 19 + dimension * 11) % 47) - 23;
                keys[head][token][dimension] = bf16(
                    static_cast<float>(key_raw) * 0.03125f);
                values[head][token][dimension] = bf16(
                    static_cast<float>(value_raw) * 0.02734375f);
            }
            apply_rope(keys[head][token], token);
        }
    }
}

HeadVector slice_head(
    const std::vector<float>& values,
    std::size_t head)
{
    auto result = HeadVector {};
    std::copy_n(
        values.begin() + static_cast<std::ptrdiff_t>(head * kHeadDim),
        kHeadDim,
        result.begin());
    return result;
}

std::vector<float> head_to_vector(const HeadVector& head)
{
    return {head.begin(), head.end()};
}

} // namespace

ftlpu::test::smollm2_layer::PhaseResult
ftlpu::test::smollm2_layer::run_decode_attention(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& hidden,
    const std::filesystem::path& trace_path,
    const std::filesystem::path& log_dir,
    const std::vector<float>& prefill_keys,
    const std::vector<float>& prefill_values)
{
    if (hidden.size() != kHidden) {
        throw std::invalid_argument("decode attention expects X[1,576]");
    }

    const auto query_weight = quantize_matrix(
        kHidden,
        kHidden,
        [](std::size_t row, std::size_t column) {
            return projection_weight(0, row, column);
        });
    const auto key_weight = quantize_matrix(
        kHidden,
        kKvWidth,
        [](std::size_t row, std::size_t column) {
            return projection_weight(1, row, column);
        });
  const auto value_weight = quantize_matrix(
        kHidden,
        kKvWidth * kOutputGroup,
        [](std::size_t row, std::size_t packed_column) {
            const auto dimension = packed_column / kOutputGroup;
            const auto lane = packed_column % kOutputGroup;
            return lane == 0
                ? projection_weight(2, row, dimension)
                : 0.0f;
        });    const auto output_weight = quantize_matrix(
        kHidden,
        kHidden,
        [](std::size_t row, std::size_t column) {
            return projection_weight(3, row, column);
        });

    system.reset_execution_state();
    auto harness = DecodeAttentionHarness {system};
    if (!log_dir.empty()) {
        harness.enable_logs(log_dir);
    }
    auto keys = KvCache {};
    auto values = KvCache {};
    initialize_prefill_cache(keys, values);
    if (!prefill_keys.empty() || !prefill_values.empty()) {
        const auto cache_elements = kPrefillLength * kKvWidth;
        if (prefill_keys.size() != cache_elements
            || prefill_values.size() != cache_elements) {
            throw std::invalid_argument(
                "decode attention prefill KV cache shape mismatch");
        }
        for (std::size_t token = 0; token < kPrefillLength; ++token) {
            for (std::size_t head = 0; head < kKvHeads; ++head) {
                for (std::size_t dimension = 0;
                     dimension < kHeadDim;
                     ++dimension) {
                    const auto source = token * kKvWidth
                        + head * kHeadDim + dimension;
                    keys[head][token][dimension] = prefill_keys[source];
                    values[head][token][dimension] = prefill_values[source];
                }
            }
        }
    }
    const auto make_cache_matrices = [&](std::size_t token_limit) {
        auto key_matrices = std::array<QuantizedMatrix, kKvHeads> {};
        auto value_matrices = std::array<QuantizedMatrix, kKvHeads> {};
        for (std::size_t head = 0; head < kKvHeads; ++head) {
            key_matrices[head] = quantize_matrix_fixed(
                2 * kDecodeBlock,
                kPaddedSequenceLength * kOutputGroup,
                kKeyCacheScales[head],
                [&](std::size_t row, std::size_t expanded_column) {
                    const auto reduction = row / kDecodeBlock;
                    const auto local_row = row % kDecodeBlock;
                    const auto stage = local_row / ftlpu::hw::kLanesPerTile;
                    const auto dimension_lane = local_row
                        % ftlpu::hw::kLanesPerTile;
                    if (stage % ftlpu::hw::kTileRows
                        != ftlpu::hw::kTileRows - 1) {
                        return 0.0f;
                    }
                    const auto column = stage / ftlpu::hw::kTileRows;
                    const auto dimension = reduction * 32
                        + column * kOutputGroup + dimension_lane;
                    const auto token = expanded_column / kOutputGroup;
                    return token < token_limit
                        ? keys[head][token][dimension]
                        : 0.0f;
                });
            value_matrices[head] = quantize_matrix_fixed(
                kProbabilityRows,
                kHeadDim,
                kValueCacheScales[head],
                [&](std::size_t row, std::size_t dimension) {
                    const auto reduction = row / kDecodeBlock;
                    const auto local_row = row % kDecodeBlock;
                    const auto stage = local_row / ftlpu::hw::kLanesPerTile;
                    const auto lane = local_row % ftlpu::hw::kLanesPerTile;
                    if (stage % ftlpu::hw::kTileRows
                            != ftlpu::hw::kTileRows - 1
                        || lane != 0) {
                        return 0.0f;
                    }
                    const auto column = stage / ftlpu::hw::kTileRows;
                    const auto token = reduction * kProbabilityTokensPerReduction
                        + column;
                    return token < token_limit
                        ? values[head][token][dimension]
                        : 0.0f;
                });
        }
        return std::pair {std::move(key_matrices), std::move(value_matrices)};
    };

    auto [key_matrices, value_matrices] =
        make_cache_matrices(kPrefillLength);
    for (std::size_t head = 0; head < kKvHeads; ++head) {
        harness.load_resident_weights_via_instructions(
            key_matrices[head],
            kKeyCacheAddressBase + head * kKvCacheHeadStride,
            "K cache head=" + std::to_string(head));
        harness.load_resident_weights_via_instructions(
            value_matrices[head],
            kValueCacheAddressBase + head * kKvCacheHeadStride,
            "V cache head=" + std::to_string(head));
    }
    harness.mark(
        "KVCache.StaticInt8",
        prefill_keys.empty()
            ? "offline fixed-scale INT8 synthetic prefill cache, tokens 0..127"
            : "offline fixed-scale INT8 cache from prefill K/V, tokens 0..127");

    const auto projected_key = harness.gemv(
        hidden, key_weight, "K projection");
    const auto resident_keys = harness.rope_heads_from_projection(
        projected_key, kKvHeads, "K");
    harness.quantize_resident_keys_to_cache();
    for (std::size_t head = 0; head < kKvHeads; ++head) {
        for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
            const auto reduction = dimension / 32;
            const auto column = (dimension % 32) / kOutputGroup;
            const auto lane = dimension % kOutputGroup;
            const auto row = reduction * kDecodeBlock
                + (column * ftlpu::hw::kTileRows
                    + ftlpu::hw::kTileRows - 1)
                    * ftlpu::hw::kLanesPerTile
                + lane;
            keys[head][kDecodePosition][dimension] = resident_keys[head][row];
        }
    }

    const auto projected_value = harness.gemv(
        hidden, value_weight, "V projection packed");
    harness.quantize_packed_values_to_cache();
    for (std::size_t head = 0; head < kKvHeads; ++head) {
        for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
            const auto packed_group = head * kHeadDim + dimension;
            values[head][kDecodePosition][dimension] =
                projected_value[packed_group * kOutputGroup];
        }
    }

    auto updated_cache_matrices = make_cache_matrices(kSequenceLength);
    key_matrices = std::move(updated_cache_matrices.first);
    value_matrices = std::move(updated_cache_matrices.second);
    const auto projected_query = harness.gemv(
        hidden, query_weight, "Q projection");
    const auto resident_queries = harness.rope_heads_from_projection(
        projected_query, kQueryHeads, "Q");


    harness.clear_resident_activation(
        kContextActivationAddressBase, kContextReductions);
    auto context = std::vector<float>(kHidden);
    for (std::size_t query_head = 0;
         query_head < kQueryHeads;
         ++query_head) {
        const auto kv_head = query_head / kQueriesPerKv;
        const auto expanded_scores = harness.gemv(
            resident_queries[query_head],
            key_matrices[kv_head],
            "QK head=" + std::to_string(query_head)
                + " kv=" + std::to_string(kv_head),
            kKeyCacheAddressBase + kv_head * kKvCacheHeadStride,
            true,
            kQueryRopeAddressBase + query_head * 2);
        auto scores = std::vector<float>(kSequenceLength);
        for (std::size_t token = 0; token < kSequenceLength; ++token) {
            scores[token] = expanded_scores[token * kOutputGroup];
        }
        const auto probabilities = harness.softmax_from_resident_scores(
            expanded_scores,
            "head=" + std::to_string(query_head));
        auto probability_sum = 0.0f;
        for (const auto probability : probabilities) {
            if (!std::isfinite(probability) || probability < 0.0f) {
                throw std::runtime_error("softmax produced an invalid probability");
            }
            probability_sum += probability;
        }
        if (std::fabs(probability_sum - 1.0f) > 0.02f) {
            throw std::runtime_error(
                "BF16 softmax sum is " + std::to_string(probability_sum));
        }


        auto resident_probability_input = std::vector<float>(kProbabilityRows);
        for (std::size_t token = 0;
             token < kPaddedSequenceLength;
             ++token) {
            const auto reduction = token / kProbabilityTokensPerReduction;
            const auto column = token % kProbabilityTokensPerReduction;
            const auto row = reduction * kDecodeBlock
                + (column * ftlpu::hw::kTileRows
                    + ftlpu::hw::kTileRows - 1)
                    * ftlpu::hw::kLanesPerTile;
            resident_probability_input[row] = probabilities[token];
        }
        const auto head_context = harness.gemv(
            resident_probability_input,
            value_matrices[kv_head],
            "PV head=" + std::to_string(query_head)
                + " kv=" + std::to_string(kv_head),
            kValueCacheAddressBase + kv_head * kKvCacheHeadStride,
            true,
            kActivationAddressBase,
            query_head * (kHeadDim / kOutputGroup),
            kContextActivationAddressBase);
        std::copy(
            head_context.begin(),
            head_context.end(),
            context.begin()
                + static_cast<std::ptrdiff_t>(query_head * kHeadDim));
    }

    auto resident_output_weight = QuantizedMatrix {
        kContextRows,
        output_weight.columns,
        output_weight.padded_columns,
        output_weight.scales,
        std::vector<std::int8_t>(
            kContextRows * output_weight.padded_columns),
        std::vector<float>(
            kContextRows * output_weight.padded_columns)};
    auto resident_context_input = std::vector<float>(kContextRows);
    for (std::size_t chunk = 0; chunk < kContextChunks; ++chunk) {
        const auto reduction = chunk
            / ftlpu::hw::kMxmSupercellsPerPlane;
        const auto column = chunk
            % ftlpu::hw::kMxmSupercellsPerPlane;
        for (std::size_t lane = 0; lane < kOutputGroup; ++lane) {
            const auto source_row = chunk * kOutputGroup + lane;
            const auto resident_row = reduction * kDecodeBlock
                + (column * ftlpu::hw::kTileRows
                    + ftlpu::hw::kTileRows - 1)
                    * ftlpu::hw::kLanesPerTile
                + lane;
            resident_context_input[resident_row] = context[source_row];
            for (std::size_t output_column = 0;
                 output_column < output_weight.padded_columns;
                 ++output_column) {
                const auto source_index = source_row
                    * output_weight.padded_columns + output_column;
                const auto resident_index = resident_row
                    * output_weight.padded_columns + output_column;
                resident_output_weight.values[resident_index] =
                    output_weight.values[source_index];
                resident_output_weight.dequantized[resident_index] =
                    output_weight.dequantized[source_index];
            }
        }
    }
    const auto output = harness.gemv(
        resident_context_input,
        resident_output_weight,
        "O projection",
        std::nullopt,
        true,
        kContextActivationAddressBase);
    auto checksum = 0.0;
    for (const auto value : output) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("attention output is not finite");
        }
        checksum += static_cast<double>(value);
    }

    if (!trace_path.empty()) {
        harness.write_trace_csv(trace_path);
    }

    return {output, {}, {}, harness.total_cycles()};
}

#ifndef FTLPU_SMOLLM2_LAYER_PHASE_ONLY
int main() try
{
    auto hidden = std::vector<float>(kHidden);
    for (std::size_t index = 0; index < kHidden; ++index) {
        const auto raw = static_cast<int>((index * 9 + 5) % 37) - 18;
        hidden[index] = bf16(static_cast<float>(raw) * 0.046875f);
    }
    auto system = ftlpu::TspSliceSystem {};
    const auto log_dir = std::filesystem::path("logs")
        / "smollm2_decode_attention_test";
    const auto* trace_env = std::getenv("FTLPU_SCHEDULE_TRACE");
    const auto result = ftlpu::test::smollm2_layer::run_decode_attention(
        system,
        hidden,
        trace_env == nullptr ? std::filesystem::path {} : trace_env,
        log_dir);
    auto checksum = 0.0;
    for (const auto value : result.output) checksum += value;
    std::cout
        << "SmolLM2 decode attention passed: prefill=128, decode=1, "
        << "Q heads=9, KV heads=3, head_dim=64, streamed_gemvs=22, "
        << "cycles=" << result.cycles
        << ", checksum=" << checksum
        << ", logs=" << log_dir.string() << '\n';
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "SmolLM2 decode attention failed: "
              << error.what() << '\n';
    return 1;
}
#endif
