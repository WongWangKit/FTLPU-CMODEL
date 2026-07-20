#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "vxm_alu_program.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSeqLen = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kKvHeads = 3;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;
constexpr std::size_t kTile = ftlpu::hw::kMxmRows;
constexpr std::size_t kWeightToIwLatency = 14;
constexpr std::size_t kActivationLatency = 5;
constexpr std::size_t kKeyAccumulatorLatency = 6;
constexpr std::size_t kValueAccumulatorLatency = 5;
constexpr std::size_t kComputeBlockCycles = 48;
constexpr std::array<std::size_t, 8> kWeightSlices {0, 4, 8, 12, 16, 20, 24, 28};
constexpr std::array<std::size_t, 4> kActivationSlices {32, 33, 34, 35};

static_assert(kSeqLen % kTile == 0);
static_assert(kHidden % kTile == 0);
static_assert(kHeadDim == 2 * kTile);
static_assert(kKvWidth == 6 * kTile);

enum class Projection : std::size_t { Key, Value };

std::size_t x_index(std::size_t token, std::size_t hidden)
{
    return token * kHidden + hidden;
}

std::size_t weight_index(std::size_t hidden, std::size_t column)
{
    return hidden * kKvWidth + column;
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

class OfflineSchedule {
public:
    explicit OfflineSchedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(std::size_t queue, std::size_t cycle, ftlpu::MemInstruction instruction)
    {
        require_available(mem_[queue], cycle, "MEM q" + std::to_string(queue));
        pad(mem_[queue], cycle, [&](std::size_t count) { icu_.enqueue_mem_nop(queue, count); });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
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
    }

    void dequant_at(std::size_t cycle, const ftlpu::test::DequantSpec& instruction)
    {
        ftlpu::test::enqueue_dequant(icu_, vxm_, cycle, instruction);
        end_cycle_ = std::max(end_cycle_, cycle + 2);
    }

    void mxm_load_at(std::size_t mxm, std::size_t cycle)
    {
        require_available(mxm_load_[mxm], cycle, "MXM load " + std::to_string(mxm));
        pad(mxm_load_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_load_nop(mxm, count);
        });
        icu_.enqueue_mxm(mxm, ftlpu::MxmControlInstruction::IW(0));
        advance(mxm_load_[mxm], cycle + 1);
    }

    void mxm_compute_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t activation_stream,
        std::size_t output_stream)
    {
        require_available(mxm_compute_[mxm], cycle, "MXM compute " + std::to_string(mxm));
        pad(mxm_compute_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_compute_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::Compute(0, activation_stream, output_stream));
        icu_.enqueue_mxm_compute_repeat(mxm, kTile - 1, 1);
        advance(mxm_compute_[mxm], cycle + kTile);
    }

    std::size_t end_cycle() const { return end_cycle_; }

private:
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

std::size_t activation_address(std::size_t k_block, std::size_t token)
{
    return k_block * kSeqLen + token;
}

std::size_t accumulator_address(std::size_t token, std::size_t column)
{
    return token * (kKvWidth / kTile) + column / kTile;
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

float read_projection(
    const ftlpu::TspSliceSystem& system,
    Projection projection,
    std::size_t token,
    std::size_t column)
{
    const auto hemisphere = static_cast<ftlpu::Hemisphere>((column / kTile) % 2);
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto group_base = projection == Projection::Key
        ? ftlpu::hw::kWestAccumulatorMemSliceBase
        : ftlpu::hw::kEastAccumulatorMemSliceBase;
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(system.read_mem_sram_lane_byte(
            hemisphere,
            group_base + byte,
            tile,
            accumulator_address(token, column),
            lane)) << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

} // namespace

int main() try
{
    std::vector<float> input(kSeqLen * kHidden);
    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            input[x_index(token, hidden)] = ftlpu::Fp16::from_float(
                input_value(token, hidden)).to_float();
        }
    }

    std::array<std::vector<float>, 2> scales {
        std::vector<float>(kKvWidth), std::vector<float>(kKvWidth)};
    std::array<std::vector<std::int8_t>, 2> weights {
        std::vector<std::int8_t>(kHidden * kKvWidth),
        std::vector<std::int8_t>(kHidden * kKvWidth)};
    std::array<std::vector<float>, 2> dequantized {
        std::vector<float>(kHidden * kKvWidth),
        std::vector<float>(kHidden * kKvWidth)};

    for (std::size_t projection = 0; projection < 2; ++projection) {
        for (std::size_t column = 0; column < kKvWidth; ++column) {
            float max_abs = 0.0f;
            for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                max_abs = std::max(max_abs, std::fabs(weight_value(
                    static_cast<Projection>(projection), hidden, column)));
            }
            scales[projection][column] = max_abs / 127.0f;
            for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                const auto quantized = std::clamp(static_cast<int>(std::lround(
                    weight_value(static_cast<Projection>(projection), hidden, column)
                    / scales[projection][column])), -127, 127);
                weights[projection][weight_index(hidden, column)] =
                    static_cast<std::int8_t>(quantized);
                dequantized[projection][weight_index(hidden, column)] =
                    ftlpu::Fp16::from_float(
                        static_cast<float>(quantized) * scales[projection][column]).to_float();
            }
        }
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system, input);
    auto schedule = OfflineSchedule(system.icu());

    std::size_t phase_start = 0;
    std::size_t weight_address = 0;
    for (std::size_t pair_base = 0; pair_base < kKvWidth; pair_base += 2 * kTile) {
        for (std::size_t k_block = 0; k_block < kHidden / kTile; ++k_block) {
            const auto dequant_start = phase_start + 10;
            for (std::size_t hemisphere_index = 0;
                 hemisphere_index < ftlpu::hw::kHemispheres;
                 ++hemisphere_index) {
                const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                const auto column_base = pair_base + hemisphere_index * kTile;
                for (std::size_t pulse = 0; pulse < 8; ++pulse) {
                    const auto projection = pulse / 4;
                    const auto block = 3 - pulse % 4;
                    const auto cycle = dequant_start + hemisphere_index * 8 + pulse;
                    auto instruction = ftlpu::test::DequantSpec {};
                    instruction.input_stream_base = ftlpu::hw::kEastStreams;
                    instruction.output_stream_base = projection * 16;
                    instruction.input_hemisphere = hemisphere;
                    instruction.output_hemisphere = hemisphere;
                    for (std::size_t stream = 0; stream < ftlpu::hw::kLanesPerTile; ++stream) {
                        const auto column = column_base
                            + block * ftlpu::hw::kLanesPerTile + stream;
                        instruction.scales[stream] = scales[projection][column];
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
                                    static_cast<std::uint8_t>(weights[projection][
                                        weight_index(hidden, column)]));
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
                        ftlpu::InstructionControlUnit::mxm_queue(hemisphere, projection),
                        cycle + kWeightToIwLatency);
                    ++weight_address;
                }
            }

            const auto first_compute = dequant_start + 32;
            for (std::size_t token_block = 0; token_block < kSeqLen / kTile; ++token_block) {
                const auto compute_cycle = first_compute + token_block * kComputeBlockCycles;
                const auto input_address = activation_address(k_block, token_block * kTile);
                for (std::size_t hemisphere_index = 0;
                     hemisphere_index < ftlpu::hw::kHemispheres;
                     ++hemisphere_index) {
                    const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                    const auto column_base = pair_base + hemisphere_index * kTile;
                    const auto output_address = accumulator_address(
                        token_block * kTile, column_base);
                    for (std::size_t byte = 0; byte < kActivationSlices.size(); ++byte) {
                        schedule.mem_repeat_at(
                            mem_queue(hemisphere, kActivationSlices[byte]),
                            compute_cycle - kActivationLatency,
                            ftlpu::MemInstruction::Read(
                                input_address, ftlpu::StreamId::East(byte)),
                            kTile,
                            1);
                    }
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, ftlpu::hw::kWestAccumulatorMemSliceBase),
                        compute_cycle + kKeyAccumulatorLatency,
                        ftlpu::MemInstruction::Accumulate(
                            output_address, ftlpu::StreamId::West(0)),
                        kTile,
                        kKvWidth / kTile);
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, ftlpu::hw::kEastAccumulatorMemSliceBase),
                        compute_cycle + kValueAccumulatorLatency,
                        ftlpu::MemInstruction::Accumulate(
                            output_address, ftlpu::StreamId::West(4)),
                        kTile,
                        kKvWidth / kTile);
                    schedule.mxm_compute_at(
                        ftlpu::InstructionControlUnit::mxm_queue(hemisphere, 0),
                        compute_cycle,
                        0,
                        0);
                    schedule.mxm_compute_at(
                        ftlpu::InstructionControlUnit::mxm_queue(hemisphere, 1),
                        compute_cycle,
                        2,
                        4);
                }
            }
            phase_start = first_compute + (kSeqLen / kTile) * kComputeBlockCycles;
        }
    }

    for (std::size_t cycle = 0; cycle < schedule.end_cycle() + 16; ++cycle) {
        system.tick({});
    }

    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t column = 0; column < kKvWidth; ++column) {
            for (std::size_t projection = 0; projection < 2; ++projection) {
                float expected = 0.0f;
                for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                    expected += input[x_index(token, hidden)]
                        * dequantized[projection][weight_index(hidden, column)];
                }
                const auto actual = read_projection(
                    system, static_cast<Projection>(projection), token, column);
                if (std::fabs(actual - expected) > 1.0e-5f) {
                    std::cerr << (projection == 0 ? 'K' : 'V')
                              << " projection mismatch at token=" << token
                              << " column=" << column
                              << " actual=" << actual
                              << " expected=" << expected << '\n';
                    return 1;
                }
            }
        }
    }

    std::cout << "SmolLM2 K/V projection passed: X[128,576] -> K/V[128,192], "
              << "W8A16 across four MXMs\n";
    return 0;
}
catch (const std::exception& ex)
{
    std::cerr << "SmolLM2 K/V projection test failed: " << ex.what() << '\n';
    return 1;
}
