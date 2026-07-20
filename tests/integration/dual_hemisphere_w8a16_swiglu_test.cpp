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

constexpr std::size_t kRows = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
constexpr std::size_t kTile = ftlpu::hw::kMxmRows;
constexpr std::size_t kWeightToIwLatency = 14;
constexpr std::size_t kActivationLatency = 5;
constexpr std::size_t kGateAccumulatorLatency = 6;
constexpr std::size_t kUpAccumulatorLatency = 5;
constexpr std::size_t kSwishWriteLatency = 13;
constexpr std::size_t kComputeBlockCycles = 48;
constexpr std::size_t kOutputLowSlice = 29;
constexpr std::size_t kOutputHighSlice = 30;
constexpr std::array<std::size_t, 8> kWeightSlices {0, 4, 8, 12, 16, 20, 24, 28};
constexpr std::array<std::size_t, 4> kActivationSlices {32, 33, 34, 35};

static_assert(kRows % kTile == 0);
static_assert(kHidden % kTile == 0);
static_assert(kIntermediate % (2 * kTile) == 0);

enum class Projection : std::size_t { Gate, Up };

std::size_t a_index(std::size_t row, std::size_t k) { return row * kHidden + k; }
std::size_t w_index(std::size_t k, std::size_t n) { return k * kIntermediate + n; }

float activation_value(std::size_t row, std::size_t k)
{
    return static_cast<float>(static_cast<int>((row * 7 + k * 5) % 23) - 11) * 0.0625f;
}

float weight_value(Projection projection, std::size_t k, std::size_t n)
{
    const auto p = static_cast<std::size_t>(projection);
    const auto raw = static_cast<int>((k * (11 + p * 6) + n * (5 + p * 2) + p * 13) % 41) - 20;
    return static_cast<float>(raw) * (0.009f + static_cast<float>((n + p) % 7) * 0.001f);
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

    void swish_at(std::size_t cycle, const ftlpu::test::SwishSpec& instruction)
    {
        ftlpu::test::enqueue_swish(icu_, vxm_, cycle, instruction);
        end_cycle_ = std::max(end_cycle_, cycle + 6);
    }

    void mxm_load_at(std::size_t mxm, std::size_t cycle)
    {
        require_available(mxm_load_[mxm], cycle, "MXM load " + std::to_string(mxm));
        pad(mxm_load_[mxm], cycle, [&](std::size_t count) { icu_.enqueue_mxm_load_nop(mxm, count); });
        icu_.enqueue_mxm(mxm, ftlpu::MxmControlInstruction::IW(0));
        advance(mxm_load_[mxm], cycle + 1);
    }

    void mxm_compute_at(std::size_t mxm, std::size_t cycle, std::size_t activation, std::size_t output)
    {
        require_available(mxm_compute_[mxm], cycle, "MXM compute " + std::to_string(mxm));
        pad(mxm_compute_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_compute_nop(mxm, count);
        });
        icu_.enqueue_mxm(mxm, ftlpu::MxmControlInstruction::Compute(0, activation, output));
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

void initialize_activations(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& activations)
{
    for (std::size_t hemisphere_index = 0; hemisphere_index < ftlpu::hw::kHemispheres;
         ++hemisphere_index) {
        const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
        for (std::size_t k_block = 0; k_block < kHidden / kTile; ++k_block) {
            for (std::size_t row = 0; row < kRows; ++row) {
                const auto address = k_block * kRows + row;
                for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                        const auto k = k_block * kTile
                            + tile * ftlpu::hw::kLanesPerTile + lane;
                        const auto bits = ftlpu::Fp16::from_float(
                            activations[a_index(row, k)]).bits();
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

float read_fp32(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    std::size_t group_base,
    std::size_t row,
    std::size_t column)
{
    const auto local_column = column % kTile;
    const auto n_base = column / kTile * kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(system.read_mem_sram_lane_byte(
            hemisphere,
            group_base + byte,
            tile,
            row * (kIntermediate / kTile) + n_base / kTile,
            lane)) << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

float read_output(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    std::size_t row,
    std::size_t column)
{
    const auto local_column = column % kTile;
    const auto n_base = column / kTile * kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, kOutputLowSlice, tile, (n_base / kTile) * kRows + row, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, kOutputHighSlice, tile, (n_base / kTile) * kRows + row, lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low) | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

} // namespace

int main() try
{
    std::vector<float> activations(kRows * kHidden);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t k = 0; k < kHidden; ++k) {
            activations[a_index(row, k)] = ftlpu::Fp16::from_float(activation_value(row, k)).to_float();
        }
    }

    std::array<std::vector<float>, 2> scales {
        std::vector<float>(kIntermediate), std::vector<float>(kIntermediate)};
    std::array<std::vector<std::int8_t>, 2> weights {
        std::vector<std::int8_t>(kHidden * kIntermediate),
        std::vector<std::int8_t>(kHidden * kIntermediate)};
    std::array<std::vector<float>, 2> dequantized {
        std::vector<float>(kHidden * kIntermediate),
        std::vector<float>(kHidden * kIntermediate)};
    for (std::size_t projection = 0; projection < 2; ++projection) {
        for (std::size_t n = 0; n < kIntermediate; ++n) {
            float max_abs = 0.0f;
            for (std::size_t k = 0; k < kHidden; ++k) {
                max_abs = std::max(max_abs, std::fabs(weight_value(
                    static_cast<Projection>(projection), k, n)));
            }
            scales[projection][n] = max_abs / 127.0f;
            for (std::size_t k = 0; k < kHidden; ++k) {
                const auto quantized = std::clamp(static_cast<int>(std::lround(
                    weight_value(static_cast<Projection>(projection), k, n)
                    / scales[projection][n])), -127, 127);
                weights[projection][w_index(k, n)] = static_cast<std::int8_t>(quantized);
                dequantized[projection][w_index(k, n)] = ftlpu::Fp16::from_float(
                    static_cast<float>(quantized) * scales[projection][n]).to_float();
            }
        }
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize_activations(system, activations);
    auto schedule = OfflineSchedule(system.icu());

    std::size_t phase_start = 0;
    std::size_t weight_address = 0;
    for (std::size_t pair_base = 0; pair_base < kIntermediate; pair_base += 2 * kTile) {
        for (std::size_t k_block = 0; k_block < kHidden / kTile; ++k_block) {
            const auto dequant_start = phase_start + 10;
            for (std::size_t hemisphere_index = 0;
                 hemisphere_index < ftlpu::hw::kHemispheres;
                 ++hemisphere_index) {
                const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                const auto n_base = pair_base + hemisphere_index * kTile;
                for (std::size_t pulse = 0; pulse < 8; ++pulse) {
                    const auto local_mxm = pulse / 4;
                    const auto global_mxm = ftlpu::InstructionControlUnit::mxm_queue(
                        hemisphere, local_mxm);
                    const auto block = 3 - pulse % 4;
                    const auto cycle = dequant_start + hemisphere_index * 8 + pulse;
                    auto instruction = ftlpu::test::DequantSpec {};
                    instruction.input_stream_base = ftlpu::hw::kEastStreams;
                    instruction.output_stream_base = local_mxm * 16;
                    instruction.input_hemisphere = hemisphere;
                    instruction.output_hemisphere = hemisphere;
                    for (std::size_t stream = 0; stream < ftlpu::hw::kLanesPerTile; ++stream) {
                        const auto n = n_base + block * ftlpu::hw::kLanesPerTile + stream;
                        instruction.scales[stream] = scales[local_mxm][n];
                        const auto slice = kWeightSlices[stream];
                        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                                const auto k = k_block * kTile
                                    + tile * ftlpu::hw::kLanesPerTile + lane;
                                system.initialize_mem_sram_lane_byte(
                                    hemisphere,
                                    slice,
                                    tile,
                                    weight_address,
                                    lane,
                                    static_cast<std::uint8_t>(
                                        weights[local_mxm][w_index(k, n)]));
                            }
                        }
                        schedule.mem_at(
                            mem_queue(hemisphere, slice),
                            cycle - west_read_latency(slice),
                            ftlpu::MemInstruction::Read(
                                weight_address, ftlpu::StreamId::West(stream)));
                    }
                    schedule.dequant_at(cycle, instruction);
                    schedule.mxm_load_at(global_mxm, cycle + kWeightToIwLatency);
                    ++weight_address;
                }
            }

            const auto first_compute = dequant_start + 32;
            for (std::size_t row_block = 0; row_block < kRows / kTile; ++row_block) {
                const auto compute_cycle = first_compute + row_block * kComputeBlockCycles;
                const auto activation_address = k_block * kRows + row_block * kTile;
                for (std::size_t hemisphere_index = 0;
                     hemisphere_index < ftlpu::hw::kHemispheres;
                     ++hemisphere_index) {
                    const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                    const auto n_base = pair_base + hemisphere_index * kTile;
                    const auto accumulator_address = row_block * kTile
                        * (kIntermediate / kTile) + n_base / kTile;
                    for (std::size_t byte = 0; byte < kActivationSlices.size(); ++byte) {
                        schedule.mem_repeat_at(
                            mem_queue(hemisphere, kActivationSlices[byte]),
                            compute_cycle - kActivationLatency,
                            ftlpu::MemInstruction::Read(
                                activation_address, ftlpu::StreamId::East(byte)),
                            kTile,
                            1);
                    }
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, ftlpu::hw::kWestAccumulatorMemSliceBase),
                        compute_cycle + kGateAccumulatorLatency,
                        ftlpu::MemInstruction::Accumulate(
                            accumulator_address, ftlpu::StreamId::West(0)),
                        kTile,
                        kIntermediate / kTile);
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, ftlpu::hw::kEastAccumulatorMemSliceBase),
                        compute_cycle + kUpAccumulatorLatency,
                        ftlpu::MemInstruction::Accumulate(
                            accumulator_address, ftlpu::StreamId::West(4)),
                        kTile,
                        kIntermediate / kTile);
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
            phase_start = first_compute + (kRows / kTile) * kComputeBlockCycles;
        }
    }

    const auto swish_start = phase_start + 16;
    for (std::size_t pair_base = 0; pair_base < kIntermediate; pair_base += 2 * kTile) {
        for (std::size_t row = 0; row < kRows; ++row) {
            for (std::size_t hemisphere_index = 0;
                 hemisphere_index < ftlpu::hw::kHemispheres;
                 ++hemisphere_index) {
                const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                const auto n_base = pair_base + hemisphere_index * kTile;
                const auto block = pair_base / (2 * kTile);
                const auto cycle = swish_start
                    + block * kRows * ftlpu::hw::kHemispheres
                    + row * ftlpu::hw::kHemispheres + hemisphere_index;
                const auto accumulator_address = row * (kIntermediate / kTile) + n_base / kTile;
                const auto swiglu_address = (n_base / kTile) * kRows + row;
                for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                    const auto gate_slice = ftlpu::hw::kWestAccumulatorMemSliceBase + byte;
                    const auto up_slice = ftlpu::hw::kEastAccumulatorMemSliceBase + byte;
                    schedule.mem_at(
                        mem_queue(hemisphere, gate_slice),
                        cycle - west_read_latency(gate_slice),
                        ftlpu::MemInstruction::Read(
                            accumulator_address, ftlpu::StreamId::West(byte)));
                    schedule.mem_at(
                        mem_queue(hemisphere, up_slice),
                        cycle - west_read_latency(up_slice),
                        ftlpu::MemInstruction::Read(
                            accumulator_address, ftlpu::StreamId::West(4 + byte)));
                }
                schedule.swish_at(cycle, ftlpu::test::SwishSpec {
                    32, 36, 0, hemisphere, hemisphere});
                schedule.mem_at(
                    mem_queue(hemisphere, kOutputLowSlice),
                    cycle + kSwishWriteLatency,
                    ftlpu::MemInstruction::Write(swiglu_address, ftlpu::StreamId::East(0)));
                schedule.mem_at(
                    mem_queue(hemisphere, kOutputHighSlice),
                    cycle + kSwishWriteLatency,
                    ftlpu::MemInstruction::Write(swiglu_address, ftlpu::StreamId::East(1)));
            }
        }
    }

    for (std::size_t cycle = 0; cycle < schedule.end_cycle() + 16; ++cycle) {
        system.tick({});
    }

    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t n = 0; n < kIntermediate; ++n) {
            const auto hemisphere = static_cast<ftlpu::Hemisphere>((n / kTile) % 2);
            std::array<float, 2> projected {};
            for (std::size_t projection = 0; projection < 2; ++projection) {
                for (std::size_t k = 0; k < kHidden; ++k) {
                    projected[projection] += activations[a_index(row, k)]
                        * dequantized[projection][w_index(k, n)];
                }
            }
            const auto gate = read_fp32(
                system, hemisphere, ftlpu::hw::kWestAccumulatorMemSliceBase, row, n);
            const auto up = read_fp32(
                system, hemisphere, ftlpu::hw::kEastAccumulatorMemSliceBase, row, n);
            if (std::fabs(gate - projected[0]) > 1.0e-5f
                || std::fabs(up - projected[1]) > 1.0e-5f) {
                std::cerr << "dual-hemisphere projection mismatch at (" << row << ',' << n << ")\n";
                return 1;
            }
            const auto expected = ftlpu::Fp16::from_float(
                (gate * up) * (1.0f / (1.0f + std::exp(-gate)))).to_float();
            const auto actual = read_output(system, hemisphere, row, n);
            if (actual != expected && !(std::isnan(actual) && std::isnan(expected))) {
                std::cerr << "dual-hemisphere SwiGLU mismatch at (" << row << ',' << n << ")\n";
                return 1;
            }
        }
    }

    std::cout << "offline dual-hemisphere W8A16 SwiGLU passed: "
              << "X[128,576], gate/up[576,1536], alternating 32-column hemispheres\n";
    return 0;
}
catch (const std::exception& ex)
{
    std::cerr << "dual-hemisphere SwiGLU test failed: " << ex.what() << '\n';
    return 1;
}
