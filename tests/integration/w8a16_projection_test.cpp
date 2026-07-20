#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "vxm_alu_program.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

constexpr std::size_t kSeqLen = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
constexpr std::size_t kTile = ftlpu::hw::kMxmRows;
constexpr std::size_t kActivationLatency = 5;
constexpr std::size_t kWeightToIwLatency = 14;
constexpr std::size_t kWestAccumulatorLatency = 6;
constexpr std::size_t kEastAccumulatorLatency = 5;
constexpr std::size_t kComputeBlockCycles = 48;
constexpr std::array<std::size_t, 8> kWeightSlices {0, 4, 8, 12, 16, 20, 24, 28};
constexpr std::array<std::size_t, 4> kActivationSlices {32, 33, 34, 35};

static_assert(kTile == 32);
static_assert(kSeqLen % kTile == 0);
static_assert(kHidden % kTile == 0);
static_assert(kIntermediate % (2 * kTile) == 0);

std::size_t a_index(std::size_t m, std::size_t k) { return m * kHidden + k; }
std::size_t w_index(std::size_t k, std::size_t n) { return k * kIntermediate + n; }

float activation_value(std::size_t m, std::size_t k)
{
    return static_cast<float>(static_cast<int>((m * 7 + k * 5) % 23) - 11) * 0.0625f;
}

float weight_value(std::size_t k, std::size_t n)
{
    const auto raw = static_cast<int>((k * 13 + n * 3) % 37) - 18;
    return static_cast<float>(raw) * (0.008f + static_cast<float>(n % 11) * 0.0015f);
}

class OfflineSchedule {
public:
    explicit OfflineSchedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(std::size_t queue, std::size_t cycle, ftlpu::MemInstruction instruction)
    {
        pad(mem_[queue], cycle, [&](std::size_t n) { icu_.enqueue_mem_nop(queue, n); });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
    }

    void mem_repeat_at(
        std::size_t queue,
        std::size_t cycle,
        ftlpu::MemInstruction first,
        std::size_t count,
        std::int64_t stride)
    {
        mem_at(queue, cycle, first);
        if (count > 1) {
            icu_.enqueue_mem_repeat(queue, count - 1, 1, stride);
        }
        advance(mem_[queue], cycle + count);
    }

    void vxm_dequant_at(std::size_t cycle, const ftlpu::test::DequantSpec& instruction)
    {
        ftlpu::test::enqueue_dequant(icu_, vxm_, cycle, instruction);
        end_cycle_ = std::max(end_cycle_, cycle + 2);
    }

    void mxm_load_at(std::size_t mxm, std::size_t cycle, ftlpu::MxmControlInstruction instruction)
    {
        pad(mxm_load_[mxm], cycle, [&](std::size_t n) { icu_.enqueue_mxm_load_nop(mxm, n); });
        icu_.enqueue_mxm(mxm, instruction);
        advance(mxm_load_[mxm], cycle + 1);
    }

    void mxm_compute_repeat_at(std::size_t mxm, std::size_t cycle, ftlpu::MxmControlInstruction instruction)
    {
        pad(mxm_compute_[mxm], cycle, [&](std::size_t n) { icu_.enqueue_mxm_compute_nop(mxm, n); });
        icu_.enqueue_mxm(mxm, instruction);
        icu_.enqueue_mxm_compute_repeat(mxm, kTile - 1, 1);
        advance(mxm_compute_[mxm], cycle + kTile);
    }

    std::size_t end_cycle() const { return end_cycle_; }

private:
    template <typename Pad>
    static void pad(std::size_t cursor, std::size_t cycle, Pad emit)
    {
        if (cycle < cursor) {
            throw std::logic_error("offline ICU queue schedule overlaps itself");
        }
        emit(cycle - cursor);
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_cycle_ = std::max(end_cycle_, next);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_{};
    std::array<std::size_t, 2> mxm_load_{};
    std::array<std::size_t, 2> mxm_compute_{};
    std::size_t end_cycle_{0};
};

std::size_t activation_address(std::size_t k_block, std::size_t m_block, std::size_t row)
{
    return (k_block * (kSeqLen / kTile) + m_block) * kTile + row;
}

std::size_t weight_read_latency(std::size_t mem_slice)
{
    return mem_slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t result_address(std::size_t row, std::size_t column)
{
    return row * (kIntermediate / kTile) + column / kTile;
}

float read_result(const ftlpu::TspSliceSystem& system, std::size_t row, std::size_t column)
{
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto group_base = column % (2 * kTile) < kTile
        ? ftlpu::hw::kWestAccumulatorMemSliceBase
        : ftlpu::hw::kEastAccumulatorMemSliceBase;
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(system.read_mem_sram_lane_byte(
            group_base + byte,
            tile,
            result_address(row, column),
            lane)) << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

void initialize_activations(ftlpu::TspSliceSystem& system, const std::vector<float>& activations)
{
    for (std::size_t kb = 0; kb < kHidden / kTile; ++kb) {
        for (std::size_t mb = 0; mb < kSeqLen / kTile; ++mb) {
            for (std::size_t row = 0; row < kTile; ++row) {
                const auto address = activation_address(kb, mb, row);
                for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                        const auto k = kb * kTile + tile * ftlpu::hw::kLanesPerTile + lane;
                        const auto bits = ftlpu::Fp16::from_float(
                            activations[a_index(mb * kTile + row, k)]).bits();
                        system.initialize_mem_sram_lane_byte(kActivationSlices[0], tile, address, lane, bits & 0xffu);
                        system.initialize_mem_sram_lane_byte(kActivationSlices[1], tile, address, lane, bits >> 8);
                        system.initialize_mem_sram_lane_byte(kActivationSlices[2], tile, address, lane, bits & 0xffu);
                        system.initialize_mem_sram_lane_byte(kActivationSlices[3], tile, address, lane, bits >> 8);
                    }
                }
            }
        }
    }
}

} // namespace

int run_test()
{
    std::vector<float> activations(kSeqLen * kHidden);
    for (std::size_t m = 0; m < kSeqLen; ++m) {
        for (std::size_t k = 0; k < kHidden; ++k) {
            activations[a_index(m, k)] = ftlpu::Fp16::from_float(activation_value(m, k)).to_float();
        }
    }

    std::vector<float> scales(kIntermediate);
    std::vector<std::int8_t> weights(kHidden * kIntermediate);
    std::vector<float> dequantized(kHidden * kIntermediate);
    for (std::size_t n = 0; n < kIntermediate; ++n) {
        float max_abs = 0.0f;
        for (std::size_t k = 0; k < kHidden; ++k) max_abs = std::max(max_abs, std::fabs(weight_value(k, n)));
        scales[n] = max_abs / 127.0f;
        for (std::size_t k = 0; k < kHidden; ++k) {
            const auto q = std::clamp(static_cast<int>(std::lround(weight_value(k, n) / scales[n])), -127, 127);
            weights[w_index(k, n)] = static_cast<std::int8_t>(q);
            dequantized[w_index(k, n)] = ftlpu::Fp16::from_float(static_cast<float>(q) * scales[n]).to_float();
        }
    }

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    initialize_activations(*system, activations);
    OfflineSchedule schedule(system->icu());

    std::size_t phase_start = 0;
    std::size_t weight_address = 0;
    for (std::size_t n_base = 0; n_base < kIntermediate; n_base += 2 * kTile) {
        for (std::size_t kb = 0; kb < kHidden / kTile; ++kb) {
            const auto k_base = kb * kTile;
            const auto dequant_start = phase_start + 10;
            for (std::size_t pulse = 0; pulse < 8; ++pulse) {
                const auto mxm = pulse / 4;
                const auto block = 3 - pulse % 4;
                const auto dequant_cycle = dequant_start + pulse;
                auto dequant = ftlpu::test::DequantSpec {};
                dequant.input_stream_base = ftlpu::hw::kEastStreams;
                dequant.output_stream_base = mxm * 16;
                for (std::size_t stream = 0; stream < 8; ++stream) {
                    const auto n = n_base + mxm * kTile + block * 8 + stream;
                    dequant.scales[stream] = scales[n];
                    const auto mem_slice = kWeightSlices[stream];
                    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                            const auto k = k_base + tile * ftlpu::hw::kLanesPerTile + lane;
                            system->initialize_mem_sram_lane_byte(
                                mem_slice, tile, weight_address, lane,
                                static_cast<std::uint8_t>(weights[w_index(k, n)]));
                        }
                    }
                    schedule.mem_at(
                        mem_slice,
                        dequant_cycle - weight_read_latency(mem_slice),
                        ftlpu::MemInstruction::Read(weight_address, ftlpu::StreamId::West(stream)));
                }
                schedule.vxm_dequant_at(dequant_cycle, dequant);
                schedule.mxm_load_at(
                    mxm,
                    dequant_cycle + kWeightToIwLatency,
                    ftlpu::MxmControlInstruction::IW(0));
                ++weight_address;
            }

            const auto first_compute = dequant_start + 24;
            for (std::size_t mb = 0; mb < kSeqLen / kTile; ++mb) {
                const auto compute_cycle = first_compute + mb * kComputeBlockCycles;
                const auto address = activation_address(kb, mb, 0);
                for (std::size_t byte = 0; byte < 4; ++byte) {
                    schedule.mem_repeat_at(
                        kActivationSlices[byte], compute_cycle - kActivationLatency,
                        ftlpu::MemInstruction::Read(address, ftlpu::StreamId::East(byte)),
                        kTile, 1);
                }
                schedule.mem_repeat_at(
                    ftlpu::hw::kWestAccumulatorMemSliceBase,
                    compute_cycle + kWestAccumulatorLatency,
                    ftlpu::MemInstruction::Accumulate(
                        result_address(mb * kTile, n_base),
                        ftlpu::StreamId::West(0)),
                    kTile,
                    kIntermediate / kTile);
                schedule.mem_repeat_at(
                    ftlpu::hw::kEastAccumulatorMemSliceBase,
                    compute_cycle + kEastAccumulatorLatency,
                    ftlpu::MemInstruction::Accumulate(
                        result_address(mb * kTile, n_base + kTile),
                        ftlpu::StreamId::West(4)),
                    kTile,
                    kIntermediate / kTile);
                schedule.mxm_compute_repeat_at(0, compute_cycle, ftlpu::MxmControlInstruction::Compute(0, 0, 0));
                schedule.mxm_compute_repeat_at(1, compute_cycle, ftlpu::MxmControlInstruction::Compute(0, 2, 4));
            }
            phase_start = first_compute + (kSeqLen / kTile) * kComputeBlockCycles;
        }
    }

    for (std::size_t cycle = 0; cycle < schedule.end_cycle() + 16; ++cycle) {
        try {
            system->tick({});
        } catch (const std::exception& error) {
            throw std::runtime_error("cycle " + std::to_string(cycle) + ": " + error.what());
        }
    }

    for (std::size_t m = 0; m < kSeqLen; ++m) {
        for (std::size_t n = 0; n < kIntermediate; ++n) {
            float expected = 0.0f;
            for (std::size_t kb = 0; kb < kHidden; kb += kTile) {
                float partial = 0.0f;
                for (std::size_t kk = 0; kk < kTile; ++kk) {
                    partial += activations[a_index(m, kb + kk)] * dequantized[w_index(kb + kk, n)];
                }
                expected += partial;
            }
            if (std::fabs(read_result(*system, m, n) - expected) > 1.0e-5f) {
                std::cerr << "offline projection mismatch at (" << m << ',' << n << ")\n";
                return 1;
            }
        }
    }
    std::cout << "offline ICU W8A16 projection passed: [128,576] x [576,1536]\n";
    return 0;
}

int main()
{
    try {
        return run_test();
    } catch (const std::exception& error) {
        std::cerr << "w8a16 projection exception: " << error.what() << '\n';
        return 1;
    }
}
