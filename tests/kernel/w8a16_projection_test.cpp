#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
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
constexpr std::size_t kActivationLatency =
    ftlpu::hw::kMxmBoundaryStreamRegisterColumn + 1
    - 32 / ftlpu::hw::kMemSlicesPerGroup;
constexpr std::size_t kComputeBlockCycles = 48;
constexpr std::size_t kWeightLoadCycles =
    ftlpu::hw::kMxmSupercellsPerPlane
    * ftlpu::hw::kMxmSupercellColumns;
constexpr std::array<std::size_t, 2> kWeightSlices {0, 8};
constexpr std::array<std::size_t, 4> kActivationSlices {32, 33, 34, 35};
constexpr std::array<std::size_t, 8> kOutputSlices {
    44, 45, 46, 47, 48, 49, 50, 51};

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

    void mem_at(std::size_t slice, std::size_t cycle, ftlpu::MemInstruction instruction)
    {
        const auto queue = ftlpu::InstructionControlUnit::mem_queue(
            ftlpu::Hemisphere::East, slice);
        pad(mem_[queue], cycle, [&](std::size_t n) { icu_.enqueue_mem_nop(queue, n); });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
    }

    void mem_repeat_at(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction first,
        std::size_t count,
        std::int64_t stride)
    {
        const auto queue = ftlpu::InstructionControlUnit::mem_queue(
            ftlpu::Hemisphere::East, slice);
        mem_at(slice, cycle, first);
        if (count > 1) {
            icu_.enqueue_mem_repeat(queue, count - 1, 1, stride);
        }
        advance(mem_[queue], cycle + count);
    }

    void mxm_dequant_at(
        std::size_t mxm,
        std::size_t cycle,
        float scale)
    {
        pad(mxm_dequant_[mxm], cycle, [&](std::size_t n) {
            icu_.enqueue_mxm_dequant_nop(mxm, n);
        });
        icu_.enqueue_mxm_dequant(
            mxm, ftlpu::MxmDequantInstruction::Scale(scale));
        advance(mxm_dequant_[mxm], cycle + 1);
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

    void mxm_accumulator_read_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t address,
        std::size_t stream_base)
    {
        pad(mxm_compute_[mxm], cycle, [&](std::size_t n) {
            icu_.enqueue_mxm_compute_nop(mxm, n);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::AccumulatorRead(
                address, stream_base, true));
        advance(mxm_compute_[mxm], cycle + 1);
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
    std::array<std::size_t, 2> mxm_load_{};
    std::array<std::size_t, 2> mxm_dequant_{};
    std::array<std::size_t, 2> mxm_compute_{};
    std::size_t end_cycle_{0};
};

std::size_t activation_address(std::size_t k_block, std::size_t m_block, std::size_t row)
{
    return (k_block * (kSeqLen / kTile) + m_block) * kTile + row;
}

std::size_t weight_read_latency(std::size_t mem_slice)
{
    return ftlpu::hw::kMxmBoundaryStreamRegisterColumn + 1
        - mem_slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mxm_to_mem_write_latency(std::size_t mem_slice)
{
    return ftlpu::hw::kSystemStreamRegisterColumns - 1
        - mem_slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t result_address(std::size_t row, std::size_t column)
{
    return row * (kIntermediate / kTile) + column / kTile;
}

float read_result(const ftlpu::TspSliceSystem& system, std::size_t row, std::size_t column)
{
    const auto mxm = (column / kTile) % 2;
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    auto raw = std::uint32_t {0};
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(
            system.read_mem_sram_lane_byte(
                kOutputSlices[mxm * sizeof(float) + byte],
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
                        const auto bits = ftlpu::Bf16::from_float(
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
            activations[a_index(m, k)] =
                ftlpu::Bf16::from_float(activation_value(m, k)).to_float();
        }
    }

    std::vector<float> scales(kIntermediate);
    std::vector<std::int8_t> weights(kHidden * kIntermediate);
    std::vector<float> dequantized(kHidden * kIntermediate);
    for (std::size_t n = 0; n < kIntermediate; ++n) {
        float max_abs = 0.0f;
        for (std::size_t k = 0; k < kHidden; ++k) max_abs = std::max(max_abs, std::fabs(weight_value(k, n)));
        scales[n] = ftlpu::Bf16::from_float(max_abs / 127.0f).to_float();
        for (std::size_t k = 0; k < kHidden; ++k) {
            const auto q = std::clamp(static_cast<int>(std::lround(weight_value(k, n) / scales[n])), -127, 127);
            weights[w_index(k, n)] = static_cast<std::int8_t>(q);
            dequantized[w_index(k, n)] = ftlpu::Bf16::from_float(
                static_cast<float>(q) * scales[n]).to_float();
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
            // Slice 0 needs 15 cycles to reach the MXM boundary.  Keep a
            // full 20-cycle read/fabric prefill before the first IWColumn.
            const auto weight_load_start = phase_start + 20;
            for (std::size_t local_column = 0;
                 local_column < kTile;
                 ++local_column) {
                const auto block =
                    local_column / ftlpu::hw::kMxmSupercellColumns;
                const auto inner_column =
                    local_column % ftlpu::hw::kMxmSupercellColumns;
                const auto load_cycle = weight_load_start + local_column;
                for (std::size_t mxm = 0; mxm < 2; ++mxm) {
                    const auto n = n_base + mxm * kTile + local_column;
                    const auto mem_slice = kWeightSlices[mxm];
                    const auto stream =
                        mxm * ftlpu::hw::kMxmInt8LoadStreamStride;
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
                        load_cycle - weight_read_latency(mem_slice),
                        ftlpu::MemInstruction::Read(
                            weight_address,
                            ftlpu::StreamId::East(stream)));
                    schedule.mxm_dequant_at(mxm, load_cycle, scales[n]);
                    schedule.mxm_load_at(
                        mxm,
                        load_cycle,
                        ftlpu::MxmControlInstruction::IWColumn(
                            0, block, inner_column));
                    ++weight_address;
                }
            }

            const auto first_compute =
                weight_load_start + kWeightLoadCycles + 4;
            for (std::size_t mb = 0; mb < kSeqLen / kTile; ++mb) {
                const auto compute_cycle = first_compute + mb * kComputeBlockCycles;
                const auto address = activation_address(kb, mb, 0);
                for (std::size_t byte = 0; byte < 4; ++byte) {
                    schedule.mem_repeat_at(
                        kActivationSlices[byte], compute_cycle - kActivationLatency,
                        ftlpu::MemInstruction::Read(address, ftlpu::StreamId::East(byte)),
                        kTile, 1);
                }
                schedule.mxm_compute_repeat_at(
                    0,
                    compute_cycle,
                    ftlpu::MxmControlInstruction::Compute(
                        0,
                        0,
                        0,
                        result_address(mb * kTile, n_base),
                        kIntermediate / kTile,
                        ftlpu::MxmAccumulatorDestination::Sram,
                        ftlpu::MxmDataFormat::BFloat16));
                schedule.mxm_compute_repeat_at(
                    1,
                    compute_cycle,
                    ftlpu::MxmControlInstruction::Compute(
                        0,
                        2,
                        4,
                        result_address(mb * kTile, n_base + kTile),
                        kIntermediate / kTile,
                        ftlpu::MxmAccumulatorDestination::Sram,
                        ftlpu::MxmDataFormat::BFloat16));
            }
            phase_start = first_compute + (kSeqLen / kTile) * kComputeBlockCycles;
        }
    }

    // Read every completed accumulator row through the architectural MXM
    // output instruction.  The two local MXMs use disjoint stream groups,
    // and MEM captures those streams into the corresponding result address.
    const auto output_read_start = phase_start + 8;
    constexpr auto kResultBlockPairs = kIntermediate / (2 * kTile);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t pair = 0; pair < kResultBlockPairs; ++pair) {
            const auto read_cycle =
                output_read_start + row * kResultBlockPairs + pair;
            for (std::size_t mxm = 0; mxm < 2; ++mxm) {
                const auto column_block = pair * 2 + mxm;
                const auto address = row * (kIntermediate / kTile)
                    + column_block;
                const auto stream_base = mxm * sizeof(float);
                schedule.mxm_accumulator_read_at(
                    mxm, read_cycle, address, stream_base);
                for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                    const auto slice =
                        kOutputSlices[mxm * sizeof(float) + byte];
                    schedule.mem_at(
                        slice,
                        read_cycle + mxm_to_mem_write_latency(slice),
                        ftlpu::MemInstruction::Write(
                            address,
                            ftlpu::StreamId::West(stream_base + byte)));
                }
            }
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
    std::cout
        << "W8A16 projection black-box passed: MEM INT8/BF16 -> "
           "MXM column dequant/MAC -> AccumulatorRead -> MEM, "
           "[128,576] x [576,1536]\n";
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
