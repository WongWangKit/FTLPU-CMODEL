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
#include <memory>
#include <vector>

namespace {

constexpr std::size_t kSeqLen = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
constexpr std::size_t kTile = ftlpu::hw::kMxmRows;
constexpr std::size_t kActivationLatency = 5;
constexpr std::size_t kWeightToIwLatency = 14;
constexpr std::size_t kGateAccumulatorLatency = 6;
constexpr std::size_t kUpAccumulatorLatency = 5;
constexpr std::size_t kComputeBlockCycles = 48;
constexpr std::size_t kSwishOutputWriteLatency = 13;
constexpr std::size_t kOutputLowSlice = 29;
constexpr std::size_t kOutputHighSlice = 30;
constexpr std::array<std::size_t, 8> kWeightSlices {0, 4, 8, 12, 16, 20, 24, 28};
constexpr std::array<std::size_t, 4> kActivationSlices {32, 33, 34, 35};

static_assert(kTile == 32);
static_assert(kSeqLen % kTile == 0);
static_assert(kHidden % kTile == 0);
static_assert(kIntermediate % kTile == 0);

enum class Projection : std::size_t { Gate, Up };

std::size_t a_index(std::size_t m, std::size_t k) { return m * kHidden + k; }
std::size_t w_index(std::size_t k, std::size_t n) { return k * kIntermediate + n; }

float activation_value(std::size_t m, std::size_t k)
{
    return static_cast<float>(static_cast<int>((m * 7 + k * 5) % 23) - 11) * 0.0625f;
}

float weight_value(Projection projection, std::size_t k, std::size_t n)
{
    const auto p = static_cast<std::size_t>(projection);
    const auto raw = static_cast<int>((k * (11 + p * 6) + n * (5 + p * 2) + p * 13) % 41) - 20;
    return static_cast<float>(raw) * (0.006f + static_cast<float>((n + p * 3) % 13) * 0.001f);
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

    void mxm_load_at(std::size_t mxm, std::size_t cycle, std::size_t weight_column)
    {
        pad(mxm_load_[mxm], cycle, [&](std::size_t n) { icu_.enqueue_mxm_load_nop(mxm, n); });
        icu_.enqueue_mxm(mxm, ftlpu::MxmControlInstruction::IW(0, weight_column));
        advance(mxm_load_[mxm], cycle + 1);
    }

    void mxm_compute_repeat_at(std::size_t mxm, std::size_t cycle, std::size_t activation, std::size_t output)
    {
        pad(mxm_compute_[mxm], cycle, [&](std::size_t n) { icu_.enqueue_mxm_compute_nop(mxm, n); });
        icu_.enqueue_mxm(mxm, ftlpu::MxmControlInstruction::Compute(0, activation, output));
        icu_.enqueue_mxm_compute_repeat(mxm, kTile - 1, 1);
        advance(mxm_compute_[mxm], cycle + kTile);
    }

    std::size_t end_cycle() const { return end_cycle_; }

private:
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
    std::array<std::size_t, 2> mxm_load_{};
    std::array<std::size_t, 2> mxm_compute_{};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_{};
    std::size_t end_cycle_{0};
};

std::size_t activation_address(std::size_t k_block, std::size_t m_block, std::size_t row)
{
    return (k_block * (kSeqLen / kTile) + m_block) * kTile + row;
}

std::size_t west_read_latency(std::size_t mem_slice)
{
    return mem_slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t accumulator_address(std::size_t row, std::size_t n_base)
{
    return row * (kIntermediate / kTile) + n_base / kTile;
}

std::size_t output_address(std::size_t row, std::size_t n_base)
{
    return (n_base / kTile) * kSeqLen + row;
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

float read_swiglu(
    const ftlpu::TspSliceSystem& system,
    std::size_t row,
    std::size_t column)
{
    const auto n_base = column / kTile * kTile;
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto address = output_address(row, n_base);
    const auto low = system.read_mem_sram_lane_byte(kOutputLowSlice, tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(kOutputHighSlice, tile, address, lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low) | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

float read_accumulator(
    const ftlpu::TspSliceSystem& system,
    std::size_t group_base,
    std::size_t row,
    std::size_t column)
{
    const auto n_base = column / kTile * kTile;
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(system.read_mem_sram_lane_byte(
            group_base + byte,
            tile,
            accumulator_address(row, n_base),
            lane)) << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

} // namespace

int main()
{
    std::vector<float> activations(kSeqLen * kHidden);
    for (std::size_t m = 0; m < kSeqLen; ++m) {
        for (std::size_t k = 0; k < kHidden; ++k) {
            activations[a_index(m, k)] = ftlpu::Fp16::from_float(activation_value(m, k)).to_float();
        }
    }

    std::array<std::vector<float>, 2> scales {
        std::vector<float>(kIntermediate),
        std::vector<float>(kIntermediate),
    };
    std::array<std::vector<std::int8_t>, 2> weights {
        std::vector<std::int8_t>(kHidden * kIntermediate),
        std::vector<std::int8_t>(kHidden * kIntermediate),
    };
    std::array<std::vector<float>, 2> dequantized {
        std::vector<float>(kHidden * kIntermediate),
        std::vector<float>(kHidden * kIntermediate),
    };
    for (std::size_t p = 0; p < 2; ++p) {
        const auto projection = static_cast<Projection>(p);
        for (std::size_t n = 0; n < kIntermediate; ++n) {
            float max_abs = 0.0f;
            for (std::size_t k = 0; k < kHidden; ++k) {
                max_abs = std::max(max_abs, std::fabs(weight_value(projection, k, n)));
            }
            scales[p][n] = max_abs / 127.0f;
            for (std::size_t k = 0; k < kHidden; ++k) {
                const auto q = std::clamp(
                    static_cast<int>(std::lround(weight_value(projection, k, n) / scales[p][n])),
                    -127,
                    127);
                weights[p][w_index(k, n)] = static_cast<std::int8_t>(q);
                dequantized[p][w_index(k, n)] = ftlpu::Fp16::from_float(
                    static_cast<float>(q) * scales[p][n]).to_float();
            }
        }
    }

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    initialize_activations(*system, activations);
    auto schedule = OfflineSchedule(system->icu());

    std::size_t phase_start = 0;
    std::size_t weight_address = 0;
    for (std::size_t n_base = 0; n_base < kIntermediate; n_base += kTile) {
        for (std::size_t kb = 0; kb < kHidden / kTile; ++kb) {
            const auto k_base = kb * kTile;
            const auto dequant_start = phase_start + 10;
            for (std::size_t pulse = 0; pulse < 8; ++pulse) {
                const auto mxm = pulse / 4;
                const auto block = 3 - pulse % 4;
                const auto dequant_cycle = dequant_start + pulse;
                auto instruction = ftlpu::test::DequantSpec {};
                instruction.input_stream_base = ftlpu::hw::kEastStreams;
                instruction.output_stream_base = mxm * 16;
                for (std::size_t stream = 0; stream < 8; ++stream) {
                    const auto n = n_base + block * 8 + stream;
                    instruction.scales[stream] = scales[mxm][n];
                    const auto mem_slice = kWeightSlices[stream];
                    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                            const auto k = k_base + tile * ftlpu::hw::kLanesPerTile + lane;
                            system->initialize_mem_sram_lane_byte(
                                mem_slice,
                                tile,
                                weight_address,
                                lane,
                                static_cast<std::uint8_t>(weights[mxm][w_index(k, n)]));
                        }
                    }
                    schedule.mem_at(
                        mem_slice,
                        dequant_cycle - west_read_latency(mem_slice),
                        ftlpu::MemInstruction::Read(weight_address, ftlpu::StreamId::West(stream)));
                }
                schedule.dequant_at(dequant_cycle, instruction);
                schedule.mxm_load_at(mxm, dequant_cycle + kWeightToIwLatency, block);
                ++weight_address;
            }

            const auto first_compute = dequant_start + 24;
            for (std::size_t mb = 0; mb < kSeqLen / kTile; ++mb) {
                const auto compute_cycle = first_compute + mb * kComputeBlockCycles;
                const auto address = activation_address(kb, mb, 0);
                for (std::size_t byte = 0; byte < 4; ++byte) {
                    schedule.mem_repeat_at(
                        kActivationSlices[byte],
                        compute_cycle - kActivationLatency,
                        ftlpu::MemInstruction::Read(address, ftlpu::StreamId::East(byte)),
                        kTile,
                        1);
                }
                schedule.mem_repeat_at(
                    ftlpu::hw::kWestAccumulatorMemSliceBase,
                    compute_cycle + kGateAccumulatorLatency,
                    ftlpu::MemInstruction::Accumulate(
                        accumulator_address(mb * kTile, n_base),
                        ftlpu::StreamId::West(0)),
                    kTile,
                    kIntermediate / kTile);
                schedule.mem_repeat_at(
                    ftlpu::hw::kEastAccumulatorMemSliceBase,
                    compute_cycle + kUpAccumulatorLatency,
                    ftlpu::MemInstruction::Accumulate(
                        accumulator_address(mb * kTile, n_base),
                        ftlpu::StreamId::West(4)),
                    kTile,
                    kIntermediate / kTile);
                schedule.mxm_compute_repeat_at(0, compute_cycle, 0, 0);
                schedule.mxm_compute_repeat_at(1, compute_cycle, 2, 4);
            }
            phase_start = first_compute + (kSeqLen / kTile) * kComputeBlockCycles;
        }
    }

    const auto swish_start = phase_start + 16;
    for (std::size_t n_base = 0; n_base < kIntermediate; n_base += kTile) {
        for (std::size_t row = 0; row < kSeqLen; ++row) {
            const auto cycle = swish_start + (n_base / kTile) * kSeqLen + row;
            const auto acc_address = accumulator_address(row, n_base);
            for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                const auto gate_slice = ftlpu::hw::kWestAccumulatorMemSliceBase + byte;
                const auto up_slice = ftlpu::hw::kEastAccumulatorMemSliceBase + byte;
                schedule.mem_at(
                    gate_slice,
                    cycle - west_read_latency(gate_slice),
                    ftlpu::MemInstruction::Read(acc_address, ftlpu::StreamId::West(byte)));
                schedule.mem_at(
                    up_slice,
                    cycle - west_read_latency(up_slice),
                    ftlpu::MemInstruction::Read(acc_address, ftlpu::StreamId::West(4 + byte)));
            }
            schedule.swish_at(cycle, ftlpu::test::SwishSpec {32, 36, 0});
            schedule.mem_at(
                kOutputLowSlice,
                cycle + kSwishOutputWriteLatency,
                ftlpu::MemInstruction::Write(output_address(row, n_base), ftlpu::StreamId::East(0)));
            schedule.mem_at(
                kOutputHighSlice,
                cycle + kSwishOutputWriteLatency,
                ftlpu::MemInstruction::Write(output_address(row, n_base), ftlpu::StreamId::East(1)));
        }
    }

    for (std::size_t cycle = 0; cycle < schedule.end_cycle() + 16; ++cycle) system->tick({});

    for (std::size_t m = 0; m < kSeqLen; ++m) {
        for (std::size_t n = 0; n < kIntermediate; ++n) {
            std::array<float, 2> projected {};
            for (std::size_t p = 0; p < 2; ++p) {
                for (std::size_t kb = 0; kb < kHidden; kb += kTile) {
                    float partial = 0.0f;
                    for (std::size_t kk = 0; kk < kTile; ++kk) {
                        partial += activations[a_index(m, kb + kk)]
                            * dequantized[p][w_index(kb + kk, n)];
                    }
                    projected[p] += partial;
                }
            }
            const auto gate = read_accumulator(
                *system, ftlpu::hw::kWestAccumulatorMemSliceBase, m, n);
            const auto up = read_accumulator(
                *system, ftlpu::hw::kEastAccumulatorMemSliceBase, m, n);
            if (std::fabs(gate - projected[0]) > 1.0e-5f
                || std::fabs(up - projected[1]) > 1.0e-5f) {
                std::cerr << "offline projection mismatch before SwiGLU at (" << m << ',' << n << ")\n";
                return 1;
            }
            const auto sigmoid = 1.0f / (1.0f + std::exp(-gate));
            const auto expected = ftlpu::Fp16::from_float((gate * up) * sigmoid).to_float();
            const auto actual = read_swiglu(*system, m, n);
            if (actual != expected && !(std::isnan(actual) && std::isnan(expected))) {
                std::cerr << "offline SwiGLU mismatch at (" << m << ',' << n << ")"
                          << " gate=" << gate << " up=" << up
                          << " actual=" << actual << " expected=" << expected
                          << " actual_bits=0x" << std::hex << ftlpu::Fp16::from_float(actual).bits()
                          << " expected_bits=0x" << ftlpu::Fp16::from_float(expected).bits()
                          << std::dec << '\n';
                return 1;
            }
        }
    }

    std::cout << "offline ICU W8A16 SwiGLU passed: X[128,576], gate/up[576,1536]\n";
    return 0;
}
