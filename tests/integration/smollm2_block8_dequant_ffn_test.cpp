#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "smollm2_layer_phases.hpp"
#include "system_gantt_trace.hpp"
#include "w8a16_swiglu_dataflow_harness.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

#if defined(FTLPU_BLOCK8_FFN_SMOKE)
#ifndef FTLPU_BLOCK8_FFN_SMOKE_ROWS
#define FTLPU_BLOCK8_FFN_SMOKE_ROWS 16
#endif
#ifndef FTLPU_BLOCK8_FFN_SMOKE_HIDDEN
#define FTLPU_BLOCK8_FFN_SMOKE_HIDDEN 64
#endif
#ifndef FTLPU_BLOCK8_FFN_SMOKE_INTERMEDIATE
#define FTLPU_BLOCK8_FFN_SMOKE_INTERMEDIATE 64
#endif
constexpr std::size_t kRows = FTLPU_BLOCK8_FFN_SMOKE_ROWS;
constexpr std::size_t kHidden = FTLPU_BLOCK8_FFN_SMOKE_HIDDEN;
constexpr std::size_t kIntermediate =
    FTLPU_BLOCK8_FFN_SMOKE_INTERMEDIATE;
#else
constexpr std::size_t kRows = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
#endif

constexpr std::size_t kTile = ftlpu::hw::kMxmRows;
constexpr std::size_t kBlockRows = ftlpu::hw::kMxmBlockRows;
constexpr std::size_t kRowBlocks = kRows / kBlockRows;
constexpr std::size_t kGateUpColumnsPerWave = 2 * kTile;
constexpr std::size_t kDownColumnsPerWave =
    ftlpu::TspSliceSystem::kMxmCount * kTile;
constexpr std::size_t kMxmOutputLatency =
    ftlpu::hw::kMxmSupercellsPerPlane
    + ftlpu::Mxm::kLocalMacStages - 2;
constexpr std::size_t kVxmSwiGluLatency = 17;
constexpr std::size_t kLoadToComputeGap = 8;
constexpr std::size_t kComputeToLoadGap = 8;
constexpr std::size_t kBlockGroupStride = 12;
constexpr std::size_t kComputeSpan =
    ((kRowBlocks + 3) / 4 - 1) * kBlockGroupStride
    + std::min<std::size_t>(4, kRowBlocks);

constexpr std::array<std::array<std::size_t, 8>, 2> kWeightSlices {{
    {{0, 1, 2, 3, 4, 5, 6, 7}},
    {{8, 9, 10, 11, 12, 13, 14, 15}},
}};
constexpr std::array<std::size_t, 16> kGateSlices {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
};
constexpr std::array<std::size_t, 16> kUpSlices {
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
};
constexpr std::array<std::size_t, 16> kActivationSlices {
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
};
constexpr std::array<std::size_t, 32> kFp32OutputSlices {
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
};

constexpr std::size_t kGateUpAddressBase = 8192;
constexpr std::size_t kSwiGluAddressBase = 16384;

static_assert(kRows % kBlockRows == 0);
static_assert(kHidden % kTile == 0);
static_assert(kIntermediate % kGateUpColumnsPerWave == 0);

enum class Projection : std::size_t { Gate, Up };

struct QuantizedMatrix {
    std::size_t rows{0};
    std::size_t columns{0};
    std::vector<float> scales{};
    std::vector<std::int8_t> weights{};
    std::vector<float> dequantized{};

    std::size_t index(std::size_t row, std::size_t column) const
    {
        return row * columns + column;
    }
};

struct PhaseSpan {
    std::size_t start{0};
    std::size_t end{0};
    std::string label{};
};

inline ftlpu::Hemisphere hemisphere(std::size_t index)
{
    return static_cast<ftlpu::Hemisphere>(index);
}

inline std::size_t mem_queue(
    ftlpu::Hemisphere side, std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(side, slice);
}

inline std::size_t mem_to_mxm_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

inline std::size_t mem_to_vxm_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

inline std::size_t west_source_to_mem_latency(std::size_t slice)
{
    return ftlpu::hw::kSystemStreamRegisterColumns - 1
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

inline std::size_t vxm_to_mem_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 1;
}

inline std::size_t block_compute_cycle(
    std::size_t start, std::size_t row_block)
{
    return start
        + (row_block / 4) * kBlockGroupStride
        + row_block % 4;
}

inline std::size_t gate_up_address(
    std::size_t wave, std::size_t row_block)
{
    return kGateUpAddressBase + wave * kRowBlocks + row_block;
}

inline std::size_t swiglu_address(
    std::size_t reduction, std::size_t row_block)
{
    return kSwiGluAddressBase + reduction * kRowBlocks + row_block;
}

inline std::size_t block_accumulator_address(
    std::size_t wave, std::size_t row_block)
{
    return wave * kRowBlocks + (row_block / 4) * 4;
}

inline std::size_t final_block_address(
    std::size_t output_block, std::size_t row_block)
{
    return output_block * kRowBlocks + row_block;
}

class Schedule {
public:
    explicit Schedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(
        ftlpu::Hemisphere side, std::size_t slice,
        std::size_t cycle, ftlpu::MemInstruction instruction)
    {
        auto& cursor = mem_[mem_queue(side, slice)];
        if (cycle < cursor) {
            throw std::logic_error(
                "Block8 FFN overlaps MEM side="
                + std::to_string(static_cast<std::size_t>(side))
                + " slice=" + std::to_string(slice)
                + " cycle=" + std::to_string(cycle)
                + " cursor=" + std::to_string(cursor));
        }
        icu_.enqueue_mem_nop(mem_queue(side, slice), cycle - cursor);
        icu_.enqueue_mem(
            mem_queue(side, slice), std::move(instruction));
        advance(cursor, cycle + 1);
    }

    void mxm_dequant_at(
        std::size_t mxm, std::size_t cycle, float scale)
    {
        auto& cursor = mxm_dequant_[mxm];
        require_available(cursor, cycle, "MXM dequant");
        icu_.enqueue_mxm_dequant_nop(mxm, cycle - cursor);
        icu_.enqueue_mxm_dequant(
            mxm, ftlpu::MxmDequantInstruction::Scale(scale));
        advance(cursor, cycle + 1);
    }

    void mxm_load_at(
        std::size_t mxm, std::size_t cycle,
        std::size_t column_block)
    {
        auto& cursor = mxm_load_[mxm];
        require_available(cursor, cycle, "MXM load");
        icu_.enqueue_mxm_load_nop(mxm, cycle - cursor);
        icu_.enqueue_mxm(
            mxm, ftlpu::MxmControlInstruction::IW(0, column_block));
        advance(cursor, cycle + 1);
    }

    void mxm_compute_at(
        std::size_t mxm, std::size_t cycle,
        std::size_t accumulator_address,
        bool final_reduction, std::size_t output_stream)
    {
        auto& cursor = mxm_compute_[mxm];
        require_available(cursor, cycle, "MXM compute");
        icu_.enqueue_mxm_compute_nop(mxm, cycle - cursor);
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::Compute(
                0, 0, output_stream, accumulator_address, 1,
                final_reduction
                    ? ftlpu::MxmAccumulatorDestination::Stream
                    : ftlpu::MxmAccumulatorDestination::Sram,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmComputeMode::Block8,
                final_reduction));
        advance(cursor, cycle + 1);
    }

    void mxm_accumulator_read_at(
        std::size_t mxm, std::size_t cycle,
        std::size_t address)
    {
        auto& cursor = mxm_compute_[mxm];
        require_available(cursor, cycle, "MXM accumulator read");
        icu_.enqueue_mxm_compute_nop(mxm, cycle - cursor);
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::AccumulatorRead(
                address, 0, true,
                ftlpu::MxmComputeMode::Block8));
        advance(cursor, cycle + 1);
    }

    void vxm_at(
        std::size_t stage, std::size_t cycle,
        const ftlpu::VxmLaneAluInstruction& instruction)
    {
        auto& cursor = vxm_[stage];
        require_available(cursor, cycle, "VXM");
        icu_.enqueue_vxm_nop(stage, cycle - cursor);
        icu_.enqueue_vxm(
            stage, ftlpu::VxmChainDepth::Eight, instruction);
        advance(cursor, cycle + 1);
    }

    std::size_t end_cycle() const noexcept { return end_cycle_; }

private:
    static void require_available(
        std::size_t cursor, std::size_t cycle, const char* queue)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                std::string("Block8 FFN overlaps ") + queue);
        }
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_cycle_ = std::max(end_cycle_, next);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMxmQueues>
        mxm_load_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMxmQueues>
        mxm_dequant_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMxmQueues>
        mxm_compute_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kVxmQueues> vxm_{};
    std::size_t end_cycle_{0};
};

inline float activation_value(std::size_t row, std::size_t column)
{
    return static_cast<float>(
        static_cast<int>((row * 7 + column * 5) % 23) - 11)
        * 0.0625f;
}

inline float gate_up_weight_value(
    Projection projection, std::size_t row, std::size_t column)
{
    const auto p = static_cast<std::size_t>(projection);
    const auto raw = static_cast<int>(
        (row * (11 + p * 6) + column * (5 + p * 2) + p * 13)
        % 41) - 20;
    return static_cast<float>(raw)
        * (0.009f + static_cast<float>((column + p) % 7) * 0.001f);
}

inline float down_weight_value(std::size_t row, std::size_t column)
{
    const auto raw =
        static_cast<int>((row * 19 + column * 11 + 7) % 47) - 23;
    return static_cast<float>(raw)
        * (0.006f + static_cast<float>((column + 3) % 9) * 0.001f);
}

template <typename Lhs, typename Rhs>
inline float block8_wavefront_dot(Lhs lhs, Rhs rhs)
{
    // A Block8 result is accumulated independently inside each 8-row
    // Supercell (row 7 reaches column 0 first), then the four tile partials
    // travel north and are added local-first.  Preserve that FP32 order in
    // the golden because the architectural result is rounded to BF16.
    auto vertical = 0.0f;
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        auto local = 0.0f;
        for (std::size_t lane = ftlpu::hw::kLanesPerTile;
             lane-- > 0;) {
            const auto index = tile * ftlpu::hw::kLanesPerTile + lane;
            local += lhs(index) * rhs(index);
        }
        vertical = tile == 0 ? local : local + vertical;
    }
    return vertical;
}

template <typename Value>
inline QuantizedMatrix quantize_group8(
    std::size_t rows, std::size_t columns, Value value)
{
    auto result = QuantizedMatrix {
        rows,
        columns,
        std::vector<float>(columns / 8),
        std::vector<std::int8_t>(rows * columns),
        std::vector<float>(rows * columns),
    };
    for (std::size_t group = 0; group < columns / 8; ++group) {
        float maximum = 0.0f;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < 8; ++column) {
                maximum = std::max(
                    maximum, std::fabs(value(row, group * 8 + column)));
            }
        }
        const auto scale =
            ftlpu::Bf16::from_float(maximum / 127.0f).to_float();
        result.scales[group] = scale;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < 8; ++column) {
                const auto n = group * 8 + column;
                const auto q = std::clamp(
                    static_cast<int>(std::lround(value(row, n) / scale)),
                    -127, 127);
                result.weights[result.index(row, n)] =
                    static_cast<std::int8_t>(q);
                result.dequantized[result.index(row, n)] =
                    ftlpu::Bf16::from_float(
                        static_cast<float>(q) * scale).to_float();
            }
        }
    }
    return result;
}

inline void initialize_block_activations(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& values, std::size_t width)
{
    for (std::size_t side_index = 0;
         side_index < ftlpu::hw::kHemispheres; ++side_index) {
        const auto side = hemisphere(side_index);
        for (std::size_t reduction = 0;
             reduction < width / kTile; ++reduction) {
            for (std::size_t row_block = 0;
                 row_block < kRowBlocks; ++row_block) {
                const auto address = reduction * kRowBlocks + row_block;
                for (std::size_t row_in_block = 0;
                     row_in_block < kBlockRows; ++row_in_block) {
                    for (std::size_t tile = 0;
                         tile < ftlpu::hw::kTileRows; ++tile) {
                        for (std::size_t lane = 0;
                             lane < ftlpu::hw::kLanesPerTile; ++lane) {
                            const auto row =
                                row_block * kBlockRows + row_in_block;
                            const auto column = reduction * kTile
                                + tile * ftlpu::hw::kLanesPerTile + lane;
                            const auto bits = ftlpu::Bf16::from_float(
                                values[row * width + column]).bits();
                            system.initialize_mem_sram_lane_byte(
                                side, kActivationSlices[row_in_block * 2],
                                tile, address, lane,
                                static_cast<std::uint8_t>(bits & 0xffu));
                            system.initialize_mem_sram_lane_byte(
                                side,
                                kActivationSlices[row_in_block * 2 + 1],
                                tile, address, lane,
                                static_cast<std::uint8_t>(bits >> 8));
                        }
                    }
                }
            }
        }
    }
}

inline void schedule_block_activation(
    Schedule& schedule, std::size_t compute_cycle,
    std::size_t reduction, std::size_t row_block,
    std::size_t address_base = 0)
{
    const auto address =
        address_base + reduction * kRowBlocks + row_block;
    for (std::size_t side_index = 0;
         side_index < ftlpu::hw::kHemispheres; ++side_index) {
        const auto side = hemisphere(side_index);
        for (std::size_t stream = 0;
             stream < kActivationSlices.size(); ++stream) {
            const auto slice = kActivationSlices[stream];
            schedule.mem_at(
                side, slice, compute_cycle - mem_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    address, ftlpu::StreamId::East(stream)));
        }
    }
}

inline void schedule_weight_block(
    ftlpu::TspSliceSystem& system, Schedule& schedule,
    ftlpu::Hemisphere side, std::size_t local_mxm,
    const QuantizedMatrix& weights,
    std::size_t reduction, std::size_t output_base,
    std::size_t address, std::size_t column_block,
    std::size_t load_cycle)
{
    const auto mxm =
        ftlpu::InstructionControlUnit::mxm_queue(side, local_mxm);
    const auto group = (output_base + column_block * 8) / 8;
    for (std::size_t stream = 0; stream < 8; ++stream) {
        const auto slice = kWeightSlices[local_mxm][stream];
        const auto column = output_base + column_block * 8 + stream;
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto row = reduction * kTile
                    + tile * ftlpu::hw::kLanesPerTile + lane;
                system.initialize_mem_sram_lane_byte(
                    side, slice, tile, address, lane,
                    static_cast<std::uint8_t>(
                        weights.weights[weights.index(row, column)]));
            }
        }
        schedule.mem_at(
            side, slice, load_cycle - mem_to_mxm_latency(slice),
            ftlpu::MemInstruction::Read(
                address,
                ftlpu::StreamId::East(
                    local_mxm * ftlpu::hw::kMxmInt8LoadStreamStride
                    + stream)));
    }
    schedule.mxm_dequant_at(mxm, load_cycle, weights.scales[group]);
    schedule.mxm_load_at(mxm, load_cycle, column_block);
}

inline std::size_t build_gate_up_schedule(
    ftlpu::TspSliceSystem& system, Schedule& schedule,
    const QuantizedMatrix& gate, const QuantizedMatrix& up)
{
    const std::array<const QuantizedMatrix*, 2> projections {&gate, &up};
    constexpr auto waves = kIntermediate / kGateUpColumnsPerWave;
    constexpr auto reductions = kHidden / kTile;
    auto cycle = std::size_t {20};
    for (std::size_t wave = 0; wave < waves; ++wave) {
        for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
            const auto load_start = cycle;
            for (std::size_t column_block = 0;
                 column_block < ftlpu::hw::kMxmSupercellsPerPlane;
                 ++column_block) {
                const auto load_cycle = load_start + column_block;
                for (std::size_t side_index = 0;
                     side_index < ftlpu::hw::kHemispheres; ++side_index) {
                    const auto side = hemisphere(side_index);
                    const auto output_base =
                        wave * kGateUpColumnsPerWave + side_index * kTile;
                    for (std::size_t local_mxm = 0;
                         local_mxm < 2; ++local_mxm) {
                        const auto address =
                            (wave * reductions + reduction) * 4
                            + column_block;
                        schedule_weight_block(
                            system, schedule, side, local_mxm,
                            *projections[local_mxm], reduction, output_base,
                            address, column_block, load_cycle);
                    }
                }
            }

            const auto compute_start = load_start + kLoadToComputeGap;
            const auto final_reduction = reduction + 1 == reductions;
            for (std::size_t row_block = 0;
                 row_block < kRowBlocks; ++row_block) {
                const auto compute_cycle =
                    block_compute_cycle(compute_start, row_block);
                schedule_block_activation(
                    schedule, compute_cycle, reduction, row_block);
                for (std::size_t side_index = 0;
                     side_index < ftlpu::hw::kHemispheres; ++side_index) {
                    const auto side = hemisphere(side_index);
                    for (std::size_t local_mxm = 0;
                         local_mxm < 2; ++local_mxm) {
                        const auto mxm =
                            ftlpu::InstructionControlUnit::mxm_queue(
                                side, local_mxm);
                        schedule.mxm_compute_at(
                            mxm, compute_cycle,
                            block_accumulator_address(wave, row_block),
                            final_reduction, local_mxm * 16);
                    }
                    if (final_reduction) {
                        const auto output_cycle =
                            compute_cycle + kMxmOutputLatency;
                        for (std::size_t local_mxm = 0;
                             local_mxm < 2; ++local_mxm) {
                            const auto& slices = local_mxm == 0
                                ? kGateSlices : kUpSlices;
                            for (std::size_t stream = 0;
                                 stream < 16; ++stream) {
                                const auto slice = slices[stream];
                                schedule.mem_at(
                                    side, slice,
                                    output_cycle
                                        + west_source_to_mem_latency(slice),
                                    ftlpu::MemInstruction::Write(
                                        gate_up_address(wave, row_block),
                                        ftlpu::StreamId::West(
                                            local_mxm * 16 + stream)));
                            }
                        }
                    }
                }
            }
            cycle = compute_start + kComputeSpan
                + kMxmOutputLatency + kComputeToLoadGap;
            if (final_reduction) cycle += 24;
        }
    }
    return std::max(cycle, schedule.end_cycle());
}

inline ftlpu::Bf16 read_staged_projection(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere side, std::size_t projection,
    std::size_t row, std::size_t column)
{
    const auto wave = column / kGateUpColumnsPerWave;
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto row_in_block = row % kBlockRows;
    const auto& slices = projection == 0 ? kGateSlices : kUpSlices;
    const auto low = system.read_mem_sram_lane_byte(
        side, slices[row_in_block * 2], tile,
        gate_up_address(wave, row / kBlockRows), lane);
    const auto high = system.read_mem_sram_lane_byte(
        side, slices[row_in_block * 2 + 1], tile,
        gate_up_address(wave, row / kBlockRows), lane);
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8));
}

inline void configure_swiglu_program(
    Schedule& schedule, std::size_t config_cycle,
    std::size_t repeat)
{
    using namespace ftlpu::test::w8a16_swiglu;
    schedule.vxm_at(
        0, config_cycle,
        basic(
            ftlpu::VxmAluOpcode::Negate,
            ftlpu::VxmLaneOperand::StreamBFloat16(),
            ftlpu::VxmLaneOperand::StreamBFloat16(), repeat));
    schedule.vxm_at(
        1, config_cycle,
        special(
            ftlpu::VxmSpecialAluOpcode::Exp,
            ftlpu::VxmLaneOperand::Previous(), repeat));
    schedule.vxm_at(
        2, config_cycle,
        basic(
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Imm(1.0f), repeat));
    schedule.vxm_at(
        3, config_cycle,
        special(
            ftlpu::VxmSpecialAluOpcode::Reciprocal,
            ftlpu::VxmLaneOperand::Previous(), repeat));
    schedule.vxm_at(
        4, config_cycle,
        basic(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Original(), repeat));
    schedule.vxm_at(
        5, config_cycle,
        basic(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Aux(), repeat));
    schedule.vxm_at(
        6, config_cycle,
        basic(
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Imm(0.0f), repeat));
    auto tail = basic(
        ftlpu::VxmAluOpcode::Bypass,
        ftlpu::VxmLaneOperand::Previous(),
        ftlpu::VxmLaneOperand::Imm(0.0f), repeat);
    tail.output_type = ftlpu::VxmCastTarget::BFloat16;
    tail.output_stream = ftlpu::VxmLane::fixed_output_stream_for_block(3);
    schedule.vxm_at(7, config_cycle, tail);
}

inline std::size_t build_swiglu_schedule(
    ftlpu::TspSliceSystem& system, Schedule& schedule,
    bool cross_hemisphere)
{
    system.configure_vxm_input_group_source(0, ftlpu::Hemisphere::East);
    system.configure_vxm_input_group_source(1, ftlpu::Hemisphere::East);
    system.configure_vxm_input_group_source(8, ftlpu::Hemisphere::West);
    system.configure_vxm_input_group_source(9, ftlpu::Hemisphere::West);
    system.configure_vxm_output_block_destination(
        3, cross_hemisphere
            ? ftlpu::Hemisphere::West : ftlpu::Hemisphere::East);
    system.configure_vxm_output_block_destination(
        7, cross_hemisphere
            ? ftlpu::Hemisphere::East : ftlpu::Hemisphere::West);

    constexpr auto gate_up_waves =
        kIntermediate / kGateUpColumnsPerWave;
    constexpr auto count = gate_up_waves * kRows;
    constexpr std::size_t config_cycle = 40;
    constexpr std::size_t input_cycle = config_cycle + 1;
    configure_swiglu_program(schedule, config_cycle, count);

    for (std::size_t wave = 0; wave < gate_up_waves; ++wave) {
        for (std::size_t row = 0; row < kRows; ++row) {
            const auto sequence = wave * kRows + row;
            const auto row_in_block = row % kBlockRows;
            for (std::size_t side_index = 0;
                 side_index < ftlpu::hw::kHemispheres; ++side_index) {
                const auto side = hemisphere(side_index);
                const auto input_stream = side_index == 0 ? 0u : 16u;
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    const auto gate_slice =
                        kGateSlices[row_in_block * 2 + byte];
                    const auto up_slice =
                        kUpSlices[row_in_block * 2 + byte];
                    schedule.mem_at(
                        side, gate_slice,
                        input_cycle + sequence
                            - mem_to_vxm_latency(gate_slice),
                        ftlpu::MemInstruction::Read(
                            gate_up_address(wave, row / kBlockRows),
                            ftlpu::StreamId::West(input_stream + byte)));
                    schedule.mem_at(
                        side, up_slice,
                        input_cycle + sequence
                            - mem_to_vxm_latency(up_slice),
                        ftlpu::MemInstruction::Read(
                            gate_up_address(wave, row / kBlockRows),
                            ftlpu::StreamId::West(
                                input_stream + 2 + byte)));
                }

                const auto destination_index = cross_hemisphere
                    ? side_index ^ 1u : side_index;
                const auto destination = hemisphere(destination_index);
                const auto output_stream = side_index == 0 ? 6u : 14u;
                const auto reduction = wave * 2 + side_index;
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    const auto slice =
                        kActivationSlices[row_in_block * 2 + byte];
                    schedule.mem_at(
                        destination, slice,
                        input_cycle + sequence + kVxmSwiGluLatency
                            + vxm_to_mem_latency(slice),
                        ftlpu::MemInstruction::Write(
                            swiglu_address(
                                reduction, row / kBlockRows),
                            ftlpu::StreamId::East(
                                output_stream + byte)));
                }
            }
        }
    }
    return std::max(
        schedule.end_cycle(), input_cycle + count + kVxmSwiGluLatency);
}

inline float read_swiglu(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere side, std::size_t row, std::size_t column)
{
    const auto reduction = column / kTile;
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto row_in_block = row % kBlockRows;
    const auto address = swiglu_address(reduction, row / kBlockRows);
    const auto low = system.read_mem_sram_lane_byte(
        side, kActivationSlices[row_in_block * 2], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        side, kActivationSlices[row_in_block * 2 + 1], tile, address, lane);
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

inline std::size_t build_down_schedule(
    ftlpu::TspSliceSystem& system, Schedule& schedule,
    const QuantizedMatrix& down)
{
    constexpr auto waves =
        (kHidden + kDownColumnsPerWave - 1) / kDownColumnsPerWave;
    constexpr auto reductions = kIntermediate / kTile;
    auto cycle = std::size_t {20};
    for (std::size_t wave = 0; wave < waves; ++wave) {
        for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
            const auto load_start = cycle;
            for (std::size_t column_block = 0;
                 column_block < ftlpu::hw::kMxmSupercellsPerPlane;
                 ++column_block) {
                const auto load_cycle = load_start + column_block;
                for (std::size_t global_mxm = 0;
                     global_mxm < ftlpu::TspSliceSystem::kMxmCount;
                     ++global_mxm) {
                    const auto output_base =
                        wave * kDownColumnsPerWave + global_mxm * kTile;
                    if (output_base >= kHidden) continue;
                    const auto side = hemisphere(global_mxm / 2);
                    const auto local_mxm = global_mxm % 2;
                    const auto address =
                        (wave * reductions + reduction) * 4
                        + column_block;
                    schedule_weight_block(
                        system, schedule, side, local_mxm, down,
                        reduction, output_base, address,
                        column_block, load_cycle);
                }
            }
            const auto compute_start = load_start + kLoadToComputeGap;
            for (std::size_t row_block = 0;
                 row_block < kRowBlocks; ++row_block) {
                const auto compute_cycle =
                    block_compute_cycle(compute_start, row_block);
                schedule_block_activation(
                    schedule, compute_cycle, reduction, row_block,
                    kSwiGluAddressBase);
                for (std::size_t global_mxm = 0;
                     global_mxm < ftlpu::TspSliceSystem::kMxmCount;
                     ++global_mxm) {
                    const auto output_base =
                        wave * kDownColumnsPerWave + global_mxm * kTile;
                    if (output_base >= kHidden) continue;
                    schedule.mxm_compute_at(
                        global_mxm, compute_cycle,
                        block_accumulator_address(wave, row_block),
                        false, 0);
                }
            }
            cycle = compute_start + kComputeSpan
                + kMxmOutputLatency + kComputeToLoadGap;
        }
    }

    auto read_cycle = std::max(cycle, schedule.end_cycle())
        + kMxmOutputLatency + 16;
    for (std::size_t wave = 0; wave < waves; ++wave) {
        for (std::size_t row_block = 0;
             row_block < kRowBlocks; ++row_block) {
            for (std::size_t local_mxm = 0; local_mxm < 2; ++local_mxm) {
                for (std::size_t side_index = 0;
                     side_index < ftlpu::hw::kHemispheres; ++side_index) {
                    const auto global_mxm = side_index * 2 + local_mxm;
                    const auto output_base =
                        wave * kDownColumnsPerWave + global_mxm * kTile;
                    if (output_base >= kHidden) continue;
                    const auto side = hemisphere(side_index);
                    schedule.mxm_accumulator_read_at(
                        global_mxm, read_cycle,
                        wave * kRowBlocks + row_block);
                    const auto output_block = wave * 4 + global_mxm;
                    for (std::size_t stream = 0;
                         stream < kFp32OutputSlices.size(); ++stream) {
                        const auto slice = kFp32OutputSlices[stream];
                        schedule.mem_at(
                            side, slice,
                            read_cycle
                                + west_source_to_mem_latency(slice),
                            ftlpu::MemInstruction::Write(
                                final_block_address(
                                    output_block, row_block),
                                ftlpu::StreamId::West(stream)));
                    }
                }
                ++read_cycle;
            }
        }
    }
    return std::max(read_cycle, schedule.end_cycle());
}

inline float read_down(
    const ftlpu::TspSliceSystem& system,
    std::size_t row, std::size_t column)
{
    const auto wave = column / kDownColumnsPerWave;
    const auto global_mxm =
        (column % kDownColumnsPerWave) / kTile;
    const auto side = hemisphere(global_mxm / 2);
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto row_in_block = row % kBlockRows;
    auto raw = std::uint32_t {0};
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(
            system.read_mem_sram_lane_byte(
                side,
                kFp32OutputSlices[row_in_block * sizeof(float) + byte],
                tile,
                final_block_address(
                    wave * 4 + global_mxm, row / kBlockRows),
                lane)) << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

inline void write_phase_trace(
    const std::filesystem::path& path,
    const std::vector<PhaseSpan>& spans)
{
    if (path.empty()) return;
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write Block8 FFN phase trace");
    }
    output << "start,end,resource,detail\n";
    for (const auto& span : spans) {
        output << span.start << ',' << span.end
               << ",\"Protocol phase\",\""
               << span.label << "\"\n";
    }
}

} // namespace

ftlpu::test::smollm2_layer::PhaseResult
ftlpu::test::smollm2_layer::run_prefill_ffn(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path)
{
    if (input.size() != kRows * ::kHidden) {
        throw std::invalid_argument("Block8 FFN input shape mismatch");
    }
    const auto gate = quantize_group8(
        ::kHidden, ::kIntermediate,
        [](std::size_t row, std::size_t column) {
            return gate_up_weight_value(Projection::Gate, row, column);
        });
    const auto up = quantize_group8(
        ::kHidden, ::kIntermediate,
        [](std::size_t row, std::size_t column) {
            return gate_up_weight_value(Projection::Up, row, column);
        });
    const auto down = quantize_group8(
        ::kIntermediate, ::kHidden, down_weight_value);

    system.reset_execution_state();
    initialize_block_activations(system, input, ::kHidden);
    const auto lut =
        ftlpu::test::w8a16_swiglu::configure_luts(system);
    auto timing = integration_timing::SystemGanttTrace {};
    auto spans = std::vector<PhaseSpan> {};
    auto total_cycles = std::size_t {0};
    const auto run_phase = [&] (
        const Schedule& schedule, std::size_t requested,
        const char* label) {
        const auto cycles = std::max(schedule.end_cycle(), requested) + 32;
        const auto start = total_cycles;
        for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
            try {
                system.tick({});
                if (!trace_path.empty()) {
                    timing.capture(system);
                }
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string(label) + " cycle "
                    + std::to_string(cycle) + ": " + error.what());
            }
        }
        total_cycles += cycles;
        timing.phase(start, total_cycles, label);
        spans.push_back({start, total_cycles, label});
    };

    auto gate_schedule = Schedule(system.icu());
    const auto gate_end =
        build_gate_up_schedule(system, gate_schedule, gate, up);
    run_phase(gate_schedule, gate_end, "Block8 Gate/Up");

    auto expected_swiglu = std::vector<float>(kRows * ::kIntermediate);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < ::kIntermediate; ++column) {
            const auto source_side = hemisphere(
                (column % kGateUpColumnsPerWave) / kTile);
            std::array<float, 2> expected_projection {};
            const std::array<const QuantizedMatrix*, 2> projections {
                &gate, &up};
            for (std::size_t projection = 0; projection < 2; ++projection) {
                for (std::size_t reduction = 0;
                     reduction < ::kHidden; reduction += kTile) {
                    const auto partial = block8_wavefront_dot(
                        [&](std::size_t inner) {
                            return input[
                                row * ::kHidden + reduction + inner];
                        },
                        [&](std::size_t inner) {
                            return projections[projection]->dequantized[
                                projections[projection]->index(
                                    reduction + inner, column)];
                        });
                    expected_projection[projection] += partial;
                }
                const auto expected = ftlpu::Bf16::from_float(
                    expected_projection[projection]);
                const auto actual = read_staged_projection(
                    system, source_side, projection, row, column);
                if (actual.bits() != expected.bits()) {
                    throw std::runtime_error(
                        "Block8 Gate/Up MEM mismatch row="
                        + std::to_string(row) + " column="
                        + std::to_string(column) + " projection="
                        + std::to_string(projection) + " actual="
                        + std::to_string(actual.to_float()) + " expected="
                        + std::to_string(expected.to_float()) + " actual_bits="
                        + std::to_string(actual.bits()) + " expected_bits="
                        + std::to_string(expected.bits()));
                }
                expected_projection[projection] = expected.to_float();
            }
            const auto exponent = lut.execute(
                ftlpu::VxmSpecialAluOpcode::Exp,
                -expected_projection[0]);
            const auto reciprocal = lut.execute(
                ftlpu::VxmSpecialAluOpcode::Reciprocal,
                1.0f + exponent);
            expected_swiglu[row * ::kIntermediate + column] =
                ftlpu::Bf16::from_float(
                    (reciprocal * expected_projection[0])
                    * expected_projection[1]).to_float();
        }
    }

    system.icu().reset();
    auto cross_schedule = Schedule(system.icu());
    const auto cross_end =
        build_swiglu_schedule(system, cross_schedule, true);
    run_phase(cross_schedule, cross_end, "VXM SwiGLU cross");

    system.icu().reset();
    auto local_schedule = Schedule(system.icu());
    const auto local_end =
        build_swiglu_schedule(system, local_schedule, false);
    run_phase(local_schedule, local_end, "VXM SwiGLU local");

    for (std::size_t side_index = 0;
         side_index < ftlpu::hw::kHemispheres; ++side_index) {
        const auto side = hemisphere(side_index);
        for (std::size_t row = 0; row < kRows; ++row) {
            for (std::size_t column = 0;
                 column < ::kIntermediate; ++column) {
                const auto actual = read_swiglu(system, side, row, column);
                const auto expected =
                    expected_swiglu[row * ::kIntermediate + column];
                if (actual != expected
                    && !(std::isnan(actual) && std::isnan(expected))) {
                    throw std::runtime_error(
                        "Block8 VXM MEM mismatch hemisphere="
                        + std::to_string(side_index) + " row="
                        + std::to_string(row) + " column="
                        + std::to_string(column));
                }
            }
        }
    }

    system.icu().reset();
    auto down_schedule = Schedule(system.icu());
    const auto down_end =
        build_down_schedule(system, down_schedule, down);
    run_phase(down_schedule, down_end, "Block8 Down");

    auto output = std::vector<float>(kRows * ::kHidden);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < ::kHidden; ++column) {
            float expected = 0.0f;
            for (std::size_t reduction = 0;
                 reduction < ::kIntermediate; reduction += kTile) {
                const auto partial = block8_wavefront_dot(
                    [&](std::size_t inner) {
                        return expected_swiglu[
                            row * ::kIntermediate + reduction + inner];
                    },
                    [&](std::size_t inner) {
                        return down.dequantized[
                            down.index(reduction + inner, column)];
                    });
                expected += partial;
            }
            const auto actual = read_down(system, row, column);
            output[row * ::kHidden + column] = actual;
            const auto tolerance =
                1.0e-5f * std::max(1.0f, std::fabs(expected));
            if (std::fabs(actual - expected) > tolerance
                && !(std::isnan(actual) && std::isnan(expected))) {
                throw std::runtime_error(
                    "Block8 Down MEM mismatch row="
                    + std::to_string(row) + " column="
                    + std::to_string(column) + " actual="
                    + std::to_string(actual) + " expected="
                    + std::to_string(expected));
            }
        }
    }

    if (!trace_path.empty()) {
        write_phase_trace(trace_path, spans);
        timing.write(
            integration_timing::SystemGanttTrace::prefix_from_name(
                trace_path.stem().string()),
            "SmolLM2 Block8 FFN system timing");
    }
    return {std::move(output), {}, {}, total_cycles};
}

#ifndef FTLPU_SMOLLM2_LAYER_PHASE_ONLY
int main()
try {
    auto input = std::vector<float>(kRows * kHidden);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            input[row * kHidden + column] = ftlpu::Bf16::from_float(
                activation_value(row, column)).to_float();
        }
    }
    auto system = ftlpu::TspSliceSystem {};
    const auto* trace = std::getenv("FTLPU_SCHEDULE_TRACE");
    const auto result = ftlpu::test::smollm2_layer::run_prefill_ffn(
        system, input,
        trace == nullptr ? std::filesystem::path {}
                         : std::filesystem::path {trace});
    std::cout
        << "SmolLM2 Block8 FFN black-box passed: MEM INT8/BF16 -> "
           "Gate/Up MXM Block8 -> MEM BF16 -> VXM SwiGLU -> "
           "dual MEM -> Down MXM Block8 -> AccumulatorRead -> MEM FP32; "
        << "X[" << kRows << ',' << kHidden << "], intermediate="
        << kIntermediate << ", cycles=" << result.cycles << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "SmolLM2 Block8 FFN failed: " << error.what() << '\n';
    return 1;
}
#endif
