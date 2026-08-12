#pragma once

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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::test::w8a16_swiglu {

#if defined(FTLPU_W8A16_SWIGLU_SMOKE)
#ifndef FTLPU_W8A16_SWIGLU_SMOKE_SEQ
#define FTLPU_W8A16_SWIGLU_SMOKE_SEQ 8
#endif
#ifndef FTLPU_W8A16_SWIGLU_SMOKE_HIDDEN
#define FTLPU_W8A16_SWIGLU_SMOKE_HIDDEN 64
#endif
#ifndef FTLPU_W8A16_SWIGLU_SMOKE_INTERMEDIATE
#define FTLPU_W8A16_SWIGLU_SMOKE_INTERMEDIATE 32
#endif
constexpr std::size_t kSeqLen = FTLPU_W8A16_SWIGLU_SMOKE_SEQ;
constexpr std::size_t kHidden = FTLPU_W8A16_SWIGLU_SMOKE_HIDDEN;
constexpr std::size_t kIntermediate =
    FTLPU_W8A16_SWIGLU_SMOKE_INTERMEDIATE;
#else
constexpr std::size_t kSeqLen = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
#endif

constexpr std::size_t kTile = hw::kMxmRows;
constexpr std::size_t kRowsPerHemisphere = kSeqLen / hw::kHemispheres;
constexpr std::size_t kWeightLoadCycles =
    hw::kMxmSupercellsPerPlane * hw::kMxmSupercellColumns;
constexpr std::size_t kMxmVectorOutputLatency =
    hw::kMxmSupercellsPerPlane + Mxm::kLocalMacStages - 2;
constexpr std::size_t kVxmSwiGluLatency = 17;

constexpr std::array<std::size_t, 2> kWeightSlices {0, 8};
constexpr std::array<std::size_t, 4> kActivationSlices {32, 33, 34, 35};
constexpr std::array<std::size_t, 4> kGateUpSlices {36, 37, 38, 39};
constexpr std::array<std::size_t, 4> kSwiGluSlices {32, 33, 34, 35};
constexpr std::array<std::size_t, 8> kDownOutputSlices {
    44, 45, 46, 47, 48, 49, 50, 51};

constexpr std::size_t kGateUpAddress = 4096;
constexpr std::size_t kSwiGluAddress = 16384;

static_assert(kSeqLen % hw::kHemispheres == 0);
static_assert(kHidden % (2 * kTile) == 0);
static_assert(kIntermediate % kTile == 0);

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

inline Hemisphere hemisphere(std::size_t index)
{
    return static_cast<Hemisphere>(index);
}

inline std::size_t mem_queue(Hemisphere side, std::size_t slice)
{
    return InstructionControlUnit::mem_queue(side, slice);
}

inline std::size_t mem_to_mxm_latency(std::size_t slice)
{
    return hw::kMemGroups + 2 - slice / hw::kMemSlicesPerGroup;
}

inline std::size_t mem_to_vxm_latency(std::size_t slice)
{
    return slice / hw::kMemSlicesPerGroup + 2;
}

inline std::size_t west_source_to_mem_latency(std::size_t slice)
{
    return hw::kSystemStreamRegisterColumns - 1
        - slice / hw::kMemSlicesPerGroup;
}

inline std::size_t vxm_to_mem_latency(std::size_t slice)
{
    return slice / hw::kMemSlicesPerGroup + 1;
}

class Schedule {
public:
    explicit Schedule(InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(
        Hemisphere side, std::size_t slice, std::size_t cycle,
        MemInstruction instruction)
    {
        auto& cursor = mem_[mem_queue(side, slice)];
        require_available(cursor, cycle, "MEM");
        icu_.enqueue_mem_nop(mem_queue(side, slice), cycle - cursor);
        icu_.enqueue_mem(
            mem_queue(side, slice), std::move(instruction));
        advance(cursor, cycle + 1);
    }

    void mem_repeat_at(
        Hemisphere side, std::size_t slice, std::size_t cycle,
        MemInstruction instruction, std::size_t count,
        std::int64_t address_stride)
    {
        if (count == 0) return;
        mem_at(side, slice, cycle, std::move(instruction));
        const auto queue = mem_queue(side, slice);
        if (count > 1) {
            icu_.enqueue_mem_repeat(
                queue, count - 1, 1, address_stride);
        }
        advance(mem_[queue], cycle + count);
    }

    void mxm_dequant_at(
        std::size_t mxm, std::size_t cycle, float scale)
    {
        auto& cursor = mxm_dequant_[mxm];
        require_available(cursor, cycle, "MXM dequant");
        icu_.enqueue_mxm_dequant_nop(mxm, cycle - cursor);
        icu_.enqueue_mxm_dequant(
            mxm, MxmDequantInstruction::Scale(scale));
        advance(cursor, cycle + 1);
    }

    void mxm_load_at(
        std::size_t mxm, std::size_t cycle,
        MxmControlInstruction instruction)
    {
        auto& cursor = mxm_load_[mxm];
        require_available(cursor, cycle, "MXM load");
        icu_.enqueue_mxm_load_nop(mxm, cycle - cursor);
        icu_.enqueue_mxm(mxm, instruction);
        advance(cursor, cycle + 1);
    }

    void mxm_compute_repeat_at(
        std::size_t mxm, std::size_t cycle,
        MxmControlInstruction instruction, std::size_t count)
    {
        if (count == 0) return;
        auto& cursor = mxm_compute_[mxm];
        require_available(cursor, cycle, "MXM compute");
        icu_.enqueue_mxm_compute_nop(mxm, cycle - cursor);
        icu_.enqueue_mxm(mxm, instruction);
        if (count > 1) {
            icu_.enqueue_mxm_compute_repeat(mxm, count - 1, 1);
        }
        advance(cursor, cycle + count);
    }

    void mxm_accumulator_read_at(
        std::size_t mxm, std::size_t cycle,
        std::size_t address, std::size_t stream_base)
    {
        auto& cursor = mxm_compute_[mxm];
        require_available(cursor, cycle, "MXM accumulator read");
        icu_.enqueue_mxm_compute_nop(mxm, cycle - cursor);
        icu_.enqueue_mxm(
            mxm,
            MxmControlInstruction::AccumulatorRead(
                address, stream_base, true));
        advance(cursor, cycle + 1);
    }

    void vxm_at(
        std::size_t stage, std::size_t cycle,
        VxmChainDepth depth,
        const VxmLaneAluInstruction& instruction)
    {
        auto& cursor = vxm_[stage];
        require_available(cursor, cycle, "VXM");
        icu_.enqueue_vxm_nop(stage, cycle - cursor);
        icu_.enqueue_vxm(stage, depth, instruction);
        advance(cursor, cycle + 1);
    }

    std::size_t end_cycle() const noexcept { return end_cycle_; }

private:
    static void require_available(
        std::size_t cursor, std::size_t cycle, const char* queue)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                std::string("dual-hemisphere W8A16 schedule overlaps ")
                + queue);
        }
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_cycle_ = std::max(end_cycle_, next);
    }

    InstructionControlUnit& icu_;
    std::array<std::size_t, InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, InstructionControlUnit::kMxmQueues> mxm_load_{};
    std::array<std::size_t, InstructionControlUnit::kMxmQueues> mxm_dequant_{};
    std::array<std::size_t, InstructionControlUnit::kMxmQueues> mxm_compute_{};
    std::array<std::size_t, InstructionControlUnit::kVxmQueues> vxm_{};
    std::size_t end_cycle_{0};
};

inline float activation_value(std::size_t row, std::size_t column)
{
    return static_cast<float>(
        static_cast<int>((row * 7 + column * 5) % 23) - 11)
        * 0.0625f;
}

inline float projection_weight_value(
    Projection projection, std::size_t row, std::size_t column)
{
    const auto p = static_cast<std::size_t>(projection);
    const auto raw = static_cast<int>(
        (row * (11 + p * 6) + column * (5 + p * 2) + p * 13)
        % 41) - 20;
    return static_cast<float>(raw)
        * (0.006f + static_cast<float>((column + p * 3) % 13)
            * 0.001f);
}

inline float down_weight_value(std::size_t row, std::size_t column)
{
    const auto raw = static_cast<int>((row * 13 + column * 3) % 37) - 18;
    return static_cast<float>(raw)
        * (0.008f + static_cast<float>(column % 11) * 0.0015f);
}

template <typename Value>
inline QuantizedMatrix quantize_matrix(
    std::size_t rows, std::size_t columns, Value value)
{
    auto result = QuantizedMatrix {
        rows,
        columns,
        std::vector<float>(columns),
        std::vector<std::int8_t>(rows * columns),
        std::vector<float>(rows * columns),
    };
    for (std::size_t column = 0; column < columns; ++column) {
        float maximum = 0.0f;
        for (std::size_t row = 0; row < rows; ++row) {
            maximum = std::max(maximum, std::fabs(value(row, column)));
        }
        result.scales[column] =
            Bf16::from_float(maximum / 127.0f).to_float();
        for (std::size_t row = 0; row < rows; ++row) {
            const auto q = std::clamp(
                static_cast<int>(std::lround(
                    value(row, column) / result.scales[column])),
                -127, 127);
            result.weights[result.index(row, column)] =
                static_cast<std::int8_t>(q);
            result.dequantized[result.index(row, column)] =
                Bf16::from_float(
                    static_cast<float>(q) * result.scales[column])
                    .to_float();
        }
    }
    return result;
}

inline std::size_t activation_address(
    std::size_t reduction, std::size_t local_row)
{
    return reduction * kRowsPerHemisphere + local_row;
}

inline std::size_t gate_up_address(
    std::size_t column_block, std::size_t local_row)
{
    return kGateUpAddress
        + column_block * kRowsPerHemisphere + local_row;
}

inline std::size_t swiglu_address(
    std::size_t column_block, std::size_t local_row)
{
    return kSwiGluAddress
        + column_block * kRowsPerHemisphere + local_row;
}

inline std::size_t gate_up_accumulator_address(
    std::size_t local_row, std::size_t column_block)
{
    return local_row * (kIntermediate / kTile) + column_block;
}

inline std::size_t down_result_address(
    std::size_t local_row, std::size_t column)
{
    return local_row * (kHidden / kTile) + column / kTile;
}

inline void initialize_activations(
    TspSliceSystem& system, const std::vector<float>& activations)
{
    for (std::size_t side_index = 0;
         side_index < hw::kHemispheres; ++side_index) {
        const auto side = hemisphere(side_index);
        for (std::size_t reduction = 0;
             reduction < kHidden / kTile; ++reduction) {
            for (std::size_t local_row = 0;
                 local_row < kRowsPerHemisphere; ++local_row) {
                const auto row = local_row * hw::kHemispheres + side_index;
                const auto address = activation_address(reduction, local_row);
                for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                    for (std::size_t lane = 0;
                         lane < hw::kLanesPerTile; ++lane) {
                        const auto column = reduction * kTile
                            + tile * hw::kLanesPerTile + lane;
                        const auto bits = Bf16::from_float(
                            activations[row * kHidden + column]).bits();
                        for (std::size_t copy = 0; copy < 2; ++copy) {
                            system.initialize_mem_sram_lane_byte(
                                side, kActivationSlices[copy * 2], tile,
                                address, lane,
                                static_cast<std::uint8_t>(bits & 0xffu));
                            system.initialize_mem_sram_lane_byte(
                                side, kActivationSlices[copy * 2 + 1], tile,
                                address, lane,
                                static_cast<std::uint8_t>(bits >> 8));
                        }
                    }
                }
            }
        }
    }
}

inline void schedule_weight_column(
    TspSliceSystem& system, Schedule& schedule,
    Hemisphere side, std::size_t local_mxm,
    const QuantizedMatrix& weights,
    std::size_t reduction_base, std::size_t output_column,
    std::size_t address, std::size_t load_cycle,
    std::size_t local_column)
{
    const auto slice = kWeightSlices[local_mxm];
    const auto mxm = InstructionControlUnit::mxm_queue(side, local_mxm);
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto row = reduction_base
                + tile * hw::kLanesPerTile + lane;
            system.initialize_mem_sram_lane_byte(
                side, slice, tile, address, lane,
                static_cast<std::uint8_t>(
                    weights.weights[weights.index(row, output_column)]));
        }
    }
    schedule.mem_at(
        side, slice, load_cycle - mem_to_mxm_latency(slice),
        MemInstruction::Read(
            address,
            StreamId::East(
                local_mxm * hw::kMxmInt8LoadStreamStride)));
    schedule.mxm_dequant_at(
        mxm, load_cycle, weights.scales[output_column]);
    schedule.mxm_load_at(
        mxm, load_cycle,
        MxmControlInstruction::IWColumn(
            0,
            local_column / hw::kMxmSupercellColumns,
            local_column % hw::kMxmSupercellColumns));
}

inline void schedule_activation_read(
    Schedule& schedule, Hemisphere side,
    std::size_t address, std::size_t compute_cycle,
    std::size_t count)
{
    for (std::size_t byte = 0; byte < kActivationSlices.size(); ++byte) {
        const auto slice = kActivationSlices[byte];
        schedule.mem_repeat_at(
            side, slice, compute_cycle - mem_to_mxm_latency(slice),
            MemInstruction::Read(address, StreamId::East(byte)),
            count, 1);
    }
}

inline void schedule_gate_up_capture(
    Schedule& schedule, Hemisphere side,
    std::size_t column_block, std::size_t local_row,
    std::size_t output_cycle)
{
    for (std::size_t projection = 0; projection < 2; ++projection) {
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kGateUpSlices[projection * 2 + byte];
            schedule.mem_at(
                side, slice,
                output_cycle + west_source_to_mem_latency(slice),
                MemInstruction::Write(
                    gate_up_address(column_block, local_row),
                    StreamId::West(projection * 4 + byte)));
        }
    }
}

inline std::size_t schedule_gate_up(
    TspSliceSystem& system, Schedule& schedule,
    const QuantizedMatrix& gate, const QuantizedMatrix& up)
{
    const std::array<const QuantizedMatrix*, 2> projections {&gate, &up};
    std::size_t phase_start = 0;
    std::size_t weight_address = 0;
    for (std::size_t column_base = 0;
         column_base < kIntermediate; column_base += kTile) {
        const auto column_block = column_base / kTile;
        for (std::size_t reduction = 0;
             reduction < kHidden / kTile; ++reduction) {
            const auto load_start = phase_start + 20;
            for (std::size_t local_column = 0;
                 local_column < kTile; ++local_column) {
                const auto load_cycle = load_start + local_column;
                for (std::size_t side_index = 0;
                     side_index < hw::kHemispheres; ++side_index) {
                    const auto side = hemisphere(side_index);
                    for (std::size_t local_mxm = 0;
                         local_mxm < 2; ++local_mxm) {
                        schedule_weight_column(
                            system, schedule, side, local_mxm,
                            *projections[local_mxm], reduction * kTile,
                            column_base + local_column,
                            weight_address, load_cycle, local_column);
                    }
                }
                ++weight_address;
            }

            const auto first_compute =
                load_start + kWeightLoadCycles + 4;
            const auto final_reduction =
                reduction + 1 == kHidden / kTile;
            for (std::size_t side_index = 0;
                 side_index < hw::kHemispheres; ++side_index) {
                const auto side = hemisphere(side_index);
                schedule_activation_read(
                    schedule, side,
                    activation_address(reduction, 0),
                    first_compute, kRowsPerHemisphere);
                for (std::size_t local_mxm = 0;
                     local_mxm < 2; ++local_mxm) {
                    const auto mxm =
                        InstructionControlUnit::mxm_queue(side, local_mxm);
                    for (std::size_t row_base = 0;
                         row_base < kRowsPerHemisphere;
                         row_base += kTile) {
                        const auto rows = std::min(
                            kTile, kRowsPerHemisphere - row_base);
                        schedule.mxm_compute_repeat_at(
                            mxm, first_compute + row_base,
                            MxmControlInstruction::Compute(
                                0, local_mxm * 2, local_mxm * 4,
                                gate_up_accumulator_address(
                                    row_base, column_block),
                                kIntermediate / kTile,
                                final_reduction
                                    ? MxmAccumulatorDestination::Stream
                                    : MxmAccumulatorDestination::Sram,
                                MxmDataFormat::BFloat16,
                                MxmComputeMode::Vector,
                                final_reduction,
                                MxmAccumulatorOutputFormat::BFloat16),
                            rows);
                    }
                }
                if (final_reduction) {
                    for (std::size_t local_row = 0;
                         local_row < kRowsPerHemisphere; ++local_row) {
                        schedule_gate_up_capture(
                            schedule, side, column_block, local_row,
                            first_compute + local_row
                                + kMxmVectorOutputLatency);
                    }
                }
            }
            phase_start = first_compute + kRowsPerHemisphere
                + kMxmVectorOutputLatency + 8;
        }
    }
    return phase_start;
}

template <typename Fn>
inline std::vector<VxmLutEntry> make_table(
    float input_min, float segment_width,
    std::size_t count, Fn fn)
{
    auto entries = std::vector<VxmLutEntry>{};
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto x0 = input_min
            + static_cast<float>(index) * segment_width;
        const auto y0 = fn(x0);
        entries.push_back(VxmLutEntry::from_float(
            (fn(x0 + segment_width) - y0) / segment_width, y0));
    }
    return entries;
}

inline VxmSpecialAlu configure_luts(TspSliceSystem& system)
{
    constexpr std::size_t kEntries = 256;
    constexpr float kLn2 = 0.6931471805599453f;
    auto reference = VxmSpecialAlu{};
    const auto configure = [&] (
        VxmSpecialAluOpcode opcode, VxmLutConfig config,
        const std::vector<VxmLutEntry>& entries) {
        system.initialize_vxm_lut(opcode, config, entries);
        reference.configure_lut(opcode, config, entries);
    };
    configure(
        VxmSpecialAluOpcode::Exp,
        {-kLn2 / 2.0f, kLn2 / static_cast<float>(kEntries)},
        make_table(
            -kLn2 / 2.0f,
            kLn2 / static_cast<float>(kEntries), kEntries,
            [](float x) { return std::exp(x); }));
    configure(
        VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f / static_cast<float>(kEntries)},
        make_table(
            1.0f, 1.0f / static_cast<float>(kEntries), kEntries,
            [](float x) { return 1.0f / x; }));
    return reference;
}

inline VxmLaneAluInstruction basic(
    VxmAluOpcode opcode, VxmLaneOperand lhs,
    VxmLaneOperand rhs = VxmLaneOperand::Imm(0.0f),
    std::size_t repeat = 1)
{
    auto instruction = VxmLaneAluInstruction {opcode, lhs, rhs};
    instruction.precision = VxmAluPrecision::Float32;
    instruction.repeat_count = repeat;
    return instruction;
}

inline VxmLaneAluInstruction special(
    VxmSpecialAluOpcode opcode, VxmLaneOperand lhs,
    std::size_t repeat)
{
    auto instruction = VxmLaneAluInstruction {opcode, lhs};
    instruction.repeat_count = repeat;
    return instruction;
}

inline std::size_t schedule_swiglu(
    TspSliceSystem& system, Schedule& schedule,
    std::size_t phase_start)
{
    // Chain C0..C7 consumes East.  Its mirrored C8..C15 chain consumes West.
    // Each result crosses the VXM and is stored in the opposite hemisphere,
    // where the local Down MXMs consume it.
    system.configure_vxm_input_group_source(0, Hemisphere::East);
    system.configure_vxm_input_group_source(1, Hemisphere::East);
    system.configure_vxm_input_group_source(8, Hemisphere::West);
    system.configure_vxm_input_group_source(9, Hemisphere::West);
    system.configure_vxm_output_block_destination(3, Hemisphere::West);
    system.configure_vxm_output_block_destination(7, Hemisphere::East);

    const auto config_cycle = phase_start + 40;
    const auto input_cycle = config_cycle + 1;
    const auto waves =
        kRowsPerHemisphere * (kIntermediate / kTile);
    schedule.vxm_at(
        0, config_cycle, VxmChainDepth::Eight,
        basic(
            VxmAluOpcode::Negate,
            VxmLaneOperand::StreamBFloat16(),
            VxmLaneOperand::StreamBFloat16(), waves));
    schedule.vxm_at(
        1, config_cycle, VxmChainDepth::Eight,
        special(
            VxmSpecialAluOpcode::Exp,
            VxmLaneOperand::Previous(), waves));
    schedule.vxm_at(
        2, config_cycle, VxmChainDepth::Eight,
        basic(
            VxmAluOpcode::Add, VxmLaneOperand::Previous(),
            VxmLaneOperand::Imm(1.0f), waves));
    schedule.vxm_at(
        3, config_cycle, VxmChainDepth::Eight,
        special(
            VxmSpecialAluOpcode::Reciprocal,
            VxmLaneOperand::Previous(), waves));
    schedule.vxm_at(
        4, config_cycle, VxmChainDepth::Eight,
        basic(
            VxmAluOpcode::Multiply, VxmLaneOperand::Previous(),
            VxmLaneOperand::Original(), waves));
    schedule.vxm_at(
        5, config_cycle, VxmChainDepth::Eight,
        basic(
            VxmAluOpcode::Multiply, VxmLaneOperand::Previous(),
            VxmLaneOperand::Aux(), waves));
    schedule.vxm_at(
        6, config_cycle, VxmChainDepth::Eight,
        basic(
            VxmAluOpcode::Bypass, VxmLaneOperand::Previous(),
            VxmLaneOperand::Imm(0.0f), waves));
    auto tail = basic(
        VxmAluOpcode::Bypass, VxmLaneOperand::Previous(),
        VxmLaneOperand::Imm(0.0f), waves);
    tail.output_type = VxmCastTarget::BFloat16;
    tail.output_stream = VxmLane::fixed_output_stream_for_block(3);
    schedule.vxm_at(
        7, config_cycle, VxmChainDepth::Eight, tail);

    for (std::size_t side_index = 0;
         side_index < hw::kHemispheres; ++side_index) {
        const auto side = hemisphere(side_index);
        const auto stream_base = side == Hemisphere::East ? 0u : 16u;
        for (std::size_t byte = 0; byte < kGateUpSlices.size(); ++byte) {
            const auto slice = kGateUpSlices[byte];
            schedule.mem_repeat_at(
                side, slice, input_cycle - mem_to_vxm_latency(slice),
                MemInstruction::Read(
                    gate_up_address(0, 0),
                    StreamId::West(stream_base + byte)),
                waves, 1);
        }
    }

    const auto output_cycle = input_cycle + kVxmSwiGluLatency;
    for (std::size_t destination_index = 0;
         destination_index < hw::kHemispheres; ++destination_index) {
        const auto destination = hemisphere(destination_index);
        const auto output_stream = destination == Hemisphere::West ? 6u : 14u;
        for (std::size_t copy = 0; copy < 2; ++copy) {
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kSwiGluSlices[copy * 2 + byte];
                schedule.mem_repeat_at(
                    destination, slice,
                    output_cycle + vxm_to_mem_latency(slice),
                    MemInstruction::Write(
                        swiglu_address(0, 0),
                        StreamId::East(output_stream + byte)),
                    waves, 1);
            }
        }
    }
    return output_cycle + waves;
}

inline std::size_t schedule_down(
    TspSliceSystem& system, Schedule& schedule,
    const QuantizedMatrix& down, std::size_t phase_start)
{
    // Down is dispatched after the Gate/Up+SwiGLU phase has completed, so
    // its weight image may reuse the phase-local MEM weight address range.
    auto weight_address = std::size_t {0};
    for (std::size_t column_base = 0;
         column_base < kHidden; column_base += 2 * kTile) {
        for (std::size_t reduction = 0;
             reduction < kIntermediate / kTile; ++reduction) {
            const auto load_start = phase_start + 20;
            for (std::size_t local_column = 0;
                 local_column < kTile; ++local_column) {
                const auto load_cycle = load_start + local_column;
                for (std::size_t side_index = 0;
                     side_index < hw::kHemispheres; ++side_index) {
                    const auto side = hemisphere(side_index);
                    for (std::size_t local_mxm = 0;
                         local_mxm < 2; ++local_mxm) {
                        schedule_weight_column(
                            system, schedule, side, local_mxm, down,
                            reduction * kTile,
                            column_base + local_mxm * kTile + local_column,
                            weight_address, load_cycle, local_column);
                    }
                }
                ++weight_address;
            }
            const auto first_compute =
                load_start + kWeightLoadCycles + 4;
            for (std::size_t side_index = 0;
                 side_index < hw::kHemispheres; ++side_index) {
                const auto side = hemisphere(side_index);
                schedule_activation_read(
                    schedule, side, swiglu_address(reduction, 0),
                    first_compute, kRowsPerHemisphere);
                for (std::size_t local_mxm = 0;
                     local_mxm < 2; ++local_mxm) {
                    const auto mxm =
                        InstructionControlUnit::mxm_queue(side, local_mxm);
                    for (std::size_t row_base = 0;
                         row_base < kRowsPerHemisphere;
                         row_base += kTile) {
                        const auto rows = std::min(
                            kTile, kRowsPerHemisphere - row_base);
                        schedule.mxm_compute_repeat_at(
                            mxm, first_compute + row_base,
                            MxmControlInstruction::Compute(
                                0, local_mxm * 2, 0,
                                down_result_address(
                                    row_base,
                                    column_base + local_mxm * kTile),
                                kHidden / kTile,
                                MxmAccumulatorDestination::Sram,
                                MxmDataFormat::BFloat16),
                            rows);
                    }
                }
            }
            phase_start = first_compute + kRowsPerHemisphere
                + kMxmVectorOutputLatency + 8;
        }
    }

    const auto read_start = phase_start + 8;
    const auto block_pairs = kHidden / (2 * kTile);
    for (std::size_t local_row = 0;
         local_row < kRowsPerHemisphere; ++local_row) {
        for (std::size_t pair = 0; pair < block_pairs; ++pair) {
            const auto read_cycle =
                read_start + local_row * block_pairs + pair;
            for (std::size_t side_index = 0;
                 side_index < hw::kHemispheres; ++side_index) {
                const auto side = hemisphere(side_index);
                for (std::size_t local_mxm = 0;
                     local_mxm < 2; ++local_mxm) {
                    const auto mxm =
                        InstructionControlUnit::mxm_queue(side, local_mxm);
                    const auto column =
                        (pair * 2 + local_mxm) * kTile;
                    const auto address =
                        down_result_address(local_row, column);
                    const auto stream_base = local_mxm * sizeof(float);
                    schedule.mxm_accumulator_read_at(
                        mxm, read_cycle, address, stream_base);
                    for (std::size_t byte = 0;
                         byte < sizeof(float); ++byte) {
                        const auto slice = kDownOutputSlices[
                            local_mxm * sizeof(float) + byte];
                        schedule.mem_at(
                            side, slice,
                            read_cycle
                                + west_source_to_mem_latency(slice),
                            MemInstruction::Write(
                                address,
                                StreamId::West(stream_base + byte)));
                    }
                }
            }
        }
    }
    return read_start + kRowsPerHemisphere * block_pairs;
}

inline Hemisphere result_hemisphere(std::size_t global_row)
{
    return global_row % 2 == 0
        ? Hemisphere::West : Hemisphere::East;
}

inline Bf16 read_swiglu(
    const TspSliceSystem& system,
    std::size_t row, std::size_t column)
{
    const auto side = result_hemisphere(row);
    const auto local_row = row / hw::kHemispheres;
    const auto tile = (column % kTile) / hw::kLanesPerTile;
    const auto lane = (column % kTile) % hw::kLanesPerTile;
    const auto address = swiglu_address(column / kTile, local_row);
    const auto low = system.read_mem_sram_lane_byte(
        side, kSwiGluSlices[0], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        side, kSwiGluSlices[1], tile, address, lane);
    return Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8));
}

inline float read_down(
    const TspSliceSystem& system,
    std::size_t row, std::size_t column)
{
    const auto side = result_hemisphere(row);
    const auto local_row = row / hw::kHemispheres;
    const auto local_mxm = (column / kTile) % 2;
    const auto tile = (column % kTile) / hw::kLanesPerTile;
    const auto lane = (column % kTile) % hw::kLanesPerTile;
    auto raw = std::uint32_t {0};
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(
            system.read_mem_sram_lane_byte(
                side,
                kDownOutputSlices[local_mxm * sizeof(float) + byte],
                tile, down_result_address(local_row, column), lane))
            << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

inline bool verify_swiglu(
    const TspSliceSystem& system,
    const std::vector<float>& activations,
    const QuantizedMatrix& gate,
    const QuantizedMatrix& up,
    const VxmSpecialAlu& lut,
    std::vector<float>& expected_swiglu)
{
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0;
             column < kIntermediate; ++column) {
            std::array<float, 2> projected {};
            for (std::size_t reduction = 0;
                 reduction < kHidden; reduction += kTile) {
                std::array<float, 2> partial {};
                for (std::size_t inner = 0; inner < kTile; ++inner) {
                    const auto x = activations[
                        row * kHidden + reduction + inner];
                    partial[0] += x * gate.dequantized[
                        gate.index(reduction + inner, column)];
                    partial[1] += x * up.dequantized[
                        up.index(reduction + inner, column)];
                }
                projected[0] += partial[0];
                projected[1] += partial[1];
            }
            const auto gate_bf16 =
                Bf16::from_float(projected[0]).to_float();
            const auto up_bf16 =
                Bf16::from_float(projected[1]).to_float();
            const auto exponent = lut.execute(
                VxmSpecialAluOpcode::Exp, -gate_bf16);
            const auto reciprocal = lut.execute(
                VxmSpecialAluOpcode::Reciprocal, 1.0f + exponent);
            const auto expected = Bf16::from_float(
                (reciprocal * gate_bf16) * up_bf16);
            expected_swiglu[row * kIntermediate + column] =
                expected.to_float();
            const auto actual = read_swiglu(system, row, column);
            if (actual.bits() != expected.bits()) {
                const auto source_side = hemisphere(row % hw::kHemispheres);
                const auto local_row = row / hw::kHemispheres;
                const auto tile = (column % kTile) / hw::kLanesPerTile;
                const auto lane = (column % kTile) % hw::kLanesPerTile;
                const auto address = gate_up_address(
                    column / kTile, local_row);
                const auto staged = [&] (std::size_t projection) {
                    const auto low = system.read_mem_sram_lane_byte(
                        source_side, kGateUpSlices[projection * 2],
                        tile, address, lane);
                    const auto high = system.read_mem_sram_lane_byte(
                        source_side, kGateUpSlices[projection * 2 + 1],
                        tile, address, lane);
                    return Bf16::from_bits(
                        static_cast<std::uint16_t>(low)
                        | (static_cast<std::uint16_t>(high) << 8));
                };
                std::cerr
                    << "dual-hemisphere SwiGLU mismatch row=" << row
                    << " column=" << column
                    << " gate=" << gate_bf16
                    << " up=" << up_bf16
                    << " staged_gate=" << staged(0).to_float()
                    << " staged_up=" << staged(1).to_float()
                    << " actual=" << actual.to_float()
                    << " expected=" << expected.to_float() << '\n';
                return false;
            }
        }
    }
    return true;
}

inline bool verify_down(
    const TspSliceSystem& system,
    const std::vector<float>& swiglu,
    const QuantizedMatrix& down)
{
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            float expected = 0.0f;
            for (std::size_t reduction = 0;
                 reduction < kIntermediate; reduction += kTile) {
                float partial = 0.0f;
                for (std::size_t inner = 0; inner < kTile; ++inner) {
                    partial += swiglu[
                        row * kIntermediate + reduction + inner]
                        * down.dequantized[
                            down.index(reduction + inner, column)];
                }
                expected += partial;
            }
            const auto actual = read_down(system, row, column);
            if (std::fabs(actual - expected) > 1.0e-5f) {
                std::cerr
                    << "dual-hemisphere Down mismatch row=" << row
                    << " column=" << column
                    << " actual=" << actual
                    << " expected=" << expected << '\n';
                return false;
            }
        }
    }
    return true;
}

inline int run(bool include_down)
try {
    auto activations = std::vector<float>(kSeqLen * kHidden);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            activations[row * kHidden + column] =
                Bf16::from_float(
                    activation_value(row, column)).to_float();
        }
    }
    const auto gate = quantize_matrix(
        kHidden, kIntermediate,
        [](std::size_t row, std::size_t column) {
            return projection_weight_value(
                Projection::Gate, row, column);
        });
    const auto up = quantize_matrix(
        kHidden, kIntermediate,
        [](std::size_t row, std::size_t column) {
            return projection_weight_value(
                Projection::Up, row, column);
        });
    auto down = QuantizedMatrix {};
    if (include_down) {
        down = quantize_matrix(
            kIntermediate, kHidden, down_weight_value);
    }

    auto system = std::make_unique<TspSliceSystem>();
    initialize_activations(*system, activations);
    const auto reference_lut = configure_luts(*system);
    const auto run_phase = [&] (
        const Schedule& schedule, std::size_t scheduled_end,
        const char* phase) {
        const auto run_cycles =
            std::max(schedule.end_cycle(), scheduled_end) + 32;
        for (std::size_t cycle = 0; cycle < run_cycles; ++cycle) {
            try {
                system->tick({});
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string(phase) + " cycle "
                    + std::to_string(cycle) + ": " + error.what());
            }
        }
    };

    auto front_schedule = Schedule(system->icu());
    const auto gate_up_end = schedule_gate_up(
        *system, front_schedule, gate, up);
    const auto swiglu_end = schedule_swiglu(
        *system, front_schedule, gate_up_end);
    run_phase(front_schedule, swiglu_end, "Gate/Up+SwiGLU");

    auto expected_swiglu =
        std::vector<float>(kSeqLen * kIntermediate);
    if (!verify_swiglu(
            *system, activations, gate, up,
            reference_lut, expected_swiglu)) {
        return 1;
    }
    if (include_down) {
        // MEM is the architectural phase boundary.  Reset only the ICU
        // program/issue state; SRAM data and all hardware units remain live.
        system->icu().reset();
        auto down_schedule = Schedule(system->icu());
        const auto down_end = schedule_down(
            *system, down_schedule, down, 0);
        run_phase(down_schedule, down_end, "Down");
        if (!verify_down(*system, expected_swiglu, down)) {
            return 1;
        }
    }

    std::cout
        << "dual-hemisphere W8A16 "
        << (include_down ? "FFN" : "SwiGLU")
        << " black-box passed: hemisphere-local Gate/Up MXM -> MEM -> "
           "two VXM depth-8 chains -> cross-hemisphere MEM";
    if (include_down) {
        std::cout
            << " -> opposite-hemisphere Down MXMs -> "
               "AccumulatorRead -> MEM FP32";
    }
    std::cout
        << ", X[" << kSeqLen << ',' << kHidden << "]"
        << ", intermediate=" << kIntermediate << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr
        << "dual-hemisphere W8A16 "
        << (include_down ? "FFN" : "SwiGLU")
        << " black-box exception: " << error.what() << '\n';
    return 1;
}

} // namespace ftlpu::test::w8a16_swiglu
