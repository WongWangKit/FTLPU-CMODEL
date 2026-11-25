#include "ftlpu/core/bf16.hpp"
#include "ftlpu/mxm/bf16_arithmetic.hpp"
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

namespace {

constexpr std::array<std::size_t, 16> kWeightSlices {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
};
constexpr std::array<std::size_t, 2> kActivationSlices {50, 51};
constexpr std::size_t kWeightAddress = 64;
constexpr std::size_t kQuantizedWeightAddress = 96;
constexpr std::size_t kActivationAddress = 80;
constexpr std::size_t kAccumulatorAddress =
    ftlpu::hw::kMxmAccumulatorRows / 2;
constexpr std::size_t kLoadStart = 20;
constexpr std::size_t kComputeCycle = 30;

std::size_t east_read_to_mxm_latency(std::size_t slice)
{
    // RegisteredR1 is the production default: one registered MEM-read cycle
    // plus the forward route from the slice boundary to sxm_mxm.
    return 1 + ftlpu::hw::kMemGroups
        + ftlpu::hw::kC2cToSxmStreamRegisterColumns + 1
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mem_queue(std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(
        ftlpu::Hemisphere::East, slice);
}

float weight_value(std::size_t row, std::size_t column)
{
    const auto signed_pattern =
        static_cast<int>((row * 5 + column * 3) % 23) - 11;
    return static_cast<float>(signed_pattern) * 0.019f + 1.00390625f;
}

float activation_value(std::size_t row)
{
    const auto signed_pattern = static_cast<int>((row * 7) % 17) - 8;
    return static_cast<float>(signed_pattern) * 0.031f - 0.50390625f;
}

class Schedule {
public:
    explicit Schedule(ftlpu::InstructionControlUnit& icu)
        : icu_(icu)
    {
    }

    void mem_at(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        auto& cursor = mem_[mem_queue(slice)];
        if (cycle < cursor) {
            throw std::logic_error("BF16 MXM test overlaps a MEM queue");
        }
        icu_.enqueue_mem_nop(mem_queue(slice), cycle - cursor);
        icu_.enqueue_mem(mem_queue(slice), instruction);
        cursor = cycle + 1;
        end_cycle_ = std::max(end_cycle_, cursor);
    }

    void mxm_load_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction)
    {
        if (cycle == 0) {
            throw std::logic_error("BF16 MXM load cannot execute before row0 is registered");
        }
        const auto issue_cycle = cycle - 1;
        if (issue_cycle < mxm_load_) {
            throw std::logic_error("BF16 MXM test overlaps the load queue");
        }
        icu_.enqueue_mxm_load_nop(0, issue_cycle - mxm_load_);
        icu_.enqueue_mxm(0, instruction);
        mxm_load_ = issue_cycle + 1;
        end_cycle_ = std::max(end_cycle_, cycle + 1);
    }

    void mxm_compute_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction)
    {
        if (cycle == 0) {
            throw std::logic_error("BF16 MXM compute cannot execute before row0 is registered");
        }
        const auto issue_cycle = cycle - 1;
        if (issue_cycle < mxm_compute_) {
            throw std::logic_error("BF16 MXM test overlaps the compute queue");
        }
        icu_.enqueue_mxm_compute_nop(0, issue_cycle - mxm_compute_);
        icu_.enqueue_mxm(0, instruction);
        mxm_compute_ = issue_cycle + 1;
        end_cycle_ = std::max(end_cycle_, cycle + 1);
    }

    void mxm_dequant_at(
        std::size_t cycle,
        ftlpu::MxmDequantInstruction instruction)
    {
        if (cycle == 0) {
            throw std::logic_error("BF16 MXM dequant cannot execute before row0 is registered");
        }
        const auto issue_cycle = cycle - 1;
        if (issue_cycle < mxm_dequant_) {
            throw std::logic_error(
                "BF16 MXM test overlaps the dequant queue");
        }
        icu_.enqueue_mxm_dequant_nop(
            0,
            issue_cycle - mxm_dequant_);
        icu_.enqueue_mxm_dequant(0, instruction);
        mxm_dequant_ = issue_cycle + 1;
        end_cycle_ = std::max(end_cycle_, cycle + 1);
    }

    void mem_repeat_at(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction,
        std::size_t count,
        std::int64_t address_stride)
    {
        mem_at(slice, cycle, instruction);
        const auto queue = mem_queue(slice);
        if (count > 1) {
            icu_.enqueue_mem_repeat(
                queue, count - 1, 1, address_stride);
        }
        mem_[queue] = cycle + count;
        end_cycle_ = std::max(end_cycle_, cycle + count);
    }

    void mxm_compute_repeat_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction,
        std::size_t count)
    {
        mxm_compute_at(cycle, instruction);
        if (count > 1) {
            icu_.enqueue_mxm_compute_repeat(0, count - 1, 1);
        }
        mxm_compute_ = cycle + count - 1;
        end_cycle_ = std::max(end_cycle_, cycle + count);
    }

    std::size_t end_cycle() const
    {
        return end_cycle_;
    }

private:
    ftlpu::InstructionControlUnit& icu_;
    std::array<
        std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::size_t mxm_load_{0};
    std::size_t mxm_dequant_{0};
    std::size_t mxm_compute_{0};
    std::size_t end_cycle_{0};
};

void initialize_inputs(ftlpu::TspSliceSystem& system)
{
    for (std::size_t block = 0;
         block < ftlpu::hw::kMxmSupercellsPerPlane;
         ++block) {
        for (std::size_t tile = 0;
             tile < ftlpu::hw::kTileRows;
             ++tile) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile;
                 ++lane) {
                const auto row =
                    tile * ftlpu::hw::kLanesPerTile + lane;
                for (std::size_t local_column = 0;
                     local_column < ftlpu::hw::kMxmSupercellColumns;
                     ++local_column) {
                    const auto column =
                        block * ftlpu::hw::kMxmSupercellColumns
                        + local_column;
                    const auto bits = ftlpu::Bf16::from_float(
                        weight_value(row, column)).bits();
                    system.initialize_mem_sram_lane_byte(
                        kWeightSlices[local_column * 2],
                        tile,
                        kWeightAddress + block,
                        lane,
                        static_cast<std::uint8_t>(bits & 0xffu));
                    system.initialize_mem_sram_lane_byte(
                        kWeightSlices[local_column * 2 + 1],
                        tile,
                        kWeightAddress + block,
                        lane,
                        static_cast<std::uint8_t>(bits >> 8));
                }
            }
        }
    }

    for (std::size_t tile = 0;
         tile < ftlpu::hw::kTileRows;
         ++tile) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            const auto row =
                tile * ftlpu::hw::kLanesPerTile + lane;
            const auto bits = ftlpu::Bf16::from_float(
                activation_value(row)).bits();
            system.initialize_mem_sram_lane_byte(
                kActivationSlices[0],
                tile,
                kActivationAddress,
                lane,
                static_cast<std::uint8_t>(bits & 0xffu));
            system.initialize_mem_sram_lane_byte(
                kActivationSlices[1],
                tile,
                kActivationAddress,
                lane,
                static_cast<std::uint8_t>(bits >> 8));
        }
    }
}

std::int8_t quantized_weight(std::size_t row, std::size_t column)
{
    return static_cast<std::int8_t>(
        static_cast<int>((row * 5 + column * 3) % 31) - 15);
}

void initialize_quantized_weights(ftlpu::TspSliceSystem& system)
{
    for (std::size_t tile = 0;
         tile < ftlpu::hw::kTileRows;
         ++tile) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            const auto row =
                tile * ftlpu::hw::kLanesPerTile + lane;
            for (std::size_t column = 0;
                 column < ftlpu::hw::kMxmSupercellColumns;
                 ++column) {
                system.initialize_mem_sram_lane_byte(
                    kWeightSlices[column],
                    tile,
                    kQuantizedWeightAddress,
                    lane,
                    static_cast<std::uint8_t>(
                        quantized_weight(row, column)));
            }
        }
    }
}

float reference(std::size_t column)
{
    float sum = 0.0f;
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        auto activations = std::array<
            std::uint16_t, ftlpu::hw::kLanesPerTile> {};
        auto weights = std::array<
            std::uint16_t, ftlpu::hw::kLanesPerTile> {};
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            const auto row = tile * ftlpu::hw::kLanesPerTile + lane;
            activations[lane] = ftlpu::Bf16::from_float(
                activation_value(row)).bits();
            weights[lane] = ftlpu::Bf16::from_float(
                weight_value(row, column)).bits();
        }
        sum = ftlpu::mxm_numeric::fp32_add(
            sum, ftlpu::mxm_numeric::bf16_dot(
                activations, weights).value());
    }
    return sum;
}

} // namespace

int main()
{
    auto quantized_system =
        std::make_unique<ftlpu::TspSliceSystem>();
    initialize_quantized_weights(*quantized_system);
    auto quantized_schedule = Schedule(quantized_system->icu());
    constexpr float kDequantScale = 0.125f;
    for (std::size_t stream = 0;
         stream < ftlpu::hw::kMxmInt8LoadStreamsPerCycle;
         ++stream) {
        const auto slice = kWeightSlices[stream];
        quantized_schedule.mem_at(
            slice,
            kLoadStart - east_read_to_mxm_latency(slice),
            ftlpu::MemInstruction::Read(
                kQuantizedWeightAddress,
                ftlpu::StreamId::East(stream)));
    }
    quantized_schedule.mxm_dequant_at(
        kLoadStart,
        ftlpu::MxmDequantInstruction::Scale(kDequantScale));
    quantized_schedule.mxm_load_at(
        kLoadStart,
        ftlpu::MxmControlInstruction::IW(0, 0));
    for (std::size_t cycle = 0;
         cycle < quantized_schedule.end_cycle() + 8;
         ++cycle) {
        quantized_system->tick({});
    }
    for (std::size_t tile = 0;
         tile < ftlpu::hw::kTileRows;
         ++tile) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            const auto row =
                tile * ftlpu::hw::kLanesPerTile + lane;
            for (std::size_t column = 0;
                 column < ftlpu::hw::kMxmSupercellColumns;
                 ++column) {
                const auto expected = ftlpu::Bf16::from_float(
                    static_cast<float>(
                        quantized_weight(row, column))
                    * kDequantScale).bits();
                const auto actual = quantized_system->mxm_unit(0)
                    .array()
                    .cell(tile, 0)
                    .weight_bits(0, lane, column);
                if (actual != expected) {
                    std::cerr
                        << "MXM INT8 dequant load mismatch at tile="
                        << tile << " lane=" << lane
                        << " column=" << column << '\n';
                    return 1;
                }
            }
        }
    }

    // TspSliceSystem owns the complete local SRAM/FU state and is larger than
    // the default Windows thread stack. Keep both subsystem instances on heap.
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    initialize_inputs(*system);
    auto schedule = Schedule(system->icu());

    for (std::size_t block = 0;
         block < ftlpu::hw::kMxmSupercellsPerPlane;
         ++block) {
        const auto iw_cycle = kLoadStart + block;
        for (std::size_t stream = 0;
             stream < kWeightSlices.size();
             ++stream) {
            const auto slice = kWeightSlices[stream];
            schedule.mem_at(
                slice,
                iw_cycle - east_read_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    kWeightAddress + block,
                    ftlpu::StreamId::East(stream)));
        }
        schedule.mxm_load_at(
            iw_cycle,
            ftlpu::MxmControlInstruction::IWDirect16(0, block));
    }

    for (std::size_t byte = 0;
         byte < kActivationSlices.size();
         ++byte) {
        const auto slice = kActivationSlices[byte];
        schedule.mem_at(
            slice,
            kComputeCycle - east_read_to_mxm_latency(slice),
            ftlpu::MemInstruction::Read(
                kActivationAddress,
                ftlpu::StreamId::East(byte)));
    }
    schedule.mxm_compute_at(
        kComputeCycle,
        ftlpu::MxmControlInstruction::Compute(
            0,
            0,
            0,
            kAccumulatorAddress,
            1,
            ftlpu::MxmAccumulatorDestination::Sram,
            ftlpu::MxmDataFormat::BFloat16));

    for (std::size_t cycle = 0;
         cycle < schedule.end_cycle() + 16;
         ++cycle) {
        system->tick({});
    }

    for (std::size_t column = 0;
         column < ftlpu::hw::kMxmColumns;
         ++column) {
        const auto actual = system->mxm_unit(0).accumulator().value(
            kAccumulatorAddress, column);
        const auto expected = reference(column);
        if (std::bit_cast<std::uint32_t>(actual)
            != std::bit_cast<std::uint32_t>(expected)) {
            std::cerr << "BF16 MXM mismatch at column=" << column
                      << " actual=" << actual
                      << " expected=" << expected << '\n';
            return 1;
        }
    }

    std::cout << "MXM BF16 passed: INT8 Dequant queue load and Vector Compute\n";
    return 0;
}
