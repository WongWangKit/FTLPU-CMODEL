#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
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
constexpr std::array<std::size_t, 16> kBlockActivationSlices {
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
};
constexpr std::size_t kWeightAddress = 64;
constexpr std::size_t kActivationAddress = 80;
constexpr std::size_t kBlockCount = 4;
constexpr std::size_t kAccumulatorAddress = 1024;
constexpr std::size_t kBlockAccumulatorAddress = 128;
constexpr std::size_t kDirectOutputAddress = 192;
constexpr std::size_t kLoadStart = 20;
constexpr std::size_t kComputeCycle = 30;

std::size_t east_read_to_mxm_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mxm_to_west_write_latency(std::size_t slice)
{
    return ftlpu::hw::kSystemStreamRegisterColumns - 1
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
        if (cycle < mxm_load_) {
            throw std::logic_error("BF16 MXM test overlaps the load queue");
        }
        icu_.enqueue_mxm_load_nop(0, cycle - mxm_load_);
        icu_.enqueue_mxm(0, instruction);
        mxm_load_ = cycle + 1;
        end_cycle_ = std::max(end_cycle_, mxm_load_);
    }

    void mxm_compute_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction)
    {
        if (cycle < mxm_compute_) {
            throw std::logic_error("BF16 MXM test overlaps the compute queue");
        }
        icu_.enqueue_mxm_compute_nop(0, cycle - mxm_compute_);
        icu_.enqueue_mxm(0, instruction);
        mxm_compute_ = cycle + 1;
        end_cycle_ = std::max(end_cycle_, mxm_compute_);
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
        mxm_compute_ = cycle + count;
        end_cycle_ = std::max(end_cycle_, mxm_compute_);
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

float reference(std::size_t column)
{
    float sum = 0.0f;
    for (std::size_t row = 0; row < ftlpu::hw::kMxmRows; ++row) {
        const auto activation = ftlpu::Bf16::from_float(
            activation_value(row)).to_float();
        const auto weight = ftlpu::Bf16::from_float(
            weight_value(row, column)).to_float();
        sum += activation * weight;
    }
    return sum;
}

float block_activation_value(std::size_t output_row, std::size_t k)
{
    const auto signed_pattern =
        static_cast<int>((output_row * 11 + k * 7) % 29) - 14;
    return static_cast<float>(signed_pattern) * 0.023f + 0.50390625f;
}

void initialize_block_activations(ftlpu::TspSliceSystem& system)
{
    for (std::size_t block = 0; block < kBlockCount; ++block) {
        for (std::size_t output_row = 0;
             output_row < ftlpu::hw::kMxmBlockRows;
             ++output_row) {
            for (std::size_t tile = 0;
                 tile < ftlpu::hw::kTileRows;
                 ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile;
                     ++lane) {
                    const auto k =
                        tile * ftlpu::hw::kLanesPerTile + lane;
                    const auto global_output_row =
                        block * ftlpu::hw::kMxmBlockRows + output_row;
                    const auto bits = ftlpu::Bf16::from_float(
                        block_activation_value(global_output_row, k)).bits();
                    system.initialize_mem_sram_lane_byte(
                        kBlockActivationSlices[output_row * 2],
                        tile,
                        kActivationAddress + block,
                        lane,
                        static_cast<std::uint8_t>(bits & 0xffu));
                    system.initialize_mem_sram_lane_byte(
                        kBlockActivationSlices[output_row * 2 + 1],
                        tile,
                        kActivationAddress + block,
                        lane,
                        static_cast<std::uint8_t>(bits >> 8));
                }
            }
        }
    }
}

float block_reference(std::size_t output_row, std::size_t column)
{
    float sum = 0.0f;
    for (std::size_t k = 0; k < ftlpu::hw::kMxmRows; ++k) {
        const auto activation = ftlpu::Bf16::from_float(
            block_activation_value(output_row, k)).to_float();
        const auto weight = ftlpu::Bf16::from_float(
            weight_value(k, column)).to_float();
        sum += activation * weight;
    }
    return sum;
}

} // namespace

int main()
{
    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system);
    auto schedule = Schedule(system.icu());

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
            ftlpu::MxmAccumulatorDestination::Stream,
            ftlpu::MxmDataFormat::BFloat16,
            ftlpu::MxmComputeMode::Vector,
            true,
            ftlpu::MxmAccumulatorOutputFormat::BFloat16));

    // Tile r starts its local diamond at issue+r.  SC column c and local
    // diagonal s therefore run at issue+r+c+s.  The common MEM write issue
    // below is aligned to the externally visible top result at issue+3+14;
    // changing the per-tile one-cycle stagger makes this black-box read fail.
    for (std::size_t byte = 0; byte < kActivationSlices.size(); ++byte) {
        const auto slice = kBlockActivationSlices[8 + byte];
        schedule.mem_at(
            slice,
            kComputeCycle
                + ftlpu::hw::kMxmSupercellsPerPlane - 1
                + ftlpu::Mxm::kLocalMacStages - 1
                + mxm_to_west_write_latency(slice),
            ftlpu::MemInstruction::Write(
                kDirectOutputAddress,
                ftlpu::StreamId::West(byte)));
    }

    auto vector_deskew_writes = std::size_t {0};
    auto vector_deskew_vectors = std::size_t {0};
    for (std::size_t cycle = 0;
         cycle < schedule.end_cycle() + 8;
         ++cycle) {
        system.tick({});
        const auto timing = system.system_timing_snapshot().mxms[0];
        vector_deskew_writes += timing.deskew_writes;
        vector_deskew_vectors += timing.deskew_vectors;
    }

    constexpr auto kOrdinaryCells =
        ftlpu::hw::kTileRows * ftlpu::hw::kMxmSupercellsPerPlane;
    if (vector_deskew_writes
            != kOrdinaryCells * ftlpu::hw::kLanesPerTile
        || vector_deskew_vectors != kOrdinaryCells) {
        std::cerr << "MXM explicit lane deskew timing mismatch: writes="
                  << vector_deskew_writes
                  << " vectors=" << vector_deskew_vectors << '\n';
        return 1;
    }

    for (std::size_t column = 0;
         column < ftlpu::hw::kMxmColumns;
         ++column) {
        const auto tile = column / ftlpu::hw::kLanesPerTile;
        const auto lane = column % ftlpu::hw::kLanesPerTile;
        const auto low = system.read_mem_sram_lane_byte(
            kBlockActivationSlices[8], tile,
            kDirectOutputAddress, lane);
        const auto high = system.read_mem_sram_lane_byte(
            kBlockActivationSlices[9], tile,
            kDirectOutputAddress, lane);
        const auto actual = ftlpu::Bf16::from_bits(
            static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8));
        const auto expected = ftlpu::Bf16::from_float(reference(column));
        if (actual.bits() != expected.bits()) {
            std::cerr << "MXM Vector black-box mismatch at column="
                      << column << " actual=" << actual.to_float()
                      << " expected=" << expected.to_float() << '\n';
            return 1;
        }
    }

    auto direct_system =
        std::make_unique<ftlpu::TspSliceSystem>();
    initialize_inputs(*direct_system);
    initialize_block_activations(*direct_system);
    auto direct_schedule = Schedule(direct_system->icu());

    for (std::size_t block = 0;
         block < ftlpu::hw::kMxmSupercellsPerPlane;
         ++block) {
        const auto iw_cycle = kLoadStart + block;
        for (std::size_t stream = 0;
             stream < kWeightSlices.size();
             ++stream) {
            const auto slice = kWeightSlices[stream];
            direct_schedule.mem_at(
                slice,
                iw_cycle - east_read_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    kWeightAddress + block,
                    ftlpu::StreamId::East(stream)));
        }
        direct_schedule.mxm_load_at(
            iw_cycle,
            ftlpu::MxmControlInstruction::IWDirect16(0, block));
    }

    constexpr std::size_t kFinalPartialCycle = kComputeCycle + 8;
    for (const auto compute_cycle : {kComputeCycle, kFinalPartialCycle}) {
        for (std::size_t stream = 0;
             stream < kBlockActivationSlices.size();
             ++stream) {
            const auto slice = kBlockActivationSlices[stream];
            direct_schedule.mem_repeat_at(
                slice,
                compute_cycle - east_read_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    kActivationAddress,
                    ftlpu::StreamId::East(stream)),
                kBlockCount,
                1);
        }
        direct_schedule.mxm_compute_repeat_at(
            compute_cycle,
            ftlpu::MxmControlInstruction::Compute(
                0,
                0,
                0,
                kBlockAccumulatorAddress,
                1,
                compute_cycle == kFinalPartialCycle
                    ? ftlpu::MxmAccumulatorDestination::Stream
                    : ftlpu::MxmAccumulatorDestination::Sram,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmComputeMode::Block8,
                true),
            kBlockCount);
    }

    for (std::size_t stream = 0;
         stream < kBlockActivationSlices.size();
         ++stream) {
        const auto slice = kBlockActivationSlices[stream];
        const auto write_cycle = kFinalPartialCycle
            + ftlpu::hw::kMxmSupercellsPerPlane - 1
            + ftlpu::Mxm::kLocalMacStages - 1
            + mxm_to_west_write_latency(slice);
        direct_schedule.mem_repeat_at(
            slice,
            write_cycle,
            ftlpu::MemInstruction::Write(
                kDirectOutputAddress,
                ftlpu::StreamId::West(stream)),
            kBlockCount,
            1);
    }

    for (std::size_t cycle = 0;
         cycle < direct_schedule.end_cycle() + 8;
         ++cycle) {
        direct_system->tick({});
    }

    for (std::size_t output_row = 0;
         output_row < kBlockCount * ftlpu::hw::kMxmBlockRows;
         ++output_row) {
        const auto row = kDirectOutputAddress
            + output_row / ftlpu::hw::kMxmBlockRows;
        const auto stream =
            (output_row % ftlpu::hw::kMxmBlockRows) * 2;
        for (std::size_t column = 0;
             column < ftlpu::hw::kMxmColumns;
             ++column) {
            const auto tile = column / ftlpu::hw::kLanesPerTile;
            const auto lane = column % ftlpu::hw::kLanesPerTile;
            const auto low = direct_system->read_mem_sram_lane_byte(
                kBlockActivationSlices[stream], tile, row, lane);
            const auto high = direct_system->read_mem_sram_lane_byte(
                kBlockActivationSlices[stream + 1], tile, row, lane);
            const auto actual = ftlpu::Bf16::from_bits(
                static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8));
            const auto expected = ftlpu::Bf16::from_float(
                2.0f * block_reference(output_row, column));
            if (actual.bits() != expected.bits()) {
                std::cerr
                    << "MXM Block8 direct BF16 stream mismatch at row="
                    << output_row << " column=" << column
                    << " actual=" << actual.to_float()
                    << " expected=" << expected.to_float() << '\n';
                return 1;
            }
        }
    }
    std::cout
        << "MXM BF16 black-box passed: MEM -> MXM Vector/Block8 -> MEM; "
           "four tile-local diamonds are launched one cycle apart and "
           "checked only through final MEM data\n";
    return 0;
}
