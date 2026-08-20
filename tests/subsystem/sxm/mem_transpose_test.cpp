#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr std::size_t kMatrixSize = ftlpu::hw::kPhysicalVectorBytes;
constexpr std::size_t kBlockSize = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kBlocks = ftlpu::hw::kTileRows;
constexpr std::size_t kStreams = 2 * kBlockSize;
constexpr std::size_t kMatrixCount = 96;
constexpr std::size_t kInputBeats = kMatrixCount * kBlocks;
constexpr std::size_t kRouteStart = 3;
constexpr std::size_t kMemToSxmLatency =
    ftlpu::hw::kMemGroups
    + ftlpu::hw::kC2cToSxmStreamRegisterColumns + 1;
constexpr std::size_t kCaptureStart = kRouteStart + kMemToSxmLatency;
constexpr std::size_t kOutputAddressBase = 32;
constexpr std::array<std::size_t, kStreams> kInputSlices {
    18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 34, 35};
constexpr std::array<std::size_t, kStreams> kOutputSlices {
    0, 1, 2, 3, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 32, 33};

static_assert(kMatrixSize == 32);
static_assert(kBlockSize == 8);
static_assert(kBlocks == 4);
static_assert(kStreams == 16);

std::uint16_t matrix_value(
    std::size_t matrix,
    std::size_t row,
    std::size_t column)
{
    return static_cast<std::uint16_t>(
        0x1000u + matrix * 0x1000u + row * kMatrixSize + column);
}

ftlpu::SxmInstruction::StreamList east_streams(std::size_t first, std::size_t count)
{
    auto result = ftlpu::SxmInstruction::StreamList {};
    for (std::size_t stream = first; stream < first + count; ++stream) {
        result.push_back(ftlpu::SxmStreamId {ftlpu::StreamId::East(stream).packed()});
    }
    return result;
}

ftlpu::SxmInstruction::StreamList west_streams(std::size_t first, std::size_t count)
{
    auto result = ftlpu::SxmInstruction::StreamList {};
    for (std::size_t stream = first; stream < first + count; ++stream) {
        result.push_back(ftlpu::SxmStreamId {ftlpu::StreamId::West(stream).packed()});
    }
    return result;
}

ftlpu::SxmInstruction::PermuteMap wavefront_map(std::size_t wave)
{
    auto map = ftlpu::Permute320::identity_map();
    for (std::size_t destination_tile = 0; destination_tile < kBlocks; ++destination_tile) {
        const auto source_tile = (wave + kBlocks - destination_tile) % kBlocks;
        for (std::size_t lane = 0; lane < kBlockSize; ++lane) {
            map[destination_tile * kBlockSize + lane] = source_tile * kBlockSize + lane;
        }
    }
    return map;
}

class Schedule {
public:
    Schedule(ftlpu::InstructionControlUnit& icu, ftlpu::Hemisphere hemisphere)
        : icu_(icu), hemisphere_(hemisphere)
    {}

    void mem_repeat(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction,
        std::size_t count,
        std::int64_t stride)
    {
        const auto queue = ftlpu::InstructionControlUnit::mem_queue(
            hemisphere_, slice, 0);
        pad(mem_[queue], cycle,
            [&](std::size_t n) { icu_.enqueue_mem_nop(queue, n); });
        icu_.enqueue_mem(queue, instruction);
        if (count > 1)
            icu_.enqueue_mem_repeat(queue, count - 1, 1, stride);
        mem_[queue] = cycle + count;
    }

    void transpose_at(std::size_t cycle, ftlpu::SxmInstruction instruction)
    {
        pad(transpose_, cycle, [&](std::size_t n) {
            icu_.enqueue_sxm_transpose_nop(hemisphere_, n);
        });
        icu_.enqueue_sxm_transpose(hemisphere_, std::move(instruction));
        transpose_ = cycle + 1;
    }

    void permute_at(std::size_t cycle, ftlpu::SxmInstruction instruction)
    {
        pad(permute_, cycle, [&](std::size_t n) {
            icu_.enqueue_sxm_permute_nop(hemisphere_, n);
        });
        icu_.enqueue_sxm_permute(hemisphere_, std::move(instruction));
        permute_ = cycle + 1;
    }

private:
    template <typename Emit>
    static void pad(std::size_t cursor, std::size_t cycle, Emit emit)
    {
        if (cycle < cursor) throw std::logic_error("offline ICU schedule overlaps a queue");
        emit(cycle - cursor);
    }

    ftlpu::InstructionControlUnit& icu_;
    ftlpu::Hemisphere hemisphere_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::size_t transpose_{0};
    std::size_t permute_{0};
};

void initialize_matrix(
    ftlpu::TspSliceSystem& system, ftlpu::Hemisphere hemisphere)
{
    for (std::size_t matrix = 0; matrix < kMatrixCount; ++matrix) {
        for (std::size_t block_row = 0; block_row < kBlocks; ++block_row) {
            for (std::size_t block_column = 0; block_column < kBlocks; ++block_column) {
                for (std::size_t local_row = 0; local_row < kBlockSize; ++local_row) {
                    for (std::size_t local_column = 0; local_column < kBlockSize; ++local_column) {
                        const auto value = matrix_value(
                            matrix,
                            block_row * kBlockSize + local_row,
                            block_column * kBlockSize + local_column);
                        for (std::size_t byte = 0; byte < 2; ++byte) {
                            system.initialize_mem_sram_lane_byte(
                                hemisphere,
                                kInputSlices[2 * local_row + byte],
                                block_column,
                                matrix * kBlocks + block_row,
                                local_column,
                                static_cast<std::uint8_t>(value >> (8 * byte)));
                        }
                    }
                }
            }
        }
    }
}

void build_schedule(Schedule& schedule)
{
    const auto transpose_src = east_streams(16, kStreams);
    const auto transpose_internal = east_streams(0, kStreams);
    const auto transpose_dst = west_streams(0, kStreams);

    for (std::size_t stream = 0; stream < kStreams; ++stream) {
        const auto input_slice = kInputSlices[stream];
        const auto output_slice = kOutputSlices[stream];
        const auto input_group =
            input_slice / ftlpu::hw::kMemSlicesPerGroup;
        const auto output_group =
            output_slice / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kRouteStart + input_group;
        schedule.mem_repeat(
            input_slice,
            read_cycle,
            ftlpu::MemInstruction::Read(
                0, ftlpu::StreamId::East(16 + stream)),
            kInputBeats,
            1);

        const auto write_cycle = kCaptureStart + 1
            + ftlpu::hw::kMemGroups
            + ftlpu::hw::kC2cToSxmStreamRegisterColumns - output_group;
        schedule.mem_repeat(
            output_slice,
            write_cycle,
            ftlpu::MemInstruction::Write(
                kOutputAddressBase, ftlpu::StreamId::West(stream)),
            kInputBeats,
            1);
    }

    // A new matrix starts every four beats.  The three-cycle northbound tail
    // overlaps the next matrix, while the Permute map repeats every four waves.
    constexpr auto kWaveCount = kInputBeats + kBlocks - 1;
    for (std::size_t wave = 0; wave < kWaveCount; ++wave) {
        if (wave < kInputBeats) {
            schedule.transpose_at(
                kCaptureStart + wave,
                ftlpu::SxmInstruction::Transpose(transpose_src, transpose_internal));
        }
        schedule.permute_at(
            kCaptureStart + wave + 1,
            ftlpu::SxmInstruction::Permute(
                transpose_internal,
                transpose_dst,
                wavefront_map(wave)));
    }
}

bool verify_transpose(const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere)
{
    for (std::size_t matrix = 0; matrix < kMatrixCount; ++matrix) {
        for (std::size_t row = 0; row < kMatrixSize; ++row) {
            for (std::size_t column = 0; column < kMatrixSize; ++column) {
                std::uint16_t actual = 0;
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    actual |= static_cast<std::uint16_t>(system.read_mem_sram_lane_byte(
                        hemisphere,
                        kOutputSlices[2 * (row % kBlockSize) + byte],
                        column / kBlockSize,
                        kOutputAddressBase + matrix * kBlocks + row / kBlockSize,
                        column % kBlockSize)) << (8 * byte);
                }
                const auto expected = matrix_value(matrix, column, row);
                if (actual != expected) {
                    std::cerr << "transpose mismatch in matrix " << matrix
                              << " at (" << row << ',' << column
                              << "): actual=0x" << std::hex << actual
                              << " expected=0x" << expected << std::dec << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

bool run_hemisphere(ftlpu::Hemisphere hemisphere)
{
    auto system = ftlpu::TspSliceSystem {};
    initialize_matrix(system, hemisphere);
    auto schedule = Schedule(system.icu(), hemisphere);
    build_schedule(schedule);

    constexpr std::size_t kRunCycles =
        kCaptureStart + kInputBeats + 16 + kBlocks;
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle)
        system.tick({});

    return verify_transpose(system, hemisphere);
}

} // namespace

int main()
try {
    if (!run_hemisphere(ftlpu::Hemisphere::East)
        || !run_hemisphere(ftlpu::Hemisphere::West))
        return 1;
    std::cout << "bidirectional local MEM -> SXM -> MEM continuous 32x32 FP16 transpose passed: matrices="
              << kMatrixCount << ", initiation_interval=" << kBlocks << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "SXM MEM transpose test failed: " << error.what() << '\n';
    return 1;
}
