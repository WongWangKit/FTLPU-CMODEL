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
constexpr std::size_t kDefaultMatrixCount = 96;
constexpr std::size_t kRouteStart = 3;
constexpr std::size_t kMemToSxmLatency =
    ftlpu::hw::kMemGroups
    + ftlpu::hw::kC2cToSxmStreamRegisterColumns;
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

    void transpose_run(std::size_t cycle, std::size_t count, ftlpu::SxmInstruction instruction)
    {
        pad(transpose_, cycle, [&](std::size_t n) {
            icu_.enqueue_sxm_transpose_nop(hemisphere_, n);
        });
        icu_.sxm_transpose_iq(hemisphere_).push_sxm_run_2d(
            ftlpu::SxmIcuRun2DInstruction::Run2D(
                0, {count, 1}, {1, 1}, std::move(instruction)));
        transpose_ = cycle + count;
    }

    void permute_run(std::size_t cycle, std::size_t count, ftlpu::SxmInstruction instruction)
    {
        pad(permute_, cycle, [&](std::size_t n) {
            icu_.enqueue_sxm_permute_nop(hemisphere_, n);
        });
        icu_.sxm_permute_iq(hemisphere_).push_sxm_run_2d(
            ftlpu::SxmIcuRun2DInstruction::Run2D(
                0, {count, 1}, {1, 1}, std::move(instruction), kBlockSize));
        permute_ = cycle + count;
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
    ftlpu::TspSliceSystem& system, ftlpu::Hemisphere hemisphere, std::size_t matrix_count)
{
    for (std::size_t matrix = 0; matrix < matrix_count; ++matrix) {
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

void build_schedule(Schedule& schedule, std::size_t matrix_count)
{
    const auto input_beats = matrix_count * kBlocks;
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
            input_beats,
            1);

        const auto write_cycle = kCaptureStart + 1
            + ftlpu::hw::kMemGroups
            + ftlpu::hw::kC2cToSxmStreamRegisterColumns - output_group;
        schedule.mem_repeat(
            output_slice,
            write_cycle,
            ftlpu::MemInstruction::Write(
                kOutputAddressBase, ftlpu::StreamId::West(stream)),
            input_beats,
            1);
    }

    // A new matrix starts every four beats.  The three-cycle northbound tail
    // overlaps the next matrix, while the Permute map repeats every four waves.
    // Do not enqueue a singleton descriptor for every beat. At T4, P consumes
    // four i-MEM words but fetch supplies one per cycle: the prefilled IQ runs
    // out at cycle 23 while T keeps capturing. One existing RUN_2D context
    // sustains II=1 and rotates the source tile by eight lanes per wave.
    schedule.transpose_run(kCaptureStart, input_beats,
        ftlpu::SxmInstruction::Transpose(transpose_src, transpose_internal));
    schedule.permute_run(kCaptureStart + 1, input_beats + kBlocks - 1,
        ftlpu::SxmInstruction::Permute(
            transpose_internal, transpose_dst, wavefront_map(0)));
}

bool verify_transpose(const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere, std::size_t matrix_count)
{
    for (std::size_t matrix = 0; matrix < matrix_count; ++matrix) {
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

bool run_hemisphere(ftlpu::Hemisphere hemisphere, std::size_t matrix_count)
{
    auto system = ftlpu::TspSliceSystem {};
    if (system.mem_array(hemisphere).read_timing() != ftlpu::MemArrayModel::ReadTiming::Main)
        throw std::runtime_error("transpose regression must retain default MEM Main timing");
    initialize_matrix(system, hemisphere, matrix_count);
    auto schedule = Schedule(system.icu(), hemisphere);
    build_schedule(schedule, matrix_count);

    const auto input_beats = matrix_count * kBlocks;
    const auto run_cycles = kCaptureStart + input_beats + 16 + kBlocks;
    for (std::size_t cycle = 0; cycle < run_cycles; ++cycle) {
        system.tick({});
        const auto& t = system.icu().sxm_transpose_iq(hemisphere).last_trace();
        const auto& p = system.icu().sxm_permute_iq(hemisphere).last_trace();
        const bool expected_t = cycle >= kCaptureStart && cycle < kCaptureStart + input_beats;
        const bool expected_p = cycle >= kCaptureStart + 1
            && cycle < kCaptureStart + input_beats + kBlocks;
        // Cycle 0 issues the initial NOP, not a T/P operation.
        if (cycle != 0 && (t.issue_pc.has_value() != expected_t || p.issue_pc.has_value() != expected_p))
            throw std::runtime_error("SXM T/P issue calendar diverged at cycle " + std::to_string(cycle));
        if (t.action == ftlpu::IcuQueueAction::ThreeDDecodeWait
            || p.action == ftlpu::IcuQueueAction::ThreeDDecodeWait)
            throw std::runtime_error("SXM RUN_2D decode starved");
    }
    if (!system.icu().program_queues_done() || !system.program_resources_drained())
        throw std::runtime_error("transpose data matched but ICU/SXM/MEM/SR did not drain");

    return verify_transpose(system, hemisphere, matrix_count);
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc > 2) throw std::invalid_argument("usage: sxm_mem_transpose_test [matrix-count: 1..96]");
    std::size_t matrix_count = kDefaultMatrixCount;
    if (argc == 2) {
        std::size_t used = 0;
        matrix_count = std::stoull(argv[1], &used);
        if (used != std::string(argv[1]).size() || matrix_count == 0 || matrix_count > kDefaultMatrixCount)
            throw std::invalid_argument("matrix-count must be in 1..96");
    }
    if (!run_hemisphere(ftlpu::Hemisphere::East, matrix_count)
        || !run_hemisphere(ftlpu::Hemisphere::West, matrix_count))
        return 1;
    std::cout << "bidirectional local MEM -> SXM -> MEM continuous 32x32 FP16 transpose passed: matrices="
              << matrix_count << ", initiation_interval=" << kBlocks
              << ", exact T/P issue calendar and full drain checked\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "SXM MEM transpose test failed: " << error.what() << '\n';
    return 1;
}
