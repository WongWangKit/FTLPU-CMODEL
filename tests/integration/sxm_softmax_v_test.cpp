#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

namespace {

constexpr std::size_t kSeqLen = 32;
constexpr std::size_t kHidden = ftlpu::hw::kMxmColumns;
constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;

constexpr std::size_t kWeightMemColumnBase = 16;
constexpr std::size_t kActivationMemColumn = 40;
constexpr std::size_t kResultMemColumnBase = 0;
constexpr std::size_t kInputStream = 1;
constexpr std::size_t kWeightStreamBase = 16;
constexpr std::size_t kOutputWestStreamBase = 0;
constexpr std::size_t kMxm = 1;

constexpr std::size_t kIwStart = 20;
constexpr std::size_t kTransposeStart = 24;
constexpr std::size_t kTransposeIssueCycles = kSeqLen + ftlpu::hw::kTileRows - 1;
constexpr std::size_t kPermuteCaptureStart =
    kTransposeStart + ftlpu::hw::kLanesPerTile - 1;
constexpr std::size_t kPermuteEmitStart =
    kPermuteCaptureStart + kTransposeIssueCycles;
constexpr std::size_t kPermuteIssueCycles = kTransposeIssueCycles + kSeqLen;
constexpr std::size_t kComputeStart = kPermuteEmitStart + 1;
constexpr std::size_t kMxmFirstOutput = kComputeStart + kBlocks - 1;
constexpr std::size_t kResultWriteStart =
    kMxmFirstOutput + ftlpu::hw::kMxmBoundaryStreamRegisterColumn;
constexpr std::size_t kTotalCycles = kResultWriteStart + kSeqLen + kBlocks + 8;

using Softmax = std::array<std::array<std::int8_t, kSeqLen>, kSeqLen>;
using Value = std::array<std::array<std::int8_t, kHidden>, kSeqLen>;
using Output = std::array<std::array<std::int32_t, kHidden>, kSeqLen>;

constexpr std::size_t east_cycles_to_mxm(std::size_t column)
{
    return ftlpu::hw::kMxmBoundaryStreamRegisterColumn
        - column / ftlpu::hw::kSlicesPerGroup;
}

constexpr std::size_t east_cycles_to_sxm(std::size_t column)
{
    return ftlpu::hw::kMemEastBoundaryStreamRegisterColumn
        - column / ftlpu::hw::kSlicesPerGroup;
}

constexpr std::size_t weight_address(std::size_t column_block)
{
    return column_block * kLanes;
}

constexpr std::size_t activation_address(std::size_t key)
{
    return key * kLanes;
}

constexpr std::size_t result_address(std::size_t query)
{
    return query * kLanes;
}

ftlpu::Permute320::Map attention_permute_map()
{
    auto map = ftlpu::Permute320::identity_map();
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            map[tile * kLanes + lane] = tile * kLanes + (kLanes - 1 - lane);
        }
    }
    return map;
}

std::int8_t softmax_value(std::size_t query, std::size_t key)
{
    return static_cast<std::int8_t>(static_cast<int>((query * 5 + key * 3 + 7) % 9) - 4);
}

std::int8_t v_value(std::size_t key, std::size_t hidden)
{
    return static_cast<std::int8_t>(static_cast<int>((key * 11 + hidden * 7 + 3) % 13) - 6);
}

Output golden(const Softmax& softmax, const Value& value)
{
    auto output = Output {};
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            auto sum = std::int32_t {0};
            for (std::size_t key = 0; key < kSeqLen; ++key) {
                sum += static_cast<std::int32_t>(softmax[query][key])
                    * static_cast<std::int32_t>(value[key][hidden]);
            }
            output[query][hidden] = sum;
        }
    }
    return output;
}

void stage_sram(
    ftlpu::TileArrayModel& mem,
    const Softmax& softmax,
    const Value& value)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = kWeightMemColumnBase + stream;
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
                const auto hidden = column_block * kLoadStreams + stream;
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    const auto key = tile * kLanes + lane;
                    const auto weight = key < kSeqLen ? value[key][hidden] : std::int8_t {0};
                    mem.set_sram_lane_byte(
                        mem_column,
                        tile,
                        weight_address(column_block),
                        lane,
                        static_cast<std::uint8_t>(weight));
                }
            }
        }
    }

    for (std::size_t key = 0; key < kSeqLen; ++key) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto query = tile * kLanes + lane;
                const auto activation = query < kSeqLen
                    ? softmax[query][key]
                    : std::int8_t {0};
                mem.set_sram_lane_byte(
                    kActivationMemColumn,
                    tile,
                    activation_address(key),
                    lane,
                    static_cast<std::uint8_t>(activation));
            }
        }
    }
}

void load_icu_program(ftlpu::InstructionControlUnit& icu)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = kWeightMemColumnBase + stream;
        const auto first_read = kIwStart - east_cycles_to_mxm(mem_column) - 1;
        icu.enqueue_mem_nop(mem_column, first_read);
        icu.enqueue_mem(
            mem_column,
            ftlpu::MemInstruction::Read(
                weight_address(kBlocks - 1),
                kWeightStreamBase + stream));
        icu.enqueue_mem_repeat(
            mem_column,
            kBlocks - 1,
            1,
            -static_cast<std::int64_t>(kLanes));
    }

    const auto first_activation_read =
        kTransposeStart - east_cycles_to_sxm(kActivationMemColumn) - 1;
    icu.enqueue_mem_nop(kActivationMemColumn, first_activation_read);
    for (std::size_t phase = 0; phase < kSeqLen; ++phase) {
        const auto block = phase / kLanes;
        const auto lane = phase % kLanes;
        const auto key = block * kLanes + (kLanes - 1 - lane);
        icu.enqueue_mem(
            kActivationMemColumn,
            ftlpu::MemInstruction::Read(activation_address(key), kInputStream));
    }

    for (std::size_t byte = 0; byte < sizeof(std::int32_t); ++byte) {
        const auto mem_column = kResultMemColumnBase + byte;
        icu.enqueue_mem_nop(mem_column, kResultWriteStart);
        icu.enqueue_mem(
            mem_column,
            ftlpu::MemInstruction::Write(
                result_address(0),
                ftlpu::hw::kEastStreams + kOutputWestStreamBase + byte));
        icu.enqueue_mem_repeat(mem_column, kSeqLen - 1, 1, kLanes);
    }

    icu.enqueue_mxm_load_nop(kMxm, kIwStart);
    icu.enqueue_mxm(kMxm, ftlpu::MxmControlInstruction::IW(0));
    icu.enqueue_mxm_load_repeat(kMxm, kBlocks - 1);

    icu.enqueue_mxm_compute_nop(kMxm, kComputeStart);
    icu.enqueue_mxm(
        kMxm,
        ftlpu::MxmControlInstruction::Compute(
            0,
            kInputStream,
            kOutputWestStreamBase));
    icu.enqueue_mxm_compute_repeat(kMxm, kSeqLen - 1);

    const auto east1 = ftlpu::SxmStreamId {ftlpu::StreamId::East(kInputStream).packed()};
    icu.enqueue_sxm_transpose_nop(kTransposeStart);
    icu.enqueue_sxm_transpose(ftlpu::SxmInstruction::TransposeStream(east1, east1));
    icu.enqueue_sxm_transpose_repeat(kTransposeIssueCycles - 1);

    icu.enqueue_sxm_permute_nop(kPermuteCaptureStart);
    icu.enqueue_sxm_permute(ftlpu::SxmInstruction::PermuteStream(
        east1,
        east1,
        attention_permute_map()));
    icu.enqueue_sxm_permute_repeat(kPermuteIssueCycles - 1);
}

std::int32_t load_result(
    const ftlpu::TileArrayModel& mem,
    std::size_t query,
    std::size_t hidden)
{
    const auto tile = hidden / kLanes;
    const auto lane = hidden % kLanes;
    auto raw = std::uint32_t {0};
    for (std::size_t byte = 0; byte < sizeof(std::int32_t); ++byte) {
        raw |= static_cast<std::uint32_t>(mem.sram_lane_byte(
                   kResultMemColumnBase + byte,
                   tile,
                   result_address(query),
                   lane))
            << (byte * 8);
    }
    return static_cast<std::int32_t>(raw);
}

} // namespace

int main()
try
{
    auto softmax = Softmax {};
    auto value = Value {};
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        for (std::size_t key = 0; key < kSeqLen; ++key) {
            softmax[query][key] = softmax_value(query, key);
        }
    }
    for (std::size_t key = 0; key < kSeqLen; ++key) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            value[key][hidden] = v_value(key, hidden);
        }
    }
    const auto expected = golden(softmax, value);

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    stage_sram(system->mem(), softmax, value);
    load_icu_program(system->icu());

    const auto log_dir = std::filesystem::path("logs") / "sxm_softmax_v";
    std::filesystem::create_directories(log_dir);
    auto icu_log = std::ofstream(log_dir / "icu.log");
    if (!icu_log.good()) {
        std::cerr << "failed to open SXM softmax x V ICU log\n";
        return 1;
    }
    icu_log << "sxm_softmax_v mem_to_sxm_to_mxm_to_mem"
            << " seq_len=" << kSeqLen
            << " hidden=" << kHidden
            << " weight_streams=E16..E31"
            << " activation_stream=E" << kInputStream
            << " output_streams=W0..W3"
            << " iw_start=" << kIwStart
            << " transpose_start=" << kTransposeStart
            << " permute_capture_start=" << kPermuteCaptureStart
            << " permute_emit_start=" << kPermuteEmitStart
            << " compute_start=" << kComputeStart
            << " write_start=" << kResultWriteStart << '\n';

    auto output_blocks = std::size_t {0};
    for (std::size_t cycle = 0; cycle < kTotalCycles; ++cycle) {
        auto sinks = ftlpu::TspSliceSystem::LogSinks {};
        sinks.icu = &icu_log;
        system->tick(sinks);
        for (const auto& output : system->mxm_unit(kMxm).last_outputs()) {
            if (output.row >= kSeqLen) {
                continue;
            }
            ++output_blocks;
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto hidden = output.column_block * kLanes + lane;
                if (output.values[lane] != expected[output.row][hidden]) {
                    std::cerr << "softmax x V MXM mismatch cycle=" << cycle
                              << " query=" << output.row
                              << " hidden=" << hidden
                              << " actual=" << output.values[lane]
                              << " expected=" << expected[output.row][hidden] << '\n';
                    return 1;
                }
            }
        }
    }

    if (output_blocks != kSeqLen * kBlocks) {
        std::cerr << "softmax x V emitted " << output_blocks
                  << " MXM blocks, expected " << kSeqLen * kBlocks << '\n';
        return 1;
    }

    for (std::size_t query = 0; query < kSeqLen; ++query) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto actual = load_result(system->mem(), query, hidden);
            if (actual != expected[query][hidden]) {
                std::cerr << "softmax x V SRAM mismatch query=" << query
                          << " hidden=" << hidden
                          << " actual=" << actual
                          << " expected=" << expected[query][hidden] << '\n';
                return 1;
            }
        }
    }

    return 0;
}
catch (const std::exception& ex)
{
    std::cerr << ex.what() << '\n';
    return 1;
}
