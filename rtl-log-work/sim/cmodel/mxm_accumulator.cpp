#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/mxm/data_format.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kWeightStreams = 16;
constexpr std::size_t kActivationStreamBase = 16;
constexpr std::array<std::size_t, 2> kActivationAddress {4, 8};
constexpr std::array<std::size_t, 5> kOutputAddress {5, 6, 7, 9, 10};
constexpr std::size_t kAccumulatorAddress = 21;
constexpr std::size_t kLoadCycle = 20;
constexpr std::size_t kAccumulatorRowStride = 3;
constexpr std::array<std::size_t, 3> kComputeCycle {30, 34, 70};
constexpr std::array<std::size_t, 5> kReadCycle {100, 130, 160, 190, 220};
constexpr std::array<std::size_t, 5> kOutputWriteCycle {
    114, 144, 174, 204, 234};
constexpr std::size_t kRunCycles = 252;
constexpr std::size_t kInitWords = 256 + 16;
constexpr std::size_t kGoldenWords = 5 * 4 * kBlocks;
constexpr std::size_t kScheduleRecords = 121;

static_assert(kBlocks == 4);
static_assert(kLanes == 8);
static_assert(kInitWords == 272);
static_assert(kGoldenWords == 80);

using Record = std::array<std::uint32_t, 15>;
using LoadSet = std::array<ftlpu::MxmControlInstruction, kBlocks>;
using ComputeSet = std::array<ftlpu::MxmControlInstruction, 3>;
using ReadSet = std::array<ftlpu::MxmControlInstruction, 5>;

std::size_t source_index(std::size_t output_index)
{
    return (5 * output_index + 3) % (kBlocks * kLanes);
}

std::size_t output_index_for_source(std::size_t input_index)
{
    return (13 * ((input_index + kBlocks * kLanes - 3)
                  % (kBlocks * kLanes))) % (kBlocks * kLanes);
}

std::uint64_t read_word(
    const ftlpu::TspSliceSystem& system,
    std::size_t column,
    std::size_t address,
    std::size_t tile)
{
    std::uint64_t result = 0;
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        result |= static_cast<std::uint64_t>(
            system.read_mem_sram_lane_byte(column, tile, address, lane))
            << (lane * 8);
    }
    return result;
}

void initialize_inputs(ftlpu::TspSliceSystem& system)
{
    for (std::size_t block = 0; block < kBlocks; ++block) {
        for (std::size_t byte_stream = 0;
             byte_stream < kWeightStreams;
             ++byte_stream) {
            const auto local_column = byte_stream / 2;
            const auto byte = byte_stream % 2;
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    const auto input_index = tile * kLanes + lane;
                    const auto output_index = block * kLanes + local_column;
                    const auto weight =
                        input_index == source_index(output_index) ? 1.0f : 0.0f;
                    const auto bits = ftlpu::encode_mxm_16bit(
                        weight, ftlpu::MxmDataFormat::BFloat16);
                    system.initialize_mem_sram_lane_byte(
                        byte_stream,
                        tile,
                        block,
                        lane,
                        static_cast<std::uint8_t>(bits >> (byte * 8)));
                }
            }
        }
    }

    for (std::size_t input_set = 0; input_set < 2; ++input_set) {
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto input_index = tile * kLanes + lane;
                auto value = static_cast<float>(
                    output_index_for_source(input_index) + 1);
                if (input_set == 1) value *= -2.0f;
                const auto bits = ftlpu::encode_mxm_16bit(
                    value, ftlpu::MxmDataFormat::BFloat16);
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    system.initialize_mem_sram_lane_byte(
                        kActivationStreamBase + byte,
                        tile,
                        kActivationAddress[input_set],
                        lane,
                        static_cast<std::uint8_t>(bits >> (byte * 8)));
                }
            }
        }
    }
}

void build_schedule(
    ftlpu::TspSliceSystem& system,
    const LoadSet& loads,
    const ComputeSet& computes,
    const ReadSet& reads)
{
    for (std::size_t column = 0; column < kWeightStreams; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kLoadCycle - (15 - group);
        system.icu().enqueue_mem_nop(column, read_cycle);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(column)));
        system.icu().enqueue_mem_repeat(column, kBlocks - 1, 1, 1);
        if (column < 4) {
            auto cursor = read_cycle + kBlocks;
            for (std::size_t output = 0; output < kOutputAddress.size(); ++output) {
                system.icu().enqueue_mem_nop(
                    column, kOutputWriteCycle[output] - cursor);
                system.icu().enqueue_mem(
                    column,
                    ftlpu::MemInstruction::Write(
                        kOutputAddress[output], ftlpu::StreamId::West(column)));
                cursor = kOutputWriteCycle[output] + 1;
            }
        }
    }

    const auto activation_group = kActivationStreamBase
        / ftlpu::hw::kMemSlicesPerGroup;
    const std::array<std::size_t, 3> activation_read_cycle {
        kComputeCycle[0] - (15 - activation_group),
        kComputeCycle[1] - (15 - activation_group),
        kComputeCycle[2] - (15 - activation_group),
    };
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto column = kActivationStreamBase + byte;
        system.icu().enqueue_mem_nop(column, activation_read_cycle[0]);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress[0], ftlpu::StreamId::East(column)));
        system.icu().enqueue_mem_nop(
            column, activation_read_cycle[1] - (activation_read_cycle[0] + 1));
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress[1], ftlpu::StreamId::East(column)));
        system.icu().enqueue_mem_nop(
            column, activation_read_cycle[2] - (activation_read_cycle[1] + 1));
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress[1], ftlpu::StreamId::East(column)));
    }

    system.icu().enqueue_mxm_load_nop(0, kLoadCycle);
    for (const auto& load : loads)
        system.icu().enqueue_mxm(0, load);

    auto cursor = std::size_t {0};
    for (std::size_t index = 0; index < computes.size(); ++index) {
        system.icu().enqueue_mxm_compute_nop(
            0, kComputeCycle[index] - cursor);
        system.icu().enqueue_mxm(0, computes[index]);
        cursor = kComputeCycle[index] + 1;
    }
    for (std::size_t index = 0; index < reads.size(); ++index) {
        system.icu().enqueue_mxm_compute_nop(
            0, kReadCycle[index] - cursor);
        system.icu().enqueue_mxm(0, reads[index]);
        cursor = kReadCycle[index] + 1;
    }
}

void write_init(const char* path, const ftlpu::TspSliceSystem& system)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open MXM accumulator init");
    output << std::hex << std::setfill('0');
    for (std::size_t block = 0; block < kBlocks; ++block) {
        for (std::size_t column = 0; column < kWeightStreams; ++column) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                output << std::setw(16)
                       << read_word(system, column, block, tile) << '\n';
            }
        }
    }
    for (const auto address : kActivationAddress) {
        for (std::size_t column = kActivationStreamBase;
             column < kActivationStreamBase + 2;
             ++column) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                output << std::setw(16)
                       << read_word(system, column, address, tile) << '\n';
            }
        }
    }
}

bool verify_output(const ftlpu::TspSliceSystem& system)
{
    for (std::size_t output = 0; output < kOutputAddress.size(); ++output) {
        for (std::size_t block = 0; block < kBlocks; ++block) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                std::uint32_t actual = 0;
                for (std::size_t byte = 0; byte < 4; ++byte) {
                    actual |= static_cast<std::uint32_t>(
                        system.read_mem_sram_lane_byte(
                            byte, block, kOutputAddress[output], lane))
                        << (byte * 8);
                }
                auto expected_value = 0.0f;
                if (output < 2) {
                    expected_value =
                        -static_cast<float>(block * kLanes + lane + 1);
                } else if (output == 3) {
                    expected_value =
                        -2.0f * static_cast<float>(block * kLanes + lane + 1);
                }
                const auto expected =
                    std::bit_cast<std::uint32_t>(expected_value);
                if (actual != expected) {
                    std::cerr << "MXM accumulator mismatch output=" << output
                              << " block=" << block << " lane=" << lane
                              << " actual=0x" << std::hex << actual
                              << " expected=0x" << expected << std::dec
                              << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

void write_golden(const char* path, const ftlpu::TspSliceSystem& system)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open MXM accumulator golden");
    output << std::hex << std::setfill('0');
    for (const auto address : kOutputAddress) {
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                output << std::setw(16)
                       << read_word(system, column, address, tile) << '\n';
            }
        }
    }
}

void set_record_bits(
    Record& record,
    std::size_t offset,
    std::uint64_t value,
    std::size_t width)
{
    for (std::size_t bit = 0; bit < width; ++bit) {
        if (((value >> bit) & 1u) != 0) {
            record[(offset + bit) / 32] |=
                std::uint32_t {1} << ((offset + bit) % 32);
        }
    }
}

Record command_record(std::uint8_t queue, std::uint32_t command)
{
    Record result {};
    set_record_bits(result, 0, queue, 8);
    set_record_bits(result, 9, command, 32);
    return result;
}

Record mem_record(std::uint8_t queue, const ftlpu::MemInstruction& instruction)
{
    Record result {};
    set_record_bits(result, 0, queue, 8);
    set_record_bits(result, 8, 1, 1);
    set_record_bits(
        result, 41, ftlpu::isa::encode_mem_instruction(instruction), 47);
    return result;
}

Record mxm_record(
    std::uint8_t queue,
    const ftlpu::MxmControlInstruction& instruction)
{
    Record result {};
    set_record_bits(result, 0, queue, 8);
    set_record_bits(result, 8, 1, 1);
    set_record_bits(
        result, 41, ftlpu::isa::encode_mxm_instruction(instruction), 48);
    return result;
}

void write_record(std::ofstream& output, const Record& record)
{
    for (std::size_t word = record.size(); word-- > 0;)
        output << std::setw(8) << record[word];
    output << '\n';
}

void write_schedule(
    const char* path,
    const LoadSet& loads,
    const ComputeSet& computes,
    const ReadSet& reads)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open MXM accumulator schedule");
    output << std::hex << std::setfill('0');
    std::size_t records = 0;

    for (std::size_t column = 0; column < kWeightStreams; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kLoadCycle - (15 - group);
        const auto queue = static_cast<std::uint8_t>(column);
        write_record(output, command_record(
            queue, ftlpu::isa::encode_icu_nop(read_cycle)));
        write_record(output, mem_record(
            queue,
            ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(column))));
        write_record(output, command_record(
            queue, ftlpu::isa::encode_icu_repeat({kBlocks - 1, 1, 1})));
        records += 3;
        if (column < 4) {
            auto cursor = read_cycle + kBlocks;
            for (std::size_t output_index = 0;
                 output_index < kOutputAddress.size();
                 ++output_index) {
                write_record(output, command_record(
                    queue,
                    ftlpu::isa::encode_icu_nop(
                        kOutputWriteCycle[output_index] - cursor)));
                write_record(output, mem_record(
                    queue,
                    ftlpu::MemInstruction::Write(
                        kOutputAddress[output_index],
                        ftlpu::StreamId::West(column))));
                records += 2;
                cursor = kOutputWriteCycle[output_index] + 1;
            }
        }
    }

    const auto activation_group = kActivationStreamBase
        / ftlpu::hw::kMemSlicesPerGroup;
    const std::array<std::size_t, 3> activation_read_cycle {
        kComputeCycle[0] - (15 - activation_group),
        kComputeCycle[1] - (15 - activation_group),
        kComputeCycle[2] - (15 - activation_group),
    };
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto column = kActivationStreamBase + byte;
        write_record(output, command_record(
            static_cast<std::uint8_t>(column),
            ftlpu::isa::encode_icu_nop(activation_read_cycle[0])));
        write_record(output, mem_record(
            static_cast<std::uint8_t>(column),
            ftlpu::MemInstruction::Read(
                kActivationAddress[0], ftlpu::StreamId::East(column))));
        write_record(output, command_record(
            static_cast<std::uint8_t>(column),
            ftlpu::isa::encode_icu_nop(
                activation_read_cycle[1] - (activation_read_cycle[0] + 1))));
        write_record(output, mem_record(
            static_cast<std::uint8_t>(column),
            ftlpu::MemInstruction::Read(
                kActivationAddress[1], ftlpu::StreamId::East(column))));
        write_record(output, command_record(
            static_cast<std::uint8_t>(column),
            ftlpu::isa::encode_icu_nop(
                activation_read_cycle[2] - (activation_read_cycle[1] + 1))));
        write_record(output, mem_record(
            static_cast<std::uint8_t>(column),
            ftlpu::MemInstruction::Read(
                kActivationAddress[1], ftlpu::StreamId::East(column))));
        records += 6;
    }

    write_record(output, command_record(
        104, ftlpu::isa::encode_icu_nop(kLoadCycle)));
    for (const auto& load : loads)
        write_record(output, mxm_record(104, load));
    records += 1 + loads.size();

    auto cursor = std::size_t {0};
    for (std::size_t index = 0; index < computes.size(); ++index) {
        write_record(output, command_record(
            108,
            ftlpu::isa::encode_icu_nop(kComputeCycle[index] - cursor)));
        write_record(output, mxm_record(108, computes[index]));
        records += 2;
        cursor = kComputeCycle[index] + 1;
    }
    for (std::size_t index = 0; index < reads.size(); ++index) {
        write_record(output, command_record(
            108,
            ftlpu::isa::encode_icu_nop(kReadCycle[index] - cursor)));
        write_record(output, mxm_record(108, reads[index]));
        records += 2;
        cursor = kReadCycle[index] + 1;
    }

    if (records != kScheduleRecords)
        throw std::logic_error("MXM accumulator schedule record count mismatch");
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4) {
        std::cerr << "usage: mxm_accumulator "
                  << "<init.hex> <golden.hex> <schedule.hex>\n";
        return 2;
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system);
    write_init(argv[1], system);

    auto loads = LoadSet {};
    for (std::size_t block = 0; block < kBlocks; ++block)
        loads[block] = ftlpu::MxmControlInstruction::IWDirect16(0, block);
    const auto sram_compute = ftlpu::MxmControlInstruction::Compute(
        0,
        kActivationStreamBase,
        0,
        kAccumulatorAddress,
        kAccumulatorRowStride,
        ftlpu::MxmAccumulatorDestination::Sram,
        ftlpu::MxmDataFormat::BFloat16,
        ftlpu::MxmComputeMode::Vector,
        true);
    const auto computes = ComputeSet {
        sram_compute, sram_compute, sram_compute};
    const auto reads = ReadSet {
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kAccumulatorAddress, 0, false),
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kAccumulatorAddress, 0, true),
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kAccumulatorAddress, 0, false),
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kAccumulatorAddress + kAccumulatorRowStride, 0, true),
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kAccumulatorAddress + kAccumulatorRowStride, 0, false),
    };
    build_schedule(system, loads, computes, reads);

    auto icu_log = std::ofstream(
        "build/cmodel_vectors/mxm_accumulator_icu.log", std::ios::trunc);
    auto mem_log = std::ofstream(
        "build/cmodel_vectors/mxm_accumulator_mem.log", std::ios::trunc);
    auto mxm_log = std::ofstream(
        "build/cmodel_vectors/mxm_accumulator_mxm.log", std::ios::trunc);
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle)
        system.tick({.icu = &icu_log, .mem = &mem_log, .mxm = &mxm_log});

    if (!verify_output(system)) return 1;
    write_golden(argv[2], system);
    write_schedule(argv[3], loads, computes, reads);
    std::cout << "C model MXM accumulator SRAM/read-clear golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
