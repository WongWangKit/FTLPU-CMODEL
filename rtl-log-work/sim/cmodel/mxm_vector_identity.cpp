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
#include <string>
#include <string_view>

namespace {

constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kCases = 2;
constexpr std::size_t kWeightStreams = 16;
constexpr std::size_t kActivationStreamBase = 16;
constexpr std::array<std::size_t, kCases> kWeightAddressBase {0, 8};
constexpr std::array<std::size_t, kCases> kActivationAddress {4, 12};
constexpr std::array<std::size_t, kCases> kOutputAddress {5, 13};
constexpr std::array<std::size_t, kCases> kLoadCycle {20, 70};
constexpr std::array<std::size_t, kCases> kComputeCycle {30, 80};
constexpr std::array<std::size_t, kCases> kOutputWriteCycle {
    kComputeCycle[0] + 3 + 14,
    kComputeCycle[1] + 3 + 14,
};
constexpr std::size_t kRunCycles = 112;
constexpr std::size_t kInitWords =
    kCases * (kBlocks * kWeightStreams * kBlocks + 2 * kBlocks);
constexpr std::size_t kGoldenWords = kCases * 4 * kBlocks;
constexpr std::size_t kScheduleRecords = 134;

static_assert(kBlocks == 4);
static_assert(kLanes == 8);
static_assert(kInitWords == 528);
static_assert(kGoldenWords == 32);

using Record = std::array<std::uint32_t, 15>;

std::size_t source_index(std::size_t test_case, std::size_t output_index)
{
    return test_case == 0
        ? (5 * output_index + 3) % (kBlocks * kLanes)
        : (7 * output_index + 1) % (kBlocks * kLanes);
}

std::size_t output_index_for_source(
    std::size_t test_case,
    std::size_t input_index)
{
    const auto modulus = kBlocks * kLanes;
    // 13 and 23 are the inverses of 5 and 7 modulo 32, respectively.
    return test_case == 0
        ? (13 * ((input_index + modulus - 3) % modulus)) % modulus
        : (23 * ((input_index + modulus - 1) % modulus)) % modulus;
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

void initialize_inputs(
    ftlpu::TspSliceSystem& system,
    ftlpu::MxmDataFormat data_format)
{
    for (std::size_t test_case = 0; test_case < kCases; ++test_case) {
        for (std::size_t block = 0; block < kBlocks; ++block) {
            for (std::size_t byte_stream = 0;
                 byte_stream < kWeightStreams;
                 ++byte_stream) {
                const auto local_column = byte_stream / 2;
                const auto byte = byte_stream % 2;
                for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                    for (std::size_t lane = 0; lane < kLanes; ++lane) {
                        const auto input_index = tile * kLanes + lane;
                        const auto output_index =
                            block * kLanes + local_column;
                        const auto weight = input_index
                                == source_index(test_case, output_index)
                            ? 1.0f
                            : 0.0f;
                        const auto bits =
                            ftlpu::encode_mxm_16bit(weight, data_format);
                        system.initialize_mem_sram_lane_byte(
                            byte_stream,
                            tile,
                            kWeightAddressBase[test_case] + block,
                            lane,
                            static_cast<std::uint8_t>(bits >> (byte * 8)));
                    }
                }
            }
        }

        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto input_index = tile * kLanes + lane;
                auto value = static_cast<float>(
                    output_index_for_source(test_case, input_index) + 1);
                if (test_case == 1) value = -value;
                const auto bits =
                    ftlpu::encode_mxm_16bit(value, data_format);
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    system.initialize_mem_sram_lane_byte(
                        kActivationStreamBase + byte,
                        tile,
                        kActivationAddress[test_case],
                        lane,
                        static_cast<std::uint8_t>(bits >> (byte * 8)));
                }
            }
        }
    }
}

void build_schedule(
    ftlpu::TspSliceSystem& system,
    const std::array<
        std::array<ftlpu::MxmControlInstruction, kBlocks>,
        kCases>& loads,
    const std::array<ftlpu::MxmControlInstruction, kCases>& computes)
{
    for (std::size_t column = 0; column < kWeightStreams; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        const auto first_read_cycle = kLoadCycle[0] - (15 - group);
        const auto second_read_cycle = kLoadCycle[1] - (15 - group);
        system.icu().enqueue_mem_nop(column, first_read_cycle);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kWeightAddressBase[0], ftlpu::StreamId::East(column)));
        system.icu().enqueue_mem_repeat(column, kBlocks - 1, 1, 1);
        if (column < 4) {
            system.icu().enqueue_mem_nop(
                column,
                kOutputWriteCycle[0] - (first_read_cycle + kBlocks));
            system.icu().enqueue_mem(
                column,
                ftlpu::MemInstruction::Write(
                    kOutputAddress[0], ftlpu::StreamId::West(column)));
            system.icu().enqueue_mem_nop(
                column, second_read_cycle - (kOutputWriteCycle[0] + 1));
        } else {
            system.icu().enqueue_mem_nop(
                column, second_read_cycle - (first_read_cycle + kBlocks));
        }
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kWeightAddressBase[1], ftlpu::StreamId::East(column)));
        system.icu().enqueue_mem_repeat(column, kBlocks - 1, 1, 1);
        if (column < 4) {
            system.icu().enqueue_mem_nop(
                column,
                kOutputWriteCycle[1] - (second_read_cycle + kBlocks));
            system.icu().enqueue_mem(
                column,
                ftlpu::MemInstruction::Write(
                    kOutputAddress[1], ftlpu::StreamId::West(column)));
        }
    }

    const auto activation_group = kActivationStreamBase
        / ftlpu::hw::kMemSlicesPerGroup;
    const std::array<std::size_t, kCases> activation_read_cycle {
        kComputeCycle[0] - (15 - activation_group),
        kComputeCycle[1] - (15 - activation_group),
    };
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto column = kActivationStreamBase + byte;
        system.icu().enqueue_mem_nop(column, activation_read_cycle[0]);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress[0],
                ftlpu::StreamId::East(kActivationStreamBase + byte)));
        system.icu().enqueue_mem_nop(
            column, activation_read_cycle[1] - (activation_read_cycle[0] + 1));
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress[1],
                ftlpu::StreamId::East(kActivationStreamBase + byte)));
    }

    system.icu().enqueue_mxm_load_nop(0, kLoadCycle[0]);
    for (const auto& load : loads[0])
        system.icu().enqueue_mxm(0, load);
    system.icu().enqueue_mxm_load_nop(
        0, kLoadCycle[1] - (kLoadCycle[0] + kBlocks));
    for (const auto& load : loads[1])
        system.icu().enqueue_mxm(0, load);

    system.icu().enqueue_mxm_compute_nop(0, kComputeCycle[0]);
    system.icu().enqueue_mxm(0, computes[0]);
    system.icu().enqueue_mxm_compute_nop(
        0, kComputeCycle[1] - (kComputeCycle[0] + 1));
    system.icu().enqueue_mxm(0, computes[1]);
}

void write_init(const char* path, const ftlpu::TspSliceSystem& system)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open MXM init output");
    output << std::hex << std::setfill('0');
    for (std::size_t test_case = 0; test_case < kCases; ++test_case) {
        for (std::size_t block = 0; block < kBlocks; ++block) {
            for (std::size_t column = 0; column < kWeightStreams; ++column) {
                for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                    output << std::setw(16)
                           << read_word(
                                  system,
                                  column,
                                  kWeightAddressBase[test_case] + block,
                                  tile)
                           << '\n';
                }
            }
        }
    }
    for (std::size_t test_case = 0; test_case < kCases; ++test_case) {
        for (std::size_t column = kActivationStreamBase;
             column < kActivationStreamBase + 2;
             ++column) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                output << std::setw(16)
                       << read_word(
                              system,
                              column,
                              kActivationAddress[test_case],
                              tile)
                       << '\n';
            }
        }
    }
}

bool verify_output(const ftlpu::TspSliceSystem& system)
{
    for (std::size_t test_case = 0; test_case < kCases; ++test_case) {
        for (std::size_t block = 0; block < kBlocks; ++block) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                std::uint32_t actual = 0;
                for (std::size_t byte = 0; byte < 4; ++byte) {
                    actual |= static_cast<std::uint32_t>(
                        system.read_mem_sram_lane_byte(
                            byte,
                            block,
                            kOutputAddress[test_case],
                            lane)) << (byte * 8);
                }
                auto expected_value =
                    static_cast<float>(block * kLanes + lane + 1);
                if (test_case == 1) expected_value = -expected_value;
                const auto expected =
                    std::bit_cast<std::uint32_t>(expected_value);
                if (actual != expected) {
                    std::cerr << "MXM double-buffer mismatch case="
                              << test_case << " block=" << block
                              << " lane=" << lane
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
    if (!output) throw std::runtime_error("failed to open MXM golden output");
    output << std::hex << std::setfill('0');
    for (std::size_t test_case = 0; test_case < kCases; ++test_case) {
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                output << std::setw(16)
                       << read_word(
                              system,
                              column,
                              kOutputAddress[test_case],
                              tile)
                       << '\n';
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
    const std::array<
        std::array<ftlpu::MxmControlInstruction, kBlocks>,
        kCases>& loads,
    const std::array<ftlpu::MxmControlInstruction, kCases>& computes)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open MXM schedule output");
    output << std::hex << std::setfill('0');
    std::size_t records = 0;

    for (std::size_t column = 0; column < kWeightStreams; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        const auto first_read_cycle = kLoadCycle[0] - (15 - group);
        const auto second_read_cycle = kLoadCycle[1] - (15 - group);
        const auto queue = static_cast<std::uint8_t>(column);
        write_record(output, command_record(
            queue, ftlpu::isa::encode_icu_nop(first_read_cycle)));
        write_record(output, mem_record(
            queue,
            ftlpu::MemInstruction::Read(
                kWeightAddressBase[0], ftlpu::StreamId::East(column))));
        write_record(output, command_record(
            queue,
            ftlpu::isa::encode_icu_repeat({kBlocks - 1, 1, 1})));
        records += 3;
        if (column < 4) {
            write_record(output, command_record(
                queue,
                ftlpu::isa::encode_icu_nop(
                    kOutputWriteCycle[0]
                    - (first_read_cycle + kBlocks))));
            write_record(output, mem_record(
                queue,
                ftlpu::MemInstruction::Write(
                    kOutputAddress[0], ftlpu::StreamId::West(column))));
            write_record(output, command_record(
                queue,
                ftlpu::isa::encode_icu_nop(
                    second_read_cycle - (kOutputWriteCycle[0] + 1))));
            records += 3;
        } else {
            write_record(output, command_record(
                queue,
                ftlpu::isa::encode_icu_nop(
                    second_read_cycle
                    - (first_read_cycle + kBlocks))));
            records += 1;
        }
        write_record(output, mem_record(
            queue,
            ftlpu::MemInstruction::Read(
                kWeightAddressBase[1], ftlpu::StreamId::East(column))));
        write_record(output, command_record(
            queue,
            ftlpu::isa::encode_icu_repeat({kBlocks - 1, 1, 1})));
        records += 2;
        if (column < 4) {
            write_record(output, command_record(
                queue,
                ftlpu::isa::encode_icu_nop(
                    kOutputWriteCycle[1]
                    - (second_read_cycle + kBlocks))));
            write_record(output, mem_record(
                queue,
                ftlpu::MemInstruction::Write(
                    kOutputAddress[1], ftlpu::StreamId::West(column))));
            records += 2;
        }
    }

    const auto activation_group = kActivationStreamBase
        / ftlpu::hw::kMemSlicesPerGroup;
    const std::array<std::size_t, kCases> activation_read_cycle {
        kComputeCycle[0] - (15 - activation_group),
        kComputeCycle[1] - (15 - activation_group),
    };
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto column = kActivationStreamBase + byte;
        write_record(output, command_record(
            static_cast<std::uint8_t>(column),
            ftlpu::isa::encode_icu_nop(activation_read_cycle[0])));
        write_record(output, mem_record(
            static_cast<std::uint8_t>(column),
            ftlpu::MemInstruction::Read(
                kActivationAddress[0],
                ftlpu::StreamId::East(kActivationStreamBase + byte))));
        write_record(output, command_record(
            static_cast<std::uint8_t>(column),
            ftlpu::isa::encode_icu_nop(
                activation_read_cycle[1] - (activation_read_cycle[0] + 1))));
        write_record(output, mem_record(
            static_cast<std::uint8_t>(column),
            ftlpu::MemInstruction::Read(
                kActivationAddress[1],
                ftlpu::StreamId::East(kActivationStreamBase + byte))));
        records += 4;
    }

    write_record(output, command_record(
        104, ftlpu::isa::encode_icu_nop(kLoadCycle[0])));
    for (const auto& load : loads[0])
        write_record(output, mxm_record(104, load));
    write_record(output, command_record(
        104,
        ftlpu::isa::encode_icu_nop(
            kLoadCycle[1] - (kLoadCycle[0] + kBlocks))));
    for (const auto& load : loads[1])
        write_record(output, mxm_record(104, load));
    records += 2 + loads[0].size() + loads[1].size();

    write_record(output, command_record(
        108, ftlpu::isa::encode_icu_nop(kComputeCycle[0])));
    write_record(output, mxm_record(108, computes[0]));
    write_record(output, command_record(
        108,
        ftlpu::isa::encode_icu_nop(
            kComputeCycle[1] - (kComputeCycle[0] + 1))));
    write_record(output, mxm_record(108, computes[1]));
    records += 4;

    if (records != kScheduleRecords)
        throw std::logic_error("MXM schedule record count mismatch");
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: mxm_vector_identity "
                  << "<init.hex> <golden.hex> <schedule.hex> [--fp16]\n";
        return 2;
    }

    auto data_format = ftlpu::MxmDataFormat::BFloat16;
    if (argc == 5) {
        if (std::string_view {argv[4]} != "--fp16") {
            std::cerr << "unknown MXM vector option: " << argv[4] << '\n';
            return 2;
        }
        data_format = ftlpu::MxmDataFormat::Float16;
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system, data_format);
    write_init(argv[1], system);

    auto loads = std::array<
        std::array<ftlpu::MxmControlInstruction, kBlocks>,
        kCases> {};
    auto computes = std::array<ftlpu::MxmControlInstruction, kCases> {};
    for (std::size_t test_case = 0; test_case < kCases; ++test_case) {
        for (std::size_t block = 0; block < kBlocks; ++block) {
            loads[test_case][block] =
                ftlpu::MxmControlInstruction::IWDirect16(test_case, block);
        }
        computes[test_case] = ftlpu::MxmControlInstruction::Compute(
            test_case,
            kActivationStreamBase,
            0,
            0,
            1,
            ftlpu::MxmAccumulatorDestination::Stream,
            data_format,
            ftlpu::MxmComputeMode::Vector,
            true);
    }
    build_schedule(system, loads, computes);

    const auto log_prefix = std::string {"build/cmodel_vectors/mxm_vector_"}
        + ftlpu::mxm_data_format_name(data_format);
    auto icu_log = std::ofstream(log_prefix + "_icu.log", std::ios::trunc);
    auto mem_log = std::ofstream(log_prefix + "_mem.log", std::ios::trunc);
    auto mxm_log = std::ofstream(log_prefix + "_mxm.log", std::ios::trunc);
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle)
        system.tick({.icu = &icu_log, .mem = &mem_log, .mxm = &mxm_log});

    if (!verify_output(system)) return 1;
    write_golden(argv[2], system);
    write_schedule(argv[3], loads, computes);
    std::cout << "C model MXM " << ftlpu::mxm_data_format_name(data_format)
              << " double-buffer golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
