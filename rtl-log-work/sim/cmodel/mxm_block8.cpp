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

constexpr std::size_t kBlocks = 4;
constexpr std::size_t kLanes = 8;
constexpr std::size_t kWeightStreams = 16;
constexpr std::size_t kActivationStreamBase = 16;
constexpr std::size_t kActivationAddress = 4;
constexpr std::size_t kStreamOutputAddress = 5;
constexpr std::array<std::size_t, 3> kReadOutputAddress {6, 7, 8};
constexpr std::size_t kAccumulatorAddress = 21;
constexpr std::size_t kLoadCycle = 20;
constexpr std::array<std::size_t, 2> kComputeCycle {30, 60};
constexpr std::array<std::size_t, 3> kReadCycle {100, 130, 160};
constexpr std::size_t kRunCycles = 180;
constexpr std::size_t kScheduleRecords = 351;

using Record = std::array<std::uint32_t, 15>;

std::size_t source_index(std::size_t output)
{
    return (5 * output + 3) % 32;
}

std::size_t output_index_for_source(std::size_t input)
{
    return (13 * ((input + 29) % 32)) % 32;
}

std::uint64_t read_word(
    const ftlpu::TspSliceSystem& system,
    std::size_t column,
    std::size_t address,
    std::size_t tile)
{
    std::uint64_t result = 0;
    for (std::size_t lane = 0; lane < kLanes; ++lane)
        result |= static_cast<std::uint64_t>(
            system.read_mem_sram_lane_byte(column, tile, address, lane))
            << (lane * 8);
    return result;
}

void initialize_inputs(ftlpu::TspSliceSystem& system)
{
    for (std::size_t block = 0; block < kBlocks; ++block) {
        for (std::size_t stream = 0; stream < kWeightStreams; ++stream) {
            const auto local_column = stream / 2;
            const auto byte = stream % 2;
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    const auto input = tile * kLanes + lane;
                    const auto output = block * kLanes + local_column;
                    const auto bits = ftlpu::encode_mxm_16bit(
                        input == source_index(output) ? 1.0f : 0.0f,
                        ftlpu::MxmDataFormat::BFloat16);
                    system.initialize_mem_sram_lane_byte(
                        stream, tile, block, lane,
                        static_cast<std::uint8_t>(bits >> (8 * byte)));
                }
            }
        }
    }

    for (std::size_t output_row = 0; output_row < 8; ++output_row) {
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto input = tile * kLanes + lane;
                const auto value = static_cast<float>(
                    output_row * 32 + output_index_for_source(input) + 1);
                const auto bits = ftlpu::encode_mxm_16bit(
                    value, ftlpu::MxmDataFormat::BFloat16);
                for (std::size_t byte = 0; byte < 2; ++byte)
                    system.initialize_mem_sram_lane_byte(
                        kActivationStreamBase + output_row*2 + byte,
                        tile, kActivationAddress, lane,
                        static_cast<std::uint8_t>(bits >> (8 * byte)));
            }
        }
    }
}

auto make_loads()
{
    auto loads = std::array<ftlpu::MxmControlInstruction, kBlocks> {};
    for (std::size_t block = 0; block < kBlocks; ++block)
        loads[block] = ftlpu::MxmControlInstruction::IWDirect16(0, block);
    return loads;
}

auto make_compute(ftlpu::MxmAccumulatorDestination destination)
{
    const auto address = destination == ftlpu::MxmAccumulatorDestination::Sram
        ? kAccumulatorAddress - 1 : kAccumulatorAddress;
    return ftlpu::MxmControlInstruction::Compute(
        0, kActivationStreamBase, 0, address, 1,
        destination,
        ftlpu::MxmDataFormat::BFloat16,
        ftlpu::MxmComputeMode::Block8, true);
}

void build_schedule(
    ftlpu::TspSliceSystem& system,
    const std::array<ftlpu::MxmControlInstruction, kBlocks>& loads,
    const std::array<ftlpu::MxmControlInstruction, 2>& computes,
    const std::array<ftlpu::MxmControlInstruction, 3>& reads)
{
    for (std::size_t column = 0; column < kWeightStreams; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kLoadCycle - (15 - group);
        const auto output_cycle = kComputeCycle[0] + 3 + (14 - group);
        system.icu().enqueue_mem_nop(column, read_cycle);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(column)));
        system.icu().enqueue_mem_repeat(column, kBlocks - 1, 1, 1);
        system.icu().enqueue_mem_nop(
            column, output_cycle - (read_cycle + kBlocks));
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Write(
                kStreamOutputAddress, ftlpu::StreamId::West(column)));
    }
    for (std::size_t column = kActivationStreamBase;
         column < kActivationStreamBase + 16;
         ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        system.icu().enqueue_mem_nop(
            column, kComputeCycle[0] - (15 - group));
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress, ftlpu::StreamId::East(column)));
        system.icu().enqueue_mem_nop(
            column, kComputeCycle[1] - kComputeCycle[0] - 1);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress, ftlpu::StreamId::East(column)));
    }
    system.icu().enqueue_mxm_load_nop(0, kLoadCycle);
    for (const auto& load : loads)
        system.icu().enqueue_mxm(0, load);
    system.icu().enqueue_mxm_compute_nop(0, kComputeCycle[0]);
    system.icu().enqueue_mxm(0, computes[0]);
    system.icu().enqueue_mxm_compute_nop(
        0, kComputeCycle[1] - kComputeCycle[0] - 1);
    system.icu().enqueue_mxm(0, computes[1]);
    auto cursor = kComputeCycle[1] + 1;
    for (std::size_t index = 0; index < reads.size(); ++index) {
        system.icu().enqueue_mxm_compute_nop(
            0, kReadCycle[index] - cursor);
        system.icu().enqueue_mxm(0, reads[index]);
        cursor = kReadCycle[index] + 1;
    }

    for (std::size_t column = 0; column < 32; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        auto mem_cursor = column < kWeightStreams
            ? kComputeCycle[0] + 3 + (14 - group) + 1
            : kComputeCycle[1] - (15 - group) + 1;
        for (std::size_t index = 0; index < reads.size(); ++index) {
            const auto output_cycle = kReadCycle[index] + (14 - group);
            system.icu().enqueue_mem_nop(
                column, output_cycle - mem_cursor);
            system.icu().enqueue_mem(
                column,
                ftlpu::MemInstruction::Write(
                    kReadOutputAddress[index],
                    ftlpu::StreamId::West(column)));
            mem_cursor = output_cycle + 1;
        }
    }
}

void write_init(const char* path, const ftlpu::TspSliceSystem& system)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open Block8 init");
    output << std::hex << std::setfill('0');
    for (std::size_t block = 0; block < kBlocks; ++block)
        for (std::size_t column = 0; column < kWeightStreams; ++column)
            for (std::size_t tile = 0; tile < kBlocks; ++tile)
                output << std::setw(16)
                       << read_word(system, column, block, tile) << '\n';
    for (std::size_t column = kActivationStreamBase;
         column < kActivationStreamBase + 16;
         ++column)
        for (std::size_t tile = 0; tile < kBlocks; ++tile)
            output << std::setw(16)
                   << read_word(system, column, kActivationAddress, tile)
                   << '\n';
}

bool verify_output(const ftlpu::TspSliceSystem& system)
{
    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t output = 0; output < 32; ++output) {
            std::uint16_t actual = 0;
            for (std::size_t byte = 0; byte < 2; ++byte)
                actual |= static_cast<std::uint16_t>(
                    system.read_mem_sram_lane_byte(
                        row*2+byte, output/8,
                        kStreamOutputAddress, output%8))
                    << (8 * byte);
            const auto expected = ftlpu::encode_mxm_16bit(
                static_cast<float>(row*32+output+1),
                ftlpu::MxmDataFormat::BFloat16);
            if (actual != expected) {
                std::cerr << "Block8 mismatch row=" << row
                          << " output=" << output << " actual=0x"
                          << std::hex << actual << " expected=0x" << expected
                          << std::dec << '\n';
                return false;
            }
        }
    }
    for (std::size_t read = 0; read < kReadOutputAddress.size(); ++read) {
        for (std::size_t row = 0; row < 8; ++row) {
            for (std::size_t output = 0; output < 32; ++output) {
                std::uint32_t actual = 0;
                for (std::size_t byte = 0; byte < 4; ++byte)
                    actual |= static_cast<std::uint32_t>(
                        system.read_mem_sram_lane_byte(
                            row*4+byte, output/8,
                            kReadOutputAddress[read], output%8))
                        << (8 * byte);
                const auto expected = read < 2
                    ? std::bit_cast<std::uint32_t>(
                        static_cast<float>(row*32+output+1))
                    : 0u;
                if (actual != expected) {
                    std::cerr << "Block8 accumulator mismatch read=" << read
                              << " row=" << row << " output=" << output
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
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open Block8 golden");
    output << std::hex << std::setfill('0');
    for (std::size_t column = 0; column < 16; ++column)
        for (std::size_t tile = 0; tile < kBlocks; ++tile)
            output << std::setw(16)
                   << read_word(system, column, kStreamOutputAddress, tile)
                   << '\n';
    for (const auto address : kReadOutputAddress)
        for (std::size_t column = 0; column < 32; ++column)
            for (std::size_t tile = 0; tile < kBlocks; ++tile)
                output << std::setw(16)
                       << read_word(system, column, address, tile) << '\n';
}

void set_bits(
    Record& record, std::size_t offset, std::uint64_t value, std::size_t width)
{
    for (std::size_t bit = 0; bit < width; ++bit)
        if ((value >> bit) & 1u)
            record[(offset + bit) / 32] |=
                std::uint32_t {1} << ((offset + bit) % 32);
}

Record command_record(std::uint8_t queue, std::uint32_t command)
{
    Record result {};
    set_bits(result, 0, queue, 8);
    set_bits(result, 9, command, 32);
    return result;
}

Record instruction_record(
    std::uint8_t queue, std::uint64_t instruction, std::size_t width)
{
    Record result {};
    set_bits(result, 0, queue, 8);
    set_bits(result, 8, 1, 1);
    set_bits(result, 41, instruction, width);
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
    const std::array<ftlpu::MxmControlInstruction, kBlocks>& loads,
    const std::array<ftlpu::MxmControlInstruction, 2>& computes,
    const std::array<ftlpu::MxmControlInstruction, 3>& reads)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open Block8 schedule");
    output << std::hex << std::setfill('0');
    std::size_t records = 0;
    for (std::size_t column = 0; column < kWeightStreams; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kLoadCycle - (15 - group);
        const auto output_cycle = kComputeCycle[0] + 3 + (14 - group);
        write_record(output, command_record(
            column, ftlpu::isa::encode_icu_nop(read_cycle)));
        write_record(output, instruction_record(
            column,
            ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Read(
                    0, ftlpu::StreamId::East(column))), 47));
        write_record(output, command_record(
            column, ftlpu::isa::encode_icu_repeat({kBlocks - 1, 1, 1})));
        write_record(output, command_record(
            column,
            ftlpu::isa::encode_icu_nop(
                output_cycle - (read_cycle + kBlocks))));
        write_record(output, instruction_record(
            column,
            ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Write(
                    kStreamOutputAddress,
                    ftlpu::StreamId::West(column))), 47));
        records += 5;
    }
    for (std::size_t column = kActivationStreamBase;
         column < kActivationStreamBase + 16;
         ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        write_record(output, command_record(
            column,
            ftlpu::isa::encode_icu_nop(kComputeCycle[0] - (15 - group))));
        write_record(output, instruction_record(
            column,
            ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Read(
                    kActivationAddress, ftlpu::StreamId::East(column))), 47));
        write_record(output, command_record(
            column,
            ftlpu::isa::encode_icu_nop(
                kComputeCycle[1] - kComputeCycle[0] - 1)));
        write_record(output, instruction_record(
            column,
            ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Read(
                    kActivationAddress, ftlpu::StreamId::East(column))), 47));
        records += 4;
    }
    write_record(output, command_record(
        104, ftlpu::isa::encode_icu_nop(kLoadCycle)));
    ++records;
    for (const auto& load : loads) {
        write_record(output, instruction_record(
            104, ftlpu::isa::encode_mxm_instruction(load), 48));
        ++records;
    }
    write_record(output, command_record(
        108, ftlpu::isa::encode_icu_nop(kComputeCycle[0])));
    write_record(output, instruction_record(
        108, ftlpu::isa::encode_mxm_instruction(computes[0]), 48));
    write_record(output, command_record(
        108, ftlpu::isa::encode_icu_nop(
            kComputeCycle[1] - kComputeCycle[0] - 1)));
    write_record(output, instruction_record(
        108, ftlpu::isa::encode_mxm_instruction(computes[1]), 48));
    records += 4;
    auto cursor = kComputeCycle[1] + 1;
    for (std::size_t index = 0; index < reads.size(); ++index) {
        write_record(output, command_record(
            108, ftlpu::isa::encode_icu_nop(
                kReadCycle[index] - cursor)));
        write_record(output, instruction_record(
            108, ftlpu::isa::encode_mxm_instruction(reads[index]), 48));
        records += 2;
        cursor = kReadCycle[index] + 1;
    }
    for (std::size_t column = 0; column < 32; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        auto mem_cursor = column < kWeightStreams
            ? kComputeCycle[0] + 3 + (14 - group) + 1
            : kComputeCycle[1] - (15 - group) + 1;
        for (std::size_t index = 0; index < reads.size(); ++index) {
            const auto output_cycle = kReadCycle[index] + (14 - group);
            write_record(output, command_record(
                column, ftlpu::isa::encode_icu_nop(
                    output_cycle - mem_cursor)));
            write_record(output, instruction_record(
                column,
                ftlpu::isa::encode_mem_instruction(
                    ftlpu::MemInstruction::Write(
                        kReadOutputAddress[index],
                        ftlpu::StreamId::West(column))), 47));
            records += 2;
            mem_cursor = output_cycle + 1;
        }
    }
    if (records != kScheduleRecords)
        throw std::logic_error("Block8 schedule record count mismatch");
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4) {
        std::cerr << "usage: mxm_block8 <init> <golden> <schedule>\n";
        return 2;
    }
    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system);
    write_init(argv[1], system);
    const auto loads = make_loads();
    const auto computes = std::array {
        make_compute(ftlpu::MxmAccumulatorDestination::Stream),
        make_compute(ftlpu::MxmAccumulatorDestination::Sram),
    };
    const auto reads = std::array {
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kAccumulatorAddress, 0, false, ftlpu::MxmComputeMode::Block8),
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kAccumulatorAddress, 0, true, ftlpu::MxmComputeMode::Block8),
        ftlpu::MxmControlInstruction::AccumulatorRead(
            kAccumulatorAddress, 0, false, ftlpu::MxmComputeMode::Block8),
    };
    build_schedule(system, loads, computes, reads);
    auto icu_log = std::ofstream(
        "build/cmodel_vectors/mxm_block8_icu.log", std::ios::trunc);
    auto mem_log = std::ofstream(
        "build/cmodel_vectors/mxm_block8_mem.log", std::ios::trunc);
    auto mxm_log = std::ofstream(
        "build/cmodel_vectors/mxm_block8_mxm.log", std::ios::trunc);
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle)
        system.tick({.icu = &icu_log, .mem = &mem_log, .mxm = &mxm_log});
    if (!verify_output(system)) return 1;
    write_golden(argv[2], system);
    write_schedule(argv[3], loads, computes, reads);
    std::cout << "C model MXM Block8 stream/SRAM/read-clear golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
