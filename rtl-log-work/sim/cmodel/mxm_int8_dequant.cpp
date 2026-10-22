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
constexpr std::size_t kWeightStreams = 8;
constexpr std::size_t kActivationStreamBase = 16;
constexpr std::size_t kActivationAddress = 4;
constexpr std::size_t kOutputAddress = 5;
constexpr std::size_t kLoadCycle = 20;
constexpr std::size_t kComputeCycle = 30;
constexpr std::size_t kOutputWriteCycle = 47;
constexpr std::size_t kRunCycles = 64;
constexpr std::size_t kScheduleRecords = 48;
constexpr std::array<std::int8_t, kBlocks> kQuantizedWeight {2, -3, 4, -5};
constexpr std::array<float, kBlocks> kScale {0.5f, -0.25f, 0.125f, -0.5f};

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
        for (std::size_t column = 0; column < kWeightStreams; ++column) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    const auto input = tile * kLanes + lane;
                    const auto output = block * kLanes + column;
                    const auto quantized = input == source_index(output)
                        ? kQuantizedWeight[block]
                        : 0;
                    system.initialize_mem_sram_lane_byte(
                        column, tile, block, lane,
                        static_cast<std::uint8_t>(quantized));
                }
            }
        }
    }

    for (std::size_t tile = 0; tile < kBlocks; ++tile) {
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const auto input = tile * kLanes + lane;
            const auto bits = ftlpu::encode_mxm_16bit(
                static_cast<float>(output_index_for_source(input) + 1),
                ftlpu::MxmDataFormat::BFloat16);
            for (std::size_t byte = 0; byte < 2; ++byte)
                system.initialize_mem_sram_lane_byte(
                    kActivationStreamBase + byte, tile,
                    kActivationAddress, lane,
                    static_cast<std::uint8_t>(bits >> (8 * byte)));
        }
    }
}

auto make_loads()
{
    auto loads = std::array<ftlpu::MxmControlInstruction, kBlocks> {};
    for (std::size_t block = 0; block < kBlocks; ++block)
        loads[block] = ftlpu::MxmControlInstruction::IW(0, block);
    return loads;
}

auto make_scales()
{
    auto scales = std::array<ftlpu::MxmDequantInstruction, kBlocks> {};
    for (std::size_t block = 0; block < kBlocks; ++block)
        scales[block] = ftlpu::MxmDequantInstruction::Scale(kScale[block]);
    return scales;
}

auto make_compute()
{
    return ftlpu::MxmControlInstruction::Compute(
        0, kActivationStreamBase, 0, 0, 1,
        ftlpu::MxmAccumulatorDestination::Stream,
        ftlpu::MxmDataFormat::BFloat16,
        ftlpu::MxmComputeMode::Vector, true);
}

void build_schedule(
    ftlpu::TspSliceSystem& system,
    const std::array<ftlpu::MxmControlInstruction, kBlocks>& loads,
    const std::array<ftlpu::MxmDequantInstruction, kBlocks>& scales,
    const ftlpu::MxmControlInstruction& compute)
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
            system.icu().enqueue_mem_nop(
                column, kOutputWriteCycle - (read_cycle + kBlocks));
            system.icu().enqueue_mem(
                column,
                ftlpu::MemInstruction::Write(
                    kOutputAddress, ftlpu::StreamId::West(column)));
        }
    }

    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto column = kActivationStreamBase + byte;
        system.icu().enqueue_mem_nop(column, kComputeCycle - 11);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress, ftlpu::StreamId::East(column)));
    }

    system.icu().enqueue_mxm_load_nop(0, kLoadCycle);
    system.icu().enqueue_mxm_dequant_nop(0, kLoadCycle);
    for (std::size_t block = 0; block < kBlocks; ++block) {
        system.icu().enqueue_mxm(0, loads[block]);
        system.icu().enqueue_mxm_dequant(0, scales[block]);
    }
    system.icu().enqueue_mxm_compute_nop(0, kComputeCycle);
    system.icu().enqueue_mxm(0, compute);
}

void write_init(const char* path, const ftlpu::TspSliceSystem& system)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open INT8 init");
    output << std::hex << std::setfill('0');
    for (std::size_t block = 0; block < kBlocks; ++block)
        for (std::size_t column = 0; column < kWeightStreams; ++column)
            for (std::size_t tile = 0; tile < kBlocks; ++tile)
                output << std::setw(16)
                       << read_word(system, column, block, tile) << '\n';
    for (std::size_t byte = 0; byte < 2; ++byte)
        for (std::size_t tile = 0; tile < kBlocks; ++tile)
            output << std::setw(16)
                   << read_word(
                          system, kActivationStreamBase + byte,
                          kActivationAddress, tile)
                   << '\n';
}

bool verify_output(const ftlpu::TspSliceSystem& system)
{
    for (std::size_t block = 0; block < kBlocks; ++block) {
        const auto weight = static_cast<float>(kQuantizedWeight[block])
            * kScale[block];
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            std::uint32_t actual = 0;
            for (std::size_t byte = 0; byte < 4; ++byte)
                actual |= static_cast<std::uint32_t>(
                    system.read_mem_sram_lane_byte(
                        byte, block, kOutputAddress, lane)) << (8 * byte);
            const auto expected = std::bit_cast<std::uint32_t>(
                weight * static_cast<float>(block * kLanes + lane + 1));
            if (actual != expected) {
                std::cerr << "INT8 dequant mismatch output="
                          << block * kLanes + lane << " actual=0x"
                          << std::hex << actual << " expected=0x" << expected
                          << std::dec << '\n';
                return false;
            }
        }
    }
    return true;
}

void write_golden(const char* path, const ftlpu::TspSliceSystem& system)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open INT8 golden");
    output << std::hex << std::setfill('0');
    for (std::size_t byte = 0; byte < 4; ++byte)
        for (std::size_t tile = 0; tile < kBlocks; ++tile)
            output << std::setw(16)
                   << read_word(system, byte, kOutputAddress, tile) << '\n';
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
    const std::array<ftlpu::MxmDequantInstruction, kBlocks>& scales,
    const ftlpu::MxmControlInstruction& compute)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open INT8 schedule");
    output << std::hex << std::setfill('0');
    std::size_t records = 0;
    for (std::size_t column = 0; column < kWeightStreams; ++column) {
        const auto group = column / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kLoadCycle - (15 - group);
        write_record(output, command_record(
            column, ftlpu::isa::encode_icu_nop(read_cycle)));
        write_record(output, instruction_record(
            column,
            ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Read(
                    0, ftlpu::StreamId::East(column))),
            47));
        write_record(output, command_record(
            column, ftlpu::isa::encode_icu_repeat({kBlocks - 1, 1, 1})));
        records += 3;
        if (column < 4) {
            write_record(output, command_record(
                column,
                ftlpu::isa::encode_icu_nop(
                    kOutputWriteCycle - (read_cycle + kBlocks))));
            write_record(output, instruction_record(
                column,
                ftlpu::isa::encode_mem_instruction(
                    ftlpu::MemInstruction::Write(
                        kOutputAddress, ftlpu::StreamId::West(column))),
                47));
            records += 2;
        }
    }
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto column = kActivationStreamBase + byte;
        write_record(output, command_record(
            column, ftlpu::isa::encode_icu_nop(kComputeCycle - 11)));
        write_record(output, instruction_record(
            column,
            ftlpu::isa::encode_mem_instruction(
                ftlpu::MemInstruction::Read(
                    kActivationAddress, ftlpu::StreamId::East(column))),
            47));
        records += 2;
    }
    write_record(output, command_record(
        104, ftlpu::isa::encode_icu_nop(kLoadCycle)));
    write_record(output, command_record(
        106, ftlpu::isa::encode_icu_nop(kLoadCycle)));
    records += 2;
    for (std::size_t block = 0; block < kBlocks; ++block) {
        write_record(output, instruction_record(
            104, ftlpu::isa::encode_mxm_instruction(loads[block]), 48));
        write_record(output, instruction_record(
            106,
            ftlpu::isa::encode_mxm_dequant_instruction(scales[block]), 16));
        records += 2;
    }
    write_record(output, command_record(
        108, ftlpu::isa::encode_icu_nop(kComputeCycle)));
    write_record(output, instruction_record(
        108, ftlpu::isa::encode_mxm_instruction(compute), 48));
    records += 2;
    if (records != kScheduleRecords)
        throw std::logic_error("INT8 dequant schedule record count mismatch");
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4) {
        std::cerr << "usage: mxm_int8_dequant <init> <golden> <schedule>\n";
        return 2;
    }
    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system);
    write_init(argv[1], system);
    const auto loads = make_loads();
    const auto scales = make_scales();
    const auto compute = make_compute();
    build_schedule(system, loads, scales, compute);
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle)
        system.tick({});
    if (!verify_output(system)) return 1;
    write_golden(argv[2], system);
    write_schedule(argv[3], loads, scales, compute);
    std::cout << "C model MXM INT8 dequant golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
