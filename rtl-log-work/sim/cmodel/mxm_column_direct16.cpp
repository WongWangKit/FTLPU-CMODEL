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
constexpr std::size_t kColumns = 32;
constexpr std::size_t kLoadCycle = 20;
constexpr std::size_t kComputeCycle = 160;
constexpr std::size_t kOutputWriteCycle = 177;
constexpr std::size_t kActivationAddress = 32;
constexpr std::size_t kOutputAddress = 33;
constexpr std::size_t kActivationStreamBase = 16;
constexpr std::size_t kRunCycles = 192;
constexpr std::size_t kScheduleRecords = 208;

using Record = std::array<std::uint32_t, 15>;

std::size_t source_index(std::size_t output_index)
{
    return (5 * output_index + 3) % kColumns;
}

std::size_t output_index_for_source(std::size_t input_index)
{
    return (13 * ((input_index + kColumns - 3) % kColumns)) % kColumns;
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
    for (std::size_t output = 0; output < kColumns; ++output) {
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto input = tile * kLanes + lane;
                const auto bits = ftlpu::encode_mxm_16bit(
                    input == source_index(output) ? 1.0f : 0.0f,
                    ftlpu::MxmDataFormat::BFloat16);
                for (std::size_t byte = 0; byte < 2; ++byte)
                    system.initialize_mem_sram_lane_byte(
                        byte, tile, output, lane,
                        static_cast<std::uint8_t>(bits >> (8 * byte)));
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

std::array<ftlpu::MxmControlInstruction, kColumns> make_loads()
{
    auto loads = std::array<ftlpu::MxmControlInstruction, kColumns> {};
    for (std::size_t output = 0; output < kColumns; ++output)
        loads[output] = ftlpu::MxmControlInstruction::IWColumnDirect16(
            0, output / kLanes, output % kLanes);
    return loads;
}

ftlpu::MxmControlInstruction make_compute()
{
    return ftlpu::MxmControlInstruction::Compute(
        0, kActivationStreamBase, 0, 0, 1,
        ftlpu::MxmAccumulatorDestination::Stream,
        ftlpu::MxmDataFormat::BFloat16,
        ftlpu::MxmComputeMode::Vector, true);
}

void build_schedule(
    ftlpu::TspSliceSystem& system,
    const std::array<ftlpu::MxmControlInstruction, kColumns>& loads,
    const ftlpu::MxmControlInstruction& compute)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        system.icu().enqueue_mem_nop(byte, kLoadCycle - 15);
        for (std::size_t output = 0; output < kColumns; ++output) {
            system.icu().enqueue_mem(
                byte,
                ftlpu::MemInstruction::Read(
                    output, ftlpu::StreamId::East(byte)));
            system.icu().enqueue_mem_repeat(byte, kBlocks - 1, 1, 1);
        }
        system.icu().enqueue_mem_nop(
            byte, kOutputWriteCycle - (kLoadCycle - 15 + kColumns * 4));
        system.icu().enqueue_mem(
            byte,
            ftlpu::MemInstruction::Write(
                kOutputAddress, ftlpu::StreamId::West(byte)));
    }
    for (std::size_t byte = 2; byte < 4; ++byte) {
        system.icu().enqueue_mem_nop(byte, kOutputWriteCycle);
        system.icu().enqueue_mem(
            byte,
            ftlpu::MemInstruction::Write(
                kOutputAddress, ftlpu::StreamId::West(byte)));
    }

    const auto activation_read_cycle = kComputeCycle - 11;
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto column = kActivationStreamBase + byte;
        system.icu().enqueue_mem_nop(column, activation_read_cycle);
        system.icu().enqueue_mem(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress, ftlpu::StreamId::East(column)));
    }

    system.icu().enqueue_mxm_load_nop(0, kLoadCycle);
    for (std::size_t output = 0; output < kColumns; ++output) {
        system.icu().enqueue_mxm(0, loads[output]);
        if (output + 1 != kColumns)
            system.icu().enqueue_mxm_load_nop(0, 3);
    }
    system.icu().enqueue_mxm_compute_nop(0, kComputeCycle);
    system.icu().enqueue_mxm(0, compute);
}

void write_init(const char* path, const ftlpu::TspSliceSystem& system)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open column IW init");
    output << std::hex << std::setfill('0');
    for (std::size_t address = 0; address < kColumns; ++address)
        for (std::size_t byte = 0; byte < 2; ++byte)
            for (std::size_t tile = 0; tile < kBlocks; ++tile)
                output << std::setw(16)
                       << read_word(system, byte, address, tile) << '\n';
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
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            std::uint32_t actual = 0;
            for (std::size_t byte = 0; byte < 4; ++byte)
                actual |= static_cast<std::uint32_t>(
                    system.read_mem_sram_lane_byte(
                        byte, block, kOutputAddress, lane)) << (8 * byte);
            const auto expected = std::bit_cast<std::uint32_t>(
                static_cast<float>(block * kLanes + lane + 1));
            if (actual != expected) {
                std::cerr << "column IW mismatch output="
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
    if (!output) throw std::runtime_error("failed to open column IW golden");
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

Record mem_record(std::uint8_t queue, const ftlpu::MemInstruction& instruction)
{
    Record result {};
    set_bits(result, 0, queue, 8);
    set_bits(result, 8, 1, 1);
    set_bits(result, 41, ftlpu::isa::encode_mem_instruction(instruction), 47);
    return result;
}

Record mxm_record(
    std::uint8_t queue, const ftlpu::MxmControlInstruction& instruction)
{
    Record result {};
    set_bits(result, 0, queue, 8);
    set_bits(result, 8, 1, 1);
    set_bits(result, 41, ftlpu::isa::encode_mxm_instruction(instruction), 48);
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
    const std::array<ftlpu::MxmControlInstruction, kColumns>& loads,
    const ftlpu::MxmControlInstruction& compute)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open column IW schedule");
    output << std::hex << std::setfill('0');
    std::size_t records = 0;

    for (std::size_t byte = 0; byte < 2; ++byte) {
        write_record(output, command_record(
            byte, ftlpu::isa::encode_icu_nop(kLoadCycle - 15)));
        ++records;
        for (std::size_t address = 0; address < kColumns; ++address) {
            write_record(output, mem_record(
                byte,
                ftlpu::MemInstruction::Read(
                    address, ftlpu::StreamId::East(byte))));
            write_record(output, command_record(
                byte, ftlpu::isa::encode_icu_repeat({kBlocks - 1, 1, 1})));
            records += 2;
        }
        write_record(output, command_record(
            byte,
            ftlpu::isa::encode_icu_nop(
                kOutputWriteCycle - (kLoadCycle - 15 + kColumns * 4))));
        write_record(output, mem_record(
            byte,
            ftlpu::MemInstruction::Write(
                kOutputAddress, ftlpu::StreamId::West(byte))));
        records += 2;
    }
    for (std::size_t byte = 2; byte < 4; ++byte) {
        write_record(output, command_record(
            byte, ftlpu::isa::encode_icu_nop(kOutputWriteCycle)));
        write_record(output, mem_record(
            byte,
            ftlpu::MemInstruction::Write(
                kOutputAddress, ftlpu::StreamId::West(byte))));
        records += 2;
    }
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto column = kActivationStreamBase + byte;
        write_record(output, command_record(
            column, ftlpu::isa::encode_icu_nop(kComputeCycle - 11)));
        write_record(output, mem_record(
            column,
            ftlpu::MemInstruction::Read(
                kActivationAddress, ftlpu::StreamId::East(column))));
        records += 2;
    }

    write_record(output, command_record(
        104, ftlpu::isa::encode_icu_nop(kLoadCycle)));
    ++records;
    for (std::size_t index = 0; index < kColumns; ++index) {
        write_record(output, mxm_record(104, loads[index]));
        ++records;
        if (index + 1 != kColumns) {
            write_record(output, command_record(
                104, ftlpu::isa::encode_icu_nop(3)));
            ++records;
        }
    }
    write_record(output, command_record(
        108, ftlpu::isa::encode_icu_nop(kComputeCycle)));
    write_record(output, mxm_record(108, compute));
    records += 2;
    if (records != kScheduleRecords)
        throw std::logic_error("column IW schedule record count mismatch");
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4) {
        std::cerr << "usage: mxm_column_direct16 <init> <golden> <schedule>\n";
        return 2;
    }
    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system);
    write_init(argv[1], system);
    const auto loads = make_loads();
    const auto compute = make_compute();
    build_schedule(system, loads, compute);
    for (std::size_t cycle = 0; cycle < kRunCycles; ++cycle)
        system.tick({});
    if (!verify_output(system)) return 1;
    write_golden(argv[2], system);
    write_schedule(argv[3], loads, compute);
    std::cout << "C model MXM Column Direct16 golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
