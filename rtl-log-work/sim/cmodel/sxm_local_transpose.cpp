#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kStreams = 16;
constexpr std::size_t kTiles = 4;
constexpr std::size_t kLanes = 8;
constexpr std::size_t kCaptureCycle = 18;
constexpr std::size_t kOutputAddress = 32;
constexpr std::size_t kScheduleRecords = kStreams * 4 + 5;

using Record = std::array<std::uint32_t, 15>;

std::uint16_t matrix_value(std::size_t tile, std::size_t row, std::size_t column)
{
    return static_cast<std::uint16_t>(
        0x1000u + tile * 0x100u + row * kLanes + column);
}

ftlpu::SxmInstruction::StreamList east_streams(std::size_t first)
{
    auto result = ftlpu::SxmInstruction::StreamList {};
    for (std::size_t stream = first; stream < first + kStreams; ++stream) {
        result.push_back(ftlpu::SxmStreamId {
            ftlpu::StreamId::East(stream).packed()});
    }
    return result;
}

ftlpu::SxmInstruction::StreamList west_streams(std::size_t first)
{
    auto result = ftlpu::SxmInstruction::StreamList {};
    for (std::size_t stream = first; stream < first + kStreams; ++stream) {
        result.push_back(ftlpu::SxmStreamId {
            ftlpu::StreamId::West(stream).packed()});
    }
    return result;
}

void initialize(ftlpu::TspSliceSystem& system)
{
    for (std::size_t stream = 0; stream < kStreams; ++stream) {
        const auto row = stream / 2;
        const auto byte = stream % 2;
        for (std::size_t tile = 0; tile < kTiles; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto value = matrix_value(tile, row, lane);
                system.initialize_mem_sram_lane_byte(
                    stream, tile, 0, lane,
                    static_cast<std::uint8_t>(value >> (byte * 8)));
            }
        }
    }
}

void build_schedule(
    ftlpu::TspSliceSystem& system,
    const ftlpu::SxmInstruction& transpose,
    const ftlpu::SxmInstruction& permute)
{
    for (std::size_t stream = 0; stream < kStreams; ++stream) {
        const auto group = stream / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kCaptureCycle - (14 - group);
        const auto write_cycle = kCaptureCycle + 14 - group;
        system.icu().enqueue_mem_nop(stream, read_cycle);
        system.icu().enqueue_mem(
            stream, ftlpu::MemInstruction::Read(
                0, ftlpu::StreamId::East(stream)));
        system.icu().enqueue_mem_nop(
            stream, write_cycle - read_cycle - 1);
        system.icu().enqueue_mem(
            stream, ftlpu::MemInstruction::Write(
                kOutputAddress, ftlpu::StreamId::West(stream)));
    }

    system.icu().enqueue_sxm_transpose_nop(kCaptureCycle);
    system.icu().enqueue_sxm_transpose(transpose);
    system.icu().enqueue_sxm_permute_nop(kCaptureCycle + 1);
    system.icu().enqueue_sxm_permute(permute);
    system.icu().enqueue_sxm_permute_repeat(3, 1);
}

std::uint64_t read_tile_word(
    const ftlpu::TspSliceSystem& system,
    std::size_t stream,
    std::size_t tile,
    std::size_t address)
{
    std::uint64_t word = 0;
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        word |= static_cast<std::uint64_t>(
            system.read_mem_sram_lane_byte(stream, tile, address, lane))
            << (lane * 8);
    }
    return word;
}

void set_record_bits(
    Record& record,
    std::size_t offset,
    std::uint64_t value,
    std::size_t width)
{
    for (std::size_t bit = 0; bit < width; ++bit) {
        if (((value >> bit) & 1u) != 0)
            record[(offset + bit) / 32] |=
                std::uint32_t {1} << ((offset + bit) % 32);
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

Record sxm_record(
    std::uint8_t queue,
    const ftlpu::isa::EncodedSxmInstruction& instruction)
{
    Record result {};
    set_record_bits(result, 0, queue, 8);
    set_record_bits(result, 8, 1, 1);
    for (std::size_t word = 0; word < instruction.words.size(); ++word)
        set_record_bits(result, 41 + word * 32, instruction.words[word], 32);
    return result;
}

void write_record(std::ofstream& output, const Record& record)
{
    for (std::size_t word = record.size(); word-- > 0;)
        output << std::setw(8) << record[word];
    output << '\n';
}

void write_vectors(
    const char* path,
    const ftlpu::TspSliceSystem& system,
    std::size_t address)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open vector output");
    output << std::hex << std::setfill('0');
    for (std::size_t stream = 0; stream < kStreams; ++stream)
        for (std::size_t tile = 0; tile < kTiles; ++tile)
            output << std::setw(16)
                   << read_tile_word(system, stream, tile, address) << '\n';
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4) {
        std::cerr << "usage: sxm_local_transpose <init.hex> <golden.hex> <schedule.hex>\n";
        return 2;
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize(system);

    const auto transpose = ftlpu::SxmInstruction::Transpose(
        east_streams(0), east_streams(16));
    const auto permute = ftlpu::SxmInstruction::Permute(
        east_streams(16), west_streams(0), ftlpu::Permute320::identity_map());
    build_schedule(system, transpose, permute);

    write_vectors(argv[1], system, 0);
    auto icu_log = std::ofstream("build/cmodel_vectors/sxm_local_icu.log", std::ios::trunc);
    auto mem_log = std::ofstream("build/cmodel_vectors/sxm_local_mem.log", std::ios::trunc);
    auto sxm_log = std::ofstream("build/cmodel_vectors/sxm_local_sxm.log", std::ios::trunc);
    for (std::size_t cycle = 0; cycle < 44; ++cycle)
        system.tick({.icu = &icu_log, .mem = &mem_log, .sxm = &sxm_log});

    for (std::size_t stream = 0; stream < kStreams; ++stream) {
        const auto row = stream / 2;
        const auto byte = stream % 2;
        for (std::size_t tile = 0; tile < kTiles; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto expected = static_cast<std::uint8_t>(
                    matrix_value(tile, lane, row) >> (byte * 8));
                const auto actual = system.read_mem_sram_lane_byte(
                    stream, tile, kOutputAddress, lane);
                if (actual != expected) {
                    std::cerr << "local transpose mismatch: stream=" << stream
                              << " tile=" << tile << " lane=" << lane
                              << " actual=0x" << std::hex
                              << static_cast<unsigned>(actual)
                              << " expected=0x"
                              << static_cast<unsigned>(expected) << std::dec
                              << '\n';
                    throw std::runtime_error("C model local transpose mismatch");
                }
            }
        }
    }
    write_vectors(argv[2], system, kOutputAddress);

    std::ofstream schedule(argv[3], std::ios::trunc);
    if (!schedule) throw std::runtime_error("failed to open schedule output");
    schedule << std::hex << std::setfill('0');
    std::size_t records = 0;
    for (std::size_t stream = 0; stream < kStreams; ++stream) {
        const auto group = stream / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kCaptureCycle - (14 - group);
        const auto write_cycle = kCaptureCycle + 14 - group;
        write_record(schedule, command_record(
            static_cast<std::uint8_t>(stream),
            ftlpu::isa::encode_icu_nop(read_cycle)));
        write_record(schedule, mem_record(
            static_cast<std::uint8_t>(stream),
            ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(stream))));
        write_record(schedule, command_record(
            static_cast<std::uint8_t>(stream),
            ftlpu::isa::encode_icu_nop(write_cycle - read_cycle - 1)));
        write_record(schedule, mem_record(
            static_cast<std::uint8_t>(stream),
            ftlpu::MemInstruction::Write(
                kOutputAddress, ftlpu::StreamId::West(stream))));
        records += 4;
    }
    write_record(schedule, command_record(
        128, ftlpu::isa::encode_icu_nop(kCaptureCycle)));
    write_record(schedule, sxm_record(128, ftlpu::isa::encode_sxm_instruction(transpose)));
    write_record(schedule, command_record(
        130, ftlpu::isa::encode_icu_nop(kCaptureCycle + 1)));
    write_record(schedule, sxm_record(130, ftlpu::isa::encode_sxm_instruction(permute)));
    write_record(schedule, command_record(
        130, ftlpu::isa::encode_icu_repeat({3, 1, 0})));
    records += 5;

    if (records != kScheduleRecords)
        throw std::logic_error("internal schedule record count mismatch");
    std::cout << "C model MEM -> SXM -> MEM local transpose golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
