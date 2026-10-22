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

constexpr std::size_t kMatrixSize = ftlpu::hw::kPhysicalVectorBytes;
constexpr std::size_t kBlockSize = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kBlocks = ftlpu::hw::kTileRows;
constexpr std::size_t kStreams = 2 * kBlockSize;
constexpr std::size_t kInputBeats = kBlocks;
constexpr std::size_t kCaptureCycle = 18;
constexpr std::size_t kOutputAddress = 32;
constexpr std::size_t kWaveCount = kInputBeats + kBlocks - 1;
constexpr std::size_t kVectorWords = kInputBeats * kStreams * kBlocks;
constexpr std::size_t kScheduleRecords = kStreams * 6 + 3 + 1 + kWaveCount;

static_assert(kMatrixSize == 32);
static_assert(kVectorWords == 256);
static_assert(kScheduleRecords == 107);

using Record = std::array<std::uint32_t, 15>;

std::uint16_t matrix_value(std::size_t row, std::size_t column)
{
    return static_cast<std::uint16_t>(0x1000u + row * kMatrixSize + column);
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

ftlpu::SxmInstruction::PermuteMap wavefront_map(std::size_t wave)
{
    auto map = ftlpu::Permute320::identity_map();
    for (std::size_t destination_tile = 0;
         destination_tile < kBlocks; ++destination_tile) {
        const auto source_tile =
            (wave + kBlocks - destination_tile) % kBlocks;
        for (std::size_t lane = 0; lane < kBlockSize; ++lane) {
            map[destination_tile * kBlockSize + lane] =
                source_tile * kBlockSize + lane;
        }
    }
    return map;
}

void initialize_matrix(ftlpu::TspSliceSystem& system)
{
    for (std::size_t block_row = 0; block_row < kBlocks; ++block_row) {
        for (std::size_t block_column = 0;
             block_column < kBlocks; ++block_column) {
            for (std::size_t local_row = 0;
                 local_row < kBlockSize; ++local_row) {
                for (std::size_t local_column = 0;
                     local_column < kBlockSize; ++local_column) {
                    const auto value = matrix_value(
                        block_row * kBlockSize + local_row,
                        block_column * kBlockSize + local_column);
                    for (std::size_t byte = 0; byte < 2; ++byte) {
                        system.initialize_mem_sram_lane_byte(
                            2 * local_row + byte,
                            block_column,
                            block_row,
                            local_column,
                            static_cast<std::uint8_t>(value >> (byte * 8)));
                    }
                }
            }
        }
    }
}

void build_schedule(
    ftlpu::TspSliceSystem& system,
    const ftlpu::SxmInstruction& transpose,
    const std::array<ftlpu::SxmInstruction, kWaveCount>& permutes)
{
    for (std::size_t stream = 0; stream < kStreams; ++stream) {
        const auto group = stream / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kCaptureCycle - (14 - group);
        const auto write_cycle = kCaptureCycle + 14 - group;

        system.icu().enqueue_mem_nop(stream, read_cycle);
        system.icu().enqueue_mem(
            stream,
            ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(stream)));
        system.icu().enqueue_mem_repeat(stream, kInputBeats - 1, 1, 1);

        system.icu().enqueue_mem_nop(
            stream, write_cycle - (read_cycle + kInputBeats));
        system.icu().enqueue_mem(
            stream,
            ftlpu::MemInstruction::Write(
                kOutputAddress, ftlpu::StreamId::West(stream)));
        system.icu().enqueue_mem_repeat(stream, kInputBeats - 1, 1, 1);
    }

    system.icu().enqueue_sxm_transpose_nop(kCaptureCycle);
    system.icu().enqueue_sxm_transpose(transpose);
    system.icu().enqueue_sxm_transpose_repeat(kInputBeats - 1, 1);

    system.icu().enqueue_sxm_permute_nop(kCaptureCycle + 1);
    for (const auto& instruction : permutes)
        system.icu().enqueue_sxm_permute(instruction);
}

bool verify_transpose(const ftlpu::TspSliceSystem& system)
{
    for (std::size_t row = 0; row < kMatrixSize; ++row) {
        for (std::size_t column = 0; column < kMatrixSize; ++column) {
            std::uint16_t actual = 0;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                actual |= static_cast<std::uint16_t>(
                    system.read_mem_sram_lane_byte(
                        2 * (row % kBlockSize) + byte,
                        column / kBlockSize,
                        kOutputAddress + row / kBlockSize,
                        column % kBlockSize)) << (8 * byte);
            }
            const auto expected = matrix_value(column, row);
            if (actual != expected) {
                std::cerr << "wavefront mismatch at (" << row << ',' << column
                          << "): actual=0x" << std::hex << actual
                          << " expected=0x" << expected << std::dec << '\n';
                return false;
            }
        }
    }
    return true;
}

std::uint64_t read_tile_word(
    const ftlpu::TspSliceSystem& system,
    std::size_t stream,
    std::size_t tile,
    std::size_t address)
{
    std::uint64_t word = 0;
    for (std::size_t lane = 0; lane < kBlockSize; ++lane) {
        word |= static_cast<std::uint64_t>(
            system.read_mem_sram_lane_byte(stream, tile, address, lane))
            << (lane * 8);
    }
    return word;
}

void write_vectors(
    const char* path,
    const ftlpu::TspSliceSystem& system,
    std::size_t address_base)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open wavefront vector output");
    output << std::hex << std::setfill('0');
    for (std::size_t beat = 0; beat < kInputBeats; ++beat) {
        for (std::size_t stream = 0; stream < kStreams; ++stream) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                output << std::setw(16)
                       << read_tile_word(
                              system, stream, tile, address_base + beat)
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

void write_schedule(
    const char* path,
    const ftlpu::SxmInstruction& transpose,
    const std::array<ftlpu::SxmInstruction, kWaveCount>& permutes)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open wavefront schedule output");
    output << std::hex << std::setfill('0');
    std::size_t records = 0;

    for (std::size_t stream = 0; stream < kStreams; ++stream) {
        const auto group = stream / ftlpu::hw::kMemSlicesPerGroup;
        const auto read_cycle = kCaptureCycle - (14 - group);
        const auto write_cycle = kCaptureCycle + 14 - group;
        const auto queue = static_cast<std::uint8_t>(stream);

        write_record(output, command_record(
            queue, ftlpu::isa::encode_icu_nop(read_cycle)));
        write_record(output, mem_record(
            queue,
            ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(stream))));
        write_record(output, command_record(
            queue,
            ftlpu::isa::encode_icu_repeat({kInputBeats - 1, 1, 1})));
        write_record(output, command_record(
            queue,
            ftlpu::isa::encode_icu_nop(
                write_cycle - (read_cycle + kInputBeats))));
        write_record(output, mem_record(
            queue,
            ftlpu::MemInstruction::Write(
                kOutputAddress, ftlpu::StreamId::West(stream))));
        write_record(output, command_record(
            queue,
            ftlpu::isa::encode_icu_repeat({kInputBeats - 1, 1, 1})));
        records += 6;
    }

    write_record(output, command_record(
        128, ftlpu::isa::encode_icu_nop(kCaptureCycle)));
    write_record(output, sxm_record(
        128, ftlpu::isa::encode_sxm_instruction(transpose)));
    write_record(output, command_record(
        128,
        ftlpu::isa::encode_icu_repeat({kInputBeats - 1, 1, 0})));
    records += 3;

    write_record(output, command_record(
        130, ftlpu::isa::encode_icu_nop(kCaptureCycle + 1)));
    ++records;
    for (const auto& instruction : permutes) {
        write_record(output, sxm_record(
            130, ftlpu::isa::encode_sxm_instruction(instruction)));
        ++records;
    }

    if (records != kScheduleRecords)
        throw std::logic_error("wavefront schedule record count mismatch");
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 4) {
        std::cerr << "usage: sxm_wavefront_transpose "
                  << "<init.hex> <golden.hex> <schedule.hex>\n";
        return 2;
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize_matrix(system);

    const auto transpose = ftlpu::SxmInstruction::Transpose(
        east_streams(0), east_streams(16));
    auto permutes = std::array<ftlpu::SxmInstruction, kWaveCount> {};
    for (std::size_t wave = 0; wave < kWaveCount; ++wave) {
        permutes[wave] = ftlpu::SxmInstruction::Permute(
            east_streams(16), west_streams(0), wavefront_map(wave));
    }

    build_schedule(system, transpose, permutes);
    write_vectors(argv[1], system, 0);

    auto icu_log = std::ofstream(
        "build/cmodel_vectors/sxm_wavefront_icu.log", std::ios::trunc);
    auto mem_log = std::ofstream(
        "build/cmodel_vectors/sxm_wavefront_mem.log", std::ios::trunc);
    auto sxm_log = std::ofstream(
        "build/cmodel_vectors/sxm_wavefront_sxm.log", std::ios::trunc);
    for (std::size_t cycle = 0; cycle < 46; ++cycle)
        system.tick({.icu = &icu_log, .mem = &mem_log, .sxm = &sxm_log});

    if (!verify_transpose(system)) return 1;
    write_vectors(argv[2], system, kOutputAddress);
    write_schedule(argv[3], transpose, permutes);

    std::cout << "C model 32x32 FP16 wavefront transpose golden generated\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
