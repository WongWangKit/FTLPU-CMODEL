#include "ftlpu/dma/dma.hpp"
#include "ftlpu/program/program.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSeqLen = 160;
constexpr std::size_t kHidden = ftlpu::hw::kMxmRows;
constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;

constexpr std::size_t kWqMatrix = 0;
constexpr std::size_t kWkMatrix = 1;
constexpr std::size_t kXMemColumn = 32;
constexpr std::size_t kQHalfLowColumn = 40;
constexpr std::size_t kQHalfHighColumn = 41;
constexpr std::size_t kKHalfLowColumn = 42;
constexpr std::size_t kKHalfHighColumn = 43;

constexpr std::size_t kQWestStreamBase = 0;
constexpr std::size_t kKWestStreamBase = 4;
constexpr std::size_t kActivationStream = 8;
constexpr std::size_t kQStreamOperand = ftlpu::hw::kEastStreams + kQWestStreamBase;
constexpr std::size_t kKStreamOperand = ftlpu::hw::kEastStreams + kKWestStreamBase;
constexpr std::size_t kQFp16OutputStream = 0;
constexpr std::size_t kKFp16OutputStream = 2;

constexpr std::size_t kIwStart = 20;
constexpr std::size_t kGemmStart = kIwStart + 2 * kBlocks;
constexpr std::size_t kMxmOutputStart = kGemmStart + kBlocks - 1;
constexpr std::size_t kVxmStart = kMxmOutputStart + ftlpu::hw::kStreamRegisterColumns;
constexpr std::size_t kVxmLatency = 2;
constexpr float kProjectionScale = 1.0f / 256.0f;

std::int8_t x_value(std::size_t row, std::size_t column)
{
    const auto mixed = row * 7 + column * 5 + ((row + 3) * (column + 11)) % 31;
    return static_cast<std::int8_t>(static_cast<int>(mixed % 17) - 8);
}

std::int8_t weight_value(std::size_t matrix, std::size_t k, std::size_t n)
{
    const auto mixed = matrix * 19 + k * 3 + n * 13 + ((k + 5) * (n + 7)) % 29;
    return static_cast<std::int8_t>(static_cast<int>(mixed % 15) - 7);
}

std::size_t x_address(std::size_t row, std::size_t lane)
{
    return row * kLanes + lane;
}

std::size_t weight_address(std::size_t matrix, std::size_t column_block)
{
    return matrix * kBlocks * kLanes + column_block * kLanes;
}

std::size_t half_address(std::size_t row)
{
    return row * kLanes;
}

std::size_t east_stream_cycles_to_sreg11(std::size_t column)
{
    return (ftlpu::hw::kStreamRegisterColumns - 1) - column / ftlpu::hw::kSlicesPerGroup;
}

std::size_t east_stream_write_latency(std::size_t column)
{
    return column / ftlpu::hw::kSlicesPerGroup + 2;
}

std::int32_t projection_value(std::size_t matrix, std::size_t row, std::size_t column)
{
    auto sum = std::int32_t {0};
    for (std::size_t k = 0; k < kHidden; ++k) {
        sum += static_cast<std::int32_t>(x_value(row, k))
            * static_cast<std::int32_t>(weight_value(matrix, k, column));
    }
    return sum;
}

std::uint16_t projection_half(std::size_t matrix, std::size_t row, std::size_t column)
{
    return ftlpu::VxmAlu::cast_scalar_to_float16_bits(
        static_cast<float>(projection_value(matrix, row, column)) * kProjectionScale);
}

ftlpu::MemGlobalAddress24 data_address(
    std::size_t mem_slice,
    std::size_t word = 0)
{
    return ftlpu::MemGlobalAddress24::FromFields(
        ftlpu::hemisphere_index(
            ftlpu::Hemisphere::East),
        mem_slice,
        ftlpu::MemLocalWordAddress13(word)
            .slice_byte_address());
}

void set_data_lane(
    std::vector<std::uint8_t>& bytes,
    std::size_t word,
    std::size_t tile,
    std::size_t lane,
    std::uint8_t value)
{
    bytes.at(
        word * ftlpu::hw::kPhysicalVectorBytes
        + tile * kLanes
        + lane) = value;
}

void append_attention_data(
    ftlpu::ProgramImage& image)
{
    const auto weight_words =
        weight_address(1, kBlocks - 1) + 1;
    for (std::size_t mem_slice = 0;
         mem_slice < 2 * kLoadStreams;
         ++mem_slice) {
        auto bytes = std::vector<std::uint8_t>(
            weight_words
                * ftlpu::hw::kPhysicalVectorBytes,
            0);
        const auto stream =
            mem_slice % kLoadStreams;
        for (std::size_t matrix = 0;
             matrix < 2;
             ++matrix) {
            for (std::size_t tile = 0;
                 tile < kBlocks;
                 ++tile) {
                for (std::size_t column_block = 0;
                     column_block < kBlocks;
                     ++column_block) {
                    const auto word =
                        weight_address(
                            matrix, column_block);
                    const auto column =
                        column_block * kLoadStreams
                        + stream;
                    for (std::size_t lane = 0;
                         lane < kLanes;
                         ++lane) {
                        set_data_lane(
                            bytes,
                            word,
                            tile,
                            lane,
                            static_cast<std::uint8_t>(
                                weight_value(
                                    matrix,
                                    tile * kLanes
                                        + lane,
                                    column)));
                    }
                }
            }
        }
        image.add_data_section(
            ftlpu::ProgramDataSection {
                "attention_weight_stream_"
                    + std::to_string(mem_slice),
                ftlpu::DmaPurpose::Model,
                data_address(mem_slice),
                std::move(bytes),
                {2, kHidden, kHidden},
                "Wq/Wk packed by MXM load stream",
            });
    }

    const auto x_words =
        x_address(kSeqLen - 1, 0) + 1;
    auto x_bytes = std::vector<std::uint8_t>(
        x_words * ftlpu::hw::kPhysicalVectorBytes,
        0);
    for (std::size_t row = 0;
         row < kSeqLen;
         ++row) {
        const auto word = x_address(row, 0);
        for (std::size_t k = 0;
             k < kHidden;
             ++k) {
            set_data_lane(
                x_bytes,
                word,
                k / kLanes,
                k % kLanes,
                static_cast<std::uint8_t>(
                    x_value(row, k)));
        }
    }
    image.add_data_section(
        ftlpu::ProgramDataSection {
            "attention_input_x",
            ftlpu::DmaPurpose::InputTensor,
            data_address(kXMemColumn),
            std::move(x_bytes),
            {kSeqLen, kHidden},
            "W8 projection input",
        });
}

void emit_weight_load(
    ftlpu::program::StaticSchedule& program,
    std::size_t mxm,
    std::size_t matrix)
{
    const auto stream_base = mxm * kLoadStreams;
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = stream_base + stream;
        const auto first_cycle = kIwStart - east_stream_cycles_to_sreg11(mem_column) - 1;
        for (std::size_t block = 0; block < kBlocks; ++block) {
            const auto column_block = kBlocks - 1 - block;
            program.mem_at(
                first_cycle + block,
                mem_column,
                ftlpu::MemInstruction::Read(weight_address(matrix, column_block), stream_base + stream));
        }
    }

    for (std::size_t block = 0; block < kBlocks; ++block) {
        program.mxm_at(
            kIwStart + block,
            mxm,
            ftlpu::MxmControlInstruction::IW(0));
    }
}

void emit_projection(
    ftlpu::program::StaticSchedule& program)
{
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        const auto read_cycle =
            kGemmStart + row
            - east_stream_cycles_to_sreg11(
                kXMemColumn)
            - 1;
        program.mem_at(
            read_cycle,
            kXMemColumn,
            ftlpu::MemInstruction::Read(
                x_address(row, 0),
                kActivationStream));

        program.mxm_at(
            kGemmStart + row,
            0,
            ftlpu::MxmControlInstruction::Compute(0, kActivationStream, kQWestStreamBase));
        program.mxm_at(
            kGemmStart + row,
            1,
            ftlpu::MxmControlInstruction::Compute(0, kActivationStream, kKWestStreamBase));

        program.vxm_at(kVxmStart + row, 0, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(kQStreamOperand),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        program.vxm_at(kVxmStart + row, 3, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(kKStreamOperand),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        program.vxm_at(kVxmStart + row + 1, 1, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::Imm(kProjectionScale),
        });
        program.vxm_at(kVxmStart + row + 1, 4, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(3),
            ftlpu::VxmLaneOperand::Imm(kProjectionScale),
        });
        program.vxm_at(kVxmStart + row + 2, 2, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(1),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float16,
            kQFp16OutputStream,
        });
        program.vxm_at(kVxmStart + row + 2, 5, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(4),
            ftlpu::VxmLaneOperand::Imm(0.0f),
        });
        program.vxm_at(kVxmStart + row + 3, 6, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(5),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float16,
            kKFp16OutputStream,
        });

        const auto q_write_cycle =
            kVxmStart + row + kVxmLatency
            + east_stream_write_latency(
                kQHalfLowColumn)
            - 1;
        const auto k_write_cycle = q_write_cycle + 1;
        program.mem_at(q_write_cycle, kQHalfLowColumn, ftlpu::MemInstruction::Write(half_address(row), kQFp16OutputStream));
        program.mem_at(q_write_cycle, kQHalfHighColumn, ftlpu::MemInstruction::Write(half_address(row), kQFp16OutputStream + 1));
        program.mem_at(k_write_cycle, kKHalfLowColumn, ftlpu::MemInstruction::Write(half_address(row), kKFp16OutputStream));
        program.mem_at(k_write_cycle, kKHalfHighColumn, ftlpu::MemInstruction::Write(half_address(row), kKFp16OutputStream + 1));
    }
}

std::uint16_t stored_half(const ftlpu::TileArrayModel& mem, std::size_t low_column, std::size_t high_column, std::size_t row, std::size_t column)
{
    const auto tile = column / kLanes;
    const auto lane = column % kLanes;
    const auto lo = mem.sram_lane_byte(low_column, tile, half_address(row), lane);
    const auto hi = mem.sram_lane_byte(high_column, tile, half_address(row), lane);
    return static_cast<std::uint16_t>(lo | (static_cast<std::uint16_t>(hi) << 8));
}

bool verify_loaded_weights(
    const ftlpu::TspSliceSystem& system,
    std::size_t mxm,
    std::size_t matrix)
{
    for (const auto tile :
         std::array<std::size_t, 3> {0, 7, 19}) {
        for (const auto column :
             std::array<std::size_t, 4> {
                 0, 31, 127, 319}) {
            const auto column_block =
                column / kLoadStreams;
            const auto local_column =
                column % kLoadStreams;
            for (const auto lane :
                 std::array<std::size_t, 2> {
                     0, 15}) {
                const auto actual =
                    system.mxm_unit(mxm)
                        .array()
                        .weight(
                            0,
                            tile,
                            column_block,
                            lane,
                            local_column);
                const auto expected =
                    weight_value(
                        matrix,
                        tile * kLanes + lane,
                        column);
                if (actual != expected) {
                    std::cerr
                        << "attention weight mismatch"
                        << " mxm=" << mxm
                        << " tile=" << tile
                        << " lane=" << lane
                        << " column=" << column
                        << " actual="
                        << static_cast<int>(actual)
                        << " expected="
                        << static_cast<int>(expected)
                        << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

int main()
try
{
    auto program =
        ftlpu::program::StaticSchedule {};
    emit_weight_load(program, 0, kWqMatrix);
    emit_weight_load(program, 1, kWkMatrix);
    emit_projection(program);

    auto workload = ftlpu::ProgramImage(
        ftlpu::ProgramImageHeader {
            ftlpu::ProgramImageHeader::kMagic,
            1,
            "W8 Q/K attention projection",
            "ProgramImage -> DMA -> local SRAM -> "
            "bootstrap -> IFetch -> finite IQ -> "
            "MEM/MXM/VXM -> MEM",
        });
    program.append_to(
        workload, "attention_projection");
    append_attention_data(workload);
    const auto launched =
        ftlpu::program::AutonomousProgramBuilder::
            Build(workload);

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ftlpu::GlobalMemoryAddressSpace memory;
    for (std::size_t hemisphere = 0;
         hemisphere < ftlpu::hw::kHemispheres;
         ++hemisphere) {
        memory.bind_hemisphere(
            hemisphere,
            system
                ->mem(static_cast<ftlpu::Hemisphere>(
                    hemisphere))
                .memory_model());
    }
    ftlpu::HostMemorySpace host;
    const auto host_buffer =
        host.register_buffer(
            launched.layout.host_bytes());
    ftlpu::DmaEngine dma(host, memory);
    const auto descriptors =
        launched.layout.make_dma_descriptors(
            host_buffer);
    for (const auto& descriptor : descriptors) {
        if (!dma.enqueue(descriptor).valid()) {
            throw std::logic_error(
                "attention DMA returned an invalid transfer ID");
        }
    }
    while (!dma.idle()) {
        if (!dma.tick()) {
            throw std::logic_error(
                "attention DMA stalled before completing its queue");
        }
    }
    std::size_t completions = 0;
    while (dma.completion_ready()) {
        if (!dma.pop_completion().id.valid()) {
            throw std::logic_error(
                "attention DMA produced an invalid completion ID");
        }
        ++completions;
    }
    assert(completions == descriptors.size());
    for (const auto row :
         std::array<std::size_t, 3> {0, 7, 159}) {
        for (const auto column :
             std::array<std::size_t, 4> {
                 0, 15, 127, 319}) {
            const auto actual =
                static_cast<std::int8_t>(
                    system->mem()
                        .sram_lane_byte(
                            kXMemColumn,
                            column / kLanes,
                            x_address(row, 0),
                            column % kLanes));
            assert(actual == x_value(row, column));
        }
    }
    ftlpu::load_bootstrap_preamble(
        system->icu(), launched.preamble);

    const auto workload_cycles =
        kVxmStart + kSeqLen + kVxmLatency
        + east_stream_write_latency(kQHalfLowColumn)
        + kBlocks + 8;
    const auto total_cycles =
        launched.schedule_epoch_cycle
        + workload_cycles;
    for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
        try {
            system->tick({});
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "cycle " << cycle << ": " << ex.what();
            system->icu().print_diagnostic_status(
                std::cerr);
            throw std::logic_error(os.str());
        }
    }

    if (!verify_loaded_weights(
            *system, 0, kWqMatrix)
        || !verify_loaded_weights(
            *system, 1, kWkMatrix)) {
        return 1;
    }

    const std::array<std::size_t, 4> sample_rows {0, 7, 79, 159};
    const std::array<std::size_t, 6> sample_columns {0, 1, 15, 16, 127, 319};
    for (const auto row : sample_rows) {
        for (const auto column : sample_columns) {
            const auto q_actual = stored_half(system->mem(), kQHalfLowColumn, kQHalfHighColumn, row, column);
            const auto k_actual = stored_half(system->mem(), kKHalfLowColumn, kKHalfHighColumn, row, column);
            const auto q_expected = projection_half(kWqMatrix, row, column);
            const auto k_expected = projection_half(kWkMatrix, row, column);
            if (q_actual != q_expected || k_actual != k_expected) {
                std::cerr << "attention projection mismatch row=" << row
                          << " column=" << column
                          << " q_actual=0x" << std::hex << q_actual
                          << " q_expected=0x" << q_expected
                          << " k_actual=0x" << k_actual
                          << " k_expected=0x" << k_expected
                          << std::dec << '\n';
                return 1;
            }
        }
    }

    return 0;
} catch (const std::exception& ex) {
    std::cerr << "attention_projection_test failed: " << ex.what() << '\n';
    return 1;
}
