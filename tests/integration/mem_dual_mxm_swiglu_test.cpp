#include "ftlpu/core/fp16.hpp"
#include "ftlpu/dma/dma.hpp"
#include "ftlpu/mxm/performance_monitor.hpp"
#include "ftlpu/program/program.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "ftlpu/vxm/data_format.hpp"
#include "ftlpu/vxm/special_alu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

static_assert(ftlpu::hw::kTileRows == 4);
static_assert(ftlpu::hw::kLanesPerTile == 8);
static_assert(ftlpu::hw::kMxmRows == 32);
static_assert(ftlpu::hw::kMxmColumns == 32);
static_assert(ftlpu::hw::kMxmK == 32);
static_assert(ftlpu::hw::kMxmCount == 4);
static_assert(ftlpu::hw::kVxmAluCount == 8);
static_assert(ftlpu::hw::kMxmActivationBytesPerValue == 2);
static_assert(ftlpu::hw::kMxmStoredWeightBytesPerValue == 1);
static_assert(ftlpu::hw::kMxmWeightBytesPerValue == 2);
static_assert(ftlpu::hw::kMxmWeightScaleStreams == 16);

constexpr std::size_t kRows = 160;
constexpr std::size_t kInputColumns = 320;
constexpr std::size_t kIntermediateColumns = 640;
constexpr std::size_t kOutputColumns = 320;
constexpr std::size_t kMBlockRows = ftlpu::hw::kMxmRows;
constexpr std::size_t kKBlockColumns = ftlpu::hw::kMxmK;
constexpr std::size_t kNBlockColumns = ftlpu::hw::kMxmColumns;
constexpr std::size_t kMBlocks = kRows / kMBlockRows;
constexpr std::size_t kGateUpKBlocks =
    kInputColumns / kKBlockColumns;
constexpr std::size_t kGateUpOutputBlocks =
    kIntermediateColumns / kNBlockColumns;
constexpr std::size_t kGateUpWaves =
    kGateUpOutputBlocks / 2;
constexpr std::size_t kDownKBlocks =
    kIntermediateColumns / kKBlockColumns;
constexpr std::size_t kDownOutputBlocks =
    kOutputColumns / kNBlockColumns;
constexpr std::size_t kDownWaves =
    (kDownOutputBlocks + ftlpu::hw::kMxmCount - 1)
    / ftlpu::hw::kMxmCount;
constexpr std::size_t kBlocks =
    ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes =
    ftlpu::hw::kLanesPerTile;

static_assert(kRows % kMBlockRows == 0);
static_assert(kInputColumns % kKBlockColumns == 0);
static_assert(kIntermediateColumns % kNBlockColumns == 0);
static_assert(kIntermediateColumns % kKBlockColumns == 0);
static_assert(kOutputColumns % kNBlockColumns == 0);
static_assert(kBlocks == ftlpu::hw::kTileRows);

enum class MatrixKind : std::size_t {
    Gate = 0,
    Up = 1,
    Down = 2,
};

constexpr std::size_t kScaleRegionRows = 256;
constexpr std::size_t kGateWeightRows =
    kGateUpOutputBlocks * kGateUpKBlocks * kBlocks;
constexpr std::size_t kUpWeightRows = kGateWeightRows;
constexpr std::size_t kDownWeightRows =
    kDownOutputBlocks * kDownKBlocks * kBlocks;
constexpr std::size_t kGateWeightBase = kScaleRegionRows;
constexpr std::size_t kUpWeightBase =
    kGateWeightBase + kGateWeightRows;
constexpr std::size_t kDownWeightBase =
    kUpWeightBase + kUpWeightRows;
constexpr std::size_t kWeightDataRows =
    kDownWeightBase + kDownWeightRows;

constexpr std::size_t kRemoteActivationLowSlice = 32;
constexpr std::size_t kRemoteActivationHighSlice = 33;
constexpr std::size_t kLocalActivationLowSlice = 34;
constexpr std::size_t kLocalActivationHighSlice = 35;
constexpr std::size_t kSwigluLowSlice = 36;
constexpr std::size_t kSwigluHighSlice = 37;
constexpr std::size_t kSwigluRemoteLowSlice = 40;
constexpr std::size_t kSwigluRemoteHighSlice = 41;
constexpr std::size_t kOutputLowSlice = 42;
constexpr std::size_t kOutputHighSlice = 43;

// IW occupies 0..7 and 16..23.  Keep FP16 activation on the top two
// streams so the tail of a long weight-load train can drain without
// colliding with the next activation wave inside the linear SR fabric.
constexpr std::size_t kWestActivationStream = 30;
constexpr std::size_t kEastActivationStream = 30;
constexpr std::size_t kGateOutputStream = 0;
constexpr std::size_t kUpOutputStream = 4;
constexpr std::size_t kVxmOutputStream = 12;
constexpr std::size_t kAccumulatorBank = 0;
constexpr std::size_t kLoadDrainCycles = 2 * kBlocks + 4;
constexpr std::size_t kComputeDrainCycles =
    2 * kBlocks + 6;
constexpr std::size_t kVxmCollectCycles =
    ftlpu::hw::kStreamRegisterColumns + kBlocks + 4;
constexpr std::size_t kSwigluLatency = 19;
constexpr std::size_t kFp16CastLatency = 9;
constexpr std::size_t kPhaseDrainCycles = 64;

using MatrixF = std::vector<float>;

bool logs_enabled()
{
    const auto* value = std::getenv("FTLPU_FFN_LOG");
    if (value == nullptr) {
        return false;
    }
    const auto text = std::string(value);
    return !(text.empty() || text == "0"
        || text == "false" || text == "FALSE"
        || text == "off" || text == "OFF");
}

const char* matrix_name(MatrixKind matrix)
{
    switch (matrix) {
    case MatrixKind::Gate: return "gate";
    case MatrixKind::Up: return "up";
    case MatrixKind::Down: return "down";
    }
    return "unknown";
}

std::size_t matrix_rows(MatrixKind matrix)
{
    return matrix == MatrixKind::Down
        ? kIntermediateColumns : kInputColumns;
}

std::size_t matrix_columns(MatrixKind matrix)
{
    return matrix == MatrixKind::Down
        ? kOutputColumns : kIntermediateColumns;
}

std::size_t matrix_k_blocks(MatrixKind matrix)
{
    return matrix_rows(matrix) / kKBlockColumns;
}

std::size_t matrix_output_blocks(MatrixKind matrix)
{
    return matrix_columns(matrix) / kNBlockColumns;
}

std::size_t matrix_weight_base(MatrixKind matrix)
{
    switch (matrix) {
    case MatrixKind::Gate: return kGateWeightBase;
    case MatrixKind::Up: return kUpWeightBase;
    case MatrixKind::Down: return kDownWeightBase;
    }
    throw std::logic_error("unknown FFN matrix");
}

std::size_t scale_address(
    MatrixKind matrix,
    std::size_t output_block,
    std::size_t column_block)
{
    constexpr std::array<std::size_t, 3> bases {
        0,
        kGateUpOutputBlocks * kBlocks,
        2 * kGateUpOutputBlocks * kBlocks,
    };
    return bases.at(static_cast<std::size_t>(matrix))
        + output_block * kBlocks + column_block;
}

std::size_t weight_address(
    MatrixKind matrix,
    std::size_t output_block,
    std::size_t k_block,
    std::size_t column_block)
{
    return matrix_weight_base(matrix)
        + (output_block * matrix_k_blocks(matrix) + k_block)
            * kBlocks
        + column_block;
}

std::size_t activation_address(
    std::size_t k_block,
    std::size_t row)
{
    return k_block * kRows + row;
}

std::size_t swiglu_address(
    std::size_t output_block,
    std::size_t row)
{
    return output_block * kRows + row;
}

std::size_t final_address(
    std::size_t output_block,
    std::size_t row)
{
    return output_block * kRows + row;
}

std::uint16_t scale_bits(
    MatrixKind matrix,
    std::size_t output_column)
{
    const auto matrix_bias =
        static_cast<std::size_t>(matrix) + 1;
    const auto value =
        0.00390625f
        * static_cast<float>(
            2 + (output_column * 3 + matrix_bias) % 7);
    return ftlpu::Fp16::from_float(value).bits();
}

float scale_value(
    MatrixKind matrix,
    std::size_t output_column)
{
    return ftlpu::Fp16::from_bits(
        scale_bits(matrix, output_column)).to_float();
}

std::int8_t stored_weight(
    MatrixKind matrix,
    std::size_t k,
    std::size_t n)
{
    const auto matrix_id =
        static_cast<std::size_t>(matrix);
    const auto mixed =
        matrix_id * 47 + k * 13 + n * 19
        + ((k + 5) * (n + 3) + matrix_id * 11) % 29;
    return static_cast<std::int8_t>(
        static_cast<int>(mixed % 31) - 15);
}

float dequantized_weight(
    MatrixKind matrix,
    std::size_t k,
    std::size_t n)
{
    return ftlpu::Fp16::from_float(
        static_cast<float>(stored_weight(matrix, k, n))
        * scale_value(matrix, n)).to_float();
}

float activation_value(
    std::size_t row,
    std::size_t column)
{
    const auto mixed =
        row * 17 + column * 7
        + ((row + 3) * (column + 5)) % 37;
    const auto value =
        static_cast<float>(
            static_cast<int>(mixed % 41) - 20)
        / 64.0f;
    return ftlpu::Fp16::from_float(value).to_float();
}

ftlpu::MemGlobalAddress24 global_address(
    ftlpu::Hemisphere hemisphere,
    std::size_t mem_slice,
    std::size_t row = 0)
{
    return ftlpu::MemGlobalAddress24::FromFields(
        ftlpu::hemisphere_index(hemisphere),
        mem_slice,
        ftlpu::MemLocalWordAddress13::FromFields(
            0, row).slice_byte_address());
}

void set_vector_lane(
    std::vector<std::uint8_t>& bytes,
    std::size_t word,
    std::size_t tile,
    std::size_t lane,
    std::uint8_t value)
{
    bytes.at(
        word * ftlpu::hw::kPhysicalVectorBytes
        + tile * kLanes + lane) = value;
}

bool slot_stores_matrix(
    std::size_t local_mxm,
    MatrixKind matrix)
{
    if (matrix == MatrixKind::Down) {
        return true;
    }
    return (local_mxm == 0 && matrix == MatrixKind::Gate)
        || (local_mxm == 1 && matrix == MatrixKind::Up);
}

void append_weight_data(ftlpu::ProgramImage& image)
{
    constexpr auto side = ftlpu::Hemisphere::East;
    for (std::size_t local_mxm = 0;
         local_mxm < ftlpu::hw::kEastMxmCount;
         ++local_mxm) {
        const auto source_base =
            local_mxm
            * ftlpu::hw::kMxmLoadStreamsPerCycle;
        for (std::size_t stream = 0;
             stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
             ++stream) {
            auto bytes = std::vector<std::uint8_t>(
                kWeightDataRows
                    * ftlpu::hw::kPhysicalVectorBytes,
                0);
            for (const auto matrix : {
                     MatrixKind::Gate,
                     MatrixKind::Up,
                     MatrixKind::Down}) {
                if (!slot_stores_matrix(
                        local_mxm, matrix)) {
                    continue;
                }
                for (std::size_t output_block = 0;
                     output_block
                         < matrix_output_blocks(matrix);
                     ++output_block) {
                    if (stream
                        < ftlpu::hw::
                            kMxmWeightScaleStreams) {
                        for (std::size_t column_block = 0;
                             column_block < kBlocks;
                             ++column_block) {
                            const auto local_column =
                                stream / 2;
                            const auto output_column =
                                output_block
                                    * kNBlockColumns
                                + column_block
                                    * ftlpu::hw::
                                        kMxmSupercellColumns
                                + local_column;
                            const auto bits = scale_bits(
                                matrix, output_column);
                            const auto value =
                                stream % 2 == 0
                                ? static_cast<std::uint8_t>(
                                      bits & 0xffu)
                                : static_cast<std::uint8_t>(
                                      bits >> 8);
                            const auto word =
                                scale_address(
                                    matrix,
                                    output_block,
                                    column_block);
                            for (std::size_t tile = 0;
                                 tile
                                     < ftlpu::hw::
                                         kTileRows;
                                 ++tile) {
                                for (std::size_t lane = 0;
                                     lane < kLanes;
                                     ++lane) {
                                    set_vector_lane(
                                        bytes,
                                        word,
                                        tile,
                                        lane,
                                        value);
                                }
                            }
                        }
                    }
                    if (stream
                        >= ftlpu::hw::
                            kMxmStoredWeightLoadStreams) {
                        continue;
                    }
                    for (std::size_t k_block = 0;
                         k_block
                             < matrix_k_blocks(matrix);
                         ++k_block) {
                        for (std::size_t column_block = 0;
                             column_block < kBlocks;
                             ++column_block) {
                            const auto output_column =
                                output_block
                                    * kNBlockColumns
                                + column_block
                                    * ftlpu::hw::
                                        kMxmSupercellColumns
                                + stream;
                            const auto word =
                                weight_address(
                                    matrix,
                                    output_block,
                                    k_block,
                                    column_block);
                            for (std::size_t tile = 0;
                                 tile
                                     < ftlpu::hw::
                                         kTileRows;
                                 ++tile) {
                                for (std::size_t lane = 0;
                                     lane < kLanes;
                                     ++lane) {
                                    const auto k =
                                        k_block
                                            * kKBlockColumns
                                        + tile * kLanes
                                        + lane;
                                    set_vector_lane(
                                        bytes,
                                        word,
                                        tile,
                                        lane,
                                        static_cast<
                                            std::uint8_t>(
                                            stored_weight(
                                                matrix,
                                                k,
                                                output_column)));
                                }
                            }
                        }
                    }
                }
            }
            image.add_data_section(
                ftlpu::ProgramDataSection {
                    "ffn_weight_slot"
                        + std::to_string(local_mxm)
                        + "_stream"
                        + std::to_string(stream),
                    ftlpu::DmaPurpose::Model,
                    global_address(
                        side,
                        source_base + stream),
                    std::move(bytes),
                    {
                        kWeightDataRows,
                        ftlpu::hw::
                            kPhysicalVectorBytes,
                    },
                    "W8 weights plus per-channel FP16 scales",
                });
        }
    }
}

void append_activation_data(ftlpu::ProgramImage& image)
{
    const std::array<std::size_t, 4> slices {
        kRemoteActivationLowSlice,
        kRemoteActivationHighSlice,
        kLocalActivationLowSlice,
        kLocalActivationHighSlice,
    };
    for (std::size_t copy = 0;
         copy < slices.size();
         ++copy) {
        auto bytes = std::vector<std::uint8_t>(
            kGateUpKBlocks * kRows
                * ftlpu::hw::kPhysicalVectorBytes,
            0);
        for (std::size_t k_block = 0;
             k_block < kGateUpKBlocks;
             ++k_block) {
            for (std::size_t row = 0;
                 row < kRows;
                 ++row) {
                const auto word =
                    activation_address(k_block, row);
                for (std::size_t tile = 0;
                     tile < ftlpu::hw::kTileRows;
                     ++tile) {
                    for (std::size_t lane = 0;
                         lane < kLanes;
                         ++lane) {
                        const auto column =
                            k_block * kKBlockColumns
                            + tile * kLanes + lane;
                        const auto bits =
                            ftlpu::Fp16::from_float(
                                activation_value(
                                    row, column)).bits();
                        const auto high =
                            copy == 1 || copy == 3;
                        set_vector_lane(
                            bytes,
                            word,
                            tile,
                            lane,
                            high
                                ? static_cast<
                                      std::uint8_t>(
                                      bits >> 8)
                                : static_cast<
                                      std::uint8_t>(
                                      bits & 0xffu));
                    }
                }
            }
        }
        image.add_data_section(
            ftlpu::ProgramDataSection {
                "ffn_activation_"
                    + std::string(
                        copy < 2 ? "remote_" : "local_")
                    + (copy % 2 == 0 ? "low" : "high"),
                ftlpu::DmaPurpose::InputTensor,
                global_address(
                    ftlpu::Hemisphere::East,
                    slices[copy]),
                std::move(bytes),
                {
                    kRows,
                    kInputColumns,
                    2,
                },
                "FP16 activation byte plane",
            });
    }
}

std::size_t east_read_to_mxm_latency(
    std::size_t mem_slice)
{
    return ftlpu::hw::kMemBoundaryStreamRegisterColumns
        - mem_slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t west_read_to_vxm_latency(
    std::size_t mem_slice)
{
    return mem_slice
            / ftlpu::hw::kMemSlicesPerGroup
        + 2;
}

std::size_t remote_read_to_west_mxm_latency(
    std::size_t mem_slice)
{
    return west_read_to_vxm_latency(mem_slice) + 1;
}

std::size_t vxm_to_mem_write_latency(
    std::size_t mem_slice)
{
    return mem_slice
            / ftlpu::hw::kMemSlicesPerGroup
        + 1;
}

ftlpu::Hemisphere mxm_hemisphere(std::size_t mxm)
{
    return mxm < ftlpu::hw::kWestMxmCount
        ? ftlpu::Hemisphere::West
        : ftlpu::Hemisphere::East;
}

std::size_t local_mxm_index(std::size_t mxm)
{
    return mxm < ftlpu::hw::kWestMxmCount
        ? mxm : mxm - ftlpu::hw::kWestMxmCount;
}

void emit_load_group(
    ftlpu::program::StaticSchedule& schedule,
    std::size_t issue_cycle,
    std::size_t mxm,
    MatrixKind matrix,
    std::size_t output_block,
    std::optional<std::size_t> k_block,
    std::size_t weight_buffer)
{
    const auto local_mxm = local_mxm_index(mxm);
    const auto source_base =
        local_mxm
        * ftlpu::hw::kMxmLoadStreamsPerCycle;
    const auto remote =
        mxm_hemisphere(mxm)
        == ftlpu::Hemisphere::West;
    const auto stream_count = k_block.has_value()
        ? ftlpu::hw::kMxmStoredWeightLoadStreams
        : ftlpu::hw::kMxmWeightScaleStreams;

    for (std::size_t step = 0;
         step < kBlocks;
         ++step) {
        const auto column_block =
            kBlocks - 1 - step;
        const auto word = k_block.has_value()
            ? weight_address(
                  matrix,
                  output_block,
                  *k_block,
                  column_block)
            : scale_address(
                  matrix,
                  output_block,
                  column_block);
        for (std::size_t stream = 0;
             stream < stream_count;
             ++stream) {
            const auto source_slice =
                source_base + stream;
            const auto latency = remote
                ? remote_read_to_west_mxm_latency(
                      source_slice)
                : east_read_to_mxm_latency(
                      source_slice);
            if (issue_cycle + step < latency) {
                throw std::logic_error(
                    "MXM load scheduled before its MEM source");
            }
            const auto stream_id = remote
                ? ftlpu::StreamId::West(
                      source_base + stream)
                : ftlpu::StreamId::East(
                      source_base + stream);
            schedule.mem_at(
                issue_cycle + step - latency,
                ftlpu::Hemisphere::East,
                source_slice,
                ftlpu::MemInstruction::Read(
                    ftlpu::MemLocalWordAddress13::
                        FromFields(0, word),
                    stream_id));
        }
        schedule.mxm_at(
            issue_cycle + step,
            mxm,
            k_block.has_value()
                ? ftlpu::MxmControlInstruction::IW(
                      weight_buffer)
                : ftlpu::MxmControlInstruction::
                      LoadScales(weight_buffer));
    }
}

void emit_activation_reads(
    ftlpu::program::StaticSchedule& schedule,
    std::size_t compute_cycle,
    std::size_t first_row,
    std::size_t k_block,
    bool swiglu_input)
{
    const auto remote_low = swiglu_input
        ? kSwigluRemoteLowSlice
        : kRemoteActivationLowSlice;
    const auto remote_high = swiglu_input
        ? kSwigluRemoteHighSlice
        : kRemoteActivationHighSlice;
    const auto local_low = swiglu_input
        ? kSwigluLowSlice
        : kLocalActivationLowSlice;
    const auto local_high = swiglu_input
        ? kSwigluHighSlice
        : kLocalActivationHighSlice;
    const auto remote_latency =
        remote_read_to_west_mxm_latency(remote_low);
    const auto local_latency =
        east_read_to_mxm_latency(local_low);

    for (std::size_t local_row = 0;
         local_row < kMBlockRows;
         ++local_row) {
        const auto global_row =
            first_row + local_row;
        const auto address = swiglu_input
            ? swiglu_address(k_block, global_row)
            : activation_address(k_block, global_row);
        schedule.mem_at(
            compute_cycle + local_row
                - remote_latency,
            ftlpu::Hemisphere::East,
            remote_low,
            ftlpu::MemInstruction::Read(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, address),
                ftlpu::StreamId::West(
                    kWestActivationStream)));
        schedule.mem_at(
            compute_cycle + local_row
                - remote_latency,
            ftlpu::Hemisphere::East,
            remote_high,
            ftlpu::MemInstruction::Read(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, address),
                ftlpu::StreamId::West(
                    kWestActivationStream + 1)));
        schedule.mem_at(
            compute_cycle + local_row
                - local_latency,
            ftlpu::Hemisphere::East,
            local_low,
            ftlpu::MemInstruction::Read(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, address),
                ftlpu::StreamId::East(
                    kEastActivationStream)));
        schedule.mem_at(
            compute_cycle + local_row
                - local_latency,
            ftlpu::Hemisphere::East,
            local_high,
            ftlpu::MemInstruction::Read(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, address),
                ftlpu::StreamId::East(
                    kEastActivationStream + 1)));
    }
}

void emit_compute_rows(
    ftlpu::program::StaticSchedule& schedule,
    std::size_t compute_cycle,
    const std::vector<std::size_t>& mxms,
    std::size_t weight_buffer,
    bool accumulate,
    bool reduce)
{
    for (const auto mxm : mxms) {
        const auto west =
            mxm_hemisphere(mxm)
            == ftlpu::Hemisphere::West;
        const auto output_stream =
            local_mxm_index(mxm) == 0
            ? kGateOutputStream : kUpOutputStream;
        for (std::size_t row = 0;
             row < kMBlockRows;
             ++row) {
            schedule.mxm_at(
                compute_cycle + row,
                mxm,
                ftlpu::MxmControlInstruction::
                    ComputeToAccumulator(
                        weight_buffer,
                        kAccumulatorBank,
                        west
                            ? kWestActivationStream
                            : kEastActivationStream,
                        output_stream,
                        accumulate,
                        reduce,
                        false));
        }
    }
}

ftlpu::VxmLaneAluInstruction swiglu_instruction(
    std::size_t stage,
    ftlpu::Hemisphere input_hemisphere)
{
    auto instruction =
        ftlpu::VxmLaneAluInstruction {};
    instruction.input_hemisphere =
        input_hemisphere;
    instruction.output_hemisphere =
        ftlpu::Hemisphere::East;
    instruction.precision =
        ftlpu::VxmAluPrecision::Float16;
    switch (stage) {
    case 0:
        instruction.operation =
            ftlpu::VxmAluOpcode::Negate;
        instruction.lhs =
            ftlpu::VxmLaneOperand::
                StreamFloat32();
        instruction.rhs =
            ftlpu::VxmLaneOperand::
                StreamFloat32();
        break;
    case 1:
        instruction.operation =
            ftlpu::VxmSpecialAluOpcode::Exp;
        instruction.lhs =
            ftlpu::VxmLaneOperand::Previous();
        break;
    case 2:
        instruction.operation =
            ftlpu::VxmAluOpcode::Add;
        instruction.lhs =
            ftlpu::VxmLaneOperand::Previous();
        instruction.rhs =
            ftlpu::VxmLaneOperand::Imm(1.0f);
        break;
    case 3:
        instruction.operation =
            ftlpu::VxmSpecialAluOpcode::
                Reciprocal;
        instruction.lhs =
            ftlpu::VxmLaneOperand::Previous();
        break;
    case 4:
        instruction.operation =
            ftlpu::VxmAluOpcode::Multiply;
        instruction.lhs =
            ftlpu::VxmLaneOperand::Previous();
        instruction.rhs =
            ftlpu::VxmLaneOperand::Original();
        break;
    case 5:
        instruction.operation =
            ftlpu::VxmAluOpcode::Multiply;
        instruction.lhs =
            ftlpu::VxmLaneOperand::Previous();
        instruction.rhs =
            ftlpu::VxmLaneOperand::Aux();
        break;
    case 6:
    case 7:
        instruction.operation =
            ftlpu::VxmAluOpcode::Bypass;
        instruction.lhs =
            ftlpu::VxmLaneOperand::Previous();
        break;
    default:
        throw std::out_of_range(
            "SwiGLU stage exceeds 8-ALU VXM");
    }
    if (stage == 7) {
        instruction.output_type =
            ftlpu::VxmCastTarget::Float16;
        instruction.output_stream =
            kVxmOutputStream;
    }
    return instruction;
}

ftlpu::VxmLaneAluInstruction fp16_cast_instruction(
    std::size_t stage,
    ftlpu::Hemisphere input_hemisphere,
    bool use_rhs_group)
{
    auto instruction =
        ftlpu::VxmLaneAluInstruction {};
    instruction.input_hemisphere =
        input_hemisphere;
    instruction.output_hemisphere =
        ftlpu::Hemisphere::East;
    instruction.precision =
        ftlpu::VxmAluPrecision::Float16;
    if (stage == 0) {
        if (use_rhs_group) {
            instruction.operation =
                ftlpu::VxmAluOpcode::Add;
            instruction.lhs =
                ftlpu::VxmLaneOperand::Imm(0.0f);
            instruction.rhs =
                ftlpu::VxmLaneOperand::
                    StreamFloat32();
        } else {
            instruction.operation =
                ftlpu::VxmAluOpcode::Bypass;
            instruction.lhs =
                ftlpu::VxmLaneOperand::
                    StreamFloat32();
        }
    } else {
        instruction.operation =
            ftlpu::VxmAluOpcode::Bypass;
        instruction.lhs =
            ftlpu::VxmLaneOperand::Previous();
    }
    if (stage == 7) {
        instruction.output_type =
            ftlpu::VxmCastTarget::Float16;
        instruction.output_stream =
            kVxmOutputStream;
    }
    return instruction;
}

void emit_vxm_wave(
    ftlpu::program::StaticSchedule& schedule,
    std::size_t issue_cycle,
    std::size_t first_row,
    std::size_t output_block,
    ftlpu::Hemisphere input_hemisphere,
    bool swiglu,
    bool use_rhs_group,
    std::size_t output_low_slice,
    std::size_t output_high_slice,
    std::size_t latency)
{
    for (std::size_t row = 0;
         row < kMBlockRows;
         ++row) {
        for (std::size_t stage = 0;
             stage < ftlpu::hw::kVxmAluCount;
             ++stage) {
            schedule.vxm_at(
                issue_cycle + row,
                stage,
                swiglu
                    ? swiglu_instruction(
                          stage,
                          input_hemisphere)
                    : fp16_cast_instruction(
                          stage,
                          input_hemisphere,
                          use_rhs_group));
        }
    }

    const auto write_cycle =
        issue_cycle + latency
        + vxm_to_mem_write_latency(
            output_low_slice);
    for (std::size_t row = 0;
         row < kMBlockRows;
         ++row) {
        const auto global_row =
            first_row + row;
        const auto address = swiglu
            ? swiglu_address(
                  output_block, global_row)
            : final_address(
                  output_block, global_row);
        schedule.mem_at(
            write_cycle + row,
            ftlpu::Hemisphere::East,
            output_low_slice,
            ftlpu::MemInstruction::Write(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, address),
                ftlpu::StreamId::East(
                    kVxmOutputStream)));
        schedule.mem_at(
            write_cycle + row,
            ftlpu::Hemisphere::East,
            output_high_slice,
            ftlpu::MemInstruction::Write(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, address),
                ftlpu::StreamId::East(
                    kVxmOutputStream + 1)));
    }
}

struct ScheduledImage {
    ftlpu::ProgramImage image{};
    std::size_t last_cycle{0};
};

ScheduledImage make_gate_up_phase(
    std::size_t m_block,
    std::size_t output_wave,
    std::size_t first_k_block,
    std::size_t k_block_count,
    bool load_scales,
    bool include_data)
{
    const auto first_row =
        m_block * kMBlockRows;
    const auto west_output_block =
        output_wave * 2;
    const auto east_output_block =
        west_output_block + 1;
    auto image = ftlpu::ProgramImage(
        ftlpu::ProgramImageHeader {
            ftlpu::ProgramImageHeader::kMagic,
            1,
            "4-MXM W8A16 gate/up phase",
            "ProgramImage -> DMA -> SRAM -> bootstrap -> "
            "ICU IFetch -> MEM/MXM/VXM -> MEM",
        });
    auto schedule =
        ftlpu::program::StaticSchedule {};
    auto cursor = std::size_t {24};

    if (load_scales) {
        for (std::size_t buffer = 0;
             buffer < ftlpu::MxmSupercell::kWeightBuffers;
             ++buffer) {
            emit_load_group(
                schedule,
                cursor,
                0,
                MatrixKind::Gate,
                west_output_block,
                std::nullopt,
                buffer);
            emit_load_group(
                schedule,
                cursor,
                1,
                MatrixKind::Up,
                west_output_block,
                std::nullopt,
                buffer);
            cursor += kLoadDrainCycles + kBlocks;
        }
        for (std::size_t buffer = 0;
             buffer < ftlpu::MxmSupercell::kWeightBuffers;
             ++buffer) {
            emit_load_group(
                schedule,
                cursor,
                2,
                MatrixKind::Gate,
                east_output_block,
                std::nullopt,
                buffer);
            emit_load_group(
                schedule,
                cursor,
                3,
                MatrixKind::Up,
                east_output_block,
                std::nullopt,
                buffer);
            cursor += kLoadDrainCycles + kBlocks;
        }
    }

    for (std::size_t local_k = 0;
         local_k < k_block_count;
         ++local_k) {
        const auto k_block =
            first_k_block + local_k;
        const auto weight_buffer =
            k_block % 2;
        emit_load_group(
            schedule,
            cursor,
            0,
            MatrixKind::Gate,
            west_output_block,
            k_block,
            weight_buffer);
        emit_load_group(
            schedule,
            cursor,
            1,
            MatrixKind::Up,
            west_output_block,
            k_block,
            weight_buffer);
        cursor += kLoadDrainCycles + kBlocks;
        emit_load_group(
            schedule,
            cursor,
            2,
            MatrixKind::Gate,
            east_output_block,
            k_block,
            weight_buffer);
        emit_load_group(
            schedule,
            cursor,
            3,
            MatrixKind::Up,
            east_output_block,
            k_block,
            weight_buffer);
        const auto compute_cycle =
            cursor + kLoadDrainCycles + kBlocks;
        emit_activation_reads(
            schedule,
            compute_cycle,
            first_row,
            k_block,
            false);
        emit_compute_rows(
            schedule,
            compute_cycle,
            {0, 1, 2, 3},
            weight_buffer,
            k_block != 0,
            k_block + 1 == kGateUpKBlocks);
        cursor = compute_cycle + kMBlockRows
            + kComputeDrainCycles;
    }

    if (first_k_block + k_block_count
        == kGateUpKBlocks) {
        const auto west_vxm =
            cursor + kVxmCollectCycles;
        emit_vxm_wave(
            schedule,
            west_vxm,
            first_row,
            west_output_block,
            ftlpu::Hemisphere::West,
            true,
            false,
            kSwigluLowSlice,
            kSwigluHighSlice,
            kSwigluLatency);
        const auto east_vxm =
            west_vxm + kMBlockRows + 8;
        emit_vxm_wave(
            schedule,
            east_vxm,
            first_row,
            east_output_block,
            ftlpu::Hemisphere::East,
            true,
            false,
            kSwigluLowSlice,
            kSwigluHighSlice,
            kSwigluLatency);
    }

    schedule.append_to(
        image,
        "W8A16 gate/up m"
            + std::to_string(m_block)
            + " wave"
            + std::to_string(output_wave)
            + " k"
            + std::to_string(first_k_block));
    if (include_data) {
        append_weight_data(image);
        append_activation_data(image);
    }
    return {
        std::move(image),
        schedule.last_cycle(),
    };
}

ScheduledImage make_swiglu_copy_phase()
{
    auto image = ftlpu::ProgramImage(
        ftlpu::ProgramImageHeader {
            ftlpu::ProgramImageHeader::kMagic,
            1,
            "SwiGLU FP16 duplication phase",
            "MEM -> Stream -> MEM, preserving hardware-produced data",
        });
    auto schedule =
        ftlpu::program::StaticSchedule {};
    constexpr std::size_t kStart = 16;
    constexpr std::size_t kCount =
        kRows * kGateUpOutputBlocks;
    for (std::size_t word = 0;
         word < kCount;
         ++word) {
        schedule.mem_at(
            kStart + word,
            ftlpu::Hemisphere::East,
            kSwigluLowSlice,
            ftlpu::MemInstruction::Read(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, word),
                ftlpu::StreamId::East(20)));
        schedule.mem_at(
            kStart + word,
            ftlpu::Hemisphere::East,
            kSwigluHighSlice,
            ftlpu::MemInstruction::Read(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, word),
                ftlpu::StreamId::East(21)));
        schedule.mem_at(
            kStart + word + 2,
            ftlpu::Hemisphere::East,
            kSwigluRemoteLowSlice,
            ftlpu::MemInstruction::Write(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, word),
                ftlpu::StreamId::East(20)));
        schedule.mem_at(
            kStart + word + 2,
            ftlpu::Hemisphere::East,
            kSwigluRemoteHighSlice,
            ftlpu::MemInstruction::Write(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, word),
                ftlpu::StreamId::East(21)));
    }
    schedule.append_to(
        image,
        "FP16 SwiGLU local duplication");
    return {
        std::move(image),
        schedule.last_cycle(),
    };
}

std::vector<std::pair<std::size_t, std::size_t>>
down_wave_assignments(std::size_t wave)
{
    auto result =
        std::vector<
            std::pair<std::size_t, std::size_t>> {};
    const auto first =
        wave * ftlpu::hw::kMxmCount;
    for (std::size_t mxm = 0;
         mxm < ftlpu::hw::kMxmCount;
         ++mxm) {
        const auto output_block = first + mxm;
        if (output_block >= kDownOutputBlocks) {
            break;
        }
        result.emplace_back(mxm, output_block);
    }
    return result;
}

ScheduledImage make_down_phase(
    std::size_t m_block,
    std::size_t output_wave,
    std::size_t first_k_block,
    std::size_t k_block_count,
    bool load_scales)
{
    const auto first_row =
        m_block * kMBlockRows;
    const auto assignments =
        down_wave_assignments(output_wave);
    auto image = ftlpu::ProgramImage(
        ftlpu::ProgramImageHeader {
            ftlpu::ProgramImageHeader::kMagic,
            1,
            "4-MXM W8A16 down phase",
            "FP16 SwiGLU -> MEM/SR -> W8A16 MXM -> VXM FP16 -> MEM",
        });
    auto schedule =
        ftlpu::program::StaticSchedule {};
    auto cursor = std::size_t {24};
    const auto emit_assignment_loads =
        [&] (std::optional<std::size_t> k_block,
             std::size_t weight_buffer) {
            for (const auto hemisphere :
                 {ftlpu::Hemisphere::West,
                  ftlpu::Hemisphere::East}) {
                auto emitted = false;
                for (const auto [mxm, output_block] :
                     assignments) {
                    if (mxm_hemisphere(mxm)
                        != hemisphere) {
                        continue;
                    }
                    emit_load_group(
                        schedule,
                        cursor,
                        mxm,
                        MatrixKind::Down,
                        output_block,
                        k_block,
                        weight_buffer);
                    emitted = true;
                }
                if (emitted) {
                    cursor +=
                        kLoadDrainCycles + kBlocks;
                }
            }
        };

    if (load_scales) {
        for (std::size_t buffer = 0;
             buffer < ftlpu::MxmSupercell::kWeightBuffers;
             ++buffer) {
            emit_assignment_loads(
                std::nullopt, buffer);
        }
    }

    for (std::size_t local_k = 0;
         local_k < k_block_count;
         ++local_k) {
        const auto k_block =
            first_k_block + local_k;
        const auto weight_buffer =
            k_block % 2;

        emit_assignment_loads(
            k_block, weight_buffer);
        const auto compute_cycle = cursor;
        emit_activation_reads(
            schedule,
            compute_cycle,
            first_row,
            k_block,
            true);
        auto active_mxms =
            std::vector<std::size_t> {};
        for (const auto [mxm, output_block] :
             assignments) {
            (void)output_block;
            active_mxms.push_back(mxm);
        }
        emit_compute_rows(
            schedule,
            compute_cycle,
            active_mxms,
            weight_buffer,
            k_block != 0,
            k_block + 1 == kDownKBlocks);
        cursor = compute_cycle + kMBlockRows
            + kComputeDrainCycles;
    }

    if (first_k_block + k_block_count
        == kDownKBlocks) {
        auto vxm_cycle =
            cursor + kVxmCollectCycles;
        for (const auto [mxm, output_block] :
             assignments) {
            emit_vxm_wave(
                schedule,
                vxm_cycle,
                first_row,
                output_block,
                mxm_hemisphere(mxm),
                false,
                local_mxm_index(mxm) == 1,
                kOutputLowSlice,
                kOutputHighSlice,
                kFp16CastLatency);
            vxm_cycle += kMBlockRows + 8;
        }
    }

    schedule.append_to(
        image,
        "W8A16 down m"
            + std::to_string(m_block)
            + " wave"
            + std::to_string(output_wave)
            + " k"
            + std::to_string(first_k_block));
    return {
        std::move(image),
        schedule.last_cycle(),
    };
}

struct ExecutionStats {
    std::array<ftlpu::MxmPerformanceMonitor,
               ftlpu::hw::kMxmCount>
        mxm{};
    std::size_t dma_descriptors{0};
    std::size_t dma_completions{0};
    std::size_t autonomous_phases{0};
    std::size_t system_cycles{0};
};

class FfnRuntime {
public:
    explicit FfnRuntime(bool enable_logs)
        : system_(std::make_unique<
                  ftlpu::TspSliceSystem>())
        , dma_(host_, memory_)
        , logs_enabled_(enable_logs)
    {
        for (std::size_t side = 0;
             side < ftlpu::hw::kHemispheres;
             ++side) {
            memory_.bind_hemisphere(
                side,
                system_
                    ->mem(static_cast<
                          ftlpu::Hemisphere>(side))
                    .memory_model());
        }
        if (logs_enabled_) {
            const auto directory =
                std::filesystem::path("logs")
                / "mem_dual_mxm_swiglu_w8a16";
            std::filesystem::create_directories(
                directory);
            icu_log_.open(directory / "icu.log");
            mem_log_.open(directory / "mem.log");
            mxm_log_.open(directory / "mxm.log");
            vxm_log_.open(directory / "vxm.log");
            if (!icu_log_ || !mem_log_
                || !mxm_log_ || !vxm_log_) {
                throw std::runtime_error(
                    "failed to open FFN trace logs");
            }
        }
    }

    void run(ScheduledImage phase)
    {
        const auto launched =
            ftlpu::program::
                AutonomousProgramBuilder::Build(
                    phase.image);
        const auto host_buffer =
            host_.register_buffer(
                launched.layout.host_bytes());
        const auto descriptors =
            launched.layout.make_dma_descriptors(
                host_buffer);
        for (const auto& descriptor :
             descriptors) {
            const auto id =
                dma_.enqueue(descriptor);
            if (!id.valid()) {
                throw std::logic_error(
                    "FFN DMA returned an invalid transfer ID");
            }
        }
        stats_.dma_descriptors +=
            descriptors.size();
        while (!dma_.idle()) {
            if (!dma_.tick()) {
                throw std::logic_error(
                    "FFN DMA stalled before completion");
            }
        }
        while (dma_.completion_ready()) {
            const auto completion =
                dma_.pop_completion();
            if (!completion.id.valid()) {
                throw std::logic_error(
                    "FFN DMA completion has invalid ID");
            }
            ++stats_.dma_completions;
        }

        // Every phase is a self-contained ProgramImage launch.  Reset only
        // ICU program/control state so notifications from an earlier phase
        // cannot satisfy this phase's Sync before its IFetch completes.
        // SRAM contents and functional-slice state (notably MXM accumulator
        // banks) deliberately remain live across K-block phases.
        system_->icu().reset();
        ftlpu::load_bootstrap_preamble(
            system_->icu(),
            launched.preamble);
        const auto cycles =
            launched.schedule_epoch_cycle
            + phase.last_cycle
            + kPhaseDrainCycles;
        for (std::size_t cycle = 0;
             cycle < cycles;
             ++cycle) {
            try {
                system_->tick(log_sinks());
            } catch (const std::exception& error) {
                std::ostringstream os;
                os << phase.image.header().workload
                   << " failed at local cycle "
                   << cycle;
                if (!phase.image.sections().empty()) {
                    os << " ["
                       << phase.image.sections()
                              .front()
                              .metadata
                       << ']';
                }
                os << ": " << error.what();
                throw std::runtime_error(os.str());
            }
            for (std::size_t mxm = 0;
                 mxm < ftlpu::hw::kMxmCount;
                 ++mxm) {
                stats_.mxm[mxm].sample(
                    system_->mxm_unit(mxm));
            }
            ++stats_.system_cycles;
        }
        ++stats_.autonomous_phases;
    }

    ftlpu::TspSliceSystem& system()
    {
        return *system_;
    }

    const ftlpu::TspSliceSystem& system() const
    {
        return *system_;
    }

    const ExecutionStats& stats() const
    {
        return stats_;
    }

private:
    ftlpu::TspSliceSystem::LogSinks
    log_sinks()
    {
        if (!logs_enabled_) {
            return {};
        }
        return {
            &icu_log_,
            &mem_log_,
            &mxm_log_,
            &vxm_log_,
            nullptr,
            0,
            0,
            0,
            false,
        };
    }

    std::unique_ptr<ftlpu::TspSliceSystem>
        system_;
    ftlpu::GlobalMemoryAddressSpace memory_;
    ftlpu::HostMemorySpace host_;
    ftlpu::DmaEngine dma_;
    ExecutionStats stats_{};
    bool logs_enabled_{false};
    std::ofstream icu_log_{};
    std::ofstream mem_log_{};
    std::ofstream mxm_log_{};
    std::ofstream vxm_log_{};
};

float reference_swiglu(float gate, float up)
{
    const auto basic = [] (
        ftlpu::VxmAluOpcode opcode,
        float lhs,
        float rhs = 0.0f) {
        return ftlpu::VxmAlu::execute(
            {opcode,
             ftlpu::VxmAluPrecision::Float16},
             lhs,
             rhs);
    };
    const auto original = gate;
    const auto auxiliary = up;
    auto value = basic(
        ftlpu::VxmAluOpcode::Negate,
        gate);
    const auto lut = ftlpu::VxmSpecialAlu {};
    value = lut.execute(
        ftlpu::VxmSpecialAluOpcode::Exp,
        value);
    value = basic(
        ftlpu::VxmAluOpcode::Add,
        value,
        1.0f);
    value = lut.execute(
        ftlpu::VxmSpecialAluOpcode::Reciprocal,
        value);
    value = basic(
        ftlpu::VxmAluOpcode::Multiply,
        value,
        original);
    value = basic(
        ftlpu::VxmAluOpcode::Multiply,
        value,
        auxiliary);
    value = basic(
        ftlpu::VxmAluOpcode::Bypass,
        value);
    value = basic(
        ftlpu::VxmAluOpcode::Bypass,
        value);
    return ftlpu::VxmDataFormat::
        fp16_bits_to_float(
            ftlpu::VxmDataFormat::
                float_to_fp16_bits(value));
}

MatrixF build_reference_swiglu()
{
    auto gate =
        MatrixF(kRows * kIntermediateColumns);
    auto up =
        MatrixF(kRows * kIntermediateColumns);
    auto swiglu =
        MatrixF(kRows * kIntermediateColumns);
    for (std::size_t row = 0;
         row < kRows;
         ++row) {
        for (std::size_t column = 0;
             column < kIntermediateColumns;
             ++column) {
            auto gate_sum = 0.0f;
            auto up_sum = 0.0f;
            for (std::size_t k = 0;
                 k < kInputColumns;
                 ++k) {
                const auto activation =
                    activation_value(row, k);
                gate_sum += activation
                    * dequantized_weight(
                        MatrixKind::Gate,
                        k,
                        column);
                up_sum += activation
                    * dequantized_weight(
                        MatrixKind::Up,
                        k,
                        column);
            }
            const auto index =
                row * kIntermediateColumns
                + column;
            gate[index] = gate_sum;
            up[index] = up_sum;
            swiglu[index] =
                reference_swiglu(
                    gate_sum, up_sum);
        }
    }
    return swiglu;
}

float reference_projection(
    MatrixKind matrix,
    std::size_t row,
    std::size_t column)
{
    auto sum = 0.0f;
    for (std::size_t k = 0;
         k < kInputColumns;
         ++k) {
        sum += activation_value(row, k)
            * dequantized_weight(
                matrix, k, column);
    }
    return sum;
}

void verify_gate_up_accumulators(
    const ftlpu::TspSliceSystem& system,
    std::size_t m_block,
    std::size_t output_wave)
{
    const auto first_row =
        m_block * kMBlockRows;
    const auto west_block =
        output_wave * 2;
    const auto east_block =
        west_block + 1;
    const std::array<MatrixKind, 4> matrices {
        MatrixKind::Gate,
        MatrixKind::Up,
        MatrixKind::Gate,
        MatrixKind::Up,
    };
    const std::array<std::size_t, 4> blocks {
        west_block,
        west_block,
        east_block,
        east_block,
    };
    for (std::size_t mxm = 0;
         mxm < ftlpu::hw::kMxmCount;
         ++mxm) {
        for (std::size_t row = 0;
             row < kMBlockRows;
             ++row) {
            for (std::size_t column = 0;
                 column < kNBlockColumns;
                 ++column) {
                const auto actual =
                    system.mxm_unit(mxm)
                        .accumulator_value(
                            kAccumulatorBank,
                            row,
                            column);
                const auto global_column =
                    blocks[mxm] * kNBlockColumns
                    + column;
                const auto expected =
                    reference_projection(
                        matrices[mxm],
                        first_row + row,
                        global_column);
                const auto tolerance =
                    1.0e-4f
                    + 2.0e-5f
                        * std::fabs(expected);
                if (std::fabs(actual - expected)
                    > tolerance) {
                    std::ostringstream os;
                    os << "MXM projection mismatch mxm="
                       << mxm
                       << " matrix="
                       << matrix_name(matrices[mxm])
                       << " row=" << first_row + row
                       << " column=" << global_column
                       << " actual=" << actual
                       << " expected=" << expected
                       << " scale_b0="
                       << system.mxm_unit(mxm)
                              .array()
                              .cell(0, column / ftlpu::hw::
                                      kMxmSupercellColumns)
                              .weight_scales(0)
                              [column % ftlpu::hw::
                                  kMxmSupercellColumns]
                       << " scale_b1="
                       << system.mxm_unit(mxm)
                              .array()
                              .cell(0, column / ftlpu::hw::
                                      kMxmSupercellColumns)
                              .weight_scales(1)
                              [column % ftlpu::hw::
                                  kMxmSupercellColumns]
                       << " expected_scale="
                       << scale_value(
                              matrices[mxm],
                              global_column)
                       << " cell_scales=";
                    for (std::size_t cell = 0;
                         cell < kBlocks;
                         ++cell) {
                        os << ' '
                           << system.mxm_unit(mxm)
                                  .array()
                                  .cell(0, cell)
                                  .weight_scales(0)[0];
                    }
                    os << " sram_scales=";
                    for (std::size_t cell = 0;
                         cell < kBlocks;
                         ++cell) {
                        const auto word =
                            scale_address(
                                matrices[mxm],
                                blocks[mxm],
                                cell);
                        const auto low =
                            system.mem(
                                ftlpu::Hemisphere::East)
                                .sram_lane_byte(
                                    local_mxm_index(mxm)
                                            * ftlpu::hw::
                                                kMxmLoadStreamsPerCycle,
                                    0,
                                    ftlpu::MemLocalWordAddress13::
                                        FromFields(0, word),
                                    0);
                        const auto high =
                            system.mem(
                                ftlpu::Hemisphere::East)
                                .sram_lane_byte(
                                    local_mxm_index(mxm)
                                            * ftlpu::hw::
                                                kMxmLoadStreamsPerCycle
                                            + 1,
                                    0,
                                    ftlpu::MemLocalWordAddress13::
                                        FromFields(0, word),
                                    0);
                        os << ' '
                           << ftlpu::Fp16::from_bits(
                                  static_cast<std::uint16_t>(
                                      low)
                                  | (static_cast<
                                         std::uint16_t>(
                                         high)
                                     << 8))
                                  .to_float();
                    }
                    throw std::runtime_error(os.str());
                }
            }
        }
    }
}

float read_fp16(
    const ftlpu::TspSliceSystem& system,
    std::size_t low_slice,
    std::size_t high_slice,
    std::size_t address,
    std::size_t physical_column);

bool close_fp16(float actual, float expected);

void verify_swiglu_wave(
    const ftlpu::TspSliceSystem& system,
    std::size_t m_block,
    std::size_t output_wave)
{
    const auto first_row =
        m_block * kMBlockRows;
    for (std::size_t block_offset = 0;
         block_offset < 2;
         ++block_offset) {
        const auto output_block =
            output_wave * 2 + block_offset;
        for (std::size_t row = 0;
             row < kMBlockRows;
             ++row) {
            for (std::size_t column = 0;
                 column < kNBlockColumns;
                 ++column) {
                const auto global_row =
                    first_row + row;
                const auto global_column =
                    output_block
                        * kNBlockColumns
                    + column;
                const auto gate =
                    reference_projection(
                        MatrixKind::Gate,
                        global_row,
                        global_column);
                const auto up =
                    reference_projection(
                        MatrixKind::Up,
                        global_row,
                        global_column);
                const auto expected =
                    reference_swiglu(gate, up);
                const auto actual = read_fp16(
                    system,
                    kSwigluLowSlice,
                    kSwigluHighSlice,
                    swiglu_address(
                        output_block,
                        global_row),
                    column);
                if (!close_fp16(actual, expected)) {
                    auto nearest_error =
                        std::numeric_limits<float>::infinity();
                    auto nearest_row = std::size_t {0};
                    auto nearest_column = std::size_t {0};
                    for (std::size_t candidate_row = first_row;
                         candidate_row < first_row + kMBlockRows;
                         ++candidate_row) {
                        for (std::size_t candidate_column =
                                 output_wave * 2 * kNBlockColumns;
                             candidate_column
                                 < (output_wave * 2 + 2)
                                     * kNBlockColumns;
                             ++candidate_column) {
                            const auto candidate =
                                reference_swiglu(
                                    reference_projection(
                                        MatrixKind::Gate,
                                        candidate_row,
                                        candidate_column),
                                    reference_projection(
                                        MatrixKind::Up,
                                        candidate_row,
                                        candidate_column));
                            const auto error =
                                std::fabs(actual - candidate);
                            if (error < nearest_error) {
                                nearest_error = error;
                                nearest_row = candidate_row;
                                nearest_column =
                                    candidate_column;
                            }
                        }
                    }
                    std::ostringstream os;
                    os << "SwiGLU wave mismatch block="
                       << output_block
                       << " row=" << global_row
                       << " column=" << global_column
                       << " actual=" << actual
                       << " expected=" << expected
                       << " nearest=(" << nearest_row
                       << ',' << nearest_column
                       << ") error=" << nearest_error;
                    throw std::runtime_error(os.str());
                }
            }
        }
    }
}

MatrixF build_reference_output(
    const MatrixF& swiglu)
{
    auto output =
        MatrixF(kRows * kOutputColumns);
    for (std::size_t row = 0;
         row < kRows;
         ++row) {
        for (std::size_t column = 0;
             column < kOutputColumns;
             ++column) {
            auto sum = 0.0f;
            for (std::size_t k = 0;
                 k < kIntermediateColumns;
                 ++k) {
                sum += swiglu[
                           row
                               * kIntermediateColumns
                           + k]
                    * dequantized_weight(
                        MatrixKind::Down,
                        k,
                        column);
            }
            output[
                row * kOutputColumns + column] =
                ftlpu::VxmDataFormat::
                    round_fp16_ftz(sum);
        }
    }
    return output;
}

float read_fp16(
    const ftlpu::TspSliceSystem& system,
    std::size_t low_slice,
    std::size_t high_slice,
    std::size_t address,
    std::size_t physical_column)
{
    const auto tile =
        physical_column / kLanes;
    const auto lane =
        physical_column % kLanes;
    const auto low =
        system.mem(ftlpu::Hemisphere::East)
            .sram_lane_byte(
                low_slice,
                tile,
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, address),
                lane);
    const auto high =
        system.mem(ftlpu::Hemisphere::East)
            .sram_lane_byte(
                high_slice,
                tile,
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, address),
                lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high)
           << 8)).to_float();
}

bool close_fp16(float actual, float expected)
{
    const auto tolerance =
        0.003f
        + 0.012f * std::fabs(expected);
    return std::fabs(actual - expected)
        <= tolerance;
}

void verify_swiglu(
    const ftlpu::TspSliceSystem& system,
    const MatrixF& expected)
{
    for (std::size_t row = 0;
         row < kRows;
         ++row) {
        for (std::size_t column = 0;
             column < kIntermediateColumns;
             ++column) {
            const auto block =
                column / kNBlockColumns;
            const auto physical =
                column % kNBlockColumns;
            const auto actual = read_fp16(
                system,
                kSwigluLowSlice,
                kSwigluHighSlice,
                swiglu_address(block, row),
                physical);
            const auto golden =
                expected[
                    row * kIntermediateColumns
                    + column];
            if (!close_fp16(actual, golden)) {
                auto nearest_error =
                    std::numeric_limits<float>::
                        infinity();
                auto nearest_row =
                    std::size_t {0};
                auto nearest_column =
                    std::size_t {0};
                for (std::size_t candidate_row = 0;
                     candidate_row < kRows;
                     ++candidate_row) {
                    for (std::size_t candidate_column = 0;
                         candidate_column
                             < kIntermediateColumns;
                         ++candidate_column) {
                        const auto error = std::fabs(
                            actual
                            - expected[
                                candidate_row
                                    * kIntermediateColumns
                                + candidate_column]);
                        if (error < nearest_error) {
                            nearest_error = error;
                            nearest_row = candidate_row;
                            nearest_column =
                                candidate_column;
                        }
                    }
                }
                std::ostringstream os;
                os << "SwiGLU mismatch row=" << row
                   << " column=" << column
                   << " actual=" << actual
                   << " expected=" << golden
                   << " nearest_expected(row="
                   << nearest_row
                   << ",column=" << nearest_column
                   << ")="
                   << expected[
                          nearest_row
                              * kIntermediateColumns
                          + nearest_column]
                   << " nearest_error="
                   << nearest_error;
                throw std::runtime_error(os.str());
            }
        }
    }
}

void verify_swiglu_copy(
    const ftlpu::TspSliceSystem& system)
{
    for (std::size_t block = 0;
         block < kGateUpOutputBlocks;
         ++block) {
        for (std::size_t row = 0;
             row < kRows;
             ++row) {
            for (std::size_t physical = 0;
                 physical < kNBlockColumns;
                 ++physical) {
                const auto address =
                    swiglu_address(block, row);
                const auto local = read_fp16(
                    system,
                    kSwigluLowSlice,
                    kSwigluHighSlice,
                    address,
                    physical);
                const auto remote = read_fp16(
                    system,
                    kSwigluRemoteLowSlice,
                    kSwigluRemoteHighSlice,
                    address,
                    physical);
                if (local != remote) {
                    std::ostringstream os;
                    os << "SwiGLU hardware copy mismatch block="
                       << block << " row=" << row
                       << " physical=" << physical
                       << " local=" << local
                       << " remote=" << remote;
                    throw std::runtime_error(os.str());
                }
            }
        }
    }
}

void verify_output(
    const ftlpu::TspSliceSystem& system,
    const MatrixF& expected)
{
    auto max_error = 0.0f;
    for (std::size_t row = 0;
         row < kRows;
         ++row) {
        for (std::size_t column = 0;
             column < kOutputColumns;
             ++column) {
            const auto block =
                column / kNBlockColumns;
            const auto physical =
                column % kNBlockColumns;
            const auto actual = read_fp16(
                system,
                kOutputLowSlice,
                kOutputHighSlice,
                final_address(block, row),
                physical);
            const auto golden =
                expected[
                    row * kOutputColumns + column];
            max_error = std::max(
                max_error,
                std::fabs(actual - golden));
            if (!close_fp16(actual, golden)) {
                std::ostringstream os;
                os << "FFN output mismatch row=" << row
                   << " column=" << column
                   << " actual=" << actual
                   << " expected=" << golden
                   << " abs_error="
                   << std::fabs(actual - golden);
                throw std::runtime_error(os.str());
            }
        }
    }
    std::cout << "FFN max FP16 error: "
              << max_error << '\n';
}

void print_stats(
    const FfnRuntime& runtime)
{
    const auto& stats = runtime.stats();
    std::cout
        << "FFN autonomous path: phases="
        << stats.autonomous_phases
        << " DMA descriptors="
        << stats.dma_descriptors
        << " completions="
        << stats.dma_completions
        << " system cycles="
        << stats.system_cycles << '\n';
    for (std::size_t mxm = 0;
         mxm < ftlpu::hw::kMxmCount;
         ++mxm) {
        stats.mxm[mxm].print(
            std::cout,
            "MXM" + std::to_string(mxm));
        if (stats.mxm[mxm].non_idle_cycles()
            == 0) {
            throw std::runtime_error(
                "one of the four MXMs was never active");
        }
    }

    const auto& vxm_stats =
        runtime.system()
            .vxm()
            .superlane(0)
            .lane(0)
            .statistics();
    std::cout << std::fixed
              << std::setprecision(2)
              << "VXM tile0/lane0 cycles="
              << vxm_stats.cycles
              << " executed_slots="
              << vxm_stats.executed_slots
              << " useful_slots="
              << vxm_stats.useful_slots
              << " active_util="
              << vxm_stats.active_utilization() * 100.0
              << "% useful_util="
              << vxm_stats.useful_utilization() * 100.0
              << "% stages=";
    for (const auto count :
         vxm_stats.stage_executions) {
        std::cout << ' ' << count;
    }
    std::cout << '\n';
}

} // namespace

int main()
try
{
    auto runtime =
        FfnRuntime(logs_enabled());
    auto first_phase = true;
    for (std::size_t m_block = 0;
         m_block < kMBlocks;
         ++m_block) {
        for (std::size_t wave = 0;
             wave < kGateUpWaves;
             ++wave) {
            // E0/E1 also serve as autonomous target loaders.  Keeping at
            // most ten scale/weight bursts in those MEM IQs leaves room for
            // the loader, common-epoch Sync prefix, and schedule packets.
            constexpr std::array<std::size_t, 3>
                kGateUpPhaseKBlocks {3, 5, 2};
            auto first_k = std::size_t {0};
            for (const auto count :
                 kGateUpPhaseKBlocks) {
                runtime.run(make_gate_up_phase(
                    m_block,
                    wave,
                    first_k,
                    count,
                    first_k == 0,
                    first_phase));
                first_phase = false;
                first_k += count;
            }
            verify_gate_up_accumulators(
                runtime.system(),
                m_block,
                wave);
            verify_swiglu_wave(
                runtime.system(),
                m_block,
                wave);
        }
    }

    runtime.run(make_swiglu_copy_phase());
    verify_swiglu_copy(runtime.system());

    const auto reference_swiglu =
        build_reference_swiglu();
    verify_swiglu(
        runtime.system(),
        reference_swiglu);

    constexpr std::array<std::size_t, 5>
        kDownPhaseKBlocks {3, 5, 5, 5, 2};
    for (std::size_t m_block = 0;
         m_block < kMBlocks;
         ++m_block) {
        for (std::size_t wave = 0;
             wave < kDownWaves;
             ++wave) {
            auto first_k = std::size_t {0};
            for (const auto count :
                 kDownPhaseKBlocks) {
                runtime.run(make_down_phase(
                    m_block,
                    wave,
                    first_k,
                    count,
                    first_k == 0));
                first_k += count;
            }
        }
    }

    const auto reference_output =
        build_reference_output(
            reference_swiglu);
    verify_output(
        runtime.system(),
        reference_output);
    print_stats(runtime);

    if (runtime.stats().dma_descriptors
        != runtime.stats().dma_completions) {
        throw std::runtime_error(
            "not every FFN DMA descriptor completed");
    }
    return 0;
} catch (const std::exception& error) {
#ifdef FTLPU_EARLY_MXM_COMPUTE_TEST
    std::cerr
        << "mem_dual_mxm_swiglu_early_compute_icu_test failed: ";
#else
    std::cerr
        << "mem_dual_mxm_swiglu_offline_icu_test failed: ";
#endif
    std::cerr << error.what() << '\n';
    return 1;
}
