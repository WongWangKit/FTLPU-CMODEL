#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kSeqLen = 160;
constexpr std::size_t kHidden = ftlpu::hw::kMxmRows;
constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;

constexpr std::size_t east_stream_cycles_to_sreg11(std::size_t column)
{
    return (ftlpu::hw::kStreamRegisterColumns - 1) - column / ftlpu::hw::kSlicesPerGroup;
}

constexpr std::size_t east_stream_write_latency(std::size_t column)
{
    return column / ftlpu::hw::kSlicesPerGroup + 2;
}

constexpr std::size_t kWqMatrix = 0;
constexpr std::size_t kWkMatrix = 1;
constexpr std::size_t kWvMatrix = 2;
constexpr std::size_t kXMemColumnBase = 0;
constexpr std::size_t kXMemColumns = kLoadStreams;
constexpr std::size_t kQInt8ColumnBase = 16;
constexpr std::size_t kQInt8Columns = kLoadStreams;
constexpr std::size_t kKInt8Column = 41;
constexpr std::size_t kVInt8ColumnBase = 0;
constexpr std::size_t kVInt8Columns = kLoadStreams;
constexpr std::size_t kSoftmaxParallelRows = 4;
constexpr std::array<std::size_t, kSoftmaxParallelRows> kScaledScoreByteColumnBases {16, 20, 24, 28};
constexpr std::array<std::size_t, kSoftmaxParallelRows> kExpScoreByteColumnBases {0, 4, 8, 12};
constexpr std::size_t kRowMaxByteColumnBase = 32;
constexpr std::size_t kRowSumByteColumnBase = 36;
constexpr std::size_t kSoftmaxInt8ColumnBase = 40;
constexpr std::size_t kAttentionOutputColumnBase = 4;
constexpr std::size_t kSoftmaxScratchAddressBase = 4096;
constexpr std::size_t kXAddressBase = 6144;

constexpr std::size_t kQWestStreamBase = 0;
constexpr std::size_t kKWestStreamBase = 4;
constexpr std::size_t kActivationStream = 8;
constexpr std::size_t kVActivationStreamBase = 16;
constexpr std::size_t kQkOutputWestStreamBase = 12;
constexpr std::size_t kVOutputWestStreamBase = 16;
constexpr std::size_t kSoftmaxWestStreamBase = ftlpu::hw::kEastStreams;
constexpr std::size_t kSoftmaxEastStreamBase = 0;
constexpr std::size_t kReductionWestStreamBase = kSoftmaxWestStreamBase + 16;
constexpr std::size_t kReductionEastStreamBase = kSoftmaxEastStreamBase + 16;
constexpr std::size_t kQStreamOperand = ftlpu::hw::kEastStreams + kQWestStreamBase;
constexpr std::size_t kKStreamOperand = ftlpu::hw::kEastStreams + kKWestStreamBase;
constexpr std::size_t kVStreamOperand = ftlpu::hw::kEastStreams + kVOutputWestStreamBase;
constexpr std::size_t kQInt8OutputStream = 0;
constexpr std::size_t kKInt8OutputStream = 1;
constexpr std::size_t kVInt8OutputStream = 20;
constexpr std::size_t kAttentionActivationStream = 1;
constexpr std::size_t kAttentionWeightStreamBase = 16;
constexpr std::size_t kAttentionOutputWestStreamBase = 0;

constexpr std::size_t kIwStart = 20;
constexpr std::size_t kGemmStart = kIwStart + 2 * kBlocks;
constexpr std::size_t kMxmOutputStart = kGemmStart + kBlocks - 1;
constexpr std::size_t kVxmStart = kMxmOutputStart + ftlpu::hw::kStreamRegisterColumns;
constexpr std::size_t kVxmLatency = 2;
constexpr std::size_t kQkIwStart = kVxmStart + kSeqLen + kVxmLatency + east_stream_write_latency(kKInt8Column) + 8;
constexpr std::size_t kQkGemmStart = kQkIwStart + kBlocks;
constexpr std::size_t kQkOutputStart = kQkGemmStart + kBlocks - 1;
constexpr std::size_t kDirectScoreVxmStart = kQkOutputStart + ftlpu::hw::kStreamRegisterColumns;
constexpr std::size_t kVProjectionGemmStart =
    kDirectScoreVxmStart + kSeqLen + east_stream_write_latency(kRowMaxByteColumnBase)
    + ftlpu::hw::kStreamRegisterColumns + 2;
constexpr std::size_t kVProjectionOutputStart =
    kVProjectionGemmStart + kBlocks - 1 + ftlpu::hw::kStreamRegisterColumns;
constexpr std::size_t kVProjectionLastWrite =
    kVProjectionOutputStart + kHidden - 1 + 2
    + east_stream_write_latency(kVInt8ColumnBase + kVInt8Columns - 1);
constexpr float kProjectionScale = 1.0f / 256.0f;
constexpr float kScoreScale = 1.0f / 17.88854381999832f; // 1 / sqrt(320)
constexpr float kSoftmaxInt8Scale = 127.0f;

static_assert(kSeqLen % kSoftmaxParallelRows == 0);

constexpr std::size_t west_stream_read_latency(std::size_t column)
{
    return column / ftlpu::hw::kSlicesPerGroup + 1;
}

constexpr std::size_t east_stream_cycles_to_sxm(std::size_t column)
{
    return ftlpu::hw::kMemEastBoundaryStreamRegisterColumn
        - column / ftlpu::hw::kSlicesPerGroup;
}

constexpr std::size_t mxm_west_stream_write_latency(std::size_t column)
{
    return ftlpu::hw::kMxmBoundaryStreamRegisterColumn
        - column / ftlpu::hw::kSlicesPerGroup;
}

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

std::size_t x_column(std::size_t row)
{
    return kXMemColumnBase + row % kXMemColumns;
}

std::size_t x_address(std::size_t row)
{
    return kXAddressBase + (row / kXMemColumns) * kLanes;
}

std::size_t weight_address(std::size_t matrix, std::size_t column_block)
{
    return matrix * kBlocks * kLanes + column_block * kLanes;
}

std::size_t projection_output_address(std::size_t row)
{
    return row * kLanes;
}

std::size_t softmax_scratch_address(std::size_t batch)
{
    return kSoftmaxScratchAddressBase + batch * kLanes;
}

std::size_t q_column(std::size_t row)
{
    return kQInt8ColumnBase + row % kQInt8Columns;
}

std::size_t q_address(std::size_t row)
{
    return (row / kQInt8Columns) * kLanes;
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

std::int8_t projection_int8(std::size_t matrix, std::size_t row, std::size_t column)
{
    return ftlpu::VxmAlu::cast_scalar_to_int8(
        static_cast<float>(projection_value(matrix, row, column)) * kProjectionScale);
}

std::int32_t qk_value(std::size_t k_row, std::size_t q_row)
{
    auto sum = std::int32_t {0};
    for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
        sum += static_cast<std::int32_t>(projection_int8(kWkMatrix, k_row, hidden))
            * static_cast<std::int32_t>(projection_int8(kWqMatrix, q_row, hidden));
    }
    return sum;
}

void stage_mem(ftlpu::TileArrayModel& mem)
{
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t k = 0; k < kHidden; ++k) {
            mem.set_sram_lane_byte(
                x_column(row),
                k / kLanes,
                x_address(row),
                k % kLanes,
                static_cast<std::uint8_t>(x_value(row, k)));
        }
    }

    for (std::size_t matrix = 0; matrix < 3; ++matrix) {
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
                const auto address = weight_address(matrix, column_block);
                for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
                    const auto column = column_block * kLoadStreams + stream;
                    for (std::size_t lane = 0; lane < kLanes; ++lane) {
                        mem.set_sram_lane_byte(
                            stream,
                            tile,
                            address,
                            lane,
                            static_cast<std::uint8_t>(weight_value(matrix, tile * kLanes + lane, column)));
                        mem.set_sram_lane_byte(
                            kLoadStreams + stream,
                            tile,
                            address,
                            lane,
                            static_cast<std::uint8_t>(weight_value(matrix, tile * kLanes + lane, column)));
                    }
                }
            }
        }
    }
}

class Program {
public:
    void mem(std::size_t cycle, std::size_t column, ftlpu::MemInstruction instruction)
    {
        mem_[column].push_back(Event<ftlpu::MemInstruction> {cycle, instruction});
    }

    void mxm(std::size_t cycle, std::size_t index, ftlpu::MxmControlInstruction instruction)
    {
        if (instruction.opcode == ftlpu::MxmControlOpcode::Compute) {
            mxm_compute_[index].push_back(Event<ftlpu::MxmControlInstruction> {cycle, instruction});
        } else {
            mxm_load_[index].push_back(Event<ftlpu::MxmControlInstruction> {cycle, instruction});
        }
    }

    void vxm(std::size_t cycle, std::size_t alu, ftlpu::VxmLaneAluInstruction instruction)
    {
        vxm_[alu].push_back(Event<ftlpu::VxmLaneAluInstruction> {cycle, instruction});
    }

    void sxm_transpose(std::size_t cycle, ftlpu::SxmInstruction instruction)
    {
        sxm_transpose_.push_back(Event<ftlpu::SxmInstruction> {cycle, std::move(instruction)});
    }

    void sxm_permute(std::size_t cycle, ftlpu::SxmInstruction instruction)
    {
        sxm_permute_.push_back(Event<ftlpu::SxmInstruction> {cycle, std::move(instruction)});
    }

    void load(ftlpu::InstructionControlUnit& icu)
    {
        for (std::size_t column = 0; column < mem_.size(); ++column) {
            load_queue(mem_[column], "MEM" + std::to_string(column), [&](std::size_t n) { icu.enqueue_mem_nop(column, n); }, [&](auto instruction) {
                icu.enqueue_mem(column, instruction);
            });
        }
        for (std::size_t mxm = 0; mxm < mxm_load_.size(); ++mxm) {
            load_queue(mxm_load_[mxm], "MXM" + std::to_string(mxm) + " load", [&](std::size_t n) { icu.enqueue_mxm_load_nop(mxm, n); }, [&](auto instruction) {
                icu.enqueue_mxm(mxm, instruction);
            });
            load_queue(mxm_compute_[mxm], "MXM" + std::to_string(mxm) + " compute", [&](std::size_t n) { icu.enqueue_mxm_compute_nop(mxm, n); }, [&](auto instruction) {
                icu.enqueue_mxm(mxm, instruction);
            });
        }
        for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
            load_queue(vxm_[alu], "VXM ALU" + std::to_string(alu), [&](std::size_t n) { icu.enqueue_vxm_nop(alu, n); }, [&](auto instruction) {
                icu.enqueue_vxm(alu, instruction);
            });
        }
        load_queue(sxm_transpose_, "SXM transpose", [&](std::size_t n) { icu.enqueue_sxm_transpose_nop(n); }, [&](auto instruction) {
            icu.enqueue_sxm_transpose(std::move(instruction));
        });
        load_queue(sxm_permute_, "SXM permute", [&](std::size_t n) { icu.enqueue_sxm_permute_nop(n); }, [&](auto instruction) {
            icu.enqueue_sxm_permute(std::move(instruction));
        });
    }

private:
    template <typename T>
    struct Event {
        std::size_t cycle{0};
        T instruction{};
    };

    template <typename T, typename Nop, typename Emit>
    static void load_queue(std::vector<Event<T>>& events, std::string_view name, Nop nop, Emit emit)
    {
        std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.cycle < rhs.cycle;
        });
        auto cursor = std::size_t {0};
        for (const auto& event : events) {
            if (event.cycle < cursor) {
                throw std::logic_error(
                    "single-head attention program queue collision in " + std::string(name)
                    + " at cycle " + std::to_string(event.cycle));
            }
            nop(event.cycle - cursor);
            emit(event.instruction);
            cursor = event.cycle + 1;
        }
    }

    std::array<std::vector<Event<ftlpu::MemInstruction>>, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::vector<Event<ftlpu::MxmControlInstruction>>, ftlpu::InstructionControlUnit::kMxmQueues> mxm_load_{};
    std::array<std::vector<Event<ftlpu::MxmControlInstruction>>, ftlpu::InstructionControlUnit::kMxmQueues> mxm_compute_{};
    std::array<std::vector<Event<ftlpu::VxmLaneAluInstruction>>, ftlpu::InstructionControlUnit::kVxmQueues> vxm_{};
    std::vector<Event<ftlpu::SxmInstruction>> sxm_transpose_{};
    std::vector<Event<ftlpu::SxmInstruction>> sxm_permute_{};
};

void emit_weight_load_at(
    Program& program,
    std::size_t mxm,
    std::size_t matrix,
    std::size_t start_cycle,
    std::size_t buffer,
    std::size_t mem_column_base,
    std::size_t stream_base)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = mem_column_base + stream;
        const auto first_cycle = start_cycle - east_stream_cycles_to_sreg11(mem_column) - 1;
        for (std::size_t block = 0; block < kBlocks; ++block) {
            const auto column_block = kBlocks - 1 - block;
            program.mem(
                first_cycle + block,
                mem_column,
                ftlpu::MemInstruction::Read(weight_address(matrix, column_block), stream_base + stream));
        }
    }

    for (std::size_t block = 0; block < kBlocks; ++block) {
        program.mxm(start_cycle + block, mxm, ftlpu::MxmControlInstruction::IW(buffer));
    }
}

void emit_weight_load(Program& program, std::size_t mxm, std::size_t matrix)
{
    const auto stream_base = mxm * kLoadStreams;
    emit_weight_load_at(
        program,
        mxm,
        matrix,
        kIwStart,
        0,
        stream_base,
        stream_base);
}

void emit_x_as_v_weight_load(Program& program)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = kXMemColumnBase + stream;
        const auto first_cycle = kQkIwStart - east_stream_cycles_to_sreg11(mem_column) - 1;
        for (std::size_t block = 0; block < kBlocks; ++block) {
            const auto row_block = kBlocks - 1 - block;
            program.mem(
                first_cycle + block,
                mem_column,
                ftlpu::MemInstruction::Read(
                    kXAddressBase + row_block * kLanes,
                    kLoadStreams + stream));
        }
    }

    for (std::size_t block = 0; block < kBlocks; ++block) {
        program.mxm(kQkIwStart + block, 1, ftlpu::MxmControlInstruction::IW(1));
    }
}

void emit_projection(Program& program)
{
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        const auto mem_column = x_column(row);
        const auto read_cycle = kGemmStart + row - east_stream_cycles_to_sreg11(mem_column);
        program.mem(read_cycle, mem_column, ftlpu::MemInstruction::Read(x_address(row), kActivationStream));

        program.mxm(
            kGemmStart + row,
            0,
            ftlpu::MxmControlInstruction::Compute(0, kActivationStream, kQWestStreamBase));
        program.mxm(
            kGemmStart + row,
            1,
            ftlpu::MxmControlInstruction::Compute(0, kActivationStream, kKWestStreamBase));

        program.vxm(kVxmStart + row, 0, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(kQStreamOperand),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        program.vxm(kVxmStart + row, 3, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(kKStreamOperand),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        program.vxm(kVxmStart + row + 1, 1, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::Imm(kProjectionScale),
        });
        program.vxm(kVxmStart + row + 1, 4, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(3),
            ftlpu::VxmLaneOperand::Imm(kProjectionScale),
        });
        program.vxm(kVxmStart + row + 2, 2, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(1),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Int8,
            kQInt8OutputStream,
        });
        program.vxm(kVxmStart + row + 2, 5, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(4),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Int8,
            kKInt8OutputStream,
        });

        const auto q_mem_column = q_column(row);
        const auto q_write_cycle = kVxmStart + row + kVxmLatency + east_stream_write_latency(q_mem_column);
        const auto k_write_cycle = kVxmStart + row + kVxmLatency + east_stream_write_latency(kKInt8Column);
        program.mem(q_write_cycle, q_mem_column, ftlpu::MemInstruction::Write(q_address(row), kQInt8OutputStream));
        program.mem(k_write_cycle, kKInt8Column, ftlpu::MemInstruction::Write(projection_output_address(row), kKInt8OutputStream));
    }
}

void emit_qk_matmul(Program& program)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = kQInt8ColumnBase + stream;
        const auto first_cycle = kQkIwStart - east_stream_cycles_to_sreg11(mem_column) - 1;
        for (std::size_t block = 0; block < kBlocks; ++block) {
            const auto column_block = kBlocks - 1 - block;
            const auto address = column_block * kLanes;
            program.mem(
                first_cycle + block,
                mem_column,
                ftlpu::MemInstruction::Read(address, stream));
        }
    }

    for (std::size_t block = 0; block < kBlocks; ++block) {
        program.mxm(kQkIwStart + block, 0, ftlpu::MxmControlInstruction::IW(1));
    }

    emit_x_as_v_weight_load(program);

    for (std::size_t row = 0; row < kSeqLen; ++row) {
        const auto read_cycle = kQkGemmStart + row - east_stream_cycles_to_sreg11(kKInt8Column);
        program.mem(
            read_cycle,
            kKInt8Column,
            ftlpu::MemInstruction::Read(projection_output_address(row), kActivationStream));
        program.mxm(
            kQkGemmStart + row,
            0,
            ftlpu::MxmControlInstruction::Compute(1, kActivationStream, kQkOutputWestStreamBase));

    }

    for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
        const auto mem_column = kLoadStreams + hidden % kLoadStreams;
        const auto activation_stream = kVActivationStreamBase + hidden % kLoadStreams;
        const auto v_read_cycle = kVProjectionGemmStart + hidden - east_stream_cycles_to_sreg11(mem_column);
        program.mem(
            v_read_cycle,
            mem_column,
            ftlpu::MemInstruction::Read(
                weight_address(kWvMatrix, hidden / kLoadStreams),
                activation_stream));
        program.mxm(
            kVProjectionGemmStart + hidden,
            1,
            ftlpu::MxmControlInstruction::Compute(1, activation_stream, kVOutputWestStreamBase));

        const auto vxm_cycle = kVProjectionOutputStart + hidden;
        program.vxm(vxm_cycle, 3, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(kVStreamOperand),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        program.vxm(vxm_cycle + 1, 4, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(3),
            ftlpu::VxmLaneOperand::Imm(kProjectionScale),
        });
        program.vxm(vxm_cycle + 2, 5, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(4),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Int8,
            kVInt8OutputStream,
        });
        const auto output_column = kVInt8ColumnBase + hidden % kVInt8Columns;
        program.mem(
            vxm_cycle + 2 + east_stream_write_latency(output_column),
            output_column,
            ftlpu::MemInstruction::Write(
                projection_output_address(hidden / kVInt8Columns),
                kVInt8OutputStream));
    }
}

struct SoftmaxSchedule {
    std::size_t pass1_max_write{0};
    std::size_t pass2_start{0};
    std::size_t pass2_sum_write{0};
    std::size_t pass3_start{0};
    std::size_t done{0};
};

void emit_fp32_mem_read(
    Program& program,
    std::size_t cycle,
    std::size_t column_base,
    std::size_t address,
    std::size_t stream_base)
{
    for (std::size_t byte = 0; byte < 4; ++byte) {
        program.mem(
            cycle,
            column_base + byte,
            ftlpu::MemInstruction::Read(address, stream_base + byte));
    }
}

void emit_fp32_mem_write(
    Program& program,
    std::size_t cycle,
    std::size_t column_base,
    std::size_t address,
    std::size_t stream_base)
{
    for (std::size_t byte = 0; byte < 4; ++byte) {
        program.mem(
            cycle,
            column_base + byte,
            ftlpu::MemInstruction::Write(address, stream_base + byte));
    }
}

SoftmaxSchedule emit_direct_score_softmax(Program& program)
{
    // MXM emits one key per cycle.  VXM lanes are queries, so feedback along
    // time reduces the 160 keys independently for every query lane.
    auto pass1_last_scaled_write = std::size_t {0};
    for (std::size_t key = 0; key < kSeqLen; ++key) {
        const auto issue_cycle = kDirectScoreVxmStart + key;
        const auto group = key % kSoftmaxParallelRows;
        const auto batch = key / kSoftmaxParallelRows;
        const auto stream_base = group * 4;
        program.vxm(issue_cycle, 0, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(kSoftmaxWestStreamBase + kQkOutputWestStreamBase),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        program.vxm(issue_cycle + 1, 1, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::Imm(kScoreScale),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
            stream_base,
        });
        auto max_instruction = ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Max,
            ftlpu::VxmLaneOperand::Alu(1),
            key == 0
                ? ftlpu::VxmLaneOperand::Imm(-std::numeric_limits<float>::infinity())
                : ftlpu::VxmLaneOperand::Alu(2),
        };
        if (key + 1 == kSeqLen) {
            max_instruction.output_stream = kReductionEastStreamBase;
        }
        program.vxm(issue_cycle + 2, 2, max_instruction);

        const auto column_base = kScaledScoreByteColumnBases[group];
        const auto write_cycle = issue_cycle + 1 + east_stream_write_latency(column_base);
        emit_fp32_mem_write(
            program,
            write_cycle,
            column_base,
            softmax_scratch_address(batch),
            stream_base);
        pass1_last_scaled_write = std::max(pass1_last_scaled_write, write_cycle);
    }

    const auto pass1_last_reduce = kDirectScoreVxmStart + kSeqLen + 1;
    const auto pass1_max_write = pass1_last_reduce + east_stream_write_latency(kRowMaxByteColumnBase);
    emit_fp32_mem_write(
        program,
        pass1_max_write,
        kRowMaxByteColumnBase,
        0,
        kReductionEastStreamBase);

    const auto pass2_start = std::max({pass1_max_write, pass1_last_scaled_write, kVProjectionLastWrite}) + 1;

    emit_fp32_mem_read(
        program,
        pass2_start,
        kRowMaxByteColumnBase,
        0,
        kReductionWestStreamBase);
    const auto pass2_max_arrival = pass2_start + west_stream_read_latency(kRowMaxByteColumnBase);
    program.vxm(pass2_max_arrival, 15, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Pass,
        ftlpu::VxmLaneOperand::StreamFloat32(kReductionWestStreamBase),
        ftlpu::VxmLaneOperand::Imm(0.0f),
    });

    const auto pass2_score_arrival = pass2_max_arrival + 1;
    auto pass2_last_exp_write = std::size_t {0};
    constexpr auto kSoftmaxBatches = kSeqLen / kSoftmaxParallelRows;
    for (std::size_t batch = 0; batch < kSoftmaxBatches; ++batch) {
        const auto sub_cycle = pass2_score_arrival + batch;
        for (std::size_t group = 0; group < kSoftmaxParallelRows; ++group) {
            const auto scaled_column_base = kScaledScoreByteColumnBases[group];
            const auto exp_column_base = kExpScoreByteColumnBases[group];
            const auto stream_base = group * 4;
            const auto alu_base = group * 4;
            emit_fp32_mem_read(
                program,
                sub_cycle - west_stream_read_latency(scaled_column_base),
                scaled_column_base,
                softmax_scratch_address(batch),
                kSoftmaxWestStreamBase + stream_base);

            program.vxm(sub_cycle, alu_base, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Subtract,
                ftlpu::VxmLaneOperand::StreamFloat32(kSoftmaxWestStreamBase + stream_base),
                ftlpu::VxmLaneOperand::Alu(15),
            });
            program.vxm(sub_cycle + 1, alu_base + 1, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Exp,
                ftlpu::VxmLaneOperand::Alu(alu_base),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                ftlpu::VxmCastTarget::Float32,
                stream_base,
            });
            program.vxm(sub_cycle + 2, alu_base + 2, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Add,
                ftlpu::VxmLaneOperand::Alu(alu_base + 1),
                batch == 0
                    ? ftlpu::VxmLaneOperand::Imm(0.0f)
                    : ftlpu::VxmLaneOperand::Alu(alu_base + 2),
            });

            const auto exp_write = sub_cycle + 1 + east_stream_write_latency(exp_column_base);
            emit_fp32_mem_write(
                program,
                exp_write,
                exp_column_base,
                softmax_scratch_address(batch),
                stream_base);
            pass2_last_exp_write = std::max(pass2_last_exp_write, exp_write);
        }
    }

    const auto pass2_last_partial_sum = pass2_score_arrival + kSoftmaxBatches + 1;
    const auto pass2_pair_sum = pass2_last_partial_sum + 1;
    program.vxm(pass2_pair_sum, 3, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::Alu(2),
        ftlpu::VxmLaneOperand::Alu(6),
    });
    program.vxm(pass2_pair_sum, 7, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::Alu(10),
        ftlpu::VxmLaneOperand::Alu(14),
    });
    const auto pass2_final_sum = pass2_pair_sum + 1;
    program.vxm(pass2_final_sum, 11, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::Alu(3),
        ftlpu::VxmLaneOperand::Alu(7),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float32,
        kReductionEastStreamBase,
    });
    const auto pass2_sum_write = pass2_final_sum + east_stream_write_latency(kRowSumByteColumnBase);
    emit_fp32_mem_write(
        program,
        pass2_sum_write,
        kRowSumByteColumnBase,
        0,
        kReductionEastStreamBase);

    const auto pass3_start = std::max(pass2_sum_write, pass2_last_exp_write) + 1;
    emit_fp32_mem_read(
        program,
        pass3_start,
        kRowSumByteColumnBase,
        0,
        kReductionWestStreamBase);
    const auto pass3_sum_arrival = pass3_start + west_stream_read_latency(kRowSumByteColumnBase);
    program.vxm(pass3_sum_arrival, 15, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Pass,
        ftlpu::VxmLaneOperand::StreamFloat32(kReductionWestStreamBase),
        ftlpu::VxmLaneOperand::Imm(0.0f),
    });

    const auto pass3_exp_arrival = pass3_sum_arrival + 1;
    auto pass3_last_write = std::size_t {0};
    for (std::size_t batch = 0; batch < kSoftmaxBatches; ++batch) {
        const auto div_cycle = pass3_exp_arrival + batch;
        for (std::size_t group = 0; group < kSoftmaxParallelRows; ++group) {
            const auto exp_column_base = kExpScoreByteColumnBases[group];
            const auto final_column = kSoftmaxInt8ColumnBase + group;
            const auto input_stream_base = group * 4;
            const auto alu_base = group * 4;
            emit_fp32_mem_read(
                program,
                div_cycle - west_stream_read_latency(exp_column_base),
                exp_column_base,
                softmax_scratch_address(batch),
                kSoftmaxWestStreamBase + input_stream_base);

            program.vxm(div_cycle, alu_base, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Divide,
                ftlpu::VxmLaneOperand::StreamFloat32(kSoftmaxWestStreamBase + input_stream_base),
                ftlpu::VxmLaneOperand::Alu(15),
            });
            program.vxm(div_cycle + 1, alu_base + 1, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Multiply,
                ftlpu::VxmLaneOperand::Alu(alu_base),
                ftlpu::VxmLaneOperand::Imm(kSoftmaxInt8Scale),
            });
            program.vxm(div_cycle + 2, alu_base + 2, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Cast,
                ftlpu::VxmLaneOperand::Alu(alu_base + 1),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                ftlpu::VxmCastTarget::Int8,
                group,
            });
            const auto write_cycle = div_cycle + 2 + east_stream_write_latency(final_column);
            program.mem(
                write_cycle,
                final_column,
                ftlpu::MemInstruction::Write(softmax_scratch_address(batch), group));
            pass3_last_write = std::max(pass3_last_write, write_cycle);
        }
    }

    const auto done = pass3_last_write + kBlocks + 8;
    return SoftmaxSchedule {pass1_max_write, pass2_start, pass2_sum_write, pass3_start, done};
}

struct AttentionOutputSchedule {
    std::size_t v_weight_read_start{0};
    std::size_t v_iw_start{0};
    std::size_t softmax_transpose_start{0};
    std::size_t softmax_emit_start{0};
    std::size_t compute_start{0};
    std::size_t write_start{0};
    std::size_t done{0};
};

ftlpu::Permute320::Map attention_lane_reverse_map()
{
    auto map = ftlpu::Permute320::identity_map();
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            map[tile * kLanes + lane] = tile * kLanes + (kLanes - 1 - lane);
        }
    }
    return map;
}

AttentionOutputSchedule emit_attention_output(Program& program)
{
    constexpr auto kWavefrontCycles = kSeqLen + ftlpu::hw::kTileRows - 1;
    constexpr auto kVIwStart = std::size_t {16};
    constexpr auto kSoftmaxTransposeStart = kVIwStart + kBlocks;
    constexpr auto kSoftmaxPermuteCaptureStart = kSoftmaxTransposeStart + kLanes - 1;
    constexpr auto kSoftmaxEmitStart = kSoftmaxPermuteCaptureStart + kWavefrontCycles;
    constexpr auto kComputeStart = kSoftmaxEmitStart + 1;
    constexpr auto kFirstOutput = kComputeStart + kBlocks - 1;
    constexpr auto kWriteStart =
        kFirstOutput + mxm_west_stream_write_latency(kAttentionOutputColumnBase);
    constexpr auto kDone = kWriteStart + kSeqLen + kBlocks + 8;

    const auto east1 = ftlpu::SxmStreamId {
        ftlpu::StreamId::East(kAttentionActivationStream).packed()};
    const auto reverse_map = attention_lane_reverse_map();

    auto first_v_weight_read = std::numeric_limits<std::size_t>::max();
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = kVInt8ColumnBase + stream;
        const auto first_cycle = kVIwStart - east_stream_cycles_to_sreg11(mem_column) - 1;
        first_v_weight_read = std::min(first_v_weight_read, first_cycle);
        for (std::size_t block = 0; block < kBlocks; ++block) {
            const auto hidden_block = kBlocks - 1 - block;
            program.mem(
                first_cycle + block,
                mem_column,
                ftlpu::MemInstruction::Read(
                    projection_output_address(hidden_block),
                    kAttentionWeightStreamBase + stream));
        }
    }
    for (std::size_t block = 0; block < kBlocks; ++block) {
        program.mxm(kVIwStart + block, 1, ftlpu::MxmControlInstruction::IW(0));
    }

    const auto first_softmax_read =
        kSoftmaxTransposeStart - east_stream_cycles_to_sxm(kSoftmaxInt8ColumnBase) - 1;
    for (std::size_t phase = 0; phase < kSeqLen; ++phase) {
        const auto block = phase / kLanes;
        const auto lane = phase % kLanes;
        const auto key = block * kLanes + (kLanes - 1 - lane);
        const auto group = key % kSoftmaxParallelRows;
        const auto batch = key / kSoftmaxParallelRows;
        program.mem(
            first_softmax_read + phase,
            kSoftmaxInt8ColumnBase + group,
            ftlpu::MemInstruction::Read(
                softmax_scratch_address(batch),
                kAttentionActivationStream));
    }
    for (std::size_t cycle = 0; cycle < kWavefrontCycles; ++cycle) {
        program.sxm_transpose(
            kSoftmaxTransposeStart + cycle,
            ftlpu::SxmInstruction::TransposeStream(east1, east1));
    }
    for (std::size_t cycle = 0; cycle < kWavefrontCycles + kSeqLen; ++cycle) {
        program.sxm_permute(
            kSoftmaxPermuteCaptureStart + cycle,
            ftlpu::SxmInstruction::PermuteStream(east1, east1, reverse_map));
    }
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        program.mxm(
            kComputeStart + query,
            1,
            ftlpu::MxmControlInstruction::Compute(
                0,
                kAttentionActivationStream,
                kAttentionOutputWestStreamBase));
        for (std::size_t byte = 0; byte < sizeof(std::int32_t); ++byte) {
            program.mem(
                kWriteStart + query,
                kAttentionOutputColumnBase + byte,
                ftlpu::MemInstruction::Write(
                    query * kLanes,
                    ftlpu::hw::kEastStreams + kAttentionOutputWestStreamBase + byte));
        }
    }

    return AttentionOutputSchedule {
        first_v_weight_read,
        kVIwStart,
        kSoftmaxTransposeStart,
        kSoftmaxEmitStart,
        kComputeStart,
        kWriteStart,
        kDone,
    };
}

std::int8_t stored_q_int8(const ftlpu::TileArrayModel& mem, std::size_t row, std::size_t column)
{
    const auto tile = column / kLanes;
    const auto lane = column % kLanes;
    return static_cast<std::int8_t>(mem.sram_lane_byte(q_column(row), tile, q_address(row), lane));
}

std::int8_t stored_k_int8(const ftlpu::TileArrayModel& mem, std::size_t row, std::size_t column)
{
    const auto tile = column / kLanes;
    const auto lane = column % kLanes;
    return static_cast<std::int8_t>(mem.sram_lane_byte(kKInt8Column, tile, projection_output_address(row), lane));
}

std::size_t score_index(std::size_t row, std::size_t column)
{
    return row * kSeqLen + column;
}

float read_float32(
    const ftlpu::TileArrayModel& mem,
    std::size_t column_base,
    std::size_t tile,
    std::size_t address,
    std::size_t lane)
{
    const auto raw = static_cast<std::uint32_t>(mem.sram_lane_byte(column_base + 0, tile, address, lane))
        | (static_cast<std::uint32_t>(mem.sram_lane_byte(column_base + 1, tile, address, lane)) << 8)
        | (static_cast<std::uint32_t>(mem.sram_lane_byte(column_base + 2, tile, address, lane)) << 16)
        | (static_cast<std::uint32_t>(mem.sram_lane_byte(column_base + 3, tile, address, lane)) << 24);
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

std::vector<std::int8_t> read_softmax_output(const ftlpu::TileArrayModel& mem)
{
    auto output = std::vector<std::int8_t>(kSeqLen * kSeqLen, 0);
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        const auto tile = query / kLanes;
        const auto lane = query % kLanes;
        for (std::size_t key = 0; key < kSeqLen; ++key) {
            const auto group = key % kSoftmaxParallelRows;
            const auto batch = key / kSoftmaxParallelRows;
            output[score_index(query, key)] = static_cast<std::int8_t>(
                mem.sram_lane_byte(
                    kSoftmaxInt8ColumnBase + group,
                    tile,
                    softmax_scratch_address(batch),
                    lane));
        }
    }
    return output;
}

std::int8_t stored_v_int8(const ftlpu::TileArrayModel& mem, std::size_t row, std::size_t column)
{
    const auto tile = row / kLanes;
    const auto lane = row % kLanes;
    return static_cast<std::int8_t>(mem.sram_lane_byte(
        kVInt8ColumnBase + column % kVInt8Columns,
        tile,
        projection_output_address(column / kVInt8Columns),
        lane));
}

std::int32_t stored_attention_output(
    const ftlpu::TileArrayModel& mem,
    std::size_t query,
    std::size_t hidden)
{
    const auto tile = hidden / kLanes;
    const auto lane = hidden % kLanes;
    auto raw = std::uint32_t {0};
    for (std::size_t byte = 0; byte < sizeof(std::int32_t); ++byte) {
        raw |= static_cast<std::uint32_t>(mem.sram_lane_byte(
                   kAttentionOutputColumnBase + byte,
                   tile,
                   query * kLanes,
                   lane))
            << (byte * 8);
    }
    return static_cast<std::int32_t>(raw);
}

std::int32_t attention_output_golden(
    const std::vector<std::int8_t>& softmax,
    const ftlpu::TileArrayModel& mem,
    std::size_t query,
    std::size_t hidden)
{
    auto sum = std::int32_t {0};
    for (std::size_t key = 0; key < kSeqLen; ++key) {
        sum += static_cast<std::int32_t>(softmax[score_index(query, key)])
            * static_cast<std::int32_t>(stored_v_int8(mem, key, hidden));
    }
    return sum;
}

struct AttentionPipelineBlock {
    std::size_t row{0};
    std::size_t start{0};
    std::size_t duration{0};
    std::string label{};
    const char* fill{"#eeeeee"};
};

void write_attention_pipeline_svg(
    const std::filesystem::path& path,
    const SoftmaxSchedule& softmax,
    const AttentionOutputSchedule& output)
{
    constexpr double kWidth = 1600.0;
    constexpr double kHeight = 650.0;
    constexpr double kLeft = 155.0;
    constexpr double kRight = 40.0;
    constexpr double kTop = 62.0;
    constexpr double kRowGap = 54.0;
    constexpr double kBlockHeight = 34.0;
    constexpr std::array<const char*, 9> kRows {
        "MEM read",
        "MEM write",
        "MXM0 load",
        "MXM0 compute",
        "MXM1 load",
        "MXM1 compute",
        "VXM",
        "SXM transpose",
        "SXM permute",
    };

    const auto final_base = softmax.done;
    const auto total_cycles = final_base + output.done;
    auto blocks = std::vector<AttentionPipelineBlock> {};
    const auto add = [&](std::size_t row, std::size_t start, std::size_t duration,
                         std::string label, const char* fill) {
        blocks.push_back(AttentionPipelineBlock {row, start, duration, std::move(label), fill});
    };

    add(0, 7, 33, "Wq/Wk", "#78b957");
    add(0, kGemmStart - east_stream_cycles_to_sreg11(kXMemColumnBase), kSeqLen + 4, "X rows", "#a8d29a");
    add(0, kQkGemmStart - east_stream_cycles_to_sreg11(kKInt8Column), kSeqLen, "K", "#a8d29a");
    add(0, kVProjectionGemmStart - east_stream_cycles_to_sreg11(kLoadStreams), kHidden, "Wv columns", "#78b957");
    add(0, softmax.pass2_start, softmax.pass3_start - softmax.pass2_start, "softmax P2", "#9bc8b4");
    add(0, softmax.pass3_start, softmax.done - softmax.pass3_start, "softmax P3", "#9bc8b4");
    add(0, final_base + output.v_weight_read_start, kBlocks + 4, "V direct", "#78b957");
    add(0, final_base + output.softmax_transpose_start - 2, kSeqLen, "softmax", "#a8d29a");

    add(1, kVxmStart, kSeqLen + 18, "Q/K", "#f5d28a");
    add(1, kVProjectionOutputStart, kHidden + 8, "V^T", "#f5d28a");
    add(1, kDirectScoreVxmStart, softmax.pass2_start - kDirectScoreVxmStart, "P1 scratch", "#f7dfaa");
    add(1, softmax.pass2_start, softmax.done - softmax.pass2_start, "P2/P3", "#f7dfaa");
    add(1, final_base + output.write_start, kSeqLen + kBlocks, "attention out", "#f5d28a");

    add(2, kIwStart, kBlocks, "Wq", "#dddddd");
    add(2, kQkIwStart, kBlocks, "Q", "#dddddd");
    add(3, kGemmStart, kSeqLen, "Q", "#eeeeee");
    add(3, kQkGemmStart, kSeqLen, "QK", "#eeeeee");

    add(4, kIwStart, kBlocks, "Wk", "#d9d9d9");
    add(4, kQkIwStart, kBlocks, "X as columns", "#d9d9d9");
    add(4, final_base + output.v_iw_start, kBlocks, "V direct", "#d9d9d9");
    add(5, kGemmStart, kSeqLen, "K", "#eeeeee");
    add(5, kVProjectionGemmStart, kHidden, "Wv columns -> V^T", "#eeeeee");
    add(5, final_base + output.compute_start, kSeqLen, "softmax x V", "#eeeeee");

    add(6, kVxmStart, kSeqLen + 3, "Q/K requant", "#6f9fe8");
    add(6, kDirectScoreVxmStart, kSeqLen + 2, "P1", "#6f9fe8");
    add(6, softmax.pass2_start, softmax.pass3_start - softmax.pass2_start, "P2", "#7da8e8");
    add(6, softmax.pass3_start, softmax.done - softmax.pass3_start, "P3", "#7da8e8");

    add(7, final_base + output.softmax_transpose_start, kSeqLen + ftlpu::hw::kTileRows - 1, "softmax", "#e7a35c");
    add(8, final_base + output.softmax_transpose_start + kLanes - 1,
        2 * kSeqLen + ftlpu::hw::kTileRows - 1, "softmax -> rows", "#8cc7c0");

    std::ofstream os(path);
    if (!os.good()) {
        throw std::runtime_error("failed to write attention pipeline SVG");
    }
    const auto scale = (kWidth - kLeft - kRight) / static_cast<double>(total_cycles);
    os << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << kWidth
       << "\" height=\"" << kHeight << "\" viewBox=\"0 0 " << kWidth << ' ' << kHeight << "\">\n";
    os << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";
    os << "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"8\" refX=\"9\" refY=\"4\" orient=\"auto\">"
          "<path d=\"M0,0 L10,4 L0,8 Z\" fill=\"#555\"/></marker></defs>\n";
    os << "<g font-family=\"Arial, Helvetica, sans-serif\" fill=\"#111\">\n";
    os << "<line x1=\"" << kLeft << "\" y1=\"28\" x2=\"" << (kWidth - kRight - 100)
       << "\" y2=\"28\" stroke=\"#555\" stroke-width=\"3\" stroke-dasharray=\"12 10\" marker-end=\"url(#arrow)\"/>\n";
    os << "<text x=\"" << (kWidth - kRight - 82) << "\" y=\"35\" font-size=\"27\">Time</text>\n";

    for (std::size_t cycle = 0; cycle <= total_cycles; cycle += 200) {
        const auto x = kLeft + static_cast<double>(cycle) * scale;
        os << "<line x1=\"" << x << "\" y1=\"48\" x2=\"" << x
           << "\" y2=\"" << (kTop + kRows.size() * kRowGap - 12)
           << "\" stroke=\"#dddddd\" stroke-width=\"1\"/>\n";
        os << "<text x=\"" << x << "\" y=\"" << (kHeight - 24)
           << "\" text-anchor=\"middle\" font-size=\"13\" fill=\"#666\">" << cycle << "</text>\n";
    }
    for (std::size_t row = 0; row < kRows.size(); ++row) {
        const auto y = kTop + static_cast<double>(row) * kRowGap;
        os << "<text x=\"18\" y=\"" << (y + 23)
           << "\" font-size=\"17\">" << kRows[row] << "</text>\n";
    }
    for (const auto& block : blocks) {
        const auto x = kLeft + static_cast<double>(block.start) * scale;
        const auto y = kTop + static_cast<double>(block.row) * kRowGap;
        const auto width = std::max(2.0, static_cast<double>(block.duration) * scale);
        os << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << width
           << "\" height=\"" << kBlockHeight << "\" fill=\"" << block.fill
           << "\" stroke=\"#222\" stroke-width=\"1.2\"/>\n";
        if (width >= 42.0) {
            os << "<text x=\"" << (x + width / 2.0) << "\" y=\"" << (y + 22)
               << "\" text-anchor=\"middle\" font-size=\"12\">" << block.label << "</text>\n";
        }
    }
    os << "<text x=\"" << kLeft << "\" y=\"" << (kHeight - 52)
       << "\" font-size=\"14\" fill=\"#555\">total: " << total_cycles << " cycles</text>\n";
    os << "</g>\n</svg>\n";
}

float softmax_golden_max(const std::vector<std::int32_t>& scores, std::size_t query)
{
    auto max_value = -std::numeric_limits<float>::infinity();
    for (std::size_t key = 0; key < kSeqLen; ++key) {
        max_value = std::max(
            max_value,
            static_cast<float>(scores[score_index(key, query)]) * kScoreScale);
    }
    return max_value;
}

float softmax_golden_sum(
    const std::vector<std::int32_t>& scores,
    std::size_t query,
    float max_value)
{
    auto sum = 0.0f;
    for (std::size_t key = 0; key < kSeqLen; ++key) {
        const auto scaled = static_cast<float>(scores[score_index(key, query)]) * kScoreScale;
        sum += std::exp(scaled - max_value);
    }
    return sum;
}

std::int8_t softmax_golden_value(
    const std::vector<std::int32_t>& scores,
    std::size_t query,
    std::size_t key,
    float max_value,
    float sum)
{
    const auto scaled = static_cast<float>(scores[score_index(key, query)]) * kScoreScale;
    const auto probability = std::exp(scaled - max_value) / sum;
    return ftlpu::VxmAlu::cast_scalar_to_int8(probability * kSoftmaxInt8Scale);
}

} // namespace

int main()
try
{
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    stage_mem(system->mem());

    auto program = Program {};
    emit_weight_load(program, 0, kWqMatrix);
    emit_weight_load(program, 1, kWkMatrix);
    emit_projection(program);
    emit_qk_matmul(program);
    const auto softmax_schedule = emit_direct_score_softmax(program);
    program.load(system->icu());

    const auto log_dir = std::filesystem::path("logs") / "single_head_attention";
    std::filesystem::create_directories(log_dir);
    auto icu_log = std::ofstream(log_dir / "icu.log");
    if (!icu_log.good()) {
        std::cerr << "failed to open single-head attention ICU log\n";
        return 1;
    }
    icu_log << "single_head_attention transpose_free_v_projection\n"
            << "  qk_output_start=" << kQkOutputStart
            << " direct_score_vxm_start=" << kDirectScoreVxmStart
            << " scaled_score_columns=16..31 (4 striped fp32 groups)"
            << " max_columns=" << kRowMaxByteColumnBase << ".." << (kRowMaxByteColumnBase + 3)
            << " exp_columns=0..15 (4 striped fp32 groups)"
            << " sum_columns=" << kRowSumByteColumnBase << ".." << (kRowSumByteColumnBase + 3)
            << " softmax_columns=" << kSoftmaxInt8ColumnBase << ".."
            << (kSoftmaxInt8ColumnBase + kSoftmaxParallelRows - 1)
            << " pass2_start=" << softmax_schedule.pass2_start
            << " pass3_start=" << softmax_schedule.pass3_start << '\n';

    std::ostringstream sink;
    auto qk_scores = std::vector<std::int32_t>(kSeqLen * kSeqLen, 0);
    const auto total_cycles = softmax_schedule.done;
    for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
        try {
            system->dispatch_icu_only(&icu_log);
            system->tick_mxm_controls_only(ftlpu::TspSliceSystem::LogSinks {nullptr, nullptr, nullptr, nullptr, nullptr});
            system->mem().tick(sink);
            system->tick_mxm_datapaths_only(ftlpu::TspSliceSystem::LogSinks {nullptr, nullptr, nullptr, nullptr, nullptr});
            if (cycle >= kQkOutputStart && cycle < kQkOutputStart + kSeqLen + kBlocks) {
                for (const auto& output : system->mxm_unit(0).last_outputs()) {
                    if (output.row >= kSeqLen) {
                        continue;
                    }
                    const auto column_base = output.column_block * kLanes;
                    if (column_base >= kSeqLen) {
                        continue;
                    }
                    for (std::size_t lane = 0; lane < kLanes && column_base + lane < kSeqLen; ++lane) {
                        qk_scores[score_index(output.row, column_base + lane)] = output.values[lane];
                    }
                }
            }
            system->tick_vxm_stream_bridge(ftlpu::TspSliceSystem::LogSinks {nullptr, &sink, nullptr, nullptr, nullptr});
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "cycle " << cycle << ": " << ex.what();
            throw std::logic_error(os.str());
        }
    }

    const std::array<std::size_t, 4> sample_rows {0, 7, 79, 159};
    const std::array<std::size_t, 6> sample_columns {0, 1, 15, 16, 127, 319};
    for (const auto row : sample_rows) {
        for (const auto column : sample_columns) {
            const auto q_actual = stored_q_int8(system->mem(), row, column);
            const auto k_actual = stored_k_int8(system->mem(), row, column);
            const auto v_actual = stored_v_int8(system->mem(), row, column);
            const auto q_expected = projection_int8(kWqMatrix, row, column);
            const auto k_expected = projection_int8(kWkMatrix, row, column);
            const auto v_expected = projection_int8(kWvMatrix, row, column);
            if (q_actual != q_expected || k_actual != k_expected || v_actual != v_expected) {
                std::cerr << "attention Q/K/V projection mismatch row=" << row
                          << " column=" << column
                          << " q_actual=" << static_cast<int>(q_actual)
                          << " q_expected=" << static_cast<int>(q_expected)
                          << " k_actual=" << static_cast<int>(k_actual)
                          << " k_expected=" << static_cast<int>(k_expected)
                          << " v_actual=" << static_cast<int>(v_actual)
                          << " v_expected=" << static_cast<int>(v_expected)
                          << '\n';
                return 1;
            }
        }
    }

    const std::array<std::size_t, 5> score_rows {0, 3, 31, 80, 159};
    const std::array<std::size_t, 5> score_columns {0, 2, 47, 128, 159};
    for (const auto row : score_rows) {
        for (const auto column : score_columns) {
            const auto actual = qk_scores[score_index(row, column)];
            const auto expected = qk_value(row, column);
            if (actual != expected) {
                std::cerr << "attention QK mismatch row=" << row
                          << " column=" << column
                          << " actual=" << actual
                          << " expected=" << expected
                          << '\n';
                return 1;
            }
        }
    }

    auto golden_max = std::array<float, kSeqLen> {};
    auto golden_sum = std::array<float, kSeqLen> {};
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        const auto tile = query / kLanes;
        const auto lane = query % kLanes;
        const auto max_actual = read_float32(system->mem(), kRowMaxByteColumnBase, tile, 0, lane);
        const auto max_expected = softmax_golden_max(qk_scores, query);
        const auto sum_actual = read_float32(system->mem(), kRowSumByteColumnBase, tile, 0, lane);
        const auto sum_expected = softmax_golden_sum(qk_scores, query, max_expected);
        golden_max[query] = max_expected;
        golden_sum[query] = sum_expected;
        const auto sum_tolerance = std::max(1.0e-5f, std::fabs(sum_expected) * 1.0e-5f);
        if (max_actual != max_expected || std::fabs(sum_actual - sum_expected) > sum_tolerance) {
            std::cerr << "attention softmax reduction mismatch query=" << query
                      << " max_actual=" << max_actual
                      << " max_expected=" << max_expected
                      << " sum_actual=" << sum_actual
                      << " sum_expected=" << sum_expected << '\n';
            return 1;
        }
    }

    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto actual = stored_v_int8(system->mem(), row, column);
            const auto expected = projection_int8(kWvMatrix, row, column);
            if (actual != expected) {
                std::cerr << "attention V projection mismatch row=" << row
                          << " column=" << column
                          << " actual=" << static_cast<int>(actual)
                          << " expected=" << static_cast<int>(expected) << '\n';
                return 1;
            }
        }
    }

    const auto softmax = read_softmax_output(system->mem());
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        for (std::size_t key = 0; key < kSeqLen; ++key) {
            const auto actual = softmax[score_index(query, key)];
            const auto expected = softmax_golden_value(
                qk_scores,
                query,
                key,
                golden_max[query],
                golden_sum[query]);
            if (actual != expected) {
                std::cerr << "attention softmax mismatch query=" << query
                          << " key=" << key
                          << " actual=" << static_cast<int>(actual)
                          << " expected=" << static_cast<int>(expected)
                          << '\n';
                return 1;
            }
        }
    }

    auto expected_attention = std::vector<std::int32_t>(kSeqLen * kHidden, 0);
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            expected_attention[query * kHidden + hidden] =
                attention_output_golden(softmax, system->mem(), query, hidden);
        }
    }

    auto attention_program = Program {};
    const auto attention_schedule = emit_attention_output(attention_program);
    attention_program.load(system->icu());
    icu_log << "attention_output_phase"
            << " v_weight_read_start=" << attention_schedule.v_weight_read_start
            << " v_iw_start=" << attention_schedule.v_iw_start
            << " softmax_transpose_start=" << attention_schedule.softmax_transpose_start
            << " softmax_emit_start=" << attention_schedule.softmax_emit_start
            << " compute_start=" << attention_schedule.compute_start
            << " write_start=" << attention_schedule.write_start << '\n';
    write_attention_pipeline_svg(
        log_dir / "pipeline.svg",
        softmax_schedule,
        attention_schedule);
    icu_log << "pipeline diagram: " << (log_dir / "pipeline.svg").string() << '\n';

    auto attention_output_blocks = std::size_t {0};
    for (std::size_t cycle = 0; cycle < attention_schedule.done; ++cycle) {
        try {
            auto sinks = ftlpu::TspSliceSystem::LogSinks {};
            sinks.icu = &icu_log;
            system->tick(sinks);
            if (cycle + 1 == attention_schedule.compute_start) {
                const auto& weights = system->mxm_unit(1).array();
                for (std::size_t key = 0; key < kSeqLen; ++key) {
                    for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                        const auto actual = weights.weight(
                            0,
                            key / kLanes,
                            hidden / kLanes,
                            key % kLanes,
                            hidden % kLanes);
                        const auto expected = stored_v_int8(system->mem(), key, hidden);
                        if (actual != expected) {
                            std::cerr << "attention V weight load mismatch key=" << key
                                      << " hidden=" << hidden
                                      << " actual=" << static_cast<int>(actual)
                                      << " expected=" << static_cast<int>(expected) << '\n';
                            return 1;
                        }
                    }
                }
            }
            for (const auto& output : system->mxm_unit(1).last_outputs()) {
                if (output.row >= kSeqLen) {
                    continue;
                }
                ++attention_output_blocks;
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    const auto hidden = output.column_block * kLanes + lane;
                    const auto expected = expected_attention[output.row * kHidden + hidden];
                    if (output.values[lane] != expected) {
                        std::cerr << "attention output MXM mismatch phase_cycle=" << cycle
                                  << " query=" << output.row
                                  << " hidden=" << hidden
                                  << " actual=" << output.values[lane]
                                  << " expected=" << expected << '\n';
                        return 1;
                    }
                }
            }
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "attention output phase cycle " << cycle << ": " << ex.what();
            throw std::logic_error(os.str());
        }
    }

    if (attention_output_blocks != kSeqLen * kBlocks) {
        std::cerr << "attention output emitted " << attention_output_blocks
                  << " MXM blocks, expected " << kSeqLen * kBlocks << '\n';
        return 1;
    }
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto actual = stored_attention_output(system->mem(), query, hidden);
            const auto expected = expected_attention[query * kHidden + hidden];
            if (actual != expected) {
                std::cerr << "attention output SRAM mismatch query=" << query
                          << " hidden=" << hidden
                          << " actual=" << actual
                          << " expected=" << expected << '\n';
                return 1;
            }
        }
    }

    return 0;
} catch (const std::exception& ex) {
    std::cerr << "single_head_attention_test failed: " << ex.what() << '\n';
    return 1;
}
