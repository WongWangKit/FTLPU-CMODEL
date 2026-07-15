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
constexpr std::size_t kXMemColumn = 32;
constexpr std::size_t kQInt8ColumnBase = 16;
constexpr std::size_t kQInt8Columns = kLoadStreams;
constexpr std::size_t kKInt8Column = 41;
constexpr std::size_t kScaledScoreByteColumnBase = 36;
constexpr std::size_t kExpScoreByteColumnBase = 0;
constexpr std::size_t kSoftmaxInt8Column = 42;

constexpr std::size_t kQWestStreamBase = 0;
constexpr std::size_t kKWestStreamBase = 4;
constexpr std::size_t kActivationStream = 8;
constexpr std::size_t kQkOutputWestStreamBase = 12;
constexpr std::size_t kSoftmaxWestStreamBase = ftlpu::hw::kEastStreams;
constexpr std::size_t kSoftmaxEastStreamBase = 0;
constexpr std::size_t kQStreamOperand = ftlpu::hw::kEastStreams + kQWestStreamBase;
constexpr std::size_t kKStreamOperand = ftlpu::hw::kEastStreams + kKWestStreamBase;
constexpr std::size_t kQInt8OutputStream = 0;
constexpr std::size_t kKInt8OutputStream = 1;

constexpr std::size_t kIwStart = 20;
constexpr std::size_t kGemmStart = kIwStart + 2 * kBlocks;
constexpr std::size_t kMxmOutputStart = kGemmStart + kBlocks - 1;
constexpr std::size_t kVxmStart = kMxmOutputStart + ftlpu::hw::kStreamRegisterColumns;
constexpr std::size_t kVxmLatency = 2;
constexpr std::size_t kQkIwStart = kVxmStart + kSeqLen + kVxmLatency + east_stream_write_latency(kKInt8Column) + 8;
constexpr std::size_t kQkGemmStart = kQkIwStart + kBlocks;
constexpr std::size_t kQkOutputStart = kQkGemmStart + kBlocks - 1;
constexpr std::size_t kDirectScoreVxmStart = kQkOutputStart + ftlpu::hw::kStreamRegisterColumns;
constexpr float kProjectionScale = 1.0f / 256.0f;
constexpr float kScoreScale = 1.0f / 17.88854381999832f; // 1 / sqrt(320)
constexpr float kSoftmaxInt8Scale = 127.0f;

constexpr std::size_t west_stream_read_latency(std::size_t column)
{
    return column / ftlpu::hw::kSlicesPerGroup + 1;
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

std::size_t x_address(std::size_t row, std::size_t lane)
{
    return row * kLanes + lane;
}

std::size_t weight_address(std::size_t matrix, std::size_t column_block)
{
    return matrix * kBlocks * kLanes + column_block * kLanes;
}

std::size_t projection_output_address(std::size_t row)
{
    return row * kLanes;
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
                kXMemColumn,
                k / kLanes,
                x_address(row, 0),
                k % kLanes,
                static_cast<std::uint8_t>(x_value(row, k)));
        }
    }

    for (std::size_t matrix = 0; matrix < 2; ++matrix) {
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

    void load(ftlpu::InstructionControlUnit& icu)
    {
        for (std::size_t column = 0; column < mem_.size(); ++column) {
            load_queue(mem_[column], [&](std::size_t n) { icu.enqueue_mem_nop(column, n); }, [&](auto instruction) {
                icu.enqueue_mem(column, instruction);
            });
        }
        for (std::size_t mxm = 0; mxm < mxm_load_.size(); ++mxm) {
            load_queue(mxm_load_[mxm], [&](std::size_t n) { icu.enqueue_mxm_load_nop(mxm, n); }, [&](auto instruction) {
                icu.enqueue_mxm(mxm, instruction);
            });
            load_queue(mxm_compute_[mxm], [&](std::size_t n) { icu.enqueue_mxm_compute_nop(mxm, n); }, [&](auto instruction) {
                icu.enqueue_mxm(mxm, instruction);
            });
        }
        for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
            load_queue(vxm_[alu], [&](std::size_t n) { icu.enqueue_vxm_nop(alu, n); }, [&](auto instruction) {
                icu.enqueue_vxm(alu, instruction);
            });
        }
    }

private:
    template <typename T>
    struct Event {
        std::size_t cycle{0};
        T instruction{};
    };

    template <typename T, typename Nop, typename Emit>
    static void load_queue(std::vector<Event<T>>& events, Nop nop, Emit emit)
    {
        std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.cycle < rhs.cycle;
        });
        auto cursor = std::size_t {0};
        for (const auto& event : events) {
            if (event.cycle < cursor) {
                throw std::logic_error("attention projection program queue collision");
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
};

void emit_weight_load(Program& program, std::size_t mxm, std::size_t matrix)
{
    const auto stream_base = mxm * kLoadStreams;
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mem_column = stream_base + stream;
        const auto first_cycle = kIwStart - east_stream_cycles_to_sreg11(mem_column) - 1;
        for (std::size_t block = 0; block < kBlocks; ++block) {
            const auto column_block = kBlocks - 1 - block;
            program.mem(
                first_cycle + block,
                mem_column,
                ftlpu::MemInstruction::Read(weight_address(matrix, column_block), stream_base + stream));
        }
    }

    for (std::size_t block = 0; block < kBlocks; ++block) {
        program.mxm(kIwStart + block, mxm, ftlpu::MxmControlInstruction::IW(0));
    }
}

void emit_projection(Program& program)
{
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        const auto read_cycle = kGemmStart + row - east_stream_cycles_to_sreg11(kXMemColumn);
        program.mem(read_cycle, kXMemColumn, ftlpu::MemInstruction::Read(x_address(row, 0), kActivationStream));

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
}

void emit_direct_score_softmax_pass1(Program& program)
{
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        const auto issue_cycle = kDirectScoreVxmStart + row;
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
        });
        program.vxm(issue_cycle + 2, 2, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(1),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
            kSoftmaxEastStreamBase,
        });

        const auto write_cycle = issue_cycle + 2 + east_stream_write_latency(kScaledScoreByteColumnBase);
        for (std::size_t byte = 0; byte < 4; ++byte) {
            program.mem(
                write_cycle,
                kScaledScoreByteColumnBase + byte,
                ftlpu::MemInstruction::Write(projection_output_address(row), kSoftmaxEastStreamBase + byte));
        }
    }
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

void tick_softmax_cycle(ftlpu::TspSliceSystem& system, std::ostream* icu_log)
{
    system.dispatch_icu_only(icu_log);
    system.mem().tick();
    system.tick_vxm_stream_bridge(ftlpu::TspSliceSystem::LogSinks {nullptr, nullptr, nullptr, nullptr, nullptr});
}

void enqueue_fp32_mem_reads(ftlpu::InstructionControlUnit& icu, std::size_t column_base, std::size_t address)
{
    for (std::size_t byte = 0; byte < 4; ++byte) {
        icu.enqueue_mem(
            column_base + byte,
            ftlpu::MemInstruction::Read(address, kSoftmaxWestStreamBase + byte));
    }
}

void enqueue_fp32_mem_writes(ftlpu::InstructionControlUnit& icu, std::size_t column_base, std::size_t address)
{
    for (std::size_t byte = 0; byte < 4; ++byte) {
        icu.enqueue_mem(
            column_base + byte,
            ftlpu::MemInstruction::Write(address, kSoftmaxEastStreamBase + byte));
    }
}

std::vector<float> compute_row_max_from_mem(const ftlpu::TileArrayModel& mem)
{
    auto row_max = std::vector<float>(kSeqLen, -std::numeric_limits<float>::infinity());
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kSeqLen; ++column) {
            const auto tile = column / kLanes;
            const auto lane = column % kLanes;
            row_max[row] = std::max(
                row_max[row],
                read_float32(mem, kScaledScoreByteColumnBase, tile, projection_output_address(row), lane));
        }
    }
    return row_max;
}

void run_softmax_row_pass2(ftlpu::TspSliceSystem& system, std::size_t row, float row_max, std::ostream* icu_log)
{
    const auto read_to_vxm = west_stream_read_latency(kScaledScoreByteColumnBase);
    const auto write_cycle = read_to_vxm + 3 + east_stream_write_latency(kExpScoreByteColumnBase);
    const auto cycles = write_cycle + kBlocks + 4;
    const auto address = projection_output_address(row);

    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        if (cycle == 0) {
            enqueue_fp32_mem_reads(system.icu(), kScaledScoreByteColumnBase, address);
        }
        if (cycle == read_to_vxm) {
            system.icu().enqueue_vxm(0, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamFloat32(kSoftmaxWestStreamBase),
                ftlpu::VxmLaneOperand::Imm(0.0f),
            });
        }
        if (cycle == read_to_vxm + 1) {
            system.icu().enqueue_vxm(1, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Subtract,
                ftlpu::VxmLaneOperand::Alu(0),
                ftlpu::VxmLaneOperand::Imm(row_max),
            });
        }
        if (cycle == read_to_vxm + 2) {
            system.icu().enqueue_vxm(2, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Exp,
                ftlpu::VxmLaneOperand::Alu(1),
                ftlpu::VxmLaneOperand::Imm(0.0f),
            });
        }
        if (cycle == read_to_vxm + 3) {
            system.icu().enqueue_vxm(3, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Cast,
                ftlpu::VxmLaneOperand::Alu(2),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                ftlpu::VxmCastTarget::Float32,
                kSoftmaxEastStreamBase,
            });
        }
        if (cycle == write_cycle) {
            enqueue_fp32_mem_writes(system.icu(), kExpScoreByteColumnBase, address);
        }
        tick_softmax_cycle(system, icu_log);
    }
}

std::vector<float> compute_row_sum_from_mem(const ftlpu::TileArrayModel& mem)
{
    auto row_sum = std::vector<float>(kSeqLen, 0.0f);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kSeqLen; ++column) {
            const auto tile = column / kLanes;
            const auto lane = column % kLanes;
            row_sum[row] += read_float32(mem, kExpScoreByteColumnBase, tile, projection_output_address(row), lane);
        }
    }
    return row_sum;
}

void run_softmax_row_pass3(ftlpu::TspSliceSystem& system, std::size_t row, float row_sum, std::ostream* icu_log)
{
    const auto read_to_vxm = west_stream_read_latency(kExpScoreByteColumnBase);
    const auto write_cycle = read_to_vxm + 3 + east_stream_write_latency(kSoftmaxInt8Column);
    const auto cycles = write_cycle + kBlocks + 4;
    const auto address = projection_output_address(row);

    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        if (cycle == 0) {
            enqueue_fp32_mem_reads(system.icu(), kExpScoreByteColumnBase, address);
        }
        if (cycle == read_to_vxm) {
            system.icu().enqueue_vxm(0, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamFloat32(kSoftmaxWestStreamBase),
                ftlpu::VxmLaneOperand::Imm(0.0f),
            });
        }
        if (cycle == read_to_vxm + 1) {
            system.icu().enqueue_vxm(1, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Divide,
                ftlpu::VxmLaneOperand::Alu(0),
                ftlpu::VxmLaneOperand::Imm(row_sum),
            });
        }
        if (cycle == read_to_vxm + 2) {
            system.icu().enqueue_vxm(2, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Multiply,
                ftlpu::VxmLaneOperand::Alu(1),
                ftlpu::VxmLaneOperand::Imm(kSoftmaxInt8Scale),
            });
        }
        if (cycle == read_to_vxm + 3) {
            system.icu().enqueue_vxm(3, ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Cast,
                ftlpu::VxmLaneOperand::Alu(2),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                ftlpu::VxmCastTarget::Int8,
                kSoftmaxEastStreamBase,
            });
        }
        if (cycle == write_cycle) {
            system.icu().enqueue_mem(
                kSoftmaxInt8Column,
                ftlpu::MemInstruction::Write(address, kSoftmaxEastStreamBase));
        }
        tick_softmax_cycle(system, icu_log);
    }
}

std::vector<std::int8_t> run_softmax_stream_program(ftlpu::TspSliceSystem& system, std::ostream* icu_log)
{
    const auto row_max = compute_row_max_from_mem(system.mem());

    for (std::size_t row = 0; row < kSeqLen; ++row) {
        run_softmax_row_pass2(system, row, row_max[row], icu_log);
    }
    const auto row_sum = compute_row_sum_from_mem(system.mem());

    for (std::size_t row = 0; row < kSeqLen; ++row) {
        run_softmax_row_pass3(system, row, row_sum[row], icu_log);
    }

    auto output = std::vector<std::int8_t>(kSeqLen * kSeqLen, 0);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kSeqLen; ++column) {
            const auto tile = column / kLanes;
            const auto lane = column % kLanes;
            output[score_index(row, column)] = static_cast<std::int8_t>(
                system.mem().sram_lane_byte(kSoftmaxInt8Column, tile, projection_output_address(row), lane));
        }
    }
    return output;
}

std::int8_t softmax_golden_value(const std::vector<std::int32_t>& scores, std::size_t row, std::size_t column)
{
    auto max_value = -std::numeric_limits<float>::infinity();
    std::array<float, kSeqLen> scaled {};
    for (std::size_t c = 0; c < kSeqLen; ++c) {
        scaled[c] = static_cast<float>(scores[score_index(row, c)]) * kScoreScale;
        max_value = std::max(max_value, scaled[c]);
    }

    auto sum = 0.0f;
    for (std::size_t c = 0; c < kSeqLen; ++c) {
        sum += std::exp(scaled[c] - max_value);
    }
    const auto probability = std::exp(scaled[column] - max_value) / sum;
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
    emit_direct_score_softmax_pass1(program);
    program.load(system->icu());

    const auto log_dir = std::filesystem::path("logs") / "attention_projection";
    std::filesystem::create_directories(log_dir);
    auto icu_log = std::ofstream(log_dir / "icu.log");
    if (!icu_log.good()) {
        std::cerr << "failed to open attention projection ICU log\n";
        return 1;
    }
    icu_log << "attention_projection direct_score_softmax\n"
            << "  qk_output_start=" << kQkOutputStart
            << " direct_score_vxm_start=" << kDirectScoreVxmStart
            << " scaled_score_columns=" << kScaledScoreByteColumnBase << ".." << (kScaledScoreByteColumnBase + 3)
            << " exp_columns=" << kExpScoreByteColumnBase << ".." << (kExpScoreByteColumnBase + 3)
            << " softmax_column=" << kSoftmaxInt8Column << '\n';

    std::ostringstream sink;
    auto qk_scores = std::vector<std::int32_t>(kSeqLen * kSeqLen, 0);
    const auto direct_pass1_done = kDirectScoreVxmStart
        + kSeqLen
        + 2
        + east_stream_write_latency(kScaledScoreByteColumnBase)
        + kBlocks
        + 8;
    const auto total_cycles = direct_pass1_done;
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
            const auto q_expected = projection_int8(kWqMatrix, row, column);
            const auto k_expected = projection_int8(kWkMatrix, row, column);
            if (q_actual != q_expected || k_actual != k_expected) {
                std::cerr << "attention projection mismatch row=" << row
                          << " column=" << column
                          << " q_actual=" << static_cast<int>(q_actual)
                          << " q_expected=" << static_cast<int>(q_expected)
                          << " k_actual=" << static_cast<int>(k_actual)
                          << " k_expected=" << static_cast<int>(k_expected)
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

    const auto softmax = run_softmax_stream_program(*system, &icu_log);
    for (const auto row : score_rows) {
        for (const auto column : score_columns) {
            const auto actual = softmax[score_index(row, column)];
            const auto expected = softmax_golden_value(qk_scores, row, column);
            if (actual != expected) {
                std::cerr << "attention softmax mismatch row=" << row
                          << " column=" << column
                          << " actual=" << static_cast<int>(actual)
                          << " expected=" << static_cast<int>(expected)
                          << '\n';
                return 1;
            }
        }
    }

    return 0;
} catch (const std::exception& ex) {
    std::cerr << "attention_projection_test failed: " << ex.what() << '\n';
    return 1;
}
