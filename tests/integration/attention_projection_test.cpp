#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
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
    std::array<std::vector<Event<ftlpu::MxmControlInstruction>>, ftlpu::InstructionControlUnit::kMxmUnitCount> mxm_load_{};
    std::array<std::vector<Event<ftlpu::MxmControlInstruction>>, ftlpu::InstructionControlUnit::kMxmUnitCount> mxm_compute_{};
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
            ftlpu::VxmCastTarget::Float16,
            kQFp16OutputStream,
        });
        program.vxm(kVxmStart + row + 2, 5, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(4),
            ftlpu::VxmLaneOperand::Imm(0.0f),
        });
        program.vxm(kVxmStart + row + 3, 6, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(5),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float16,
            kKFp16OutputStream,
        });

        const auto q_write_cycle = kVxmStart + row + kVxmLatency + east_stream_write_latency(kQHalfLowColumn);
        const auto k_write_cycle = q_write_cycle + 1;
        program.mem(q_write_cycle, kQHalfLowColumn, ftlpu::MemInstruction::Write(half_address(row), kQFp16OutputStream));
        program.mem(q_write_cycle, kQHalfHighColumn, ftlpu::MemInstruction::Write(half_address(row), kQFp16OutputStream + 1));
        program.mem(k_write_cycle, kKHalfLowColumn, ftlpu::MemInstruction::Write(half_address(row), kKFp16OutputStream));
        program.mem(k_write_cycle, kKHalfHighColumn, ftlpu::MemInstruction::Write(half_address(row), kKFp16OutputStream + 1));
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
    program.load(system->icu());

    std::ostringstream sink;
    const auto total_cycles = kVxmStart + kSeqLen + kVxmLatency + east_stream_write_latency(kQHalfLowColumn) + kBlocks + 8;
    for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
        try {
            system->dispatch_icu_only(&sink);
            system->tick_mxm_controls_only(ftlpu::TspSliceSystem::LogSinks {nullptr, nullptr, nullptr, nullptr, nullptr});
            system->mem().tick(sink);
            system->tick_mxm_datapaths_only(ftlpu::TspSliceSystem::LogSinks {nullptr, nullptr, nullptr, nullptr, nullptr});
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
