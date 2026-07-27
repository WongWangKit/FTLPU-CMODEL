#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "vxm_alu_program.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kRows = ftlpu::hw::kMxmRows;
constexpr std::size_t kHidden = ftlpu::hw::kMxmColumns;
constexpr float kEpsilon = 1.0e-5f;

constexpr std::array<std::size_t, 2> kInputSlices {0, 1};
constexpr std::array<std::size_t, 2> kGammaSlices {2, 3};
constexpr std::array<std::size_t, 2> kOutputSlices {24, 25};
constexpr std::size_t kInputAddressBase = 0;
constexpr std::size_t kGammaAddress = 96;
constexpr std::size_t kOutputAddressBase = 320;

static_assert(kRows == 32);
static_assert(kHidden == 32);

std::size_t mem_queue(std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(ftlpu::Hemisphere::East, slice);
}

std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t east_read_to_mxm_latency(std::size_t slice)
{
    return 13 - slice / ftlpu::hw::kMemSlicesPerGroup;
}

float input_value(std::size_t row, std::size_t column)
{
    static_cast<void>(column);
    return 0.25f + static_cast<float>(row % 7) * 0.0625f;
}

float gamma_value(std::size_t column)
{
    static_cast<void>(column);
    return 0.875f;
}

class OfflineSchedule {
public:
    explicit OfflineSchedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(std::size_t queue, std::size_t cycle, ftlpu::MemInstruction instruction)
    {
        pad(mem_[queue], cycle, [&](std::size_t count) { icu_.enqueue_mem_nop(queue, count); });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
    }

    void mem_repeat_at(
        std::size_t queue,
        std::size_t cycle,
        ftlpu::MemInstruction instruction,
        std::size_t count,
        std::int64_t stride)
    {
        mem_at(queue, cycle, instruction);
        if (count > 1) icu_.enqueue_mem_repeat(queue, count - 1, 1, stride);
        advance(mem_[queue], cycle + count);
    }

    void vxm_at(std::size_t alu, std::size_t cycle, ftlpu::VxmLaneAluInstruction instruction)
    {
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu, cycle, std::move(instruction));
        end_cycle_ = std::max(end_cycle_, cycle + 1);
    }

    std::size_t end_cycle() const { return end_cycle_; }

private:
    template <typename Emit>
    static void pad(std::size_t cursor, std::size_t cycle, Emit emit)
    {
        if (cycle < cursor) {
            throw std::logic_error("offline RMSNorm schedule overlaps an ICU queue");
        }
        emit(cycle - cursor);
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_cycle_ = std::max(end_cycle_, next);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_ {};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_ {};
    std::size_t end_cycle_{0};
};

ftlpu::VxmLaneAluInstruction alu_instruction(
    ftlpu::VxmAluOpcode opcode,
    ftlpu::VxmLaneOperand lhs,
    ftlpu::VxmLaneOperand rhs,
    ftlpu::VxmCastTarget cast = ftlpu::VxmCastTarget::Float32,
    std::optional<std::size_t> output = std::nullopt)
{
    return {opcode, lhs, rhs, 1.0f, 0, cast, output,
        ftlpu::Hemisphere::East, ftlpu::Hemisphere::East};
}

void initialize_data(ftlpu::TspSliceSystem& system, const std::vector<float>& input)
{
    // At address c, every physical lane holds x[row, c].  A VXM vector is
    // therefore one hidden-column across all rows: VXM feedback lane r can
    // reduce sum_c x[r,c]^2 in place without routing through MEM or MXM.
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto row = tile * ftlpu::hw::kLanesPerTile + lane;
            for (std::size_t column = 0; column < kHidden; ++column) {
                const auto input_bits = ftlpu::Fp16::from_float(input[row * kHidden + column]).bits();
                const auto gamma_bits = ftlpu::Fp16::from_float(gamma_value(column)).bits();
                system.initialize_mem_sram_lane_byte(
                    ftlpu::Hemisphere::East, kInputSlices[0], tile,
                    kInputAddressBase + column, lane, input_bits & 0xffu);
                system.initialize_mem_sram_lane_byte(
                    ftlpu::Hemisphere::East, kInputSlices[1], tile,
                    kInputAddressBase + column, lane, input_bits >> 8);
                system.initialize_mem_sram_lane_byte(
                    ftlpu::Hemisphere::East, kGammaSlices[0], tile,
                    kGammaAddress + column, lane, gamma_bits & 0xffu);
                system.initialize_mem_sram_lane_byte(
                    ftlpu::Hemisphere::East, kGammaSlices[1], tile,
                    kGammaAddress + column, lane, gamma_bits >> 8);
            }
        }
    }
}

float read_output(const ftlpu::TspSliceSystem& system, std::size_t row, std::size_t column)
{
    const auto tile = row / ftlpu::hw::kLanesPerTile;
    const auto lane = row % ftlpu::hw::kLanesPerTile;
    const auto low = system.read_mem_sram_lane_byte(
        ftlpu::Hemisphere::East, kOutputSlices[0], tile, kOutputAddressBase + column, lane);
    const auto high = system.read_mem_sram_lane_byte(
        ftlpu::Hemisphere::East, kOutputSlices[1], tile, kOutputAddressBase + column, lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low) | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

} // namespace

int main() try
{
    auto input = std::vector<float>(kRows * kHidden);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            input[row * kHidden + column] = ftlpu::Fp16::from_float(
                input_value(row, column)).to_float();
        }
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize_data(system, input);
    auto schedule = OfflineSchedule(system.icu());

    // Pass 1: square one hidden column per cycle, then use ALU1 as the
    // feedback accumulator.  Each physical lane owns one logical row, so all
    // 32 row reductions proceed in parallel without MEM accumulation or MXM.
    constexpr std::size_t kSquareStart = 4;
    for (std::size_t column = 0; column < kHidden; ++column) {
        const auto cycle = kSquareStart + column;
        for (std::size_t byte = 0; byte < kInputSlices.size(); ++byte) {
            const auto slice = kInputSlices[byte];
            schedule.mem_at(
                mem_queue(slice), cycle - west_read_latency(slice),
                ftlpu::MemInstruction::Read(kInputAddressBase + column, ftlpu::StreamId::West(byte)));
        }
        schedule.vxm_at(0, cycle, alu_instruction(
            ftlpu::VxmAluOpcode::Square, ftlpu::VxmLaneOperand::StreamFloat16(32),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
    }
    schedule.vxm_at(1, kSquareStart, alu_instruction(
        ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::Imm(0.0f),
        ftlpu::VxmLaneOperand::Imm(0.0f)));
    for (std::size_t column = 0; column < kHidden; ++column) {
        schedule.vxm_at(1, kSquareStart + column + 1, alu_instruction(
            ftlpu::VxmAluOpcode::Add, ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::Alu(1)));
    }

    // Pass 2: ALU1 now holds sum(x^2).  Build inverse RMS once and leave it
    // resident in ALU5 while x and gamma stream by hidden column.
    constexpr std::size_t kNormalizeStart = kSquareStart + kHidden + 1;
    schedule.vxm_at(2, kNormalizeStart, alu_instruction(
        ftlpu::VxmAluOpcode::Divide, ftlpu::VxmLaneOperand::Alu(1),
        ftlpu::VxmLaneOperand::Imm(static_cast<float>(kHidden))));
    schedule.vxm_at(3, kNormalizeStart + 1, alu_instruction(
        ftlpu::VxmAluOpcode::Add, ftlpu::VxmLaneOperand::Alu(2),
        ftlpu::VxmLaneOperand::Imm(kEpsilon)));
    schedule.vxm_at(4, kNormalizeStart + 2, alu_instruction(
        ftlpu::VxmAluOpcode::Sqrt, ftlpu::VxmLaneOperand::Alu(3),
        ftlpu::VxmLaneOperand::Imm(0.0f)));
    schedule.vxm_at(5, kNormalizeStart + 3, alu_instruction(
        ftlpu::VxmAluOpcode::Divide, ftlpu::VxmLaneOperand::Imm(1.0f),
        ftlpu::VxmLaneOperand::Alu(4)));
    for (std::size_t column = 0; column < kHidden; ++column) {
        const auto cycle = kNormalizeStart + column;
        const auto tail_delay = column + 1 == kHidden ? std::size_t {1} : std::size_t {0};
        const auto data_cycle = cycle + tail_delay;
        for (std::size_t byte = 0; byte < kInputSlices.size(); ++byte) {
            const auto slice = kInputSlices[byte];
            schedule.mem_at(
                mem_queue(slice), cycle - west_read_latency(slice) + 1 + tail_delay,
                ftlpu::MemInstruction::Read(kInputAddressBase + column, ftlpu::StreamId::West(byte)));
        }
        for (std::size_t byte = 0; byte < kGammaSlices.size(); ++byte) {
            const auto slice = kGammaSlices[byte];
            schedule.mem_at(
                mem_queue(slice), cycle - west_read_latency(slice) + 2,
                ftlpu::MemInstruction::Read(kGammaAddress + column, ftlpu::StreamId::West(2 + byte)));
        }
        schedule.vxm_at(6, data_cycle, alu_instruction(
            ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::StreamFloat16(32),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(7, data_cycle + 1, alu_instruction(
            ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::Alu(6),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(8, data_cycle + 2, alu_instruction(
            ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::Alu(7),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(9, data_cycle + 3, alu_instruction(
            ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::Alu(8),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(10, data_cycle, alu_instruction(
            ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::StreamFloat16(34),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(11, data_cycle + 1, alu_instruction(
            ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::Alu(10),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(12, data_cycle + 2, alu_instruction(
            ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::Alu(11),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(13, data_cycle + 3, alu_instruction(
            ftlpu::VxmAluOpcode::Pass, ftlpu::VxmLaneOperand::Alu(12),
            ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(14, data_cycle + 5, alu_instruction(
            ftlpu::VxmAluOpcode::Multiply, ftlpu::VxmLaneOperand::Alu(9),
            ftlpu::VxmLaneOperand::Alu(5)));
        schedule.vxm_at(15, data_cycle + 6, alu_instruction(
            ftlpu::VxmAluOpcode::Multiply, ftlpu::VxmLaneOperand::Alu(14),
            ftlpu::VxmLaneOperand::Alu(13), ftlpu::VxmCastTarget::Float16, 0));
        for (std::size_t byte = 0; byte < kOutputSlices.size(); ++byte) {
            const auto slice = kOutputSlices[byte];
            schedule.mem_at(
                mem_queue(slice), data_cycle + 7 + slice / ftlpu::hw::kMemSlicesPerGroup,
                ftlpu::MemInstruction::Write(kOutputAddressBase + column, ftlpu::StreamId::East(byte)));
        }
    }

    for (std::size_t cycle = 0; cycle < schedule.end_cycle() + 20; ++cycle) {
        system.tick({});
    }

    for (std::size_t row = 0; row < kRows; ++row) {
        auto sum_squares = 0.0f;
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto squared = ftlpu::Fp16::from_float(
                input[row * kHidden + column] * input[row * kHidden + column]).to_float();
            sum_squares += squared;
        }
        const auto inverse_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(kHidden) + kEpsilon);
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto expected = ftlpu::Fp16::from_float(
                input[row * kHidden + column] * inverse_rms * gamma_value(column)).to_float();
            const auto actual = read_output(system, row, column);
            if (std::fabs(actual - expected) > 2.0e-3f) {
                std::cerr << "RMSNorm mismatch at row=" << row << " column=" << column
                          << " actual=" << actual << " expected=" << expected
                          << " expected_sum=" << sum_squares << '\n';
                return 1;
            }
        }
    }

    std::cout << "RMSNorm passed: X[32,32] fp16, gamma[32] fp16, VXM feedback sum(x^2), VXM rsqrt\n";
    return 0;
}
catch (const std::exception& ex)
{
    std::cerr << "RMSNorm test failed: " << ex.what() << '\n';
    return 1;
}
