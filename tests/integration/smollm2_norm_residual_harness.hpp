#pragma once

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "smollm2_layer_phases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::test::smollm2_norm_residual {

constexpr std::size_t kRowsPerChunk = hw::kTileRows * hw::kLanesPerTile;
constexpr float kEpsilon = 1.0e-5f;
constexpr std::array<std::array<std::size_t, 2>, 4> kInputSlices {{
    {{0, 1}}, {{2, 3}}, {{16, 17}}, {{18, 19}},
}};
constexpr std::array<std::array<std::size_t, 2>, 2> kScalarSlices {{
    {{24, 25}}, {{26, 27}},
}};
constexpr std::array<std::size_t, 2> kOutputSlices {28, 29};
constexpr std::size_t kInputAddress = 1024;
constexpr std::size_t kSecondAddress = 2048;
constexpr std::size_t kSquareAddress = 3072;
constexpr std::size_t kInverseAddress = 3073;
constexpr std::size_t kOutputAddress = 4096;

inline std::size_t mem_queue(std::size_t slice)
{
    return InstructionControlUnit::mem_queue(Hemisphere::East, slice);
}

inline std::size_t mem_to_vxm_latency(std::size_t slice)
{
    return slice / hw::kMemSlicesPerGroup + 2;
}

inline std::size_t vxm_to_mem_latency(std::size_t slice)
{
    return slice / hw::kMemSlicesPerGroup + 1;
}

class Schedule {
public:
    explicit Schedule(InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(
        std::size_t slice, std::size_t cycle,
        MemInstruction instruction)
    {
        auto& cursor = mem_[mem_queue(slice)];
        require(cursor, cycle, "MEM");
        icu_.enqueue_mem_nop(mem_queue(slice), cycle - cursor);
        icu_.enqueue_mem(mem_queue(slice), std::move(instruction));
        cursor = cycle + 1;
        end_ = std::max(end_, cursor);
    }

    void mem_repeat_at(
        std::size_t slice, std::size_t cycle,
        MemInstruction instruction, std::size_t count,
        std::int64_t address_stride)
    {
        if (count == 0) return;
        mem_at(slice, cycle, std::move(instruction));
        if (count > 1) {
            icu_.enqueue_mem_repeat(
                mem_queue(slice), count - 1, 1, address_stride);
        }
        mem_[mem_queue(slice)] = cycle + count;
        end_ = std::max(end_, cycle + count);
    }

    void vxm_at(
        std::size_t stage, std::size_t cycle,
        VxmChainDepth depth, VxmLaneAluInstruction instruction)
    {
        auto& cursor = vxm_[stage];
        require(cursor, cycle, "VXM");
        icu_.enqueue_vxm_nop(stage, cycle - cursor);
        icu_.enqueue_vxm(stage, depth, instruction);
        cursor = cycle + 1;
        end_ = std::max(end_, cursor);
    }

    std::size_t end_cycle() const noexcept { return end_; }

private:
    static void require(
        std::size_t cursor, std::size_t cycle, const char* resource)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                std::string("SmolLM2 norm/residual overlaps ") + resource);
        }
    }

    InstructionControlUnit& icu_;
    std::array<std::size_t, InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, InstructionControlUnit::kVxmQueues> vxm_{};
    std::size_t end_{0};
};

inline VxmLaneAluInstruction basic(
    VxmAluOpcode opcode, VxmLaneOperand lhs,
    VxmLaneOperand rhs = VxmLaneOperand::Imm(0.0f),
    std::size_t repeat = 1)
{
    auto instruction = VxmLaneAluInstruction {opcode, lhs, rhs};
    instruction.precision = VxmAluPrecision::Float32;
    instruction.repeat_count = repeat;
    return instruction;
}

inline VxmLaneAluInstruction accumulator(
    bool reset, bool emit, std::size_t repeat)
{
    auto instruction = basic(
        VxmAluOpcode::Add,
        VxmLaneOperand::Previous(), VxmLaneOperand::Acc(), repeat);
    instruction.accumulator_reset = reset;
    instruction.accumulator_write = true;
    instruction.accumulator_emit = emit;
    return instruction;
}

template <typename Fn>
inline std::vector<VxmLutEntry> make_table(
    float input_min, float width, std::size_t count, Fn fn)
{
    auto entries = std::vector<VxmLutEntry> {};
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto x0 = input_min + static_cast<float>(index) * width;
        const auto y0 = fn(x0);
        entries.push_back(VxmLutEntry::from_float(
            (fn(x0 + width) - y0) / width, y0));
    }
    return entries;
}

inline VxmSpecialAlu configure_rsqrt(TspSliceSystem& system)
{
    constexpr std::size_t kEntries = 256;
    constexpr float kWidth = 3.0f / static_cast<float>(kEntries);
    const auto entries = make_table(
        1.0f, kWidth, kEntries,
        [](float value) { return 1.0f / std::sqrt(value); });
    system.initialize_vxm_lut(
        VxmSpecialAluOpcode::Rsqrt, {1.0f, kWidth}, entries);
    auto reference = VxmSpecialAlu {};
    reference.configure_lut(
        VxmSpecialAluOpcode::Rsqrt, {1.0f, kWidth}, entries);
    return reference;
}

inline void initialize_pair(
    TspSliceSystem& system,
    const std::array<std::size_t, 2>& slices,
    std::size_t address, std::size_t column,
    const std::vector<float>& values, std::size_t row_base)
{
    for (std::size_t physical = 0; physical < kRowsPerChunk; ++physical) {
        const auto row = row_base + physical;
        const auto value = row * smollm2_layer::kHidden + column
                < values.size()
            ? values[row * smollm2_layer::kHidden + column] : 0.0f;
        const auto bits = Bf16::from_float(value).bits();
        const auto tile = physical / hw::kLanesPerTile;
        const auto lane = physical % hw::kLanesPerTile;
        system.initialize_mem_sram_lane_byte(
            Hemisphere::East, slices[0], tile, address, lane,
            static_cast<std::uint8_t>(bits & 0xffu));
        system.initialize_mem_sram_lane_byte(
            Hemisphere::East, slices[1], tile, address, lane,
            static_cast<std::uint8_t>(bits >> 8));
    }
}

inline void read_pair_repeat(
    Schedule& schedule,
    const std::array<std::size_t, 2>& slices,
    std::size_t address, std::size_t stream,
    std::size_t input_cycle, std::size_t count)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto slice = slices[byte];
        schedule.mem_repeat_at(
            slice, input_cycle - mem_to_vxm_latency(slice),
            MemInstruction::Read(
                address, StreamId::West(stream + byte)),
            count, 1);
    }
}

inline void write_pair(
    Schedule& schedule,
    const std::array<std::size_t, 2>& slices,
    std::size_t address, std::size_t stream,
    std::size_t output_cycle)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto slice = slices[byte];
        schedule.mem_at(
            slice, output_cycle + vxm_to_mem_latency(slice),
            MemInstruction::Write(
                address, StreamId::East(stream + byte)));
    }
}

inline float read_bf16(
    const TspSliceSystem& system, std::size_t physical,
    std::size_t column)
{
    const auto tile = physical / hw::kLanesPerTile;
    const auto lane = physical % hw::kLanesPerTile;
    const auto low = system.read_mem_sram_lane_byte(
        Hemisphere::East, kOutputSlices[0], tile,
        kOutputAddress + column, lane);
    const auto high = system.read_mem_sram_lane_byte(
        Hemisphere::East, kOutputSlices[1], tile,
        kOutputAddress + column, lane);
    return Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

inline float read_bf16_pair(
    const TspSliceSystem& system,
    const std::array<std::size_t, 2>& slices,
    std::size_t address, std::size_t physical)
{
    const auto tile = physical / hw::kLanesPerTile;
    const auto lane = physical % hw::kLanesPerTile;
    const auto low = system.read_mem_sram_lane_byte(
        Hemisphere::East, slices[0], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        Hemisphere::East, slices[1], tile, address, lane);
    return Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

inline void write_phase_trace(
    const std::filesystem::path& path,
    std::size_t cycles, const std::string& phase)
{
    if (path.empty()) return;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write phase trace");
    output << "start,end,resource,detail\n0," << cycles
           << ",\"Protocol phase\",\"" << phase << "\"\n";
}

inline smollm2_layer::PhaseResult run_rmsnorm(
    TspSliceSystem& system, const std::vector<float>& input,
    const std::filesystem::path& trace_path,
    const std::string& phase)
{
    if (input.empty()
        || input.size() % smollm2_layer::kHidden != 0) {
        throw std::invalid_argument("RMSNorm input must be [rows,576]");
    }
    const auto rows = input.size() / smollm2_layer::kHidden;
    auto output = std::vector<float>(input.size());
    auto total_cycles = std::size_t {0};
    for (std::size_t row_base = 0;
         row_base < rows; row_base += kRowsPerChunk) {
        system.reset_execution_state();
        const auto reference = configure_rsqrt(system);
        system.configure_vxm_input_group_source(0, Hemisphere::East);
        system.configure_vxm_input_group_source(1, Hemisphere::East);
        system.configure_vxm_input_group_source(8, Hemisphere::East);
        system.configure_vxm_input_group_source(9, Hemisphere::East);
        system.configure_vxm_output_block_destination(0, Hemisphere::East);
        system.configure_vxm_output_block_destination(1, Hemisphere::East);
        system.configure_vxm_output_block_destination(4, Hemisphere::East);
        system.configure_vxm_output_block_destination(5, Hemisphere::East);
        for (std::size_t column = 0;
             column < smollm2_layer::kHidden; ++column) {
            for (const auto& slices : kInputSlices) {
                initialize_pair(
                    system, slices, kInputAddress + column,
                    column, input, row_base);
            }
        }

        auto schedule = Schedule {system.icu()};
        constexpr std::size_t kSquareConfig = 40;
        constexpr std::size_t kSquareInput = kSquareConfig + 1;
        schedule.vxm_at(
            0, kSquareConfig, VxmChainDepth::Two,
            basic(
                VxmAluOpcode::Multiply,
                VxmLaneOperand::StreamBFloat16(),
                VxmLaneOperand::StreamBFloat16(),
                smollm2_layer::kHidden));
        schedule.vxm_at(
            1, kSquareConfig, VxmChainDepth::Two,
            accumulator(true, false, 1));
        schedule.vxm_at(
            1, kSquareConfig + 1, VxmChainDepth::Two,
            accumulator(
                false, false, smollm2_layer::kHidden - 2));
        auto final_sum = accumulator(false, true, 1);
        final_sum.output_type = VxmCastTarget::BFloat16;
        final_sum.output_stream =
            VxmLane::fixed_output_stream_for_block(0);
        schedule.vxm_at(
            1, kSquareConfig + 2,
            VxmChainDepth::Two, final_sum);
        read_pair_repeat(
            schedule, kInputSlices[0], kInputAddress,
            0, kSquareInput, smollm2_layer::kHidden);
        read_pair_repeat(
            schedule, kInputSlices[1], kInputAddress,
            2, kSquareInput, smollm2_layer::kHidden);
        read_pair_repeat(
            schedule, kInputSlices[2], kInputAddress,
            16, kSquareInput, smollm2_layer::kHidden);
        read_pair_repeat(
            schedule, kInputSlices[3], kInputAddress,
            18, kSquareInput, smollm2_layer::kHidden);
        constexpr std::size_t kSquareOutput =
            kSquareInput + smollm2_layer::kHidden + 1;
        write_pair(
            schedule, kScalarSlices[0], kSquareAddress,
            0, kSquareOutput);
        write_pair(
            schedule, kScalarSlices[1], kSquareAddress,
            8, kSquareOutput);

        constexpr std::size_t kRsqrtConfig = kSquareOutput + 40;
        constexpr std::size_t kRsqrtInput = kRsqrtConfig + 1;
        schedule.vxm_at(
            0, kRsqrtConfig, VxmChainDepth::Four,
            basic(
                VxmAluOpcode::Multiply,
                VxmLaneOperand::StreamBFloat16(),
                VxmLaneOperand::Imm(
                    1.0f / static_cast<float>(
                        smollm2_layer::kHidden))));
        schedule.vxm_at(
            1, kRsqrtConfig, VxmChainDepth::Four,
            basic(
                VxmAluOpcode::Add, VxmLaneOperand::Previous(),
                VxmLaneOperand::Imm(kEpsilon)));
        schedule.vxm_at(
            2, kRsqrtConfig, VxmChainDepth::Four,
            basic(VxmAluOpcode::Bypass, VxmLaneOperand::Previous()));
        auto rsqrt = VxmLaneAluInstruction {
            VxmSpecialAluOpcode::Rsqrt, VxmLaneOperand::Previous()};
        rsqrt.output_type = VxmCastTarget::BFloat16;
        rsqrt.output_stream =
            VxmLane::fixed_output_stream_for_block(1);
        schedule.vxm_at(
            3, kRsqrtConfig, VxmChainDepth::Four, rsqrt);
        read_pair_repeat(
            schedule, kScalarSlices[0], kSquareAddress,
            0, kRsqrtInput, 1);
        read_pair_repeat(
            schedule, kScalarSlices[1], kSquareAddress,
            16, kRsqrtInput, 1);
        constexpr std::size_t kRsqrtOutput = kRsqrtInput + 8;
        write_pair(
            schedule, kScalarSlices[0], kInverseAddress,
            2, kRsqrtOutput);
        write_pair(
            schedule, kScalarSlices[1], kInverseAddress,
            10, kRsqrtOutput);

        constexpr std::size_t kScalarConfig = kRsqrtOutput + 40;
        constexpr std::size_t kScalarInput = kScalarConfig + 1;
        schedule.vxm_at(
            0, kScalarConfig, VxmChainDepth::Two,
            basic(VxmAluOpcode::Bypass,
                  VxmLaneOperand::StreamBFloat16()));
        auto capture = basic(
            VxmAluOpcode::Bypass, VxmLaneOperand::Previous());
        capture.local_scalar_write = true;
        schedule.vxm_at(
            1, kScalarConfig, VxmChainDepth::Two, capture);
        read_pair_repeat(
            schedule, kScalarSlices[0], kInverseAddress,
            0, kScalarInput, 1);
        read_pair_repeat(
            schedule, kScalarSlices[1], kInverseAddress,
            16, kScalarInput, 1);

        constexpr std::size_t kNormalizeConfig = kScalarInput + 20;
        constexpr std::size_t kNormalizeInput = kNormalizeConfig + 1;
        schedule.vxm_at(
            0, kNormalizeConfig, VxmChainDepth::Two,
            basic(
                VxmAluOpcode::Bypass,
                VxmLaneOperand::StreamBFloat16(),
                VxmLaneOperand::Imm(0.0f),
                smollm2_layer::kHidden));
        auto normalize = basic(
            VxmAluOpcode::Multiply,
            VxmLaneOperand::Previous(), VxmLaneOperand::Acc(),
            smollm2_layer::kHidden);
        normalize.output_type = VxmCastTarget::BFloat16;
        normalize.output_stream =
            VxmLane::fixed_output_stream_for_block(0);
        schedule.vxm_at(
            1, kNormalizeConfig, VxmChainDepth::Two, normalize);
        read_pair_repeat(
            schedule, kInputSlices[0], kInputAddress,
            0, kNormalizeInput, smollm2_layer::kHidden);
        read_pair_repeat(
            schedule, kInputSlices[2], kInputAddress,
            16, kNormalizeInput, smollm2_layer::kHidden);
        constexpr std::size_t kNormalizeOutput = kNormalizeInput + 2;
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kOutputSlices[byte];
            schedule.mem_repeat_at(
                slice, kNormalizeOutput + vxm_to_mem_latency(slice),
                MemInstruction::Write(
                    kOutputAddress, StreamId::East(byte)),
                smollm2_layer::kHidden, 1);
        }

        const auto cycles = schedule.end_cycle() + 16;
        for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
            system.tick({});
        }
        for (std::size_t physical = 0;
             physical < kRowsPerChunk && row_base + physical < rows;
             ++physical) {
            auto sum = 0.0f;
            for (std::size_t column = 0;
                 column < smollm2_layer::kHidden; ++column) {
                const auto value = Bf16::from_float(input[
                    (row_base + physical) * smollm2_layer::kHidden
                    + column]).to_float();
                sum += value * value;
            }
            // The reduction is materialized in MEM as BF16 before the
            // scalar Rsqrt phase.  The golden must include that architectural
            // boundary; using the unrounded FP32 sum only happened to agree
            // for the original single-row directed pattern.
            const auto stored_sum = Bf16::from_float(sum).to_float();
            const auto inverse = Bf16::from_float(reference.execute(
                VxmSpecialAluOpcode::Rsqrt,
                stored_sum / static_cast<float>(smollm2_layer::kHidden)
                    + kEpsilon)).to_float();
            for (std::size_t column = 0;
                 column < smollm2_layer::kHidden; ++column) {
                const auto actual = read_bf16(system, physical, column);
                const auto value = Bf16::from_float(input[
                    (row_base + physical) * smollm2_layer::kHidden
                    + column]).to_float();
                const auto expected =
                    Bf16::from_float(value * inverse).to_float();
                if (actual != expected) {
                    throw std::runtime_error(
                        phase + " RMSNorm mismatch row="
                        + std::to_string(row_base + physical)
                        + " column=" + std::to_string(column)
                        + " actual=" + std::to_string(actual)
                        + " expected=" + std::to_string(expected)
                        + " inverse=" + std::to_string(inverse)
                        + " hw_sum=" + std::to_string(read_bf16_pair(
                            system, kScalarSlices[0],
                            kSquareAddress, physical))
                        + " hw_inverse=" + std::to_string(read_bf16_pair(
                            system, kScalarSlices[0],
                            kInverseAddress, physical)));
                }
                output[(row_base + physical) * smollm2_layer::kHidden
                       + column] = actual;
            }
        }
        total_cycles += cycles;
    }
    write_phase_trace(trace_path, total_cycles, phase);
    return {std::move(output), {}, {}, total_cycles};
}

inline smollm2_layer::PhaseResult run_residual(
    TspSliceSystem& system,
    const std::vector<float>& lhs, const std::vector<float>& rhs,
    const std::filesystem::path& trace_path,
    const std::string& phase)
{
    if (lhs.size() != rhs.size() || lhs.empty()
        || lhs.size() % smollm2_layer::kHidden != 0) {
        throw std::invalid_argument("residual operands must be [rows,576]");
    }
    const auto rows = lhs.size() / smollm2_layer::kHidden;
    auto output = std::vector<float>(lhs.size());
    auto total_cycles = std::size_t {0};
    for (std::size_t row_base = 0;
         row_base < rows; row_base += kRowsPerChunk) {
        system.reset_execution_state();
        system.configure_vxm_input_group_source(0, Hemisphere::East);
        system.configure_vxm_input_group_source(1, Hemisphere::East);
        system.configure_vxm_input_group_source(8, Hemisphere::East);
        system.configure_vxm_input_group_source(9, Hemisphere::East);
        system.configure_vxm_output_block_destination(0, Hemisphere::East);
        system.configure_vxm_output_block_destination(4, Hemisphere::East);
        for (std::size_t column = 0;
             column < smollm2_layer::kHidden; ++column) {
            initialize_pair(
                system, kInputSlices[0], kInputAddress + column,
                column, lhs, row_base);
            initialize_pair(
                system, kInputSlices[1], kSecondAddress + column,
                column, rhs, row_base);
            initialize_pair(
                system, kInputSlices[2], kInputAddress + column,
                column, lhs, row_base);
            initialize_pair(
                system, kInputSlices[3], kSecondAddress + column,
                column, rhs, row_base);
        }
        auto schedule = Schedule {system.icu()};
        constexpr std::size_t kConfig = 40;
        constexpr std::size_t kInput = kConfig + 1;
        schedule.vxm_at(
            0, kConfig, VxmChainDepth::Two,
            basic(
                VxmAluOpcode::Add,
                VxmLaneOperand::StreamBFloat16(),
                VxmLaneOperand::StreamBFloat16(),
                smollm2_layer::kHidden));
        auto tail = basic(
            VxmAluOpcode::Bypass, VxmLaneOperand::Previous(),
            VxmLaneOperand::Imm(0.0f), smollm2_layer::kHidden);
        tail.output_type = VxmCastTarget::BFloat16;
        tail.output_stream = VxmLane::fixed_output_stream_for_block(0);
        schedule.vxm_at(1, kConfig, VxmChainDepth::Two, tail);
        read_pair_repeat(
            schedule, kInputSlices[0], kInputAddress,
            0, kInput, smollm2_layer::kHidden);
        read_pair_repeat(
            schedule, kInputSlices[1], kSecondAddress,
            2, kInput, smollm2_layer::kHidden);
        read_pair_repeat(
            schedule, kInputSlices[2], kInputAddress,
            16, kInput, smollm2_layer::kHidden);
        read_pair_repeat(
            schedule, kInputSlices[3], kSecondAddress,
            18, kInput, smollm2_layer::kHidden);
        constexpr std::size_t kOutput = kInput + 1;
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kOutputSlices[byte];
            schedule.mem_repeat_at(
                slice, kOutput + vxm_to_mem_latency(slice),
                MemInstruction::Write(
                    kOutputAddress, StreamId::East(byte)),
                smollm2_layer::kHidden, 1);
        }
        const auto cycles = schedule.end_cycle() + 16;
        for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
            system.tick({});
        }
        for (std::size_t physical = 0;
             physical < kRowsPerChunk && row_base + physical < rows;
             ++physical) {
            for (std::size_t column = 0;
                 column < smollm2_layer::kHidden; ++column) {
                const auto index =
                    (row_base + physical) * smollm2_layer::kHidden
                    + column;
                const auto actual = read_bf16(system, physical, column);
                const auto expected = Bf16::from_float(
                    Bf16::from_float(lhs[index]).to_float()
                    + Bf16::from_float(rhs[index]).to_float()).to_float();
                if (actual != expected) {
                    throw std::runtime_error(
                        phase + " residual mismatch index="
                        + std::to_string(index)
                        + " actual=" + std::to_string(actual)
                        + " expected=" + std::to_string(expected));
                }
                output[index] = actual;
            }
        }
        total_cycles += cycles;
    }
    write_phase_trace(trace_path, total_cycles, phase);
    return {std::move(output), {}, {}, total_cycles};
}

} // namespace ftlpu::test::smollm2_norm_residual
