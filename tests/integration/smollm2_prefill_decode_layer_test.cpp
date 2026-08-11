#include "ftlpu/system/tsp_slice_system.hpp"
#include "smollm2_layer_phases.hpp"
#include "vxm_alu_program.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ftlpu::test::smollm2_layer::PhaseResult;
constexpr std::size_t kHidden = ftlpu::test::smollm2_layer::kHidden;
constexpr std::size_t kPrefillLength =
    ftlpu::test::smollm2_layer::kPrefillLength;
constexpr std::size_t kVectorRows = ftlpu::hw::kMxmRows;
constexpr float kRmsEpsilon = 1.0e-5f;
constexpr std::array<std::size_t, 2> kInputSlices {0, 1};
constexpr std::array<std::size_t, 2> kSecondSlices {2, 3};
constexpr std::array<std::size_t, 2> kGammaSlices {4, 5};
constexpr std::array<std::size_t, 2> kOutputSlices {8, 9};
constexpr std::size_t kInputAddressBase = 0;
constexpr std::size_t kSecondAddressBase = 1024;
constexpr std::size_t kGammaAddressBase = 2048;
constexpr std::size_t kOutputAddressBase = 3072;

struct TraceEvent {
    std::size_t start;
    std::size_t end;
    std::string resource;
    std::string detail;
};

std::size_t mem_queue(std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(
        ftlpu::Hemisphere::East, slice);
}

std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

class VxmSchedule {
public:
    explicit VxmSchedule(ftlpu::InstructionControlUnit& icu)
        : icu_(icu)
    {
    }

    void mem_at(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction,
        std::string detail)
    {
        auto& cursor = mem_[mem_queue(slice)];
        pad(cursor, cycle, [&](std::size_t count) {
            icu_.enqueue_mem_nop(mem_queue(slice), count);
        });
        icu_.enqueue_mem(mem_queue(slice), instruction);
        cursor = cycle + 1;
        end_ = std::max(end_, cursor);
        trace_.push_back({
            cycle,
            cycle + 1,
            instruction.opcode == ftlpu::MemOpcode::Read
                ? "MEM.E.Read"
                : "MEM.E.Write",
            std::move(detail)});
    }

    void alu_at(
        std::size_t alu,
        std::size_t cycle,
        ftlpu::VxmLaneAluInstruction instruction,
        std::string detail)
    {
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu, cycle, instruction);
        end_ = std::max(end_, cycle + 1);
        trace_.push_back({
            cycle,
            cycle + 1,
            "VXM.ALU" + std::to_string(alu),
            std::move(detail)});
    }

    std::size_t end_cycle() const noexcept { return end_; }
    const std::vector<TraceEvent>& trace() const noexcept { return trace_; }

private:
    template <typename Emit>
    static void pad(std::size_t cursor, std::size_t cycle, Emit emit)
    {
        if (cycle < cursor) {
            throw std::logic_error("layer VXM schedule overlaps an ICU queue");
        }
        emit(cycle - cursor);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_ {};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_ {};
    std::vector<TraceEvent> trace_ {};
    std::size_t end_{0};
};

ftlpu::VxmLaneAluInstruction alu_instruction(
    ftlpu::VxmAluOpcode opcode,
    ftlpu::VxmLaneOperand lhs,
    ftlpu::VxmLaneOperand rhs,
    ftlpu::VxmCastTarget cast = ftlpu::VxmCastTarget::Float32,
    std::optional<std::size_t> output = std::nullopt)
{
    return {
        opcode,
        lhs,
        rhs,
        1.0f,
        0,
        cast,
        output,
        ftlpu::Hemisphere::East,
        ftlpu::Hemisphere::East};
}

void append_trace(
    std::vector<TraceEvent>& destination,
    const std::vector<TraceEvent>& source,
    std::size_t offset,
    const std::string& phase)
{
    for (const auto& event : source) {
        destination.push_back({
            offset + event.start,
            offset + event.end,
            event.resource,
            phase + ": " + event.detail});
    }
}

void write_trace(
    const std::filesystem::path& path,
    std::vector<TraceEvent> events)
{
    std::stable_sort(
        events.begin(),
        events.end(),
        [](const TraceEvent& lhs, const TraceEvent& rhs) {
            if (lhs.start != rhs.start) return lhs.start < rhs.start;
            if (lhs.end != rhs.end) return lhs.end < rhs.end;
            return lhs.resource < rhs.resource;
        });
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write layer-op trace");
    }
    const auto quote = [](const std::string& value) {
        auto result = std::string {"\""};
        for (const auto ch : value) {
            result.push_back(ch);
            if (ch == '"') result.push_back('"');
        }
        result.push_back('"');
        return result;
    };
    output << "start,end,resource,detail\n";
    for (const auto& event : events) {
        output << event.start << ',' << event.end << ','
               << quote(event.resource) << ',' << quote(event.detail) << '\n';
    }
}

void initialize_bf16_vector(
    ftlpu::TspSliceSystem& system,
    const std::array<std::size_t, 2>& slices,
    std::size_t address,
    std::size_t column,
    const std::vector<float>& values,
    std::size_t row_base)
{
    for (std::size_t physical = 0; physical < kVectorRows; ++physical) {
        const auto tile = physical / ftlpu::hw::kLanesPerTile;
        const auto lane = physical % ftlpu::hw::kLanesPerTile;
        const auto row = row_base + physical;
        const auto bits = ftlpu::Bf16::from_float(
            row < values.size() / kHidden ? values[row * kHidden + column] : 0.0f)
                              .bits();
        system.initialize_mem_sram_lane_byte(
            ftlpu::Hemisphere::East,
            slices[0],
            tile,
            address,
            lane,
            static_cast<std::uint8_t>(bits & 0xffu));
        system.initialize_mem_sram_lane_byte(
            ftlpu::Hemisphere::East,
            slices[1],
            tile,
            address,
            lane,
            static_cast<std::uint8_t>(bits >> 8));
    }
}

float read_bf16(
    const ftlpu::TspSliceSystem& system,
    std::size_t row,
    std::size_t column)
{
    const auto tile = row / ftlpu::hw::kLanesPerTile;
    const auto lane = row % ftlpu::hw::kLanesPerTile;
    const auto low = system.read_mem_sram_lane_byte(
        ftlpu::Hemisphere::East,
        kOutputSlices[0],
        tile,
        kOutputAddressBase + column,
        lane);
    const auto high = system.read_mem_sram_lane_byte(
        ftlpu::Hemisphere::East,
        kOutputSlices[1],
        tile,
        kOutputAddressBase + column,
        lane);
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8))
        .to_float();
}

PhaseResult run_rmsnorm(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path,
    const std::string& phase)
{
    if (input.empty() || input.size() % kHidden != 0) {
        throw std::invalid_argument("RMSNorm input must be [rows,576]");
    }
    const auto rows = input.size() / kHidden;
    auto output = std::vector<float>(input.size());
    auto trace = std::vector<TraceEvent> {};
    auto total_cycles = std::size_t {0};

    for (std::size_t row_base = 0; row_base < rows; row_base += kVectorRows) {
        system.reset_execution_state();
        for (std::size_t column = 0; column < kHidden; ++column) {
            initialize_bf16_vector(
                system, kInputSlices, kInputAddressBase + column, column, input, row_base);
            for (std::size_t physical = 0; physical < kVectorRows; ++physical) {
                const auto tile = physical / ftlpu::hw::kLanesPerTile;
                const auto lane = physical % ftlpu::hw::kLanesPerTile;
                const auto bits = ftlpu::Bf16::from_float(1.0f).bits();
                system.initialize_mem_sram_lane_byte(
                    ftlpu::Hemisphere::East,
                    kGammaSlices[0],
                    tile,
                    kGammaAddressBase + column,
                    lane,
                    static_cast<std::uint8_t>(bits & 0xffu));
                system.initialize_mem_sram_lane_byte(
                    ftlpu::Hemisphere::East,
                    kGammaSlices[1],
                    tile,
                    kGammaAddressBase + column,
                    lane,
                    static_cast<std::uint8_t>(bits >> 8));
            }
        }

        auto schedule = VxmSchedule(system.icu());
        constexpr auto kSquareStart = std::size_t {8};
        schedule.alu_at(
            1,
            kSquareStart,
            alu_instruction(
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::Imm(0.0f),
                ftlpu::VxmLaneOperand::Imm(0.0f)),
            "sum(x^2) init");
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto cycle = kSquareStart + column;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kInputSlices[byte];
                schedule.mem_at(
                    slice,
                    cycle - west_read_latency(slice),
                    ftlpu::MemInstruction::Read(
                        kInputAddressBase + column,
                        ftlpu::StreamId::West(byte)),
                    "RMSNorm x column=" + std::to_string(column));
            }
            schedule.alu_at(
                0,
                cycle,
                alu_instruction(
                    ftlpu::VxmAluOpcode::Square,
                    ftlpu::VxmLaneOperand::StreamBFloat16(32),
                    ftlpu::VxmLaneOperand::Imm(0.0f)),
                "square column=" + std::to_string(column));
            schedule.alu_at(
                1,
                cycle + 1,
                alu_instruction(
                    ftlpu::VxmAluOpcode::Add,
                    ftlpu::VxmLaneOperand::Alu(0),
                    ftlpu::VxmLaneOperand::Alu(1)),
                "sum(x^2) column=" + std::to_string(column));
        }

        constexpr auto kNormalizeGap = std::size_t {1};
        const auto normalize = kSquareStart + kHidden + kNormalizeGap;
        schedule.alu_at(
            2,
            normalize,
            alu_instruction(
                ftlpu::VxmAluOpcode::Divide,
                ftlpu::VxmLaneOperand::Alu(1),
                ftlpu::VxmLaneOperand::Imm(static_cast<float>(kHidden))),
            "mean(x^2)");
        schedule.alu_at(
            3,
            normalize + 1,
            alu_instruction(
                ftlpu::VxmAluOpcode::Add,
                ftlpu::VxmLaneOperand::Alu(2),
                ftlpu::VxmLaneOperand::Imm(kRmsEpsilon)),
            "mean(x^2)+epsilon");
        schedule.alu_at(
            4,
            normalize + 2,
            alu_instruction(
                ftlpu::VxmAluOpcode::Sqrt,
                ftlpu::VxmLaneOperand::Alu(3),
                ftlpu::VxmLaneOperand::Imm(0.0f)),
            "sqrt");
        schedule.alu_at(
            5,
            normalize + 3,
            alu_instruction(
                ftlpu::VxmAluOpcode::Divide,
                ftlpu::VxmLaneOperand::Imm(1.0f),
                ftlpu::VxmLaneOperand::Alu(4)),
            "inverse RMS");

        const auto scale_start = normalize + 4;
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto x_cycle = scale_start + column;
            const auto gamma_cycle = x_cycle + 1;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto x_slice = kInputSlices[byte];
                schedule.mem_at(
                    x_slice,
                    x_cycle - west_read_latency(x_slice),
                    ftlpu::MemInstruction::Read(
                        kInputAddressBase + column,
                        ftlpu::StreamId::West(byte)),
                    "RMSNorm x scale column=" + std::to_string(column));
                const auto gamma_slice = kGammaSlices[byte];
                schedule.mem_at(
                    gamma_slice,
                    gamma_cycle - west_read_latency(gamma_slice),
                    ftlpu::MemInstruction::Read(
                        kGammaAddressBase + column,
                        ftlpu::StreamId::West(2 + byte)),
                    "RMSNorm gamma column=" + std::to_string(column));
            }
            schedule.alu_at(
                6,
                x_cycle,
                alu_instruction(
                    ftlpu::VxmAluOpcode::Multiply,
                    ftlpu::VxmLaneOperand::StreamBFloat16(32),
                    ftlpu::VxmLaneOperand::Alu(5)),
                "x * inverse_rms column=" + std::to_string(column));
            schedule.alu_at(
                7,
                gamma_cycle,
                alu_instruction(
                    ftlpu::VxmAluOpcode::Multiply,
                    ftlpu::VxmLaneOperand::Alu(6),
                    ftlpu::VxmLaneOperand::StreamBFloat16(34),
                    ftlpu::VxmCastTarget::BFloat16,
                    0),
                "RMSNorm BF16 output column=" + std::to_string(column));
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kOutputSlices[byte];
                schedule.mem_at(
                    slice,
                    gamma_cycle + 1
                        + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        kOutputAddressBase + column,
                        ftlpu::StreamId::East(byte)),
                    "RMSNorm write column=" + std::to_string(column));
            }
        }

        const auto chunk_cycles = schedule.end_cycle() + 12;
        for (std::size_t cycle = 0; cycle < chunk_cycles; ++cycle) {
            system.tick({});
        }
        append_trace(trace, schedule.trace(), total_cycles, phase);
        for (std::size_t physical = 0;
             physical < kVectorRows && row_base + physical < rows;
             ++physical) {
            for (std::size_t column = 0; column < kHidden; ++column) {
                output[(row_base + physical) * kHidden + column] =
                    read_bf16(system, physical, column);
            }
        }
        total_cycles += chunk_cycles;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        auto sum_squares = 0.0f;
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto value = ftlpu::Bf16::from_float(
                input[row * kHidden + column]).to_float();
            sum_squares += value * value;
        }
        const auto inverse = 1.0f / std::sqrt(
            sum_squares / static_cast<float>(kHidden) + kRmsEpsilon);
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto value = ftlpu::Bf16::from_float(
                input[row * kHidden + column]).to_float();
            const auto expected =
                ftlpu::Bf16::from_float(value * inverse).to_float();
            if (output[row * kHidden + column] != expected) {
                throw std::runtime_error(
                    phase + " RMSNorm mismatch row=" + std::to_string(row)
                    + " column=" + std::to_string(column));
            }
        }
    }
    if (!trace_path.empty()) write_trace(trace_path, std::move(trace));
    return {std::move(output), {}, {}, total_cycles};
}

PhaseResult run_residual(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const std::filesystem::path& trace_path,
    const std::string& phase)
{
    if (lhs.size() != rhs.size() || lhs.empty() || lhs.size() % kHidden != 0) {
        throw std::invalid_argument("residual operands must both be [rows,576]");
    }
    const auto rows = lhs.size() / kHidden;
    auto output = std::vector<float>(lhs.size());
    auto trace = std::vector<TraceEvent> {};
    auto total_cycles = std::size_t {0};

    for (std::size_t row_base = 0; row_base < rows; row_base += kVectorRows) {
        system.reset_execution_state();
        for (std::size_t column = 0; column < kHidden; ++column) {
            initialize_bf16_vector(
                system, kInputSlices, kInputAddressBase + column, column, lhs, row_base);
            initialize_bf16_vector(
                system, kSecondSlices, kSecondAddressBase + column, column, rhs, row_base);
        }
        auto schedule = VxmSchedule(system.icu());
        constexpr auto kStart = std::size_t {8};
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto cycle = kStart + column;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto lhs_slice = kInputSlices[byte];
                schedule.mem_at(
                    lhs_slice,
                    cycle - west_read_latency(lhs_slice),
                    ftlpu::MemInstruction::Read(
                        kInputAddressBase + column,
                        ftlpu::StreamId::West(byte)),
                    "residual lhs column=" + std::to_string(column));
                const auto rhs_slice = kSecondSlices[byte];
                schedule.mem_at(
                    rhs_slice,
                    cycle - west_read_latency(rhs_slice),
                    ftlpu::MemInstruction::Read(
                        kSecondAddressBase + column,
                        ftlpu::StreamId::West(2 + byte)),
                    "residual rhs column=" + std::to_string(column));
            }
            schedule.alu_at(
                0,
                cycle,
                alu_instruction(
                    ftlpu::VxmAluOpcode::Add,
                    ftlpu::VxmLaneOperand::StreamBFloat16(32),
                    ftlpu::VxmLaneOperand::StreamBFloat16(34),
                    ftlpu::VxmCastTarget::BFloat16,
                    0),
                "BF16 residual add column=" + std::to_string(column));
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kOutputSlices[byte];
                schedule.mem_at(
                    slice,
                    cycle + 1
                        + slice / ftlpu::hw::kMemSlicesPerGroup,
                    ftlpu::MemInstruction::Write(
                        kOutputAddressBase + column,
                        ftlpu::StreamId::East(byte)),
                    "residual write column=" + std::to_string(column));
            }
        }
        const auto chunk_cycles = schedule.end_cycle() + 12;
        for (std::size_t cycle = 0; cycle < chunk_cycles; ++cycle) {
            system.tick({});
        }
        append_trace(trace, schedule.trace(), total_cycles, phase);
        for (std::size_t physical = 0;
             physical < kVectorRows && row_base + physical < rows;
             ++physical) {
            for (std::size_t column = 0; column < kHidden; ++column) {
                const auto actual = read_bf16(system, physical, column);
                const auto index = (row_base + physical) * kHidden + column;
                const auto expected = ftlpu::Bf16::from_float(
                    ftlpu::Bf16::from_float(lhs[index]).to_float()
                    + ftlpu::Bf16::from_float(rhs[index]).to_float())
                                          .to_float();
                if (actual != expected) {
                    throw std::runtime_error(
                        phase + " residual mismatch index="
                        + std::to_string(index));
                }
                output[index] = actual;
            }
        }
        total_cycles += chunk_cycles;
    }
    if (!trace_path.empty()) write_trace(trace_path, std::move(trace));
    return {std::move(output), {}, {}, total_cycles};
}

void merge_trace(
    const std::filesystem::path& output_path,
    const std::vector<std::pair<std::filesystem::path, std::size_t>>& phases)
{
    auto output = std::ofstream(output_path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write combined layer trace");
    output << "start,end,resource,detail\n";
    for (const auto& [path, offset] : phases) {
        auto input = std::ifstream(path);
        if (!input) {
            throw std::runtime_error("cannot read phase trace: " + path.string());
        }
        auto line = std::string {};
        std::getline(input, line);
        while (std::getline(input, line)) {
            const auto first = line.find(',');
            const auto second = line.find(',', first + 1);
            if (first == std::string::npos || second == std::string::npos) {
                throw std::runtime_error("invalid phase trace row");
            }
            const auto start = std::stoull(line.substr(0, first));
            const auto end = std::stoull(line.substr(first + 1, second - first - 1));
            output << start + offset << ',' << end + offset
                   << line.substr(second) << '\n';
        }
    }
}

std::vector<float> make_prefill_input()
{
    auto input = std::vector<float>(kPrefillLength * kHidden);
    for (std::size_t token = 0; token < kPrefillLength; ++token) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto raw = static_cast<int>((token * 7 + hidden * 5) % 29) - 14;
            input[token * kHidden + hidden] = ftlpu::Bf16::from_float(
                static_cast<float>(raw) * 0.046875f)
                                                  .to_float();
        }
    }
    return input;
}

std::vector<float> make_decode_input()
{
    auto input = std::vector<float>(kHidden);
    constexpr auto token = kPrefillLength;
    for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
        const auto raw = static_cast<int>((token * 7 + hidden * 5) % 29) - 14;
        input[hidden] = ftlpu::Bf16::from_float(
            static_cast<float>(raw) * 0.046875f)
                            .to_float();
    }
    return input;
}

} // namespace

int main() try
{
    const auto docs = std::filesystem::path("docs") / "traces";
    std::filesystem::create_directories(docs);
    const auto trace_paths = std::array<std::filesystem::path, 13> {
        docs / "smollm2_layer_00_prefill_norm1.csv",
        docs / "smollm2_layer_01_prefill_attention.csv",
        docs / "smollm2_layer_02_prefill_attention_residual.csv",
        docs / "smollm2_layer_03_prefill_norm2.csv",
        docs / "smollm2_layer_04_prefill_ffn.csv",
        docs / "smollm2_layer_05_prefill_ffn_residual.csv",
        docs / "smollm2_layer_06_decode_norm1.csv",
        docs / "smollm2_layer_07_decode_attention.csv",
        docs / "smollm2_layer_08_decode_attention_residual.csv",
        docs / "smollm2_layer_09_decode_norm2.csv",
        docs / "smollm2_layer_10_decode_ffn.csv",
        docs / "smollm2_layer_11_decode_ffn_residual.csv",
        docs / "smollm2_prefill_decode_layer_schedule.csv"};

    auto system = ftlpu::TspSliceSystem {};
    const auto prefill_input = make_prefill_input();
    const auto prefill_norm1 = run_rmsnorm(
        system, prefill_input, trace_paths[0], "prefill norm1");
    const auto prefill_attention =
        ftlpu::test::smollm2_layer::run_prefill_attention(
            system, prefill_norm1.output, trace_paths[1]);
    const auto prefill_attention_residual = run_residual(
        system,
        prefill_input,
        prefill_attention.output,
        trace_paths[2],
        "prefill attention residual");
    const auto prefill_norm2 = run_rmsnorm(
        system,
        prefill_attention_residual.output,
        trace_paths[3],
        "prefill norm2");
    const auto prefill_ffn = ftlpu::test::smollm2_layer::run_prefill_ffn(
        system, prefill_norm2.output, trace_paths[4]);
    const auto prefill_output = run_residual(
        system,
        prefill_attention_residual.output,
        prefill_ffn.output,
        trace_paths[5],
        "prefill FFN residual");

    const auto decode_input = make_decode_input();
    const auto decode_norm1 = run_rmsnorm(
        system, decode_input, trace_paths[6], "decode norm1");
    const auto decode_attention =
        ftlpu::test::smollm2_layer::run_decode_attention(
            system,
            decode_norm1.output,
            trace_paths[7],
            std::filesystem::path("logs")
                / "smollm2_prefill_decode_layer_test",
            prefill_attention.key_cache,
            prefill_attention.value_cache);
    const auto decode_attention_residual = run_residual(
        system,
        decode_input,
        decode_attention.output,
        trace_paths[8],
        "decode attention residual");
    const auto decode_norm2 = run_rmsnorm(
        system,
        decode_attention_residual.output,
        trace_paths[9],
        "decode norm2");
    const auto decode_ffn = ftlpu::test::smollm2_layer::run_decode_ffn(
        system, decode_norm2.output, trace_paths[10]);
    const auto decode_output = run_residual(
        system,
        decode_attention_residual.output,
        decode_ffn.output,
        trace_paths[11],
        "decode FFN residual");

    const std::array<std::size_t, 12> cycles {
        prefill_norm1.cycles,
        prefill_attention.cycles,
        prefill_attention_residual.cycles,
        prefill_norm2.cycles,
        prefill_ffn.cycles,
        prefill_output.cycles,
        decode_norm1.cycles,
        decode_attention.cycles,
        decode_attention_residual.cycles,
        decode_norm2.cycles,
        decode_ffn.cycles,
        decode_output.cycles};
    auto phases =
        std::vector<std::pair<std::filesystem::path, std::size_t>> {};
    auto total_cycles = std::size_t {0};
    for (std::size_t phase = 0; phase < cycles.size(); ++phase) {
        phases.emplace_back(trace_paths[phase], total_cycles);
        total_cycles += cycles[phase];
    }
    merge_trace(trace_paths[12], phases);

    auto checksum = 0.0;
    for (const auto value : decode_output.output) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("final decode output is not finite");
        }
        checksum += value;
    }
    std::cout
        << "SmolLM2 full layer prefill+decode passed: "
        << "prefill=[128,576], decode=[1,576], "
        << "RMSNorm/residual/attention/SwiGLU/FFN verified; cycles="
        << total_cycles << ", checksum=" << checksum
        << ", trace=" << trace_paths[12].string() << '\n';
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "SmolLM2 full layer prefill+decode failed: "
              << error.what() << '\n';
    return 1;
}
