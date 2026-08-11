#pragma once

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "system_gantt_trace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::test::softmax_dataflow {

constexpr std::size_t kRows = 8;
constexpr std::size_t kBytePlanes = 2;
constexpr std::size_t kRawAddress = 64;
constexpr std::size_t kXAddress = 256;
constexpr std::size_t kMaxAddress = 512;
constexpr std::size_t kResultAddress = 768;
constexpr float kMaskValue = -32.0f;

constexpr std::array<std::size_t, 16> kRawSlices {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
};
constexpr std::array<std::size_t, 16> kXSlices {
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
};
constexpr std::array<std::size_t, 16> kMaxSlices {
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
};

struct Scenario {
    std::string name;
    std::size_t elements{0};
    float attention_scale{1.0f};
    std::array<std::size_t, kRows> valid_elements{};
};

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

    void mem_at(std::size_t slice, std::size_t cycle, MemInstruction instruction)
    {
        auto& cursor = mem_[mem_queue(slice)];
        require_available(cursor, cycle, "MEM");
        icu_.enqueue_mem_nop(mem_queue(slice), cycle - cursor);
        icu_.enqueue_mem(mem_queue(slice), std::move(instruction));
        cursor = cycle + 1;
        end_cycle_ = std::max(end_cycle_, cursor);
    }

    void mem_repeat_at(
        std::size_t slice, std::size_t cycle, MemInstruction instruction,
        std::size_t count, std::int64_t address_stride)
    {
        if (count == 0) return;
        mem_at(slice, cycle, std::move(instruction));
        if (count > 1) {
            icu_.enqueue_mem_repeat(
                mem_queue(slice), count - 1, 1, address_stride);
        }
        mem_[mem_queue(slice)] = cycle + count;
        end_cycle_ = std::max(end_cycle_, cycle + count);
    }

    void vxm_at(
        std::size_t stage, std::size_t cycle, VxmChainDepth depth,
        const VxmLaneAluInstruction& instruction)
    {
        auto& cursor = vxm_[stage];
        require_available(cursor, cycle, "VXM");
        icu_.enqueue_vxm_nop(stage, cycle - cursor);
        icu_.enqueue_vxm(stage, depth, instruction);
        cursor = cycle + 1;
        end_cycle_ = std::max(end_cycle_, cursor);
    }

    std::size_t end_cycle() const noexcept { return end_cycle_; }

private:
    static void require_available(
        std::size_t cursor, std::size_t cycle, const char* queue)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                std::string("Softmax schedule overlaps ") + queue + " queue");
        }
    }

    InstructionControlUnit& icu_;
    std::array<std::size_t, InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, InstructionControlUnit::kVxmQueues> vxm_{};
    std::size_t end_cycle_{0};
};

template <typename Fn>
inline std::vector<VxmLutEntry> make_table(
    float input_min, float segment_width, std::size_t count, Fn fn)
{
    auto entries = std::vector<VxmLutEntry>{};
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto x0 = input_min + static_cast<float>(index) * segment_width;
        const auto y0 = fn(x0);
        entries.push_back(VxmLutEntry::from_float(
            (fn(x0 + segment_width) - y0) / segment_width, y0));
    }
    return entries;
}

inline VxmSpecialAlu configure_luts(TspSliceSystem& system)
{
    constexpr std::size_t kEntries = 256;
    constexpr float kLn2 = 0.6931471805599453f;
    auto reference = VxmSpecialAlu{};
    const auto configure = [&](VxmSpecialAluOpcode opcode,
                               VxmLutConfig config,
                               const std::vector<VxmLutEntry>& entries) {
        system.initialize_vxm_lut(opcode, config, entries);
        reference.configure_lut(opcode, config, entries);
    };
    configure(
        VxmSpecialAluOpcode::Exp,
        {-kLn2 / 2.0f, kLn2 / static_cast<float>(kEntries)},
        make_table(
            -kLn2 / 2.0f, kLn2 / static_cast<float>(kEntries), kEntries,
            [](float x) { return std::exp(x); }));
    configure(
        VxmSpecialAluOpcode::Reciprocal,
        {1.0f, 1.0f / static_cast<float>(kEntries)},
        make_table(
            1.0f, 1.0f / static_cast<float>(kEntries), kEntries,
            [](float x) { return 1.0f / x; }));
    return reference;
}

inline float raw_score(
    std::size_t row, std::size_t element,
    std::size_t tile, std::size_t lane)
{
    const auto pattern = static_cast<int>(
        (row * 29 + element * 17 + tile * 11 + lane * 7) % 97) - 48;
    return static_cast<float>(pattern) / 4.0f;
}

inline void initialize_raw_scores(
    TspSliceSystem& system, const Scenario& scenario)
{
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t element = 0; element < scenario.elements; ++element) {
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    const auto bits = Bf16::from_float(
                        raw_score(row, element, tile, lane)).bits();
                    system.initialize_mem_sram_lane_byte(
                        kRawSlices[row * 2], tile,
                        kRawAddress + element, lane,
                        static_cast<std::uint8_t>(bits & 0xffu));
                    system.initialize_mem_sram_lane_byte(
                        kRawSlices[row * 2 + 1], tile,
                        kRawAddress + element, lane,
                        static_cast<std::uint8_t>(bits >> 8));
                }
            }
        }
    }
}

inline VxmLaneAluInstruction basic(
    VxmAluOpcode opcode, VxmLaneOperand lhs,
    VxmLaneOperand rhs = VxmLaneOperand::Imm(0.0f),
    std::size_t repeat = 1)
{
    auto instruction = VxmLaneAluInstruction{opcode, lhs, rhs};
    instruction.precision = VxmAluPrecision::Float32;
    instruction.repeat_count = repeat;
    return instruction;
}

inline VxmLaneAluInstruction special(
    VxmSpecialAluOpcode opcode, VxmLaneOperand lhs,
    std::size_t repeat = 1)
{
    auto instruction = VxmLaneAluInstruction{opcode, lhs};
    instruction.repeat_count = repeat;
    return instruction;
}

inline VxmLaneAluInstruction accumulator(
    VxmAluOpcode opcode, bool reset, bool emit, std::size_t repeat,
    bool output)
{
    auto instruction = basic(
        opcode, VxmLaneOperand::Previous(), VxmLaneOperand::Acc(), repeat);
    instruction.accumulator_reset = reset;
    instruction.accumulator_write = true;
    instruction.accumulator_emit = emit;
    if (output) {
        instruction.output_type = VxmCastTarget::BFloat16;
    }
    return instruction;
}

inline void schedule_stream_read(
    Schedule& schedule, const std::array<std::size_t, 16>& slices,
    std::size_t row, std::size_t address, std::size_t input_stream,
    std::size_t first_input_cycle, std::size_t count,
    std::int64_t address_stride)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto slice = slices[row * 2 + byte];
        schedule.mem_repeat_at(
            slice, first_input_cycle - mem_to_vxm_latency(slice),
            MemInstruction::Read(
                address, StreamId::West(input_stream + byte)),
            count, address_stride);
    }
}

inline void schedule_stream_write(
    Schedule& schedule, const std::array<std::size_t, 16>& slices,
    std::size_t row, std::size_t address, std::size_t output_stream,
    std::size_t first_output_cycle, std::size_t count,
    std::int64_t address_stride)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto slice = slices[row * 2 + byte];
        schedule.mem_repeat_at(
            slice, first_output_cycle + vxm_to_mem_latency(slice),
            MemInstruction::Write(
                address, StreamId::East(output_stream + byte)),
            count, address_stride);
    }
}

inline void schedule_generate_x(
    Schedule& schedule, const Scenario& scenario, std::size_t config_cycle)
{
    const auto input_cycle = config_cycle + 1;
    for (std::size_t logical_head = 0; logical_head < 8; logical_head += 2) {
        const auto pair = logical_head / 2;
        auto multiply = basic(
            VxmAluOpcode::Multiply,
            VxmLaneOperand::StreamBFloat16(),
            VxmLaneOperand::Imm(scenario.attention_scale),
            scenario.elements);
        schedule.vxm_at(
            logical_head, config_cycle, VxmChainDepth::Two, multiply);

        const auto valid = scenario.valid_elements[pair];
        auto valid_add = basic(
            VxmAluOpcode::Add, VxmLaneOperand::Previous(),
            VxmLaneOperand::Imm(0.0f), valid);
        valid_add.output_type = VxmCastTarget::BFloat16;
        valid_add.output_stream =
            VxmLane::fixed_output_stream_for_block(
                VxmLane::block_for_stage(logical_head + 1));
        schedule.vxm_at(
            logical_head + 1, config_cycle,
            VxmChainDepth::Two, valid_add);
        if (valid < scenario.elements) {
            auto masked_add = valid_add;
            masked_add.rhs = VxmLaneOperand::Imm(kMaskValue);
            masked_add.repeat_count = scenario.elements - valid;
            schedule.vxm_at(
                logical_head + 1, config_cycle + 1,
                VxmChainDepth::Two, masked_add);
        }
    }

    for (std::size_t row = 0; row < kRows; ++row) {
        schedule_stream_read(
            schedule, kRawSlices, row, kRawAddress,
            row * 4, input_cycle, scenario.elements, 1);
        schedule_stream_write(
            schedule, kXSlices, row, kXAddress,
            row * 2, input_cycle + 2, scenario.elements, 1);
    }
}

inline void schedule_max(
    Schedule& schedule, const Scenario& scenario, std::size_t config_cycle)
{
    const auto input_cycle = config_cycle + 1;
    for (std::size_t logical_head = 0; logical_head < 8; logical_head += 2) {
        schedule.vxm_at(
            logical_head, config_cycle, VxmChainDepth::Two,
            basic(
                VxmAluOpcode::Bypass,
                VxmLaneOperand::StreamBFloat16(),
                VxmLaneOperand::Imm(0.0f), scenario.elements));

        const auto tail = logical_head + 1;
        schedule.vxm_at(
            tail, config_cycle, VxmChainDepth::Two,
            accumulator(VxmAluOpcode::Max, true, false, 1, false));
        if (scenario.elements > 2) {
            schedule.vxm_at(
                tail, config_cycle + 1, VxmChainDepth::Two,
                accumulator(
                    VxmAluOpcode::Max, false, false,
                    scenario.elements - 2, false));
        }
        auto final = accumulator(VxmAluOpcode::Max, false, true, 1, true);
        final.output_stream = VxmLane::fixed_output_stream_for_block(
            VxmLane::block_for_stage(tail));
        schedule.vxm_at(
            tail, config_cycle + 2, VxmChainDepth::Two, final);
    }

    for (std::size_t row = 0; row < kRows; ++row) {
        schedule_stream_read(
            schedule, kXSlices, row, kXAddress,
            row * 4, input_cycle, scenario.elements, 1);
        schedule_stream_write(
            schedule, kMaxSlices, row, kMaxAddress,
            row * 2, input_cycle + scenario.elements, 1, 0);
    }
}

inline void schedule_depth4_inputs(
    Schedule& schedule, const Scenario& scenario, std::size_t wave,
    std::size_t input_cycle)
{
    for (std::size_t chain = 0; chain < 4; ++chain) {
        const auto row = wave * 4 + chain;
        const auto stream_base = chain * 8;
        schedule_stream_read(
            schedule, kXSlices, row, kXAddress,
            stream_base, input_cycle, scenario.elements, 1);
        schedule_stream_read(
            schedule, kMaxSlices, row, kMaxAddress,
            stream_base + 2, input_cycle, scenario.elements, 0);
    }
}

inline void schedule_sum_reciprocal(
    Schedule& schedule, const Scenario& scenario,
    std::size_t wave, std::size_t config_cycle)
{
    const auto input_cycle = config_cycle + 1;
    for (const auto head : {std::size_t{0}, std::size_t{4}}) {
        schedule.vxm_at(
            head, config_cycle, VxmChainDepth::Four,
            basic(
                VxmAluOpcode::Subtract,
                VxmLaneOperand::StreamBFloat16(),
                VxmLaneOperand::StreamBFloat16(), scenario.elements));
        schedule.vxm_at(
            head + 1, config_cycle, VxmChainDepth::Four,
            special(
                VxmSpecialAluOpcode::Exp,
                VxmLaneOperand::Previous(), scenario.elements));
        schedule.vxm_at(
            head + 2, config_cycle, VxmChainDepth::Four,
            basic(
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous(),
                VxmLaneOperand::Imm(0.0f), scenario.elements));
        schedule.vxm_at(
            head + 3, config_cycle, VxmChainDepth::Four,
            accumulator(VxmAluOpcode::Add, true, false, 1, false));
        if (scenario.elements > 2) {
            schedule.vxm_at(
                head + 3, config_cycle + 1, VxmChainDepth::Four,
                accumulator(
                    VxmAluOpcode::Add, false, false,
                    scenario.elements - 2, false));
        }
        schedule.vxm_at(
            head + 3, config_cycle + 2, VxmChainDepth::Four,
            accumulator(VxmAluOpcode::Add, false, true, 1, false));

        schedule.vxm_at(
            head, config_cycle + 1, VxmChainDepth::Four,
            basic(
                VxmAluOpcode::Bypass, VxmLaneOperand::Feedback(),
                VxmLaneOperand::Imm(0.0f), 1));
        schedule.vxm_at(
            head + 1, config_cycle + 1, VxmChainDepth::Four,
            basic(VxmAluOpcode::Bypass, VxmLaneOperand::Previous()));
        schedule.vxm_at(
            head + 2, config_cycle + 1, VxmChainDepth::Four,
            basic(VxmAluOpcode::Bypass, VxmLaneOperand::Previous()));
        auto reciprocal = special(
            VxmSpecialAluOpcode::Reciprocal,
            VxmLaneOperand::Previous());
        reciprocal.local_scalar_write = true;
        schedule.vxm_at(
            head + 3, config_cycle + 3,
            VxmChainDepth::Four, reciprocal);
    }
    schedule_depth4_inputs(schedule, scenario, wave, input_cycle);
}

inline void schedule_normalize(
    Schedule& schedule, const Scenario& scenario,
    std::size_t wave, std::size_t config_cycle)
{
    const auto input_cycle = config_cycle + 1;
    for (const auto head : {std::size_t{0}, std::size_t{4}}) {
        schedule.vxm_at(
            head, config_cycle, VxmChainDepth::Four,
            basic(
                VxmAluOpcode::Subtract,
                VxmLaneOperand::StreamBFloat16(),
                VxmLaneOperand::StreamBFloat16(), scenario.elements));
        schedule.vxm_at(
            head + 1, config_cycle, VxmChainDepth::Four,
            special(
                VxmSpecialAluOpcode::Exp,
                VxmLaneOperand::Previous(), scenario.elements));
        schedule.vxm_at(
            head + 2, config_cycle, VxmChainDepth::Four,
            basic(
                VxmAluOpcode::Bypass,
                VxmLaneOperand::Previous(),
                VxmLaneOperand::Imm(0.0f), scenario.elements));
        auto multiply = basic(
            VxmAluOpcode::Multiply,
            VxmLaneOperand::Previous(),
            VxmLaneOperand::Acc(), scenario.elements);
        multiply.output_type = VxmCastTarget::BFloat16;
        multiply.output_stream =
            VxmLane::fixed_output_stream_for_block(
                VxmLane::block_for_stage(head + 3));
        schedule.vxm_at(
            head + 3, config_cycle,
            VxmChainDepth::Four, multiply);
    }
    schedule_depth4_inputs(schedule, scenario, wave, input_cycle);
    constexpr std::array<std::size_t, 4> kOutputStreams {2, 6, 10, 14};
    for (std::size_t chain = 0; chain < 4; ++chain) {
        const auto row = wave * 4 + chain;
        schedule_stream_write(
            schedule, kRawSlices, row, kResultAddress,
            kOutputStreams[chain], input_cycle + 8,
            scenario.elements, 1);
    }
}

inline std::size_t build_schedule(
    Schedule& schedule, const Scenario& scenario)
{
    constexpr std::size_t kGenerateCycle = 30;
    const auto max_cycle = kGenerateCycle + scenario.elements + 40;
    schedule_generate_x(schedule, scenario, kGenerateCycle);
    schedule_max(schedule, scenario, max_cycle);

    auto wave_cycle = max_cycle + scenario.elements + 50;
    for (std::size_t wave = 0; wave < 2; ++wave) {
        schedule_sum_reciprocal(schedule, scenario, wave, wave_cycle);
        const auto normalize_cycle = wave_cycle + scenario.elements + 24;
        schedule_normalize(
            schedule, scenario, wave, normalize_cycle);
        wave_cycle = normalize_cycle + scenario.elements + 32;
    }
    return wave_cycle;
}

inline Bf16 read_bf16(
    const TspSliceSystem& system,
    const std::array<std::size_t, 16>& slices,
    std::size_t row, std::size_t tile,
    std::size_t address, std::size_t lane)
{
    const auto low = system.read_mem_sram_lane_byte(
        slices[row * 2], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        slices[row * 2 + 1], tile, address, lane);
    return Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8));
}

inline bool verify(
    const TspSliceSystem& system, const Scenario& scenario,
    const VxmSpecialAlu& lut)
{
    float largest_math_error = 0.0f;
    float largest_sum_error = 0.0f;
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                auto x = std::vector<float>(scenario.elements);
                for (std::size_t element = 0;
                     element < scenario.elements; ++element) {
                    const auto raw = Bf16::from_float(
                        raw_score(row, element, tile, lane)).to_float();
                    const auto mask = element < scenario.valid_elements[row]
                        ? 0.0f : kMaskValue;
                    x[element] = Bf16::from_float(
                        raw * scenario.attention_scale + mask).to_float();
                }
                const auto maximum =
                    *std::max_element(x.begin(), x.end());
                auto approximate_exp = std::vector<float>(scenario.elements);
                double approximate_sum = 0.0;
                double mathematical_sum = 0.0;
                for (std::size_t element = 0;
                     element < scenario.elements; ++element) {
                    approximate_exp[element] = lut.execute(
                        VxmSpecialAluOpcode::Exp,
                        x[element] - maximum);
                    approximate_sum += approximate_exp[element];
                    mathematical_sum += std::exp(x[element] - maximum);
                }
                const auto inverse_sum = lut.execute(
                    VxmSpecialAluOpcode::Reciprocal,
                    static_cast<float>(approximate_sum));
                double actual_sum = 0.0;
                for (std::size_t element = 0;
                     element < scenario.elements; ++element) {
                    const auto expected = Bf16::from_float(
                        approximate_exp[element] * inverse_sum);
                    const auto actual = read_bf16(
                        system, kRawSlices, row, tile,
                        kResultAddress + element, lane);
                    if (actual.bits() != expected.bits()) {
                        std::cerr
                            << scenario.name << " Softmax mismatch"
                            << " row=" << row
                            << " tile=" << tile
                            << " lane=" << lane
                            << " element=" << element
                            << " actual=" << actual.to_float()
                            << " expected=" << expected.to_float()
                            << '\n';
                        return false;
                    }
                    const auto mathematical = static_cast<float>(
                        std::exp(x[element] - maximum) / mathematical_sum);
                    largest_math_error = std::max(
                        largest_math_error,
                        std::fabs(actual.to_float() - mathematical));
                    actual_sum += actual.to_float();
                }
                largest_sum_error = std::max(
                    largest_sum_error,
                    static_cast<float>(std::fabs(actual_sum - 1.0)));
            }
        }
    }
    if (largest_math_error > 0.02f || largest_sum_error > 0.03f) {
        std::cerr
            << scenario.name << " Softmax numerical tolerance failed"
            << " max_abs_error=" << largest_math_error
            << " max_sum_error=" << largest_sum_error << '\n';
        return false;
    }
    std::cout
        << scenario.name
        << " Softmax passed: MEM raw score -> VXM scale/mask -> MEM x -> "
           "VXM max -> MEM scalar -> VXM exp/sum -> feedback reciprocal -> "
           "local scalar -> VXM normalize -> MEM BF16"
        << ", max_abs_error=" << largest_math_error
        << ", max_sum_error=" << largest_sum_error << '\n';
    return true;
}

inline int run(const Scenario& scenario)
try {
    if (scenario.elements < 3) {
        throw std::invalid_argument("Softmax test requires at least three elements");
    }
    for (std::size_t row = 0; row < kRows; ++row) {
        if (scenario.valid_elements[row] == 0
            || scenario.valid_elements[row] > scenario.elements
            || scenario.valid_elements[row]
                != scenario.valid_elements[row % 4]) {
            throw std::invalid_argument(
                "mirrored VXM row pairs require matching non-zero valid lengths");
        }
    }

    auto system = TspSliceSystem{};
    const auto reference_lut = configure_luts(system);
    initialize_raw_scores(system, scenario);
    auto schedule = Schedule(system.icu());
    const auto scheduled_end = build_schedule(schedule, scenario);
    auto timing = integration_timing::SystemGanttTrace {};
    const auto run_cycles = std::max(schedule.end_cycle(), scheduled_end)
        + hw::kTileRows + 24;
    for (std::size_t cycle = 0; cycle < run_cycles; ++cycle) {
        try {
            system.tick({});
            timing.capture(system);
        } catch (const std::exception& error) {
            std::cerr << scenario.name
                      << " Softmax hardware schedule failed at cycle "
                      << cycle << ": " << error.what() << '\n';
            return 1;
        }
    }
    const auto passed = verify(system, scenario, reference_lut);
    timing.write(
        integration_timing::SystemGanttTrace::prefix_from_name(
            scenario.name) + "_system",
        scenario.name + " Softmax system timing");
    return passed ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << scenario.name << " Softmax setup failed: "
              << error.what() << '\n';
    return 1;
}

} // namespace ftlpu::test::softmax_dataflow
