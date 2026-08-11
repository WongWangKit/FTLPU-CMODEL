#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "system_gantt_trace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kTokens = 8;
constexpr std::size_t kElements = 32;
constexpr float kEpsilon = 1.0e-5f;

// X occupies two independent byte-plane sets because the depth-2 square
// stage consumes x on both fixed head operands in the same cycle.  The second
// set is reused for gamma during the final phase. Scalar scratch and output
// reuse the same physical slices at disjoint SRAM addresses; one hemisphere
// owns 32 slices, and the phases are intentionally non-overlapping.
constexpr std::array<std::size_t, 16> kXLhsSlices {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
};
constexpr std::array<std::size_t, 16> kXRhsGammaSlices {
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
};
constexpr auto kScalarSlices = kXLhsSlices;
constexpr auto kOutputSlices = kXLhsSlices;

constexpr std::size_t kXAddress = 64;
constexpr std::size_t kGammaAddress = 128;
constexpr std::size_t kSquareSumAddress = 256;
constexpr std::size_t kInverseRmsAddress = 320;
constexpr std::size_t kOutputAddress = 512;

std::size_t mem_queue(std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(
        ftlpu::Hemisphere::East, slice);
}

std::size_t mem_to_vxm_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t vxm_to_mem_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 1;
}

class Schedule {
public:
    explicit Schedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(
        std::size_t slice, std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        auto& cursor = mem_[mem_queue(slice)];
        if (cycle < cursor) {
            throw std::logic_error(
                "RMSNorm schedule overlaps MEM slice="
                + std::to_string(slice) + " cursor="
                + std::to_string(cursor) + " cycle="
                + std::to_string(cycle));
        }
        icu_.enqueue_mem_nop(mem_queue(slice), cycle - cursor);
        icu_.enqueue_mem(mem_queue(slice), std::move(instruction));
        cursor = cycle + 1;
        end_cycle_ = std::max(end_cycle_, cursor);
    }

    void mem_repeat_at(
        std::size_t slice, std::size_t cycle,
        ftlpu::MemInstruction instruction, std::size_t count,
        std::int64_t address_stride)
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
        std::size_t stage, std::size_t cycle, ftlpu::VxmChainDepth depth,
        const ftlpu::VxmLaneAluInstruction& instruction)
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
                std::string("RMSNorm schedule overlaps ") + queue
                + " queue");
        }
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_ {};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kVxmQueues> vxm_ {};
    std::size_t end_cycle_{0};
};

ftlpu::VxmLaneAluInstruction basic(
    ftlpu::VxmAluOpcode opcode, ftlpu::VxmLaneOperand lhs,
    ftlpu::VxmLaneOperand rhs = ftlpu::VxmLaneOperand::Imm(0.0f),
    std::size_t repeat = 1)
{
    auto instruction = ftlpu::VxmLaneAluInstruction {opcode, lhs, rhs};
    instruction.precision = ftlpu::VxmAluPrecision::Float32;
    instruction.repeat_count = repeat;
    return instruction;
}

ftlpu::VxmLaneAluInstruction accumulator(
    bool reset, bool emit, std::size_t repeat)
{
    auto instruction = basic(
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::Previous(),
        ftlpu::VxmLaneOperand::Acc(), repeat);
    instruction.accumulator_reset = reset;
    instruction.accumulator_write = true;
    instruction.accumulator_emit = emit;
    return instruction;
}

template <typename Fn>
std::vector<ftlpu::VxmLutEntry> make_table(
    float input_min, float segment_width, std::size_t count, Fn fn)
{
    auto entries = std::vector<ftlpu::VxmLutEntry> {};
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto x0 = input_min + static_cast<float>(index) * segment_width;
        const auto y0 = fn(x0);
        entries.push_back(ftlpu::VxmLutEntry::from_float(
            (fn(x0 + segment_width) - y0) / segment_width, y0));
    }
    return entries;
}

void configure_rsqrt_lut(ftlpu::TspSliceSystem& system)
{
    constexpr std::size_t kEntries = 256;
    constexpr float kWidth = 3.0f / static_cast<float>(kEntries);
    system.initialize_vxm_lut(
        ftlpu::VxmSpecialAluOpcode::Rsqrt,
        {1.0f, kWidth},
        make_table(
            1.0f, kWidth, kEntries,
            [](float x) { return 1.0f / std::sqrt(x); }));
}

float input_value(
    std::size_t token, std::size_t element,
    std::size_t tile, std::size_t lane)
{
    const auto wave = std::sin(
        static_cast<float>(element) * 0.173f
        + static_cast<float>(token) * 0.311f
        + static_cast<float>(tile) * 0.127f
        + static_cast<float>(lane) * 0.071f);
    return 0.75f + static_cast<float>(token) * 0.035f + wave * 0.25f;
}

float gamma_value(std::size_t token, std::size_t element)
{
    return 0.875f
        + static_cast<float>((token * 7 + element * 3) % 13) * 0.0078125f;
}

void initialize_data(ftlpu::TspSliceSystem& system)
{
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t element = 0; element < kElements; ++element) {
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile; ++lane) {
                    const auto x = ftlpu::Fp16::from_float(
                        input_value(token, element, tile, lane)).bits();
                    const auto gamma = ftlpu::Fp16::from_float(
                        gamma_value(token, element)).bits();
                    for (std::size_t byte = 0; byte < 2; ++byte) {
                        const auto x_byte = static_cast<std::uint8_t>(
                            byte == 0 ? x & 0xffu : x >> 8);
                        const auto gamma_byte = static_cast<std::uint8_t>(
                            byte == 0 ? gamma & 0xffu : gamma >> 8);
                        system.initialize_mem_sram_lane_byte(
                            kXLhsSlices[token * 2 + byte], tile,
                            kXAddress + element, lane, x_byte);
                        system.initialize_mem_sram_lane_byte(
                            kXRhsGammaSlices[token * 2 + byte], tile,
                            kXAddress + element, lane, x_byte);
                        system.initialize_mem_sram_lane_byte(
                            kXRhsGammaSlices[token * 2 + byte], tile,
                            kGammaAddress + element, lane, gamma_byte);
                    }
                }
            }
        }
    }
}

void schedule_read_pair(
    Schedule& schedule, const std::array<std::size_t, 16>& slices,
    std::size_t token, std::size_t address, std::size_t stream,
    std::size_t input_cycle, std::size_t count,
    std::int64_t address_stride)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto slice = slices[token * 2 + byte];
        schedule.mem_repeat_at(
            slice, input_cycle - mem_to_vxm_latency(slice),
            ftlpu::MemInstruction::Read(
                address, ftlpu::StreamId::West(stream + byte)),
            count, address_stride);
    }
}

void schedule_write_pair(
    Schedule& schedule, const std::array<std::size_t, 16>& slices,
    std::size_t token, std::size_t address, std::size_t stream,
    std::size_t output_cycle, std::size_t count,
    std::int64_t address_stride)
{
    for (std::size_t byte = 0; byte < 2; ++byte) {
        const auto slice = slices[token * 2 + byte];
        schedule.mem_repeat_at(
            slice, output_cycle + vxm_to_mem_latency(slice),
            ftlpu::MemInstruction::Write(
                address, ftlpu::StreamId::East(stream + byte)),
            count, address_stride);
    }
}

std::size_t build_schedule(Schedule& schedule)
{
    // Phase 1: eight depth-2 chains compute eight independent square sums.
    constexpr std::size_t kSquareConfigCycle = 40;
    constexpr std::size_t kSquareInputCycle = kSquareConfigCycle + 1;
    for (std::size_t head = 0; head < 8; head += 2) {
        schedule.vxm_at(
            head, kSquareConfigCycle, ftlpu::VxmChainDepth::Two,
            basic(
                ftlpu::VxmAluOpcode::Multiply,
                ftlpu::VxmLaneOperand::StreamFloat16(),
                ftlpu::VxmLaneOperand::StreamFloat16(), kElements));

        const auto tail = head + 1;
        schedule.vxm_at(
            tail, kSquareConfigCycle, ftlpu::VxmChainDepth::Two,
            accumulator(true, false, 1));
        if (kElements > 2) {
            schedule.vxm_at(
                tail, kSquareConfigCycle + 1, ftlpu::VxmChainDepth::Two,
                accumulator(false, false, kElements - 2));
        }
        auto final = accumulator(false, true, 1);
        final.output_type = ftlpu::VxmCastTarget::Float16;
        final.output_stream =
            ftlpu::VxmLane::fixed_output_stream_for_block(head / 2);
        schedule.vxm_at(
            tail, kSquareConfigCycle + 2,
            ftlpu::VxmChainDepth::Two, final);
    }

    for (std::size_t token = 0; token < kTokens; ++token) {
        schedule_read_pair(
            schedule, kXLhsSlices, token, kXAddress,
            token * 4, kSquareInputCycle, kElements, 1);
        schedule_read_pair(
            schedule, kXRhsGammaSlices, token, kXAddress,
            token * 4 + 2, kSquareInputCycle, kElements, 1);
    }
    constexpr std::size_t kSquareOutputCycle =
        kSquareInputCycle + kElements + 1;
    for (std::size_t token = 0; token < kTokens; ++token) {
        schedule_write_pair(
            schedule, kScalarSlices, token, kSquareSumAddress,
            token * 2, kSquareOutputCycle, 1, 0);
    }
    // Phase 2: four depth-4 chains process two scalar waves.  C3/C7 host
    // the fixed Rsqrt LUT; their mirrored copies are C11/C15.
    constexpr std::size_t kRsqrtConfigCycle = kSquareOutputCycle + 40;
    constexpr std::size_t kRsqrtInputCycle = kRsqrtConfigCycle + 1;
    for (const auto head : {std::size_t {0}, std::size_t {4}}) {
        schedule.vxm_at(
            head, kRsqrtConfigCycle, ftlpu::VxmChainDepth::Four,
            basic(
                ftlpu::VxmAluOpcode::Multiply,
                ftlpu::VxmLaneOperand::StreamFloat16(),
                ftlpu::VxmLaneOperand::Imm(
                    1.0f / static_cast<float>(kElements)),
                2));
        schedule.vxm_at(
            head + 1, kRsqrtConfigCycle, ftlpu::VxmChainDepth::Four,
            basic(
                ftlpu::VxmAluOpcode::Add,
                ftlpu::VxmLaneOperand::Previous(),
                ftlpu::VxmLaneOperand::Imm(kEpsilon), 2));
        schedule.vxm_at(
            head + 2, kRsqrtConfigCycle, ftlpu::VxmChainDepth::Four,
            basic(
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::Previous(),
                ftlpu::VxmLaneOperand::Imm(0.0f), 2));
        auto rsqrt = ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmSpecialAluOpcode::Rsqrt,
            ftlpu::VxmLaneOperand::Previous()};
        rsqrt.repeat_count = 2;
        rsqrt.output_type = ftlpu::VxmCastTarget::Float16;
        rsqrt.output_stream =
            ftlpu::VxmLane::fixed_output_stream_for_block((head + 3) / 2);
        schedule.vxm_at(
            head + 3, kRsqrtConfigCycle,
            ftlpu::VxmChainDepth::Four, rsqrt);
    }

    constexpr std::array<std::size_t, 4> kDepth4InputStreams {0, 8, 16, 24};
    constexpr std::array<std::size_t, 4> kDepth4OutputStreams {2, 6, 10, 14};
    for (std::size_t wave = 0; wave < 2; ++wave) {
        for (std::size_t chain = 0; chain < 4; ++chain) {
            const auto token = wave * 4 + chain;
            schedule_read_pair(
                schedule, kScalarSlices, token, kSquareSumAddress,
                kDepth4InputStreams[chain], kRsqrtInputCycle + wave, 1, 0);
        }
    }
    constexpr std::size_t kRsqrtOutputCycle = kRsqrtInputCycle + 8;
    for (std::size_t wave = 0; wave < 2; ++wave) {
        for (std::size_t chain = 0; chain < 4; ++chain) {
            const auto token = wave * 4 + chain;
            schedule_write_pair(
                schedule, kScalarSlices, token, kInverseRmsAddress,
                kDepth4OutputStreams[chain], kRsqrtOutputCycle + wave, 1, 0);
        }
    }
    // Phase 3: redistribute the eight scalar results into the eight C1/C3
    // local registers using only stream data and instructions.
    constexpr std::size_t kScalarLoadConfigCycle = kRsqrtOutputCycle + 40;
    constexpr std::size_t kScalarLoadInputCycle = kScalarLoadConfigCycle + 1;
    for (std::size_t head = 0; head < 8; head += 2) {
        schedule.vxm_at(
            head, kScalarLoadConfigCycle, ftlpu::VxmChainDepth::Two,
            basic(
                ftlpu::VxmAluOpcode::Bypass,
                ftlpu::VxmLaneOperand::StreamFloat16()));
        auto capture = basic(
            ftlpu::VxmAluOpcode::Bypass,
            ftlpu::VxmLaneOperand::Previous());
        capture.local_scalar_write = true;
        schedule.vxm_at(
            head + 1, kScalarLoadConfigCycle,
            ftlpu::VxmChainDepth::Two, capture);
    }
    for (std::size_t token = 0; token < kTokens; ++token) {
        schedule_read_pair(
            schedule, kScalarSlices, token, kInverseRmsAddress,
            token * 4, kScalarLoadInputCycle, 1, 0);
    }
    // Phase 4: all eight depth-2 chains normalize vector elements in parallel.
    constexpr std::size_t kNormalizeConfigCycle = kScalarLoadInputCycle + 20;
    constexpr std::size_t kNormalizeInputCycle = kNormalizeConfigCycle + 1;
    for (std::size_t head = 0; head < 8; head += 2) {
        schedule.vxm_at(
            head, kNormalizeConfigCycle, ftlpu::VxmChainDepth::Two,
            basic(
                ftlpu::VxmAluOpcode::Multiply,
                ftlpu::VxmLaneOperand::StreamFloat16(),
                ftlpu::VxmLaneOperand::StreamFloat16(), kElements));
        auto normalize = basic(
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Previous(),
            ftlpu::VxmLaneOperand::Acc(), kElements);
        normalize.output_type = ftlpu::VxmCastTarget::Float16;
        normalize.output_stream =
            ftlpu::VxmLane::fixed_output_stream_for_block(head / 2);
        schedule.vxm_at(
            head + 1, kNormalizeConfigCycle,
            ftlpu::VxmChainDepth::Two, normalize);
    }
    constexpr std::size_t kNormalizeOutputCycle = kNormalizeInputCycle + 3;
    for (std::size_t token = 0; token < kTokens; ++token) {
        // Gamma owns independent slices and is a normal read stream.
        schedule_read_pair(
            schedule, kXRhsGammaSlices, token, kGammaAddress,
            token * 4 + 2, kNormalizeInputCycle, kElements, 1);

        // Every East-hemisphere slice is already occupied by x or gamma.
        // The dual-port MEM instruction overlaps the middle of the x read
        // stream with the delayed VXM result write.  Prefix/suffix transfers
        // cover the non-overlapping pipeline fill and drain cycles.
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kXLhsSlices[token * 2 + byte];
            const auto read_cycle =
                kNormalizeInputCycle - mem_to_vxm_latency(slice);
            const auto write_cycle =
                kNormalizeOutputCycle + vxm_to_mem_latency(slice);
            const auto overlap_offset = write_cycle - read_cycle;
            if (overlap_offset >= kElements) {
                throw std::logic_error(
                    "RMSNorm vector is too short to overlap MEM ReadWrite");
            }

            schedule.mem_repeat_at(
                slice, read_cycle,
                ftlpu::MemInstruction::Read(
                    kXAddress,
                    ftlpu::StreamId::West(token * 4 + byte)),
                overlap_offset, 1);
            schedule.mem_repeat_at(
                slice, write_cycle,
                ftlpu::MemInstruction::ReadWrite(
                    kXAddress + overlap_offset,
                    ftlpu::StreamId::West(token * 4 + byte),
                    kOutputAddress,
                    ftlpu::StreamId::East(token * 2 + byte)),
                kElements - overlap_offset, 1);
            schedule.mem_repeat_at(
                slice, read_cycle + kElements,
                ftlpu::MemInstruction::Write(
                    kOutputAddress + kElements - overlap_offset,
                    ftlpu::StreamId::East(token * 2 + byte)),
                overlap_offset, 1);
        }
    }
    return schedule.end_cycle() + 24;
}

float read_output(
    const ftlpu::TspSliceSystem& system, std::size_t token,
    std::size_t element, std::size_t tile, std::size_t lane)
{
    const auto low = system.read_mem_sram_lane_byte(
        kOutputSlices[token * 2], tile, kOutputAddress + element, lane);
    const auto high = system.read_mem_sram_lane_byte(
        kOutputSlices[token * 2 + 1], tile, kOutputAddress + element, lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

} // namespace

int main() try
{
    auto system = ftlpu::TspSliceSystem {};
    initialize_data(system);
    configure_rsqrt_lut(system);

    auto schedule = Schedule {system.icu()};
    const auto cycles = build_schedule(schedule);
    auto timing = integration_timing::SystemGanttTrace {};
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        system.tick({});
        timing.capture(system);
    }

    auto maximum_error = 0.0f;
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile; ++lane) {
                auto square_sum = 0.0f;
                for (std::size_t element = 0; element < kElements; ++element) {
                    const auto x = ftlpu::Fp16::from_float(
                        input_value(token, element, tile, lane)).to_float();
                    square_sum += x * x;
                }
                const auto inverse_rms = 1.0f / std::sqrt(
                    square_sum / static_cast<float>(kElements) + kEpsilon);
                for (std::size_t element = 0; element < kElements; ++element) {
                    const auto x = ftlpu::Fp16::from_float(
                        input_value(token, element, tile, lane)).to_float();
                    const auto gamma = ftlpu::Fp16::from_float(
                        gamma_value(token, element)).to_float();
                    const auto expected = ftlpu::Fp16::from_float(
                        x * gamma * inverse_rms).to_float();
                    const auto actual = read_output(
                        system, token, element, tile, lane);
                    maximum_error = std::max(
                        maximum_error, std::fabs(actual - expected));
                    if (std::fabs(actual - expected) > 1.5e-2f) {
                        std::cerr
                            << "RMSNorm mismatch token=" << token
                            << " element=" << element
                            << " tile=" << tile
                            << " lane=" << lane
                            << " actual=" << actual
                            << " expected=" << expected
                            << " square_sum=" << square_sum << '\n';
                        return 1;
                    }
                }
            }
        }
    }

    std::cout
        << "RMSNorm black-box passed: MEM -> VXM(depth2 square reduction)"
        << " -> MEM -> VXM(depth4 rsqrt) -> MEM"
        << " -> VXM(depth2 scalar load/normalize) -> MEM"
        << ", tokens=8 elements=32 max_error=" << maximum_error << '\n';
    timing.write(
        "rmsnorm_system", "RMSNorm black-box system timing");
    return 0;
}
catch (const std::exception& ex)
{
    std::cerr << "RMSNorm black-box test failed: " << ex.what() << '\n';
    return 1;
}
