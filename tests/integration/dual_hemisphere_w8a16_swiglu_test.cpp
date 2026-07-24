#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "vxm_alu_program.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
constexpr std::size_t kTile = ftlpu::hw::kMxmRows;
constexpr std::size_t kWeightToIwLatency = 14;
constexpr std::size_t kActivationLatency = 5;
constexpr std::size_t kGateAccumulatorLatency = 6;
constexpr std::size_t kUpAccumulatorLatency = 5;
constexpr std::size_t kSwishWriteLatency = 13;
constexpr std::size_t kMemToMxmLatency = 13;
constexpr std::size_t kDownOutputCastLatency = 16;
constexpr std::size_t kDownActivationStreamBase = 16;
constexpr std::size_t kDownActivationOutputBase = 8;
constexpr std::size_t kDownAccumulatorAddressBase = 7000;
constexpr std::size_t kPrefetchWeightStreamBase = 8;
constexpr std::size_t kComputeBlockCycles = kTile;
constexpr std::size_t kOutputLowSlice = 29;
constexpr std::size_t kOutputHighSlice = 30;
constexpr std::array<std::size_t, 8> kWeightSlices {0, 4, 8, 12, 16, 20, 24, 28};
constexpr std::array<std::size_t, 8> kDownWeightSlices {1, 5, 9, 13, 17, 21, 25, 31};
constexpr std::array<std::size_t, 4> kActivationSlices {32, 33, 34, 35};
constexpr std::array<std::size_t, 4> kDownOutputSlices {32, 33, 34, 35};

static_assert(kRows % kTile == 0);
static_assert(kHidden % kTile == 0);
static_assert(kIntermediate % (2 * kTile) == 0);

enum class Projection : std::size_t { Gate, Up };

std::size_t a_index(std::size_t row, std::size_t k) { return row * kHidden + k; }
std::size_t w_index(std::size_t k, std::size_t n) { return k * kIntermediate + n; }
std::size_t down_w_index(std::size_t k, std::size_t n) { return k * kHidden + n; }

float activation_value(std::size_t row, std::size_t k)
{
    return static_cast<float>(static_cast<int>((row * 7 + k * 5) % 23) - 11) * 0.0625f;
}

float weight_value(Projection projection, std::size_t k, std::size_t n)
{
    const auto p = static_cast<std::size_t>(projection);
    const auto raw = static_cast<int>((k * (11 + p * 6) + n * (5 + p * 2) + p * 13) % 41) - 20;
    return static_cast<float>(raw) * (0.009f + static_cast<float>((n + p) % 7) * 0.001f);
}

float down_weight_value(std::size_t k, std::size_t n)
{
    const auto raw = static_cast<int>((k * 19 + n * 11 + 7) % 47) - 23;
    return static_cast<float>(raw)
        * (0.006f + static_cast<float>((n + 3) % 9) * 0.001f);
}

class OfflineSchedule {
public:
    struct TraceEvent {
        std::size_t start{0};
        std::size_t end{0};
        std::string resource;
        std::string detail;
    };

    explicit OfflineSchedule(ftlpu::InstructionControlUnit& icu)
        : icu_(icu)
        , trace_enabled_(std::getenv("FTLPU_SCHEDULE_TRACE") != nullptr)
    {
    }

    void mem_at(std::size_t queue, std::size_t cycle, ftlpu::MemInstruction instruction)
    {
        require_available(mem_[queue], cycle, "MEM q" + std::to_string(queue));
        pad(mem_[queue], cycle, [&](std::size_t count) { icu_.enqueue_mem_nop(queue, count); });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
        trace_mem(queue, cycle, cycle + 1, instruction);
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
        if (count > 1) trace_mem(queue, cycle + 1, cycle + count, instruction, stride);
    }

    void dequant_at(std::size_t cycle, const ftlpu::test::DequantSpec& instruction)
    {
        ftlpu::test::enqueue_dequant(icu_, vxm_, cycle, instruction);
        end_cycle_ = std::max(end_cycle_, cycle + 2);
        trace(cycle, cycle + 1, "VXM.ALU0-7", "dequant int8 multiply");
        trace(cycle + 1, cycle + 2, "VXM.ALU8-15", "dequant cast FP16");
    }

    void down_prefetch_dequant_at(
        std::size_t cycle, const ftlpu::test::DequantSpec& instruction)
    {
        constexpr std::array<std::size_t, ftlpu::hw::kLanesPerTile> kAlus {
            0, 1, 6, 7, 8, 9, 14, 15};
        for (std::size_t element = 0;
             element < ftlpu::hw::kLanesPerTile;
             ++element) {
            const auto alu = kAlus[element];
            ftlpu::test::enqueue_alu_at(icu_, vxm_, alu, cycle, {
                ftlpu::VxmAluOpcode::Multiply,
                ftlpu::VxmLaneOperand::StreamInt8(
                    instruction.input_stream_base + element),
                ftlpu::VxmLaneOperand::Imm(instruction.scales[element]),
                1.0f, 0, ftlpu::VxmCastTarget::Float32, std::nullopt,
                instruction.input_hemisphere,
                instruction.output_hemisphere});
            ftlpu::test::enqueue_alu_at(icu_, vxm_, alu, cycle + 1, {
                ftlpu::VxmAluOpcode::Cast,
                ftlpu::VxmLaneOperand::Alu(alu),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Float16,
                instruction.output_stream_base + element * 2,
                instruction.input_hemisphere,
                instruction.output_hemisphere});
        }
        end_cycle_ = std::max(end_cycle_, cycle + 2);
        trace(cycle, cycle + 2, "VXM.DownPrefetch",
            "two-cycle down weight dequant on fanout-idle ALUs");
    }

    void swish_at(std::size_t cycle, const ftlpu::test::SwishSpec& instruction)
    {
        ftlpu::test::enqueue_swish(icu_, vxm_, cycle, instruction);
        const auto duplicate_hemisphere = instruction.output_hemisphere
                == ftlpu::Hemisphere::East
            ? ftlpu::Hemisphere::West
            : ftlpu::Hemisphere::East;
        ftlpu::test::enqueue_alu_at(icu_, vxm_, 10, cycle + 5, {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(8),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f, 0, ftlpu::VxmCastTarget::Float16,
            instruction.output_stream_base,
            instruction.input_hemisphere,
            duplicate_hemisphere});
        end_cycle_ = std::max(end_cycle_, cycle + 6);
        trace(cycle, cycle + 1, "VXM.ALU0", "SwiGLU negate gate");
        trace(cycle, cycle + 1, "VXM.ALU1", "SwiGLU gate * up");
        trace(cycle + 1, cycle + 2, "VXM.ALU2", "SwiGLU exp(-gate)");
        trace(cycle + 1, cycle + 2, "VXM.ALU5", "SwiGLU product delay");
        trace(cycle + 2, cycle + 3, "VXM.ALU3", "SwiGLU 1 + exp(-gate)");
        trace(cycle + 2, cycle + 3, "VXM.ALU6", "SwiGLU product delay");
        trace(cycle + 3, cycle + 4, "VXM.ALU4", "SwiGLU reciprocal sigmoid denominator");
        trace(cycle + 3, cycle + 4, "VXM.ALU7", "SwiGLU product delay");
        trace(cycle + 4, cycle + 5, "VXM.ALU8", "SwiGLU product * sigmoid");
        trace(cycle + 5, cycle + 6, "VXM.ALU9", "SwiGLU cast FP16 -> local E0");
        trace(cycle + 5, cycle + 6, "VXM.ALU10", "SwiGLU cast FP16 -> remote E0");
    }

    void route_local_down_activation_at(
        std::size_t cycle,
        std::size_t destination_count,
        std::size_t output_stream_base)
    {
        constexpr std::array<std::size_t, 2> kAluBases {2, 10};
        for (std::size_t destination = 0; destination < destination_count; ++destination) {
            const auto hemisphere = static_cast<ftlpu::Hemisphere>(destination);
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto alu = kAluBases[destination] + byte;
                ftlpu::test::enqueue_alu_at(icu_, vxm_, alu, cycle, {
                    ftlpu::VxmAluOpcode::Pass,
                    ftlpu::VxmLaneOperand::StreamInt8(
                        ftlpu::hw::kEastStreams + kDownActivationStreamBase + byte),
                    ftlpu::VxmLaneOperand::Imm(0.0f),
                    1.0f, 0, ftlpu::VxmCastTarget::Int8,
                    output_stream_base + byte,
                    hemisphere, hemisphere});
                trace(cycle, cycle + 1, "VXM.ALU" + std::to_string(alu),
                    "local SwiGLU FP16 broadcast to down MXM0/1");
            }
        }
        end_cycle_ = std::max(end_cycle_, cycle + 1);
    }

    void cast_down_pair_at(std::size_t cycle, ftlpu::Hemisphere hemisphere)
    {
        const auto alu_base = hemisphere == ftlpu::Hemisphere::East ? 0u : 8u;
        const auto instruction = [&](std::size_t input_stream, std::size_t output_stream) {
            return ftlpu::VxmLaneAluInstruction {
                ftlpu::VxmAluOpcode::Pass,
                ftlpu::VxmLaneOperand::StreamFloat32(input_stream),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f, 0, ftlpu::VxmCastTarget::Float16, output_stream,
                hemisphere, hemisphere};
        };
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base, cycle, instruction(32, 0));
        ftlpu::test::enqueue_alu_at(icu_, vxm_, alu_base + 1, cycle, instruction(36, 2));
        trace(cycle, cycle + 1, "VXM.ALU" + std::to_string(alu_base),
            "down MXM0 FP32 -> FP16");
        trace(cycle, cycle + 1, "VXM.ALU" + std::to_string(alu_base + 1),
            "down MXM1 FP32 -> FP16");
        end_cycle_ = std::max(end_cycle_, cycle + 1);
    }

    void mxm_load_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t weight_column,
        std::size_t weight_buffer = 0)
    {
        require_available(mxm_load_[mxm], cycle, "MXM load " + std::to_string(mxm));
        pad(mxm_load_[mxm], cycle, [&](std::size_t count) { icu_.enqueue_mxm_load_nop(mxm, count); });
        icu_.enqueue_mxm(
            mxm, ftlpu::MxmControlInstruction::IW(weight_buffer, weight_column));
        advance(mxm_load_[mxm], cycle + 1);
        trace(cycle, cycle + 1, mxm_name(mxm) + ".Load",
            "IW buffer=" + std::to_string(weight_buffer)
                + " column=" + std::to_string(weight_column));
    }

    void mxm_compute_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t activation,
        std::size_t output,
        std::size_t weight_buffer = 0)
    {
        require_available(mxm_compute_[mxm], cycle, "MXM compute " + std::to_string(mxm));
        pad(mxm_compute_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_compute_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::Compute(
                weight_buffer, activation, output));
        icu_.enqueue_mxm_compute_repeat(mxm, kTile - 1, 1);
        advance(mxm_compute_[mxm], cycle + kTile);
        trace(cycle, cycle + kTile, mxm_name(mxm) + ".Compute",
            "Compute buffer=" + std::to_string(weight_buffer)
                + " act=E" + std::to_string(activation)
                + " out=W" + std::to_string(output));
        if (kComputeBlockCycles > kTile) {
            trace(cycle + kTile, cycle + kComputeBlockCycles,
                mxm_name(mxm) + ".Tail", "control + datapath drain");
        }
    }

    std::size_t end_cycle() const { return end_cycle_; }

    std::size_t vxm_end_cycle() const
    {
        return *std::max_element(vxm_.begin(), vxm_.end());
    }

    void write_trace_csv(const std::string& path) const
    {
        auto output = std::ofstream(path, std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open schedule trace output: " + path);
        output << "start,end,resource,detail\n";
        for (const auto& event : trace_events_) {
            output << event.start << ',' << event.end << ','
                   << csv_field(event.resource) << ',' << csv_field(event.detail) << '\n';
        }
    }

private:
    static const char* mem_opcode_name(ftlpu::MemOpcode opcode)
    {
        switch (opcode) {
        case ftlpu::MemOpcode::Read: return "Read";
        case ftlpu::MemOpcode::Write: return "Write";
        case ftlpu::MemOpcode::ReadWrite: return "ReadWrite";
        case ftlpu::MemOpcode::Gather: return "Gather";
        case ftlpu::MemOpcode::Scatter: return "Scatter";
        case ftlpu::MemOpcode::Accumulate: return "Accumulate";
        }
        return "Unknown";
    }

    static std::string stream_name(std::size_t packed)
    {
        const auto stream = ftlpu::StreamId::from_packed(packed);
        return std::string(stream.direction() == ftlpu::StreamDirection::East ? "E" : "W")
            + std::to_string(stream.index());
    }

    static std::string mxm_name(std::size_t mxm)
    {
        const auto hemisphere = mxm / ftlpu::TspSliceSystem::kMxmCountPerHemisphere;
        const auto local = mxm % ftlpu::TspSliceSystem::kMxmCountPerHemisphere;
        return std::string("MXM.") + (hemisphere == 0 ? "E" : "W")
            + std::to_string(local);
    }

    static std::string csv_field(const std::string& value)
    {
        auto result = std::string {"\""};
        for (const auto ch : value) {
            if (ch == '"') result += '"';
            result += ch;
        }
        result += '"';
        return result;
    }

    void trace(std::size_t start, std::size_t end, std::string resource, std::string detail)
    {
        if (trace_enabled_) {
            trace_events_.push_back(TraceEvent {start, end, std::move(resource), std::move(detail)});
        }
    }

    void trace_mem(
        std::size_t queue,
        std::size_t start,
        std::size_t end,
        const ftlpu::MemInstruction& instruction,
        std::int64_t stride = 0)
    {
        const auto hemisphere = queue / ftlpu::InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto slice = queue % ftlpu::InstructionControlUnit::kMemQueuesPerHemisphere;
        auto detail = std::string("slice=") + std::to_string(slice)
            + " addr=" + std::to_string(instruction.address)
            + " stream=" + stream_name(instruction.stream);
        if (end > start + 1) {
            detail += " count=" + std::to_string(end - start)
                + " stride=" + std::to_string(stride);
        }
        if (instruction.opcode == ftlpu::MemOpcode::Accumulate) {
            detail += instruction.accumulator_destination
                    == ftlpu::MemAccumulatorDestination::Stream
                ? " dst=stream+clear" : " dst=sram";
        }
        trace(start, end,
            std::string("MEM.") + (hemisphere == 0 ? "E." : "W.")
                + mem_opcode_name(instruction.opcode),
            std::move(detail));
    }

    static void require_available(std::size_t cursor, std::size_t cycle, const std::string& queue)
    {
        if (cycle < cursor) {
            throw std::logic_error(queue + " requests cycle " + std::to_string(cycle)
                + " but its cursor is " + std::to_string(cursor));
        }
    }

    template <typename Emit>
    static void pad(std::size_t cursor, std::size_t cycle, Emit emit)
    {
        if (cycle < cursor) throw std::logic_error("offline ICU queue schedule overlaps itself");
        emit(cycle - cursor);
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_cycle_ = std::max(end_cycle_, next);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMxmQueues> mxm_load_{};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMxmQueues> mxm_compute_{};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_{};
    std::vector<TraceEvent> trace_events_{};
    bool trace_enabled_{false};
    std::size_t end_cycle_{0};
};

std::size_t mem_queue(ftlpu::Hemisphere hemisphere, std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(hemisphere, slice);
}

std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

void initialize_activations(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& activations)
{
    for (std::size_t hemisphere_index = 0; hemisphere_index < ftlpu::hw::kHemispheres;
         ++hemisphere_index) {
        const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
        for (std::size_t k_block = 0; k_block < kHidden / kTile; ++k_block) {
            for (std::size_t row = 0; row < kRows; ++row) {
                const auto address = k_block * kRows + row;
                for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                        const auto k = k_block * kTile
                            + tile * ftlpu::hw::kLanesPerTile + lane;
                        const auto bits = ftlpu::Fp16::from_float(
                            activations[a_index(row, k)]).bits();
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kActivationSlices[0], tile, address, lane, bits & 0xffu);
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kActivationSlices[1], tile, address, lane, bits >> 8);
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kActivationSlices[2], tile, address, lane, bits & 0xffu);
                        system.initialize_mem_sram_lane_byte(
                            hemisphere, kActivationSlices[3], tile, address, lane, bits >> 8);
                    }
                }
            }
        }
    }
}

float read_fp32(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    std::size_t group_base,
    std::size_t row,
    std::size_t column)
{
    const auto local_column = column % kTile;
    const auto n_base = column / kTile * kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(system.read_mem_sram_lane_byte(
            hemisphere,
            group_base + byte,
            tile,
            row * (kIntermediate / kTile) + n_base / kTile,
            lane)) << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

float read_output(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    std::size_t row,
    std::size_t column)
{
    const auto local_column = column % kTile;
    const auto n_base = column / kTile * kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, kOutputLowSlice, tile, (n_base / kTile) * kRows + row, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, kOutputHighSlice, tile, (n_base / kTile) * kRows + row, lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low) | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

float read_down_output(
    const ftlpu::TspSliceSystem& system,
    std::size_t row,
    std::size_t column)
{
    const auto wave = column / (4 * kTile);
    const auto global_mxm = (column % (4 * kTile)) / kTile;
    const auto hemisphere = static_cast<ftlpu::Hemisphere>(global_mxm / 2);
    const auto local_mxm = global_mxm % 2;
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto address = wave * kRows + row;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, kDownOutputSlices[local_mxm * 2], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, kDownOutputSlices[local_mxm * 2 + 1], tile, address, lane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low) | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

} // namespace

int main() try
{
    std::vector<float> activations(kRows * kHidden);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t k = 0; k < kHidden; ++k) {
            activations[a_index(row, k)] = ftlpu::Fp16::from_float(activation_value(row, k)).to_float();
        }
    }

    std::array<std::vector<float>, 2> scales {
        std::vector<float>(kIntermediate), std::vector<float>(kIntermediate)};
    std::array<std::vector<std::int8_t>, 2> weights {
        std::vector<std::int8_t>(kHidden * kIntermediate),
        std::vector<std::int8_t>(kHidden * kIntermediate)};
    std::array<std::vector<float>, 2> dequantized {
        std::vector<float>(kHidden * kIntermediate),
        std::vector<float>(kHidden * kIntermediate)};
    for (std::size_t projection = 0; projection < 2; ++projection) {
        for (std::size_t n = 0; n < kIntermediate; ++n) {
            float max_abs = 0.0f;
            for (std::size_t k = 0; k < kHidden; ++k) {
                max_abs = std::max(max_abs, std::fabs(weight_value(
                    static_cast<Projection>(projection), k, n)));
            }
            scales[projection][n] = max_abs / 127.0f;
            for (std::size_t k = 0; k < kHidden; ++k) {
                const auto quantized = std::clamp(static_cast<int>(std::lround(
                    weight_value(static_cast<Projection>(projection), k, n)
                    / scales[projection][n])), -127, 127);
                weights[projection][w_index(k, n)] = static_cast<std::int8_t>(quantized);
                dequantized[projection][w_index(k, n)] = ftlpu::Fp16::from_float(
                    static_cast<float>(quantized) * scales[projection][n]).to_float();
            }
        }
    }

    auto down_scales = std::vector<float>(kHidden);
    auto down_weights = std::vector<std::int8_t>(kIntermediate * kHidden);
    auto down_dequantized = std::vector<float>(kIntermediate * kHidden);
    for (std::size_t n = 0; n < kHidden; ++n) {
        float max_abs = 0.0f;
        for (std::size_t k = 0; k < kIntermediate; ++k) {
            max_abs = std::max(max_abs, std::fabs(down_weight_value(k, n)));
        }
        down_scales[n] = max_abs / 127.0f;
        for (std::size_t k = 0; k < kIntermediate; ++k) {
            const auto quantized = std::clamp(static_cast<int>(std::lround(
                down_weight_value(k, n) / down_scales[n])), -127, 127);
            down_weights[down_w_index(k, n)] = static_cast<std::int8_t>(quantized);
            down_dequantized[down_w_index(k, n)] = ftlpu::Fp16::from_float(
                static_cast<float>(quantized) * down_scales[n]).to_float();
        }
    }

    auto system = ftlpu::TspSliceSystem {};
    initialize_activations(system, activations);
    auto schedule = OfflineSchedule(system.icu());

    std::size_t phase_start = 0;
    auto fused_swish_start = std::optional<std::size_t> {};
    std::size_t weight_address = 0;
    for (std::size_t pair_base = 0; pair_base < kIntermediate; pair_base += 2 * kTile) {
        const auto load_projection_mxm = [&](std::size_t k_block,
                                             std::size_t weight_buffer,
                                             std::size_t local_mxm,
                                             std::size_t start_cycle) {
            for (std::size_t hemisphere_index = 0;
                 hemisphere_index < ftlpu::hw::kHemispheres;
                 ++hemisphere_index) {
                const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                const auto n_base = pair_base + hemisphere_index * kTile;
                for (std::size_t pulse = 0; pulse < 4; ++pulse) {
                    const auto global_mxm = ftlpu::InstructionControlUnit::mxm_queue(
                        hemisphere, local_mxm);
                    const auto block = 3 - pulse;
                    const auto cycle = start_cycle + hemisphere_index * 4 + pulse;
                    auto instruction = ftlpu::test::DequantSpec {};
                    instruction.input_stream_base = ftlpu::hw::kEastStreams
                        + kPrefetchWeightStreamBase;
                    instruction.output_stream_base = local_mxm * 16;
                    instruction.input_hemisphere = hemisphere;
                    instruction.output_hemisphere = hemisphere;
                    for (std::size_t stream = 0; stream < ftlpu::hw::kLanesPerTile; ++stream) {
                        const auto n = n_base + block * ftlpu::hw::kLanesPerTile + stream;
                        instruction.scales[stream] = scales[local_mxm][n];
                        const auto slice = kWeightSlices[stream];
                        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                                const auto k = k_block * kTile
                                    + tile * ftlpu::hw::kLanesPerTile + lane;
                                system.initialize_mem_sram_lane_byte(
                                    hemisphere,
                                    slice,
                                    tile,
                                    weight_address,
                                    lane,
                                    static_cast<std::uint8_t>(
                                        weights[local_mxm][w_index(k, n)]));
                            }
                        }
                        schedule.mem_at(
                            mem_queue(hemisphere, slice),
                            cycle - west_read_latency(slice),
                            ftlpu::MemInstruction::Read(
                                weight_address,
                                ftlpu::StreamId::West(
                                    kPrefetchWeightStreamBase + stream)));
                    }
                    schedule.dequant_at(cycle, instruction);
                    schedule.mxm_load_at(
                        global_mxm,
                        cycle + kWeightToIwLatency,
                        block,
                        weight_buffer);
                    ++weight_address;
                }
            }
        };

        const auto initial_dequant_start = std::max(
            phase_start + 10, schedule.vxm_end_cycle());
        load_projection_mxm(0, 0, 0, initial_dequant_start);
        load_projection_mxm(0, 0, 1, initial_dequant_start + 8);
        auto first_compute = initial_dequant_start + 32;
        for (std::size_t k_block = 0; k_block < kHidden / kTile; ++k_block) {
            const auto weight_buffer = k_block % 2;
            const auto has_next_block = k_block + 1 < kHidden / kTile;
            if (has_next_block) {
                const auto next_buffer = (k_block + 1) % 2;
                load_projection_mxm(
                    k_block + 1, next_buffer, 0, first_compute);
                load_projection_mxm(
                    k_block + 1, next_buffer, 1, first_compute + kTile);
            }
            for (std::size_t row_block = 0; row_block < kRows / kTile; ++row_block) {
                const auto activation_address = k_block * kRows + row_block * kTile;
                const auto activation_stream_base = has_next_block && row_block == 0
                    ? 16u : 0u;
                for (std::size_t hemisphere_index = 0;
                     hemisphere_index < ftlpu::hw::kHemispheres;
                     ++hemisphere_index) {
                    const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                    const auto final_reduction = !has_next_block;
                    const auto compute_cycle = first_compute
                        + (final_reduction
                            ? (row_block * ftlpu::hw::kHemispheres + hemisphere_index)
                                * kComputeBlockCycles
                            : row_block * kComputeBlockCycles);
                    const auto n_base = pair_base + hemisphere_index * kTile;
                    const auto accumulator_address = row_block * kTile
                        * (kIntermediate / kTile) + n_base / kTile;
                    for (std::size_t byte = 0; byte < 2; ++byte) {
                        schedule.mem_repeat_at(
                            mem_queue(hemisphere, kActivationSlices[byte]),
                            compute_cycle - kActivationLatency,
                            ftlpu::MemInstruction::Read(
                                activation_address,
                                ftlpu::StreamId::East(
                                    activation_stream_base + byte)),
                            kTile,
                            1);
                    }
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, ftlpu::hw::kWestAccumulatorMemSliceBase),
                        compute_cycle + kGateAccumulatorLatency,
                        ftlpu::MemInstruction::Accumulate(
                            accumulator_address,
                            ftlpu::StreamId::West(0),
                            final_reduction
                                ? ftlpu::MemAccumulatorDestination::Stream
                                : ftlpu::MemAccumulatorDestination::Sram),
                        kTile,
                        kIntermediate / kTile);
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, ftlpu::hw::kEastAccumulatorMemSliceBase),
                        compute_cycle + kUpAccumulatorLatency,
                        ftlpu::MemInstruction::Accumulate(
                            accumulator_address,
                            ftlpu::StreamId::West(4),
                            final_reduction
                                ? ftlpu::MemAccumulatorDestination::Stream
                                : ftlpu::MemAccumulatorDestination::Sram),
                        kTile,
                        kIntermediate / kTile);
                    schedule.mxm_compute_at(
                        ftlpu::InstructionControlUnit::mxm_queue(hemisphere, 0),
                        compute_cycle,
                        activation_stream_base,
                        0,
                        weight_buffer);
                    schedule.mxm_compute_at(
                        ftlpu::InstructionControlUnit::mxm_queue(hemisphere, 1),
                        compute_cycle,
                        activation_stream_base,
                        4,
                        weight_buffer);

                    if (final_reduction) {
                        constexpr auto kAccumulatorToVxmLatency =
                            kGateAccumulatorLatency
                            + (ftlpu::hw::kWestAccumulatorMemSliceBase
                                / ftlpu::hw::kMemSlicesPerGroup + 1);
                        for (std::size_t offset = 0; offset < kTile; ++offset) {
                            const auto row = row_block * kTile + offset;
                            const auto vxm_cycle = compute_cycle
                                + kAccumulatorToVxmLatency + offset;
                            if (!fused_swish_start) fused_swish_start = vxm_cycle;
                            const auto swiglu_address = (n_base / kTile) * kRows + row;
                            schedule.swish_at(vxm_cycle, ftlpu::test::SwishSpec {
                                32, 36, 0, hemisphere, hemisphere});
                            for (std::size_t output_index = 0;
                                 output_index < ftlpu::hw::kHemispheres;
                                 ++output_index) {
                                const auto output_hemisphere =
                                    static_cast<ftlpu::Hemisphere>(output_index);
                                schedule.mem_at(
                                    mem_queue(output_hemisphere, kOutputLowSlice),
                                    vxm_cycle + kSwishWriteLatency,
                                    ftlpu::MemInstruction::Write(
                                        swiglu_address, ftlpu::StreamId::East(0)));
                                schedule.mem_at(
                                    mem_queue(output_hemisphere, kOutputHighSlice),
                                    vxm_cycle + kSwishWriteLatency,
                                    ftlpu::MemInstruction::Write(
                                        swiglu_address, ftlpu::StreamId::East(1)));
                            }
                        }
                    }
                }
            }
            first_compute += (kRows / kTile) * kComputeBlockCycles
                * (has_next_block ? 1 : ftlpu::hw::kHemispheres);
        }
        phase_start = first_compute;
    }

    const auto projection_end = schedule.end_cycle();
    const auto swish_start = fused_swish_start.value_or(projection_end);
    const auto swish_end = projection_end + 16;
    const auto down_start = swish_end;
    phase_start = down_start;
    std::size_t down_weight_address = 0;
    const auto load_down_mxm_weights = [&] (
        std::size_t output_base,
        std::size_t reduction_block,
        std::size_t destination_count,
        std::size_t local_mxm,
        std::size_t weight_buffer,
        std::size_t start_cycle,
        bool overlapped) {
        for (std::size_t destination = 0;
             destination < destination_count;
             ++destination) {
            const auto hemisphere = static_cast<ftlpu::Hemisphere>(destination);
            const auto weight_stream_base = local_mxm == 0 ? 8u : 24u;
            for (std::size_t pulse = 0; pulse < 4; ++pulse) {
                const auto block = 3 - pulse;
                const auto sequence = destination * 4 + pulse;
                const auto cycle = start_cycle + sequence * (overlapped ? 2 : 1);
                auto instruction = ftlpu::test::DequantSpec {};
                instruction.input_stream_base =
                    ftlpu::hw::kEastStreams + weight_stream_base;
                instruction.output_stream_base = local_mxm * 16;
                instruction.input_hemisphere = hemisphere;
                instruction.output_hemisphere = hemisphere;
                for (std::size_t stream = 0;
                     stream < ftlpu::hw::kLanesPerTile;
                     ++stream) {
                    const auto column = output_base + destination * 2 * kTile
                        + local_mxm * kTile
                        + block * ftlpu::hw::kLanesPerTile + stream;
                    instruction.scales[stream] = down_scales[column];
                    const auto slice = kDownWeightSlices[stream];
                    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                        for (std::size_t lane = 0;
                             lane < ftlpu::hw::kLanesPerTile;
                             ++lane) {
                            const auto k = reduction_block * kTile
                                + tile * ftlpu::hw::kLanesPerTile + lane;
                            system.initialize_mem_sram_lane_byte(
                                hemisphere, slice, tile, down_weight_address, lane,
                                static_cast<std::uint8_t>(
                                    down_weights[down_w_index(k, column)]));
                        }
                    }
                    schedule.mem_at(
                        mem_queue(hemisphere, slice),
                        cycle - west_read_latency(slice),
                        ftlpu::MemInstruction::Read(
                            down_weight_address,
                            ftlpu::StreamId::West(weight_stream_base + stream)));
                }
                if (overlapped) {
                    schedule.down_prefetch_dequant_at(cycle, instruction);
                } else {
                    schedule.dequant_at(cycle, instruction);
                }
                schedule.mxm_load_at(
                    ftlpu::InstructionControlUnit::mxm_queue(
                        hemisphere, local_mxm),
                    cycle + kWeightToIwLatency,
                    block,
                    weight_buffer);
                ++down_weight_address;
            }
        }
    };

    for (std::size_t output_base = 0; output_base < kHidden; output_base += 4 * kTile) {
        const auto remaining_columns = kHidden - output_base;
        const auto destination_count = remaining_columns > 2 * kTile ? 2u : 1u;
        const auto wave = output_base / (4 * kTile);
        const auto initial_dequant_start = phase_start + 10;
        load_down_mxm_weights(
            output_base, 0, destination_count, 0, 0,
            initial_dequant_start, false);
        load_down_mxm_weights(
            output_base, 0, destination_count, 1, 0,
            initial_dequant_start + destination_count * 4, false);
        auto first_compute = initial_dequant_start + 32
            + (destination_count == 2 ? 8 : 0);

        for (std::size_t reduction_block = 0;
             reduction_block < kIntermediate / kTile;
             ++reduction_block) {
            const auto has_next_reduction =
                reduction_block + 1 < kIntermediate / kTile;
            auto prefetch_mxm1_row = std::size_t {1};
            auto prefetch_mxm0_row = std::size_t {2};
            if (has_next_reduction) {
                const auto next_buffer = (reduction_block + 1) % 2;
                const auto mxm1_route = first_compute
                    + prefetch_mxm1_row * kTile - kMemToMxmLatency;
                const auto mxm0_route = first_compute
                    + prefetch_mxm0_row * kTile - kMemToMxmLatency;
                load_down_mxm_weights(
                    output_base, reduction_block + 1, destination_count,
                    1, next_buffer, mxm1_route, true);
                load_down_mxm_weights(
                    output_base, reduction_block + 1, destination_count,
                    0, next_buffer, mxm0_route, true);
            }

            const auto weight_buffer = reduction_block % 2;
            for (std::size_t row_block = 0; row_block < kRows / kTile; ++row_block) {
                const auto compute_cycle = first_compute + row_block * kComputeBlockCycles;
                const auto route_cycle = compute_cycle - kMemToMxmLatency;
                const auto swiglu_address = reduction_block * kRows + row_block * kTile;
                auto activation_base = kDownActivationOutputBase;
                if (has_next_reduction && row_block == prefetch_mxm1_row) {
                    activation_base = 12;
                }
                if (has_next_reduction && row_block == prefetch_mxm0_row) {
                    activation_base = 16;
                }
                for (std::size_t destination = 0;
                     destination < destination_count;
                     ++destination) {
                    const auto hemisphere = static_cast<ftlpu::Hemisphere>(destination);
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, kOutputLowSlice),
                        route_cycle - west_read_latency(kOutputLowSlice),
                        ftlpu::MemInstruction::Read(
                            swiglu_address,
                            ftlpu::StreamId::West(kDownActivationStreamBase)),
                        kTile, 1);
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, kOutputHighSlice),
                        route_cycle - west_read_latency(kOutputHighSlice),
                        ftlpu::MemInstruction::Read(
                            swiglu_address,
                            ftlpu::StreamId::West(kDownActivationStreamBase + 1)),
                        kTile, 1);
                }
                for (std::size_t offset = 0; offset < kTile; ++offset) {
                    schedule.route_local_down_activation_at(
                        route_cycle + offset, destination_count, activation_base);
                }

                const auto accumulator_address = kDownAccumulatorAddressBase
                    + wave * kRows + row_block * kTile;
                const auto destination = reduction_block + 1 == kIntermediate / kTile
                    ? ftlpu::MemAccumulatorDestination::Stream
                    : ftlpu::MemAccumulatorDestination::Sram;
                for (std::size_t hemisphere_index = 0;
                     hemisphere_index < destination_count;
                     ++hemisphere_index) {
                    const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, ftlpu::hw::kWestAccumulatorMemSliceBase),
                        compute_cycle + kGateAccumulatorLatency,
                        ftlpu::MemInstruction::Accumulate(
                            accumulator_address, ftlpu::StreamId::West(0), destination),
                        kTile, 1);
                    schedule.mem_repeat_at(
                        mem_queue(hemisphere, ftlpu::hw::kEastAccumulatorMemSliceBase),
                        compute_cycle + kUpAccumulatorLatency,
                        ftlpu::MemInstruction::Accumulate(
                            accumulator_address, ftlpu::StreamId::West(4), destination),
                        kTile, 1);
                    schedule.mxm_compute_at(
                        ftlpu::InstructionControlUnit::mxm_queue(hemisphere, 0),
                        compute_cycle, activation_base, 0, weight_buffer);
                    schedule.mxm_compute_at(
                        ftlpu::InstructionControlUnit::mxm_queue(hemisphere, 1),
                        compute_cycle, activation_base, 4, weight_buffer);
                    if (destination == ftlpu::MemAccumulatorDestination::Stream) {
                        for (std::size_t offset = 0; offset < kTile; ++offset) {
                            const auto row = row_block * kTile + offset;
                            const auto cast_cycle = compute_cycle
                                + kDownOutputCastLatency + offset;
                            schedule.cast_down_pair_at(cast_cycle, hemisphere);
                            for (std::size_t byte = 0;
                                 byte < kDownOutputSlices.size();
                                 ++byte) {
                                const auto slice = kDownOutputSlices[byte];
                                schedule.mem_at(
                                    mem_queue(hemisphere, slice),
                                    cast_cycle + 1
                                        + slice / ftlpu::hw::kMemSlicesPerGroup,
                                    ftlpu::MemInstruction::Write(
                                        wave * kRows + row,
                                        ftlpu::StreamId::East(byte)));
                            }
                        }
                    }
                }
            }
            first_compute += (kRows / kTile) * kComputeBlockCycles;
        }
        phase_start = schedule.end_cycle() + 10;
    }

    const auto scheduled_cycles = schedule.end_cycle() + 16;
    if (const auto* trace_path = std::getenv("FTLPU_SCHEDULE_TRACE")) {
        schedule.write_trace_csv(trace_path);
    }
    if (std::getenv("FTLPU_SCHEDULE_REPORT") != nullptr) {
        std::cout << "schedule_phase cycle=0 name=gate/up projection start\n"
                  << "schedule_phase cycle=" << swish_start
                  << " name=first fused SwiGLU block starts\n"
                  << "schedule_phase cycle=" << projection_end
                  << " name=gate/up plus fused SwiGLU issue end\n"
                  << "schedule_phase cycle=" << swish_end << " name=SwiGLU end\n"
                  << "schedule_phase cycle=" << down_start << " name=down projection start\n"
                  << "schedule_phase cycle=" << scheduled_cycles << " name=down projection end\n"
                  << std::flush;
    }
    if (std::getenv("FTLPU_SCHEDULE_TRACE_ONLY") != nullptr) {
        std::cout << "dual-hemisphere W8A16 FFN schedule trace generated; scheduled_cycles="
                  << scheduled_cycles << '\n';
        return 0;
    }

    for (std::size_t cycle = 0; cycle < scheduled_cycles; ++cycle) {
        try {
            system.tick({});
        } catch (const std::exception& ex) {
            throw std::runtime_error(
                "system cycle " + std::to_string(cycle) + ": " + ex.what());
        }
    }
    auto swiglu_values = std::vector<float>(kRows * kIntermediate);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t n = 0; n < kIntermediate; ++n) {
            std::array<float, 2> projected {};
            for (std::size_t projection = 0; projection < 2; ++projection) {
                for (std::size_t k = 0; k < kHidden; ++k) {
                    projected[projection] += activations[a_index(row, k)]
                        * dequantized[projection][w_index(k, n)];
                }
            }
            const auto expected = ftlpu::Fp16::from_float(
                (projected[0] * projected[1])
                    * (1.0f / (1.0f + std::exp(-projected[0])))).to_float();
            for (std::size_t hemisphere_index = 0;
                 hemisphere_index < ftlpu::hw::kHemispheres;
                 ++hemisphere_index) {
                const auto hemisphere = static_cast<ftlpu::Hemisphere>(hemisphere_index);
                const auto actual = read_output(system, hemisphere, row, n);
                if (actual != expected && !(std::isnan(actual) && std::isnan(expected))) {
                    std::cerr << "dual-hemisphere SwiGLU copy mismatch at ("
                              << row << ',' << n << ") hemisphere="
                              << hemisphere_index << '\n';
                    return 1;
                }
            }
            swiglu_values[row * kIntermediate + n] = expected;
        }
    }

    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t n = 0; n < kHidden; ++n) {
            float projected = 0.0f;
            for (std::size_t reduction_block = 0;
                 reduction_block < kIntermediate / kTile;
                 ++reduction_block) {
                float partial = 0.0f;
                for (std::size_t lane = 0; lane < kTile; ++lane) {
                    const auto k = reduction_block * kTile + lane;
                    partial += swiglu_values[row * kIntermediate + k]
                        * down_dequantized[down_w_index(k, n)];
                }
                projected += partial;
            }
            const auto expected = ftlpu::Fp16::from_float(projected).to_float();
            const auto actual = read_down_output(system, row, n);
            if (actual != expected && !(std::isnan(actual) && std::isnan(expected))) {
                std::cerr << "dual-hemisphere down projection mismatch at ("
                          << row << ',' << n << "): expected=" << expected
                          << " actual=" << actual << '\n';
                return 1;
            }
        }
    }

    std::cout << "offline dual-hemisphere W8A16 FFN passed: "
              << "X[128,576], gate/up[576,1536], SwiGLU[128,1536], "
              << "down[1536,576]\n";
    return 0;
}
catch (const std::exception& ex)
{
    std::cerr << "dual-hemisphere SwiGLU test failed: " << ex.what() << '\n';
    return 1;
}
