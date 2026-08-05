#include "ftlpu/core/bf16.hpp"
#include "smollm2_layer_phases.hpp"
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
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
constexpr std::size_t kTile = ftlpu::hw::kMxmRows;
constexpr std::size_t kBlockRows = ftlpu::hw::kMxmBlockRows;
constexpr std::size_t kRowBlocks = kRows / kBlockRows;
constexpr std::size_t kGateUpColumnsPerWave = 2 * kTile;
constexpr std::size_t kDownColumnsPerWave =
    ftlpu::TspSliceSystem::kMxmCount * kTile;
constexpr std::size_t kLoadToComputeGap = 8;
constexpr std::size_t kComputeToLoadGap = 8;
constexpr std::size_t kBlockGroupStride = 12;
constexpr std::size_t kComputeSpan =
    (kRows / kTile - 1) * kBlockGroupStride + 4;
constexpr std::array<std::array<std::size_t, 8>, 2> kWeightSlices {{
    {{0, 1, 2, 3, 4, 5, 6, 7}},
    {{8, 9, 10, 11, 12, 13, 14, 15}},
}};
constexpr std::array<std::size_t, 16> kActivationSlices {
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
};
constexpr std::array<std::size_t, 8> kSwigluInputSlices {
    0, 1, 2, 3, 4, 5, 6, 7,
};

static_assert(kRows % kBlockRows == 0);
static_assert(kHidden % kTile == 0);
static_assert(kIntermediate % kTile == 0);

enum class Projection : std::size_t {
    Gate = 0,
    Up = 1,
};

std::size_t activation_index(
    std::size_t row,
    std::size_t column,
    std::size_t width)
{
    return row * width + column;
}

std::size_t gate_up_weight_index(std::size_t k, std::size_t n)
{
    return k * kIntermediate + n;
}

std::size_t down_weight_index(std::size_t k, std::size_t n)
{
    return k * kHidden + n;
}

float activation_value(std::size_t row, std::size_t k)
{
    return static_cast<float>(
        static_cast<int>((row * 7 + k * 5) % 23) - 11)
        * 0.0625f;
}

float gate_up_weight_value(
    Projection projection,
    std::size_t k,
    std::size_t n)
{
    const auto p = static_cast<std::size_t>(projection);
    const auto raw = static_cast<int>(
        (k * (11 + p * 6) + n * (5 + p * 2) + p * 13) % 41)
        - 20;
    return static_cast<float>(raw)
        * (0.009f + static_cast<float>((n + p) % 7) * 0.001f);
}

float down_weight_value(std::size_t k, std::size_t n)
{
    const auto raw =
        static_cast<int>((k * 19 + n * 11 + 7) % 47) - 23;
    return static_cast<float>(raw)
        * (0.006f + static_cast<float>((n + 3) % 9) * 0.001f);
}

std::size_t mem_queue(
    ftlpu::Hemisphere hemisphere,
    std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(
        hemisphere,
        slice);
}

std::size_t east_read_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t west_read_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t swiglu_write_latency(std::size_t slice)
{
    constexpr std::size_t kSwigluPipelineCycles = 6;
    return kSwigluPipelineCycles
        + slice / ftlpu::hw::kMemSlicesPerGroup;
}

struct TraceEvent {
    std::size_t start{0};
    std::size_t end{0};
    std::string resource;
    std::string detail;
};

class OfflineSchedule {
public:
    OfflineSchedule(
        ftlpu::InstructionControlUnit& icu,
        std::vector<TraceEvent>& trace,
        std::size_t trace_offset,
        std::string phase)
        : icu_(icu)
        , trace_(trace)
        , trace_offset_(trace_offset)
        , phase_(std::move(phase))
    {
    }

    void mem_at(
        std::size_t queue,
        std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        require_available(mem_[queue], cycle, "MEM");
        icu_.enqueue_mem_nop(queue, cycle - mem_[queue]);
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
    }

    void dequant_at(
        std::size_t mxm,
        std::size_t cycle,
        float scale,
        std::size_t output_group)
    {
        require_available(
            mxm_dequant_[mxm],
            cycle,
            "MXM Dequant");
        icu_.enqueue_mxm_dequant_nop(
            mxm,
            cycle - mxm_dequant_[mxm]);
        icu_.enqueue_mxm_dequant(
            mxm,
            ftlpu::MxmDequantInstruction::Scale(scale));
        advance(mxm_dequant_[mxm], cycle + 1);
        add_trace(
            cycle,
            cycle + 1,
            mxm_name(mxm) + ".Dequant",
            "BF16 scale group=" + std::to_string(output_group));
    }

    void load_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t column_block)
    {
        require_available(mxm_load_[mxm], cycle, "MXM load");
        icu_.enqueue_mxm_load_nop(
            mxm,
            cycle - mxm_load_[mxm]);
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::IW(0, column_block));
        advance(mxm_load_[mxm], cycle + 1);
        add_trace(
            cycle,
            cycle + 1,
            mxm_name(mxm) + ".Load",
            "INT8 IW column_block=" + std::to_string(column_block));
    }

    void compute_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t accumulator_base)
    {
        require_available(
            mxm_compute_[mxm],
            cycle,
            "MXM compute");
        icu_.enqueue_mxm_compute_nop(
            mxm,
            cycle - mxm_compute_[mxm]);
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::Compute(
                0,
                0,
                0,
                accumulator_base,
                1,
                ftlpu::MxmAccumulatorDestination::Sram,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmComputeMode::Block8));
        advance(mxm_compute_[mxm], cycle + 1);
    }

    void trace_compute_window(
        std::size_t mxm,
        std::size_t start,
        std::size_t end,
        std::size_t wave,
        std::size_t reduction,
        bool final_reduction)
    {
        add_trace(
            start,
            end,
            mxm_name(mxm) + ".Compute",
            "Block8 wave=" + std::to_string(wave)
                + " reduction=" + std::to_string(reduction)
                + " rows=128 "
                + (final_reduction
                    ? "final sum ready dst=sram"
                    : "partial sum dst=sram"));
    }

    void trace_mem_window(
        ftlpu::Hemisphere hemisphere,
        std::size_t start,
        std::size_t end,
        std::string detail)
    {
        add_trace(
            start,
            end,
            std::string("MEM.") + ftlpu::hemisphere_short_name(hemisphere)
                + ".Read",
            std::move(detail));
    }

    void trace_mem_write(
        ftlpu::Hemisphere hemisphere,
        std::size_t start,
        std::size_t end,
        std::string detail)
    {
        add_trace(
            start,
            end,
            std::string("MEM.") + ftlpu::hemisphere_short_name(hemisphere)
                + ".Write",
            std::move(detail));
    }

    void swiglu_at(
        std::size_t cycle,
        ftlpu::Hemisphere input_hemisphere,
        std::size_t wave,
        std::size_t row)
    {
        ftlpu::test::enqueue_swish(
            icu_,
            vxm_,
            cycle,
            ftlpu::test::SwishSpec {
                ftlpu::hw::kEastStreams,
                ftlpu::hw::kEastStreams + 4,
                0,
                input_hemisphere,
                input_hemisphere,
                ftlpu::VxmCastTarget::BFloat16,
            });
        const auto duplicate_hemisphere =
            input_hemisphere == ftlpu::Hemisphere::East
            ? ftlpu::Hemisphere::West
            : ftlpu::Hemisphere::East;
        ftlpu::test::enqueue_alu_at(
            icu_,
            vxm_,
            10,
            cycle + 5,
            {
                ftlpu::VxmAluOpcode::Cast,
                ftlpu::VxmLaneOperand::Alu(8),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                ftlpu::VxmCastTarget::BFloat16,
                0,
                input_hemisphere,
                duplicate_hemisphere,
            });
        end_cycle_ = std::max(end_cycle_, cycle + 6);

        const auto suffix =
            "wave=" + std::to_string(wave)
            + " row=" + std::to_string(row);
        add_trace(cycle, cycle + 1, "VXM.ALU0", "negate gate " + suffix);
        add_trace(cycle, cycle + 1, "VXM.ALU1", "gate * up " + suffix);
        add_trace(cycle + 1, cycle + 2, "VXM.ALU2", "exp(-gate) " + suffix);
        add_trace(cycle + 1, cycle + 2, "VXM.ALU5", "product delay " + suffix);
        add_trace(cycle + 2, cycle + 3, "VXM.ALU3", "1 + exp(-gate) " + suffix);
        add_trace(cycle + 2, cycle + 3, "VXM.ALU6", "product delay " + suffix);
        add_trace(cycle + 3, cycle + 4, "VXM.ALU4", "reciprocal " + suffix);
        add_trace(cycle + 3, cycle + 4, "VXM.ALU7", "product delay " + suffix);
        add_trace(cycle + 4, cycle + 5, "VXM.ALU8", "SwiGLU multiply " + suffix);
        add_trace(cycle + 5, cycle + 6, "VXM.ALU9", "BF16 local write " + suffix);
        add_trace(cycle + 5, cycle + 6, "VXM.ALU10", "BF16 remote write " + suffix);
    }

    std::size_t end_cycle() const
    {
        return end_cycle_;
    }

private:
    static void require_available(
        std::size_t cursor,
        std::size_t cycle,
        const char* resource)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                std::string(resource) + " queue overlap at cycle "
                + std::to_string(cycle));
        }
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_cycle_ = std::max(end_cycle_, next);
    }

    static std::string mxm_name(std::size_t mxm)
    {
        const auto hemisphere =
            mxm / ftlpu::TspSliceSystem::kMxmCountPerHemisphere;
        const auto local =
            mxm % ftlpu::TspSliceSystem::kMxmCountPerHemisphere;
        return std::string("MXM.")
            + (hemisphere == 0 ? "E" : "W")
            + std::to_string(local);
    }

    void add_trace(
        std::size_t start,
        std::size_t end,
        std::string resource,
        std::string detail)
    {
        trace_.push_back(TraceEvent {
            trace_offset_ + start,
            trace_offset_ + end,
            std::move(resource),
            phase_ + ": " + detail,
        });
    }

    ftlpu::InstructionControlUnit& icu_;
    std::vector<TraceEvent>& trace_;
    std::size_t trace_offset_{0};
    std::string phase_;
    std::array<
        std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues> mem_ {};
    std::array<
        std::size_t,
        ftlpu::InstructionControlUnit::kMxmQueues> mxm_load_ {};
    std::array<
        std::size_t,
        ftlpu::InstructionControlUnit::kMxmQueues> mxm_dequant_ {};
    std::array<
        std::size_t,
        ftlpu::InstructionControlUnit::kMxmQueues> mxm_compute_ {};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_ {};
    std::size_t end_cycle_{0};
};

void write_trace_csv(
    const std::string& path,
    const std::vector<TraceEvent>& events)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot open FFN schedule trace output: " + path);
    }
    auto ordered_events = events;
    std::stable_sort(
        ordered_events.begin(),
        ordered_events.end(),
        [](const TraceEvent& lhs, const TraceEvent& rhs) {
            if (lhs.start != rhs.start) {
                return lhs.start < rhs.start;
            }
            if (lhs.end != rhs.end) {
                return lhs.end < rhs.end;
            }
            return lhs.resource < rhs.resource;
        });

    const auto quote_csv = [](const std::string& value) {
        auto quoted = std::string {"\""};
        quoted.reserve(value.size() + 2);
        for (const auto ch : value) {
            quoted.push_back(ch);
            if (ch == '"') {
                quoted.push_back('"');
            }
        }
        quoted.push_back('"');
        return quoted;
    };

    output << "start,end,resource,detail\n";
    for (const auto& event : ordered_events) {
        output << event.start << ',' << event.end
               << ',' << quote_csv(event.resource)
               << ',' << quote_csv(event.detail) << '\n';
    }
}

void initialize_block_activations(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& values,
    std::size_t width)
{
    const auto reduction_blocks = width / kTile;
    for (std::size_t hemisphere_index = 0;
         hemisphere_index < ftlpu::hw::kHemispheres;
         ++hemisphere_index) {
        const auto hemisphere =
            static_cast<ftlpu::Hemisphere>(hemisphere_index);
        for (std::size_t reduction = 0;
             reduction < reduction_blocks;
             ++reduction) {
            for (std::size_t row_block = 0;
                 row_block < kRowBlocks;
                 ++row_block) {
                const auto address =
                    reduction * kRowBlocks + row_block;
                for (std::size_t stream = 0;
                     stream < kActivationSlices.size();
                     ++stream) {
                    const auto output_row = stream / 2;
                    const auto byte = stream % 2;
                    for (std::size_t tile = 0;
                         tile < ftlpu::hw::kTileRows;
                         ++tile) {
                        for (std::size_t lane = 0;
                             lane < ftlpu::hw::kLanesPerTile;
                             ++lane) {
                            const auto row =
                                row_block * kBlockRows + output_row;
                            const auto k = reduction * kTile
                                + tile * ftlpu::hw::kLanesPerTile
                                + lane;
                            const auto bits = ftlpu::Bf16::from_float(
                                values[activation_index(
                                    row,
                                    k,
                                    width)]).bits();
                            system.initialize_mem_sram_lane_byte(
                                hemisphere,
                                kActivationSlices[stream],
                                tile,
                                address,
                                lane,
                                static_cast<std::uint8_t>(
                                    (bits >> (byte * 8)) & 0xffu));
                        }
                    }
                }
            }
        }
    }
}

void initialize_swiglu_inputs(
    ftlpu::TspSliceSystem& system,
    const std::array<std::vector<float>, 2>& projections)
{
    constexpr auto kWaves = kIntermediate / kGateUpColumnsPerWave;
    for (std::size_t wave = 0; wave < kWaves; ++wave) {
        for (std::size_t hemisphere_index = 0;
             hemisphere_index < ftlpu::hw::kHemispheres;
             ++hemisphere_index) {
            const auto hemisphere =
                static_cast<ftlpu::Hemisphere>(hemisphere_index);
            const auto output_base =
                wave * kGateUpColumnsPerWave
                + hemisphere_index * kTile;
            for (std::size_t row = 0; row < kRows; ++row) {
                const auto address = wave * kRows + row;
                for (std::size_t projection = 0;
                     projection < projections.size();
                     ++projection) {
                    for (std::size_t tile = 0;
                         tile < ftlpu::hw::kTileRows;
                         ++tile) {
                        for (std::size_t lane = 0;
                             lane < ftlpu::hw::kLanesPerTile;
                             ++lane) {
                            const auto column =
                                output_base
                                + tile * ftlpu::hw::kLanesPerTile
                                + lane;
                            const auto raw = std::bit_cast<std::uint32_t>(
                                projections[projection][
                                    activation_index(
                                        row,
                                        column,
                                        kIntermediate)]);
                            for (std::size_t byte = 0;
                                 byte < sizeof(float);
                                 ++byte) {
                                system.initialize_mem_sram_lane_byte(
                                    hemisphere,
                                    kSwigluInputSlices[
                                        projection * sizeof(float) + byte],
                                    tile,
                                    address,
                                    lane,
                                    static_cast<std::uint8_t>(
                                        (raw >> (byte * 8)) & 0xffu));
                            }
                        }
                    }
                }
            }
        }
    }
}

float read_block_activation(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    std::size_t row,
    std::size_t column)
{
    const auto reduction = column / kTile;
    const auto local_column = column % kTile;
    const auto tile = local_column / ftlpu::hw::kLanesPerTile;
    const auto lane = local_column % ftlpu::hw::kLanesPerTile;
    const auto address = reduction * kRowBlocks + row / kBlockRows;
    const auto stream = (row % kBlockRows) * 2;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere,
        kActivationSlices[stream],
        tile,
        address,
        lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere,
        kActivationSlices[stream + 1],
        tile,
        address,
        lane);
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

void schedule_block_activations(
    OfflineSchedule& schedule,
    std::size_t compute_cycle,
    std::size_t reduction,
    std::size_t row_block)
{
    const auto address = reduction * kRowBlocks + row_block;
    for (std::size_t hemisphere_index = 0;
         hemisphere_index < ftlpu::hw::kHemispheres;
         ++hemisphere_index) {
        const auto hemisphere =
            static_cast<ftlpu::Hemisphere>(hemisphere_index);
        auto first_read = compute_cycle;
        for (std::size_t stream = 0;
             stream < kActivationSlices.size();
             ++stream) {
            const auto slice = kActivationSlices[stream];
            const auto read_cycle =
                compute_cycle - east_read_latency(slice);
            first_read = std::min(first_read, read_cycle);
            schedule.mem_at(
                mem_queue(hemisphere, slice),
                read_cycle,
                ftlpu::MemInstruction::Read(
                    address,
                    ftlpu::StreamId::East(stream)));
        }
        schedule.trace_mem_window(
            hemisphere,
            first_read,
            compute_cycle,
            "Block8 activation reduction="
                + std::to_string(reduction)
                + " row_block=" + std::to_string(row_block));
    }
}

std::size_t block_compute_cycle(
    std::size_t compute_start,
    std::size_t row_block)
{
    return compute_start
        + (row_block / 4) * kBlockGroupStride
        + row_block % 4;
}

void run_system(
    ftlpu::TspSliceSystem& system,
    std::size_t cycles,
    const char* phase)
{
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        try {
            system.tick({});
        } catch (const std::exception& ex) {
            throw std::runtime_error(
                std::string(phase) + " cycle "
                + std::to_string(cycle) + ": " + ex.what());
        }
    }
}

} // namespace

ftlpu::test::smollm2_layer::PhaseResult
ftlpu::test::smollm2_layer::run_prefill_ffn(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path)
{
    if (input.size() != kRows * kHidden) {
        throw std::invalid_argument(
            "prefill FFN input must be [128,576]");
    }
    auto activations = input;


    std::array<std::vector<float>, 2> gate_up_scales {
        std::vector<float>(kIntermediate / 8),
        std::vector<float>(kIntermediate / 8),
    };
    std::array<std::vector<std::int8_t>, 2> gate_up_weights {
        std::vector<std::int8_t>(kHidden * kIntermediate),
        std::vector<std::int8_t>(kHidden * kIntermediate),
    };
    std::array<std::vector<float>, 2> gate_up_dequantized {
        std::vector<float>(kHidden * kIntermediate),
        std::vector<float>(kHidden * kIntermediate),
    };
    for (std::size_t projection = 0; projection < 2; ++projection) {
        for (std::size_t group = 0;
             group < kIntermediate / 8;
             ++group) {
            float max_abs = 0.0f;
            for (std::size_t k = 0; k < kHidden; ++k) {
                for (std::size_t column = 0; column < 8; ++column) {
                    max_abs = std::max(
                        max_abs,
                        std::fabs(gate_up_weight_value(
                            static_cast<Projection>(projection),
                            k,
                            group * 8 + column)));
                }
            }
            const auto scale = ftlpu::Bf16::from_float(
                max_abs / 127.0f).to_float();
            gate_up_scales[projection][group] = scale;
            for (std::size_t k = 0; k < kHidden; ++k) {
                for (std::size_t column = 0; column < 8; ++column) {
                    const auto n = group * 8 + column;
                    const auto quantized = std::clamp(
                        static_cast<int>(std::lround(
                            gate_up_weight_value(
                                static_cast<Projection>(projection),
                                k,
                                n)
                            / scale)),
                        -127,
                        127);
                    gate_up_weights[projection][
                        gate_up_weight_index(k, n)] =
                        static_cast<std::int8_t>(quantized);
                    gate_up_dequantized[projection][
                        gate_up_weight_index(k, n)] =
                        ftlpu::Bf16::from_float(
                            static_cast<float>(quantized) * scale)
                            .to_float();
                }
            }
        }
    }

    auto down_scales =
        std::vector<float>(kHidden / 8);
    auto down_weights =
        std::vector<std::int8_t>(kIntermediate * kHidden);
    auto down_dequantized =
        std::vector<float>(kIntermediate * kHidden);
    for (std::size_t group = 0; group < kHidden / 8; ++group) {
        float max_abs = 0.0f;
        for (std::size_t k = 0; k < kIntermediate; ++k) {
            for (std::size_t column = 0; column < 8; ++column) {
                max_abs = std::max(
                    max_abs,
                    std::fabs(down_weight_value(
                        k,
                        group * 8 + column)));
            }
        }
        const auto scale = ftlpu::Bf16::from_float(
            max_abs / 127.0f).to_float();
        down_scales[group] = scale;
        for (std::size_t k = 0; k < kIntermediate; ++k) {
            for (std::size_t column = 0; column < 8; ++column) {
                const auto n = group * 8 + column;
                const auto quantized = std::clamp(
                    static_cast<int>(std::lround(
                        down_weight_value(k, n) / scale)),
                    -127,
                    127);
                down_weights[down_weight_index(k, n)] =
                    static_cast<std::int8_t>(quantized);
                down_dequantized[down_weight_index(k, n)] =
                    ftlpu::Bf16::from_float(
                        static_cast<float>(quantized) * scale)
                        .to_float();
            }
        }
    }

    system.reset_execution_state();
    auto trace = std::vector<TraceEvent> {};
    initialize_block_activations(system, activations, kHidden);
    auto gate_up_schedule = OfflineSchedule(
        system.icu(),
        trace,
        0,
        "gate/up");

    auto cycle = std::size_t {20};
    constexpr auto kGateUpWaves =
        kIntermediate / kGateUpColumnsPerWave;
    constexpr auto kGateUpReductions = kHidden / kTile;
    for (std::size_t wave = 0; wave < kGateUpWaves; ++wave) {
        for (std::size_t reduction = 0;
             reduction < kGateUpReductions;
             ++reduction) {
            const auto load_start = cycle;
            for (std::size_t column_block = 0;
                 column_block < ftlpu::hw::kMxmSupercellsPerPlane;
                 ++column_block) {
                const auto load_cycle = load_start + column_block;
                for (std::size_t hemisphere_index = 0;
                     hemisphere_index < ftlpu::hw::kHemispheres;
                     ++hemisphere_index) {
                    const auto hemisphere =
                        static_cast<ftlpu::Hemisphere>(
                            hemisphere_index);
                    const auto output_base =
                        wave * kGateUpColumnsPerWave
                        + hemisphere_index * kTile;
                    for (std::size_t local_mxm = 0;
                         local_mxm < 2;
                         ++local_mxm) {
                        const auto global_mxm =
                            ftlpu::InstructionControlUnit::mxm_queue(
                                hemisphere,
                                local_mxm);
                        const auto projection = local_mxm;
                        const auto address =
                            (wave * kGateUpReductions + reduction) * 4
                            + column_block;
                        const auto group =
                            (output_base + column_block * 8) / 8;
                        for (std::size_t stream = 0;
                             stream < 8;
                             ++stream) {
                            const auto slice =
                                kWeightSlices[local_mxm][stream];
                            for (std::size_t tile = 0;
                                 tile < ftlpu::hw::kTileRows;
                                 ++tile) {
                                for (std::size_t lane = 0;
                                     lane < ftlpu::hw::kLanesPerTile;
                                     ++lane) {
                                    const auto k = reduction * kTile
                                        + tile
                                            * ftlpu::hw::kLanesPerTile
                                        + lane;
                                    const auto n = output_base
                                        + column_block * 8 + stream;
                                    system.initialize_mem_sram_lane_byte(
                                        hemisphere,
                                        slice,
                                        tile,
                                        address,
                                        lane,
                                        static_cast<std::uint8_t>(
                                            gate_up_weights[projection][
                                                gate_up_weight_index(
                                                    k,
                                                    n)]));
                                }
                            }
                            gate_up_schedule.mem_at(
                                mem_queue(hemisphere, slice),
                                load_cycle
                                    - east_read_latency(slice),
                                ftlpu::MemInstruction::Read(
                                    address,
                                    ftlpu::StreamId::East(
                                        local_mxm
                                            * ftlpu::hw::
                                                kMxmInt8LoadStreamStride
                                        + stream)));
                        }
                        gate_up_schedule.dequant_at(
                            global_mxm,
                            load_cycle,
                            gate_up_scales[projection][group],
                            group);
                        gate_up_schedule.load_at(
                            global_mxm,
                            load_cycle,
                            column_block);
                    }
                    gate_up_schedule.trace_mem_window(
                        hemisphere,
                        load_cycle - 15,
                        load_cycle,
                        "INT8 gate/up weights wave="
                            + std::to_string(wave)
                            + " reduction="
                            + std::to_string(reduction)
                            + " column_block="
                            + std::to_string(column_block));
                }
            }

            const auto compute_start =
                load_start + kLoadToComputeGap;
            for (std::size_t row_block = 0;
                 row_block < kRowBlocks;
                 ++row_block) {
                const auto compute_cycle =
                    block_compute_cycle(compute_start, row_block);
                schedule_block_activations(
                    gate_up_schedule,
                    compute_cycle,
                    reduction,
                    row_block);
                const auto accumulator_base =
                    wave * kRowBlocks
                    + (row_block / 4) * 4;
                for (std::size_t mxm = 0;
                     mxm < ftlpu::TspSliceSystem::kMxmCount;
                     ++mxm) {
                    gate_up_schedule.compute_at(
                        mxm,
                        compute_cycle,
                        accumulator_base);
                }
            }
            for (std::size_t mxm = 0;
                 mxm < ftlpu::TspSliceSystem::kMxmCount;
                 ++mxm) {
                gate_up_schedule.trace_compute_window(
                    mxm,
                    compute_start,
                    compute_start + kComputeSpan,
                    wave,
                    reduction,
                    reduction + 1 == kGateUpReductions);
            }
            cycle = compute_start + kComputeSpan
                + kComputeToLoadGap;
        }
    }

    const auto gate_up_cycles =
        gate_up_schedule.end_cycle() + 16;
    run_system(system, gate_up_cycles, "gate/up");

    std::array<std::vector<float>, 2> gate_up_outputs {
        std::vector<float>(kRows * kIntermediate),
        std::vector<float>(kRows * kIntermediate),
    };
    auto swiglu = std::vector<float>(kRows * kIntermediate);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t n = 0; n < kIntermediate; ++n) {
            const auto wave = n / kGateUpColumnsPerWave;
            const auto hemisphere_index =
                (n % kGateUpColumnsPerWave) / kTile;
            const auto local_column = n % kTile;
            const auto address =
                wave * kRowBlocks + row / kBlockRows;
            std::array<float, 2> actual {};
            std::array<float, 2> expected {};
            for (std::size_t projection = 0;
                 projection < 2;
                 ++projection) {
                const auto global_mxm =
                    hemisphere_index * 2 + projection;
                actual[projection] =
                    system.mxm_unit(global_mxm)
                        .block_accumulator()
                        .value(
                            address,
                            row % kBlockRows,
                            local_column);
                for (std::size_t k = 0; k < kHidden; ++k) {
                    expected[projection] +=
                        activations[activation_index(
                            row,
                            k,
                            kHidden)]
                        * gate_up_dequantized[projection][
                            gate_up_weight_index(k, n)];
                }
                const auto tolerance = 1.0e-5f * std::max(
                    1.0f,
                    std::fabs(expected[projection]));
                if (std::fabs(actual[projection] - expected[projection])
                    > tolerance) {
                    std::cerr
                        << "Block8 gate/up mismatch at projection="
                        << projection << " row=" << row
                        << " column=" << n
                        << " expected=" << expected[projection]
                        << " actual=" << actual[projection] << '\n';
                    throw std::runtime_error(
                        "prefill FFN hardware mismatch");
                }
                gate_up_outputs[projection][
                    activation_index(row, n, kIntermediate)] =
                    actual[projection];
            }
            swiglu[activation_index(row, n, kIntermediate)] =
                ftlpu::Bf16::from_float(
                    (actual[0] * actual[1])
                    / (1.0f + std::exp(-actual[0])))
                    .to_float();
        }
    }

    system.reset_execution_state();
    initialize_swiglu_inputs(system, gate_up_outputs);
    auto swiglu_schedule = OfflineSchedule(
        system.icu(),
        trace,
        gate_up_cycles,
        "swiglu");
    cycle = west_read_latency(kSwigluInputSlices.back());
    constexpr auto kSwigluWaves =
        kIntermediate / kGateUpColumnsPerWave;
    for (std::size_t wave = 0; wave < kSwigluWaves; ++wave) {
        for (std::size_t source_index = 0;
             source_index < ftlpu::hw::kHemispheres;
             ++source_index) {
            const auto source =
                static_cast<ftlpu::Hemisphere>(source_index);
            const auto reduction =
                wave * ftlpu::hw::kHemispheres + source_index;
            for (std::size_t row = 0; row < kRows; ++row) {
                const auto input_address = wave * kRows + row;
                auto first_read = cycle;
                for (std::size_t stream = 0;
                     stream < kSwigluInputSlices.size();
                     ++stream) {
                    const auto slice = kSwigluInputSlices[stream];
                    const auto read_cycle =
                        cycle - west_read_latency(slice);
                    first_read = std::min(first_read, read_cycle);
                    swiglu_schedule.mem_at(
                        mem_queue(source, slice),
                        read_cycle,
                        ftlpu::MemInstruction::Read(
                            input_address,
                            ftlpu::StreamId::West(stream)));
                }
                swiglu_schedule.trace_mem_window(
                    source,
                    first_read,
                    cycle,
                    "FP32 gate/up wave=" + std::to_string(wave)
                        + " row=" + std::to_string(row));
                swiglu_schedule.swiglu_at(
                    cycle,
                    source,
                    wave,
                    row);

                const auto output_address =
                    reduction * kRowBlocks + row / kBlockRows;
                const auto output_stream =
                    (row % kBlockRows) * 2;
                for (std::size_t destination_index = 0;
                     destination_index < ftlpu::hw::kHemispheres;
                     ++destination_index) {
                    const auto destination =
                        static_cast<ftlpu::Hemisphere>(
                            destination_index);
                    auto first_write =
                        std::numeric_limits<std::size_t>::max();
                    auto last_write = std::size_t {0};
                    for (std::size_t byte = 0;
                         byte < sizeof(std::uint16_t);
                         ++byte) {
                        const auto slice =
                            kActivationSlices[output_stream + byte];
                        const auto write_cycle =
                            cycle + swiglu_write_latency(slice);
                        first_write = std::min(
                            first_write,
                            write_cycle);
                        last_write = std::max(
                            last_write,
                            write_cycle);
                        swiglu_schedule.mem_at(
                            mem_queue(destination, slice),
                            write_cycle,
                            ftlpu::MemInstruction::Write(
                                output_address,
                                ftlpu::StreamId::East(byte)));
                    }
                    swiglu_schedule.trace_mem_write(
                        destination,
                        first_write,
                        last_write + 1,
                        "BF16 packed activation reduction="
                            + std::to_string(reduction)
                            + " row_block="
                            + std::to_string(row / kBlockRows));
                }
                ++cycle;
            }
        }
    }
    const auto swiglu_cycles =
        swiglu_schedule.end_cycle() + 16;
    run_system(system, swiglu_cycles, "swiglu");

    for (std::size_t hemisphere_index = 0;
         hemisphere_index < ftlpu::hw::kHemispheres;
         ++hemisphere_index) {
        const auto hemisphere =
            static_cast<ftlpu::Hemisphere>(hemisphere_index);
        for (std::size_t row = 0; row < kRows; ++row) {
            for (std::size_t n = 0; n < kIntermediate; ++n) {
                const auto actual =
                    read_block_activation(
                        system,
                        hemisphere,
                        row,
                        n);
                const auto expected =
                    swiglu[activation_index(
                        row,
                        n,
                        kIntermediate)];
                if (actual != expected
                    && !(std::isnan(actual) && std::isnan(expected))) {
                    std::cerr
                        << "VXM BF16 SwiGLU mismatch at hemisphere="
                        << hemisphere_index
                        << " row=" << row
                        << " column=" << n
                        << " expected=" << expected
                        << " actual=" << actual << '\n';
                    throw std::runtime_error(
                        "prefill FFN hardware mismatch");
                }
            }
        }
    }

    system.reset_execution_state();
    const auto down_trace_offset =
        gate_up_cycles + swiglu_cycles;
    auto down_schedule = OfflineSchedule(
        system.icu(),
        trace,
        down_trace_offset,
        "down");
    cycle = 20;
    constexpr auto kDownWaves =
        (kHidden + kDownColumnsPerWave - 1)
        / kDownColumnsPerWave;
    constexpr auto kDownReductions = kIntermediate / kTile;
    for (std::size_t wave = 0; wave < kDownWaves; ++wave) {
        for (std::size_t reduction = 0;
             reduction < kDownReductions;
             ++reduction) {
            const auto load_start = cycle;
            for (std::size_t column_block = 0;
                 column_block < ftlpu::hw::kMxmSupercellsPerPlane;
                 ++column_block) {
                const auto load_cycle = load_start + column_block;
                for (std::size_t global_mxm = 0;
                     global_mxm < ftlpu::TspSliceSystem::kMxmCount;
                     ++global_mxm) {
                    const auto output_base =
                        wave * kDownColumnsPerWave
                        + global_mxm * kTile;
                    if (output_base >= kHidden) {
                        continue;
                    }
                    const auto hemisphere =
                        static_cast<ftlpu::Hemisphere>(
                            global_mxm / 2);
                    const auto local_mxm = global_mxm % 2;
                    const auto address =
                        (wave * kDownReductions + reduction) * 4
                        + column_block;
                    const auto group =
                        (output_base + column_block * 8) / 8;
                    for (std::size_t stream = 0;
                         stream < 8;
                         ++stream) {
                        const auto slice =
                            kWeightSlices[local_mxm][stream];
                        for (std::size_t tile = 0;
                             tile < ftlpu::hw::kTileRows;
                             ++tile) {
                            for (std::size_t lane = 0;
                                 lane < ftlpu::hw::kLanesPerTile;
                                 ++lane) {
                                const auto k = reduction * kTile
                                    + tile
                                        * ftlpu::hw::kLanesPerTile
                                    + lane;
                                const auto n = output_base
                                    + column_block * 8 + stream;
                                system.initialize_mem_sram_lane_byte(
                                    hemisphere,
                                    slice,
                                    tile,
                                    address,
                                    lane,
                                    static_cast<std::uint8_t>(
                                        down_weights[
                                            down_weight_index(k, n)]));
                            }
                        }
                        down_schedule.mem_at(
                            mem_queue(hemisphere, slice),
                            load_cycle - east_read_latency(slice),
                            ftlpu::MemInstruction::Read(
                                address,
                                ftlpu::StreamId::East(
                                    local_mxm
                                        * ftlpu::hw::
                                            kMxmInt8LoadStreamStride
                                    + stream)));
                    }
                    down_schedule.dequant_at(
                        global_mxm,
                        load_cycle,
                        down_scales[group],
                        group);
                    down_schedule.load_at(
                        global_mxm,
                        load_cycle,
                        column_block);
                }
            }

            const auto compute_start =
                load_start + kLoadToComputeGap;
            for (std::size_t row_block = 0;
                 row_block < kRowBlocks;
                 ++row_block) {
                const auto compute_cycle =
                    block_compute_cycle(compute_start, row_block);
                schedule_block_activations(
                    down_schedule,
                    compute_cycle,
                    reduction,
                    row_block);
                const auto accumulator_base =
                    wave * kRowBlocks
                    + (row_block / 4) * 4;
                for (std::size_t global_mxm = 0;
                     global_mxm < ftlpu::TspSliceSystem::kMxmCount;
                     ++global_mxm) {
                    const auto output_base =
                        wave * kDownColumnsPerWave
                        + global_mxm * kTile;
                    if (output_base < kHidden) {
                        down_schedule.compute_at(
                            global_mxm,
                            compute_cycle,
                            accumulator_base);
                    }
                }
            }
            for (std::size_t global_mxm = 0;
                 global_mxm < ftlpu::TspSliceSystem::kMxmCount;
                 ++global_mxm) {
                const auto output_base =
                    wave * kDownColumnsPerWave
                    + global_mxm * kTile;
                if (output_base < kHidden) {
                    down_schedule.trace_compute_window(
                        global_mxm,
                        compute_start,
                        compute_start + kComputeSpan,
                        wave,
                        reduction,
                        reduction + 1 == kDownReductions);
                }
            }
            cycle = compute_start + kComputeSpan
                + kComputeToLoadGap;
        }
    }

    const auto down_cycles = down_schedule.end_cycle() + 16;
    run_system(system, down_cycles, "down");

    auto output = std::vector<float>(kRows * kHidden);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t n = 0; n < kHidden; ++n) {
            const auto wave = n / kDownColumnsPerWave;
            const auto global_mxm =
                (n % kDownColumnsPerWave) / kTile;
            const auto local_column = n % kTile;
            const auto address =
                wave * kRowBlocks + row / kBlockRows;
            const auto actual =
                system.mxm_unit(global_mxm)
                    .block_accumulator()
                    .value(
                        address,
                        row % kBlockRows,
                        local_column);
            output[activation_index(row, n, kHidden)] = actual;
            float expected = 0.0f;
            for (std::size_t k = 0; k < kIntermediate; ++k) {
                expected +=
                    swiglu[activation_index(
                        row,
                        k,
                        kIntermediate)]
                    * down_dequantized[
                        down_weight_index(k, n)];
            }
            const auto tolerance =
                1.0e-5f * std::max(1.0f, std::fabs(expected));
            if (std::fabs(actual - expected) > tolerance
                && !(std::isnan(actual) && std::isnan(expected))) {
                std::cerr << "Block8 down mismatch at row="
                          << row << " column=" << n
                          << " expected=" << expected
                          << " actual=" << actual << '\n';
                throw std::runtime_error(
                    "prefill FFN down mismatch");
            }
        }
    }

    if (!trace_path.empty()) {
        write_trace_csv(trace_path.string(), trace);
    }

    return {
        std::move(output),
        {},
        {},
        gate_up_cycles + swiglu_cycles + down_cycles};
}

#ifndef FTLPU_SMOLLM2_LAYER_PHASE_ONLY
int main()
try {
    auto input = std::vector<float>(kRows * kHidden);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t k = 0; k < kHidden; ++k) {
            input[activation_index(row, k, kHidden)] =
                ftlpu::Bf16::from_float(
                    activation_value(row, k)).to_float();
        }
    }
    auto system = ftlpu::TspSliceSystem {};
    const auto* trace = std::getenv("FTLPU_SCHEDULE_TRACE");
    const auto result = ftlpu::test::smollm2_layer::run_prefill_ffn(
        system, input, trace == nullptr ? std::filesystem::path {} : trace);
    std::cout
        << "SmolLM2 Block8 Dequant FFN passed: "
        << "X[128,576], gate/up[576,1536], "
        << "BF16 SwiGLU[128,1536], down[1536,576]; cycles="
        << result.cycles << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "SmolLM2 Block8 Dequant FFN failed: "
              << ex.what() << '\n';
    return 1;
}
#endif
