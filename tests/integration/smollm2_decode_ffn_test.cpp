#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "vxm_alu_program.hpp"
#include "smollm2_layer_phases.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
constexpr std::size_t kDecodeBlock =
    ftlpu::hw::kTileRows * ftlpu::hw::kLanesPerTile;
constexpr std::size_t kOutputGroup = ftlpu::hw::kMxmSupercellColumns;
constexpr std::size_t kDecodeOutputWidth = ftlpu::hw::kMxmColumns;
constexpr std::size_t kDecodeWaveStages =
    ftlpu::hw::kTileRows
    + ftlpu::hw::kMxmSupercellsPerPlane - 1;
constexpr std::size_t kGroupsPerAccumulatorRow =
    ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kGateUpGroups = kIntermediate / kOutputGroup;
constexpr std::size_t kGateUpWaves = kIntermediate / kDecodeOutputWidth;
constexpr std::size_t kGateUpAccumulatorRows =
    kGateUpGroups / kGroupsPerAccumulatorRow;
constexpr std::size_t kGateUpReductions =
    (kHidden + kDecodeBlock - 1) / kDecodeBlock;
constexpr std::size_t kDownGroupsPerMxm =
    (kHidden / 2) / kOutputGroup;
constexpr std::size_t kDownWavesPerMxm =
    (kHidden / 2) / kDecodeOutputWidth;
constexpr std::size_t kDownAccumulatorRows =
    kDownGroupsPerMxm / kGroupsPerAccumulatorRow;
constexpr std::size_t kDownReductions = kIntermediate / kDecodeBlock;

constexpr std::size_t kActivationAddressBase = 6000;
constexpr std::size_t kGateWeightAddressBase = 8000;
constexpr std::size_t kUpWeightAddressBase = 10000;
constexpr std::size_t kDownWeightAddressBase = 12000;
constexpr std::size_t kFinalOutputAddressBase = 16000;
constexpr std::size_t kDownAccumulatorAddressBase =
    ftlpu::hw::kMxmAccumulatorRows / 2;
constexpr std::array<std::size_t, 2> kActivationSlices {50, 51};
constexpr std::array<std::size_t, 8> kSwigluSlices {
    36, 37, 38, 39, 40, 41, 42, 43};
constexpr std::array<std::size_t, 4> kFinalSlices {48, 49, 50, 51};
constexpr std::size_t kMxmToVxmLatency =
    ftlpu::hw::kMxmBoundaryStreamRegisterColumn + 1;
constexpr std::size_t kEastMxm =
    ftlpu::InstructionControlUnit::mxm_queue(
        ftlpu::Hemisphere::East, 0);
constexpr std::size_t kWestMxm =
    ftlpu::InstructionControlUnit::mxm_queue(
        ftlpu::Hemisphere::West, 0);

static_assert(kDecodeBlock == 32);
static_assert(kGateUpGroups == 192);
static_assert(kGateUpWaves == 48);
static_assert(kGateUpAccumulatorRows == 48);
static_assert(kDownGroupsPerMxm == 36);
static_assert(kDownAccumulatorRows == 9);

enum class Projection : std::size_t { Gate, Up };

struct TraceEvent {
    std::size_t start{0};
    std::size_t end{0};
    std::string resource;
    std::string detail;
};

float activation_value(std::size_t k)
{
    return static_cast<float>(static_cast<int>((k * 5 + 3) % 23) - 11)
        * 0.0625f;
}

float gate_up_weight_value(Projection projection, std::size_t k, std::size_t n)
{
    const auto p = static_cast<std::size_t>(projection);
    const auto raw = static_cast<int>(
        (k * (11 + p * 6) + n * (5 + p * 2) + p * 13) % 41) - 20;
    return static_cast<float>(raw)
        * (0.009f + static_cast<float>((n + p) % 7) * 0.001f);
}

float down_weight_value(std::size_t k, std::size_t n)
{
    const auto raw = static_cast<int>((k * 19 + n * 11 + 7) % 47) - 23;
    return static_cast<float>(raw)
        * (0.006f + static_cast<float>((n + 3) % 9) * 0.001f);
}

std::size_t east_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mxm_west_latency(std::size_t slice)
{
    return ftlpu::hw::kSystemStreamRegisterColumns - 1
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t vxm_east_write_latency(std::size_t slice)
{
    constexpr std::size_t kSwishPipeline = 6;
    return kSwishPipeline
        + slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mem_queue(ftlpu::Hemisphere hemisphere, std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(hemisphere, slice);
}

class Schedule {
public:
    explicit Schedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(
        ftlpu::Hemisphere hemisphere,
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        const auto queue = mem_queue(hemisphere, slice);
        pad(mem_[queue], cycle, [&](std::size_t count) {
            icu_.enqueue_mem_nop(queue, count);
        });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
    }

    void load_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t buffer)
    {
        pad(mxm_load_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_load_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::DecodeLoadActivation(
                buffer,
                0,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmDecodeLayout::Native4x4));
        advance(mxm_load_[mxm], cycle + 1);
    }

    void decode_compute_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t buffer,
        float scale,
        std::size_t accumulator_address,
        std::size_t accumulator_column)
    {
        pad(mxm_dequant_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_dequant_nop(mxm, count);
        });
        icu_.enqueue_mxm_dequant(
            mxm, ftlpu::MxmDequantInstruction::Scale(scale));
        advance(mxm_dequant_[mxm], cycle + 1);

        pad(mxm_compute_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_compute_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::DecodeStreamCompute(
                buffer,
                0,
                ftlpu::MxmDataFormat::BFloat16,
                accumulator_address,
                accumulator_column,
                ftlpu::MxmAccumulatorDestination::Sram,
                false,
                ftlpu::MxmDecodeLayout::Native4x4));
        advance(mxm_compute_[mxm], cycle + 1);
    }

    void accumulator_read_at(
        std::size_t mxm,
        std::size_t cycle,
        std::size_t address,
        std::size_t stream_base,
        bool clear)
    {
        pad(mxm_compute_[mxm], cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_compute_nop(mxm, count);
        });
        icu_.enqueue_mxm(
            mxm,
            ftlpu::MxmControlInstruction::AccumulatorRead(
                address, stream_base, clear));
        advance(mxm_compute_[mxm], cycle + 1);
    }

    void swiglu_at(std::size_t cycle)
    {
        ftlpu::test::enqueue_swish(
            icu_, vxm_, cycle,
            ftlpu::test::SwishSpec {
                ftlpu::hw::kEastStreams,
                4,
                0,
                ftlpu::Hemisphere::East,
                ftlpu::Hemisphere::East,
                ftlpu::VxmCastTarget::BFloat16});
        ftlpu::test::enqueue_alu_at(
            icu_, vxm_, 10, cycle + 5,
            {
                ftlpu::VxmAluOpcode::Cast,
                ftlpu::VxmLaneOperand::Alu(8),
                ftlpu::VxmLaneOperand::Imm(0.0f),
                1.0f,
                0,
                ftlpu::VxmCastTarget::BFloat16,
                0,
                ftlpu::Hemisphere::East,
                ftlpu::Hemisphere::West,
            });
        end_ = std::max(end_, cycle + 6);
    }

    void trace(
        std::size_t start,
        std::size_t end,
        std::string resource,
        std::string detail)
    {
        trace_.push_back(TraceEvent {
            start, end, std::move(resource), std::move(detail)});
    }

    void write_trace_csv(const std::filesystem::path& path) const
    {
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        auto output = std::ofstream(path, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot open decode FFN schedule trace: " + path.string());
        }
        auto events = trace_;
        std::stable_sort(
            events.begin(), events.end(),
            [](const TraceEvent& lhs, const TraceEvent& rhs) {
                if (lhs.start != rhs.start) return lhs.start < rhs.start;
                if (lhs.end != rhs.end) return lhs.end < rhs.end;
                return lhs.resource < rhs.resource;
            });
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
                   << quote(event.resource) << ','
                   << quote(event.detail) << '\n';
        }
    }

    std::size_t end_cycle() const noexcept { return end_; }

private:
    template <typename Emit>
    static void pad(std::size_t cursor, std::size_t cycle, Emit emit)
    {
        if (cycle < cursor) {
            throw std::logic_error("offline decode FFN schedule overlaps an ICU queue");
        }
        emit(cycle - cursor);
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_ = std::max(end_, next);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::size_t, ftlpu::TspSliceSystem::kMxmCount> mxm_load_{};
    std::array<std::size_t, ftlpu::TspSliceSystem::kMxmCount> mxm_compute_{};
    std::array<std::size_t, ftlpu::TspSliceSystem::kMxmCount> mxm_dequant_{};
    std::array<std::size_t, ftlpu::VxmLane::kAluCount> vxm_{};
    std::vector<TraceEvent> trace_{};
    std::size_t end_{0};
};

struct QuantizedMatrix {
    std::vector<float> scales;
    std::vector<std::int8_t> values;
    std::vector<float> dequantized;
};

template <typename Generator>
QuantizedMatrix quantize_matrix(
    std::size_t rows,
    std::size_t columns,
    Generator generator)
{
    auto result = QuantizedMatrix {
        std::vector<float>(columns / kDecodeOutputWidth),
        std::vector<std::int8_t>(rows * columns),
        std::vector<float>(rows * columns)};
    for (std::size_t group = 0;
         group < columns / kDecodeOutputWidth;
         ++group) {
        float max_abs = 0.0f;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t lane = 0; lane < kDecodeOutputWidth; ++lane) {
                max_abs = std::max(
                    max_abs,
                    std::fabs(generator(
                        row, group * kDecodeOutputWidth + lane)));
            }
        }
        const auto scale = ftlpu::Bf16::from_float(max_abs / 127.0f).to_float();
        result.scales[group] = scale;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t lane = 0; lane < kDecodeOutputWidth; ++lane) {
                const auto column = group * kDecodeOutputWidth + lane;
                const auto quantized = std::clamp(
                    static_cast<int>(std::lround(generator(row, column) / scale)),
                    -127,
                    127);
                const auto index = row * columns + column;
                result.values[index] = static_cast<std::int8_t>(quantized);
                result.dequantized[index] = ftlpu::Bf16::from_float(
                    static_cast<float>(quantized) * scale).to_float();
            }
        }
    }
    return result;
}

void initialize_activation_source(
    ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    const std::vector<float>& values,
    std::size_t reduction_count)
{
    for (std::size_t reduction = 0; reduction < reduction_count; ++reduction) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto index = reduction * kDecodeBlock
                    + tile * ftlpu::hw::kLanesPerTile + lane;
                const auto value = index < values.size() ? values[index] : 0.0f;
                const auto bits = ftlpu::Bf16::from_float(value).bits();
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    system.initialize_mem_sram_lane_byte(
                        hemisphere,
                        kActivationSlices[byte],
                        tile,
                        kActivationAddressBase + reduction,
                        lane,
                        static_cast<std::uint8_t>((bits >> (byte * 8)) & 0xffu));
                }
            }
        }
    }
}

void initialize_weight_streams(
    ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    const QuantizedMatrix& matrix,
    std::size_t matrix_rows,
    std::size_t matrix_columns,
    std::size_t global_output_base,
    std::size_t output_waves,
    std::size_t address_base)
{
    const auto reductions = (matrix_rows + kDecodeBlock - 1) / kDecodeBlock;
    for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
        for (std::size_t wave = 0; wave < output_waves; ++wave) {
            const auto address = address_base + reduction * output_waves + wave;
            for (std::size_t stream = 0; stream < ftlpu::hw::kEastStreams; ++stream) {
                const auto column = stream / kOutputGroup;
                const auto output_lane = stream % kOutputGroup;
                for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                    const auto output = global_output_base
                        + wave * kDecodeOutputWidth
                        + column * kOutputGroup + output_lane;
                    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                        const auto row = reduction * kDecodeBlock
                            + tile * ftlpu::hw::kLanesPerTile + lane;
                        const auto value = row < matrix_rows
                                && output < matrix_columns
                            ? matrix.values[row * matrix_columns + output]
                            : std::int8_t {0};
                        system.initialize_mem_sram_lane_byte(
                            hemisphere,
                            stream,
                            tile,
                            address,
                            lane,
                            static_cast<std::uint8_t>(value));
                    }
                }
            }
        }
    }
}

void schedule_activation_load(
    Schedule& schedule,
    ftlpu::Hemisphere hemisphere,
    std::size_t mxm,
    std::size_t cycle,
    std::size_t reduction,
    std::size_t buffer)
{
    for (std::size_t stream = 0; stream < kActivationSlices.size(); ++stream) {
        const auto slice = kActivationSlices[stream];
        schedule.mem_at(
            hemisphere,
            slice,
            cycle - east_latency(slice),
            ftlpu::MemInstruction::Read(
                kActivationAddressBase + reduction,
                ftlpu::StreamId::East(stream)));
    }
    schedule.load_at(mxm, cycle, buffer);
}

void schedule_weight_waves(
    Schedule& schedule,
    ftlpu::Hemisphere hemisphere,
    std::size_t mxm,
    std::size_t compute_start,
    std::size_t reduction,
    std::size_t output_waves,
    std::size_t address_base,
    const std::vector<float>& scales,
    std::size_t global_wave_base,
    std::size_t accumulator_address_base,
    std::size_t buffer)
{
    for (std::size_t column = 0;
         column < ftlpu::hw::kMxmSupercellsPerPlane;
         ++column) {
        for (std::size_t wave = 0; wave < output_waves; ++wave) {
            const auto boundary_cycle = compute_start + wave + column;
            for (std::size_t output_lane = 0;
                 output_lane < kOutputGroup;
                 ++output_lane) {
                const auto stream = column * kOutputGroup + output_lane;
                schedule.mem_at(
                    hemisphere,
                    stream,
                    boundary_cycle - east_latency(stream),
                    ftlpu::MemInstruction::Read(
                        address_base + reduction * output_waves + wave,
                        ftlpu::StreamId::East(stream)));
            }
        }
    }
    for (std::size_t wave = 0; wave < output_waves; ++wave) {
        schedule.decode_compute_at(
            mxm,
            compute_start + wave,
            buffer,
            scales[global_wave_base + wave],
            accumulator_address_base + wave,
            0);
    }
}

void run(ftlpu::TspSliceSystem& system, std::size_t cycles)
{
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        try {
            system.tick({});
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "decode FFN cycle " + std::to_string(cycle)
                + ": " + error.what());
        }
    }
}

bool close_enough(float actual, float expected)
{
    const auto tolerance = 2.0e-5f * std::max(1.0f, std::fabs(expected));
    return std::fabs(actual - expected) <= tolerance
        || (std::isnan(actual) && std::isnan(expected));
}

float read_final(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    std::size_t local_output)
{
    const auto address = local_output / ftlpu::hw::kMxmColumns;
    const auto in_row = local_output % ftlpu::hw::kMxmColumns;
    const auto tile = in_row / ftlpu::hw::kLanesPerTile;
    const auto lane = in_row % ftlpu::hw::kLanesPerTile;
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(
            system.read_mem_sram_lane_byte(
                hemisphere,
                kFinalSlices[byte],
                tile,
                kFinalOutputAddressBase + address,
                lane)) << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

float read_swiglu(
    const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere,
    std::size_t index)
{
    const auto vector = index / ftlpu::hw::kMxmColumns;
    const auto in_vector = index % ftlpu::hw::kMxmColumns;
    const auto column = vector % ftlpu::hw::kMxmSupercellsPerPlane;
    const auto address = vector / ftlpu::hw::kMxmSupercellsPerPlane;
    const auto tile = in_vector / ftlpu::hw::kLanesPerTile;
    const auto lane = in_vector % ftlpu::hw::kLanesPerTile;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere,
        kSwigluSlices[column * 2],
        tile,
        address,
        lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere,
        kSwigluSlices[column * 2 + 1],
        tile,
        address,
        lane);
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

} // namespace

ftlpu::test::smollm2_layer::PhaseResult
ftlpu::test::smollm2_layer::run_decode_ffn(
    ftlpu::TspSliceSystem& system,
    const std::vector<float>& activation,
    const std::filesystem::path& trace_path)
{
    if (activation.size() != kHidden) {
        throw std::invalid_argument("decode FFN expects X[1,576]");
    }

    const auto gate = quantize_matrix(
        kHidden,
        kIntermediate,
        [](std::size_t k, std::size_t n) {
            return gate_up_weight_value(Projection::Gate, k, n);
        });
    const auto up = quantize_matrix(
        kHidden,
        kIntermediate,
        [](std::size_t k, std::size_t n) {
            return gate_up_weight_value(Projection::Up, k, n);
        });
    const auto down = quantize_matrix(
        kIntermediate,
        kHidden,
        [](std::size_t k, std::size_t n) {
            return down_weight_value(k, n);
        });

    system.reset_execution_state();
    initialize_activation_source(
        system, ftlpu::Hemisphere::East, activation, kGateUpReductions);
    initialize_activation_source(
        system, ftlpu::Hemisphere::West, activation, kGateUpReductions);
    initialize_weight_streams(
        system,
        ftlpu::Hemisphere::East,
        gate,
        kHidden,
        kIntermediate,
        0,
        kGateUpWaves,
        kGateWeightAddressBase);
    initialize_weight_streams(
        system,
        ftlpu::Hemisphere::West,
        up,
        kHidden,
        kIntermediate,
        0,
        kGateUpWaves,
        kUpWeightAddressBase);
    initialize_weight_streams(
        system,
        ftlpu::Hemisphere::East,
        down,
        kIntermediate,
        kHidden,
        0,
        kDownWavesPerMxm,
        kDownWeightAddressBase);
    initialize_weight_streams(
        system,
        ftlpu::Hemisphere::West,
        down,
        kIntermediate,
        kHidden,
        kHidden / 2,
        kDownWavesPerMxm,
        kDownWeightAddressBase);

    auto schedule = Schedule(system.icu());
    auto phase_cycle = std::size_t {30};
    for (std::size_t reduction = 0; reduction < kGateUpReductions; ++reduction) {
        const auto buffer = reduction % 2;
        schedule_activation_load(
            schedule, ftlpu::Hemisphere::East, kEastMxm,
            phase_cycle, reduction, buffer);
        schedule_activation_load(
            schedule, ftlpu::Hemisphere::West, kWestMxm,
            phase_cycle, reduction, buffer);
        schedule.trace(
            phase_cycle, phase_cycle + 1,
            "MXM.E0.ActivationLoad",
            "gate activation block=" + std::to_string(reduction));
        schedule.trace(
            phase_cycle, phase_cycle + 1,
            "MXM.W0.ActivationLoad",
            "up activation block=" + std::to_string(reduction));
        const auto compute_start = phase_cycle + ftlpu::hw::kTileRows;
        schedule_weight_waves(
            schedule,
            ftlpu::Hemisphere::East,
            kEastMxm,
            compute_start,
            reduction,
            kGateUpWaves,
            kGateWeightAddressBase,
            gate.scales,
            0,
            0,
            buffer);
        schedule_weight_waves(
            schedule,
            ftlpu::Hemisphere::West,
            kWestMxm,
            compute_start,
            reduction,
            kGateUpWaves,
            kUpWeightAddressBase,
            up.scales,
            0,
            0,
            buffer);
        schedule.trace(
            compute_start - east_latency(0),
            compute_start + kGateUpWaves
                + ftlpu::hw::kMxmSupercellsPerPlane - 1
                - east_latency(ftlpu::hw::kEastStreams - 1),
            "MEM.E.Read",
            "gate streamed INT8 weights reduction="
                + std::to_string(reduction));
        schedule.trace(
            compute_start - east_latency(0),
            compute_start + kGateUpWaves
                + ftlpu::hw::kMxmSupercellsPerPlane - 1
                - east_latency(ftlpu::hw::kEastStreams - 1),
            "MEM.W.Read",
            "up streamed INT8 weights reduction="
                + std::to_string(reduction));
        schedule.trace(
            compute_start, compute_start + kGateUpWaves,
            "MXM.E0.Compute",
            "gate decode waves reduction=" + std::to_string(reduction)
                + " dst=sram");
        schedule.trace(
            compute_start, compute_start + kGateUpWaves,
            "MXM.W0.Compute",
            "up decode waves reduction=" + std::to_string(reduction)
                + " dst=sram");
        phase_cycle = compute_start + kGateUpWaves + kDecodeWaveStages + 4;
    }

    const auto swiglu_read_start = phase_cycle + 4;
    for (std::size_t address = 0;
         address < kGateUpAccumulatorRows;
         ++address) {
        const auto read_cycle = swiglu_read_start + address;
        const auto vxm_cycle = read_cycle + kMxmToVxmLatency;
        schedule.accumulator_read_at(kEastMxm, read_cycle, address, 0, false);
        schedule.accumulator_read_at(kWestMxm, read_cycle - 1, address, 4, false);
        schedule.swiglu_at(vxm_cycle);
        const auto column = address % ftlpu::hw::kMxmSupercellsPerPlane;
        const auto sram_address = address / ftlpu::hw::kMxmSupercellsPerPlane;
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kSwigluSlices[column * 2 + byte];
            const auto write_cycle = vxm_cycle + vxm_east_write_latency(slice);
            schedule.mem_at(
                ftlpu::Hemisphere::East,
                slice,
                write_cycle,
                ftlpu::MemInstruction::Write(
                    sram_address, ftlpu::StreamId::East(byte)));
            schedule.mem_at(
                ftlpu::Hemisphere::West,
                slice,
                write_cycle,
                ftlpu::MemInstruction::Write(
                    sram_address, ftlpu::StreamId::East(byte)));
        }
    }
    const auto first_swiglu_cycle =
        swiglu_read_start + kMxmToVxmLatency;
    schedule.trace(
        swiglu_read_start,
        swiglu_read_start + kGateUpAccumulatorRows,
        "MXM.E0.AccumulatorRead",
        "gate FP32 dst=stream retain");
    schedule.trace(
        swiglu_read_start - 1,
        swiglu_read_start - 1 + kGateUpAccumulatorRows,
        "MXM.W0.AccumulatorRead",
        "up FP32 dst=stream retain; one-cycle bridge skew");
    schedule.trace(
        first_swiglu_cycle,
        first_swiglu_cycle + kGateUpAccumulatorRows + 5,
        "VXM.SwiGLU",
        "48 vector waves; gate W0..W3 + bridged up E4..E7");
    schedule.trace(
        first_swiglu_cycle + vxm_east_write_latency(kSwigluSlices.front()),
        first_swiglu_cycle + kGateUpAccumulatorRows
            + vxm_east_write_latency(kSwigluSlices.back()),
        "MEM.E.Write",
        "BF16 SwiGLU packed for down activation");
    schedule.trace(
        first_swiglu_cycle + vxm_east_write_latency(kSwigluSlices.front()),
        first_swiglu_cycle + kGateUpAccumulatorRows
            + vxm_east_write_latency(kSwigluSlices.back()),
        "MEM.W.Write",
        "BF16 SwiGLU duplicate for down activation");

    const auto down_start = swiglu_read_start
        + kGateUpAccumulatorRows + kMxmToVxmLatency + 32;
    phase_cycle = down_start;
    for (std::size_t reduction = 0; reduction < kDownReductions; ++reduction) {
        const auto buffer = reduction % 2;
        const auto swiglu_vector = reduction;
        const auto swiglu_slice_pair =
            swiglu_vector % ftlpu::hw::kMxmSupercellsPerPlane;
        const auto swiglu_address =
            swiglu_vector / ftlpu::hw::kMxmSupercellsPerPlane;
        for (std::size_t stream = 0; stream < 2; ++stream) {
            const auto slice =
                kSwigluSlices[swiglu_slice_pair * 2 + stream];
            for (std::size_t hemisphere_index = 0;
                 hemisphere_index < ftlpu::hw::kHemispheres;
                 ++hemisphere_index) {
                schedule.mem_at(
                    static_cast<ftlpu::Hemisphere>(hemisphere_index),
                    slice,
                    phase_cycle - east_latency(slice),
                    ftlpu::MemInstruction::Read(
                        swiglu_address, ftlpu::StreamId::East(stream)));
            }
        }
        schedule.load_at(kEastMxm, phase_cycle, buffer);
        schedule.load_at(kWestMxm, phase_cycle, buffer);
        schedule.trace(
            phase_cycle, phase_cycle + 1,
            "MXM.E0.ActivationLoad",
            "down SwiGLU block=" + std::to_string(reduction));
        schedule.trace(
            phase_cycle, phase_cycle + 1,
            "MXM.W0.ActivationLoad",
            "down SwiGLU block=" + std::to_string(reduction));
        const auto compute_start = phase_cycle + ftlpu::hw::kTileRows;
        schedule_weight_waves(
            schedule,
            ftlpu::Hemisphere::East,
            kEastMxm,
            compute_start,
            reduction,
            kDownWavesPerMxm,
            kDownWeightAddressBase,
            down.scales,
            0,
            kDownAccumulatorAddressBase,
            buffer);
        schedule_weight_waves(
            schedule,
            ftlpu::Hemisphere::West,
            kWestMxm,
            compute_start,
            reduction,
            kDownWavesPerMxm,
            kDownWeightAddressBase,
            down.scales,
            kDownWavesPerMxm,
            kDownAccumulatorAddressBase,
            buffer);
        schedule.trace(
            compute_start - east_latency(0),
            compute_start + kDownWavesPerMxm
                + ftlpu::hw::kMxmSupercellsPerPlane - 1
                - east_latency(ftlpu::hw::kEastStreams - 1),
            "MEM.E.Read",
            "down weights columns 0..287 reduction="
                + std::to_string(reduction));
        schedule.trace(
            compute_start - east_latency(0),
            compute_start + kDownWavesPerMxm
                + ftlpu::hw::kMxmSupercellsPerPlane - 1
                - east_latency(ftlpu::hw::kEastStreams - 1),
            "MEM.W.Read",
            "down weights columns 288..575 reduction="
                + std::to_string(reduction));
        schedule.trace(
            compute_start, compute_start + kDownWavesPerMxm,
            "MXM.E0.Compute",
            "down columns 0..287 reduction=" + std::to_string(reduction)
                + " dst=sram");
        schedule.trace(
            compute_start, compute_start + kDownWavesPerMxm,
            "MXM.W0.Compute",
            "down columns 288..575 reduction=" + std::to_string(reduction)
                + " dst=sram");
        phase_cycle = compute_start + kDownWavesPerMxm
            + kDecodeWaveStages + 4;
    }

    const auto output_read_start = phase_cycle + 4;
    for (std::size_t address = 0; address < kDownAccumulatorRows; ++address) {
        const auto read_cycle = output_read_start + address;
        schedule.accumulator_read_at(
            kEastMxm, read_cycle, kDownAccumulatorAddressBase + address, 0, true);
        schedule.accumulator_read_at(
            kWestMxm, read_cycle, kDownAccumulatorAddressBase + address, 0, true);
        for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
            const auto slice = kFinalSlices[byte];
            const auto write_cycle = read_cycle + mxm_west_latency(slice);
            schedule.mem_at(
                ftlpu::Hemisphere::East,
                slice,
                write_cycle,
                ftlpu::MemInstruction::Write(
                    kFinalOutputAddressBase + address,
                    ftlpu::StreamId::West(byte)));
            schedule.mem_at(
                ftlpu::Hemisphere::West,
                slice,
                write_cycle,
                ftlpu::MemInstruction::Write(
                    kFinalOutputAddressBase + address,
                    ftlpu::StreamId::West(byte)));
        }
    }

    schedule.trace(
        output_read_start,
        output_read_start + kDownAccumulatorRows,
        "MXM.E0.AccumulatorRead",
        "down columns 0..287 dst=stream+clear");
    schedule.trace(
        output_read_start,
        output_read_start + kDownAccumulatorRows,
        "MXM.W0.AccumulatorRead",
        "down columns 288..575 dst=stream+clear");
    schedule.trace(
        output_read_start + mxm_west_latency(kFinalSlices.front()),
        output_read_start + kDownAccumulatorRows
            + mxm_west_latency(kFinalSlices.back()),
        "MEM.E.Write",
        "final FP32 down output columns 0..287");
    schedule.trace(
        output_read_start + mxm_west_latency(kFinalSlices.front()),
        output_read_start + kDownAccumulatorRows
            + mxm_west_latency(kFinalSlices.back()),
        "MEM.W.Write",
        "final FP32 down output columns 288..575");

    if (!trace_path.empty()) {
        schedule.write_trace_csv(trace_path);
    }
    run(system, schedule.end_cycle() + 12);

    auto projected_gate = std::vector<float>(kIntermediate);
    auto projected_up = std::vector<float>(kIntermediate);
    auto swiglu = std::vector<float>(kIntermediate);
    for (std::size_t n = 0; n < kIntermediate; ++n) {
        for (std::size_t k = 0; k < kHidden; ++k) {
            projected_gate[n] += activation[k]
                * gate.dequantized[k * kIntermediate + n];
            projected_up[n] += activation[k]
                * up.dequantized[k * kIntermediate + n];
        }
        const auto accumulator_address =
            n / ftlpu::hw::kMxmColumns;
        const auto accumulator_column =
            n % ftlpu::hw::kMxmColumns;
        const auto actual_gate = system.mxm_unit(kEastMxm).accumulator().value(
            accumulator_address, accumulator_column);
        const auto actual_up = system.mxm_unit(kWestMxm).accumulator().value(
            accumulator_address, accumulator_column);
        if (!close_enough(actual_gate, projected_gate[n])
            || !close_enough(actual_up, projected_up[n])) {
            std::cerr << "decode FFN gate/up mismatch at " << n
                      << " gate=" << actual_gate << '/' << projected_gate[n]
                      << " up=" << actual_up << '/' << projected_up[n] << '\n';
            throw std::runtime_error("decode FFN hardware mismatch");
        }
        swiglu[n] = ftlpu::Bf16::from_float(
            (actual_gate * actual_up)
            / (1.0f + std::exp(-actual_gate))).to_float();
        const auto actual_swiglu = read_swiglu(
            system, ftlpu::Hemisphere::East, n);
        if (actual_swiglu != swiglu[n]) {
            std::cerr << "decode FFN SwiGLU mismatch at " << n
                      << " actual=" << actual_swiglu
                      << " expected=" << swiglu[n]
                      << " gate=" << actual_gate
                      << " up=" << actual_up << '\n';
            throw std::runtime_error("decode FFN hardware mismatch");
        }
    }

    auto output = std::vector<float>(kHidden);
    for (std::size_t n = 0; n < kHidden; ++n) {
        float expected = 0.0f;
        for (std::size_t k = 0; k < kIntermediate; ++k) {
            expected += swiglu[k]
                * down.dequantized[k * kHidden + n];
        }
        const auto hemisphere = n < kHidden / 2
            ? ftlpu::Hemisphere::East
            : ftlpu::Hemisphere::West;
        const auto local_output = n % (kHidden / 2);
        output[n] = read_final(system, hemisphere, local_output);
        if (!close_enough(output[n], expected)) {
            throw std::runtime_error(
                "decode FFN down mismatch at output=" + std::to_string(n));
        }
    }
    return {std::move(output), {}, {}, system.cycle()};
}

#ifndef FTLPU_SMOLLM2_LAYER_PHASE_ONLY
int main() try
{
    auto activation = std::vector<float>(kHidden);
    for (std::size_t k = 0; k < kHidden; ++k) {
        activation[k] = ftlpu::Bf16::from_float(activation_value(k)).to_float();
    }
    auto system = ftlpu::TspSliceSystem {};
    const auto* trace_env = std::getenv("FTLPU_SCHEDULE_TRACE");
    const auto result = ftlpu::test::smollm2_layer::run_decode_ffn(
        system,
        activation,
        trace_env == nullptr ? std::filesystem::path {} : trace_env);
    std::cout
        << "SmolLM2 decode FFN passed: X[1,576], gate/up[576,1536], "
        << "BF16 SwiGLU[1,1536], down[1536,576]; cycles="
        << result.cycles << '\n';
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "SmolLM2 decode FFN failed: " << error.what() << '\n';
    return 1;
}
#endif
