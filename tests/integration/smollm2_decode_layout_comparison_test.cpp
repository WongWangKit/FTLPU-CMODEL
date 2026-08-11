#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t kActivationAddressBase = 32;
constexpr std::size_t kWeightAddressBase = 1024;
constexpr std::size_t kOutputAddressBase = 4096;
constexpr float kScale = 0.125f;
constexpr std::array<std::size_t, 8> kLinearActivationSlices {
    44, 45, 46, 47, 48, 49, 50, 51};
constexpr std::array<std::size_t, 2> kNativeActivationSlices {50, 51};
constexpr std::array<std::size_t, 2> kOutputSlices {40, 41};
constexpr std::size_t kLinearBlock =
    ftlpu::hw::kTileRows * ftlpu::hw::kMxmSupercellsPerPlane
    * ftlpu::hw::kLanesPerTile;
constexpr std::size_t kNativeBlock =
    ftlpu::hw::kTileRows * ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLinearStages =
    ftlpu::hw::kTileRows * ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kNativeStages =
    ftlpu::hw::kTileRows + ftlpu::hw::kMxmSupercellsPerPlane - 1;

struct Workload {
    const char* name;
    std::size_t k;
    std::size_t n;
};

struct TraceEvent {
    std::size_t start{};
    std::size_t end{};
    std::string resource{};
    std::string detail{};
};

struct RunResult {
    std::size_t cycles{};
    std::vector<std::uint16_t> output{};
    std::vector<TraceEvent> trace{};
};

std::size_t ceil_div(std::size_t value, std::size_t divisor)
{
    return (value + divisor - 1) / divisor;
}

float activation_value(std::size_t k)
{
    return static_cast<float>(static_cast<int>((k * 5 + 3) % 17) - 8)
        * 0.125f;
}

std::int8_t weight_value(std::size_t k, std::size_t n)
{
    return static_cast<std::int8_t>(
        static_cast<int>((k * 13 + n * 7 + 5) % 15) - 7);
}

std::size_t east_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t west_latency(std::size_t slice)
{
    return ftlpu::hw::kSystemStreamRegisterColumns - 1
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

class Schedule {
public:
    explicit Schedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        const auto queue = ftlpu::InstructionControlUnit::mem_queue(
            ftlpu::Hemisphere::East, slice);
        if (cycle < mem_[queue]) throw std::logic_error("overlapping MEM queue");
        icu_.enqueue_mem_nop(queue, cycle - mem_[queue]);
        icu_.enqueue_mem(queue, instruction);
        mem_[queue] = cycle + 1;
        end_ = std::max(end_, mem_[queue]);
    }

    void load_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction)
    {
        if (cycle < load_) throw std::logic_error("overlapping MXM load queue");
        icu_.enqueue_mxm_load_nop(0, cycle - load_);
        icu_.enqueue_mxm(0, instruction);
        load_ = cycle + 1;
        end_ = std::max(end_, load_);
    }

    void compute_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction)
    {
        if (cycle < compute_) {
            throw std::logic_error("overlapping MXM compute queue");
        }
        icu_.enqueue_mxm_compute_nop(0, cycle - compute_);
        icu_.enqueue_mxm(0, instruction);
        compute_ = cycle + 1;
        end_ = std::max(end_, compute_);
    }

    void dequant_at(std::size_t cycle)
    {
        if (cycle < dequant_) {
            throw std::logic_error("overlapping MXM dequant queue");
        }
        icu_.enqueue_mxm_dequant_nop(0, cycle - dequant_);
        icu_.enqueue_mxm_dequant(
            0, ftlpu::MxmDequantInstruction::Scale(kScale));
        dequant_ = cycle + 1;
        end_ = std::max(end_, dequant_);
    }

    void trace(
        std::size_t start,
        std::size_t end,
        std::string resource,
        std::string detail)
    {
        trace_.push_back(
            {start, end, std::move(resource), std::move(detail)});
    }

    std::vector<TraceEvent> take_trace() { return std::move(trace_); }

    std::size_t end_cycle() const noexcept { return end_; }

private:
    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::size_t load_{0};
    std::size_t compute_{0};
    std::size_t dequant_{0};
    std::size_t end_{0};
    std::vector<TraceEvent> trace_{};
};

void write_bf16(
    ftlpu::TspSliceSystem& system,
    const std::array<std::size_t, 2>& slices,
    std::size_t tile,
    std::size_t address,
    std::size_t lane,
    float value)
{
    const auto bits = ftlpu::Bf16::from_float(value).bits();
    for (std::size_t byte = 0; byte < slices.size(); ++byte) {
        system.initialize_mem_sram_lane_byte(
            slices[byte], tile, address, lane,
            static_cast<std::uint8_t>((bits >> (byte * 8)) & 0xffu));
    }
}

void initialize_linear(ftlpu::TspSliceSystem& system, const Workload& workload)
{
    const auto reductions = ceil_div(workload.k, kLinearBlock);
    const auto waves = ceil_div(workload.n, std::size_t {8});
    for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t tile = 0; tile < 4; ++tile) {
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const auto k = reduction * kLinearBlock
                        + (column * 4 + tile) * 8 + lane;
                    write_bf16(
                        system,
                        {kLinearActivationSlices[column * 2],
                         kLinearActivationSlices[column * 2 + 1]},
                        tile, kActivationAddressBase + reduction, lane,
                        k < workload.k ? activation_value(k) : 0.0f);
                    for (std::size_t wave = 0; wave < waves; ++wave) {
                        for (std::size_t output_lane = 0;
                             output_lane < 8;
                             ++output_lane) {
                            const auto stream = column * 8 + output_lane;
                            const auto n = wave * 8 + output_lane;
                            const auto value = k < workload.k && n < workload.n
                                ? weight_value(k, n)
                                : std::int8_t {0};
                            system.initialize_mem_sram_lane_byte(
                                stream, tile,
                                kWeightAddressBase + reduction * waves + wave,
                                lane, static_cast<std::uint8_t>(value));
                        }
                    }
                }
            }
        }
    }
}

void initialize_native(ftlpu::TspSliceSystem& system, const Workload& workload)
{
    const auto reductions = ceil_div(workload.k, kNativeBlock);
    const auto waves = ceil_div(workload.n, std::size_t {32});
    for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
        for (std::size_t tile = 0; tile < 4; ++tile) {
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const auto k = reduction * kNativeBlock + tile * 8 + lane;
                write_bf16(
                    system, kNativeActivationSlices, tile,
                    kActivationAddressBase + reduction, lane,
                    k < workload.k ? activation_value(k) : 0.0f);
                for (std::size_t wave = 0; wave < waves; ++wave) {
                    for (std::size_t stream = 0; stream < 32; ++stream) {
                        const auto n = wave * 32 + stream;
                        const auto value = k < workload.k && n < workload.n
                            ? weight_value(k, n)
                            : std::int8_t {0};
                        system.initialize_mem_sram_lane_byte(
                            stream, tile,
                            kWeightAddressBase + reduction * waves + wave,
                            lane, static_cast<std::uint8_t>(value));
                    }
                }
            }
        }
    }
}

std::size_t program_linear(Schedule& schedule, const Workload& workload)
{
    const auto reductions = ceil_div(workload.k, kLinearBlock);
    const auto waves = ceil_div(workload.n, std::size_t {8});
    auto phase = std::size_t {20};
    for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
        auto activation_read_start = phase;
        for (std::size_t stream = 0; stream < 8; ++stream) {
            const auto slice = kLinearActivationSlices[stream];
            activation_read_start = std::min(
                activation_read_start, phase - east_latency(slice));
            schedule.mem_at(
                slice, phase - east_latency(slice),
                ftlpu::MemInstruction::Read(
                    kActivationAddressBase + reduction,
                    ftlpu::StreamId::East(stream)));
        }
        schedule.load_at(
            phase,
            ftlpu::MxmControlInstruction::DecodeLoadActivation(
                reduction % 2,
                0,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmDecodeLayout::Linear1x16));
        schedule.trace(
            activation_read_start, phase,
            "MEM.Activation.Read",
            "reduction=" + std::to_string(reduction)
                + " K=" + std::to_string(reduction * kLinearBlock)
                + ".." + std::to_string(
                    std::min(workload.k, (reduction + 1) * kLinearBlock) - 1));
        schedule.trace(
            phase, phase + 1,
            "MXM.Load",
            "activation buffer=" + std::to_string(reduction % 2));

        const auto compute_start = phase + 2;
        auto weight_read_start = compute_start;
        auto weight_read_end = compute_start;
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t wave = 0; wave < waves; ++wave) {
                const auto boundary = compute_start + wave + column * 4;
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const auto stream = column * 8 + lane;
                    const auto read_cycle = boundary - east_latency(stream);
                    weight_read_start = std::min(weight_read_start, read_cycle);
                    weight_read_end = std::max(weight_read_end, read_cycle + 1);
                    schedule.mem_at(
                        stream, read_cycle,
                        ftlpu::MemInstruction::Read(
                            kWeightAddressBase + reduction * waves + wave,
                            ftlpu::StreamId::East(stream)));
                }
            }
        }

        const auto final = reduction + 1 == reductions;
        auto output_write_start = std::numeric_limits<std::size_t>::max();
        auto output_write_end = std::size_t {0};
        for (std::size_t wave = 0; wave < waves; ++wave) {
            const auto cycle = compute_start + wave;
            schedule.dequant_at(cycle);
            schedule.compute_at(
                cycle,
                ftlpu::MxmControlInstruction::DecodeStreamCompute(
                    reduction % 2,
                    0,
                    ftlpu::MxmDataFormat::BFloat16,
                    wave / 4,
                    wave % 4,
                    final ? ftlpu::MxmAccumulatorDestination::Stream
                          : ftlpu::MxmAccumulatorDestination::Sram,
                    final,
                    ftlpu::MxmDecodeLayout::Linear1x16));
            if (final) {
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    const auto slice = kOutputSlices[byte];
                    const auto write_cycle = cycle + kLinearStages - 1
                        + west_latency(slice);
                    output_write_start = std::min(
                        output_write_start, write_cycle);
                    output_write_end = std::max(
                        output_write_end, write_cycle + 1);
                    schedule.mem_at(
                        slice,
                        write_cycle,
                        ftlpu::MemInstruction::Write(
                            kOutputAddressBase + wave,
                            ftlpu::StreamId::West(byte)));
                }
            }
        }
        schedule.trace(
            weight_read_start, weight_read_end,
            "MEM.Weight.Read",
            "reduction=" + std::to_string(reduction)
                + " waves=" + std::to_string(waves)
                + " columns stagger=4 cycles");
        schedule.trace(
            compute_start, compute_start + waves,
            "MXM.Compute",
            "reduction=" + std::to_string(reduction)
                + (final ? " dst=stream+clear" : " dst=sram"));
        schedule.trace(
            compute_start + waves,
            compute_start + waves + kLinearStages - 1,
            "MXM.Tail",
            "final launched wave drains through 16-cell chain");
        if (final) {
            schedule.trace(
                output_write_start, output_write_end,
                "MEM.Output.Write",
                "BF16 outputs=" + std::to_string(workload.n));
        }
        // E0..E7 are free one cycle after the final column-0 weight. The
        // alternating activation buffers let the next reduction overlap the
        // remaining 1x16 pipeline tail.
        phase = compute_start + waves;
    }
    return schedule.end_cycle() + 8;
}

std::size_t program_native(Schedule& schedule, const Workload& workload)
{
    const auto reductions = ceil_div(workload.k, kNativeBlock);
    const auto waves = ceil_div(workload.n, std::size_t {32});
    auto phase = std::size_t {20};
    for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
        auto activation_read_start = phase;
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kNativeActivationSlices[byte];
            activation_read_start = std::min(
                activation_read_start, phase - east_latency(slice));
            schedule.mem_at(
                slice, phase - east_latency(slice),
                ftlpu::MemInstruction::Read(
                    kActivationAddressBase + reduction,
                    ftlpu::StreamId::East(byte)));
        }
        schedule.load_at(
            phase,
            ftlpu::MxmControlInstruction::DecodeLoadActivation(
                reduction % 2,
                0,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmDecodeLayout::Native4x4));
        schedule.trace(
            activation_read_start, phase,
            "MEM.Activation.Read",
            "reduction=" + std::to_string(reduction)
                + " K=" + std::to_string(reduction * kNativeBlock)
                + ".." + std::to_string(
                    std::min(workload.k, (reduction + 1) * kNativeBlock) - 1));
        schedule.trace(
            phase, phase + 1,
            "MXM.Load",
            "activation buffer=" + std::to_string(reduction % 2));

        const auto compute_start = phase + 4;
        auto weight_read_start = compute_start;
        auto weight_read_end = compute_start;
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t wave = 0; wave < waves; ++wave) {
                const auto boundary = compute_start + wave + column;
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const auto stream = column * 8 + lane;
                    const auto read_cycle = boundary - east_latency(stream);
                    weight_read_start = std::min(weight_read_start, read_cycle);
                    weight_read_end = std::max(weight_read_end, read_cycle + 1);
                    schedule.mem_at(
                        stream, read_cycle,
                        ftlpu::MemInstruction::Read(
                            kWeightAddressBase + reduction * waves + wave,
                            ftlpu::StreamId::East(stream)));
                }
            }
        }

        const auto final = reduction + 1 == reductions;
        auto output_write_start = std::numeric_limits<std::size_t>::max();
        auto output_write_end = std::size_t {0};
        for (std::size_t wave = 0; wave < waves; ++wave) {
            const auto cycle = compute_start + wave;
            schedule.dequant_at(cycle);
            schedule.compute_at(
                cycle,
                ftlpu::MxmControlInstruction::DecodeStreamCompute(
                    reduction % 2,
                    0,
                    ftlpu::MxmDataFormat::BFloat16,
                    wave,
                    0,
                    final ? ftlpu::MxmAccumulatorDestination::Stream
                          : ftlpu::MxmAccumulatorDestination::Sram,
                    final,
                    ftlpu::MxmDecodeLayout::Native4x4));
            if (final) {
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    const auto slice = kOutputSlices[byte];
                    const auto write_cycle = cycle + kNativeStages - 1
                        + west_latency(slice);
                    output_write_start = std::min(
                        output_write_start, write_cycle);
                    output_write_end = std::max(
                        output_write_end, write_cycle + 1);
                    schedule.mem_at(
                        slice,
                        write_cycle,
                        ftlpu::MemInstruction::Write(
                            kOutputAddressBase + wave,
                            ftlpu::StreamId::West(byte)));
                }
            }
        }
        schedule.trace(
            weight_read_start, weight_read_end,
            "MEM.Weight.Read",
            "reduction=" + std::to_string(reduction)
                + " waves=" + std::to_string(waves)
                + " columns stagger=1 cycle");
        schedule.trace(
            compute_start, compute_start + waves,
            "MXM.Compute",
            "reduction=" + std::to_string(reduction)
                + (final ? " dst=stream+clear" : " dst=sram"));
        schedule.trace(
            compute_start + waves,
            compute_start + waves + kNativeStages - 1,
            "MXM.Tail",
            "final launched wave drains through 4x4 diagonal");
        if (final) {
            schedule.trace(
                output_write_start, output_write_end,
                "MEM.Output.Write",
                "BF16 outputs=" + std::to_string(workload.n));
        }
        // The next activation enters while the four diagonal columns of the
        // current reduction finish draining.
        phase = compute_start + waves;
    }
    return schedule.end_cycle() + 8;
}

RunResult run(ftlpu::MxmDecodeLayout layout, const Workload& workload)
{
    auto system = ftlpu::TspSliceSystem {};
    auto schedule = Schedule(system.icu());
    if (layout == ftlpu::MxmDecodeLayout::Native4x4) {
        initialize_native(system, workload);
    } else {
        initialize_linear(system, workload);
    }

    const auto cycles = layout == ftlpu::MxmDecodeLayout::Native4x4
        ? program_native(schedule, workload)
        : program_linear(schedule, workload);
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) system.tick({});

    auto output = std::vector<std::uint16_t>(workload.n);
    for (std::size_t n = 0; n < workload.n; ++n) {
        const auto tile = layout == ftlpu::MxmDecodeLayout::Native4x4
            ? (n % 32) / 8
            : ftlpu::hw::kTileRows - 1;
        const auto address = layout == ftlpu::MxmDecodeLayout::Native4x4
            ? kOutputAddressBase + n / 32
            : kOutputAddressBase + n / 8;
        const auto lane = n % 8;
        const auto low = system.read_mem_sram_lane_byte(
            kOutputSlices[0], tile, address, lane);
        const auto high = system.read_mem_sram_lane_byte(
            kOutputSlices[1], tile, address, lane);
        output[n] = static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8);
    }
    return {cycles, std::move(output), schedule.take_trace()};
}

std::uint16_t reference(const Workload& workload, std::size_t n)
{
    auto sum = 0.0f;
    for (std::size_t k = 0; k < workload.k; ++k) {
        sum += ftlpu::Bf16::from_float(activation_value(k)).to_float()
            * ftlpu::Bf16::from_float(
                  static_cast<float>(weight_value(k, n)) * kScale)
                  .to_float();
    }
    return ftlpu::Bf16::from_float(sum).bits();
}

void verify(
    const Workload& workload,
    const RunResult& linear,
    const RunResult& native)
{
    for (std::size_t n = 0; n < workload.n; ++n) {
        const auto expected = reference(workload, n);
        if (linear.output[n] != expected || native.output[n] != expected) {
            throw std::runtime_error(
                std::string(workload.name) + " mismatch at output "
                + std::to_string(n)
                + " linear=" + std::to_string(
                    ftlpu::Bf16::from_bits(linear.output[n]).to_float())
                + " native=" + std::to_string(
                    ftlpu::Bf16::from_bits(native.output[n]).to_float())
                + " expected=" + std::to_string(
                    ftlpu::Bf16::from_bits(expected).to_float()));
        }
    }
}

void append_trace(
    std::vector<TraceEvent>& destination,
    const std::vector<TraceEvent>& source,
    const std::string& prefix)
{
    for (const auto& event : source) {
        auto qualified = event;
        const auto separator = qualified.resource.find('.');
        qualified.resource.insert(separator + 1, prefix + '.');
        destination.push_back(std::move(qualified));
    }
}

void write_trace_csv(
    const std::filesystem::path& path,
    const std::vector<TraceEvent>& events)
{
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot open decode layout schedule trace: " + path.string());
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
               << quote(event.resource) << ','
               << quote(event.detail) << '\n';
    }
}
} // namespace

int main()
try {
    constexpr std::array workloads {
        Workload {"gate/up projection", 576, 1536},
        Workload {"down projection half", 1536, 288},
    };

    auto all_linear = std::size_t {0};
    auto all_native = std::size_t {0};
    auto hybrid = std::size_t {0};
    auto trace = std::vector<TraceEvent> {};
    for (std::size_t index = 0; index < workloads.size(); ++index) {
        const auto& workload = workloads[index];
        const auto linear = run(ftlpu::MxmDecodeLayout::Linear1x16, workload);
        const auto native = run(ftlpu::MxmDecodeLayout::Native4x4, workload);
        verify(workload, linear, native);
        const auto workload_name = index == 0 ? "GateUp" : "DownHalf";
        append_trace(
            trace, linear.trace,
            std::string(workload_name) + ".Linear1x16");
        append_trace(
            trace, native.trace,
            std::string(workload_name) + ".Native4x4");
        all_linear += linear.cycles;
        all_native += native.cycles;
        hybrid += index == 0 ? native.cycles : linear.cycles;
        std::cout << workload.name << " K=" << workload.k
                  << " N=" << workload.n
                  << ": Linear1x16=" << linear.cycles
                  << " cycles, Native4x4=" << native.cycles
                  << " cycles, winner="
                  << (linear.cycles <= native.cycles
                          ? "Linear1x16"
                          : "Native4x4")
                  << ", BF16 verified\n";
    }

    if (const auto* trace_path = std::getenv("FTLPU_SCHEDULE_TRACE")) {
        write_trace_csv(trace_path, trace);
        std::cout << "schedule trace=" << trace_path << '\n';
    }

    std::cout << "SmolLM2 FFN projection critical-path totals: all-linear="
              << all_linear << ", all-native=" << all_native
              << ", hybrid(native gate/up + linear down)=" << hybrid
              << " cycles\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "SmolLM2 decode layout comparison failed: "
              << error.what() << '\n';
    return 1;
}
