#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::size_t kK = 128;
constexpr std::size_t kN = 32;
constexpr std::size_t kActivationAddress = 32;
constexpr std::size_t kWeightAddress = 64;
constexpr std::size_t kOutputAddress = 256;
constexpr float kScale = 0.125f;
constexpr std::array<std::size_t, 8> kLinearActivationSlices {
    44, 45, 46, 47, 48, 49, 50, 51};
constexpr std::array<std::size_t, 2> kNativeActivationSlices {50, 51};
constexpr std::array<std::size_t, 2> kOutputSlices {40, 41};
constexpr std::size_t kLinearStages =
    ftlpu::hw::kTileRows * ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kNativeStages =
    ftlpu::hw::kTileRows + ftlpu::hw::kMxmSupercellsPerPlane - 1;
constexpr std::size_t kNativeBlock =
    ftlpu::hw::kTileRows * ftlpu::hw::kLanesPerTile;

float activation_value(std::size_t k)
{
    return 0.75f
        + static_cast<float>(static_cast<int>((k * 5 + 3) % 17) - 8)
            * 0.125f;
}

std::int8_t weight_value(std::size_t k, std::size_t n)
{
    return static_cast<std::int8_t>(
        static_cast<int>((k * 13 + n * 7 + 5) % 31) - 15);
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
        if (cycle < compute_) throw std::logic_error("overlapping MXM compute queue");
        icu_.enqueue_mxm_compute_nop(0, cycle - compute_);
        icu_.enqueue_mxm(0, instruction);
        compute_ = cycle + 1;
        end_ = std::max(end_, compute_);
    }

    void dequant_at(std::size_t cycle)
    {
        if (cycle < dequant_) throw std::logic_error("overlapping MXM dequant queue");
        icu_.enqueue_mxm_dequant_nop(0, cycle - dequant_);
        icu_.enqueue_mxm_dequant(
            0, ftlpu::MxmDequantInstruction::Scale(kScale));
        dequant_ = cycle + 1;
        end_ = std::max(end_, dequant_);
    }

    std::size_t end_cycle() const noexcept { return end_; }

private:
    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::size_t load_{0};
    std::size_t compute_{0};
    std::size_t dequant_{0};
    std::size_t end_{0};
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

void initialize_linear(ftlpu::TspSliceSystem& system)
{
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t tile = 0; tile < 4; ++tile) {
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const auto k = (column * 4 + tile) * 8 + lane;
                write_bf16(
                    system,
                    {kLinearActivationSlices[column * 2],
                     kLinearActivationSlices[column * 2 + 1]},
                    tile, kActivationAddress, lane, activation_value(k));
                for (std::size_t wave = 0; wave < 4; ++wave) {
                    for (std::size_t output_lane = 0;
                         output_lane < 8;
                         ++output_lane) {
                        const auto stream = column * 8 + output_lane;
                        system.initialize_mem_sram_lane_byte(
                            stream, tile, kWeightAddress + wave, lane,
                            static_cast<std::uint8_t>(
                                weight_value(k, wave * 8 + output_lane)));
                    }
                }
            }
        }
    }
}

void initialize_native(ftlpu::TspSliceSystem& system)
{
    for (std::size_t reduction = 0; reduction < kK / kNativeBlock; ++reduction) {
        for (std::size_t tile = 0; tile < 4; ++tile) {
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const auto k = reduction * kNativeBlock + tile * 8 + lane;
                write_bf16(
                    system, kNativeActivationSlices, tile,
                    kActivationAddress + reduction, lane, activation_value(k));
                for (std::size_t stream = 0; stream < 32; ++stream) {
                    system.initialize_mem_sram_lane_byte(
                        stream, tile, kWeightAddress + reduction, lane,
                        static_cast<std::uint8_t>(weight_value(k, stream)));
                }
            }
        }
    }
}

std::size_t program_linear(Schedule& schedule)
{
    constexpr auto activation_cycle = std::size_t {20};
    constexpr auto compute_start = std::size_t {30};
    for (std::size_t stream = 0; stream < 8; ++stream) {
        const auto slice = kLinearActivationSlices[stream];
        schedule.mem_at(
            slice, activation_cycle - east_latency(slice),
            ftlpu::MemInstruction::Read(
                kActivationAddress, ftlpu::StreamId::East(stream)));
    }
    schedule.load_at(
        activation_cycle,
        ftlpu::MxmControlInstruction::DecodeLoadActivation());
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t wave = 0; wave < 4; ++wave) {
            const auto boundary = compute_start + wave + column * 4;
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const auto stream = column * 8 + lane;
                schedule.mem_at(
                    stream, boundary - east_latency(stream),
                    ftlpu::MemInstruction::Read(
                        kWeightAddress + wave,
                        ftlpu::StreamId::East(stream)));
            }
        }
    }
    for (std::size_t wave = 0; wave < 4; ++wave) {
        const auto cycle = compute_start + wave;
        schedule.dequant_at(cycle);
        schedule.compute_at(
            cycle,
            ftlpu::MxmControlInstruction::DecodeStreamCompute(
                0, 0, ftlpu::MxmDataFormat::BFloat16));
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kOutputSlices[byte];
            schedule.mem_at(
                slice, cycle + kLinearStages - 1 + west_latency(slice),
                ftlpu::MemInstruction::Write(
                    kOutputAddress + wave,
                    ftlpu::StreamId::West(byte)));
        }
    }
    return schedule.end_cycle() + 8;
}

std::size_t program_native(Schedule& schedule)
{
    auto phase = std::size_t {20};
    for (std::size_t reduction = 0; reduction < kK / kNativeBlock; ++reduction) {
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = kNativeActivationSlices[byte];
            schedule.mem_at(
                slice, phase - east_latency(slice),
                ftlpu::MemInstruction::Read(
                    kActivationAddress + reduction,
                    ftlpu::StreamId::East(byte)));
        }
        schedule.load_at(
            phase,
            ftlpu::MxmControlInstruction::DecodeLoadActivation(
                reduction % 2,
                0,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmDecodeLayout::Native4x4));
        const auto compute = phase + 4;
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const auto stream = column * 8 + lane;
                schedule.mem_at(
                    stream, compute + column - east_latency(stream),
                    ftlpu::MemInstruction::Read(
                        kWeightAddress + reduction,
                        ftlpu::StreamId::East(stream)));
            }
        }
        schedule.dequant_at(compute);
        const auto final = reduction + 1 == kK / kNativeBlock;
        schedule.compute_at(
            compute,
            ftlpu::MxmControlInstruction::DecodeStreamCompute(
                reduction % 2,
                0,
                ftlpu::MxmDataFormat::BFloat16,
                0,
                0,
                final ? ftlpu::MxmAccumulatorDestination::Stream
                      : ftlpu::MxmAccumulatorDestination::Sram,
                final,
                ftlpu::MxmDecodeLayout::Native4x4));
        if (final) {
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto slice = kOutputSlices[byte];
                schedule.mem_at(
                    slice, compute + kNativeStages - 1 + west_latency(slice),
                    ftlpu::MemInstruction::Write(
                        kOutputAddress,
                        ftlpu::StreamId::West(byte)));
            }
        }
        // Column 3 is the last east-stream user at compute+3. The next
        // activation starts at compute+4 while the current diagonal wave
        // drains through its final three stages.
        phase = compute + ftlpu::hw::kMxmSupercellsPerPlane;
    }
    return schedule.end_cycle() + 8;
}

std::vector<std::uint16_t> run(ftlpu::MxmDecodeLayout layout, std::size_t& cycles)
{
    auto system = ftlpu::TspSliceSystem {};
    auto schedule = Schedule(system.icu());
    if (layout == ftlpu::MxmDecodeLayout::Native4x4) {
        initialize_native(system);
        cycles = program_native(schedule);
    } else {
        initialize_linear(system);
        cycles = program_linear(schedule);
    }
    const auto name = layout == ftlpu::MxmDecodeLayout::Native4x4
        ? "native4x4"
        : "linear1x16";
    const auto log_dir = std::filesystem::path("logs")
        / "mxm_decode_layout_comparison_test";
    std::filesystem::create_directories(log_dir);
    auto mxm_log = std::ofstream(log_dir / (std::string(name) + "_mxm.log"));
    auto mem_log = std::ofstream(log_dir / (std::string(name) + "_mem.log"));
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        system.tick({.mem = &mem_log, .mxm = &mxm_log});
    }

    auto output = std::vector<std::uint16_t>(kN);
    for (std::size_t n = 0; n < kN; ++n) {
        const auto tile = layout == ftlpu::MxmDecodeLayout::Native4x4
            ? n / 8
            : ftlpu::hw::kTileRows - 1;
        const auto address = layout == ftlpu::MxmDecodeLayout::Native4x4
            ? kOutputAddress
            : kOutputAddress + n / 8;
        const auto lane = n % 8;
        const auto low = system.read_mem_sram_lane_byte(
            kOutputSlices[0], tile, address, lane);
        const auto high = system.read_mem_sram_lane_byte(
            kOutputSlices[1], tile, address, lane);
        output[n] = static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8);
    }
    return output;
}

std::uint16_t reference(std::size_t n)
{
    auto sum = 0.0f;
    for (std::size_t k = 0; k < kK; ++k) {
        sum += ftlpu::Bf16::from_float(activation_value(k)).to_float()
            * ftlpu::Bf16::from_float(
                  static_cast<float>(weight_value(k, n)) * kScale)
                  .to_float();
    }
    return ftlpu::Bf16::from_float(sum).bits();
}
} // namespace

int main()
try {
    auto linear_cycles = std::size_t {0};
    auto native_cycles = std::size_t {0};
    const auto linear = run(
        ftlpu::MxmDecodeLayout::Linear1x16, linear_cycles);
    const auto native = run(
        ftlpu::MxmDecodeLayout::Native4x4, native_cycles);
    for (std::size_t n = 0; n < kN; ++n) {
        const auto expected = reference(n);
        if (linear[n] != expected || native[n] != expected) {
            throw std::runtime_error(
                "decode layout mismatch at output " + std::to_string(n)
                + " linear=" + std::to_string(
                    ftlpu::Bf16::from_bits(linear[n]).to_float())
                + " native=" + std::to_string(
                    ftlpu::Bf16::from_bits(native[n]).to_float())
                + " expected=" + std::to_string(
                    ftlpu::Bf16::from_bits(expected).to_float()));
        }
    }
    std::cout << "MXM decode layout comparison passed: GEMV K=128 N=32, "
              << "Linear1x16 cycles=" << linear_cycles
              << ", Native4x4 cycles=" << native_cycles
              << ", outputs are bit-identical BF16\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "MXM decode layout comparison failed: "
              << error.what() << '\n';
    return 1;
}
