#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
constexpr std::size_t kActivationAddress = 32;
constexpr std::size_t kWeightAddress = 64;
constexpr std::size_t kOutputAddress = 96;
constexpr std::size_t kActivationCycle = 20;
constexpr std::size_t kComputeCycle = 30;
constexpr float kScale = 0.125f;
constexpr std::array<std::size_t, 2> kActivationSlices {50, 51};
constexpr std::array<std::size_t, 2> kOutputSlices {40, 41};

static_assert(ftlpu::hw::kMxmColumns <= ftlpu::hw::kEastStreams,
    "this streamed-weight fixture requires one stream per output column");
constexpr std::size_t kDecodeStages =
    ftlpu::hw::kTileRows
    + ftlpu::hw::kMxmIsColumns - 1;
constexpr std::size_t kDecodeK =
    ftlpu::hw::kTileRows * ftlpu::hw::kLanesPerTile;
constexpr std::size_t kDecodeN =
    ftlpu::hw::kMxmIsColumns * ftlpu::hw::kLanesPerTile;
constexpr std::size_t kOutputBlocks =
    2;
constexpr std::size_t kOutputWidth = kDecodeK;
constexpr std::size_t kWeightWaves = kOutputBlocks * kOutputWidth / kDecodeN;

std::size_t east_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups
        + ftlpu::hw::kC2cToSxmStreamRegisterColumns + 1
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t west_latency(std::size_t slice)
{
    return ftlpu::hw::kSystemStreamRegisterColumns - 1
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mem_queue(std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(
        ftlpu::Hemisphere::East, slice);
}

float activation_value(std::size_t k)
{
    return 0.75f
        + static_cast<float>(static_cast<int>((k * 5 + 3) % 17) - 8)
            * 0.125f;
}

std::int8_t quantized_weight(std::size_t k, std::size_t output)
{
    return static_cast<std::int8_t>(
        static_cast<int>((k * 13 + output * 7 + 5) % 31) - 15);
}

class Schedule {
public:
    explicit Schedule(ftlpu::InstructionControlUnit& icu) : icu_(icu) {}

    void mem_at(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        const auto queue = mem_queue(slice);
        auto& cursor = mem_[queue];
        if (cycle < cursor) throw std::logic_error("overlapping MEM queue");
        icu_.enqueue_mem_nop(queue, cycle - cursor);
        icu_.enqueue_mem(queue, instruction);
        cursor = cycle + 1;
        end_ = std::max(end_, cursor);
    }

    void load_at(std::size_t cycle, ftlpu::MxmControlInstruction instruction)
    {
        if (cycle < load_) throw std::logic_error("overlapping load queue");
        icu_.enqueue_mxm_load_nop(0, cycle - load_);
        icu_.enqueue_mxm(0, instruction);
        load_ = cycle + 1;
        end_ = std::max(end_, load_);
    }

    void compute_at(std::size_t cycle, ftlpu::MxmControlInstruction instruction)
    {
        if (cycle < compute_) throw std::logic_error("overlapping compute queue");
        icu_.enqueue_mxm_compute_nop(0, cycle-compute_);
        icu_.enqueue_mxm(0,instruction);
        compute_ = cycle + 1;
        end_ = std::max(end_, compute_);
    }

    void dequant_at(std::size_t cycle)
    {
        if (cycle < dequant_) throw std::logic_error("overlapping dequant queue");
        icu_.enqueue_mxm_dequant_nop(0, cycle - dequant_);
        icu_.enqueue_mxm_dequant(
            0, ftlpu::MxmDequantInstruction::Scale(kScale));
        dequant_ = cycle + 1;
        end_ = std::max(end_, dequant_);
    }

    std::size_t end_cycle() const noexcept { return end_; }

private:
    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_ {};
    std::size_t load_{0};
    std::size_t compute_{0};
    std::size_t dequant_{0};
    std::size_t end_{0};
};

void initialize(ftlpu::TspSliceSystem& system, bool vector)
{
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto k = tile * ftlpu::hw::kLanesPerTile + lane;
            const auto activation =
                ftlpu::Bf16::from_float(activation_value(k)).bits();
            for (std::size_t byte = 0; byte < kActivationSlices.size(); ++byte) {
                system.initialize_mem_sram_lane_byte(
                    kActivationSlices[byte],
                    tile,
                    kActivationAddress,
                    lane,
                    static_cast<std::uint8_t>(
                        (activation >> (byte * 8)) & 0xffu));
            }

            for (std::size_t output_block = 0;
                 output_block < kWeightWaves;
                 ++output_block) {
                for (std::size_t stream = 0;
                     stream < kDecodeN;
                     ++stream) {
                    const auto column =
                        stream / ftlpu::hw::kMxmSupercellColumns;
                    const auto output_lane =
                        stream % ftlpu::hw::kMxmSupercellColumns;
                    const auto weight_k =
                        tile * ftlpu::hw::kLanesPerTile + lane;
                    const auto output =
                        output_block * kDecodeN
                        + column * ftlpu::hw::kMxmSupercellColumns
                        + output_lane;
                    const auto quantized = quantized_weight(weight_k, output);
                    system.initialize_mem_sram_lane_byte(
                        vector ? column : stream,
                        tile,
                        kWeightAddress + (vector ? output_block * ftlpu::hw::kLanesPerTile + output_lane : output_block),
                        lane,
                        static_cast<std::uint8_t>(quantized));
                }
            }
        }
    }
}

ftlpu::Bf16 reference(std::size_t output)
{
    float sum = 0.0f;
    for (std::size_t stage = 0; stage < ftlpu::hw::kTileRows; ++stage) {
        float partial = 0.0f;
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto k = stage * ftlpu::hw::kLanesPerTile + lane;
            const auto activation =
                ftlpu::Bf16::from_float(activation_value(k)).to_float();
            const auto weight = ftlpu::Bf16::from_float(
                static_cast<float>(quantized_weight(k, output)) * kScale)
                                    .to_float();
            partial += activation * weight;
        }
        sum += partial;
    }
    return ftlpu::Bf16::from_float(sum);
}

void program(Schedule& schedule, bool vector, bool incomplete = false)
{
    const auto waves = incomplete ? 1 : kWeightWaves;
    for (std::size_t byte = 0; byte < kActivationSlices.size(); ++byte) {
        const auto slice = kActivationSlices[byte];
        schedule.mem_at(
            slice, kActivationCycle - east_latency(slice),
            ftlpu::MemInstruction::Read(
                kActivationAddress, ftlpu::StreamId::East(byte)));
    }
    schedule.load_at(
        kActivationCycle,
        ftlpu::MxmControlInstruction::IsLoadActivation(
            0,
            0,
            ftlpu::MxmDataFormat::BFloat16,
            vector ? ftlpu::MxmIsMode::Vector : ftlpu::MxmIsMode::Block));

    for (std::size_t column = 0;
         column < ftlpu::hw::kMxmIsColumns;
         ++column) {
        for (std::size_t output_block = 0;
             output_block < waves;
             ++output_block) {
            const auto cycle = kComputeCycle + output_block * (vector ? ftlpu::hw::kLanesPerTile : 1) + column;
            for (std::size_t output_lane = 0;
                 output_lane < ftlpu::hw::kMxmSupercellColumns;
                 ++output_lane) {
                const auto stream =
                    column * ftlpu::hw::kMxmSupercellColumns
                    + output_lane;
                schedule.mem_at(
                    vector ? column : stream,
                    cycle + (vector ? output_lane : 0) - east_latency(vector ? column : stream),
                    ftlpu::MemInstruction::Read(
                        kWeightAddress + (vector ? output_block * ftlpu::hw::kLanesPerTile + output_lane : output_block),
                        ftlpu::StreamId::East(vector ? column : stream)));
            }
        }
    }

    for (std::size_t output_block = 0;
         output_block < waves;
         ++output_block) {
        const auto cycle = kComputeCycle + output_block * (vector ? ftlpu::hw::kLanesPerTile : 1);
        schedule.dequant_at(cycle);
        schedule.compute_at(
            cycle,
            ftlpu::MxmControlInstruction::IsStreamCompute(
                0,
                0,
                ftlpu::MxmDataFormat::BFloat16,
                output_block * ftlpu::hw::kMxmIsColumns / ftlpu::hw::kTileRows,
                incomplete ? 1 : output_block * ftlpu::hw::kMxmIsColumns % ftlpu::hw::kTileRows,
                ftlpu::MxmAccumulatorDestination::Sram,
                false,
                vector ? ftlpu::MxmIsMode::Vector : ftlpu::MxmIsMode::Block));
    }

    // No SR output until every segment in each ACC row has been filled.
    const auto read_start = kComputeCycle
        + (waves - 1) * (vector ? ftlpu::hw::kLanesPerTile : 1)
        + kDecodeStages + (vector ? ftlpu::hw::kLanesPerTile - 1 : 0);
    for (std::size_t output_block = 0;
         output_block < (incomplete ? 1 : kOutputBlocks);
         ++output_block) {
        schedule.compute_at(read_start + output_block,
            ftlpu::MxmControlInstruction::AccumulatorRead(output_block, 0, true,
                ftlpu::MxmAccumulatorOutputFormat::BFloat16));
        for (std::size_t stream = 0;
             stream < kOutputSlices.size();
             ++stream) {
            const auto slice = kOutputSlices[stream];
            schedule.mem_at(
                slice,
                read_start + output_block + west_latency(slice),
                ftlpu::MemInstruction::Write(
                    kOutputAddress + output_block,
                    ftlpu::StreamId::West(stream)));
        }
    }
}

bool verify(const ftlpu::TspSliceSystem& system)
{
    for (std::size_t output = 0; output < kOutputWidth * kOutputBlocks; ++output) {
        const auto output_block = output / kOutputWidth;
        const auto output_tile =
            (output % kOutputWidth)
            / ftlpu::hw::kMxmSupercellColumns;
        const auto lane = output % ftlpu::hw::kMxmSupercellColumns;
        const auto low = system.read_mem_sram_lane_byte(
            kOutputSlices[0],
            output_tile,
            kOutputAddress + output_block,
            lane);
        const auto high = system.read_mem_sram_lane_byte(
            kOutputSlices[1],
            output_tile,
            kOutputAddress + output_block,
            lane);
        const auto actual = ftlpu::Bf16::from_bits(
            static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8));
        const auto expected = reference(output);
        if (actual.bits() != expected.bits()) {
            std::cerr << "decode output mismatch at " << output
                      << ": actual=" << actual.to_float()
                      << " expected=" << expected.to_float() << '\n';
            return false;
        }
    }
    return true;
}
} // namespace

int main()
try {
    const auto log_dir =
        std::filesystem::path("logs")
        / "mxm_decode_streamed_weight_test";
    std::filesystem::create_directories(log_dir);
    auto icu_log = std::ofstream(log_dir / "icu.log", std::ios::trunc);
    auto mem_log = std::ofstream(log_dir / "mem.log", std::ios::trunc);
    auto mxm_log = std::ofstream(log_dir / "mxm.log", std::ios::trunc);
    if (!icu_log || !mem_log || !mxm_log) {
        throw std::runtime_error("cannot open decode streamed-weight logs");
    }

    for (bool vector : {false, true}) {
    auto system = ftlpu::TspSliceSystem {};
    initialize(system, vector);
    auto schedule = Schedule(system.icu());
    program(schedule, vector);
    for (std::size_t cycle = 0; cycle < schedule.end_cycle() + 8; ++cycle) {
        system.tick({
            .icu = &icu_log,
            .mem = &mem_log,
            .mxm = &mxm_log,
            .mxm_log_tile = 0,
        });
    }
    if (!verify(system)) return 1;
    std::cout << (vector ? "IS-vector" : "IS-block") << " columns=" << ftlpu::hw::kMxmIsColumns << " passed\n";

    // A retained partial row must not leak zero-filled lanes onto SR.
    auto partial_system = std::make_unique<ftlpu::TspSliceSystem>();
    initialize(*partial_system, vector);
    auto partial_schedule = Schedule(partial_system->icu());
    program(partial_schedule, vector, true);
    bool rejected = false;
    try {
        for (std::size_t cycle = 0; cycle < partial_schedule.end_cycle() + 8; ++cycle)
            partial_system->tick({});
    } catch (const std::logic_error& error) {
        rejected = std::string(error.what()).find("incomplete IS result vector") != std::string::npos;
        if (!rejected) throw;
    }
    if (!rejected) throw std::runtime_error("incomplete IS ACC read was accepted");
    }
    std::cout << "MXM decode streamed-weight passed: K=" << kDecodeK
              << " N=" << kDecodeN << ", " << ftlpu::hw::kTileRows
              << " activation vectors, " << kDecodeN
              << " INT8 streams, wave_cycles=" << kDecodeStages
              << ", INT8 to BF16 dequant and BF16 MAC; logs="
              << log_dir.string() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "MXM decode streamed-weight test failed: "
              << error.what() << '\n';
    return 1;
}
