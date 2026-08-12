#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "system_gantt_trace.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr std::size_t kReductions = 2;
constexpr std::size_t kOutputRows = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kK = ftlpu::hw::kMxmRows;
constexpr std::size_t kN = ftlpu::hw::kMxmColumns;
constexpr std::size_t kBytePlanes = ftlpu::SxmSlice::kTransposeBytePlanes;
constexpr std::size_t kSxmStreams = kOutputRows * kBytePlanes;

constexpr std::array<std::size_t, 16> kWeightSlices {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
};
constexpr std::array<std::size_t, 2> kActivationSlices {32, 33};
constexpr std::array<std::size_t, 16> kOutputSlices {
    36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51,
};
constexpr std::array<std::size_t, 4> kFinalOutputSlices {0, 1, 2, 3};

constexpr std::size_t kWeightAddress = 64;
constexpr std::size_t kActivationAddress = 96;
constexpr std::size_t kOutputAddress = 160;
constexpr std::size_t kFinalOutputAddress = 320;
constexpr std::size_t kAccumulatorAddress = 256;
constexpr std::array<std::size_t, kReductions> kLoadCycles {20, 50};
constexpr std::array<std::size_t, kReductions> kComputeCycles {30, 60};
constexpr std::size_t kVxmConfigCycle = 112;
constexpr std::size_t kVxmWaveInterval = ftlpu::hw::kTileRows;
constexpr std::size_t kVxmWaves = kOutputRows / 2;

static_assert(kOutputRows == 8);
static_assert(kK == 32);
static_assert(kN == 32);
static_assert(kSxmStreams == 16);

std::size_t mem_queue(std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(
        ftlpu::Hemisphere::East, slice);
}

std::size_t mem_to_mxm_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t sxm_to_mem_latency(std::size_t slice)
{
    return ftlpu::hw::kMemEastBoundaryStreamRegisterColumn
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mem_to_vxm_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 2;
}

std::size_t vxm_to_mem_latency(std::size_t slice)
{
    return slice / ftlpu::hw::kMemSlicesPerGroup + 1;
}

float activation_value(
    std::size_t reduction,
    std::size_t output_row,
    std::size_t k)
{
    const auto pattern = static_cast<int>(
        (reduction * 5 + output_row * 3 + k * 7) % 17) - 8;
    return static_cast<float>(pattern) / 8.0f;
}

float weight_value(
    std::size_t reduction,
    std::size_t k,
    std::size_t column)
{
    const auto pattern = static_cast<int>(
        (reduction * 11 + k * 5 + column * 3) % 19) - 9;
    return static_cast<float>(pattern) / 16.0f;
}

ftlpu::Bf16 expected_mxm_value(
    std::size_t output_row,
    std::size_t column)
{
    float sum = 0.0f;
    for (std::size_t reduction = 0; reduction < kReductions; ++reduction) {
        for (std::size_t k = 0; k < kK; ++k) {
            const auto activation = ftlpu::Bf16::from_float(
                activation_value(reduction, output_row, k)).to_float();
            const auto weight = ftlpu::Bf16::from_float(
                weight_value(reduction, k, column)).to_float();
            sum += activation * weight;
        }
    }
    return ftlpu::Bf16::from_float(sum);
}

ftlpu::SxmInstruction::StreamList west_streams(
    std::size_t first,
    std::size_t count)
{
    auto streams = ftlpu::SxmInstruction::StreamList {};
    streams.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        streams.push_back(ftlpu::SxmStreamId {
            ftlpu::StreamId::West(first + index).packed()});
    }
    return streams;
}

class Schedule {
public:
    explicit Schedule(ftlpu::InstructionControlUnit& icu)
        : icu_(icu)
    {
    }

    void mem_at(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        auto& cursor = mem_[mem_queue(slice)];
        require_available(cursor, cycle, "MEM");
        icu_.enqueue_mem_nop(mem_queue(slice), cycle - cursor);
        icu_.enqueue_mem(mem_queue(slice), instruction);
        cursor = cycle + 1;
        end_cycle_ = std::max(end_cycle_, cursor);
    }

    void mem_repeat_at(
        std::size_t slice,
        std::size_t cycle,
        ftlpu::MemInstruction instruction,
        std::size_t count,
        std::int64_t address_stride)
    {
        mem_at(slice, cycle, instruction);
        if (count > 1) {
            icu_.enqueue_mem_repeat(
                mem_queue(slice), count - 1, 1, address_stride);
        }
        mem_[mem_queue(slice)] = cycle + count;
        end_cycle_ = std::max(end_cycle_, cycle + count);
    }

    void mxm_load_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction)
    {
        require_available(mxm_load_, cycle, "MXM load");
        icu_.enqueue_mxm_load_nop(0, cycle - mxm_load_);
        icu_.enqueue_mxm(0, instruction);
        mxm_load_ = cycle + 1;
        end_cycle_ = std::max(end_cycle_, mxm_load_);
    }

    void mxm_compute_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction)
    {
        require_available(mxm_compute_, cycle, "MXM compute");
        icu_.enqueue_mxm_compute_nop(0, cycle - mxm_compute_);
        icu_.enqueue_mxm(0, instruction);
        mxm_compute_ = cycle + 1;
        end_cycle_ = std::max(end_cycle_, mxm_compute_);
    }

    void sxm_transpose_at(
        std::size_t cycle,
        ftlpu::SxmInstruction instruction)
    {
        require_available(sxm_transpose_, cycle, "SXM transpose");
        icu_.enqueue_sxm_transpose_nop(cycle - sxm_transpose_);
        icu_.enqueue_sxm_transpose(std::move(instruction));
        sxm_transpose_ = cycle + 1;
        end_cycle_ = std::max(end_cycle_, sxm_transpose_);
    }

    void sxm_permute_at(
        std::size_t cycle,
        ftlpu::SxmInstruction instruction)
    {
        require_available(sxm_permute_, cycle, "SXM permute");
        icu_.enqueue_sxm_permute_nop(cycle - sxm_permute_);
        icu_.enqueue_sxm_permute(std::move(instruction));
        sxm_permute_ = cycle + 1;
        end_cycle_ = std::max(end_cycle_, sxm_permute_);
    }

    void vxm_at(
        std::size_t alu,
        std::size_t cycle,
        ftlpu::VxmChainDepth depth,
        const ftlpu::VxmLaneAluInstruction& instruction)
    {
        auto& cursor = vxm_[alu];
        require_available(cursor, cycle, "VXM");
        icu_.enqueue_vxm_nop(alu, cycle - cursor);
        icu_.enqueue_vxm(alu, depth, instruction);
        cursor = cycle + 1;
        end_cycle_ = std::max(end_cycle_, cursor);
    }

    std::size_t end_cycle() const noexcept
    {
        return end_cycle_;
    }

private:
    static void require_available(
        std::size_t cursor,
        std::size_t cycle,
        const char* queue)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                std::string("MXM-SXM integration schedule overlaps ")
                + queue + " queue");
        }
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<std::size_t, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::size_t mxm_load_{0};
    std::size_t mxm_compute_{0};
    std::size_t sxm_transpose_{0};
    std::size_t sxm_permute_{0};
    std::array<std::size_t, ftlpu::InstructionControlUnit::kVxmQueues> vxm_{};
    std::size_t end_cycle_{0};
};

void initialize_inputs(ftlpu::TspSliceSystem& system)
{
    for (std::size_t reduction = 0; reduction < kReductions; ++reduction) {
        for (std::size_t column_block = 0;
             column_block < ftlpu::hw::kMxmSupercellsPerPlane;
             ++column_block) {
            const auto address = kWeightAddress
                + reduction * ftlpu::hw::kMxmSupercellsPerPlane
                + column_block;
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile;
                     ++lane) {
                    const auto k = tile * ftlpu::hw::kLanesPerTile + lane;
                    for (std::size_t local_column = 0;
                         local_column < ftlpu::hw::kMxmSupercellColumns;
                         ++local_column) {
                        const auto column = column_block
                                * ftlpu::hw::kMxmSupercellColumns
                            + local_column;
                        const auto bits = ftlpu::Bf16::from_float(
                            weight_value(reduction, k, column)).bits();
                        system.initialize_mem_sram_lane_byte(
                            kWeightSlices[local_column * kBytePlanes],
                            tile,
                            address,
                            lane,
                            static_cast<std::uint8_t>(bits & 0xffu));
                        system.initialize_mem_sram_lane_byte(
                            kWeightSlices[local_column * kBytePlanes + 1],
                            tile,
                            address,
                            lane,
                            static_cast<std::uint8_t>(bits >> 8));
                    }
                }
            }
        }

        for (std::size_t output_row = 0;
             output_row < kOutputRows;
             ++output_row) {
            const auto address = kActivationAddress
                + reduction * kOutputRows + output_row;
            for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile;
                     ++lane) {
                    const auto k = tile * ftlpu::hw::kLanesPerTile + lane;
                    const auto bits = ftlpu::Bf16::from_float(
                        activation_value(reduction, output_row, k)).bits();
                    system.initialize_mem_sram_lane_byte(
                        kActivationSlices[0],
                        tile,
                        address,
                        lane,
                        static_cast<std::uint8_t>(bits & 0xffu));
                    system.initialize_mem_sram_lane_byte(
                        kActivationSlices[1],
                        tile,
                        address,
                        lane,
                        static_cast<std::uint8_t>(bits >> 8));
                }
            }
        }
    }
}

void build_schedule(Schedule& schedule)
{
    for (std::size_t reduction = 0; reduction < kReductions; ++reduction) {
        for (std::size_t column_block = 0;
             column_block < ftlpu::hw::kMxmSupercellsPerPlane;
             ++column_block) {
            const auto load_cycle = kLoadCycles[reduction] + column_block;
            const auto address = kWeightAddress
                + reduction * ftlpu::hw::kMxmSupercellsPerPlane
                + column_block;
            for (std::size_t stream = 0;
                 stream < kWeightSlices.size();
                 ++stream) {
                const auto slice = kWeightSlices[stream];
                schedule.mem_at(
                    slice,
                    load_cycle - mem_to_mxm_latency(slice),
                    ftlpu::MemInstruction::Read(
                        address, ftlpu::StreamId::East(stream)));
            }
            schedule.mxm_load_at(
                load_cycle,
                ftlpu::MxmControlInstruction::IWDirect16(
                    reduction % ftlpu::Mxm::kWeightBuffers,
                    column_block));
        }

        for (std::size_t byte = 0;
             byte < kActivationSlices.size();
             ++byte) {
            const auto slice = kActivationSlices[byte];
            schedule.mem_repeat_at(
                slice,
                kComputeCycles[reduction] - mem_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    kActivationAddress + reduction * kOutputRows,
                    ftlpu::StreamId::East(byte)),
                kOutputRows,
                1);
        }

        for (std::size_t output_row = 0;
             output_row < kOutputRows;
             ++output_row) {
            const auto final_reduction = reduction + 1 == kReductions;
            schedule.mxm_compute_at(
                kComputeCycles[reduction] + output_row,
                ftlpu::MxmControlInstruction::Compute(
                    reduction % ftlpu::Mxm::kWeightBuffers,
                    0,
                    final_reduction ? output_row * kBytePlanes : 0,
                    kAccumulatorAddress,
                    1,
                    final_reduction
                        ? ftlpu::MxmAccumulatorDestination::Stream
                        : ftlpu::MxmAccumulatorDestination::Sram,
                    ftlpu::MxmDataFormat::BFloat16,
                    ftlpu::MxmComputeMode::Vector,
                    final_reduction,
                    ftlpu::MxmAccumulatorOutputFormat::BFloat16));
        }
    }

    // The first final-result row is produced by MXM at sreg14 after the four
    // supercell columns.  It becomes visible to SXM on the following cycle.
    constexpr auto kSxmCaptureCycle =
        kComputeCycles.back()
        + ftlpu::hw::kMxmSupercellsPerPlane
        + ftlpu::Mxm::kLocalMacStages - 1;
    const auto transpose_source = west_streams(0, kSxmStreams);
    const auto transpose_internal = west_streams(16, kSxmStreams);
    const auto transpose_output = west_streams(0, kSxmStreams);
    schedule.sxm_transpose_at(
        kSxmCaptureCycle,
        ftlpu::SxmInstruction::Transpose(
            transpose_source, transpose_internal));
    schedule.sxm_permute_at(
        kSxmCaptureCycle,
        ftlpu::SxmInstruction::Permute(
            transpose_internal,
            transpose_output,
            ftlpu::Permute320::identity_map()));

    // Eight capture cycles plus explicit Transpose and Permute pipeline stages.
    constexpr auto kFirstSxmOutputCycle =
        kSxmCaptureCycle + kOutputRows + 2;
    for (std::size_t stream = 0; stream < kSxmStreams; ++stream) {
        const auto output_row = stream / kBytePlanes;
        const auto slice = kOutputSlices[stream];
        schedule.mem_at(
            slice,
            kFirstSxmOutputCycle + output_row
                + sxm_to_mem_latency(slice),
            ftlpu::MemInstruction::Write(
                kOutputAddress + output_row,
                ftlpu::StreamId::West(stream)));
    }

    // The SXM block is now resident in MEM.  Four VXM waves read two rows at
    // a time into the two mirrored fixed chains.  The compact instruction
    // selects BF16 input/output and chain depth; no VXM internal API is used.
    auto head = ftlpu::VxmLaneAluInstruction {};
    head.operation = ftlpu::VxmAluOpcode::Add;
    head.lhs = ftlpu::VxmLaneOperand::StreamBFloat16();
    head.rhs = ftlpu::VxmLaneOperand::Imm(1.0f);
    head.precision = ftlpu::VxmAluPrecision::Float32;

    auto tail = ftlpu::VxmLaneAluInstruction {};
    tail.operation = ftlpu::VxmAluOpcode::Bypass;
    tail.lhs = ftlpu::VxmLaneOperand::Previous();
    tail.rhs = ftlpu::VxmLaneOperand::Imm(0.0f);
    tail.precision = ftlpu::VxmAluPrecision::Float32;
    tail.output_type = ftlpu::VxmCastTarget::BFloat16;
    tail.output_stream = 0;

    constexpr std::array<std::size_t, 4> kVxmInputStreams {0, 1, 16, 17};
    constexpr std::array<std::size_t, 4> kVxmOutputStreams {0, 1, 8, 9};
    for (std::size_t wave = 0; wave < kVxmWaves; ++wave) {
        const auto config_cycle = kVxmConfigCycle + wave * kVxmWaveInterval;
        const auto input_cycle = config_cycle + 1;
        const auto first_output_cycle = input_cycle + 1;

        schedule.vxm_at(
            0, config_cycle, ftlpu::VxmChainDepth::Two, head);
        schedule.vxm_at(
            1, config_cycle, ftlpu::VxmChainDepth::Two, tail);

        for (std::size_t chain = 0; chain < 2; ++chain) {
            const auto output_row = wave * 2 + chain;
            for (std::size_t byte = 0; byte < kBytePlanes; ++byte) {
                const auto source_index = output_row * kBytePlanes + byte;
                const auto source_slice = kOutputSlices[source_index];
                schedule.mem_at(
                    source_slice,
                    input_cycle - mem_to_vxm_latency(source_slice),
                    ftlpu::MemInstruction::Read(
                        kOutputAddress + output_row,
                        ftlpu::StreamId::West(
                            kVxmInputStreams[chain * kBytePlanes + byte])));
            }
        }

        for (std::size_t byte = 0;
             byte < kFinalOutputSlices.size();
             ++byte) {
            const auto slice = kFinalOutputSlices[byte];
            schedule.mem_at(
                slice,
                first_output_cycle + vxm_to_mem_latency(slice),
                ftlpu::MemInstruction::Write(
                    kFinalOutputAddress + wave,
                    ftlpu::StreamId::East(kVxmOutputStreams[byte])));
        }
    }
}

bool verify_final_output(const ftlpu::TspSliceSystem& system)
{
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            for (std::size_t wave = 0; wave < kVxmWaves; ++wave) {
                for (std::size_t chain = 0; chain < 2; ++chain) {
                    const auto low = system.read_mem_sram_lane_byte(
                        kFinalOutputSlices[chain * kBytePlanes],
                        tile,
                        kFinalOutputAddress + wave,
                        lane);
                    const auto high = system.read_mem_sram_lane_byte(
                        kFinalOutputSlices[chain * kBytePlanes + 1],
                        tile,
                        kFinalOutputAddress + wave,
                        lane);
                    const auto actual_bits = static_cast<std::uint16_t>(low)
                        | (static_cast<std::uint16_t>(high) << 8);
                    const auto output_row = wave * 2 + chain;
                    const auto mxm = expected_mxm_value(
                        lane,
                        tile * ftlpu::hw::kLanesPerTile + output_row);
                    const auto expected = ftlpu::Bf16::from_float(
                        mxm.to_float() + 1.0f);
                    if (actual_bits != expected.bits()) {
                        const auto actual = ftlpu::Bf16::from_bits(actual_bits);
                        std::cerr
                            << "MEM -> MXM -> SXM -> MEM -> VXM -> MEM mismatch"
                            << " tile=" << tile
                            << " lane=" << lane
                            << " output_row=" << output_row
                            << " actual=" << actual.to_float()
                            << " expected=" << expected.to_float()
                            << '\n';
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

} // namespace

int main()
try {
    auto system = ftlpu::TspSliceSystem {};
    initialize_inputs(system);
    auto schedule = Schedule(system.icu());
    build_schedule(schedule);
    auto timing = integration_timing::SystemGanttTrace {};
    const auto collect_timing =
        integration_timing::SystemGanttTrace::enabled();

    const auto run_cycles = schedule.end_cycle()
        + ftlpu::hw::kTileRows + 4;
    for (std::size_t cycle = 0; cycle < run_cycles; ++cycle) {
        try {
            system.tick({});
            if (collect_timing) timing.capture(system);
        }
        catch (const std::exception& error) {
            std::cerr
                << "MEM -> MXM -> SXM -> MEM -> VXM -> MEM failed at cycle "
                      << cycle << ": " << error.what() << '\n';
            return 1;
        }
    }

    if (!verify_final_output(system)) return 1;

    std::cout
        << "MEM -> MXM -> SXM -> MEM -> VXM -> MEM passed: "
        << "one 8x32 output block, two K reductions, serial 8-cycle "
        << "SXM transpose, MEM staging, four dual-chain BF16 VXM waves\n";
    if (collect_timing) {
        timing.write(
            "attention_dataflow_system",
            "MEM-MXM-SXM-MEM-VXM-MEM system timing");
    }
    return 0;
}
catch (const std::exception& error) {
    std::cerr
        << "MEM -> MXM -> SXM -> MEM -> VXM -> MEM setup failed: "
              << error.what() << '\n';
    return 1;
}
