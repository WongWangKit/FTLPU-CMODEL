#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/icu/stream_nd_packet.hpp"

#include <cstddef>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main()
try {
    using namespace ftlpu;

    // Qwen2.5-1.5B FFN Up projection, one hemisphere:
    // M=32, K=1536 (48 K tiles), N=8960/2 (140 N tiles).
    constexpr std::size_t kRows = 32;
    constexpr std::size_t kReductionTiles = 1536 / 32;
    constexpr std::size_t kOutputTilesPerHemisphere = 8960 / 32 / 2;
    constexpr std::size_t kTileCycles = 32;
    constexpr std::size_t kProjectionCycles =
        kRows * kReductionTiles * kOutputTilesPerHemisphere;
    constexpr std::size_t kStart = 4;

    using MemQueue = DistributedIcuQueue<
        MemInstruction, 96, 16, 8, 1>;
    using MxmQueue = DistributedIcuQueue<
        MxmControlInstruction, 128, 16, 8, 1>;
    using DequantQueue = DistributedIcuQueue<
        MxmDequantInstruction, 128, 16, 8, 1>;

    // A weight slice emits four 32-byte rows per 32x32 INT8 tile.  The
    // descriptor covers row, K tile and N tile with three hardware counters.
    const IcuStreamNdSchedule weightSchedule {
        kStart, 3,
        {4, kReductionTiles, kOutputTilesPerHemisphere},
        {1, kTileCycles, kReductionTiles * kTileCycles},
        {1, 4, 4 * kReductionTiles},
        IcuInductionTarget::MemAddress,
    };
    const auto memPacket = encode_icu_stream_nd_packet({
        IcuStreamNdUnit::Mem,
        weightSchedule,
        isa::encode_mem_instruction(MemInstruction::Read(100, 8)),
    });
    MemQueue mem;
    mem.push_stream_nd_packet(memPacket);

    // MXM supercell load and dequant each consume four cycles per weight tile.
    const IcuStreamNdSchedule loadSchedule {
        kStart, 3,
        {4, kReductionTiles, kOutputTilesPerHemisphere},
        {1, kTileCycles, kReductionTiles * kTileCycles},
        {1, 0, 0},
        IcuInductionTarget::MxmWeightColumn,
    };
    MxmQueue load;
    load.push_stream_nd_packet(encode_icu_stream_nd_packet({
        IcuStreamNdUnit::MxmLoad,
        loadSchedule,
        isa::encode_mxm_instruction(MxmControlInstruction::IW(0, 0)),
    }));

    auto noInductionSchedule = loadSchedule;
    noInductionSchedule.operand_strides = {0, 0, 0};
    noInductionSchedule.induction_target = IcuInductionTarget::None;
    DequantQueue dequant;
    const auto scale = MxmDequantInstruction::Scale(0.0078125f);
    dequant.push_stream_nd_packet(encode_icu_stream_nd_packet({
        IcuStreamNdUnit::MxmDequant,
        noInductionSchedule,
        isa::encode_mxm_dequant_instruction(scale),
    }));

    // Compute issues one fine instruction per token row.  Accumulator rows
    // 32..63 are reused for every K/N tile, exactly as the Up projection does.
    const IcuStreamNdSchedule computeSchedule {
        kStart, 3,
        {kRows, kReductionTiles, kOutputTilesPerHemisphere},
        {1, kTileCycles, kReductionTiles * kTileCycles},
        {1, 0, 0},
        IcuInductionTarget::MxmAccumulatorAddress,
    };
    MxmQueue compute;
    const auto computeNative = MxmControlInstruction::Compute(
        0, 16, 4, 32, 1, MxmAccumulatorDestination::Sram,
        MxmDataFormat::BFloat16, false,
        MxmAccumulatorOutputFormat::Float32);

    // Exercise the fixed-field decoder independently of the Up loop.  Signed
    // induction must round-trip, while reserved bits must fail closed.
    const IcuStreamNdSchedule signedSchedule {
        7, 2, {4, 3, 1}, {1, 8, 1}, {-1, 32, 0},
        IcuInductionTarget::MxmAccumulatorAddress,
    };
    const auto signedPacket = encode_icu_stream_nd_packet({
        IcuStreamNdUnit::MxmCompute,
        signedSchedule,
        isa::encode_mxm_instruction(computeNative),
    });
    const auto signedDecoded = decode_icu_stream_nd_packet(signedPacket);
    require(signedDecoded.schedule.operand_strides[0] == -1
            && signedDecoded.schedule.operand_strides[1] == 32
            && encode_icu_stream_nd_packet(signedDecoded).words
                == signedPacket.words,
        "STREAM_ND fixed packet did not round-trip signed induction");
    auto corruptPacket = signedPacket;
    corruptPacket.words[0] |= std::uint32_t {1} << 31;
    bool rejectedReservedBit = false;
    try {
        static_cast<void>(decode_icu_stream_nd_packet(corruptPacket));
    } catch (const std::invalid_argument&) {
        rejectedReservedBit = true;
    }
    require(rejectedReservedBit,
        "STREAM_ND decoder accepted a non-zero reserved header bit");

    compute.push_stream_nd_packet(encode_icu_stream_nd_packet({
        IcuStreamNdUnit::MxmCompute,
        computeSchedule,
        isa::encode_mxm_instruction(computeNative),
    }));

    require(mem.imem_occupancy() == 1 && load.imem_occupancy() == 1
            && dequant.imem_occupancy() == 1
            && compute.imem_occupancy() == 1,
        "Qwen Up projection did not remain one coarse instruction per ICU queue");

    std::size_t memIssues = 0;
    std::size_t loadIssues = 0;
    std::size_t dequantIssues = 0;
    std::size_t computeIssues = 0;
    std::optional<MemInstruction> firstMem;
    std::optional<MemInstruction> lastMem;
    std::optional<MxmControlInstruction> firstLoad;
    std::optional<MxmControlInstruction> lastLoad;
    std::optional<MxmControlInstruction> firstCompute;
    std::optional<MxmControlInstruction> lastCompute;
    const std::size_t finalCycle = kStart + kProjectionCycles - 1;
    for (std::size_t cycle = 0; cycle <= finalCycle; ++cycle) {
        if (auto instruction = mem.tick()) {
            if (!firstMem) firstMem = instruction;
            lastMem = instruction;
            ++memIssues;
        }
        if (auto instruction = load.tick()) {
            if (!firstLoad) firstLoad = instruction;
            lastLoad = instruction;
            ++loadIssues;
        }
        if (auto instruction = dequant.tick()) {
            require(instruction->scale_bf16 == scale.scale_bf16,
                "Qwen Up dequant packet changed its BF16 scale");
            ++dequantIssues;
        }
        if (auto instruction = compute.tick()) {
            if (!firstCompute) firstCompute = instruction;
            lastCompute = instruction;
            ++computeIssues;
        }
    }

    constexpr std::size_t kWeightIssues =
        4 * kReductionTiles * kOutputTilesPerHemisphere;
    require(memIssues == kWeightIssues && loadIssues == kWeightIssues
            && dequantIssues == kWeightIssues,
        "Qwen Up weight-side ICU expansion count is incorrect");
    require(computeIssues == kProjectionCycles,
        "Qwen Up compute ICU expansion count is incorrect");
    require(firstMem && lastMem && firstMem->address == 100
            && lastMem->address == 100 + kWeightIssues - 1,
        "Qwen Up MEM ICU did not generate the complete affine address range");
    require(firstLoad && lastLoad && firstLoad->weight_column == 0
            && lastLoad->weight_column == 3,
        "Qwen Up load ICU did not regenerate supercell columns");
    require(firstCompute && lastCompute
            && firstCompute->accumulator_address == 32
            && lastCompute->accumulator_address == 63,
        "Qwen Up compute ICU did not regenerate token accumulator rows");
    require(mem.done() && load.done() && dequant.done() && compute.done(),
        "Qwen Up coarse ICU programs did not complete");

    std::cout << "Qwen2.5 Up projection hardware ICU packet passed: "
              << "one command/queue, M=" << kRows
              << " K_tiles=" << kReductionTiles
              << " N_tiles/hemisphere=" << kOutputTilesPerHemisphere
              << " compute_issues=" << computeIssues << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Qwen2.5 Up projection hardware ICU packet failed: "
              << error.what() << '\n';
    return 1;
}
