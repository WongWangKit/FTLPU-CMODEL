#include "ftlpu/icu/distributed_queue.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main()
try {
    using namespace ftlpu;
    using ControlQueue = DistributedIcuQueue<
        MxmControlInstruction, 128, 64, 16, 1>;

    ControlQueue load;
    load.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        2, 3, {2, 2, 2}, {2, 8, 20}, {1, 4, 8},
        IcuInductionTarget::MxmWeightColumn},
        MxmControlInstruction::IW(0, 0));
    std::vector<std::pair<std::size_t, std::size_t>> loadIssues;
    for (std::size_t cycle = 0; cycle <= 32; ++cycle) {
        if (const auto instruction = load.tick())
            loadIssues.emplace_back(cycle, instruction->weight_column);
    }
    require(loadIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {2, 0}, {4, 1}, {10, 4}, {12, 5},
                {22, 8}, {24, 9}, {30, 12}, {32, 13}},
        "MXM load STREAM_ND emitted incorrect cycles or columns");
    require(load.done(), "MXM load STREAM_ND did not complete");

    ControlQueue compute;
    const auto finalPartial = MxmControlInstruction::Compute(
        0, 0, 0, 4, 1, MxmAccumulatorDestination::Stream,
        MxmDataFormat::BFloat16, true,
        MxmAccumulatorOutputFormat::BFloat16);
    compute.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        3, 2, {3, 2, 1}, {1, 8, 1}, {0, 4, 0},
        IcuInductionTarget::MxmAccumulatorAddress}, finalPartial);
    std::vector<std::pair<std::size_t, std::size_t>> computeIssues;
    for (std::size_t cycle = 0; cycle <= 13; ++cycle) {
        if (const auto instruction = compute.tick()) {
            require(instruction->accumulator_destination
                    == MxmAccumulatorDestination::Stream
                    && instruction->accumulator_clear
                    && instruction->accumulator_output_format
                        == MxmAccumulatorOutputFormat::BFloat16,
                "MXM compute STREAM_ND changed final-partial behavior");
            computeIssues.emplace_back(
                cycle, instruction->accumulator_address);
        }
    }
    require(computeIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {3, 4}, {4, 4}, {5, 4},
                {11, 8}, {12, 8}, {13, 8}},
        "MXM compute STREAM_ND emitted incorrect cycles or accumulator addresses");

    // Native4 FFN keeps one descriptor active in each physical MXM ICU
    // queue for an entire K parity.  The load queue advances once per K
    // reduction; the compute queue visits all output waves before advancing.
    using DecodeLoadQueue = DistributedIcuQueue<
        MxmControlInstruction, 128, 64, 16, 1, 1,
        IcuQueueRole::MxmLoad>;
    using DecodeComputeQueue = DistributedIcuQueue<
        MxmControlInstruction, 128, 64, 16, 1, 1,
        IcuQueueRole::MxmCompute>;
    DecodeLoadQueue decodeLoad;
    decodeLoad.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        2, 3, {1, 1, 3}, {1, 1, 7}, {0, 0, 0},
        IcuInductionTarget::None},
        MxmControlInstruction::DecodeLoadActivation(
            1, 0, MxmDataFormat::BFloat16, MxmDecodeLayout::Native4x4));
    DecodeComputeQueue decodeCompute;
    decodeCompute.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        6, 3, {3, 1, 3}, {1, 1, 7}, {1, 0, 0},
        IcuInductionTarget::MxmAccumulatorAddress},
        MxmControlInstruction::DecodeStreamCompute(
            1, 0, MxmDataFormat::BFloat16, 0, 0,
            MxmAccumulatorDestination::Sram, false,
            MxmDecodeLayout::Native4x4));
    std::vector<std::size_t> decodeLoadCycles;
    std::vector<std::pair<std::size_t, std::size_t>> decodeComputeIssues;
    for (std::size_t cycle = 0; cycle <= 22; ++cycle) {
        if (const auto instruction = decodeLoad.tick()) {
            require(instruction->decode_operation
                    == MxmDecodeOperation::LoadActivation
                    && instruction->decode_layout == MxmDecodeLayout::Native4x4,
                "Native4 K-loop load changed its FU operation");
            decodeLoadCycles.push_back(cycle);
        }
        if (const auto instruction = decodeCompute.tick()) {
            require(instruction->decode_operation
                    == MxmDecodeOperation::StreamCompute
                    && instruction->decode_layout == MxmDecodeLayout::Native4x4,
                "Native4 K-loop compute changed its FU operation");
            decodeComputeIssues.emplace_back(
                cycle, instruction->accumulator_address);
        }
    }
    require(decodeLoadCycles == std::vector<std::size_t> {2, 9, 16},
        "Native4 K-loop load emitted incorrect reduction cycles");
    require(decodeComputeIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {6, 0}, {7, 1}, {8, 2},
                {13, 0}, {14, 1}, {15, 2},
                {20, 0}, {21, 1}, {22, 2}},
        "Native4 K-loop compute emitted incorrect wave/reduction cycles");
    require(decodeLoad.done() && decodeCompute.done(),
        "Native4 K-loop descriptors did not complete");

    using DequantQueue = DistributedIcuQueue<
        MxmDequantInstruction, 32, 32, 8, 1>;
    DequantQueue dequant;
    const auto scale = MxmDequantInstruction::Scale(0.125f);
    dequant.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        2, 2, {4, 2, 1}, {1, 8, 1}, {0, 0, 0},
        IcuInductionTarget::None}, scale);
    std::vector<std::size_t> dequantCycles;
    for (std::size_t cycle = 0; cycle <= 13; ++cycle) {
        if (const auto instruction = dequant.tick()) {
            require(instruction->scale_bf16 == scale.scale_bf16,
                "MXM dequant STREAM_ND changed the scale");
            dequantCycles.push_back(cycle);
        }
    }
    require(dequantCycles == std::vector<std::size_t> {
                2, 3, 4, 5, 10, 11, 12, 13},
        "MXM dequant STREAM_ND emitted at incorrect cycles");

    std::cout << "icu_mxm_stream_nd_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_mxm_stream_nd_test failed: "
              << error.what() << '\n';
    return 1;
}
