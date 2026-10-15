#include "ftlpu/icu/distributed_queue.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

ftlpu::SxmInstruction::StreamList stream_range(
    std::size_t base, std::size_t count)
{
    auto streams = ftlpu::SxmInstruction::StreamList{};
    for (std::size_t index = 0; index < count; ++index)
        streams.push_back(ftlpu::SxmStreamId {base + index});
    return streams;
}

} // namespace

int main() try
{
    using namespace ftlpu;
    using PermuteQueue = DistributedIcuQueue<SxmInstruction,
        hw::kIcuSxmInstructionBits, 64, 16, 1, 1,
        IcuQueueRole::SxmPermute>;
    using TransposeQueue = DistributedIcuQueue<SxmInstruction,
        hw::kIcuSxmInstructionBits, 64, 16, 1, 1,
        IcuQueueRole::SxmTranspose>;

    auto transpose = SxmInstruction::Transpose(
        stream_range(0, 16), stream_range(16, 16));
    transpose.input_row = 3;
    transpose.src_streams = stream_range(4, 2);
    const auto transposeRun = SxmIcuRun2DInstruction::Run2D(
        0, {2, 3}, {4, 19}, transpose);
    const auto transposePacket =
        isa::encode_sxm_transpose_icu_run_2d_instruction(transposeRun);
    const auto decodedTranspose =
        isa::decode_sxm_transpose_icu_run_2d_instruction(transposePacket);
    require(decodedTranspose.loop.counts
                == std::array<std::size_t, 3> {2, 3, 1}
            && decodedTranspose.instruction.input_row == 3
            && decodedTranspose.instruction.src_streams
                == stream_range(4, 2)
            && decodedTranspose.instruction.dst_streams
                == stream_range(16, 16),
        "SXM transpose compact packet lost a consumed field");
    TransposeQueue transposeQueue;
    transposeQueue.push_sxm_run_2d(transposeRun);
    require(transposeQueue.imem_occupancy() == 2,
        "SXM transpose RUN_2D must occupy two local i-MEM words");

    auto map = SxmInstruction::PermuteMap {};
    for (std::size_t lane = 0; lane < map.size(); ++lane)
        map[lane] = (lane + hw::kLanesPerTile) % map.size();
    const auto tile = SxmInstruction::Permute(
        stream_range(0, 16), stream_range(16, 16), map);
    const auto run = SxmIcuRun2DInstruction::Run2D(
        0, {3, 2}, {2, 13}, tile, 8);
    const auto packet = isa::encode_sxm_permute_icu_run_2d_instruction(run);
    const auto decoded = isa::decode_sxm_permute_icu_run_2d_instruction(packet);
    require(decoded.loop.start_cycle == 0
            && decoded.loop.counts
                == std::array<std::size_t, 3> {3, 2, 1}
            && decoded.loop.cycle_strides
                == std::array<std::size_t, 3> {2, 13, 1}
            && decoded.instruction.opcode == SxmOpcode::Permute
            && decoded.instruction.permute_map == map
            && decoded.permute_map_stride == 8,
        "SXM RUN_2D codec round trip changed the packet");

    PermuteQueue queue;
    queue.push_nop(7);
    queue.push_sxm_run_2d(run);
    require(queue.imem_occupancy() == 1 + isa::EncodedSxmPermuteIcuRun2DPacket::kWordCount,
        "SXM launch delay plus RUN_2D has the wrong local i-MEM size");
    std::vector<std::size_t> issueCycles;
    std::size_t launch = 0;
    for (std::size_t cycle = 0; cycle <= 24; ++cycle) {
        if (const auto issued = queue.tick()) {
            require(issued->opcode == SxmOpcode::Permute
                    && issued->permute_map[0]
                        == (map[0] + launch * 8)
                            % SxmInstruction::kTotalLanes,
                "SXM RUN_2D changed the tile-local instruction");
            issueCycles.push_back(cycle);
            ++launch;
        }
    }
    require(issueCycles
            == std::vector<std::size_t> {7, 9, 11, 20, 22, 24},
        "SXM RUN_2D issued at incorrect cycles");
    auto waitedRun = run;
    waitedRun.loop.wait_cycle = 7;
    const auto waitedPacket =
        isa::encode_sxm_permute_icu_run_2d_instruction(waitedRun);
    require(isa::decode_sxm_permute_icu_run_2d_instruction(
                waitedPacket).loop.wait_cycle == 7,
        "SXM RUN_2D lost its encoded wait_cycle");
    PermuteQueue waitedQueue;
    waitedQueue.push_sxm_run_2d(waitedRun);
    require(waitedQueue.imem_occupancy() == isa::EncodedSxmPermuteIcuRun2DPacket::kWordCount,
        "SXM wait_cycle should replace the NOP i-MEM word");
    std::vector<std::size_t> waitedIssueCycles;
    for (std::size_t cycle = 0; cycle <= 24; ++cycle)
        if (waitedQueue.tick()) waitedIssueCycles.push_back(cycle);
    require(waitedIssueCycles == issueCycles,
        "SXM wait_cycle changed functional issue timing");
    require(queue.peak_active_3d_contexts() == 1
            && queue.active_3d_context_count() == 0,
        "SXM RUN_2D did not use and retire one decoded context");

    auto absoluteStart = run;
    absoluteStart.loop.start_cycle = 7;
    bool rejectedAbsoluteStart = false;
    try {
        (void)isa::encode_sxm_permute_icu_run_2d_instruction(absoluteStart);
    } catch (const std::invalid_argument&) {
        rejectedAbsoluteStart = true;
    }
    require(rejectedAbsoluteStart,
        "SXM RUN_2D accepted an absolute start cycle");

    PermuteQueue sequential;
    for (std::size_t phase = 0; phase < 4; ++phase) {
        sequential.push_sxm_run_2d(SxmIcuRun2DInstruction::Run2D(
            0, {2, 1}, {8, 1}, tile));
    }
    std::vector<std::size_t> sequentialIssueCycles;
    for (std::size_t cycle = 0; cycle <= 41; ++cycle) {
        if (sequential.tick()) sequentialIssueCycles.push_back(cycle);
    }
    require(sequentialIssueCycles
            == std::vector<std::size_t> {0, 8, 9, 17, 18, 26, 27, 35},
        "one-context SXM ICU did not serialize relative RUN_2D packets");

    auto broadcast = SxmInstruction::Transpose(
        {{4},{5}}, {{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28},{29},{30},{31}});
    broadcast.input_row = SxmInstruction::kBroadcastInputRows;
    auto broadcastRun = SxmIcuRun2DInstruction::Run2D(0,{2,1},{16,1},broadcast);
    const auto broadcastDecoded = isa::decode_sxm_transpose_icu_run_2d_instruction(
        isa::encode_sxm_transpose_icu_run_2d_instruction(broadcastRun));
    require(broadcastDecoded.instruction.input_row == SxmInstruction::kBroadcastInputRows
        && broadcastDecoded.instruction.src_streams.size()==2,
        "SXM broadcast-row selector did not round trip");

    auto consumeRow = tile;
    consumeRow.output_row = SxmInstruction::kConsumeOutputRowBase + 2;
    const auto consumeRun = SxmIcuRun2DInstruction::Run2D(0,{2,1},{16,1},consumeRow);
    require(isa::decode_sxm_permute_icu_run_2d_instruction(
        isa::encode_sxm_permute_icu_run_2d_instruction(consumeRun)).instruction.output_row == 11,
        "SXM consuming output-row selector did not round trip");

    std::cout << "sxm_run_2d_instruction_test passed: "
              << "launches=6 active_contexts="
              << hw::kIcuSxmRun2DContextDepth
              << " packet_words=" << isa::EncodedSxmPermuteIcuRun2DPacket::kWordCount
              << " program_words=" << 1 + isa::EncodedSxmPermuteIcuRun2DPacket::kWordCount
              << " context_bits="
              << hw::kIcuSxmPermuteRun2DContextBits << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "sxm_run_2d_instruction_test failed: "
              << error.what() << '\n';
    return 1;
}
