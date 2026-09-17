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

template <typename Function>
void require_invalid(Function&& function, const char* message)
{
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

template <typename Function>
void require_static_schedule(Function&& function, const char* message)
{
    bool rejected = false;
    try {
        function();
    } catch (const ftlpu::StaticScheduleError&) {
        rejected = true;
    }
    require(rejected, message);
}

} // namespace

int main()
try {
    using namespace ftlpu;
    using MemQueue = DistributedIcuQueue<MemInstruction,
        hw::kIcuMemInstructionBits, 64, 8, 1, 1, IcuQueueRole::Mem>;
    using LoadQueue = DistributedIcuQueue<MxmControlInstruction,
        hw::kIcuMxmInstructionBits, 64, 8, 1, 1,
        IcuQueueRole::MxmLoad>;
    using DequantQueue = DistributedIcuQueue<MxmDequantInstruction,
        hw::kIcuMxmInstructionBits, 64, 8, 1, 1,
        IcuQueueRole::MxmDequant>;
    using ComputeQueue = DistributedIcuQueue<MxmControlInstruction,
        hw::kIcuMxmInstructionBits, 64, 8, 1, 1,
        IcuQueueRole::MxmCompute>;

    MemQueue memStream;
    memStream.push_mem_stream_nd(IcuMemStreamNdSchedule {
        2, 2, {2, 2, 1}, {2, 8, 1}, {1, 16, 0},
        IcuInductionTarget::MemAddress},
        MemInstruction::Read(100, 0));
    require(memStream.imem_occupancy()
            == MemQueue::three_d_packet_word_count + 1,
        "legacy MEM_STREAM_ND did not lower to NOP plus one MEM 3-D packet");
    std::vector<std::pair<std::size_t, std::size_t>> memIssues;
    for (std::size_t cycle = 0; cycle <= 12; ++cycle) {
        if (const auto instruction = memStream.tick())
            memIssues.emplace_back(cycle, instruction->address);
    }
    require(memIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {2, 100}, {4, 101}, {10, 116}, {12, 117}},
        "raw MEM 3-D adapter changed STREAM_ND cycle/address induction");

    // QueueRole::Mem is a physical raw-word queue. Its compatibility loader
    // converts successive absolute descriptors to gaps from one queue-local
    // cursor; it must not add each absolute start as a fresh delay.
    MemQueue sequentialMem;
    sequentialMem.push_mem_stream_nd(IcuMemStreamNdSchedule {
        2, 1, {2, 1, 1}, {2, 1, 1}, {1, 0, 0},
        IcuInductionTarget::MemAddress},
        MemInstruction::Read(20, 0));
    sequentialMem.push_mem_stream_nd(IcuMemStreamNdSchedule {
        7, 1, {2, 1, 1}, {1, 1, 1}, {1, 0, 0},
        IcuInductionTarget::MemAddress},
        MemInstruction::Read(30, 0));
    require(sequentialMem.imem_occupancy()
            == 2 * MemQueue::three_d_packet_word_count + 2,
        "successive legacy descriptors did not encode two queue-local NOP gaps");
    const auto occupancyBeforeOverlap = sequentialMem.imem_occupancy();
    require_static_schedule([&] {
        sequentialMem.push_mem_stream_nd(IcuMemStreamNdSchedule {
            8, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0},
            IcuInductionTarget::MemAddress},
            MemInstruction::Read(40, 0));
    }, "raw compatibility adapter accepted an overlapping absolute start");
    require(sequentialMem.imem_occupancy() == occupancyBeforeOverlap,
        "rejected absolute start modified the raw queue program");
    std::vector<std::pair<std::size_t, std::size_t>> sequentialIssues;
    for (std::size_t cycle = 0; cycle <= 8; ++cycle) {
        if (const auto instruction = sequentialMem.tick())
            sequentialIssues.emplace_back(cycle, instruction->address);
    }
    require(sequentialIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {2, 20}, {4, 21}, {7, 30}, {8, 31}},
        "successive legacy descriptors accumulated absolute starts");

    MemQueue memMacro;
    memMacro.push_macro(IcuMacroSchedule {
        1, 2, 2, 2, 2, 8, 20,
        IcuInductionTarget::MemAddress},
        MemInstruction::WriteTap(10, 1));
    require(memMacro.imem_occupancy()
            == MemQueue::three_d_packet_word_count + 1,
        "legacy MEM macro did not encode its start as a preceding NOP");
    std::vector<std::pair<std::size_t, std::size_t>> macroIssues;
    for (std::size_t cycle = 0; cycle <= 11; ++cycle) {
        if (const auto instruction = memMacro.tick()) {
            require(instruction->opcode == MemOpcode::Write
                    && instruction->preserve_stream,
                "raw MEM 3-D adapter lost write-tap semantics");
            macroIssues.emplace_back(cycle, instruction->address);
        }
    }
    require(macroIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {1, 10}, {3, 12}, {9, 30}, {11, 32}},
        "raw MEM 3-D adapter changed Macro cycle/address induction");

    LoadQueue load;
    load.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        2, 2, {2, 2, 1}, {2, 8, 1}, {1, 2, 0},
        IcuInductionTarget::MxmWeightColumn},
        MxmControlInstruction::IW(0, 0,
            MxmWeightInputMode::Int8DequantBf16, 8));
    require(load.imem_occupancy()
            == LoadQueue::three_d_packet_word_count + 1,
        "legacy MXM_STREAM_ND did not lower to NOP plus LOAD_3D");
    std::vector<std::pair<std::size_t, std::size_t>> loadIssues;
    for (std::size_t cycle = 0; cycle <= 12; ++cycle) {
        if (const auto instruction = load.tick()) {
            require(instruction->weight_stream_base == 8,
                "raw LOAD_3D adapter changed the weight stream");
            loadIssues.emplace_back(cycle, instruction->weight_column);
        }
    }
    require(loadIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {2, 0}, {4, 1}, {10, 2}, {12, 3}},
        "raw LOAD_3D adapter changed STREAM_ND induction");

    LoadQueue loadMacro;
    loadMacro.push_macro(IcuMacroSchedule {
        1, 2, 2, 1, 2, 8, 2,
        IcuInductionTarget::MxmWeightColumn},
        MxmControlInstruction::IW(1, 0));
    require(loadMacro.imem_occupancy()
            == LoadQueue::three_d_packet_word_count + 1,
        "legacy MXM macro did not encode its start as a preceding NOP");
    std::vector<std::size_t> macroColumns;
    for (std::size_t cycle = 0; cycle <= 11; ++cycle) {
        if (const auto instruction = loadMacro.tick())
            macroColumns.push_back(instruction->weight_column);
    }
    require(macroColumns == std::vector<std::size_t> {0, 1, 2, 3},
        "raw LOAD_3D adapter changed Macro induction");

    DequantQueue dequant;
    const auto scale = MxmDequantInstruction::Scale(0.125f);
    dequant.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        1, 1, {3, 1, 1}, {2, 1, 1}, {0, 0, 0},
        IcuInductionTarget::None}, scale);
    require(dequant.imem_occupancy()
            == DequantQueue::three_d_packet_word_count + 1,
        "legacy dequant schedule did not encode its start as a preceding NOP");
    std::vector<std::size_t> dequantCycles;
    for (std::size_t cycle = 0; cycle <= 5; ++cycle) {
        if (const auto instruction = dequant.tick()) {
            require(instruction->scale_bf16 == scale.scale_bf16,
                "raw DEQUANT_3D adapter changed the scale");
            dequantCycles.push_back(cycle);
        }
    }
    require(dequantCycles == std::vector<std::size_t> {1, 3, 5},
        "raw DEQUANT_3D adapter changed STREAM_ND timing");

    ComputeQueue compute;
    const auto computeTemplate = MxmControlInstruction::Compute(
        1, 4, 12, 20, 2, MxmAccumulatorDestination::Stream,
        MxmDataFormat::BFloat16, true,
        MxmAccumulatorOutputFormat::BFloat16);
    compute.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        3, 2, {3, 2, 1}, {1, 8, 1}, {0, 4, 0},
        IcuInductionTarget::MxmAccumulatorAddress}, computeTemplate);
    require(compute.imem_occupancy()
            == ComputeQueue::three_d_packet_word_count + 1,
        "legacy compute schedule did not encode its start as a preceding NOP");
    std::vector<std::pair<std::size_t, std::size_t>> computeIssues;
    for (std::size_t cycle = 0; cycle <= 13; ++cycle) {
        if (const auto instruction = compute.tick()) {
            require(instruction->weight_buffer == 1
                    && instruction->activation_stream_base == 4
                    && instruction->stream_base == 12
                    && instruction->accumulator_row_stride == 2
                    && instruction->accumulator_destination
                        == MxmAccumulatorDestination::Stream
                    && instruction->accumulator_clear
                    && instruction->data_format == MxmDataFormat::BFloat16
                    && instruction->accumulator_output_format
                        == MxmAccumulatorOutputFormat::BFloat16,
                "raw COMPUTE_3D adapter changed fixed compute operands");
            computeIssues.emplace_back(
                cycle, instruction->accumulator_address);
        }
    }
    require(computeIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {3, 20}, {4, 20}, {5, 20},
                {11, 24}, {12, 24}, {13, 24}},
        "raw COMPUTE_3D adapter changed STREAM_ND timing/induction");

    ComputeQueue accumulatorRead;
    const auto accumulatorReadTemplate =
        MxmControlInstruction::AccumulatorRead(
            40,
            20,
            false,
            MxmAccumulatorOutputFormat::BFloat16,
            MxmAccumulatorDestination::Stream);
    accumulatorRead.push_mxm_stream_nd(IcuMxmStreamNdSchedule {
        4, 2, {2, 3, 1}, {1, 4, 1}, {1, 8, 0},
        IcuInductionTarget::MxmAccumulatorAddress},
        accumulatorReadTemplate);
    require(accumulatorRead.imem_occupancy()
            == ComputeQueue::three_d_packet_word_count + 1,
        "legacy accumulator read did not lower to NOP plus one 3-D packet");
    std::vector<std::pair<std::size_t, std::size_t>> accumulatorReadIssues;
    for (std::size_t cycle = 0; cycle <= 13; ++cycle) {
        if (const auto instruction = accumulatorRead.tick()) {
            require(instruction->opcode == MxmControlOpcode::AccumulatorRead
                    && instruction->stream_base == 20
                    && !instruction->accumulator_clear
                    && instruction->accumulator_output_format
                        == MxmAccumulatorOutputFormat::BFloat16
                    && instruction->accumulator_destination
                        == MxmAccumulatorDestination::Stream,
                "raw ACCUMULATOR_READ_3D adapter changed fixed operands");
            accumulatorReadIssues.emplace_back(
                cycle, instruction->accumulator_address);
        }
    }
    require(accumulatorReadIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {4, 40}, {5, 41}, {8, 48},
                {9, 49}, {12, 56}, {13, 57}},
        "raw ACCUMULATOR_READ_3D adapter changed timing/address induction");

    require_invalid([&] {
        MemQueue unsupported;
        unsupported.push_mem_stream_nd(IcuMemStreamNdSchedule {},
            MemInstruction::Gather(0, 1));
    }, "raw MEM adapter accepted gather/scatter");
    require_invalid([&] {
        LoadQueue unsupported;
        unsupported.push_mxm_stream_nd(IcuMxmStreamNdSchedule {},
            MxmControlInstruction::IWColumn(0, 0, 0));
    }, "raw MXM load adapter accepted IWColumn");
    require_invalid([&] {
        ComputeQueue unsupported;
        unsupported.push_mxm_stream_nd(IcuMxmStreamNdSchedule {},
            MxmControlInstruction::DecodeLoadActivation());
    }, "raw MXM compute adapter accepted Decode");
    require_invalid([&] {
        MemQueue unsupported;
        unsupported.push_mem_slice_program(IcuMemSliceProgram {});
    }, "raw MEM queue silently accepted an unrepresentable slice program");

    std::cout << "legacy software descriptors lower to NOP plus zero-relative FU packets\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "legacy FU 3-D adapter test failed: "
              << error.what() << '\n';
    return 1;
}
