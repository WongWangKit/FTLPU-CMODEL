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

} // namespace

int main()
{
    try {
        using Queue = ftlpu::DistributedIcuQueue<
            ftlpu::VxmCompactInstruction,
            ftlpu::hw::kIcuVxmInstructionBits,
            64, 8, 1, 1, ftlpu::IcuQueueRole::Vxm>;

        auto alu = ftlpu::VxmLaneAluInstruction {};
        alu.operation = ftlpu::VxmAluOpcode::Multiply;
        alu.lhs = ftlpu::VxmLaneOperand::StreamBFloat16();
        alu.rhs = ftlpu::VxmLaneOperand::Imm(0.5f);
        constexpr auto stage = std::size_t {0};
        alu.repeat_count = 32;
        const auto compact = ftlpu::VxmCompactInstructionCodec::encode(
            stage, ftlpu::VxmChainDepth::Two, alu);
        const auto run = ftlpu::VxmIcuRun2DInstruction::Run2D(
            0, {3, 2}, {2, 12}, compact);
        const auto packet =
            ftlpu::isa::encode_vxm_icu_run_2d_instruction(run);
        const auto decoded =
            ftlpu::isa::decode_vxm_icu_run_2d_instruction(packet);
        require(decoded.loop.start_cycle == 0
                && decoded.loop.counts
                    == std::array<std::size_t, 3> {3, 2, 1}
                && decoded.loop.cycle_strides
                    == std::array<std::size_t, 3> {2, 12, 1}
                && decoded.instruction == compact,
            "VXM RUN_2D codec round trip changed the packet");

        Queue queue;
        queue.push_nop(4);
        queue.push_vxm_run_2d(run);
        require(queue.imem_occupancy() == 4,
            "VXM launch delay plus RUN_2D must occupy four local i-MEM words");
        std::vector<std::size_t> issueCycles;
        for (std::size_t cycle = 0; cycle < 32; ++cycle) {
            const auto instruction = queue.dispatch_next();
            if (!instruction) continue;
            require(*instruction == compact,
                "VXM RUN_2D expanded a different compact config");
            issueCycles.push_back(cycle);
        }
        require(issueCycles
                == std::vector<std::size_t> {4, 6, 8, 16, 18, 20},
            "VXM RUN_2D issued at the wrong cycles");
        auto waitedRun = run;
        waitedRun.loop.wait_cycle = 4;
        const auto waitedPacket =
            ftlpu::isa::encode_vxm_icu_run_2d_instruction(waitedRun);
        require(ftlpu::isa::decode_vxm_icu_run_2d_instruction(
                    waitedPacket).loop.wait_cycle == 4,
            "VXM RUN_2D lost its encoded wait_cycle");
        Queue waitedQueue;
        waitedQueue.push_vxm_run_2d(waitedRun);
        require(waitedQueue.imem_occupancy() == 3,
            "VXM wait_cycle should replace the NOP i-MEM word");
        std::vector<std::size_t> waitedIssueCycles;
        for (std::size_t cycle = 0; cycle < 32; ++cycle)
            if (waitedQueue.dispatch_next())
                waitedIssueCycles.push_back(cycle);
        require(waitedIssueCycles == issueCycles,
            "VXM wait_cycle changed functional issue timing");
        require(queue.peak_active_3d_contexts() == 1,
            "VXM RUN_2D must use one decoded context");
        require(queue.active_3d_context_count() == 0,
            "VXM RUN_2D did not retire its context");

        auto absoluteStart = run;
        absoluteStart.loop.start_cycle = 4;
        bool rejectedAbsoluteStart = false;
        try {
            (void)ftlpu::isa::encode_vxm_icu_run_2d_instruction(
                absoluteStart);
        } catch (const std::invalid_argument&) {
            rejectedAbsoluteStart = true;
        }
        require(rejectedAbsoluteStart,
            "VXM RUN_2D accepted an absolute start cycle");

        auto invalid = run;
        invalid.loop.counts[2] = 2;
        bool rejected = false;
        try {
            (void)ftlpu::isa::encode_vxm_icu_run_2d_instruction(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected,
            "VXM RUN_2D accepted a hidden third launch counter");

        std::cout << "vxm_run_2d_instruction_test passed: "
                  << "launches=6 run_length=32 packet_words=3 program_words=4\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vxm_run_2d_instruction_test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
