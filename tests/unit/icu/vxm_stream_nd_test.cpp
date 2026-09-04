#include "ftlpu/icu/distributed_queue.hpp"

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
try {
    using namespace ftlpu;
    using Queue = DistributedIcuQueue<
        VxmCompactInstruction, 96, 64, 16, 1>;

    VxmLaneAluInstruction lane {};
    lane.operation = VxmAluOpcode::Multiply;
    lane.lhs = VxmLaneOperand::StreamBFloat16(1.0f, 0);
    lane.rhs = VxmLaneOperand::Imm(0.5f);
    lane.output_type = VxmCastTarget::BFloat16;
    lane.repeat_count = 32;
    const auto packet = VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Eight, lane);

    Queue queue;
    queue.push_vxm_stream_nd(IcuVxmStreamNdSchedule {
        3, 3, {2, 2, 2}, {7, 19, 43}, {0, 0, 0},
        IcuInductionTarget::None}, packet);

    std::vector<std::size_t> issueCycles;
    for (std::size_t cycle = 0; cycle <= 72; ++cycle) {
        if (const auto instruction = queue.tick()) {
            require(*instruction == packet,
                "VXM_STREAM_ND changed the compact config packet");
            const auto decoded = VxmCompactInstructionCodec::decode(
                0, *instruction);
            require(decoded.instruction.repeat_count == 32,
                "VXM_STREAM_ND expanded the Superlane repeat count");
            require(queue.last_trace().action
                    == IcuQueueAction::VxmStreamNdIssue,
                "VXM_STREAM_ND emitted the wrong queue trace action");
            issueCycles.push_back(cycle);
        }
    }
    require(issueCycles == std::vector<std::size_t> {
                3, 10, 22, 29, 46, 53, 65, 72},
        "VXM_STREAM_ND emitted at incorrect cycles");
    require(queue.done(), "VXM_STREAM_ND queue did not complete");

    std::cout << "icu_vxm_stream_nd_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_vxm_stream_nd_test failed: "
              << error.what() << '\n';
    return 1;
}
