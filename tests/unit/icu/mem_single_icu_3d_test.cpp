#include "ftlpu/system/icu.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() try
{
    using namespace ftlpu;

    InstructionControlUnit icu;
    const auto queueIndex = InstructionControlUnit::mem_queue(
        Hemisphere::East, 3, 1);
    const IcuLoop3D loop {0, {4, 1, 1}, {1, 1, 1}};
    icu.enqueue_mem_3d(queueIndex, MemIcuInstruction::Read3D(
        loop, MemIcuAddress3D::Affine(10, {1, 0, 0}),
        StreamId::East(0)));
    auto waitedWrite = loop;
    waitedWrite.wait_cycle = 5;
    icu.enqueue_mem_3d(queueIndex, MemIcuInstruction::Write3D(
        waitedWrite, MemIcuAddress3D::Affine(20, {1, 0, 0}),
        StreamId::West(0)));

    auto& queue = icu.mem_iq(queueIndex);
    require(queue.imem_occupancy()
            == 2 * InstructionControlUnit::MemIcu::three_d_packet_word_count,
        "read and write 3-D commands did not share one MEM i-MEM");

    std::size_t issued = 0;
    std::vector<std::size_t> issueCycles;
    for (std::size_t cycle = 0; cycle < 32 && !queue.done(); ++cycle) {
        const auto instruction = queue.tick();
        if (!instruction) continue;
        issueCycles.push_back(cycle);
        if (issued < 4) {
            require(instruction->opcode == MemOpcode::Read
                    && instruction->address == 10 + issued,
                "single MEM ICU did not finish the read domain first");
        } else {
            require(instruction->opcode == MemOpcode::Write
                    && instruction->address == 20 + issued - 4,
                "single MEM ICU did not serialize the following write domain");
        }
        ++issued;
    }
    require(issued == 8 && queue.done(),
        "single MEM ICU did not complete both serialized 3-D domains");
    require(issueCycles == std::vector<std::size_t> {0, 1, 2, 3,
                9, 10, 11, 12},
        "MEM wait_cycle did not replace an inter-domain NOP");

    std::cout << "MEM single-ICU 3-D serialization test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "MEM single-ICU test failed: " << error.what() << '\n';
    return 1;
}
