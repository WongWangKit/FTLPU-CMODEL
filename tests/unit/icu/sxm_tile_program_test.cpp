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
    using Queue = DistributedIcuQueue<SxmInstruction, 96, 64, 16, 1>;

    auto map = SxmInstruction::PermuteMap {};
    for (std::size_t lane = 0; lane < map.size(); ++lane)
        map[lane] = (lane + hw::kLanesPerTile) % map.size();
    const auto instruction = SxmInstruction::Permute(
        {{0}, {1}}, {{16}, {17}}, map);

    Queue queue;
    queue.push_sxm_tile_program(IcuSxmTileProgramSchedule {
        4, 2, {3, 2, 1}, {2, 11, 1}, {0, 0, 0},
        IcuInductionTarget::None}, instruction);
    std::vector<std::size_t> issueCycles;
    for (std::size_t cycle = 0; cycle <= 19; ++cycle) {
        if (const auto issued = queue.tick()) {
            require(issued->opcode == SxmOpcode::Permute
                    && issued->permute_map == map,
                "SXM_TILE_PROGRAM changed the tile map");
            require(queue.last_trace().action
                    == IcuQueueAction::SxmTileProgramIssue,
                "SXM_TILE_PROGRAM emitted the wrong trace action");
            issueCycles.push_back(cycle);
        }
    }
    require(issueCycles == std::vector<std::size_t> {
                4, 6, 8, 15, 17, 19},
        "SXM_TILE_PROGRAM emitted at incorrect cycles");
    require(queue.done(), "SXM_TILE_PROGRAM queue did not complete");

    std::cout << "icu_sxm_tile_program_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_sxm_tile_program_test failed: "
              << error.what() << '\n';
    return 1;
}
