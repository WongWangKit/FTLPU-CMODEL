#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

int main()
{
    using namespace ftlpu;
    auto system = std::make_unique<TspSliceSystem>();
    constexpr auto stream = MxmActivationDequantizer::kBroadcastInputStream;
    for (const auto side : {Hemisphere::West, Hemisphere::East}) {
        auto& mem = system->mem(side).memory_model();
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            mem.set_sram_lane_byte(0, 0, 0, lane, 3);
        }
        mem.enqueue_instruction(
            0, MemInstruction::Read(
                0, side == Hemisphere::West
                    ? StreamId::West(stream)
                    : StreamId::East(stream)));
    }
    system->tick({});
    assert(system->mem(Hemisphere::West).stream_fabric().segment_valid(
        0, 0, StreamId::West(stream)));
    // East slice zero emerges at column one and then takes ten passive hops.
    for (std::size_t cycle = 0; cycle < 10; ++cycle) system->tick({});
    assert(system->mem(Hemisphere::East).stream_fabric().segment_valid(
        hw::kMemBoundaryStreamRegisterColumns - 1,
        0, StreamId::East(stream)));
    return 0;
}
