#include "ftlpu/system/tsp_slice_system.hpp"
#include "ftlpu/vxm/contract.hpp"

#include <cassert>
#include <memory>

int main()
{
    static_assert(ftlpu::hw::kMxmCount == 4);
    static_assert(ftlpu::hw::kWestMxmCount == 2);
    static_assert(ftlpu::hw::kEastMxmCount == 2);
    auto system = std::make_unique<ftlpu::TspSliceSystem>();

    const auto& west0 = system->mxm_unit(0).ports().weight_input();
    const auto& west1 = system->mxm_unit(1).ports().weight_input();
    const auto& east0 = system->mxm_unit(2).ports().weight_input();
    const auto& east1 = system->mxm_unit(3).ports().weight_input();
    assert(west0.direction == ftlpu::StreamDirection::West);
    assert(west0.stream_base == 0);
    assert(west1.direction == ftlpu::StreamDirection::West);
    assert(west1.stream_base == 16);
    assert(east0.direction == ftlpu::StreamDirection::East);
    assert(east0.stream_base == 0);
    assert(east1.direction == ftlpu::StreamDirection::East);
    assert(east1.stream_base == 16);

    assert(system->mem().memory_model().sram_lane_byte(
        0,
        0,
        ftlpu::MemLocalWordAddress13::FromFields(1, 40959),
        0) == 0);
    static_assert(ftlpu::VxmInterfaceContract::lane_count == 8);
    system->tick(ftlpu::TspSliceSystem::LogSinks {});
    assert(system->cycle() == 1);
    return 0;
}
