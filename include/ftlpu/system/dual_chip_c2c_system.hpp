#pragma once

#include "ftlpu/c2c/c2c.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace ftlpu {

class DualChipC2cSystem {
public:
    explicit DualChipC2cSystem(C2cLinkConfig link_config = {})
        : chip_0_to_1_(link_config)
        , chip_1_to_0_(link_config)
    {
        chips_[0].attach_c2c(
            Hemisphere::East,
            C2cStreamPortMap::EastEdge(
                hw::kMxmBoundaryStreamRegisterColumn),
            chip_0_to_1_,
            chip_1_to_0_);
        chips_[1].attach_c2c(
            Hemisphere::West,
            C2cStreamPortMap::WestEdge(
                hw::kMemWestBoundaryStreamRegisterColumn),
            chip_1_to_0_,
            chip_0_to_1_);
    }

    TspSliceSystem& chip(std::size_t index) { return chips_.at(index); }
    const TspSliceSystem& chip(std::size_t index) const
    {
        return chips_.at(index);
    }

    C2cLink& link_0_to_1() noexcept { return chip_0_to_1_; }
    C2cLink& link_1_to_0() noexcept { return chip_1_to_0_; }

    std::size_t cycle() const noexcept { return cycle_; }

    void tick()
    {
        chips_[0].tick(TspSliceSystem::LogSinks {});
        chips_[1].tick(TspSliceSystem::LogSinks {});
        chip_0_to_1_.tick();
        chip_1_to_0_.tick();
        ++cycle_;
    }

    void reset_execution_state()
    {
        chips_[0].reset_execution_state();
        chips_[1].reset_execution_state();
        chip_0_to_1_.reset();
        chip_1_to_0_.reset();
        cycle_ = 0;
    }

private:
    std::array<TspSliceSystem, 2> chips_ {
        TspSliceSystem {}, TspSliceSystem {}};
    C2cLink chip_0_to_1_;
    C2cLink chip_1_to_0_;
    std::size_t cycle_{0};
};

} // namespace ftlpu
