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
        : chip_0_to_1_ {
            C2cLink(link_config), C2cLink(link_config)}
        , chip_1_to_0_ {
            C2cLink(link_config), C2cLink(link_config)}
    {
        for (std::size_t source_index = 0;
             source_index < hw::kHemispheres;
             ++source_index) {
            const auto source = static_cast<Hemisphere>(source_index);
            const auto peer = static_cast<Hemisphere>(
                hw::kHemispheres - 1 - source_index);
            chips_[0].attach_c2c(
                source,
                chip_0_to_1_[source_index],
                chip_1_to_0_[hemisphere_index(peer)]);
            chips_[1].attach_c2c(
                peer,
                chip_1_to_0_[hemisphere_index(peer)],
                chip_0_to_1_[source_index]);
        }
    }

    TspSliceSystem& chip(std::size_t index) { return chips_.at(index); }
    const TspSliceSystem& chip(std::size_t index) const
    {
        return chips_.at(index);
    }

    C2cLink& link_0_to_1(Hemisphere source) noexcept
    {
        return chip_0_to_1_[hemisphere_index(source)];
    }
    C2cLink& link_1_to_0(Hemisphere source) noexcept
    {
        return chip_1_to_0_[hemisphere_index(source)];
    }
    C2cLink& link_0_to_1() noexcept
    {
        return link_0_to_1(Hemisphere::East);
    }
    C2cLink& link_1_to_0() noexcept
    {
        return link_1_to_0(Hemisphere::West);
    }

    std::size_t cycle() const noexcept { return cycle_; }

    void tick()
    {
        chips_[0].tick(TspSliceSystem::LogSinks {});
        chips_[1].tick(TspSliceSystem::LogSinks {});
        for (auto& link : chip_0_to_1_) link.tick();
        for (auto& link : chip_1_to_0_) link.tick();
        ++cycle_;
    }

    void reset_execution_state()
    {
        chips_[0].reset_execution_state();
        chips_[1].reset_execution_state();
        for (auto& link : chip_0_to_1_) link.reset();
        for (auto& link : chip_1_to_0_) link.reset();
        cycle_ = 0;
    }

private:
    std::array<TspSliceSystem, 2> chips_ {
        TspSliceSystem {}, TspSliceSystem {}};
    std::array<C2cLink, hw::kHemispheres> chip_0_to_1_;
    std::array<C2cLink, hw::kHemispheres> chip_1_to_0_;
    std::size_t cycle_{0};
};

} // namespace ftlpu
