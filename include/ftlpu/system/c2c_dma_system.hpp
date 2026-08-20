#pragma once

#include "ftlpu/c2c/c2c.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>

namespace ftlpu {

// One chip with a DMA-controlled external C2C port on each hemisphere.
// Both DMA engines share the modeled DDR4 command/data interface.
class C2cDmaSystem {
public:
    explicit C2cDmaSystem(
        Ddr4Config ddr4_config = {},
        std::size_t dma_fifo_depth_vectors = 16,
        SystemHardwareConfiguration hardware = {})
        : ddr4_(ddr4_config)
        , dmas_ {
            C2cDmaEngine(ddr4_, dma_fifo_depth_vectors,
                hardware.c2c_streams_per_direction),
            C2cDmaEngine(ddr4_, dma_fifo_depth_vectors,
                hardware.c2c_streams_per_direction)}
        , chip_(hardware)
    {
        for (std::size_t index = 0; index < hw::kHemispheres; ++index) {
            const auto hemisphere = static_cast<Hemisphere>(index);
            chip_.attach_c2c_dma(hemisphere, dmas_[index]);
        }
    }

    TspSliceSystem& chip() noexcept { return chip_; }
    const TspSliceSystem& chip() const noexcept { return chip_; }

    Ddr4Model& ddr4() noexcept { return ddr4_; }
    const Ddr4Model& ddr4() const noexcept { return ddr4_; }

    C2cDmaEngine& dma(Hemisphere hemisphere) noexcept
    {
        return dmas_[hemisphere_index(hemisphere)];
    }
    const C2cDmaEngine& dma(Hemisphere hemisphere) const noexcept
    {
        return dmas_[hemisphere_index(hemisphere)];
    }

    std::size_t cycle() const noexcept { return cycle_; }

    void tick(TspSliceSystem::LogSinks sinks = {})
    {
        chip_.tick(sinks);
        ddr4_.tick();
        ++cycle_;
    }

    void reset_execution_state()
    {
        chip_.reset_execution_state();
        ddr4_.reset_execution_state();
        cycle_ = 0;
    }

private:
    Ddr4Model ddr4_;
    std::array<C2cDmaEngine, hw::kHemispheres> dmas_;
    TspSliceSystem chip_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
