#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/gemm_engine.hpp"

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <string>

namespace ftlpu {

class MxmPerformanceMonitor {
public:
    void sample(const MxmGemmEngine& gemm)
    {
        std::size_t active = 0;
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            for (std::size_t column = 0; column < hw::kMxmSupercellsPerPlane; ++column) {
                if (gemm.computing_cell(tile, column)) {
                    ++active;
                }
            }
        }

        sample_active_cells(active);
    }

    void sample_idle()
    {
        sample_active_cells(0);
    }

    void sample_active_cells(std::size_t active)
    {
        ++sampled_cycles_;
        active_cell_cycles_ += active;
        if (active != 0) {
            ++non_idle_cycles_;
        }
        if (active > peak_active_cells_) {
            peak_active_cells_ = active;
        }
    }

    std::size_t sampled_cycles() const
    {
        return sampled_cycles_;
    }

    std::size_t non_idle_cycles() const
    {
        return non_idle_cycles_;
    }

    std::size_t active_cell_cycles() const
    {
        return active_cell_cycles_;
    }

    std::size_t peak_active_cells() const
    {
        return peak_active_cells_;
    }

    double array_utilization() const
    {
        if (sampled_cycles_ == 0) {
            return 0.0;
        }
        return static_cast<double>(active_cell_cycles_)
            / static_cast<double>(sampled_cycles_ * kArrayCells);
    }

    double active_cycle_density() const
    {
        if (non_idle_cycles_ == 0) {
            return 0.0;
        }
        return static_cast<double>(active_cell_cycles_)
            / static_cast<double>(non_idle_cycles_ * kArrayCells);
    }

    void print(std::ostream& os, const std::string& label) const
    {
        const auto old_flags = os.flags();
        const auto old_precision = os.precision();
        os << std::fixed << std::setprecision(2)
           << label
           << " perf cycles=" << sampled_cycles_
           << " active_cycles=" << non_idle_cycles_
           << " active_cell_cycles=" << active_cell_cycles_
           << " peak_active_cells=" << peak_active_cells_ << "/" << kArrayCells
           << " array_util=" << (array_utilization() * 100.0) << "%"
           << " active_density=" << (active_cycle_density() * 100.0) << "%"
           << '\n';
        os.flags(old_flags);
        os.precision(old_precision);
    }

private:
    static constexpr std::size_t kArrayCells = hw::kMxmSupercellsPerPlane * hw::kMxmSupercellsPerPlane;

    std::size_t sampled_cycles_{0};
    std::size_t non_idle_cycles_{0};
    std::size_t active_cell_cycles_{0};
    std::size_t peak_active_cells_{0};
};

} // namespace ftlpu
