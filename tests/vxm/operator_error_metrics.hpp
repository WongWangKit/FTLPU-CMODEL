#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>

// Reusable element-wise accuracy accumulator for VXM operator tests.
// A new workload only needs to call observe(actual, expected) per output.
struct VxmOperatorErrorMetrics {
    std::size_t samples{0};
    double max_absolute_error{0.0};
    double sum_absolute_error{0.0};
    double sum_squared_error{0.0};
    double max_relative_error{0.0};
    double sum_relative_error{0.0};
    double sum_squared_relative_error{0.0};

    void observe(double actual, double expected)
    {
        const auto absolute = std::fabs(actual - expected);
        // The floor keeps values whose mathematical reference is near zero
        // from producing an infinite or undefined relative error.
        const auto relative = absolute
            / std::max(std::fabs(expected), 1.0e-6);
        ++samples;
        max_absolute_error = std::max(max_absolute_error, absolute);
        sum_absolute_error += absolute;
        sum_squared_error += absolute * absolute;
        max_relative_error = std::max(max_relative_error, relative);
        sum_relative_error += relative;
        sum_squared_relative_error += relative * relative;
    }

    double mean_absolute_error() const
    {
        return samples == 0 ? 0.0
                            : sum_absolute_error / static_cast<double>(samples);
    }

    double root_mean_squared_error() const
    {
        return samples == 0 ? 0.0
            : std::sqrt(sum_squared_error / static_cast<double>(samples));
    }

    double mean_relative_error() const
    {
        return samples == 0 ? 0.0
                            : sum_relative_error / static_cast<double>(samples);
    }

    double root_mean_squared_relative_error() const
    {
        return samples == 0 ? 0.0
            : std::sqrt(sum_squared_relative_error
                        / static_cast<double>(samples));
    }
};

inline void print_vxm_operator_error(
    std::ostream& os, const VxmOperatorErrorMetrics& error)
{
    const auto old_flags = os.flags();
    const auto old_precision = os.precision();
    os << "  Error samples=" << error.samples
       << " mean_rel=" << std::scientific << std::setprecision(6)
       << error.mean_relative_error()
       << " rms_rel=" << error.root_mean_squared_relative_error()
       << " max_rel=" << error.max_relative_error
       << " max_abs=" << error.max_absolute_error << '\n';
    os.flags(old_flags);
    os.precision(old_precision);
}
