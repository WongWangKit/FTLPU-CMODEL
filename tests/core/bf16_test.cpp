#include "ftlpu/core/bf16.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

int main()
{
    assert(ftlpu::Bf16::from_float(1.0f).bits() == 0x3f80u);
    assert(ftlpu::Bf16::from_float(-2.0f).bits() == 0xc000u);
    assert(ftlpu::Bf16::from_bits(0x4049u).to_float() == 3.140625f);

    // Halfway values round to an even retained mantissa bit.
    assert(ftlpu::Bf16::from_float(1.00390625f).bits() == 0x3f80u);
    assert(ftlpu::Bf16::from_float(1.01171875f).bits() == 0x3f82u);

    const auto wide = ftlpu::Bf16::from_float(1.0e20f).to_float();
    assert(std::isfinite(wide));
    assert(std::isinf(
        ftlpu::Bf16::from_float(std::numeric_limits<float>::infinity())
            .to_float()));
    assert(std::isnan(
        ftlpu::Bf16::from_float(std::numeric_limits<float>::quiet_NaN())
            .to_float()));
    return 0;
}
