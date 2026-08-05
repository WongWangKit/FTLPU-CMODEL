#include "ftlpu/mxm/mxm.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

int main()
{
    using ftlpu::MxmOutputCast;
    using ftlpu::VxmDataFormat;

    auto mxm = std::make_unique<ftlpu::Mxm>();
    (void)mxm;
    static_assert(MxmOutputCast::kLatency == 1);
    static_assert(MxmOutputCast::kOutputBytes == 2);

    for (const auto value :
         {std::int32_t{-17}, std::int32_t{0}, std::int32_t{23},
          std::int32_t{1024}, std::int32_t{32767}}) {
        const auto bits = MxmOutputCast::cast(value);
        const auto bytes = MxmOutputCast::bytes(value);
        assert(bits == (static_cast<std::uint16_t>(bytes[0])
                     | (static_cast<std::uint16_t>(bytes[1]) << 8)));
        const auto restored = VxmDataFormat::fp16_bits_to_float(bits);
        const auto expected =
            VxmDataFormat::round_fp16_ftz(static_cast<float>(value));
        assert(restored == expected);
    }

    const auto overflow = MxmOutputCast::cast(
        std::numeric_limits<std::int32_t>::max());
    assert(std::isinf(VxmDataFormat::fp16_bits_to_float(overflow)));

    const auto float_bits = MxmOutputCast::cast(1.5f);
    const auto float_bytes = MxmOutputCast::bytes(1.5f);
    assert(
        float_bits
        == (static_cast<std::uint16_t>(float_bytes[0])
            | (static_cast<std::uint16_t>(float_bytes[1]) << 8)));
    assert(VxmDataFormat::fp16_bits_to_float(float_bits) == 1.5f);

    const auto combined = MxmOutputCast::combined_dequant_scale(
        0.25f, 0.5f);
    assert(combined == 0.125f);
    assert(VxmDataFormat::fp16_bits_to_float(
               MxmOutputCast::cast(std::int32_t {40}, combined))
        == 5.0f);
    return 0;
}
