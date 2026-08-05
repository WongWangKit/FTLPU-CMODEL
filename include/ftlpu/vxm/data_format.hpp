#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ftlpu {

enum class VxmCastTarget {
    Float32,
    Float16,
    Int8,
};

// Input/output format conversion is boundary hardware, not an ALU operation.
class VxmDataFormat {
public:
    static std::int8_t quantize_int8(float input, float scale,
                                     std::int32_t zero_point = 0)
    {
        if (scale == 0.0f) {
            throw std::invalid_argument("VXM quantize scale must be non-zero");
        }
        return cast_int8(input / scale + static_cast<float>(zero_point));
    }

    static std::int8_t cast_int8(float input)
    {
        if (std::isnan(input)) return 0;
        const auto rounded = static_cast<long>(std::nearbyint(input));
        const auto clipped = std::clamp<long>(rounded,
            std::numeric_limits<std::int8_t>::min(),
            std::numeric_limits<std::int8_t>::max());
        return static_cast<std::int8_t>(clipped);
    }

    static std::uint16_t float_to_fp16_bits(float input)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &input, sizeof(bits));
        const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
        const auto exponent = static_cast<int>((bits >> 23) & 0xffu);
        auto fraction = bits & 0x7fffffu;

        if (exponent == 0xff) {
            return static_cast<std::uint16_t>(sign | (fraction ? 0x7e00u : 0x7c00u));
        }
        int half_exponent = exponent - 127 + 15;
        if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        if (half_exponent <= 0) {
            if (half_exponent < -10) return sign;
            fraction |= 0x800000u;
            const auto shift = static_cast<unsigned>(14 - half_exponent);
            auto half_fraction = fraction >> shift;
            const auto remainder = fraction & ((std::uint32_t {1} << shift) - 1u);//提取剩余部分
            const auto halfway = std::uint32_t {1} << (shift - 1u);
            if (remainder > halfway || (remainder == halfway && (half_fraction & 1u))) {
                ++half_fraction;
            }
            return static_cast<std::uint16_t>(sign | half_fraction);
        }

        auto half_fraction = fraction >> 13;
        const auto remainder = fraction & 0x1fffu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (half_fraction & 1u))) {
            ++half_fraction;
            if (half_fraction == 0x400u) {
                half_fraction = 0;
                ++half_exponent;
                if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
        }
        return static_cast<std::uint16_t>(sign
            | (static_cast<std::uint16_t>(half_exponent) << 10)
            | static_cast<std::uint16_t>(half_fraction));
    }

    static float fp16_bits_to_float(std::uint16_t bits)
    {
        const auto sign = (bits & 0x8000u) ? -1.0f : 1.0f;
        const auto exponent = static_cast<unsigned>((bits >> 10) & 0x1fu);
        const auto fraction = static_cast<unsigned>(bits & 0x03ffu);
        if (exponent == 0) {
            return fraction == 0 ? std::copysign(0.0f, sign)
                                 : sign * std::ldexp(static_cast<float>(fraction), -24);
        }
        if (exponent == 31) {
            return fraction == 0 ? sign * std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::quiet_NaN();
        }
        return sign * std::ldexp(1.0f + static_cast<float>(fraction) / 1024.0f,
                                  static_cast<int>(exponent) - 15);
    }

    static float round_fp16_ftz(float input)
    {
        constexpr float kMinNormal = 0x1p-14f;
        if (std::isfinite(input) && std::fabs(input) < kMinNormal) {
            return std::copysign(0.0f, input);
        }
        const auto bits = float_to_fp16_bits(input);
        if ((bits & 0x7c00u) == 0) return std::copysign(0.0f, input);
        return fp16_bits_to_float(bits);
    }
};

} // namespace ftlpu
