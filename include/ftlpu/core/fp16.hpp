#pragma once

#include <bit>
#include <cstdint>

namespace ftlpu {

// IEEE 754 binary16 storage with deterministic conversion on hosts that do
// not expose a native half-precision arithmetic type.
class Fp16 {
public:
    constexpr Fp16() = default;

    static constexpr Fp16 from_bits(std::uint16_t bits) noexcept
    {
        return Fp16(bits);
    }

    static Fp16 from_float(float value) noexcept
    {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
        const auto exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
        const auto mantissa = bits & 0x7fffffu;

        if (((bits >> 23) & 0xffu) == 0xffu) {
            const auto payload = mantissa == 0 ? 0u : ((mantissa >> 13) | 1u);
            return Fp16(static_cast<std::uint16_t>(sign | 0x7c00u | payload));
        }
        if (exponent >= 31) {
            return Fp16(static_cast<std::uint16_t>(sign | 0x7c00u));
        }
        if (exponent <= 0) {
            if (exponent < -10) {
                return Fp16(sign);
            }
            const auto normalized = mantissa | 0x800000u;
            const auto shift = static_cast<unsigned>(14 - exponent);
            auto half_mantissa = normalized >> shift;
            const auto remainder = normalized & ((1u << shift) - 1u);
            const auto halfway = 1u << (shift - 1u);
            if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u) != 0)) {
                ++half_mantissa;
            }
            return Fp16(static_cast<std::uint16_t>(sign | half_mantissa));
        }

        auto half = static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(exponent) << 10)
            | static_cast<std::uint16_t>(mantissa >> 13));
        const auto remainder = mantissa & 0x1fffu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u) != 0)) {
            ++half;
        }
        return Fp16(half);
    }

    constexpr std::uint16_t bits() const noexcept
    {
        return bits_;
    }

    float to_float() const noexcept
    {
        const auto sign = static_cast<std::uint32_t>(bits_ & 0x8000u) << 16;
        const auto exponent = (bits_ >> 10) & 0x1fu;
        auto mantissa = static_cast<std::uint32_t>(bits_ & 0x03ffu);
        std::uint32_t result = 0;

        if (exponent == 0) {
            if (mantissa == 0) {
                result = sign;
            } else {
                auto normalized_exponent = -14;
                while ((mantissa & 0x0400u) == 0) {
                    mantissa <<= 1;
                    --normalized_exponent;
                }
                mantissa &= 0x03ffu;
                result = sign
                    | (static_cast<std::uint32_t>(normalized_exponent + 127) << 23)
                    | (mantissa << 13);
            }
        } else if (exponent == 0x1fu) {
            result = sign | 0x7f800000u | (mantissa << 13);
        } else {
            result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
        return std::bit_cast<float>(result);
    }

private:
    explicit constexpr Fp16(std::uint16_t bits) noexcept
        : bits_(bits)
    {
    }

    std::uint16_t bits_{0};
};

} // namespace ftlpu
