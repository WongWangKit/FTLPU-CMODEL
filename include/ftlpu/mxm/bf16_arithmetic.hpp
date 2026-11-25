#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ftlpu::mxm_numeric {

struct Bf16DotResult {
    std::uint32_t bits{};
    bool alignment_lost{};

    float value() const noexcept
    {
        return std::bit_cast<float>(bits);
    }
};

inline unsigned leading_one(std::uint64_t value) noexcept
{
    return value == 0 ? 0u : 63u - static_cast<unsigned>(std::countl_zero(value));
}

inline std::uint64_t shift_right_sticky(
    std::uint64_t value, unsigned distance) noexcept
{
    if (distance == 0) return value;
    if (distance >= 64) return value == 0 ? 0u : 1u;
    const auto discarded_mask = (std::uint64_t {1} << distance) - 1u;
    return (value >> distance) | ((value & discarded_mask) != 0 ? 1u : 0u);
}

// Bit-exact equivalent of lpu_mxm_supercell_math_pkg::pack_fp32_rne.
inline std::uint32_t pack_fp32_rne(
    bool sign, int exponent, std::uint64_t magnitude) noexcept
{
    auto adjusted = magnitude;
    auto working_exponent = exponent;
    if (working_exponent < -126) {
        adjusted = shift_right_sticky(
            adjusted, static_cast<unsigned>(-126 - working_exponent));
        working_exponent = -126;
    }

    auto rounded = adjusted >> 40;
    const auto guard = ((adjusted >> 39) & 1u) != 0;
    const auto sticky = (adjusted & ((std::uint64_t {1} << 39) - 1u)) != 0;
    if (guard && (sticky || (rounded & 1u) != 0)) ++rounded;
    if ((rounded & (std::uint64_t {1} << 24)) != 0) {
        rounded >>= 1;
        ++working_exponent;
    }

    if (magnitude == 0 || rounded == 0) {
        return static_cast<std::uint32_t>(sign) << 31;
    }
    if (working_exponent > 127) {
        return (static_cast<std::uint32_t>(sign) << 31) | 0x7f800000u;
    }
    auto encoded_exponent = static_cast<std::uint32_t>(working_exponent + 127);
    if (working_exponent == -126 && (rounded & (std::uint64_t {1} << 23)) == 0) {
        encoded_exponent = 0;
    }
    return (static_cast<std::uint32_t>(sign) << 31)
        | (encoded_exponent << 23)
        | static_cast<std::uint32_t>(rounded & 0x7fffffu);
}

// BF16 input DAZ, common-exponent alignment, finite 64-bit reduction and one
// final FP32 RNE boundary, matching lpu_mxm_bf16_dot_csa.
inline Bf16DotResult bf16_dot(
    std::span<const std::uint16_t> activations,
    std::span<const std::uint16_t> weights)
{
    struct Product {
        std::uint16_t significand{};
        int exponent{};
        bool sign{};
    };

    if (activations.size() != weights.size() || activations.empty()) {
        return {0x7fc00000u, false};
    }

    std::vector<Product> products;
    products.reserve(activations.size());
    bool nan = false;
    bool positive_infinity = false;
    bool negative_infinity = false;
    for (std::size_t term = 0; term < activations.size(); ++term) {
        const auto a = activations[term];
        const auto b = weights[term];
        const auto a_exponent = static_cast<unsigned>((a >> 7) & 0xffu);
        const auto b_exponent = static_cast<unsigned>((b >> 7) & 0xffu);
        const auto a_fraction = static_cast<unsigned>(a & 0x7fu);
        const auto b_fraction = static_cast<unsigned>(b & 0x7fu);
        const auto sign = ((a ^ b) & 0x8000u) != 0;
        const auto a_zero = a_exponent == 0; // DAZ includes BF16 subnormals.
        const auto b_zero = b_exponent == 0;
        const auto a_infinity = a_exponent == 0xffu && a_fraction == 0;
        const auto b_infinity = b_exponent == 0xffu && b_fraction == 0;
        const auto a_nan = a_exponent == 0xffu && a_fraction != 0;
        const auto b_nan = b_exponent == 0xffu && b_fraction != 0;

        nan = nan || a_nan || b_nan
            || (a_zero && b_infinity) || (b_zero && a_infinity);
        if (!a_nan && !b_nan && !a_zero && !b_zero
            && (a_infinity || b_infinity)) {
            positive_infinity = positive_infinity || !sign;
            negative_infinity = negative_infinity || sign;
        }
        if (a_nan || b_nan || a_infinity || b_infinity || a_zero || b_zero) {
            continue;
        }

        auto product = static_cast<std::uint16_t>(
            (0x80u + a_fraction) * (0x80u + b_fraction));
        auto exponent = static_cast<int>(a_exponent)
            + static_cast<int>(b_exponent) - 254;
        if ((product & 0x8000u) == 0) {
            product = static_cast<std::uint16_t>(product << 1);
        } else {
            ++exponent;
        }
        products.push_back(Product {product, exponent, sign});
    }

    const auto growth = static_cast<unsigned>(
        std::bit_width(activations.size() - 1));
    const auto aligned_bits = 63u - growth;
    int common_exponent = 0;
    if (!products.empty()) {
        common_exponent = products.front().exponent;
        for (const auto& product : products) {
            if (product.exponent > common_exponent) {
                common_exponent = product.exponent;
            }
        }
    }

    std::int64_t total = 0;
    bool lost = false;
    for (const auto& product : products) {
        const auto unshifted = static_cast<std::uint64_t>(product.significand)
            << (aligned_bits - 16u);
        const auto distance = static_cast<unsigned>(
            common_exponent - product.exponent);
        const auto magnitude = distance >= 64 ? 0u : unshifted >> distance;
        if (distance >= 64) {
            lost = lost || unshifted != 0;
        } else if (distance != 0) {
            const auto mask = (std::uint64_t {1} << distance) - 1u;
            lost = lost || (unshifted & mask) != 0;
        }
        const auto signed_magnitude = static_cast<std::int64_t>(magnitude);
        total += product.sign ? -signed_magnitude : signed_magnitude;
    }

    if (nan || (positive_infinity && negative_infinity)) {
        return {0x7fc00000u, lost};
    }
    if (positive_infinity || negative_infinity) {
        return {
            (static_cast<std::uint32_t>(negative_infinity) << 31)
                | 0x7f800000u,
            lost};
    }

    const auto sign = total < 0;
    const auto magnitude = sign
        ? static_cast<std::uint64_t>(-(total + 1)) + 1u
        : static_cast<std::uint64_t>(total);
    if (magnitude == 0) return {0u, lost};
    const auto leading = leading_one(magnitude);
    const auto normalized = magnitude << (63u - leading);
    const auto exponent = common_exponent
        + static_cast<int>(leading)
        - static_cast<int>(aligned_bits - 1u);
    return {pack_fp32_rne(sign, exponent, normalized), lost};
}

template <std::size_t Terms>
inline Bf16DotResult bf16_dot(
    const std::array<std::uint16_t, Terms>& activations,
    const std::array<std::uint16_t, Terms>& weights)
{
    return bf16_dot(
        std::span<const std::uint16_t>(activations),
        std::span<const std::uint16_t>(weights));
}

// Bit-exact three-stage MXM FP32 adder arithmetic, excluding its latency.
inline std::uint32_t fp32_add_bits(
    std::uint32_t lhs, std::uint32_t rhs) noexcept
{
    const auto lhs_nan = (lhs & 0x7f800000u) == 0x7f800000u
        && (lhs & 0x007fffffu) != 0;
    const auto rhs_nan = (rhs & 0x7f800000u) == 0x7f800000u
        && (rhs & 0x007fffffu) != 0;
    const auto lhs_inf = (lhs & 0x7fffffffu) == 0x7f800000u;
    const auto rhs_inf = (rhs & 0x7fffffffu) == 0x7f800000u;
    if (lhs_nan || rhs_nan
        || (lhs_inf && rhs_inf && ((lhs ^ rhs) & 0x80000000u) != 0)) {
        return 0x7fc00000u;
    }
    if (lhs_inf) return lhs;
    if (rhs_inf) return rhs;
    if ((lhs & 0x7fffffffu) == 0 && (rhs & 0x7fffffffu) == 0) {
        return (lhs & rhs) & 0x80000000u;
    }
    if ((lhs & 0x7fffffffu) == 0) return rhs;
    if ((rhs & 0x7fffffffu) == 0) return lhs;

    const auto lhs_magnitude = lhs & 0x7fffffffu;
    const auto rhs_magnitude = rhs & 0x7fffffffu;
    const auto big = lhs_magnitude >= rhs_magnitude ? lhs : rhs;
    const auto small = lhs_magnitude >= rhs_magnitude ? rhs : lhs;
    const auto big_field = static_cast<unsigned>((big >> 23) & 0xffu);
    const auto small_field = static_cast<unsigned>((small >> 23) & 0xffu);
    const auto big_exponent = big_field == 0 ? 1u : big_field;
    const auto small_exponent = small_field == 0 ? 1u : small_field;
    const auto big_significand = (static_cast<std::uint64_t>(big_field != 0) << 26)
        | (static_cast<std::uint64_t>(big & 0x7fffffu) << 3);
    const auto small_significand = (static_cast<std::uint64_t>(small_field != 0) << 26)
        | (static_cast<std::uint64_t>(small & 0x7fffffu) << 3);
    const auto aligned_small = shift_right_sticky(
        small_significand, big_exponent - small_exponent);
    const auto subtract = ((big ^ small) & 0x80000000u) != 0;
    const auto magnitude = subtract
        ? big_significand - aligned_small
        : big_significand + aligned_small;
    if (magnitude == 0) return 0;
    const auto leading = leading_one(magnitude);
    const auto normalized = magnitude << (63u - leading);
    const auto exponent = static_cast<int>(big_exponent) - 127
        + static_cast<int>(leading) - 26;
    return pack_fp32_rne((big & 0x80000000u) != 0, exponent, normalized);
}

inline float fp32_add(float lhs, float rhs) noexcept
{
    return std::bit_cast<float>(fp32_add_bits(
        std::bit_cast<std::uint32_t>(lhs),
        std::bit_cast<std::uint32_t>(rhs)));
}

inline std::uint16_t fp32_to_bf16(std::uint32_t value) noexcept
{
    if ((value & 0x7f800000u) == 0x7f800000u) {
        return static_cast<std::uint16_t>(value >> 16)
            | ((value & 0x007fffffu) != 0 ? 0x0040u : 0u);
    }
    const auto increment = (value & 0x00008000u) != 0
        && ((value & 0x00007fffu) != 0 || (value & 0x00010000u) != 0);
    return static_cast<std::uint16_t>(value >> 16)
        + static_cast<std::uint16_t>(increment);
}

inline std::uint16_t fp32_to_bf16(float value) noexcept
{
    return fp32_to_bf16(std::bit_cast<std::uint32_t>(value));
}

// Bit-exact combinational result of lpu_mxm_weight_dequant_pipe for one lane.
inline std::uint16_t int8_dequant_bf16(
    std::int8_t quantized,
    std::int8_t zero_point,
    std::uint16_t scale) noexcept
{
    const auto difference = static_cast<int>(quantized)
        - static_cast<int>(zero_point);
    const auto sign = (difference < 0) != ((scale & 0x8000u) != 0);
    const auto exponent_field = static_cast<unsigned>((scale >> 7) & 0xffu);
    const auto fraction = static_cast<unsigned>(scale & 0x7fu);
    if (exponent_field == 0xffu) {
        if (fraction != 0 || difference == 0) return 0x7fc0u;
        return (static_cast<std::uint16_t>(sign) << 15) | 0x7f80u;
    }
    if (difference == 0 || (scale & 0x7fffu) == 0) {
        return static_cast<std::uint16_t>(sign) << 15;
    }

    const auto magnitude = static_cast<unsigned>(
        difference < 0 ? -difference : difference);
    const auto scale_significand = (exponent_field != 0 ? 0x80u : 0u)
        | fraction;
    const auto product = magnitude * scale_significand;
    auto exponent = exponent_field == 0 ? 1u : exponent_field;
    if ((product & 0xff80u) == 0) {
        return (static_cast<std::uint16_t>(sign) << 15)
            | static_cast<std::uint16_t>(product & 0x7fu);
    }

    const auto distance = static_cast<unsigned>(std::bit_width(product) - 8u);
    auto rounded = product >> distance;
    if (distance != 0) {
        const auto guard = ((product >> (distance - 1u)) & 1u) != 0;
        const auto sticky_mask = distance <= 1
            ? 0u : (1u << (distance - 1u)) - 1u;
        const auto sticky = (product & sticky_mask) != 0;
        if (guard && (sticky || (rounded & 1u) != 0)) ++rounded;
    }
    exponent += distance;
    if ((rounded & 0x100u) != 0) {
        rounded >>= 1;
        ++exponent;
    }
    if (exponent >= 0xffu) {
        return (static_cast<std::uint16_t>(sign) << 15) | 0x7f80u;
    }
    return (static_cast<std::uint16_t>(sign) << 15)
        | static_cast<std::uint16_t>(exponent << 7)
        | static_cast<std::uint16_t>(rounded & 0x7fu);
}

} // namespace ftlpu::mxm_numeric
