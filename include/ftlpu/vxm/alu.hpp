#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ftlpu {

enum class VxmAluOpcode {
    Pass,
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
    Abs,
    Min,
    Max,
    Clamp,
    Square,
    Sqrt,
    Exp,
    Log,
    Relu,
    Cast,
};

enum class VxmCastTarget {
    Float32,
    Float16,
    Int8,
};

struct VxmAluInstruction {
    VxmAluOpcode opcode{VxmAluOpcode::Pass};
    float immediate0{0.0f};
    float immediate1{0.0f};
    std::int32_t output_zero_point{0};
    float scale{1.0f};
    VxmCastTarget cast_target{VxmCastTarget::Float32};
};

class VxmAlu {
public:
    using Vector = std::array<float, hw::kLanesPerTile>;
    using Int8Vector = std::array<std::int8_t, hw::kLanesPerTile>;

    static Vector execute(const VxmAluInstruction& instruction, const Vector& a)
    {
        return execute(instruction, a, zero_vector(), zero_vector());
    }

    static Vector execute(const VxmAluInstruction& instruction, const Vector& a, const Vector& b)
    {
        return execute(instruction, a, b, zero_vector());
    }

    static Vector execute(const VxmAluInstruction& instruction, const Vector& a, const Vector& b, const Vector& c)
    {
        Vector out{};
        switch (instruction.opcode) {
        case VxmAluOpcode::Pass:
            return a;
        case VxmAluOpcode::Add:
            apply_binary(out, a, b, [](float x, float y) { return x + y; });
            return out;
        case VxmAluOpcode::Subtract:
            apply_binary(out, a, b, [](float x, float y) { return x - y; });
            return out;
        case VxmAluOpcode::Multiply:
            apply_binary(out, a, b, [](float x, float y) { return x * y; });
            return out;
        case VxmAluOpcode::Divide:
            apply_binary(out, a, b, [](float x, float y) { return x / y; });
            return out;
        case VxmAluOpcode::Negate:
            apply_unary(out, a, [](float x) { return -x; });
            return out;
        case VxmAluOpcode::Abs:
            apply_unary(out, a, [](float x) { return std::fabs(x); });
            return out;
        case VxmAluOpcode::Min:
            apply_binary(out, a, b, [](float x, float y) { return std::min(x, y); });
            return out;
        case VxmAluOpcode::Max:
            apply_binary(out, a, b, [](float x, float y) { return std::max(x, y); });
            return out;
        case VxmAluOpcode::Clamp:
            apply_unary(out, a, [lo = instruction.immediate0, hi = instruction.immediate1](float x) {
                return std::min(std::max(x, lo), hi);
            });
            return out;
        case VxmAluOpcode::Square:
            apply_unary(out, a, [](float x) { return x * x; });
            return out;
        case VxmAluOpcode::Sqrt:
            apply_unary(out, a, [](float x) { return std::sqrt(x); });
            return out;
        case VxmAluOpcode::Exp:
            apply_unary(out, a, [](float x) { return std::exp(x); });
            return out;
        case VxmAluOpcode::Log:
            apply_unary(out, a, [](float x) { return std::log(x); });
            return out;
        case VxmAluOpcode::Relu:
            apply_unary(out, a, [](float x) { return std::max(0.0f, x); });
            return out;
        case VxmAluOpcode::Cast:
            return cast_to_float(a, instruction.cast_target);
        }
        throw std::logic_error("unsupported VXM ALU opcode");
    }

    static Int8Vector quantize(const Vector& input, float scale, std::int32_t output_zero_point = 0)
    {
        if (scale == 0.0f) {
            throw std::invalid_argument("VXM quantize scale must be non-zero");
        }
        Int8Vector out{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            out[lane] = quantize_scalar(input[lane], scale, output_zero_point);
        }
        return out;
    }

    static std::int8_t quantize_scalar(float input, float scale, std::int32_t output_zero_point = 0)
    {
        if (scale == 0.0f) {
            throw std::invalid_argument("VXM quantize scale must be non-zero");
        }
        return cast_scalar_to_int8(input / scale + static_cast<float>(output_zero_point));
    }

    static std::int8_t cast_scalar_to_int8(float input)
    {
        const auto rounded = static_cast<std::int32_t>(std::nearbyint(input));
        return saturate_int8(rounded);
    }

    static std::uint16_t cast_scalar_to_float16_bits(float input)
    {
        if (std::isnan(input)) {
            return 0x7e00u;
        }

        const auto sign = std::signbit(input) ? std::uint16_t {0x8000u} : std::uint16_t {0};
        auto value = std::fabs(input);
        if (std::isinf(value)) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        if (value == 0.0f) {
            return sign;
        }

        int exponent = 0;
        const auto mantissa = std::frexp(value, &exponent);
        auto half_exponent = exponent + 14;
        if (half_exponent >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        if (half_exponent <= 0) {
            const auto scaled = std::ldexp(value, 24);
            auto subnormal = static_cast<std::uint32_t>(std::nearbyint(scaled));
            if (subnormal == 0) {
                return sign;
            }
            if (subnormal >= 0x400u) {
                return static_cast<std::uint16_t>(sign | 0x0400u);
            }
            return static_cast<std::uint16_t>(sign | subnormal);
        }

        auto fraction = static_cast<std::uint32_t>(std::nearbyint((mantissa * 2.0f - 1.0f) * 1024.0f));
        if (fraction == 1024u) {
            fraction = 0;
            ++half_exponent;
            if (half_exponent >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
        }
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(half_exponent) << 10) | fraction);
    }

    static Vector dequantize(const Int8Vector& input, float scale, std::int32_t input_zero_point = 0)
    {
        Vector out{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            out[lane] = static_cast<float>(static_cast<std::int32_t>(input[lane]) - input_zero_point) * scale;
        }
        return out;
    }

private:
    static const Vector& zero_vector()
    {
        static const Vector zero{};
        return zero;
    }

    template <typename Fn>
    static void apply_unary(Vector& out, const Vector& a, Fn fn)
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            out[lane] = fn(a[lane]);
        }
    }

    template <typename Fn>
    static void apply_binary(Vector& out, const Vector& a, const Vector& b, Fn fn)
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            out[lane] = fn(a[lane], b[lane]);
        }
    }

    static float lane_sum(const Vector& a)
    {
        float sum = 0.0f;
        for (const auto value : a) {
            sum += value;
        }
        return sum;
    }

    static Vector cast_to_float(const Vector& input, VxmCastTarget target)
    {
        Vector out{};
        switch (target) {
        case VxmCastTarget::Float32:
            return input;
        case VxmCastTarget::Float16:
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                out[lane] = input[lane];
            }
            return out;
        case VxmCastTarget::Int8:
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                out[lane] = static_cast<float>(cast_scalar_to_int8(input[lane]));
            }
            return out;
        }
        throw std::logic_error("unsupported VXM cast target");
    }

    static std::int8_t saturate_int8(std::int32_t value)
    {
        value = std::min<std::int32_t>(value, std::numeric_limits<std::int8_t>::max());
        value = std::max<std::int32_t>(value, std::numeric_limits<std::int8_t>::min());
        return static_cast<std::int8_t>(value);
    }
};

} // namespace ftlpu
