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
    Rsqrt,
    Exp,
    Log,
    Tanh,
    Relu,
    Gelu,
    Equal,
    LessThan,
    LessEqual,
    Select,
    Accumulate,
    Cast,
};

enum class VxmCastTarget {
    Float32,
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
        case VxmAluOpcode::Rsqrt:
            apply_unary(out, a, [](float x) { return 1.0f / std::sqrt(x); });
            return out;
        case VxmAluOpcode::Exp:
            apply_unary(out, a, [](float x) { return std::exp(x); });
            return out;
        case VxmAluOpcode::Log:
            apply_unary(out, a, [](float x) { return std::log(x); });
            return out;
        case VxmAluOpcode::Tanh:
            apply_unary(out, a, [](float x) { return std::tanh(x); });
            return out;
        case VxmAluOpcode::Relu:
            apply_unary(out, a, [](float x) { return std::max(0.0f, x); });
            return out;
        case VxmAluOpcode::Gelu:
            apply_unary(out, a, [](float x) {
                constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
                return 0.5f * x * (1.0f + std::tanh(kSqrtTwoOverPi * (x + 0.044715f * x * x * x)));
            });
            return out;
        case VxmAluOpcode::Equal:
            apply_binary(out, a, b, [](float x, float y) { return x == y ? 1.0f : 0.0f; });
            return out;
        case VxmAluOpcode::LessThan:
            apply_binary(out, a, b, [](float x, float y) { return x < y ? 1.0f : 0.0f; });
            return out;
        case VxmAluOpcode::LessEqual:
            apply_binary(out, a, b, [](float x, float y) { return x <= y ? 1.0f : 0.0f; });
            return out;
        case VxmAluOpcode::Select:
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                out[lane] = a[lane] != 0.0f ? b[lane] : c[lane];
            }
            return out;
        case VxmAluOpcode::Accumulate: {
            const auto sum = lane_sum(a);
            out.fill(sum);
            return out;
        }
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
