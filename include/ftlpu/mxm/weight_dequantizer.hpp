#pragma once

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/bf16_arithmetic.hpp"
#include "ftlpu/mxm/supercell.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu {

enum class MxmWeightInputMode {
    Int8DequantBf16 = 0,
    Direct16 = 1,
    Fp8DequantBf16 = 2,
};

struct MxmDequantInstruction {
    std::uint16_t scale_bf16{Bf16::from_float(1.0f).bits()};
    std::int8_t zero_point{0};

    static MxmDequantInstruction Scale(float scale)
    {
        return MxmDequantInstruction {Bf16::from_float(scale).bits(), 0};
    }

    static MxmDequantInstruction ScaleZeroPoint(
        float scale, std::int8_t zero_point)
    {
        return MxmDequantInstruction {
            Bf16::from_float(scale).bits(), zero_point};
    }

    static constexpr MxmDequantInstruction ScaleBits(
        std::uint16_t scale_bf16,
        std::int8_t zero_point = 0) noexcept
    {
        return MxmDequantInstruction {scale_bf16, zero_point};
    }

    float scale() const noexcept
    {
        return Bf16::from_bits(scale_bf16).to_float();
    }
};

struct MxmWeightInput {
    using ValueMatrix = MxmArray::InputVector;

    ValueMatrix values {};

    auto& operator[](std::size_t lane)
    {
        return values.at(lane);
    }

    const auto& operator[](std::size_t lane) const
    {
        return values.at(lane);
    }
};

class MxmWeightDequantizer {
public:
    using Output = MxmArray::InputVector;

    Output convert(
        const MxmWeightInput& input,
        MxmWeightLoadMode load_mode,
        std::size_t inner_column,
        MxmWeightInputMode input_mode,
        MxmDequantInstruction instruction) const
    {
        if (input_mode == MxmWeightInputMode::Direct16) {
            return input.values;
        }
        if (input_mode != MxmWeightInputMode::Int8DequantBf16
            && input_mode != MxmWeightInputMode::Fp8DequantBf16) {
            throw std::invalid_argument("MXM weight input mode is invalid");
        }

        auto output = Output {};
        if (load_mode == MxmWeightLoadMode::Vector) {
            check_column(inner_column);
            convert_column(
                input,
                inner_column,
                input_mode, instruction,
                output);
            return output;
        }
        if (load_mode != MxmWeightLoadMode::Block) {
            throw std::invalid_argument("MXM weight load mode is invalid");
        }
        for (std::size_t column = 0;
             column < hw::kMxmSupercellColumns;
             ++column) {
            convert_column(
                input,
                column,
                input_mode, instruction,
                output);
        }
        return output;
    }

private:
    static void check_column(std::size_t column)
    {
        if (column >= hw::kMxmSupercellColumns) {
            throw std::out_of_range(
                "MXM dequantizer column is outside the supercell");
        }
    }

    static float decode_e4m3fn(std::uint8_t byte)
    {
        const unsigned magnitude = byte & 0x7f;
        if (magnitude == 0x7f) return std::numeric_limits<float>::quiet_NaN();
        const int exponent = magnitude >> 3;
        const int fraction = magnitude & 7;
        const float value = exponent == 0 ? std::ldexp(float(fraction), -9)
            : std::ldexp(float(8 + fraction), exponent - 10);
        return byte & 0x80 ? -value : value;
    }

    static void convert_column(
        const MxmWeightInput& input,
        std::size_t column,
        MxmWeightInputMode mode,
        MxmDequantInstruction instruction,
        Output& output)
    {
        const auto scale = Bf16::from_bits(instruction.scale_bf16).to_float();
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto& word = input.values[lane][column];
            if (!word.has_value()) {
                throw std::logic_error(
                    "MXM INT8 IW requires every quantized weight lane");
            }
            if ((word->data & 0xff00u) != 0) {
                throw std::logic_error(
                    "MXM INT8 IW weight input contains more than one byte");
            }
            const auto byte = static_cast<std::uint8_t>(word->data);
            const auto bits = mode == MxmWeightInputMode::Int8DequantBf16
                ? mxm_numeric::int8_dequant_bf16(
                    static_cast<std::int8_t>(byte),
                    instruction.zero_point,
                    instruction.scale_bf16)
                : Bf16::from_float(decode_e4m3fn(byte) * scale).bits();
            output[lane][column] = MxmArray::Supercell::InputWord {
                bits,
                word->last,
            };
        }
    }
};

} // namespace ftlpu
