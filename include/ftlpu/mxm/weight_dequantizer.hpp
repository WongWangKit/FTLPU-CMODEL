#pragma once

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/supercell.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu {

enum class MxmWeightInputMode {
    Int8DequantBf16 = 0,
    Direct16 = 1,
};

struct MxmDequantInstruction {
    std::uint16_t scale_bf16{Bf16::from_float(1.0f).bits()};

    static MxmDequantInstruction Scale(float scale)
    {
        return MxmDequantInstruction {Bf16::from_float(scale).bits()};
    }

    static constexpr MxmDequantInstruction ScaleBits(
        std::uint16_t scale_bf16) noexcept
    {
        return MxmDequantInstruction {scale_bf16};
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
        if (input_mode != MxmWeightInputMode::Int8DequantBf16) {
            throw std::invalid_argument("MXM weight input mode is invalid");
        }

        auto output = Output {};
        if (load_mode == MxmWeightLoadMode::Column) {
            check_column(inner_column);
            convert_column(
                input,
                inner_column,
                instruction.scale_bf16,
                output);
            return output;
        }
        if (load_mode != MxmWeightLoadMode::Supercell) {
            throw std::invalid_argument("MXM weight load mode is invalid");
        }
        for (std::size_t column = 0;
             column < hw::kMxmSupercellColumns;
             ++column) {
            convert_column(
                input,
                column,
                instruction.scale_bf16,
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

    static void convert_column(
        const MxmWeightInput& input,
        std::size_t column,
        std::uint16_t dequant_scale_bf16,
        Output& output)
    {
        const auto scale = Bf16::from_bits(dequant_scale_bf16).to_float();
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
            const auto quantized = static_cast<std::int8_t>(
                static_cast<std::uint8_t>(word->data));
            output[lane][column] = MxmArray::Supercell::InputWord {
                Bf16::from_float(
                    static_cast<float>(quantized) * scale).bits(),
                word->last,
            };
        }
    }
};

} // namespace ftlpu
