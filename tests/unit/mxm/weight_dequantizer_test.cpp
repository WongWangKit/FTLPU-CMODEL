#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/mxm/weight_dequantizer.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

ftlpu::MxmWeightInput quantized_input()
{
    auto input = ftlpu::MxmWeightInput {};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t column = 0;
             column < ftlpu::hw::kMxmSupercellColumns;
             ++column) {
            const auto value = static_cast<std::int8_t>(
                static_cast<int>(lane) - static_cast<int>(column) - 3);
            input[lane][column] =
                ftlpu::MxmArray::Supercell::InputWord {
                    static_cast<std::uint8_t>(value),
                    column + 1 == ftlpu::hw::kMxmSupercellColumns,
                };
        }
    }
    return input;
}

} // namespace

int main()
{
    using Mode = ftlpu::MxmWeightInputMode;
    for (const auto mode : {Mode::Direct16, Mode::Int8DequantBf16, Mode::Fp8DequantBf16}) {
        const auto iw = ftlpu::MxmControlInstruction::IWColumn(1, 0, 7, mode, 4);
        const auto decoded = ftlpu::isa::decode_mxm_instruction(ftlpu::isa::encode_mxm_instruction(iw));
        if (decoded.weight_input_mode != mode || decoded.weight_inner_column != 7
            || decoded.weight_stream_base != 4 || decoded.weight_buffer != 1)
            throw std::runtime_error("IW format round trip failed");
    }
    // Independent BF16 bit oracle at scale 2^-12, covering both zeros,
    // subnormals, finite exponent-15 values and the two E4M3FN NaN codes.
    for (unsigned code = 0; code < 256; ++code) {
        auto data = quantized_input();
        data[0][0]->data = code;
        const auto result = ftlpu::MxmWeightDequantizer{}.convert(data,
            ftlpu::MxmWeightLoadMode::Block, 0, Mode::Fp8DequantBf16,
            ftlpu::MxmDequantInstruction::Scale(0x1p-12f));
        const unsigned mag = code & 127;
        const unsigned sign = (code & 128) << 8;
        unsigned expected = sign;
        if (mag == 127) {
            if (!std::isnan(ftlpu::Bf16::from_bits(result[0][0]->data).to_float()))
                throw std::runtime_error("FP8 NaN decode failed");
            continue;
        }
        if (mag >= 8) {
            expected |= ((mag / 8 + 108) << 7) | ((mag % 8) << 4);
        } else if (mag != 0) {
            const unsigned leading = mag >= 4 ? 2 : mag >= 2 ? 1 : 0;
            expected |= ((106 + leading) << 7) | ((mag << (7 - leading)) & 127);
        }
        if (result[0][0]->data != expected)
            throw std::runtime_error("E4M3FN weight decode failed");
    }
    const auto dequant = ftlpu::MxmDequantInstruction::Scale(0.125f);
    const auto input = quantized_input();
    const auto converted = ftlpu::MxmWeightDequantizer {}.convert(
        input,
        ftlpu::MxmWeightLoadMode::Block,
        0,
        ftlpu::MxmWeightInputMode::Int8DequantBf16,
        dequant);

    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t column = 0;
             column < ftlpu::hw::kMxmSupercellColumns;
             ++column) {
            const auto quantized = static_cast<std::int8_t>(
                static_cast<std::uint8_t>(input[lane][column]->data));
            const auto expected = ftlpu::Bf16::from_float(
                static_cast<float>(quantized) * dequant.scale()).bits();
            assert(converted[lane][column]->data == expected);
        }
    }

    auto array = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice control(*array);
    control.issue_dequant_south(dequant);
    control.issue_south(ftlpu::MxmControlInstruction::IW(0, 0));
    control.set_weight_input(0, input);
    std::ostringstream log;
    control.tick(log);
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t column = 0;
             column < ftlpu::hw::kMxmSupercellColumns;
             ++column) {
            assert(
                array->cell(0, 0).weight_bits(0, lane, column)
                == converted[lane][column]->data);
        }
    }

    auto missing_array = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice missing(*missing_array);
    missing.issue_south(ftlpu::MxmControlInstruction::IW(0, 0));
    missing.set_weight_input(0, input);
    bool rejected = false;
    try {
        missing.tick(log);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);

    return 0;
}
