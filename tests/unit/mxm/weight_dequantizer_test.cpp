#include "ftlpu/core/bf16.hpp"
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
    const auto dequant = ftlpu::MxmDequantInstruction::Scale(0.125f);
    const auto input = quantized_input();
    const auto converted = ftlpu::MxmWeightDequantizer {}.convert(
        input,
        ftlpu::MxmWeightLoadMode::Supercell,
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
