#include "ftlpu/core/fp16.hpp"
#include "ftlpu/mxm/supercell.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <type_traits>

namespace {

using Supercell = ftlpu::MxmSupercell;

Supercell::InputVector scale_input(
    const Supercell::WeightScales& scales)
{
    auto input = Supercell::InputVector{};
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        for (std::size_t column = 0;
             column < ftlpu::hw::kMxmSupercellColumns;
             ++column) {
            const auto bits =
                ftlpu::Fp16::from_float(
                    scales[column]).bits();
            input[lane][column * 2] =
                Supercell::InputWord {
                    static_cast<
                        Supercell::EncodedWeightByte>(
                        bits & 0xffu),
                    false};
            input[lane][column * 2 + 1] =
                Supercell::InputWord {
                    static_cast<
                        Supercell::EncodedWeightByte>(
                        bits >> 8),
                    column + 1
                        == ftlpu::hw::
                            kMxmSupercellColumns};
        }
    }
    return input;
}

Supercell::InputVector weight_input()
{
    auto input = Supercell::InputVector{};
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        for (std::size_t column = 0;
             column < ftlpu::hw::kMxmSupercellColumns;
             ++column) {
            const auto value = static_cast<std::int8_t>(
                static_cast<int>(lane * 9 + column * 5)
                % 31 - 15);
            input[lane][column] =
                Supercell::InputWord {
                    static_cast<
                        Supercell::EncodedWeightByte>(
                        value),
                    column + 1
                        == ftlpu::hw::
                            kMxmSupercellColumns};
        }
    }
    return input;
}

} // namespace

int main()
{
    static_assert(ftlpu::hw::kTileRows == 4);
    static_assert(ftlpu::hw::kLanesPerTile == 8);
    static_assert(ftlpu::hw::kMxmCount == 4);
    static_assert(ftlpu::hw::kMxmRows == 32);
    static_assert(ftlpu::hw::kMxmColumns == 32);
    static_assert(
        ftlpu::hw::kMxmActivationBytesPerValue == 2);
    static_assert(
        ftlpu::hw::kMxmStoredWeightBytesPerValue == 1);
    static_assert(
        ftlpu::hw::kMxmWeightBytesPerValue == 2);
    static_assert(
        Supercell::kRequiresWeightDequantization);
    static_assert(
        std::is_same_v<
            Supercell::StoredWeight,
            std::int8_t>);
    static_assert(
        std::is_same_v<
            Supercell::Weight,
            float>);

    auto scales = Supercell::WeightScales{};
    for (std::size_t column = 0;
         column < scales.size();
         ++column) {
        scales[column] =
            ftlpu::Fp16::from_float(
                0.03125f
                * static_cast<float>(column + 1))
                .to_float();
    }

    auto supercell = Supercell{};
    auto log = std::ostringstream{};
    supercell.set_input(scale_input(scales));
    supercell.issue(
        ftlpu::MxmInstruction::LoadScales(0));
    supercell.tick(log);

    const auto encoded_weights = weight_input();
    supercell.set_input(encoded_weights);
    supercell.issue(ftlpu::MxmInstruction::IW(0));
    supercell.tick(log);

    auto activations = Supercell::ActivationData{};
    for (std::size_t lane = 0;
         lane < activations.size();
         ++lane) {
        activations[lane] =
            ftlpu::Fp16::from_float(
                0.125f
                * static_cast<float>(
                    static_cast<int>(lane) - 3))
                .to_float();
    }

    const auto result =
        supercell.compute_partial(activations, 0);
    for (std::size_t column = 0;
         column < ftlpu::hw::kMxmSupercellColumns;
         ++column) {
        float expected = 0.0f;
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            const auto quantized =
                static_cast<std::int8_t>(
                    encoded_weights[lane][column]
                        ->data);
            assert(
                supercell.stored_weight(
                    0, lane, column)
                == quantized);
            const auto dequantized =
                ftlpu::Fp16::from_float(
                    static_cast<float>(quantized)
                    * scales[column])
                    .to_float();
            assert(
                supercell.weight(
                    0, lane, column)
                == dequantized);
            expected +=
                activations[lane] * dequantized;
        }
        assert(
            std::fabs(result[column] - expected)
            < 1.0e-6f);
    }

    // Scale and W8 weight state is independently addressable in both
    // ping-pong buffers; loading the next K block must not alter buffer 0.
    auto second_scales = scales;
    for (auto& scale : second_scales) {
        scale = ftlpu::Fp16::from_float(
            scale * 0.5f).to_float();
    }
    supercell.set_input(scale_input(second_scales));
    supercell.issue(
        ftlpu::MxmInstruction::LoadScales(1));
    supercell.tick(log);
    supercell.set_input(encoded_weights);
    supercell.issue(ftlpu::MxmInstruction::IW(1));
    supercell.tick(log);

    const auto second_result =
        supercell.compute_partial(activations, 1);
    for (std::size_t column = 0;
         column < ftlpu::hw::kMxmSupercellColumns;
         ++column) {
        auto expected = 0.0f;
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            const auto quantized =
                static_cast<std::int8_t>(
                    encoded_weights[lane][column]
                        ->data);
            expected += activations[lane]
                * ftlpu::Fp16::from_float(
                      static_cast<float>(quantized)
                      * second_scales[column])
                      .to_float();
            assert(
                supercell.weight(0, lane, column)
                == ftlpu::Fp16::from_float(
                       static_cast<float>(quantized)
                       * scales[column])
                       .to_float());
        }
        assert(
            std::fabs(
                second_result[column] - expected)
            < 1.0e-6f);
    }
    return 0;
}
