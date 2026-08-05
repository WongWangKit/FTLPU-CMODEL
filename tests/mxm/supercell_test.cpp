#include "ftlpu/mxm/supercell.hpp"

#include <cassert>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

ftlpu::MxmSupercell::InputVector full_input()
{
    ftlpu::MxmSupercell::InputVector input{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t stream = 0; stream < ftlpu::hw::kMxmLoadStreamsPerCycle; ++stream) {
            input[lane][stream] = ftlpu::MxmSupercell::InputWord {
                static_cast<std::int8_t>(lane * 16 + stream),
                stream + 1 == ftlpu::hw::kMxmLoadStreamsPerCycle,
            };
        }
    }
    return input;
}

std::int32_t expected_dot(std::int8_t activation_base, std::size_t column)
{
    std::int32_t sum = 0;
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        const auto activation = static_cast<std::int32_t>(
            static_cast<std::int8_t>(activation_base + static_cast<std::int8_t>(lane)));
        const auto weight = static_cast<std::int32_t>(static_cast<std::int8_t>(lane * 16 + column));
        sum += activation * weight;
    }
    return sum;
}

} // namespace

int main()
{
    ftlpu::MxmSupercell supercell;
    std::ostringstream log;

    supercell.set_input(full_input());
    supercell.issue(ftlpu::MxmInstruction::IW(1));
    supercell.tick(log);
    assert(supercell.weight_buffer_valid(1));
    assert(!supercell.weight_buffer_valid(0));
    assert(supercell.weight(0, 15, 15) == 0);
    assert(supercell.weight(1, 15, 15) == static_cast<std::int8_t>(15 * 16 + 15));

    // A background reload is a lower-first protocol. Repeated pulses for a
    // half must leave its validity state stable; only the first upper pulse
    // completes the buffer, and later upper pulses must keep it complete.
    for (std::size_t pulse = 0; pulse < 2; ++pulse) {
        supercell.set_input(full_input());
        supercell.issue(ftlpu::MxmInstruction::IW(
            1, ftlpu::MxmWeightLoadMode::BackgroundLowerHalf));
        supercell.tick(log);
        assert(!supercell.weight_buffer_valid(1));
    }
    for (std::size_t pulse = 0; pulse < 2; ++pulse) {
        supercell.set_input(full_input());
        supercell.issue(ftlpu::MxmInstruction::IW(
            1, ftlpu::MxmWeightLoadMode::BackgroundUpperHalf));
        supercell.tick(log);
        assert(supercell.weight_buffer_valid(1));
    }

    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t stream = 0; stream < ftlpu::hw::kMxmLoadStreamsPerCycle; ++stream) {
            assert(supercell.weight(1, lane, stream) == static_cast<std::int8_t>(lane * 16 + stream));
        }
    }

    ftlpu::MxmSupercell::ActivationData activation{};
    ftlpu::MxmSupercell::PartialSum south_partial{};
    for (std::size_t lane = 0; lane < activation.size(); ++lane) {
        activation[lane] = static_cast<std::int8_t>(lane + 1);
    }
    south_partial.fill(1000);
    const auto north_partial = supercell.compute_partial(activation, 1, south_partial);
    for (std::size_t column = 0; column < north_partial.size(); ++column) {
        assert(north_partial[column] == 1000 + expected_dot(1, column));
    }

    supercell.reset();
    bool caught = false;
    try {
        supercell.compute_partial(activation, 0);
    } catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    supercell.reset();
    ftlpu::MxmSupercell::InputVector missing = full_input();
    missing[15][15].reset();
    caught = false;
    try {
        supercell.set_input(missing);
        supercell.issue(ftlpu::MxmInstruction::IW(0));
        supercell.tick(log);
    } catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    const auto text = log.str();
    assert(text.find("IW buffer1=0x000102030405060708090a0b0c0d0e0f") != std::string::npos);
    assert(text.find("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff") != std::string::npos);

    return 0;
}
