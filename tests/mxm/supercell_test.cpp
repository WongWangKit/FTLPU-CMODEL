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

ftlpu::MxmSupercell::ActivationVector activation_input(std::int8_t base = 1)
{
    ftlpu::MxmSupercell::ActivationVector input{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        input[lane] = ftlpu::MxmSupercell::InputWord {
            static_cast<std::int8_t>(base + static_cast<std::int8_t>(lane)),
            lane + 1 == ftlpu::hw::kLanesPerTile,
        };
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

    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t stream = 0; stream < ftlpu::hw::kMxmLoadStreamsPerCycle; ++stream) {
            assert(supercell.weight(1, lane, stream) == static_cast<std::int8_t>(lane * 16 + stream));
        }
    }

    supercell.set_activation_input(activation_input(), 1);
    for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellColumns; ++column) {
        supercell.tick(log);
        const auto& outputs = supercell.outputs();
        assert(outputs.size() == 1);
        assert(outputs[0].column == column);
        assert(outputs[0].value == expected_dot(1, column));
    }

    supercell.tick(log);
    assert(supercell.outputs().empty());

    for (std::size_t cycle = 0; cycle < ftlpu::hw::kMxmSupercellColumns; ++cycle) {
        const auto base = static_cast<std::int8_t>(cycle + 1);
        supercell.set_activation_input(activation_input(base), 1);
        supercell.tick(log);

        const auto& outputs = supercell.outputs();
        assert(outputs.size() == cycle + 1);
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const auto column = outputs[i].column;
            const auto source_cycle = cycle - column;
            const auto expected_base = static_cast<std::int8_t>(source_cycle + 1);
            assert(outputs[i].value == expected_dot(expected_base, column));
        }
    }

    assert(supercell.outputs().size() == ftlpu::hw::kMxmSupercellColumns);

    supercell.reset();
    bool caught = false;
    try {
        supercell.set_activation_input(activation_input(), 0);
        supercell.tick(log);
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
    assert(text.find("MAC col=0 result=" + std::to_string(expected_dot(1, 0))) != std::string::npos);
    assert(text.find("MAC col=15 result=" + std::to_string(expected_dot(1, 15))) != std::string::npos);

    return 0;
}
