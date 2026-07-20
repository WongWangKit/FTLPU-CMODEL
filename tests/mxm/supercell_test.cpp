#include "ftlpu/mxm/supercell.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace {

ftlpu::MxmSupercell::InputVector full_input()
{
    ftlpu::MxmSupercell::InputVector input{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t stream = 0; stream < ftlpu::hw::kMxmSupercellColumns; ++stream) {
            input[lane][stream] = ftlpu::MxmSupercell::InputWord {
                static_cast<float>(static_cast<int>(lane * 8 + stream) - 31),
                stream + 1 == ftlpu::hw::kMxmSupercellColumns,
            };
        }
    }
    return input;
}

ftlpu::MxmSupercell::ActivationVector activation_input(float base = 0.5f)
{
    ftlpu::MxmSupercell::ActivationVector input{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        input[lane] = ftlpu::MxmSupercell::ActivationWord {
            base + static_cast<float>(lane) * 0.25f,
            lane + 1 == ftlpu::hw::kLanesPerTile,
        };
    }
    return input;
}

float expected_dot(float activation_base, std::size_t column)
{
    float sum = 0.0f;
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        const auto activation = activation_base + static_cast<float>(lane) * 0.25f;
        const auto weight = static_cast<float>(static_cast<int>(lane * 8 + column) - 31);
        sum += activation * weight;
    }
    return sum;
}

bool nearly_equal(float lhs, float rhs)
{
    return std::fabs(lhs - rhs) < 1.0e-5f;
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
    assert(supercell.weight(0, 7, 7) == 0);
    assert(supercell.weight(1, 7, 7) == static_cast<std::int8_t>(32));

    supercell.set_activation_input(activation_input(), 1);
    for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellColumns; ++column) {
        supercell.tick(log);
        const auto& outputs = supercell.outputs();
        assert(outputs.size() == 1);
        assert(outputs[0].column == column);
        assert(nearly_equal(outputs[0].value, expected_dot(0.5f, column)));
    }

    supercell.tick(log);
    assert(supercell.outputs().empty());

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
    auto missing = full_input();
    missing[7][7].reset();
    caught = false;
    try {
        supercell.set_input(missing);
        supercell.issue(ftlpu::MxmInstruction::IW(0));
        supercell.tick(log);
    } catch (const std::logic_error&) {
        caught = true;
    }
    assert(caught);

    assert(log.str().find("MAC col=7 result=") != std::string::npos);
    return 0;
}
