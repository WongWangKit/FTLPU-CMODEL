#include "ftlpu/vxm_alu.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

bool nearly_equal(float a, float b, float eps = 1.0e-5f)
{
    return std::fabs(a - b) <= eps;
}

ftlpu::VxmAlu::Vector ramp(float start)
{
    ftlpu::VxmAlu::Vector out{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        out[lane] = start + static_cast<float>(lane);
    }
    return out;
}

} // namespace

int main()
{
    const auto a = ramp(-8.0f);
    const auto b = ramp(1.0f);

    auto add = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Add}, a, b);
    assert(add[0] == -7.0f);
    assert(add[15] == 23.0f);

    auto mul = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Multiply}, a, b);
    assert(mul[0] == -8.0f);
    assert(mul[15] == 112.0f);

    auto square = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Square}, b);
    assert(square[3] == 16.0f);

    auto rsqrt = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Rsqrt}, square);
    assert(nearly_equal(rsqrt[3], 0.25f));

    auto tanh = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Tanh}, a);
    assert(tanh[0] < -0.99f);
    assert(nearly_equal(tanh[8], 0.0f));

    ftlpu::VxmAlu::Vector ones{};
    ones.fill(1.0f);
    const auto neg = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Negate}, a);
    const auto exp_neg = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Exp}, neg);
    const auto denom = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Add}, ones, exp_neg);
    const auto sigmoid = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Divide}, ones, denom);
    assert(sigmoid[0] < 0.001f);
    assert(nearly_equal(sigmoid[8], 0.5f));

    auto relu = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Relu}, a);
    assert(relu[0] == 0.0f);
    assert(relu[15] == 7.0f);

    ftlpu::VxmAlu::Vector condition{};
    condition[0] = 1.0f;
    condition[1] = 0.0f;
    auto selected = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Select}, condition, a, b);
    assert(selected[0] == a[0]);
    assert(selected[1] == b[1]);

    auto accumulated = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Accumulate}, b);
    assert(accumulated[0] == 136.0f);
    assert(accumulated[15] == 136.0f);

    auto quantized = ftlpu::VxmAlu::quantize(b, 0.5f, -3);
    assert(quantized[0] == -1);
    assert(quantized[15] == 29);

    auto dequantized = ftlpu::VxmAlu::dequantize(quantized, 0.5f, -3);
    assert(nearly_equal(dequantized[0], 1.0f));
    assert(nearly_equal(dequantized[15], 16.0f));

    ftlpu::VxmAlu::Vector large{};
    large.fill(1000.0f);
    auto saturated = ftlpu::VxmAlu::quantize(large, 1.0f, 0);
    assert(saturated[0] == 127);

    auto compare = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::LessThan}, a, b);
    assert(compare[0] == 1.0f);
    assert(compare[15] == 1.0f);

    return 0;
}
