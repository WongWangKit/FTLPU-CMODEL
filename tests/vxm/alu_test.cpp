#include "ftlpu/vxm/alu.hpp"

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

    auto sqrt = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Sqrt}, square);
    assert(nearly_equal(sqrt[3], 4.0f));

    auto min = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Min}, a, b);
    assert(min[0] == -8.0f);
    assert(min[15] == 7.0f);

    auto max = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Max}, a, b);
    assert(max[0] == 1.0f);
    assert(max[15] == 16.0f);

    auto clamp = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Clamp, -2.0f, 3.0f}, a);
    assert(clamp[0] == -2.0f);
    assert(clamp[15] == 3.0f);

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

    auto log = ftlpu::VxmAlu::execute({ftlpu::VxmAluOpcode::Log}, b);
    assert(nearly_equal(log[0], 0.0f));

    auto quantized = ftlpu::VxmAlu::quantize(b, 0.5f, -3);
    assert(quantized[0] == -1);
    assert(quantized[15] == 29);

    auto dequantized = ftlpu::VxmAlu::dequantize(quantized, 0.5f, -3);
    assert(nearly_equal(dequantized[0], 1.0f));
    assert(nearly_equal(dequantized[15], 16.0f));

    auto cast_fp32 = ftlpu::VxmAlu::execute(
        {ftlpu::VxmAluOpcode::Cast, 0.0f, 0.0f, 0, 1.0f, ftlpu::VxmCastTarget::Float32},
        b);
    assert(nearly_equal(cast_fp32[0], 1.0f));
    assert(nearly_equal(cast_fp32[15], 16.0f));

    ftlpu::VxmAlu::Vector int8_cast_input {};
    int8_cast_input[0] = -129.0f;
    int8_cast_input[1] = -2.4f;
    int8_cast_input[2] = 127.6f;
    const auto cast_int8 = ftlpu::VxmAlu::execute(
        {ftlpu::VxmAluOpcode::Cast, 0.0f, 0.0f, 0, 1.0f, ftlpu::VxmCastTarget::Int8},
        int8_cast_input);
    assert(cast_int8[0] == -128.0f);
    assert(cast_int8[1] == -2.0f);
    assert(cast_int8[2] == 127.0f);

    ftlpu::VxmAlu::Vector large{};
    large.fill(1000.0f);
    auto saturated = ftlpu::VxmAlu::quantize(large, 1.0f, 0);
    assert(saturated[0] == 127);

    return 0;
}
