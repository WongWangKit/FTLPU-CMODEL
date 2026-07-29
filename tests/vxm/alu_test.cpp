#include "ftlpu/vxm/alu.hpp"

#include <cassert>
#include <cmath>

namespace {
bool near(float a, float b, float eps = 1.0e-3f)
{
    return std::fabs(a - b) <= eps;
}
}

int main()
{
    using namespace ftlpu;
    const auto fp32 = VxmAluPrecision::Float32;

    // One call represents one scalar Basic ALU operation.
    assert(VxmAlu::execute({VxmAluOpcode::Bypass, fp32}, 2.0f) == 2.0f);
    assert(VxmAlu::execute({VxmAluOpcode::Add, fp32}, 2.0f, 3.0f) == 5.0f);
    assert(VxmAlu::execute({VxmAluOpcode::Subtract, fp32}, 2.0f, 3.0f) == -1.0f);
    assert(VxmAlu::execute({VxmAluOpcode::Multiply, fp32}, 2.0f, 3.0f) == 6.0f);
    assert(VxmAlu::execute({VxmAluOpcode::Negate, fp32}, 2.0f) == -2.0f);
    assert(VxmAlu::execute({VxmAluOpcode::Max, fp32}, -2.0f, 0.0f) == 0.0f);

    // Compiler lowering examples.
    assert(VxmAlu::execute({VxmAluOpcode::Multiply, fp32}, 1.5f, 1.5f) == 2.25f);
    assert(VxmAlu::execute({VxmAluOpcode::Max, fp32}, -3.0f, 0.0f) == 0.0f);

    const auto value = 1.0004f;
    const auto rounded = VxmAlu::execute(
        {VxmAluOpcode::Bypass, VxmAluPrecision::Float16}, value);
    assert(rounded != value);
    assert(near(rounded, value));

    // Cast and quantization belong to boundary-format hardware.
    assert(VxmDataFormat::round_fp16_ftz(0x1p-20f) == 0.0f);
    assert(VxmDataFormat::quantize_int8(12.7f, 0.1f) == 127);
    assert(VxmDataFormat::quantize_int8(-20.0f, 0.1f) == -128);
    assert(VxmDataFormat::quantize_int8(1.0f, 0.5f, 3) == 5);
    const auto half = VxmDataFormat::float_to_fp16_bits(1.5f);
    assert(near(VxmDataFormat::fp16_bits_to_float(half), 1.5f));

    // Timing model: ordinary operations complete in one tick. Multiply has
    // latency 2 but accepts a new request every tick (II=1).
    static_assert(VxmAlu::kDefaultLatency == 1);
    static_assert(VxmAlu::kMultiplyLatency == 2);
    static_assert(VxmAlu::kInitiationInterval == 1);
    using Pipeline = VxmAlu::Pipeline<int>;
    auto pipeline = Pipeline{};
    auto add = pipeline.tick(Pipeline::Request{
        {VxmAluOpcode::Add, fp32}, 2.0f, 3.0f, 10});
    assert(add && add->value == 5.0f && add->metadata == 10);
    auto first_mul = pipeline.tick(Pipeline::Request{
        {VxmAluOpcode::Multiply, fp32}, 2.0f, 4.0f, 20});
    assert(!first_mul);
    auto second_mul = pipeline.tick(Pipeline::Request{
        {VxmAluOpcode::Multiply, fp32}, 3.0f, 5.0f, 30});
    assert(second_mul && second_mul->value == 8.0f
           && second_mul->metadata == 20);
    auto drained_mul = pipeline.tick();
    assert(drained_mul && drained_mul->value == 15.0f
           && drained_mul->metadata == 30);
    assert(pipeline.empty());
    return 0;
}
