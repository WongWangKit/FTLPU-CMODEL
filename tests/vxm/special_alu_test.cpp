#include "ftlpu/vxm/special_alu.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

template <typename Fn>
std::vector<ftlpu::VxmLutEntry> make_table(float input_min, float width,
                                           std::size_t count, Fn fn)
{
    std::vector<ftlpu::VxmLutEntry> entries;
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto x0 = input_min + static_cast<float>(index) * width;
        const auto x1 = x0 + width;
        const auto y0 = fn(x0);
        const auto k = (fn(x1) - y0) / width;
        entries.push_back(ftlpu::VxmLutEntry::from_float(k, y0));
    }
    return entries;
}

bool relative_near(float actual, float expected, float relative = 0.015f)
{
    return std::fabs(actual - expected) <= relative * std::max(1.0f, std::fabs(expected));
}

void configure(ftlpu::VxmSpecialAlu& alu)
{
    constexpr std::size_t count = 128;
    constexpr float ln2 = 0.6931471805599453f;
    const auto exp_min = -ln2 / 2.0f;
    const auto exp_width = ln2 / static_cast<float>(count);
    alu.configure_lut(ftlpu::VxmSpecialAluOpcode::Exp,
        {exp_min, exp_width}, make_table(exp_min, exp_width, count,
            [](float x) { return std::exp(x); }));

    const auto reciprocal_width = 1.0f / static_cast<float>(count);
    alu.configure_lut(ftlpu::VxmSpecialAluOpcode::Reciprocal,
        {1.0f, reciprocal_width}, make_table(1.0f, reciprocal_width, count,
            [](float x) { return 1.0f / x; }));

    const auto rsqrt_width = 3.0f / static_cast<float>(count);
    alu.configure_lut(ftlpu::VxmSpecialAluOpcode::Rsqrt,
        {1.0f, rsqrt_width}, make_table(1.0f, rsqrt_width, count,
            [](float x) { return 1.0f / std::sqrt(x); }));
}

}

int main()
{
    auto alu = ftlpu::VxmSpecialAlu{};
    configure(alu);

    assert(relative_near(alu.execute(ftlpu::VxmSpecialAluOpcode::Exp, 1.25f),
                         std::exp(1.25f)));
    assert(relative_near(alu.execute(ftlpu::VxmSpecialAluOpcode::Reciprocal, -3.25f),
                         1.0f / -3.25f));
    assert(relative_near(alu.execute(ftlpu::VxmSpecialAluOpcode::Rsqrt, 9.0f),
                         1.0f / 3.0f));

    const auto a = alu.make_lookup(ftlpu::VxmSpecialAluOpcode::Reciprocal, 1.1f);
    const auto b = alu.make_lookup(ftlpu::VxmSpecialAluOpcode::Reciprocal, 1.8f);
    assert(a.index != b.index); // lane-local address; shared table contents.
    assert(alu.execute(ftlpu::VxmSpecialAluOpcode::Exp, 0x1p-20f) == 1.0f);
    assert(std::isinf(alu.execute(ftlpu::VxmSpecialAluOpcode::Reciprocal, 0.0f)));
    assert(std::isnan(alu.execute(ftlpu::VxmSpecialAluOpcode::Rsqrt, -1.0f)));

    // The LUT datapath has five registered calculation stages and II=1.
    static_assert(ftlpu::VxmSpecialAlu::kPipelineLatency == 5);
    static_assert(ftlpu::VxmSpecialAlu::kInitiationInterval == 1);
    using Pipeline = ftlpu::VxmSpecialAlu::Pipeline<int>;
    auto pipeline = Pipeline{};
    auto first = pipeline.tick(alu, Pipeline::Request{
        ftlpu::VxmSpecialAluOpcode::Exp, 1.0f, 10});
    assert(!first);
    auto second = pipeline.tick(alu, Pipeline::Request{
        ftlpu::VxmSpecialAluOpcode::Exp, 2.0f, 20});
    assert(!second);
    assert(!pipeline.tick(alu));
    assert(!pipeline.tick(alu));
    first = pipeline.tick(alu);
    assert(first && first->metadata == 10);
    assert(relative_near(first->value, std::exp(1.0f)));
    second = pipeline.tick(alu);
    assert(second && second->metadata == 20);
    assert(relative_near(second->value, std::exp(2.0f)));
    assert(pipeline.empty());
    return 0;
}
