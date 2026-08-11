#include "ftlpu/mxm/accumulator.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

template <typename Fn>
bool throws(Fn&& fn)
{
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    static_assert(ftlpu::hw::kMxmAccumulatorRows
        == ftlpu::hw::kMxmAccumulatorBlockCount * ftlpu::hw::kMxmRows);
    static_assert(ftlpu::hw::kMxmAccumulatorBytes
        == ftlpu::hw::kMxmAccumulatorBlockCount * 32 * 32 * sizeof(float));

    ftlpu::MxmAccumulator accumulator;
    ftlpu::MxmAccumulator::Segment first{};
    ftlpu::MxmAccumulator::Segment second{};
    for (std::size_t lane = 0; lane < first.size(); ++lane) {
        first[lane] = static_cast<float>(lane + 1);
        second[lane] = static_cast<float>(2 * lane);
    }
    accumulator.accumulate(17, 2, first);
    accumulator.accumulate(17, 2, second);
    const auto result = accumulator.read(17, 2);
    for (std::size_t lane = 0; lane < result.size(); ++lane) {
        assert(result[lane] == first[lane] + second[lane]);
        assert(accumulator.value(17, 16 + lane) == result[lane]);
    }

    accumulator.clear_segment(17, 2);
    for (const auto value : accumulator.read(17, 2)) {
        assert(value == 0.0f);
    }
    assert(throws([&] {
        (void)accumulator.read(ftlpu::hw::kMxmAccumulatorRows, 0);
    }));
    return 0;
}
