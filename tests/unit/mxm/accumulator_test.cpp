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
        == ftlpu::hw::kMxmAccumulatorBlockCount * ftlpu::hw::kMxmRows
            * ftlpu::hw::kMxmColumns * sizeof(float));

    ftlpu::MxmAccumulator accumulator;
    ftlpu::MxmAccumulator::Segment first{};
    ftlpu::MxmAccumulator::Segment second{};
    for (std::size_t lane = 0; lane < first.size(); ++lane) {
        first[lane] = static_cast<float>(lane + 1);
        second[lane] = static_cast<float>(2 * lane);
    }
    constexpr auto row = ftlpu::hw::kMxmAccumulatorRows - 1;
    constexpr auto column = ftlpu::hw::kMxmSupercellsPerPlane - 1;
    accumulator.accumulate(row, column, first);
    accumulator.accumulate(row, column, second);
    const auto result = accumulator.read(row, column);
    for (std::size_t lane = 0; lane < result.size(); ++lane) {
        assert(result[lane] == first[lane] + second[lane]);
        assert(accumulator.value(row, column * first.size() + lane) == result[lane]);
    }

    accumulator.clear_segment(row, column);
    for (const auto value : accumulator.read(row, column)) {
        assert(value == 0.0f);
    }
    assert(throws([&] {
        (void)accumulator.read(ftlpu::hw::kMxmAccumulatorRows, 0);
    }));
    return 0;
}
