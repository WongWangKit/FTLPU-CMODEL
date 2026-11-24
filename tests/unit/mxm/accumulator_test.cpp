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
        == ftlpu::hw::kMxmAccumulatorBlockCount * ftlpu::hw::kMxmRows * ftlpu::hw::kMxmColumns * sizeof(float));

    ftlpu::MxmAccumulator accumulator;
    ftlpu::MxmAccumulator::Segment first{};
    ftlpu::MxmAccumulator::Segment second{};
    for (std::size_t lane = 0; lane < first.size(); ++lane) {
        first[lane] = static_cast<float>(lane + 1);
        second[lane] = static_cast<float>(2 * lane);
    }
    constexpr std::size_t last_segment=ftlpu::hw::kMxmColumns/ftlpu::hw::kLanesPerTile-1;
    accumulator.low();
    accumulator.accumulate(17, last_segment, first);
    accumulator.accumulate(17, last_segment, second);
    const auto result = accumulator.read(17, last_segment);
    for (std::size_t lane = 0; lane < result.size(); ++lane) {
        assert(result[lane] == first[lane] + second[lane]);
        assert(accumulator.value(17, last_segment*ftlpu::hw::kLanesPerTile + lane) == result[lane]);
    }
    accumulator.validate_posedge();
    accumulator.posedge();

    accumulator.low();
    accumulator.clear_segment(17, last_segment);
    for (const auto value : accumulator.read(17, last_segment)) {
        assert(value == 0.0f);
    }
    accumulator.validate_posedge();
    accumulator.posedge();
    assert(throws([&] {
        (void)accumulator.read(ftlpu::hw::kMxmAccumulatorRows, 0);
    }));
    assert(throws([&]{(void)accumulator.read(17,last_segment+1);}));
    return 0;
}
