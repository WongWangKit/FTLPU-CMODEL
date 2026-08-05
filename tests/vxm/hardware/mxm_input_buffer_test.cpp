#include "ftlpu/vxm/mxm_input_buffer.hpp"

#include <cassert>
#include <cstddef>

int main()
{
    using namespace ftlpu;

    constexpr auto features = 2 * hw::kMxmColumns;
    auto collector = MxmVxmInputBuffer{
        features, VxmChainDepth::Eight, 1};
    assert(collector.chain_count() == 2);

    for (std::size_t chain = 0; chain < collector.chain_count(); ++chain) {
        for (std::size_t operand = 0;
             operand < MxmVxmInputBuffer::kOperandsPerToken; ++operand) {
            for (std::size_t block = 0; block < 2; ++block) {
                for (std::size_t row = 0; row < hw::kMxmRows; ++row) {
                    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                        auto output = MxmAccumulatorSlice::Output{};
                        output.row = row;
                        output.tile = tile;
                        output.final = true;
                        output.dequant_scale = 0.5f;
                        for (std::size_t lane = 0;
                             lane < hw::kLanesPerTile; ++lane) {
                            output.values[lane] = static_cast<
                                MxmAccumulatorSlice::Value>(
                                    2 * (1 + chain * 4 + operand * 2
                                        + block + row + tile + lane));
                        }
                        collector.capture(
                            7, chain, operand, block, output);
                    }
                }
            }
        }
    }
    assert(collector.ready(7));

    auto vxm = VxmSlice{};
    collector.feed_feature_tile(7, 0, 0, vxm);
    const auto& bundle = vxm.input_buffer(0).bundle();
    for (std::size_t chain = 0; chain < collector.chain_count(); ++chain) {
        for (std::size_t operand = 0;
             operand < MxmVxmInputBuffer::kOperandsPerToken; ++operand) {
            const auto group = chain * 8 + operand;
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile; ++lane) {
                const auto value = static_cast<MxmAccumulatorSlice::Value>(
                    2 * (1 + chain * 4 + operand * 2 + lane));
                const auto expected = MxmOutputCast::cast(value, 0.5f);
                assert(bundle[lane][2 * group]
                    == static_cast<std::uint8_t>(expected));
                assert(bundle[lane][2 * group + 1]
                    == static_cast<std::uint8_t>(expected >> 8));
            }
        }
    }

    collector.release_group(7);
    assert(collector.resident_groups() == 0);
    return 0;
}
