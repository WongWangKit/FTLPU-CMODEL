#include "ftlpu/mxm/activation_dequantizer.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

int main()
{
    using namespace ftlpu;

    constexpr auto kColumn =
        hw::kMemBoundaryStreamRegisterColumns - 1;
    auto fabric = StreamRegisterFabric(
        hw::kMemBoundaryStreamRegisterColumns);
    auto dequantizer = MxmActivationDequantizer(
        MxmActivationDequantizer::Endpoint {
            kColumn,
            StreamDirection::East,
            0,
            StreamDirection::West,
            0});
    dequantizer.configure_scale(0.5f);
    dequantizer.issue_south();

    const auto final_output = 1
        + (hw::kTileRows - 1)
            * MxmControlSlice::kComputeTileLatency;
    for (std::size_t cycle = 0;
         cycle <= final_output; ++cycle) {
        if (cycle < hw::kTileRows) {
            const auto tile = cycle;
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile; ++lane) {
                const auto value = static_cast<std::int8_t>(
                    static_cast<int>(tile * hw::kLanesPerTile + lane) - 12);
                fabric.initialize_cell(
                    kColumn, tile, lane, StreamId::East(0),
                    StreamCell::Valid(
                        static_cast<std::uint8_t>(value),
                        lane + 1 == hw::kLanesPerTile,
                        100 + tile));
            }
        }

        if (cycle >= 1
            && (cycle - 1) % MxmControlSlice::kComputeTileLatency == 0) {
                const auto tile = (cycle - 1)
                    / MxmControlSlice::kComputeTileLatency;
            if (tile < hw::kTileRows) {
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile; ++lane) {
                    const auto signed_value = static_cast<std::int8_t>(
                        static_cast<int>(tile * hw::kLanesPerTile + lane) - 12);
                    const auto expected = Fp16::from_float(
                        static_cast<float>(signed_value) * 0.5f).bits();
                    const auto& low = fabric.cell(
                        kColumn, tile, lane, StreamId::West(0));
                    const auto& high = fabric.cell(
                        kColumn, tile, lane, StreamId::West(1));
                    assert(low.valid && high.valid);
                    assert(low.data == static_cast<std::uint8_t>(expected));
                    assert(high.data == static_cast<std::uint8_t>(expected >> 8));
                    assert(low.vector_tag == 100 + tile);
                    assert(high.vector_tag == 100 + tile);
                }
            }
        }

        fabric.begin_cycle();
        dequantizer.evaluate(fabric);
        fabric.commit_cycle();
    }

    assert(dequantizer.statistics().accepted_segments == hw::kTileRows);
    assert(dequantizer.statistics().emitted_segments == hw::kTileRows);
    return 0;
}
