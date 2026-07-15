#include "ftlpu/mxm/mxm.hpp"

#include <cassert>
#include <cstdint>
#include <memory>

namespace {

constexpr std::size_t kWeightColumn = 3;
constexpr std::size_t kWeightStreamBase = 8;

std::uint8_t pattern(std::size_t tile, std::size_t lane, std::size_t stream)
{
    return static_cast<std::uint8_t>(tile + 2 * lane + 3 * stream);
}

} // namespace

int main()
{
    // The endpoints deliberately use non-legacy columns and mixed directions.
    ftlpu::StreamRegisterFabric fabric(7);
    auto mxm = std::make_unique<ftlpu::Mxm>(ftlpu::MxmStreamPortMap {
        ftlpu::MxmStreamPortMap::WeightEndpoint {
            {kWeightColumn, ftlpu::StreamDirection::West, false},
            kWeightStreamBase},
        ftlpu::MxmStreamPortMap::InputEndpoint {
            5, ftlpu::StreamDirection::East, false},
        ftlpu::MxmStreamPortMap::OutputEndpoint {
            1, ftlpu::StreamDirection::West},
    });

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t stream = 0;
             stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
             ++stream) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                fabric.initialize_cell(
                    kWeightColumn,
                    tile,
                    lane,
                    ftlpu::StreamId::West(kWeightStreamBase + stream),
                    ftlpu::StreamCell::Valid(pattern(tile, lane, stream)));
            }
        }
    }

    mxm->control().issue_south(ftlpu::MxmControlInstruction::IW(0));
    for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows; ++cycle) {
        fabric.begin_cycle();
        mxm->evaluate_control(fabric, 0);

        // Keep only inputs that MXM did not consume.  If any lane in any one
        // of the 16 input streams is missed, it remains visible after cycle 20.
        fabric.stage_link(ftlpu::StreamRegisterFabric::Link {
            kWeightColumn,
            kWeightColumn,
            ftlpu::StreamDirection::West,
            true});
        fabric.commit_cycle();
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        assert(mxm->control().loaded_cell(0, tile, 0));
        for (std::size_t stream = 0;
             stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
             ++stream) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto id = ftlpu::StreamId::West(kWeightStreamBase + stream);
                assert(!fabric.cell(kWeightColumn, tile, lane, id).valid);
                assert(mxm->array().weight(0, tile, 0, lane, stream)
                    == static_cast<std::int8_t>(pattern(tile, lane, stream)));
            }
        }
    }

    return 0;
}
