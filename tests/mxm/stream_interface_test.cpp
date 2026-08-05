#include "ftlpu/mxm/mxm.hpp"

#include <cassert>
#include <cstdint>
#include <memory>

namespace {

constexpr std::size_t kWeightColumn = 3;
constexpr std::size_t kWeightStreamBase = 16;

std::uint8_t pattern(std::size_t tile, std::size_t lane, std::size_t stream)
{
    return static_cast<std::uint8_t>(tile + 2 * lane + 3 * stream);
}

} // namespace

int main()
{
    // Weight and activation share one fixed, aligned 16-stream MXM window.
    ftlpu::StreamRegisterFabric fabric(7);
    auto mxm = std::make_unique<ftlpu::Mxm>(ftlpu::MxmStreamPortMap {
        ftlpu::MxmStreamPortMap::InputEndpoint {
            kWeightColumn,
            ftlpu::StreamDirection::West,
            kWeightStreamBase},
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

    // Background loading always reads the fixed upper half of this MXM's
    // 16-stream window. Two explicit half writes rebuild one valid buffer.
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t stream = 8; stream < 16; ++stream) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile;
                 ++lane) {
                fabric.initialize_cell(
                    kWeightColumn,
                    tile,
                    lane,
                    ftlpu::StreamId::West(
                        kWeightStreamBase + stream),
                    ftlpu::StreamCell::Valid(
                        pattern(tile, lane, stream)));
            }
        }
    }
    mxm->control().issue_south(ftlpu::MxmControlInstruction::IW(
        0, ftlpu::MxmWeightLoadMode::BackgroundLowerHalf));
    for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows; ++cycle) {
        fabric.begin_cycle();
        mxm->evaluate_control(fabric, 0);
        fabric.stage_link(ftlpu::StreamRegisterFabric::Link {
            kWeightColumn,
            kWeightColumn,
            ftlpu::StreamDirection::West,
            true});
        fabric.commit_cycle();
    }
    assert(!mxm->control().loaded_cell(0, 0, 0));
    for (std::size_t column = 0; column < 8; ++column) {
        assert(mxm->array().weight(0, 0, 0, 0, column)
            == static_cast<std::int8_t>(pattern(0, 0, 8 + column)));
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t stream = 8; stream < 16; ++stream) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile;
                 ++lane) {
                fabric.initialize_cell(
                    kWeightColumn,
                    tile,
                    lane,
                    ftlpu::StreamId::West(
                        kWeightStreamBase + stream),
                    ftlpu::StreamCell::Valid(static_cast<std::uint8_t>(
                        64 + pattern(tile, lane, stream))));
            }
        }
    }
    mxm->control().issue_south(ftlpu::MxmControlInstruction::IW(
        0, ftlpu::MxmWeightLoadMode::BackgroundUpperHalf));
    for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows; ++cycle) {
        fabric.begin_cycle();
        mxm->evaluate_control(fabric, 0);
        fabric.stage_link(ftlpu::StreamRegisterFabric::Link {
            kWeightColumn,
            kWeightColumn,
            ftlpu::StreamDirection::West,
            true});
        fabric.commit_cycle();
    }
    assert(mxm->control().loaded_cell(0, 0, 0));
    for (std::size_t column = 8; column < 16; ++column) {
        assert(mxm->array().weight(0, 0, 0, 0, column)
            == static_cast<std::int8_t>(
                64 + pattern(0, 0, column)));
    }

    // Compute has no activation-stream field: it reads the first stream in
    // the MXM-local 16-stream window (base 16 for this test instance).
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        fabric.initialize_cell(
            kWeightColumn,
            0,
            lane,
            ftlpu::StreamId::West(kWeightStreamBase),
            ftlpu::StreamCell::Valid(
                static_cast<std::uint8_t>(lane + 1)));
    }
    for (std::size_t stream = 8; stream < 16; ++stream) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            fabric.initialize_cell(
                kWeightColumn,
                0,
                lane,
                ftlpu::StreamId::West(
                    kWeightStreamBase + stream),
                ftlpu::StreamCell::Valid(static_cast<std::uint8_t>(
                    96 + lane + stream)));
        }
    }
    mxm->control().issue_south(ftlpu::MxmControlInstruction::IW(
        1, ftlpu::MxmWeightLoadMode::BackgroundLowerHalf));
    mxm->control().issue_south(
        ftlpu::MxmControlInstruction::Compute(0, 0));
    fabric.begin_cycle();
    mxm->evaluate(fabric, 0);
    fabric.commit_cycle();
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        assert(!fabric.cell(
            kWeightColumn,
            0,
            lane,
            ftlpu::StreamId::West(kWeightStreamBase)).valid);
    }
    for (std::size_t stream = 8; stream < 16; ++stream) {
        assert(!fabric.cell(
            kWeightColumn,
            0,
            0,
            ftlpu::StreamId::West(
                kWeightStreamBase + stream)).valid);
    }
    assert(!mxm->control().loaded_cell(1, 0, 0));

    return 0;
}
