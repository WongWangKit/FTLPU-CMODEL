#include "ftlpu/core/stream_fabric.hpp"
#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/mxm/mxm.hpp"

#include <cstdint>
#include <iostream>
#include <memory>

namespace {

constexpr std::size_t kSramRow = 256;
constexpr std::size_t kMxmWeightColumn = 4;

std::uint8_t pattern(std::size_t tile, std::size_t lane, std::size_t stream)
{
    return static_cast<std::uint8_t>(1 + tile + 2 * lane + 3 * stream);
}

} // namespace

int main()
{
    // All MEM groups in this small demo inject their eastbound Read results at
    // physical SR column 4.  A larger system may map every boundary separately.
    ftlpu::MemStreamPortMap::BoundaryColumns mem_boundaries{};
    mem_boundaries.fill(kMxmWeightColumn);
    ftlpu::MemArrayModel mem{ftlpu::MemStreamPortMap(mem_boundaries)};
    ftlpu::StreamRegisterFabric fabric(8);

    // MXM is connected only through explicit SR endpoints.  There is no
    // TileArrayModel compatibility facade in this composition.
    auto mxm = std::make_unique<ftlpu::Mxm>(ftlpu::MxmStreamPortMap {
        ftlpu::MxmStreamPortMap::WeightEndpoint {
            {kMxmWeightColumn, ftlpu::StreamDirection::East, false}, 0},
        ftlpu::MxmStreamPortMap::InputEndpoint {
            6, ftlpu::StreamDirection::West, false},
        ftlpu::MxmStreamPortMap::OutputEndpoint {
            2, ftlpu::StreamDirection::West},
    });

    for (std::size_t stream = 0;
         stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
         ++stream) {
        for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                mem.set_sram_lane_byte(
                    stream, tile, kSramRow, lane, pattern(tile, lane, stream));
            }
        }
        mem.enqueue_instruction(
            stream,
            ftlpu::MemInstruction::Read(kSramRow, ftlpu::StreamId::East(stream)));
    }

    // Prime SR with MEM tile 0.  Producers stage next-state values and the
    // single global commit makes them visible to MXM on the following cycle.
    fabric.begin_cycle();
    mem.evaluate(fabric);
    mxm->evaluate_control(fabric, 0);
    fabric.commit_cycle();

    mxm->control().issue_south(ftlpu::MxmControlInstruction::IW(0));
    for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows; ++cycle) {
        fabric.begin_cycle();
        mem.evaluate(fabric);
        mxm->evaluate_control(fabric, 0);
        fabric.commit_cycle();
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        if (!mxm->control().loaded_cell(0, tile, 0)) {
            std::cerr << "MXM did not load tile " << tile << '\n';
            return 1;
        }
        for (std::size_t stream = 0;
             stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
             ++stream) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto actual = mxm->array().weight(0, tile, 0, lane, stream);
                const auto expected = static_cast<std::int8_t>(pattern(tile, lane, stream));
                if (actual != expected) {
                    std::cerr << "weight mismatch at tile=" << tile
                              << " lane=" << lane
                              << " stream=" << stream << '\n';
                    return 1;
                }
            }
        }
    }

    std::cout << "MEM row " << kSramRow
              << " -> SR column " << kMxmWeightColumn
              << " -> MXM buffer 0: loaded 20 tiles x 16 streams x 16 lanes\n";
    return 0;
}
