#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/core/topology.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

constexpr std::size_t kTileRows = ftlpu::hw::kTileRows;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kSramAddress = 256;
constexpr std::size_t kTargetSreg = ftlpu::hw::kStreamRegisterColumns - 1;
constexpr std::size_t kReadExecuteBaseCycle = 6;
constexpr std::size_t kMxmHandoffBaseCycle = 17;
constexpr std::size_t kMxmColumn = 6;

bool require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

std::uint8_t pattern(std::size_t tile, std::size_t stream, std::size_t lane)
{
    return static_cast<std::uint8_t>(tile * 4 + stream * 2 + lane);
}

void inject_weight_streams(ftlpu::TileArrayModel& mem, std::size_t tile)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            mem.set_east_stream_input(
                tile,
                lane,
                stream,
                ftlpu::TileArrayModel::DataWord {
                    pattern(tile, stream, lane),
                    lane + 1 == ftlpu::hw::kLanesPerTile,
                });
        }
    }
}

void issue_mem_writes_for_cycle(ftlpu::TileArrayModel& mem, std::size_t cycle)
{
    for (std::size_t slice = 0; slice < kLoadStreams; ++slice) {
        if (ftlpu::stream_register_before_slice(slice) + 1 == cycle) {
            mem.enqueue_instruction(slice, ftlpu::MemInstruction::Write(kSramAddress, slice));
        }
    }
}

void issue_mem_reads_for_cycle(ftlpu::TileArrayModel& mem, std::size_t cycle)
{
    for (std::size_t slice = 0; slice < kLoadStreams; ++slice) {
        const auto source_sreg = ftlpu::stream_register_before_slice(slice);
        const auto read_cycle = kReadExecuteBaseCycle + source_sreg;
        if (cycle == read_cycle) {
            mem.enqueue_instruction(slice, ftlpu::MemInstruction::Read(kSramAddress, slice));
        }
    }
}

ftlpu::MxmControlSlice::WeightInput collect_mxm_weight_input(
    const ftlpu::TileArrayModel& mem,
    std::size_t tile)
{
    ftlpu::MxmControlSlice::WeightInput input{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
            const auto& slot = mem.east_register(tile, lane, kTargetSreg, stream);
            if (!slot.has_value()) {
                throw std::logic_error("MEM to MXM handoff reached an empty stream slot");
            }
            input[lane][stream] = ftlpu::MxmArray::Supercell::InputWord {
                static_cast<std::int8_t>(slot->data),
                stream + 1 == kLoadStreams,
            };
        }
    }
    return input;
}

std::string marker_for_loaded_row(std::size_t column)
{
    std::string marker(ftlpu::hw::kMxmSupercellsPerPlane, '.');
    marker[column] = 'L';
    return marker;
}

} // namespace

int main()
{
    auto mem = std::make_unique<ftlpu::TileArrayModel>();
    auto mxm = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice control(*mxm);
    std::ostringstream log;

    log << "scenario mem_to_mxm streams=16 slices=0..15 target_sreg=" << kTargetSreg
        << " mxm_iw_cycle=" << kMxmHandoffBaseCycle
        << " mxm_column=" << kMxmColumn << '\n';

    for (std::size_t cycle = 0; cycle < kMxmHandoffBaseCycle + kTileRows; ++cycle) {
        log << "global cycle " << cycle << '\n';

        if (cycle < kTileRows) {
            inject_weight_streams(*mem, cycle);
        }

        issue_mem_writes_for_cycle(*mem, cycle);
        issue_mem_reads_for_cycle(*mem, cycle);
        mem->tick(log);

        if (cycle == kMxmHandoffBaseCycle) {
            control.issue_south(ftlpu::MxmControlInstruction::IW(kMxmColumn));
        }

        if (cycle >= kMxmHandoffBaseCycle) {
            const auto tile = cycle - kMxmHandoffBaseCycle;
            control.set_weight_input(tile, collect_mxm_weight_input(*mem, tile));
        }

        control.tick(log);
    }

    for (std::size_t tile = 0; tile < kTileRows; ++tile) {
        for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto expected = pattern(tile, stream, lane);
                if (!require(mem->sram_byte(stream, tile, kSramAddress + lane) == expected, "MEM SRAM byte mismatch")) {
                    return 1;
                }
                if (!require(
                        mxm->weight(tile, kMxmColumn, lane, stream) == static_cast<std::int8_t>(expected),
                        "MXM loaded weight mismatch")) {
                    return 1;
                }
            }
        }
    }

    const auto text = log.str();
    if (!require(text.find("scenario mem_to_mxm streams=16") != std::string::npos, "missing scenario log")) {
        return 1;
    }
    if (!require(text.find("global cycle 17") != std::string::npos, "missing handoff cycle log")) {
        return 1;
    }
    if (!require(text.find("c0.t0=Write(a=256,s=0)") != std::string::npos, "missing first MEM write")) {
        return 1;
    }
    if (!require(text.find("c15.t0=Read(a=256,s=15)") != std::string::npos, "missing last MEM read")) {
        return 1;
    }
    if (!require(text.find("tile 0 IW col=6") != std::string::npos, "missing tile 0 IW")) {
        return 1;
    }
    if (!require(text.find("tile 19 IW col=6") != std::string::npos, "missing tile 19 IW")) {
        return 1;
    }
    if (!require(
            text.find("row 19: " + marker_for_loaded_row(kMxmColumn)) != std::string::npos,
            "missing final MXM row load marker")) {
        return 1;
    }

    return 0;
}
