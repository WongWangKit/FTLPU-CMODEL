#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/core/topology.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr std::size_t kTileRows = ftlpu::hw::kTileRows;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kSramAddress = 256;
constexpr std::size_t kTargetSreg = ftlpu::hw::kStreamRegisterColumns - 1;
constexpr std::size_t kReadExecuteBaseCycle = 6;
constexpr std::size_t kMxmHandoffBaseCycle = 17;
constexpr std::size_t kMxmColumn = 6;

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
        if (cycle == kReadExecuteBaseCycle + source_sreg) {
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

} // namespace

int main(int argc, char** argv)
{
    const std::string log_path = argc > 1 ? argv[1] : "mem_mxm_trace.log";
    std::ofstream log(log_path);
    if (!log) {
        std::cerr << "failed to open log file: " << log_path << '\n';
        return 1;
    }

    auto mem = std::make_unique<ftlpu::TileArrayModel>();
    auto mxm = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice control(*mxm);

    log << "scenario mem_to_mxm streams=16 slices=0..15 target_sreg=" << kTargetSreg
        << " read_base_cycle=" << kReadExecuteBaseCycle
        << " mxm_iw_cycle=" << kMxmHandoffBaseCycle
        << " mxm_column=" << kMxmColumn << '\n';
    log << "schedule: input latches into sreg0 at the end of cycle 0; "
        << "write slices by source sreg at cycles 1..4; "
        << "read slices by source sreg at cycles 6..9; "
        << "streams continue east through the full MEM and reach MXM at sreg "
        << kTargetSreg << "; IW starts at cycle " << kMxmHandoffBaseCycle
        << " and moves north one tile per cycle; LW starts at cycle "
        << (kMxmHandoffBaseCycle + kTileRows) << '\n';

    for (std::size_t cycle = 0; cycle < kMxmHandoffBaseCycle + 2 * kTileRows; ++cycle) {
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
        if (cycle == kMxmHandoffBaseCycle + kTileRows) {
            control.issue_south(ftlpu::MxmControlInstruction::LW(1u << kMxmColumn));
        }

        if (cycle >= kMxmHandoffBaseCycle && cycle < kMxmHandoffBaseCycle + kTileRows) {
            const auto tile = cycle - kMxmHandoffBaseCycle;
            log << "handoff tile " << tile << " sreg " << kTargetSreg
                << " -> MXM row " << tile << " col " << kMxmColumn << '\n';
            control.set_weight_input(tile, collect_mxm_weight_input(*mem, tile));
        }

        control.tick(log);
    }

    std::cout << "wrote MEM to MXM trace log: " << log_path << '\n';
    std::cout << "IW starts at cycle " << kMxmHandoffBaseCycle
              << ", LW starts at cycle " << (kMxmHandoffBaseCycle + kTileRows)
              << ", target_sreg=" << kTargetSreg
              << ", mxm_column=" << kMxmColumn << '\n';
    return 0;
}
