#include "ftlpu/core/topology.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <filesystem>
#include <string>

namespace {

constexpr std::size_t kTileRows = ftlpu::hw::kTileRows;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kSramAddress = 256;
constexpr std::size_t kTargetSreg = ftlpu::hw::kStreamRegisterColumns - 1;
constexpr std::size_t kReadExecuteBaseCycle = 6;
constexpr std::size_t kMxmHandoffBaseCycle = 18;
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

void inject_weight_streams(ftlpu::TspSliceSystem& system, std::size_t tile)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            system.mem().set_east_stream_input(
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

void issue_mem_writes_for_cycle(ftlpu::TspSliceSystem& system, std::size_t cycle)
{
    for (std::size_t slice = 0; slice < kLoadStreams; ++slice) {
        if (ftlpu::stream_register_before_slice(slice) + 1 == cycle) {
            system.icu().enqueue_mem(slice, ftlpu::MemInstruction::Write(kSramAddress, slice));
        }
    }
}

void issue_mem_reads_for_cycle(ftlpu::TspSliceSystem& system, std::size_t cycle)
{
    for (std::size_t slice = 0; slice < kLoadStreams; ++slice) {
        const auto read_cycle = kReadExecuteBaseCycle + ftlpu::stream_register_before_slice(slice);
        if (cycle == read_cycle) {
            system.icu().enqueue_mem(slice, ftlpu::MemInstruction::Read(kSramAddress, slice));
        }
    }
}

std::string marker_for_loaded_row(std::size_t column)
{
    std::string marker(ftlpu::hw::kMxmSupercellsPerPlane, '.');
    marker[column] = 'L';
    return marker;
}

} // namespace

int main()
try
{
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    const auto log_dir = std::filesystem::path("logs") / "mem_mxm";
    std::filesystem::create_directories(log_dir);

    std::ofstream icu_log(log_dir / "icu.log");
    std::ofstream mem_log(log_dir / "mem.log");
    std::ofstream mxm_log(log_dir / "mxm.log");
    std::ofstream vxm_log(log_dir / "vxm.log");
    if (!icu_log || !mem_log || !mxm_log || !vxm_log) {
        std::cerr << "failed to open mem_mxm log files\n";
        return 1;
    }
    auto logs = ftlpu::TspSliceSystem::LogSinks {
        &icu_log,
        &mem_log,
        &mxm_log,
        &vxm_log,
        nullptr,
    };

    for (std::size_t cycle = 0; cycle < kMxmHandoffBaseCycle + kTileRows; ++cycle) {
        if (cycle < kTileRows) {
            inject_weight_streams(*system, cycle);
        }

        issue_mem_writes_for_cycle(*system, cycle);
        issue_mem_reads_for_cycle(*system, cycle);
        if (cycle == kMxmHandoffBaseCycle) {
            system->icu().enqueue_mxm(0, ftlpu::MxmControlInstruction::IW(kMxmColumn));
        }

        system->tick(logs);
    }

    for (std::size_t tile = 0; tile < kTileRows; ++tile) {
        for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                const auto expected = pattern(tile, stream, lane);
                if (!require(
                        system->mem().sram_byte(stream, tile, kSramAddress + lane) == expected,
                        "MEM SRAM byte mismatch")) {
                    return 1;
                }
                if (!require(
                        system->mxm_unit(0).array().weight(tile, kMxmColumn, lane, stream)
                            == static_cast<std::int8_t>(expected),
                        "MXM loaded weight mismatch")) {
                    return 1;
                }
            }
        }
    }

    for (std::size_t tile = 0; tile < kTileRows; ++tile) {
        if (!require(system->mxm_unit(0).control().loaded_cell(tile, kMxmColumn), "missing MXM loaded-cell marker")) {
            return 1;
        }
    }
    if (!require(!marker_for_loaded_row(kMxmColumn).empty(), "invalid marker helper")) {
        return 1;
    }

    std::cout << "wrote logs under: " << log_dir.string() << '\n';

    return 0;
}
catch (const std::exception& ex) {
    std::cerr << "mem_mxm_test failed: " << ex.what() << '\n';
    return 1;
}
