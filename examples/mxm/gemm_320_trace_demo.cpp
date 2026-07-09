#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/mxm/gemm_engine.hpp"
#include "ftlpu/core/topology.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kWeightSramBaseAddress = 6144;
constexpr std::size_t kTargetSreg = ftlpu::hw::kStreamRegisterColumns - 1;
constexpr std::size_t kReadExecuteBaseCycle = 6;
constexpr std::size_t kMxmHandoffBaseCycle = 17;
constexpr std::size_t kComputeCycles = ftlpu::hw::kMxmColumns;
constexpr std::size_t kComputeFlushCycles = 2 * (kBlocks - 1);
constexpr std::size_t kActivationSlice = 16;
constexpr std::size_t kActivationStream = 0;
constexpr std::size_t kResultSliceBase = 40;
constexpr std::size_t kResultStreamBase = 0;
constexpr std::size_t kResultSramBaseAddress = 0;

std::int8_t weight_value(std::size_t k, std::size_t n)
{
    return static_cast<std::int8_t>(static_cast<int>((k * 3 + n * 5) % 7) - 3);
}

std::int8_t activation_value(std::size_t m, std::size_t k)
{
    return static_cast<std::int8_t>(static_cast<int>((m * 2 + k * 3) % 5) - 2);
}

std::size_t weight_slice(std::size_t column_block, std::size_t stream)
{
    (void)column_block;
    return stream;
}

std::size_t weight_address(std::size_t column_block)
{
    return kWeightSramBaseAddress + column_block * kLanes;
}

void inject_weight_streams(ftlpu::TileArrayModel& mem, std::size_t tile, std::size_t column_block)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const auto k = tile * kLanes + lane;
            const auto n = column_block * kLoadStreams + stream;
            mem.set_east_stream_input(
                tile,
                lane,
                stream,
                ftlpu::TileArrayModel::DataWord {
                    static_cast<std::uint8_t>(weight_value(k, n)),
                    lane + 1 == kLanes,
                });
        }
    }
}

void issue_weight_writes_for_cycle(ftlpu::TileArrayModel& mem, std::size_t column_block, std::size_t cycle)
{
    for (std::size_t slice = 0; slice < kLoadStreams; ++slice) {
        const auto absolute_slice = weight_slice(column_block, slice);
        if (ftlpu::stream_register_before_slice(absolute_slice) + 1 == cycle) {
            mem.enqueue_instruction(absolute_slice, ftlpu::MemInstruction::Write(weight_address(column_block), slice));
        }
    }
}

void issue_weight_reads_for_cycle(
    ftlpu::TileArrayModel& mem,
    std::size_t column_block,
    std::size_t cycle,
    std::size_t column_issue_cycle)
{
    for (std::size_t slice = 0; slice < kLoadStreams; ++slice) {
        const auto absolute_slice = weight_slice(column_block, slice);
        if (cycle == column_issue_cycle + ftlpu::stream_register_before_slice(absolute_slice)) {
            mem.enqueue_instruction(absolute_slice, ftlpu::MemInstruction::Read(weight_address(column_block), slice));
        }
    }
}

void issue_pipelined_weight_reads_for_cycle(ftlpu::TileArrayModel& mem, std::size_t cycle)
{
    for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
        issue_weight_reads_for_cycle(mem, column_block, cycle, kReadExecuteBaseCycle + column_block);
    }
}

ftlpu::MxmControlSlice::WeightInput collect_weight_input(const ftlpu::TileArrayModel& mem, std::size_t tile)
{
    ftlpu::MxmControlSlice::WeightInput input{};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
            const auto& slot = mem.east_register(tile, lane, kTargetSreg, stream);
            if (!slot.has_value()) {
                throw std::logic_error("weight handoff reached an empty MEM stream slot");
            }
            input[lane][stream] = ftlpu::MxmArray::Supercell::InputWord {
                static_cast<std::int8_t>(slot->data),
                stream + 1 == kLoadStreams,
            };
        }
    }
    return input;
}

void load_all_weights_through_mem(
    ftlpu::MxmArray& array,
    ftlpu::TileArrayModel& weight_mem,
    std::ostream& mem_log,
    std::ostream& mxm_log)
{
    ftlpu::MxmControlSlice control(array);
    constexpr std::size_t kLastLoadCycle = kMxmHandoffBaseCycle + 2 * (kBlocks - 1);
    for (std::size_t cycle = 0; cycle <= kLastLoadCycle; ++cycle) {
        issue_pipelined_weight_reads_for_cycle(weight_mem, cycle);
        weight_mem.tick(mem_log);

        if (cycle >= kMxmHandoffBaseCycle && cycle < kMxmHandoffBaseCycle + kBlocks) {
            const auto column_block = cycle - kMxmHandoffBaseCycle;
            control.issue_south(ftlpu::MxmControlInstruction::IW(column_block));
        }

        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (cycle < kMxmHandoffBaseCycle + tile) {
                continue;
            }
            const auto column_block = cycle - kMxmHandoffBaseCycle - tile;
            if (column_block < kBlocks) {
                control.set_weight_input(tile, collect_weight_input(weight_mem, tile));
            }
        }
        control.tick(mxm_log);
    }
}

ftlpu::MxmGemmEngine::ActivationVector activation_vector(std::size_t tile, std::size_t k)
{
    ftlpu::MxmGemmEngine::ActivationVector input{};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        input[lane] = ftlpu::MxmGemmEngine::ActivationWord {
            activation_value(tile * kLanes + lane, k),
            lane + 1 == kLanes,
        };
    }
    return input;
}

void store_activation_matrix_to_mem(ftlpu::TileArrayModel& mem, std::ostream& mem_log)
{
    const auto activation_source_sreg = ftlpu::stream_register_before_slice(kActivationSlice);

    mem_log << "store A[320,320] into MEM slice " << kActivationSlice << " stream E0\n";
    for (std::size_t cycle = 0; cycle < kComputeCycles + kBlocks + activation_source_sreg + 1; ++cycle) {
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (cycle >= tile && cycle - tile < kComputeCycles) {
                const auto k = cycle - tile;
                const auto input = activation_vector(tile, k);
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    mem.set_east_stream_input(
                        tile,
                        lane,
                        kActivationStream,
                        ftlpu::TileArrayModel::DataWord {
                            static_cast<std::uint8_t>(input[lane]->data),
                            lane + 1 == kLanes,
                        });
                }
            }
        }

        if (cycle > activation_source_sreg && cycle - activation_source_sreg - 1 < kComputeCycles) {
            mem.enqueue_instruction(
                kActivationSlice,
                ftlpu::MemInstruction::Write((cycle - activation_source_sreg - 1) * kLanes, kActivationStream));
        }
        mem.tick(mem_log);
    }
}

void store_weight_matrix_to_mem(ftlpu::TileArrayModel& mem, std::ostream& mem_log)
{
    mem_log << "store full B[320,320] into MEM slices 0..15 with continuous column-block streaming\n";
    for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
        mem_log << "  B column_block " << column_block
                << " uses slices " << weight_slice(column_block, 0)
                << ".." << weight_slice(column_block, kLoadStreams - 1)
                << " addr=" << weight_address(column_block) << '\n';
    }

    const auto last_source_sreg = ftlpu::stream_register_before_slice(weight_slice(kBlocks - 1, kLoadStreams - 1));
    constexpr std::size_t kLastColumnBlock = kBlocks - 1;
    const auto last_cycle = kLastColumnBlock + kBlocks + last_source_sreg + 1;
    for (std::size_t cycle = 0; cycle <= last_cycle; ++cycle) {
        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            if (cycle >= column_block && cycle - column_block < kBlocks) {
                inject_weight_streams(mem, cycle - column_block, column_block);
            }
            if (cycle >= column_block) {
                issue_weight_writes_for_cycle(mem, column_block, cycle - column_block);
            }
        }
        mem.tick(mem_log);
    }
}

ftlpu::MxmGemmEngine::ActivationVector collect_activation_from_mem(
    const ftlpu::TileArrayModel& mem,
    std::size_t tile)
{
    ftlpu::MxmGemmEngine::ActivationVector input{};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        const auto& slot = mem.east_register(tile, lane, kTargetSreg, kActivationStream);
        if (!slot.has_value()) {
            throw std::logic_error("activation handoff reached an empty MEM stream slot");
        }
        input[lane] = ftlpu::MxmGemmEngine::ActivationWord {
            static_cast<std::int8_t>(slot->data),
            lane + 1 == kLanes,
        };
    }
    return input;
}

void issue_activation_read_for_cycle(ftlpu::TileArrayModel& mem, std::size_t phase_cycle, std::size_t compute_start_cycle)
{
    const auto handoff_sreg = ftlpu::stream_register_after_slice(kActivationSlice);
    const auto read_to_handoff_cycles = kTargetSreg - handoff_sreg + 1;
    if (phase_cycle + read_to_handoff_cycles < compute_start_cycle) {
        return;
    }

    const auto k = phase_cycle + read_to_handoff_cycles - compute_start_cycle;
    if (k >= kComputeCycles) {
        return;
    }

    mem.enqueue_instruction(kActivationSlice, ftlpu::MemInstruction::Read(k * kLanes, kActivationStream));
}

void write_mxm_outputs_to_mem(
    ftlpu::TileArrayModel& result_mem,
    const std::vector<ftlpu::MxmGemmEngine::ColumnOutput>& outputs,
    std::array<std::size_t, kBlocks>& result_counts,
    std::ostream& mem_log)
{
    for (const auto& output : outputs) {
        const auto tile = output.target_tile;
        const auto address = kResultSramBaseAddress + result_counts[tile] * kLanes;
        ++result_counts[tile];

        mem_log << "mxm_result_output tile=" << tile
                << " col_block=" << output.column_block
                << " sreg=11W streams=W" << kResultStreamBase
                << "..W" << (kResultStreamBase + 3)
                << " slices=" << kResultSliceBase
                << ".." << (kResultSliceBase + 3)
                << " addr=" << address
                << " int32=";

        const auto old_flags = mem_log.flags();
        const auto old_fill = mem_log.fill();
        mem_log << std::hex << std::setfill('0');
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const auto raw = static_cast<std::uint32_t>(output.values[lane]);
            if (lane != 0) {
                mem_log << '_';
            }
            mem_log << std::setw(8) << raw;
            for (std::size_t byte = 0; byte < 4; ++byte) {
                const auto value = static_cast<std::uint8_t>((raw >> (8 * byte)) & 0xffu);
                result_mem.set_west_stream_input(
                    tile,
                    lane,
                    kResultStreamBase + byte,
                    ftlpu::TileArrayModel::DataWord {value, lane + 1 == kLanes});
                result_mem.set_sram_byte(kResultSliceBase + byte, tile, address + lane, value);
            }
        }
        mem_log.flags(old_flags);
        mem_log.fill(old_fill);
        mem_log << '\n';
    }
}

void print_mxm_array_state(
    std::ostream& os,
    const ftlpu::MxmControlSlice& control,
    const ftlpu::MxmGemmEngine* gemm)
{
    os << "  array_state:\n";
    for (std::size_t tile = 0; tile < kBlocks; ++tile) {
        os << "    row " << tile << ": ";
        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            if (gemm != nullptr && gemm->computing_cell(tile, column_block)) {
                os << 'C';
            } else if (gemm != nullptr && gemm->completed_cell(tile, column_block)) {
                os << '.';
            } else if (control.loaded_cell(tile, column_block)) {
                os << 'L';
            } else {
                os << '.';
            }
        }
        os << '\n';
    }
}

} // namespace

int run_demo(int argc, char** argv)
{
    const std::string mem_log_path = argc > 1 ? argv[1] : "gemm_320_mem.log";
    const std::string mxm_log_path = argc > 2 ? argv[2] : "gemm_320_mxm.log";
    std::ofstream mem_log(mem_log_path);
    std::ofstream mxm_log(mxm_log_path);
    if (!mem_log) {
        std::cerr << "failed to open MEM log file: " << mem_log_path << '\n';
        return 1;
    }
    if (!mxm_log) {
        std::cerr << "failed to open MXM log file: " << mxm_log_path << '\n';
        return 1;
    }

    auto array = std::make_unique<ftlpu::MxmArray>();
    auto memory = std::make_unique<ftlpu::TileArrayModel>();

    store_activation_matrix_to_mem(*memory, mem_log);
    store_weight_matrix_to_mem(*memory, mem_log);

    constexpr std::size_t kTile0WeightsReadyPhaseCycle = kMxmHandoffBaseCycle + kBlocks - 1;
    constexpr std::size_t kComputeStartPhaseCycle = kTile0WeightsReadyPhaseCycle + 1;
    constexpr std::size_t kLastLoadPhaseCycle = kMxmHandoffBaseCycle + 2 * (kBlocks - 1);
    constexpr std::size_t kLastComputePhaseCycle = kComputeStartPhaseCycle + kComputeCycles + kComputeFlushCycles;
    const auto phase_base_mem_cycle = memory->cycle();

    mxm_log << "loading resident B[320,320] from MEM into MXM through continuous IW pipeline\n";
    mxm_log << "compute A[320,320] x B[320,320], Compute is issued every cycle for " << kComputeCycles
            << " cycles after tile0 weights are loaded at phase cycle " << kComputeStartPhaseCycle
            << " (MEM cycle " << (phase_base_mem_cycle + kComputeStartPhaseCycle) << ")\n";
    mxm_log << "legend: L=loaded idle, C=valid computing 16MAC, .=completed/no valid\n";
    mem_log << "stream A[320,320] from MEM slice " << kActivationSlice
            << " stream E0 to MXM starting as soon as tile0 weights are ready\n";
    mem_log << "connect MXM column outputs to the same MEM sreg11 west streams W0..W3, slices 40..43\n";

    ftlpu::MxmControlSlice control(*array);
    auto gemm = std::make_unique<ftlpu::MxmGemmEngine>(*array);
    bool gemm_started = false;
    std::array<std::size_t, kBlocks> result_counts{};

    for (std::size_t phase_cycle = 0; phase_cycle <= kLastComputePhaseCycle; ++phase_cycle) {
        if (phase_cycle <= kLastLoadPhaseCycle) {
            issue_pipelined_weight_reads_for_cycle(*memory, phase_cycle);
        }
        issue_activation_read_for_cycle(*memory, phase_cycle, kComputeStartPhaseCycle);
        memory->tick(mem_log);

        if (phase_cycle >= kMxmHandoffBaseCycle && phase_cycle < kMxmHandoffBaseCycle + kBlocks) {
            const auto column_block = phase_cycle - kMxmHandoffBaseCycle;
            control.issue_south(ftlpu::MxmControlInstruction::IW(column_block));
        }
        if (phase_cycle >= kComputeStartPhaseCycle && phase_cycle < kComputeStartPhaseCycle + kComputeCycles) {
            control.issue_south(ftlpu::MxmControlInstruction::Compute());
        }
        if (phase_cycle == kComputeStartPhaseCycle) {
            gemm->start_compute(kComputeCycles);
            gemm_started = true;
        }

        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (phase_cycle < kMxmHandoffBaseCycle + tile) {
                continue;
            }
            const auto column_block = phase_cycle - kMxmHandoffBaseCycle - tile;
            if (column_block < kBlocks) {
                control.set_weight_input(tile, collect_weight_input(*memory, tile));
            }
        }
        control.tick(mxm_log, false);

        if (!gemm_started) {
            print_mxm_array_state(mxm_log, control, nullptr);
            continue;
        }

        const auto compute_cycle = phase_cycle - kComputeStartPhaseCycle;
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (compute_cycle >= tile && compute_cycle - tile < kComputeCycles) {
                gemm->set_activation_input(tile, collect_activation_from_mem(*memory, tile));
            }
        }
        gemm->tick(mxm_log, false, false);
        print_mxm_array_state(mxm_log, control, gemm.get());
        write_mxm_outputs_to_mem(*memory, gemm->column_outputs(), result_counts, mem_log);
    }

    std::cout << "wrote GEMM MEM log: " << mem_log_path << '\n';
    std::cout << "wrote GEMM MXM log: " << mxm_log_path << '\n';
    return 0;
}

int main(int argc, char** argv)
{
    try {
        return run_demo(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "gemm_320_trace_demo failed: " << ex.what() << '\n';
        return 1;
    }
}
