#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/mxm/gemm_engine.hpp"
#include "ftlpu/core/topology.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>

namespace {

constexpr std::size_t kRows = ftlpu::hw::kMxmRows;
constexpr std::size_t kColumns = ftlpu::hw::kMxmColumns;
constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kSramAddress = 320;
constexpr std::size_t kTargetSreg = ftlpu::hw::kStreamRegisterColumns - 1;
constexpr std::size_t kReadExecuteBaseCycle = 6;
constexpr std::size_t kMxmHandoffBaseCycle = 17;
constexpr std::size_t kComputeCycles = kColumns;
constexpr std::size_t kComputeFlushCycles = 2 * (kBlocks - 1);

class NullBuffer : public std::streambuf {
public:
    int overflow(int c) override
    {
        return c;
    }
};

std::int8_t weight_value(std::size_t k, std::size_t n)
{
    return static_cast<std::int8_t>(static_cast<int>((k * 3 + n * 5) % 7) - 3);
}

std::int8_t activation_value(std::size_t m, std::size_t k)
{
    return static_cast<std::int8_t>(static_cast<int>((m * 2 + k * 3) % 5) - 2);
}

void inject_weight_streams(ftlpu::TileArrayModel& mem, std::size_t tile, std::size_t column_block)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const auto k = tile * kLanes + lane;
            const auto n = column_block * kLoadStreams + stream;
            const auto value = static_cast<std::uint8_t>(weight_value(k, n));
            mem.set_east_stream_input(
                tile,
                lane,
                stream,
                ftlpu::TileArrayModel::DataWord {value, lane + 1 == kLanes});
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
        const auto read_cycle = kReadExecuteBaseCycle + ftlpu::stream_register_before_slice(slice);
        if (cycle == read_cycle) {
            mem.enqueue_instruction(slice, ftlpu::MemInstruction::Read(kSramAddress, slice));
        }
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

void load_weight_column_through_mem(
    ftlpu::MxmArray& array,
    std::size_t column_block,
    std::ostream& os)
{
    auto mem = std::make_unique<ftlpu::TileArrayModel>();
    ftlpu::MxmControlSlice control(array);
    for (std::size_t cycle = 0; cycle < kMxmHandoffBaseCycle + 2 * kBlocks; ++cycle) {
        if (cycle < kBlocks) {
            inject_weight_streams(*mem, cycle, column_block);
        }

        issue_mem_writes_for_cycle(*mem, cycle);
        issue_mem_reads_for_cycle(*mem, cycle);
        mem->tick(os);

        if (cycle == kMxmHandoffBaseCycle) {
            control.issue_south(ftlpu::MxmControlInstruction::IW(column_block));
        }
        if (cycle == kMxmHandoffBaseCycle + kBlocks) {
            control.issue_south(ftlpu::MxmControlInstruction::LW(1u << column_block));
        }
        if (cycle >= kMxmHandoffBaseCycle && cycle < kMxmHandoffBaseCycle + kBlocks) {
            const auto tile = cycle - kMxmHandoffBaseCycle;
            control.set_weight_input(tile, collect_weight_input(*mem, tile));
        }
        control.tick(os);
    }
}

ftlpu::MxmGemmEngine::ActivationVector activation_vector(std::size_t row, std::size_t tile)
{
    ftlpu::MxmGemmEngine::ActivationVector input{};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        input[lane] = ftlpu::MxmGemmEngine::ActivationWord {
            activation_value(row, tile * kLanes + lane),
            lane + 1 == kLanes,
        };
    }
    return input;
}

std::int32_t expected_result(std::size_t row, std::size_t column)
{
    std::int32_t sum = 0;
    for (std::size_t k = 0; k < kRows; ++k) {
        sum += static_cast<std::int32_t>(activation_value(row, k))
            * static_cast<std::int32_t>(weight_value(k, column));
    }
    return sum;
}

bool require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    NullBuffer null_buffer;
    std::ostream null_stream(&null_buffer);
    auto array = std::make_unique<ftlpu::MxmArray>();

    for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
        load_weight_column_through_mem(*array, column_block, null_stream);
    }

    for (std::size_t k_block = 0; k_block < kBlocks; ++k_block) {
        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            assert(array->weight(k_block, column_block, 0, 0)
                == weight_value(k_block * kLanes, column_block * kLoadStreams));
            assert(array->weight(k_block, column_block, 15, 15)
                == weight_value(k_block * kLanes + 15, column_block * kLoadStreams + 15));
        }
    }

    ftlpu::MxmControlSlice control(*array);
    std::ostringstream control_log;
    control.issue_south(ftlpu::MxmControlInstruction::Compute());
    control.tick(control_log);
    if (!require(control.compute_active(0), "compute pulse did not reach tile 0")) {
        return 1;
    }

    auto gemm = std::make_unique<ftlpu::MxmGemmEngine>(*array);
    std::ostringstream gemm_log;
    gemm->start_compute(kComputeCycles);

    for (std::size_t cycle = 0; cycle <= kComputeCycles + kComputeFlushCycles; ++cycle) {
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (cycle >= tile && cycle - tile < kComputeCycles) {
                gemm->set_activation_input(tile, activation_vector(cycle - tile, tile));
            }
        }
        gemm->tick(gemm_log);
    }

    for (std::size_t row = 0; row < kRows; ++row) {
        const auto& result = gemm->accumulator_row(row);
        for (std::size_t column = 0; column < kColumns; ++column) {
            if (!require(result[column] == expected_result(row, column), "GEMM result mismatch")) {
                return 1;
            }
        }
    }

    const auto text = gemm_log.str();
    if (!require(text.find("consume activation tile=0 row=0") != std::string::npos, "missing activation consume log")) {
        return 1;
    }
    if (!require(text.find("valid tile=0 row=0 col_block=0 16MAC") != std::string::npos, "missing valid MAC log")) {
        return 1;
    }
    if (!require(text.find("north_output row=319") != std::string::npos, "missing final north output log")) {
        return 1;
    }
    if (!require(text.find("row 19: ....................") != std::string::npos, "missing final idle row state")) {
        return 1;
    }
    if (!require(control_log.str().find("Compute") != std::string::npos, "missing control compute log")) {
        return 1;
    }

    return 0;
}
