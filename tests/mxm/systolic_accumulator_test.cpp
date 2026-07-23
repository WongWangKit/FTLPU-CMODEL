#include "ftlpu/mxm/mxm.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <sstream>

namespace {

ftlpu::MxmArray::InputVector uniform_weights(std::int8_t value)
{
    ftlpu::MxmArray::InputVector input{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t stream = 0; stream < ftlpu::hw::kMxmLoadStreamsPerCycle; ++stream) {
            input[lane][stream] = ftlpu::MxmSupercell::InputWord {
                value,
                stream + 1 == ftlpu::hw::kMxmLoadStreamsPerCycle,
            };
        }
    }
    return input;
}

void load_uniform_weight_buffer(ftlpu::Mxm& mxm, std::size_t buffer, std::int8_t value)
{
    std::ostringstream ignored;
    const auto input = uniform_weights(value);
    for (std::size_t row = 0; row < ftlpu::hw::kMxmSupercellsPerPlane; ++row) {
        for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellsPerPlane; ++column) {
            mxm.array().tick_cell_iw_load(row, column, buffer, input, ignored);
        }
    }
}

void stage_uniform_activation(ftlpu::StreamRegisterFabric& fabric, std::int8_t value)
{
    const auto column = ftlpu::hw::kMemBoundaryStreamRegisterColumns - 1;
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            fabric.initialize_cell(
                column,
                tile,
                lane,
                ftlpu::StreamId::East(0),
                ftlpu::StreamCell::Valid(
                    static_cast<std::uint8_t>(value),
                    lane + 1 == ftlpu::hw::kLanesPerTile));
        }
    }
}

struct RunResult {
    std::size_t outputs{0};
    std::size_t first_output_cycle{static_cast<std::size_t>(-1)};
    std::size_t last_output_cycle{0};
};

RunResult run_adjacent_k_blocks(
    ftlpu::Mxm& mxm,
    ftlpu::StreamRegisterFabric& fabric,
    ftlpu::MxmControlInstruction first,
    ftlpu::MxmControlInstruction second)
{
    // No bubble: the second block enters tile 0 while the first block is
    // still advancing through tiles 1..19.
    mxm.control().issue_south(first);
    mxm.control().issue_south(second);
    RunResult result{};
    constexpr std::size_t kCycles = 2 * ftlpu::hw::kMxmSupercellsPerPlane + 5;
    for (std::size_t cycle = 0; cycle < kCycles; ++cycle) {
        stage_uniform_activation(fabric, 1);
        fabric.begin_cycle();
        mxm.evaluate(fabric, 0);
        if (!mxm.last_outputs().empty()) {
            if (result.outputs == 0) {
                result.first_output_cycle = cycle;
            }
            result.last_output_cycle = cycle;
            for (const auto& output : mxm.last_outputs()) {
                for (const auto value : output.values) {
                    assert(value == 960);
                }
                assert(output.accumulator_bank == 0);
                ++result.outputs;
            }
        }
        fabric.commit_cycle();
    }
    return result;
}

} // namespace

int main()
{
    auto mxm = std::make_unique<ftlpu::Mxm>();
    auto fabric = std::make_unique<ftlpu::StreamRegisterFabric>(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    load_uniform_weight_buffer(*mxm, 0, 1);
    load_uniform_weight_buffer(*mxm, 1, 2);

    // Two adjacent K=320 blocks.  The first retains 320 in acc0; on the very
    // next cycle the second starts and adds 640 before reducing 960 to SR.
    const auto reduced = run_adjacent_k_blocks(
        *mxm,
        *fabric,
        ftlpu::MxmControlInstruction::ComputeToAccumulator(
            0, 0, 0, 0, false, false, true),
        ftlpu::MxmControlInstruction::ComputeToAccumulator(
            1, 0, 0, 0, true, true, true));
    assert(reduced.outputs == ftlpu::hw::kMxmSupercellsPerPlane);
    assert(reduced.first_output_cycle == ftlpu::hw::kMxmSupercellsPerPlane);
    assert(reduced.last_output_cycle == 2 * ftlpu::hw::kMxmSupercellsPerPlane - 1);
    assert(mxm->accumulator_value(0, 0, 0) == 960);
    assert(mxm->accumulator_value(0, 0, 319) == 960);

    return 0;
}
